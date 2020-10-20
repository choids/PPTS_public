#include <util/Network.h>
#include "query/Operator.h"
#include "array/Metadata.h"
#include "array/Array.h"
#include "PPTS.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdio.h>

namespace scidb {

    using namespace boost;
    using namespace std;

    int MaximalCacheCounts = 500;

    class PhysicalPPTS: public PhysicalOperator
    {
    private:
        std::vector<WindowBoundaries> _window;
        int _subarraySize = 1;
        int _unitSize = 1;
        Coordinate _subarray_theMostMinorDimSize = 1;
        Coordinate _subarray_theNextMinorDimSize = 1;
        const int EXTRA_PARAMETERS = 8;
    public:
        PhysicalPPTS(const std::string& logicalName, const std::string& physicalName,
                       const Parameters& parameters, const ArrayDesc& schema)
                : PhysicalOperator(logicalName, physicalName, parameters, schema)
        {
            size_t nDims = (parameters.size() - EXTRA_PARAMETERS)/2;
            _window = std::vector<WindowBoundaries>(nDims);
            for (size_t i = 0, size = nDims * 2, boundaryNo = 0; i < size; i+=2, ++boundaryNo)
            {
                _window[boundaryNo] = WindowBoundaries(
                        ((std::shared_ptr<OperatorParamPhysicalExpression>&)_parameters[i])->getExpression()->evaluate().getInt64(),
                        ((std::shared_ptr<OperatorParamPhysicalExpression>&)_parameters[i+1])->getExpression()->evaluate().getInt64()
                );
                Coordinate temp = _window[boundaryNo]._boundaries.first + _window[boundaryNo]._boundaries.second + 1;
                _subarraySize *= (temp);

                if(boundaryNo != (nDims-1)) {
                    _unitSize *= (temp);

                    if(boundaryNo == (nDims-2)){
                        _subarray_theNextMinorDimSize = temp;
                    }
                }
                else{
                    _subarray_theMostMinorDimSize = temp;
                }
            }
        }

        virtual void requiresRedimensionOrRepartition(
                std::vector<ArrayDesc> const& inputSchemas,
                std::vector<ArrayDesc const*>& modifiedPtrs) const
        {
            SCIDB_ASSERT(inputSchemas.size() == 1);
            SCIDB_ASSERT(modifiedPtrs.size() == 1);

            if (inputNeedsRepart(inputSchemas[0])) {
                modifiedPtrs[0] = getRedimensionOrRepartitionSchema(inputSchemas[0]);
            } else {
                modifiedPtrs.clear();
            }
        }

        ArrayDesc* getRedimensionOrRepartitionSchema(
                ArrayDesc const& inputSchema) const
        {
            _redimRepartSchemas.clear();
            Attributes attrs = inputSchema.getAttributes();

            Dimensions dims;
            for (size_t i =0; i<inputSchema.getDimensions().size(); i++)
            {
                DimensionDesc inDim = inputSchema.getDimensions()[i];

                int64_t overlap = inDim.getChunkOverlap();
                int64_t const neededOverlap = std::max(_window[i]._boundaries.first, _window[i]._boundaries.second);
                if ( neededOverlap > inDim.getChunkOverlap())
                {
                    overlap = neededOverlap;
                }

                int64_t rawChunkInterval = inDim.getRawChunkInterval();
                if (inDim.isAutochunked()) {
                    rawChunkInterval = DimensionDesc::PASSTHRU;
                }

                dims.push_back( DimensionDesc(inDim.getBaseName(),
                                              inDim.getNamesAndAliases(),
                                              inDim.getStartMin(),
                                              inDim.getCurrStart(),
                                              inDim.getCurrEnd(),
                                              inDim.getEndMax(),
                                              rawChunkInterval,
                                              overlap));
            }

            _redimRepartSchemas.push_back(std::make_shared<ArrayDesc>(inputSchema.getName(), attrs, dims,
                                                                      inputSchema.getDistribution(),
                                                                      inputSchema.getResidency()));
            return _redimRepartSchemas.back().get();
        }

        bool inputNeedsRepart(ArrayDesc const& input) const
        {
            Dimensions const& dims = input.getDimensions();
            for (size_t i = 0, n = dims.size(); i < n; i++)
            {
                DimensionDesc const& srcDim = dims[i];
                bool justOneChunk = !srcDim.isAutochunked() &&
                                    static_cast<uint64_t>(srcDim.getChunkInterval()) == srcDim.getLength();
                if (!justOneChunk &&
                    srcDim.getChunkOverlap() < std::max(_window[i]._boundaries.first, _window[i]._boundaries.second))
                {
                    return true;
                }
            }
            return false;
        }

        std::shared_ptr<Array> execute(std::vector< std::shared_ptr<Array> >& inputArrays, std::shared_ptr<Query> query) override
        {
            SCIDB_ASSERT(inputArrays.size() == 3);

            std::shared_ptr<Array> inputArray = ensureRandomAccess(inputArrays[0], query);
            std::shared_ptr<Array> partitions = ensureRandomAccess(inputArrays[1], query);
            std::shared_ptr<Array> maxDensityArray = ensureRandomAccess(inputArrays[2], query);

            ArrayDesc const& inDesc = inputArray->getArrayDesc();

            std::vector<AttributeID> inputAttrIDs;
            std::vector<AggregatePtr> aggregates;
            std::string method;
            int topK = 0;
            size_t nDims = inDesc.getDimensions().size();
            bool disjoint = true;//disjoint = true , overlap-allowing = false
            bool maxDensityOpt;
            bool materialize;

            size_t start = inDesc.getDimensions().size() * 2;
            for (size_t i = start, size = _parameters.size(); i < size; i++)
            {
                std::shared_ptr<scidb::OperatorParam> & param = _parameters[i];

                if(i == start){ // k
                    topK = ((std::shared_ptr<OperatorParamPhysicalExpression>&)param)->getExpression()->evaluate().getInt32();
                }
                else if(i == start + 1){ // scoring function
                    AttributeID inAttId;
                    AggregatePtr agg = resolveAggregate((std::shared_ptr <OperatorParamAggregateCall> const&) _parameters[i],
                                                        inDesc.getAttributes(),
                                                        &inAttId,
                                                        0);

                    aggregates.push_back(agg);

                    if (inAttId == INVALID_ATTRIBUTE_ID)
                    {
                        inputAttrIDs.push_back(0);
                    } else
                    {
                        inputAttrIDs.push_back(inAttId);
                    }
                }
                else if(i == start + 2){ // overlap-allowing or disjoint
                    std::string disjointString = ((std::shared_ptr<OperatorParamPhysicalExpression> &) param)->getExpression()->evaluate().getString();
                    std::transform(disjointString.begin(), disjointString.end(), disjointString.begin(), ::tolower);
                    if(disjointString == "disjoint"){
                        disjoint = true;
                    }
                    else{
                        disjoint = false;
                    }
                }
                else if(i == start + 3){ // Maximal density estimation
                    std::string maxDensityString = ((std::shared_ptr<OperatorParamPhysicalExpression> &) param)->getExpression()->evaluate().getString();
                    std::transform(maxDensityString.begin(), maxDensityString.end(), maxDensityString.begin(), ::tolower);
                    if(maxDensityString == "maxdensity"){
                        maxDensityOpt = true;
                    }
                    else{
                        maxDensityOpt = false;
                    }
                }
            }

            int myNumChunks = 0;
            std::shared_ptr<ConstArrayIterator> arrayIterator = inputArray->getConstIterator(0);
            while (!arrayIterator->end()) {
                myNumChunks++;
                ++(*arrayIterator);
            }
            arrayIterator->restart();

            if(MaximalCacheCounts > myNumChunks)
                MaximalCacheCounts = myNumChunks;

            std::shared_ptr<Materials> materials = std::make_shared<Materials>(nDims);
            computeLocalTopK(inputArray, partitions, maxDensityArray, topK, inputAttrIDs, aggregates, method, query,
                             disjoint, materials, myNumChunks, maxDensityOpt);

            if (query->getInstanceID() == 0)
            {
                return makeFinalTopKArray(materials->_finalTopK,inDesc.getDimensions().size(),query,disjoint);
            }
            else
            {
                return std::shared_ptr<Array>(new MemArray(_schema, query));
            }
        }

        /**
         * getMaxScore()
         * aggregate 종류에 따라 현재 partition에서 가질 수 있는 최대 score를 반환하는 함수
         */
        double getMaxScore(int aggregateType,double representative,int windowSize,bool maxDensityOpt,double maxDensity){
            switch(aggregateType){
                case 0: {//sum
                    if (maxDensityOpt) {
                        return representative * _subarraySize * maxDensity;
                    } else {
                        return representative * _subarraySize;
                    }
                }
                case 1: {//avg
                    return representative;
                }
            }
            return 0;
        }

        void computeLocalTopK(std::shared_ptr<Array>& inputArray, std::shared_ptr<Array>& partitions,
                std::shared_ptr<Array>& maxDensityArray, int topK, const std::vector<AttributeID> & inputAttrIDs,
                const std::vector<AggregatePtr>& aggregates, const std::string &method,std::shared_ptr<Query>& query,
                bool const disjoint,std::shared_ptr<Materials>& materials, int const myNumChunks,
                bool const maxDensityOpt)
        {
            Dimensions dimensions = inputArray->getArrayDesc().getDimensions();
            size_t _nDims = dimensions.size();

            PPTSArray array = PPTSArray(inputArray->getArrayDesc(), inputArray, _window, inputAttrIDs, aggregates, method, topK, disjoint);
            std::shared_ptr<PPTSArrayIterator> arrayIterator = array.getProgressiveArrayIterator(0);
            std::shared_ptr<ConstArrayIterator> maxDensityArrayIterator;
            if(maxDensityOpt) {
                maxDensityArrayIterator = maxDensityArray->getConstIterator(0);
            }
            array.unitSize = _unitSize;
            array.subarraySize = _subarraySize;
            array.subarray_theMostMinorDimSize = _subarray_theMostMinorDimSize;
            array.subarray_theNextMinorDimSize = _subarray_theNextMinorDimSize;

            std::shared_ptr<ConstArrayIterator> pStartingCellIter = partitions->getConstIterator(0);
            std::shared_ptr<ConstArrayIterator> pEndingCellIter = partitions->getConstIterator(1);
            std::shared_ptr<ConstArrayIterator> pMaxIter = partitions->getConstIterator(2);
            InstanceID myID = query->getInstanceID();

            vector<int> chunkOrder;
            for(int i=0;i<myNumChunks;i++){
                chunkOrder.push_back(i);
            }

            Coordinates const& firstInArray = array.getFirstPosition();
            Coordinates const& lastInArray = array.getLastPosition();

            int totalPartition_test = 0;
            while(!pStartingCellIter->end()){
                Coordinates chunkCoordinate = pStartingCellIter->getPosition();
                double maxDensity = 1;
                if(maxDensityOpt) {
                    maxDensityArrayIterator->setPosition(chunkCoordinate);
                    ConstChunk const &maxDensityChunk = maxDensityArrayIterator->getChunk();
                    std::shared_ptr<ConstChunkIterator> maxDensityChunkIterator = maxDensityChunk.getConstIterator();
                    maxDensity = maxDensityChunkIterator->getItem().getDouble();
                    array.chunkInfos.maxDensities.push_back(maxDensity);
                }
                else{
                    array.chunkInfos.maxDensities.push_back(1);
                }

                arrayIterator->setPosition(chunkCoordinate);
                pEndingCellIter->setPosition(chunkCoordinate);
                pMaxIter->setPosition(chunkCoordinate);
                PPTSChunk chunk = arrayIterator->getProgressiveChunk();
                array.chunkInfos.chunkStartCoordinates.push_back(chunk.getFirstPosition(false));
                array.chunkInfos.nextPartitionPosition.push_back(0);

                ConstChunk const &pMaxChunk = pMaxIter->getChunk();
                std::shared_ptr<ConstChunkIterator> pMaxChunkIterator = pMaxChunk.getConstIterator();

                double pMaxString = pMaxChunkIterator->getItem().getDouble();
                double UBS = getMaxScore(array.getAggregateType(),pMaxString,_subarraySize,maxDensityOpt,maxDensity);
                array.chunkInfos.nextPartitionUBS.push_back(UBS);

                unsigned long temp_totalSubarrays = 1;
                unsigned long temp_totalUnits = 1;
                unsigned long temp_totalCells = 1;
                Coordinates temp_newChunkInterval_forUnitStartPosition;
                Coordinates temp_newChunkInterval_forSubarrayStartPosition;
                Coordinates temp_newChunkInterval_EndToEnd;
                Coordinates firstPos = chunk.getFirstPosition(false);
                Coordinates lastPos = chunk.getLastPosition(false);
                Coordinates temp_actualFirstPos;
                Coordinates temp_actualLastPos;
                Coordinates temp_realLastPos;
                Coordinates temp_realFirstPos;

                for (size_t i = 0; i < _nDims; i++) {
                    int64_t chunkOverlap = array.getChunkOverlap(i);
                    temp_realFirstPos.push_back(std::max(firstPos[i] - chunkOverlap, firstInArray[i]));
                    temp_realLastPos.push_back(std::min(lastPos[i] + chunkOverlap, lastInArray[i]));
                    Coordinate tempReal = temp_realLastPos[i] - temp_realFirstPos[i] + 1;
                    temp_actualFirstPos.push_back(temp_realFirstPos[i] + _window[i]._boundaries.first);
                    temp_actualLastPos.push_back(temp_realLastPos[i] - _window[i]._boundaries.second);
                    temp_totalSubarrays *= (temp_actualLastPos[i] - temp_actualFirstPos[i] + 1);
                    temp_totalCells *= (tempReal);
                    if (i != _nDims - 1) {
                        Coordinate temp = temp_actualLastPos[i] - temp_realFirstPos[i] + 1;
                        temp_newChunkInterval_forUnitStartPosition.push_back(temp);
                        temp_totalUnits *= temp;
                    } else {
                        Coordinate temp = tempReal;
                        temp_newChunkInterval_forUnitStartPosition.push_back(temp);
                        temp_totalUnits *= temp;
                    }
                    temp_newChunkInterval_forSubarrayStartPosition.push_back(temp_actualLastPos[i] - temp_actualFirstPos[i] + 1);
                    temp_newChunkInterval_EndToEnd.push_back(tempReal);
                }

                array.chunkInfos.actualFirstPos.push_back(temp_actualFirstPos);
                array.chunkInfos.actualLastPos.push_back(temp_actualLastPos);
                array.chunkInfos.realLastPos.push_back(temp_realLastPos);
                array.chunkInfos.realFirstPos.push_back(temp_realFirstPos);
                array.chunkInfos.totalSubarrays.push_back(temp_totalSubarrays);
                array.chunkInfos.totalCells.push_back(temp_totalCells);
                array.chunkInfos.totalUnits.push_back(temp_totalUnits);
                array.chunkInfos.newChunkInterval_forUnitStartPosition.push_back(temp_newChunkInterval_forUnitStartPosition);
                array.chunkInfos.newChunkInterval_forSubarrayStartPosition.push_back(temp_newChunkInterval_forSubarrayStartPosition);
                array.chunkInfos.newChunkInterval_EndToEnd.push_back(temp_newChunkInterval_EndToEnd);
                if(temp_totalSubarrays > array.chunkInfos.max_TotalSubarrays){
                    array.chunkInfos.max_TotalSubarrays = temp_totalSubarrays;
                    array.chunkInfos.max_TotalCells = temp_totalCells;
                    array.chunkInfos.max_TotalUnits = temp_totalUnits;
                }

                ConstChunk const &pStartingCellChunk = pStartingCellIter->getChunk();
                std::shared_ptr<ConstChunkIterator> pStartingCellChunkIterator = pStartingCellChunk.getConstIterator();
                std::vector<string> pStartingCellTemp;
                while (!pStartingCellChunkIterator->end()) {
                    pStartingCellTemp.push_back(pStartingCellChunkIterator->getItem().getString());
                    ++(*pStartingCellChunkIterator);
                }

                ConstChunk const &pEndingCellChunk = pEndingCellIter->getChunk();
                std::shared_ptr<ConstChunkIterator> pEndingCellChunkIterator = pEndingCellChunk.getConstIterator();
                std::vector<string> pEndingCellTemp;
                while (!pEndingCellChunkIterator->end()) {
                    pEndingCellTemp.push_back(pEndingCellChunkIterator->getItem().getString());
                    ++(*pEndingCellChunkIterator);
                }

                std::vector<double> pMaxTemp;
                while (!pMaxChunkIterator->end()) {
                    pMaxTemp.push_back(pMaxChunkIterator->getItem().getDouble());
                    ++(*pMaxChunkIterator);
                }

                array.chunkInfos.pStartingCells.push_back(pStartingCellTemp);
                array.chunkInfos.pEndingCells.push_back(pEndingCellTemp);
                array.chunkInfos.pMaxs.push_back(pMaxTemp);
                totalPartition_test += pStartingCellTemp.size();

                ++(*pStartingCellIter);
            }

            priority_queue<struct_NextPartitionUBS,std::vector<struct_NextPartitionUBS>,std::less<struct_NextPartitionUBS> > temp_UBS;
            for(size_t i=0;i<array.chunkInfos.nextPartitionUBS.size();i++){
                temp_UBS.emplace(i,array.chunkInfos.nextPartitionUBS[i]);
            }
            chunkOrder.clear();
            while(!temp_UBS.empty()){
                chunkOrder.push_back(temp_UBS.top().chunkNum);
                temp_UBS.pop();
            }

            ChunkCaches chunkCaches(MaximalCacheCounts,array.chunkInfos.max_TotalSubarrays,array.chunkInfos.max_TotalUnits,
                                    array.chunkInfos.max_TotalCells);

            int howManyPartitionsChecked = 0;
            int currentK = 1;
            for(;currentK <= topK ; currentK++) {
                for (size_t i = 0; i < chunkOrder.size(); i++) {
                    array.currentChunk = chunkOrder[i];
                    if (materials->localAnswer.value > array.chunkInfos.nextPartitionUBS[array.currentChunk])
                        continue;

                    Coordinates currentChunkStartCoordinates = array.chunkInfos.chunkStartCoordinates[array.currentChunk];
                    arrayIterator->setPosition(currentChunkStartCoordinates);

                    PPTSChunk& chunk = arrayIterator->getProgressiveChunk();
                    std::shared_ptr<PPTSChunkIterator> chunkIterator = std::make_shared<PPTSChunkIterator>(array, *arrayIterator, chunk, 1026);

                    chunk.setActualFirstLastPosition(array.chunkInfos.actualFirstPos[array.currentChunk],array.chunkInfos.actualLastPos[array.currentChunk]);
                    chunk.setRealFirstLastPosition(array.chunkInfos.realFirstPos[array.currentChunk],array.chunkInfos.realLastPos[array.currentChunk]);
                    chunk.set_newChunkInterval_forSubarrayStartPosition(array.chunkInfos.newChunkInterval_forSubarrayStartPosition[array.currentChunk]);
                    chunk.set_newChunkInterval_forUnitStartPosition(array.chunkInfos.newChunkInterval_forUnitStartPosition[array.currentChunk]);
                    chunk.set_newChunkInterval_EndToEnd(array.chunkInfos.newChunkInterval_EndToEnd[array.currentChunk]);
                    chunk.theMostMinorDimSize = array.chunkInfos.newChunkInterval_EndToEnd[array.currentChunk][_nDims-1];
                    bool update = chunkCaches.cacheUpdate(array.currentChunk);

                    if(update){
                        Coordinates currGridPos = array.chunkInfos.realFirstPos[array.currentChunk];
                        currGridPos[_nDims-1] -= 1;
                        Coordinates currRealLastPos = array.chunkInfos.realLastPos[array.currentChunk];
                        // This code assumes that a chunk is loaded into memory using c++std::map.
                        // In the case of dense chunks, this code can be optimized by using c++std::vector.
                        std::map<uint64_t, double>& materializedChunk = chunkCaches.caches[array.currentChunk].materializedChunk;

                        while(!chunkIterator->_inputIterator->end()) {
                            Coordinates const& currPos = chunkIterator->_inputIterator->getPosition();
                            Value const& currVal = chunkIterator->_inputIterator->getItem();

                            position_t position = chunk.coord2pos_withOverlap(currPos);
                            materializedChunk[position] = currVal.getDouble();
                            ++(*chunkIterator->_inputIterator);
                        }
                    }

                    size_t p = array.chunkInfos.nextPartitionPosition[array.currentChunk];
                    size_t pCount = array.chunkInfos.pStartingCells[array.currentChunk].size();
                    for(; p < pCount; p++){
                        Coordinates pStartingCell;
                        Coordinates pEndingCell;

                        std::string pStartingCellString = array.chunkInfos.pStartingCells[array.currentChunk][p];
                        std::string pEndingCellString = array.chunkInfos.pEndingCells[array.currentChunk][p];
                        double pMaxString = array.chunkInfos.pMaxs[array.currentChunk][p];
                        chunkIterator->info.setRepresentative(pMaxString);

                        std::istringstream ss(pStartingCellString);
                        std::string token;
                        while (std::getline(ss, token, ',')) {
                            pStartingCell.push_back(std::stoi(token));
                        }
                        std::istringstream ss2(pEndingCellString);
                        while (std::getline(ss2, token, ',')) {
                            pEndingCell.push_back(std::stoi(token));
                        }

                        double maxDensity = array.chunkInfos.maxDensities[array.currentChunk];
                        double UBS = getMaxScore(array.getAggregateType(),pMaxString,_subarraySize,maxDensityOpt,maxDensity);
                        chunkIterator->info.setMaxScore(UBS);

                        if (materials->localAnswer.value > UBS) {// Answer-returning condition
                            array.chunkInfos.nextPartitionPosition[array.currentChunk] = p;
                            array.chunkInfos.nextPartitionUBS[array.currentChunk] = UBS;
                            break;
                        }

                        chunk.checkOnePartition_IC(pStartingCell, pEndingCell, chunkIterator, materials,
                                                                   chunkCaches.caches[array.currentChunk]);

                        howManyPartitionsChecked++;
                    }
                    if(p == array.chunkInfos.pStartingCells[array.currentChunk].size()){
                        array.chunkInfos.nextPartitionUBS[array.currentChunk] = -INFINITY;
                    }
                }

                if (query->getInstanceID() != 0) { // worker node
                    progressiveTopKCandidate temp(materials->localAnswer);
                    std::shared_ptr<SharedBuffer> buf2 = temp.marshall(_nDims);
                    BufSend(0, buf2, query);
                } else { // coordinator node
                    for (InstanceID i = 1; i < query->getInstancesCount(); ++i) {
                        std::shared_ptr<SharedBuffer> buf2 = BufReceive(i, query);
                        progressiveTopKCandidate otherIth(buf2, _nDims);
                        if (materials->localAnswer.value < otherIth.value) {
                            materials->localAnswer = otherIth;
                        }
                    }
                }

                if (query->getInstanceID() != 0) { // worker node
                    std::shared_ptr<SharedBuffer> buf2 = BufReceive(0, query);
                    progressiveTopKCandidate finalIth(buf2, _nDims);
                    materials->localAnswer = finalIth;
                } else { // coordinator node
                    progressiveTopKCandidate temp(materials->localAnswer);
                    std::shared_ptr<SharedBuffer> buf2 = temp.marshall(_nDims);
                    for (InstanceID i = 1; i < query->getInstancesCount(); ++i) {
                        BufSend(i, buf2, query);
                    }
                }
                materials->_finalTopK.push_back(materials->localAnswer);

                while(!materials->_backup.empty()){
                    if(materials->_backup.top().value > materials->localAnswer.value){
                        materials->_backup.pop();
                    }
                    else{
                        break;
                    }
                }

                vector<progressiveTopKCandidate> tie;
                while(!materials->_backup.empty()){
                    if(materials->_backup.top().value == materials->localAnswer.value){
                        tie.push_back(materials->_backup.top());
                        materials->_backup.pop();
                    }
                    else{
                        break;
                    }
                }

                int index = 0;
                for(; index < tie.size() ; index++){
                    bool same = true;
                    for(size_t j = 0; j<_nDims ; j++){
                        if(tie[index].coordinate[j] != materials->localAnswer.coordinate[j]){
                            same = false;
                            break;
                        }
                    }
                    if(same)
                        break;
                }
                for(int f = 0;f < tie.size(); f++) {
                    if (f != index)
                        materials->_backup.push(tie[f]);
                }

                materials->localAnswer = progressiveTopKCandidate(_nDims);
                while(!materials->_backup.empty()){
                    bool accept = true;
                    if(disjoint) {
                        for (auto it2 = materials->_finalTopK.begin(); it2 != materials->_finalTopK.end(); it2++) {
                            if (materials->_backup.top().range.intersects(it2->range)) {
                                accept = false;
                            }
                        }
                    }
                    if(accept){
                        materials->localAnswer = materials->_backup.top();
                        break;
                    }
                    materials->_backup.pop();
                }

                priority_queue<struct_NextPartitionUBS,std::vector<struct_NextPartitionUBS>,std::less<struct_NextPartitionUBS> > temp_UBS;
                for(size_t i=0;i<array.chunkInfos.nextPartitionUBS.size();i++){
                    temp_UBS.emplace(i,array.chunkInfos.nextPartitionUBS[i]);
                }
                chunkOrder.clear();
                while(!temp_UBS.empty()){
                    chunkOrder.push_back(temp_UBS.top().chunkNum);
                    temp_UBS.pop();
                }
            }
        }

        std::shared_ptr<Array> makeFinalTopKArray
                (std::list<progressiveTopKCandidate>& finalTopK,size_t numDims, std::shared_ptr<Query>& query, bool disjoint)
        {
            std::shared_ptr<Array> outputArray(new MemArray(_schema, query));
            Coordinates startingPosition(1, query->getInstanceID());

            std::shared_ptr<ArrayIterator> outputArrayIter_coordinate = outputArray->getIterator(0);
            std::shared_ptr<ChunkIterator> outputChunkIter_coordinate = outputArrayIter_coordinate->newChunk(startingPosition).getIterator(query,
                                                                                                                                           ChunkIterator::SEQUENTIAL_WRITE);
            std::shared_ptr<ArrayIterator> outputArrayIter_value = outputArray->getIterator(1);
            std::shared_ptr<ChunkIterator> outputChunkIter_value = outputArrayIter_value->newChunk(startingPosition).getIterator(query, ChunkIterator::SEQUENTIAL_WRITE |
                                                                                                                                        ChunkIterator::NO_EMPTY_CHECK);
            outputChunkIter_coordinate->setPosition(startingPosition);
            outputChunkIter_value->setPosition(startingPosition);

            int i = 0;
            for(auto it = finalTopK.begin(); it != finalTopK.end() ; it++, i++){

                std::string s;
                for(size_t j=0;j<numDims;j++) {
                    s.append(to_string(it->coordinate[j]));
                    if(j != numDims-1)
                        s.append(",");
                }
                Value value;
                value.setString(s);
                outputChunkIter_coordinate->writeItem(value);

                Value value2;
                value2.setDouble(it->value);
                outputChunkIter_value->writeItem(value2);

                ++(*outputChunkIter_coordinate);
                ++(*outputChunkIter_value);

            }
            outputChunkIter_coordinate->flush();
            outputChunkIter_value->flush();

            return outputArray;
        }
    };

    REGISTER_PHYSICAL_OPERATOR_FACTORY(PhysicalPPTS, "PPTS", "physicalPPTS");
}
