#include <boost/numeric/conversion/cast.hpp>
#include <log4cxx/logger.h>
#include <math.h>
#include "PPTS.h"
#include <util/RegionCoordinatesIterator.h>
#include "query/Operator.h"
#include "array/Metadata.h"
#include "array/Array.h"

using namespace std;
using namespace boost;

namespace scidb
{
    PPTSChunkIterator::PPTSChunkIterator(PPTSArray const& array,PPTSArrayIterator const& arrayIterator, PPTSChunk const& chunk, int mode)
            : _array(array),
              _chunk(chunk),
              _firstPos(_chunk.getFirstPosition(false)),
              _lastPos(_chunk.getLastPosition(false)),
              _currPos(_firstPos.size()),
              _attrID(_chunk._attrID),
              _aggregate(_array._aggregates[_attrID]->clone()),
              _iterationMode(mode),
              _nextValue(TypeLibrary::getType(_chunk.getAttributeDesc().getType())),
              sumInManagingList(0),
              countInManagingList(0),
              _nDims(_array._dimensions.size()),
              _subarrayStartCoords(_nDims),
              _subarrayEndCoords(_nDims)
    {
        if ((_iterationMode & IGNORE_EMPTY_CELLS) == false)
        {
            throw SYSTEM_EXCEPTION(SCIDB_SE_INTERNAL, SCIDB_LE_CHUNK_WRONG_ITERATION_MODE);
        }

        int iterMode = IGNORE_EMPTY_CELLS;
        if (_aggregate->ignoreNulls())
        {
            iterMode |= IGNORE_NULL_VALUES;
        }

        if ( _aggregate->ignoreZeroes() && attributeDefaultIsSameAsTypeDefault() )
        {
            iterMode |= IGNORE_DEFAULT_VALUES;
        }

        _inputIterator = arrayIterator.iterator->getChunk().getConstIterator(iterMode);

        if (_array.getArrayDesc().getEmptyBitmapAttribute())
        {
            AttributeID eAttrId = _array._inputArray->getArrayDesc().getEmptyBitmapAttribute()->getId();
            _emptyTagArrayIterator = _array._inputArray->getConstIterator(eAttrId);

            if (! _emptyTagArrayIterator->setPosition(_firstPos))
            {
                throw SYSTEM_EXCEPTION(SCIDB_SE_EXECUTION, SCIDB_LE_OPERATION_FAILED) << "setPosition";
            }
            _emptyTagIterator = _emptyTagArrayIterator->getChunk().getConstIterator(IGNORE_EMPTY_CELLS);
        }

        _windowSize = 1;
        for(size_t i=0;i<_nDims;i++){
            _windowSize *= (_array._window[i]._boundaries.first+_array._window[i]._boundaries.second+1);
        }

        restart();
    }
    const std::shared_ptr<ConstChunkIterator>& PPTSChunkIterator::getInputIterator() const
    {
        return _inputIterator;
    }

    void PPTSChunkIterator::addToTotal(unitInfo& unitResult)
    {
        switch(_array.getAggregateType()){
            case 0://sum
            case 1: {//avg
                sumInManagingList += unitResult.value;
                countInManagingList += unitResult.cellCount;
                break;
            }
        }
    }

    void PPTSChunkIterator::removeToTotal(unitInfo& unitResult)
    {
        switch(_array.getAggregateType()){
            case 0://sum
            case 1: {//avg
                sumInManagingList -= unitResult.value;
                countInManagingList -= unitResult.cellCount;
                break;
            }
        }
    }

    void PPTSChunkIterator::initialize(unitInfo& unitResult,std::shared_ptr<Materials>& materials)
    {
        switch(_array.getAggregateType()){
            case 0://sum
            case 1: {//avg
                unitResult.value = 0;
                break;
            }
        }
    }

    void PPTSChunkIterator::finalizeIC()
    {
        _nextCellCount = countInManagingList;
        switch(_array.getAggregateType()){
            case 0: {//sum
                _nextValue.setDouble(sumInManagingList);
                break;
            }
            case 1: {//avg
                if (countInManagingList == 0 || sumInManagingList == DNEGINF)
                    _nextValue.setDouble(DNEGINF);
                else {
                    _nextValue.setDouble(sumInManagingList / (double) countInManagingList);
                }
                break;
            }
        }
    }

    bool PPTSChunkIterator::attributeDefaultIsSameAsTypeDefault() const
    {
        const AttributeDesc& a = _array._inputArray->getArrayDesc().getAttributes()[_array._inputAttrIDs[_attrID]];
        return isDefaultFor(a.getDefaultValue(),a.getType());
    }

    int PPTSChunkIterator::getMode() const
    {
        return _iterationMode;
    }


    // This code assumes that a chunk is loaded using c++std::map.
    // In the case of dense chunks, this code can be optimized by using c++std::vector.
    unitInfo PPTSChunkIterator::calculateUnit(Coordinates first, Coordinates last, int lastDimValue,chunkCache& cache)
    {
        unitInfo unitResult;

        first[_nDims - 1] = lastDimValue;
        last[_nDims - 1] = lastDimValue;
        Coordinates currentCoords = first;

        std::map<uint64_t, double> &materializedChunk = cache.materializedChunk;

        map<uint64_t, double>::const_iterator unitStart;
        map<uint64_t, double>::const_iterator unitEnd;
        position_t unitFirstPosition = _chunk.coord2pos_withOverlap_EndToEnd(first);
        unitStart = materializedChunk.lower_bound(unitFirstPosition);
        position_t unitLastPosition = _chunk.coord2pos_withOverlap_EndToEnd(last);
        unitEnd = materializedChunk.upper_bound(unitLastPosition);
        if ( unitStart == unitEnd )
            return unitResult;

        int jumpNum = (_array.subarraySize /
                       (_array.subarray_theMostMinorDimSize * _array.subarray_theNextMinorDimSize));
        for (int j = 0; j < jumpNum; j++) {
            position_t currentPosition = _chunk.coord2pos_withOverlap_EndToEnd(currentCoords);

            for (size_t i = 0; i < _array.subarray_theNextMinorDimSize; i++) {
                map<uint64_t, double>::const_iterator it = materializedChunk.find(currentPosition);
                if (it != materializedChunk.end()) {
                    unitResult.cellCount++;
                    unitResult.value += it->second;
                }

                currentPosition += _chunk.theMostMinorDimSize;
            }

            // This code does not consider a 1-dimensional array.
            currentCoords[_nDims - 2] += (_array.subarray_theNextMinorDimSize - 1);
            for (size_t i = _nDims - 2; ++currentCoords[i] > _subarrayEndCoords[i]; i--) {
                if (i == 0) {
                    return unitResult;
                }
                currentCoords[i] = _subarrayStartCoords[i];
            }
        }
    }

    void PPTSChunkIterator::calculateSubarrayScoreIC
            (Coordinates const& currPos, std::shared_ptr<Materials>& materials, Coordinates const& realLeftBottom,chunkCache& cache,bool& managingList)
    {
        info.setLocalTopKIsEmpty(materials->_finalTopK.empty());

        int lastLength = _array._window[_nDims-1]._boundaries.first + _array._window[_nDims-1]._boundaries.second;
        for (size_t i = 0; i < _nDims; i++) {
            _subarrayStartCoords[i] = currPos[i]-_array._window[i]._boundaries.first;
            _subarrayEndCoords[i] = currPos[i]+_array._window[i]._boundaries.second;
        }
        position_t currentPosition = _chunk.coord2pos_withOverlap_forUnitStartPosition(_subarrayStartCoords);

        if(managingList)
        {
            for (size_t lastDimValue = _subarrayStartCoords[_nDims-1]; lastDimValue <= _subarrayEndCoords[_nDims-1]; lastDimValue++,currentPosition++)
            {
                if(cache.unitMemo[currentPosition].value == DNEGINF) {
                    unitInfo unitResult = calculateUnit(_subarrayStartCoords, _subarrayEndCoords, lastDimValue, cache);
                    cache.unitMemo[currentPosition].value = unitResult.value;
                    cache.unitMemo[currentPosition].cellCount = unitResult.cellCount;
                }
                addToTotal(cache.unitMemo[currentPosition]);
                managingList = false;
            }
        }
        else {
            removeToTotal(cache.unitMemo[currentPosition - 1]);
            if(cache.unitMemo[currentPosition + lastLength].value == DNEGINF || !_array._chunkCache){
                unitInfo unitResult = calculateUnit(_subarrayStartCoords,_subarrayEndCoords,_subarrayEndCoords[_nDims-1], cache);
                cache.unitMemo[currentPosition + lastLength].value = unitResult.value;
                cache.unitMemo[currentPosition + lastLength].cellCount = unitResult.cellCount;
            }
            addToTotal(cache.unitMemo[currentPosition + lastLength]);
        }

        finalizeIC();
    }

    Value const& PPTSChunkIterator::getItem()
    {
        if (!_hasCurrent)
            throw USER_EXCEPTION(SCIDB_SE_EXECUTION, SCIDB_LE_NO_CURRENT_ELEMENT);
        return _nextValue;
    }

    const Coordinates& PPTSChunkIterator::getPosition()
    {
        if(!_hasCurrent){
            _currPos[0] = -1;
        }
        return _currPos;
    }

    bool PPTSChunkIterator::setPosition(Coordinates const& pos)
    {
        for (size_t i = 0, n = _currPos.size(); i < n; i++)
        {
            if (pos[i] < _firstPos[i] || pos[i] > _lastPos[i])
            {
                return false;
            }
        }
        _currPos = pos;

        if (_emptyTagIterator.get() && !_emptyTagIterator->setPosition(_currPos))
        {
            return false;
        }

        if (_iterationMode & IGNORE_NULL_VALUES && _nextValue.isNull())
        {
            return false;
        }
        if (_iterationMode & IGNORE_DEFAULT_VALUES && _nextValue == _defaultValue)
        {
            return false;
        }
        return true;
    }

    bool PPTSChunkIterator::isEmpty() const
    {
        return false;
    }

    void PPTSChunkIterator::restart()
    {
        if (setPosition(_firstPos))
        {
            _hasCurrent = true;
            return;
        }
        ++(*this);
    }

    void PPTSChunkIterator::operator ++()
    {
        bool done = false;
        while (!done)
        {
            size_t nDims = _firstPos.size();
            for (size_t i = nDims-1; ++_currPos[i] > _lastPos[i]; i--)
            {
                if (i == 0)
                {
                    _hasCurrent = false;
                    return;
                }
                _currPos[i] = _firstPos[i];
            }

            if (_emptyTagIterator.get() && !_emptyTagIterator->setPosition(_currPos))
            {
                continue;
            }

            if (_iterationMode & IGNORE_NULL_VALUES && _nextValue.isNull())
            {
                continue;
            }
            if (_iterationMode & IGNORE_DEFAULT_VALUES && _nextValue == _defaultValue)
            {
                continue;
            }

            done = true;
            _hasCurrent = true;
        }
    }

    bool PPTSChunkIterator::end()
    {
        return !_hasCurrent;
    }

    ConstChunk const& PPTSChunkIterator::getChunk()
    {
        return _chunk;
    }

    PPTSChunk::PPTSChunk(PPTSArray const& arr, AttributeID attr)
            : _array(arr),
              _arrayIterator(NULL),
              _nDims(arr._desc.getDimensions().size()),
              _firstPos(_nDims),
              _lastPos(_nDims),
              _attrID(attr),
              _materialized(false),
              _mapper()
    {
        if (arr._desc.getEmptyBitmapAttribute() == 0 || attr!=arr._desc.getEmptyBitmapAttribute()->getId())
        {
            _aggregate = arr._aggregates[_attrID]->clone();
        }
    }

    position_t PPTSChunk::coord2pos_withOverlap_forSubarrayStartPosition(const Coordinates& coord) const
    {
        position_t pos(-1);
        if (_nDims == 1) {
            pos = coord[0] - _actualFirstPos[0];
        } else if (_nDims == 2) {
            pos = (coord[0] - _actualFirstPos[0])*_newChunkInterval_forSubarrayStartPosition[1] + (coord[1] - _actualFirstPos[1]);
        } else {
            pos = 0;
            for (size_t i = 0, n = _nDims; i < n; i++) {
                pos *= _newChunkInterval_forSubarrayStartPosition[i];
                pos += coord[i] - _actualFirstPos[i];
            }
        }
        return pos;
    }



    position_t PPTSChunk::coord2pos_withOverlap_forUnitStartPosition(const Coordinates& coord) const
    {
        position_t pos(-1);
        if (_nDims == 1) {
            pos = coord[0] - _actualFirstPos[0];
        } else if (_nDims == 2) {
            pos = (coord[0] - _realFirstPos[0])*_newChunkInterval_forUnitStartPosition[1] + (coord[1] - _realFirstPos[1]);
        } else {
            pos = 0;
            for (size_t i = 0, n = _nDims; i < n; i++) {
                pos *= _newChunkInterval_forUnitStartPosition[i];
                pos += coord[i] - _realFirstPos[i];
            }
        }
        return pos;
    }

    position_t PPTSChunk::coord2pos_withOverlap_EndToEnd(const Coordinates& coord) const
    {
        position_t pos(-1);
        if (_nDims == 1) {
            pos = coord[0] - _realFirstPos[0];
        } else if (_nDims == 2) {
            pos = (coord[0] - _realFirstPos[0])*_newChunkInterval_EndToEnd[1] + (coord[1] - _realFirstPos[1]);
        } else {
            pos = 0;
            for (size_t i = 0, n = _nDims; i < n; i++) {
                pos *= _newChunkInterval_EndToEnd[i];
                pos += coord[i] - _realFirstPos[i];
            }
        }
        return pos;
    }

    void PPTSChunk::setActualFirstLastPosition(Coordinates const& actualFirstPos,Coordinates const& actualLastPos){
        _actualFirstPos = actualFirstPos;
        _actualLastPos = actualLastPos;
    }
    void PPTSChunk::setRealFirstLastPosition(Coordinates const& realFirstPos,Coordinates const& realLastPos){
        _realFirstPos = realFirstPos;
    }

   void PPTSChunk::checkOnePartition_IC(Coordinates& leftBottom,
                                        Coordinates& rightTop,
                                        std::shared_ptr<PPTSChunkIterator>& chunkIterator,
                                        std::shared_ptr<Materials>& materials,
                                        chunkCache& cache)
   {
        size_t nDims = _array._window.size();
        Coordinates realLeftBottom;
        for(long i=0;i<nDims;i++){
            leftBottom[i] = std::max(leftBottom[i] - _array._window[i]._boundaries.second,_actualFirstPos[i]);
            rightTop[i] = std::min(rightTop[i] + _array._window[i]._boundaries.first,_actualLastPos[i]);
            realLeftBottom.push_back(leftBottom[i] - _array._window[i]._boundaries.first);
        }

        Coordinates current = leftBottom;
        current[nDims-1]--;
        bool managingList = true;

        chunkIterator->initializeManagedValue();
        chunkIterator->initializeCellCount();

        while(true){
            current[nDims-1]++;
            bool checkedAllSubarraysInThisPartition = false;
            for (long i = nDims - 1; i >= 0; i--) {
                if (current[i] > rightTop[i]) {
                    if (i == 0) {
                        checkedAllSubarraysInThisPartition = true;
                        break;
                    }
                    current[i] = leftBottom[i];
                    current[i - 1] += 1;

                    if (i == nDims - 1) {
                        managingList = true;
                        chunkIterator->initializeManagedValue();
                        chunkIterator->initializeCellCount();
                    }
                }
                else
                    break;
            }
            if(checkedAllSubarraysInThisPartition){
                managingList = true;
                break;
            }
            position_t currentPosition = coord2pos_withOverlap_forUnitStartPosition(current);
            position_t currPosition_forSubarray = coord2pos_withOverlap_forSubarrayStartPosition(current);

            if(cache.isChecked[currPosition_forSubarray]){
                continue;
            }

            chunkIterator->calculateSubarrayScoreIC(current,materials,realLeftBottom,cache,managingList);
            cache.isChecked[currPosition_forSubarray] = true;

            Value calculatedValue = chunkIterator->getNextValue();

            if(calculatedValue.getDouble() == 0)
                continue;

            double ithScore = materials->localAnswer.value;
            progressiveTopKCandidate newCandidate;


            Coordinates _leftBottom;
            Coordinates _rightTop;

            for(int j=0;j<nDims;j++) {
                _leftBottom.push_back(current[j] - _array._window[j]._boundaries.first);
                _rightTop.push_back(current[j] + _array._window[j]._boundaries.second);
            }

            SpatialRange candidateRange = SpatialRange(_leftBottom,_rightTop);
            newCandidate = progressiveTopKCandidate(current, calculatedValue.getDouble(), chunkIterator->getNextCellCount(),candidateRange);
            materials->_backup.push(newCandidate);

            if ((calculatedValue.getDouble() >= ithScore) && calculatedValue.getDouble() != DNEGINF) {
                bool newCandidateIsOK = true;
                if (_array._disjoint) {
                    for (auto it = materials->_finalTopK.begin(); it != materials->_finalTopK.end(); it++) {
                        if (it->range.intersects(newCandidate.range)) {
                            newCandidateIsOK = false;
                            break;
                        }
                    }
                }

                if (newCandidateIsOK) {
                    materials->localAnswer = newCandidate;
                }
            }
        }
    }

    Array const& PPTSChunk::getArray() const
    {
        return _array;
    }

    const ArrayDesc& PPTSChunk::getArrayDesc() const
    {
        return _array._desc;
    }

    const AttributeDesc& PPTSChunk::getAttributeDesc() const
    {
        return _array._desc.getAttributes()[_attrID];
    }

    Coordinates const& PPTSChunk::getFirstPosition(bool withOverlap) const
    {
        return _firstPos;
    }

    Coordinates const& PPTSChunk::getLastPosition(bool withOverlap) const
    {
        return _lastPos;
    }

    std::shared_ptr<ConstChunkIterator> PPTSChunk::getConstIterator(int iterationMode) const
    {
        SCIDB_ASSERT(( NULL != _arrayIterator ));
        ConstChunk const& inputChunk = _arrayIterator->iterator->getChunk();
        if (_array.getArrayDesc().getEmptyBitmapAttribute() && _attrID == _array.getArrayDesc().getEmptyBitmapAttribute()->getId())
        {
            return inputChunk.getConstIterator((iterationMode & ~ChunkIterator::INTENDED_TILE_MODE) | ChunkIterator::IGNORE_OVERLAPS);
        }
        return std::shared_ptr<ConstChunkIterator>(new PPTSChunkIterator(_array, *_arrayIterator, *this, iterationMode));
    }

    CompressorType PPTSChunk::getCompressionMethod() const
    {
        return _array._desc.getAttributes()[_attrID].getDefaultCompressionMethod();
    }

    inline uint64_t PPTSChunk::coord2pos(Coordinates const& coord) const
    {
        SCIDB_ASSERT(_materialized);
        position_t pos = _mapper->coord2pos(coord);
        SCIDB_ASSERT(pos >= 0);
        return pos;
    }

    inline void PPTSChunk::pos2coord(uint64_t pos, Coordinates& coord) const
    {
        SCIDB_ASSERT(_materialized);
        _mapper->pos2coord(pos, coord);
    }

    inline bool PPTSChunk::valueIsNeededForAggregate ( const Value & val, const ConstChunk & inputChunk ) const
    {
        return (!((val.isNull() && _aggregate->ignoreNulls()) ||
                  (isDefaultFor(val,inputChunk.getAttributeDesc().getType()) && _aggregate->ignoreZeroes())));
    }

    // This code assumes that a chunk is loaded into memory using c++std::map.
    // In the case of dense chunks, this code can be optimized by using c++std::vector.
    void PPTSChunk::materialize() {
        _materialized = true;
        _stateMap.clear();
        _inputMap.clear();

        int64_t nInputElements = 0;
        int64_t nResultElements = 0;

        int iterMode = ChunkIterator::IGNORE_EMPTY_CELLS;
        ConstChunk const &chunk = _arrayIterator->iterator->getChunk();
        _mapper = std::shared_ptr<CoordinatesMapper>(new CoordinatesMapper(chunk));
        Coordinates const &firstPos = chunk.getFirstPosition(false);
        Coordinates const &lastPos = chunk.getLastPosition(false);

        std::shared_ptr<ConstChunkIterator> chunkIter = chunk.getConstIterator(iterMode);
        while (!chunkIter->end()) {
            Coordinates const &currPos = chunkIter->getPosition();
            Value const &currVal = chunkIter->getItem();
            uint64_t pos = _mapper->coord2pos(currPos);

            bool insideOverlap = true;
            for (size_t i = 0; i < _nDims; i++) {
                if (currPos[i] < firstPos[i] || currPos[i] > lastPos[i]) {
                    insideOverlap = false;
                    break;
                }
            }

            if (insideOverlap) {
                _stateMap[pos] = true;
                nResultElements += 1;
            }

            if (valueIsNeededForAggregate(currVal, chunk)) {
                _inputMap[pos] = currVal;
                nInputElements += 1;
            }
            ++(*chunkIter);
        }
    }

    void PPTSChunk::setPosition(PPTSArrayIterator* iterator, Coordinates const& pos)
    {
        _arrayIterator = iterator;
        _firstPos = pos;
        Dimensions const& dims = _array._desc.getDimensions();

        for (size_t i = 0, n = dims.size(); i < n; i++) {
            _lastPos[i] = _firstPos[i] + dims[i].getChunkInterval() - 1;
            if (_lastPos[i] > dims[i].getEndMax())
            {
                _lastPos[i] = dims[i].getEndMax();
            }
        }
        _materialized = false;
        if (_aggregate.get() == 0)
        {
            return;
        }
    }

    void PPTSChunk::clearNewChunkInterval() {
        _newChunkInterval_forSubarrayStartPosition.clear();
        _newChunkInterval_forUnitStartPosition.clear();
    }

    void PPTSChunk::set_newChunkInterval_forSubarrayStartPosition(Coordinates & temp) {
        _newChunkInterval_forSubarrayStartPosition = temp;
    }

    void PPTSChunk::set_newChunkInterval_forUnitStartPosition(Coordinates & temp) {
        _newChunkInterval_forUnitStartPosition = temp;
    }

    void PPTSChunk::set_newChunkInterval_EndToEnd(Coordinates & temp) {
        _newChunkInterval_EndToEnd = temp;
    }

    position_t PPTSChunk::coord2pos_withOverlap(const Coordinates& coord) const
    {
        position_t pos(-1);

        if (_nDims == 1) {
            pos = coord[0] - _realFirstPos[0];
        } else if (_nDims == 2) {
            pos = (coord[0] - _realFirstPos[0])*_newChunkInterval_EndToEnd[1] + (coord[1] - _realFirstPos[1]);
        } else {
            pos = 0;
            for (size_t i = 0, n = _nDims; i < n; i++) {
                pos *= _newChunkInterval_EndToEnd[i];
                pos += coord[i] - _realFirstPos[i];
            }
        }
        return pos;
    }

    PPTSArrayIterator::PPTSArrayIterator(PPTSArray const& arr, AttributeID attrID, AttributeID input,  std::string const& method)
            : array(arr),
              iterator(arr._inputArray->getConstIterator(input)),
              currPos(arr._dimensions.size()),
              chunk(arr, attrID),
              _method(method)
    {
        restart();
    }

    PPTSChunk& PPTSArrayIterator::getProgressiveChunk()
    {
        if (!chunkInitialized)
        {
            assert(iterator->getPosition() == currPos);
            chunk.setPosition(this, currPos);
            chunkInitialized = true;
        }
        chunk.clearNewChunkInterval();
        return chunk;
    }

    void PPTSArrayIterator::operator ++()
    {
        if (!hasCurrent)
            throw USER_EXCEPTION(SCIDB_SE_EXECUTION, SCIDB_LE_NO_CURRENT_ELEMENT);
        chunkInitialized = false;
        ++(*iterator);
        hasCurrent = !iterator->end();
        if (hasCurrent)
        {
            currPos = iterator->getPosition();
        }
    }

    bool PPTSArrayIterator::end()
    {
        return !hasCurrent;
    }

    Coordinates const& PPTSArrayIterator::getPosition()
    {
        if (!hasCurrent)
            throw USER_EXCEPTION(SCIDB_SE_EXECUTION, SCIDB_LE_NO_CURRENT_ELEMENT);
        return currPos;
    }

    bool PPTSArrayIterator::setPosition(Coordinates const& pos)
    {
        chunkInitialized = false;
        if (!iterator->setPosition(pos))
        {
            return hasCurrent = false;
        }
        currPos = iterator->getPosition();
        return hasCurrent = true;
    }

    void PPTSArrayIterator::restart()
    {
        chunkInitialized = false;
        iterator->restart();
        hasCurrent = !iterator->end();
        if (hasCurrent)
        {
            currPos = iterator->getPosition();
        }
    }

    ConstChunk const& PPTSArrayIterator::getChunk()
    {
        if (!chunkInitialized)
        {
            assert(iterator->getPosition() == currPos);
            chunk.setPosition(this, currPos);
            chunkInitialized = true;
        }
        return chunk;
    }

    PPTSArray::PPTSArray(ArrayDesc const& desc, std::shared_ptr<Array> const& inputArray,
                     std::vector<WindowBoundaries> const& window, std::vector<AttributeID> const& inputAttrIDs, std::vector<AggregatePtr> const& aggregates,
                     std::string const& method,int const topK, bool const disjoint):
            _desc(desc),
            _inputDesc(inputArray->getArrayDesc()),
            _window(window),
            _dimensions(_desc.getDimensions()),
            _inputArray(inputArray),
            _inputAttrIDs(inputAttrIDs),
            _aggregates(aggregates),
            _method(method),
            _topK(topK),
            _disjoint(disjoint),
            _chunkCache(chunkCache)
    {
        nDims = _dimensions.size();
        for(size_t i=0;i<nDims;i++){
            lastPosition.push_back(_dimensions[i].getEndMax());
            firstPosition.push_back(_dimensions[i].getStartMin());
        }
        setAggregateType(_aggregates[0]->getName());
    }

    void PPTSArray::setAggregateType(const std::string& aggregateName)
    {
        if (aggregateName == "sum")
            aggregateType = 0;
        else if (aggregateName == "avg")
            aggregateType = 1;
        else if (aggregateName == "min")
            aggregateType = 2;
        else if (aggregateName == "max")
            aggregateType = 3;
    }

    int PPTSArray::getAggregateType() const
    {
        return aggregateType;
    }

    ArrayDesc const& PPTSArray::getArrayDesc() const
    {
        return _desc;
    }

    Coordinates const& PPTSArray::getLastPosition() const
    {
        return lastPosition;
    }

    Coordinates const& PPTSArray::getFirstPosition() const
    {
        return firstPosition;
    }

    int64_t PPTSArray::getChunkOverlap(int dimensionNum) const
    {
        return _dimensions[dimensionNum].getChunkOverlap();
    }

    std::shared_ptr<ConstArrayIterator> PPTSArray::getConstIterator(AttributeID attr) const
    {
        if (_desc.getEmptyBitmapAttribute() && attr == _desc.getEmptyBitmapAttribute()->getId())
        {
            return std::shared_ptr<ConstArrayIterator>(new PPTSArrayIterator(*this, attr, _inputArray->getArrayDesc().getEmptyBitmapAttribute()->getId(), _method));
        }

        return std::shared_ptr<ConstArrayIterator>(new PPTSArrayIterator(*this, attr, _inputAttrIDs[attr], _method ));
    }

    std::shared_ptr<PPTSArrayIterator> PPTSArray::getProgressiveArrayIterator(AttributeID attr)
    {
        if (_desc.getEmptyBitmapAttribute() && attr == _desc.getEmptyBitmapAttribute()->getId())
        {
            return std::shared_ptr<PPTSArrayIterator>(new PPTSArrayIterator(*this, attr, _inputArray->getArrayDesc().getEmptyBitmapAttribute()->getId(), _method));
        }

        return std::shared_ptr<PPTSArrayIterator>(new PPTSArrayIterator(*this, attr, _inputAttrIDs[attr], _method));
    }

    Materials::Materials(size_t _nDims)
    {
        localAnswer = progressiveTopKCandidate(_nDims);
    }

    progressiveTopKCandidate::progressiveTopKCandidate(int nDims)
    {
        coordinate.clear();
        for(int i=0;i<nDims;i++){
            coordinate.push_back(-1);
        }
        value = -1;
        Coordinates lowPosition;
        Coordinates highPosition;
        for(int i=0;i<nDims;i++){
            lowPosition.push_back(-2);
            highPosition.push_back(-1);
        }
        cellCount = 0;
        range = SpatialRange(lowPosition,highPosition);
    }

    progressiveTopKCandidate::progressiveTopKCandidate(Coordinates const& _coordinate, double _value, const int _cellCount)
            :coordinate(_coordinate),
             value(_value),
             cellCount(_cellCount)
    {
        Coordinates lowPosition;
        Coordinates highPosition;
        for(size_t i=0;i < _coordinate.size();i++){
            lowPosition.push_back(-2);
            highPosition.push_back(-1);
        }

        range = SpatialRange(lowPosition,highPosition);
    }

    progressiveTopKCandidate::progressiveTopKCandidate(const Coordinates  &_coordinate,double _value, const int _cellCount,const SpatialRange  &_range)
            :coordinate(_coordinate),
             value(_value),
             cellCount(_cellCount),
             range(_range){}

    progressiveTopKCandidate::progressiveTopKCandidate(std::shared_ptr<SharedBuffer> const &candidate, int numOfDims)
    {
        coordinate.clear();
        if (candidate->getSize() != getMarshalledSize(numOfDims))
        {
            for(int i=0;i<numOfDims;i++){
                coordinate.push_back(-1);
            }
            value = -1;
            cellCount = 0;
            Coordinates lowPosition;
            Coordinates highPosition;
            for(int i=0;i<numOfDims;i++){
                lowPosition.push_back(-2);
                highPosition.push_back(-1);
            }

            range = SpatialRange(lowPosition,highPosition);
            return;
        }

        int64_t* ptr = static_cast<int64_t*> (candidate->getData());

        coordinate.reserve(numOfDims);
        for(int i=0;i<numOfDims;i++){
            coordinate.push_back(*ptr);
            ++ptr;
        }

        double* ptrDouble = reinterpret_cast<double*>(ptr);
        value = *ptrDouble;
        ++ptrDouble;

        int* ptrInt = reinterpret_cast<int*>(ptrDouble);
        cellCount = (*ptrInt);
        ++ptrInt;

        range = SpatialRange();
        ptr = reinterpret_cast<int64_t*>(ptrInt);
        for(size_t i=0;i<coordinate.size();i++){
            range._low.push_back(*ptr);
            ++ptr;
        }
        for(size_t i=0;i<coordinate.size();i++){
            range._high.push_back(*ptr);
            ++ptr;
        }
    }

    std::shared_ptr<SharedBuffer> progressiveTopKCandidate::marshall(const int numOfDims)
    {
        std::shared_ptr <SharedBuffer> result (new MemoryBuffer(NULL, getMarshalledSize(numOfDims)));
        int64_t* ptr = static_cast<int64_t*> (result->getData());

        for(size_t i=0;i<coordinate.size();i++){
            *ptr = coordinate[i];
            ++ptr;
        }

        double* ptrDouble = reinterpret_cast<double*>(ptr);
        *ptrDouble = value;
        ++ptrDouble;

        int* ptrInt = reinterpret_cast<int*>(ptrDouble);
        *ptrInt = cellCount;
        ++ptrInt;

        ptr = reinterpret_cast<int64_t*>(ptrInt);
        for(size_t i=0;i<coordinate.size();i++){
            *ptr =  range._low[i];
            ++ptr;
        }
        for(size_t i=0;i<coordinate.size();i++){
            *ptr = range._high[i];
            ++ptr;
        }

        return result;
    }

    bool progressiveTopKCandidate::operator<(const progressiveTopKCandidate &other) const
    {
        return (value < other.value);
    }

    bool progressiveTopKCandidate::operator>(const progressiveTopKCandidate &other) const
    {
        return (value > other.value);
    }

    bool progressiveTopKCandidate::operator == (const progressiveTopKCandidate& other)
    {
        size_t nDims = coordinate.size();
        for(size_t i=0 ; i < nDims ; i++){
            if(coordinate[i] != other.coordinate[i])
                return false;
        }
        return true;
    }

    bool progressiveTopKCandidate::equal(const progressiveTopKCandidate& other)
    {
        size_t nDims = coordinate.size();
        for(size_t i=0 ; i < nDims ; i++){
            if(coordinate[i] != other.coordinate[i])
                return false;
        }
        return true;
    }

    size_t progressiveTopKCandidate::getMarshalledSize(const int numOfDims)
    {
        return 3*numOfDims*sizeof(int64_t) + sizeof(double) + sizeof(int);
    }

    unitInfo::unitInfo(double _value, int _cellCount)
    {
        value = _value;
        cellCount = _cellCount;
    }

    double Information::getMaxScore() {
        return maxScore;
    }

    std::vector<unitInfo> &Information::getUnitMemo()
    {
        return unitMemo;
    }

    void Information::setLocalTopKIsEmpty(bool localTopKIsEmpty)
    {
        this->localTopKIsEmpty = localTopKIsEmpty;
    }

    WindowBoundaries::WindowBoundaries()
    {
        _boundaries.first = _boundaries.second = 0;
    }

    WindowBoundaries::WindowBoundaries(Coordinate preceding, Coordinate following)
    {
        SCIDB_ASSERT(preceding >= 0);
        SCIDB_ASSERT(following >= 0);
        _boundaries.first = preceding;
        _boundaries.second = following;
    }

    partition::partition(size_t nDims)
    {
        for(size_t i=0;i<nDims;i++){
            leftBottom.push_back(-1);
            rightTop.push_back(-1);
        }
        representative = -1;
    }

    partition::partition(Coordinates& _leftBottom, Coordinates& _rightTop, double _representative):
            leftBottom(_leftBottom),
            rightTop(_rightTop),
            representative(_representative)
    {}

    bool partition::operator> (const partition &other) const
    {
        return (representative > other.representative);
    }

    bool partition::operator< (const partition& other) const
    {
        return (representative < other.representative);
    }
}