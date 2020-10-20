#include <string>
#include <utility>
#include <vector>
#include <queue>
#include <map>
#include <limits>
#include <list>
#include <functional>
#include <algorithm>
#include <hash_map>
#include <util/RegionCoordinatesIterator.h>
#include <array/DelegateArray.h>
#include <array/Metadata.h>
#include <query/FunctionDescription.h>
#include <query/Expression.h>
#include <query/Aggregate.h>
#include <array/MemArray.h>
#include <util/SpatialType.h>
#include <array/Coordinate.h>

namespace scidb
{
    const double DNEGINF = std::numeric_limits<double>::min();

    struct ChunkInformations;
    struct unitInfo;
    struct chunkCache;
    struct ChunkCaches;
    struct partition;
    struct progressiveTopKCandidate;
    struct Materials;
    struct DescendingOrder;
    struct AscendingOrder;
    struct RemoveDuplicate;
    class Information;
    struct WindowBoundaries;
    struct struct_NextPartitionUBS;
    class PPTSArray;
    class PPTSArrayIterator;
    class PPTSChunk;
    class PPTSChunkIterator;
    class Information;

    struct ChunkInformations{
        std::vector<Coordinates> chunkStartCoordinates;
        std::vector<Coordinates> actualFirstPos;
        std::vector<Coordinates> actualLastPos;
        std::vector<Coordinates> realLastPos;
        std::vector<Coordinates> realFirstPos;
        std::vector<unsigned long> totalSubarrays;
        std::vector<unsigned long> totalCells;
        std::vector<unsigned long> totalUnits;
        std::vector<Coordinates> newChunkInterval_forUnitStartPosition;
        std::vector<Coordinates> newChunkInterval_forSubarrayStartPosition;
        std::vector<Coordinates> newChunkInterval_EndToEnd;
        unsigned long max_TotalSubarrays = 0;
        unsigned long max_TotalUnits = 0;
        unsigned long max_TotalCells = 0;
        std::vector<size_t> nextPartitionPosition;
        std::vector<double> nextPartitionUBS;
        std::vector<double> maxDensities;
        std::vector<std::vector<std::string>> pStartingCells;
        std::vector<std::vector<std::string>> pEndingCells;
        std::vector<std::vector<double>> pMaxs;
    };

    struct unitInfo{
        double value = 0;
        int cellCount = 0;
        unitInfo(){
            value = 0;
            cellCount = 0;
        }
        unitInfo(double _value,int _cellCount);
    };

    struct chunkCache{
        std::vector<bool> isChecked;
        std::vector<unitInfo> unitMemo;

        // This code assumes that a chunk is loaded into memory using c++std::map.
        // In the case of dense chunks, this code can be optimized by using c++std::vector.
        std::map<uint64_t, double> materializedChunk;

        chunkCache() = default;
        chunkCache(unsigned long max_TotalSubarrays,unsigned long max_TotalUnits,
                   unsigned long max_TotalCells) {
            unitInfo tempUnitInfo(DNEGINF, 0);
            isChecked = std::vector<bool>(max_TotalSubarrays, false);
            unitMemo = std::vector<unitInfo>(max_TotalUnits, tempUnitInfo);
        }
    };

    // Keep several chunks on memory by LRU policy to reduce repetitive chunk I/O.
    struct ChunkCaches{
        std::unordered_map<int,chunkCache> caches;
        std::list<int> LRU;
        chunkCache newCache;
        int maximalCounts;

        ChunkCaches(int _maximalCounts, unsigned long max_TotalSubarrays,
                    unsigned long max_TotalUnits, unsigned long max_TotalCells){
            maximalCounts = _maximalCounts;
            newCache = chunkCache(max_TotalSubarrays,max_TotalUnits,max_TotalCells);
        }

        bool cacheUpdate(int chunkID){
            if(caches.find(chunkID) != caches.end()){
                for(auto it = LRU.begin(); it != LRU.end(); it++){
                    if(*it == chunkID){
                        LRU.erase(it);
                        break;
                    }
                }
                LRU.push_front(chunkID);
                return false;
            }
            else{
                if(caches.size() == maximalCounts) {
                    int eraseCandidate = LRU.back();
                    caches[eraseCandidate].materializedChunk.clear();
                    LRU.pop_back();
                    caches.erase(eraseCandidate);
                }

                LRU.push_front(chunkID);
                caches.insert({chunkID,newCache});
                return true;
            }
        }
    };

    struct partition{
        Coordinates leftBottom;
        Coordinates rightTop;
        double representative;

        partition(size_t nDims);
        partition(Coordinates& _leftBottom,Coordinates& _rightTop,double _representative);

        bool operator < (const partition& other) const;
        bool operator > (const partition& other) const;
    };

    // a single subarray
    struct progressiveTopKCandidate{
        Coordinates coordinate; //starting cell's coordinate
        double value = 0; //score
        int cellCount = 0;
        SpatialRange range;

        progressiveTopKCandidate() = default;
        explicit progressiveTopKCandidate(int nDims);
        progressiveTopKCandidate(const Coordinates & _coordinate,double _value, const int _cellCount);
        progressiveTopKCandidate(const Coordinates & _coordinate,double _value, const int _cellCount,const SpatialRange & _range);
        progressiveTopKCandidate(std::shared_ptr<SharedBuffer> const& candidate,int numOfDims);

        size_t getMarshalledSize(const int numOfDims);
        std::shared_ptr<SharedBuffer> marshall(const int numOfDims);
        bool operator < (const progressiveTopKCandidate& other) const;
        bool operator > (const progressiveTopKCandidate& other) const;
        bool operator == (const progressiveTopKCandidate& other);
        bool equal(const progressiveTopKCandidate& other);
    };

    struct Materials{
        std::list<progressiveTopKCandidate> _finalTopK;
        std::priority_queue<progressiveTopKCandidate,std::vector<progressiveTopKCandidate>,std::less<progressiveTopKCandidate> > _backup;
        progressiveTopKCandidate localAnswer;

        Materials() = default;
        explicit Materials(size_t _nDims);
    };

    struct DescendingOrder {
        inline bool operator()(const progressiveTopKCandidate& struct1, const progressiveTopKCandidate& struct2){
            return (struct1.value > struct2.value);
        }
    };

    struct AscendingOrder {
        inline bool operator()(const progressiveTopKCandidate& struct1, const progressiveTopKCandidate& struct2){
            return (struct1.value < struct2.value);
        }
    };

    struct RemoveDuplicate {
        inline bool operator()(progressiveTopKCandidate const& struct1, progressiveTopKCandidate const& struct2){
            for(size_t i=0;i<struct1.coordinate.size();i++){
                if(struct1.coordinate[i] != struct2.coordinate[i])
                    return false;
            }
            return true;
        }
    };

    class Information
    {

    public:
        double getMaxScore();
        void setLocalTopKIsEmpty(bool localTopKIsEmpty);
        std::vector<unitInfo>& getUnitMemo();
    private:
        double representative;
        double maxScore;
        bool localTopKIsEmpty;
        std::vector<unitInfo> unitMemo;
        std::vector<bool> isChecked;
    };

    struct WindowBoundaries
    {
        WindowBoundaries();
        WindowBoundaries(Coordinate preceding, Coordinate following);
        std::pair<Coordinate, Coordinate> _boundaries;
    };

    class PPTSChunk : public ConstChunk
    {
        friend class PPTSChunkIterator;

    public:
        PPTSChunk(PPTSArray const& array, AttributeID attrID);
        virtual const ArrayDesc& getArrayDesc() const;
        virtual const AttributeDesc& getAttributeDesc() const;
        virtual Coordinates const& getFirstPosition(bool withOverlap) const;
        virtual Coordinates const& getLastPosition(bool withOverlap) const;
        virtual std::shared_ptr<ConstChunkIterator> getConstIterator(int iterationMode) const;
        virtual CompressorType getCompressionMethod() const;
        virtual Array const& getArray() const;
        void setPosition(PPTSArrayIterator* iterator, Coordinates const& pos);
        void checkOnePartition_IC(Coordinates&,Coordinates&,std::shared_ptr<PPTSChunkIterator>&,std::shared_ptr<Materials>&,chunkCache& cache);
        void setActualFirstLastPosition(Coordinates const& actualFirstPos,Coordinates const& actualLastPos);
        void setRealFirstLastPosition(Coordinates const& realFirstPos,Coordinates const& realLastPos);
        void clearNewChunkInterval();
        position_t coord2pos_withOverlap_forSubarrayStartPosition(const Coordinates& coord) const;
        position_t coord2pos_withOverlap_forUnitStartPosition(const Coordinates& coord) const;
        position_t coord2pos_withOverlap_EndToEnd(const Coordinates& coord) const;
        void set_newChunkInterval_forSubarrayStartPosition(Coordinates&);
        void set_newChunkInterval_forUnitStartPosition(Coordinates&);
        void set_newChunkInterval_EndToEnd(Coordinates & temp);
        Coordinate theMostMinorDimSize;
        position_t coord2pos_withOverlap(const Coordinates& coord) const;
    private:
        void materialize();
        void pos2coord(uint64_t pos, Coordinates& coord) const;
        uint64_t coord2pos(const Coordinates& coord) const;
        inline bool valueIsNeededForAggregate (const Value & val, const ConstChunk & inputChunk) const;
        PPTSArray const& _array;
        PPTSArrayIterator const* _arrayIterator;
        size_t _nDims;
        Coordinates _firstPos;
        Coordinates _lastPos;
        Coordinates _actualFirstPos;
        Coordinates _actualLastPos;
        Coordinates _realFirstPos;
        AttributeID _attrID;
        AggregatePtr _aggregate;
        std::map<uint64_t, bool> _stateMap;
        std::map<uint64_t, Value> _inputMap;
        bool _materialized;
        std::shared_ptr<CoordinatesMapper> _mapper;
        inline bool isMaterialized() const { return _materialized; };
        Value _nextValue;
        Coordinates _newChunkInterval_forSubarrayStartPosition;
        Coordinates _newChunkInterval_forUnitStartPosition;
        Coordinates _newChunkInterval_EndToEnd;
    };

    class PPTSChunkIterator : public ConstChunkIterator
    {
        friend class PPTSChunk;

    public:
        virtual int getMode() const;
        virtual bool isEmpty() const;
        virtual Value const& getItem();
        virtual void operator++();
        virtual bool end();
        virtual const Coordinates& getPosition();
        virtual bool setPosition(Coordinates const &pos);
        virtual void restart();
        ConstChunk const &getChunk();
        PPTSChunkIterator(PPTSArray const& array,PPTSArrayIterator const& arrayIterator, PPTSChunk const& chunk, int mode);
        void addToTotal(unitInfo& unitResult);
        void removeToTotal(unitInfo& unitResult);
        void calculateSubarrayScoreIC(Coordinates const&, std::shared_ptr<Materials>&, Coordinates const&,chunkCache& cache,bool& managingList);
        const std::shared_ptr<ConstChunkIterator>& getInputIterator() const;
        std::shared_ptr<ConstChunkIterator> _inputIterator;
        Information info;
    private:
        bool attributeDefaultIsSameAsTypeDefault() const;
        PPTSArray const& _array;
        PPTSChunk const& _chunk;
        Coordinates const& _firstPos;
        Coordinates const& _lastPos;
        Coordinates _currPos;
        bool _hasCurrent;
        AttributeID _attrID;
        AggregatePtr _aggregate;
        Value _defaultValue;
        size_t _nDims;
        int _iterationMode;
        std::shared_ptr<ConstArrayIterator> _emptyTagArrayIterator;
        std::shared_ptr<ConstChunkIterator> _emptyTagIterator;
        Value _nextValue;
        double sumInManagingList = 0;
        int countInManagingList = 0;
        Coordinates _subarrayStartCoords;
        Coordinates _subarrayEndCoords;
        int _windowSize;

        unitInfo calculateUnit(Coordinates, Coordinates, int, chunkCache&);
        void finalizeIC();
        void initialize(unitInfo &unitResult, std::shared_ptr<Materials>& materials);
    };

    class PPTSArrayIterator : public ConstArrayIterator
    {
        friend class PPTSChunk;
        friend class PPTSChunkIterator;
    public:
        virtual ConstChunk const& getChunk();
        virtual bool end();
        virtual void operator ++();
        virtual Coordinates const& getPosition();
        virtual bool setPosition(Coordinates const& pos);
        virtual void restart();

        PPTSArrayIterator(PPTSArray const& array, AttributeID id, AttributeID input, std::string const& method);
        PPTSChunk& getProgressiveChunk();
    private:
        PPTSArray const& array;
        std::shared_ptr<ConstArrayIterator> iterator;
        Coordinates currPos;
        bool hasCurrent;
        PPTSChunk chunk;
        bool chunkInitialized;
        std::string _method;
    };

    class PPTSArray : public Array
    {
        friend class PPTSArrayIterator;
        friend class PPTSChunkIterator;
        friend class PPTSChunk;

    public:
        virtual ArrayDesc const &getArrayDesc() const;
        virtual std::shared_ptr<ConstArrayIterator> getConstIterator(AttributeID attr) const;
        std::shared_ptr<PPTSArrayIterator> getProgressiveArrayIterator(AttributeID attr);
        PPTSArray(ArrayDesc const &desc,
                std::shared_ptr<Array> const &inputArray,
                std::vector<WindowBoundaries> const &window,
                std::vector<AttributeID> const &inputAttrIDs,
                std::vector<AggregatePtr> const &aggregates,
                std::string const &method,
                int const topK,
                bool const disjoint);
        Coordinates const& getLastPosition() const;
        Coordinates const& getFirstPosition() const;
        int64_t getChunkOverlap(int dimensionNum) const;
        int getAggregateType() const;

        bool const _disjoint;
        bool const _chunkCache;
        bool const _materialize;
        ChunkInformations chunkInfos;
        int currentChunk;
        int unitSize;
        int subarraySize;
        Coordinate subarray_theMostMinorDimSize;
        Coordinate subarray_theNextMinorDimSize;
        size_t nDims;
    private:
        ArrayDesc _desc;
        ArrayDesc _inputDesc;
        std::vector<WindowBoundaries> _window;
        Dimensions _dimensions;
        std::shared_ptr<Array> _inputArray;
        std::vector<AttributeID> _inputAttrIDs;
        std::vector<AggregatePtr> _aggregates;
        std::string _method;
        Coordinates lastPosition;
        Coordinates firstPosition;
        int aggregateType;
        int const _topK;

        void setAggregateType(const std::string &aggregateName);
    };

    struct struct_NextPartitionUBS
    {
        int chunkNum;
        double nextPartitionUBS;

        struct_NextPartitionUBS(int chunkNum, double nextPartitionUBS){
            this->chunkNum = chunkNum;
            this->nextPartitionUBS = nextPartitionUBS;
        }

        bool operator<(const struct_NextPartitionUBS &other) const {
            return (nextPartitionUBS < other.nextPartitionUBS);
        }

        bool operator>(const struct_NextPartitionUBS &other) const {
            return (nextPartitionUBS > other.nextPartitionUBS);
        }
    };
}
