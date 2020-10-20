#include <memory>
#include <log4cxx/logger.h>
#include "query/Operator.h"
#include "system/Exceptions.h"
#include "query/LogicalExpression.h"
#include "PPTS.h"

namespace scidb {

    using namespace std;

    class LogicalPPTS : public LogicalOperator {
    public:

        LogicalPPTS(const std::string &logicalName, const std::string &alias) :
                LogicalOperator(logicalName, alias) {
            ADD_PARAM_INPUT(); // input array
            ADD_PARAM_INPUT(); // partition array
            ADD_PARAM_INPUT(); // maxdensity array
            ADD_PARAM_VARIES(); // additional inputs
        }

        std::vector<std::shared_ptr<OperatorParamPlaceholder> >
        nextVaryParamPlaceholder(const std::vector<ArrayDesc> &schemas)
        {
            std::vector<std::shared_ptr<OperatorParamPlaceholder> > res;
            if (_parameters.size() < schemas[0].getDimensions().size() * 2) { // subarray size
                res.push_back(PARAM_CONSTANT("int64"));
            } else if (_parameters.size() == schemas[0].getDimensions().size() * 2) { // K
                res.push_back(PARAM_CONSTANT("int32"));
            } else if (_parameters.size() == schemas[0].getDimensions().size() * 2 + 1) { // Scoring function
                res.push_back(PARAM_AGGREGATE_CALL());
            }
            else if (_parameters.size() == schemas[0].getDimensions().size() * 2 + 2) { // disjoint or overlap-allowing
                res.push_back(PARAM_CONSTANT(TID_STRING));
            }
            else if (_parameters.size() == schemas[0].getDimensions().size() * 2 + 3) { // maximal density estimation
                res.push_back(PARAM_CONSTANT(TID_STRING));
            }
            else{
                res.push_back(END_OF_VARIES_PARAMS());
            }
            return res;
        }

        ArrayDesc inferSchema(std::vector<ArrayDesc> schemas, std::shared_ptr<Query> query) {
            SCIDB_ASSERT(schemas.size() == 3);

            ArrayDesc const &desc = schemas[0];
            ArrayDesc const &desc2 = schemas[1];

            size_t nDims = desc.getDimensions().size();
            std::vector<WindowBoundaries> window(nDims);
            size_t windowSize = 1;
            for (size_t i = 0, size = nDims * 2, boundaryNo = 0; i < size; i += 2, ++boundaryNo) {
                int64_t boundaryLower =
                        evaluate(((std::shared_ptr<OperatorParamLogicalExpression> &) _parameters[i])->getExpression(), TID_INT64).getInt64();

                if (boundaryLower < 0)
                    throw USER_QUERY_EXCEPTION(SCIDB_SE_INFER_SCHEMA, SCIDB_LE_OP_WINDOW_ERROR3,
                                               _parameters[i]->getParsingContext());

                int64_t boundaryUpper =
                        evaluate(((std::shared_ptr<OperatorParamLogicalExpression> &) _parameters[i+1])->getExpression(), TID_INT64).getInt64();

                if (boundaryUpper < 0)
                    throw USER_QUERY_EXCEPTION(SCIDB_SE_INFER_SCHEMA, SCIDB_LE_OP_WINDOW_ERROR3,
                                               _parameters[i]->getParsingContext());

                window[boundaryNo] = WindowBoundaries(boundaryLower, boundaryUpper);
                windowSize *= window[boundaryNo]._boundaries.second + window[boundaryNo]._boundaries.first + 1;

            }
            if (windowSize <= 1)
                throw USER_QUERY_EXCEPTION(SCIDB_SE_INFER_SCHEMA, SCIDB_LE_OP_WINDOW_ERROR4,
                                           _parameters[0]->getParsingContext());

            int64_t k =
                    evaluate(((std::shared_ptr<OperatorParamLogicalExpression> &) _parameters[nDims * 2])->getExpression(), TID_INT64).getInt64();

            Attributes outputAttributes;
            outputAttributes.push_back(AttributeDesc(0, "coordinate", TID_STRING, 0, CompressorType::NONE));
            outputAttributes.push_back(AttributeDesc(1, "score", TID_DOUBLE, 0, CompressorType::NONE));
            outputAttributes = addEmptyTagAttribute(outputAttributes);

            Dimensions outputDimensions;
            outputDimensions.push_back(DimensionDesc("i", 0, k - 1, k, 0));

            return ArrayDesc("topK", outputAttributes, outputDimensions, defaultPartitioning(), query->getDefaultArrayResidency());
        }
    };

    REGISTER_LOGICAL_OPERATOR_FACTORY(LogicalPPTS, "PPTS");

}
