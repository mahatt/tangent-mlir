#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Tangent/TangentDialect.h"
#include "Tangent/TangentOps.h"

using namespace mlir;
using namespace tangent;

//===----------------------------------------------------------------------===//
// Tangent dialect.
//===----------------------------------------------------------------------===//

#include "Tangent/TangentOpsDialect.cpp.inc"

void TangentDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tangent/TangentOps.cpp.inc"
      >();
}

void tangent::ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  tangent::ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::Operation *TangentDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
    return builder.create<tangent::ConstantOp>(loc, type,
                                      value.cast<mlir::DenseElementsAttr>());
}