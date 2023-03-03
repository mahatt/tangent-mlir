
include "TangentDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : Tangent_Op<"constant", []> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal 
  }];

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}


def PrintOp : Tangent_Op<"print", []> {
    let summary = "print operation";
    let description = [{ The "print" builtin operation}];
    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

