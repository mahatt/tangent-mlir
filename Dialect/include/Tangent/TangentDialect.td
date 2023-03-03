include "mlir/IR/OpBase.td"


// dialect definition.

def Tangent_Dialect : Dialect {
    let name = "tangent";
    let summary = "A Tangent out-of-tree MLIR dialect.";
    let description = [{
        This dialect is minimal example to implement Tangent-world kind of sample code
        for MLIR.
    }];
    let cppNamespace = "::tangent";
    let hasConstantMaterializer = 1;
}


// Base  operation definition.

class Tangent_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tangent_Dialect, mnemonic, traits>;
