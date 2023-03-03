#ifndef MLIR_TANGENT_PASSES_H
#define MLIR_TANGENT_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace tangent {
  std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
}

#endif