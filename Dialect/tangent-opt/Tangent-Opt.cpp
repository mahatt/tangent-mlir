

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Tangent/TangentDialect.h"
#include "Tangent/TangentPasses.h"

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input Tangent file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int dumpLLVMIR(mlir::ModuleOp module) {
  mlir::registerLLVMDialectTranslation(*module->getContext());
  // Convert the module to LLVM IR in a new LLVM IR context.

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  // Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::outs() << *llvmModule << "\n";
  return 0;
}

int loadMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
  if (int error = loadMLIR(context, module)) {
    return error;
  }

  // Register passes to be applied in this compile process
  mlir::PassManager passManager(&context);
  mlir::applyPassManagerCLOptions(passManager);

  //passManager.addPass(tangent::createLowerToAffinePass());
  //passManager.addPass(tangent::createLowerToLLVMPass());

  if (mlir::failed(passManager.run(*module))) {
    return 4;
  }

  return 0;
}

int runJit(mlir::ModuleOp module) {
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Register the translation from MLIR to LLVM IR, which must happen before we
    // can JIT-compile.
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(0, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}

int main(int argc, char **argv) {
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "Tangent compiler\n");
  mlir::MLIRContext context;
  context.getOrLoadDialect<tangent::TangentDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module)) {
    return error;
  }

  dumpLLVMIR(*module);
//  runJit(*module);

  return 0;
}
