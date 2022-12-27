
#! /bin/bash

mkdir build 
cd build

cmake ../ -G "Unix Makefiles" \
-DLLVM_DIR=/mnt/c/UbuntuWork/Projects/llvm-project/rbuild/lib/cmake/llvm \
-DMLIR_DIR=/mnt/c/UbuntuWork/Projects/llvm-project/mlbuild/lib/cmake/mlir

cmake --build . --target tangent-opt