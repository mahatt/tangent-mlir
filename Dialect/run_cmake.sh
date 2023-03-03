
#! /bin/bash

mkdir build 
cd build

cmake ../ -G "Unix Makefiles" \
-DLLVM_DIR=/mnt/c/UbuntuWork/Projects/repo/llvm/build/lib/cmake/llvm \
-DMLIR_DIR=/mnt/c/UbuntuWork/Projects/repo/llvm/build/lib/cmake/mlir

#cmake ../ -G "Unix Makefiles" -DLLVM_DIR=/src/llvm-project/rbuild/lib/cmake/llvm  -DMLIR_DIR=/src/llvm-project/mlbuild/lib/cmake/mlir

cmake --build . --target tangent-opt
