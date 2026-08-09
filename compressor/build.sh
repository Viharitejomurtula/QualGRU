#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# Thin wrapper around the CMake build (see CMakeLists.txt at repo root for
# the actual compiler flags — -msse4.2 fixed, never -march=native). Kept for
# people who just want `bash compressor/build.sh` to produce
# compressor/qualgru without thinking about build directories.

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQUALGRU_BUILD_TESTS=OFF
cmake --build build -j"$(nproc)" --target qualgru

echo "built compressor/qualgru"
