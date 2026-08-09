
#!/bin/bash

set -e

cd "$(dirname "$0")/.."

# -msse4.2 is load-bearing, NOT a tuning choice. -march=native emits whatever

# SIMD the build machine has, which changes matmul accumulation order, which

# shifts the low bits of the probabilities and can desync the rANS coder.

# Archives must be readable by binaries built elsewhere.

g++ -O2 -std=c++17 -msse4.2 -DNDEBUG -I /usr/include/eigen3 -I . -I compressor/include compressor/src/qualgru.cpp -o compressor/qualgru -lz

echo "built compressor/qualgru"

