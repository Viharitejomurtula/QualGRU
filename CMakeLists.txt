cmake_minimum_required(VERSION 3.16)
project(QualGRU CXX)

# -----------------------------------------------------------------------------
# -msse4.2 is load-bearing, NOT a tuning choice. -march=native emits whatever
# SIMD the build machine has, which changes matmul accumulation order, which
# shifts the low bits of the probabilities and can desync the rANS coder.
# Archives must remain decodable by a binary built elsewhere, so this project
# never enables -march=native and warns if the user tries to force it in.
# -----------------------------------------------------------------------------
if(CMAKE_CXX_FLAGS MATCHES "-march=native" OR CMAKE_CXX_FLAGS_INIT MATCHES "-march=native")
    message(WARNING "CMAKE_CXX_FLAGS requests -march=native, which QualGRU does not "
                     "support: it changes floating-point accumulation order and can "
                     "desync archives between machines. -msse4.2 is used instead.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(ZLIB REQUIRED)

# Fixed compressor/qualgru output flags, independent of CMAKE_BUILD_TYPE.
set(QUALGRU_COMPILE_OPTIONS -O2 -msse4.2 -DNDEBUG)

option(QUALGRU_BUILD_TESTS "Build the rANS and format unit tests" ON)

# -----------------------------------------------------------------------------
# Main CLI: qualgru compress / qualgru decompress
# -----------------------------------------------------------------------------
add_executable(qualgru compressor/src/qualgru.cpp)
target_include_directories(qualgru PRIVATE inference compressor/include)
target_compile_options(qualgru PRIVATE ${QUALGRU_COMPILE_OPTIONS})
target_link_libraries(qualgru PRIVATE Eigen3::Eigen ZLIB::ZLIB)
# Model weights are resolved at runtime relative to the binary
# (exe_dir() + "/models/<name>", via /proc/self/exe), so qualgru must land
# next to compressor/models/ — pin the output directory there rather than
# the usual out-of-tree build/ dir.
set_target_properties(qualgru PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/compressor
)

# For `cmake --install`, models/ must land as a sibling of the installed
# binary for the same exe_dir()-relative lookup to find it.
install(TARGETS qualgru RUNTIME DESTINATION bin)
install(DIRECTORY compressor/models/ DESTINATION bin/models)

# -----------------------------------------------------------------------------
# Tests: rANS coder correctness + format round-trip. Plain exit-code checks,
# no test framework — matches how they were written to be run standalone.
# -----------------------------------------------------------------------------
if(QUALGRU_BUILD_TESTS)
    enable_testing()

    add_executable(rans_test compressor/encoder/rans_test.cpp)
    target_compile_options(rans_test PRIVATE -O2 -msse4.2)
    add_test(NAME rans_test COMMAND rans_test)

    add_executable(rans_adversarial compressor/encoder/rans_adversarial.cpp)
    target_compile_options(rans_adversarial PRIVATE -O2 -msse4.2)
    add_test(NAME rans_adversarial COMMAND rans_adversarial)

    add_executable(test_format compressor/tests/test_format.cpp)
    target_include_directories(test_format PRIVATE compressor/include)
    target_compile_options(test_format PRIVATE -O2 -msse4.2)
    target_link_libraries(test_format PRIVATE ZLIB::ZLIB)
    add_test(NAME test_format COMMAND test_format)
endif()
