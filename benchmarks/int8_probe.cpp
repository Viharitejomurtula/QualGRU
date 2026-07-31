#include <iostream>

#include <chrono>

#include <vector>

#include <cstdint>

#include <random>

#include <immintrin.h>

#include <Eigen/Dense>



// Simple AVX2 int8 dot product using vpmaddubsw (unsigned x signed 8-bit

// multiply-add, packed into 16-bit results). This is the realistic AVX2

// (non-VNNI) int8 primitive -- NOT as fast as AVX-512 VNNI's vpdpbusd,

// but it's what this CPU can actually do.

int32_t int8_dot(const uint8_t* a, const int8_t* b, int n) {

    __m256i acc = _mm256_setzero_si256();

    int i = 0;

    for (; i + 32 <= n; i += 32) {

        __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));

        __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));

        __m256i prod16 = _mm256_maddubs_epi16(va, vb);          // 32x int8 -> 16x int16

        __m256i prod32 = _mm256_madd_epi16(prod16, _mm256_set1_epi16(1)); // 16x int16 -> 8x int32

        acc = _mm256_add_epi32(acc, prod32);

    }

    int32_t buf[8];

    _mm256_storeu_si256((__m256i*)buf, acc);

    int32_t sum = 0;

    for (int j = 0; j < 8; j++) sum += buf[j];

    for (; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];  // tail

    return sum;

}



int main() {

    int hidden_size = 64;  // matches your h64 config

    int n_iters = 1000000;



    std::mt19937 rng(42);

    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);

    std::uniform_int_distribution<int> idist(-100, 100);



    // --- FP32 baseline: dot product of two hidden_size vectors ---

    std::vector<float> a_f(hidden_size), b_f(hidden_size);

    for (int i = 0; i < hidden_size; i++) { a_f[i] = fdist(rng); b_f[i] = fdist(rng); }

    Eigen::Map<Eigen::VectorXf> ea(a_f.data(), hidden_size);

    Eigen::Map<Eigen::VectorXf> eb(b_f.data(), hidden_size);



    volatile float fp32_result = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_iters; i++) {

        fp32_result = ea.dot(eb);

    }

    auto t1 = std::chrono::high_resolution_clock::now();

    double fp32_time = std::chrono::duration<double>(t1 - t0).count();



    // --- INT8 version: same size, quantized data ---

    std::vector<uint8_t> a_i8(hidden_size);

    std::vector<int8_t> b_i8(hidden_size);

    for (int i = 0; i < hidden_size; i++) {

        a_i8[i] = (uint8_t)(idist(rng) + 128);

        b_i8[i] = (int8_t)idist(rng);

    }



    volatile int32_t int8_result = 0;

    auto t2 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_iters; i++) {

        int8_result = int8_dot(a_i8.data(), b_i8.data(), hidden_size);

    }

    auto t3 = std::chrono::high_resolution_clock::now();

    double int8_time = std::chrono::duration<double>(t3 - t2).count();



    std::cout << "=== INT8 vs FP32 DOT PRODUCT MICROBENCHMARK (hidden_size=" << hidden_size << ") ===" << std::endl;

    std::cout << "FP32: " << fp32_time << "s  (" << (fp32_time/n_iters)*1e9 << " ns/call)" << std::endl;

    std::cout << "INT8: " << int8_time << "s  (" << (int8_time/n_iters)*1e9 << " ns/call)" << std::endl;

    std::cout << "Speedup: " << (fp32_time / int8_time) << "x" << std::endl;



    return 0;

}
