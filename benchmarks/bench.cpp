#include <iostream>

#include "gru_cell.hpp"

#include <chrono>



int main() {

    QualGRU model("export_h256", 50, 256, 32, 8);



    int hidden_size = model.cell.hidden_size;

    int qual_emb_dim = model.cell.qual_emb_dim;

    int base_emb_dim = model.cell.base_emb_dim;

    int input_dim = qual_emb_dim + 2 * base_emb_dim;



    GRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim);

    Eigen::VectorXf h = Eigen::VectorXf::Zero(hidden_size);



    int n_iters = 100000;



    // Warm up (first calls can be slower due to cold cache / lazy allocation)

    for (int i = 0; i < 1000; i++) {

        model.cell.forward_into(0, 0, 1, h, buf);

        h = buf.h_new;

    }



    // --- Benchmark 1: forward_into (full gate computation) ---

    h = Eigen::VectorXf::Zero(hidden_size);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_iters; i++) {

        model.cell.forward_into(0, 0, 1, h, buf);

        h = buf.h_new;

    }

    auto t1 = std::chrono::high_resolution_clock::now();

    double forward_time = std::chrono::duration<double>(t1 - t0).count();



    // --- Benchmark 2: predict_probs (output projection + softmax) ---

    auto t2 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_iters; i++) {

        Eigen::VectorXf probs = model.predict_probs(h);

    }

    auto t3 = std::chrono::high_resolution_clock::now();

    double predict_time = std::chrono::duration<double>(t3 - t2).count();



    // --- Benchmark 3: raw matmuls only (no gates, no embeddings, no nonlinearities) ---

    // Isolates pure FLOP cost of the 3 W_* hidden-to-hidden matmuls, to see

    // how much of forward_into's cost is "just the linear algebra" vs.

    // everything else (embedding lookups, concatenation, elementwise ops).

    Eigen::VectorXf dummy_out(hidden_size);

    auto t4 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_iters; i++) {

        dummy_out.noalias() = model.cell.W_r * h;

        dummy_out.noalias() = model.cell.W_u * h;

        dummy_out.noalias() = model.cell.W_c * h;

    }

    auto t5 = std::chrono::high_resolution_clock::now();

    double matmul_time = std::chrono::duration<double>(t5 - t4).count();



    std::cout << "=== MICROBENCHMARK (single-threaded, " << n_iters << " iterations) ===" << std::endl;

    std::cout << "forward_into total:  " << forward_time << "s  ("

              << (forward_time / n_iters) * 1e9 << " ns/call)" << std::endl;

    std::cout << "predict_probs total: " << predict_time << "s  ("

              << (predict_time / n_iters) * 1e9 << " ns/call)" << std::endl;

    std::cout << "raw W_* matmuls only: " << matmul_time << "s  ("

              << (matmul_time / n_iters) * 1e9 << " ns/call)" << std::endl;

    std::cout << std::endl;

    std::cout << "matmul as % of forward_into: " << (matmul_time / forward_time) * 100.0 << "%" << std::endl;

    std::cout << "predict_probs as % of (forward_into + predict_probs): "

              << (predict_time / (forward_time + predict_time)) * 100.0 << "%" << std::endl;



    return 0;

}
