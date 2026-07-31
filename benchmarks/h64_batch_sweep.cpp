#include <iostream>

#include "gru_cell.hpp"

#include <fstream>

#include <sstream>

#include <cmath>

#include <chrono>

#include <thread>

#include <algorithm>

#include <iomanip>



struct ReadPair {

    std::string bases;

    std::string quals;

};



struct ThreadResult {

    float total_loss = 0.0f;

    long total_chars = 0;

};



void process_read_batch(QualGRU& model,

                         const std::unordered_map<char, int>& qual_stoi,

                         const std::unordered_map<char, int>& base_stoi,

                         const std::vector<ReadPair>& reads,

                         int start_idx, int end_idx,

                         int batch_size,

                         ThreadResult& result) {



    int hidden_size = model.cell.hidden_size;

    int qual_emb_dim = model.cell.qual_emb_dim;

    int base_emb_dim = model.cell.base_emb_dim;

    int input_dim = qual_emb_dim + 2 * base_emb_dim;



    for (int b_start = start_idx; b_start < end_idx; b_start += batch_size) {

        int b_end = std::min(b_start + batch_size, end_idx);

        int this_batch_size = b_end - b_start;



        std::vector<std::vector<int>> qual_ids(this_batch_size);

        std::vector<std::vector<int>> base_ids(this_batch_size);

        int max_len = 0;



        for (int i = 0; i < this_batch_size; i++) {

            const std::string& quals = reads[b_start + i].quals;

            const std::string& bases = reads[b_start + i].bases;

            for (char c : quals) qual_ids[i].push_back(qual_stoi.at(c));

            for (char c : bases) base_ids[i].push_back(base_stoi.at(c));

            max_len = std::max(max_len, (int)quals.size());

        }



        BatchGRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim, this_batch_size);

        Eigen::MatrixXf H = Eigen::MatrixXf::Zero(hidden_size, this_batch_size);



        for (int t = 0; t < max_len - 1; t++) {

            Eigen::RowVectorXf mask(this_batch_size);

            std::vector<int> q_idx(this_batch_size), b_idx(this_batch_size), bn_idx(this_batch_size);

            for (int i = 0; i < this_batch_size; i++) {

                int len = qual_ids[i].size();

                bool active = (t < len - 1);

                mask(i) = active ? 1.0f : 0.0f;

                int tt = active ? t : std::max(0, len - 2);

                q_idx[i] = qual_ids[i][tt];

                b_idx[i] = base_ids[i][tt];

                bn_idx[i] = base_ids[i][tt + 1];

            }



            model.cell.forward_batch(q_idx, b_idx, bn_idx, H, mask, buf);

            H = buf.H_new;



            Eigen::MatrixXf probs = model.predict_probs_batch(H);



            for (int i = 0; i < this_batch_size; i++) {

                if (mask(i) > 0.5f) {

                    int target_idx = qual_ids[i][t + 1];

                    float p = probs(target_idx, i);

                    result.total_loss += -std::log(p);

                    result.total_chars++;

                }

            }

        }

    }

}



std::vector<ReadPair> load_paired_file(const std::string& path) {

    std::vector<ReadPair> reads;

    std::ifstream file(path);

    std::string line;

    while (std::getline(file, line)) {

        size_t tab_pos = line.find('\t');

        if (tab_pos == std::string::npos) continue;

        ReadPair rp;

        rp.bases = line.substr(0, tab_pos);

        rp.quals = line.substr(tab_pos + 1);

        reads.push_back(rp);

    }

    return reads;

}



int main() {

    QualGRU model("export_h256", 50, 256, 32, 8);



    std::ifstream vocab_file("qual_vocab.txt");

    std::string qual_vocab((std::istreambuf_iterator<char>(vocab_file)), std::istreambuf_iterator<char>());

    std::unordered_map<char, int> qual_stoi;

    for (size_t i = 0; i < qual_vocab.size(); i++) qual_stoi[qual_vocab[i]] = i;



    std::string base_vocab = "ACGTN";

    std::unordered_map<char, int> base_stoi;

    for (size_t i = 0; i < base_vocab.size(); i++) base_stoi[base_vocab[i]] = i;



    std::vector<ReadPair> reads = load_paired_file("paired_chr20_10k.txt");

    std::sort(reads.begin(), reads.end(), [](const ReadPair& a, const ReadPair& b) {

        return a.quals.size() < b.quals.size();

    });



    int n_reads_to_test = 640;

    int n_threads = 64;

    int stride = reads.size() / n_reads_to_test;

    std::vector<ReadPair> test_reads;

    for (int i = 0; i < n_reads_to_test; i++) test_reads.push_back(reads[i * stride]);

    std::sort(test_reads.begin(), test_reads.end(), [](const ReadPair& a, const ReadPair& b) {

        return a.quals.size() < b.quals.size();

    });



    std::vector<int> batch_sizes = {8, 16, 32, 64, 128, 256};



    std::cout << "=== H64 BATCH SIZE SWEEP (64 threads) ===" << std::endl;

    std::cout << std::left << std::setw(12) << "BatchSize"

              << std::setw(14) << "Elapsed(s)"

              << std::setw(16) << "Chars/sec"

              << std::setw(12) << "MB/s"

              << std::setw(10) << "bpc" << std::endl;

    std::cout << std::string(64, '-') << std::endl;



    for (int bs : batch_sizes) {

        std::vector<std::thread> threads;

        std::vector<ThreadResult> results(n_threads);

        int reads_per_thread = n_reads_to_test / n_threads;



        auto t_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_threads; i++) {

            int start_idx = i * reads_per_thread;

            int end_idx = (i == n_threads - 1) ? n_reads_to_test : start_idx + reads_per_thread;

            threads.emplace_back(process_read_batch, std::ref(model), std::ref(qual_stoi), std::ref(base_stoi),

                                  std::ref(test_reads), start_idx, end_idx, bs, std::ref(results[i]));

        }

        for (auto& t : threads) t.join();

        auto t_end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(t_end - t_start).count();



        float total_loss = 0.0f;

        long total_chars = 0;

        for (const auto& r : results) {

            total_loss += r.total_loss;

            total_chars += r.total_chars;

        }

        float bpc = (total_loss / total_chars) / std::log(2.0f);

        double chars_per_sec = total_chars / elapsed;

        double mb_per_sec = chars_per_sec / (1024.0 * 1024.0);



        std::cout << std::left << std::setw(12) << bs

                   << std::setw(14) << std::fixed << std::setprecision(3) << elapsed

                   << std::setw(16) << std::fixed << std::setprecision(0) << chars_per_sec

                   << std::setw(12) << std::fixed << std::setprecision(3) << mb_per_sec

                   << std::setw(10) << std::fixed << std::setprecision(4) << bpc

                   << std::endl;

    }



    return 0;

}
