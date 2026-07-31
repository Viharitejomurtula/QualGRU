#include <iostream>

#include "gru_cell.hpp"

#include <fstream>

#include <chrono>

#include <algorithm>



struct ReadPair {

    std::string bases;

    std::string quals;

};



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

    QualGRU model("export_h32", 50, 32, 32, 8);



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



    int n_reads = 640;

    int stride = reads.size() / n_reads;

    std::vector<ReadPair> test_reads;

    for (int i = 0; i < n_reads; i++) test_reads.push_back(reads[i * stride]);

    std::sort(test_reads.begin(), test_reads.end(), [](const ReadPair& a, const ReadPair& b) {

        return a.quals.size() < b.quals.size();

    });

    reads = test_reads;



    int hidden_size = model.cell.hidden_size;

    int qual_emb_dim = model.cell.qual_emb_dim;

    int base_emb_dim = model.cell.base_emb_dim;

    int input_dim = qual_emb_dim + 2 * base_emb_dim;

    int batch_size = 64;



    // Accumulated timers for each stage, across ALL batches.

    double t_index_conv = 0, t_buf_alloc = 0, t_xt_build = 0;

    double t_ru_matmul = 0, t_c_matmul = 0, t_mask = 0, t_predict = 0;



    long total_chars = 0;



    auto total_t0 = std::chrono::high_resolution_clock::now();



    for (int b_start = 0; b_start < n_reads; b_start += batch_size) {

        int b_end = std::min(b_start + batch_size, n_reads);

        int this_batch_size = b_end - b_start;



        // --- Stage: index conversion ---

        auto s0 = std::chrono::high_resolution_clock::now();

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

        auto s1 = std::chrono::high_resolution_clock::now();

        t_index_conv += std::chrono::duration<double>(s1 - s0).count();



        // --- Stage: buffer allocation ---

        BatchGRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim, this_batch_size);

        Eigen::MatrixXf H = Eigen::MatrixXf::Zero(hidden_size, this_batch_size);

        auto s2 = std::chrono::high_resolution_clock::now();

        t_buf_alloc += std::chrono::duration<double>(s2 - s1).count();



        for (int t = 0; t < max_len - 1; t++) {

            // --- Stage: mask + index selection ---

            auto m0 = std::chrono::high_resolution_clock::now();

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

            auto m1 = std::chrono::high_resolution_clock::now();

            t_mask += std::chrono::duration<double>(m1 - m0).count();



            // --- Stage: X_t build (embedding lookups + concat) ---

            for (int i = 0; i < this_batch_size; i++) {

                buf.QE.col(i) = model.cell.qual_emb.row(q_idx[i]).transpose();

                buf.BE_0.col(i) = model.cell.base_emb.row(b_idx[i]).transpose();

                buf.BE_1.col(i) = model.cell.base_emb.row(bn_idx[i]).transpose();

            }

            buf.X_t.block(0, 0, qual_emb_dim, this_batch_size) = buf.QE;

            buf.X_t.block(qual_emb_dim, 0, base_emb_dim, this_batch_size) = buf.BE_0;

            buf.X_t.block(qual_emb_dim + base_emb_dim, 0, base_emb_dim, this_batch_size) = buf.BE_1;

            auto m2 = std::chrono::high_resolution_clock::now();

            t_xt_build += std::chrono::duration<double>(m2 - m1).count();



            // --- Stage: RU matmul (fused r/u gates) ---

            buf.RU_pre.noalias() = model.cell.W_ru * H + model.cell.U_ru * buf.X_t;

            buf.RU_pre.colwise() += model.cell.b_ru;

            buf.R = (1.0f / (1.0f + (-buf.RU_pre.topRows(hidden_size).array()).exp())).matrix();

            buf.U = (1.0f / (1.0f + (-buf.RU_pre.bottomRows(hidden_size).array()).exp())).matrix();

            auto m3 = std::chrono::high_resolution_clock::now();

            t_ru_matmul += std::chrono::duration<double>(m3 - m2).count();



            // --- Stage: C matmul (candidate gate) ---

            buf.C_pre.noalias() = model.cell.W_c * (buf.R.cwiseProduct(H)) + model.cell.U_c * buf.X_t;

            buf.C_pre.colwise() += model.cell.b_c;

            buf.C = buf.C_pre.array().tanh().matrix();

            Eigen::MatrixXf H_update = (1.0f - buf.U.array()).matrix().cwiseProduct(H) + buf.U.cwiseProduct(buf.C);

            for (int i = 0; i < this_batch_size; i++) {

                buf.H_new.col(i) = mask(i) * H_update.col(i) + (1.0f - mask(i)) * H.col(i);

            }

            H = buf.H_new;

            auto m4 = std::chrono::high_resolution_clock::now();

            t_c_matmul += std::chrono::duration<double>(m4 - m3).count();



            // --- Stage: predict_probs_batch ---

            Eigen::MatrixXf probs = model.predict_probs_batch(H);

            for (int i = 0; i < this_batch_size; i++) {

                if (mask(i) > 0.5f) total_chars++;

            }

            auto m5 = std::chrono::high_resolution_clock::now();

            t_predict += std::chrono::duration<double>(m5 - m4).count();

        }

    }



    auto total_t1 = std::chrono::high_resolution_clock::now();

    double total_time = std::chrono::duration<double>(total_t1 - total_t0).count();



    std::cout << "=== STAGE BREAKDOWN (single-threaded, " << n_reads << " reads, batch=" << batch_size << ") ===" << std::endl;

    std::cout << "Total time:        " << total_time << "s" << std::endl;

    std::cout << "Total chars:       " << total_chars << std::endl;

    std::cout << std::endl;

    std::cout << "index_conv:  " << t_index_conv << "s  (" << (t_index_conv/total_time)*100 << "%)" << std::endl;

    std::cout << "buf_alloc:   " << t_buf_alloc << "s  (" << (t_buf_alloc/total_time)*100 << "%)" << std::endl;

    std::cout << "mask:        " << t_mask << "s  (" << (t_mask/total_time)*100 << "%)" << std::endl;

    std::cout << "xt_build:    " << t_xt_build << "s  (" << (t_xt_build/total_time)*100 << "%)" << std::endl;

    std::cout << "ru_matmul:   " << t_ru_matmul << "s  (" << (t_ru_matmul/total_time)*100 << "%)" << std::endl;

    std::cout << "c_matmul:    " << t_c_matmul << "s  (" << (t_c_matmul/total_time)*100 << "%)" << std::endl;

    std::cout << "predict:     " << t_predict << "s  (" << (t_predict/total_time)*100 << "%)" << std::endl;

    std::cout << std::endl;

    std::cout << "Throughput: " << (total_chars / total_time) << " chars/sec" << std::endl;



    return 0;

}
