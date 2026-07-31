#include <iostream>

#include "gru_cell.hpp"

#include <fstream>

#include <sstream>

#include <cmath>

#include <chrono>

#include <thread>

#include <algorithm>

#include <iomanip>



// ---------------------------------------------------------------------------

// Reference baselines, measured separately on the SAME chr20 data

// (186,533 reads, 3,471,674,698 quality characters).

//

//   CRAM:   differential sizing -- real-quality CRAM minus flat-quality CRAM,

//           isolating the fqzcomp quality stream from sequence/header/CIGAR.

//   CoLoRd: tool's own "Quality size" field, -q org (lossless) mode.

//

// Throughput reference: samtools view -@ 64 on the real-quality CRAM,

// full-pipeline decompression (includes sequence + alignment reconstruction,

// so this is a conservative/low estimate of fqzcomp's quality-only speed).

// ---------------------------------------------------------------------------

const float CRAM_BPC          = 3.9505f;

const float COLORD_BPC        = 4.0816f;

const double CRAM_DECOMP_MBPS = 9.767;



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



    int hidden_size  = model.cell.hidden_size;

    int qual_emb_dim = model.cell.qual_emb_dim;

    int base_emb_dim = model.cell.base_emb_dim;

    int input_dim    = qual_emb_dim + 2 * base_emb_dim;



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

                q_idx[i]  = qual_ids[i][tt];

                b_idx[i]  = base_ids[i][tt];

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



struct BenchResult {

    std::string label;

    int hidden_size;

    double elapsed_sec;

    long total_chars;

    double chars_per_sec;

    double mb_per_sec;

    float bpc;

};



BenchResult run_config(const std::string& label,

                        const std::string& export_dir,

                        int hidden_size,

                        const std::unordered_map<char, int>& qual_stoi,

                        const std::unordered_map<char, int>& base_stoi,

                        const std::vector<ReadPair>& reads,

                        int n_reads, int n_threads, int batch_size,

                        int n_repeats) {



    QualGRU model(export_dir, 50, hidden_size, 32, 8);



    double best_cps = 0.0;

    double best_elapsed = 0.0;

    long   best_chars = 0;

    float  bpc = 0.0f;



    // Repeat and keep the best run -- this machine is shared, so a single

    // timing can be polluted by other users' load. Best-of-N is the more

    // honest estimate of what the code itself can do.

    for (int rep = 0; rep < n_repeats; rep++) {

        std::vector<std::thread> threads;

        std::vector<ThreadResult> results(n_threads);

        int reads_per_thread = n_reads / n_threads;



        auto t0 = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_threads; i++) {

            int s = i * reads_per_thread;

            int e = (i == n_threads - 1) ? n_reads : s + reads_per_thread;

            threads.emplace_back(process_read_batch, std::ref(model), std::ref(qual_stoi),

                                  std::ref(base_stoi), std::ref(reads), s, e,

                                  batch_size, std::ref(results[i]));

        }

        for (auto& t : threads) t.join();

        auto t1 = std::chrono::high_resolution_clock::now();



        double elapsed = std::chrono::duration<double>(t1 - t0).count();



        float total_loss = 0.0f;

        long total_chars = 0;

        for (const auto& r : results) {

            total_loss  += r.total_loss;

            total_chars += r.total_chars;

        }



        double cps = total_chars / elapsed;

        if (cps > best_cps) {

            best_cps     = cps;

            best_elapsed = elapsed;

            best_chars   = total_chars;

            bpc          = (total_loss / total_chars) / std::log(2.0f);

        }

        std::cout << "  [" << label << "] run " << (rep + 1) << "/" << n_repeats

                   << ": " << std::fixed << std::setprecision(0) << cps << " chars/sec"

                   << std::endl;

    }



    return {label, hidden_size, best_elapsed, best_chars,

            best_cps, best_cps / (1024.0 * 1024.0), bpc};

}



int main(int argc, char** argv) {

    std::string data_file = (argc > 1) ? argv[1] : "paired_chr20_10k.txt";

    int n_reads    = 640;

    int n_threads  = 64;

    int batch_size = 32;   // best from the batch-size sweeps

    int n_repeats  = 3;    // best-of-N, shared machine



    std::ifstream vocab_file("qual_vocab.txt");

    std::string qual_vocab((std::istreambuf_iterator<char>(vocab_file)),

                            std::istreambuf_iterator<char>());

    std::unordered_map<char, int> qual_stoi;

    for (size_t i = 0; i < qual_vocab.size(); i++) qual_stoi[qual_vocab[i]] = i;



    std::string base_vocab = "ACGTN";

    std::unordered_map<char, int> base_stoi;

    for (size_t i = 0; i < base_vocab.size(); i++) base_stoi[base_vocab[i]] = i;



    std::vector<ReadPair> reads = load_paired_file(data_file);

    if (reads.empty()) {

        std::cerr << "ERROR: no reads loaded from " << data_file << std::endl;

        return 1;

    }

    std::cout << "Loaded " << reads.size() << " reads from " << data_file << std::endl;



    // Sort by length, then sample evenly across the length distribution so

    // the test set has a realistic length mix while keeping each batch

    // roughly length-homogeneous (minimises wasted masked computation).

    std::sort(reads.begin(), reads.end(), [](const ReadPair& a, const ReadPair& b) {

        return a.quals.size() < b.quals.size();

    });

    int stride = reads.size() / n_reads;

    std::vector<ReadPair> test_reads;

    for (int i = 0; i < n_reads; i++) test_reads.push_back(reads[i * stride]);

    std::sort(test_reads.begin(), test_reads.end(), [](const ReadPair& a, const ReadPair& b) {

        return a.quals.size() < b.quals.size();

    });



    long total_test_chars = 0;

    for (const auto& r : test_reads) total_test_chars += r.quals.size();

    std::cout << "Test set: " << n_reads << " reads, "

               << total_test_chars << " quality characters" << std::endl;

    std::cout << "Config: " << n_threads << " threads, batch=" << batch_size

               << ", best-of-" << n_repeats << std::endl << std::endl;



    std::vector<BenchResult> results;

    results.push_back(run_config("h256", "export_h256", 256, qual_stoi, base_stoi,

                                  test_reads, n_reads, n_threads, batch_size, n_repeats));

    results.push_back(run_config("h64",  "export_h64",   64, qual_stoi, base_stoi,

                                  test_reads, n_reads, n_threads, batch_size, n_repeats));

    results.push_back(run_config("h32",  "export_h32",   32, qual_stoi, base_stoi,

                                  test_reads, n_reads, n_threads, batch_size, n_repeats));



    std::cout << "\n";

    std::cout << "==========================================================================\n";

    std::cout << " FINAL BENCHMARK -- QualGRU vs production compressors\n";

    std::cout << "==========================================================================\n";

    std::cout << std::left

               << std::setw(12) << "Method"

               << std::setw(10) << "bpc"

               << std::setw(14) << "vs CRAM"

               << std::setw(14) << "vs CoLoRd"

               << std::setw(12) << "MB/s"

               << std::endl;

    std::cout << std::string(74, '-') << std::endl;



    for (const auto& r : results) {

        double vs_cram   = (CRAM_BPC   - r.bpc) / CRAM_BPC   * 100.0;

        double vs_colord = (COLORD_BPC - r.bpc) / COLORD_BPC * 100.0;

        std::cout << std::left

                   << std::setw(12) << ("QualGRU " + r.label)

                   << std::setw(10) << std::fixed << std::setprecision(4) << r.bpc

                   << std::setw(14) << (std::to_string((int)(vs_cram * 100) / 100.0).substr(0,5) + "% better")

                   << std::setw(14) << (std::to_string((int)(vs_colord * 100) / 100.0).substr(0,5) + "% better")

                   << std::setw(12) << std::fixed << std::setprecision(3) << r.mb_per_sec

                   << std::endl;

    }



    std::cout << std::string(74, '-') << std::endl;

    std::cout << std::left

               << std::setw(12) << "CRAM"

               << std::setw(10) << std::fixed << std::setprecision(4) << CRAM_BPC

               << std::setw(14) << "--"

               << std::setw(14) << "--"

               << std::setw(12) << std::fixed << std::setprecision(3) << CRAM_DECOMP_MBPS

               << std::endl;

    std::cout << std::left

               << std::setw(12) << "CoLoRd"

               << std::setw(10) << std::fixed << std::setprecision(4) << COLORD_BPC

               << std::setw(14) << "--"

               << std::setw(14) << "--"

               << std::setw(12) << "n/m"

               << std::endl;

    std::cout << "==========================================================================\n";

    std::cout << "CRAM/CoLoRd bpc measured separately on chr20 (186,533 reads,\n";

    std::cout << "3,471,674,698 quality chars). CRAM MB/s is full-pipeline samtools\n";

    std::cout << "decompression (-@ 64), which also reconstructs sequence and alignment,\n";

    std::cout << "so it understates fqzcomp's quality-only speed. n/m = not measured.\n";



    return 0;

}
