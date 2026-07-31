#include <iostream>

#include "gru_cell.hpp"  //includes header file

#include <fstream>

#include <sstream>

#include <cmath>

#include <chrono>

#include <thread>

#include <mutex>





struct ReadPair {

        std::string bases;

        std::string quals;

};



struct ThreadResult {

        float total_loss = 0.0f;

        long total_chars = 0;

};



void process_read_range(QualGRU& model,

                         const std::unordered_map<char, int>& qual_stoi,

                         const std::unordered_map<char, int>& base_stoi,

                         const std::vector<ReadPair>& reads,

                         int start_idx, int end_idx,

                         ThreadResult& result) {



        int hidden_size = model.cell.hidden_size;

        int qual_emb_dim = model.cell.qual_emb_dim;

        int base_emb_dim = model.cell.base_emb_dim;

        int input_dim = qual_emb_dim + 2 * base_emb_dim;



        GRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim);  // created ONCE per thread



        for (int r = start_idx; r < end_idx; r++) {

                const std::string& bases = reads[r].bases;

                const std::string& quals = reads[r].quals;



                Eigen::VectorXf h = Eigen::VectorXf::Zero(hidden_size);



                for (size_t t = 0; t < quals.size() - 1; t++) {

                        int qual_idx = qual_stoi.at(quals[t]);

                        int base_idx = base_stoi.at(bases[t]);

                        int base_next_idx = base_stoi.at(bases[t + 1]);



                        model.cell.forward_into(qual_idx, base_idx, base_next_idx, h, buf);

                        h = buf.h_new;



                        Eigen::VectorXf probs = model.predict_probs(h);

                        int target_idx = qual_stoi.at(quals[t + 1]);

                        float p = probs[target_idx];



                        result.total_loss += -std::log(p);

                        result.total_chars++;

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

void process_read_batch(QualGRU& model, const std::unordered_map<char, int>& qual_stoi, const std::unordered_map<char, int>& base_stoi, const std::vector<ReadPair>& reads, int start_idx, int end_idx, int batch_size, ThreadResult& result) {
	int hidden_size = model.cell.hidden_size;
	int qual_emb_dim = model.cell.qual_emb_dim;
	int base_emb_dim = model.cell.base_emb_dim;
	int input_dim = qual_emb_dim + 2 * base_emb_dim;

	for (int b_start = start_idx; b_start < end_idx; b_start += batch_size) {
		int b_end = std::min(b_start+batch_size, end_idx);
		int this_batch_size = b_end - b_start;

		std::vector<std::vector<int>> qual_ids(this_batch_size);
		std::vector<std::vector<int>> base_ids(this_batch_size);
		int max_len = 0;

		for (int i=0; i < this_batch_size; i++) {
		const std::string& quals = reads[b_start + i].quals;
		const std::string& bases = reads[b_start + i].bases;
		for (char c: quals) qual_ids[i].push_back(qual_stoi.at(c));
		for (char c: bases) base_ids[i].push_back(base_stoi.at(c));
		max_len = std::max(max_len, (int)quals.size());

		}

		BatchGRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim, this_batch_size);
		Eigen::MatrixXf H = Eigen::MatrixXf::Zero(hidden_size, this_batch_size);

		for (int t = 0; t < max_len -1; t++) {
			Eigen::RowVectorXf mask(this_batch_size);
			std::vector<int> q_idx(this_batch_size), b_idx(this_batch_size), bn_idx(this_batch_size);
			for (int i =0;i < this_batch_size;i++) {
				int len = qual_ids[i].size();
				bool active = (t < len-1);
				mask(i) = active ? 1.0f : 0.0f;
				int tt = active ? t : std::max(0, len - 2);
                		q_idx[i] = qual_ids[i][tt];
                		b_idx[i] = base_ids[i][tt];
                		bn_idx[i] = base_ids[i][tt + 1];
			}

			model.cell.forward_batch(q_idx, b_idx, bn_idx, H, mask, buf);
			H = buf.H_new;

			Eigen::MatrixXf probs = model.predict_probs_batch(H);

			for (int i=0; i < this_batch_size; i++) {
				if (mask(i) > 0.5f) {
					int target_idx = qual_ids[i][t+1];
					float p = probs(target_idx, i);
					result.total_loss += -std::log(p);
					result.total_chars++;
				}
			}

		}

	}




}
std::vector<Eigen::MatrixXf> precompute_batch_inputs(

        QualGRU& model,

        const std::vector<std::vector<int>>& qual_ids,

        const std::vector<std::vector<int>>& base_ids,

        int max_len) {



        int qual_emb_dim = model.cell.qual_emb_dim;

        int base_emb_dim = model.cell.base_emb_dim;

        int input_dim = qual_emb_dim + 2 * base_emb_dim;

        int batch_size = qual_ids.size();



        std::vector<Eigen::MatrixXf> X_all(max_len - 1, Eigen::MatrixXf(input_dim, batch_size));



        for (int t = 0; t < max_len - 1; t++) {

                for (int i = 0; i < batch_size; i++) {

                        int len = qual_ids[i].size();

                        int tt = (t < len - 1) ? t : std::max(0, len - 2);



                        Eigen::VectorXf qe = model.cell.qual_emb.row(qual_ids[i][tt]).transpose();

                        Eigen::VectorXf be_0 = model.cell.base_emb.row(base_ids[i][tt]).transpose();

                        Eigen::VectorXf be_1 = model.cell.base_emb.row(base_ids[i][tt + 1]).transpose();



                        X_all[t].block(0, i, qual_emb_dim, 1) = qe;

                        X_all[t].block(qual_emb_dim, i, base_emb_dim, 1) = be_0;

                        X_all[t].block(qual_emb_dim + base_emb_dim, i, base_emb_dim, 1) = be_1;

                }

        }



        return X_all;

}
void process_read_batch_v2(QualGRU& model,

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



                std::vector<Eigen::MatrixXf> X_all = precompute_batch_inputs(model, qual_ids, base_ids, max_len);



                BatchGRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim, this_batch_size);

                Eigen::MatrixXf H = Eigen::MatrixXf::Zero(hidden_size, this_batch_size);



                for (int t = 0; t < max_len - 1; t++) {

                        Eigen::RowVectorXf mask(this_batch_size);

                        for (int i = 0; i < this_batch_size; i++) {

                                int len = qual_ids[i].size();

                                mask(i) = (t < len - 1) ? 1.0f : 0.0f;

                        }



                        model.cell.forward_batch_precomputed(X_all[t], H, mask, buf);

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

int main() {

        // Load the trained model (cell + output projection) once.

        QualGRU model("export_h32", 50, 32, 32, 8);



        // Vocab lookup tables.

        std::ifstream vocab_file("qual_vocab.txt");

        std::string qual_vocab((std::istreambuf_iterator<char>(vocab_file)), std::istreambuf_iterator<char>());

        std::unordered_map<char, int> qual_stoi;

        for (size_t i = 0; i < qual_vocab.size(); i++) {

                qual_stoi[qual_vocab[i]] = i;

        }



        std::string base_vocab = "ACGTN";

        std::unordered_map<char, int> base_stoi;

        for (size_t i = 0; i < base_vocab.size(); i++) {

                base_stoi[base_vocab[i]] = i;

        }



        // Load real data.
	std::vector<ReadPair> reads = load_paired_file("paired_chr20_10k.txt");
        std::cout << "Loaded " << reads.size() << " reads." << std::endl;
        std::sort(reads.begin(), reads.end(), [](const ReadPair& a, const ReadPair& b) {
    		return a.quals.size() < b.quals.size();

	});

	int n_reads_to_test = 640;

	int n_threads = 64;



	// Sample from a randomized subset spread across the full (sorted) dataset,

	// rather than just the first 640 entries (which would all be short reads).

	std::vector<ReadPair> test_reads;

	int stride = reads.size() / n_reads_to_test;

	for (int i = 0; i < n_reads_to_test; i++) {

		test_reads.push_back(reads[i * stride]);

	}

	std::sort(test_reads.begin(), test_reads.end(), [](const ReadPair& a, const ReadPair& b) {

		return a.quals.size() < b.quals.size();

	});

	reads = test_reads;






        std::vector<std::thread> threads;

        std::vector<ThreadResult> results(n_threads);



        int reads_per_thread = n_reads_to_test / n_threads;



        auto t_start = std::chrono::high_resolution_clock::now();



        for (int i = 0; i < n_threads; i++) {

                int start_idx = i * reads_per_thread;

                int end_idx = (i == n_threads - 1) ? n_reads_to_test : start_idx + reads_per_thread;

                threads.emplace_back(process_read_range, std::ref(model), std::ref(qual_stoi), std::ref(base_stoi), std::ref(reads), start_idx, end_idx, std::ref(results[i]));

        }



        for (auto& t : threads) {

                t.join();

        }



        auto t_end = std::chrono::high_resolution_clock::now();

        double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();



        float total_loss = 0.0f;

        long total_chars = 0;

        for (const auto& r : results) {

                total_loss += r.total_loss;

                total_chars += r.total_chars;

        }



        float avg_loss = total_loss / total_chars;

        float bpc = avg_loss / std::log(2.0f);

        double chars_per_sec = total_chars / elapsed_sec;

        double mb_per_sec = chars_per_sec / (1024.0 * 1024.0);



        std::cout << "\n=== MULTI-THREADED RESULT ===" << std::endl;

        std::cout << "Threads: " << n_threads << std::endl;

        std::cout << "Reads processed: " << n_reads_to_test << std::endl;

        std::cout << "Total chars: " << total_chars << std::endl;

        std::cout << "Elapsed: " << elapsed_sec << "s" << std::endl;

        std::cout << "Throughput: " << chars_per_sec << " chars/sec = " << mb_per_sec << " MB/s" << std::endl;

        std::cout << "bpc: " << bpc << std::endl;



	int batch_size_per_thread = 64;
	std::vector<std::thread> batch_threads;
	std::vector<ThreadResult> batch_results(n_threads);

	auto bt_start = std::chrono::high_resolution_clock::now();

	for (int i=0; i < n_threads;i++) {
                int start_idx = i * reads_per_thread;
                int end_idx = (i == n_threads - 1) ? n_reads_to_test : start_idx + reads_per_thread;
                batch_threads.emplace_back(process_read_batch, std::ref(model), std::ref(qual_stoi), std::ref(base_stoi),
		std::ref(reads), start_idx, end_idx, batch_size_per_thread, std::ref(batch_results[i]));
	}
	for (auto& t : batch_threads) {
                t.join();
        }

        auto bt_end = std::chrono::high_resolution_clock::now();
        double bt_elapsed = std::chrono::duration<double>(bt_end - bt_start).count();

        float bt_total_loss = 0.0f;
        long bt_total_chars = 0;
        for (const auto& r : batch_results) {
                bt_total_loss += r.total_loss;
                bt_total_chars += r.total_chars;
        }

        float bt_avg_loss = bt_total_loss / bt_total_chars;
        float bt_bpc = bt_avg_loss / std::log(2.0f);
        double bt_chars_per_sec = bt_total_chars / bt_elapsed;
        double bt_mb_per_sec = bt_chars_per_sec / (1024.0 * 1024.0);

        std::cout << "\n=== BATCHED MULTI-THREADED RESULT ===" << std::endl;
        std::cout << "Threads: " << n_threads << "  Batch size: " << batch_size_per_thread << std::endl;
        std::cout << "Reads processed: " << n_reads_to_test << std::endl;
        std::cout << "Total chars: " << bt_total_chars << std::endl;
        std::cout << "Elapsed: " << bt_elapsed << "s" << std::endl;
        std::cout << "Throughput: " << bt_chars_per_sec << " chars/sec = " << bt_mb_per_sec << " MB/s" << std::endl;
        std::cout << "bpc: " << bt_bpc << std::endl;




	// --- Precomputed-embedding batched run ---

        std::vector<std::thread> v2_threads;

        std::vector<ThreadResult> v2_results(n_threads);



        auto v2_start = std::chrono::high_resolution_clock::now();



        for (int i = 0; i < n_threads; i++) {

                int start_idx = i * reads_per_thread;

                int end_idx = (i == n_threads - 1) ? n_reads_to_test : start_idx + reads_per_thread;

                v2_threads.emplace_back(process_read_batch_v2, std::ref(model), std::ref(qual_stoi), std::ref(base_stoi),

                                         std::ref(reads), start_idx, end_idx, batch_size_per_thread, std::ref(v2_results[i]));

        }



        for (auto& t : v2_threads) {

                t.join();

        }



        auto v2_end = std::chrono::high_resolution_clock::now();

        double v2_elapsed = std::chrono::duration<double>(v2_end - v2_start).count();



        float v2_total_loss = 0.0f;

        long v2_total_chars = 0;

        for (const auto& r : v2_results) {

                v2_total_loss += r.total_loss;

                v2_total_chars += r.total_chars;

        }



        float v2_avg_loss = v2_total_loss / v2_total_chars;

        float v2_bpc = v2_avg_loss / std::log(2.0f);

        double v2_chars_per_sec = v2_total_chars / v2_elapsed;

        double v2_mb_per_sec = v2_chars_per_sec / (1024.0 * 1024.0);



        std::cout << "\n=== PRECOMPUTED-EMBEDDING BATCHED RESULT ===" << std::endl;

        std::cout << "Threads: " << n_threads << "  Batch size: " << batch_size_per_thread << std::endl;

        std::cout << "Reads processed: " << n_reads_to_test << std::endl;

        std::cout << "Total chars: " << v2_total_chars << std::endl;

        std::cout << "Elapsed: " << v2_elapsed << "s" << std::endl;

        std::cout << "Throughput: " << v2_chars_per_sec << " chars/sec = " << v2_mb_per_sec << " MB/s" << std::endl;

        std::cout << "bpc: " << v2_bpc << std::endl;

        return 0;

}
