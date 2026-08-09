// encode_read.cpp — Phase 3: one real read, QualGRU driving the distributions.

//

// This is the first point where the model and the coder meet. The coder itself

// is unchanged from rans_test.cpp / rans_adversarial.cpp, which already proved

// it round-trips. What is new is that the frequency table is regenerated at

// every timestep from the model's output instead of being fixed.

//

// Exit test: bits/symbol should land close to the bpc the benchmarks reported

// for this model (3.744 for h64, 3.678 for h256, 3.792 for h32). A large

// discrepancy means the quantisation or the symbol/table pairing is wrong.

//

// Build (from repo root, on courtyard):

//   g++ -O2 -std=c++17 -march=native -DNDEBUG -I /usr/include/eigen3 -I inference \

//       encode_read.cpp -o encode_read

// Run:

//   ./encode_read weights/h64 weights/qual_vocab.txt data/paired_chr20_10k.txt 64 0



#include <cstdint>

#include <cmath>

#include <cstdio>

#include <cassert>

#include <vector>

#include <string>

#include <fstream>

#include <sstream>

#include <numeric>

#include <algorithm>

#include <unordered_map>

#include <chrono>


#include "gru_cell.hpp"



static constexpr uint32_t PROB_BITS = 16;

static constexpr uint32_t M         = 1u << PROB_BITS;

static constexpr uint32_t LOWER     = 1u << 23;



static inline uint32_t x_max(uint32_t f_s) {

    return ((LOWER >> PROB_BITS) << 8) * f_s;

}



// Vocabulary size is no longer a compile-time constant -- it comes from the

// vocab file (50 for lossless, 4 for the binned lossy models).

struct FreqTable {

    std::vector<uint32_t> freq;

    std::vector<uint32_t> cum;    // cum[s] = start of s; cum[n] == M

};



// - We use floors and largest remainders when converting the probabilities into

//   frequencies, we also calculate the cumulative frequencies. We also enforce f>=1 for all s

static FreqTable quantise(const Eigen::VectorXf &probs) {

    int n = (int)probs.size();

    FreqTable t;

    t.freq.assign(n, 0);

    t.cum.assign(n + 1, 0);



    std::vector<double> scaled(n);

    uint32_t total = 0;

    for (int s = 0; s < n; s++) {

        scaled[s] = (double)probs[s] * M;

        t.freq[s] = (uint32_t)scaled[s];

        if (t.freq[s] < 1) t.freq[s] = 1;

        total += t.freq[s];

    }



    if (total < M) {

        std::vector<int> order(n);

        std::iota(order.begin(), order.end(), 0);

        std::sort(order.begin(), order.end(), [&](int a, int b) {

            return (scaled[a] - (double)(uint32_t)scaled[a]) >

                   (scaled[b] - (double)(uint32_t)scaled[b]);

        });

        for (uint32_t i = 0; total < M; i++, total++)

            t.freq[order[i % n]]++;

    } else {

        while (total > M) {

            int biggest = 0;

            for (int s = 1; s < n; s++)

                if (t.freq[s] > t.freq[biggest]) biggest = s;

            if (t.freq[biggest] <= 1) break;

            t.freq[biggest]--;

            total--;

        }

    }



    t.cum[0] = 0;

    for (int s = 0; s < n; s++) t.cum[s + 1] = t.cum[s] + t.freq[s];

    assert(t.cum[n] == M);

    return t;

}



// - At each timestep, the frequency table and the encoding formula are used to

//   push each symbol into the state.

//     - If the state is going to exceed the upper bound if the next symbol is

//       pushed then we write to the output buffer

static inline void encode_symbol(uint32_t &x, uint8_t **ptr, int s, const FreqTable &t) {

    uint32_t f_s = t.freq[s];

    uint32_t c_s = t.cum[s];

    uint32_t bound = x_max(f_s);

    while (x >= bound) {

        *--(*ptr) = (uint8_t)(x & 0xff);

        x >>= 8;

    }

    // use encode formula x = M*floor(x/fs) + (x%fs) + cs

    x = ((x / f_s) << PROB_BITS) + (x % f_s) + c_s;

}

static inline int decode_symbol(uint32_t &x, const uint8_t **ptr, const FreqTable &t, int n) {

    uint32_t slot = x & (M - 1);

    int s = 0;

    while (s < n - 1 && t.cum[s + 1] <= slot) s++;



    uint32_t f_s = t.freq[s];

    uint32_t c_s = t.cum[s];



    x = f_s * (x >> PROB_BITS) + slot - c_s;



    while (x < LOWER) {

        x = (x << 8) | **ptr;

        (*ptr)++;

    }

    return s;

}



struct ReadPair { std::string bases, quals; };



static std::vector<ReadPair> load_paired_file(const std::string &path, int max_reads) {

    std::vector<ReadPair> reads;

    std::ifstream f(path);

    std::string line;

    while (std::getline(f, line)) {

        size_t tab = line.find('\t');

        if (tab == std::string::npos) continue;

        reads.push_back({line.substr(0, tab), line.substr(tab + 1)});

        if (max_reads > 0 && (int)reads.size() >= max_reads) break;

    }

    return reads;

}

struct ReadResult {

    long   n_symbols;      // len - 1, the number actually coded

    long   buffer_bytes;

    double model_bits;

    bool   roundtrip_ok;

};



static ReadResult process_read(QualGRU &model,

                                const std::vector<int> &q_idx,

                                const std::vector<int> &b_idx,

                                int hidden_size, int vocab_size,

                                bool verify) {

    int len = (int)q_idx.size();

    int qual_emb_dim = model.cell.qual_emb_dim;

    int base_emb_dim = model.cell.base_emb_dim;

    int input_dim    = qual_emb_dim + 2 * base_emb_dim;



    GRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim);

    Eigen::VectorXf h = Eigen::VectorXf::Zero(hidden_size);



    std::vector<FreqTable> tables;

    tables.reserve(len - 1);

    double model_bits = 0.0;



    for (int t = 0; t < len - 1; t++) {

        model.cell.forward_into(q_idx[t], b_idx[t], b_idx[t + 1], h, buf);

        h = buf.h_new;

        Eigen::VectorXf probs = model.predict_probs(h);

        model_bits += -std::log((double)probs[q_idx[t + 1]]) / std::log(2.0);

        tables.push_back(quantise(probs));

    }



    std::vector<uint8_t> out(len * 8 + 4096);

    uint8_t *ptr = out.data() + out.size();

    uint32_t x = LOWER;

    for (int t = len - 2; t >= 0; t--)

        encode_symbol(x, &ptr, q_idx[t + 1], tables[t]);



    ReadResult r;

    r.n_symbols    = len - 1;

    r.buffer_bytes = (long)((out.data() + out.size()) - ptr);

    r.model_bits   = model_bits;

    r.roundtrip_ok = true;



    if (verify) {

        std::vector<int> decoded(len);

        decoded[0] = q_idx[0];

        GRUBuffers dbuf(hidden_size, input_dim, qual_emb_dim, base_emb_dim);

        Eigen::VectorXf dh = Eigen::VectorXf::Zero(hidden_size);

        uint32_t dx = x;

        const uint8_t *rptr = ptr;

        for (int t = 0; t < len - 1; t++) {

            model.cell.forward_into(decoded[t], b_idx[t], b_idx[t + 1], dh, dbuf);

            dh = dbuf.h_new;

            FreqTable table = quantise(model.predict_probs(dh));

            decoded[t + 1] = decode_symbol(dx, &rptr, table, vocab_size);

        }

        r.roundtrip_ok = (decoded == q_idx) && (dx == LOWER);

    }

    return r;

}



int main(int argc, char **argv) {

    if (argc < 6) {

        fprintf(stderr,

            "usage: %s <weights_dir> <vocab_file> <paired_data> <hidden_size> <read_index>\n"

            "  e.g. %s weights/h64 weights/qual_vocab.txt data/paired_chr20_10k.txt 64 0\n",

            argv[0], argv[0]);

        return 1;

    }

    std::string weights_dir = argv[1];

    std::string vocab_file  = argv[2];

    std::string data_file   = argv[3];

    int hidden_size         = std::stoi(argv[4]);




    // vocabulary

    std::ifstream vf(vocab_file);

    std::string qual_vocab((std::istreambuf_iterator<char>(vf)),

                            std::istreambuf_iterator<char>());

    while (!qual_vocab.empty() &&

           (qual_vocab.back() == '\n' || qual_vocab.back() == '\r'))

        qual_vocab.pop_back();

    int vocab_size = (int)qual_vocab.size();



    std::unordered_map<char,int> qual_stoi;

    for (int i = 0; i < vocab_size; i++) qual_stoi[qual_vocab[i]] = i;



    std::string base_vocab = "ACGTN";

    std::unordered_map<char,int> base_stoi;

    for (int i = 0; i < (int)base_vocab.size(); i++) base_stoi[base_vocab[i]] = i;



    QualGRU model(weights_dir, vocab_size, hidden_size, 32, 8);

    int n_reads = std::stoi(argv[5]);          // now a COUNT, not an index

    bool verify = (argc > 6 && std::string(argv[6]) == "verify");



    auto reads = load_paired_file(data_file, n_reads);

    printf("loaded %zu reads, verify=%s\n\n", reads.size(), verify ? "yes" : "no");



    long   total_symbols = 0, total_bytes = 0, n_failed = 0;

    double total_model_bits = 0.0;

    auto   t_start = std::chrono::high_resolution_clock::now();



    for (size_t i = 0; i < reads.size(); i++) {

        const std::string &bases = reads[i].bases;

        const std::string &quals = reads[i].quals;

        int len = (int)quals.size();

        if (len < 2) continue;



        std::vector<int> q_idx(len), b_idx(len);

        bool ok = true;

        for (int t = 0; t < len; t++) {

            auto qi = qual_stoi.find(quals[t]);

            if (qi == qual_stoi.end()) { ok = false; break; }

            q_idx[t] = qi->second;

            auto bi = base_stoi.find(bases[t]);

            b_idx[t] = (bi == base_stoi.end()) ? base_stoi['N'] : bi->second;

        }

        if (!ok) { printf("read %zu: out-of-vocab char, skipped\n", i); continue; }



        ReadResult r = process_read(model, q_idx, b_idx, hidden_size, vocab_size, verify);

        total_symbols    += r.n_symbols;

        total_bytes      += r.buffer_bytes;

        total_model_bits += r.model_bits;

        if (!r.roundtrip_ok) { n_failed++; printf("read %zu: ROUND-TRIP FAILED\n", i); }



        if ((i + 1) % 100 == 0)

            printf("  %zu reads, %ld symbols, %ld bytes\n", i + 1, total_symbols, total_bytes);

    }



    double elapsed = std::chrono::duration<double>(

        std::chrono::high_resolution_clock::now() - t_start).count();



    // 32 bits of state + 8 bits of symbol[0] per read

    double overhead_bits = 40.0 * reads.size();

    double coder_bits    = total_bytes * 8.0 + overhead_bits;



    printf("\n=== AGGREGATE ===\n");

    printf("reads:            %zu\n", reads.size());

    printf("symbols coded:    %ld\n", total_symbols);

    printf("compressed bytes: %ld  (+%.0f bits framing)\n", total_bytes, overhead_bits);

    printf("\n");

    printf("model bpc:        %.4f\n", total_model_bits / total_symbols);

    printf("coder bpc:        %.4f\n", coder_bits / total_symbols);

    printf("coder overhead:   %+.3f%%\n",

            100.0 * (coder_bits / total_symbols - total_model_bits / total_symbols)

                  / (total_model_bits / total_symbols));

    printf("\n");

    printf("vs CRAM 3.9505:   %+.2f%%\n", 100.0 * (coder_bits / total_symbols - 3.9505) / 3.9505);

    printf("vs CoLoRd 4.0816: %+.2f%%\n", 100.0 * (coder_bits / total_symbols - 4.0816) / 4.0816);

    printf("\n");

    printf("elapsed:          %.1fs  (%.0f symbols/sec)\n", elapsed, total_symbols / elapsed);

    if (verify) printf("round-trip:       %s\n", n_failed == 0 ? "ALL PASS" : "FAILURES");
    return 0;
}
