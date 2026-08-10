// qualgru_chunked.cpp — chunked compress/decompress, standalone for testing.

//

// Same coder, same model interface, same threading pattern as qualgru.cpp.

// The only thing this changes is the FILE LAYOUT: reads are processed in

// blocks of CHUNK_SIZE, each block fully self-contained (own tables, own

// zlib'd names/sequences, own quality bytes), written and freed before the

// next block starts. Memory is bounded by chunk size, not file size.

//

//   [outer header]

//   [chunk 0: chunk header | tables | names(z) | seqs(z) | quals]

//   [chunk 1: ...]

//   ...

//

// This is deliberately a SEPARATE binary from qualgru.cpp. Compare its ratio

// and speed against the unchunked format on the same input before deciding

// whether to merge it in.

//

// Build (from compressor_ont):

//   g++ -O2 -std=c++17 -msse4.2 -DNDEBUG -I /usr/include/eigen3 -I . \

//       -I compressor/include compressor/src/qualgru_chunked.cpp \

//       -o compressor/qualgru_chunked -lz -lpthread



#include <cstdint>

#include <cmath>

#include <cstdio>

#include <cstring>

#include <cstdlib>

#include <vector>

#include <string>

#include <fstream>

#include <numeric>

#include <algorithm>

#include <unordered_map>

#include <chrono>

#include <thread>

#include <atomic>

#include <unistd.h>

#include <libgen.h>



#include "gru_cell.hpp"

#include "format.hpp"



static constexpr uint32_t PROB_BITS = 16;

static constexpr uint32_t M         = 1u << PROB_BITS;

static constexpr uint32_t LOWER     = 1u << 23;

static constexpr size_t   CHUNK_SIZE = 10000;



static inline uint32_t x_max(uint32_t f_s) {

    return ((LOWER >> PROB_BITS) << 8) * f_s;

}



// ---------------------------------------------------------------------------

// Coder — identical to qualgru.cpp. quantise() must stay byte-identical

// between compress and decompress or the stream desyncs.

// ---------------------------------------------------------------------------



struct FreqTable {

    std::vector<uint32_t> freq;

    std::vector<uint32_t> cum;

};



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

    return t;

}



static inline void encode_symbol(uint32_t &x, uint8_t **ptr, int s, const FreqTable &t) {

    uint32_t f_s = t.freq[s];

    uint32_t c_s = t.cum[s];

    uint32_t bound = x_max(f_s);

    while (x >= bound) {

        *--(*ptr) = (uint8_t)(x & 0xff);

        x >>= 8;

    }

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



// ---------------------------------------------------------------------------

// Model registry

// ---------------------------------------------------------------------------



struct ModelInfo { const char *name; int hidden; bool lossy; };

static const ModelInfo MODELS[] = {

    { "h64",         64,  false },

    { "h256",        256, false },

    { "h32",         32,  false },

    { "lossy4_h64",  64,  true  },

    { "lossy4_h256", 256, true  },

    { "lossy4_h32",  32,  true  },

};

static constexpr int N_MODELS = 6;



static int model_id_from_name(const std::string &name) {

    for (int i = 0; i < N_MODELS; i++)

        if (name == MODELS[i].name) return i;

    return -1;

}



// The lossy4_* models were trained on quality strings pre-quantized to 4
// bins matching CoLoRd's default ONT scheme (see
// training/rnn_ont_torch_v3_lossy_custom.py): edges [7, 14, 26] in Q-value
// space, bin representatives [3, 10, 18, 35]. This is the only lossy scheme
// shipped, so it's hardcoded here rather than read from per-model metadata.
static const int LOSSY4_BIN_EDGES[3] = {7, 14, 26};
static const int LOSSY4_BIN_REPS[4]  = {3, 10, 18, 35};

static inline char lossy4_quantize(char qual_char) {
    int q = (int)(unsigned char)qual_char - 33;
    int idx = (int)(std::upper_bound(LOSSY4_BIN_EDGES, LOSSY4_BIN_EDGES + 3, q)
                     - LOSSY4_BIN_EDGES);
    return (char)(LOSSY4_BIN_REPS[idx] + 33);
}



static std::string exe_dir() {

    char buf[4096];

    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);

    if (n <= 0) return ".";

    buf[n] = '\0';

    return std::string(dirname(buf));

}



// ---------------------------------------------------------------------------

// FASTQ — a STATEFUL reader. Unlike qualgru.cpp's read_fastq (which slurps

// the whole file), this keeps the gzFile open across calls and hands back

// one chunk at a time. That's the core change chunking requires: the reader

// itself must be incremental.

// ---------------------------------------------------------------------------



struct FastqRead {

    std::string name;

    std::string seq;

    std::string qual;

};



class FastqReader {

public:

    explicit FastqReader(const std::string &path) {

        f_ = gzopen(path.c_str(), "rb");

        ok_ = (f_ != nullptr);

        line_.resize(BUFSZ);

    }

    ~FastqReader() { if (f_) gzclose(f_); }



    bool good() const { return ok_; }



    // Fills `out` with up to CHUNK_SIZE reads. Returns false only on a read

    // error; an empty `out` with true means clean EOF.

    bool next_chunk(std::vector<FastqRead> &out, size_t max_reads) {

        out.clear();

        std::string l1, l2, l3, l4;

        while (out.size() < max_reads &&

               next_line(l1) && next_line(l2) && next_line(l3) && next_line(l4)) {

            if (l1.empty() || l1[0] != '@') {

                fprintf(stderr, "malformed FASTQ near read %zu\n", total_read_);

                return false;

            }

            if (l2.size() != l4.size()) {

                fprintf(stderr, "seq/qual length mismatch at read %zu\n", total_read_);

                return false;

            }

            out.push_back({l1.substr(1), l2, l4});

            total_read_++;

        }

        return true;

    }



private:

    static constexpr size_t BUFSZ = 1 << 16;

    gzFile f_ = nullptr;

    bool ok_ = false;

    std::vector<char> line_;

    size_t total_read_ = 0;



    bool next_line(std::string &out) {

        out.clear();

        while (true) {

            if (gzgets(f_, line_.data(), (int)BUFSZ) == nullptr)

                return !out.empty();

            out += line_.data();

            if (!out.empty() && out.back() == '\n') break;

        }

        while (!out.empty() && (out.back() == '\n' || out.back() == '\r'))

            out.pop_back();

        return true;

    }

};



// ---------------------------------------------------------------------------

// Chunk header — same fields as the old whole-file Header, scoped to one

// chunk. Written/read exactly like format.hpp's Header, same discipline:

// fixed field order is the entire contract.

// ---------------------------------------------------------------------------



struct ChunkHeader {

    uint32_t n_reads;

    uint64_t names_z_len;

    uint64_t names_uncompressed;

    uint64_t seq_z_len;

    uint64_t seq_uncompressed;

    uint64_t qual_block_len;

};



static constexpr size_t CHUNK_HEADER_SIZE = 4 + 8 + 8 + 8 + 8 + 8;



static void write_chunk_header(std::vector<uint8_t> &out, const ChunkHeader &c) {

    write_u32(out, c.n_reads);

    write_u64(out, c.names_z_len);

    write_u64(out, c.names_uncompressed);

    write_u64(out, c.seq_z_len);

    write_u64(out, c.seq_uncompressed);

    write_u64(out, c.qual_block_len);

}



static bool read_chunk_header(const uint8_t *&ptr, size_t available, ChunkHeader &c) {

    if (available < CHUNK_HEADER_SIZE) return false;

    c.n_reads             = read_u32(ptr);

    c.names_z_len         = read_u64(ptr);

    c.names_uncompressed  = read_u64(ptr);

    c.seq_z_len            = read_u64(ptr);

    c.seq_uncompressed     = read_u64(ptr);

    c.qual_block_len       = read_u64(ptr);

    return true;

}



// ---------------------------------------------------------------------------

// Outer archive header. Same shape as format.hpp's Header, minus the

// per-file block-length fields (those are per-CHUNK now) and plus a chunk

// count so the decoder knows when to stop.

// ---------------------------------------------------------------------------



struct ArchiveHeader {

    uint8_t  version;

    uint8_t  model_id;

    uint8_t  flags;

    uint64_t n_reads;

    uint64_t n_symbols;

    uint64_t n_chunks;

    uint32_t crc_quals;

    uint32_t crc_seqs;

};



static constexpr size_t ARCHIVE_HEADER_SIZE = 4 + 1 + 1 + 1 + 8 + 8 + 8 + 4 + 4;



static void write_archive_header(std::vector<uint8_t> &out, const ArchiveHeader &h) {

    // distinct magic: this is NOT the same format as .qgru

    out.push_back('Q'); out.push_back('G'); out.push_back('R'); out.push_back('U');

    out.push_back(h.version);

    out.push_back(h.model_id);

    out.push_back(h.flags);

    write_u64(out, h.n_reads);

    write_u64(out, h.n_symbols);

    write_u64(out, h.n_chunks);

    write_u32(out, h.crc_quals);

    write_u32(out, h.crc_seqs);

}



static bool read_archive_header(const uint8_t *&ptr, size_t available, ArchiveHeader &h) {

    if (available < ARCHIVE_HEADER_SIZE) return false;

    if (ptr[0] != 'Q' || ptr[1] != 'G' || ptr[2] != 'R' || ptr[3] != 'U') return false;

    ptr += 4;

    h.version = *ptr++;

    if (h.version != 1) return false;

    h.model_id  = *ptr++;

    h.flags     = *ptr++;

    h.n_reads   = read_u64(ptr);

    h.n_symbols = read_u64(ptr);

    h.n_chunks  = read_u64(ptr);

    h.crc_quals = read_u32(ptr);

    h.crc_seqs  = read_u32(ptr);

    return true;

}





// ---------------------------------------------------------------------------

// compress

// ---------------------------------------------------------------------------



static int cmd_compress(const std::string &in_path,

                         const std::string &out_path,

                         const std::string &weights_dir,

                         int model_id,

                         int n_threads,

                         bool verbose) {



    int hidden_size = MODELS[model_id].hidden;
    bool lossy = MODELS[model_id].lossy;



    std::ifstream vf(weights_dir + "/vocab.txt");

    if (!vf) { fprintf(stderr, "cannot read %s/vocab.txt\n", weights_dir.c_str()); return 1; }

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



    FastqReader reader(in_path);

    if (!reader.good()) { fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }



    FILE *out = fopen(out_path.c_str(), "wb");

    if (!out) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }



    std::vector<uint8_t> placeholder(ARCHIVE_HEADER_SIZE, 0);

    fwrite(placeholder.data(), 1, ARCHIVE_HEADER_SIZE, out);



    int qual_emb_dim = model.cell.qual_emb_dim;

    int base_emb_dim = model.cell.base_emb_dim;

    int input_dim    = qual_emb_dim + 2 * base_emb_dim;



    uint32_t crc_quals = crc32(0L, Z_NULL, 0);

    uint32_t crc_seqs  = crc32(0L, Z_NULL, 0);

    uint64_t total_reads = 0, total_symbols = 0, total_chunks = 0;



    auto t0 = std::chrono::high_resolution_clock::now();



    std::vector<FastqRead> chunk_reads;

    while (true) {

        if (!reader.next_chunk(chunk_reads, CHUNK_SIZE)) { fclose(out); return 1; }

        if (chunk_reads.empty()) break;   // clean EOF



        size_t n_reads = chunk_reads.size();



        // ---- per-read slots, scoped to THIS chunk only ----

        std::vector<std::vector<uint8_t>> results(n_reads);

        std::vector<uint32_t> lengths(n_reads);

        std::vector<uint32_t> comp_lengths(n_reads);

        std::vector<uint8_t>  first_syms(n_reads);

        std::vector<uint32_t> states(n_reads);

        // The stream actually encoded -- identical to chunk_reads[r].qual in
        // lossless mode, but the post-quantization (binned) string in lossy
        // mode. Used for the CRC below so the checksum verifies against what
        // decompression can actually reproduce, not the unrecoverable original.
        std::vector<std::string> canon_quals(n_reads);

        std::atomic<bool> failed{false};

        std::atomic<size_t> next_read{0};



        auto worker = [&]() {

            GRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim);

            while (true) {

                size_t r = next_read.fetch_add(1);

                if (r >= n_reads || failed.load()) return;



                const std::string &seq  = chunk_reads[r].seq;

                const std::string &qual = chunk_reads[r].qual;

                int len = (int)qual.size();

                lengths[r] = (uint32_t)len;



                if (len < 2) {

                    if (len == 1) {
                        char qc = lossy ? lossy4_quantize(qual[0]) : qual[0];
                        int idx = (int)qual_stoi.at(qc);
                        first_syms[r]  = (uint8_t)idx;
                        canon_quals[r] = std::string(1, qual_vocab[idx]);
                    } else {
                        first_syms[r] = 0;
                    }

                    states[r]       = LOWER;

                    comp_lengths[r] = 0;

                    continue;

                }



                std::vector<int> q_idx(len), b_idx(len);

                bool ok = true;

                for (int t = 0; t < len; t++) {

                    char qc = lossy ? lossy4_quantize(qual[t]) : qual[t];

                    auto qi = qual_stoi.find(qc);

                    if (qi == qual_stoi.end()) {

                        fprintf(stderr, "quality char 0x%02x not in vocabulary\n",

                                (unsigned char)qc);

                        failed.store(true);

                        ok = false;

                        break;

                    }

                    q_idx[t] = qi->second;

                    auto bi = base_stoi.find(seq[t]);

                    b_idx[t] = (bi == base_stoi.end()) ? base_stoi.at('N') : bi->second;

                }

                if (!ok) return;



                first_syms[r] = (uint8_t)q_idx[0];



                canon_quals[r].resize(len);

                for (int t = 0; t < len; t++) canon_quals[r][t] = qual_vocab[q_idx[t]];



                Eigen::VectorXf h = Eigen::VectorXf::Zero(hidden_size);

                std::vector<FreqTable> tables;

                tables.reserve(len - 1);

                for (int t = 0; t < len - 1; t++) {

                    model.cell.forward_into(q_idx[t], b_idx[t], b_idx[t + 1], h, buf);

                    h = buf.h_new;

                    tables.push_back(quantise(model.predict_probs(h)));

                }



                std::vector<uint8_t> tmp(len * 8 + 4096);

                uint8_t *ptr = tmp.data() + tmp.size();

                uint32_t x = LOWER;

                for (int t = len - 2; t >= 0; t--)

                    encode_symbol(x, &ptr, q_idx[t + 1], tables[t]);



                size_t n = (tmp.data() + tmp.size()) - ptr;

                results[r].assign(ptr, ptr + n);

                comp_lengths[r] = (uint32_t)n;

                states[r]       = x;

            }

        };



        std::vector<std::thread> pool;

        for (int i = 0; i < n_threads; i++) pool.emplace_back(worker);

        for (auto &t : pool) t.join();

        if (failed.load()) { fclose(out); return 1; }



        // ---- assemble this chunk, in read order ----

        std::vector<uint8_t> all_bases, all_names, qual_stream;

        for (size_t r = 0; r < n_reads; r++) {

            all_bases.insert(all_bases.end(), chunk_reads[r].seq.begin(), chunk_reads[r].seq.end());

            all_names.insert(all_names.end(), chunk_reads[r].name.begin(), chunk_reads[r].name.end());

            all_names.push_back('\n');



            crc_quals = crc32(crc_quals, (const Bytef*)canon_quals[r].data(), canon_quals[r].size());

            crc_seqs  = crc32(crc_seqs,  (const Bytef*)chunk_reads[r].seq.data(),  chunk_reads[r].seq.size());



            total_symbols += chunk_reads[r].qual.size();

            qual_stream.insert(qual_stream.end(), results[r].begin(), results[r].end());

        }

        total_reads += n_reads;



        std::vector<uint8_t> tables_buf;

        for (uint32_t l : lengths)      write_varint(tables_buf, l);

        for (uint32_t c : comp_lengths) write_varint(tables_buf, c);

        for (uint8_t  s : first_syms)   tables_buf.push_back(s);

        for (uint32_t s : states)       write_u32(tables_buf, s);



        auto names_z = zlib_compress(all_names);

        auto seqs_z  = zlib_compress(all_bases);



        ChunkHeader ch{};

        ch.n_reads             = (uint32_t)n_reads;

        ch.names_z_len         = names_z.size();

        ch.names_uncompressed  = all_names.size();

        ch.seq_z_len            = seqs_z.size();

        ch.seq_uncompressed     = all_bases.size();

        ch.qual_block_len       = qual_stream.size();



        std::vector<uint8_t> chdr;

        write_chunk_header(chdr, ch);

        fwrite(chdr.data(), 1, chdr.size(), out);

        fwrite(tables_buf.data(), 1, tables_buf.size(), out);

        fwrite(names_z.data(), 1, names_z.size(), out);

        fwrite(seqs_z.data(), 1, seqs_z.size(), out);

        fwrite(qual_stream.data(), 1, qual_stream.size(), out);



        total_chunks++;

        if (verbose)

            fprintf(stderr, "  chunk %llu: %zu reads, %zu quality bytes\n",

                    (unsigned long long)total_chunks, n_reads, qual_stream.size());

        // chunk_reads, results, tables_buf, names_z, seqs_z, qual_stream all

        // go out of scope here -- freed before the next chunk is read

    }



    double elapsed = std::chrono::duration<double>(

        std::chrono::high_resolution_clock::now() - t0).count();



    ArchiveHeader ah{};

    ah.version   = 1;

    ah.model_id  = (uint8_t)model_id;

    ah.flags     = 1;

    ah.n_reads   = total_reads;

    ah.n_symbols = total_symbols;

    ah.n_chunks  = total_chunks;

    ah.crc_quals = crc_quals;

    ah.crc_seqs  = crc_seqs;



    std::vector<uint8_t> hbuf;

    write_archive_header(hbuf, ah);

    fseek(out, 0, SEEK_SET);

    fwrite(hbuf.data(), 1, hbuf.size(), out);

    fclose(out);



    long total_bytes;

    {

        std::ifstream check(out_path, std::ios::binary | std::ios::ate);

        total_bytes = (long)check.tellg();

    }



    printf("%s -> %s\n", in_path.c_str(), out_path.c_str());

    printf("  model      %s%s\n", MODELS[model_id].name, lossy ? "  (lossy, 4-bin)" : "");

    if (lossy)

        fprintf(stderr, "note: lossy mode -- decompressed quality values are quantized to "

                         "4 bins (edges Q7/14/26), not byte-identical to the input\n");

    printf("  reads      %llu  (%llu symbols, %llu chunks)\n",

            (unsigned long long)total_reads, (unsigned long long)total_symbols,

            (unsigned long long)total_chunks);

    printf("  (includes sequences + tables) total      %ld bytes  (%.4f bits/symbol overall)\n",

            total_bytes, 8.0 * total_bytes / total_symbols);

    printf("  %.1fs  (%.0f symbols/sec)\n", elapsed, total_symbols / elapsed);



    return 0;

}





// ---------------------------------------------------------------------------

// decompress

// ---------------------------------------------------------------------------



static int cmd_decompress(const std::string &in_path,

                           const std::string &out_path,

                           const std::string &weights_override,

                           int n_threads,

                           bool verbose) {



    std::ifstream fin(in_path, std::ios::binary);

    if (!fin) { fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }



    // The archive itself is still read whole here for simplicity -- chunking

    // bounds DECODED-DATA memory, which was the actual problem (decoded

    // strings are much larger than their compressed form). Streaming the

    // compressed bytes in too is a further refinement, not required for the

    // memory bound that matters.

    std::vector<uint8_t> file((std::istreambuf_iterator<char>(fin)),

                               std::istreambuf_iterator<char>());



    const uint8_t *p = file.data();

    const uint8_t *end = file.data() + file.size();



    ArchiveHeader h{};

    if (!read_archive_header(p, file.size(), h)) {

        fprintf(stderr, "%s is not a valid QCHK archive\n", in_path.c_str());

        return 1;

    }

    if (h.model_id >= N_MODELS) {

        fprintf(stderr, "archive names unknown model id %u\n", h.model_id);

        return 1;

    }



    std::string weights_dir = weights_override.empty()

        ? exe_dir() + "/models/" + MODELS[h.model_id].name

        : weights_override;

    int hidden_size = MODELS[h.model_id].hidden;



    std::ifstream vf(weights_dir + "/vocab.txt");

    if (!vf) { fprintf(stderr, "cannot read %s/vocab.txt\n", weights_dir.c_str()); return 1; }

    std::string qual_vocab((std::istreambuf_iterator<char>(vf)),

                            std::istreambuf_iterator<char>());

    while (!qual_vocab.empty() &&

           (qual_vocab.back() == '\n' || qual_vocab.back() == '\r'))

        qual_vocab.pop_back();

    int vocab_size = (int)qual_vocab.size();



    std::string base_vocab = "ACGTN";

    std::unordered_map<char,int> base_stoi;

    for (int i = 0; i < (int)base_vocab.size(); i++) base_stoi[base_vocab[i]] = i;



    QualGRU model(weights_dir, vocab_size, hidden_size, 32, 8);



    int qual_emb_dim = model.cell.qual_emb_dim;

    int base_emb_dim = model.cell.base_emb_dim;

    int input_dim    = qual_emb_dim + 2 * base_emb_dim;



    FILE *out = fopen(out_path.c_str(), "wb");

    if (!out) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }



    uint32_t crc_quals = crc32(0L, Z_NULL, 0);

    uint32_t crc_seqs  = crc32(0L, Z_NULL, 0);



    auto t0 = std::chrono::high_resolution_clock::now();



    for (uint64_t chunk_i = 0; chunk_i < h.n_chunks; chunk_i++) {

        ChunkHeader ch{};

        if (!read_chunk_header(p, end - p, ch)) {

            fprintf(stderr, "corrupt archive: bad chunk header at chunk %llu\n",

                    (unsigned long long)chunk_i);

            return 1;

        }



        uint64_t n_reads = ch.n_reads;



        std::vector<uint32_t> lengths(n_reads);

        for (uint64_t i = 0; i < n_reads; i++) lengths[i] = read_varint(p);

        std::vector<uint32_t> comp_lengths(n_reads);

        for (uint64_t i = 0; i < n_reads; i++) comp_lengths[i] = read_varint(p);

        std::vector<uint8_t> first_syms(n_reads);

        for (uint64_t i = 0; i < n_reads; i++) first_syms[i] = *p++;

        std::vector<uint32_t> states(n_reads);

        for (uint64_t i = 0; i < n_reads; i++) states[i] = read_u32(p);



        std::vector<uint8_t> names_z(p, p + ch.names_z_len);

        p += ch.names_z_len;

        auto names_flat = zlib_decompress(names_z, ch.names_uncompressed);



        std::vector<std::string> names;

        {

            std::string cur;

            for (uint8_t c : names_flat) {

                if (c == '\n') { names.push_back(cur); cur.clear(); }

                else cur.push_back((char)c);

            }

        }

        if (names.size() != n_reads) {

            fprintf(stderr, "corrupt archive: %zu names for %llu reads in chunk %llu\n",

                    names.size(), (unsigned long long)n_reads, (unsigned long long)chunk_i);

            return 1;

        }



        std::vector<uint8_t> seqs_z(p, p + ch.seq_z_len);

        p += ch.seq_z_len;

        auto all_bases = zlib_decompress(seqs_z, ch.seq_uncompressed);



        const uint8_t *qual_block = p;

        p += ch.qual_block_len;   // advance past this chunk's quality bytes now



        std::vector<uint64_t> offsets(n_reads);

        {

            uint64_t acc = 0;

            for (uint64_t r = 0; r < n_reads; r++) { offsets[r] = acc; acc += comp_lengths[r]; }

            if (acc != ch.qual_block_len) {

                fprintf(stderr, "corrupt archive: chunk %llu comp_lengths mismatch\n",

                        (unsigned long long)chunk_i);

                return 1;

            }

        }

        std::vector<uint64_t> seq_offsets(n_reads);

        {

            uint64_t acc = 0;

            for (uint64_t r = 0; r < n_reads; r++) { seq_offsets[r] = acc; acc += lengths[r]; }

        }



        std::vector<std::string> out_quals(n_reads), out_seqs(n_reads);

        std::atomic<uint64_t> next_read{0};



        auto worker = [&]() {

            GRUBuffers buf(hidden_size, input_dim, qual_emb_dim, base_emb_dim);

            while (true) {

                uint64_t r = next_read.fetch_add(1);

                if (r >= n_reads) return;



                int len = (int)lengths[r];

                std::string seq((const char*)all_bases.data() + seq_offsets[r], len);



                std::vector<int> b_idx(len);

                for (int t = 0; t < len; t++) {

                    auto bi = base_stoi.find(seq[t]);

                    b_idx[t] = (bi == base_stoi.end()) ? base_stoi.at('N') : bi->second;

                }



                std::string qual(len, '\0');

                if (len >= 1) qual[0] = qual_vocab[first_syms[r]];



                if (len >= 2) {

                    const uint8_t *qptr = qual_block + offsets[r];

                    uint32_t x = states[r];

                    std::vector<int> decoded(len);

                    decoded[0] = first_syms[r];

                    Eigen::VectorXf dh = Eigen::VectorXf::Zero(hidden_size);



                    for (int t = 0; t < len - 1; t++) {

                        model.cell.forward_into(decoded[t], b_idx[t], b_idx[t + 1], dh, buf);

                        dh = buf.h_new;

                        FreqTable table = quantise(model.predict_probs(dh));

                        decoded[t + 1] = decode_symbol(x, &qptr, table, vocab_size);

                        qual[t + 1] = qual_vocab[decoded[t + 1]];

                    }

                }



                out_seqs[r]  = std::move(seq);

                out_quals[r] = std::move(qual);

            }

        };



        std::vector<std::thread> pool;

        for (int i = 0; i < n_threads; i++) pool.emplace_back(worker);

        for (auto &t : pool) t.join();



        for (uint64_t r = 0; r < n_reads; r++) {

            crc_quals = crc32(crc_quals, (const Bytef*)out_quals[r].data(), out_quals[r].size());

            crc_seqs  = crc32(crc_seqs,  (const Bytef*)out_seqs[r].data(),  out_seqs[r].size());

            fprintf(out, "@%s\n%s\n+\n%s\n", names[r].c_str(), out_seqs[r].c_str(), out_quals[r].c_str());

        }



        if (verbose)

            fprintf(stderr, "  chunk %llu/%llu: %llu reads decoded\n",

                    (unsigned long long)(chunk_i + 1), (unsigned long long)h.n_chunks,

                    (unsigned long long)n_reads);

        // lengths, comp_lengths, out_quals, out_seqs, all_bases etc. all go

        // out of scope here -- freed before the next chunk is read

    }

    fclose(out);



    double elapsed = std::chrono::duration<double>(

        std::chrono::high_resolution_clock::now() - t0).count();



    bool ok = (crc_quals == h.crc_quals) && (crc_seqs == h.crc_seqs);



    printf("%s -> %s\n", in_path.c_str(), out_path.c_str());

    printf("  model      %s%s\n", MODELS[h.model_id].name,

            MODELS[h.model_id].lossy ? "  (lossy, 4-bin)" : "");

    printf("  reads      %llu  (%llu symbols, %llu chunks)\n",

            (unsigned long long)h.n_reads, (unsigned long long)h.n_symbols,

            (unsigned long long)h.n_chunks);

    printf("  checksum   %s\n", ok ? "OK" : "FAILED");

    printf("  %.1fs  (%.0f symbols/sec)\n", elapsed, h.n_symbols / elapsed);



    if (!ok) {

        fprintf(stderr, "\nCHECKSUM MISMATCH -- output does not match the original.\n");

        return 1;

    }

    return 0;

}





// ---------------------------------------------------------------------------



static void usage() {

    fprintf(stderr,

        "qualgru_chunked -- standalone test of chunked compression\n"

        "\n"

        "  qualgru_chunked compress   <in.fastq[.gz]> <out.qchk> [--model NAME] [--threads N]\n"

        "  qualgru_chunked decompress <in.qchk>       <out.fastq> [--threads N]\n"

        "\n"

        "Models: h64 (default), h256, h32 -- lossless, byte-identical round trip.\n"

        "        lossy4_h64, lossy4_h256, lossy4_h32 -- 4-bin quantized (CoLoRd's\n"

        "        default ONT scheme, edges Q7/14/26); decompressed output matches\n"

        "        the binned quality values, not the original bytes.\n"

        "\n"

        "Archives use a DIFFERENT format from .qgru (magic QCHK, not QGRU) --\n"

        "the two are not interchangeable. This binary exists to compare ratio\n"

        "and speed against the unchunked format before merging.\n");

}



int main(int argc, char **argv) {

    if (argc < 4) { usage(); return 1; }



    std::string cmd = argv[1];

    if (cmd != "compress" && cmd != "decompress") {

        fprintf(stderr, "unknown command '%s'\n\n", cmd.c_str());

        usage();

        return 1;

    }



    std::string in_path  = argv[2];

    std::string out_path = argv[3];

    std::string model_name = "h64";

    std::string weights_override;

    int n_threads = 4;

    bool verbose = false;



    for (int i = 4; i < argc; i++) {

        std::string a = argv[i];

        if (a == "--model" && i + 1 < argc)        model_name = argv[++i];

        else if (a == "--weights" && i + 1 < argc) weights_override = argv[++i];

        else if (a == "--threads" && i + 1 < argc) n_threads = std::stoi(argv[++i]);

        else if (a == "-v" || a == "--verbose")    verbose = true;

        else { fprintf(stderr, "unknown option '%s'\n", a.c_str()); return 1; }

    }

    if (n_threads == 0) n_threads = (int)std::thread::hardware_concurrency();

    if (n_threads < 1)  n_threads = 1;



    if (cmd == "compress") {

        int model_id = model_id_from_name(model_name);

        if (model_id < 0) {

            fprintf(stderr, "unknown model '%s'\n", model_name.c_str());

            return 1;

        }

        std::string weights_dir = weights_override.empty()

            ? exe_dir() + "/models/" + model_name

            : weights_override;

        return cmd_compress(in_path, out_path, weights_dir, model_id, n_threads, verbose);

    }



    return cmd_decompress(in_path, out_path, weights_override, n_threads, verbose);

}
