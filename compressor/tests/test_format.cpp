#include "format.hpp"

#include <cstdio>

#include <vector>

#include <string>

#include <fstream>



int main() {

    // -----------------------------------------------------------------------

    // 1. every value round-trips

    // -----------------------------------------------------------------------

    for (uint32_t v = 0; v <= 100000; v++) {

        std::vector<uint8_t> buf;

        write_varint(buf, v);

        const uint8_t *p = buf.data();

        uint32_t got = read_varint(p);

        if (got != v) { printf("FAIL: wrote %u, read %u\n", v, got); return 1; }

    }

    printf("single round-trip 0..100000: PASS\n");



    // -----------------------------------------------------------------------

    // 2. many varints back to back, read sequentially

    //    this is what catches a broken pointer advance

    // -----------------------------------------------------------------------

    std::vector<uint32_t> vals = {0, 1, 127, 128, 300, 16383, 16384, 50000, 2097151};

    std::vector<uint8_t> buf;

    for (uint32_t v : vals) write_varint(buf, v);



    const uint8_t *p = buf.data();

    for (uint32_t v : vals) {

        uint32_t got = read_varint(p);

        if (got != v) { printf("FAIL in sequence: wrote %u, read %u\n", v, got); return 1; }

    }

    printf("sequential round-trip: PASS (%zu values in %zu bytes)\n", vals.size(), buf.size());



    // -----------------------------------------------------------------------

    // 3. sizes are what we expect

    // -----------------------------------------------------------------------

    auto sz = [](uint32_t v) { std::vector<uint8_t> b; write_varint(b, v); return b.size(); };

    printf("size check: 127->%zu  128->%zu  16383->%zu  16384->%zu\n",

            sz(127), sz(128), sz(16383), sz(16384));



    // -----------------------------------------------------------------------

    // 4. fixed-width integers -- max values catch cast/shift bugs

    // -----------------------------------------------------------------------

    std::vector<uint8_t> ibuf;

    write_u32(ibuf, 0);

    write_u32(ibuf, 10000);

    write_u32(ibuf, 0xFFFFFFFF);

    write_u64(ibuf, 0);

    write_u64(ibuf, 186198600);

    write_u64(ibuf, 0xFFFFFFFFFFFFFFFFull);



    const uint8_t *ip = ibuf.data();

    bool ok = read_u32(ip) == 0

           && read_u32(ip) == 10000

           && read_u32(ip) == 0xFFFFFFFF

           && read_u64(ip) == 0

           && read_u64(ip) == 186198600

           && read_u64(ip) == 0xFFFFFFFFFFFFFFFFull;

    printf("u32/u64 round-trip: %s (%zu bytes)\n", ok ? "PASS" : "FAIL", ibuf.size());



    // -----------------------------------------------------------------------

    // 5. header round-trip -- every field must survive

    // -----------------------------------------------------------------------

    Header hw{};

    hw.version              = 1;

    hw.model_id             = 0;

    hw.flags                = 1;

    hw.n_reads              = 10000;

    hw.n_symbols            = 186198600;

    hw.crc_quals            = 0xDEADBEEF;

    hw.crc_seqs             = 0x12345678;

    hw.seq_block_len        = 50000000;

    hw.seq_uncompressed_len = 186198600;

    hw.qual_block_len       = 87328334;



    std::vector<uint8_t> hbuf;

    write_header(hbuf, hw);



    Header hr{};

    const uint8_t *hp = hbuf.data();

    bool hok = read_header(hp, hbuf.size(), hr);



    bool fields_ok = hok

        && hr.version              == hw.version

        && hr.model_id             == hw.model_id

        && hr.flags                == hw.flags

        && hr.n_reads              == hw.n_reads

        && hr.n_symbols            == hw.n_symbols

        && hr.crc_quals            == hw.crc_quals

        && hr.crc_seqs             == hw.crc_seqs

        && hr.seq_block_len        == hw.seq_block_len

        && hr.seq_uncompressed_len == hw.seq_uncompressed_len

        && hr.qual_block_len       == hw.qual_block_len;



    printf("header: %s (%zu bytes, expected %zu)\n",

            fields_ok ? "PASS" : "FAIL", hbuf.size(), HEADER_SIZE);



    // the pointer must have advanced exactly HEADER_SIZE

    printf("header ptr advance: %s (%td)\n",

            (hp - hbuf.data()) == (ptrdiff_t)HEADER_SIZE ? "PASS" : "FAIL",

            hp - hbuf.data());



    // -----------------------------------------------------------------------

    // 6. rejection cases -- a read_header that accepts anything is worse

    //    than none at all

    // -----------------------------------------------------------------------

    std::vector<uint8_t> bad = hbuf;

    bad[0] = 'X';

    const uint8_t *bp = bad.data();

    printf("bad magic rejected:  %s\n", !read_header(bp, bad.size(), hr) ? "PASS" : "FAIL");



    const uint8_t *tp = hbuf.data();

    printf("truncated rejected:  %s\n", !read_header(tp, 10, hr) ? "PASS" : "FAIL");



    std::vector<uint8_t> badver = hbuf;

    badver[4] = 99;                       // version byte

    const uint8_t *vp = badver.data();

    printf("bad version rejected: %s\n", !read_header(vp, badver.size(), hr) ? "PASS" : "FAIL");



    // -----------------------------------------------------------------------

    // 7. zlib -- synthetic, then real sequence data

    // -----------------------------------------------------------------------

    {

        std::vector<uint8_t> orig;

        for (int i = 0; i < 100000; i++) orig.push_back("ACGT"[i % 4]);



        auto comp = zlib_compress(orig);

        auto back = zlib_decompress(comp, orig.size());



        printf("\nzlib (synthetic ACGT): %s  %zu -> %zu bytes (%.2f%%)\n",

                back == orig ? "PASS" : "FAIL",

                orig.size(), comp.size(),

                100.0 * comp.size() / orig.size());

    }



    // Real sequence data -- the synthetic case above compresses absurdly well

    // and tells you nothing about the ratio you will actually see. This is the

    // number that determines how much of the archive the sequence block eats.

    {

        std::ifstream f("data/paired_chr20_10k.txt");

        std::vector<uint8_t> seqs;

        std::string line;

        int n = 0;

        while (std::getline(f, line) && n < 200) {

            size_t tab = line.find('\t');

            if (tab == std::string::npos) continue;

            seqs.insert(seqs.end(), line.begin(), line.begin() + tab);

            n++;

        }



        if (seqs.empty()) {

            printf("zlib (real bases): SKIPPED (run from compressor_ont, "

                    "data/paired_chr20_10k.txt not found)\n");

        } else {

            auto comp = zlib_compress(seqs);

            auto back = zlib_decompress(comp, seqs.size());

            printf("zlib (real bases):     %s  %zu -> %zu bytes (%.2f%%, %.3f bits/base)\n",

                    back == seqs ? "PASS" : "FAIL",

                    seqs.size(), comp.size(),

                    100.0 * comp.size() / seqs.size(),

                    8.0 * comp.size() / seqs.size());

        }

    }



    return 0;

}
