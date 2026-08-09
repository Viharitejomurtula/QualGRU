// rans_adversarial.cpp — Phase 1b: adversarial tests for the rANS coder.

//

// rans_test.cpp proved the coder works on a well-behaved Zipf distribution.

// This file tries to break it. Each case targets a specific way the frequency

// quantiser or the renormalisation can fail, and every one of these is a

// distribution QualGRU could plausibly produce on real data.

//

// Build: g++ -O2 -std=c++17 rans_adversarial.cpp -o rans_adversarial



#include <cstdint>

#include <cmath>

#include <cstdio>

#include <cassert>

#include <vector>

#include <random>

#include <numeric>

#include <algorithm>

#include <string>



static constexpr uint32_t PROB_BITS = 16;

static constexpr uint32_t M         = 1u << PROB_BITS;

static constexpr uint32_t LOWER     = 1u << 23;

static constexpr int      N_SYMBOLS = 50;



static inline uint32_t x_max(uint32_t f_s) {

    return ((LOWER >> PROB_BITS) << 8) * f_s;

}



struct FreqTable {

    uint32_t freq[N_SYMBOLS];

    uint32_t cum[N_SYMBOLS + 1];

};



static FreqTable quantise(const double *probs, int n) {

    FreqTable t{};



    double scaled[N_SYMBOLS];

    uint32_t total = 0;

    for (int s = 0; s < n; s++) {

        scaled[s] = probs[s] * M;

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



    total = 0;

    for (int s = 0; s < n; s++) total += t.freq[s];



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

// Test harness

// ---------------------------------------------------------------------------



struct Result { bool pass; std::string detail; };



// Checks the quantiser's invariants, then round-trips an explicit symbol

// sequence through the coder.

static Result run_case(const std::string &name,

                        const double *probs,

                        const std::vector<int> &symbols) {

    FreqTable t = quantise(probs, N_SYMBOLS);



    // invariant 1: every frequency >= 1 (a zero-frequency symbol is unencodable)

    for (int s = 0; s < N_SYMBOLS; s++) {

        if (t.freq[s] < 1)

            return {false, "freq[" + std::to_string(s) + "] == 0"};

    }



    // invariant 2: frequencies sum to exactly M

    uint32_t total = 0;

    for (int s = 0; s < N_SYMBOLS; s++) total += t.freq[s];

    if (total != M)

        return {false, "sum == " + std::to_string(total) + ", want " + std::to_string(M)};



    // invariant 3: cumulative table terminates at M

    if (t.cum[N_SYMBOLS] != M)

        return {false, "cum[n] == " + std::to_string(t.cum[N_SYMBOLS])};



    // round-trip

    const int N = (int)symbols.size();

    std::vector<uint8_t> buf(N * 8 + 4096);

    uint8_t *ptr = buf.data() + buf.size();

    uint32_t x = LOWER;



    for (int i = N - 1; i >= 0; i--)

        encode_symbol(x, &ptr, symbols[i], t);



    size_t n_bytes = (buf.data() + buf.size()) - ptr;

    const uint8_t *rptr = ptr;

    uint32_t xd = x;



    std::vector<int> decoded(N);

    for (int i = 0; i < N; i++)

        decoded[i] = decode_symbol(xd, &rptr, t, N_SYMBOLS);



    if (decoded != symbols) {

        for (int i = 0; i < N; i++)

            if (decoded[i] != symbols[i])

                return {false, "mismatch at " + std::to_string(i) +

                                ": got " + std::to_string(decoded[i]) +

                                ", want " + std::to_string(symbols[i])};

    }



    if (xd != LOWER)

        return {false, "final state " + std::to_string(xd) + ", want " + std::to_string(LOWER)};



    double bps = (n_bytes * 8.0 + 32.0) / N;

    char detail[128];

    snprintf(detail, sizeof detail, "%zu bytes, %.4f bits/sym", n_bytes, bps);

    return {true, detail};

}



static void report(const std::string &name, const Result &r) {

    printf("%-46s %s", name.c_str(), r.pass ? "PASS" : "FAIL");

    printf("  %s\n", r.detail.c_str());

}





int main() {

    std::mt19937 rng(999);

    int failures = 0;



    auto check = [&](const std::string &name, const double *probs,

                      const std::vector<int> &syms) {

        Result r = run_case(name, probs, syms);

        report(name, r);

        if (!r.pass) failures++;

    };



    // -----------------------------------------------------------------------

    // Case 1: one symbol at 1e-9. Below 1/65536, so it floors to 0 and must be

    // rescued to 1. Then we force it to actually occur -- if the rescue failed,

    // the encoder produces an invalid stream.

    // -----------------------------------------------------------------------

    {

        double p[N_SYMBOLS];

        for (int s = 0; s < N_SYMBOLS; s++) p[s] = (1.0 - 1e-9) / (N_SYMBOLS - 1);

        p[7] = 1e-9;



        std::vector<int> syms;

        for (int i = 0; i < 5000; i++) syms.push_back(rng() % N_SYMBOLS);

        for (int i = 0; i < 20; i++) syms[i * 37] = 7;   // force the rare symbol in

        check("sub-representable prob (1e-9), symbol occurs", p, syms);

    }



    // -----------------------------------------------------------------------

    // Case 2: exact zero probability. The model can emit this after softmax

    // underflow. Must still floor to 1 and remain encodable.

    // -----------------------------------------------------------------------

    {

        double p[N_SYMBOLS] = {0};

        for (int s = 0; s < N_SYMBOLS; s++) p[s] = 1.0 / (N_SYMBOLS - 3);

        p[0] = p[1] = p[2] = 0.0;



        std::vector<int> syms;

        for (int i = 0; i < 5000; i++) syms.push_back(3 + rng() % (N_SYMBOLS - 3));

        syms[100] = 0; syms[200] = 1; syms[300] = 2;   // zero-prob symbols occur

        check("exact zero probabilities, symbols occur", p, syms);

    }



    // -----------------------------------------------------------------------

    // Case 3: near-degenerate. One symbol at 0.9999 -- its x_max is enormous,

    // everything else's is tiny, so renormalisation behaviour swings wildly

    // between consecutive symbols.

    // -----------------------------------------------------------------------

    {

        double p[N_SYMBOLS];

        for (int s = 0; s < N_SYMBOLS; s++) p[s] = 0.0001 / (N_SYMBOLS - 1);

        p[23] = 0.9999;



        std::vector<int> syms;

        for (int i = 0; i < 20000; i++) syms.push_back(23);

        for (int i = 0; i < 50; i++) syms[i * 313] = (int)(rng() % N_SYMBOLS);

        check("near-degenerate (one symbol at 0.9999)", p, syms);

    }



    // -----------------------------------------------------------------------

    // Case 4: perfectly uniform. Every freq is 65536/50 = 1310.72, so all 50

    // have a fractional remainder and the largest-remainder pass has to

    // distribute 36 leftover units.

    // -----------------------------------------------------------------------

    {

        double p[N_SYMBOLS];

        for (int s = 0; s < N_SYMBOLS; s++) p[s] = 1.0 / N_SYMBOLS;



        std::vector<int> syms;

        for (int i = 0; i < 10000; i++) syms.push_back(rng() % N_SYMBOLS);

        check("perfectly uniform (36 units to redistribute)", p, syms);

    }



    // -----------------------------------------------------------------------

    // Case 5: all mass on one symbol, everything else exactly zero. The most

    // degenerate table possible -- 49 symbols floored up to 1, so the dominant

    // symbol has to give back 49 units.

    // -----------------------------------------------------------------------

    {

        double p[N_SYMBOLS] = {0};

        p[0] = 1.0;



        std::vector<int> syms(5000, 0);

        syms[2500] = 17;                              // the impossible symbol occurs

        check("all mass on one symbol, another occurs", p, syms);

    }



    // -----------------------------------------------------------------------

    // Case 6: every symbol in the alphabet appears at least once, under a

    // skewed distribution. Exercises the full cumulative table.

    // -----------------------------------------------------------------------

    {

        double p[N_SYMBOLS];

        double sum = 0;

        for (int s = 0; s < N_SYMBOLS; s++) { p[s] = 1.0 / (s + 1); sum += p[s]; }

        for (int s = 0; s < N_SYMBOLS; s++) p[s] /= sum;



        std::vector<int> syms;

        for (int s = 0; s < N_SYMBOLS; s++)

            for (int k = 0; k < 20; k++) syms.push_back(s);

        std::shuffle(syms.begin(), syms.end(), rng);

        check("every symbol occurs, skewed distribution", p, syms);

    }



    // -----------------------------------------------------------------------

    // Case 7: single symbol total. Shortest possible read.

    // -----------------------------------------------------------------------

    {

        double p[N_SYMBOLS];

        for (int s = 0; s < N_SYMBOLS; s++) p[s] = 1.0 / N_SYMBOLS;

        check("single symbol", p, std::vector<int>{31});

    }



    // -----------------------------------------------------------------------

    // Case 8: long run, realistic length for an ONT read.

    // -----------------------------------------------------------------------

    {

        double p[N_SYMBOLS];

        double sum = 0;

        for (int s = 0; s < N_SYMBOLS; s++) {

            p[s] = std::exp(-0.5 * std::pow((s - 35) / 8.0, 2));   // bell around s=35

            sum += p[s];

        }

        for (int s = 0; s < N_SYMBOLS; s++) p[s] /= sum;



        std::vector<int> syms;

        std::discrete_distribution<int> d(p, p + N_SYMBOLS);

        for (int i = 0; i < 30000; i++) syms.push_back(d(rng));

        check("30k symbols, bell-shaped (ONT-like)", p, syms);

    }



    printf("\n%s\n", failures == 0 ? "ALL PASS" : (std::to_string(failures) + " FAILED").c_str());

    return failures == 0 ? 0 : 1;

}
