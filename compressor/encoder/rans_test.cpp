// rans_test.cpp — Phase 1: standalone rANS coder, no model.

//

// Purpose: prove the coder round-trips correctly using a fixed, hardcoded

// probability distribution. If this passes, the coder arithmetic, the

// frequency conversion, and the buffer direction are all correct, and none

// of them need to be suspected when the model is wired in later.

//

// Build: g++ -O2 -std=c++17 rans_test.cpp -o rans_test



#include <cstdint>

#include <cmath>

#include <cstdio>

#include <cassert>

#include <vector>

#include <random>

#include <numeric>

#include <algorithm>



// - We need to set upper and lower limits for the state integer x

// - M = 2^16, there are 50 symbols, and we need to set our initial x0 to the lower limit l

static constexpr uint32_t PROB_BITS = 16;

static constexpr uint32_t M         = 1u << PROB_BITS;   // 65536  (NOT 2^16 -- ^ is XOR)

static constexpr uint32_t LOWER     = 1u << 23;          // 8388608

static constexpr int      N_SYMBOLS = 50;



// Note there is no single fixed upper limit. The bound depends on the symbol's

// frequency: rare symbols (small f_s) overflow far sooner, which is *why* they

// cost more bits. See x_max() below.

static inline uint32_t x_max(uint32_t f_s) {

    return ((LOWER >> PROB_BITS) << 8) * f_s;

}





// - At each timestep we take the probability distribution and turn it into a

//   frequency table in which the sum of all frequencies adds up to M

// - We use floors and largest remainders when converting the probabilities into

//   frequencies, we also calculate the cumulative frequencies. We also enforce f>=1 for all s

struct FreqTable {

    uint32_t freq[N_SYMBOLS];

    uint32_t cum[N_SYMBOLS + 1];   // cum[s] = start of symbol s; cum[N] == M

};



static FreqTable quantise(const double *probs, int n) {

    FreqTable t{};



    // floors

    double scaled[N_SYMBOLS];

    uint32_t total = 0;

    for (int s = 0; s < n; s++) {

        scaled[s] = probs[s] * M;

        t.freq[s] = (uint32_t)scaled[s];          // floor

        if (t.freq[s] < 1) t.freq[s] = 1;         // enforce f_s >= 1

        total += t.freq[s];

    }



    // largest remainders: hand out what's left to the symbols that lost the most

    // to flooring. If we overshot M (because of the >=1 floor), take back from

    // the largest frequencies instead.

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

            if (t.freq[biggest] <= 1) break;      // cannot go below 1

            t.freq[biggest]--;

            total--;

        }

    }



    // Assert all frequencies add up to M

    total = 0;

    for (int s = 0; s < n; s++) {

        assert(t.freq[s] >= 1);

        total += t.freq[s];

    }

    assert(total == M);



    // and this also gets cumfreq

    t.cum[0] = 0;

    for (int s = 0; s < n; s++) t.cum[s + 1] = t.cum[s] + t.freq[s];

    assert(t.cum[n] == M);



    return t;

}





// - At each timestep, the frequency table and the encoding formula are used to

//   push each symbol into the state.

//     - If the state is going to exceed the upper bound if the next symbol is

//       pushed then we write to the output buffer

//

// The output buffer is written BACKWARDS and read forwards -- ptr starts at the

// end of the allocation and walks down. (Your pseudocode had this inverted.)

static inline void encode_symbol(uint32_t &x, uint8_t **ptr, int s, const FreqTable &t) {

    uint32_t f_s = t.freq[s];

    uint32_t c_s = t.cum[s];



    // renormalise BEFORE encoding, not after -- once the state has overflowed

    // the information is already lost

    uint32_t bound = x_max(f_s);

    while (x >= bound) {

        *--(*ptr) = (uint8_t)(x & 0xff);

        x >>= 8;

    }



    // use encode formula x = M*floor(x/fs) + (x%fs) + cs

    x = ((x / f_s) << PROB_BITS) + (x % f_s) + c_s;

}





// - use decoding formula x = fs*floor(x/M) + (x % M - cs)

//     - again there is a limit for the state integer x, it can't stay below the lower limit l

//     - if the lower limit is hit then we read from the output buffer

static inline int decode_symbol(uint32_t &x, const uint8_t **ptr, const FreqTable &t, int n) {

    // find which symbol's slot the low 16 bits land in

    uint32_t slot = x & (M - 1);

    int s = 0;

    while (s < n - 1 && t.cum[s + 1] <= slot) s++;



    uint32_t f_s = t.freq[s];

    uint32_t c_s = t.cum[s];



    x = f_s * (x >> PROB_BITS) + slot - c_s;



    // while, not if -- may need more than one byte

    while (x < LOWER) {

        x = (x << 8) | **ptr;

        (*ptr)++;

    }

    return s;

}





int main() {

    // Fixed distribution standing in for QualGRU's output. Skewed, so that

    // rare symbols exercise the frequent-renormalisation path.

    double probs[N_SYMBOLS];

    {

        double sum = 0;

        for (int s = 0; s < N_SYMBOLS; s++) {

            probs[s] = 1.0 / (s + 1);            // Zipf-ish

            sum += probs[s];

        }

        for (int s = 0; s < N_SYMBOLS; s++) probs[s] /= sum;

    }

    FreqTable table = quantise(probs, N_SYMBOLS);



    // Draw a symbol sequence from that distribution.

    const int N = 100000;

    std::vector<int> symbols(N);

    {

        std::mt19937 rng(12345);

        std::discrete_distribution<int> dist(probs, probs + N_SYMBOLS);

        for (int i = 0; i < N; i++) symbols[i] = dist(rng);

    }



    // ---------------- encode ----------------

    std::vector<uint8_t> buf(N * 4 + 1024);

    uint8_t *ptr = buf.data() + buf.size();      // start at the END



    uint32_t x = LOWER;                          // x0 = lower limit



    // We walk through the q scores in reverse when encoding

    for (int i = N - 1; i >= 0; i--)

        encode_symbol(x, &ptr, symbols[i], table);



    // The result of encoding is the final state integer x and the entirety of

    // the output buffer

    size_t n_bytes = (buf.data() + buf.size()) - ptr;

    uint32_t final_state = x;



    double bits_total  = n_bytes * 8.0 + 32.0;   // +32 for the stored state

    double bits_per_sym = bits_total / N;



    // theoretical entropy of the distribution, for comparison

    double H = 0;

    for (int s = 0; s < N_SYMBOLS; s++)

        if (probs[s] > 0) H -= probs[s] * (log(probs[s]) / log(2.0));



    printf("symbols:      %d\n", N);

    printf("bytes out:    %zu\n", n_bytes);

    printf("bits/symbol:  %.4f\n", bits_per_sym);

    printf("entropy:      %.4f\n", H);

    printf("overhead:     %.3f%%\n", 100.0 * (bits_per_sym - H) / H);



    // ---------------- decode ----------------

    const uint8_t *rptr = ptr;                   // read FORWARDS from where encode stopped

    x = final_state;



    std::vector<int> decoded(N);

    // Loop over the known symbol count -- the decoder is told the length,

    // it cannot infer it. (Your pseudocode had `while x != lower_limit`,

    // which does not terminate correctly.)

    for (int i = 0; i < N; i++)

        decoded[i] = decode_symbol(x, &rptr, table, N_SYMBOLS);



    // ---------------- verify ----------------

    bool ok = (decoded == symbols);

    printf("\nround-trip:   %s\n", ok ? "PASS" : "FAIL");

    if (!ok) {

        for (int i = 0; i < N; i++) {

            if (decoded[i] != symbols[i]) {

                printf("  first mismatch at %d: got %d, want %d\n",

                        i, decoded[i], symbols[i]);

                break;

            }

        }

        return 1;

    }



    // final state should be back at the starting value

    printf("final state:  %u (started %u) %s\n",

            x, LOWER, x == LOWER ? "OK" : "MISMATCH");



    return 0;

}
