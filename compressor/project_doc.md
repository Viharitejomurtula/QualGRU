# QualGRU Compressor — Design and Implementation Plan



Status: planning. Nothing in this document is built yet.



---



## 1. Scope



Turn the verified QualGRU inference engine into a working command-line compressor:



```

qualgru compress   reads.fastq  reads.qgru

qualgru decompress reads.qgru   reads_out.fastq

```



with `reads_out.fastq` byte-identical to `reads.fastq` in lossless mode.



**Decisions taken:**



- **Self-contained archives.** Sequences are stored alongside qualities. The model

  conditions on `base_t` and `base_t+1`, so qualities cannot be decoded without the

  sequence; requiring the user to supply the original FASTA would make this a component

  rather than a tool.

- **Default model: h64 lossless** (3.744 bpc, 2.75 MB/s forward pass). Lossless gives a

  byte-for-byte round-trip, which is both the strictest correctness test and the strongest

  claim. Lossy models ship behind `--model`, where "correct" means matching the *binned*

  qualities rather than the originals.



**Out of scope for v1:** BAM/CRAM input (needs htslib), reference-based sequence

compression, GPU inference, streaming/pipe support.



---



## 2. Architecture



```

compress:

  FASTQ ──parse──> reads

                     ├─ sequences ──zlib──────────────> sequence block

                     └─ qualities ─┐

                                   ├─ GRU forward ──> probabilities

                                   │                       │

                                   │                  quantise to

                                   │                  16-bit freqs

                                   │                       │

                                   └───────────────> range coder ──> quality block



decompress:

  archive ──> sequence block ──unzlib──> sequences

                                            │

              quality block ──> range coder ┤

                                            ├─ GRU forward (one step)

                                            └─ decoded symbol ──feeds back──┐

                                                       ▲                     │

                                                       └─────────────────────┘

```



The feedback loop on decode is the structural difference from everything built so far.

`forward_batch` assumes the entire read is known upfront; decode cannot know timestep

*t+1*'s input until *t* has been decoded.



---



## 3. Correctness requirements



### 3.1 Encoder/decoder determinism



Encoder and decoder must produce **bit-identical** probability distributions at every

timestep. A single differing bit desynchronises the coder and every subsequent symbol is

garbage.



Mitigations, in order of importance:



1. **Quantise probabilities to 16-bit integer frequencies before the coder.** A float

   difference of 1e-7 almost never flips a 16-bit bucket. This is the main defence.

2. **Fixed instruction set at build time** — `-msse4.2` or baseline `x86-64`, never

   `-march=native`. Differing SIMD widths change matmul accumulation order, which changes

   low bits.

3. **Checksum the original qualities in the header.** Decompression verifies and fails

   loudly rather than silently emitting corruption.



Residual risk: "almost never" is not never. The bulletproof fix is fixed-point integer

inference throughout, which is a substantially larger change. Not attempted in v1.



### 3.2 Frequency quantisation



Model probabilities are `float` and sum to 1. The coder needs integer frequencies summing

to exactly `2^16`. Requirements:



- Every symbol gets frequency ≥ 1. A zero frequency makes that symbol unencodable — if it

  then occurs, the encoder produces an invalid stream.

- Frequencies must sum to exactly `2^16`, not approximately. Rounding error is absorbed by

  adjusting the largest bucket.

- The conversion must be deterministic and identical on both sides. Same function, same

  input, same output.



---



## 4. File format (`.qgru`)



```

offset  size    field

0       4       magic "QGRU"

4       1       format version (1)

5       1       model id (0=h64 lossless, 1=h256 lossless, 2=h32 lossless,

                          3=h64 lossy4, 4=h256 lossy4, 5=h32 lossy4)

6       1       flags (bit 0: sequences present)

7       1       reserved

8       8       read count (uint64)

16      8       total quality characters (uint64)

24      4       CRC32 of the original quality stream

28      4       CRC32 of the original sequence stream

32      8       sequence block length (uint64)

40      8       quality block length (uint64)

48      ...     read lengths, varint-encoded

...     ...     read names block, zlib

...     ...     sequence block, zlib

...     ...     quality block, range-coded

```



Read names are stored separately and zlib-compressed; they are not modelled.



---



## 5. Implementation phases



Each phase has an explicit exit test. Do not start the next phase until the current one

passes.



### Phase 1 — Range coder, standalone

`include/qualgru/range_coder.hpp`



Integer arithmetic only. No model, no data files.



**Exit test:** encode 100,000 symbols drawn from a known fixed distribution, decode,

assert the output sequence is identical. Then repeat with a skewed distribution

(one symbol at 99%) and with a uniform one.



*Estimated: half a day.*



### Phase 2 — Frequency quantisation

`include/qualgru/frequencies.hpp`



Convert a `float` probability vector to integer frequencies summing to `2^16`, all ≥ 1.



**Exit test:** for 10,000 random probability vectors, assert sum is exactly `2^16` and

every entry ≥ 1. Include adversarial cases: near-uniform, one symbol at 0.9999, and

symbols with probability below `1/2^16`.



*Estimated: 1–2 hours.*



### Phase 3 — Encode one read

`src/compress.cpp`



Wire model → probabilities → frequencies → coder. Single read, single-threaded.



**Exit test:** encoded size in bits, divided by symbol count, is within a few percent of

the bpc the benchmark predicts for that model (3.744 for h64). A large discrepancy means

the frequency conversion is wrong.



*Estimated: half a day.*



### Phase 4 — Decode one read (**the gate**)

`src/decompress.cpp`



Sequential loop: decode symbol, feed back into GRU, decode next. Requires a single-step

forward call that takes one symbol at a time — `forward_into` already has this shape.



**Exit test:** round-trip one read, byte-compare against the original. Then ten reads of

differing lengths. Then a read containing every symbol in the vocabulary.



If this passes, determinism holds and the remainder is plumbing. If it fails, the failure

is isolated to one read with no infrastructure built on top of it.



*Estimated: 1 day, most of it debugging.*



### Phase 5 — File format and multi-read

`include/qualgru/format.hpp`



Header, varint read lengths, sequence block via zlib, CRC32 verification.



**Exit test:** round-trip a 1,000-read FASTQ. Verify the CRC catches deliberate corruption

of a byte in the quality block.



*Estimated: half a day.*



### Phase 6 — FASTQ I/O

`src/fastq.cpp`



Parse plain and gzipped FASTQ. Preserve read names and the `+` line exactly.



**Exit test:** `diff` original and round-tripped FASTQ on a real 10,000-read file.



*Estimated: 2–3 hours.*



### Phase 7 — CLI, build, docs

`src/main.cpp`, `CMakeLists.txt`, README



`compress`, `decompress`, `--model`, `--threads`, `--verbose`. CMake with FetchContent for

Eigen so a clone builds without system dependencies. Weights committed directly (~68 KB for

h64).



**Exit test:** clone into a clean directory, `cmake && make`, compress and decompress a

test file successfully.



*Estimated: half a day.*



### Phase 8 — Threading

Parallelise across reads. Each read's decode chain is independent.



**Exit test:** identical output to the single-threaded path, byte for byte, at 1, 4, and 64

threads. Any difference means a determinism bug.



*Estimated: 2–3 hours.*



### Phase 9 — Benchmark

Measure real compressed size and real wall-clock compress/decompress time against CRAM and

CoLoRd on identical input.



**This is the number the project has been missing.** Every ratio figure to date is

cross-entropy; every throughput figure is forward-pass only. This phase produces measured

bytes and measured seconds.



*Estimated: 2–3 hours, plus runtime.*








## 7. Risks



| Risk | Likelihood | Mitigation |

|---|---|---|

| Float non-determinism desyncs coder | Medium | 16-bit frequency quantisation; fixed ISA; CRC verification |

| Decode much slower than forward-pass benchmarks | **High** | Expected — sequential within a read. Batch across reads; measure and report honestly |

| Frequency rounding produces zero-probability symbol | Medium | Floor at 1; explicit test with adversarial distributions |

| Compressed size exceeds predicted bpc | Low | Coder overhead is small; a large gap indicates a bug, caught in Phase 3 |

| Memory blowup on long reads | Low | ONT reads reach ~30 KB; hidden state is per-read and small |



The second row deserves emphasis. Current throughput figures batch 2,048 reads into each

matrix multiply. Decode can still batch across independent reads, but the effective batch

may be smaller and the coder adds per-symbol cost that is currently uncounted. **The

decompression number will be lower than 2.75 MB/s and it is not yet known by how much.**



---



## 8. Success criteria



**Minimum:** lossless round-trip on a real FASTQ, byte-identical, with measured compressed

size within a few percent of the predicted 3.744 bpc.



**Target:** the above plus measured decompression throughput, benchmarked against CRAM and

CoLoRd on identical data, with all three numbers produced by the same measurement method.



**Stretch:** lossy mode, threading, and a compression ratio that holds its advantage over

CRAM once real coder overhead is included.
