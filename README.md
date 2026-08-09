# QualGRU

A neural sequence model for compressing Oxford Nanopore quality scores, paired with a
complete, working, threaded compressor built around it. On real reads it beats CRAM and
CoLoRd on quality-score entropy, and that advantage survives contact with a real rANS
entropy coder — not just a model-level cross-entropy projection.

This repository holds the model, training code, a C++ inference engine, benchmarks, and the
`compressor/` CLI (`qualgru compress` / `qualgru decompress`) that turns the model into a
byte-identical, round-trip-verified FASTQ compressor. See [Status](#status) for what's built
versus what remains.

---

## Build and run

Requires Eigen3 and zlib headers:

```bash
git clone https://github.com/Viharitejomurtula/QualGRU.git
cd QualGRU
sudo apt install libeigen3-dev zlib1g-dev g++
```

Build:

```bash
bash compressor/build.sh
```

Compress a FASTQ (plain or `.gz`):

```bash
./compressor/qualgru compress reads.fastq reads.qgru --model h64 --threads 8
```

Decompress:

```bash
./compressor/qualgru decompress reads.qgru reads_out.fastq --threads 8
```

`--model` selects `h64` (default, balanced), `h256` (best ratio), or `h32` (fastest).
Decompression doesn't take `--model` — it's read from the archive header. `--threads 0` uses
all available cores; default is 4.

---

## Results — measured, not projected

Every number below is bytes on disk and wall-clock seconds, not cross-entropy. That gap —
flagged as the project's main open risk while only the model existed — is closed.

### Compression ratio, quality scores only (1,000 real ONT reads)

| Method | bits/symbol | vs. QualGRU |
|---|---|---|
| **QualGRU (h64)** | **3.7352** | — |
| CRAM 3.0/3.1 (full 186,533-read dataset) | 3.95–3.96 | 5–6% smaller |
| CoLoRd (`-q org`, same 1,000 reads) | 4.1612 | 10% smaller |

### Compression ratio, total archive (quality + sequence + framing, same 1,000 reads)

| Method | Total size |
|---|---|
| QualGRU | 14,301,333 bytes |
| CoLoRd | 14,508,401 bytes |

QualGRU wins on quality (its actual contribution); CoLoRd wins on sequence compression
(reference-free assembly, not something QualGRU attempts — sequences here are plain zlib).
The two roughly cancel at the archive level; reported separately above because conflating
them would misstate both results.

### Throughput

| Direction | QualGRU (h64, 16 threads) | CRAM |
|---|---|---|
| Decompress, quality only | ~1.18 MB/s | ~800–1,800 MB/s* |

\*CRAM's figure is a differential-timing estimate (full decode time minus decode time with
quality flattened), imprecise because it subtracts two ~20-second measurements to recover a
~3-second difference — repeated trials gave estimates from 781 to 1,761 MB/s. The imprecision
doesn't change the finding: CRAM's purpose-built codec is two to three orders of magnitude
faster than this from-scratch neural approach.

### What this means

The compression-ratio advantage from the model results (~5–10%, model-size dependent)
survives contact with a real entropy coder — coder overhead measured at 0.02–0.4% depending
on read length, negligible at ONT read lengths.

The throughput cost of that advantage is real and large. A purpose-built C codec with decades
of optimization decodes roughly 1,000× faster than a neural model run one GRU step at a time.
This is the honest trade the project makes: better compression, much slower — worth stating
plainly rather than hoping nobody asks.

---

## How it works

At each position *t*, the model predicts quality<sub>t+1</sub> conditioned on
quality<sub>t</sub>, base<sub>t</sub>, and base<sub>t+1</sub>, plus a GRU hidden state
summarising all prior positions. Using base<sub>t+1</sub> is legitimate: in a real pipeline
the nucleotide sequence is decoded before the quality stream is compressed.

**The base conditioning is the result.** Quality-only variants plateaued at 3.88–3.91 bpc
and would not move — scaling hidden size from 64 to 256 changed held-out bpc by under 0.02,
and adding windowed causal attention over the hidden states changed nothing measurable
(3.8791 → 3.8811). Conditioning on adjacent base calls broke the plateau to ~3.70,
consistent with quality scores carrying sequence-context dependence that a quality-only
model structurally cannot access.

Training on 50K reads rather than 10K converged to essentially the same value
(3.6998 vs. 3.6989), suggesting the model is near its architectural ceiling rather than
data-limited.

---

## Repository layout

```
compressor/    qualgru CLI — rANS coder, threaded encode/decode, build.sh, tests
inference/     gru_cell.hpp — C++/Eigen forward pass, verified bit-exact vs. PyTorch
training/      PyTorch trainers (lossless, lossy, custom binning) and weight export
weights/       .bin weight files, 7 configurations, plus vocabularies
checkpoints/   .pt training checkpoints, and .npz from the original NumPy implementation
benchmarks/    the C++ programs used to produce the throughput and profiling numbers
```

Weight sets: `h256`, `h64`, `h32` (lossless); `lossy4_h256`, `lossy4_h64`, `lossy4_h32`
(4-bin); `h64_smallemb` (reduced embedding dimensions, a negative result).

---

## The C++ engine

`inference/gru_cell.hpp` reimplements the forward pass from scratch in Eigen — no PyTorch
dependency at inference time. It was validated bit-exactly against the PyTorch reference at
the level of individual gate activations, then end-to-end on real reads.

Throughput went from ~2,000 chars/sec (PyTorch, JIT-compiled, single-threaded) to ~9.2M
chars/sec. What actually mattered, in order:

| Change | Effect |
|---|---|
| `-O3 -march=native` (from `-O0`) | **38×** |
| 64-thread parallelism across reads | ~23× |
| Reducing hidden size 256 → 64 → 32 | 9.5× per step |
| Batching reads into GEMM instead of GEMV | ~9% |
| Vectorising the output softmax | small |

Stage profiling put **79–97% of runtime in matrix multiplication**, with indexing, masking,
and allocation under 3% combined. Optimizations aimed anywhere else did nothing:

- Fusing the reset/update gate matmuls: no measurable change
- Precomputing embeddings for a whole batch upfront: **2× slower**
- Pinning threads to cores: **worse**
- `-Ofast`, `-flto`, `-funroll-loops`: no change over `-O3`
- OpenBLAS: 14% at h256, nothing at h64 or h32
- Halving embedding dimensions: no measurable speedup, ~0.005 bpc worse

The evaluation node is an Intel Xeon E7-4830 (2.13 GHz, 2011). Its AVX2 instructions **fault
at runtime** despite `lscpu` reporting support — likely a virtualised CPU model. That rules
out int8 quantisation, MKL, and bfloat16 on this hardware, not as low-payoff but as
impossible.

---

## Methods

**Data.** GIAB 2025.01 HG002 nanopore, `hac` basecalling. chr20 from flowcell PAW70337
(186,533 reads; 3.47×10⁹ quality characters) and chr21 from PAW71238 (121,498 reads).
Model evaluation uses a 640-read sample drawn evenly across the length distribution.

**How bpc is computed.** For CRAM: encode the same reads twice, once with true qualities and
once with qualities flattened to a constant, and attribute the size difference to the quality
stream. For CoLoRd: the tool reports quality-stream size directly. For QualGRU: at each
position take −log₂ of the probability the model assigned to the character that actually
occurred, and average.

**CRAM configurations tested.** samtools 1.13 at CRAM 3.0 default (3.9505) and 3.1 default
(3.9607) and 3.1 archive profile (3.9569); samtools 1.21 at 3.1 archive (3.9583). All four
fall within 0.01 bpc. fqzcomp-qual entered htslib's CRAM writer in 1.14, so only the 1.21 run
could produce it; `use_fqz=1` was not accepted as an option by either build, and we did not
establish whether the codec was selected in the archive profile.

---

## Status

**What exists:** a complete, working, threaded compressor — `qualgru compress` /
`qualgru decompress`, FASTQ in (plain or gzipped) and out, byte-identical round trip verified.

- An rANS entropy coder, hand-implemented, verified against a fixed-distribution test suite
  covering degenerate and adversarial cases before the model was ever involved
- Threaded encode and decode, dynamic work assignment across an atomic counter (read lengths
  vary too widely — 380 to 50,000+ characters — for static splitting to balance), verified
  byte-identical output at 1/4/16/64 threads
- Chunked, processing reads in blocks of 10,000 so memory stays bounded regardless of input
  size, rather than holding an entire dataset in RAM
- CRC32-verified — the decoder checks its output against a checksum stored at compress time,
  so a desync between encoder and decoder (e.g. from differing build flags) fails loudly
  instead of silently producing plausible-but-wrong quality scores
- Self-locating model weights (resolved relative to the binary via `/proc/self/exe`), so the
  tool runs correctly from any working directory
- Fixed instruction set at build time (`-msse4.2`, never `-march=native`) — archives must
  remain decodable by a binary built elsewhere, and differing SIMD width changes floating-point
  accumulation order enough to desync the coder

**Known limitations:**

- Cross-machine determinism (compress on one machine, decompress on another, confirm the
  checksum) has not been tested — the fixed-ISA build flag and CRC exist specifically to make
  this safe, but the claim is unverified
- No CMake — a shell build script (`build.sh`) covers building the tool, not packaging it for
  distribution
- Lossy models (4-bin, matching CoLoRd's quantization scheme) are trained and benchmarked at
  the model level but not wired into the CLI
- Sequence compression is plain zlib; a reference-based approach (as CRAM and CoLoRd use)
  would likely close or reverse the total-archive-size gap, but reference-based sequence
  compression was never this project's contribution
- The CRAM throughput figure is an order-of-magnitude estimate, not a precise one, for the
  reasons noted above

The benchmark programs in `benchmarks/` use hardcoded relative paths and expect to run from
a working directory that no longer exists in this layout. They are included as a record of
what was measured, not as a build target.

---

## Context

Undergraduate research in the Sirén / Paten Lab at UC Santa Cruz, advised by
Prof. Jouni Sirén (UCSC) and Prof. Tsachy Weissman (Stanford).
