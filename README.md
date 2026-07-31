# QualGRU



A neural sequence model for compressing Oxford Nanopore quality scores. On HG002 data it

reaches lower entropy than both CRAM and CoLoRd, in lossless and lossy modes.



This repository is a **research archive** — models, training code, a C++ inference engine,

and the benchmarks used to produce the results below. It is not yet a usable compressor;

see [Status](#status).



---



## Results



All figures on HG002 nanopore reads, GIAB 2025.01 release, `hac` basecalling.



### Lossless — bits per quality symbol



| Method | chr20 | chr21 (held-out flowcell) |

|---|---|---|

| **QualGRU h256** | **3.678** | **3.690** |

| QualGRU h64 | 3.744 | 3.749 |

| QualGRU h32 | 3.792 | 3.793 |

| CRAM 3.0 | 3.951 | — |

| CRAM 3.1, archive profile | 3.958 | — |

| CoLoRd `-q org` | 4.082 | — |



### Lossy — four bins, CoLoRd's default ONT scheme (thresholds 7, 14, 26)



| Method | bpc |

|---|---|

| **QualGRU h256** | **0.378** |

| QualGRU h64 | 0.385 |

| QualGRU h32 | 0.391 |

| CoLoRd `-q 4-avg` | 0.408 |



### Throughput



| Method | MB/s | bpc |

|---|---|---|

| QualGRU h32, lossy | 8.79 | 0.391 |

| QualGRU h32, lossless | 5.26 | 3.792 |

| QualGRU h64, lossless | 2.75 | 3.744 |

| QualGRU h256, lossless | 0.33 | 3.678 |

| CoLoRd decompress, lossy | 14.91 | 0.408 |

| CoLoRd decompress, lossless | 9.25 | 4.082 |

| CRAM 3.0 decompress | 9.77 | 3.951 |



QualGRU figures are **model forward pass only** — no entropy coder, and reads are batched

freely. Real decompression is sequential within a read and will be slower. CoLoRd and CRAM

figures are end-to-end decompression including sequence and header reconstruction and a

6.95 GB disk write, so they understate their quality codecs in isolation.



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



**What exists:** a trained model that beats production compressors on entropy, a verified

C++ inference engine, and benchmarks characterising the speed/ratio frontier.



**What does not exist:** an arithmetic coder. The bpc figures here are cross-entropy — the

bound a coder would approach — not bytes measured on disk. CRAM and CoLoRd figures *are*

bytes on disk and include real coder overhead, so the comparison currently favours QualGRU

by a small margin.



Building the coder, a round-trip encode/decode harness, and a file format is the next

milestone. Until then these are projected rather than realised compression ratios, and the

throughput figures are an upper bound on decompression speed rather than a measurement of it.



The benchmark programs in `benchmarks/` use hardcoded relative paths and expect to run from

a working directory that no longer exists in this layout. They are included as a record of

what was measured, not as a build target.



---



## Context



Undergraduate research in the Sirén / Paten Lab at UC Santa Cruz, advised by

Prof. Jouni Sirén (UCSC) and Prof. Tsachy Weissman (Stanford).
