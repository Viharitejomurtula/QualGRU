# QualGRU GPU + CPU Optimization Checkpoint — 2026-08-15

Branch: checkpoint/full-gpu-cpu-optimization-2026-08-15
Parent commit: d612f51faffd458821f9539fac573c2aef6c38de

## GPU whole-file ONT benchmarks

### h64 — A100, 8192-symbol independent chunks
- Dataset: chr20_slice.fastq
- Quality symbols: 3,471,674,698
- Compression: 51.794 M quality symbols/s
- Decompression: 54.684 M quality symbols/s
- Codec-required quality rate: 3.7200 bits/symbol
- Self-contained archive rate: 3.7249 bits/symbol
- Reconstruction mismatches: 0
- GPU rANS roundtrip: PASS

### h256 — A100, 8192-symbol independent chunks
- Quality symbols: 3,471,674,698
- Compression: 23.696 M quality symbols/s
- Decompression: 24.651 M quality symbols/s
- Codec-required quality rate: 3.6333 bits/symbol
- Self-contained archive rate: 3.6382 bits/symbol
- Reconstruction mismatches: 0
- GPU rANS roundtrip: PASS

h256 reduces codec bpc by about 2.3% versus h64 but is about 2.2x slower.

## GPU optimizations established
- Deterministic probability quantization.
- Compact rANS (frequency, cumulative) state.
- GPU-batched rANS.
- TF32 GRU inference experiments.
- 8192-symbol independent chunking to eliminate the low-parallelism long-read tail.
- Whole-file processing across 527,163 independent chunks.

## CPU h64/h256 benchmarks

### h64, 64 threads, O2 + SSE4.2
- Compression: about 1.47 M symbols/s
- Decompression: about 6.57 M symbols/s

### h64, 64 threads, O3 + SSE4.2
- Compression: about 1.52 M symbols/s
- Decompression: about 6.75 M symbols/s

### h64, 64 threads, O3 + AVX2/FMA
- Compression: about 1.60 M symbols/s
- Decompression: about 8.85 M symbols/s

### h256, 64 threads
- Compression: about 0.645 M symbols/s
- Decompression: about 0.987 M symbols/s

## ISA compatibility
- O2 SSE4.2 encode -> O3 SSE4.2 decode: PASS
- O3 SSE4.2 encode -> O2 SSE4.2 decode: PASS
- SSE4.2 encode -> AVX2 decode: FAIL
- AVX2 encode -> SSE4.2 decode: FAIL
- AVX2 encode -> AVX2 decode: PASS
- Therefore SSE4.2 remains the portable archive-compatible implementation.

## External compressor benchmarks

### SPRING
- Full FASTQ compression archive: 1,987,225,600 bytes
- Quality stream: 1,739,495,150 bytes
- Quality rate: about 4.008 bits/symbol
- Compression wall time: 179.86 s
- Decompression wall time: 127.42 s
- Byte-identical FASTQ roundtrip: PASS

### ENANO
- Quality stream: 1,900,608,182 bytes
- Quality rate: about 4.38 bits/symbol
- Compression wall time: 42.31 s
- Decompression wall time: 75.12 s
- Quality and sequence reconstruction: PASS
- FASTQ serialization differs because + separator lines are rewritten.

## CPU optimization target
- Target: 20 M quality symbols/s compression and decompression.
- Current best experimental h64 compression: about 1.60 M/s.
- Current best experimental h64 decompression: about 8.85 M/s.

## Next CPU optimization
Compression currently constructs and retains a full FreqTable for every transition before reverse rANS encoding.
The encoder ultimately consumes only frequency and cumulative frequency for the actual next symbol.
Planned optimization: retain compact (freq,cum) pairs only, then eliminate per-symbol heap allocations inside quantise().
