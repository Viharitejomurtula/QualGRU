#!/bin/bash

#SBATCH -p short

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 8

#SBATCH --mem=64G

#SBATCH --time=01:00:00

#SBATCH -J spring-ont

#SBATCH -o /private/home/vtejomur/compression/spring-ont-%j.out

set -euo pipefail



BIN=/private/home/vtejomur/compression/bench_tools/SPRING/build/spring

IN=/private/home/vtejomur/compression/compressor_ont/CoLoRd/chr20_slice.fastq

OUT=/private/home/vtejomur/compression/bench_results/chr20.spring

DEC=/private/home/vtejomur/compression/bench_results/chr20.spring.dec.fastq

WORK=/private/home/vtejomur/compression/bench_results/spring_tmp_$SLURM_JOB_ID



mkdir -p /private/home/vtejomur/compression/bench_results "$WORK"

rm -f "$OUT" "$DEC"



echo "===== INPUT ====="

stat -c 'bytes=%s' "$IN"



echo "===== COMPRESS ====="

/usr/bin/time -v "$BIN" -c -l -t 8 -w "$WORK" -i "$IN" -o "$OUT"



echo "===== COMPRESSED ====="

stat -c 'bytes=%s' "$OUT"



echo "===== DECOMPRESS ====="

/usr/bin/time -v "$BIN" -d -t 8 -w "$WORK" -i "$OUT" -o "$DEC"



echo "===== VERIFY ====="

cmp "$IN" "$DEC"

echo "ROUNDTRIP=PASS"



rm -rf "$WORK" "$DEC"

