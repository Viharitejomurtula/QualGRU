#!/bin/bash

#SBATCH -p short

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 8

#SBATCH --mem=32G

#SBATCH --time=01:00:00

#SBATCH -J enano-ont

#SBATCH -o /private/home/vtejomur/compression/enano-ont-%j.out

set -euo pipefail



BIN=/private/home/vtejomur/compression/bench_tools/EnanoFASTQ/enano/enano

IN=/private/home/vtejomur/compression/compressor_ont/CoLoRd/chr20_slice.fastq

OUT=/private/home/vtejomur/compression/bench_results/chr20.enano

DEC=/private/home/vtejomur/compression/bench_results/chr20.enano.dec.fastq



mkdir -p /private/home/vtejomur/compression/bench_results

rm -f "$OUT" "$DEC"



echo "===== INPUT ====="

stat -c 'bytes=%s' "$IN"



echo "===== COMPRESS ====="

/usr/bin/time -v "$BIN" -t 8 "$IN" "$OUT"



echo "===== COMPRESSED ====="

stat -c 'bytes=%s' "$OUT"



echo "===== DECOMPRESS ====="

/usr/bin/time -v "$BIN" -d -t 8 "$OUT" "$DEC"



echo "===== VERIFY ====="

cmp "$IN" "$DEC"

echo "ROUNDTRIP=PASS"



rm -f "$DEC"

