#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -J spring-dec
#SBATCH -o /private/home/vtejomur/compression/spring-dec-%j.out
set -euo pipefail
BIN=/private/home/vtejomur/compression/bench_tools/SPRING/build/spring
ARCH=/private/home/vtejomur/compression/bench_results/chr20.spring
ORIG=/private/home/vtejomur/compression/compressor_ont/CoLoRd/chr20_slice.fastq
TMP=/tmp/spring_dec_$SLURM_JOB_ID
mkdir -p "$TMP"
cp "$ARCH" "$TMP/input.spring"
echo "===== SYSTEM ====="
hostname
lscpu | grep "Model name" | head -1
echo "===== DECOMPRESS ====="
/usr/bin/time -v "$BIN" -d -t 8 -w "$TMP" -i "$TMP/input.spring" -o "$TMP/decoded.fastq"
echo "===== VERIFY ====="
cmp "$ORIG" "$TMP/decoded.fastq"
echo "ROUNDTRIP=PASS"
stat -c "decoded_bytes=%s" "$TMP/decoded.fastq"
rm -rf "$TMP"
