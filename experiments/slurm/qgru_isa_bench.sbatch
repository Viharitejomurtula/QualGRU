#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -J qgru-isa
#SBATCH -o /private/home/vtejomur/compression/qgru-isa-%j.out
set -euo pipefail
ROOT=/private/home/vtejomur/compression/QualGRU-gpu
DATA=/private/home/vtejomur/compression/qualgru_profile_10k.fastq
TMP=/tmp/qgru_isa_$SLURM_JOB_ID
mkdir -p "$TMP"
cp "$DATA" "$TMP/input.fastq"
QUAL=$(awk 'NR%4==0{s+=length($0)}END{print s}' "$TMP/input.fastq")
cd "$ROOT"
echo "===== MACHINE ====="
hostname
lscpu | grep -E "Model name|avx2|fma" | head -3
echo "QUALITY_SYMBOLS=$QUAL"
for B in qualgru_o2_sse42 qualgru_o3_sse42 qualgru_o3_avx2; do
echo "========================================"
echo "BINARY=$B"
echo "========================================"
/usr/bin/time -f "COMPRESS wall=%e user=%U sys=%S cpu=%P" ./compressor/$B compress "$TMP/input.fastq" "$TMP/$B.qgru" --model h64 --threads 64
stat -c "archive_bytes=%s" "$TMP/$B.qgru"
/usr/bin/time -f "DECOMPRESS wall=%e user=%U sys=%S cpu=%P" ./compressor/$B decompress "$TMP/$B.qgru" "$TMP/$B.fastq" --threads 64
cmp "$TMP/input.fastq" "$TMP/$B.fastq"
echo "SAME_BUILD_ROUNDTRIP=PASS"
done
rm -rf "$TMP"
