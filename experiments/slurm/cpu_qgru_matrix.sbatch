#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH -J qgru-cpu
#SBATCH -o /private/home/vtejomur/compression/cpu-qgru-%A_%a.out
set -euo pipefail
ROOT=/private/home/vtejomur/compression/QualGRU-gpu
DATA=/private/home/vtejomur/compression/qualgru_profile_10k.fastq
MODELS=(h64 h256)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
THREADS=${THREADS}
TMP=/tmp/qgru_${MODEL}_${THREADS}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TMP"
cp "$DATA" "$TMP/input.fastq"
QUAL=$(awk 'NR%4==0{s+=length($0)}END{print s}' "$TMP/input.fastq")
echo "model=$MODEL threads=$THREADS quality_symbols=$QUAL"
hostname
lscpu | grep "Model name" | head -1
cd "$ROOT"
/usr/bin/time -f "COMPRESS wall=%e user=%U sys=%S maxrss_kb=%M" ./compressor/qualgru compress "$TMP/input.fastq" "$TMP/out.qgru" --model "$MODEL" --threads "$THREADS"
stat -c "archive_bytes=%s" "$TMP/out.qgru"
/usr/bin/time -f "DECOMPRESS wall=%e user=%U sys=%S maxrss_kb=%M" ./compressor/qualgru decompress "$TMP/out.qgru" "$TMP/decoded.fastq" --threads "$THREADS"
cmp "$TMP/input.fastq" "$TMP/decoded.fastq"
echo "ROUNDTRIP=PASS"
rm -rf "$TMP"
