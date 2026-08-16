#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -J qgru-compat
#SBATCH -o /private/home/vtejomur/compression/qgru-compat-%j.out
set -u
ROOT=/private/home/vtejomur/compression/QualGRU-gpu
DATA=/private/home/vtejomur/compression/qualgru_profile_10k.fastq
TMP=/tmp/qgru_compat_$SLURM_JOB_ID
mkdir -p "$TMP"
head -n 4000 "$DATA" > "$TMP/input.fastq"
cd "$ROOT"
BINS=(qualgru_o2_sse42 qualgru_o3_sse42 qualgru_o3_avx2)
for ENC in "${BINS[@]}"; do
./compressor/$ENC compress "$TMP/input.fastq" "$TMP/$ENC.qgru" --model h64 --threads 64 >/dev/null
done
for ENC in "${BINS[@]}"; do for DEC in "${BINS[@]}"; do
rm -f "$TMP/decoded.fastq"
./compressor/$DEC decompress "$TMP/$ENC.qgru" "$TMP/decoded.fastq" --threads 64 >/dev/null 2>&1
RC=$?
if [ "$RC" -eq 0 ] && cmp -s "$TMP/input.fastq" "$TMP/decoded.fastq"; then RESULT=PASS; else RESULT=FAIL; fi
echo "ENC=$ENC DEC=$DEC RESULT=$RESULT exit=$RC"
done; done
rm -rf "$TMP"
