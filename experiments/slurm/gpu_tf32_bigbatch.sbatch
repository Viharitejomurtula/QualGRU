#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=48G

#SBATCH --time=00:30:00

#SBATCH -J qgru-bigbatch

#SBATCH -o /private/home/vtejomur/compression/gpu-bigbatch-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "===== TF32 BATCH 98304 ====="

python3 benchmarks/gpu_codec_fastquant_tf32_98304.py

echo

echo "===== TF32 BATCH 131072 ====="

python3 benchmarks/gpu_codec_fastquant_tf32_131072.py

