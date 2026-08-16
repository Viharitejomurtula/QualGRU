#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=64G

#SBATCH --time=01:00:00

#SBATCH -J qgru-h256cmp

#SBATCH -o /private/home/vtejomur/compression/gpu-h256cmp-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u benchmarks/gpu_codec_compare_h256_32k_1024.py

