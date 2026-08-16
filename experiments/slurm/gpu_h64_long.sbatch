#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 6

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=64G

#SBATCH --time=01:30:00

#SBATCH -J qgru-h64long

#SBATCH -o /private/home/vtejomur/compression/gpu-h64long-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u benchmarks/gpu_codec_compare_h64_8k_8192.py

