#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:1

#SBATCH --mem=8G

#SBATCH --time=00:10:00

#SBATCH -J qgru-tail

#SBATCH -o /private/home/vtejomur/compression/gpu-tail-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 benchmarks/gpu_tail_bench.py

