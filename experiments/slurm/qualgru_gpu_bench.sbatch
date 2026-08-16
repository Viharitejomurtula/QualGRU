#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:1

#SBATCH --mem=8G

#SBATCH --time=00:15:00

#SBATCH -J qgru-gpubench

#SBATCH -o /private/home/vtejomur/compression/gpu-bench-%j.out

set -euo pipefail



cd /private/home/vtejomur/compression/QualGRU-gpu



echo "===== NODE ====="

hostname



echo "===== GPU ====="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader



echo "===== PYTORCH ====="

python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'



echo "===== QUALGRU GPU BENCHMARK ====="

python3 benchmarks/gpu_h64_bench.py

