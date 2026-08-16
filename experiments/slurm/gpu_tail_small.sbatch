#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=32G

#SBATCH --time=01:30:00

#SBATCH -J qgru-tails

#SBATCH -o /private/home/vtejomur/compression/gpu-tails-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 -u benchmarks/gpu_tail_small.py

