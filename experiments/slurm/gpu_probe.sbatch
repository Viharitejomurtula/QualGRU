#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:1

#SBATCH --mem=8G

#SBATCH --time=00:10:00

#SBATCH -J qgru-gpuprobe

#SBATCH -o /private/home/vtejomur/compression/gpu-probe-%j.out

set -euo pipefail

echo "NODE=$(hostname)"

echo "===== GPU ====="

nvidia-smi

echo "===== NVCC BEFORE MODULE ====="

which nvcc || true

echo "===== CUDA MODULES ====="

module avail 2>&1 | grep -i cuda || true

echo "===== MODULE LIST ====="

module list 2>&1 || true

