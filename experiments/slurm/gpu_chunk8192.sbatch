#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 8

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=96G

#SBATCH --time=01:30:00

#SBATCH -J qgru-ch8k

#SBATCH -o /private/home/vtejomur/compression/gpu-ch8k-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

python3 -u benchmarks/gpu_wholefile_chunk8192.py

