#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=20G

#SBATCH --time=00:30:00

#SBATCH -J qgru-fq-a100

#SBATCH -o /private/home/vtejomur/compression/gpu-fastquant-a100-%j.out

set -euo pipefail



cd /private/home/vtejomur/compression/QualGRU-gpu



nvidia-smi --query-gpu=name,memory.total --format=csv,noheader



echo "===== A100 FASTQUANT 32K ====="

python3 benchmarks/gpu_codec_fastquant_32k.py



echo

echo "===== A100 FASTQUANT 65K ====="

python3 benchmarks/gpu_codec_fastquant_65k.py

