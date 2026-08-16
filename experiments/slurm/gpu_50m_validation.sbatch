#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=48G

#SBATCH --time=00:30:00

#SBATCH -J qgru-50m-val

#SBATCH -o /private/home/vtejomur/compression/gpu-50m-validation-%j.out

set -euo pipefail



cd /private/home/vtejomur/compression/QualGRU-gpu



nvidia-smi --query-gpu=name,memory.total --format=csv,noheader



echo "========================================"

echo "65K x 256 REPRODUCTION RUN 1"

echo "========================================"

python3 benchmarks/gpu_codec_fastquant_65k.py



echo

echo "========================================"

echo "65K x 256 REPRODUCTION RUN 2"

echo "========================================"

python3 benchmarks/gpu_codec_fastquant_65k.py



echo

echo "========================================"

echo "65K x 256 REPRODUCTION RUN 3"

echo "========================================"

python3 benchmarks/gpu_codec_fastquant_65k.py



echo

echo "========================================"

echo "65K x 1024 LONG-SEQUENCE TEST"

echo "========================================"

python3 benchmarks/gpu_codec_fastquant_65k_1024.py

