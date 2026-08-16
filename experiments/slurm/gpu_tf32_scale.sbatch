#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=24G

#SBATCH --time=00:30:00

#SBATCH -J qgru-tf32

#SBATCH -o /private/home/vtejomur/compression/gpu-tf32-%j.out

set -euo pipefail



cd /private/home/vtejomur/compression/QualGRU-gpu



nvidia-smi --query-gpu=name,memory.total --format=csv,noheader



for B in 8192 16384 32768 65536

do

    echo

    echo "========================================"

    echo "TF32 BATCH $B"

    echo "========================================"

    python3 benchmarks/gpu_codec_fastquant_tf32_${B}.py

done

