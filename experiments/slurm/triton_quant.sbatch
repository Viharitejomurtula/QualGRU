#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:1

#SBATCH --mem=8G

#SBATCH --time=00:10:00

#SBATCH -J qgru-triton

#SBATCH -o /private/home/vtejomur/compression/triton-quant-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

python3 benchmarks/triton_quant_bench.py

