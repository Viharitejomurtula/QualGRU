#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=16G

#SBATCH --time=00:20:00

#SBATCH -J qgru-codec16k

#SBATCH -o /private/home/vtejomur/compression/gpu-codec16k-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

hostname

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 benchmarks/gpu_codec_proto.py

