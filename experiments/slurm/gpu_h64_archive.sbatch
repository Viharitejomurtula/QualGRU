#!/bin/bash

#SBATCH -p gpu

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 4

#SBATCH --gres=gpu:A100:1

#SBATCH --mem=48G

#SBATCH --time=01:00:00

#SBATCH -J qgru-h64arc

#SBATCH -o /private/home/vtejomur/compression/gpu-h64arc-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

python3 -u benchmarks/gpu_codec_compare_h64_archive_32k_1024.py

