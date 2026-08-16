#!/bin/bash

#SBATCH -p short

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 8

#SBATCH --mem=16G

#SBATCH --time=00:30:00

#SBATCH -J build-qgru

#SBATCH -o /private/home/vtejomur/compression/build-qgru-%j.out

set -euo pipefail

cd /private/home/vtejomur/compression/QualGRU-gpu

bash compressor/build.sh

find compressor -maxdepth 3 -type f -name qualgru -perm -111 -ls

