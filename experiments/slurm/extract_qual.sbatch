#!/bin/bash

#SBATCH -p short

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 2

#SBATCH --mem=8G

#SBATCH --time=01:00:00

#SBATCH -J extract-qual

#SBATCH -o /private/home/vtejomur/compression/extract-qual-%j.out

set -euo pipefail



IN=/private/home/vtejomur/compression/compressor_ont/CoLoRd/chr20_slice.fastq

OUT=/private/home/vtejomur/compression/chr20_slice.qualities.txt



awk 'NR % 4 == 0' "$IN" > "$OUT"



echo "file:"

ls -lh "$OUT"



echo "reads:"

wc -l "$OUT"



echo "symbols:"

awk '{n += length($0)} END {print n}' "$OUT"

