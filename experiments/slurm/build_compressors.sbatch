#!/bin/bash

#SBATCH -p short

#SBATCH -N 1

#SBATCH -n 1

#SBATCH -c 8

#SBATCH --mem=32G

#SBATCH --time=01:00:00

#SBATCH -J build-codecs

#SBATCH -o /private/home/vtejomur/compression/build-codecs-%j.out



set -u



ROOT=/private/home/vtejomur/compression/bench_tools

mkdir -p "$ROOT"

cd "$ROOT"



echo "===== fqzcomp5 ====="

if [ ! -d fqzcomp5 ]; then

    git clone --recursive https://github.com/jkbonfield/fqzcomp5.git

fi

cd fqzcomp5

git submodule update --init --recursive

make -j8 || true

cd "$ROOT"



echo "===== ENANO ====="

if [ ! -d EnanoFASTQ ]; then

    git clone https://github.com/guilledufort/EnanoFASTQ.git

fi

cd EnanoFASTQ/enano

make -j8 || true

cd "$ROOT"



echo "===== SPRING ====="

if [ ! -d SPRING ]; then

    git clone https://github.com/shubhamchandak94/SPRING.git

fi

mkdir -p SPRING/build

cd SPRING/build

cmake .. || true

make -j8 || true

cd "$ROOT"



echo "===== QVZ ====="

if [ ! -d qvz ]; then

    git clone https://github.com/mikelhernaez/qvz.git

fi

cd qvz

make -j8 || true

cd "$ROOT"



echo

echo "===== BUILT EXECUTABLES ====="

find "$ROOT" -maxdepth 4 -type f -perm -111 | sort

