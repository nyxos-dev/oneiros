#!/bin/bash
# Reproduce the current Oneiros champion: MLP token model over a BPE vocab of 2048.
# Best held-out result so far: 2.32 bits/byte (I5). All from-scratch, no ML libs.
set -e
cd "$(dirname "$0")"
STEPS="${1:-2000000}"

bash data/make_corpus.sh                                  # regenerate corpus from nyx-os
gcc -O2 -march=native -o build/bpe      src/bpe.c
gcc -O2 -march=native -DV=2048 -o build/oneiros_2048 src/oneiros.c -lm

./build/bpe learn data/corpus_clean.txt 1792 data/vocab2048.bin data/corpus2048.tok
./build/oneiros_2048 train data/corpus2048.tok "$STEPS" data/oneiros_2048.bin

echo "== sample =="
./build/oneiros_2048 sample data/oneiros_2048.bin 60 0.6 "static void " data/vocab2048.bin
