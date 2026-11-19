#!/bin/bash
# Each model uncapped, then at each phone tier's not-visible limit, highest first.
cd "$(dirname "$0")"
run() { echo "### $1 cap=$2 $(date +%T)"; python3 -I memtest.py "$1" "$2" 2>&1; }
L=lfm2.5-1.2b-instruct-8da4w-gptq-4k; Q=qwen3-1.7b-8da4w-gptq-4k; B=qwen3-4b-8da4w-gptq-4k
run $L none; run $L 2048; run $L 1024
run $Q none; run $Q 3072; run $Q 2048; run $Q 1024
run $B none; run $B 5120; run $B 4096; run $B 3072; run $B 2048
echo "### done $(date +%T)"
