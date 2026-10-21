#!/bin/bash
# ppl.sh LABEL "engine args" [ENV=V ...]: perplexity of prompts/ppl6x768.txt through the decode router (windows of 16)
B=$(cd "$(dirname "$0")" && pwd); L=$1; A=$2; shift 2
O=$B/runs/ppl-$L; mkdir -p $O; cp $B/profiles/live-2026-10-21.bin $O/ep.bin
cd "$B/../../.." || exit 1
t0=$(date +%s.%N)
env LD_LIBRARY_PATH=/usr/local/cuda-13.4/lib64 "$@" ./build/strata-ds4 --pack ${DS4_BENCH_MODEL:-/home/aiuser/models/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw}/maya-ds4 \
  --max-context 65536 --tp 2 --prefill-chunk 0 --expert-profile $O/ep.bin $A --ppl $B/prompts/ppl6x768.txt > $O/out.txt 2> $O/err.txt
t1=$(date +%s.%N)
echo "$L: $(cat $O/out.txt | tail -1) [$(grep -h '^ppl: sequence' $O/err.txt | awk '{printf "%s ", $NF}')] ($(echo "$t1 - $t0" | bc | cut -d. -f1) s)"
