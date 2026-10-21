#!/bin/bash
# monitor.sh OUTDIR: samples GPU (util, mem, PCIe gen/throughput), NVMe, memory/swap/page faults and the engine's
# CPU / page faults once a second until killed (kill the process group).  summarize.py reads the files.
OUT=${1:?outdir}
mkdir -p "$OUT"
ENG=${2:-$(pgrep -x strata-ds4 | head -1)}
nvidia-smi --query-gpu=timestamp,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem \
  --format=csv,noheader,nounits -lms 1000 > "$OUT/gpu.csv" 2>&1 &
nvidia-smi dmon -s t -d 1 > "$OUT/pcie.txt" 2>&1 &
iostat -dxm -y nvme0n1 1 > "$OUT/iostat.txt" 2>&1 &
vmstat -w -n 1 > "$OUT/vmstat.txt" 2>&1 &
[ -n "$ENG" ] && pidstat -r -u -h -p "$ENG" 1 > "$OUT/pidstat.txt" 2>&1 &
echo "$ENG" > "$OUT/engine.pid"
trap 'kill $(jobs -p) 2>/dev/null; exit 0' TERM INT
wait
