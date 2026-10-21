#!/usr/bin/env python3
"""summarize.py OUTDIR: averages of monitor.sh's samples over the run (GPU, PCIe, NVMe, memory, page faults)."""
import re, statistics as st, sys
from pathlib import Path

d = Path(sys.argv[1])


def mean(xs):
    return round(st.mean(xs), 1) if xs else None


out = {}
gen, util, memu, vram, pw = [], [], [], [], []
for line in (d / "gpu.csv").read_text().splitlines():
    p = [x.strip() for x in line.split(",")]
    if len(p) < 9 or not p[1].isdigit():
        continue
    gen.append(int(p[1])); util.append(float(p[3])); memu.append(float(p[4])); vram.append(float(p[5]))
    pw.append(float(p[6]))
if gen:
    out["pcie_gen"] = f"min {min(gen)} max {max(gen)} mode {st.mode(gen)}"
    out["gpu_util_%"] = mean(util); out["gpu_mem_bw_util_%"] = mean(memu); out["vram_MiB"] = max(vram)
    out["power_W"] = mean(pw)
rx, tx = [], []
for line in (d / "pcie.txt").read_text().splitlines():
    p = line.split()
    if p and p[0].isdigit() and len(p) >= 3:
        rx.append(float(p[1])); tx.append(float(p[2]))
out["pcie_rx_MBps(host->gpu)"] = mean(rx); out["pcie_tx_MBps"] = mean(tx)
rmb, rps, aw = [], [], []
hdr = None
for line in (d / "iostat.txt").read_text().splitlines():
    p = line.split()
    if p and p[0] == "Device":
        hdr = p
    elif hdr and p and p[0].startswith("nvme"):
        row = dict(zip(hdr, p))
        rmb.append(float(row.get("rMB/s", 0))); rps.append(float(row.get("r/s", 0)))
        aw.append(float(row.get("r_await", 0)))
out["nvme_read_MBps"] = mean(rmb); out["nvme_read_iops"] = mean(rps); out["nvme_r_await_ms"] = mean(aw)
si, so, free, cache, us, sy, wa = [], [], [], [], [], [], []
hdr = None
for line in (d / "vmstat.txt").read_text().splitlines():
    p = line.split()
    if "si" in p and "so" in p:
        hdr = p
    elif hdr and p and p[0].isdigit():
        row = dict(zip(hdr, p))
        si.append(float(row["si"])); so.append(float(row["so"])); free.append(float(row["free"]) / 1e6)
        cache.append(float(row["cache"]) / 1e6); us.append(float(row["us"])); sy.append(float(row["sy"]))
        wa.append(float(row["wa"]))
out["swap_in_KBps"] = mean(si); out["swap_out_KBps"] = mean(so); out["ram_free_GB"] = mean(free)
out["page_cache_GB"] = mean(cache); out["cpu_user_%"] = mean(us); out["cpu_sys_%"] = mean(sy); out["cpu_iowait_%"] = mean(wa)
majf, minf, cpu, rss = [], [], [], []
hdr = None
for line in (d / "pidstat.txt").read_text().splitlines():
    p = line.split()
    if p and p[0] == "#":
        hdr = p[1:]
    elif hdr and p and p[0][:1].isdigit():
        if len(p) == len(hdr) + 1:   # "06:48:57 AM": the time is two fields
            p = [p[0]] + p[2:]
        row = dict(zip(hdr, p))
        try:
            majf.append(float(row["majflt/s"])); minf.append(float(row["minflt/s"])); cpu.append(float(row["%CPU"]))
            rss.append(float(row["RSS"]) / 1e6)
        except (KeyError, ValueError):
            pass
out["engine_majflt_per_s"] = mean(majf); out["engine_minflt_per_s"] = mean(minf); out["engine_cpu_%"] = mean(cpu)
out["engine_rss_GB"] = max(rss) if rss else None
for k, v in out.items():
    print(f"{k:28s} {v}")
