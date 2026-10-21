#!/usr/bin/env python3
"""threads.py LOG [seconds]: once the engine (strata-ds4) has answered its first request (LOG has a 'maya ds4 serve:
prompt' line), sample every thread's CPU time over `seconds` and print CPU% per thread with the CPU it last ran on."""
import os, sys, time

log, secs = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
hz = os.sysconf("SC_CLK_TCK")


def engine():
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                if open(f"/proc/{p}/comm").read().strip().startswith("strata-ds4"):
                    return int(p)
            except OSError:
                pass
    return None


t_end = time.time() + 600
pid = None
while time.time() < t_end:
    pid = engine()
    try:
        if pid and "maya ds4 serve: prompt" in open(log).read():
            break
    except OSError:
        pass
    time.sleep(0.5)
time.sleep(1.0)


def snap(pid):
    out = {}
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            f = open(f"/proc/{pid}/task/{t}/stat").read()
            name = f[f.index("(") + 1:f.rindex(")")]
            rest = f[f.rindex(")") + 2:].split()
            out[int(t)] = (name, int(rest[11]) + int(rest[12]), int(rest[36]))
        except (OSError, ValueError):
            pass
    return out


a = snap(pid)
time.sleep(secs)
b = snap(pid)
rows = []
for t, (name, cpu1, psr) in b.items():
    if t in a:
        rows.append((100.0 * (cpu1 - a[t][1]) / hz / secs, t, name, psr))
rows.sort(reverse=True)
print(f"engine {pid}: {len(rows)} threads, total {sum(r[0] for r in rows):.0f}% over {secs:.0f} s")
for pc, t, name, psr in rows[:24]:
    print(f"  tid {t:8d} {name:16s} cpu{psr:3d}  {pc:6.1f}%")
