#!/usr/bin/env python3
"""drive.py: one benchmark configuration of strata-ds4, through its serve protocol (the path Maya's server uses).

    drive.py --label NAME [--exe BIN] [--args "..."] [--env K=V ...] [--prompts p1,p2,...] [--max-new 512]
             [--cold] [--seed 1234] [--profile FILE] [--repeat 1] [--no-monitor]

Starts the engine with the config's base args (pack, context, --tp 2) plus --args, a fresh copy of the frozen expert
profile (the engine rewrites the one it is given), DS4_SERVE_PROF=1; waits for READY; sends each prompt as
GEN <max_new> temperature=1 top_p=1 top_k=64 seed=S <ids> (the server's sampling); reads DONE; QUITs.
--cold: the model files' pages dropped from the page cache first (posix_fadvise DONTNEED; no root needed).
Writes runs/<label>/{engine.log,results.json,monitor files} and prints a summary.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # (the repo: tools/ds4/bench3090/..)
MODEL = Path(os.environ.get("DS4_BENCH_MODEL", "/home/aiuser/models/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw"))
PACK = MODEL / "maya-ds4"
BASE_ARGS = ["--pack", str(PACK), "--max-context", "65536", "--tp", "2"]
LIBS = "/usr/local/cuda-13.4/lib64:/usr/local/cuda-13.4/targets/x86_64-linux/lib"

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--exe", default=str(ROOT / "build" / "strata-ds4"))
ap.add_argument("--args", default="")
ap.add_argument("--base-args", default=None, help="replace the base args (pack, context, tp)")
ap.add_argument("--env", action="append", default=[])
ap.add_argument("--prompts", default="code,math,essay,tcp,puzzle,zh")
ap.add_argument("--max-new", type=int, default=512)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--temperature", default="1")
ap.add_argument("--profile", default=str(HERE / "profiles" / "live-2026-10-21.bin"))
ap.add_argument("--cold", action="store_true")
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--no-monitor", action="store_true")
ap.add_argument("--scope", default="", help="run inside systemd-run --user --scope with these -p properties, "
                                            "e.g. 'MemorySwapMax=0'")
a = ap.parse_args()

out = HERE / "runs" / a.label
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)


def drop_cache():
    n = 0
    for f in sorted(MODEL.glob("model-*.safetensors")):
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        n += 1
    return n


if a.cold:
    print(f"[drive] dropped the page cache of {drop_cache()} model files", flush=True)

prof = out / "expert-profile.bin"
shutil.copy(a.profile, prof)
base = a.base_args.split() if a.base_args is not None else BASE_ARGS
cmd = [a.exe, "--serve"] + base + a.args.split() + ["--expert-profile", str(prof)]
if a.scope:
    cmd = ["systemd-run", "--user", "--scope", "--quiet"] + [x for p in a.scope.split() for x in ("-p", p)] + cmd
env = dict(os.environ)
env["LD_LIBRARY_PATH"] = LIBS + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
env["DS4_SERVE_PROF"] = "1"
for kv in a.env:
    k, v = kv.split("=", 1)
    env[k] = v
(out / "cmd.txt").write_text(" ".join(cmd) + "\n" + "\n".join(a.env) + "\n")
log = open(out / "engine.log", "w")
t_start = time.time()
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                     cwd=str(ROOT), env=env)
for line in p.stdout:
    if line.startswith("READY"):
        break
else:
    print("[drive] engine exited before READY; see", out / "engine.log")
    sys.exit(1)
load_s = time.time() - t_start
print(f"[drive] {a.label}: ready in {load_s:.1f} s", flush=True)

mon = None
if not a.no_monitor:
    mon = subprocess.Popen(["setsid", str(HERE / "monitor.sh"), str(out / "mon"), str(p.pid)], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

results = []
names = a.prompts.split(",")
for rep in range(a.repeat):
    for name in names:
        ids = (HERE / "prompts" / f"{name}.ids").read_text().strip()
        n_ids = ids.count(",") + 1
        keys = f"temperature={a.temperature} top_p=1 top_k=64 seed={a.seed}"
        p.stdin.write(f"GEN {a.max_new} {keys} {ids}\n")
        p.stdin.flush()
        done = None
        for line in p.stdout:
            if line.startswith("DONE "):
                done = line.split()
                break
            if line.startswith("ERR "):
                print("[drive]", name, line.strip(), flush=True)
        if done is None:
            print("[drive] engine died; see", out / "engine.log")
            sys.exit(1)
        produced, plen, pms, dms = int(done[1]), int(done[2]), float(done[3]), float(done[4])
        acc, off, reused, hits, lookups = int(done[6]), int(done[7]), int(done[8]), int(done[9]), int(done[10])
        time.sleep(0.3)   # (the engine's stderr lines for the request)
        r = {"rep": rep, "prompt": name, "prompt_tokens": plen, "reused": reused, "prefill_ms": pms,
             "prefill_tok_s": round(1000 * (plen - reused) / pms, 1) if pms > 0 else None,
             "generated": produced, "decode_ms": dms, "tok_s": round(1000 * produced / dms, 2),
             "finish": done[5], "drafts_accepted": acc, "drafts_offered": off,
             "vram_hit": round(hits / lookups, 4) if lookups else None}
        results.append(r)
        print(f"[drive] {name:8s} prompt {plen:6d} ({r['prefill_tok_s']} tok/s)  gen {produced:4d}  "
              f"{r['tok_s']:6.2f} tok/s  drafts {acc}/{off}  vram_hit {r['vram_hit']}", flush=True)
p.stdin.write("QUIT\n")
p.stdin.flush()
try:
    p.wait(timeout=120)
except subprocess.TimeoutExpired:
    p.kill()
if mon is not None:
    os.killpg(mon.pid, 15)
    mon.wait()
log.close()
gen = sum(r["generated"] for r in results)
ms = sum(r["decode_ms"] for r in results)
summary = {"label": a.label, "cmd": " ".join(cmd), "env": a.env, "load_s": round(load_s, 1), "tokens": gen,
           "decode_s": round(ms / 1000, 1), "tok_s": round(1000 * gen / ms, 2) if ms else None,
           "per_prompt_tok_s": [r["tok_s"] for r in results],
           "accept_rate": round(sum(r["drafts_accepted"] for r in results) /
                                max(1, sum(r["drafts_offered"] for r in results)), 3),
           "vram_hit": round(sum((r["vram_hit"] or 0) for r in results) / len(results), 4)}
(out / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=1))
print(f"[drive] {a.label}: {summary['tok_s']} tok/s over {gen} tokens ({summary['decode_s']} s decode); "
      f"per prompt {summary['per_prompt_tok_s']}; drafts kept {summary['accept_rate']}", flush=True)
if mon is not None:
    subprocess.run([sys.executable, "-I", str(HERE / "summarize.py"), str(out / "mon")])
