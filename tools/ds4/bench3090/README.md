# DeepSeek-V4.1-Flash-1017 SAGE 1.62 bpw on one RTX 3090: benchmark harness and results

The box: RTX 3090 24 GB (PCIe 4.0 x16 on the CPU's root port), Core i5-13400 (6 P-cores + 4 E-cores, AVX2, no
AVX-512), 62 GiB DDR5, model on a Kingston NV3 NVMe (DRAM-less), 8 GB swap file on the same disk. Engine:
`build/strata-ds4` (branch `ds4-perf-3090`). Model: `/home/aiuser/models/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw`
(set `DS4_BENCH_MODEL` for another folder).

## Build

```sh
PATH=$PWD/.venv/bin:/usr/local/cuda-13.4/bin:$PATH ninja -C build strata-ds4
```

## Decode benchmark (the serve path Maya's server uses)

`drive.py` starts the engine with the given args plus a fresh copy of the frozen expert profile
(`profiles/live-2026-10-21.bin`; the engine rewrites the one it is given, and that file decides what lands in VRAM),
sends each prompt as `GEN 512 temperature=1 top_p=1 top_k=64 seed=1234 <ids>` (the server's sampling), and reports the
engine's own DONE timings plus a `prof:` line per request (`DS4_SERVE_PROF=1`). Stop Maya first (it needs the GPU).

```sh
cd tools/ds4/bench3090
# --cold: the model files dropped from the page cache first (no root needed), so runs start alike
python3 drive.py --label rec --cold --prompts code,math,essay,tcp,puzzle,zh --max-new 512 \
    --args "--prefill-chunk 1024 --cache-routing 3.0,1,r --ram-routing 0.3 --dspark-conf 0.65"
python3 drive.py --label ctx --prompts ctx4k,ctx32k --max-new 512 --args "..."     # long context, TTFT
```

Run-to-run noise at temperature 1 is about +-10% for 3,072 tokens (the sampled text changes which experts route and
how many drafts stand), and one prompt can range 10-30 tok/s: compare configurations over 6 x 512 tokens or more, and
repeat the close ones. The page cache's state matters as much: a box that served for hours (~58 GB of expert pages
cached) decoded the same flags ~25-35% faster than a cold start.

## Quality (perplexity through the decode router)

```sh
./ppl.sh exact ""                                                   # 7.3003 (the reference)
./ppl.sh r3 "--cache-routing 3.0,1,r --ram-routing 0.3"             # 7.5288 (+3.1%)
```

`prompts/ppl6x768.txt`: six 768-token sequences of text written after the model's release (this repo's docs and code:
3 prose, 3 code). `--prefill-chunk 0` runs them in windows of 16 through the decode path, so cache routing applies.

## Results (2026-10-21)

Perplexity (exact routing 7.3003):

| --cache-routing / --ram-routing | perplexity | cost |
|---|---|---|
| exact | 7.300 | - |
| exact, DS4_RAM_TIER_GB=20 | 7.267 | -0.5% (numerics) |
| 6.0,2,r / 0.2 | 7.483 | +2.5% |
| **3.0,1,r / 0.3** | **7.529** | **+3.1%** |
| 0.8,1 | 7.820 | +7.1% |
| 6.0,1,r / 0.2 (the original setup) | 7.856 | +7.6% |
| 10.0,1,r / 0.2 | 7.994 | +9.5% |

Decode, 6 prompts x 512 tokens, cold start, sustained (engine-measured; vision on unless noted):

| configuration | tok/s | quality cost |
|---|---|---|
| exact routing (+ RAM tier 28 GB) | 6.8 | 0 |
| 6.0,1,r + ram 0.2 (original) | 17.5 (two passes: 18.5, 16.5) | +7.6% |
| 3.0,1,r + ram 0.3 | 17.6 / 19.3 (vision off) | +3.1% |
| **3.0,1,r + ram 0.3 + --dspark-conf 0.65** | **19.9** (passes 19.6, 20.4) | **+3.1%** |
| 6.0,2,r + ram 0.2 | 15.9 (vision off) | +2.5% |

Long context, recommended flags: 4K prompt prefill 35-39 tok/s (~3 min), decode 17-18 tok/s; 32K prompt prefill 52-56
tok/s (~10 min), decode ~10.5 tok/s (the prompt path streams every CPU-lane expert through the page cache once a chunk,
and the Engram rows fall out of it).

## Where the time goes (decode)

- The GPU waits for the CPU lane 40-60% of decode (device-side timers in `wait_seq` / `win_wait`, `prof:` line): the
  CPU lane runs ~20-34 routed experts a token at ~1.1-1.3 ms each on 10 cores (EXL3 trellis decode, compute-bound:
  ~2.6 G weights/s a thread in the engine, 9.75 G/s a P-core in a hot loop).
- The GPU's own work is near its memory-bandwidth floor: the dense FP8 GEMVs (8.6 GB) ~12.7 ms a round, VRAM experts
  ~9 ms, attention/mHC ~6 ms; host gaps (Engram reads, syncs) ~5 ms.
- PCIe: zero-copy reads of pinned host memory reach 17-19.6 GB/s, cudaMemcpy 13 GB/s (`zc_probe.cu`); CUDA cannot
  register the read-only file mappings (no in-place GPU reads of the page cache).

## Engine options added on this branch (all opt-in)

- `DS4_SERVE_PROF=1`: a `prof:` line per request (rounds, drafts, per-position split, the GPU's waits for the CPU lane).
- `DS4_RAM_TIER_GB=G`: the next-hottest CPU-lane experts as pinned, GPU-mapped blobs. Alone (the GPU always runs them
  over PCIe) it lost from a cold start (15.3 vs 17.2 tok/s): the zero-copy reads sit on the GPU's critical path.
- `DS4_RT_SPLIT=1` (with the tier): the service thread hands the GPU a balanced share of each window's RAM-tier experts
  first (ds_seq) and the CPU lane computes the rest from the pinned blobs - correct (greedy output identical), lane
  waits ~10 points lower, net within noise (+-8%).
- `DS4_MX_RT_MIN=N`: the windows a RAM-tier expert must be in before the cache moves it into VRAM (default 2).
- `--cpu-threads N` below the core count now puts the helper threads on the cores the lane leaves (here the E-cores)
  instead of the lane cores' hyperthreads.
- The prompt path drops a streamed expert's pages again unless the load read it ahead or decode used it lately
  (`DS4_PROMPT_DONTNEED=0`: off): prefill +8-11%, the following decode's CPU lane 64 -> 50 ms a position at 32K.

Tried and not kept (within noise or slower): `DS4_DSTREAM=1` (-6%: the staging copy costs what the CPU would),
`DS4_DRAFT_COST=1`, `--dspark none` (-16%), `--cpu-threads 6/7`, `--dspark-conf` 0.6 / 0.7 / 0.8, `DS4_EG_URING=1`
(Engram 11.3 -> 4.8 ms a position at 32K, decode unchanged: the CPU lane is the long pole there),
`--cache-routing 10.0,1,r` (no faster than 6.0, +9.5%).

The live server (`maya-ds4.json`) runs `--prefill-chunk 1024 --cache-routing 3.0,1,r --ram-routing 0.3 --dspark-conf 0.65`.
