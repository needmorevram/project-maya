#!/usr/bin/env python3
"""Send one chat request to the Maya server and report decode speed from the client side.

usage: req.py [--port 8080] [--max-tokens N] [--seed S] [--prompt-file F | --prompt TEXT] [--effort high|none]
The engine's own line in maya-ds4.log ("N generated in X ms") is the authoritative decode number; this prints the
client-side view (time to first token, tokens after the first / time after the first) as a cross-check.
"""
import argparse, json, time, urllib.request

DEFAULT_PROMPT = ("Write a long, detailed technical essay on the history of the steam engine, from Hero of Alexandria "
                  "to the triple-expansion marine engine. Cover the physics, the key inventors, and the economic impact. "
                  "Use many sections and go into depth.")

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--max-tokens", type=int, default=2048)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--prompt", default=DEFAULT_PROMPT)
ap.add_argument("--prompt-file")
ap.add_argument("--effort", default=None, help="reasoning_effort to send (default: the server's)")
a = ap.parse_args()
prompt = open(a.prompt_file, encoding="utf-8").read() if a.prompt_file else a.prompt
body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": a.max_tokens, "seed": a.seed, "stream": True, "stream_options": {"include_usage": True}}
if a.effort:
    body["reasoning_effort"] = a.effort
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time(); t_first = None; n_chunks = 0; usage = None
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                n_chunks += 1
                if t_first is None:
                    t_first = time.time()
t1 = time.time()
gen = (usage or {}).get("completion_tokens") or n_chunks
ptok = (usage or {}).get("prompt_tokens")
ttft = (t_first - t0) if t_first else float("nan")
dec = (gen - 1) / (t1 - t_first) if t_first and gen > 1 else float("nan")
print(json.dumps({"prompt_tokens": ptok, "completion_tokens": gen, "ttft_s": round(ttft, 2),
                  "decode_tok_s_client": round(dec, 2), "wall_s": round(t1 - t0, 1)}))
