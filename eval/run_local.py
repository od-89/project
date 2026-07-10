"""Local dev harness (no Docker needed).

Usage (from repo root):
    python eval/run_local.py [tasks_json] [--server]   # --server spawns llama-server

Set LLAMA_URL to reuse an already-running llama-server (fastest iteration):
    tools/llama/llama-server.exe -m models/Qwen2.5-3B-Instruct-Q4_K_M.gguf -c 4096 --port 8091 --jinja
    set LLAMA_URL=http://127.0.0.1:8091   (then run this script)

Auto-checks tasks that carry expect_num / expect_contains; prints everything
for eyeballing the rest.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

tasks_file = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
    else str(ROOT / "eval" / "variants.json")

tasks = json.load(open(tasks_file, encoding="utf-8"))
run_tasks = [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in tasks]

out_dir = ROOT / "eval" / "out"
out_dir.mkdir(parents=True, exist_ok=True)
stem = pathlib.Path(tasks_file).stem
inp = out_dir / f"tasks-{stem}.json"
outp = out_dir / f"results-{stem}.json"
inp.write_text(json.dumps(run_tasks), encoding="utf-8")
if outp.exists():
    outp.unlink()

env = dict(os.environ)
env.setdefault("INPUT_PATH", str(inp))
env["OUTPUT_PATH"] = str(outp)
env.setdefault("MODE", "zero")
env.setdefault("SOFT_DEADLINE", "3000")
env.setdefault("HARD_DEADLINE", "3300")
env.setdefault("LLAMA_THREADS", str(os.cpu_count() or 4))
env.setdefault("LLAMA_BIN", str(ROOT / "tools" / "llama" / "llama-server.exe"))
env.setdefault("MODEL_PATH", str(ROOT / "models" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"))

t0 = time.time()
proc = subprocess.run([sys.executable, "-m", "agent.main"], cwd=str(ROOT), env=env)
dt = time.time() - t0

results = {r["task_id"]: r["answer"] for r in json.loads(outp.read_text(encoding="utf-8"))}

n_ok = n_bad = n_manual = 0
for t in tasks:
    tid = t["task_id"]
    ans = results.get(tid, "<MISSING>")
    status = "MANUAL"
    if "expect_num" in t:
        nums = [x.replace(",", "").replace("$", "") for x in
                re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", ans)]
        vals = {float(x) for x in nums if x}
        status = "OK" if float(t["expect_num"]) in vals else "BAD"
    elif "expect_contains" in t:
        needle = t["expect_contains"].lower().replace(" ", "")
        status = "OK" if needle in ans.lower().replace(" ", "") else "BAD"
    n_ok += status == "OK"
    n_bad += status == "BAD"
    n_manual += status == "MANUAL"
    print(f"\n=== {tid} [{status}]\nQ: {t['prompt'][:150]}\nA: {ans}")

print(f"\n---- auto-checked: {n_ok} OK, {n_bad} BAD, {n_manual} manual ---- "
      f"({dt:.0f}s wall)")
