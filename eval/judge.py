"""LLM-judge for local answers — mimics the hackathon's accuracy gate.

Usage:
    python eval/judge.py [tasks_json] [results_json]

Reads FIREWORKS_API_KEY from .env (repo root) or the environment. Uses a
strong Fireworks serverless model to judge each answer against the task's
intent, prints per-task verdicts and the total pass rate.
"""
import json
import os
import pathlib
import re
import ssl
import sys
import time
import urllib.request

# Avast Web Shield MITMs HTTPS with a CA cert that OpenSSL 3 rejects
# ("Basic Constraints of CA cert not marked critical"). This is a local dev
# tool, so fall back to an unverified context when that happens.
_SSL_CTX = None


def _urlopen(req, timeout):
    global _SSL_CTX
    try:
        return urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX)
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e) and _SSL_CTX is None:
            _SSL_CTX = ssl.create_default_context()
            _SSL_CTX.check_hostname = False
            _SSL_CTX.verify_mode = ssl.CERT_NONE
            sys.stderr.write("[judge] AV MITM cert detected - continuing unverified\n")
            return urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX)
        raise

ROOT = pathlib.Path(__file__).resolve().parent.parent

# load .env
envf = ROOT / ".env"
if envf.exists():
    for ln in envf.read_text(encoding="utf-8").splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

KEY = os.environ.get("FIREWORKS_API_KEY", "")
if not KEY:
    sys.exit("FIREWORKS_API_KEY missing (put it in .env)")

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "accounts/fireworks/models/deepseek-v4-pro")
URL = "https://api.fireworks.ai/inference/v1/chat/completions"

JUDGE_SYS = (
    "You are a strict but fair evaluation judge. You are given a task prompt "
    "and a candidate answer produced by an AI agent. Decide whether the answer "
    "correctly and adequately fulfills the task's intent: factual correctness, "
    "correct final results for math/logic, working code that satisfies the "
    "stated requirements, correct labels, obeyed format/length constraints. "
    "Minor stylistic differences are fine. Reply with ONLY a JSON object: "
    '{"pass": true|false, "reason": "<one short sentence>"}'
)


def judge(task, answer, retries=2):
    body = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user",
             "content": f"TASK:\n{task}\n\nCANDIDATE ANSWER:\n{answer}"},
        ],
        "max_tokens": 2000,
        "temperature": 0,
    }
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                URL, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {KEY}"})
            with _urlopen(req, timeout=120) as r:
                resp = json.loads(r.read().decode("utf-8", "replace"))
            txt = resp["choices"][0]["message"]["content"] or ""
            m = re.search(r'\{[^{}]*"pass"[^{}]*\}', txt, re.S)
            if m:
                v = json.loads(m.group(0))
                return bool(v.get("pass")), str(v.get("reason", ""))[:200]
            return None, f"unparsed: {txt[:120]}"
        except Exception as e:
            if attempt == retries:
                return None, f"judge error: {e}"
            time.sleep(2)


def main():
    tasks_file = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "eval" / "full_sweep.json")
    results_file = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "eval" / "out" / "results.json")
    tasks = {t["task_id"]: t["prompt"] for t in json.load(open(tasks_file, encoding="utf-8"))}
    results = {r["task_id"]: r["answer"] for r in json.load(open(results_file, encoding="utf-8"))}

    n_pass = n_fail = n_err = 0
    fails = []
    for tid, prompt in tasks.items():
        ans = results.get(tid, "")
        ok, reason = judge(prompt, ans)
        tag = "PASS" if ok else ("FAIL" if ok is False else "ERR ")
        if ok: n_pass += 1
        elif ok is False:
            n_fail += 1
            fails.append((tid, reason))
        else: n_err += 1
        print(f"[{tag}] {tid}: {reason}")

    total = n_pass + n_fail
    print(f"\n==== JUDGE: {n_pass}/{total} pass "
          f"({100.0 * n_pass / max(1, total):.1f}%), errors: {n_err} ====")
    print("gate needs 84.2% (16/19) — margin matters, aim for 100%")
    if fails:
        print("\nFAILED:")
        for tid, r in fails:
            print(f"  {tid}: {r}")


if __name__ == "__main__":
    main()
