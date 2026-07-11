"""Entrypoint: read /input/tasks.json -> answer all tasks -> /output/results.json.

Design: answer-then-improve under a strict wall-clock budget.
  Pass 1 banks a best-shot answer for every task (cheap categories first).
  Pass 2 spends remaining time re-verifying low-confidence answers.
  A watchdog guarantees a complete, valid results.json and exit code 0.

MODE=zero    -> never touches Fireworks (0 scored tokens).
MODE=hybrid  -> escalates still-low-confidence tasks via FIREWORKS_BASE_URL.
"""
import json
import os
import sys
import threading
import time

from .classify import classify
from .local_llm import LocalLLM
from .solvers import HANDLERS

T0 = time.time()
SOFT_DEADLINE = float(os.environ.get("SOFT_DEADLINE", "500"))   # stop optional work
HARD_DEADLINE = float(os.environ.get("HARD_DEADLINE", "560"))   # flush + exit
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
MODE = os.environ.get("MODE", "zero").lower()
ESC_MAX = int(os.environ.get("ESC_MAX", "6"))
ESC_CONF = float(os.environ.get("ESC_CONF", "0.55"))

_lock = threading.Lock()
_results = {}          # task_id -> answer str
_order = []            # task ids in input order


def elapsed():
    return time.time() - T0


def log(msg):
    sys.stderr.write(f"[main +{elapsed():6.1f}s] {msg}\n")


def flush():
    with _lock:
        data = [{"task_id": tid, "answer": _results.get(tid, "") or
                 "Unable to determine within the time limit."}
                for tid in _order]
    tmp = OUTPUT_PATH + ".tmp"
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=1)
    os.replace(tmp, OUTPUT_PATH)


def _watchdog():
    while True:
        left = HARD_DEADLINE - elapsed()
        if left <= 0:
            break
        time.sleep(min(left, 1.0))
    log("watchdog fired: flushing and exiting")
    try:
        flush()
    finally:
        os._exit(0)


class Ctx:
    """What handlers see: chat access + time awareness.

    fast=True is the Pass-1 mode: only cheap verification (<=18s asks) is
    allowed so every task banks an answer quickly; the expensive checks run
    in Pass 2 with whatever wall-clock remains.
    """

    def __init__(self, llm):
        self.llm = llm
        self.task_deadline = None  # absolute time.time() cap for current task
        self.fast = False

    def chat(self, system, user, **kw):
        if elapsed() > HARD_DEADLINE - 6:
            return ""
        return self.llm.chat(system, user, **kw)

    def have_time(self, seconds_needed: float) -> bool:
        if self.fast and seconds_needed > 18:
            return False
        if elapsed() + seconds_needed > SOFT_DEADLINE:
            return False
        if self.task_deadline and time.time() + seconds_needed > self.task_deadline:
            return False
        return True


# Cheap/fast categories first so answers get banked early.
_CAT_ORDER = ["sentiment", "ner", "summarize", "factual",
              "math", "code_gen", "code_debug", "logic"]


def run():
    global _order
    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        assert isinstance(tasks, list)
    except Exception as e:
        log(f"FATAL: cannot read tasks: {e}")
        _order = []
        flush()
        return 0

    items = []
    for t in tasks:
        tid = str(t.get("task_id", "")) or f"task-{len(items)+1}"
        prompt = str(t.get("prompt", "") or "")
        items.append({"id": tid, "prompt": prompt, "cat": classify(prompt)})
    _order[:] = [it["id"] for it in items]
    log(f"loaded {len(items)} tasks: " +
        ", ".join(f"{it['id']}={it['cat']}" for it in items))
    flush()  # valid (placeholder) file exists from the very start

    threading.Thread(target=_watchdog, daemon=True).start()

    llm = LocalLLM()
    llm.start()

    # hybrid: escalate-first. Fireworks answers the hard categories in a
    # parallel wave at t~0 (immune to local CPU speed); the local pipeline
    # covers easy categories and any failed calls.
    fw_done = {}
    if MODE == "hybrid":
        try:
            from concurrent.futures import ThreadPoolExecutor
            from .fireworks import fw_answer
            HARD = {"factual", "math", "logic", "code_debug", "code_gen"}
            hard_items = [it for it in items if it["cat"] in HARD][:ESC_MAX]

            def _esc(it):
                text, tok = fw_answer(it["prompt"], it["cat"])
                return it["id"], text

            if hard_items:
                with ThreadPoolExecutor(max_workers=4) as ex:
                    for tid, text in ex.map(_esc, hard_items):
                        if text:
                            fw_done[tid] = text
                log(f"escalate-first wave: {len(fw_done)}/{len(hard_items)} answered by Fireworks")
                with _lock:
                    _results.update(fw_done)
                flush()
        except Exception as e:
            log(f"escalate-first error (falling back to local): {e}")

    if not llm.wait_ready(timeout=90):
        log("FATAL: local model failed to start")
        flush()
        return 0
    log("local model ready")

    ctx = Ctx(llm)
    work = sorted((it for it in items if it["id"] not in fw_done),
                  key=lambda it: _CAT_ORDER.index(it["cat"]))
    confs = {it["id"]: 0.85 for it in items if it["id"] in fw_done}

    # ---- Pass 1 (fast): bank an answer for everything quickly
    ctx.fast = True
    for i, it in enumerate(work):
        remaining = max(1, len(work) - i)
        budget = max(12.0, (SOFT_DEADLINE * 0.62 - elapsed()) / remaining * 1.25)
        ctx.task_deadline = time.time() + budget
        try:
            res = HANDLERS[it["cat"]](it["prompt"], ctx)
        except Exception as e:
            log(f"handler error on {it['id']}: {e}")
            res = {"answer": "", "conf": 0.1, "cat": it["cat"]}
        if not res.get("answer"):
            try:
                res["answer"] = ctx.chat(
                    "Answer the task as well as you can, concisely.",
                    it["prompt"], temperature=0.0, max_tokens=180) or ""
                res["conf"] = min(res.get("conf", 0.3), 0.4)
            except Exception:
                pass
        with _lock:
            _results[it["id"]] = res.get("answer", "")
        confs[it["id"]] = res.get("conf", 0.3)
        flush()
        log(f"[{i+1}/{len(work)}] {it['id']} cat={it['cat']} "
            f"conf={confs[it['id']]:.2f} len={len(res.get('answer',''))}")

    # ---- Pass 2 (full): re-verify ascending by confidence while time remains
    ctx.fast = False
    ctx.task_deadline = None
    weak = sorted((it for it in work if confs[it["id"]] < 0.9),
                  key=lambda it: confs[it["id"]])
    pass2_cut = SOFT_DEADLINE - (100 if MODE == "hybrid" else 25)
    for it in weak:
        if elapsed() > pass2_cut:
            break
        log(f"pass2 verify {it['id']} (conf={confs[it['id']]:.2f})")
        try:
            res = HANDLERS[it["cat"]](it["prompt"], ctx)
        except Exception as e:
            log(f"pass2 error {it['id']}: {e}")
            continue
        if res.get("answer") and res.get("conf", 0) > confs[it["id"]]:
            with _lock:
                _results[it["id"]] = res["answer"]
            confs[it["id"]] = res["conf"]
            flush()

    # ---- Hybrid escalation (never in zero mode)
    if MODE == "hybrid":
        from .fireworks import fw_answer, spent
        esc = [it for it in work if confs[it["id"]] < ESC_CONF]
        esc.sort(key=lambda it: confs[it["id"]])
        for it in esc[:ESC_MAX]:
            if elapsed() > HARD_DEADLINE - 35:
                break
            text, tok = fw_answer(it["prompt"], it["cat"])
            if text:
                with _lock:
                    _results[it["id"]] = text
                confs[it["id"]] = 0.8
                flush()
                log(f"escalated {it['id']} via Fireworks ({tok} tokens)")
        log(f"fireworks usage: {spent()}")

    flush()
    low = [f"{k}={v:.2f}" for k, v in confs.items() if v < 0.6]
    log(f"done in {elapsed():.1f}s; low-conf: {low or 'none'}")
    llm.stop()
    return 0


if __name__ == "__main__":
    try:
        code = run()
    except Exception as e:  # absolute last resort: still emit valid output
        log(f"UNCAUGHT: {e}")
        try:
            flush()
        except Exception:
            pass
        code = 0
    sys.exit(code)
