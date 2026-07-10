"""Standalone checker: python eval/check.py <tasks.json> <results.json>"""
import json
import re
import sys

tasks = {t["task_id"]: t for t in json.load(open(sys.argv[1], encoding="utf-8"))}
res = {r["task_id"]: r["answer"] for r in json.load(open(sys.argv[2], encoding="utf-8"))}

n_ok = n_bad = n_manual = 0
for tid, t in tasks.items():
    a = res.get(tid, "")
    status = "MANUAL"
    if "expect_num" in t:
        nums = set()
        for x in re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", a):
            try:
                nums.add(float(x.replace(",", "").replace("$", "")))
            except ValueError:
                pass
        status = "OK" if float(t["expect_num"]) in nums else "BAD"
    elif "expect_contains" in t:
        needle = t["expect_contains"].lower().replace(" ", "")
        status = "OK" if needle in a.lower().replace(" ", "") else "BAD"
    n_ok += status == "OK"
    n_bad += status == "BAD"
    n_manual += status == "MANUAL"
    snippet = a.replace("\n", " | ")[:140]
    print(f"[{status}] {tid}: {snippet}")
print(f"---- {n_ok} OK, {n_bad} BAD, {n_manual} manual ----")
