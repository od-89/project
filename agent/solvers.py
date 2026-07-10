"""Per-category handlers.

Every handler returns a dict:
    {"answer": str, "conf": float, "cat": str}

conf semantics:
    >= 0.85  deterministically verified (programs agree / tests pass / majority)
    ~  0.6   single clean LLM answer, no independent check yet
    <= 0.35  something went wrong; candidate for escalation / retry

Handlers must never raise; they degrade to a best-effort answer.
"""
import re
import sys

from .pyexec import run_python

# ---------------------------------------------------------------- helpers

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S | re.I)


def log(msg):
    sys.stderr.write(f"[solver] {msg}\n")


def extract_code(text: str) -> str:
    m = _FENCE.findall(text or "")
    if m:
        return max(m, key=len).strip()
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").lstrip("python").strip()
    if "def " in t or "print(" in t or "import " in t:
        return t
    return t


def compiles(code: str):
    try:
        compile(code, "<gen>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?%?")


def normalize_num(tok: str):
    if tok is None:
        return None
    t = tok.strip().replace(",", "").replace("$", "").rstrip("%").rstrip(".")
    try:
        v = float(t)
        return v
    except ValueError:
        return None


def nums_equal(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def fmt_num(v: float) -> str:
    if v is None:
        return ""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s


def last_number(text: str):
    if not text:
        return None
    toks = _NUM.findall(text)
    return normalize_num(toks[-1]) if toks else None


def answer_line(text: str):
    """Extract the payload of the last 'ANSWER: ...' line."""
    if not text:
        return None
    hits = re.findall(r"ANSWER:\s*(.+)", text)
    if hits:
        return hits[-1].strip().rstrip(".")
    return None


def norm_short(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------- handlers


def h_factual(task, ctx):
    sysmsg = ("Answer the question accurately and directly in 1-3 short sentences, "
              "covering every part of it. No preamble.")
    a1 = ctx.chat(sysmsg, task, temperature=0.0, max_tokens=150)
    if not a1:
        return {"answer": "", "conf": 0.2, "cat": "factual"}
    if not ctx.have_time(20):
        return {"answer": a1, "conf": 0.6, "cat": "factual"}
    a2 = ctx.chat("Answer precisely and completely in 1-3 short sentences. "
                  "Double-check facts before answering.",
                  task, temperature=0.55, max_tokens=150, seed=7)
    if not a2:
        return {"answer": a1, "conf": 0.6, "cat": "factual"}
    same = _facts_agree(ctx, task, a1, a2)
    if same:
        return {"answer": a1, "conf": 0.9, "cat": "factual"}
    if not ctx.have_time(25):
        return {"answer": a1, "conf": 0.45, "cat": "factual"}
    a3 = ctx.chat("You are a careful fact-checker. Answer the question correctly "
                  "and completely in 1-3 short sentences.",
                  task, temperature=0.25, max_tokens=150, seed=99)
    if a3 and _facts_agree(ctx, task, a1, a3):
        return {"answer": a1, "conf": 0.85, "cat": "factual"}
    if a3 and _facts_agree(ctx, task, a2, a3):
        return {"answer": a3, "conf": 0.8, "cat": "factual"}
    return {"answer": a1, "conf": 0.4, "cat": "factual"}


def _facts_agree(ctx, q, a, b) -> bool:
    v = ctx.chat("You compare two candidate answers to the same question. "
                 "Reply with exactly YES if they agree on all key facts, otherwise NO.",
                 f"Question: {q}\n\nAnswer A: {a}\n\nAnswer B: {b}",
                 temperature=0.0, max_tokens=4)
    return v.strip().upper().startswith("Y")


_PROG_SYS = ("You convert word problems into Python. Write a minimal Python 3 "
             "program that computes the requested result and prints ONLY the final "
             "value with print(). Integer results must print as integers. Use exact "
             "arithmetic (integers or the fractions module) where possible. "
             "Output only the code. No markdown, no comments, no explanations.")


def h_math(task, ctx):
    c1 = extract_code(ctx.chat(_PROG_SYS, task, temperature=0.0, max_tokens=220))
    ok1, out1, err1 = run_python(c1) if c1 else (False, "", "empty")
    v1 = last_number(out1) if ok1 else None

    v2 = None
    if ctx.have_time(25):
        c2 = extract_code(ctx.chat(
            "Solve the problem by writing a tiny Python 3 program. It must print "
            "only the final numeric answer. Output only code, nothing else.",
            task, temperature=0.5, max_tokens=220, seed=11))
        ok2, out2, _ = run_python(c2) if c2 else (False, "", "")
        v2 = last_number(out2) if ok2 else None

    if v1 is not None and v2 is not None and nums_equal(v1, v2):
        return {"answer": _phrase_math(ctx, task, v1), "conf": 0.95, "cat": "math"}

    # tie-break / fallback: brief chain-of-thought
    v3 = None
    if ctx.have_time(35):
        cot = ctx.chat("Solve the problem with brief step-by-step reasoning "
                       "(at most 6 short lines). End with one final line exactly: "
                       "ANSWER: <number>",
                       task, temperature=0.0, max_tokens=300)
        v3 = normalize_num(answer_line(cot) or "") or last_number(cot)

    for x, y in ((v1, v3), (v2, v3), (v1, v2)):
        if x is not None and y is not None and nums_equal(x, y):
            return {"answer": _phrase_math(ctx, task, x), "conf": 0.85, "cat": "math"}

    best = next((v for v in (v1, v3, v2) if v is not None), None)
    if best is None:
        direct = ctx.chat("Answer the question. Give the final number and one short "
                          "sentence.", task, temperature=0.0, max_tokens=200)
        return {"answer": direct or "", "conf": 0.3, "cat": "math"}
    return {"answer": _phrase_math(ctx, task, best), "conf": 0.5, "cat": "math"}


def _phrase_math(ctx, task, value) -> str:
    v = fmt_num(value)
    if ctx.have_time(12):
        s = ctx.chat("Given a question and its correct computed result, reply with "
                     "one short sentence that answers the question using that exact "
                     "number. No working, no extra text.",
                     f"Question: {task}\nResult: {v}",
                     temperature=0.0, max_tokens=60)
        if s and re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", s.replace(",", "")):
            return s
    return f"The answer is {v}."


def _parse_aspects(text):
    """Parse 'POSITIVES: ...' / 'NEGATIVES: ...' lines -> (pos, neg) lists,
    or (None, None) when the format is absent."""
    if not text:
        return None, None
    pos = neg = None
    for ln in text.splitlines():
        m = re.match(r"\s*positives?\s*[:\-]\s*(.*)", ln, re.I)
        if m:
            pos = _aspect_list(m.group(1))
            continue
        m = re.match(r"\s*negatives?\s*[:\-]\s*(.*)", ln, re.I)
        if m:
            neg = _aspect_list(m.group(1))
    if pos is None or neg is None:
        return None, None
    return pos, neg


def _aspect_list(s):
    s = (s or "").strip().strip(".")
    if not s or s.lower() in ("none", "n/a", "-", "none.", "nothing"):
        return []
    return [p.strip() for p in s.split(";") if p.strip()]


def h_sentiment(task, ctx):
    ext = ctx.chat(
        "From the review/text in the task, extract the sentiment-bearing aspects. "
        "Reply in exactly this format (two lines):\n"
        "POSITIVES: <positive aspects separated by ';' or the word none>\n"
        "NEGATIVES: <negative aspects separated by ';' or the word none>",
        task, temperature=0.0, max_tokens=120)
    pos, neg = _parse_aspects(ext)
    if pos is None:
        a = ctx.chat("Classify the sentiment of the text given in the task as "
                     "Positive, Negative, Neutral, or Mixed (Mixed = both clearly "
                     "positive and clearly negative aspects present). Reply in the "
                     "format: <Label> - <one-sentence justification>.",
                     task, temperature=0.0, max_tokens=100)
        conf = 0.6 if re.match(r"\s*(positive|negative|neutral|mixed)\b", a or "", re.I) else 0.35
        return {"answer": a or "", "conf": conf, "cat": "sentiment"}
    # label is computed, not guessed — no label/justification contradictions
    if pos and neg:
        label = "Mixed"
        just = f"the text praises {'; '.join(pos)}, but criticizes {'; '.join(neg)}"
    elif pos:
        label = "Positive"
        just = f"the text expresses satisfaction: {'; '.join(pos)}"
    elif neg:
        label = "Negative"
        just = f"the text expresses dissatisfaction: {'; '.join(neg)}"
    else:
        label = "Neutral"
        just = "the text states information without a clear positive or negative stance"
    answer = f"{label} - {just}."
    return {"answer": answer, "conf": 0.9, "cat": "sentiment"}


def _target_words(task):
    m = re.search(r"(?:at most|maximum of|no more than|in|to|within)\s+(\d+)\s+words", task, re.I)
    return int(m.group(1)) if m else None


def _wants_one_sentence(task):
    return bool(re.search(r"(exactly\s+)?one\s+sentence|single\s+sentence|1\s+sentence", task, re.I))


def _target_sentences(task):
    m = re.search(r"(?:exactly\s+|in\s+|at most\s+)?(\d+)\s+sentences", task, re.I)
    return int(m.group(1)) if m else None


def h_summarize(task, ctx):
    a = ctx.chat("You are a precise summarizer. Follow the length/format constraint "
                 "stated in the task EXACTLY. Output only the summary, nothing else.",
                 task, temperature=0.0, max_tokens=170)
    if not a:
        return {"answer": "", "conf": 0.2, "cat": "summarize"}
    conf = 0.85
    if _wants_one_sentence(task):
        ss = sentences(a)
        if len(ss) != 1 and ctx.have_time(15):
            a2 = ctx.chat("Rewrite the text as exactly ONE sentence, preserving all "
                          "key information. Output only that sentence.",
                          a, temperature=0.0, max_tokens=120)
            if a2 and len(sentences(a2)) == 1:
                a = a2
            else:
                a = " ".join(s.rstrip(".!?") + "," for s in ss[:-1]) + " and " + ss[-1]
                a = a[0].upper() + a[1:]
        elif len(ss) != 1:
            a = " ".join(ss)
    n_s = _target_sentences(task)
    if n_s and len(sentences(a)) > n_s and ctx.have_time(15):
        a2 = ctx.chat(f"Rewrite the text in exactly {n_s} sentences, preserving key "
                      "information. Output only the rewritten text.",
                      a, temperature=0.0, max_tokens=170)
        if a2:
            a = a2
    n_w = _target_words(task)
    if n_w:
        words = a.split()
        if len(words) > n_w and ctx.have_time(15):
            a2 = ctx.chat(f"Shorten to at most {n_w} words, keep it one grammatical "
                          "sentence if possible. Output only the shortened text.",
                          a, temperature=0.0, max_tokens=n_w * 3)
            if a2 and len(a2.split()) <= n_w:
                a = a2
            else:
                a = " ".join(words[:n_w]).rstrip(",;:") + "."
        elif len(words) > n_w:
            a = " ".join(words[:n_w]).rstrip(",;:") + "."
    return {"answer": a.strip(), "conf": conf, "cat": "summarize"}


def h_ner(task, ctx):
    sysmsg = ("Extract ALL named entities from the text given in the task and label "
              "each with its type: Person, Organization, Location, Date, Time, Event, "
              "Product, Money, Percent, or Other. Reply with one entity per line in "
              "the format: Entity - Type. No other text.")
    a = ctx.chat(sysmsg, task, temperature=0.0, max_tokens=140)
    lines = _ner_lines(a)
    if not lines and ctx.have_time(15):
        a = ctx.chat(sysmsg, task, temperature=0.4, max_tokens=140, seed=13)
        lines = _ner_lines(a)
    if not lines:
        return {"answer": a or "", "conf": 0.3, "cat": "ner"}
    seen, out = set(), []
    for ln in lines:
        k = norm_short(ln)
        if k and k not in seen:
            seen.add(k)
            out.append(ln)
    return {"answer": "\n".join(out), "conf": 0.85, "cat": "ner"}


def _ner_lines(text):
    out = []
    for ln in (text or "").splitlines():
        ln = ln.strip().strip("-*• ").strip()
        if not ln:
            continue
        if re.search(r".+\s[-—–:]\s*.+", ln):
            out.append(re.sub(r"\s[-—–:]\s*", " - ", ln, count=1))
    return out


_FUZZ_POOLS = [
    (re.compile(r"sentence|text|string|phrase|paragraph|message", re.I),
     ["Hello world hello", "Cat cat CAT dog", "a B a b A"]),
    (re.compile(r"word|target|term|key$|substr|pattern|char", re.I),
     ["hello", "cat", "a"]),
    (re.compile(r"nums|numbers|lst|list|arr|array|values|items|data|elements|seq", re.I),
     [[3, 1, 4, 1, 5, 9, 2, 6], [2, 2, 2], [-5, 0, 5, 10], [7]]),
    (re.compile(r"words|strings|names|tokens", re.I),
     [["apple", "banana", "apple"], ["x"], []]),
    (re.compile(r"^s$|^st$", re.I), ["Level madam", "abc CBA", "Noon"]),
    (re.compile(r"^n$|^num$|count|limit|size|^k$|^m$|^x$|^a$|^b$|number", re.I),
     [0, 1, 2, 7, 10, -3]),
    (re.compile(r"dict|mapping|^d$|^map$", re.I), [{"a": 1, "b": 2}, {}]),
]


def _fn_signature(code):
    m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", code or "")
    if not m:
        return None, None
    params = []
    for p in m.group(2).split(","):
        p = p.strip()
        if not p or p.startswith("*") or "=" in p:
            continue
        params.append(p.split(":")[0].strip())
    return m.group(1), params


def _fuzz_argsets(params, n=4):
    pools = []
    for name in params:
        pool = None
        for rx, vals in _FUZZ_POOLS:
            if rx.search(name):
                pool = vals
                break
        if pool is None:
            return None
        pools.append(pool)
    if not pools:
        return None
    sets_ = []
    for i in range(n):
        sets_.append([pool[i % len(pool)] for pool in pools])
    return sets_


def _differential_compare(code_a, code_b, fname, argsets):
    """Execute two implementations on the same inputs in a sandbox.
    Returns (comparable, mismatches, first_mismatch_desc)."""
    import json as _json
    script = (
        "import json\n"
        f"ns1, ns2 = {{}}, {{}}\n"
        f"exec({code_a!r}, ns1)\n"
        f"exec({code_b!r}, ns2)\n"
        f"argsets = json.loads({_json.dumps(argsets)!r})\n"
        "rows = []\n"
        "for a in argsets:\n"
        "    try:\n"
        f"        r1 = repr(ns1[{fname!r}](*[__import__('copy').deepcopy(x) for x in a]))\n"
        "    except Exception as e:\n"
        "        r1 = 'ERR:' + type(e).__name__\n"
        "    try:\n"
        f"        r2 = repr(ns2[{fname!r}](*[__import__('copy').deepcopy(x) for x in a]))\n"
        "    except Exception as e:\n"
        "        r2 = 'ERR:' + type(e).__name__\n"
        "    rows.append([repr(a), r1, r2])\n"
        "print(json.dumps(rows))\n"
    )
    ok, out, err = run_python(script, timeout=10)
    if not ok or not out:
        return 0, 0, ""
    try:
        rows = _json.loads(out.splitlines()[-1])
    except Exception:
        return 0, 0, ""
    comparable = mism = 0
    desc = ""
    for args_r, r1, r2 in rows:
        if r1.startswith("ERR:") or r2.startswith("ERR:"):
            continue
        comparable += 1
        if r1 != r2:
            mism += 1
            if not desc:
                desc = f"input {args_r}: candidate returns {r1}, reference returns {r2}"
    return comparable, mism, desc


def _reference_impl(ctx, task, fname, seed=77):
    r = ctx.chat("Write a correct, self-contained Python function that does "
                 f"exactly what the task requires. Name it '{fname}'. Handle the "
                 "specific requirements stated in the task precisely. "
                 "Output only the code, no explanations.",
                 task, temperature=0.4, max_tokens=380, seed=seed)
    ref = extract_code(r)
    if ref and compiles(ref)[0] and re.search(rf"def\s+{re.escape(fname)}\s*\(", ref):
        return ref
    return None


def _differential_verify(ctx, task, code, sysmsg, max_tok):
    """Cross-check candidate code against an independently written reference
    implementation on fuzz inputs. Returns (code, conf) or None if inconclusive."""
    fname, params = _fn_signature(code)
    if not fname or not params:
        return None
    argsets = _fuzz_argsets(params)
    if not argsets or not ctx.have_time(25):
        return None
    ref = _reference_impl(ctx, task, fname)
    if not ref:
        return None
    comparable, mism, desc = _differential_compare(code, ref, fname, argsets)
    log(f"differential {fname}: comparable={comparable} mismatches={mism} {desc[:120]}")
    if comparable >= 2 and mism == 0:
        return (code, 0.93)
    if mism == 0:
        return None  # not enough signal
    if not ctx.have_time(35):
        return None
    reply2 = ctx.chat(sysmsg,
                      f"{task}\n\nNote: two candidate implementations disagree — "
                      f"{desc}. Re-read the task requirements carefully and provide "
                      "the correct code.",
                      temperature=0.2, max_tokens=max_tok, seed=43)
    code2 = extract_code(reply2)
    if code2 and compiles(code2)[0]:
        c2, m2, _ = _differential_compare(code2, ref, fname, argsets)
        if c2 >= 2 and m2 == 0:
            return (code2, 0.88)
    # the fresh reference tends to beat an anchored fix for small models
    return (ref, 0.6)


def h_code_debug(task, ctx):
    sysmsg = ("You are an expert Python debugger. The task contains buggy code. "
              "Reply with one sentence naming the bug, then the FULL corrected code "
              "in a ```python fence. Keep the original function name and signature.")
    reply = ctx.chat(sysmsg, task, temperature=0.0, max_tokens=380)
    code = extract_code(reply)
    ok, err = compiles(code) if code else (False, "no code")
    smoke_err = None
    if ok:
        smoke_err = _smoke_call(code)
    if ok and smoke_err is None:
        verified = _differential_verify(ctx, task, code, sysmsg, 380)
        if verified:
            c, conf = verified
            ans = (_debug_answer(reply, c) if c == code
                   else _debug_answer_recut(ctx, task, reply, c))
            return {"answer": ans, "conf": conf, "cat": "code_debug"}
        return {"answer": _debug_answer(reply, code), "conf": 0.75, "cat": "code_debug"}
    if ctx.have_time(35):
        reply2 = ctx.chat(sysmsg,
                          f"{task}\n\nNote: a previous fix attempt failed a check "
                          f"({err or smoke_err}). Provide the corrected code again, "
                          "carefully.",
                          temperature=0.35, max_tokens=380, seed=21)
        code2 = extract_code(reply2)
        ok2, err2 = compiles(code2) if code2 else (False, "no code")
        if ok2 and _smoke_call(code2) is None:
            verified = _differential_verify(ctx, task, code2, sysmsg, 380)
            if verified:
                c, conf = verified
                ans = (_debug_answer(reply2, c) if c == code2
                       else _debug_answer_recut(ctx, task, reply2, c))
                return {"answer": ans, "conf": conf, "cat": "code_debug"}
            return {"answer": _debug_answer(reply2, code2), "conf": 0.7, "cat": "code_debug"}
        if ok2:
            return {"answer": _debug_answer(reply2, code2), "conf": 0.55, "cat": "code_debug"}
    if ok:
        return {"answer": _debug_answer(reply, code), "conf": 0.55, "cat": "code_debug"}
    return {"answer": reply or "", "conf": 0.3, "cat": "code_debug"}


def _debug_answer(reply, code):
    first = ""
    for ln in (reply or "").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("```"):
            first = ln
            break
    return (first + "\n\n" if first else "") + f"```python\n{code}\n```"


def _debug_answer_recut(ctx, task, reply, final_code):
    """Final code was replaced during verification — restate the bug for the
    code we actually ship."""
    sent = ctx.chat("In one short sentence, state the bug in the original code "
                    "shown in the task (what it fails to do).",
                    task, temperature=0.0, max_tokens=60)
    if not sent:
        sent = "The original code does not implement the stated requirement."
    return sent.strip() + "\n\n" + f"```python\n{final_code}\n```"


_ARG_GUESS = [
    (re.compile(r"nums|numbers|lst|list|arr|values|items|data", re.I), "[3, 1, 4, 1, 5, 9, 2, 6]"),
    (re.compile(r"words|strings|names", re.I), "['apple', 'banana', 'cherry']"),
    (re.compile(r"s\b|text|string|word|sentence|phrase", re.I), "'level madam hello'"),
    (re.compile(r"n\b|num|count|limit|size|k\b|x\b|a\b|b\b", re.I), "5"),
    (re.compile(r"d\b|dict|mapping|map\b", re.I), "{'a': 1, 'b': 2}"),
]


def _smoke_call(code: str):
    """Best-effort: call the first defined function with guessed args.
    Returns None if it runs without raising, else the error text."""
    m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", code)
    if not m:
        ok, _, err = run_python(code)
        return None if ok else err[-300:]
    fname, params = m.group(1), m.group(2)
    args = []
    for p in params.split(","):
        p = p.strip()
        if not p or p.startswith("*") or "=" in p:
            continue
        name = p.split(":")[0].strip()
        for rx, val in _ARG_GUESS:
            if rx.search(name):
                args.append(val)
                break
        else:
            args.append("3")
    harness = code + f"\n\nresult = {fname}({', '.join(args)})\nprint('SMOKE_OK', repr(result))\n"
    ok, out, err = run_python(harness)
    if ok and "SMOKE_OK" in out:
        return None
    return (err or "no output")[-300:]


def h_code_gen(task, ctx):
    sysmsg = ("You are an expert Python developer. Write correct, clean Python for "
              "the request, handling edge cases (empty input, duplicates, invalid "
              "values) sensibly. Reply with ONLY one ```python fence containing the "
              "code followed by exactly 2 assert statements that test it.")
    reply = ctx.chat(sysmsg, task, temperature=0.0, max_tokens=420)
    code = extract_code(reply)
    ok, err = compiles(code) if code else (False, "no code")
    ran = False
    if ok:
        ran, out, rerr = run_python(code)
        err = rerr[-300:] if not ran else ""
    if ok and ran:
        verified = _differential_verify(ctx, task, code, sysmsg, 420)
        if verified:
            c, conf = verified
            return {"answer": _gen_answer(c), "conf": conf, "cat": "code_gen"}
        return {"answer": _gen_answer(code), "conf": 0.8, "cat": "code_gen"}
    if ctx.have_time(35):
        reply2 = ctx.chat(sysmsg,
                          f"{task}\n\nNote: a previous attempt failed with: {err}. "
                          "Write the code again carefully.",
                          temperature=0.3, max_tokens=420, seed=31)
        code2 = extract_code(reply2)
        ok2, _ = compiles(code2) if code2 else (False, "")
        if ok2:
            ran2, _, _ = run_python(code2)
            if ran2:
                return {"answer": _gen_answer(code2), "conf": 0.85, "cat": "code_gen"}
            return {"answer": _gen_answer(code2), "conf": 0.55, "cat": "code_gen"}
    if ok:
        return {"answer": _gen_answer(code), "conf": 0.5, "cat": "code_gen"}
    return {"answer": reply or "", "conf": 0.3, "cat": "code_gen"}


def _gen_answer(code):
    lines = [ln for ln in code.splitlines() if not ln.strip().startswith("assert")]
    while lines and (not lines[-1].strip() or lines[-1].strip().startswith("#")):
        lines.pop()
    body = "\n".join(lines).strip()
    return f"```python\n{body}\n```"


def h_logic(task, ctx):
    sysmsg = ("Solve the logic puzzle with brief careful reasoning (at most 8 short "
              "lines), checking every stated condition. End with one final line "
              "exactly: ANSWER: <the answer>")
    r1 = ctx.chat(sysmsg, task, temperature=0.0, max_tokens=280)
    a1 = answer_line(r1)
    a2 = None
    if ctx.have_time(30):
        r2 = ctx.chat(sysmsg, task, temperature=0.7, max_tokens=280, seed=17)
        a2 = answer_line(r2)
    if a1 and a2 and norm_short(a1) == norm_short(a2):
        return {"answer": _logic_answer(ctx, task, a1), "conf": 0.9, "cat": "logic"}
    a3 = None
    if ctx.have_time(30):
        r3 = ctx.chat("Carefully solve this constraint puzzle. Enumerate the "
                      "possibilities briefly and eliminate those violating any "
                      "condition. End with one final line exactly: ANSWER: <the answer>",
                      task, temperature=0.3, max_tokens=300, seed=29)
        a3 = answer_line(r3)
    for x, y in ((a1, a3), (a2, a3)):
        if x and y and norm_short(x) == norm_short(y):
            return {"answer": _logic_answer(ctx, task, x), "conf": 0.85, "cat": "logic"}
    pick = a1 or a3 or a2
    if pick:
        return {"answer": _logic_answer(ctx, task, pick), "conf": 0.45, "cat": "logic"}
    return {"answer": (r1 or "").strip(), "conf": 0.3, "cat": "logic"}


def _logic_answer(ctx, task, ans):
    if ctx.have_time(12):
        s = ctx.chat("Given a puzzle and its correct answer, reply with the answer "
                     "stated as one sentence plus one brief sentence of justification. "
                     "No other text.",
                     f"Puzzle: {task}\nCorrect answer: {ans}",
                     temperature=0.0, max_tokens=80)
        if s and norm_short(ans) in norm_short(s):
            return s
    return f"{ans}."


HANDLERS = {
    "factual": h_factual,
    "math": h_math,
    "sentiment": h_sentiment,
    "summarize": h_summarize,
    "ner": h_ner,
    "code_debug": h_code_debug,
    "code_gen": h_code_gen,
    "logic": h_logic,
}
