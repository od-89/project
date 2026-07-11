"""Fireworks escalation client (hybrid mode only).

All calls go through FIREWORKS_BASE_URL exactly as the harness requires, use
only models from ALLOWED_MODELS (read at runtime, never hardcoded), and are
token-accounted. Zero mode never imports/calls this.
"""
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

# Local-dev only: Avast MITMs HTTPS with a cert OpenSSL rejects. The judge VM
# never sets this. Default: full verification.
_SSL_CTX = None
if os.environ.get("FW_INSECURE_SSL") == "1":
    _SSL_CTX = ssl.create_default_context()
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = ssl.CERT_NONE

_GENERAL_PREF = ("gemma", "llama", "qwen", "mini", "glm", "deepseek", "kimi")
_CODE_PREF = ("code", "coder", "kimi", "qwen", "deepseek", "glm", "gemma")

_spent = {"total": 0, "calls": 0}


def allowed_models():
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def pick_model(cat: str) -> str:
    models = allowed_models()
    if not models:
        return ""
    prefs = _CODE_PREF if cat in ("code_debug", "code_gen") else _GENERAL_PREF
    for key in prefs:
        for m in models:
            if key in m.lower():
                return m
    return models[0]


def _endpoint_candidates():
    base = (os.environ.get("FIREWORKS_BASE_URL", "") or "").rstrip("/")
    if not base:
        return []
    if base.endswith("/chat/completions"):
        return [base]
    cands = [f"{base}/chat/completions"]
    if not base.endswith("/v1"):
        cands.append(f"{base}/v1/chat/completions")
    return cands


def fw_answer(task_prompt: str, cat: str, max_tokens: int = 380):
    """Return (text, tokens) or ("", 0) on failure."""
    key = os.environ.get("FIREWORKS_API_KEY", "")
    model = pick_model(cat)
    if not key or not model:
        return "", 0
    msg = task_prompt + "\n\nAnswer directly and concisely."
    body = {
        "model": model,
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    for url in _endpoint_candidates():
        for extra in ({"reasoning_effort": "none"}, {}):
            payload = dict(body, **extra)
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {key}",
                    },
                )
                with urllib.request.urlopen(req, timeout=18, context=_SSL_CTX) as r:
                    resp = json.loads(r.read().decode("utf-8", "replace"))
                text = (resp["choices"][0]["message"]["content"] or "").strip()
                usage = resp.get("usage") or {}
                tok = int(usage.get("total_tokens")
                          or (usage.get("prompt_tokens", 0)
                              + usage.get("completion_tokens", 0)))
                _spent["total"] += tok
                _spent["calls"] += 1
                sys.stderr.write(
                    f"[fw] model={model} tokens={tok} running_total={_spent['total']}\n")
                return text, tok
            except urllib.error.HTTPError as e:
                sys.stderr.write(f"[fw] HTTP {e.code} at {url} extra={extra}\n")
                if e.code in (400, 404, 422):
                    continue
                return "", 0
            except Exception as e:
                sys.stderr.write(f"[fw] error: {e}\n")
                return "", 0
    return "", 0


def spent():
    return dict(_spent)
