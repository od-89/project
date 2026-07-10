"""llama.cpp server lifecycle + minimal OpenAI-compatible chat client.

Zero external dependencies (urllib only) so the runtime image stays lean.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


class LocalLLM:
    def __init__(self, bin_path=None, model_path=None, port=8091, threads=None,
                 ctx=4096, base_url=None):
        self.bin_path = bin_path or os.environ.get("LLAMA_BIN", "/opt/llama/llama-server")
        self.model_path = model_path or os.environ.get("MODEL_PATH", "/models/model.gguf")
        self.port = int(os.environ.get("LLAMA_PORT", port))
        self.threads = int(threads or os.environ.get("LLAMA_THREADS", "2"))
        self.ctx = int(os.environ.get("LLAMA_CTX", ctx))
        # If LLAMA_URL is set we attach to an already-running server (local dev).
        self.base_url = base_url or os.environ.get("LLAMA_URL") or f"http://127.0.0.1:{self.port}"
        self.proc = None
        self.external = bool(os.environ.get("LLAMA_URL"))

    def start(self):
        if self.external:
            return
        args = [
            self.bin_path,
            "-m", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.ctx),
            "-t", str(self.threads),
            "-tb", str(self.threads),
            "--jinja",
            "-np", "1",
            "--no-webui",
            "--cache-reuse", "256",
        ]
        log = sys.stderr
        self.proc = subprocess.Popen(args, stdout=log, stderr=log)

    def wait_ready(self, timeout=55.0) -> bool:
        deadline = time.time() + timeout
        url = f"{self.base_url}/health"
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(0.4)
        return False

    def chat(self, system: str, user: str, temperature: float = 0.0,
             max_tokens: int = 160, timeout: float = 120.0, seed: int = 42) -> str:
        body = {
            "model": "local",
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        data = json.dumps(body).encode("utf-8")
        url = f"{self.base_url}/v1/chat/completions"
        for attempt in (1, 2):
            try:
                req = urllib.request.Request(
                    url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    resp = json.loads(r.read().decode("utf-8", "replace"))
                return (resp["choices"][0]["message"]["content"] or "").strip()
            except Exception as e:
                sys.stderr.write(f"[llm] chat attempt {attempt} failed: {e}\n")
                time.sleep(0.5)
        return ""

    def stop(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
