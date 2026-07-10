"""Run model-written Python in an isolated subprocess with hard limits."""
import os
import subprocess
import sys


def run_python(code: str, timeout: float = 8.0):
    """Execute code with `python -I -c`. Returns (ok, stdout, stderr)."""
    cmd = [sys.executable, "-I", "-c", code]
    kwargs = {}
    if os.name == "posix":
        import resource

        def _limits():
            resource.setrlimit(resource.RLIMIT_AS, (1_500_000_000, 1_500_000_000))
            cpu = int(timeout) + 2
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))

        kwargs["preexec_fn"] = _limits
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={"PYTHONIOENCODING": "utf-8"}, **kwargs,
        )
        return (r.returncode == 0, (r.stdout or "").strip(), (r.stderr or "").strip())
    except subprocess.TimeoutExpired:
        return (False, "", "timeout")
    except Exception as e:  # pragma: no cover
        return (False, "", f"{type(e).__name__}: {e}")
