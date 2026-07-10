# ZeroFire — Zero-Token Local-First Agent (AMD Hackathon ACT II, Track 1)

A general-purpose AI agent for the AMD Developer Hackathon ACT II **Track 1
(Hybrid Token-Efficient Routing Agent)**. It answers all eight task categories
**entirely inside the container** with a local Qwen2.5-3B-Instruct model
(llama.cpp) plus deterministic verification — spending **zero Fireworks
tokens** in its primary mode, the best possible token score, while clearing
the accuracy gate.

## How it works

```
/input/tasks.json
      │
      ▼
┌──────────────┐   regex, 0 cost
│  classifier   │──────────────► category (factual / math / sentiment /
└──────────────┘                summarize / NER / code-debug / code-gen / logic)
      │
      ▼
┌───────────────────────────────────────────────────────────┐
│ Pass 1 — bank a best-shot answer for every task           │
│   • math  : LLM writes a tiny Python program → executed   │
│             twice independently → results must agree      │
│   • code  : generated/fixed code is compiled AND run      │
│   • logic : two independent chains-of-thought must agree  │
│   • facts : two independent answers cross-checked         │
│   • text  : format constraints enforced programmatically  │
├───────────────────────────────────────────────────────────┤
│ Pass 2 — remaining time re-verifies low-confidence tasks  │
├───────────────────────────────────────────────────────────┤
│ Watchdog — guarantees valid, complete results.json and    │
│            exit 0 well before the 10-minute limit         │
└───────────────────────────────────────────────────────────┘
      │
      ▼
/output/results.json
```

* **Local model**: Qwen2.5-3B-Instruct Q4_K_M (1.9 GB) served by `llama-server`
  with 2 threads — sized for the 4 GB RAM / 2 vCPU judging environment.
* **Zero mode (default, tag `latest`)**: never calls Fireworks. Local models
  are explicitly a valid strategy; local tokens score 0.
* **Hybrid mode (tag `hybrid`)**: identical pipeline, but tasks whose answers
  could not be independently verified escalate — one terse call each — to the
  cheapest suitable model from `ALLOWED_MODELS` via `FIREWORKS_BASE_URL`
  (env-injected, never hardcoded), with reasoning disabled and `max_tokens`
  capped.

## Container contract (Track 1)

* Reads `/input/tasks.json`, writes `/output/results.json`
  (`[{"task_id": ..., "answer": ...}]`), exits 0.
* linux/amd64, starts in seconds, finishes far inside the 10-minute cap;
  a watchdog flushes results and exits 0 even in worst-case stalls.
* Env consumed at runtime: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`,
  `ALLOWED_MODELS` (hybrid mode only; zero mode ignores them).

## Run it

```bash
docker run --rm --cpus=2 --memory=4g \
  -v /path/to/input:/input:ro -v /path/to/output:/output \
  ghcr.io/od-89/zerofire-agent:latest
```

`/path/to/input/tasks.json`:

```json
[ { "task_id": "t1", "prompt": "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?" } ]
```

## Develop locally (no Docker)

```bash
# 1) download a llama.cpp release for your OS into tools/llama/
# 2) download the model:
#    https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF (Q4_K_M)
#    into models/Qwen2.5-3B-Instruct-Q4_K_M.gguf
python eval/run_local.py                 # runs the 29-variant local eval
python eval/run_local.py eval/practice_tasks.json
```

`eval/run_local.py` spawns the server itself, or reuses one you started when
`LLAMA_URL` is set. Auto-checks math/logic/fact variants against expected
values.

## CI

Every push to `main` builds the linux/amd64 image, pushes
`ghcr.io/od-89/zerofire-agent:{latest,hybrid,<sha>}`, then runs the practice
task set inside the freshly built image under judge-like limits
(`--cpus=2 --memory=4g`) and validates the output schema.

## Repository layout

```
agent/            the agent (stdlib-only Python)
  main.py         orchestrator: passes, pacing, watchdog, atomic flushes
  classify.py     zero-cost category router
  solvers.py      per-category handlers + verification
  local_llm.py    llama-server lifecycle + OpenAI-compatible client
  pyexec.py       sandboxed execution of model-written Python
  fireworks.py    hybrid-mode escalation (token-accounted)
Dockerfile        3-stage build: llama.cpp release + GGUF weights + slim runtime
eval/             practice tasks, 29 checkable variants, local harness
```
