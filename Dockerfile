# syntax=docker/dockerfile:1
# Track 1 agent: local llama.cpp (Qwen2.5-3B-Instruct Q4_K_M) + Python orchestrator.
# Final image ~2.3 GB compressed — far under the 10 GB limit.

ARG LLAMA_TAG=b9950

# ---- stage 1: llama.cpp server binaries -------------------------------------
FROM debian:bookworm-slim AS llama
ARG LLAMA_TAG
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fL -o /tmp/llama.tgz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" \
    && mkdir -p /tmp/l /opt/llama \
    && tar -xzf /tmp/llama.tgz -C /tmp/l \
    && SERVER="$(find /tmp/l -name llama-server -type f | head -1)" \
    && cp -a "$(dirname "$SERVER")/." /opt/llama/ \
    && /opt/llama/llama-server --version || true

# ---- stage 2: model weights --------------------------------------------------
FROM debian:bookworm-slim AS model
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /models && curl -fL -o /models/model.gguf \
      "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf"

# ---- stage 3: runtime --------------------------------------------------------
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libcurl4 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=llama /opt/llama /opt/llama
COPY --from=model /models /models
COPY agent /app/agent

ARG MODE=zero
ARG ESC_MAX=6
ARG ESC_CONF=0.55
ENV LLAMA_BIN=/opt/llama/llama-server \
    MODEL_PATH=/models/model.gguf \
    LLAMA_THREADS=2 \
    MODE=${MODE} \
    ESC_MAX=${ESC_MAX} \
    ESC_CONF=${ESC_CONF} \
    PYTHONUNBUFFERED=1

WORKDIR /app
ENTRYPOINT ["python", "-m", "agent.main"]
