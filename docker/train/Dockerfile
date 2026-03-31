FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv --break-system-packages

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

RUN uv venv --python python3.11

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install deps without dev tools, pulling torch from the CUDA index
RUN uv sync --no-dev --index-url https://download.pytorch.org/whl/cu132

VOLUME ["/app/data", "/app/artifacts", "/app/checkpoints"]

ENTRYPOINT ["python", "-m", "minimal_llm.train"]
