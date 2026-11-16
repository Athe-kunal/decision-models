# Training images for decision-models, one Dockerfile with three targets:
#
#   encoder  UltraBERT / Ettin encoders (make train-encoder, DDP)
#   decoder  Qwen3 decoders with LLM-as-Jev (make train-decoder, FSDP2)
#   qwen35   decoder + the qwen35 extra (Gated DeltaNet kernels, causal-conv1d)
#
# encoder and decoder share one environment; the targets only name the image.
# Any NVIDIA GPU torch 2.9 (CUDA 12.8) supports, Turing to Blackwell, on x86_64
# or aarch64 (Grace Hopper / Blackwell: add --platform linux/arm64). FA2 varlen
# needs Ampere or newer; on older GPUs train with model.attention=sdpa (docs/docker.md).
#
#   docker build -f docker/Dockerfile --target encoder -t decision-models:encoder .
#   docker run --rm -it --gpus all --ipc=host --shm-size=32g \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -v $PWD/runs:/root/decision-models/runs \
#     decision-models:encoder
#
# See docs/docker.md.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

ARG UV_VERSION=0.9.4
ARG PYTHON_VERSION=3.11

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git git-lfs make \
        nvtop rsync tmux \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh \
        | env UV_INSTALL_DIR=/usr/local/bin sh

# uv-managed Python inside the image; the venv lives in the project
# (.venv), where the Makefile expects it.
ENV UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    CUDA_HOME=/usr/local/cuda

WORKDIR /root/decision-models

# Dependencies first, so code edits do not reinstall torch.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra dev

COPY Makefile ./
COPY configs ./configs
COPY scripts ./scripts
COPY analysis ./analysis
COPY src ./src
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev

ENV PATH=/root/decision-models/.venv/bin:${PATH}
CMD ["/bin/bash"]


FROM base AS encoder
LABEL org.opencontainers.image.title="decision-models encoder training"


FROM base AS decoder
LABEL org.opencontainers.image.title="decision-models decoder training"


# causal-conv1d builds against the installed torch (no build isolation).
# Its setup first tries a prebuilt wheel and compiles with nvcc otherwise.
# TORCH_CUDA_ARCH_LIST covers every generation CUDA 12.8 targets: 7.5 Turing
# (T4), 8.0/8.6 Ampere (A100, A10, 3090), 8.9 Ada (L4, L40S, 4090), 9.0 Hopper
# (H100, H200), 10.0/12.0 Blackwell (B200, RTX 50). Narrow it to build faster.
FROM decoder AS qwen35
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0"
ARG MAX_JOBS=8
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev --extra qwen35
LABEL org.opencontainers.image.title="decision-models Qwen3.5 decoder training"
