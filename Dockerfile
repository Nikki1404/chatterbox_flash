FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"
ENV HTTP_PROXY="http://163.116.128.80:8080"
ENV HTTPS_PROXY="http://163.116.128.80:8080"

ENV no_proxy="localhost,127.0.0.1"
ENV NO_PROXY="localhost,127.0.0.1"


ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    MODEL_ID=ResembleAI/chatterbox-flash \
    DEVICE=cuda \
    CHATTERBOX_FLASH_ENGINE=flashinfer \
    BLOCK_SIZE=16 \
    NUM_STEPS=10 \
    TEMPERATURE=0.2 \
    TIME_SHIFT_TAU=0.5 \
    CFG_SCALE=1.0 \
    POSITION_TEMPERATURE=5.0 \
    N_CFM_TIMESTEPS=2 \
    REFERENCE_AUDIO=/app/reference.wav \
    CUDA_MODULE_LOADING=LAZY \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    git \
    curl \
    ca-certificates \
    ffmpeg \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*


RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

RUN uv venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN git clone --depth 1 \
    https://github.com/resemble-ai/chatterbox-flash.git \
    /opt/chatterbox-flash

WORKDIR /opt/chatterbox-flash


RUN uv pip install -e ".[flashinfer]"


RUN python - <<'PY'
import torch
import flashinfer

print("=" * 70)
print("PyTorch version      :", torch.__version__)
print("PyTorch CUDA version :", torch.version.cuda)
print("FlashInfer version   :", flashinfer.__version__)
print("=" * 70)
PY


RUN uv pip install \
    flashinfer-cubin \
    --index-url https://flashinfer.ai/whl


RUN uv pip install \
    flashinfer-jit-cache \
    --index-url https://flashinfer.ai/whl/cu130


RUN flashinfer show-config

WORKDIR /app

COPY requirements.txt .

RUN uv pip install -r requirements.txt


COPY patch_streaming.py /tmp/patch_streaming.py

RUN python /tmp/patch_streaming.py


# Fail build immediately if patch was not applied.
RUN grep -n "block_callback" \
    /opt/chatterbox-flash/chatterbox_flash/model.py


COPY server.py .
COPY client.py .
COPY openai_client.py .


RUN python - <<'PY'
import torch
import flashinfer

print("")
print("============================================================")
print(" FINAL CHATTERBOX-FLASH ENVIRONMENT")
print("============================================================")
print("Torch          :", torch.__version__)
print("Torch CUDA     :", torch.version.cuda)
print("FlashInfer     :", flashinfer.__version__)
print("CUDA available :", torch.cuda.is_available())
print("Backend        : flashinfer")
print("============================================================")
PY


# ============================================================
# SERVER
# ============================================================

EXPOSE 8000

# IMPORTANT:
# One worker = one model instance on the A10G.
# Do not increase workers for lowest single-request latency.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
