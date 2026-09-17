# Chatterbox-Flash CUDA 13.0 + FlashInfer

Latency-focused standalone Chatterbox-Flash server for NVIDIA A10G.

## APIs

- `GET /health`
- `WS /ws/tts` — block-level PCM16 streaming
- `POST /v1/audio/speech` — OpenAI-compatible TTS

## Stack

- CUDA 13.0.1 runtime
- Chatterbox-Flash
- FlashInfer backend
- FlashInfer CUDA 13.0 cubin/JIT cache
- BF16
- CUDA Graph
- T3 block size 16
- reference conditioning at startup
- one Uvicorn worker per GPU

## Add your reference voice

Place `reference.wav` in this project directory.

## Build

```bash
docker build --no-cache -t chatterbox-flash:cu130-flashinfer .
```

## Run

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p 8000:8000 \
  -v "$(pwd)/reference.wav:/app/reference.wav:ro" \
  chatterbox-flash:cu130-flashinfer
```

## Health

```bash
curl http://127.0.0.1:8000/health
```

Verify `"backend": "flashinfer"`.

## WebSocket

```bash
pip install websockets

python client.py \
  --url ws://127.0.0.1:8000/ws/tts \
  --text "Hello, thank you for calling. How may I help you today?" \
  --output ws_output.wav
```

## OpenAI-compatible endpoint

```bash
pip install openai
python openai_client.py
```

Endpoint:

`POST /v1/audio/speech`

Example model/voice:

- model: `chatterbox-flash`
- voice: `reference`
- response_format: `wav`

The preloaded `reference.wav` is used for the voice.

## Important

The WS path patches the upstream internal T3 committed-block loop so packets can
be emitted before the full utterance has completed. The current S3Gen stage
still re-decodes the accumulated committed token prefix and sends only its new
audio tail. That is the next optimization area if sustained streaming RTF is
still too high.
