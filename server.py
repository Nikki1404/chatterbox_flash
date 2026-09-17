import asyncio
import io
import json
import logging
import os
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel
from chatterbox_flash import ChatterboxFlashTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("chatterbox-flash")

MODEL_ID = os.getenv("MODEL_ID", "ResembleAI/chatterbox-flash")
DEVICE = os.getenv("DEVICE", "cuda")
BACKEND = os.getenv("CHATTERBOX_FLASH_ENGINE", "flashinfer")
REFERENCE_AUDIO = os.getenv("REFERENCE_AUDIO", "/app/reference.wav")

BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "16"))
NUM_STEPS = int(os.getenv("NUM_STEPS", "10"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TIME_SHIFT_TAU = float(os.getenv("TIME_SHIFT_TAU", "0.5"))
CFG_SCALE = float(os.getenv("CFG_SCALE", "1.0"))
POSITION_TEMPERATURE = float(os.getenv("POSITION_TEMPERATURE", "5.0"))
N_CFM_TIMESTEPS = int(os.getenv("N_CFM_TIMESTEPS", "2"))

app = FastAPI(title="Chatterbox-Flash TTS", version="1.0.0")
tts = None
gpu_lock = threading.Lock()


class OpenAISpeechRequest(BaseModel):
    model: str = "chatterbox-flash"
    input: str
    voice: str = "reference"
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = 1.0


def ms():
    return time.perf_counter() * 1000.0


def to_pcm16(wav_tensor) -> bytes:
    if torch.is_tensor(wav_tensor):
        arr = wav_tensor.detach().float().cpu().numpy()
    else:
        arr = np.asarray(wav_tensor)
    arr = np.asarray(arr).squeeze()
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def generate_full(text: str):
    with gpu_lock, torch.inference_mode():
        return tts.generate(
            text,
            num_steps=NUM_STEPS,
            temperature=TEMPERATURE,
            time_shift_tau=TIME_SHIFT_TAU,
            cfg_scale=CFG_SCALE,
            position_temperature=POSITION_TEMPERATURE,
            pmi_uncond_prior_precompute=True,
            use_cuda_graph=True,
            backend=BACKEND,
            n_cfm_timesteps=N_CFM_TIMESTEPS,
        )


@app.on_event("startup")
async def startup():
    global tts

    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda but CUDA is unavailable.")

    if torch.cuda.is_available():
        log.info("GPU: %s", torch.cuda.get_device_name(0))
        log.info("Torch CUDA: %s", torch.version.cuda)

    started = ms()
    tts = ChatterboxFlashTTS.from_pretrained(
        MODEL_ID,
        device=DEVICE,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        drf_block_size=BLOCK_SIZE,
    )
    log.info("Model loaded in %.2f ms | sample_rate=%s", ms() - started, tts.sr)

    if not Path(REFERENCE_AUDIO).exists():
        log.warning("Reference audio missing: %s", REFERENCE_AUDIO)
        return

    started = ms()
    tts.prepare_conditionals(REFERENCE_AUDIO)
    log.info("Reference conditioning prepared in %.2f ms", ms() - started)

    try:
        log.info("Warming model...")
        _ = generate_full("Hello.")
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        log.info("Warmup complete.")
    except Exception:
        log.exception("Warmup failed; server remains available.")


@app.get("/")
async def root():
    return {
        "service": "chatterbox-flash",
        "websocket": "/ws/tts",
        "openai_compatible": "/v1/audio/speech",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "backend": BACKEND,
        "sample_rate": tts.sr if tts else None,
        "block_size": BLOCK_SIZE,
        "reference_ready": bool(tts is not None and tts.conds is not None),
    }


@app.post("/v1/audio/speech")
async def openai_audio_speech(req: OpenAISpeechRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="input must not be empty")
    if tts is None or tts.conds is None:
        raise HTTPException(status_code=503, detail="Reference voice is not prepared")

    # Accepted for OpenAI SDK compatibility. Chatterbox-Flash does not expose
    # OpenAI's named voices or a native speed parameter, so the preloaded
    # reference voice is used and speed must remain 1.0.
    if req.speed != 1.0:
        raise HTTPException(status_code=400, detail="Only speed=1.0 is supported")

    started = ms()
    try:
        wav_tensor = await asyncio.to_thread(generate_full, req.input.strip())
    except Exception as exc:
        log.exception("OpenAI-compatible generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    pcm = to_pcm16(wav_tensor)
    elapsed = ms() - started
    log.info("HTTP generation complete in %.2f ms", elapsed)

    headers = {
        "X-Model": "chatterbox-flash",
        "X-Generation-Ms": f"{elapsed:.2f}",
        "Cache-Control": "no-store",
    }

    if req.response_format == "pcm":
        return Response(
            content=pcm,
            media_type="application/octet-stream",
            headers=headers,
        )

    return Response(
        content=pcm_to_wav(pcm, tts.sr),
        media_type="audio/wav",
        headers=headers,
    )


def streaming_generate(text: str, out_q: queue.Queue):
    """
    True T3 block-time streaming.

    The Dockerfile adds `block_callback` to the upstream T3 block loop.
    Each committed speech-token block is immediately vocoded and the new
    PCM tail is pushed to the WebSocket queue while T3 continues generating.
    """
    started = ms()
    sent_samples = 0

    try:
        with gpu_lock, torch.inference_mode():
            text_tokens = tts._encode_text(text, normalize_text=True)
            n_tokens = int(text_tokens.size(1))

            # Mirrors upstream Chatterbox-Flash's speech length heuristic.
            # A generous ceiling avoids truncating normal utterances.
            total_speech_len = max(n_tokens * 6, 300)

            def on_block(committed_tokens, block_index=None, is_final=False):
                nonlocal sent_samples

                tokens = committed_tokens[0] if committed_tokens.ndim == 2 else committed_tokens
                stop_token = tts.t3.hp.stop_speech_token
                eos = (tokens == stop_token).nonzero(as_tuple=True)[0]
                if len(eos):
                    tokens = tokens[:eos[0].item()]
                if tokens.numel() == 0:
                    return

                wav_tensor, _ = tts.s3gen.inference(
                    speech_tokens=tokens.to(tts.device),
                    ref_dict=tts.conds.gen,
                    n_cfm_timesteps=N_CFM_TIMESTEPS,
                )
                wav_tensor = wav_tensor.detach().float().cpu().squeeze()

                total_samples = int(wav_tensor.numel())
                if total_samples <= sent_samples:
                    return

                new_pcm = to_pcm16(wav_tensor[sent_samples:])
                sent_samples = total_samples

                if new_pcm:
                    out_q.put({
                        "type": "audio",
                        "pcm": new_pcm,
                        "block": block_index,
                        "generation_ms": ms() - started,
                    })

            speech_tokens = tts.t3.generate(
                t3_cond=tts.conds.t3,
                text_tokens=text_tokens,
                text_token_lens=torch.tensor([n_tokens], device=tts.device),
                total_speech_len=total_speech_len,
                num_steps=NUM_STEPS,
                temperature=TEMPERATURE,
                time_shift_tau=TIME_SHIFT_TAU,
                omnivoice_schedule_t_shift=0.5,
                cfg_scale=CFG_SCALE,
                position_temperature=POSITION_TEMPERATURE,
                pmi_uncond_prior_precompute=True,
                use_cuda_graph=True,
                backend=BACKEND,
                batch_size=1,
                block_callback=on_block,
            )

            # Final pass ensures no tail after the final callback is missed.
            final_tokens = speech_tokens[0] if speech_tokens.ndim == 2 else speech_tokens
            stop_token = tts.t3.hp.stop_speech_token
            eos = (final_tokens == stop_token).nonzero(as_tuple=True)[0]
            if len(eos):
                final_tokens = final_tokens[:eos[0].item()]

            if final_tokens.numel():
                wav_tensor, _ = tts.s3gen.inference(
                    speech_tokens=final_tokens.to(tts.device),
                    ref_dict=tts.conds.gen,
                    n_cfm_timesteps=N_CFM_TIMESTEPS,
                )
                wav_tensor = wav_tensor.detach().float().cpu().squeeze()
                if wav_tensor.numel() > sent_samples:
                    tail = to_pcm16(wav_tensor[sent_samples:])
                    if tail:
                        out_q.put({
                            "type": "audio",
                            "pcm": tail,
                            "block": "final",
                            "generation_ms": ms() - started,
                        })

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            out_q.put({"type": "end", "total_ms": ms() - started})

    except Exception as exc:
        log.exception("WebSocket generation failed")
        out_q.put({"type": "error", "message": str(exc)})


@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected")

    try:
        while True:
            payload = json.loads(await ws.receive_text())
            text = payload.get("text", "").strip()

            if not text:
                await ws.send_json({"type": "error", "message": "text is required"})
                continue

            if tts is None or tts.conds is None:
                await ws.send_json({
                    "type": "error",
                    "message": "Reference voice is not prepared",
                })
                continue

            request_started = ms()

            await ws.send_json({
                "type": "start",
                "model": "chatterbox-flash",
                "sample_rate": tts.sr,
                "channels": 1,
                "sample_width": 2,
                "format": "pcm_s16le",
                "backend": BACKEND,
                "block_size": BLOCK_SIZE,
            })

            out_q = queue.Queue()
            threading.Thread(
                target=streaming_generate,
                args=(text, out_q),
                daemon=True,
            ).start()

            first_audio = True

            while True:
                item = await asyncio.to_thread(out_q.get)

                if item["type"] == "audio":
                    if first_audio:
                        await ws.send_json({
                            "type": "first_audio",
                            "ttfb_ms": ms() - request_started,
                            "model_generation_ms": item["generation_ms"],
                        })
                        first_audio = False
                    await ws.send_bytes(item["pcm"])

                elif item["type"] == "end":
                    await ws.send_json({
                        "type": "end",
                        "generation_ms": item["total_ms"],
                        "request_ms": ms() - request_started,
                    })
                    break

                elif item["type"] == "error":
                    await ws.send_json({"type": "error", "message": item["message"]})
                    break

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception:
        log.exception("WebSocket connection failed")
