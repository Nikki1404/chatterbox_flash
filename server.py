import asyncio
import io
import json
import logging
import os
import queue
import re
import threading
import time
import wave
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response
from pydantic import BaseModel

from chatterbox_flash import ChatterboxFlashTTS


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("chatterbox-flash")


# ============================================================
# Configuration
# ============================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "ResembleAI/chatterbox-flash",
)

DEVICE = os.getenv(
    "DEVICE",
    "cuda",
)

BACKEND = os.getenv(
    "CHATTERBOX_FLASH_ENGINE",
    "flashinfer",
)

REFERENCE_AUDIO = os.getenv(
    "REFERENCE_AUDIO",
    "/app/reference.wav",
)

BLOCK_SIZE = int(
    os.getenv("BLOCK_SIZE", "16")
)

NUM_STEPS = int(
    os.getenv("NUM_STEPS", "10")
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.2")
)

TIME_SHIFT_TAU = float(
    os.getenv("TIME_SHIFT_TAU", "0.5")
)

CFG_SCALE = float(
    os.getenv("CFG_SCALE", "1.0")
)

POSITION_TEMPERATURE = float(
    os.getenv("POSITION_TEMPERATURE", "5.0")
)

N_CFM_TIMESTEPS = int(
    os.getenv("N_CFM_TIMESTEPS", "2")
)

# Keep individual generation requests reasonably short.
# The client can still send an arbitrarily longer text file.
MAX_CHUNK_CHARS = int(
    os.getenv("MAX_CHUNK_CHARS", "180")
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Chatterbox-Flash TTS",
    version="1.2.0",
)

tts = None

# Only one inference operation should access this model/GPU
# at a time.
gpu_lock = threading.Lock()


# ============================================================
# OpenAI request
# ============================================================

class OpenAISpeechRequest(BaseModel):
    model: str = "chatterbox-flash"
    input: str
    voice: str = "reference"
    response_format: Literal["wav", "pcm"] = "wav"
    speed: float = 1.0


# ============================================================
# Utilities
# ============================================================

def ms():
    return time.perf_counter() * 1000.0


def to_pcm16(wav_tensor) -> bytes:

    if torch.is_tensor(wav_tensor):
        arr = (
            wav_tensor
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    else:
        arr = np.asarray(wav_tensor)

    arr = np.asarray(arr).squeeze()

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )

    arr = np.clip(
        arr,
        -1.0,
        1.0,
    )

    return (
        arr * 32767.0
    ).astype("<i2").tobytes()


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int,
) -> bytes:

    buf = io.BytesIO()

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)

    return buf.getvalue()


# ============================================================
# Long-text splitting
# ============================================================

def split_text_for_tts(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
):
    """
    Split long input into smaller TTS requests.

    Split priority:

        sentence
           ↓
        comma / semicolon / colon
           ↓
        whitespace

    The client can therefore send the complete text file.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    chunks = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # ----------------------------------------------------
        # Sentence fits by itself
        # ----------------------------------------------------

        if len(sentence) <= max_chars:

            candidate = (
                f"{current} {sentence}".strip()
                if current
                else sentence
            )

            if len(candidate) <= max_chars:
                current = candidate

            else:
                if current:
                    chunks.append(current)

                current = sentence

            continue

        # ----------------------------------------------------
        # Long sentence
        # ----------------------------------------------------

        if current:
            chunks.append(current)
            current = ""

        pieces = re.split(
            r"(?<=[,;:])\s+",
            sentence,
        )

        piece_buffer = ""

        for piece in pieces:

            piece = piece.strip()

            if not piece:
                continue

            candidate = (
                f"{piece_buffer} {piece}".strip()
                if piece_buffer
                else piece
            )

            if len(candidate) <= max_chars:
                piece_buffer = candidate
                continue

            if piece_buffer:
                chunks.append(piece_buffer)
                piece_buffer = ""

            # -----------------------------------------------
            # Still too long -> split on words
            # -----------------------------------------------

            if len(piece) > max_chars:

                words = piece.split()
                word_buffer = ""

                for word in words:

                    candidate = (
                        f"{word_buffer} {word}".strip()
                        if word_buffer
                        else word
                    )

                    if len(candidate) <= max_chars:
                        word_buffer = candidate

                    else:
                        if word_buffer:
                            chunks.append(word_buffer)

                        word_buffer = word

                if word_buffer:
                    piece_buffer = word_buffer

            else:
                piece_buffer = piece

        if piece_buffer:
            current = piece_buffer

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# Model generation
# ============================================================

def generate_full(text: str):
    """
    Use Chatterbox-Flash's high-level generation path.

    Important:
    We intentionally do NOT manually invoke tts.t3.generate()
    here. The library handles its own token preparation,
    generation length, FlashInfer execution and S3Gen.
    """

    with gpu_lock, torch.inference_mode():

        wav = tts.generate(
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

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        return wav


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
async def startup():

    global tts

    # --------------------------------------------------------
    # CUDA validation
    # --------------------------------------------------------

    if (
        DEVICE == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "DEVICE=cuda but CUDA is unavailable."
        )

    if torch.cuda.is_available():

        log.info(
            "GPU: %s",
            torch.cuda.get_device_name(0),
        )

        log.info(
            "Torch version: %s",
            torch.__version__,
        )

        log.info(
            "Torch CUDA: %s",
            torch.version.cuda,
        )

    # --------------------------------------------------------
    # Load Chatterbox
    # --------------------------------------------------------

    started = ms()

    tts = ChatterboxFlashTTS.from_pretrained(
        MODEL_ID,
        device=DEVICE,
        dtype=(
            torch.bfloat16
            if DEVICE == "cuda"
            else torch.float32
        ),
        drf_block_size=BLOCK_SIZE,
    )

    log.info(
        "Model loaded in %.2f ms | sample_rate=%s",
        ms() - started,
        tts.sr,
    )

    # --------------------------------------------------------
    # Reference voice
    # --------------------------------------------------------

    reference_path = Path(
        REFERENCE_AUDIO
    )

    if not reference_path.exists():

        raise RuntimeError(
            f"Reference audio missing: "
            f"{REFERENCE_AUDIO}"
        )

    if not reference_path.is_file():

        raise RuntimeError(
            f"Reference audio is not a file: "
            f"{REFERENCE_AUDIO}"
        )

    started = ms()

    tts.prepare_conditionals(
        REFERENCE_AUDIO
    )

    log.info(
        "Reference conditioning prepared "
        "in %.2f ms",
        ms() - started,
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    try:

        log.info(
            "Warming model..."
        )

        _ = generate_full(
            "Hello."
        )

        log.info(
            "Warmup complete."
        )

    except Exception:

        log.exception(
            "Warmup failed; server remains available."
        )


# ============================================================
# Root
# ============================================================

@app.get("/")
async def root():

    return {
        "service": "chatterbox-flash",
        "version": "1.2.0",
        "websocket": "/ws/tts",
        "openai_compatible": "/v1/audio/speech",
        "health": "/health",
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,

        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),

        "backend": BACKEND,

        "sample_rate": (
            tts.sr
            if tts
            else None
        ),

        "block_size": BLOCK_SIZE,

        "max_chunk_chars": (
            MAX_CHUNK_CHARS
        ),

        "reference_ready": bool(
            tts is not None
            and tts.conds is not None
        ),
    }


# ============================================================
# OpenAI-compatible endpoint
# ============================================================

@app.post("/v1/audio/speech")
async def openai_audio_speech(
    req: OpenAISpeechRequest,
):

    text = req.input.strip()

    if not text:

        raise HTTPException(
            status_code=400,
            detail="input must not be empty",
        )

    if (
        tts is None
        or tts.conds is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Reference voice is not prepared",
        )

    if req.speed != 1.0:

        raise HTTPException(
            status_code=400,
            detail="Only speed=1.0 is supported",
        )

    started = ms()

    try:

        # Long HTTP input is also split.
        text_chunks = split_text_for_tts(
            text
        )

        if not text_chunks:
            raise ValueError(
                "No text available for generation."
            )

        log.info(
            "HTTP TTS | chars=%d | chunks=%d",
            len(text),
            len(text_chunks),
        )

        pcm_parts = []

        for index, chunk_text in enumerate(
            text_chunks,
            start=1,
        ):

            log.info(
                "HTTP generating chunk %d/%d | "
                "chars=%d | text=%r",
                index,
                len(text_chunks),
                len(chunk_text),
                chunk_text,
            )

            wav_tensor = (
                await asyncio.to_thread(
                    generate_full,
                    chunk_text,
                )
            )

            pcm = to_pcm16(
                wav_tensor
            )

            if pcm:
                pcm_parts.append(
                    pcm
                )

            log.info(
                "HTTP chunk %d/%d complete | "
                "audio_bytes=%d",
                index,
                len(text_chunks),
                len(pcm),
            )

        if not pcm_parts:
            raise RuntimeError(
                "No audio generated."
            )

        pcm = b"".join(
            pcm_parts
        )

    except Exception as exc:

        log.exception(
            "OpenAI-compatible generation failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    elapsed = (
        ms() - started
    )

    log.info(
        "HTTP generation complete | "
        "chars=%d | chunks=%d | %.2f ms",
        len(text),
        len(text_chunks),
        elapsed,
    )

    headers = {
        "X-Model": "chatterbox-flash",
        "X-Generation-Ms": f"{elapsed:.2f}",
        "X-Text-Chunks": str(
            len(text_chunks)
        ),
        "Cache-Control": "no-store",
    }

    if req.response_format == "pcm":

        return Response(
            content=pcm,
            media_type="application/octet-stream",
            headers=headers,
        )

    return Response(
        content=pcm_to_wav(
            pcm,
            tts.sr,
        ),
        media_type="audio/wav",
        headers=headers,
    )


# ============================================================
# WebSocket generation worker
# ============================================================

def streaming_generate(
    text: str,
    out_q: queue.Queue,
):
    """
    Reliable long-text WebSocket generation.

    Full input:
          ↓
    sentence-aware splitting
          ↓
    tts.generate(chunk)
          ↓
    PCM16
          ↓
    WebSocket

    Reference conditioning remains loaded in the same model
    for every chunk.
    """

    started = ms()

    try:

        text_chunks = split_text_for_tts(
            text
        )

        if not text_chunks:

            raise ValueError(
                "No text available for generation."
            )

        log.info(
            "WebSocket TTS | "
            "chars=%d | chunks=%d",
            len(text),
            len(text_chunks),
        )

        successful_chunks = 0

        # ----------------------------------------------------
        # Generate every text chunk
        # ----------------------------------------------------

        for index, chunk_text in enumerate(
            text_chunks,
            start=1,
        ):

            chunk_started = ms()

            log.info(
                "Generating chunk %d/%d | "
                "chars=%d | text=%r",
                index,
                len(text_chunks),
                len(chunk_text),
                chunk_text,
            )

            # IMPORTANT:
            #
            # Use Chatterbox's normal generation path.
            #
            # Do not manually call:
            #
            # tts._encode_text()
            # tts.t3.generate()
            # tts.s3gen.inference()

            wav_tensor = generate_full(
                chunk_text
            )

            pcm = to_pcm16(
                wav_tensor
            )

            if not pcm:

                log.warning(
                    "Chunk %d/%d returned empty audio.",
                    index,
                    len(text_chunks),
                )

                continue

            successful_chunks += 1

            generation_ms = (
                ms() - started
            )

            # Send completed chunk to websocket task.
            out_q.put({
                "type": "audio",
                "pcm": pcm,
                "chunk": index,
                "total_chunks": len(
                    text_chunks
                ),
                "generation_ms": (
                    generation_ms
                ),
            })

            log.info(
                "Chunk %d/%d complete | "
                "chars=%d | "
                "audio_bytes=%d | "
                "chunk_ms=%.2f",
                index,
                len(text_chunks),
                len(chunk_text),
                len(pcm),
                ms() - chunk_started,
            )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if successful_chunks == 0:

            raise RuntimeError(
                "No audio was generated."
            )

        total_ms = (
            ms() - started
        )

        log.info(
            "WebSocket TTS complete | "
            "chars=%d | "
            "chunks=%d | "
            "successful=%d | "
            "total_ms=%.2f",
            len(text),
            len(text_chunks),
            successful_chunks,
            total_ms,
        )

        out_q.put({
            "type": "end",
            "total_ms": total_ms,
            "text_chunks": len(
                text_chunks
            ),
        })

    except Exception as exc:

        log.exception(
            "WebSocket generation failed"
        )

        out_q.put({
            "type": "error",
            "message": str(exc),
        })


# ============================================================
# WebSocket endpoint
# ============================================================

@app.websocket("/ws/tts")
async def websocket_tts(
    ws: WebSocket,
):

    await ws.accept()

    log.info(
        "WebSocket connected"
    )

    try:

        while True:

            # ------------------------------------------------
            # Receive JSON request
            # ------------------------------------------------

            raw_payload = (
                await ws.receive_text()
            )

            payload = json.loads(
                raw_payload
            )

            text = payload.get(
                "text",
                "",
            ).strip()

            if not text:

                await ws.send_json({
                    "type": "error",
                    "message": "text is required",
                })

                continue

            if (
                tts is None
                or tts.conds is None
            ):

                await ws.send_json({
                    "type": "error",
                    "message": (
                        "Reference voice is not prepared"
                    ),
                })

                continue

            request_started = ms()

            # ------------------------------------------------
            # Determine chunk count
            # ------------------------------------------------

            text_chunks = split_text_for_tts(
                text
            )

            log.info(
                "Received WebSocket request | "
                "chars=%d | chunks=%d",
                len(text),
                len(text_chunks),
            )

            # ------------------------------------------------
            # Start response
            # ------------------------------------------------

            await ws.send_json({
                "type": "start",
                "model": "chatterbox-flash",
                "sample_rate": tts.sr,
                "channels": 1,
                "sample_width": 2,
                "format": "pcm_s16le",
                "backend": BACKEND,
                "block_size": BLOCK_SIZE,
                "text_chars": len(text),
                "text_chunks": len(
                    text_chunks
                ),
            })

            # ------------------------------------------------
            # Start generation thread
            # ------------------------------------------------

            out_q = queue.Queue()

            worker = threading.Thread(
                target=streaming_generate,
                args=(
                    text,
                    out_q,
                ),
                daemon=True,
            )

            worker.start()

            first_audio = True

            # ------------------------------------------------
            # Forward worker output
            # ------------------------------------------------

            while True:

                item = await asyncio.to_thread(
                    out_q.get
                )

                # --------------------------------------------
                # Audio
                # --------------------------------------------

                if item["type"] == "audio":

                    if first_audio:

                        await ws.send_json({
                            "type": "first_audio",

                            "ttfb_ms": (
                                ms()
                                - request_started
                            ),

                            "model_generation_ms": (
                                item["generation_ms"]
                            ),
                        })

                        first_audio = False

                    await ws.send_bytes(
                        item["pcm"]
                    )

                    await ws.send_json({
                        "type": "progress",
                        "chunk": item.get(
                            "chunk"
                        ),
                        "total_chunks": item.get(
                            "total_chunks"
                        ),
                    })

                # --------------------------------------------
                # Finished
                # --------------------------------------------

                elif item["type"] == "end":

                    await ws.send_json({
                        "type": "done",

                        "generation_ms": (
                            item["total_ms"]
                        ),

                        "request_ms": (
                            ms()
                            - request_started
                        ),

                        "text_chunks": (
                            item.get(
                                "text_chunks",
                                1,
                            )
                        ),
                    })

                    break

                # --------------------------------------------
                # Error
                # --------------------------------------------

                elif item["type"] == "error":

                    await ws.send_json({
                        "type": "error",
                        "message": item[
                            "message"
                        ],
                    })

                    break

    except WebSocketDisconnect:

        log.info(
            "WebSocket disconnected"
        )

    except Exception:

        log.exception(
            "WebSocket connection failed"
        )
