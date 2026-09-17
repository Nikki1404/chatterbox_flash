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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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


# Maximum text size for one T3 generation.
#
# Long text sent by the client is automatically split into
# multiple TTS-safe utterances.
MAX_CHUNK_CHARS = int(
    os.getenv("MAX_CHUNK_CHARS", "280")
)


# Safety ceiling for generated speech tokens for one chunk.
#
# This is deliberately below very large model position limits.
MAX_SPEECH_TOKENS = int(
    os.getenv("MAX_SPEECH_TOKENS", "2048")
)


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="Chatterbox-Flash TTS",
    version="1.1.0",
)

tts = None

# One model / GPU.
#
# Prevent simultaneous inference requests from trying to use
# the same model state/GPU buffers at the same time.
gpu_lock = threading.Lock()


# ============================================================
# OpenAI request schema
# ============================================================

class OpenAISpeechRequest(BaseModel):

    model: str = "chatterbox-flash"

    input: str

    voice: str = "reference"

    response_format: Literal[
        "wav",
        "pcm",
    ] = "wav"

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

        arr = np.asarray(
            wav_tensor
        )

    arr = np.asarray(
        arr
    ).squeeze()

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
    ).astype(
        "<i2"
    ).tobytes()


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int,
) -> bytes:

    buf = io.BytesIO()

    with wave.open(
        buf,
        "wb",
    ) as wf:

        wf.setnchannels(1)

        wf.setsampwidth(2)

        wf.setframerate(
            sample_rate
        )

        wf.writeframes(
            pcm
        )

    return buf.getvalue()


# ============================================================
# Text splitting
# ============================================================

def split_text_for_tts(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
):
    """
    Split long text into safe TTS chunks.

    Priority:

        sentence boundary
              ↓
        comma/semicolon
              ↓
        whitespace

    No text is intentionally discarded.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    # --------------------------------------------------------
    # Sentence split
    # --------------------------------------------------------

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
        # Normal sentence
        # ----------------------------------------------------

        if len(sentence) <= max_chars:

            if current:

                candidate = (
                    current
                    + " "
                    + sentence
                )

            else:

                candidate = sentence

            if len(candidate) <= max_chars:

                current = candidate

            else:

                if current:
                    chunks.append(
                        current
                    )

                current = sentence

            continue

        # ----------------------------------------------------
        # Sentence itself is too long
        # ----------------------------------------------------

        if current:

            chunks.append(
                current
            )

            current = ""

        # Try punctuation boundaries.
        pieces = re.split(
            r"(?<=[,;:])\s+",
            sentence,
        )

        piece_buffer = ""

        for piece in pieces:

            piece = piece.strip()

            if not piece:
                continue

            if piece_buffer:

                candidate = (
                    piece_buffer
                    + " "
                    + piece
                )

            else:

                candidate = piece

            if len(candidate) <= max_chars:

                piece_buffer = candidate

                continue

            if piece_buffer:

                chunks.append(
                    piece_buffer
                )

                piece_buffer = ""

            # ------------------------------------------------
            # Piece is still too large.
            # Split by words.
            # ------------------------------------------------

            if len(piece) > max_chars:

                words = piece.split()

                word_buffer = ""

                for word in words:

                    if word_buffer:

                        candidate = (
                            word_buffer
                            + " "
                            + word
                        )

                    else:

                        candidate = word

                    if len(candidate) <= max_chars:

                        word_buffer = candidate

                    else:

                        if word_buffer:

                            chunks.append(
                                word_buffer
                            )

                        word_buffer = word

                if word_buffer:

                    piece_buffer = (
                        word_buffer
                    )

            else:

                piece_buffer = piece

        if piece_buffer:

            current = (
                piece_buffer
            )

    if current:

        chunks.append(
            current
        )

    return chunks


# ============================================================
# Standard full generation
# ============================================================

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


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
async def startup():

    global tts

    # --------------------------------------------------------
    # CUDA check
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
            "Torch CUDA: %s",
            torch.version.cuda,
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    started = ms()

    tts = (
        ChatterboxFlashTTS
        .from_pretrained(
            MODEL_ID,

            device=DEVICE,

            dtype=(
                torch.bfloat16
                if DEVICE == "cuda"
                else torch.float32
            ),

            drf_block_size=BLOCK_SIZE,
        )
    )

    log.info(
        "Model loaded in %.2f ms | sample_rate=%s",
        ms() - started,
        tts.sr,
    )

    # --------------------------------------------------------
    # Reference audio
    # --------------------------------------------------------

    reference_path = Path(
        REFERENCE_AUDIO
    )

    if not reference_path.exists():

        log.warning(
            "Reference audio missing: %s",
            REFERENCE_AUDIO,
        )

        return

    if not reference_path.is_file():

        raise RuntimeError(
            f"REFERENCE_AUDIO is not a file: "
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

        if DEVICE == "cuda":
            torch.cuda.synchronize()

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
        "version": "1.1.0",
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

        "max_chunk_chars": MAX_CHUNK_CHARS,

        "max_speech_tokens": MAX_SPEECH_TOKENS,

        "reference_ready": bool(
            tts is not None
            and tts.conds is not None
        ),
    }


# ============================================================
# OpenAI-compatible HTTP endpoint
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
            detail=(
                "Reference voice is not prepared"
            ),
        )

    if req.speed != 1.0:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only speed=1.0 is supported"
            ),
        )

    started = ms()

    try:

        # ----------------------------------------------------
        # Split long HTTP input as well.
        # ----------------------------------------------------

        chunks = split_text_for_tts(
            text
        )

        log.info(
            "HTTP TTS: %d chars -> %d chunk(s)",
            len(text),
            len(chunks),
        )

        pcm_parts = []

        for index, chunk_text in enumerate(
            chunks,
            start=1,
        ):

            log.info(
                "HTTP chunk %d/%d | chars=%d",
                index,
                len(chunks),
                len(chunk_text),
            )

            wav_tensor = await asyncio.to_thread(
                generate_full,
                chunk_text,
            )

            pcm_parts.append(
                to_pcm16(
                    wav_tensor
                )
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
        "chunks=%d | %.2f ms",
        len(chunks),
        elapsed,
    )

    headers = {
        "X-Model": "chatterbox-flash",
        "X-Generation-Ms": f"{elapsed:.2f}",
        "X-Text-Chunks": str(
            len(chunks)
        ),
        "Cache-Control": "no-store",
    }

    if req.response_format == "pcm":

        return Response(
            content=pcm,

            media_type=(
                "application/octet-stream"
            ),

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
# WebSocket generation
# ============================================================

def streaming_generate(
    text: str,
    out_q: queue.Queue,
):
    """
    Long-text-safe WebSocket generation.

    Client can send the entire text/file.

    Server:

        full text
           ↓
        split text
           ↓
        T3 chunk
           ↓
        S3Gen
           ↓
        PCM
           ↓
        WebSocket

    The same reference conditioning is reused for every
    generated chunk.
    """

    started = ms()

    try:

        # ----------------------------------------------------
        # Split input
        # ----------------------------------------------------

        chunks = split_text_for_tts(
            text
        )

        if not chunks:

            raise ValueError(
                "No text available for generation."
            )

        log.info(
            "WebSocket TTS request: "
            "%d chars -> %d chunk(s)",
            len(text),
            len(chunks),
        )

        # ----------------------------------------------------
        # Generate chunks sequentially
        # ----------------------------------------------------

        for index, chunk_text in enumerate(
            chunks,
            start=1,
        ):

            chunk_started = ms()

            log.info(
                "Generating chunk %d/%d | "
                "chars=%d | text=%r",
                index,
                len(chunks),
                len(chunk_text),
                chunk_text[:120],
            )

            with gpu_lock, torch.inference_mode():

                # --------------------------------------------
                # Encode text
                # --------------------------------------------

                text_tokens = (
                    tts._encode_text(
                        chunk_text,
                        normalize_text=True,
                    )
                )

                n_tokens = int(
                    text_tokens.size(1)
                )

                # --------------------------------------------
                # Speech token budget
                # --------------------------------------------

                estimated_speech_len = max(
                    n_tokens * 6,
                    300,
                )

                total_speech_len = min(
                    estimated_speech_len,
                    MAX_SPEECH_TOKENS,
                )

                log.info(
                    "Chunk %d/%d | "
                    "text_tokens=%d | "
                    "speech_budget=%d",
                    index,
                    len(chunks),
                    n_tokens,
                    total_speech_len,
                )

                # --------------------------------------------
                # T3 generation
                # --------------------------------------------

                speech_tokens = (
                    tts.t3.generate(
                        t3_cond=tts.conds.t3,

                        text_tokens=text_tokens,

                        text_token_lens=(
                            torch.tensor(
                                [n_tokens],
                                device=tts.device,
                                dtype=torch.long,
                            )
                        ),

                        total_speech_len=(
                            total_speech_len
                        ),

                        num_steps=NUM_STEPS,

                        temperature=TEMPERATURE,

                        time_shift_tau=TIME_SHIFT_TAU,

                        omnivoice_schedule_t_shift=0.5,

                        cfg_scale=CFG_SCALE,

                        position_temperature=(
                            POSITION_TEMPERATURE
                        ),

                        pmi_uncond_prior_precompute=True,

                        use_cuda_graph=True,

                        backend=BACKEND,

                        batch_size=1,
                    )
                )

                # --------------------------------------------
                # Normalize speech token shape
                # --------------------------------------------

                if speech_tokens.ndim == 2:

                    tokens = (
                        speech_tokens[0]
                    )

                else:

                    tokens = (
                        speech_tokens
                    )

                # --------------------------------------------
                # Remove stop token and anything after it
                # --------------------------------------------

                stop_token = (
                    tts.t3.hp.stop_speech_token
                )

                eos = (
                    tokens
                    == stop_token
                ).nonzero(
                    as_tuple=True
                )[0]

                if len(eos):

                    tokens = tokens[
                        :eos[0].item()
                    ]

                if tokens.numel() == 0:

                    log.warning(
                        "Chunk %d/%d produced "
                        "no speech tokens.",
                        index,
                        len(chunks),
                    )

                    continue

                # --------------------------------------------
                # S3Gen
                # --------------------------------------------

                wav_tensor, _ = (
                    tts.s3gen.inference(
                        speech_tokens=(
                            tokens.to(
                                tts.device
                            )
                        ),

                        ref_dict=(
                            tts.conds.gen
                        ),

                        n_cfm_timesteps=(
                            N_CFM_TIMESTEPS
                        ),
                    )
                )

                if DEVICE == "cuda":
                    torch.cuda.synchronize()

                # --------------------------------------------
                # Convert to PCM16
                # --------------------------------------------

                pcm = to_pcm16(
                    wav_tensor
                )

            # ------------------------------------------------
            # Send generated utterance
            # ------------------------------------------------

            if pcm:

                out_q.put({
                    "type": "audio",

                    "pcm": pcm,

                    "chunk": index,

                    "total_chunks": len(
                        chunks
                    ),

                    "generation_ms": (
                        ms() - started
                    ),
                })

            log.info(
                "Chunk %d/%d complete | "
                "text_chars=%d | "
                "speech_tokens=%d | "
                "audio_bytes=%d | "
                "%.2f ms",
                index,
                len(chunks),
                len(chunk_text),
                int(tokens.numel()),
                len(pcm),
                ms() - chunk_started,
            )

        # ----------------------------------------------------
        # Complete
        # ----------------------------------------------------

        if DEVICE == "cuda":
            torch.cuda.synchronize()

        total_ms = (
            ms() - started
        )

        log.info(
            "Complete WebSocket TTS finished | "
            "text_chars=%d | "
            "chunks=%d | "
            "%.2f ms",
            len(text),
            len(chunks),
            total_ms,
        )

        out_q.put({
            "type": "end",

            "total_ms": total_ms,

            "text_chunks": len(
                chunks
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
            # Receive request
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
                    "message": (
                        "text is required"
                    ),
                })

                continue

            if (
                tts is None
                or tts.conds is None
            ):

                await ws.send_json({
                    "type": "error",
                    "message": (
                        "Reference voice "
                        "is not prepared"
                    ),
                })

                continue

            request_started = ms()

            # ------------------------------------------------
            # Calculate chunk count before generation
            # ------------------------------------------------

            text_chunks = (
                split_text_for_tts(
                    text
                )
            )

            log.info(
                "Received WebSocket request | "
                "chars=%d | chunks=%d",
                len(text),
                len(text_chunks),
            )

            # ------------------------------------------------
            # Start metadata
            # ------------------------------------------------

            await ws.send_json({
                "type": "start",

                "model": (
                    "chatterbox-flash"
                ),

                "sample_rate": (
                    tts.sr
                ),

                "channels": 1,

                "sample_width": 2,

                "format": (
                    "pcm_s16le"
                ),

                "backend": (
                    BACKEND
                ),

                "block_size": (
                    BLOCK_SIZE
                ),

                "text_chars": (
                    len(text)
                ),

                "text_chunks": (
                    len(text_chunks)
                ),
            })

            # ------------------------------------------------
            # Worker
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
            # Stream generated chunks
            # ------------------------------------------------

            while True:

                item = (
                    await asyncio.to_thread(
                        out_q.get
                    )
                )

                # --------------------------------------------
                # Audio
                # --------------------------------------------

                if item["type"] == "audio":

                    if first_audio:

                        await ws.send_json({
                            "type": (
                                "first_audio"
                            ),

                            "ttfb_ms": (
                                ms()
                                - request_started
                            ),

                            "model_generation_ms": (
                                item[
                                    "generation_ms"
                                ]
                            ),
                        })

                        first_audio = False

                    # Send PCM16
                    await ws.send_bytes(
                        item["pcm"]
                    )

                    # Optional progress metadata
                    await ws.send_json({
                        "type": "progress",

                        "chunk": item.get(
                            "chunk"
                        ),

                        "total_chunks": (
                            item.get(
                                "total_chunks"
                            )
                        ),
                    })

                # --------------------------------------------
                # End
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

                        "message": (
                            item["message"]
                        ),
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
