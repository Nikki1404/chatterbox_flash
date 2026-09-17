#!/usr/bin/env python3

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import numpy as np
import websockets

try:
    import sounddevice as sd
except ImportError:
    sd = None


DEFAULT_URL = "ws://127.0.0.1:8000/ws/tts"


def ms(seconds: float) -> float:
    return seconds * 1000.0


def load_text(args) -> str:
    """
    Load text either directly from --text
    or from a UTF-8 text file using --text-file.
    """

    if args.text_file:
        text_path = Path(args.text_file)

        if not text_path.exists():
            raise FileNotFoundError(
                f"Text file not found: {text_path}"
            )

        if not text_path.is_file():
            raise ValueError(
                f"Text path is not a file: {text_path}"
            )

        text = text_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            raise ValueError(
                f"Text file is empty: {text_path}"
            )

        print(f"[INFO] Text file  : {text_path}")
        print(f"[INFO] Characters : {len(text)}")

        return text

    text = args.text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    return text


async def run_tts(
    url: str,
    text: str,
    output: str,
    play: bool = False,
):
    print("\n" + "=" * 70)
    print("CHATTERBOX FLASH - WEBSOCKET CLIENT")
    print("=" * 70)

    print(f"Server     : {url}")
    print(f"Characters : {len(text)}")
    print(f"Output     : {output}")

    preview = text[:150].replace("\n", " ")

    if len(text) > 150:
        preview += "..."

    print(f"Text       : {preview}")
    print("=" * 70)

    overall_start = time.perf_counter()

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------

    print("\n[1] Connecting to WebSocket...")

    connect_start = time.perf_counter()

    async with websockets.connect(
        url,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=10,
    ) as websocket:

        connected_at = time.perf_counter()

        connection_ms = ms(
            connected_at - connect_start
        )

        print(
            f"[OK] Connected in "
            f"{connection_ms:.2f} ms"
        )

        # -----------------------------------------------------
        # Send request
        # -----------------------------------------------------

        request = {
            "text": text
        }

        await websocket.send(
            json.dumps(request)
        )

        request_sent_at = time.perf_counter()

        print("[2] TTS request sent")
        print("[3] Waiting for audio...")

        # -----------------------------------------------------
        # Receive stream
        # -----------------------------------------------------

        audio_chunks = []

        first_audio_at = None
        last_audio_at = None

        previous_chunk_at = None

        chunk_intervals = []
        chunk_sizes = []

        chunk_count = 0
        total_audio_bytes = 0

        sample_rate = 24000
        channels = 1
        sample_width = 2

        server_metadata = {}

        while True:

            try:
                message = await websocket.recv()

            except websockets.ConnectionClosedOK:
                print("[INFO] Server closed connection.")
                break

            except websockets.ConnectionClosed as exc:
                print(
                    f"[WARN] WebSocket closed: "
                    f"{exc.code} {exc.reason}"
                )
                break

            now = time.perf_counter()

            # -------------------------------------------------
            # Binary message = PCM16 audio
            # -------------------------------------------------

            if isinstance(message, bytes):

                if first_audio_at is None:

                    first_audio_at = now

                    ttfb_ms = ms(
                        first_audio_at
                        - request_sent_at
                    )

                    print(
                        "\n[AUDIO] First audio received"
                    )

                    print(
                        f"[TTFB] {ttfb_ms:.2f} ms\n"
                    )

                if previous_chunk_at is not None:

                    interval = ms(
                        now - previous_chunk_at
                    )

                    chunk_intervals.append(
                        interval
                    )

                previous_chunk_at = now
                last_audio_at = now

                chunk_count += 1

                chunk_size = len(message)

                chunk_sizes.append(
                    chunk_size
                )

                total_audio_bytes += (
                    chunk_size
                )

                audio_chunks.append(
                    message
                )

                elapsed = (
                    now - request_sent_at
                )

                print(
                    f"[chunk {chunk_count:03d}] "
                    f"{chunk_size:8d} bytes | "
                    f"total={total_audio_bytes:10d} | "
                    f"t={elapsed:7.3f}s"
                )

                continue

            # -------------------------------------------------
            # Text message = control / metadata
            # -------------------------------------------------

            try:
                data = json.loads(
                    message
                )

            except json.JSONDecodeError:

                print(
                    f"[SERVER] {message}"
                )

                continue

            message_type = data.get(
                "type"
            )

            if message_type == "start":

                server_metadata.update(
                    data
                )

                sample_rate = int(
                    data.get(
                        "sample_rate",
                        sample_rate,
                    )
                )

                channels = int(
                    data.get(
                        "channels",
                        channels,
                    )
                )

                print(
                    f"[START] "
                    f"sample_rate={sample_rate} Hz | "
                    f"channels={channels}"
                )

            elif message_type == "done":

                server_metadata.update(
                    data
                )

                print(
                    "\n[DONE] Server completed generation"
                )

                break

            elif message_type == "error":

                error_message = data.get(
                    "message",
                    data.get(
                        "error",
                        "Unknown server error",
                    ),
                )

                raise RuntimeError(
                    error_message
                )

            else:

                print(
                    f"[SERVER] {data}"
                )

        finished_at = time.perf_counter()

    # ---------------------------------------------------------
    # Validate audio
    # ---------------------------------------------------------

    if not audio_chunks:

        raise RuntimeError(
            "Server returned no audio."
        )

    pcm = b"".join(
        audio_chunks
    )

    # ---------------------------------------------------------
    # Calculate metrics
    # ---------------------------------------------------------

    samples = (
        len(pcm)
        / sample_width
        / channels
    )

    audio_duration = (
        samples
        / sample_rate
    )

    ttfb_ms = ms(
        first_audio_at
        - request_sent_at
    )

    generation_seconds = (
        finished_at
        - request_sent_at
    )

    total_seconds = (
        finished_at
        - overall_start
    )

    streaming_window = 0.0

    if (
        first_audio_at is not None
        and last_audio_at is not None
    ):
        streaming_window = (
            last_audio_at
            - first_audio_at
        )

    # End-to-end generation RTF
    rtf = (
        generation_seconds
        / audio_duration
        if audio_duration > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # Save WAV
    # ---------------------------------------------------------

    output_path = Path(
        output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(
        str(output_path),
        "wb",
    ) as wav_file:

        wav_file.setnchannels(
            channels
        )

        wav_file.setsampwidth(
            sample_width
        )

        wav_file.setframerate(
            sample_rate
        )

        wav_file.writeframes(
            pcm
        )

    # ---------------------------------------------------------
    # Print metrics
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("LATENCY REPORT")
    print("=" * 70)

    print(
        f"Connection time   : "
        f"{connection_ms:10.2f} ms"
    )

    print(
        f"TTFB              : "
        f"{ttfb_ms:10.2f} ms"
    )

    print(
        f"Generation time   : "
        f"{generation_seconds:10.3f} sec"
    )

    print(
        f"Total latency     : "
        f"{total_seconds:10.3f} sec"
    )

    print(
        f"Audio duration    : "
        f"{audio_duration:10.3f} sec"
    )

    print(
        f"RTF               : "
        f"{rtf:10.3f}"
    )

    print(
        f"Streaming window  : "
        f"{streaming_window:10.3f} sec"
    )

    print(
        f"Audio chunks      : "
        f"{chunk_count:10d}"
    )

    print(
        f"Audio bytes       : "
        f"{len(pcm):10d}"
    )

    print(
        f"Sample rate       : "
        f"{sample_rate:10d} Hz"
    )

    # ---------------------------------------------------------
    # Chunk statistics
    # ---------------------------------------------------------

    if chunk_sizes:

        sizes = np.asarray(
            chunk_sizes,
            dtype=np.float64,
        )

        print("\nChunk sizes:")

        print(
            f"  Average         : "
            f"{sizes.mean():10.2f} bytes"
        )

        print(
            f"  Minimum         : "
            f"{sizes.min():10.0f} bytes"
        )

        print(
            f"  Maximum         : "
            f"{sizes.max():10.0f} bytes"
        )

    if chunk_intervals:

        intervals = np.asarray(
            chunk_intervals,
            dtype=np.float64,
        )

        print("\nChunk intervals:")

        print(
            f"  Average         : "
            f"{intervals.mean():10.2f} ms"
        )

        print(
            f"  Minimum         : "
            f"{intervals.min():10.2f} ms"
        )

        print(
            f"  Maximum         : "
            f"{intervals.max():10.2f} ms"
        )

        print(
            f"  P50             : "
            f"{np.percentile(intervals, 50):10.2f} ms"
        )

        print(
            f"  P95             : "
            f"{np.percentile(intervals, 95):10.2f} ms"
        )

    # ---------------------------------------------------------
    # Server metadata
    # ---------------------------------------------------------

    if server_metadata:

        print("\nServer metadata:")

        for key, value in (
            server_metadata.items()
        ):

            if key != "type":

                print(
                    f"  {key}: {value}"
                )

    print("\nSaved WAV:")

    print(
        output_path.resolve()
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # Optional playback
    # ---------------------------------------------------------

    if play:

        if sd is None:

            print(
                "\n[WARN] sounddevice is not installed."
            )

            print(
                "Install using: pip install sounddevice"
            )

            return

        print("\n[PLAY] Playing audio...")

        audio_np = np.frombuffer(
            pcm,
            dtype="<i2",
        ).astype(
            np.float32
        )

        audio_np /= 32768.0

        if channels > 1:

            audio_np = audio_np.reshape(
                -1,
                channels,
            )

        sd.play(
            audio_np,
            samplerate=sample_rate,
        )

        sd.wait()

        print("[PLAY] Finished.")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Chatterbox Flash WebSocket "
            "streaming latency client"
        )
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=(
            "WebSocket URL. "
            f"Default: {DEFAULT_URL}"
        ),
    )

    # ---------------------------------------------------------
    # Either --text OR --text-file
    # ---------------------------------------------------------

    text_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    text_group.add_argument(
        "--text",
        help="Text to synthesize",
    )

    text_group.add_argument(
        "--text-file",
        help=(
            "Path to UTF-8 text file "
            "to synthesize"
        ),
    )

    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV filename",
    )

    parser.add_argument(
        "--play",
        action="store_true",
        help=(
            "Play audio after generation"
        ),
    )

    args = parser.parse_args()

    text = load_text(
        args
    )

    asyncio.run(
        run_tts(
            url=args.url,
            text=text,
            output=args.output,
            play=args.play,
        )
    )


if __name__ == "__main__":
    main()
