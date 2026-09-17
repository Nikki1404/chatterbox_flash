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


async def run_tts(
    url: str,
    text: str,
    output: str,
    play: bool = False,
):
    print("=" * 70)
    print("Chatterbox Flash WebSocket Client")
    print("=" * 70)
    print(f"Server : {url}")
    print(f"Text   : {text}")
    print(f"Output : {output}")
    print("=" * 70)

    request_start = time.perf_counter()

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------
    print("\n[1] Connecting...")

    connect_start = time.perf_counter()

    async with websockets.connect(
        url,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    ) as websocket:

        connected_at = time.perf_counter()
        connection_ms = ms(connected_at - connect_start)

        print(f"[OK] Connected in {connection_ms:.2f} ms")

        # ---------------------------------------------------------
        # Send request
        # ---------------------------------------------------------
        request = {
            "text": text
        }

        await websocket.send(json.dumps(request))

        request_sent_at = time.perf_counter()

        print("[2] TTS request sent")
        print("[3] Waiting for first audio chunk...")

        # ---------------------------------------------------------
        # Receive streaming audio
        # ---------------------------------------------------------
        audio_chunks = []

        first_audio_at = None
        last_audio_at = None

        sample_rate = 24000
        channels = 1
        sample_width = 2

        chunk_count = 0
        total_audio_bytes = 0

        previous_chunk_time = None
        chunk_intervals = []

        server_metadata = {}

        while True:
            message = await websocket.recv()

            current_time = time.perf_counter()

            # -----------------------------------------------------
            # Binary = PCM16 audio
            # -----------------------------------------------------
            if isinstance(message, bytes):

                if first_audio_at is None:
                    first_audio_at = current_time

                    ttfb_ms = ms(first_audio_at - request_sent_at)

                    print(
                        f"[AUDIO] First chunk received "
                        f"after {ttfb_ms:.2f} ms"
                    )

                if previous_chunk_time is not None:
                    chunk_intervals.append(
                        ms(current_time - previous_chunk_time)
                    )

                previous_chunk_time = current_time
                last_audio_at = current_time

                chunk_count += 1
                total_audio_bytes += len(message)

                audio_chunks.append(message)

                print(
                    f"[chunk {chunk_count:03d}] "
                    f"{len(message):8d} bytes | "
                    f"total={total_audio_bytes:10d} bytes"
                )

                continue

            # -----------------------------------------------------
            # Text = metadata/control message
            # -----------------------------------------------------
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                print(f"[server] {message}")
                continue

            message_type = data.get("type")

            if message_type == "start":
                server_metadata.update(data)

                sample_rate = data.get(
                    "sample_rate",
                    sample_rate,
                )

                print(
                    f"[START] sample_rate={sample_rate}"
                )

            elif message_type == "done":
                server_metadata.update(data)

                print("[DONE] Server finished generation")
                break

            elif message_type == "error":
                raise RuntimeError(
                    data.get("message", "Unknown server error")
                )

            else:
                print(f"[server] {data}")

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------
        finished_at = time.perf_counter()

    if not audio_chunks:
        raise RuntimeError(
            "Server returned no audio."
        )

    pcm = b"".join(audio_chunks)

    # PCM16 = 2 bytes/sample
    number_of_samples = (
        len(pcm)
        / sample_width
        / channels
    )

    audio_duration = (
        number_of_samples
        / sample_rate
    )

    connection_ms = ms(
        connected_at - connect_start
    )

    ttfb_ms = ms(
        first_audio_at - request_sent_at
    )

    generation_seconds = (
        finished_at - request_sent_at
    )

    total_seconds = (
        finished_at - request_start
    )

    streaming_seconds = (
        last_audio_at - first_audio_at
        if last_audio_at and first_audio_at
        else 0
    )

    # Real Time Factor
    # < 1 means faster than realtime
    rtf = (
        generation_seconds / audio_duration
        if audio_duration > 0
        else 0
    )

    # ---------------------------------------------------------
    # Save WAV
    # ---------------------------------------------------------
    output_path = Path(output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("LATENCY REPORT")
    print("=" * 70)

    print(f"Connection time : {connection_ms:10.2f} ms")
    print(f"TTFB             : {ttfb_ms:10.2f} ms")
    print(f"Generation time  : {generation_seconds:10.3f} sec")
    print(f"Total latency    : {total_seconds:10.3f} sec")
    print(f"Audio duration   : {audio_duration:10.3f} sec")
    print(f"RTF              : {rtf:10.3f}")
    print(f"Audio chunks     : {chunk_count:10d}")
    print(f"Audio bytes      : {len(pcm):10d}")
    print(f"Sample rate      : {sample_rate:10d} Hz")

    if streaming_seconds > 0:
        print(
            f"Streaming window : "
            f"{streaming_seconds:10.3f} sec"
        )

    if chunk_intervals:
        intervals = np.asarray(
            chunk_intervals,
            dtype=np.float64,
        )

        print("\nChunk timing:")
        print(
            f"  Average interval : "
            f"{intervals.mean():.2f} ms"
        )
        print(
            f"  Minimum interval : "
            f"{intervals.min():.2f} ms"
        )
        print(
            f"  Maximum interval : "
            f"{intervals.max():.2f} ms"
        )

    # Server-side metrics if provided
    if server_metadata:
        print("\nServer metadata:")

        for key, value in server_metadata.items():
            if key != "type":
                print(f"  {key}: {value}")

    print("\nSaved:")
    print(output_path.resolve())

    print("=" * 70)

    # ---------------------------------------------------------
    # Optional playback
    # ---------------------------------------------------------
    if play:

        if sd is None:
            print(
                "\n[WARN] sounddevice is not installed. "
                "Skipping playback."
            )
            return

        print("\nPlaying audio...")

        audio_np = np.frombuffer(
            pcm,
            dtype="<i2",
        ).astype(np.float32)

        audio_np /= 32768.0

        sd.play(
            audio_np,
            samplerate=sample_rate,
        )

        sd.wait()


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
        help="WebSocket TTS URL",
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Text to synthesize",
    )

    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV file",
    )

    parser.add_argument(
        "--play",
        action="store_true",
        help="Play generated audio",
    )

    args = parser.parse_args()

    asyncio.run(
        run_tts(
            url=args.url,
            text=args.text,
            output=args.output,
            play=args.play,
        )
    )


if __name__ == "__main__":
    main()
