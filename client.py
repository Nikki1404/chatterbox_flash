import argparse
import asyncio
import json
import time
import wave

import websockets


async def main(args):
    async with websockets.connect(
        args.url,
        max_size=None,
        compression=None,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        started = time.perf_counter()
        await ws.send(json.dumps({"text": args.text}))

        sample_rate = None
        chunks = []
        first_audio = None
        server_ttfb = None
        generation_ms = None

        while True:
            message = await ws.recv()

            if isinstance(message, bytes):
                if first_audio is None:
                    first_audio = time.perf_counter()
                    print(f"Client TTFB: {(first_audio-started)*1000:.2f} ms")
                chunks.append(message)
                print(f"chunk {len(chunks)}: {len(message)} bytes")
                continue

            data = json.loads(message)

            if data["type"] == "start":
                sample_rate = data["sample_rate"]
                print(
                    f"{sample_rate} Hz | {data['format']} | "
                    f"backend={data['backend']} | block={data['block_size']}"
                )

            elif data["type"] == "first_audio":
                server_ttfb = data["ttfb_ms"]
                print(f"Server TTFB: {server_ttfb:.2f} ms")
                print(f"Model first audio: {data['model_generation_ms']:.2f} ms")

            elif data["type"] == "end":
                generation_ms = data["generation_ms"]
                break

            elif data["type"] == "error":
                raise RuntimeError(data["message"])

        pcm = b"".join(chunks)

        with wave.open(args.output, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)

        duration = (len(pcm) // 2) / sample_rate
        total_ms = (time.perf_counter() - started) * 1000
        rtf = (generation_ms / 1000) / duration if duration else 0

        print("\n--- Metrics ---")
        print(f"Chunks:        {len(chunks)}")
        print(f"Audio:         {duration:.3f} sec")
        if first_audio:
            print(f"Client TTFB:   {(first_audio-started)*1000:.2f} ms")
        if server_ttfb is not None:
            print(f"Server TTFB:   {server_ttfb:.2f} ms")
        print(f"Generation:    {generation_ms:.2f} ms")
        print(f"Total:         {total_ms:.2f} ms")
        print(f"RTF:           {rtf:.3f}")
        print(f"Saved:         {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/tts")
    p.add_argument("--text", required=True)
    p.add_argument("--output", default="ws_output.wav")
    asyncio.run(main(p.parse_args()))
