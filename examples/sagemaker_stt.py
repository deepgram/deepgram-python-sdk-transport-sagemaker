"""Streaming STT transcription via SageMaker (V1 Listen, e.g. Nova-3, Nova-2).

Paces audio to real-time to match how a live microphone would behave.

Usage:
    export SAGEMAKER_ENDPOINT=my-deepgram-stt-endpoint
    export AWS_REGION=us-east-2
    python examples/sagemaker_stt.py
"""

import asyncio
import os
import struct
import sys
from pathlib import Path

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram_sagemaker import SageMakerTransportFactory

ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "deepgram-nova-3")
REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
AUDIO_FILE = Path(__file__).resolve().parent.parent / "spacewalk.wav"


async def main():
    factory = SageMakerTransportFactory(endpoint_name=ENDPOINT, region=REGION)
    client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

    print(f"Streaming STT via SageMaker (V1 Listen)")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Model:    {MODEL}")
    print(f"Region:   {REGION}")
    print()

    async with client.listen.v1.connect(model=MODEL, interim_results="true") as connection:

        def on_message(message):
            try:
                channel = message.channel
                if channel and channel.alternatives:
                    transcript = channel.alternatives[0].transcript
                    if transcript:
                        is_final = getattr(message, "is_final", False)
                        prefix = "[final]  " if is_final else "[interim]"
                        print(f"{prefix} {transcript}")
            except Exception:
                pass

        close_sent = False

        def on_error(error):
            if not close_sent:
                print(f"Error: {error}", file=sys.stderr)

        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)

        listen_task = asyncio.create_task(connection.start_listening())
        await asyncio.sleep(0.5)

        if not AUDIO_FILE.exists():
            print(f"Audio file not found: {AUDIO_FILE}")
            print("Download from: https://dpgr.am/spacewalk.wav")
            return

        audio = AUDIO_FILE.read_bytes()

        # Parse WAV header for pacing
        sample_rate = struct.unpack_from("<I", audio, 24)[0]
        block_align = struct.unpack_from("<H", audio, 32)[0]

        chunk_size = 8192
        frames_per_chunk = chunk_size / block_align
        sleep_secs = frames_per_chunk / sample_rate

        print(f"Streaming WAV: {sample_rate} Hz, block align {block_align}, "
              f"pacing {sleep_secs * 1000:.0f} ms per chunk\n")

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            await connection.send_media(chunk)
            await asyncio.sleep(sleep_secs)

        close_sent = True
        await connection.send_close_stream()
        await asyncio.sleep(3)
        listen_task.cancel()

    print("Done.")


asyncio.run(main())
