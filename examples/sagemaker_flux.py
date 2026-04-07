"""Flux STT transcription via SageMaker (V2 Listen).

Flux is a V2 model using turn-based transcription. Audio is paced to real-time.

Usage:
    export SAGEMAKER_ENDPOINT=my-deepgram-flux-endpoint
    export AWS_REGION=us-east-2
    python examples/sagemaker_flux.py
"""

import asyncio
import os
import struct
import sys
from pathlib import Path

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram_sagemaker import SageMakerTransportFactory

ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "deepgram-flux")
REGION = os.getenv("AWS_REGION", "us-west-2")
AUDIO_FILE = Path(__file__).resolve().parent.parent / "spacewalk.wav"


async def main():
    factory = SageMakerTransportFactory(endpoint_name=ENDPOINT, region=REGION)
    client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

    print("Flux transcription via SageMaker (V2 Listen)")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Region:   {REGION}")
    print()

    if not AUDIO_FILE.exists():
        print(f"Audio file not found: {AUDIO_FILE}")
        print("Download from: https://dpgr.am/spacewalk.wav")
        return

    async with client.listen.v2.connect(model="flux-general-en") as connection:

        def on_turn_info(turn_info):
            try:
                transcript = turn_info.transcript
                event = turn_info.event
                turn_index = turn_info.turn_index
                if transcript:
                    print(f"[{event}] turn={turn_index:.0f}  {transcript}")
            except Exception:
                pass

        def on_error(error):
            print(f"Error: {error}", file=sys.stderr)

        connection.on(EventType.MESSAGE, on_turn_info)
        connection.on(EventType.ERROR, on_error)

        listen_task = asyncio.create_task(connection.start_listening())
        await asyncio.sleep(0.5)

        audio = AUDIO_FILE.read_bytes()

        # Parse WAV header for pacing
        sample_rate = struct.unpack_from("<I", audio, 24)[0]
        block_align = struct.unpack_from("<H", audio, 32)[0]

        chunk_size = 8192
        frames_per_chunk = chunk_size / block_align
        sleep_secs = frames_per_chunk / sample_rate

        print(f"Streaming WAV: {sample_rate} Hz, block align {block_align}\n")

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            await connection.send_media(chunk)
            await asyncio.sleep(sleep_secs)

        await connection.send_close_stream()
        await asyncio.sleep(3)
        listen_task.cancel()

    print("\nDone.")


asyncio.run(main())
