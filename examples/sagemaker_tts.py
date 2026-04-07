"""Streaming TTS via SageMaker (V1 Speak, Aura 2).

Sends text to a Deepgram TTS model on SageMaker and saves the resulting
audio to a WAV file, then plays it.

Usage:
    export SAGEMAKER_ENDPOINT=my-deepgram-tts-endpoint
    export AWS_REGION=us-east-2
    python examples/sagemaker_tts.py
"""

import asyncio
import os
import struct
import subprocess
import sys

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types import SpeakV1Text, SpeakV1Flush, SpeakV1FlushType, SpeakV1Close, SpeakV1CloseType
from deepgram_sagemaker import SageMakerTransportFactory

ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "deepgram-tts")
REGION = os.getenv("AWS_REGION", "us-west-2")
OUTPUT_FILE = "tts_output.wav"


async def main():
    factory = SageMakerTransportFactory(endpoint_name=ENDPOINT, region=REGION)
    client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

    print("Text-to-Speech via SageMaker")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Region:   {REGION}")
    print(f"Output:   {OUTPUT_FILE}")
    print()

    audio_chunks = []
    close_sent = False

    async with client.speak.v1.connect(model="aura-2-atlas-en") as connection:

        def on_message(data):
            if isinstance(data, (bytes, bytearray)):
                audio_chunks.append(data)
                count = len(audio_chunks)
                if count % 50 == 1 or count <= 5:
                    print(f"Received audio chunk #{count} ({len(data)} bytes)")

        def on_error(error):
            if not close_sent:
                print(f"Error: {error}", file=sys.stderr)

        def on_close(_):
            if not close_sent:
                print("Connection closed")

        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)
        connection.on(EventType.CLOSE, on_close)

        listen_task = asyncio.create_task(connection.start_listening())
        await asyncio.sleep(1)

        sentences = [
            "Hello, this is a text-to-speech test running on Amazon SageMaker.",
            "The Deepgram model is generating audio from text in real time.",
            "This audio is being streamed back through the Python SDK transport layer.",
        ]

        for sentence in sentences:
            print(f'Sending: "{sentence}"')
            await connection.send_text(SpeakV1Text(text=sentence, type="Speak"))

        await connection.send_flush(SpeakV1Flush(type="Flush"))
        print("Waiting for audio...")

        await asyncio.sleep(10)

        close_sent = True
        await connection.send_close(SpeakV1Close(type="Close"))
        await asyncio.sleep(2)
        listen_task.cancel()

    total = len(audio_chunks)
    total_bytes = sum(len(c) for c in audio_chunks)
    print(f"\nTotal audio chunks: {total}")
    print(f"Total audio bytes: {total_bytes}")

    if audio_chunks:
        pcm = b"".join(audio_chunks)
        sr, ch, bps = 24000, 1, 16
        with open(OUTPUT_FILE, "wb") as f:
            f.write(struct.pack("<4sI4s4sIHHIIHH4sI",
                b"RIFF", 36 + len(pcm), b"WAVE",
                b"fmt ", 16, 1, ch, sr, sr * ch * bps // 8, ch * bps // 8, bps,
                b"data", len(pcm)))
            f.write(pcm)
        print(f"Audio saved to {OUTPUT_FILE}")
        print("Playing audio...")
        subprocess.run(["afplay", OUTPUT_FILE])

    print("Done.")


asyncio.run(main())
