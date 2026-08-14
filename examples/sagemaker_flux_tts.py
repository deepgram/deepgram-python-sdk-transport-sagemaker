"""Flux TTS synthesis via SageMaker (V2 Speak).

Flux is a V2 model using turn-based synthesis: each `Speak` message opens or
extends a turn, `Flush` closes the turn and forces the model to emit the
remaining audio. Model strings are `flux-{voice}-{language}`; an Aura string
is rejected on `/v2/speak` (use `examples/sagemaker_tts.py` for Aura).

This example assumes the Flux TTS model package serves the `v2/speak`
invocation path -- the transport derives that path from the WebSocket URL the
SDK builds. Run with `LOG_LEVEL=INFO` to see the path the transport actually
used on the connect line:

    Connecting to SageMaker endpoint: <name> in <region> (path=v2/speak query=...)

Usage:
    export SAGEMAKER_ENDPOINT=my-deepgram-flux-tts-endpoint
    export AWS_REGION=us-east-2
    python examples/sagemaker_flux_tts.py
"""

import asyncio
import logging
import os
import struct
import sys

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v2.types import SpeakV2Speak

from deepgram_sagemaker import SageMakerTransportFactory

ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "deepgram-flux-tts")
REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL = os.getenv("DEEPGRAM_MODEL", "flux-alexis-en")
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING")
OUTPUT_FILE = "flux_tts_output.wav"

# Must match the `sample_rate` passed to connect() -- it drives the WAV header
# written at the end. SpeakV2SampleRate is a string type on the wire.
SAMPLE_RATE = 24000


async def main():
    logging.basicConfig(
        level=LOG_LEVEL.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    factory = SageMakerTransportFactory(endpoint_name=ENDPOINT, region=REGION)
    client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

    print("Flux text-to-speech via SageMaker (V2 Speak)")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Region:   {REGION}")
    print(f"Model:    {MODEL}")
    print(f"Output:   {OUTPUT_FILE}")
    print()

    audio_chunks = []
    close_sent = False

    async with client.speak.v2.connect(
        model=MODEL,
        encoding="linear16",
        sample_rate=str(SAMPLE_RATE),
    ) as connection:

        def on_message(data):
            if isinstance(data, (bytes, bytearray)):
                audio_chunks.append(data)
                count = len(audio_chunks)
                if count % 50 == 1 or count <= 5:
                    print(f"Received audio chunk #{count} ({len(data)} bytes)")
                return

            msg_type = getattr(data, "type", None)
            if msg_type == "SpeechStarted":
                print("SpeechStarted")
            elif msg_type == "SpeechMetadata":
                # Per-turn billing and timing.
                print(
                    f"SpeechMetadata: speech_id={getattr(data, 'speech_id', '?')} "
                    f"audio={getattr(data, 'audio_duration_ms', 0)} ms "
                    f"billable_chars={getattr(data, 'billable_character_count', 0)}"
                )
            elif msg_type == "Flushed":
                print("Flushed")
            elif msg_type == "SessionMetadata":
                # Cumulative session totals, emitted as the session winds down.
                print(
                    f"SessionMetadata: total_audio="
                    f"{getattr(data, 'total_audio_duration_ms', 0)} ms "
                    f"total_billable_chars="
                    f"{getattr(data, 'total_billable_character_count', 0)}"
                )
            elif msg_type == "Warning":
                print(
                    f"Warning [{getattr(data, 'code', '?')}]: "
                    f"{getattr(data, 'description', '')}",
                    file=sys.stderr,
                )
            elif msg_type == "Error":
                print(f"Error: {data}", file=sys.stderr)

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
            "Hello, this is Flux text-to-speech running on Amazon SageMaker.",
            "The model streams audio back turn by turn as the text arrives.",
            "This audio came through the Deepgram Python SDK transport layer.",
        ]

        for sentence in sentences:
            print(f'Sending: "{sentence}"')
            await connection.send_speak(SpeakV2Speak(type="Speak", text=sentence))

        # Flush closes the turn and forces out any buffered audio.
        await connection.send_flush()
        print("Waiting for audio...")

        await asyncio.sleep(10)

        close_sent = True
        await connection.send_close()
        await asyncio.sleep(2)
        listen_task.cancel()

    total = len(audio_chunks)
    total_bytes = sum(len(c) for c in audio_chunks)
    print(f"\nTotal audio chunks: {total}")
    print(f"Total audio bytes: {total_bytes}")

    if audio_chunks:
        pcm = b"".join(audio_chunks)
        channels, bits_per_sample = 1, 16
        byte_rate = SAMPLE_RATE * channels * bits_per_sample // 8
        with open(OUTPUT_FILE, "wb") as f:
            f.write(struct.pack("<4sI4s4sIHHIIHH4sI",
                b"RIFF", 36 + len(pcm), b"WAVE",
                b"fmt ", 16, 1, channels, SAMPLE_RATE, byte_rate,
                channels * bits_per_sample // 8, bits_per_sample,
                b"data", len(pcm)))
            f.write(pcm)
        print(f"Audio saved to {OUTPUT_FILE}")
    else:
        print("No audio received.")

    print("Done.")


asyncio.run(main())
