"""Live microphone transcription via SageMaker using Flux (V2 Listen).

Captures audio from the system microphone and streams to a Flux model on
SageMaker. Uses turn-based transcription. Press Ctrl+C to stop.

Usage:
    export SAGEMAKER_ENDPOINT=my-deepgram-flux-endpoint
    export AWS_REGION=us-east-2
    python examples/sagemaker_live_mic_flux.py
"""

import asyncio
import os
import sys
import signal

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram_sagemaker import SageMakerTransportFactory

ENDPOINT = os.getenv("SAGEMAKER_ENDPOINT", "deepgram-flux")
REGION = os.getenv("AWS_REGION", "us-west-2")

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 8192


async def main():
    try:
        import pyaudio
    except ImportError:
        print("pyaudio is required: pip install pyaudio")
        print("On macOS: brew install portaudio && pip install pyaudio")
        sys.exit(1)

    factory = SageMakerTransportFactory(endpoint_name=ENDPOINT, region=REGION)
    client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

    print("Live Microphone Transcription via SageMaker (Flux V2)")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Region:   {REGION}")
    print(f"Audio:    {SAMPLE_RATE} Hz, 16-bit, mono")
    print()

    running = True

    def stop(*_):
        nonlocal running
        print("\nStopping...")
        running = False

    signal.signal(signal.SIGINT, stop)

    async with client.listen.v2.connect(
        model="flux-general-en",
        encoding="linear16",
        sample_rate=str(SAMPLE_RATE),
    ) as connection:

        def on_turn_info(turn_info):
            try:
                transcript = turn_info.transcript
                event = turn_info.event
                if transcript:
                    if str(event) == "EndOfTurn":
                        print(f"\033[2K\r{transcript}")
                    else:
                        print(f"\033[2K\r  ... {transcript}", end="", flush=True)
            except Exception:
                pass

        def on_error(error):
            print(f"\nError: {error}", file=sys.stderr)

        connection.on(EventType.MESSAGE, on_turn_info)
        connection.on(EventType.ERROR, on_error)

        listen_task = asyncio.create_task(connection.start_listening())
        await asyncio.sleep(0.5)

        print("Listening... speak into your microphone. Press Ctrl+C to stop.\n")

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )

        while running:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            await connection.send_media(data)
            await asyncio.sleep(0)

        stream.stop_stream()
        stream.close()
        pa.terminate()

        await connection.send_close_stream()
        await asyncio.sleep(2)
        listen_task.cancel()

    print("Done.")


asyncio.run(main())
