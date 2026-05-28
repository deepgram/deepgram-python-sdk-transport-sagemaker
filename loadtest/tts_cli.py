"""SageMaker TTS (Aura speak.v1) load test for the Python SDK.

Mirrors the JS dg-sdk-tts-loadtest.mjs script. Per-connection flow is
text-in, audio-out: send N sentences, send Flush, capture audio chunks
until the model emits a tail event (closed) or the await-flush timeout.

Usage:
    python -m loadtest.tts_cli <endpoint-name> --connections 400 \
        --region us-east-2 --transcripts-dir /tmp/tts-loadtest
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
import time
from pathlib import Path

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types import SpeakV1Text, SpeakV1Flush, SpeakV1Close
from deepgram_sagemaker import SageMakerConfig, SageMakerTransportFactory

logger = logging.getLogger(__name__)

DEFAULT_SENTENCES = [
    "Hello, this is a text-to-speech load test running on Amazon SageMaker.",
    "The Deepgram model is generating audio from text in real time.",
    "This audio is being streamed back through the Python SDK transport layer.",
]


class TtsConnection:
    def __init__(
        self,
        *,
        connection_id: int,
        client: AsyncDeepgramClient,
        model: str,
        encoding: str,
        await_flush_s: float,
        max_retries: int,
    ) -> None:
        self.connection_id = connection_id
        self._client = client
        self._model = model
        self._encoding = encoding
        self._await_flush_s = await_flush_s
        self._max_retries = max_retries
        self.stats = {
            "active": False,
            "errored": False,
            "audio_chunks": 0,
            "audio_bytes": 0,
            "flushed": False,
            "retry_count": 0,
            "start_time": 0.0,
            "end_time": 0.0,
            "error_messages": [],
        }

    def duration_seconds(self) -> float:
        if self.stats["start_time"] == 0.0:
            return 0.0
        end = self.stats["end_time"] or time.monotonic()
        return end - self.stats["start_time"]

    async def run(self) -> None:
        self.stats["start_time"] = time.monotonic()
        self.stats["active"] = True
        attempt = 0

        while True:
            try:
                async with self._client.speak.v1.connect(
                    model=self._model, encoding=self._encoding
                ) as connection:
                    errors: list[str] = []
                    close_event = asyncio.Event()

                    def on_message(data):
                        if isinstance(data, (bytes, bytearray)):
                            self.stats["audio_chunks"] += 1
                            self.stats["audio_bytes"] += len(data)
                            return
                        if data is None:
                            return
                        # The "Flushed" event surfaces via a Pydantic message
                        # type; getattr is the safe way to peek.
                        if getattr(data, "type", None) == "Flushed":
                            self.stats["flushed"] = True
                            # Mirror JS: exit the per-connection wait the
                            # moment we receive Flushed; no need to burn
                            # the full await-flush timeout.
                            close_event.set()

                    def on_error(err):
                        errors.append(str(err))
                        close_event.set()

                    def on_close(_):
                        close_event.set()

                    connection.on(EventType.MESSAGE, on_message)
                    connection.on(EventType.ERROR, on_error)
                    connection.on(EventType.CLOSE, on_close)

                    listen_task = asyncio.create_task(connection.start_listening())
                    await asyncio.sleep(0.5)

                    for sentence in DEFAULT_SENTENCES:
                        await connection.send_text(SpeakV1Text(text=sentence, type="Speak"))
                    await connection.send_flush(SpeakV1Flush(type="Flush"))

                    try:
                        await asyncio.wait_for(close_event.wait(), timeout=self._await_flush_s)
                    except asyncio.TimeoutError:
                        pass

                    try:
                        await connection.send_close(SpeakV1Close(type="Close"))
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)
                    listen_task.cancel()
                    try:
                        await listen_task
                    except (asyncio.CancelledError, Exception):
                        pass

                    if errors and self.stats["audio_chunks"] == 0:
                        raise RuntimeError("; ".join(errors))

                self.stats["end_time"] = time.monotonic()
                self.stats["active"] = False
                return

            except Exception as exc:
                msg = str(exc) or type(exc).__name__
                if attempt < self._max_retries:
                    attempt += 1
                    self.stats["retry_count"] += 1
                    backoff = min(1.0 * (2 ** (attempt - 1)), 30.0)
                    logger.warning(
                        "[Conn %d] retry %d/%d in %.1fs: %s",
                        self.connection_id, attempt, self._max_retries, backoff, msg[:100],
                    )
                    await asyncio.sleep(backoff)
                    continue
                self.stats["active"] = False
                self.stats["errored"] = True
                self.stats["error_messages"].append(msg)
                self.stats["end_time"] = time.monotonic()
                return


async def print_dashboard(conns, start_time, stop_event):
    while not stop_event.is_set():
        active = sum(1 for c in conns if c.stats["active"])
        errored = sum(1 for c in conns if c.stats["errored"])
        flushed = sum(1 for c in conns if c.stats["flushed"])
        chunks = sum(c.stats["audio_chunks"] for c in conns)
        elapsed = time.monotonic() - start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        sys.stderr.write(
            f"\r[{h:02d}:{m:02d}:{s:02d}] Active: {active}/{len(conns)} | "
            f"Errored: {errored} | Flushed: {flushed} | AudioChunks: {chunks}    "
        )
        sys.stderr.flush()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("endpoint_name")
    p.add_argument("--connections", "-c", type=int, default=1)
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--model", default="aura-2-atlas-en")
    p.add_argument("--encoding", default="linear16")
    p.add_argument("--await-flush", type=float, default=30.0)
    p.add_argument("--max-retries", type=int, default=10)
    p.add_argument("--transcripts-dir", default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def amain():
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    sm_config = SageMakerConfig(
        endpoint_name=args.endpoint_name,
        region=args.region,
        max_retries=100,
        max_backoff=30.0,
        retry_budget=3600.0,
    )
    factory = SageMakerTransportFactory(config=sm_config)
    client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

    sys.stderr.write("=== Deepgram SDK SageMaker TTS Load Test (Python) ===\n")
    sys.stderr.write(f"Endpoint:       {args.endpoint_name}\n")
    sys.stderr.write(f"Connections:    {args.connections}\n")
    sys.stderr.write(f"Region:         {args.region}\n")
    sys.stderr.write(f"Model:          {args.model}\n")
    sys.stderr.write(f"Encoding:       {args.encoding}\n\n")

    conns = [
        TtsConnection(
            connection_id=i + 1,
            client=client,
            model=args.model,
            encoding=args.encoding,
            await_flush_s=args.await_flush,
            max_retries=args.max_retries,
        )
        for i in range(args.connections)
    ]

    start = time.monotonic()
    stop = asyncio.Event()
    dashboard_task = asyncio.create_task(print_dashboard(conns, start, stop))

    await asyncio.gather(*(c.run() for c in conns))
    stop.set()
    await dashboard_task

    wall_time = time.monotonic() - start
    sys.stderr.write("\n\n=== TTS SUMMARY ===\n")
    successful = sum(1 for c in conns if not c.stats["errored"])
    errored = sum(1 for c in conns if c.stats["errored"])
    flushed = sum(1 for c in conns if c.stats["flushed"])
    total_chunks = sum(c.stats["audio_chunks"] for c in conns)
    total_bytes = sum(c.stats["audio_bytes"] for c in conns)
    total_retries = sum(c.stats["retry_count"] for c in conns)
    got_audio = sum(1 for c in conns if c.stats["audio_chunks"] > 0 and not c.stats["errored"])

    sys.stderr.write(f"Total connections:  {args.connections}\n")
    sys.stderr.write(f"Successful:         {successful}\n")
    sys.stderr.write(f"Errored:            {errored}\n")
    sys.stderr.write(f"Flushed (got Flushed event): {flushed}\n")
    sys.stderr.write(f"Got audio (>=1 chunk): {got_audio}\n")
    sys.stderr.write(f"Total audio chunks: {total_chunks}\n")
    sys.stderr.write(f"Total audio bytes:  {total_bytes}\n")
    sys.stderr.write(f"Total retries:      {total_retries}\n")
    sys.stderr.write(f"Wall time:          {wall_time:.2f}s\n")

    dump_dir = args.transcripts_dir
    if dump_dir and dump_dir != "-":
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        csv_path = Path(dump_dir) / "tts-summary.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["conn_id", "errored", "audio_chunks", "audio_bytes", "flushed", "retries", "duration_s", "err_msg"])
            for c in conns:
                err_msg = (" | ".join(c.stats["error_messages"]))[:300].replace("\n", " ").replace(",", ";")
                w.writerow([
                    c.connection_id,
                    c.stats["errored"],
                    c.stats["audio_chunks"],
                    c.stats["audio_bytes"],
                    c.stats["flushed"],
                    c.stats["retry_count"],
                    f"{c.duration_seconds():.2f}",
                    err_msg,
                ])
        sys.stderr.write(f"Wrote per-connection summary to {csv_path}\n")

    if errored > 0:
        sys.stderr.write("\n--- Errors ---\n")
        counts: dict[str, int] = {}
        for c in conns:
            for m in c.stats["error_messages"]:
                key = (m or "")[:120]
                counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            sys.stderr.write(f"  [{count}x] {key}\n")


if __name__ == "__main__":
    asyncio.run(amain())
