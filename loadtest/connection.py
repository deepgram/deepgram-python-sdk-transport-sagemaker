"""A single streaming connection through the Deepgram SDK + SageMaker transport.

Mirrors the Java SdkStreamingConnection load-test helper. Each instance owns
its own SageMakerTransportFactory (and therefore its own AWS HTTP/2 client)
to ensure stream isolation.
"""

from __future__ import annotations

import asyncio
import logging
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types import ListenV1CloseStream, ListenV1Results
from deepgram.listen.v2.types import ListenV2CloseStream, ListenV2TurnInfo

logger = logging.getLogger(__name__)

# Chunk size matches the Java load test (8 KB).
CHUNK_SIZE = 8192

# If real-time pacing drifts more than this far behind, reset the baseline
# so the loop doesn't burst-send catch-up audio at line speed (which overruns
# the model and truncates tail-end transcripts after CloseStream).
PACING_DRIFT_THRESHOLD_S = 1.0

# Default-retryable, narrow terminal allowlist. Matches the JS harness's
# isRetryable() approach: the retry budget is the safety net, not the
# classifier. We only give up immediately on 4xx-coded caller-side rejections
# (auth, validation, missing endpoint) where retrying would just waste budget.
_TERMINAL_TOKENS = (
    "AccessDeniedException",
    "UnrecognizedClientException",
    "ValidationException",
    "InvalidEndpointException",
    "ResourceNotFoundException",
    "EndpointNotFoundException",
)


def _is_retryable(message: str) -> bool:
    if not message:
        return True
    return not any(token in message for token in _TERMINAL_TOKENS)


@dataclass
class ConnectionStats:
    """Snapshot fields the dashboard and summary read from each connection."""

    connection_id: int
    chunk_count: int = 0
    transcript_count: int = 0
    retry_count: int = 0
    errored: bool = False
    error_messages: list[str] = field(default_factory=list)
    final_transcripts: list[str] = field(default_factory=list)
    active: bool = False
    stopped: bool = False
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        end = self.end_time if self.end_time != 0.0 else time.monotonic()
        return end - self.start_time


class SdkStreamingConnection:
    """One streaming connection through DeepgramClient -> SageMaker transport.

    Each connection creates its own SageMakerTransportFactory (and therefore
    its own underlying AWS HTTP/2 client) so streams are fully isolated.
    """

    def __init__(
        self,
        *,
        connection_id: int,
        client: AsyncDeepgramClient,
        service: str,  # "listen.v1" or "listen.v2"
        wav_path: Path,
        connect_options: dict,
        loop: bool,
        max_retries: int,
        await_final_results_s: float,
        on_first_pass_complete,
    ) -> None:
        self.stats = ConnectionStats(connection_id=connection_id)
        self._client = client
        self._service = service or "listen.v1"
        self._wav_path = wav_path
        self._connect_options = connect_options
        self._loop = loop
        self._max_retries = max_retries
        self._await_final_results_s = await_final_results_s
        self._on_first_pass_complete = on_first_pass_complete
        self._first_pass_fired = False
        self._stop_event = asyncio.Event()
        # Cached WAV metadata so each loop iteration doesn't re-open the file.
        with wave.open(str(wav_path), "rb") as wf:
            self._sample_rate = wf.getframerate()
            self._channels = wf.getnchannels()
            self._sample_width = wf.getsampwidth()
            self._frames = wf.getnframes()
        self._bytes_per_frame = self._channels * self._sample_width
        self._frames_per_chunk = max(1, CHUNK_SIZE // self._bytes_per_frame)
        self._chunk_duration_s = self._frames_per_chunk / self._sample_rate

    @property
    def connection_id(self) -> int:
        return self.stats.connection_id

    def stop(self) -> None:
        self.stats.stopped = True
        self._stop_event.set()

    async def run(self) -> None:
        """Connect, stream the WAV, capture transcripts, retry transient errors."""
        self.stats.start_time = time.monotonic()
        self.stats.active = True
        attempt = 0

        while not self.stats.stopped:
            try:
                ctx = (
                    self._client.listen.v2.connect(**self._connect_options)
                    if self._service == "listen.v2"
                    else self._client.listen.v1.connect(**self._connect_options)
                )
                async with ctx as connection:
                    stream_errored = False
                    stream_errors: list[str] = []
                    listen_done = asyncio.Event()

                    def on_open(_):
                        logger.debug("[Conn %d] open", self.connection_id)

                    def on_message(message):
                        import os
                        if os.environ.get("DG_LOADTEST_TRACE_MSG"):
                            logger.info(
                                "[Conn %d] msg type=%s event=%s",
                                self.connection_id,
                                type(message).__name__,
                                getattr(message, "event", None),
                            )
                        # listen.v2 (Flux): TurnInfo events arrive on various
                        # types (ListenV2TurnInfo, or sometimes deserialized as
                        # ListenV2Connected when the discriminator misroutes).
                        # Dispatch on the `event` attribute rather than the
                        # wrapper class to capture both shapes.
                        if self._service == "listen.v2":
                            event_name = getattr(message, "event", None)
                            if event_name != "EndOfTurn":
                                return
                            transcript = getattr(message, "transcript", "") or ""
                            if transcript:
                                self.stats.transcript_count += 1
                                self.stats.final_transcripts.append(transcript)
                                logger.info(
                                    "[Conn %d] #%d: %s",
                                    self.connection_id,
                                    self.stats.transcript_count,
                                    transcript,
                                )
                            return
                        # listen.v1: final transcript on is_final=True Results.
                        if isinstance(message, ListenV1Results):
                            is_final = bool(getattr(message, "is_final", False))
                            channel = getattr(message, "channel", None)
                            if not is_final or channel is None:
                                return
                            alts = getattr(channel, "alternatives", None) or []
                            if not alts:
                                return
                            transcript = getattr(alts[0], "transcript", "") or ""
                            if transcript:
                                self.stats.transcript_count += 1
                                self.stats.final_transcripts.append(transcript)
                                logger.info(
                                    "[Conn %d] #%d: %s",
                                    self.connection_id,
                                    self.stats.transcript_count,
                                    transcript,
                                )

                    def on_error(err):
                        nonlocal stream_errored
                        stream_errored = True
                        stream_errors.append(str(err))
                        logger.error("[Conn %d] error: %s", self.connection_id, err)
                        listen_done.set()

                    def on_close(_):
                        logger.debug("[Conn %d] close", self.connection_id)
                        listen_done.set()

                    connection.on(EventType.OPEN, on_open)
                    connection.on(EventType.MESSAGE, on_message)
                    connection.on(EventType.ERROR, on_error)
                    connection.on(EventType.CLOSE, on_close)

                    listen_task = asyncio.create_task(connection.start_listening())

                    try:
                        await self._stream_audio(connection)

                        if not self.stats.stopped:
                            close_msg = (
                                ListenV2CloseStream(type="CloseStream")
                                if self._service == "listen.v2"
                                else ListenV1CloseStream(type="CloseStream")
                            )
                            await connection.send_close_stream(close_msg)

                        try:
                            await asyncio.wait_for(
                                listen_done.wait(), timeout=self._await_final_results_s
                            )
                        except asyncio.TimeoutError:
                            pass
                    finally:
                        listen_task.cancel()
                        try:
                            await listen_task
                        except (asyncio.CancelledError, Exception):
                            pass

                    if stream_errored:
                        # If we already captured at least one transcript, the
                        # error is a tail-end "model idle after CloseStream"
                        # condition -- the audio was processed, the transcript
                        # is good, no point retrying. Treat as success.
                        if self.stats.transcript_count > 0:
                            logger.info(
                                "[Conn %d] post-transcript error suppressed: %s",
                                self.connection_id,
                                stream_errors[0][:100] if stream_errors else "",
                            )
                        else:
                            raise RuntimeError("; ".join(stream_errors))

                # Connection completed cleanly.
                self.stats.end_time = time.monotonic()

                if not self._loop or self.stats.stopped:
                    self.stats.active = False
                    return
                attempt = 0  # reset on a clean pass when looping

            except Exception as exc:  # noqa: BLE001 -- classified below
                if self.stats.stopped:
                    self.stats.active = False
                    return
                msg = str(exc) or type(exc).__name__
                if _is_retryable(msg) and attempt < self._max_retries:
                    attempt += 1
                    self.stats.retry_count += 1
                    backoff = min(1.0 * (2 ** (attempt - 1)), 30.0)
                    logger.warning(
                        "[Conn %d] Retryable error (attempt %d/%d), retrying in %.1fs: %s",
                        self.connection_id,
                        attempt,
                        self._max_retries,
                        backoff,
                        msg[:100],
                    )
                    try:
                        await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                        # If wait returned without timing out, stop was requested.
                        self.stats.active = False
                        return
                    except asyncio.TimeoutError:
                        continue

                self.stats.active = False
                self.stats.errored = True
                self.stats.error_messages.append(msg)
                self.stats.end_time = time.monotonic()
                if attempt >= self._max_retries:
                    logger.error(
                        "[Conn %d] Retries exhausted (%d/%d): %s",
                        self.connection_id,
                        attempt,
                        self._max_retries,
                        msg[:100],
                    )
                else:
                    logger.error(
                        "[Conn %d] Non-retryable error: %s", self.connection_id, msg[:100]
                    )
                return

        self.stats.active = False

    async def _stream_audio(self, connection) -> None:
        """Stream the WAV in real-time, with a drift-reset guard for retry storms.

        listen.v1 takes audio format via query params (encoding/sample_rate/
        channels), so we send raw PCM (wave.readframes strips the header).
        Flux V2 auto-detects format from the WAV/RIFF header, so we stream
        the entire file from byte 0 -- sending raw PCM fails with
        UNPARSABLE_CLIENT_MESSAGE.
        """
        total_chunks = self.stats.chunk_count
        play_count = 0
        stream_start = time.monotonic()
        chunks_at_pacing_start = total_chunks

        # For Flux V2: pre-read the full file (including header) once, then
        # iterate fixed-size byte slices each pass.
        full_wav_bytes = (
            self._wav_path.read_bytes() if self._service == "listen.v2" else b""
        )

        while not self.stats.stopped:
            play_count += 1
            logger.debug("[Conn %d] streaming WAV (pass %d)...", self.connection_id, play_count)

            if self._service == "listen.v2":
                pos = 0
                while not self.stats.stopped and pos < len(full_wav_bytes):
                    chunk = full_wav_bytes[pos : pos + CHUNK_SIZE]
                    pos += CHUNK_SIZE
                    await connection.send_media(chunk)
                    total_chunks += 1
                    self.stats.chunk_count = total_chunks
                    elapsed = time.monotonic() - stream_start
                    target = (total_chunks - chunks_at_pacing_start) * self._chunk_duration_s
                    sleep_s = target - elapsed
                    if sleep_s < -PACING_DRIFT_THRESHOLD_S:
                        logger.info(
                            "[Conn %d] pacing drift %.0fms detected -- resetting baseline",
                            self.connection_id,
                            -sleep_s * 1000,
                        )
                        stream_start = time.monotonic()
                        chunks_at_pacing_start = total_chunks
                        sleep_s = 0.0
                    if sleep_s > 0:
                        try:
                            await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_s)
                            return
                        except asyncio.TimeoutError:
                            pass

                if not self.stats.stopped and not self._first_pass_fired:
                    self._first_pass_fired = True
                    if self._on_first_pass_complete is not None:
                        self._on_first_pass_complete()
                if not self._loop:
                    break
                continue

            with wave.open(str(self._wav_path), "rb") as wf:
                while not self.stats.stopped:
                    chunk = wf.readframes(self._frames_per_chunk)
                    if not chunk:
                        break

                    await connection.send_media(chunk)
                    total_chunks += 1
                    self.stats.chunk_count = total_chunks

                    # Real-time pacing with drift reset. If send_media blocked
                    # for a long time (SDK retry storm, throttle backoff, etc.),
                    # the naive pacing math would burst-send catch-up audio at
                    # line speed -- overruns the model's input buffer and
                    # inflates tail-end WER. Reset the baseline when drift
                    # exceeds the threshold.
                    elapsed = time.monotonic() - stream_start
                    target = (total_chunks - chunks_at_pacing_start) * self._chunk_duration_s
                    sleep_s = target - elapsed
                    if sleep_s < -PACING_DRIFT_THRESHOLD_S:
                        logger.info(
                            "[Conn %d] pacing drift %.0fms detected -- resetting baseline",
                            self.connection_id,
                            -sleep_s * 1000,
                        )
                        stream_start = time.monotonic()
                        chunks_at_pacing_start = total_chunks
                        sleep_s = 0.0
                    if sleep_s > 0:
                        try:
                            await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_s)
                            # Stop signalled mid-sleep.
                            return
                        except asyncio.TimeoutError:
                            pass

            # One full pass of the file completed (reached EOF, not stopped mid-file).
            if not self.stats.stopped and not self._first_pass_fired:
                self._first_pass_fired = True
                if self._on_first_pass_complete is not None:
                    self._on_first_pass_complete()

            if not self._loop:
                break

        logger.info(
            "[Conn %d] audio streaming done (%d passes, %d chunks)",
            self.connection_id,
            play_count,
            total_chunks,
        )


def build_connect_options(
    *,
    service: str,
    model: str,
    sample_rate: int,
    channels: int,
    interim_results: bool,
) -> dict:
    """Build the kwargs dict for client.listen.{v1,v2}.connect()."""
    if service == "listen.v2":
        # Flux V2: model is the only required arg; audio format is inferred
        # from the WAV/RIFF header in the byte stream. Caller can override
        # via --model, otherwise default to the multi-language variant the
        # deepgram-streaming-flux-multi model package serves.
        return {
            "model": model if (model and model != "nova-3") else "flux-general-multi"
        }
    return {
        "model": model,
        "encoding": "linear16",
        "sample_rate": sample_rate,
        "channels": channels,
        "interim_results": interim_results,
    }
