"""SageMaker transport for the Deepgram Python SDK.

Uses AWS SageMaker's HTTP/2 bidirectional streaming API as an alternative to
WebSocket, allowing transparent switching between Deepgram Cloud and Deepgram
on SageMaker.

**Async-only** -- the underlying ``sagemaker-runtime-http2`` library is
entirely async, so this transport implements the ``AsyncTransport`` protocol
and must be used with ``AsyncDeepgramClient``.

Requirements::

    pip install deepgram-sagemaker

Usage::

    from deepgram import AsyncDeepgramClient
    from deepgram_sagemaker import SageMakerTransportFactory

    factory = SageMakerTransportFactory(
        endpoint_name="my-deepgram-endpoint",
        region="us-west-2",
    )
    client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

    async with client.listen.v1.connect(model="nova-3") as connection:
        connection.on(EventType.MESSAGE, handler)
        await connection.start_listening()

For burst-tuned timeouts and retry behavior, build and pass a
:class:`~deepgram_sagemaker.config.SageMakerConfig`::

    from deepgram_sagemaker import SageMakerConfig, SageMakerTransportFactory

    config = SageMakerConfig(
        endpoint_name="my-deepgram-endpoint",
        region="us-east-1",
        connection_timeout=5.0,        # tighten for fail-fast pipelines
        connection_acquire_timeout=15.0,
    )
    factory = SageMakerTransportFactory(config=config)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from collections import deque
from typing import Any, Callable
from urllib.parse import urlparse

from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
)
from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
from smithy_aws_core.identity import EnvironmentCredentialsResolver
from smithy_http.interfaces import HTTPRequestConfiguration

from .config import SageMakerConfig

logger = logging.getLogger(__name__)


_RETRYABLE = "RETRYABLE"
_TERMINAL = "TERMINAL"


def compute_backoff(
    initial_s: float,
    max_s: float,
    multiplier: float,
    attempt: int,
    random_uniform: Callable[[float, float], float] | None = None,
) -> float:
    """Full-jitter exponential backoff. Pure function for testability.

    Without jitter, N streams failing simultaneously all compute the same
    exponential delay and retry in lockstep, hammering the endpoint in waves.
    Full jitter -- uniform in ``[initial_s, ceiling]`` -- spreads the retry
    load continuously over the backoff window. See AWS Architecture Blog
    "Exponential Backoff and Jitter".
    """
    try:
        scaled = initial_s * (multiplier ** attempt)
    except OverflowError:
        scaled = float("inf")
    if scaled > max_s or scaled != scaled:  # NaN guard
        ceiling = max_s
    else:
        ceiling = max(initial_s, scaled)
    if ceiling <= initial_s:
        return ceiling
    rng = random_uniform if random_uniform is not None else random.uniform
    return rng(initial_s, ceiling)


def classify(error: BaseException) -> str:
    """Classify an exception as RETRYABLE (transient) or TERMINAL.

    Default is RETRYABLE -- the retry budget is the safety net, not the
    classifier. The narrow set of TERMINAL errors is caller-side rejections
    from AWS: 4xx status codes (other than 429 throttling and 424 Failed
    Dependency, which under burst load we've observed SageMaker emit for
    transient upstream model errors). Anything throttling-coded is RETRYABLE
    regardless of status. Causes are walked; unknown cause-chain types
    default to RETRYABLE.
    """
    seen: set[int] = set()
    cur: BaseException | None = error
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status = _status_code(cur)
        code = _error_code(cur)
        if code and "throttl" in code.lower():
            return _RETRYABLE
        if status is not None and 400 <= status < 500 and status not in (429, 424):
            return _TERMINAL
        cur = cur.__cause__ or cur.__context__
    return _RETRYABLE


def _status_code(error: BaseException) -> int | None:
    for attr in ("status_code", "http_status_code", "statusCode"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        meta = response.get("ResponseMetadata") or {}
        status = meta.get("HTTPStatusCode") if isinstance(meta, dict) else None
        if isinstance(status, int):
            return status
    return None


def _error_code(error: BaseException) -> str | None:
    for attr in ("error_code", "code", "errorCode"):
        value = getattr(error, attr, None)
        if isinstance(value, str):
            return value
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        err = response.get("Error") or {}
        code = err.get("Code") if isinstance(err, dict) else None
        if isinstance(code, str):
            return code
    name = type(error).__name__
    if "Throttl" in name or "Throttle" in name:
        return name
    return None


class _BufferedEvent:
    """Replay-buffer entry. Keeps the byte length alongside the event so the
    cap-and-evict path doesn't have to peek inside the payload wrapper."""

    __slots__ = ("event", "bytes_len")

    def __init__(self, event: Any, bytes_len: int) -> None:
        self.event = event
        self.bytes_len = bytes_len


class SageMakerTransport:
    """SageMaker BiDi streaming transport satisfying the ``AsyncTransport`` protocol.

    Connection is established lazily on the first ``send()`` or ``recv()``
    call, since the transport is constructed synchronously by the SDK's shim
    layer. The transport internally retries transient AWS errors (throttling,
    pool-acquire timeouts, transient connect/timeout failures) with jittered
    exponential backoff, bounded by :attr:`SageMakerConfig.max_retries` and
    :attr:`SageMakerConfig.retry_budget`. Terminal errors (auth, validation)
    surface to the caller immediately.

    Parameters
    ----------
    config : SageMakerConfig
        Endpoint, region, timeouts, and retry tuning.
    invocation_path : str
        Model invocation path (e.g. ``"v1/listen"``). Extracted from the
        WebSocket URL by :class:`SageMakerTransportFactory`.
    query_string : str
        URL-encoded query parameters. Extracted from the WebSocket URL by
        :class:`SageMakerTransportFactory`.
    """

    def __init__(
        self,
        config: SageMakerConfig,
        invocation_path: str,
        query_string: str,
    ) -> None:
        self._config = config
        self.endpoint_name = config.endpoint_name
        self.region = config.region
        self.invocation_path = invocation_path
        self.query_string = query_string
        self._stream: Any = None
        self._output_stream: Any = None
        self._connected = False
        self._closed = False
        self._close_sent = False
        self._connect_lock: asyncio.Lock | None = None

        # Events queued before the next connection lands (e.g. while a
        # reconnect is in flight). Drained onto the new input_stream during
        # _do_connect, in order.
        self._pending: deque[Any] = deque()

        # Replay buffer: events sent on the current stream that AWS hasn't
        # acked yet (no payload part received since they were sent). On
        # internal reset, the next _do_connect drains this buffer onto the
        # new stream so audio sent on the rejected stream isn't lost. Trimmed
        # in _handle_payload_part_ack (a transcript proves AWS consumed prior
        # audio) and capped at config.max_replay_buffer_bytes with FIFO
        # eviction so unbounded throttle storms can't OOM.
        self._replay: deque[_BufferedEvent] = deque()
        self._replay_bytes = 0

        # Retry budget tracking. Reset to 0 once real downstream data flows
        # back to the application -- NOT on connect success. Connect succeeds
        # in TLS+HTTP/2 setup terms even when the bidi-stream request will be
        # throttled milliseconds later.
        self._retry_attempt = 0
        self._retry_window_start = 0.0
        # Earliest wall-clock at which the next connect attempt is allowed to
        # proceed. Set by recv/send error paths so post-subscription throttles
        # still pace the next attempt.
        self._retry_not_before = 0.0

        self._setup_credentials()

    def _setup_credentials(self) -> None:
        """Resolve AWS credentials via boto3 (if available) into env vars.

        The ``EnvironmentCredentialsResolver`` used by the SageMaker HTTP/2
        client reads ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and
        optionally ``AWS_SESSION_TOKEN`` from the environment. Use boto3's
        credential chain (env vars, shared credentials file, IAM role, etc.)
        and write the resolved values back so the resolver can find them.
        """
        try:
            import boto3
        except ImportError:
            return
        try:
            session = boto3.Session(region_name=self.region)
            creds = session.get_credentials()
            if creds is None:
                return
            frozen = creds.get_frozen_credentials()
            os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
            if frozen.token:
                os.environ["AWS_SESSION_TOKEN"] = frozen.token
        except Exception as exc:
            logger.debug("Could not load boto3 credentials: %s", exc)

    async def _ensure_connected(self) -> None:
        """Lazily establish the SageMaker BiDi stream, with retry/backoff.

        Internally retries transient AWS errors (throttling, pool exhaustion,
        transient connect/timeout) bounded by ``config.max_retries`` and
        ``config.retry_budget``. Terminal errors and budget exhaustion bubble
        out and surface to the application.
        """
        if self._connected:
            return

        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()

        async with self._connect_lock:
            if self._connected:
                return

            if self._retry_window_start == 0.0:
                self._retry_window_start = time.monotonic()

            last_error: BaseException | None = None
            while True:
                # Honor backoff scheduled by error handlers so post-subscription
                # throttles pace the next attempt.
                now = time.monotonic()
                if self._retry_not_before > now:
                    sleep_s = self._retry_not_before - now
                    logger.info(
                        "ensureConnected: honoring scheduled backoff (%.3fs) before next attempt",
                        sleep_s,
                    )
                    await asyncio.sleep(sleep_s)

                attempt_before = self._retry_attempt
                try:
                    await self._do_connect()
                    # Connect succeeded -- but DO NOT reset retry counters
                    # here. Connect success only proves TLS+HTTP/2 setup; the
                    # actual bidi-stream request can still be throttled. Reset
                    # happens in _handle_payload_part_ack when real data flows.
                    logger.info(
                        "ensureConnected: connect SUCCEEDED (attempt=%d -- counter NOT reset; "
                        "waits for first real payload before resetting)",
                        attempt_before,
                    )
                    self._connected = True
                    return
                except BaseException as exc:  # noqa: BLE001 -- classify below
                    last_error = exc
                    cls = classify(exc)
                    attempt = self._retry_attempt
                    elapsed = time.monotonic() - self._retry_window_start
                    budget_left = (
                        attempt < self._config.max_retries
                        and elapsed < self._config.retry_budget
                    )
                    logger.info(
                        "ensureConnected: connect FAILED class=%s attempt=%d/%d "
                        "elapsed=%.3fs/%.3fs budgetLeft=%s err=%s",
                        cls,
                        attempt,
                        self._config.max_retries,
                        elapsed,
                        self._config.retry_budget,
                        budget_left,
                        _summarize(exc),
                    )
                    if cls == _TERMINAL or not budget_left:
                        logger.warning(
                            "ensureConnected: SURFACING (class=%s budgetLeft=%s) err=%s",
                            cls,
                            budget_left,
                            _summarize(exc),
                        )
                        raise
                    backoff = self._compute_backoff(attempt)
                    logger.info(
                        "ensureConnected: backoff=%.3fs before retry attempt %d",
                        backoff,
                        attempt + 1,
                    )
                    self._retry_attempt += 1
                    await asyncio.sleep(backoff)

            # Unreachable -- loop returns or raises.
            del last_error

    async def _do_connect(self) -> None:
        """Single connect attempt -- invokes the bidi stream and waits for the
        output stream to become available."""

        logger.info(
            "Connecting to SageMaker endpoint: %s in %s", self.endpoint_name, self.region
        )

        # The underlying smithy HTTP/2 stack doesn't yet expose dedicated
        # connect / acquire timeouts the way Java Netty does; the closest knob
        # is request-level read_timeout. Map subscription_timeout to that --
        # all three configured timeouts apply to roughly "how long are we
        # willing to wait before declaring this attempt failed."
        config = Config(
            endpoint_uri=f"https://runtime.sagemaker.{self.region}.amazonaws.com:8443",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
            http_request_config=HTTPRequestConfiguration(
                read_timeout=self._config.connection_timeout,
            ),
        )
        client = SageMakerRuntimeHTTP2Client(config=config)

        stream_input = InvokeEndpointWithBidirectionalStreamInput(
            endpoint_name=self.endpoint_name,
            model_invocation_path=self.invocation_path,
            model_query_string=self.query_string,
        )

        self._stream = await client.invoke_endpoint_with_bidirectional_stream(
            stream_input
        )

        output = await asyncio.wait_for(
            self._stream.await_output(), timeout=self._config.subscription_timeout
        )
        self._output_stream = output[1]

        # Drain replay buffer onto the new stream so audio sent on a prior
        # rejected attempt isn't lost.
        if self._replay:
            count = len(self._replay)
            logger.info(
                "do_connect: replaying %d buffered events (%d bytes) onto new stream",
                count,
                self._replay_bytes,
            )
            for be in list(self._replay):
                await self._stream.input_stream.send(be.event)

        # Drain anything queued before this attempt (e.g. sends issued while
        # _connect_lock was held by a previous attempt that failed).
        while self._pending:
            event = self._pending.popleft()
            await self._stream.input_stream.send(event)

        logger.info("Connected to SageMaker endpoint: %s", self.endpoint_name)

    def _compute_backoff(self, attempt: int) -> float:
        return compute_backoff(
            self._config.initial_backoff,
            self._config.max_backoff,
            self._config.backoff_multiplier,
            attempt,
        )

    def _buffer_for_replay(self, event: Any, byte_len: int) -> None:
        """Append event to the replay buffer with FIFO eviction at the byte cap."""
        cap = self._config.max_replay_buffer_bytes
        if cap == 0:
            return
        self._replay.append(_BufferedEvent(event, byte_len))
        self._replay_bytes += byte_len
        while self._replay_bytes > cap and self._replay:
            dropped = self._replay.popleft()
            self._replay_bytes -= dropped.bytes_len

    def _clear_replay(self) -> None:
        self._replay.clear()
        self._replay_bytes = 0

    def _handle_retryable_error(self, exc: BaseException) -> bool:
        """Process a runtime error and decide whether to reset the stream.

        Returns True if the error was classified as RETRYABLE and budget
        remains -- caller should drop the current stream and let the next
        operation reconnect. Returns False if the error is TERMINAL or budget
        exhausted -- caller should re-raise.
        """
        if self._close_sent:
            logger.info("handle error: closeSent=True -> treating as normal close")
            return False

        cls = classify(exc)
        attempt = self._retry_attempt
        elapsed = (
            0.0
            if self._retry_window_start == 0.0
            else time.monotonic() - self._retry_window_start
        )
        budget_left = (
            attempt < self._config.max_retries
            and elapsed < self._config.retry_budget
        )
        logger.info(
            "handle error: class=%s attempt=%d/%d elapsed=%.3fs/%.3fs budgetLeft=%s err=%s",
            cls,
            attempt,
            self._config.max_retries,
            elapsed,
            self._config.retry_budget,
            budget_left,
            _summarize(exc),
        )
        if cls != _RETRYABLE or not budget_left:
            logger.warning(
                "handle error: SURFACING (class=%s budgetLeft=%s) err=%s",
                cls,
                budget_left,
                _summarize(exc),
            )
            return False

        # Internal reset: drop current stream, schedule backoff so the next
        # _ensure_connected pauses rather than immediately hammering AWS.
        if self._retry_window_start == 0.0:
            self._retry_window_start = time.monotonic()
        attempt_for_backoff = self._retry_attempt
        self._retry_attempt += 1
        backoff = self._compute_backoff(attempt_for_backoff)
        self._retry_not_before = time.monotonic() + backoff
        logger.info(
            "handle error: RETRYABLE -> internal reset, attempt %d scheduled backoff %.3fs",
            attempt_for_backoff + 1,
            backoff,
        )
        self._connected = False
        self._output_stream = None
        return True

    async def send(self, data: Any) -> None:
        """Send text, bytes, or dict data to SageMaker.

        The SDK calls this with:
        - ``bytes`` for audio media
        - ``str`` for JSON control messages
        """
        if self._closed:
            raise RuntimeError("Transport is closed")

        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
            data_type: str | None = None
        elif isinstance(data, str):
            raw = data.encode("utf-8")
            data_type = "UTF8"
        elif isinstance(data, dict):
            raw = json.dumps(data).encode("utf-8")
            data_type = "UTF8"
        else:
            raw = str(data).encode("utf-8")
            data_type = None

        payload = (
            RequestPayloadPart(bytes_=raw, data_type=data_type)
            if data_type
            else RequestPayloadPart(bytes_=raw)
        )
        event = RequestStreamEventPayloadPart(value=payload)

        # Track close signals so we treat the model's idle-timeout as a
        # normal close rather than an error. Match both the SDK's default
        # `json.dumps` output (`"type": "CloseStream"`, with space) and the
        # whitespace-stripped form so this stays robust if the serializer
        # changes. Covers listen.v1/v2's `CloseStream`/`Finalize` and
        # speak.v1's `Close` (TTS).
        if isinstance(data, str):
            normalized = data.replace(" ", "")
            if (
                '"type":"CloseStream"' in normalized
                or '"type":"Finalize"' in normalized
                or '"type":"Close"' in normalized
            ):
                self._close_sent = True

        while True:
            await self._ensure_connected()
            try:
                await self._stream.input_stream.send(event)
                self._buffer_for_replay(event, len(raw))
                return
            except BaseException as exc:  # noqa: BLE001 -- classified below
                if not self._handle_retryable_error(exc):
                    raise

    async def recv(self) -> Any:
        """Receive the next message from SageMaker.

        Returns a decoded UTF-8 string when possible (so the SDK can
        JSON-parse it), or raw bytes for binary data. Returns ``None`` when
        the stream ends.
        """
        if self._closed:
            return None

        while True:
            await self._ensure_connected()
            try:
                result = await self._output_stream.receive()
            except BaseException as exc:  # noqa: BLE001 -- classified below
                if self._handle_retryable_error(exc):
                    continue
                raise

            if result is None:
                return None

            if result.value and result.value.bytes_:
                raw = result.value.bytes_
                # JSON messages start with '{"' -- return as string for the
                # SDK to parse. Everything else is binary (e.g. TTS audio).
                if len(raw) > 1 and raw[0:1] == b'{' and raw[1:2] == b'"':
                    text = raw.decode("utf-8")
                    self._handle_payload_part_ack(raw, text)
                    return text
                self._handle_payload_part_ack(raw, None)
                return raw

            return None

    def _handle_payload_part_ack(self, raw: bytes, text: str | None) -> None:
        """Reset retry counters and trim the replay buffer when the model has
        produced real downstream content.

        Rather than enumerate every downstream message type per Deepgram
        product, invert the check: assume ANY payload counts as an ack EXCEPT
        ``Metadata`` and ``Error`` -- both can fire at close without the
        model having consumed input. Trusting Metadata as an ack causes
        front-loss when the model errored before producing transcript and
        the replay buffer is cleared prematurely.
        """
        is_close_only = text is not None and (
            '"type":"Metadata"' in text or '"type":"Error"' in text
        )
        if is_close_only:
            return
        if self._retry_attempt != 0 or self._retry_window_start != 0.0 or self._retry_not_before != 0.0:
            logger.info(
                "payload ack: data received (%dB) -> resetting retry counters "
                "(was attempt=%d, windowStart=%.3f, notBefore=%.3f)",
                len(raw),
                self._retry_attempt,
                self._retry_window_start,
                self._retry_not_before,
            )
            self._retry_attempt = 0
            self._retry_window_start = 0.0
            self._retry_not_before = 0.0
        self._clear_replay()

    async def __aiter__(self):
        """Async-iterate over messages until the stream ends."""
        while not self._closed:
            msg = await self.recv()
            if msg is None:
                break
            yield msg

    async def close(self) -> None:
        """Close the SageMaker stream and release resources."""
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        self._clear_replay()
        if self._stream:
            try:
                await self._stream.input_stream.close()
            except Exception:
                pass
        logger.info("Closed SageMaker connection: %s", self.endpoint_name)


def _summarize(error: BaseException) -> str:
    msg = str(error)
    if len(msg) > 160:
        msg = msg[:157] + "..."
    status = _status_code(error)
    code = _error_code(error)
    suffix = ""
    if status is not None or code is not None:
        parts = []
        if status is not None:
            parts.append(f"status={status}")
        if code is not None:
            parts.append(f"code={code}")
        suffix = " [" + " ".join(parts) + "]"
    return f"{type(error).__name__}: {msg}{suffix}"


class SageMakerTransportFactory:
    """Factory callable for ``AsyncDeepgramClient(transport_factory=...)``.

    **Async-only** -- must be used with ``AsyncDeepgramClient``, not the sync
    ``DeepgramClient``. Passing this factory to ``DeepgramClient`` raises
    ``TypeError``.

    When the SDK calls ``factory(url, headers)``, this extracts the
    invocation path and query string from the URL and creates a
    :class:`SageMakerTransport`. The URL is the WebSocket URL the SDK would
    normally connect to, e.g.::

        wss://api.deepgram.com/v1/listen?model=nova-3&interim_results=true

    From this, the factory extracts:
    - ``invocation_path`` = ``"v1/listen"``
    - ``query_string`` = ``"model=nova-3&interim_results=true"``

    Pass either a fully-built :class:`SageMakerConfig` via ``config=`` for
    burst-tuned timeouts and retry behavior, or the shortcut
    ``endpoint_name=`` / ``region=`` when only the endpoint identity needs
    to be set. Mixing the two forms is rejected.
    """

    _SYNC_ERROR = (
        "SageMakerTransportFactory is async-only and cannot be used with the "
        "sync DeepgramClient. Use AsyncDeepgramClient instead."
    )

    def __init__(
        self,
        endpoint_name: str | None = None,
        region: str | None = None,
        *,
        config: SageMakerConfig | None = None,
    ) -> None:
        if config is not None:
            if endpoint_name is not None or region is not None:
                raise ValueError(
                    "Pass either `config=SageMakerConfig(...)` or the "
                    "shortcut `endpoint_name=`/`region=`, not both."
                )
            self._config = config
        else:
            if endpoint_name is None:
                raise TypeError("endpoint_name is required")
            self._config = SageMakerConfig(
                endpoint_name=endpoint_name,
                region=region if region is not None else "us-west-2",
            )
        self.endpoint_name = self._config.endpoint_name
        self.region = self._config.region

    @property
    def config(self) -> SageMakerConfig:
        return self._config

    def __call__(self, url: str, headers: dict) -> SageMakerTransport:
        """Create a transport instance from the SDK-provided WebSocket URL.

        Raises
        ------
        TypeError
            If called outside an async context (i.e. from the sync
            ``DeepgramClient``), since SageMaker streaming is async-only.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            raise TypeError(self._SYNC_ERROR) from None

        parsed = urlparse(url)
        # Strip leading slash -- SageMaker expects "v1/listen" not "/v1/listen"
        invocation_path = parsed.path.lstrip("/")
        query_string = parsed.query  # already URL-encoded

        return SageMakerTransport(
            config=self._config,
            invocation_path=invocation_path,
            query_string=query_string,
        )
