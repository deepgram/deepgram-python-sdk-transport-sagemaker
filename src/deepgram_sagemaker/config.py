"""Configuration for the SageMaker transport.

Defaults are tuned for high-burst workloads (large numbers of streams opened in
a tight loop against an endpoint that may need to scale up). They are
intentionally more lenient than the AWS SDK defaults so that 200--500-stream
bursts don't trip connect-acquire / connect-handshake timeouts before the
endpoint has had a chance to accept the inbound TLS handshakes. Tighten them
if you want fail-fast behavior in low-latency pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

# AWS SDK default for the underlying HTTP/2 connect is ~2 s. Cold endpoints
# under a 200-500 stream burst can't accept TLS handshakes in 2 s.
DEFAULT_CONNECTION_TIMEOUT = 30.0

# AWS SDK default acquire is ~10 s. A 400-stream burst drains the acquire pool
# past 10 s.
DEFAULT_CONNECTION_ACQUIRE_TIMEOUT = 60.0

# Time to wait for the SageMaker bidi stream to open before failing the first
# send. Was previously hardcoded to 10 s; raised to match the Java 0.1.3 default.
DEFAULT_SUBSCRIPTION_TIMEOUT = 60.0

# Cap on simultaneous in-flight HTTP/2 streams. Advisory in Python today
# (the underlying smithy HTTP/2 stack does not expose a hard cap), but kept
# for surface parity with the Java config and to feed any future Python-side
# concurrency limiter.
DEFAULT_MAX_CONCURRENCY = 500

# Max retries on transient AWS errors per stream invocation. Terminal errors
# (auth, validation) bypass this and surface immediately.
DEFAULT_MAX_RETRIES = 5

# First backoff delay after the initial failure.
DEFAULT_INITIAL_BACKOFF = 0.1

# Cap on per-attempt backoff delay regardless of multiplier.
DEFAULT_MAX_BACKOFF = 5.0

# Exponential growth factor between retry attempts.
DEFAULT_BACKOFF_MULTIPLIER = 2.0

# Total wall-clock budget across all retry attempts before giving up.
DEFAULT_RETRY_BUDGET = 30.0

# Cap on the in-memory replay buffer that holds sent-but-unacked stream events
# for the current bidi stream attempt. On internal reset (retryable error), the
# buffer is drained onto the new stream so AWS sees a continuous audio sequence
# rather than the gap created by the discarded events. Trimmed when a real
# downstream payload arrives (model has consumed prior audio). 8 MiB ~= 256 s
# of 16 kHz mono 16-bit PCM, which covers the longest throttle storms we've
# seen in practice.
DEFAULT_MAX_REPLAY_BUFFER_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class SageMakerConfig:
    """Configuration for connecting to a Deepgram model hosted on SageMaker.

    Pass an instance to :class:`SageMakerTransportFactory` via its ``config=``
    parameter. The factory also accepts ``endpoint_name=`` / ``region=`` as a
    shortcut when only the endpoint identity needs to be set; mixing the two
    forms is rejected.

    All time-based fields are ``float`` seconds, matching the convention of
    :mod:`asyncio` timeouts.
    """

    endpoint_name: str
    region: str = "us-west-2"
    content_type: str = "application/octet-stream"
    accept_type: str = "application/json"
    connection_timeout: float = DEFAULT_CONNECTION_TIMEOUT
    connection_acquire_timeout: float = DEFAULT_CONNECTION_ACQUIRE_TIMEOUT
    subscription_timeout: float = DEFAULT_SUBSCRIPTION_TIMEOUT
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF
    max_backoff: float = DEFAULT_MAX_BACKOFF
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER
    retry_budget: float = DEFAULT_RETRY_BUDGET
    max_replay_buffer_bytes: int = DEFAULT_MAX_REPLAY_BUFFER_BYTES

    def __post_init__(self) -> None:
        if not self.endpoint_name or not self.endpoint_name.strip():
            raise ValueError("endpoint_name is required")
        _require_positive("connection_timeout", self.connection_timeout)
        _require_positive("connection_acquire_timeout", self.connection_acquire_timeout)
        _require_positive("subscription_timeout", self.subscription_timeout)
        _require_positive("max_concurrency", self.max_concurrency)
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        _require_positive("initial_backoff", self.initial_backoff)
        _require_positive("max_backoff", self.max_backoff)
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")
        _require_positive("retry_budget", self.retry_budget)
        if self.max_replay_buffer_bytes < 0:
            raise ValueError("max_replay_buffer_bytes must be non-negative")
        if self.initial_backoff > self.max_backoff:
            raise ValueError(
                f"initial_backoff ({self.initial_backoff}s) must not exceed "
                f"max_backoff ({self.max_backoff}s)"
            )


def _require_positive(name: str, value: float | int) -> None:
    if value is None or value <= 0:
        raise ValueError(f"{name} must be positive")
