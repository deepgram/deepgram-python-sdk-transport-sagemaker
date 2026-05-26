"""Tests for the retry classifier, jitter backoff, and replay buffer.

End-to-end retry against the real smithy bidi-stream is not covered here (the
async handler indirection makes it hard to deterministically stub); those
paths are exercised by burst tests against a real SageMaker endpoint.
"""

import statistics

import pytest

from deepgram_sagemaker import SageMakerConfig, SageMakerTransport
from deepgram_sagemaker.transport import (
    _RETRYABLE,
    _TERMINAL,
    classify,
    compute_backoff,
)


def _new_transport(max_replay_buffer_bytes: int = 1024) -> SageMakerTransport:
    cfg = SageMakerConfig(
        endpoint_name="test",
        region="us-east-1",
        max_replay_buffer_bytes=max_replay_buffer_bytes,
    )
    return SageMakerTransport(config=cfg, invocation_path="v1/listen", query_string="")


# RNG stubs — return the floor, the ceiling, or the midpoint of the range.
def _min_rng(lo: float, hi: float) -> float:
    return lo


def _max_rng(lo: float, hi: float) -> float:
    return hi


def _mid_rng(lo: float, hi: float) -> float:
    return lo + (hi - lo) / 2


class TestClassify:
    """Defaults to RETRYABLE; TERMINAL only for caller-side AWS 4xx rejections."""

    def test_timeout_is_retryable(self):
        assert classify(TimeoutError("acquire timeout")) == _RETRYABLE

    def test_connection_error_is_retryable(self):
        assert classify(ConnectionError("connection refused")) == _RETRYABLE

    def test_oserror_is_retryable(self):
        assert classify(OSError("network error")) == _RETRYABLE

    def test_cancelled_is_retryable(self):
        import asyncio

        assert classify(asyncio.CancelledError()) == _RETRYABLE

    def test_aws_429_is_retryable(self):
        class FakeAws(Exception):
            status_code = 429
            error_code = "ThrottlingException"

        assert classify(FakeAws("Rate exceeded")) == _RETRYABLE

    def test_aws_5xx_is_retryable(self):
        class FakeAws(Exception):
            status_code = 503

        assert classify(FakeAws("internal")) == _RETRYABLE

    def test_throttling_error_code_overrides_status(self):
        """Throttling-coded errors retry regardless of (non-throttling) status."""

        class FakeAws(Exception):
            status_code = 400
            error_code = "ThrottlingException"

        assert classify(FakeAws("Rate exceeded")) == _RETRYABLE

    def test_aws_401_is_terminal(self):
        class FakeAws(Exception):
            status_code = 401

        assert classify(FakeAws("Forbidden")) == _TERMINAL

    def test_aws_403_is_terminal(self):
        class FakeAws(Exception):
            status_code = 403

        assert classify(FakeAws("Forbidden")) == _TERMINAL

    def test_aws_400_validation_is_terminal(self):
        class FakeAws(Exception):
            status_code = 400
            error_code = "ValidationException"

        assert classify(FakeAws("invalid input")) == _TERMINAL

    def test_aws_404_is_terminal(self):
        class FakeAws(Exception):
            status_code = 404

        assert classify(FakeAws("endpoint not found")) == _TERMINAL

    def test_aws_424_is_retryable(self):
        """ModelError 424 from primary container is transient under burst load."""

        class FakeAws(Exception):
            status_code = 424

        assert classify(FakeAws('Failed to establish WebSocket connection')) == _RETRYABLE

    def test_walks_cause_chain(self):
        inner = OSError("netty")
        wrapper = RuntimeError("oops")
        wrapper.__cause__ = inner
        assert classify(wrapper) == _RETRYABLE

    def test_unknown_defaults_to_retryable(self):
        assert classify(RuntimeError("mystery")) == _RETRYABLE

    def test_throttling_in_classname_is_retryable(self):
        """Exceptions named like ThrottlingException retry even without explicit code attr."""

        class ThrottlingException(Exception):
            pass

        assert classify(ThrottlingException("Rate exceeded")) == _RETRYABLE


class TestComputeBackoff:
    """Full-jitter exponential backoff."""

    def test_attempt_zero_is_floor(self):
        # initial=0.1, multiplier=2: at attempt=0 scaled==initial so range collapses.
        assert compute_backoff(0.1, 1.0, 2.0, 0, _mid_rng) == pytest.approx(0.1)

    def test_attempt_1_midpoint(self):
        # initial=0.1, multiplier=2 -> ceiling=0.2. mid = 0.1 + 0.05 = 0.15.
        assert compute_backoff(0.1, 1.0, 2.0, 1, _mid_rng) == pytest.approx(0.15)

    def test_attempt_4_capped(self):
        # scaled = 0.1 * 16 = 1.6, capped to 1.0. Range [0.1, 1.0], midpoint = 0.55.
        assert compute_backoff(0.1, 1.0, 2.0, 4, _mid_rng) == pytest.approx(0.55)

    def test_rng_bounds_respected(self):
        # attempt=2: scaled = 0.1 * 4 = 0.4. Range [0.1, 0.4].
        assert compute_backoff(0.1, 1.0, 2.0, 2, _min_rng) == pytest.approx(0.1)
        assert compute_backoff(0.1, 1.0, 2.0, 2, _max_rng) == pytest.approx(0.4)

    def test_ceiling_caps_at_max(self):
        assert compute_backoff(0.1, 5.0, 2.0, 100, _max_rng) == pytest.approx(5.0)
        assert compute_backoff(0.1, 5.0, 2.0, 10_000, _max_rng) == pytest.approx(5.0)

    def test_degenerate_range_skips_rng(self):
        """When ceiling == initial, return the floor without invoking RNG."""
        calls = []

        def counting_rng(lo: float, hi: float) -> float:
            calls.append((lo, hi))
            return lo

        assert compute_backoff(0.1, 1.0, 2.0, 0, counting_rng) == pytest.approx(0.1)
        assert compute_backoff(0.1, 1.0, 1.0, 5, counting_rng) == pytest.approx(0.1)
        assert calls == []

    def test_production_rng_spreads_retries(self):
        """The whole point of jitter: many concurrent retries must spread across the window."""
        samples = [compute_backoff(0.1, 1.0, 2.0, 4) for _ in range(1000)]
        assert min(samples) < 0.2, f"min should land near floor; got {min(samples)}"
        assert max(samples) > 0.9, f"max should land near ceiling; got {max(samples)}"
        mean = statistics.mean(samples)
        assert 0.4 < mean < 0.7, f"mean of uniform [0.1, 1.0] should be near 0.55; got {mean}"


class TestReplayBuffer:
    """Replay buffer with FIFO eviction at the byte cap."""

    def test_accumulates_with_byte_count(self):
        t = _new_transport(1024)
        t._buffer_for_replay("aaa", 3)
        t._buffer_for_replay("bbbbb", 5)
        assert len(t._replay) == 2
        assert t._replay_bytes == 8

    def test_clear_drops_all(self):
        t = _new_transport(1024)
        t._buffer_for_replay("a", 1)
        t._buffer_for_replay("b", 1)
        t._clear_replay()
        assert len(t._replay) == 0
        assert t._replay_bytes == 0

    def test_fifo_eviction_at_cap(self):
        t = _new_transport(10)  # cap = 10 bytes
        t._buffer_for_replay("a", 4)  # total 4
        t._buffer_for_replay("b", 4)  # total 8
        t._buffer_for_replay("c", 4)  # total 12 -> evict 'a', total 8
        t._buffer_for_replay("d", 4)  # total 12 -> evict 'b', total 8
        events = [be.event for be in t._replay]
        assert events == ["c", "d"]
        assert t._replay_bytes == 8

    def test_zero_cap_disables_buffer(self):
        t = _new_transport(0)
        t._buffer_for_replay("a", 1)
        t._buffer_for_replay("b", 1)
        assert len(t._replay) == 0
        assert t._replay_bytes == 0

    def test_oversized_single_event_dropped(self):
        t = _new_transport(10)
        t._buffer_for_replay("0123456789ABCDEF", 16)
        assert len(t._replay) == 0
        assert t._replay_bytes == 0


class TestPayloadAckResets:
    """Real payloads reset retry counters and clear the replay buffer; Metadata/Error don't."""

    def test_real_payload_resets_counters(self):
        t = _new_transport(1024)
        t._retry_attempt = 3
        t._retry_window_start = 1.0
        t._retry_not_before = 1.5
        t._buffer_for_replay("audio", 5)
        t._handle_payload_part_ack(b'{"channel":...}', '{"channel":"abc"}')
        assert t._retry_attempt == 0
        assert t._retry_window_start == 0.0
        assert t._retry_not_before == 0.0
        assert t._replay_bytes == 0

    def test_metadata_does_not_count_as_ack(self):
        t = _new_transport(1024)
        t._retry_attempt = 3
        t._buffer_for_replay("audio", 5)
        t._handle_payload_part_ack(
            b'{"type":"Metadata"}', '{"type":"Metadata","duration":0}'
        )
        # Counters and buffer must survive a Metadata-only payload.
        assert t._retry_attempt == 3
        assert t._replay_bytes == 5

    def test_error_does_not_count_as_ack(self):
        t = _new_transport(1024)
        t._retry_attempt = 2
        t._buffer_for_replay("audio", 5)
        t._handle_payload_part_ack(b'{"type":"Error"}', '{"type":"Error"}')
        assert t._retry_attempt == 2
        assert t._replay_bytes == 5

    def test_binary_payload_counts_as_ack(self):
        """TTS audio chunks (binary, no JSON envelope) prove the model produced output."""
        t = _new_transport(1024)
        t._retry_attempt = 1
        t._buffer_for_replay("audio", 5)
        t._handle_payload_part_ack(b"\x00\x01\x02", None)
        assert t._retry_attempt == 0
        assert t._replay_bytes == 0
