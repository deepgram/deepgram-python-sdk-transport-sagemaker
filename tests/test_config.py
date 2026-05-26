"""Tests for SageMakerConfig defaults and validation."""

import pytest

from deepgram_sagemaker import SageMakerConfig
from deepgram_sagemaker.config import (
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_CONNECTION_ACQUIRE_TIMEOUT,
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_INITIAL_BACKOFF,
    DEFAULT_MAX_BACKOFF,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_REPLAY_BUFFER_BYTES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BUDGET,
    DEFAULT_SUBSCRIPTION_TIMEOUT,
)


class TestDefaults:
    """Burst-tuned defaults must match the Java 0.1.3 transport."""

    def test_defaults_applied(self):
        c = SageMakerConfig(endpoint_name="ep")
        assert c.connection_timeout == DEFAULT_CONNECTION_TIMEOUT == 30.0
        assert c.connection_acquire_timeout == DEFAULT_CONNECTION_ACQUIRE_TIMEOUT == 60.0
        assert c.subscription_timeout == DEFAULT_SUBSCRIPTION_TIMEOUT == 60.0
        assert c.max_concurrency == DEFAULT_MAX_CONCURRENCY == 500
        assert c.max_retries == DEFAULT_MAX_RETRIES == 5
        assert c.initial_backoff == DEFAULT_INITIAL_BACKOFF == 0.1
        assert c.max_backoff == DEFAULT_MAX_BACKOFF == 5.0
        assert c.backoff_multiplier == DEFAULT_BACKOFF_MULTIPLIER == 2.0
        assert c.retry_budget == DEFAULT_RETRY_BUDGET == 30.0
        assert c.max_replay_buffer_bytes == DEFAULT_MAX_REPLAY_BUFFER_BYTES == 8 * 1024 * 1024

    def test_default_region(self):
        c = SageMakerConfig(endpoint_name="ep")
        assert c.region == "us-west-2"


class TestValidation:
    """Each setter validates >0 / non-empty / non-negative."""

    def test_endpoint_required(self):
        with pytest.raises(ValueError, match="endpoint_name is required"):
            SageMakerConfig(endpoint_name="")

    def test_endpoint_blank_rejected(self):
        with pytest.raises(ValueError, match="endpoint_name is required"):
            SageMakerConfig(endpoint_name="   ")

    @pytest.mark.parametrize(
        "kw",
        [
            "connection_timeout",
            "connection_acquire_timeout",
            "subscription_timeout",
            "initial_backoff",
            "max_backoff",
            "retry_budget",
        ],
    )
    def test_zero_or_negative_durations_rejected(self, kw):
        with pytest.raises(ValueError, match="must be positive"):
            SageMakerConfig(endpoint_name="ep", **{kw: 0})
        with pytest.raises(ValueError, match="must be positive"):
            SageMakerConfig(endpoint_name="ep", **{kw: -1.5})

    def test_max_concurrency_must_be_positive(self):
        with pytest.raises(ValueError, match="max_concurrency must be positive"):
            SageMakerConfig(endpoint_name="ep", max_concurrency=0)

    def test_max_retries_zero_ok(self):
        c = SageMakerConfig(endpoint_name="ep", max_retries=0)
        assert c.max_retries == 0

    def test_max_retries_negative_rejected(self):
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            SageMakerConfig(endpoint_name="ep", max_retries=-1)

    def test_backoff_multiplier_below_one_rejected(self):
        with pytest.raises(ValueError, match="backoff_multiplier must be >= 1.0"):
            SageMakerConfig(endpoint_name="ep", backoff_multiplier=0.5)

    def test_initial_above_max_backoff_rejected(self):
        with pytest.raises(ValueError, match="must not exceed max_backoff"):
            SageMakerConfig(endpoint_name="ep", initial_backoff=10.0, max_backoff=1.0)

    def test_replay_buffer_zero_ok(self):
        c = SageMakerConfig(endpoint_name="ep", max_replay_buffer_bytes=0)
        assert c.max_replay_buffer_bytes == 0

    def test_replay_buffer_negative_rejected(self):
        with pytest.raises(ValueError, match="max_replay_buffer_bytes must be non-negative"):
            SageMakerConfig(endpoint_name="ep", max_replay_buffer_bytes=-1)


class TestImmutability:
    """SageMakerConfig is frozen — once built, fields cannot be reassigned."""

    def test_frozen(self):
        c = SageMakerConfig(endpoint_name="ep")
        with pytest.raises(Exception):  # FrozenInstanceError, subclass of AttributeError
            c.connection_timeout = 1.0  # type: ignore[misc]
