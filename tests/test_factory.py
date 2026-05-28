"""Tests for SageMakerTransportFactory URL parsing and sync guard."""

import asyncio

import pytest

from deepgram_sagemaker import (
    SageMakerConfig,
    SageMakerTransport,
    SageMakerTransportFactory,
)


class TestSageMakerTransportFactory:
    """Tests for the factory's URL parsing and transport creation."""

    def test_factory_parses_standard_url(self):
        """Factory extracts invocation_path and query_string from a WebSocket URL."""
        factory = SageMakerTransportFactory("my-endpoint", region="us-east-1")

        async def _run():
            transport = factory(
                "wss://api.deepgram.com/v1/listen?model=nova-3&interim_results=true",
                {},
            )
            assert isinstance(transport, SageMakerTransport)
            assert transport.endpoint_name == "my-endpoint"
            assert transport.region == "us-east-1"
            assert transport.invocation_path == "v1/listen"
            assert transport.query_string == "model=nova-3&interim_results=true"

        asyncio.run(_run())

    def test_factory_strips_leading_slash(self):
        """Invocation path should not have a leading slash."""
        factory = SageMakerTransportFactory("ep")

        async def _run():
            transport = factory("wss://host/v1/listen", {})
            assert transport.invocation_path == "v1/listen"

        asyncio.run(_run())

    def test_factory_handles_no_query_string(self):
        """Factory handles URLs with no query parameters."""
        factory = SageMakerTransportFactory("ep")

        async def _run():
            transport = factory("wss://host/v1/listen", {})
            assert transport.query_string == ""

        asyncio.run(_run())

    def test_factory_default_region(self):
        """Default region is us-west-2."""
        factory = SageMakerTransportFactory("ep")
        assert factory.region == "us-west-2"

    def test_factory_rejects_sync_context(self):
        """Factory raises TypeError when called outside an async context."""
        factory = SageMakerTransportFactory("ep")
        with pytest.raises(TypeError, match="async-only"):
            factory("wss://host/v1/listen", {})

    def test_factory_accepts_config(self):
        """Factory accepts a fully-built SageMakerConfig via the config= keyword."""
        cfg = SageMakerConfig(
            endpoint_name="custom-ep",
            region="us-east-2",
            connection_timeout=5.0,
            subscription_timeout=15.0,
        )
        factory = SageMakerTransportFactory(config=cfg)
        assert factory.endpoint_name == "custom-ep"
        assert factory.region == "us-east-2"
        assert factory.config.connection_timeout == 5.0
        assert factory.config.subscription_timeout == 15.0

    def test_factory_rejects_config_mixed_with_shortcut(self):
        """Mixing config= with endpoint_name=/region= is rejected to avoid ambiguity."""
        cfg = SageMakerConfig(endpoint_name="from-config")
        with pytest.raises(ValueError, match="not both"):
            SageMakerTransportFactory("from-shortcut", config=cfg)

    def test_factory_requires_endpoint(self):
        """Factory without endpoint_name and without config raises TypeError."""
        with pytest.raises(TypeError, match="endpoint_name is required"):
            SageMakerTransportFactory()


class TestSageMakerTransportInit:
    """Tests for SageMakerTransport initialization (no connection)."""

    def test_initial_state(self):
        """Transport starts disconnected and not closed."""
        transport = SageMakerTransport(
            config=SageMakerConfig(endpoint_name="ep"),
            invocation_path="v1/listen",
            query_string="model=nova-3",
        )
        assert transport._connected is False
        assert transport._closed is False
        assert transport._stream is None

    async def test_close_idempotent(self):
        """Calling close() multiple times is safe."""
        transport = SageMakerTransport(
            config=SageMakerConfig(endpoint_name="ep"),
            invocation_path="v1/listen",
            query_string="",
        )
        await transport.close()
        await transport.close()  # should not raise
        assert transport._closed is True


class TestExports:
    """Tests for package-level exports."""

    def test_public_exports(self):
        """Package exports SageMakerConfig, SageMakerTransport, and SageMakerTransportFactory."""
        import deepgram_sagemaker

        assert hasattr(deepgram_sagemaker, "SageMakerConfig")
        assert hasattr(deepgram_sagemaker, "SageMakerTransport")
        assert hasattr(deepgram_sagemaker, "SageMakerTransportFactory")
        for name in ("SageMakerConfig", "SageMakerTransport", "SageMakerTransportFactory"):
            assert name in deepgram_sagemaker.__all__
