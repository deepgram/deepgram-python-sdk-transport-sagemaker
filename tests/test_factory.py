"""Tests for SageMakerTransportFactory URL parsing and sync guard."""

import asyncio

import pytest

import deepgram_sagemaker.transport as transport_module
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

    @pytest.mark.asyncio
    async def test_connect_resolves_async_aws_config(self, monkeypatch):
        """AWS runtime 0.11 accepts the resolved config arguments used by the transport."""
        captured: dict[str, object] = {}

        class FakeInputStream:
            async def close(self):
                captured["input_stream_closed"] = True

        class FakeStream:
            input_stream = FakeInputStream()

            async def await_output(self):
                return (None, object())

        class FakeClient:
            def __init__(self, config):
                captured["client_config"] = config

            async def invoke_endpoint_with_bidirectional_stream(self, stream_input):
                captured["stream_input"] = stream_input
                return FakeStream()

            async def close(self):
                captured["client_closed"] = True

        monkeypatch.setattr(transport_module, "AsyncSageMakerRuntimeHTTP2Client", FakeClient)
        config = SageMakerConfig(endpoint_name="ep", connection_timeout=12.5)
        transport = SageMakerTransport(config, "v1/listen", "model=nova-3")

        await transport._do_connect()

        resolved_config = captured["client_config"]
        assert captured["client_config"] is not None
        assert type(resolved_config).__name__ == "AsyncSageMakerRuntimeHTTP2Config"
        assert resolved_config.endpoint_uri == "https://runtime.sagemaker.us-west-2.amazonaws.com:8443"
        assert resolved_config.region == "us-west-2"
        assert type(resolved_config.transport).__name__ == "AWSCRTHTTPClient"
        assert resolved_config.http_request_config.read_timeout == 12.5

        await transport.close()
        assert captured["input_stream_closed"] is True
        assert captured["client_closed"] is True
        assert transport._client is None

    @pytest.mark.asyncio
    async def test_connect_cleanup_preserves_setup_error_when_client_close_fails(self, monkeypatch, caplog):
        """Client-close failures must not mask the setup failure that triggered cleanup."""

        class FailingClient:
            def __init__(self, config):
                pass

            async def invoke_endpoint_with_bidirectional_stream(self, stream_input):
                raise RuntimeError("stream setup failed")

            async def close(self):
                raise RuntimeError("client close failed")

        monkeypatch.setattr(transport_module, "AsyncSageMakerRuntimeHTTP2Client", FailingClient)
        transport = SageMakerTransport(
            SageMakerConfig(endpoint_name="ep", connection_timeout=0.01), "v1/listen", ""
        )

        with pytest.raises(RuntimeError, match="stream setup failed"):
            await asyncio.wait_for(transport._do_connect(), timeout=0.1)

        assert "client shutdown failed" in caplog.text
        assert transport._client is None

    @pytest.mark.asyncio
    async def test_connect_cleanup_bounds_a_hung_client_close(self, monkeypatch, caplog):
        """A stalled client close must not prevent the original setup error from surfacing."""

        class HangingCloseClient:
            def __init__(self, config):
                pass

            async def invoke_endpoint_with_bidirectional_stream(self, stream_input):
                raise RuntimeError("stream setup failed")

            async def close(self):
                await asyncio.Event().wait()

        monkeypatch.setattr(transport_module, "AsyncSageMakerRuntimeHTTP2Client", HangingCloseClient)
        transport = SageMakerTransport(
            SageMakerConfig(endpoint_name="ep", connection_timeout=0.01), "v1/listen", ""
        )

        with pytest.raises(RuntimeError, match="stream setup failed"):
            await asyncio.wait_for(transport._do_connect(), timeout=0.1)

        assert "client shutdown timed out" in caplog.text
        assert transport._client is None

    @pytest.mark.asyncio
    async def test_close_bounds_a_hung_input_stream_and_closes_the_client(self, monkeypatch, caplog):
        """A stalled stream close must still release the client connection."""
        captured: dict[str, object] = {}

        class HangingInputStream:
            async def close(self):
                await asyncio.Event().wait()

        class FakeStream:
            input_stream = HangingInputStream()

            async def await_output(self):
                return (None, object())

        class FakeClient:
            def __init__(self, config):
                pass

            async def invoke_endpoint_with_bidirectional_stream(self, stream_input):
                return FakeStream()

            async def close(self):
                captured["client_closed"] = True

        monkeypatch.setattr(transport_module, "AsyncSageMakerRuntimeHTTP2Client", FakeClient)
        transport = SageMakerTransport(
            SageMakerConfig(endpoint_name="ep", connection_timeout=0.01), "v1/listen", ""
        )
        await transport._do_connect()

        await asyncio.wait_for(transport.close(), timeout=0.1)

        assert "input stream shutdown timed out" in caplog.text
        assert captured["client_closed"] is True
        assert transport._client is None

    @pytest.mark.asyncio
    async def test_connect_closes_a_client_constructed_after_transport_close(self, monkeypatch):
        """A close during config resolution must release the client constructed afterwards."""
        resolve_started = asyncio.Event()
        finish_resolve = asyncio.Event()
        captured: dict[str, object] = {}

        class SlowConfig:
            @classmethod
            async def resolve(cls, **kwargs):
                resolve_started.set()
                await finish_resolve.wait()
                return object()

        class FakeClient:
            def __init__(self, config):
                captured["client_created"] = True

            async def close(self):
                captured["client_closed"] = True

        monkeypatch.setattr(transport_module, "AsyncSageMakerRuntimeHTTP2Config", SlowConfig)
        monkeypatch.setattr(transport_module, "AsyncSageMakerRuntimeHTTP2Client", FakeClient)
        transport = SageMakerTransport(SageMakerConfig(endpoint_name="ep"), "v1/listen", "")
        connect = asyncio.create_task(transport._do_connect())
        await resolve_started.wait()

        await transport.close()
        finish_resolve.set()

        with pytest.raises(RuntimeError, match="Transport is closed"):
            await connect

        assert captured["client_created"] is True
        assert captured["client_closed"] is True
        assert transport._client is None

    @pytest.mark.asyncio
    async def test_close_logs_input_stream_failure_and_closes_the_client(self, caplog):
        """An input stream close error must not prevent client cleanup."""
        captured: dict[str, object] = {}

        class FailingInputStream:
            async def close(self):
                raise RuntimeError("input close failed")

        class FakeStream:
            input_stream = FailingInputStream()

        class FakeClient:
            async def close(self):
                captured["client_closed"] = True

        transport = SageMakerTransport(SageMakerConfig(endpoint_name="ep"), "v1/listen", "")
        transport._stream = FakeStream()
        transport._client = FakeClient()

        await transport.close()

        assert "input stream shutdown failed" in caplog.text
        assert captured["client_closed"] is True
        assert transport._client is None

    @pytest.mark.asyncio
    async def test_retry_reset_closes_and_clears_the_client(self):
        """A retryable error must release the client before the next connection attempt."""
        captured: dict[str, object] = {}

        class FakeClient:
            async def close(self):
                captured["client_closed"] = True

        transport = SageMakerTransport(SageMakerConfig(endpoint_name="ep"), "v1/listen", "")
        transport._stream = object()
        transport._output_stream = object()
        transport._client = FakeClient()

        assert await transport._handle_retryable_error(ConnectionError("connection lost")) is True
        assert captured["client_closed"] is True
        assert transport._client is None
        assert transport._stream is None
        assert transport._output_stream is None


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
