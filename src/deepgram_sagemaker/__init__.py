"""SageMaker transport for the Deepgram Python SDK.

Install::

    pip install deepgram-sagemaker

Usage::

    from deepgram import AsyncDeepgramClient
    from deepgram_sagemaker import SageMakerTransportFactory

    factory = SageMakerTransportFactory(
        endpoint_name="my-deepgram-endpoint",
        region="us-west-2",
    )
    client = AsyncDeepgramClient(
        api_key="unused",
        transport_factory=factory,
    )

For burst-tuned timeouts and retry behavior::

    from deepgram_sagemaker import SageMakerConfig, SageMakerTransportFactory

    config = SageMakerConfig(
        endpoint_name="my-deepgram-endpoint",
        region="us-east-1",
        connection_timeout=5.0,
        connection_acquire_timeout=15.0,
    )
    factory = SageMakerTransportFactory(config=config)
"""

from .config import SageMakerConfig
from .transport import SageMakerTransport, SageMakerTransportFactory

__all__ = ["SageMakerConfig", "SageMakerTransport", "SageMakerTransportFactory"]
