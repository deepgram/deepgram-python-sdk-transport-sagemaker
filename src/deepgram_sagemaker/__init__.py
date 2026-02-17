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
"""

from .transport import SageMakerTransport, SageMakerTransportFactory

__all__ = ["SageMakerTransport", "SageMakerTransportFactory"]
