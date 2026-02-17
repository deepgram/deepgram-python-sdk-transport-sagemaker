# Deepgram SageMaker Transport

[![PyPI version](https://img.shields.io/pypi/v/deepgram-sagemaker)](https://pypi.python.org/pypi/deepgram-sagemaker)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

SageMaker transport for the [Deepgram Python SDK](https://github.com/deepgram/deepgram-python-sdk). Uses AWS SageMaker's HTTP/2 bidirectional streaming API as an alternative to WebSocket, allowing transparent switching between Deepgram Cloud and Deepgram on SageMaker.

**Requires Python 3.12+** (due to AWS SDK dependencies).

## Installation

```bash
pip install deepgram-sagemaker
```

This installs `aws-sdk-sagemaker-runtime-http2` and `boto3` automatically.

## Usage

The SageMaker transport is **async-only** and must be used with `AsyncDeepgramClient`:

```python
import asyncio
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram_sagemaker import SageMakerTransportFactory

factory = SageMakerTransportFactory(
    endpoint_name="my-deepgram-endpoint",
    region="us-west-2",
)

# SageMaker uses AWS credentials (not Deepgram API keys)
client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

async def main():
    async with client.listen.v1.connect(model="nova-3") as connection:
        connection.on(EventType.MESSAGE, lambda msg: print(msg))
        await connection.start_listening()

asyncio.run(main())
```

## AWS Credentials

The transport resolves AWS credentials using boto3's credential chain:

- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- Shared credentials file (`~/.aws/credentials`)
- IAM role (EC2, ECS, Lambda)

## Links

- [Deepgram Python SDK](https://github.com/deepgram/deepgram-python-sdk)
- [SageMaker example](https://github.com/deepgram/deepgram-python-sdk/blob/main/examples/27-transcription-live-sagemaker.py)
- [Deepgram documentation](https://developers.deepgram.com)
