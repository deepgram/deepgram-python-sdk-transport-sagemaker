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

## Configuration

For burst-tuned timeouts and retry behavior, build a `SageMakerConfig` and pass
it via `config=`:

```python
from deepgram_sagemaker import SageMakerConfig, SageMakerTransportFactory

config = SageMakerConfig(
    endpoint_name="my-deepgram-endpoint",
    region="us-east-2",
    connection_timeout=5.0,
    connection_acquire_timeout=15.0,
)
factory = SageMakerTransportFactory(config=config)
```

All time-based fields are `float` seconds (matching `asyncio` convention).

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `endpoint_name` | Yes | — | SageMaker endpoint name |
| `region` | No | `us-west-2` | AWS region |
| `connection_timeout` | No | `30.0` | Max time for the underlying TCP/TLS connect (AWS default is ~2 s — bumped here so cold-start endpoints under burst load have time to accept TLS handshakes). |
| `connection_acquire_timeout` | No | `60.0` | Max time to acquire a connection from the underlying HTTP/2 pool (AWS default is ~10 s — bumped so a 200–500-stream burst doesn't drain the acquire pool). |
| `subscription_timeout` | No | `60.0` | Max time the transport waits for the SageMaker bidi stream to open before failing. A timeout here is treated as a transient connect failure and counts against `max_retries` / `retry_budget`. |
| `max_concurrency` | No | `500` | Cap on simultaneous in-flight HTTP/2 streams. Advisory in Python today (the underlying smithy HTTP/2 stack does not expose a hard cap), but kept for surface parity with the Java transport. |
| `max_retries` | No | `5` | Max retries on transient AWS errors (throttling, pool-exhausted, transient connect/timeout). Set to `0` to disable internal retry. Terminal errors (auth, validation) bypass this. |
| `initial_backoff` | No | `0.1` | First backoff delay applied after the initial failure. |
| `max_backoff` | No | `5.0` | Cap on the per-attempt backoff delay regardless of multiplier. |
| `backoff_multiplier` | No | `2.0` | Exponential growth factor between retry attempts. Must be `>= 1.0`. |
| `retry_budget` | No | `30.0` | Total wall-clock cap across all retry attempts before giving up and surfacing the error to the application. |
| `max_replay_buffer_bytes` | No | `8 * 1024 * 1024` | Cap on the in-memory replay buffer that holds sent-but-unacked stream events. Set to `0` to disable replay (sent events are dropped on internal reset). |

### High-concurrency notes

The transport's defaults are tuned for high-burst workloads (large numbers of
streams opened in a tight loop against an endpoint that may need to scale up).
If you're opening 200–500 streams simultaneously against a cold endpoint, the
AWS SDK's general-purpose defaults (~2 s connect, ~10 s acquire) will fire
before the load balancer has accepted all of the inbound TLS handshakes — you
will see a wave of acquire / connect timeouts that look like server-side
problems but are really client-side fail-fast tripping early.

This transport ships with more lenient defaults (30 s / 60 s) so the common
high-concurrency path works out of the box. Tighten them if you need fail-fast
behavior in low-latency pipelines:

```python
config = SageMakerConfig(
    endpoint_name="my-deepgram-endpoint",
    region="us-east-2",
    connection_timeout=5.0,
    connection_acquire_timeout=15.0,
)
```

### Retry & storm absorption

Transient AWS-side failures (`ThrottlingException`, connection-pool exhaustion,
transient connect/timeout failures) are absorbed by the transport itself:
classified as retryable, retried with jittered exponential backoff up to
`max_retries` and `retry_budget`, with messages buffered during the reset
window replayed onto the new stream so audio isn't dropped. Only **terminal**
errors (auth, validation, resource-not-found) and budget-exhausted retryable
errors propagate to the application.

```python
config = SageMakerConfig(
    endpoint_name="my-deepgram-endpoint",
    max_retries=10,
    initial_backoff=0.2,
    max_backoff=10.0,
    retry_budget=60.0,
)
```

Set `max_retries=0` to disable internal retry entirely (every transient AWS
error then surfaces immediately to the application).

## AWS Credentials

The transport resolves AWS credentials using boto3's credential chain:

- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- Shared credentials file (`~/.aws/credentials`)
- IAM role (EC2, ECS, Lambda)

## Links

- [Deepgram Python SDK](https://github.com/deepgram/deepgram-python-sdk)
- [SageMaker example](https://github.com/deepgram/deepgram-python-sdk/blob/main/examples/27-transcription-live-sagemaker.py)
- [Deepgram documentation](https://developers.deepgram.com)
