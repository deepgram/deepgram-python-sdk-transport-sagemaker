# Changelog

## [0.5.0](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/compare/v0.4.0...v0.5.0) (2026-09-18)


### Features

* configurable timeouts and storm absorption for high-burst workloads ([#6](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/6)) ([f2925fc](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/f2925fcea84f8575a20cb30f0bef818b4c9b262d))
* initial deepgram-sagemaker transport package ([8ff6a7d](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/8ff6a7d37f25c5a1226144fb65ca953af0819bec))
* support SageMaker runtime HTTP2 0.11 ([#9](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/9)) ([8296de3](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/8296de3a86ccc2166901a9a73970dc77212431d8))


### Bug Fixes

* live mic example display issues ([#4](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/4)) ([150217a](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/150217a0db84e768b50295b543f68ddf09991335))
* TTS support and transport improvements ([#2](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/2)) ([1f9d19f](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/1f9d19fd078883fe8341b8b4cc94bc20c31a6251))

## [0.4.0](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/compare/v0.3.0...v0.4.0) (2026-09-18)

Updates the SageMaker transport for AWS runtime HTTP2 `0.11`. Existing `SageMakerTransportFactory(endpoint_name=..., region=...)` usage remains unchanged.

### Features

* Use the runtime HTTP2 `0.11` async client and resolved configuration API, restoring SageMaker streaming for installations that resolve the current AWS runtime package. ([#9](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/9)) ([8296de3](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/8296de3a86ccc2166901a9a73970dc77212431d8))
* Install the required `awscrt` extra and configure its HTTP/2 client for bidirectional SageMaker streaming. `awscrt` uses a compiled extension: supported platforms install a wheel; source builds require a C toolchain. ([#9](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/9)) ([8296de3](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/8296de3a86ccc2166901a9a73970dc77212431d8))
* Close AWS clients during connection setup, retry resets, and transport shutdown. The STT example now waits up to 30 seconds for final transcripts before exiting. ([#9](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/9)) ([8296de3](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/8296de3a86ccc2166901a9a73970dc77212431d8))

## [0.3.0](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/compare/v0.2.2...v0.3.0) (2026-06-01)

High-burst hardening: a configurable `SageMakerConfig` plus internal storm absorption, so transient AWS-side failures are retried inside the transport instead of surfacing to callers. Backwards-compatible — existing `SageMakerTransportFactory(endpoint_name=..., region=...)` callers keep working and pick up the new lenient defaults.


### Features

* **transport:** configurable timeouts and storm absorption for high-burst workloads. New `SageMakerConfig` dataclass exposes tunable `connection_timeout` (30s, up from the AWS client's ~2s), `connection_acquire_timeout` (60s), `subscription_timeout` (60s), `max_concurrency` (500), and a retry stack: `max_retries` (5; `0` disables), full-jitter exponential backoff (`initial_backoff` / `max_backoff` / `backoff_multiplier`), a `retry_budget` (30s) ceiling, and an 8 MiB replay buffer (`max_replay_buffer_bytes`; `0` disables). The transport now absorbs transient AWS-side failures internally — the retry classifier defaults to RETRYABLE and treats 429 throttling and 424 `ModelError` as transient under burst, replays unacked events onto a fresh bidi stream so audio isn't lost on reset, and resets retry counters only on real downstream payloads (not Metadata/Error). Also adds whitespace-tolerant user-close detection for `listen.v1`/`v2` (`CloseStream` / `Finalize`) and `speak.v1` (`Close`) to stop post-completion retry storms. Pass `config=SageMakerConfig(...)` to tune. Mirrors the storm-absorption design in the Java transport; validated end-to-end against live Nova-3, Flux, and Aura endpoints at up to 400 concurrent connections ([#6](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/6)) ([f2925fc](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/f2925fcea84f8575a20cb30f0bef818b4c9b262d))

## [0.2.2](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/compare/v0.2.1...v0.2.2) (2026-04-09)


### Bug Fixes

* live mic example display issues ([#4](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/4)) ([150217a](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/150217a0db84e768b50295b543f68ddf09991335))

## [0.2.1](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/compare/v0.2.0...v0.2.1) (2026-04-07)


### Bug Fixes

* TTS support and transport improvements ([#2](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/issues/2)) ([1f9d19f](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/1f9d19fd078883fe8341b8b4cc94bc20c31a6251))

## [0.2.0](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/compare/v0.1.0...v0.2.0) (2026-02-17)


### Features

* initial deepgram-sagemaker transport package ([8ff6a7d](https://github.com/deepgram/deepgram-python-sdk-transport-sagemaker/commit/8ff6a7d37f25c5a1226144fb65ca953af0819bec))

## Changelog
