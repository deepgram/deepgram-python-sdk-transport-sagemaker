# SageMaker load test (Python)

End-to-end load harness that streams a known WAV across N concurrent
connections through the Deepgram Python SDK + the SageMaker transport, and
reports per-connection WER against a reference transcript. Validates that
burst-load fixes to the transport stay within 1 percentage point of the
single-concurrency baseline WER.

Mirrors the Java reference harness flag-for-flag so numbers compare
directly across languages.

## Install

This package's normal `pip install` covers the runtime deps. The load test
needs both the SDK and the transport on the same venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e /path/to/deepgram-python-sdk
pip install -e /path/to/deepgram-python-sdk-transport-sagemaker
```

Both repos need to be on the branches that include the reconnect /
storm-absorption work for the burst path to be exercised.

## AWS credentials

The transport resolves credentials via boto3's chain (env vars, shared
credentials file, IAM role). Use whatever you'd normally use for the AWS
dev account, e.g.:

```bash
export AWS_SHARED_CREDENTIALS_FILE=$HOME/.aws/creds.dev
```

## Generate a reference transcript

The WER pass needs a ground-truth transcript to compare each connection
against. If you don't already have one, run the harness at
`--connections 1` and capture what the model emits:

```bash
python -m loadtest <endpoint-name> \
    --file ./english.wav \
    --connections 1 \
    --region us-east-2 \
    --write-reference ./english.txt
```

This dumps the model's transcript to `./english.txt`. Subsequent
high-concurrency runs will measure WER against that file.

## Run the burst test

Matches the Java reference harness's single-process burst configuration:

```bash
python -m loadtest <endpoint-name> \
    --file ./english.wav \
    --reference ./english.txt \
    --connections 400 \
    --no-loop \
    --region us-east-2 \
    --transcripts-dir /tmp/proc-01-transcripts
```

Output: dashboard line every 2 s, then a summary table, per-connection
`conn-NNNN.txt` files in `--transcripts-dir`, a WER report per connection,
and `summary.csv` with `conn_id, errored, transcripts, chunks, duration_s,
wer_pct, note` columns.

## Sharded variant

For the 1000-connection / 10-instance sharded configuration, spawn N parallel
processes, each with its own slice of connections and output directory:

```bash
for i in $(seq 1 20); do
    python -m loadtest <endpoint-name> \
        --file ./english.wav \
        --reference ./english.txt \
        --connections 50 \
        --no-loop \
        --region us-east-2 \
        --transcripts-dir /tmp/proc-$(printf '%02d' "$i")-transcripts &
done
wait
```

## Success criterion

Every connection's WER should land within 1 percentage point of the
single-concurrency baseline (the run that produced `english.txt`). If WER
spikes for any connection, the burst path is dropping audio mid-stream and
the storm-absorption logic needs another look.

**Wall-clock is not a stable metric.** The SDK does retry + buffering, so
some connections finish earlier than others. Don't chase wall-clock parity
across runs.

## CLI reference

```
python -m loadtest --help
```

Flag-for-flag with the Java reference. Notable flags:

- `--connections N` (default 1) -- simultaneous streams
- `--batch-size N` (default 0 = all at once) / `--batch-delay S` --
  stagger connection opens
- `--loop` / `--no-loop` (default no-loop, matches the typical high-burst
  production pattern of one play per stream)
- `--duration S` (default 0 = run until audio ends)
- `--max-retries N` (default 10) -- caller-side retry on top of the
  transport's own
- `--await-final-results S` (default 15) -- post-CloseStream flush window
- `--transcripts-dir DIR` -- per-connection transcript dump + summary.csv
- `--write-reference PATH` -- capture model transcripts at concurrency 1
- `--reference ''` -- explicitly disable WER (smoke runs)
