"""SageMaker load test CLI for the Python Deepgram SDK + sagemaker transport.

Mirrors the Java DeepgramSdkLoadTest reference harness flag-for-flag:

  python -m loadtest <endpoint-name> \\
      --file ./english.wav \\
      --reference ./english.txt \\
      --connections 400 \\
      --no-loop \\
      --region us-east-2 \\
      --transcripts-dir /tmp/proc-01-transcripts

The harness opens N concurrent streaming connections through the Deepgram
SDK and the SageMaker transport, streams the WAV in real-time on each,
captures final transcripts per connection, and reports per-connection WER
against a reference transcript. Same numbers shape as the Java run -- WER
within 1 percentage point of single-concurrency baseline is the success
criterion for a burst-load fix.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from deepgram import AsyncDeepgramClient
from deepgram_sagemaker import SageMakerConfig, SageMakerTransportFactory

from .connection import SdkStreamingConnection, build_connect_options
from .wer import compute_wer

logger = logging.getLogger("loadtest")


def _git_head(repo_path: Path) -> str:
    """Return a short branch + commit identifier for a checked-out repo,
    or an empty string if the path isn't inside a git repo."""
    head_file = repo_path / ".git" / "HEAD"
    if not head_file.exists():
        return ""
    try:
        head = head_file.read_text().strip()
    except OSError:
        return ""
    if head.startswith("ref: "):
        ref = head[5:]
        branch = ref.rsplit("/", 1)[-1]
        sha_file = repo_path / ".git" / ref
        if sha_file.exists():
            try:
                sha = sha_file.read_text().strip()[:7]
                return f"{branch} @ {sha}"
            except OSError:
                pass
        return branch
    return head[:7]


def _print_resolution_banner() -> None:
    """Print where the SDK + transport are being imported from so the user
    can eyeball whether the local branches are linked correctly."""
    try:
        import deepgram
        import deepgram_sagemaker
    except ImportError as exc:  # pragma: no cover -- only on a broken install
        sys.stderr.write(f"ERROR: {exc}\n")
        return
    sdk_path = Path(deepgram.__file__)
    tr_path = Path(deepgram_sagemaker.__file__)
    sys.stderr.write(f"SDK:        {sdk_path}\n")
    # Walk up to find the repo root (containing .git)
    for parent in sdk_path.parents:
        if (parent / ".git").exists():
            sys.stderr.write(f"            branch: {_git_head(parent)}\n")
            break
    sys.stderr.write(f"Transport:  {tr_path}\n")
    for parent in tr_path.parents:
        if (parent / ".git").exists():
            sys.stderr.write(f"            branch: {_git_head(parent)}\n")
            break


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dg-sdk-loadtest",
        description="Deepgram SDK SageMaker load test (Python)",
    )
    p.add_argument("endpoint_name", help="SageMaker endpoint name")
    p.add_argument(
        "--file", "-f", required=True, type=Path,
        help="Path to 16-bit PCM WAV file",
    )
    p.add_argument(
        "--reference", default=None,
        help=(
            "Path to a plain-text reference transcript used to compute WER "
            "per connection. Defaults to <wav-stem>.txt next to the WAV. "
            "Pass '' to disable."
        ),
    )
    p.add_argument(
        "--connections", "-c", type=int, default=1,
        help="Total simultaneous streaming connections (default: 1)",
    )
    p.add_argument(
        "--batch-size", type=int, default=0,
        help="Connections to open per batch (0 = all at once, default: 0)",
    )
    p.add_argument(
        "--batch-delay", type=float, default=0.0,
        help="Seconds to wait between batches (default: 0)",
    )
    loop_group = p.add_mutually_exclusive_group()
    loop_group.add_argument(
        "--loop", dest="loop", action="store_true",
        help=(
            "Loop audio file continuously until --duration elapses. "
            "Default plays the file once per stream (matches the typical "
            "high-burst production usage pattern of one call per stream)."
        ),
    )
    loop_group.add_argument(
        "--no-loop", dest="loop", action="store_false",
        help="Play the file once per stream (default).",
    )
    p.set_defaults(loop=False)
    p.add_argument(
        "--duration", type=int, default=0,
        help="Stop after N seconds (0 = run until audio ends or Ctrl+C, default: 0)",
    )
    p.add_argument(
        "--region", default="us-west-2",
        help="AWS region (default: us-west-2)",
    )
    p.add_argument(
        "--max-retries", type=int, default=10,
        help="Max retries per connection on retryable errors (default: 10)",
    )
    p.add_argument(
        "--await-final-results", type=float, default=15.0,
        help=(
            "After send_close_stream, wait up to N seconds for the model to "
            "flush remaining transcripts and close. Returns sooner if the "
            "server closes early. Bump when retries cause large model "
            "backlogs and the default clips tail-end transcripts (default: 15)."
        ),
    )
    p.add_argument(
        "--model", default="nova-3",
        help="Deepgram model (default: nova-3; use flux-general-en for Flux)",
    )
    p.add_argument(
        "--service", default="listen.v1", choices=["listen.v1", "listen.v2"],
        help="listen.v1 (default, Nova/Nova-3 STT) or listen.v2 (Flux turn-based)",
    )
    p.add_argument(
        "--interim-results", action="store_true", default=False,
        help="Enable interim/partial results (default: off; listen.v1 only)",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARN", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    p.add_argument(
        "--transcripts-dir", default=None,
        help=(
            "Directory to write per-connection final transcripts "
            "(conn-NNNN.txt) plus summary.csv with WER per connection. "
            "Pass '-' to disable."
        ),
    )
    p.add_argument(
        "--write-reference", default=None,
        help=(
            "After a --connections 1 run, write the captured final "
            "transcripts as the reference file at the given path. Useful "
            "for generating a ground-truth transcript at concurrency 1 "
            "before the high-concurrency run."
        ),
    )
    return p


def _configure_logging(level_name: str) -> None:
    if level_name.upper() == "WARN":
        level_name = "WARNING"
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _print_wav_info(wav: Path) -> None:
    try:
        with wave.open(str(wav), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            duration = frames / rate if rate else 0
            sys.stderr.write(
                f"WAV: {wav.name} | {rate} Hz | {channels}ch | {duration:.2f}s\n"
            )
    except Exception:
        sys.stderr.write(f"WAV: {wav} (could not read format)\n")


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


async def _dashboard(connections, start: float, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
            return
        except asyncio.TimeoutError:
            pass
        if not connections:
            continue
        active = sum(1 for c in connections if c.stats.active)
        errored = sum(1 for c in connections if c.stats.errored)
        transcripts = sum(c.stats.transcript_count for c in connections)
        chunks = sum(c.stats.chunk_count for c in connections)
        retries = sum(c.stats.retry_count for c in connections)
        elapsed = time.monotonic() - start
        sys.stderr.write(
            f"\r[{_format_duration(elapsed)}] Active: {active}/{len(connections)} | "
            f"Errored: {errored} | Retries: {retries} | Transcripts: {transcripts} | "
            f"Chunks: {chunks}    "
        )
        sys.stderr.flush()


def _write_reference_if_requested(connections, path: str | None) -> None:
    if not path:
        return
    parts: list[str] = []
    for c in connections:
        for t in c.stats.final_transcripts:
            if t:
                parts.append(t)
    body = " ".join(parts)
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    sys.stderr.write(f"\nWrote reference transcript to {out.resolve()} ({len(body)} chars)\n")


def _write_transcripts_if_requested(connections, dir_path: str | None) -> Path | None:
    if not dir_path or dir_path == "-":
        return None
    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in connections:
        body = " ".join(c.stats.final_transcripts)
        (out_dir / f"conn-{c.connection_id:04d}.txt").write_text(body)
    sys.stderr.write(
        f"\nWrote {len(connections)} per-connection transcript file(s) to {out_dir.resolve()}\n"
    )
    return out_dir


def _load_reference(reference_arg: str | None, wav_path: Path) -> str | None:
    """Return reference text, or None when WER should be skipped."""
    if reference_arg is not None:
        if reference_arg == "":
            return None  # explicit disable
        path = Path(reference_arg)
    else:
        path = wav_path.with_suffix(".txt")
        if not path.exists():
            logger.debug("No reference transcript at %s; skipping WER", path)
            return None
    try:
        return path.read_text()
    except OSError as exc:
        logger.warning("Reference transcript not readable at %s: %s; skipping WER", path, exc)
        return None


def _wer_worker(args):
    conn_id, errored, hypothesis, reference = args
    if not hypothesis.strip():
        return (conn_id, None, "errored" if errored else "no transcript")
    wer = compute_wer(reference, hypothesis)
    if wer is None:
        return (conn_id, None, "errored" if errored else "no transcript")
    return (conn_id, wer, "")


def _print_wer_report(connections, reference: str, dump_dir: Path | None) -> None:
    sys.stderr.write("\nComputing WER in parallel...\n")
    worker_input = [
        (
            c.connection_id,
            c.stats.errored,
            " ".join(c.stats.final_transcripts),
            reference,
        )
        for c in connections
    ]
    workers = max(1, min(os.cpu_count() or 1, len(connections)))
    results: list[tuple[int, float | None, str]] = []
    if workers == 1 or len(connections) == 1:
        results = [_wer_worker(i) for i in worker_input]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_wer_worker, worker_input))
    results.sort(key=lambda r: r[0])

    sys.stderr.write("\nWER per connection (vs. reference transcript):\n")
    for conn_id, wer, note in results:
        if wer is None:
            sys.stderr.write(f"  [Conn {conn_id:4d}] -    ({note})\n")
        else:
            sys.stderr.write(f"  [Conn {conn_id:4d}] {wer * 100:.2f}%\n")

    values = sorted(w for _, w, _ in results if w is not None)
    if values:
        mean = sum(values) / len(values)
        mid = len(values) // 2
        median = values[mid] if len(values) % 2 == 1 else (values[mid - 1] + values[mid]) / 2
        p95_idx = max(0, int(len(values) * 0.95) - 1)
        sys.stderr.write(
            f"\n  Mean WER:   {mean * 100:.2f}%\n"
            f"  Median WER: {median * 100:.2f}%\n"
            f"  Min WER:    {values[0] * 100:.2f}%\n"
            f"  P95 WER:    {values[p95_idx] * 100:.2f}%\n"
            f"  Max WER:    {values[-1] * 100:.2f}%\n"
        )

    if dump_dir is not None:
        results_by_id = {cid: (cid, w, n) for cid, w, n in results}
        csv_path = dump_dir / "summary.csv"
        with csv_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                ["conn_id", "errored", "transcripts", "chunks", "duration_s", "wer_pct", "note"]
            )
            for c in connections:
                _, w, note = results_by_id.get(c.connection_id, (c.connection_id, None, ""))
                wer_pct = "" if w is None else f"{w * 100:.4f}"
                writer.writerow(
                    [
                        c.connection_id,
                        c.stats.errored,
                        c.stats.transcript_count,
                        c.stats.chunk_count,
                        f"{c.stats.duration_seconds:.2f}",
                        wer_pct,
                        note.replace(",", ";"),
                    ]
                )
        sys.stderr.write(f"\nWrote per-connection summary to {csv_path.resolve()}\n")


def _print_summary(connections, wall_time: float) -> None:
    sys.stderr.write("\n\n=== STREAM SUMMARY ===\n")
    successful = sum(1 for c in connections if not c.stats.errored)
    errored = sum(1 for c in connections if c.stats.errored)
    total_transcripts = sum(c.stats.transcript_count for c in connections)
    total_chunks = sum(c.stats.chunk_count for c in connections)
    total_retries = sum(c.stats.retry_count for c in connections)
    sys.stderr.write(f"Total connections:  {len(connections)}\n")
    sys.stderr.write(f"Successful:         {successful}\n")
    sys.stderr.write(f"Errored:            {errored}\n")
    sys.stderr.write(f"Total retries:      {total_retries}\n")
    sys.stderr.write(f"Total transcripts:  {total_transcripts}\n")
    sys.stderr.write(f"Total chunks sent:  {total_chunks}\n")
    sys.stderr.write(f"Wall time:          {wall_time:.2f}s\n\n")

    durations = sorted(
        c.stats.duration_seconds for c in connections if not c.stats.errored
    )
    if durations:
        sys.stderr.write("--- Connection Durations (successful) ---\n")
        sys.stderr.write(f"Min:    {durations[0]:.2f}s\n")
        sys.stderr.write(f"Median: {durations[len(durations) // 2]:.2f}s\n")
        sys.stderr.write(f"Max:    {durations[-1]:.2f}s\n")
        sys.stderr.write(f"Mean:   {sum(durations) / len(durations):.2f}s\n")

    if errored:
        sys.stderr.write("\n--- Errors ---\n")
        counts: dict[str, int] = {}
        for c in connections:
            if not c.stats.errored:
                continue
            for m in c.stats.error_messages:
                key = (m or "")[:120]
                counts[key] = counts.get(key, 0) + 1
        for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            sys.stderr.write(f"  [{count}x] {key}\n")


async def _run(args: argparse.Namespace) -> int:
    _configure_logging(args.log_level)
    if not args.file.exists():
        sys.stderr.write(f"ERROR: WAV file not found: {args.file}\n")
        return 1
    if args.connections < 1:
        sys.stderr.write("ERROR: --connections must be >= 1\n")
        return 1

    effective_batch_size = args.batch_size if args.batch_size > 0 else args.connections

    sys.stderr.write("=== Deepgram SDK SageMaker Load Test (Python) ===\n")
    sys.stderr.write(f"Endpoint:       {args.endpoint_name}\n")
    sys.stderr.write(f"WAV file:       {args.file}\n")
    sys.stderr.write(f"Connections:    {args.connections}\n")
    sys.stderr.write(f"Batch size:     {effective_batch_size}\n")
    sys.stderr.write(f"Batch delay:    {args.batch_delay}s\n")
    sys.stderr.write(f"Region:         {args.region}\n")
    sys.stderr.write(f"Model:          {args.model}\n")
    sys.stderr.write(f"Loop:           {args.loop}\n")
    sys.stderr.write(f"Max retries:    {args.max_retries}\n")
    _print_resolution_banner()
    sys.stderr.write("\n")
    _print_wav_info(args.file)

    with wave.open(str(args.file), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()

    connect_options = build_connect_options(
        service=args.service,
        model=args.model,
        sample_rate=sample_rate,
        channels=channels,
        interim_results=args.interim_results,
    )

    # One shared factory + client for the whole process. The Python SDK's
    # install_transport() patches the websocket layer process-wide, so
    # multiple AsyncDeepgramClient instances with different transport
    # factories can't coexist. Each connect() still gets its own underlying
    # SageMakerTransport instance (via factory.__call__), so stream
    # isolation is preserved.
    #
    # Load-test retry tuning: dialed far above the SDK defaults so a single
    # transient AWS-side condition doesn't end the connection. NOT
    # production defaults.
    sm_config = SageMakerConfig(
        endpoint_name=args.endpoint_name,
        region=args.region,
        max_retries=100,
        max_backoff=30.0,
        retry_budget=3600.0,
    )
    factory = SageMakerTransportFactory(config=sm_config)
    shared_client = AsyncDeepgramClient(api_key="unused", transport_factory=factory)

    connections: list[SdkStreamingConnection] = []

    # Default loop-stop policy: when --duration is unset and --loop is on,
    # stop all connections once every connection has completed one full read
    # of the WAV file. Bounds the run without Ctrl+C.
    stop_on_all_first_passes = args.loop and args.duration == 0
    first_pass_count = [0]
    stop_event_dashboard = asyncio.Event()

    def on_first_pass():
        first_pass_count[0] += 1
        if stop_on_all_first_passes and first_pass_count[0] >= args.connections:
            sys.stderr.write(
                f"\nAll {args.connections} connection(s) completed one full pass. Stopping...\n"
            )
            for c in connections:
                c.stop()

    test_start = time.monotonic()
    num_batches = (args.connections + effective_batch_size - 1) // effective_batch_size
    sys.stderr.write(
        f"Opening {args.connections} connection(s) in {num_batches} batch(es) "
        "(one SageMaker client per connection)...\n"
    )

    dashboard_task = asyncio.create_task(
        _dashboard(connections, test_start, stop_event_dashboard)
    )

    duration_task = None
    if args.duration > 0:
        async def _stop_after_duration():
            await asyncio.sleep(args.duration)
            sys.stderr.write(f"\nDuration limit reached ({args.duration}s). Stopping...\n")
            for c in connections:
                c.stop()
        duration_task = asyncio.create_task(_stop_after_duration())

    run_tasks: list[asyncio.Task] = []
    try:
        for batch_start in range(0, args.connections, effective_batch_size):
            batch_end = min(batch_start + effective_batch_size, args.connections)
            batch_num = batch_start // effective_batch_size + 1
            sys.stderr.write(
                f"Opening batch {batch_num}/{num_batches}: "
                f"connections {batch_start + 1}-{batch_end}...\n"
            )

            for i in range(batch_start, batch_end):
                conn = SdkStreamingConnection(
                    connection_id=i + 1,
                    client=shared_client,
                    service=args.service,
                    wav_path=args.file,
                    connect_options=connect_options,
                    loop=args.loop,
                    max_retries=args.max_retries,
                    await_final_results_s=args.await_final_results,
                    on_first_pass_complete=on_first_pass,
                )
                connections.append(conn)
                run_tasks.append(asyncio.create_task(conn.run()))

            if batch_end < args.connections and args.batch_delay > 0:
                await asyncio.sleep(args.batch_delay)

        sys.stderr.write(
            f"All {args.connections} connection(s) launched. Streaming...\n\n"
        )

        await asyncio.gather(*run_tasks, return_exceptions=True)
    finally:
        stop_event_dashboard.set()
        if duration_task is not None:
            duration_task.cancel()
            try:
                await duration_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await dashboard_task
        except (asyncio.CancelledError, Exception):
            pass
        for c in connections:
            c.stop()

    wall_time = time.monotonic() - test_start
    _print_summary(connections, wall_time)

    # Dump transcripts / write reference BEFORE WER so the artifacts are on
    # disk even if WER is skipped.
    _write_reference_if_requested(connections, args.write_reference)
    dump_dir = _write_transcripts_if_requested(connections, args.transcripts_dir)

    reference = _load_reference(args.reference, args.file)
    if reference is not None:
        _print_wer_report(connections, reference, dump_dir)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
