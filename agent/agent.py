#!/usr/bin/env python3
"""
AluminatAI GPU Agent v0.2.0 — unified production daemon.

Combines the signal-handling / CSV reliability of aluminati_agent.py with
the attribution engine, scheduler detection, and API uploader from main.py.

Usage:
    aluminatai-agent                        # reads env vars, runs forever
    aluminatai-agent --interval 2           # 0.5 Hz sampling
    aluminatai-agent --duration 3600        # run 1 h then exit 0
    aluminatai-agent --output /data/m.csv   # local CSV manifest too
    aluminatai-agent --help

Signal handling:
    SIGINT / SIGTERM → flush buffer → close CSV → signal_job_complete → exit 0
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import signal
import socket
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("aluminatai-agent")

# ── Optional rich console ──────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False

# ── Local module imports (all optional for graceful degradation) ───────────────

try:
    from collector import GPUCollector, CSV_HEADER
    _COLLECTOR = True
except ImportError:
    _COLLECTOR = False
    log.warning("collector.py not found — NVML collection disabled")

try:
    from uploader import MetricsUploader
    from config import (
        UPLOAD_ENABLED, UPLOAD_INTERVAL, API_KEY, API_ENDPOINT,
        SCHEDULER_POLL_INTERVAL, SAMPLE_INTERVAL,
        AGENT_VERSION, HEARTBEAT_INTERVAL,
    )
    _UPLOADER = True
except ImportError:
    _UPLOADER = False
    UPLOAD_ENABLED = False
    API_KEY = ""
    API_ENDPOINT = "https://aluminatiai.com/api/metrics/ingest"
    UPLOAD_INTERVAL = 60
    SCHEDULER_POLL_INTERVAL = 30
    SAMPLE_INTERVAL = 5.0
    AGENT_VERSION = "0.2.0"
    HEARTBEAT_INTERVAL = 300

try:
    from schedulers import detect_scheduler
    _SCHEDULER = True
except ImportError:
    _SCHEDULER = False

try:
    from attribution import AttributionEngine
    from attribution.process_probe import ProcessProbe
    from attribution.pid_resolver import PidResolver
    _ATTRIBUTION = True
except ImportError:
    _ATTRIBUTION = False

try:
    from metrics_server import MetricsServer
    _METRICS_SERVER = True
except ImportError:
    _METRICS_SERVER = False

# ── ManifestWriter — atomic-flush CSV output ──────────────────────────────────

CSV_MANIFEST_COLUMNS = [
    "timestamp", "job_id", "gpu_uuid", "gpu_index", "gpu_name",
    "power_w", "energy_delta_j", "util_pct", "temp_c",
    "mem_used_mb", "team_id", "model_tag", "gpu_fraction",
    "attribution_confidence",
]


class ManifestWriter:
    """Append-only CSV manifest with line-buffering + fsync on close."""

    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[io.TextIOWrapper] = None
        self._writer = None
        self._row_count = 0

    def open(self):
        self._file = open(self._path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_MANIFEST_COLUMNS)
        self._file.flush()
        log.info("Manifest CSV: %s", self._path)

    def write_row(self, row: list):
        if self._writer:
            self._writer.writerow(row)
            self._row_count += 1

    def flush(self):
        if self._file and not self._file.closed:
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except OSError:
                pass

    def close(self):
        if self._file and not self._file.closed:
            self.flush()
            self._file.close()
            log.info("Manifest closed: %s (%d rows)", self._path, self._row_count)

    @property
    def row_count(self) -> int:
        return self._row_count


# ── Job completion signal ─────────────────────────────────────────────────────


def signal_job_complete(endpoint: str, api_key: str, job_uuid: str,
                        end_time: Optional[str] = None) -> bool:
    url = endpoint.rstrip("/") + "/api/metrics/jobs/complete"
    payload: dict = {"job_id": job_uuid}
    if end_time:
        payload["end_time"] = end_time
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            log.info("Job complete signalled: job=%s msg=%s", job_uuid, body.get("message", ""))
            return True
    except Exception as exc:
        log.warning("Job complete signal failed (non-fatal): %s", exc)
        return False


# ── Heartbeat sender ──────────────────────────────────────────────────────────


def send_heartbeat(endpoint: str, api_key: str, gpu_count: int,
                   gpu_uuids: List[str], scheduler_name: str) -> None:
    url = endpoint.rstrip("/") + "/api/agent/heartbeat"
    payload = {
        "agent_version": AGENT_VERSION,
        "hostname": socket.gethostname(),
        "gpu_count": gpu_count,
        "gpu_uuids": gpu_uuids,
        "scheduler": scheduler_name,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as exc:
        log.debug("Heartbeat failed (non-fatal): %s", exc)


# ── Unified Agent ─────────────────────────────────────────────────────────────


class Agent:
    """
    Production GPU energy monitoring daemon.

    Poll loop:
      1. Poll scheduler every SCHEDULER_POLL_INTERVAL
      2. Collect NVML metrics via GPUCollector
      3. AttributionEngine.resolve() per GPU handle
      4. Append fractional rows to uploader buffer + CSV manifest
      5. Flush uploader every UPLOAD_INTERVAL
      6. Send heartbeat every HEARTBEAT_INTERVAL
      7. On SIGTERM/SIGINT: flush → close CSV → signal_job_complete → exit 0
    """

    def __init__(
        self,
        interval: float = SAMPLE_INTERVAL,
        output_csv: Optional[str] = None,
        duration: Optional[float] = None,
        quiet: bool = False,
        job_uuid: Optional[str] = None,
    ):
        self.interval = interval
        self.output_csv = output_csv
        self.duration = duration
        self.quiet = quiet
        self.job_uuid = job_uuid or os.getenv("ALUMINATAI_JOB_UUID")

        self.running = False
        self.sample_count = 0
        self.total_energy: dict[int, float] = {}

        # Upload
        self.uploader: Optional[MetricsUploader] = None
        self.last_upload_time = 0.0
        if _UPLOADER and UPLOAD_ENABLED and API_KEY:
            self.uploader = MetricsUploader()
            log.info("API upload enabled → %s", API_ENDPOINT)
        elif not quiet:
            log.info("API upload disabled (no API key)")

        # Scheduler
        self.scheduler = None
        self.last_scheduler_poll = 0.0
        if _SCHEDULER:
            self.scheduler = detect_scheduler()
            log.info("Scheduler: %s", self.scheduler.name)
        elif not quiet:
            log.info("Scheduler integration unavailable")

        # Attribution engine
        self.attribution_engine = None
        if _ATTRIBUTION and self.scheduler:
            probe = ProcessProbe()
            resolver = PidResolver(self.scheduler)
            self.attribution_engine = AttributionEngine(probe, resolver, self.scheduler)
            log.info("Attribution: process-level GPU attribution enabled")

        # Prometheus metrics server
        self.metrics_server = None
        if _METRICS_SERVER:
            self.metrics_server = MetricsServer()
            self.metrics_server.start()

        # Rich console
        self.console = Console() if (_RICH and not quiet) else None

        # Heartbeat state
        self.last_heartbeat = 0.0

        # Signal handlers
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    # ── Signal handling ──────────────────────────────────────────────────

    def _on_signal(self, signum, _frame):
        sig_name = signal.Signals(signum).name
        log.info("Received %s — shutting down gracefully.", sig_name)
        self.running = False

    # ── Main run loop ────────────────────────────────────────────────────

    def run(self) -> int:
        if not _COLLECTOR:
            log.error("GPUCollector unavailable — install nvidia-ml-py3")
            return 3

        try:
            collector = GPUCollector(collect_clocks=False)
        except Exception as exc:
            log.error("Failed to initialize GPU collector: %s", exc)
            return 3

        gpu_count = collector.get_gpu_count()
        gpu_uuids = [info["uuid"] for info in collector.get_gpu_info()]
        scheduler_name = self.scheduler.name if self.scheduler else "none"

        for i in range(gpu_count):
            self.total_energy[i] = 0.0

        # Replay WAL on startup
        if self.uploader:
            retried = self.uploader.retry_failed_uploads()
            if retried > 0:
                log.info("WAL replay: %d metrics re-uploaded", retried)

        # CSV manifest
        manifest: Optional[ManifestWriter] = None
        if self.output_csv:
            manifest = ManifestWriter(Path(self.output_csv))
            manifest.open()

        if not self.quiet:
            self._print_banner(gpu_count, scheduler_name)

        self.running = True
        start_time = time.monotonic()
        self.last_upload_time = time.time()

        # Initial heartbeat
        if self.uploader and API_KEY:
            send_heartbeat(API_ENDPOINT, API_KEY, gpu_count, gpu_uuids, scheduler_name)
            self.last_heartbeat = time.time()

        try:
            while self.running:
                loop_start = time.monotonic()
                now = time.time()

                # Scheduler poll
                if self.scheduler and (now - self.last_scheduler_poll >= SCHEDULER_POLL_INTERVAL):
                    try:
                        self.scheduler.discover_jobs()
                        self.last_scheduler_poll = now
                    except Exception as exc:
                        log.warning("Scheduler poll failed: %s", exc)

                # Collect
                try:
                    metrics = collector.collect()
                except Exception as exc:
                    log.warning("Collection error: %s", exc)
                    time.sleep(self.interval)
                    continue

                self.sample_count += 1

                # Attribution + upload buffering
                attributed_rows: list[dict] = []
                for m in metrics:
                    handle = collector.gpu_handles[m.gpu_index]
                    if self.attribution_engine:
                        attributions = self.attribution_engine.resolve(
                            handle=handle,
                            gpu_index=m.gpu_index,
                            total_power_w=m.power_draw_w,
                            energy_delta_j=m.energy_delta_j,
                        )
                        if attributions:
                            for attr in attributions:
                                d = m.to_dict()
                                d.update(
                                    team_id=attr.team_id,
                                    model_tag=attr.model_tag,
                                    job_id=attr.job_id,
                                    scheduler_source=attr.scheduler_source,
                                    power_draw_w=attr.power_w,
                                    energy_delta_j=attr.energy_delta_j,
                                    gpu_fraction=attr.gpu_fraction,
                                    attribution_confidence=attr.confidence,
                                )
                                attributed_rows.append(d)
                                if manifest:
                                    manifest.write_row([
                                        d.get("timestamp"), d.get("job_id"),
                                        d.get("gpu_uuid"), d.get("gpu_index"),
                                        d.get("gpu_name"), d.get("power_draw_w"),
                                        d.get("energy_delta_j"), d.get("utilization_gpu_pct"),
                                        d.get("temperature_c"), d.get("memory_used_mb"),
                                        attr.team_id, attr.model_tag,
                                        attr.gpu_fraction, attr.confidence,
                                    ])
                        else:
                            d = m.to_dict()
                            attributed_rows.append(d)
                            if manifest:
                                manifest.write_row([
                                    d.get("timestamp"), d.get("job_id"),
                                    d.get("gpu_uuid"), d.get("gpu_index"),
                                    d.get("gpu_name"), d.get("power_draw_w"),
                                    d.get("energy_delta_j"), d.get("utilization_gpu_pct"),
                                    d.get("temperature_c"), d.get("memory_used_mb"),
                                    None, None, None, None,
                                ])
                    else:
                        if self.scheduler:
                            job = self.scheduler.gpu_to_job(m.gpu_index)
                            if job:
                                m.job_id = job.job_id
                                m.team_id = job.team_id
                                m.model_tag = job.model_tag
                                m.scheduler_source = job.scheduler_source
                        d = m.to_dict()
                        attributed_rows.append(d)
                        if manifest:
                            manifest.write_row([
                                d.get("timestamp"), d.get("job_id"),
                                d.get("gpu_uuid"), d.get("gpu_index"),
                                d.get("gpu_name"), d.get("power_draw_w"),
                                d.get("energy_delta_j"), d.get("utilization_gpu_pct"),
                                d.get("temperature_c"), d.get("memory_used_mb"),
                                d.get("team_id"), d.get("model_tag"), None, None,
                            ])

                if self.uploader and attributed_rows:
                    self.uploader.add_metrics(attributed_rows)

                # Prometheus metrics update
                if self.metrics_server:
                    self.metrics_server.update(metrics, attributed_rows)

                # Energy accumulation
                for m in metrics:
                    if m.energy_delta_j:
                        self.total_energy[m.gpu_index] += m.energy_delta_j / 3_600_000

                # Periodic upload flush
                if self.uploader and (time.time() - self.last_upload_time >= UPLOAD_INTERVAL):
                    n = self.uploader.flush()
                    self.last_upload_time = time.time()
                    if n and not self.quiet:
                        log.info("Uploaded %d metrics", n)

                # Periodic heartbeat
                if self.uploader and API_KEY and (time.time() - self.last_heartbeat >= HEARTBEAT_INTERVAL):
                    send_heartbeat(API_ENDPOINT, API_KEY, gpu_count, gpu_uuids, scheduler_name)
                    self.last_heartbeat = time.time()

                # Console display
                if not self.quiet and self.sample_count % 1 == 0:
                    self._display(metrics)

                # Duration guard
                if self.duration and (time.monotonic() - start_time) >= self.duration:
                    log.info("Duration limit reached (%.0fs).", self.duration)
                    break

                # Sleep remainder
                elapsed = time.monotonic() - loop_start
                sleep_s = max(0.0, self.interval - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except Exception as exc:
            log.exception("Unhandled error in agent loop: %s", exc)
            return 1

        finally:
            # Flush remaining metrics
            if self.uploader:
                remaining = len(self.uploader.buffer)
                if remaining > 0:
                    log.info("Flushing %d remaining metrics…", remaining)
                    self.uploader.flush()

            # Close CSV manifest (fsync)
            if manifest:
                manifest.close()

            # Shutdown collector
            try:
                collector.shutdown()
            except Exception:
                pass

            # Stop Prometheus server
            if self.metrics_server:
                self.metrics_server.stop()

            # Signal job completion
            if API_KEY and self.job_uuid:
                signal_job_complete(
                    endpoint=API_ENDPOINT,
                    api_key=API_KEY,
                    job_uuid=self.job_uuid,
                    end_time=datetime.now(timezone.utc).isoformat(),
                )

            if not self.quiet:
                self._print_summary(time.monotonic() - start_time)

        return 0

    # ── Display helpers ──────────────────────────────────────────────────

    def _print_banner(self, gpu_count: int, scheduler: str):
        log.info("=" * 60)
        log.info("  AluminatAI GPU Agent v%s", AGENT_VERSION)
        log.info("=" * 60)
        log.info("  GPUs        : %d", gpu_count)
        log.info("  Interval    : %.2fs", self.interval)
        log.info("  Scheduler   : %s", scheduler)
        log.info("  Attribution : %s", "process-level" if self.attribution_engine else "scheduler-poll")
        log.info("  Upload      : %s", "enabled" if self.uploader else "disabled")
        if self.duration:
            log.info("  Duration    : %.0fs", self.duration)
        if self.output_csv:
            log.info("  Manifest    : %s", self.output_csv)
        log.info("=" * 60)

    def _display(self, metrics):
        if self.console and _RICH:
            table = Table(title=f"Sample #{self.sample_count}")
            table.add_column("GPU", style="cyan")
            table.add_column("Power", justify="right")
            table.add_column("Util", justify="right")
            table.add_column("Temp", justify="right")
            table.add_column("Energy Δ", justify="right")
            table.add_column("Total kWh", justify="right")
            for m in metrics:
                e_str = f"{m.energy_delta_j:.1f}J" if m.energy_delta_j else "N/A"
                table.add_row(
                    f"GPU {m.gpu_index}",
                    f"{m.power_draw_w:.1f}W",
                    f"{m.utilization_gpu_pct}%",
                    f"{m.temperature_c}°C",
                    e_str,
                    f"{self.total_energy.get(m.gpu_index, 0):.4f}",
                )
            self.console.clear()
            self.console.print(table)

    def _print_summary(self, runtime: float):
        log.info("=" * 60)
        log.info("  SESSION SUMMARY")
        log.info("=" * 60)
        log.info("  Runtime  : %.1fs  Samples: %d", runtime, self.sample_count)
        total_all = 0.0
        for idx, kwh in sorted(self.total_energy.items()):
            log.info("  GPU %-2d   : %.6f kWh (%.1f J)", idx, kwh, kwh * 3_600_000)
            total_all += kwh
        log.info("  TOTAL    : %.6f kWh ($%.4f @ $0.12/kWh)", total_all, total_all * 0.12)
        log.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aluminatai-agent",
        description="AluminatAI GPU Energy Agent v0.2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables (override CLI defaults):
  ALUMINATAI_API_KEY        API key (alum_...)
  ALUMINATAI_API_ENDPOINT   Ingest URL
  ALUMINATAI_JOB_UUID       DB job UUID for completion signal
  SAMPLE_INTERVAL           Sampling interval in seconds
  UPLOAD_INTERVAL           Upload flush interval in seconds

Examples:
  aluminatai-agent
  aluminatai-agent --interval 2 --duration 3600
  aluminatai-agent --output /data/manifests/run.csv --quiet
        """,
    )
    parser.add_argument("--interval", "-i", type=float, default=None,
                        help="Sampling interval in seconds (default: SAMPLE_INTERVAL env or 5.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV manifest path")
    parser.add_argument("--duration", "-d", type=float, default=None,
                        help="Run for N seconds then exit 0 (default: infinite)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress console output")
    parser.add_argument("--job-uuid", type=str, default=None,
                        help="DB job UUID — signals completion on exit")
    parser.add_argument("--version", action="version", version=f"aluminatai-agent {AGENT_VERSION}")

    args = parser.parse_args()

    interval = args.interval if args.interval is not None else SAMPLE_INTERVAL
    if interval < 0.1:
        log.error("Interval must be >= 0.1s")
        return 2

    agent = Agent(
        interval=interval,
        output_csv=args.output,
        duration=args.duration,
        quiet=args.quiet,
        job_uuid=args.job_uuid,
    )
    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
