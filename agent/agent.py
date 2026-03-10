# Copyright 2026 Kevin (AluminatiAI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AluminatiAI — https://github.com/AgentMulder404/AluminatAI
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
import hashlib
import shutil
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

# ── Logging helpers ────────────────────────────────────────────────────────────

# Standard LogRecord attributes that should not be treated as user "extra" fields.
_STANDARD_LOG_ATTRS: frozenset = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "process", "processName", "taskName", "message", "asctime",
})


class _JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single-line JSON object for ELK / Grafana Loki.

    Standard fields: ts, level, logger, msg.
    Extra fields passed via logger.xxx(..., extra={...}) are merged in at the
    top level, enabling structured events like:
      {"ts":"…","level":"WARNING","event":"upload_timeout","attempt":2, …}
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge any extra={} fields the caller attached
        for key, val in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    """
    Configure the root logger with the requested level and format.

    Args:
        level: Standard logging level name ("DEBUG", "INFO", "WARNING", …).
        fmt:   "text" (human-readable) or "json" (newline-delimited JSON).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
    root.addHandler(handler)


# ── Config hash ───────────────────────────────────────────────────────────────


def _compute_config_hash() -> str:
    """
    Return an 8-char hex digest that changes when key agent config values drift.

    Useful for fleet-wide drift detection: if two agent nodes report the same
    version but different config_hash values in their heartbeats, their configs
    have diverged (e.g., one has a stale SAMPLE_INTERVAL).
    """
    try:
        from config import (
            API_ENDPOINT, SAMPLE_INTERVAL, UPLOAD_INTERVAL, UPLOAD_BATCH_SIZE,
            METRICS_PORT, WAL_MAX_MB, LOG_LEVEL, DRY_RUN, PROMETHEUS_ONLY, OFFLINE_MODE,
        )
    except ImportError:
        return "00000000"
    canonical = json.dumps({
        "api_endpoint":    API_ENDPOINT,
        "sample_interval": SAMPLE_INTERVAL,
        "upload_interval": UPLOAD_INTERVAL,
        "batch_size":      UPLOAD_BATCH_SIZE,
        "metrics_port":    METRICS_PORT,
        "wal_max_mb":      WAL_MAX_MB,
        "log_level":       LOG_LEVEL,
        "dry_run":         DRY_RUN,
        "prometheus_only": PROMETHEUS_ONLY,
        "offline_mode":    OFFLINE_MODE,
    }, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:8]


# Minimal early setup so import-time warnings are visible; overridden in main().
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
except (ImportError, SyntaxError) as _e:
    _COLLECTOR = False
    log.warning("collector.py unavailable (%s) — NVML collection disabled", type(_e).__name__)

try:
    from uploader import MetricsUploader
    from config import (
        UPLOAD_ENABLED, UPLOAD_INTERVAL, API_KEY, API_ENDPOINT,
        SCHEDULER_POLL_INTERVAL, SAMPLE_INTERVAL,
        AGENT_VERSION, HEARTBEAT_INTERVAL,
        DRY_RUN, PROMETHEUS_ONLY, LOG_LEVEL, LOG_FORMAT,
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
    AGENT_VERSION = "0.2.2"
    HEARTBEAT_INTERVAL = 300
    DRY_RUN = False
    PROMETHEUS_ONLY = False
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "text"

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


def send_heartbeat(
    endpoint: str,
    api_key: str,
    gpu_count: int,
    gpu_uuids: List[str],
    scheduler_name: str,
    uptime_sec: float = 0.0,
    config_hash: str = "",
) -> None:
    url = endpoint.rstrip("/") + "/api/agent/heartbeat"
    payload = {
        "agent_version": AGENT_VERSION,
        "hostname": socket.gethostname(),
        "gpu_count": gpu_count,
        "gpu_uuids": gpu_uuids,
        "scheduler": scheduler_name,
        "uptime_sec": round(uptime_sec, 1),
        "config_hash": config_hash,
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
        dry_run: bool = DRY_RUN,
        prometheus_only: bool = PROMETHEUS_ONLY,
    ):
        self.interval = interval
        self.output_csv = output_csv
        self.duration = duration
        self.quiet = quiet
        self.job_uuid = job_uuid or os.getenv("ALUMINATAI_JOB_UUID")
        self.dry_run = dry_run
        self.prometheus_only = prometheus_only

        self.running = False
        self.sample_count = 0
        self.total_energy: dict[int, float] = {}
        self._start_time = time.monotonic()
        self._config_hash = _compute_config_hash()

        if self.dry_run:
            log.warning("DRY RUN — collecting and attributing, but no data will be uploaded or written to WAL")
        if self.prometheus_only:
            log.warning("PROMETHEUS ONLY — cloud uploads disabled; Prometheus metrics served locally")

        # Upload — disabled in prometheus_only mode (dry_run still creates uploader for logging)
        self.uploader: Optional[MetricsUploader] = None
        self.last_upload_time = 0.0
        if self.prometheus_only:
            pass  # no uploader; Prometheus is the only sink
        elif _UPLOADER and UPLOAD_ENABLED and API_KEY:
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
            # Push initial WAL stats to Prometheus
            if self.metrics_server:
                status = self.uploader.get_status()
                self.metrics_server.update_wal_stats(
                    wal_size_bytes=status["wal_bytes"],
                    wal_entries_pending=status["wal_entries_pending"],
                    replay_uploaded_delta=self.uploader._wal_replay_uploaded,
                    replay_failed_delta=self.uploader._wal_replay_failed,
                )

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

        # Initial heartbeat (skipped in dry-run / prometheus-only modes)
        if self.uploader and API_KEY and not self.dry_run and not self.prometheus_only:
            send_heartbeat(
                API_ENDPOINT, API_KEY, gpu_count, gpu_uuids, scheduler_name,
                uptime_sec=0.0,
                config_hash=self._config_hash,
            )
            self.last_heartbeat = time.time()

        # Determine run mode label for Prometheus agent_info
        _mode = "dry_run" if self.dry_run else ("prometheus_only" if self.prometheus_only else "normal")
        _hostname = socket.gethostname()

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
                            if self.metrics_server:
                                self.metrics_server.record_attribution_unresolved()
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

                # Prometheus metrics update (GPU + attribution)
                if self.metrics_server:
                    self.metrics_server.update(metrics, attributed_rows)
                    # Agent uptime and info — updated every collection cycle
                    self.metrics_server.update_agent_stats(
                        uptime_sec=time.monotonic() - self._start_time,
                        version=AGENT_VERSION,
                        hostname=_hostname,
                        mode=_mode,
                    )

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
                    # Update WAL + upload stats in Prometheus after every flush
                    if self.metrics_server:
                        status = self.uploader.get_status()
                        self.metrics_server.update_upload_stats(
                            success_delta=0,   # counters cumulative — delta tracked by uploader
                            failure_delta=0,
                            buffer_size=status["buffer_size"],
                        )
                        self.metrics_server.update_wal_stats(
                            wal_size_bytes=status["wal_bytes"],
                            wal_entries_pending=status["wal_entries_pending"],
                        )

                # Periodic heartbeat (skipped in dry-run / prometheus-only modes)
                if (self.uploader and API_KEY
                        and not self.dry_run and not self.prometheus_only
                        and time.time() - self.last_heartbeat >= HEARTBEAT_INTERVAL):
                    send_heartbeat(
                        API_ENDPOINT, API_KEY, gpu_count, gpu_uuids, scheduler_name,
                        uptime_sec=time.monotonic() - self._start_time,
                        config_hash=self._config_hash,
                    )
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

            # Signal job completion (skipped in dry-run / prometheus-only modes)
            if API_KEY and self.job_uuid and not self.dry_run and not self.prometheus_only:
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
        if self.dry_run:
            log.info("  Mode        : DRY RUN (no uploads, no WAL)")
        elif self.prometheus_only:
            log.info("  Mode        : PROMETHEUS ONLY (no cloud uploads)")
        else:
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


# ── Replay subcommand ─────────────────────────────────────────────────────────


_SERVICE_UNIT = """\
[Unit]
Description=AluminatAI GPU Energy Monitoring Agent
Documentation=https://aluminatiai.com/docs/agent
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=120
StartLimitBurst=5

[Service]
Type=simple
User=aluminatai
Group=aluminatai
EnvironmentFile=/etc/aluminatai/agent.env
Environment=DATA_DIR=/var/lib/aluminatai
Environment=LOG_DIR=/var/log/aluminatai
ExecStart={bin_path}
Restart=on-failure
RestartSec=10s
TimeoutStopSec=30s
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/aluminatai /var/log/aluminatai
SystemCallFilter=@system-service
SystemCallErrorNumber=EPERM
CapabilityBoundingSet=
MemoryMax=256M
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
"""

_ENV_TEMPLATE = """\
# AluminatAI Agent Configuration
# Edit this file, then restart: sudo systemctl restart aluminatai-agent
# Full reference: https://aluminatiai.com/docs/agent#configuration

ALUMINATAI_API_KEY={api_key}
ALUMINATAI_API_ENDPOINT=https://aluminatiai.com/api/metrics/ingest
SAMPLE_INTERVAL=5.0
UPLOAD_INTERVAL=60
METRICS_PORT=9100
LOG_LEVEL=INFO
"""

_UNIT_PATH = Path("/etc/systemd/system/aluminatai-agent.service")
_ENV_PATH  = Path("/etc/aluminatai/agent.env")


def _cmd_service(args) -> int:
    """service install | uninstall | status."""
    action = args.service_action

    if action == "status":
        ret = os.system("systemctl status aluminatai-agent")
        return 0 if ret == 0 else 1

    if action == "uninstall":
        if os.geteuid() != 0:
            print("error: 'service uninstall' must be run as root (try: sudo aluminatiai service uninstall)")
            return 1
        os.system("systemctl stop aluminatai-agent 2>/dev/null")
        os.system("systemctl disable aluminatai-agent 2>/dev/null")
        for path in [_UNIT_PATH]:
            if path.exists():
                path.unlink()
                print(f"Removed {path}")
        os.system("systemctl daemon-reload")
        print("Service uninstalled.  Config and data directories were NOT removed.")
        print(f"  Config: {_ENV_PATH}  (remove manually if desired)")
        return 0

    # install
    if os.geteuid() != 0:
        print("error: 'service install' must be run as root (try: sudo aluminatiai service install)")
        return 1

    if not hasattr(args, "api_key") or not args.api_key:
        existing_key = ""
        if _ENV_PATH.exists():
            for line in _ENV_PATH.read_text().splitlines():
                if line.startswith("ALUMINATAI_API_KEY="):
                    existing_key = line.split("=", 1)[1].strip()
                    break
        if existing_key:
            print(f"Found existing API key in {_ENV_PATH}")
            api_key = existing_key
        else:
            print("Get your API key at: https://aluminatiai.com/dashboard/setup")
            try:
                api_key = input("Enter API Key: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.")
                return 1
            if not api_key:
                print("error: API key cannot be empty")
                return 1

    else:
        api_key = args.api_key

    # Find the installed binary
    bin_path = sys.executable.replace("python", "aluminatiai").replace("python3", "aluminatiai")
    bin_path = shutil.which("aluminatiai") or bin_path

    # Create user if missing
    if os.system("id aluminatai &>/dev/null") != 0:
        os.system("useradd --system --no-create-home --shell /usr/sbin/nologin "
                  "--comment 'AluminatAI GPU agent' aluminatai")
        print("Created system user 'aluminatai'")

    # Directories
    for d, mode in [
        (Path("/var/lib/aluminatai"), 0o700),
        (Path("/var/log/aluminatai"), 0o755),
        (Path("/etc/aluminatai"),     0o750),
    ]:
        d.mkdir(parents=True, exist_ok=True, mode=mode)

    # Env file (write only if not present or explicitly updating)
    if not _ENV_PATH.exists() or getattr(args, "update_env", False):
        _ENV_PATH.write_text(_ENV_TEMPLATE.format(api_key=api_key))
        _ENV_PATH.chmod(0o600)
        print(f"Wrote {_ENV_PATH}")
    else:
        print(f"Keeping existing {_ENV_PATH} (pass --update-env to overwrite)")

    # Unit file
    _UNIT_PATH.write_text(_SERVICE_UNIT.format(bin_path=bin_path))
    _UNIT_PATH.chmod(0o644)
    print(f"Wrote {_UNIT_PATH}")

    os.system("systemctl daemon-reload")
    os.system("systemctl enable aluminatai-agent")
    os.system("systemctl restart aluminatai-agent")

    import time as _t
    _t.sleep(3)

    active = os.system("systemctl is-active --quiet aluminatai-agent") == 0
    if active:
        print("\nAluminatAI Agent is running!")
        print("  Status:    sudo systemctl status aluminatai-agent")
        print("  Logs:      sudo journalctl -u aluminatai-agent -f")
        print("  Metrics:   curl -s localhost:9100/metrics | head -20")
        print("  Dashboard: https://aluminatiai.com/dashboard")
        return 0
    else:
        print("\nService failed to start.")
        print("  Logs: sudo journalctl -u aluminatai-agent -n 50")
        return 1


def _cmd_replay(args) -> int:
    """Export WAL contents to a CSV file, optionally clearing the WAL."""
    try:
        from uploader import _wal_read_valid, _wal_clear
    except ImportError:
        log.error("uploader.py not available — cannot replay WAL")
        return 1

    rows = _wal_read_valid()
    if not rows:
        print("WAL is empty.")
        return 0

    out = Path(args.output)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows → {out}")

    if args.clear:
        _wal_clear()
        print("WAL cleared.")

    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aluminatai-agent",
        description="AluminatAI GPU Energy Agent v0.2.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables and config file (lowest to highest priority):
  Config file  (--config path.json or ALUMINATAI_CONFIG env var)
  ALUMINATAI_API_KEY        API key (alum_...)
  ALUMINATAI_API_ENDPOINT   Ingest URL
  ALUMINATAI_JOB_UUID       DB job UUID for completion signal
  SAMPLE_INTERVAL           Sampling interval in seconds (default: 5.0)
  UPLOAD_INTERVAL           Upload flush interval in seconds (default: 60)
  LOG_LEVEL                 DEBUG / INFO / WARNING / ERROR (default: INFO)
  LOG_FORMAT                text | json  (default: text)
  DRY_RUN                   1 — collect/attribute but do not upload
  PROMETHEUS_ONLY           1 — disable cloud uploads; serve Prometheus only
  OFFLINE_MODE              1 — write WAL only, no HTTP uploads

Config file keys (JSON/YAML): sample_interval, upload_interval,
  metrics_port, wal_max_mb, log_level, log_format, dry_run,
  prometheus_only, offline_mode, … (see docs)

Examples:
  aluminatiai
  aluminatiai --interval 1 --duration 3600
  aluminatiai --dry-run --log-format json
  aluminatiai --prometheus-only --interval 2
  aluminatiai --config /etc/aluminatai.json
  aluminatiai replay --output /data/metrics.csv --clear
  aluminatiai service install
  aluminatiai service status
  aluminatiai service uninstall
        """,
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="JSON or YAML config file path (also: ALUMINATAI_CONFIG env var)")
    parser.add_argument("--interval", "-i", type=float, default=None,
                        help="Sampling interval in seconds (default: SAMPLE_INTERVAL or 5.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV manifest path")
    parser.add_argument("--duration", "-d", type=float, default=None,
                        help="Run for N seconds then exit 0 (default: infinite)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress console output")
    parser.add_argument("--job-uuid", type=str, default=None,
                        help="DB job UUID — signals completion on exit")
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN,
                        help="Collect and attribute but skip all uploads/WAL writes")
    parser.add_argument("--prometheus-only", action="store_true", default=PROMETHEUS_ONLY,
                        help="Disable cloud uploads; serve Prometheus metrics only")
    parser.add_argument("--log-format", choices=["text", "json"], default=LOG_FORMAT,
                        help="Log output format: 'text' (default) or 'json' for ELK/Loki")
    parser.add_argument("--log-level", default=None,
                        help="Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: LOG_LEVEL or INFO)")
    parser.add_argument("--version", action="version", version=f"aluminatai-agent {AGENT_VERSION}")

    subparsers = parser.add_subparsers(dest="command")

    # replay subcommand: export WAL → CSV
    replay_parser = subparsers.add_parser(
        "replay",
        help="Export WAL contents to CSV (for offline/air-gapped clusters)",
    )
    replay_parser.add_argument(
        "--output", "-o", default="metrics.csv",
        help="Output CSV file path (default: metrics.csv)",
    )
    replay_parser.add_argument(
        "--clear", action="store_true",
        help="Clear the WAL after successful export",
    )

    # service subcommand: install / uninstall / status for systemd
    service_parser = subparsers.add_parser(
        "service",
        help="Manage the aluminatai-agent systemd service (requires root for install/uninstall)",
    )
    service_sub = service_parser.add_subparsers(dest="service_action")
    service_sub.required = True

    svc_install = service_sub.add_parser("install", help="Install and start the systemd service")
    svc_install.add_argument(
        "--api-key", dest="api_key", default=os.environ.get("ALUMINATAI_API_KEY", ""),
        help="API key (default: ALUMINATAI_API_KEY env var)",
    )
    svc_install.add_argument(
        "--update-env", action="store_true",
        help="Overwrite existing /etc/aluminatai/agent.env",
    )

    service_sub.add_parser("uninstall", help="Stop, disable, and remove the systemd service")
    service_sub.add_parser("status", help="Show systemd service status")

    args = parser.parse_args()

    # Apply --config if provided (already applied in cli.py for installed entry-point;
    # this handles the case where agent.py is run directly with python agent.py).
    if args.command not in ("replay", "service") and getattr(args, "config", None):
        if not os.environ.get("ALUMINATAI_CONFIG"):
            os.environ["ALUMINATAI_CONFIG"] = args.config

    if args.command == "replay":
        return _cmd_replay(args)

    if args.command == "service":
        return _cmd_service(args)

    # Re-configure logging now that we have the final level + format
    effective_level = args.log_level or LOG_LEVEL
    effective_fmt = args.log_format
    _setup_logging(level=effective_level, fmt=effective_fmt)

    if not UPLOAD_ENABLED and not args.dry_run and not args.prometheus_only:
        log.warning(
            "ALUMINATAI_API_KEY is not set — metrics will NOT be uploaded to the dashboard. "
            "Get your API key at https://aluminatiai.com/dashboard"
        )

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
        dry_run=args.dry_run,
        prometheus_only=args.prometheus_only,
    )
    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
