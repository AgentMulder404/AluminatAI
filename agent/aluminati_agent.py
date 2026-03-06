#!/usr/bin/env python3
# DEPRECATED: use agent/agent.py (aluminatai-agent CLI) instead.
# This file is kept for reference and will be deleted in v0.3.0.
"""
AluminatAI Background Monitoring Agent v0.1

A zero-inference-overhead GPU telemetry daemon that captures power draw,
utilization, memory, and temperature via NVML, writes structured CSV
Energy Manifests, and prints a rolling energy heartbeat to stderr.

Architecture
────────────
  ┌─────────────────────────────────────────────────────────┐
  │                  aluminati_agent.py                      │
  │                                                         │
  │  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
  │  │  NVMLProbe│──▶│ ManifestWriter│──▶│  Heartbeat     │  │
  │  │  (pynvml) │   │  (.csv)       │   │  (rolling kWh) │  │
  │  └──────────┘   └──────────────┘   └────────────────┘  │
  │          ▲                                    │         │
  │          │           AgentCore                │         │
  │          └────────── (main loop) ◀────────────┘         │
  │                         │                               │
  │                    SIGINT/SIGTERM                        │
  │                    graceful stop                         │
  └─────────────────────────────────────────────────────────┘

Usage:
    python aluminati_agent.py --job-id TEST_001
    python aluminati_agent.py --job-id TRAIN_RUN_42 --freq 2 --duration 600
    python aluminati_agent.py --job-id BENCH --gpu 0,2 --output /data/manifests/

Self-Healing:
    - No GPU?  Logs a warning and exits cleanly (exit code 3).
    - NVML init fails?  Same — no crash, no traceback spam.
    - SIGTERM mid-write?  CSV is flushed and closed before exit.

Dependencies:
    pip install nvidia-ml-py3   (pynvml)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import signal
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("aluminati-agent")

# ── NVML Import (graceful) ───────────────────────────────────────────────────

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class GPUSample:
    """Single telemetry snapshot from one GPU."""

    timestamp_utc: str
    epoch_s: float
    job_id: str
    gpu_index: int
    power_w: float
    util_pct: int
    mem_util_pct: int
    temp_c: int
    mem_used_mb: float
    mem_total_mb: float
    fan_pct: int
    gpu_uuid: str
    gpu_name: str

    # Derived by the heartbeat tracker, not NVML
    energy_delta_j: float = 0.0


CSV_COLUMNS = [
    "timestamp",
    "job_id",
    "gpu_index",
    "power_w",
    "util_pct",
    "temp_c",
    "mem_used_mb",
    "mem_total_mb",
    "mem_util_pct",
    "fan_pct",
    "energy_delta_j",
    "rolling_kwh",
    "gpu_uuid",
    "gpu_name",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  NVMLProbe — thin pynvml wrapper with self-healing
# ═══════════════════════════════════════════════════════════════════════════════


class NVMLProbe:
    """
    Low-level NVML interface.

    Initialises the driver once, caches device handles and static info,
    and exposes a single ``sample()`` method that returns a list of
    GPUSample objects for the requested GPU indices.

    Self-healing: if a single GPU read fails mid-sample, it is skipped
    rather than crashing the entire agent.
    """

    def __init__(self, gpu_indices: Optional[List[int]] = None):
        """
        Args:
            gpu_indices: Specific GPUs to monitor.  None = all GPUs.
        """
        if not _NVML_AVAILABLE:
            raise RuntimeError(
                "pynvml is not installed.  "
                "Install with: pip install nvidia-ml-py3"
            )

        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:
            raise RuntimeError(f"NVML initialisation failed: {exc}") from exc

        total_gpus = pynvml.nvmlDeviceGetCount()
        if total_gpus == 0:
            pynvml.nvmlShutdown()
            raise RuntimeError("No NVIDIA GPUs detected by NVML.")

        # Resolve requested indices
        if gpu_indices is not None:
            for idx in gpu_indices:
                if idx < 0 or idx >= total_gpus:
                    pynvml.nvmlShutdown()
                    raise ValueError(
                        f"GPU index {idx} out of range (0..{total_gpus - 1})"
                    )
            self._indices = list(gpu_indices)
        else:
            self._indices = list(range(total_gpus))

        # Cache handles + static info
        self._handles: dict[int, object] = {}
        self._static: dict[int, dict] = {}

        for idx in self._indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            self._handles[idx] = handle

            uuid = pynvml.nvmlDeviceGetUUID(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            self._static[idx] = {"uuid": uuid, "name": name}

        log.info(
            "NVML ready — monitoring GPU(s): %s  [%s]",
            ", ".join(str(i) for i in self._indices),
            ", ".join(self._static[i]["name"] for i in self._indices),
        )

    @property
    def gpu_indices(self) -> List[int]:
        return list(self._indices)

    @property
    def gpu_count(self) -> int:
        return len(self._indices)

    def sample(self, job_id: str) -> List[GPUSample]:
        """Read all monitored GPUs and return a list of GPUSamples."""
        now_utc = datetime.now(timezone.utc).isoformat()
        epoch = time.time()
        results: List[GPUSample] = []

        for idx in self._indices:
            handle = self._handles[idx]
            try:
                results.append(self._read_one(handle, idx, job_id, now_utc, epoch))
            except pynvml.NVMLError as exc:
                log.warning("GPU %d read failed (skipped): %s", idx, exc)

        return results

    def shutdown(self):
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    # ── internal ──────────────────────────────────────────────────────────

    def _read_one(
        self, handle, idx: int, job_id: str, ts: str, epoch: float
    ) -> GPUSample:
        power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util_gpu, util_mem = util.gpu, util.memory
        except pynvml.NVMLError:
            util_gpu, util_mem = 0, 0

        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError:
            temp = 0

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = mem.used / (1024 * 1024)
            mem_total_mb = mem.total / (1024 * 1024)
        except pynvml.NVMLError:
            mem_used_mb, mem_total_mb = 0.0, 0.0

        try:
            fan = pynvml.nvmlDeviceGetFanSpeed(handle)
        except pynvml.NVMLError:
            fan = 0

        static = self._static[idx]

        return GPUSample(
            timestamp_utc=ts,
            epoch_s=epoch,
            job_id=job_id,
            gpu_index=idx,
            power_w=round(power_w, 2),
            util_pct=util_gpu,
            mem_util_pct=util_mem,
            temp_c=temp,
            mem_used_mb=round(mem_used_mb, 1),
            mem_total_mb=round(mem_total_mb, 1),
            fan_pct=fan,
            gpu_uuid=static["uuid"],
            gpu_name=static["name"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  ManifestWriter — structured CSV output with atomic flush
# ═══════════════════════════════════════════════════════════════════════════════


class ManifestWriter:
    """
    Appends GPU telemetry rows to a structured CSV file.

    The manifest file is the unit of energy accountability — every row
    is a timestamped, job-tagged, per-GPU power reading that downstream
    systems can aggregate into Energy Manifests.

    The writer keeps the file handle open and flushes after every batch
    so that data is recoverable even after an unclean shutdown.
    """

    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

        self._file: Optional[io.TextIOWrapper] = None
        self._writer: Optional[csv.writer] = None
        self._row_count = 0

    def open(self):
        """Open (or create) the CSV file and write the header."""
        self._file = open(self._path, "w", newline="", buffering=1)  # line-buffered
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)
        self._file.flush()
        log.info("Manifest file opened: %s", self._path)

    def write(self, sample: GPUSample, rolling_kwh: float):
        """Append a single sample row to the manifest."""
        if self._writer is None:
            raise RuntimeError("ManifestWriter is not open")

        self._writer.writerow([
            sample.timestamp_utc,
            sample.job_id,
            sample.gpu_index,
            sample.power_w,
            sample.util_pct,
            sample.temp_c,
            sample.mem_used_mb,
            sample.mem_total_mb,
            sample.mem_util_pct,
            sample.fan_pct,
            round(sample.energy_delta_j, 4),
            round(rolling_kwh, 8),
            sample.gpu_uuid,
            sample.gpu_name,
        ])
        self._row_count += 1

    def flush(self):
        """Force an OS-level flush to disk."""
        if self._file and not self._file.closed:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self):
        """Flush and close the CSV file."""
        if self._file and not self._file.closed:
            self.flush()
            self._file.close()
            log.info(
                "Manifest closed: %s  (%d rows written)", self._path, self._row_count
            )

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def path(self) -> Path:
        return self._path


# ═══════════════════════════════════════════════════════════════════════════════
#  HeartbeatTracker — rolling energy (kWh) computation
# ═══════════════════════════════════════════════════════════════════════════════


class HeartbeatTracker:
    """
    Maintains per-GPU rolling energy consumption using trapezoidal integration.

    Energy between two consecutive samples is computed as:

        ΔE = (P₁ + P₂) / 2 × Δt   [Joules]

    The tracker accumulates ΔE per GPU and provides the total in kWh.

    The heartbeat is printed to stderr at a configurable interval so it
    never contaminates stdout (which can be redirected to /dev/null or
    piped into another process).
    """

    def __init__(self, heartbeat_interval_s: float = 30.0):
        # Per-GPU state
        self._last_epoch: dict[int, float] = {}
        self._last_power: dict[int, float] = {}
        self._total_joules: dict[int, float] = {}

        self._heartbeat_interval = heartbeat_interval_s
        self._last_heartbeat = 0.0
        self._total_samples = 0

    def update(self, sample: GPUSample) -> GPUSample:
        """
        Ingest a new sample, compute ΔE, and return a copy with
        ``energy_delta_j`` filled in.
        """
        idx = sample.gpu_index
        epoch = sample.epoch_s
        power = sample.power_w

        delta_j = 0.0
        if idx in self._last_epoch:
            dt = epoch - self._last_epoch[idx]
            if 0 < dt < 60:  # reject stale gaps > 60s
                avg_p = (power + self._last_power[idx]) / 2.0
                delta_j = avg_p * dt

        self._last_epoch[idx] = epoch
        self._last_power[idx] = power
        self._total_joules.setdefault(idx, 0.0)
        self._total_joules[idx] += delta_j
        self._total_samples += 1

        # Return an updated copy (GPUSample is frozen)
        return GPUSample(
            timestamp_utc=sample.timestamp_utc,
            epoch_s=sample.epoch_s,
            job_id=sample.job_id,
            gpu_index=sample.gpu_index,
            power_w=sample.power_w,
            util_pct=sample.util_pct,
            mem_util_pct=sample.mem_util_pct,
            temp_c=sample.temp_c,
            mem_used_mb=sample.mem_used_mb,
            mem_total_mb=sample.mem_total_mb,
            fan_pct=sample.fan_pct,
            gpu_uuid=sample.gpu_uuid,
            gpu_name=sample.gpu_name,
            energy_delta_j=delta_j,
        )

    def rolling_kwh(self, gpu_index: int) -> float:
        """Total accumulated energy in kWh for a GPU."""
        return self._total_joules.get(gpu_index, 0.0) / 3_600_000.0

    def total_kwh(self) -> float:
        """Total accumulated energy across all GPUs."""
        return sum(self._total_joules.values()) / 3_600_000.0

    def total_joules(self) -> float:
        return sum(self._total_joules.values())

    def maybe_print_heartbeat(self, force: bool = False):
        """Print a rolling energy heartbeat to stderr if the interval has elapsed."""
        now = time.time()
        if not force and (now - self._last_heartbeat < self._heartbeat_interval):
            return

        self._last_heartbeat = now
        total_j = self.total_joules()
        total_kwh = self.total_kwh()

        parts = []
        for idx in sorted(self._total_joules):
            gpu_kwh = self.rolling_kwh(idx)
            parts.append(f"GPU{idx}={gpu_kwh:.6f}kWh")

        gpu_detail = "  ".join(parts) if parts else "no data"
        log.info(
            "HEARTBEAT  samples=%d  total=%.4fJ (%.8f kWh)  [%s]",
            self._total_samples,
            total_j,
            total_kwh,
            gpu_detail,
        )

    @property
    def sample_count(self) -> int:
        return self._total_samples


# ═══════════════════════════════════════════════════════════════════════════════
#  AgentCore — main loop orchestrating probe → writer → heartbeat
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  API helper — signal job completion to AluminatiAI backend
# ═══════════════════════════════════════════════════════════════════════════════


def signal_job_complete(
    endpoint: str,
    api_key: str,
    job_uuid: str,
    end_time: Optional[str] = None,
) -> bool:
    """
    POST /api/metrics/jobs/complete to mark a job finished in the DB.
    The server-side DB trigger then auto-generates the energy manifest.

    Returns True on success, False on any error (non-fatal).
    """
    url = endpoint.rstrip("/") + "/api/metrics/jobs/complete"
    payload = {"job_id": job_uuid}
    if end_time:
        payload["end_time"] = end_time

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            log.info(
                "Job completion signalled: job_uuid=%s  end_time=%s  msg=%s",
                job_uuid,
                body.get("end_time", "?"),
                body.get("message", ""),
            )
            return True
    except urllib.error.HTTPError as exc:
        log.warning("Job complete HTTP %s: %s", exc.code, exc.read().decode())
    except Exception as exc:
        log.warning("Job complete signal failed (non-fatal): %s", exc)
    return False


class AgentCore:
    """
    Production background monitoring agent.

    Orchestrates the NVMLProbe, ManifestWriter, and HeartbeatTracker in
    a simple poll loop.  Designed to run as a standalone background
    process (``nohup``, ``systemd``, or ``multiprocessing.Process``)
    alongside active training/inference workloads.

    Signal handling:
        SIGINT  (Ctrl-C) → graceful stop
        SIGTERM (kill)    → graceful stop
    """

    def __init__(
        self,
        job_id: str,
        freq_hz: float = 1.0,
        gpu_indices: Optional[List[int]] = None,
        output_dir: Optional[Path] = None,
        duration_s: Optional[float] = None,
        heartbeat_s: float = 30.0,
        quiet: bool = False,
        api_key: Optional[str] = None,
        endpoint: str = "https://aluminatiai.com",
        job_uuid: Optional[str] = None,
    ):
        self._job_id = job_id
        self._interval = 1.0 / freq_hz
        self._gpu_indices = gpu_indices
        self._duration = duration_s
        self._quiet = quiet
        self._api_key = api_key
        self._endpoint = endpoint
        self._job_uuid = job_uuid

        # Resolve output path
        if output_dir is None:
            output_dir = Path("./data/manifests")
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"manifest_{job_id}_{ts_tag}.csv"
        self._csv_path = output_dir / csv_name

        # Components (initialised in run())
        self._probe: Optional[NVMLProbe] = None
        self._writer: Optional[ManifestWriter] = None
        self._heartbeat = HeartbeatTracker(heartbeat_interval_s=heartbeat_s)

        # Shutdown flag
        self._running = False

    def run(self) -> int:
        """
        Execute the agent loop.  Returns an exit code:
            0 — clean shutdown (signal or duration reached)
            1 — runtime error
            3 — no GPU / NVML unavailable
        """
        # Install signal handlers
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        # ── Initialise NVML ──────────────────────────────────────────────
        try:
            self._probe = NVMLProbe(gpu_indices=self._gpu_indices)
        except RuntimeError as exc:
            log.error("GPU probe failed: %s", exc)
            return 3

        # ── Open manifest CSV ────────────────────────────────────────────
        self._writer = ManifestWriter(self._csv_path)
        try:
            self._writer.open()
        except OSError as exc:
            log.error("Cannot open manifest file: %s", exc)
            self._probe.shutdown()
            return 1

        # ── Banner ───────────────────────────────────────────────────────
        if not self._quiet:
            self._print_banner()

        # ── Main loop ────────────────────────────────────────────────────
        self._running = True
        start_time = time.monotonic()

        try:
            while self._running:
                loop_t0 = time.monotonic()

                samples = self._probe.sample(self._job_id)

                for raw in samples:
                    enriched = self._heartbeat.update(raw)
                    rolling = self._heartbeat.rolling_kwh(enriched.gpu_index)
                    self._writer.write(enriched, rolling)

                if not self._quiet:
                    self._heartbeat.maybe_print_heartbeat()

                # Duration guard
                if self._duration and (time.monotonic() - start_time) >= self._duration:
                    log.info("Duration limit reached (%.0fs). Stopping.", self._duration)
                    break

                # Sleep remainder of interval
                elapsed = time.monotonic() - loop_t0
                sleep_s = max(0.0, self._interval - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)

        except Exception as exc:
            log.exception("Unhandled error in agent loop: %s", exc)
            return 1

        finally:
            # ── Cleanup (always runs) ────────────────────────────────────
            self._writer.close()
            self._probe.shutdown()

            if not self._quiet:
                self._heartbeat.maybe_print_heartbeat(force=True)
                self._print_summary(time.monotonic() - start_time)

            # ── Signal job completion to AluminatiAI backend ─────────────
            if self._api_key and self._job_uuid:
                signal_job_complete(
                    endpoint=self._endpoint,
                    api_key=self._api_key,
                    job_uuid=self._job_uuid,
                    end_time=datetime.now(timezone.utc).isoformat(),
                )

        return 0

    # ── Signal handling ──────────────────────────────────────────────────

    def _on_signal(self, signum, _frame):
        sig_name = signal.Signals(signum).name
        log.info("Received %s — shutting down gracefully.", sig_name)
        self._running = False

    # ── Display helpers ──────────────────────────────────────────────────

    def _print_banner(self):
        log.info("=" * 62)
        log.info("  AluminatAI Background Monitoring Agent v0.1")
        log.info("=" * 62)
        log.info("  Job ID      : %s", self._job_id)
        log.info("  GPU(s)      : %s", self._probe.gpu_indices)
        log.info("  Frequency   : %.1f Hz (%.2fs interval)", 1.0 / self._interval, self._interval)
        log.info("  Manifest    : %s", self._csv_path)
        if self._duration:
            log.info("  Duration    : %.0fs", self._duration)
        else:
            log.info("  Duration    : indefinite (until SIGINT/SIGTERM)")
        log.info("=" * 62)

    def _print_summary(self, runtime_s: float):
        hb = self._heartbeat
        log.info("=" * 62)
        log.info("  SESSION SUMMARY")
        log.info("=" * 62)
        log.info("  Runtime     : %.1fs", runtime_s)
        log.info("  Samples     : %d", hb.sample_count)
        log.info("  Manifest    : %s  (%d rows)", self._csv_path, self._writer.row_count)
        log.info("  ────────────────────────────────────────")

        total_j = 0.0
        for idx in sorted(self._heartbeat._total_joules):
            j = self._heartbeat._total_joules[idx]
            kwh = j / 3_600_000.0
            total_j += j
            log.info("  GPU %-2d      : %.2f J  (%.8f kWh)", idx, j, kwh)

        total_kwh = total_j / 3_600_000.0
        cost = total_kwh * 0.12  # $0.12/kWh US average
        log.info("  ────────────────────────────────────────")
        log.info("  TOTAL       : %.2f J  (%.8f kWh)", total_j, total_kwh)
        log.info("  Est. cost   : $%.6f  (@ $0.12/kWh)", cost)
        log.info("=" * 62)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aluminati_agent",
        description=(
            "AluminatAI Background Monitoring Agent v0.1 — "
            "zero-overhead GPU telemetry via NVML."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python aluminati_agent.py --job-id TEST_001
  python aluminati_agent.py --job-id TRAIN_42 --freq 2 --duration 600
  python aluminati_agent.py --job-id BENCH --gpu 0,2 --output /data/manifests/
  python aluminati_agent.py --job-id SWEEP --quiet --duration 3600
""",
    )

    parser.add_argument(
        "--job-id",
        required=True,
        help="Unique identifier for this monitoring session.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=1.0,
        help="Sampling frequency in Hz (default: 1.0 = once per second).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Comma-separated GPU indices to monitor (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for manifest CSV (default: ./data/manifests/).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N seconds then stop (default: indefinite).",
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="Heartbeat log interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress periodic heartbeat output (summary still printed on exit).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="AluminatiAI API key (alum_...). When set, signals job completion on exit.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://aluminatiai.com",
        help="AluminatiAI API base URL (default: production).",
    )
    parser.add_argument(
        "--job-uuid",
        type=str,
        default=None,
        help="UUID of the gpu_jobs row in the database. Required for completion signal.",
    )

    args = parser.parse_args()

    # Validate frequency
    if args.freq <= 0 or args.freq > 100:
        log.error("Frequency must be between 0 and 100 Hz, got %.1f", args.freq)
        return 2

    # Parse GPU indices
    gpu_indices = None
    if args.gpu:
        try:
            gpu_indices = [int(x.strip()) for x in args.gpu.split(",")]
        except ValueError:
            log.error("Invalid --gpu value: %r  (expected comma-separated ints)", args.gpu)
            return 2

    output_dir = Path(args.output) if args.output else None

    agent = AgentCore(
        job_id=args.job_id,
        freq_hz=args.freq,
        gpu_indices=gpu_indices,
        output_dir=output_dir,
        duration_s=args.duration,
        heartbeat_s=args.heartbeat,
        quiet=args.quiet,
        api_key=args.api_key,
        endpoint=args.endpoint,
        job_uuid=args.job_uuid,
    )

    return agent.run()


if __name__ == "__main__":
    sys.exit(main())
