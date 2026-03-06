"""
Prometheus /metrics endpoint for the AluminatAI GPU agent (Phase 2).

Activated when METRICS_PORT != 0 (default 9100).
Optional dep: pip install aluminatai-agent[prometheus]

Gauges / counters exposed:
  aluminatai_gpu_power_watts{gpu_uuid, gpu_index}
  aluminatai_gpu_energy_joules_total{gpu_uuid}
  aluminatai_gpu_utilization_pct{gpu_uuid, gpu_index}
  aluminatai_gpu_temperature_c{gpu_uuid, gpu_index}
  aluminatai_upload_success_total
  aluminatai_upload_failure_total
  aluminatai_buffer_size
  aluminatai_attribution_confidence{confidence, gpu_uuid}
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List

logger = logging.getLogger(__name__)

try:
    import prometheus_client as prom
    from prometheus_client import start_http_server, Gauge, Counter, REGISTRY
    _PROM = True
except ImportError:
    _PROM = False


class MetricsServer:
    """Background thread serving Prometheus metrics."""

    def __init__(self):
        self._port: int = 0
        self._started = False

        try:
            from config import METRICS_PORT
            self._port = METRICS_PORT
        except ImportError:
            self._port = 9100

        if not _PROM:
            logger.info("prometheus-client not installed — metrics server disabled")
            return

        if self._port == 0:
            logger.info("METRICS_PORT=0 — metrics server disabled")
            return

        # Define metrics
        labels = ["gpu_uuid", "gpu_index"]

        self._power = Gauge(
            "aluminatai_gpu_power_watts",
            "GPU power draw in watts",
            labels,
        )
        self._energy = Counter(
            "aluminatai_gpu_energy_joules_total",
            "Cumulative GPU energy in joules",
            ["gpu_uuid"],
        )
        self._util = Gauge(
            "aluminatai_gpu_utilization_pct",
            "GPU compute utilization percent",
            labels,
        )
        self._temp = Gauge(
            "aluminatai_gpu_temperature_c",
            "GPU temperature in Celsius",
            labels,
        )
        self._upload_success = Counter(
            "aluminatai_upload_success_total",
            "Total metrics successfully uploaded",
        )
        self._upload_failure = Counter(
            "aluminatai_upload_failure_total",
            "Total metric batches that failed upload",
        )
        self._buffer_size = Gauge(
            "aluminatai_buffer_size",
            "Current in-memory upload buffer size",
        )
        self._confidence = Gauge(
            "aluminatai_attribution_confidence",
            "Attribution confidence (1=process, 0.5=scheduler_poll, 0=idle)",
            ["confidence", "gpu_uuid"],
        )

    def start(self) -> None:
        if not _PROM or self._port == 0 or self._started:
            return
        try:
            start_http_server(self._port)
            self._started = True
            logger.info("Prometheus metrics server started on :%d/metrics", self._port)
        except OSError as exc:
            logger.warning("Could not start metrics server on :%d: %s", self._port, exc)

    def stop(self) -> None:
        pass  # prometheus_client HTTP server has no clean shutdown API

    def update(self, metrics: List[Any], attributed_rows: List[dict]) -> None:
        """Called from the main loop after each collection cycle."""
        if not _PROM or not self._started:
            return

        for m in metrics:
            uuid = m.gpu_uuid
            idx = str(m.gpu_index)
            self._power.labels(gpu_uuid=uuid, gpu_index=idx).set(m.power_draw_w)
            self._util.labels(gpu_uuid=uuid, gpu_index=idx).set(m.utilization_gpu_pct)
            self._temp.labels(gpu_uuid=uuid, gpu_index=idx).set(m.temperature_c)
            if m.energy_delta_j:
                self._energy.labels(gpu_uuid=uuid).inc(m.energy_delta_j)

        for row in attributed_rows:
            conf = row.get("attribution_confidence", "unknown")
            uuid = row.get("gpu_uuid", "unknown")
            conf_score = {"process": 1.0, "scheduler_poll": 0.5, "inferred": 0.3, "idle": 0.1}.get(conf, 0.0)
            self._confidence.labels(confidence=conf, gpu_uuid=uuid).set(conf_score)

    def update_upload_stats(self, success_delta: int, failure_delta: int, buffer_size: int) -> None:
        if not _PROM or not self._started:
            return
        if success_delta > 0:
            self._upload_success.inc(success_delta)
        if failure_delta > 0:
            self._upload_failure.inc(failure_delta)
        self._buffer_size.set(buffer_size)
