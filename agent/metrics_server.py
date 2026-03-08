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
"""
Prometheus /metrics endpoint for the AluminatAI GPU agent.

Activated when METRICS_PORT != 0 (default 9100).
Optional dep: pip install aluminatiai[prometheus]

GPU metrics:
  aluminatai_gpu_power_watts{gpu_uuid, gpu_index}
  aluminatai_gpu_energy_joules_total{gpu_uuid}
  aluminatai_gpu_utilization_pct{gpu_uuid, gpu_index}
  aluminatai_gpu_temperature_c{gpu_uuid, gpu_index}

Upload / WAL health:
  aluminatai_upload_success_total
  aluminatai_upload_failure_total
  aluminatai_buffer_size
  aluminatai_wal_size_bytes
  aluminatai_wal_entries_pending
  aluminatai_wal_replay_uploaded_total
  aluminatai_wal_replay_failed_total

Attribution:
  aluminatai_attribution_confidence{gpu_uuid, method}
  aluminatai_attribution_unresolved_total

Agent health:
  aluminatai_agent_uptime_seconds
  aluminatai_agent_info{version, hostname, mode}
"""

from __future__ import annotations

import base64
import logging
import threading
from typing import Any, List
from wsgiref.simple_server import WSGIServer, WSGIRequestHandler, make_server

logger = logging.getLogger(__name__)

try:
    import prometheus_client as prom
    from prometheus_client import Gauge, Counter, REGISTRY, make_wsgi_app
    _PROM = True
except ImportError:
    _PROM = False


class _QuietHandler(WSGIRequestHandler):
    """Suppress per-request stdout logs from wsgiref."""

    def log_message(self, *a, **kw):
        pass


def _basic_auth_middleware(app, credentials: str):
    """WSGI wrapper enforcing HTTP Basic Auth."""
    expected = b"Basic " + base64.b64encode(credentials.encode())

    def _inner(environ, start_response):
        auth = environ.get("HTTP_AUTHORIZATION", "").encode()
        if auth != expected:
            start_response("401 Unauthorized", [
                ("WWW-Authenticate", 'Basic realm="aluminatai"'),
                ("Content-Type", "text/plain"),
            ])
            return [b"Unauthorized\n"]
        return app(environ, start_response)

    return _inner


class MetricsServer:
    """Background thread serving Prometheus metrics."""

    def __init__(self):
        self._port: int = 0
        self._bind_host: str = ""
        self._basic_auth: str = ""
        self._started = False
        self._srv = None

        try:
            from config import METRICS_PORT, METRICS_BIND_HOST, METRICS_BASIC_AUTH
            self._port = METRICS_PORT
            self._bind_host = METRICS_BIND_HOST
            self._basic_auth = METRICS_BASIC_AUTH
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
        self._wal_size_bytes = Gauge(
            "aluminatai_wal_size_bytes",
            "WAL file size in bytes",
        )
        self._wal_entries_pending = Gauge(
            "aluminatai_wal_entries_pending",
            "Approximate number of metric rows waiting in the WAL",
        )
        self._wal_replay_uploaded = Counter(
            "aluminatai_wal_replay_uploaded_total",
            "Total WAL rows successfully re-uploaded during replay",
        )
        self._wal_replay_failed = Counter(
            "aluminatai_wal_replay_failed_total",
            "Total WAL rows that failed replay and remain pending",
        )
        self._confidence = Gauge(
            "aluminatai_attribution_confidence",
            "Attribution confidence score, labelled by resolution method",
            ["method", "gpu_uuid"],
        )
        self._attribution_unresolved = Counter(
            "aluminatai_attribution_unresolved_total",
            "Collection cycles where the attribution engine returned no result for a GPU",
        )
        self._agent_uptime = Gauge(
            "aluminatai_agent_uptime_seconds",
            "Seconds since the agent process started",
        )
        # Info-pattern gauge: always 1.0, metadata in labels
        self._agent_info = Gauge(
            "aluminatai_agent_info",
            "Agent metadata (version, hostname, run mode); value is always 1",
            ["version", "hostname", "mode"],
        )

    def start(self) -> None:
        if not _PROM or self._port == 0 or self._started:
            return
        try:
            app = make_wsgi_app()
            if self._basic_auth:
                app = _basic_auth_middleware(app, self._basic_auth)
                logger.warning(
                    "Prometheus Basic Auth active — use a TLS proxy in production"
                )
            bind = self._bind_host or "0.0.0.0"
            srv = make_server(bind, self._port, app, WSGIServer, _QuietHandler)
            self._srv = srv
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            self._started = True
            logger.info("Prometheus metrics server on %s:%d/metrics", bind, self._port)
        except OSError as exc:
            logger.warning("Could not start metrics server on :%d: %s", self._port, exc)

    def stop(self) -> None:
        if self._srv is not None:
            self._srv.shutdown()

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

        _CONF_SCORES = {
            "tagged":          1.0,
            "scheduler":       0.9,
            "scheduler_poll":  0.7,
            "rules":           0.6,
            "heuristic":       0.4,
            "memory_split":    0.2,
            "idle":            0.1,
        }
        for row in attributed_rows:
            conf = row.get("attribution_confidence", "unknown")
            uuid = row.get("gpu_uuid", "unknown")
            conf_score = _CONF_SCORES.get(conf, 0.0)
            self._confidence.labels(method=conf, gpu_uuid=uuid).set(conf_score)

    def update_upload_stats(self, success_delta: int, failure_delta: int, buffer_size: int) -> None:
        if not _PROM or not self._started:
            return
        if success_delta > 0:
            self._upload_success.inc(success_delta)
        if failure_delta > 0:
            self._upload_failure.inc(failure_delta)
        self._buffer_size.set(buffer_size)

    def update_wal_stats(
        self,
        wal_size_bytes: int,
        wal_entries_pending: int,
        replay_uploaded_delta: int = 0,
        replay_failed_delta: int = 0,
    ) -> None:
        """Update WAL health metrics. Called after each flush/replay cycle."""
        if not _PROM or not self._started:
            return
        self._wal_size_bytes.set(wal_size_bytes)
        self._wal_entries_pending.set(wal_entries_pending)
        if replay_uploaded_delta > 0:
            self._wal_replay_uploaded.inc(replay_uploaded_delta)
        if replay_failed_delta > 0:
            self._wal_replay_failed.inc(replay_failed_delta)

    def update_agent_stats(
        self,
        uptime_sec: float,
        version: str = "",
        hostname: str = "",
        mode: str = "normal",
    ) -> None:
        """Update agent uptime and info label. Called once per collection cycle."""
        if not _PROM or not self._started:
            return
        self._agent_uptime.set(uptime_sec)
        if version and hostname:
            self._agent_info.labels(version=version, hostname=hostname, mode=mode).set(1.0)

    def record_attribution_unresolved(self, count: int = 1) -> None:
        """Increment the unresolved attribution counter."""
        if not _PROM or not self._started or count <= 0:
            return
        self._attribution_unresolved.inc(count)
