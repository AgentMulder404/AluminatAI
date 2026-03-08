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
Configuration for AluminatAI GPU Agent v0.2.2

Priority (highest to lowest):
  1. Environment variables (ALUMINATAI_*, SAMPLE_INTERVAL, …)
  2. Config file  (ALUMINATAI_CONFIG=/path/to/file.json|yaml)
  3. Built-in defaults

Config file is JSON or YAML (YAML requires pip install aluminatiai[observability]).
Default search order when ALUMINATAI_CONFIG is unset:
  ./aluminatai.json → ./aluminatai.yaml → ~/.config/aluminatai/config.json
"""
import json
import os
import sys
from pathlib import Path

# ── Version ───────────────────────────────────────────────────────────────────

AGENT_VERSION = "0.2.2"

# ── Config-file → env-var mapping ─────────────────────────────────────────────

_CONFIG_KEY_TO_ENV: dict[str, str] = {
    "api_key":                  "ALUMINATAI_API_KEY",
    "api_endpoint":             "ALUMINATAI_API_ENDPOINT",
    "sample_interval":          "SAMPLE_INTERVAL",
    "upload_interval":          "UPLOAD_INTERVAL",
    "upload_batch_size":        "UPLOAD_BATCH_SIZE",
    "upload_max_retries":       "UPLOAD_MAX_RETRIES",
    "upload_max_retry_delay":   "UPLOAD_MAX_RETRY_DELAY",
    "wal_max_age_hours":        "WAL_MAX_AGE_HOURS",
    "wal_max_mb":               "WAL_MAX_MB",
    "scheduler_poll_interval":  "SCHEDULER_POLL_INTERVAL",
    "heartbeat_interval":       "HEARTBEAT_INTERVAL",
    "metrics_port":             "METRICS_PORT",
    "metrics_bind_host":        "METRICS_BIND_HOST",
    "metrics_basic_auth":       "METRICS_BASIC_AUTH",
    "offline_mode":             "OFFLINE_MODE",
    "dry_run":                  "DRY_RUN",
    "prometheus_only":          "PROMETHEUS_ONLY",
    "log_level":                "LOG_LEVEL",
    "log_format":               "LOG_FORMAT",
    "data_dir":                 "DATA_DIR",
    "log_dir":                  "LOG_DIR",
    "https_proxy":              "HTTPS_PROXY",
    "ca_bundle":                "ALUMINATAI_CA_BUNDLE",
    "client_cert":              "ALUMINATAI_CLIENT_CERT",
    "client_key":               "ALUMINATAI_CLIENT_KEY",
    "attribution_config":       "ALUMINATAI_ATTRIBUTION_CONFIG",
    "trusted_uids":             "ALUMINATAI_TRUSTED_UIDS",
}


def _load_config_file() -> None:
    """
    Read a JSON or YAML config file and apply values as env-var fallbacks.

    Env vars already set in the process environment are never overridden —
    explicit env vars always take precedence over the config file.
    """
    path = os.getenv("ALUMINATAI_CONFIG", "")
    if not path:
        candidates = [
            "aluminatai.json",
            "aluminatai.yaml",
            "aluminatai.yml",
            os.path.expanduser("~/.config/aluminatai/config.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
    if not path:
        return

    data: dict = {}
    try:
        with open(path) as f:
            raw = f.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore[import-untyped]
                data = yaml.safe_load(raw) or {}
            except ImportError:
                print(
                    "[aluminatai] YAML config requires PyYAML — "
                    "install with: pip install 'aluminatiai[observability]'",
                    file=sys.stderr,
                )
                return
        else:
            data = json.loads(raw)
    except Exception as exc:
        print(f"[aluminatai] Failed to load config file {path!r}: {exc}", file=sys.stderr)
        return

    applied = []
    for key, value in data.items():
        env_var = _CONFIG_KEY_TO_ENV.get(key)
        if not env_var:
            continue
        if env_var in os.environ:
            continue  # env var wins
        # Booleans → "1" or "" so downstream bool() and .lower() parsing works
        if isinstance(value, bool):
            os.environ[env_var] = "1" if value else ""
        else:
            os.environ[env_var] = str(value)
        applied.append(f"{key}={value!r}")

    if applied:
        print(f"[aluminatai] Config file {path!r}: applied {', '.join(applied)}", file=sys.stderr)


# Apply file-based config before any constants are evaluated so that all
# os.getenv() calls below see the merged environment.
_load_config_file()

# ── API Configuration ─────────────────────────────────────────────────────────

API_ENDPOINT = os.getenv("ALUMINATAI_API_ENDPOINT", "https://aluminatiai.com/v1/metrics/ingest")
API_KEY = os.getenv("ALUMINATAI_API_KEY", "")

# ── Upload Configuration ──────────────────────────────────────────────────────

UPLOAD_ENABLED = bool(API_KEY)
UPLOAD_INTERVAL = int(os.getenv("UPLOAD_INTERVAL", "60"))        # seconds between flush calls
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))   # metrics per HTTP request

# Exponential backoff
UPLOAD_MAX_RETRIES = int(os.getenv("UPLOAD_MAX_RETRIES", "5"))
UPLOAD_MAX_RETRY_DELAY = int(os.getenv("UPLOAD_MAX_RETRY_DELAY", "60"))  # seconds cap

# ── WAL (Write-Ahead Log) ─────────────────────────────────────────────────────

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
WAL_DIR = DATA_DIR / "wal"
WAL_MAX_AGE_HOURS = int(os.getenv("WAL_MAX_AGE_HOURS", "24"))
WAL_MAX_MB = int(os.getenv("WAL_MAX_MB", "512"))

# Legacy — kept for backward compat with old uploader.py imports
ENABLE_LOCAL_BACKUP = True  # WAL is always active when DATA_DIR is writable

# ── Sampling ──────────────────────────────────────────────────────────────────

SAMPLE_INTERVAL = float(os.getenv("SAMPLE_INTERVAL", "5.0"))     # seconds between NVML reads

# ── Scheduler Integration ─────────────────────────────────────────────────────

SCHEDULER_POLL_INTERVAL = int(os.getenv("SCHEDULER_POLL_INTERVAL", "30"))

# ── Heartbeat ─────────────────────────────────────────────────────────────────

HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "300"))  # 5 min

# ── TLS / Proxy ───────────────────────────────────────────────────────────────

HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")
CA_BUNDLE = os.getenv("ALUMINATAI_CA_BUNDLE", "")      # path to company CA PEM
CLIENT_CERT = os.getenv("ALUMINATAI_CLIENT_CERT", "")  # mTLS client cert path
CLIENT_KEY = os.getenv("ALUMINATAI_CLIENT_KEY", "")    # mTLS client key path

# ── Prometheus Metrics Server ─────────────────────────────────────────────────

METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))         # 0 = disabled
METRICS_BIND_HOST = os.getenv("METRICS_BIND_HOST", "")        # "" = 0.0.0.0 (all interfaces)
METRICS_BASIC_AUTH = os.getenv("METRICS_BASIC_AUTH", "")      # "user:pass" or "" (no auth)

# ── Run Modes ─────────────────────────────────────────────────────────────────

OFFLINE_MODE = os.getenv("OFFLINE_MODE", "").lower() in ("1", "true", "yes")

# Collect + attribute + Prometheus, but skip all HTTP uploads and WAL writes.
# Useful for debugging attribution and config without sending data.
DRY_RUN = os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes")

# Disable cloud uploads entirely; run only local Prometheus metrics.
# Implies no WAL writes (unlike OFFLINE_MODE which writes to WAL).
PROMETHEUS_ONLY = os.getenv("PROMETHEUS_ONLY", "").lower() in ("1", "true", "yes")

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# "text" (human-readable, default) or "json" (newline-delimited JSON for ELK/Loki)
LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))

# ── Attribution ───────────────────────────────────────────────────────────────

ATTRIBUTION_CONFIG = os.getenv("ALUMINATAI_ATTRIBUTION_CONFIG", "")


def _parse_trusted_uids() -> set:
    """Parse ALUMINATAI_TRUSTED_UIDS=0,1000,1001 into a set of ints."""
    result: set[int] = set()
    for part in os.getenv("ALUMINATAI_TRUSTED_UIDS", "").split(","):
        part = part.strip()
        if part:
            try:
                result.add(int(part))
            except ValueError:
                pass
    return result


TRUSTED_UIDS: set[int] = _parse_trusted_uids()

# ── Ensure directories exist ──────────────────────────────────────────────────

DATA_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
WAL_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
LOG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
