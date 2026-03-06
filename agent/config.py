"""
Configuration for AluminatAI GPU Agent v0.2.0
All settings read from environment variables with sensible defaults.
"""
import os
from pathlib import Path

# ── Version ───────────────────────────────────────────────────────────────────

AGENT_VERSION = "0.2.0"

# ── API Configuration ─────────────────────────────────────────────────────────

API_ENDPOINT = os.getenv("ALUMINATAI_API_ENDPOINT", "https://aluminatiai.com/api/metrics/ingest")
API_KEY = os.getenv("ALUMINATAI_API_KEY", "")

# ── Upload Configuration ──────────────────────────────────────────────────────

UPLOAD_ENABLED = bool(API_KEY)
UPLOAD_INTERVAL = int(os.getenv("UPLOAD_INTERVAL", "60"))        # seconds between flush calls
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))   # metrics per HTTP request

# Exponential backoff (Phase 2)
UPLOAD_MAX_RETRIES = int(os.getenv("UPLOAD_MAX_RETRIES", "5"))
UPLOAD_MAX_RETRY_DELAY = int(os.getenv("UPLOAD_MAX_RETRY_DELAY", "60"))  # seconds cap

# ── WAL (Write-Ahead Log) ─────────────────────────────────────────────────────
# Replaces the old failed_uploads JSON file heap (Phase 2)

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

# ── TLS / Proxy (Phase 2) ─────────────────────────────────────────────────────

HTTPS_PROXY = os.getenv("HTTPS_PROXY", "")
CA_BUNDLE = os.getenv("ALUMINATAI_CA_BUNDLE", "")      # path to company CA PEM
CLIENT_CERT = os.getenv("ALUMINATAI_CLIENT_CERT", "")  # mTLS client cert path
CLIENT_KEY = os.getenv("ALUMINATAI_CLIENT_KEY", "")    # mTLS client key path

# ── Prometheus Metrics Server (Phase 2) ───────────────────────────────────────

METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))  # 0 = disabled

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))

# ── Ensure directories exist ──────────────────────────────────────────────────

DATA_DIR.mkdir(parents=True, exist_ok=True)
WAL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
