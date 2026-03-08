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
Metrics uploader for AluminatAI API — v0.2.0

Features:
  - Exponential backoff with jitter (1s → 2s → 4s → 8s → 16s, capped 60s)
  - Respects Retry-After on 429
  - Permanent failure on 401/403 → writes to WAL immediately
  - WAL-based local buffer (append-only newline-delimited JSON)
  - TLS / mTLS / proxy support from config
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Dict, List

try:
    import fcntl  # POSIX only — not available on Windows
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

import requests

from config import (
    API_ENDPOINT, API_KEY, UPLOAD_BATCH_SIZE,
    WAL_DIR, WAL_MAX_AGE_HOURS, WAL_MAX_MB,
    UPLOAD_MAX_RETRIES, UPLOAD_MAX_RETRY_DELAY,
    HTTPS_PROXY, CA_BUNDLE, CLIENT_CERT, CLIENT_KEY,
)

logger = logging.getLogger(__name__)

# ── WAL helpers ───────────────────────────────────────────────────────────────

WAL_FILE = WAL_DIR / "metrics.wal"


def _wal_append(batch: List[Dict]) -> None:
    """Append a batch to the WAL as newline-delimited JSON entries.

    Uses an exclusive flock on POSIX to prevent concurrent writers (e.g. two
    agent processes during a K8s rolling update) from interleaving partial
    writes and corrupting the WAL.
    """
    WAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(WAL_FILE, "a") as f:
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for row in batch:
                    entry = {"ts": time.time(), "row": row}
                    f.write(json.dumps(entry) + "\n")
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)
        logger.info("WAL: appended %d rows → %s", len(batch), WAL_FILE)
    except OSError as exc:
        logger.error("WAL write failed: %s", exc)


def _wal_read_valid() -> List[Dict]:
    """Read WAL, filter by TTL, enforce size cap, return metric dicts."""
    if not WAL_FILE.exists():
        return []

    cutoff = time.time() - WAL_MAX_AGE_HOURS * 3600
    rows: list[dict] = []
    raw_lines: list[str] = []

    try:
        with open(WAL_FILE) as f:
            raw_lines = f.readlines()
    except OSError:
        return []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("ts", 0) >= cutoff:
                rows.append(entry["row"])
        except (json.JSONDecodeError, KeyError):
            pass

    # Size cap: drop oldest if WAL is too large
    wal_mb = WAL_FILE.stat().st_size / (1024 * 1024) if WAL_FILE.exists() else 0
    if wal_mb > WAL_MAX_MB:
        drop = max(0, len(rows) - len(rows) // 2)
        rows = rows[drop:]
        logger.warning("WAL exceeded %dMB — dropped %d oldest rows", WAL_MAX_MB, drop)

    return rows


def _wal_clear() -> None:
    """Delete the WAL after a successful full replay."""
    try:
        WAL_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _wal_rewrite(rows: List[Dict]) -> None:
    """Atomically replace the WAL with only the given rows.

    Writes to a .tmp sibling file first, then renames over the WAL so that a
    crash between _wal_clear() and _wal_append() can never lose data — the
    old WAL remains intact until the rename succeeds.
    """
    WAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = WAL_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            if _HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for row in rows:
                    entry = {"ts": time.time(), "row": row}
                    f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                if _HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)
        tmp.replace(WAL_FILE)  # atomic on POSIX
        logger.info("WAL: rewrote %d rows → %s", len(rows), WAL_FILE)
    except OSError as exc:
        logger.error("WAL rewrite failed: %s", exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# ── Session factory ───────────────────────────────────────────────────────────


def _build_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    })
    if HTTPS_PROXY:
        session.proxies = {"https": HTTPS_PROXY, "http": HTTPS_PROXY}
    # TLS verification: always enabled. Never set verify=False — self-signed certs
    # are rejected by default. CA_BUNDLE overrides the system CA store (e.g. for
    # corporate proxy MITM certs); omit it to use the bundled Mozilla CA store.
    session.verify = CA_BUNDLE if CA_BUNDLE else True
    if CLIENT_CERT and CLIENT_KEY:
        session.cert = (CLIENT_CERT, CLIENT_KEY)
    return session


# ── Uploader ──────────────────────────────────────────────────────────────────


class MetricsUploader:
    """Upload GPU metrics to the AluminatAI API with backoff + WAL durability."""

    def __init__(self, api_endpoint: str = API_ENDPOINT, api_key: str = API_KEY):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.session = _build_session(api_key)
        self.buffer: List[Dict] = []
        self._upload_success = 0
        self._upload_failure = 0
        logger.info("Uploader initialised → %s", api_endpoint)

    def add_metrics(self, metrics: List[Dict]) -> None:
        self.buffer.extend(metrics)

    def upload_batch(self, metrics: List[Dict]) -> bool:
        """
        Upload one batch with exponential backoff.

        Returns True on success.  Writes to WAL on permanent failure.
        """
        delay = 1.0
        for attempt in range(1, UPLOAD_MAX_RETRIES + 1):
            try:
                resp = self.session.post(self.api_endpoint, json=metrics, timeout=30)
            except requests.Timeout:
                logger.warning("Upload timeout (attempt %d/%d)", attempt, UPLOAD_MAX_RETRIES)
                self._sleep_with_jitter(delay)
                delay = min(delay * 2, UPLOAD_MAX_RETRY_DELAY)
                continue
            except requests.ConnectionError as exc:
                logger.warning("Connection error (attempt %d/%d): %s", attempt, UPLOAD_MAX_RETRIES, exc)
                self._sleep_with_jitter(delay)
                delay = min(delay * 2, UPLOAD_MAX_RETRY_DELAY)
                continue
            except requests.RequestException as exc:
                logger.error("Unrecoverable request error: %s", exc)
                _wal_append(metrics)
                self._upload_failure += 1
                return False

            if resp.status_code == 200:
                self._upload_success += len(metrics)
                return True

            if resp.status_code in (401, 403):
                logger.error("Permanent auth failure (%d) — check API key", resp.status_code)
                _wal_append(metrics)
                self._upload_failure += 1
                return False

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", delay))
                logger.warning("Rate-limited — waiting %.0fs", retry_after)
                time.sleep(retry_after + random.uniform(0, 1))
                delay = min(delay * 2, UPLOAD_MAX_RETRY_DELAY)
                continue

            # 5xx or other transient error
            logger.warning("HTTP %d (attempt %d/%d)", resp.status_code, attempt, UPLOAD_MAX_RETRIES)
            self._sleep_with_jitter(delay)
            delay = min(delay * 2, UPLOAD_MAX_RETRY_DELAY)

        # Exhausted retries
        logger.error("Upload failed after %d attempts — writing to WAL", UPLOAD_MAX_RETRIES)
        _wal_append(metrics)
        self._upload_failure += 1
        return False

    @staticmethod
    def _sleep_with_jitter(base_delay: float) -> None:
        jitter = random.uniform(-0.2 * base_delay, 0.2 * base_delay)
        time.sleep(max(0, base_delay + jitter))

    def flush(self) -> int:
        """Upload all buffered metrics. Returns number successfully uploaded."""
        if not self.buffer:
            return 0

        uploaded = 0
        remaining: list[dict] = []

        for i in range(0, len(self.buffer), UPLOAD_BATCH_SIZE):
            batch = self.buffer[i:i + UPLOAD_BATCH_SIZE]
            if self.upload_batch(batch):
                uploaded += len(batch)
            else:
                remaining.extend(batch)

        self.buffer = remaining
        if uploaded:
            logger.info("Flushed %d metrics", uploaded)
        return uploaded

    def retry_failed_uploads(self) -> int:
        """
        Replay the WAL at startup.  Returns number of metrics successfully re-uploaded.
        Clears the WAL if all entries are replayed.
        """
        rows = _wal_read_valid()
        if not rows:
            return 0

        logger.info("WAL replay: %d rows pending", len(rows))
        uploaded = 0
        failed: list[dict] = []

        for i in range(0, len(rows), UPLOAD_BATCH_SIZE):
            batch = rows[i:i + UPLOAD_BATCH_SIZE]
            if self.upload_batch(batch):
                uploaded += len(batch)
            else:
                failed.extend(batch)

        if not failed:
            _wal_clear()
            logger.info("WAL replay complete — all %d rows uploaded", uploaded)
        else:
            # Atomically rewrite WAL with only the rows that still need upload.
            # _wal_rewrite() writes to a .tmp file then renames, so a crash
            # between these two steps cannot silently drop metrics.
            _wal_rewrite(failed)
            logger.warning("WAL replay partial: %d uploaded, %d remain", uploaded, len(failed))

        return uploaded

    def get_status(self) -> Dict:
        wal_size = WAL_FILE.stat().st_size if WAL_FILE.exists() else 0
        return {
            "buffer_size": len(self.buffer),
            "wal_bytes": wal_size,
            "upload_success_total": self._upload_success,
            "upload_failure_total": self._upload_failure,
            "api_endpoint": self.api_endpoint,
            "has_api_key": bool(self.api_key),
        }
