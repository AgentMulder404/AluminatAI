"""
Metrics uploader for AluminatAI API
Handles uploading GPU metrics with retry logic and local backup
"""
import requests
import json
import time
from typing import List, Dict
from pathlib import Path
import logging

from config import API_ENDPOINT, API_KEY, UPLOAD_BATCH_SIZE, DATA_DIR, ENABLE_LOCAL_BACKUP

logger = logging.getLogger(__name__)


class MetricsUploader:
    """Handles uploading metrics to AluminatAI API"""

    def __init__(self, api_endpoint: str = API_ENDPOINT, api_key: str = API_KEY):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'X-API-Key': api_key,
        })

        # Metrics buffer
        self.buffer: List[Dict] = []
        self.failed_uploads_dir = DATA_DIR / 'failed_uploads'

        if ENABLE_LOCAL_BACKUP:
            self.failed_uploads_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📤 Uploader initialized: {api_endpoint}")

    def add_metrics(self, metrics: List[Dict]):
        """Add metrics to upload buffer"""
        self.buffer.extend(metrics)
        logger.debug(f"Added {len(metrics)} metrics to buffer (total: {len(self.buffer)})")

    def upload_batch(self, metrics: List[Dict]) -> bool:
        """Upload a batch of metrics to API"""
        try:
            response = self.session.post(
                self.api_endpoint,
                json=metrics,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Uploaded {len(metrics)} metrics successfully")
                return True
            elif response.status_code == 401:
                logger.error("❌ API key invalid or expired")
                return False
            elif response.status_code == 429:
                logger.warning("⚠️  Rate limit exceeded, will retry later")
                return False
            else:
                logger.warning(f"⚠️  Upload failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    logger.warning(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    pass
                return False

        except requests.Timeout:
            logger.error("❌ Upload timeout (30s)")
            return False
        except requests.ConnectionError as e:
            logger.error(f"❌ Connection error: {e}")
            return False
        except requests.RequestException as e:
            logger.error(f"❌ Upload error: {e}")
            return False

    def flush(self) -> int:
        """Upload all buffered metrics"""
        if not self.buffer:
            return 0

        uploaded = 0
        failed_batches = []

        # Split into batches
        for i in range(0, len(self.buffer), UPLOAD_BATCH_SIZE):
            batch = self.buffer[i:i + UPLOAD_BATCH_SIZE]

            if self.upload_batch(batch):
                uploaded += len(batch)
            else:
                failed_batches.append(batch)

        # Save failed uploads locally
        if failed_batches and ENABLE_LOCAL_BACKUP:
            self._save_failed_uploads(failed_batches)

        # Clear buffer
        total_metrics = len(self.buffer)
        self.buffer = []

        if uploaded > 0:
            logger.info(f"📊 Upload summary: {uploaded}/{total_metrics} succeeded")

        return uploaded

    def _save_failed_uploads(self, batches: List[List[Dict]]):
        """Save failed uploads to disk for retry"""
        timestamp = int(time.time())
        filepath = self.failed_uploads_dir / f"failed_{timestamp}.json"

        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'batches': batches,
                }, f)
            logger.info(f"💾 Saved {sum(len(b) for b in batches)} failed metrics to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save failed uploads: {e}")

    def retry_failed_uploads(self) -> int:
        """Retry uploading previously failed batches"""
        if not ENABLE_LOCAL_BACKUP:
            return 0

        uploaded = 0
        failed_files = list(self.failed_uploads_dir.glob('failed_*.json'))

        if not failed_files:
            return 0

        logger.info(f"🔄 Retrying {len(failed_files)} failed upload batch(es)")

        for filepath in failed_files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                all_succeeded = True
                for batch in data['batches']:
                    if self.upload_batch(batch):
                        uploaded += len(batch)
                    else:
                        all_succeeded = False
                        break

                # Delete file if all batches succeeded
                if all_succeeded:
                    filepath.unlink()
                    logger.info(f"✅ Retried and cleared {filepath.name}")

            except Exception as e:
                logger.error(f"Failed to retry {filepath}: {e}")

        return uploaded

    def get_status(self) -> Dict:
        """Get uploader status"""
        failed_count = 0
        if ENABLE_LOCAL_BACKUP:
            failed_files = list(self.failed_uploads_dir.glob('failed_*.json'))
            for filepath in failed_files:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        for batch in data['batches']:
                            failed_count += len(batch)
                except:
                    pass

        return {
            'buffer_size': len(self.buffer),
            'failed_metrics_count': failed_count,
            'api_endpoint': self.api_endpoint,
            'has_api_key': bool(self.api_key),
        }
