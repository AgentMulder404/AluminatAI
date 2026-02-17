"""
Configuration for GPU Energy Agent
Reads from environment variables with sensible defaults
"""
import os
from pathlib import Path

# API Configuration
API_ENDPOINT = os.getenv('ALUMINATAI_API_ENDPOINT', 'https://aluminatiai-landing.vercel.app/api/metrics/ingest')
API_KEY = os.getenv('ALUMINATAI_API_KEY', '')

# Upload Configuration
UPLOAD_ENABLED = bool(API_KEY)
UPLOAD_INTERVAL = int(os.getenv('UPLOAD_INTERVAL', '60'))  # seconds
UPLOAD_BATCH_SIZE = int(os.getenv('UPLOAD_BATCH_SIZE', '100'))  # metrics per request

# Local Storage (fallback when upload fails)
DATA_DIR = Path(os.getenv('DATA_DIR', './data'))
ENABLE_LOCAL_BACKUP = os.getenv('ENABLE_LOCAL_BACKUP', 'true').lower() == 'true'

# Sampling
SAMPLE_INTERVAL = float(os.getenv('SAMPLE_INTERVAL', '5.0'))  # seconds

# Scheduler integration
SCHEDULER_POLL_INTERVAL = int(os.getenv('SCHEDULER_POLL_INTERVAL', '30'))  # seconds

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_DIR = Path(os.getenv('LOG_DIR', './logs'))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
