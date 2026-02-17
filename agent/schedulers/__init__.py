"""
Scheduler adapters for GPU job attribution.

Detects which scheduler is managing compute resources and intercepts
job metadata to link GPU power metrics to specific jobs, teams, and models.
"""

from .base import SchedulerAdapter, JobMetadata, NullAdapter
from .detect import detect_scheduler

__all__ = [
    'SchedulerAdapter',
    'JobMetadata',
    'NullAdapter',
    'detect_scheduler',
]
