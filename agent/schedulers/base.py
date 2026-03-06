"""
Base classes for scheduler adapters.

Every scheduler adapter must implement SchedulerAdapter to provide
a consistent interface for mapping GPU indices to job metadata.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class JobMetadata:
    """Metadata for a single GPU job, used for energy attribution."""

    job_id: str
    job_name: str
    team_id: str
    model_tag: str
    scheduler_source: str       # "kubernetes" | "slurm" | "runai" | "manual"
    gpu_indices: list[int]
    user_email: str = "unknown"
    start_time: str = ""


class SchedulerAdapter(ABC):
    """
    Abstract base class for scheduler adapters.

    Implementations intercept job metadata from a specific scheduler
    (Kubernetes, Slurm, Run:ai) and maintain a mapping from GPU index
    to the job currently using that GPU.
    """

    @abstractmethod
    def discover_jobs(self) -> list[JobMetadata]:
        """
        Scan for active GPU jobs on this node.

        Returns a list of all jobs that are currently using GPUs.
        Also updates the internal gpu_index → job mapping.
        """
        ...

    @abstractmethod
    def gpu_to_job(self, gpu_index: int) -> Optional[JobMetadata]:
        """
        Look up which job owns a specific GPU index.

        Returns None if the GPU is idle or attribution is unavailable.
        """
        ...

    def resolve_job(self, job_id: str) -> Optional["JobMetadata"]:
        """
        Back-lookup a specific job by ID or name (for PidResolver).

        Called when a process's environ contains a scheduler-specific
        job identifier. Override in adapters that support it.
        Default returns None (unsupported).
        """
        return None

    def resolve_pod_by_uid(self, pod_uid: str) -> Optional["JobMetadata"]:
        """
        Back-lookup a Kubernetes pod by its UID (for PidResolver).

        Override in KubernetesAdapter. Default returns None.
        """
        return None

    @property
    def name(self) -> str:
        """Human-readable adapter name."""
        return self.__class__.__name__


class NullAdapter(SchedulerAdapter):
    """
    Fallback adapter for standalone mode — no scheduler detected.

    Returns no attribution. The agent still collects power metrics,
    they just won't be tagged with job/team/model metadata.
    """

    def discover_jobs(self) -> list[JobMetadata]:
        return []

    def gpu_to_job(self, gpu_index: int) -> Optional[JobMetadata]:
        return None

    @property
    def name(self) -> str:
        return "none (standalone)"
