"""
AttributionEngine: Resolve GPU power attribution per sample.

For each GPU handle + power reading, returns one or more AttributionResult
objects representing each job's fractional share of the GPU power.

Attribution confidence levels:
  "process"        — resolved via nvmlDeviceGetComputeRunningProcesses + PID env
  "scheduler_poll" — resolved via scheduler.gpu_to_job() (old behaviour)
  "inferred"       — heuristic / partial resolution
  "idle"           — GPU is idle; billed to ALUMINATAI_IDLE_TEAM
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .process_probe import ProcessProbe
from .pid_resolver import PidResolver

if TYPE_CHECKING:
    from schedulers.base import SchedulerAdapter, JobMetadata

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    team_id: str
    model_tag: str
    job_id: str
    scheduler_source: str
    power_w: float
    gpu_fraction: float                   # 0.0–1.0
    energy_delta_j: Optional[float]
    confidence: str                       # "process" | "scheduler_poll" | "inferred" | "idle"


class AttributionEngine:
    def __init__(
        self,
        probe: ProcessProbe,
        resolver: PidResolver,
        scheduler: "SchedulerAdapter",
    ):
        self._probe = probe
        self._resolver = resolver
        self._scheduler = scheduler

    def resolve(
        self,
        handle,
        gpu_index: int,
        total_power_w: float,
        energy_delta_j: Optional[float],
    ) -> list[AttributionResult]:
        """
        Return attribution result(s) for one GPU at one sample time.

        Steps:
          1. Query running compute processes via NVML
          2. Resolve each process to a job, group by job, split power by memory fraction
          3. Fallback: scheduler poll (single winner)
          4. Fallback: idle attribution if ALUMINATAI_IDLE_TEAM is set
          5. Return [] if no attribution configured (backward compat)
        """
        processes = self._probe.query(handle, gpu_index)

        if processes:
            # Group by resolved job key, accumulate GPU memory bytes
            by_key: dict[str, tuple[Optional["JobMetadata"], int]] = {}
            for proc in processes:
                job = self._resolver.resolve(proc)
                key = job.job_id if job else f"pid:{proc.pid}"
                _, mem = by_key.get(key, (job, 0))
                by_key[key] = (job, mem + proc.gpu_memory_bytes)

            total_mem = sum(m for _, m in by_key.values()) or 1
            results: list[AttributionResult] = []

            for key, (job, mem) in by_key.items():
                frac = mem / total_mem
                if job:
                    team_id = job.team_id
                    model_tag = job.model_tag
                    job_id = job.job_id
                    scheduler_source = job.scheduler_source
                else:
                    # Unresolved process — emit under sentinel values
                    team_id = os.getenv("ALUMINATAI_IDLE_TEAM", "unresolved")
                    model_tag = "untagged"
                    job_id = key
                    scheduler_source = "manual"

                results.append(AttributionResult(
                    team_id=team_id,
                    model_tag=model_tag,
                    job_id=job_id,
                    scheduler_source=scheduler_source,
                    power_w=round(total_power_w * frac, 3),
                    gpu_fraction=round(frac, 4),
                    energy_delta_j=round(energy_delta_j * frac, 4) if energy_delta_j is not None else None,
                    confidence="process",
                ))

            return results

        # Fallback: scheduler poll (current/old behaviour)
        job = self._scheduler.gpu_to_job(gpu_index)
        if job:
            return [AttributionResult(
                team_id=job.team_id,
                model_tag=job.model_tag,
                job_id=job.job_id,
                scheduler_source=job.scheduler_source,
                power_w=round(total_power_w, 3),
                gpu_fraction=1.0,
                energy_delta_j=energy_delta_j,
                confidence="scheduler_poll",
            )]

        # Fallback: idle
        idle_team = os.getenv("ALUMINATAI_IDLE_TEAM")
        if idle_team:
            return [AttributionResult(
                team_id=idle_team,
                model_tag="idle",
                job_id="idle",
                scheduler_source="manual",
                power_w=round(total_power_w, 3),
                gpu_fraction=1.0,
                energy_delta_j=energy_delta_j,
                confidence="idle",
            )]

        # No attribution configured — emit raw (backward compat)
        return []
