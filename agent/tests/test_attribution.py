"""
Attribution engine unit tests (Phase 3 — DDP grouping).

Tests:
  1. DDP: 8 PIDs with same SLURM_JOB_ID → 1 result, gpu_fraction=1.0
  2. Multi-tenant: 2 PIDs from different jobs (60/40 mem split) → 2 results summing to 1.0
  3. No-PID scheduler fallback → confidence="scheduler_poll", gpu_fraction=1.0
  4. No-PID, no scheduler, ALUMINATAI_IDLE_TEAM set → confidence="idle"
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock, patch

# Allow running from repo root or agent/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attribution.engine import AttributionEngine, AttributionResult
from attribution.process_probe import ProcessInfo, ProcessProbe
from attribution.pid_resolver import PidResolver
from schedulers.base import JobMetadata, NullAdapter, SchedulerAdapter


# ── Test helpers ──────────────────────────────────────────────────────────────


def make_job(job_id: str, team: str = "team-a", model: str = "gpt") -> JobMetadata:
    return JobMetadata(
        job_id=job_id,
        job_name=f"job-{job_id}",
        team_id=team,
        model_tag=model,
        scheduler_source="slurm",
        gpu_indices=[0],
    )


def make_proc(pid: int, gpu_mem: int, slurm_job_id: str = "") -> ProcessInfo:
    environ = {}
    if slurm_job_id:
        environ["SLURM_JOB_ID"] = slurm_job_id
    return ProcessInfo(pid=pid, gpu_memory_bytes=gpu_mem, environ=environ)


class MockScheduler(SchedulerAdapter):
    """Minimal scheduler that resolves known job IDs."""

    def __init__(self, jobs: List[JobMetadata]):
        self._jobs = {j.job_id: j for j in jobs}
        self._gpu_map: dict[int, JobMetadata] = {}
        for j in jobs:
            for idx in j.gpu_indices:
                self._gpu_map[idx] = j

    def discover_jobs(self) -> List[JobMetadata]:
        return list(self._jobs.values())

    def gpu_to_job(self, gpu_index: int) -> Optional[JobMetadata]:
        return self._gpu_map.get(gpu_index)

    def resolve_job(self, job_id: str) -> Optional[JobMetadata]:
        return self._jobs.get(job_id)

    @property
    def name(self) -> str:
        return "mock-slurm"


def build_engine(procs: List[ProcessInfo], scheduler: SchedulerAdapter) -> AttributionEngine:
    probe = MagicMock(spec=ProcessProbe)
    probe.query.return_value = procs
    resolver = PidResolver(scheduler)
    return AttributionEngine(probe, resolver, scheduler)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestDDPGrouping(unittest.TestCase):
    """Case 1: Multiple PIDs from the same Slurm job → single attribution result."""

    def test_8_pids_same_job_yields_fraction_1(self):
        job = make_job("JOB_001", team="ml-team", model="llama3")
        scheduler = MockScheduler([job])

        # 8 DDP workers, all in the same Slurm job, equal memory usage
        procs = [make_proc(pid=1000 + i, gpu_mem=10 * 1024**3, slurm_job_id="JOB_001")
                 for i in range(8)]

        engine = build_engine(procs, scheduler)
        results = engine.resolve(handle=None, gpu_index=0, total_power_w=300.0, energy_delta_j=1500.0)

        self.assertEqual(len(results), 1, "8 DDP workers should collapse to 1 attribution result")
        r = results[0]
        self.assertEqual(r.job_id, "JOB_001")
        self.assertEqual(r.team_id, "ml-team")
        self.assertAlmostEqual(r.gpu_fraction, 1.0, places=3)
        self.assertAlmostEqual(r.power_w, 300.0, places=1)
        self.assertEqual(r.confidence, "process")


class TestMultiTenantSplit(unittest.TestCase):
    """Case 2: 2 jobs sharing a GPU with 60/40 memory split."""

    def test_two_jobs_60_40_split(self):
        job_a = make_job("JOB_A", team="team-alpha")
        job_b = make_job("JOB_B", team="team-beta")
        scheduler = MockScheduler([job_a, job_b])

        total_mem = 80 * 1024**3  # 80 GB GPU
        procs = [
            make_proc(pid=2001, gpu_mem=int(total_mem * 0.60), slurm_job_id="JOB_A"),
            make_proc(pid=2002, gpu_mem=int(total_mem * 0.40), slurm_job_id="JOB_B"),
        ]

        engine = build_engine(procs, scheduler)
        results = engine.resolve(handle=None, gpu_index=0, total_power_w=400.0, energy_delta_j=2000.0)

        self.assertEqual(len(results), 2)

        fracs = {r.team_id: r.gpu_fraction for r in results}
        self.assertAlmostEqual(fracs["team-alpha"], 0.60, places=2)
        self.assertAlmostEqual(fracs["team-beta"], 0.40, places=2)

        total_frac = sum(r.gpu_fraction for r in results)
        self.assertAlmostEqual(total_frac, 1.0, places=3, msg="Fractions must sum to 1.0")

        total_energy = sum(r.energy_delta_j for r in results if r.energy_delta_j)
        self.assertAlmostEqual(total_energy, 2000.0, places=2)

        for r in results:
            self.assertEqual(r.confidence, "process")


class TestSchedulerFallback(unittest.TestCase):
    """Case 3: No running processes → fall back to scheduler.gpu_to_job()."""

    def test_no_procs_uses_scheduler_poll(self):
        job = make_job("JOB_SCHED", team="ops-team")
        job = JobMetadata(
            job_id="JOB_SCHED",
            job_name="training",
            team_id="ops-team",
            model_tag="bert",
            scheduler_source="slurm",
            gpu_indices=[0],
        )
        scheduler = MockScheduler([job])

        engine = build_engine([], scheduler)  # no processes
        results = engine.resolve(handle=None, gpu_index=0, total_power_w=200.0, energy_delta_j=1000.0)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.job_id, "JOB_SCHED")
        self.assertEqual(r.confidence, "scheduler_poll")
        self.assertAlmostEqual(r.gpu_fraction, 1.0)


class TestIdleFallback(unittest.TestCase):
    """Case 4: No processes and no scheduler job → idle attribution."""

    def test_idle_attribution(self):
        scheduler = NullAdapter()
        engine = build_engine([], scheduler)

        with patch.dict(os.environ, {"ALUMINATAI_IDLE_TEAM": "infra"}):
            results = engine.resolve(handle=None, gpu_index=0, total_power_w=50.0, energy_delta_j=250.0)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.team_id, "infra")
        self.assertEqual(r.confidence, "idle")
        self.assertEqual(r.model_tag, "idle")
        self.assertAlmostEqual(r.gpu_fraction, 1.0)

    def test_no_idle_team_returns_empty(self):
        scheduler = NullAdapter()
        engine = build_engine([], scheduler)

        env = {k: v for k, v in os.environ.items() if k != "ALUMINATAI_IDLE_TEAM"}
        with patch.dict(os.environ, env, clear=True):
            results = engine.resolve(handle=None, gpu_index=0, total_power_w=50.0, energy_delta_j=250.0)

        self.assertEqual(results, [], "No attribution config → empty list (backward compat)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
