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
PidResolver: Map a running process to a JobMetadata via scheduler back-lookup.

Resolution priority (first match wins):
  1. SLURM_JOB_ID in environ  → SlurmAdapter.resolve_job()
  2. RUNAI_JOB_NAME in environ → RunaiAdapter.resolve_job()
  3. KUBERNETES_SERVICE_HOST   → read cgroup for pod UID → KubernetesAdapter.resolve_pod_by_uid()
  4. ALUMINATAI_TEAM + ALUMINATAI_MODEL env vars (manual override)
  5. None (unresolved — power is still tracked under "pid:<pid>")
"""

import logging
import re
import sys
from typing import Optional, TYPE_CHECKING

from .process_probe import ProcessInfo

if TYPE_CHECKING:
    from schedulers.base import SchedulerAdapter, JobMetadata

logger = logging.getLogger(__name__)

_IS_LINUX = sys.platform.startswith("linux")


class PidResolver:
    def __init__(self, scheduler: "SchedulerAdapter"):
        self._scheduler = scheduler

    def resolve(self, proc: ProcessInfo) -> "Optional[JobMetadata]":
        env = proc.environ

        # 1. Slurm
        slurm_job_id = env.get("SLURM_JOB_ID")
        if slurm_job_id:
            job = self._scheduler.resolve_job(slurm_job_id)
            if job:
                return job

        # 2. Run:ai
        runai_job_name = env.get("RUNAI_JOB_NAME")
        if runai_job_name:
            job = self._scheduler.resolve_job(runai_job_name)
            if job:
                return job

        # 3. Kubernetes (via cgroup → pod UID)
        if env.get("KUBERNETES_SERVICE_HOST") and _IS_LINUX:
            pod_uid = self._read_pod_uid_from_cgroup(proc.pid)
            if pod_uid:
                job = self._scheduler.resolve_pod_by_uid(pod_uid)
                if job:
                    return job

        # 4. Manual ALUMINATAI env vars
        team = env.get("ALUMINATAI_TEAM")
        model = env.get("ALUMINATAI_MODEL", "untagged")
        if team:
            from schedulers.base import JobMetadata
            return JobMetadata(
                job_id=f"manual-pid-{proc.pid}",
                job_name=f"pid-{proc.pid}",
                team_id=team,
                model_tag=model,
                scheduler_source="manual",
                gpu_indices=[],
            )

        return None

    def _read_pod_uid_from_cgroup(self, pid: int) -> Optional[str]:
        """
        Parse the Kubernetes pod UID from /proc/<pid>/cgroup.

        cgroup v1 example:
          12:devices:/kubepods/burstable/pod<uid>/container-<id>
        cgroup v2 example:
          0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod<uid>.slice/...
        """
        path = f"/proc/{pid}/cgroup"
        try:
            with open(path, "r") as f:
                content = f.read()
        except OSError:
            return None

        # Match "pod<uuid>" pattern (RFC 4122 UUID)
        match = re.search(
            r"pod([0-9a-f]{8}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{4}[-_][0-9a-f]{12})",
            content,
            re.IGNORECASE,
        )
        if match:
            # Normalise underscores → hyphens (cgroup v2 sometimes uses underscores)
            return match.group(1).replace("_", "-")
        return None
