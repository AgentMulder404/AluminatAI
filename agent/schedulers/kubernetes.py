"""
Kubernetes scheduler adapter.

Watches GPU-requesting pods on the local node and maps GPU indices
to job metadata for energy attribution.

Labeling convention:
  aluminatai.io/team:  Team identifier (falls back to namespace)
  aluminatai.io/model: ML model tag (falls back to "untagged")
  aluminatai.io/user:  Submitter email (annotation, optional)

Deployment: Runs as a DaemonSet sidecar alongside the AluminatAI agent.
Requires RBAC: pods (get, list, watch) cluster-wide or per-namespace.
"""

import os
import logging
from typing import Optional

from .base import SchedulerAdapter, JobMetadata

logger = logging.getLogger(__name__)

try:
    from kubernetes import client, config
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    logger.debug("kubernetes package not installed — K8s adapter unavailable")


TEAM_LABEL = "aluminatai.io/team"
MODEL_LABEL = "aluminatai.io/model"
USER_ANNOTATION = "aluminatai.io/user"
GPU_RESOURCE = "nvidia.com/gpu"


class KubernetesAdapter(SchedulerAdapter):
    """
    Intercepts job metadata from Kubernetes pods requesting GPU resources.

    Correlation strategy:
    1. List running pods on this node that request nvidia.com/gpu
    2. Read NVIDIA_VISIBLE_DEVICES env var for GPU index mapping
    3. Extract team from label or namespace, model from label
    4. Derive stable job ID from owner references (Job/CronJob UID)
    """

    def __init__(self):
        if not K8S_AVAILABLE:
            raise RuntimeError(
                "kubernetes package required. Install with: pip install kubernetes"
            )

        # Detect cluster vs local kubeconfig
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        else:
            config.load_kube_config()
            logger.info("Loaded local kubeconfig")

        self._v1 = client.CoreV1Api()
        self._node_name = os.getenv("NODE_NAME", self._detect_node_name())
        self._gpu_job_map: dict[int, JobMetadata] = {}

        logger.info(f"KubernetesAdapter initialized for node: {self._node_name}")

    def discover_jobs(self) -> list[JobMetadata]:
        """Scan all running pods on this node that request GPU resources."""
        try:
            pods = self._v1.list_pod_for_all_namespaces(
                field_selector=(
                    f"spec.nodeName={self._node_name},"
                    f"status.phase=Running"
                ),
            )
        except Exception as e:
            logger.error(f"Failed to list pods: {e}")
            return []

        jobs: list[JobMetadata] = []
        new_map: dict[int, JobMetadata] = {}

        for pod in pods.items:
            gpu_count = self._get_gpu_request(pod)
            if gpu_count == 0:
                continue

            gpu_indices = self._resolve_gpu_indices(pod, gpu_count)
            labels = pod.metadata.labels or {}
            annotations = pod.metadata.annotations or {}

            start_time = ""
            if pod.status and pod.status.start_time:
                start_time = pod.status.start_time.isoformat()

            metadata = JobMetadata(
                job_id=self._derive_job_id(pod),
                job_name=pod.metadata.name,
                team_id=self._extract_team(pod),
                model_tag=labels.get(MODEL_LABEL, "untagged"),
                scheduler_source="kubernetes",
                gpu_indices=gpu_indices,
                user_email=annotations.get(USER_ANNOTATION, "unknown"),
                start_time=start_time,
            )

            jobs.append(metadata)
            for idx in gpu_indices:
                new_map[idx] = metadata

        self._gpu_job_map = new_map
        logger.debug(f"Discovered {len(jobs)} GPU jobs across {len(new_map)} GPUs")
        return jobs

    def gpu_to_job(self, gpu_index: int) -> Optional[JobMetadata]:
        return self._gpu_job_map.get(gpu_index)

    # ── Private helpers ──────────────────────────────────────────────

    def _get_gpu_request(self, pod) -> int:
        """Sum GPU requests across all containers in the pod."""
        total = 0
        for container in (pod.spec.containers or []):
            if container.resources and container.resources.requests:
                gpu_req = container.resources.requests.get(GPU_RESOURCE, 0)
                total += int(gpu_req)
            if container.resources and container.resources.limits:
                gpu_limit = container.resources.limits.get(GPU_RESOURCE, 0)
                total = max(total, int(gpu_limit))
        return total

    def _resolve_gpu_indices(self, pod, gpu_count: int) -> list[int]:
        """
        Resolve actual GPU indices from NVIDIA_VISIBLE_DEVICES.

        Priority:
        1. NVIDIA_VISIBLE_DEVICES env var (set by device plugin)
        2. Sequential allocation fallback
        """
        for container in (pod.spec.containers or []):
            for env_var in (container.env or []):
                if env_var.name == "NVIDIA_VISIBLE_DEVICES" and env_var.value:
                    value = env_var.value.strip()
                    if value == "all":
                        return list(range(gpu_count))
                    if value == "none" or value == "void":
                        return []
                    try:
                        return [int(i.strip()) for i in value.split(",")]
                    except ValueError:
                        logger.warning(
                            f"Could not parse NVIDIA_VISIBLE_DEVICES='{value}' "
                            f"for pod {pod.metadata.name}"
                        )

        # Fallback: assume sequential from 0
        logger.debug(
            f"No NVIDIA_VISIBLE_DEVICES for pod {pod.metadata.name}, "
            f"assuming indices 0..{gpu_count - 1}"
        )
        return list(range(gpu_count))

    def _extract_team(self, pod) -> str:
        """
        Derive team identifier.

        Priority: aluminatai.io/team label > namespace > "default"
        """
        labels = pod.metadata.labels or {}
        if TEAM_LABEL in labels:
            return labels[TEAM_LABEL]
        return pod.metadata.namespace or "default"

    def _derive_job_id(self, pod) -> str:
        """
        Derive a stable job identifier.

        If the pod is owned by a Job or CronJob, use the owner's UID
        so all pods in the same job share an ID. Otherwise use the pod UID.
        """
        for ref in (pod.metadata.owner_references or []):
            if ref.kind in ("Job", "CronJob", "MPIJob", "PyTorchJob", "TFJob"):
                return str(ref.uid)
        return str(pod.metadata.uid)

    def _detect_node_name(self) -> str:
        """Detect current node name from hostname."""
        import socket
        hostname = socket.gethostname()
        logger.debug(f"Detected node name from hostname: {hostname}")
        return hostname

    @property
    def name(self) -> str:
        return f"kubernetes (node={self._node_name})"
