"""
ProcessProbe: Query NVML for compute processes on a GPU device and
read their environment variables from /proc/<pid>/environ (Linux).
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)

_IS_LINUX = sys.platform.startswith("linux")


@dataclass
class ProcessInfo:
    pid: int
    gpu_memory_bytes: int
    environ: dict[str, str] = field(default_factory=dict)


class ProcessProbe:
    """
    Queries NVML for compute processes on a GPU handle and enriches each
    with environment variables read from /proc/<pid>/environ.

    On non-Linux platforms (Windows, macOS) environ is always empty and
    the caller falls back to scheduler-poll attribution gracefully.
    """

    def query(self, handle, gpu_index: int) -> list[ProcessInfo]:
        """
        Return a list of ProcessInfo for each compute process on this GPU.

        Args:
            handle:    pynvml device handle
            gpu_index: for logging only

        Returns:
            List of ProcessInfo (may be empty if no processes or NVML error)
        """
        if not NVML_AVAILABLE:
            return []

        try:
            nvml_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError as e:
            logger.debug(f"GPU {gpu_index}: nvmlDeviceGetComputeRunningProcesses failed: {e}")
            nvml_procs = []

        # Fallback: some MIG configurations and inference servers (TensorRT, graphics APIs)
        # only appear under graphics processes, not compute.
        if not nvml_procs:
            try:
                nvml_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                if nvml_procs:
                    logger.debug(f"GPU {gpu_index}: using graphics processes fallback ({len(nvml_procs)} procs)")
            except pynvml.NVMLError:
                pass

        if not nvml_procs:
            return []

        results: list[ProcessInfo] = []
        for p in nvml_procs:
            environ = self._read_environ(p.pid) if _IS_LINUX else {}
            results.append(ProcessInfo(
                pid=p.pid,
                gpu_memory_bytes=p.usedGpuMemory or 0,
                environ=environ,
            ))

        return results

    def _read_environ(self, pid: int) -> dict[str, str]:
        """
        Read /proc/<pid>/environ and parse into a key=value dict.

        Returns empty dict on PermissionError (different user) or if
        the process has already exited (FileNotFoundError).
        """
        path = f"/proc/{pid}/environ"
        try:
            with open(path, "rb") as f:
                data = f.read()
        except PermissionError:
            logger.debug(f"PID {pid}: no permission to read environ (different user)")
            return {}
        except FileNotFoundError:
            logger.debug(f"PID {pid}: process exited before environ read")
            return {}
        except OSError as e:
            logger.debug(f"PID {pid}: error reading environ: {e}")
            return {}

        env: dict[str, str] = {}
        for entry in data.split(b"\x00"):
            if b"=" in entry:
                k, _, v = entry.partition(b"=")
                try:
                    env[k.decode("utf-8", errors="replace")] = v.decode("utf-8", errors="replace")
                except Exception:
                    pass
        return env
