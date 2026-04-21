"""
Intel RAPL energy reader — CPU + DRAM power monitoring via sysfs.

Reads /sys/class/powercap/intel-rapl:*/energy_uj counters with overflow
handling. Auto-disabled on non-Linux or when sysfs is not readable.

Usage:
    reader = RaplReader()
    if reader.available:
        snapshot = reader.read()
        # ... wait ...
        delta = reader.read()
        cpu_watts = delta.package_watts
        ram_watts = delta.dram_watts
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_RAPL_BASE = Path("/sys/class/powercap")


@dataclass
class RaplReading:
    package_energy_uj: int
    dram_energy_uj: int
    timestamp: float
    package_watts: float = 0.0
    dram_watts: float = 0.0


class RaplReader:
    """Read Intel RAPL energy counters from sysfs."""

    def __init__(self):
        self._available = False
        self._package_path: Optional[Path] = None
        self._dram_path: Optional[Path] = None
        self._max_energy_uj: int = 0
        self._last: Optional[RaplReading] = None

        if not sys.platform.startswith("linux"):
            logger.debug("RAPL unavailable: not Linux")
            return

        pkg = _RAPL_BASE / "intel-rapl:0" / "energy_uj"
        if not pkg.exists():
            logger.debug("RAPL unavailable: %s not found", pkg)
            return

        try:
            pkg.read_text()
        except PermissionError:
            logger.debug("RAPL unavailable: permission denied on %s", pkg)
            return

        self._package_path = pkg
        self._available = True

        # Try to find DRAM domain (usually intel-rapl:0:2 but varies)
        for subdir in sorted((_RAPL_BASE / "intel-rapl:0").iterdir()):
            name_file = subdir / "name"
            if name_file.exists():
                try:
                    name = name_file.read_text().strip()
                    if name == "dram":
                        self._dram_path = subdir / "energy_uj"
                        break
                except (PermissionError, OSError):
                    continue

        # Read max energy range for overflow handling
        max_path = _RAPL_BASE / "intel-rapl:0" / "max_energy_range_uj"
        if max_path.exists():
            try:
                self._max_energy_uj = int(max_path.read_text().strip())
            except (ValueError, PermissionError):
                self._max_energy_uj = 2**32

        logger.info("RAPL enabled: package=%s dram=%s",
                     self._package_path, self._dram_path or "not found")

    @property
    def available(self) -> bool:
        return self._available

    def read(self) -> Optional[RaplReading]:
        """Read current RAPL counters and compute watts since last reading."""
        if not self._available or not self._package_path:
            return None

        try:
            pkg_uj = int(self._package_path.read_text().strip())
        except (ValueError, PermissionError, OSError):
            return None

        dram_uj = 0
        if self._dram_path:
            try:
                dram_uj = int(self._dram_path.read_text().strip())
            except (ValueError, PermissionError, OSError):
                pass

        now = time.monotonic()
        reading = RaplReading(
            package_energy_uj=pkg_uj,
            dram_energy_uj=dram_uj,
            timestamp=now,
        )

        if self._last is not None:
            dt = now - self._last.timestamp
            if dt > 0:
                pkg_delta = self._delta_with_overflow(
                    self._last.package_energy_uj, pkg_uj
                )
                dram_delta = self._delta_with_overflow(
                    self._last.dram_energy_uj, dram_uj
                )
                reading.package_watts = pkg_delta / (dt * 1_000_000)
                reading.dram_watts = dram_delta / (dt * 1_000_000)

        self._last = reading
        return reading

    def _delta_with_overflow(self, prev: int, curr: int) -> int:
        """Handle counter overflow (wraps at max_energy_range_uj)."""
        if curr >= prev:
            return curr - prev
        return (self._max_energy_uj - prev) + curr
