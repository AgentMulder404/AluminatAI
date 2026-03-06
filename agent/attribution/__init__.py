"""
GPU Attribution Engine

Resolves which job(s) own a GPU at each sample and splits power
proportionally by GPU memory fraction when multiple processes share a device.
"""

from .engine import AttributionEngine, AttributionResult

__all__ = ["AttributionEngine", "AttributionResult"]
