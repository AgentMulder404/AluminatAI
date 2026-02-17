"""
Energy efficiency analysis for GPU workloads.

Provides:
- GPU architecture specs and roofline model calculations
- Efficiency curve building from observed fleet metrics
- Hardware Match Score computation for workload-GPU pairing
"""

from .gpu_specs import ArchSpec, ModelProfile, GPU_ARCHITECTURES, MODEL_PROFILES
from .curve_builder import EfficiencyCurveBuilder
from .hardware_match import HardwareMatchScorer, MatchResult

__all__ = [
    'ArchSpec',
    'ModelProfile',
    'GPU_ARCHITECTURES',
    'MODEL_PROFILES',
    'EfficiencyCurveBuilder',
    'HardwareMatchScorer',
    'MatchResult',
]
