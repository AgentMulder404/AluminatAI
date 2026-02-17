"""
GPU architecture specifications and ML model profiles.

This module is the single source of truth for hardware specs used in
efficiency calculations. Values mirror the gpu_architectures and
model_profiles tables seeded in migration 007.

Roofline Model:
  For a given workload with arithmetic intensity I (FLOP/byte):
    Attainable TFLOPS = min(peak_tflops, memory_bw * I)

  Memory-bound workloads (low I): limited by bandwidth, compute underutilized
  Compute-bound workloads (high I): limited by TFLOPS, bandwidth sufficient
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchSpec:
    """Immutable GPU architecture specification."""

    name: str
    family: str
    tdp_w: float               # Thermal Design Power (Watts)
    fp16_tflops: float          # Peak FP16 throughput
    fp32_tflops: float          # Peak FP32 throughput
    bf16_tflops: float          # Peak BF16 throughput (0 if unsupported)
    memory_gb: float
    memory_bw_gbps: float       # Peak memory bandwidth (GB/s)
    has_transformer_engine: bool

    @property
    def idle_power_w(self) -> float:
        """Estimated idle power (~30% of TDP based on empirical data)."""
        return self.tdp_w * 0.30

    def peak_tflops_for_precision(self, precision: str) -> float:
        """Return peak TFLOPS for a given precision."""
        mapping = {
            'fp16': self.fp16_tflops,
            'bf16': self.bf16_tflops if self.bf16_tflops > 0 else self.fp16_tflops,
            'fp32': self.fp32_tflops,
            'fp8': self.fp16_tflops * 2.0,  # FP8 ~2x FP16 on Hopper
            'int8': self.fp16_tflops * 2.0,
        }
        return mapping.get(precision, self.fp16_tflops)

    def estimated_power_at_utilization(self, util_frac: float) -> float:
        """
        Power draw at a given utilization fraction (0.0 - 1.0).

        Uses linear interpolation between idle power and TDP.
        Validated against NVIDIA datacenter power curves:
          P(u) = P_idle + (P_tdp - P_idle) * u
        """
        return self.idle_power_w + (self.tdp_w - self.idle_power_w) * util_frac

    def roofline_tflops(
        self,
        math_intensity: float,
        util_frac: float,
        precision: str = 'fp16',
    ) -> float:
        """
        Roofline model: attainable TFLOPS given math intensity and utilization.

        Args:
            math_intensity: FLOP/byte of the workload
            util_frac: GPU utilization fraction (0.0 - 1.0)
            precision: Compute precision ('fp16', 'bf16', 'fp32')

        Returns:
            Effective TFLOPS this GPU can deliver for this workload.
        """
        peak = self.peak_tflops_for_precision(precision) * util_frac

        # Bandwidth ceiling: memory_bw (GB/s) * math_intensity (FLOP/byte) / 1e3 → TFLOPS
        bandwidth_ceiling = (self.memory_bw_gbps * math_intensity) / 1000.0

        return min(peak, bandwidth_ceiling)

    def joules_per_tflop(
        self,
        math_intensity: float,
        util_frac: float,
        precision: str = 'fp16',
    ) -> float:
        """
        Core efficiency metric: energy cost per unit of useful compute.

        Lower is better. Returns float('inf') if no useful work is done.
        """
        effective_tflops = self.roofline_tflops(math_intensity, util_frac, precision)
        if effective_tflops <= 0:
            return float('inf')

        power_w = self.estimated_power_at_utilization(util_frac)
        return power_w / effective_tflops


@dataclass(frozen=True)
class ModelProfile:
    """ML model workload characteristics for hardware matching."""

    tag: str
    family: str
    math_intensity: float       # FLOP/byte — the key roofline parameter
    precision: str              # Dominant compute precision
    is_memory_bound: bool       # True if math_intensity < ~50 FLOP/byte
    typical_util_min: int       # Expected utilization range lower bound
    typical_util_max: int       # Expected utilization range upper bound

    @property
    def typical_util_mid(self) -> float:
        """Midpoint of expected utilization range as fraction."""
        return (self.typical_util_min + self.typical_util_max) / 200.0


# ═══════════════════════════════════════════════════════════════════════
# GPU ARCHITECTURES — mirrors migration 007 seed data
# ═══════════════════════════════════════════════════════════════════════

GPU_ARCHITECTURES: dict[str, ArchSpec] = {
    spec.name: spec for spec in [
        # Ampere
        ArchSpec('A100-SXM4-80GB',  'Ampere',        400,  312,  19.5,  312,  80, 2039, False),
        ArchSpec('A100-SXM4-40GB',  'Ampere',        400,  312,  19.5,  312,  40, 1555, False),
        ArchSpec('A100-PCIe-80GB',  'Ampere',        300,  312,  19.5,  312,  80, 2039, False),
        ArchSpec('A100-PCIe-40GB',  'Ampere',        250,  312,  19.5,  312,  40, 1555, False),
        # Hopper
        ArchSpec('H100-SXM5-80GB',  'Hopper',        700,  989,  67.0,  989,  80, 3350, True),
        ArchSpec('H100-PCIe-80GB',  'Hopper',        350,  756,  51.0,  756,  80, 2039, True),
        ArchSpec('H200-SXM-141GB',  'Hopper',        700,  989,  67.0,  989, 141, 4800, True),
        # Ada Lovelace
        ArchSpec('L40S',            'Ada Lovelace',  350,  362,  91.6,  362,  48,  864, False),
        ArchSpec('L40',             'Ada Lovelace',  300,  181,  90.5,  181,  48,  864, False),
        # Lower-tier
        ArchSpec('A10G',            'Ampere',        150,   70,  31.2,   70,  24,  600, False),
        ArchSpec('T4',              'Turing',         70,   65,   8.1,    0,  16,  300, False),
        # Volta
        ArchSpec('V100-SXM2-32GB',  'Volta',        300,  125,  15.7,    0,  32,  900, False),
        ArchSpec('V100-SXM2-16GB',  'Volta',        300,  125,  15.7,    0,  16,  900, False),
    ]
}


# ═══════════════════════════════════════════════════════════════════════
# MODEL PROFILES — mirrors migration 007 seed data
# ═══════════════════════════════════════════════════════════════════════

MODEL_PROFILES: dict[str, ModelProfile] = {
    p.tag: p for p in [
        ModelProfile('bert-base',       'BERT',        10,  'fp16',  True,  35, 65),
        ModelProfile('bert-large',      'BERT',        12,  'fp16',  True,  40, 70),
        ModelProfile('llama-3-8b',      'Llama',       95,  'bf16',  False, 60, 85),
        ModelProfile('llama-3-70b',     'Llama',      185,  'bf16',  False, 75, 95),
        ModelProfile('llama-3-405b',    'Llama',      220,  'bf16',  False, 80, 95),
        ModelProfile('mistral-7b',      'Mistral',     90,  'bf16',  False, 55, 80),
        ModelProfile('mixtral-8x7b',    'Mistral',    110,  'bf16',  False, 60, 85),
        ModelProfile('gpt-neox-20b',    'GPT-NeoX',   130,  'fp16',  False, 65, 90),
        ModelProfile('sdxl',            'Diffusion',   48,  'fp16',  True,  50, 80),
        ModelProfile('sd-3',            'Diffusion',   55,  'fp16',  False, 50, 80),
        ModelProfile('whisper-large',   'Whisper',     28,  'fp16',  True,  35, 65),
        ModelProfile('whisper-medium',  'Whisper',     22,  'fp16',  True,  30, 60),
        ModelProfile('vit-large',       'ViT',         35,  'fp16',  True,  40, 70),
        ModelProfile('t5-xxl',          'T5',         100,  'bf16',  False, 55, 85),
        ModelProfile('falcon-40b',      'Falcon',     140,  'bf16',  False, 65, 90),
        ModelProfile('deepseek-v3',     'DeepSeek',   200,  'bf16',  False, 75, 95),
    ]
}


def resolve_arch(gpu_name: str) -> ArchSpec | None:
    """
    Match a gpu_name string (from NVML) to a known architecture.

    NVML returns names like "NVIDIA A100-SXM4-80GB" or "Tesla T4".
    We strip the vendor prefix and try exact match, then substring match.
    """
    # Exact match
    if gpu_name in GPU_ARCHITECTURES:
        return GPU_ARCHITECTURES[gpu_name]

    # Strip common prefixes
    cleaned = gpu_name.replace('NVIDIA ', '').replace('Tesla ', '')
    if cleaned in GPU_ARCHITECTURES:
        return GPU_ARCHITECTURES[cleaned]

    # Substring match (e.g., "A100" matches "A100-SXM4-80GB")
    for arch_name, spec in GPU_ARCHITECTURES.items():
        if arch_name in cleaned or cleaned in arch_name:
            return spec

    # Family match (e.g., "A100" anywhere in the string)
    for arch_name, spec in GPU_ARCHITECTURES.items():
        # Extract the base model (e.g., "A100" from "A100-SXM4-80GB")
        base = arch_name.split('-')[0]
        if base in gpu_name:
            return spec

    return None
