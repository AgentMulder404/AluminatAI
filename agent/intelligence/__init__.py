# Copyright 2026 Kevin (NemulAI)
# SPDX-License-Identifier: Apache-2.0
#
# NemulAI — https://github.com/AgentMulder404/NemulAI

"""Model Intelligence Pipeline — auto-discover, profile, and rank GPU pairings for new AI models."""

from intelligence.detector import ModelDetector, DetectedModel
from intelligence.profiler import ModelProfiler, ProfileResult
from intelligence.estimator import BenchmarkEstimator, EstimationResult, GPUEstimate
from intelligence.quantization import QuantizationAdvisor, QuantizationVariant, QuantizedModelProfile, QuantizationRecommendation
from intelligence.pricing import GPUPricingTracker, PricePerformanceMetrics, PriceAlert
from intelligence.pipeline import IntelligencePipeline, PipelineResult
from intelligence.registry import ModelRegistry, RegistryEntry

__all__ = [
    "ModelDetector", "DetectedModel",
    "ModelProfiler", "ProfileResult",
    "BenchmarkEstimator", "EstimationResult", "GPUEstimate",
    "QuantizationAdvisor", "QuantizationVariant", "QuantizedModelProfile", "QuantizationRecommendation",
    "GPUPricingTracker", "PricePerformanceMetrics", "PriceAlert",
    "IntelligencePipeline", "PipelineResult",
    "ModelRegistry", "RegistryEntry",
]
