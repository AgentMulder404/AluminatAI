"""
GPU Metrics Collector using NVIDIA Management Library (NVML)

This module provides low-overhead GPU monitoring with energy calculation.
"""

import time
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py3")


@dataclass
class GPUMetrics:
    """Single GPU metrics snapshot"""
    timestamp: str
    gpu_index: int
    gpu_uuid: str
    gpu_name: str

    # Power metrics
    power_draw_w: float
    power_limit_w: float
    energy_delta_j: Optional[float] = None

    # Utilization
    utilization_gpu_pct: int = 0
    utilization_memory_pct: int = 0

    # Thermal
    temperature_c: int = 0
    fan_speed_pct: int = 0

    # Clocks (optional, can add overhead)
    sm_clock_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None

    # Memory
    memory_used_mb: float = 0
    memory_total_mb: float = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def to_csv_row(self) -> List:
        """Convert to CSV row (without optional fields)"""
        return [
            self.timestamp,
            self.gpu_index,
            self.gpu_uuid,
            self.power_draw_w,
            self.energy_delta_j or 0,
            self.utilization_gpu_pct,
            self.utilization_memory_pct,
            self.temperature_c,
            self.memory_used_mb,
        ]


class GPUCollector:
    """
    Collects metrics from all NVIDIA GPUs using NVML.

    Features:
    - Low overhead (<0.5ms per GPU)
    - Energy delta calculation (E = P × Δt)
    - Configurable metric collection
    - Graceful error handling
    """

    def __init__(self, collect_clocks: bool = False):
        """
        Initialize GPU collector.

        Args:
            collect_clocks: If True, collect clock speeds (adds ~0.1ms overhead)
        """
        if not NVML_AVAILABLE:
            raise RuntimeError("NVML not available. Install nvidia-ml-py3")

        self.collect_clocks = collect_clocks
        self.initialized = False
        self.gpu_count = 0
        self.gpu_handles = []
        self.gpu_info = []

        # Track last sample for energy calculation
        self.last_sample_time = {}
        self.last_power_draw = {}

        self._initialize()

    def _initialize(self):
        """Initialize NVML and discover GPUs"""
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()

            # Get handles and basic info for each GPU
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.gpu_handles.append(handle)

                # Get static info (only needs to be fetched once)
                gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
                gpu_name = pynvml.nvmlDeviceGetName(handle)

                # Decode if bytes (Python 3)
                if isinstance(gpu_uuid, bytes):
                    gpu_uuid = gpu_uuid.decode('utf-8')
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')

                self.gpu_info.append({
                    'index': i,
                    'uuid': gpu_uuid,
                    'name': gpu_name
                })

            self.initialized = True

        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize NVML: {e}")

    def collect(self) -> List[GPUMetrics]:
        """
        Collect current metrics from all GPUs.

        Returns:
            List of GPUMetrics, one per GPU
        """
        if not self.initialized:
            raise RuntimeError("Collector not initialized")

        metrics = []
        timestamp = datetime.now(timezone.utc).isoformat()
        current_time = time.time()

        for i, handle in enumerate(self.gpu_handles):
            try:
                gpu_metrics = self._collect_single_gpu(
                    handle,
                    i,
                    timestamp,
                    current_time
                )
                metrics.append(gpu_metrics)
            except pynvml.NVMLError as e:
                print(f"Warning: Failed to collect metrics for GPU {i}: {e}")
                continue

        return metrics

    def _collect_single_gpu(
        self,
        handle,
        gpu_index: int,
        timestamp: str,
        current_time: float
    ) -> GPUMetrics:
        """Collect metrics from a single GPU"""

        # Power metrics
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

        # Calculate energy delta: E = P × Δt
        energy_delta = None
        if gpu_index in self.last_sample_time:
            time_delta = current_time - self.last_sample_time[gpu_index]
            # Use average power between samples for better accuracy
            avg_power = (power_draw + self.last_power_draw[gpu_index]) / 2.0
            energy_delta = avg_power * time_delta  # Joules

        self.last_sample_time[gpu_index] = current_time
        self.last_power_draw[gpu_index] = power_draw

        # Utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            util_gpu = util.gpu
            util_mem = util.memory
        except pynvml.NVMLError:
            util_gpu = 0
            util_mem = 0

        # Temperature
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle,
                pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            temperature = 0

        # Fan speed
        try:
            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
        except pynvml.NVMLError:
            fan_speed = 0

        # Memory
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = mem_info.used / 1024 / 1024  # bytes -> MB
            mem_total = mem_info.total / 1024 / 1024
        except pynvml.NVMLError:
            mem_used = 0
            mem_total = 0

        # Clocks (optional - adds overhead)
        sm_clock = None
        mem_clock = None
        if self.collect_clocks:
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle,
                    pynvml.NVML_CLOCK_SM
                )
                mem_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle,
                    pynvml.NVML_CLOCK_MEM
                )
            except pynvml.NVMLError:
                pass

        return GPUMetrics(
            timestamp=timestamp,
            gpu_index=gpu_index,
            gpu_uuid=self.gpu_info[gpu_index]['uuid'],
            gpu_name=self.gpu_info[gpu_index]['name'],
            power_draw_w=power_draw,
            power_limit_w=power_limit,
            energy_delta_j=energy_delta,
            utilization_gpu_pct=util_gpu,
            utilization_memory_pct=util_mem,
            temperature_c=temperature,
            fan_speed_pct=fan_speed,
            sm_clock_mhz=sm_clock,
            memory_clock_mhz=mem_clock,
            memory_used_mb=mem_used,
            memory_total_mb=mem_total,
        )

    def get_gpu_count(self) -> int:
        """Return number of GPUs detected"""
        return self.gpu_count

    def get_gpu_info(self) -> List[Dict]:
        """Return static GPU information"""
        return self.gpu_info

    def shutdown(self):
        """Cleanup NVML resources"""
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
                self.initialized = False
            except pynvml.NVMLError:
                pass

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        self.shutdown()


# CSV header for export
CSV_HEADER = [
    'timestamp',
    'gpu_index',
    'gpu_uuid',
    'power_w',
    'energy_j',
    'util_gpu_pct',
    'util_mem_pct',
    'temp_c',
    'memory_used_mb',
]


if __name__ == '__main__':
    # Simple test
    print("Testing GPU Collector...")

    try:
        with GPUCollector() as collector:
            print(f"Found {collector.get_gpu_count()} GPUs:")
            for info in collector.get_gpu_info():
                print(f"  GPU {info['index']}: {info['name']} ({info['uuid']})")

            print("\nCollecting 3 samples (2s intervals)...")
            for i in range(3):
                metrics = collector.collect()
                for m in metrics:
                    print(f"  GPU {m.gpu_index}: {m.power_draw_w:.1f}W, "
                          f"{m.utilization_gpu_pct}% util, "
                          f"{m.temperature_c}°C, "
                          f"{m.energy_delta_j:.1f}J" if m.energy_delta_j else "N/A")

                if i < 2:  # Don't sleep after last sample
                    time.sleep(2)

            print("\n✅ Collector test passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
