"""
Metrics collection for BERT model benchmarking.
Tracks inference speed, memory usage, and model characteristics.
"""

import time
import psutil
import torch
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class BenchmarkMetrics:
    """Container for all benchmark metrics."""

    # Model info
    model_name: str
    num_parameters: int
    model_size_mb: float

    # Test configuration
    device: str
    batch_size: int
    sequence_length: int

    # Inference speed
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_samples_per_sec: float

    # Memory usage
    peak_memory_mb: float
    memory_allocated_mb: Optional[float] = None  # GPU only

    # System utilization (optional)
    cpu_percent: Optional[float] = None
    gpu_percent: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class MetricsCollector:
    """Collects performance metrics for model benchmarking."""

    def __init__(self, device: str = "cpu"):
        """
        Initialize metrics collector.

        Args:
            device: Device to monitor ('cpu' or 'cuda')
        """
        self.device = device
        self.gpu_available = torch.cuda.is_available() and device == "cuda"

        # Initialize NVML for GPU monitoring
        if self.gpu_available and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_initialized = True
            except:
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False

    def get_model_size(self, model) -> tuple[int, float]:
        """
        Get model parameter count and size in MB.

        Args:
            model: PyTorch model

        Returns:
            Tuple of (num_parameters, size_in_mb)
        """
        num_params = sum(p.numel() for p in model.parameters())

        # Calculate size (assuming float32 = 4 bytes)
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)

        return num_params, size_mb

    def measure_latency(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> List[float]:
        """
        Measure inference latency over multiple iterations.

        Args:
            model: Model to benchmark
            inputs: Input tensors
            num_iterations: Number of measurement iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            List of latency measurements in seconds
        """
        model.eval()
        latencies = []

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(**inputs)
                if self.gpu_available:
                    torch.cuda.synchronize()

        # Measurement
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(**inputs)

                if self.gpu_available:
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append(end - start)

        return latencies

    def get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        if self.gpu_available:
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 ** 2)

    def get_gpu_memory_allocated(self) -> Optional[float]:
        """Get GPU memory allocated in MB."""
        if self.gpu_available:
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return None

    def get_cpu_percent(self) -> float:
        """Get current CPU utilization percentage."""
        return psutil.cpu_percent(interval=0.1)

    def get_gpu_percent(self) -> Optional[float]:
        """Get current GPU utilization percentage."""
        if self.nvml_initialized:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                return float(util.gpu)
            except:
                return None
        return None

    def reset_memory_stats(self):
        """Reset memory statistics."""
        if self.gpu_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def benchmark_model(
        self,
        model,
        model_name: str,
        inputs: Dict[str, torch.Tensor],
        batch_size: int,
        sequence_length: int,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> BenchmarkMetrics:
        """
        Run complete benchmark on a model.

        Args:
            model: Model to benchmark
            model_name: Name of the model
            inputs: Input tensors
            batch_size: Batch size
            sequence_length: Sequence length
            num_iterations: Number of measurement iterations
            warmup_iterations: Number of warmup iterations

        Returns:
            BenchmarkMetrics object
        """
        # Get model characteristics
        num_params, size_mb = self.get_model_size(model)

        # Reset memory tracking
        self.reset_memory_stats()

        # Measure latency
        latencies = self.measure_latency(
            model, inputs, num_iterations, warmup_iterations
        )

        # Convert to milliseconds
        latencies_ms = [l * 1000 for l in latencies]

        # Calculate statistics
        import numpy as np
        mean_latency = np.mean(latencies_ms)
        median_latency = np.median(latencies_ms)
        p95_latency = np.percentile(latencies_ms, 95)
        p99_latency = np.percentile(latencies_ms, 99)

        # Calculate throughput
        throughput = batch_size / (mean_latency / 1000)  # samples per second

        # Get memory usage
        peak_memory = self.get_memory_usage()
        memory_allocated = self.get_gpu_memory_allocated()

        # Get utilization (optional, can be noisy)
        cpu_percent = self.get_cpu_percent()
        gpu_percent = self.get_gpu_percent()

        return BenchmarkMetrics(
            model_name=model_name,
            num_parameters=num_params,
            model_size_mb=size_mb,
            device=self.device,
            batch_size=batch_size,
            sequence_length=sequence_length,
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_samples_per_sec=throughput,
            peak_memory_mb=peak_memory,
            memory_allocated_mb=memory_allocated,
            cpu_percent=cpu_percent,
            gpu_percent=gpu_percent
        )

    def __del__(self):
        """Cleanup NVML."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
