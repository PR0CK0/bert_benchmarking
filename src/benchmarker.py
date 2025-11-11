"""
Core benchmarking system for BERT models.
Orchestrates testing across multiple models and configurations.
"""

import torch
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel
)
from tqdm import tqdm
import warnings

from .metrics import MetricsCollector, BenchmarkMetrics

warnings.filterwarnings('ignore')


class BERTBenchmarker:
    """Main benchmarking orchestrator for BERT models."""

    def __init__(
        self,
        models_config_path: str = "configs/models.yaml",
        benchmark_config_path: str = "configs/benchmark_config.yaml"
    ):
        """
        Initialize benchmarker.

        Args:
            models_config_path: Path to models configuration
            benchmark_config_path: Path to benchmark configuration
        """
        self.models_config = self._load_config(models_config_path)
        self.benchmark_config = self._load_config(benchmark_config_path)

        self.results: List[BenchmarkMetrics] = []

    def _load_config(self, path: str) -> Dict:
        """Load YAML configuration file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _prepare_dummy_inputs(
        self,
        tokenizer,
        batch_size: int,
        sequence_length: int,
        device: str
    ) -> Dict[str, torch.Tensor]:
        """
        Create dummy inputs for benchmarking.

        Args:
            tokenizer: Tokenizer instance
            batch_size: Batch size
            sequence_length: Sequence length
            device: Device to place tensors on

        Returns:
            Dictionary of input tensors
        """
        # Create dummy text
        dummy_text = ["cybersecurity threat detection analysis"] * batch_size

        # Tokenize
        inputs = tokenizer(
            dummy_text,
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        return inputs

    def _load_model(
        self,
        model_name: str,
        model_type: str,
        num_classes: int,
        device: str
    ):
        """
        Load model and tokenizer.

        Args:
            model_name: Model name/path
            model_type: Type of model
            num_classes: Number of classification classes
            device: Device to load model on

        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading {model_name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if model_type == "sequence-classification":
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True  # Handle different num_labels
                )
            else:
                # For sentence-transformers or other types
                model = AutoModel.from_pretrained(model_name)

            model = model.to(device)
            model.eval()

            return model, tokenizer

        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            return None, None

    def benchmark_single_model(
        self,
        model_name: str,
        model_type: str,
        device: str,
        batch_sizes: List[int],
        sequence_lengths: List[int],
        num_classes: int,
        warmup_iterations: int = 10,
        test_iterations: int = 100
    ) -> List[BenchmarkMetrics]:
        """
        Benchmark a single model across different configurations.

        Args:
            model_name: Model name/path
            model_type: Type of model
            device: Device to test on
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            num_classes: Number of classification classes
            warmup_iterations: Number of warmup iterations
            test_iterations: Number of test iterations

        Returns:
            List of BenchmarkMetrics for each configuration
        """
        results = []

        # Load model
        model, tokenizer = self._load_model(
            model_name, model_type, num_classes, device
        )

        if model is None or tokenizer is None:
            return results

        # Initialize metrics collector
        collector = MetricsCollector(device=device)

        # Test across configurations
        configs = [
            (bs, seq_len)
            for bs in batch_sizes
            for seq_len in sequence_lengths
        ]

        for batch_size, seq_length in tqdm(configs, desc=f"Testing configs"):
            try:
                # Prepare inputs
                inputs = self._prepare_dummy_inputs(
                    tokenizer, batch_size, seq_length, device
                )

                # Benchmark
                metrics = collector.benchmark_model(
                    model=model,
                    model_name=model_name,
                    inputs=inputs,
                    batch_size=batch_size,
                    sequence_length=seq_length,
                    num_iterations=test_iterations,
                    warmup_iterations=warmup_iterations
                )

                results.append(metrics)

            except Exception as e:
                print(f"Error testing {model_name} with bs={batch_size}, "
                      f"seq_len={seq_length}: {str(e)}")
                continue

        # Cleanup
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

        return results

    def run_benchmark(
        self,
        models_subset: Optional[List[str]] = None,
        devices_subset: Optional[List[str]] = None
    ) -> List[BenchmarkMetrics]:
        """
        Run benchmark across all configured models and settings.

        Args:
            models_subset: Optional list of model names to test (tests all if None)
            devices_subset: Optional list of devices to test (uses config if None)

        Returns:
            List of all benchmark results
        """
        # Get configuration
        benchmark_cfg = self.benchmark_config['benchmark']
        batch_sizes = benchmark_cfg['batch_sizes']
        sequence_lengths = benchmark_cfg['sequence_lengths']
        devices = devices_subset or benchmark_cfg['devices']
        num_classes = benchmark_cfg['num_classes']
        warmup = benchmark_cfg['warmup_iterations']
        test_iters = benchmark_cfg['test_iterations']

        # Filter devices based on availability
        if 'cuda' in devices and not torch.cuda.is_available():
            print("CUDA not available, skipping GPU tests")
            devices = [d for d in devices if d != 'cuda']

        # Get models to test
        models = self.models_config['models']
        if models_subset:
            models = [m for m in models if m['name'] in models_subset]

        # Run benchmarks
        all_results = []

        for model_info in models:
            model_name = model_info['name']
            model_type = model_info['type']

            print(f"\n{'='*60}")
            print(f"Benchmarking: {model_name}")
            print(f"Description: {model_info['description']}")
            print(f"{'='*60}")

            for device in devices:
                print(f"\nTesting on {device.upper()}...")

                results = self.benchmark_single_model(
                    model_name=model_name,
                    model_type=model_type,
                    device=device,
                    batch_sizes=batch_sizes,
                    sequence_lengths=sequence_lengths,
                    num_classes=num_classes,
                    warmup_iterations=warmup,
                    test_iterations=test_iters
                )

                all_results.extend(results)

        self.results = all_results
        return all_results

    def get_results(self) -> List[BenchmarkMetrics]:
        """Get all benchmark results."""
        return self.results
