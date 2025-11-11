"""
Example usage of the BERT Benchmarking Suite.
This script demonstrates different ways to use the framework.
"""

from src import BERTBenchmarker, ResultsManager


def example_1_fast_test():
    """Example 1: Fast test of a single model."""
    print("="*60)
    print("EXAMPLE 1: Fast Test of Single Model")
    print("="*60)

    benchmarker = BERTBenchmarker()

    # Override config for fast test
    benchmarker.benchmark_config['benchmark']['warmup_iterations'] = 5
    benchmarker.benchmark_config['benchmark']['test_iterations'] = 20
    benchmarker.benchmark_config['benchmark']['batch_sizes'] = [1, 8]
    benchmarker.benchmark_config['benchmark']['sequence_lengths'] = [64]

    # Test just one model
    results = benchmarker.run_benchmark(
        models_subset=["distilbert-base-uncased"],
        devices_subset=["cpu"]
    )

    # Print results
    for result in results:
        print(f"\nModel: {result.model_name}")
        print(f"Device: {result.device}")
        print(f"Batch size: {result.batch_size}")
        print(f"Latency: {result.mean_latency_ms:.2f}ms")
        print(f"Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")
        print(f"Memory: {result.peak_memory_mb:.2f}MB")


def example_2_compare_models():
    """Example 2: Compare multiple models on CPU."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Compare Multiple Models")
    print("="*60)

    benchmarker = BERTBenchmarker()

    # Fast config
    benchmarker.benchmark_config['benchmark']['test_iterations'] = 20
    benchmarker.benchmark_config['benchmark']['batch_sizes'] = [1]
    benchmarker.benchmark_config['benchmark']['sequence_lengths'] = [128]

    # Compare distilled vs base models
    results = benchmarker.run_benchmark(
        models_subset=[
            "distilbert-base-uncased",
            "bert-base-uncased",
            "prajjwal1/bert-small"
        ],
        devices_subset=["cpu"]
    )

    # Analyze
    results_manager = ResultsManager()

    print("\nFastest models:")
    best = results_manager.get_best_models(results, criterion="latency", top_k=3)
    print(best)

    print("\nSmallest models:")
    smallest = results_manager.get_best_models(results, criterion="size", top_k=3)
    print(smallest)


def example_3_save_and_visualize():
    """Example 3: Run benchmark and save results with visualizations."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Full Benchmark with Visualization")
    print("="*60)

    benchmarker = BERTBenchmarker()

    # Fast config
    benchmarker.benchmark_config['benchmark']['test_iterations'] = 30
    benchmarker.benchmark_config['benchmark']['batch_sizes'] = [1, 4, 8]
    benchmarker.benchmark_config['benchmark']['sequence_lengths'] = [64, 128]

    # Run on subset
    results = benchmarker.run_benchmark(
        models_subset=[
            "distilbert-base-uncased",
            "prajjwal1/bert-mini"
        ],
        devices_subset=["cpu"]
    )

    # Save results
    results_manager = ResultsManager()

    print("\nSaving results...")
    saved_files = results_manager.save_results(results, formats=["csv", "json"])
    print(f"Saved: {saved_files}")

    print("\nGenerating report...")
    report = results_manager.create_comparison_report(results)
    print(f"Report: {report}")

    print("\nGenerating comparison plots...")
    results_manager.create_comparison_plots(results)

    print("\nGenerating top performers plots...")
    results_manager.create_top_performers_plots(results, top_k=5)

    print("\nDone! Check the results/ directory")


def example_4_custom_metrics():
    """Example 4: Extract specific metrics from results."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Metrics Analysis")
    print("="*60)

    benchmarker = BERTBenchmarker()

    # Minimal config
    benchmarker.benchmark_config['benchmark']['test_iterations'] = 10
    benchmarker.benchmark_config['benchmark']['batch_sizes'] = [1]
    benchmarker.benchmark_config['benchmark']['sequence_lengths'] = [128]

    results = benchmarker.run_benchmark(
        models_subset=["distilbert-base-uncased"],
        devices_subset=["cpu"]
    )

    # Extract specific metrics
    result = results[0]

    print(f"\nDetailed metrics for {result.model_name}:")
    print(f"  Parameters: {result.num_parameters:,}")
    print(f"  Size: {result.model_size_mb:.2f} MB")
    print(f"  Mean latency: {result.mean_latency_ms:.2f}ms")
    print(f"  Median latency: {result.median_latency_ms:.2f}ms")
    print(f"  P95 latency: {result.p95_latency_ms:.2f}ms")
    print(f"  P99 latency: {result.p99_latency_ms:.2f}ms")
    print(f"  Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")
    print(f"  Peak memory: {result.peak_memory_mb:.2f} MB")

    # Calculate efficiency score (lower is better)
    efficiency_score = (result.mean_latency_ms * result.peak_memory_mb) / 1000
    print(f"  Efficiency score: {efficiency_score:.2f} (latency * memory)")


if __name__ == "__main__":
    import sys

    examples = {
        "1": example_1_fast_test,
        "2": example_2_compare_models,
        "3": example_3_save_and_visualize,
        "4": example_4_custom_metrics,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Usage: python example_usage.py [1|2|3|4]")
        print("\nExamples:")
        print("  1: Fast test of single model")
        print("  2: Compare multiple models")
        print("  3: Full benchmark with visualization")
        print("  4: Custom metrics analysis")
        print("\nRunning all examples...")

        for name, func in examples.items():
            try:
                func()
            except Exception as e:
                print(f"\nError in example {name}: {e}")
                continue
