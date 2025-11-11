#!/usr/bin/env python
"""
Main script to run BERT model benchmarks.

Usage:
    # Run full benchmark on all models
    python run_benchmark.py

    # Run on specific models only
    python run_benchmark.py --models distilbert-base-uncased bert-base-uncased

    # Run on CPU only
    python run_benchmark.py --devices cpu

    # Fast test (fewer iterations)
    python run_benchmark.py --fast
"""

import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from src import BERTBenchmarker, ResultsManager

# Load environment variables from .env file
# This loads .env but does NOT override existing system environment variables
load_dotenv()


def setup_cache_directories(force_override=False):
    """
    Set up cache directories with proper precedence.

    Args:
        force_override: If True, override system environment variables with .env settings
    """
    # Get base cache directory from .env or default
    cache_dir = os.getenv('BERT_CACHE_DIR', './cache')
    cache_path = Path(cache_dir)

    cache_vars = {
        'HF_HOME': cache_path / 'huggingface',
        'HF_HUB_CACHE': cache_path / 'huggingface' / 'hub',
        'TRANSFORMERS_CACHE': cache_path / 'transformers',
        'HF_DATASETS_CACHE': cache_path / 'datasets',
        'TORCH_HOME': cache_path / 'torch'
    }

    for var_name, var_path in cache_vars.items():
        if force_override:
            # Force override mode: use .env settings regardless of system env
            os.environ[var_name] = str(var_path)
        else:
            # Respect system environment variables, only set if not already set
            if var_name not in os.environ:
                os.environ[var_name] = str(var_path)

    # Print cache configuration (helps with debugging)
    print(f"Cache Configuration:")
    print(f"  HF_HOME: {os.environ['HF_HOME']}")
    print(f"  HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")
    if force_override:
        print(f"  (Forced override mode - ignoring system environment variables)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark BERT models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of model names to benchmark (default: all models in config)"
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Devices to test on: cpu, cuda (default: from config)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast benchmark with fewer iterations"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--force-cache-dir",
        action="store_true",
        help="Force use of BERT_CACHE_DIR from .env, overriding system environment variables"
    )

    args = parser.parse_args()

    # Set up cache directories based on .env and system environment variables
    setup_cache_directories(force_override=args.force_cache_dir)

    # Initialize benchmarker
    print("Initializing BERT Benchmarker...")
    benchmarker = BERTBenchmarker()

    # Override config for fast benchmark
    if args.fast:
        print("Running in FAST mode (reduced iterations)")
        benchmarker.benchmark_config['benchmark']['warmup_iterations'] = 5
        benchmarker.benchmark_config['benchmark']['test_iterations'] = 20
        benchmarker.benchmark_config['benchmark']['batch_sizes'] = [1, 8]
        benchmarker.benchmark_config['benchmark']['sequence_lengths'] = [64, 128]

    # Run benchmark
    print("\nStarting benchmark...")
    print(f"Models: {args.models if args.models else 'ALL'}")
    print(f"Devices: {args.devices if args.devices else 'From config'}")

    results = benchmarker.run_benchmark(
        models_subset=args.models,
        devices_subset=args.devices
    )

    if not results:
        print("\nNo results generated!")
        return

    # Save results
    print(f"\n\nBenchmark complete! Generated {len(results)} result entries.")
    print("\nSaving results...")

    # Determine benchmark type for folder naming
    benchmark_type = "fast" if args.fast else "full"
    results_manager = ResultsManager(
        output_dir=args.output_dir,
        benchmark_type=benchmark_type
    )

    # Save raw data
    saved_files = results_manager.save_results(
        results,
        formats=["csv", "json"]
    )

    # Generate report (always run)
    print("\nGenerating comparison report...")
    try:
        report_path = results_manager.create_comparison_report(results)
    except Exception as e:
        print(f"Warning: Report generation failed: {e}")
        report_path = None

    # Generate plots (always run unless explicitly disabled)
    if not args.no_plots:
        print("\nGenerating comparison plots...")
        try:
            results_manager.create_comparison_plots(results)
        except Exception as e:
            print(f"Warning: Plot generation failed: {e}")

        print("\nGenerating top performers plots...")
        try:
            results_manager.create_top_performers_plots(results, top_k=5)
        except Exception as e:
            print(f"Warning: Top performers plot generation failed: {e}")

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_manager.output_dir}/")
    print(f"  Run timestamp: {results_manager.run_timestamp}")
    print(f"\nFiles generated:")
    print(f"  - CSV: benchmark_results.csv")
    print(f"  - JSON: benchmark_results.json")
    if report_path:
        print(f"  - Report: benchmark_report.md")
    if not args.no_plots:
        print(f"  - Comparison plots: plots_latency.png, plots_throughput.png, plots_memory.png, plots_size_vs_latency.png")
        print(f"  - Top performers: top_performers_fastest.png, top_performers_memory.png, top_performers_smallest.png, top_performers_summary.png")

    # Show top performers
    print("\n" + "="*60)
    print("TOP PERFORMERS")
    print("="*60)

    print("\nFastest inference (by latency):")
    top_fast = results_manager.get_best_models(results, criterion="latency", top_k=3)
    print(top_fast.to_string())

    print("\nMost memory efficient:")
    top_memory = results_manager.get_best_models(results, criterion="memory", top_k=3)
    print(top_memory.to_string())

    print("\nSmallest models:")
    top_small = results_manager.get_best_models(results, criterion="size", top_k=3)
    print(top_small.to_string())

    print("\n" + "="*60)
    print(f"Full results available in: {results_manager.output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
