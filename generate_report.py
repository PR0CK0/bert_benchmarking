#!/usr/bin/env python
"""
Generate report and plots from existing benchmark results.
Useful if the benchmark completed but report generation failed.

Usage:
    python generate_report.py results/20251111_101910_fast/benchmark_results.json
    python generate_report.py results/20251111_143022_full/benchmark_results.csv
"""

import sys
import json
from pathlib import Path
from src import ResultsManager
from src.metrics import BenchmarkMetrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py <results_file>")
        print("\nAvailable results files:")
        results_dir = Path("results")
        if results_dir.exists():
            # Search recursively in timestamped subdirectories
            for file in sorted(results_dir.glob("*/benchmark_results.json")):
                print(f"  {file}")
        else:
            print("  No results directory found!")
        sys.exit(1)

    results_file = Path(sys.argv[1])

    if not results_file.exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

    print(f"Loading results from: {results_file}")

    # Load results
    if results_file.suffix == ".json":
        with open(results_file, 'r') as f:
            data = json.load(f)
        results = [BenchmarkMetrics(**r) for r in data]
    elif results_file.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(results_file)
        results = [BenchmarkMetrics(**row.to_dict()) for _, row in df.iterrows()]
    else:
        print(f"Error: Unsupported file format: {results_file.suffix}")
        sys.exit(1)

    print(f"Loaded {len(results)} result entries")

    # Initialize results manager
    results_manager = ResultsManager()

    # Generate report
    print("\nGenerating comparison report...")
    try:
        report_path = results_manager.create_comparison_report(results)
        print(f"✓ Report saved: {report_path}")
    except Exception as e:
        print(f"✗ Report generation failed: {e}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    try:
        results_manager.create_comparison_plots(results)
        print("✓ Comparison plots generated successfully")
    except Exception as e:
        print(f"✗ Comparison plot generation failed: {e}")

    # Generate top performers plots
    print("\nGenerating top performers plots...")
    try:
        results_manager.create_top_performers_plots(results, top_k=5)
        print("✓ Top performers plots generated successfully")
    except Exception as e:
        print(f"✗ Top performers plot generation failed: {e}")

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
    print("Check the results/ directory for all outputs!")
    print("="*60)


if __name__ == "__main__":
    main()
