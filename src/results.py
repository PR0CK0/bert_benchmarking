"""
Results storage, analysis, and visualization for benchmark data.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import BenchmarkMetrics


class ResultsManager:
    """Manages storage and analysis of benchmark results."""

    def __init__(
        self,
        output_dir: str = "results",
        run_timestamp: Optional[str] = None,
        benchmark_type: str = "full"
    ):
        """
        Initialize results manager.

        Args:
            output_dir: Base directory to store results
            run_timestamp: Optional timestamp for this run (will create subdirectory)
            benchmark_type: Type of benchmark ('fast' or 'full')
        """
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)

        # Create timestamped subdirectory for this run
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Append benchmark type to timestamp
        self.run_timestamp = f"{run_timestamp}_{benchmark_type}"
        self.benchmark_type = benchmark_type
        self.output_dir = self.base_output_dir / self.run_timestamp
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def save_results(
        self,
        results: List[BenchmarkMetrics],
        formats: List[str] = ["csv", "json"]
    ) -> Dict[str, Path]:
        """
        Save benchmark results in specified formats.

        Args:
            results: List of benchmark metrics
            formats: Output formats ('csv', 'json')

        Returns:
            Dictionary mapping format to file path
        """
        if not results:
            print("No results to save")
            return {}

        saved_files = {}

        # Convert to DataFrame
        df = self._results_to_dataframe(results)

        # Save in requested formats (no timestamp in filename - already in directory)
        if "csv" in formats:
            csv_path = self.output_dir / "benchmark_results.csv"
            df.to_csv(csv_path, index=False)
            saved_files["csv"] = csv_path
            print(f"Saved CSV: {csv_path}")

        if "json" in formats:
            json_path = self.output_dir / "benchmark_results.json"
            results_dict = [r.to_dict() for r in results]
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            saved_files["json"] = json_path
            print(f"Saved JSON: {json_path}")

        return saved_files

    def _results_to_dataframe(self, results: List[BenchmarkMetrics]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame([r.to_dict() for r in results])

    def load_results(self, file_path: str) -> pd.DataFrame:
        """
        Load results from file.

        Args:
            file_path: Path to results file (CSV or JSON)

        Returns:
            DataFrame of results
        """
        path = Path(file_path)

        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def create_comparison_report(
        self,
        results: List[BenchmarkMetrics],
        output_name: Optional[str] = None
    ) -> str:
        """
        Create a markdown comparison report.

        Args:
            results: List of benchmark metrics
            output_name: Optional output filename

        Returns:
            Path to generated report
        """
        df = self._results_to_dataframe(results)

        # Generate report
        report_lines = [
            "# BERT Model Benchmark Results",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTotal models tested: {df['model_name'].nunique()}",
            f"Total configurations: {len(df)}",
            "\n## Summary Statistics\n",
        ]

        # Group by model and device
        summary = df.groupby(['model_name', 'device']).agg({
            'num_parameters': 'first',
            'model_size_mb': 'first',
            'mean_latency_ms': 'mean',
            'throughput_samples_per_sec': 'mean',
            'peak_memory_mb': 'mean'
        }).round(2)

        report_lines.append(summary.to_markdown())

        # Best performers
        report_lines.append("\n## Best Performers\n")

        # Fastest inference (CPU)
        cpu_results = df[df['device'] == 'cpu']
        if not cpu_results.empty:
            fastest_cpu = cpu_results.loc[cpu_results['mean_latency_ms'].idxmin()]
            report_lines.append(f"\n**Fastest CPU Inference:** {fastest_cpu['model_name']}")
            report_lines.append(f"- Latency: {fastest_cpu['mean_latency_ms']:.2f}ms")
            report_lines.append(f"- Throughput: {fastest_cpu['throughput_samples_per_sec']:.2f} samples/sec")

        # Fastest inference (GPU)
        gpu_results = df[df['device'] == 'cuda']
        if not gpu_results.empty:
            fastest_gpu = gpu_results.loc[gpu_results['mean_latency_ms'].idxmin()]
            report_lines.append(f"\n**Fastest GPU Inference:** {fastest_gpu['model_name']}")
            report_lines.append(f"- Latency: {fastest_gpu['mean_latency_ms']:.2f}ms")
            report_lines.append(f"- Throughput: {fastest_gpu['throughput_samples_per_sec']:.2f} samples/sec")

        # Smallest model
        smallest = df.loc[df['model_size_mb'].idxmin()]
        report_lines.append(f"\n**Smallest Model:** {smallest['model_name']}")
        report_lines.append(f"- Size: {smallest['model_size_mb']:.2f} MB")
        report_lines.append(f"- Parameters: {smallest['num_parameters']:,}")

        # Most memory efficient
        most_efficient = df.loc[df['peak_memory_mb'].idxmin()]
        report_lines.append(f"\n**Most Memory Efficient:** {most_efficient['model_name']}")
        report_lines.append(f"- Peak memory: {most_efficient['peak_memory_mb']:.2f} MB")

        # Save report (no timestamp - already in directory name)
        report_name = output_name or "benchmark_report.md"
        report_path = self.output_dir / report_name

        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Report saved: {report_path}")
        return str(report_path)

    def create_comparison_plots(
        self,
        results: List[BenchmarkMetrics],
        output_name: Optional[str] = None
    ):
        """
        Create visualization plots comparing models.

        Args:
            results: List of benchmark metrics
            output_name: Optional base name for output files
        """
        df = self._results_to_dataframe(results)

        # No timestamp - already in directory name
        base_name = output_name or "plots"

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)

        # Plot 1: Latency comparison (average across batch sizes) - sorted by latency
        fig, ax = plt.subplots()
        latency_data = df.groupby(['model_name', 'device'])['mean_latency_ms'].mean().reset_index()

        # Sort by average latency across devices
        model_order = latency_data.groupby('model_name')['mean_latency_ms'].mean().sort_values().index

        sns.barplot(data=latency_data, x='model_name', y='mean_latency_ms', hue='device',
                   order=model_order, ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel('Mean Latency (ms)')
        ax.set_title('Average Inference Latency by Model (Sorted by Speed)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot1_path = self.output_dir / f"{base_name}_latency.png"
        plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot1_path}")

        # Plot 2: Throughput comparison - sorted by throughput (descending)
        fig, ax = plt.subplots()
        throughput_data = df.groupby(['model_name', 'device'])['throughput_samples_per_sec'].mean().reset_index()

        # Sort by average throughput across devices (descending - higher is better)
        model_order = throughput_data.groupby('model_name')['throughput_samples_per_sec'].mean().sort_values(ascending=False).index

        sns.barplot(data=throughput_data, x='model_name', y='throughput_samples_per_sec', hue='device',
                   order=model_order, ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title('Average Throughput by Model (Sorted by Performance)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot2_path = self.output_dir / f"{base_name}_throughput.png"
        plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot2_path}")

        # Plot 3: Memory usage - sorted by memory (ascending)
        fig, ax = plt.subplots()
        memory_data = df.groupby(['model_name', 'device'])['peak_memory_mb'].mean().reset_index()

        # Sort by average memory across devices (ascending - lower is better)
        model_order = memory_data.groupby('model_name')['peak_memory_mb'].mean().sort_values().index

        sns.barplot(data=memory_data, x='model_name', y='peak_memory_mb', hue='device',
                   order=model_order, ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Average Peak Memory by Model (Sorted by Efficiency)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot3_path = self.output_dir / f"{base_name}_memory.png"
        plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot3_path}")

        # Plot 4: Model size vs latency scatter (CPU only, all models)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get CPU data only, aggregated by model
        cpu_data = df[df['device'] == 'cpu'].groupby('model_name').agg({
            'model_size_mb': 'first',
            'mean_latency_ms': 'mean'
        }).reset_index()

        if not cpu_data.empty:
            color_cpu = '#1f77b4'

            # Plot scatter points
            ax.scatter(cpu_data['model_size_mb'], cpu_data['mean_latency_ms'],
                      s=100, alpha=0.6, color=color_cpu)

            # Add model name labels
            for idx, row in cpu_data.iterrows():
                ax.annotate(row['model_name'],
                          (row['model_size_mb'], row['mean_latency_ms']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.8,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor=color_cpu,
                                   alpha=0.2, edgecolor='none'))

            # Add trend line
            if len(cpu_data) > 1:
                z = np.polyfit(cpu_data['model_size_mb'], cpu_data['mean_latency_ms'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(cpu_data['model_size_mb'].min(),
                                     cpu_data['model_size_mb'].max(), 100)
                ax.plot(x_trend, p(x_trend), '--', color=color_cpu, alpha=0.3, linewidth=2)

            ax.set_xlabel('Model Size (MB)', fontsize=11)
            ax.set_ylabel('Mean Latency (ms)', fontsize=11)
            ax.set_title('Model Size vs Inference Latency (CPU)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot4_path = self.output_dir / f"{base_name}_size_vs_latency.png"
        plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot4_path}")

    def get_best_models(
        self,
        results: List[BenchmarkMetrics],
        criterion: str = "latency",
        device: Optional[str] = None,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Get top performing models by criterion.

        Args:
            results: List of benchmark metrics
            criterion: Criterion to rank by ('latency', 'throughput', 'memory', 'size')
            device: Filter by device (None for all)
            top_k: Number of top models to return

        Returns:
            DataFrame of top models
        """
        df = self._results_to_dataframe(results)

        if device:
            df = df[df['device'] == device]

        # Aggregate by model
        if criterion == "latency":
            df = df.groupby('model_name').agg({
                'mean_latency_ms': 'mean',
                'model_size_mb': 'first',
                'num_parameters': 'first'
            }).sort_values('mean_latency_ms').head(top_k)
        elif criterion == "throughput":
            df = df.groupby('model_name').agg({
                'throughput_samples_per_sec': 'mean',
                'model_size_mb': 'first',
                'num_parameters': 'first'
            }).sort_values('throughput_samples_per_sec', ascending=False).head(top_k)
        elif criterion == "memory":
            df = df.groupby('model_name').agg({
                'peak_memory_mb': 'mean',
                'model_size_mb': 'first',
                'num_parameters': 'first'
            }).sort_values('peak_memory_mb').head(top_k)
        elif criterion == "size":
            df = df.groupby('model_name').agg({
                'model_size_mb': 'first',
                'num_parameters': 'first',
                'mean_latency_ms': 'mean'
            }).sort_values('model_size_mb').head(top_k)

        return df

    def create_top_performers_plots(
        self,
        results: List[BenchmarkMetrics],
        output_name: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Create visual plots of top performers in each category.

        Args:
            results: List of benchmark metrics
            output_name: Optional base name for output files
            top_k: Number of top models to show (default: 5)
        """
        base_name = output_name or "top_performers"

        # Set style for clean plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

        # 1. Fastest Inference
        fig, ax = plt.subplots()
        top_fast = self.get_best_models(results, criterion="latency", top_k=top_k)

        # Sort by latency (ascending) for plotting
        top_fast_sorted = top_fast.sort_values('mean_latency_ms', ascending=True)

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_fast_sorted)))
        bars = ax.barh(range(len(top_fast_sorted)), top_fast_sorted['mean_latency_ms'],
                       color=colors, alpha=0.8)

        ax.set_yticks(range(len(top_fast_sorted)))
        ax.set_yticklabels(top_fast_sorted.index, fontsize=10)
        ax.set_xlabel('Mean Latency (ms)', fontsize=11)
        ax.set_title(f'Top {top_k} Fastest Models (Lower is Better)', fontsize=13, fontweight='bold')
        ax.invert_yaxis()  # Best at top

        # Add value labels on bars
        for i, (idx, row) in enumerate(top_fast_sorted.iterrows()):
            ax.text(row['mean_latency_ms'] + 0.5, i, f"{row['mean_latency_ms']:.2f}ms",
                   va='center', fontsize=9)

        plt.tight_layout()
        plot_path = self.output_dir / f"{base_name}_fastest.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

        # 2. Most Memory Efficient
        fig, ax = plt.subplots()
        top_memory = self.get_best_models(results, criterion="memory", top_k=top_k)

        # Sort by memory (ascending) for plotting
        top_memory_sorted = top_memory.sort_values('peak_memory_mb', ascending=True)

        colors = plt.cm.coolwarm_r(np.linspace(0.2, 0.8, len(top_memory_sorted)))
        bars = ax.barh(range(len(top_memory_sorted)), top_memory_sorted['peak_memory_mb'],
                       color=colors, alpha=0.8)

        ax.set_yticks(range(len(top_memory_sorted)))
        ax.set_yticklabels(top_memory_sorted.index, fontsize=10)
        ax.set_xlabel('Peak Memory (MB)', fontsize=11)
        ax.set_title(f'Top {top_k} Most Memory Efficient (Lower is Better)',
                    fontsize=13, fontweight='bold')
        ax.invert_yaxis()  # Best at top

        # Add value labels on bars
        for i, (idx, row) in enumerate(top_memory_sorted.iterrows()):
            ax.text(row['peak_memory_mb'] + 5, i, f"{row['peak_memory_mb']:.1f}MB",
                   va='center', fontsize=9)

        plt.tight_layout()
        plot_path = self.output_dir / f"{base_name}_memory.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

        # 3. Smallest Models
        fig, ax = plt.subplots()
        top_small = self.get_best_models(results, criterion="size", top_k=top_k)

        # Sort by size (ascending) for plotting
        top_small_sorted = top_small.sort_values('model_size_mb', ascending=True)

        colors = plt.cm.plasma_r(np.linspace(0.2, 0.8, len(top_small_sorted)))
        bars = ax.barh(range(len(top_small_sorted)), top_small_sorted['model_size_mb'],
                       color=colors, alpha=0.8)

        ax.set_yticks(range(len(top_small_sorted)))
        ax.set_yticklabels(top_small_sorted.index, fontsize=10)
        ax.set_xlabel('Model Size (MB)', fontsize=11)
        ax.set_title(f'Top {top_k} Smallest Models (Lower is Better)',
                    fontsize=13, fontweight='bold')
        ax.invert_yaxis()  # Best at top

        # Add value labels on bars
        for i, (idx, row) in enumerate(top_small_sorted.iterrows()):
            ax.text(row['model_size_mb'] + 2, i,
                   f"{row['model_size_mb']:.1f}MB ({row['num_parameters']/1e6:.1f}M params)",
                   va='center', fontsize=9)

        plt.tight_layout()
        plot_path = self.output_dir / f"{base_name}_smallest.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

        # 4. Combined Summary (3-panel figure)
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        # Panel 1: Fastest (top 5) - sorted low to high, best at top
        top_fast_5 = self.get_best_models(results, criterion="latency", top_k=5)
        top_fast_5_sorted = top_fast_5.sort_values('mean_latency_ms', ascending=True)
        colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_fast_5_sorted)))
        axes[0].barh(range(len(top_fast_5_sorted)), top_fast_5_sorted['mean_latency_ms'],
                     color=colors1, alpha=0.8)
        axes[0].set_yticks(range(len(top_fast_5_sorted)))
        axes[0].set_yticklabels(top_fast_5_sorted.index, fontsize=9)
        axes[0].set_xlabel('Latency (ms)', fontsize=10)
        axes[0].set_title('Fastest Inference', fontsize=11, fontweight='bold')
        axes[0].invert_yaxis()  # Best (lowest) at top
        for i, (idx, row) in enumerate(top_fast_5_sorted.iterrows()):
            axes[0].text(row['mean_latency_ms'] + 0.5, i, f"{row['mean_latency_ms']:.1f}ms",
                        va='center', fontsize=8)

        # Panel 2: Memory Efficient (top 5) - sorted low to high, best at top
        top_mem_5 = self.get_best_models(results, criterion="memory", top_k=5)
        top_mem_5_sorted = top_mem_5.sort_values('peak_memory_mb', ascending=True)
        colors2 = plt.cm.coolwarm_r(np.linspace(0.2, 0.8, len(top_mem_5_sorted)))
        axes[1].barh(range(len(top_mem_5_sorted)), top_mem_5_sorted['peak_memory_mb'],
                     color=colors2, alpha=0.8)
        axes[1].set_yticks(range(len(top_mem_5_sorted)))
        axes[1].set_yticklabels(top_mem_5_sorted.index, fontsize=9)
        axes[1].set_xlabel('Memory (MB)', fontsize=10)
        axes[1].set_title('Most Memory Efficient', fontsize=11, fontweight='bold')
        axes[1].invert_yaxis()  # Best (lowest) at top
        for i, (idx, row) in enumerate(top_mem_5_sorted.iterrows()):
            axes[1].text(row['peak_memory_mb'] + 5, i, f"{row['peak_memory_mb']:.0f}MB",
                        va='center', fontsize=8)

        # Panel 3: Smallest (top 5) - sorted low to high, best at top
        top_size_5 = self.get_best_models(results, criterion="size", top_k=5)
        top_size_5_sorted = top_size_5.sort_values('model_size_mb', ascending=True)
        colors3 = plt.cm.plasma_r(np.linspace(0.2, 0.8, len(top_size_5_sorted)))
        axes[2].barh(range(len(top_size_5_sorted)), top_size_5_sorted['model_size_mb'],
                     color=colors3, alpha=0.8)
        axes[2].set_yticks(range(len(top_size_5_sorted)))
        axes[2].set_yticklabels(top_size_5_sorted.index, fontsize=9)
        axes[2].set_xlabel('Size (MB)', fontsize=10)
        axes[2].set_title('Smallest Models', fontsize=11, fontweight='bold')
        axes[2].invert_yaxis()  # Best (lowest) at top
        for i, (idx, row) in enumerate(top_size_5_sorted.iterrows()):
            axes[2].text(row['model_size_mb'] + 2, i, f"{row['model_size_mb']:.0f}MB",
                        va='center', fontsize=8)

        plt.suptitle('Top 5 Performers Summary', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_path = self.output_dir / f"{base_name}_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")
