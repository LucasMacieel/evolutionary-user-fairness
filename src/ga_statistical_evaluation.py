"""
GA Statistical Evaluation Module

Runs the Genetic Algorithm across all datasets, models, and groupings
for multiple iterations (default: 30) WITHOUT a fixed random seed to
ensure statistical rigor of results.

Collects key metrics from each run and computes statistical summaries:
- Mean, Standard Deviation
- Median
- 95% Confidence Interval
- Interquartile Range (IQR)
- Coefficient of Variation (CV)
- Success Rate (constraint satisfaction)

Usage:
    python ga_statistical_evaluation.py --runs 30 --output ../results/statistical
    python ga_statistical_evaluation.py --runs 5 --dataset 5Beauty-rand --model biasedMF
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_loader import DataLoader
from ga_optimizer import GAOptimizer
from utils.tools import create_logger


class GAStatisticalEvaluator:
    """
    Runs GA multiple times per configuration and computes statistical summaries.
    """

    def __init__(
        self,
        n_runs: int = 30,
        output_dir: str = "../results/statistical",
        datasets: List[str] = None,
        models: List[str] = None,
        groupings: List[Tuple[str, str, str]] = None,
        dataset_folder: str = "../dataset",
        k: int = 10,
        epsilon: str = "auto",
        verbose: bool = True,
        # GA parameters (matching latest ga_optimizer.py defaults)
        population_size: int = 10,
        generations: int = 200,
        mutation_rate: float = 0.3030,
        crossover_rate: float = 0.9715,
        elitism_count: int = 9,
        penalty_lambda: float = None,
        # Adaptive penalty parameters (Bean & Hadj-Alouane method)
        adaptive_penalty: bool = True,
        penalty_beta1: float = 2.54,
        penalty_beta2: float = 3.00,
        penalty_history_k: int = 9,
        # Early stopping
        early_stopping: bool = True,
    ):
        """
        Initialize the statistical evaluator.

        Args:
            n_runs: Number of runs per configuration (default: 30)
            output_dir: Directory to save results
            datasets: List of dataset names (None = all)
            models: List of model names (None = all)
            groupings: List of (group_name, g1_file, g2_file) tuples (None = all)
            dataset_folder: Path to dataset folder
            k: Top-K for recommendations
            epsilon: Epsilon for fairness constraint ('auto' recommended)
            verbose: Print progress to console
            population_size: GA population size
            generations: Number of GA generations
            mutation_rate: Mutation rate for GA
            crossover_rate: Crossover rate for GA
            elitism_count: Number of elites to preserve
            penalty_lambda: Penalty coefficient (None = auto-calculate)
            adaptive_penalty: Enable adaptive penalty (Bean & Hadj-Alouane)
            penalty_beta1: Tightening factor when all feasible
            penalty_beta2: Relaxation factor when all infeasible
            penalty_history_k: Lookback window for feasibility history
            early_stopping: Halt when feasible solution found
        """
        self.n_runs = n_runs
        self.output_dir = output_dir
        self.dataset_folder = dataset_folder
        self.k = k
        self.epsilon = epsilon
        self.verbose = verbose

        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.penalty_lambda = penalty_lambda
        self.adaptive_penalty = adaptive_penalty
        self.penalty_beta1 = penalty_beta1
        self.penalty_beta2 = penalty_beta2
        self.penalty_history_k = penalty_history_k
        self.early_stopping = early_stopping

        # Default configurations (matching ga_optimizer.py main block)
        self.datasets = datasets or ["5Beauty-rand", "5Grocery-rand", "5Health-rand"]
        self.models = models or ["biasedMF", "NCF"]
        self.groupings = groupings or [
            (
                "0.05_count",
                "0.05_count_active_test_ratings.txt",
                "0.05_count_inactive_test_ratings.txt",
            ),
            (
                "sum_0.05",
                "sum_0.05_price_active_test_ratings.txt",
                "sum_0.05_price_inactive_test_ratings.txt",
            ),
            (
                "max_0.05",
                "max_0.05_price_active_test_ratings.txt",
                "max_0.05_price_inactive_test_ratings.txt",
            ),
        ]

        self.metrics_list = ["ndcg@10", "f1@10"]

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Results storage
        self.raw_results: List[Dict] = []
        self.summary_results: List[Dict] = []

    def _run_single_ga(
        self,
        data_loader: DataLoader,
        model_name: str,
        group_name: str,
        run_id: int,
        logger,
        prebuilt_data: Dict = None,
    ) -> Dict:
        """
        Execute a single GA run without setting a random seed.

        Args:
            data_loader: DataLoader instance
            model_name: Name of the model
            group_name: Name of the grouping method
            run_id: Run identifier (1 to n_runs)
            logger: Logger instance
            prebuilt_data: Pre-built vectorized data for faster execution

        Returns:
            Dictionary with run results
        """
        # Create GA optimizer WITHOUT seed (seed=None allows natural randomness)
        ga = GAOptimizer(
            data_loader=data_loader,
            k=self.k,
            eval_metric_list=self.metrics_list,
            fairness_metric="f1",
            epsilon=self.epsilon,
            logger=logger,
            model_name=model_name,
            group_name=group_name,
            seed=None,  # No seed for statistical rigor
            prebuilt_data=prebuilt_data,
            # GA parameters from evaluator config
            population_size=self.population_size,
            generations=self.generations,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elitism_count=self.elitism_count,
            penalty_lambda=self.penalty_lambda,
            # Adaptive penalty parameters
            adaptive_penalty=self.adaptive_penalty,
            penalty_beta1=self.penalty_beta1,
            penalty_beta2=self.penalty_beta2,
            penalty_history_k=self.penalty_history_k,
            # Early stopping
            early_stopping=self.early_stopping,
        )

        # Run optimization
        results = ga.train()

        return {
            "run_id": run_id,
            "final_ugf": float(results["final_ugf"]),
            "original_ugf": float(results["original_ugf"]),
            "epsilon": float(results["epsilon"]),
            "constraint_satisfied": bool(results["constraint_satisfied"]),
            "cpu_time": float(results["cpu_time"]),
            "final_ndcg": float(results["final_metrics"][0]),  # ndcg@10
            "final_f1": float(results["final_metrics"][1]),  # f1@10
            "baseline_ndcg": float(results["baseline_metrics"][0]),
            "baseline_f1": float(results["baseline_metrics"][1]),
            "ugf_reduction": float(results["original_ugf"] - results["final_ugf"]),
            "ugf_reduction_pct": float(
                (results["original_ugf"] - results["final_ugf"])
                / results["original_ugf"]
                * 100
                if results["original_ugf"] > 0
                else 0
            ),
        }

    def _calculate_statistics(self, runs: List[Dict]) -> Dict:
        """
        Calculate statistical summaries for a list of run results.

        Args:
            runs: List of run result dictionaries

        Returns:
            Dictionary with statistical summaries
        """
        n = len(runs)
        if n == 0:
            return {}

        # Extract numeric metrics
        metrics = {
            "final_ugf": [r["final_ugf"] for r in runs],
            "cpu_time": [r["cpu_time"] for r in runs],
            "final_f1": [r["final_f1"] for r in runs],
            "final_ndcg": [r["final_ndcg"] for r in runs],
            "ugf_reduction_pct": [r["ugf_reduction_pct"] for r in runs],
        }

        stats = {}

        for metric_name, values in metrics.items():
            arr = np.array(values)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if n > 1 else 0  # Sample std
            median = np.median(arr)
            min_val = np.min(arr)
            max_val = np.max(arr)

            # 95% Confidence Interval: mean ± 1.96 * std / sqrt(n)
            ci_margin = 1.96 * std / np.sqrt(n) if n > 1 else 0
            ci_lower = mean - ci_margin
            ci_upper = mean + ci_margin

            # Interquartile Range
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1

            # Coefficient of Variation (relative variability)
            cv = (std / mean * 100) if mean != 0 else 0

            stats[metric_name] = {
                "mean": float(mean),
                "std": float(std),
                "median": float(median),
                "min": float(min_val),
                "max": float(max_val),
                "ci_95_lower": float(ci_lower),
                "ci_95_upper": float(ci_upper),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "cv_percent": float(cv),
            }

        # Success rate (constraint satisfaction)
        success_count = sum(1 for r in runs if r["constraint_satisfied"])
        stats["success_rate"] = float(success_count / n * 100)

        # Add sample size
        stats["n_runs"] = n

        # Add constant values from first run
        stats["original_ugf"] = float(runs[0]["original_ugf"])
        stats["epsilon"] = float(runs[0]["epsilon"])
        stats["baseline_f1"] = float(runs[0]["baseline_f1"])
        stats["baseline_ndcg"] = float(runs[0]["baseline_ndcg"])

        return stats

    def run_evaluation(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Run the full statistical evaluation across all configurations.

        Returns:
            Tuple of (raw_results, summary_results)
        """
        total_configs = len(self.datasets) * len(self.models) * len(self.groupings)
        total_runs = total_configs * self.n_runs
        current_config = 0
        current_run_total = 0

        start_time = time.time()

        print("=" * 80)
        print("GA STATISTICAL EVALUATION")
        print("=" * 80)
        print(f"Configurations: {total_configs}")
        print(f"Runs per configuration: {self.n_runs}")
        print(f"Total runs: {total_runs}")
        print(f"Datasets: {self.datasets}")
        print(f"Models: {self.models}")
        print(f"Groupings: {[g[0] for g in self.groupings]}")
        print("=" * 80)

        for dataset_name in self.datasets:
            for model_name in self.models:
                for group_name, group_1_file, group_2_file in self.groupings:
                    current_config += 1
                    config_key = f"{dataset_name}_{model_name}_{group_name}"

                    print(f"\n{'=' * 70}")
                    print(
                        f"CONFIG {current_config}/{total_configs}: {dataset_name} | {model_name} | {group_name}"
                    )
                    print(f"{'=' * 70}")

                    # Setup paths
                    data_path = os.path.join(self.dataset_folder, dataset_name)
                    rank_file = f"{model_name}_rank.csv"

                    # Check if rank file exists
                    if not os.path.exists(os.path.join(data_path, rank_file)):
                        print(f"  SKIPPED: {rank_file} not found")
                        continue

                    try:
                        # Load data once per configuration
                        dl = DataLoader(
                            data_path,
                            rank_file=rank_file,
                            group_1_file=group_1_file,
                            group_2_file=group_2_file,
                        )

                        # Use cached vectorized data from disk if available
                        # Include dataset name in cache key to prevent cross-dataset cache collision
                        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
                        cache_file = os.path.join(
                            cache_dir,
                            f"vectorized_cache_{dataset_name}_{model_name}_{group_name}_k{self.k}.pkl",
                        )

                        if os.path.exists(cache_file):
                            print(f"  Loading cached data from: {cache_file}")
                            prebuilt_data = GAOptimizer.load_vectorized_data(cache_file)
                        else:
                            print("  Building vectorized data (no cache found)...")
                            prebuilt_data = GAOptimizer.build_vectorized_data(
                                dl, self.k
                            )
                            # Optionally save for future use
                            os.makedirs(cache_dir, exist_ok=True)
                            GAOptimizer.save_vectorized_data(prebuilt_data, cache_file)

                        # Setup logger for this configuration
                        logger_dir = os.path.join(self.output_dir, "logs")
                        os.makedirs(logger_dir, exist_ok=True)
                        logger_path = os.path.join(logger_dir, f"stat_{config_key}.log")
                        logger = create_logger(
                            name=f"stat_{config_key}", path=logger_path
                        )

                        # Store runs for this configuration
                        config_runs = []

                        # Run GA n_runs times
                        for run_id in range(1, self.n_runs + 1):
                            current_run_total += 1

                            if self.verbose:
                                print(
                                    f"  Run {run_id}/{self.n_runs} "
                                    f"(Total: {current_run_total}/{total_runs})...",
                                    end="",
                                    flush=True,
                                )

                            run_result = self._run_single_ga(
                                data_loader=dl,
                                model_name=model_name,
                                group_name=group_name,
                                run_id=run_id,
                                logger=logger,
                                prebuilt_data=prebuilt_data,
                            )

                            # Add config info
                            run_result["dataset"] = dataset_name
                            run_result["model"] = model_name
                            run_result["grouping"] = group_name

                            config_runs.append(run_result)
                            self.raw_results.append(run_result)

                            if self.verbose:
                                print(
                                    f" UGF={run_result['final_ugf']:.4f}, "
                                    f"F1={run_result['final_f1']:.4f}, "
                                    f"Time={run_result['cpu_time']:.1f}s"
                                )

                        # Calculate statistics for this configuration
                        stats = self._calculate_statistics(config_runs)
                        stats["dataset"] = dataset_name
                        stats["model"] = model_name
                        stats["grouping"] = group_name
                        self.summary_results.append(stats)

                        # Print summary for this configuration
                        print("\n  --- Configuration Summary ---")
                        print(
                            f"  Final UGF: {stats['final_ugf']['mean']:.4f} +/- {stats['final_ugf']['std']:.4f} "
                            f"(95% CI: [{stats['final_ugf']['ci_95_lower']:.4f}, {stats['final_ugf']['ci_95_upper']:.4f}])"
                        )
                        print(
                            f"  Final F1: {stats['final_f1']['mean']:.4f} +/- {stats['final_f1']['std']:.4f}"
                        )
                        print(
                            f"  CPU Time: {stats['cpu_time']['mean']:.1f} +/- {stats['cpu_time']['std']:.1f}s"
                        )
                        print(f"  Success Rate: {stats['success_rate']:.1f}%")

                    except Exception as e:
                        print(f"  ERROR: {str(e)}")
                        import traceback

                        traceback.print_exc()

        total_time = time.time() - start_time
        print(f"\n{'=' * 80}")
        print("EVALUATION COMPLETE")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"{'=' * 80}")

        return self.raw_results, self.summary_results

    def export_results(self) -> Tuple[str, str]:
        """
        Export results to JSON and CSV files.

        Returns:
            Tuple of (json_path, csv_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export raw results to JSON
        json_path = os.path.join(self.output_dir, f"raw_results_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(self.raw_results, f, indent=2)
        print(f"Raw results saved to: {json_path}")

        # Export summary to CSV (flattened)
        csv_data = []
        for stats in self.summary_results:
            row = {
                "dataset": stats["dataset"],
                "model": stats["model"],
                "grouping": stats["grouping"],
                "n_runs": stats["n_runs"],
                "success_rate": stats["success_rate"],
                "original_ugf": stats["original_ugf"],
                "epsilon": stats["epsilon"],
                "baseline_f1": stats["baseline_f1"],
                "baseline_ndcg": stats["baseline_ndcg"],
            }
            # Add metric statistics
            for metric in [
                "final_ugf",
                "final_f1",
                "final_ndcg",
                "cpu_time",
                "ugf_reduction_pct",
            ]:
                if metric in stats:
                    for stat_name, stat_value in stats[metric].items():
                        row[f"{metric}_{stat_name}"] = stat_value

            csv_data.append(row)

        csv_path = os.path.join(self.output_dir, f"summary_{timestamp}.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")

        return json_path, csv_path

    def print_summary_table(self):
        """Print a formatted summary table to console."""
        print("\n" + "=" * 120)
        print("STATISTICAL SUMMARY")
        print("=" * 120)

        header = (
            f"{'Dataset':<15} {'Model':<10} {'Group':<12} "
            f"{'UGF Mean':>10} {'UGF Std':>9} {'UGF 95% CI':>20} "
            f"{'F1 Mean':>9} {'Success%':>9} {'Time(s)':>9}"
        )
        print(header)
        print("-" * 120)

        for stats in self.summary_results:
            ugf_ci = f"[{stats['final_ugf']['ci_95_lower']:.4f}, {stats['final_ugf']['ci_95_upper']:.4f}]"
            row = (
                f"{stats['dataset'].replace('5', '').replace('-rand', ''):<15} "
                f"{stats['model']:<10} "
                f"{stats['grouping']:<12} "
                f"{stats['final_ugf']['mean']:>10.4f} "
                f"{stats['final_ugf']['std']:>9.4f} "
                f"{ugf_ci:>20} "
                f"{stats['final_f1']['mean']:>9.4f} "
                f"{stats['success_rate']:>8.1f}% "
                f"{stats['cpu_time']['mean']:>9.1f}"
            )
            print(row)

        print("=" * 120)


def main():
    """CLI entry point for statistical evaluation."""
    parser = argparse.ArgumentParser(
        description="GA Statistical Evaluation - Run GA multiple times for statistical rigor"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of runs per configuration (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/statistical",
        help="Output directory for results (default: ../results/statistical)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to run (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to run (default: all)",
    )
    parser.add_argument(
        "--grouping",
        type=str,
        default=None,
        help="Specific grouping to run (default: all). Use group name like '0.05_count'",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run output",
    )

    args = parser.parse_args()

    # Filter configurations based on CLI args
    datasets = [args.dataset] if args.dataset else None
    models = [args.model] if args.model else None

    # Handle grouping filter
    groupings = None
    if args.grouping:
        all_groupings = [
            (
                "0.05_count",
                "0.05_count_active_test_ratings.txt",
                "0.05_count_inactive_test_ratings.txt",
            ),
            (
                "sum_0.05",
                "sum_0.05_price_active_test_ratings.txt",
                "sum_0.05_price_inactive_test_ratings.txt",
            ),
            (
                "max_0.05",
                "max_0.05_price_active_test_ratings.txt",
                "max_0.05_price_inactive_test_ratings.txt",
            ),
        ]
        groupings = [g for g in all_groupings if g[0] == args.grouping]
        if not groupings:
            print(f"Error: Unknown grouping '{args.grouping}'")
            print(f"Available: {[g[0] for g in all_groupings]}")
            sys.exit(1)

    # Create evaluator
    evaluator = GAStatisticalEvaluator(
        n_runs=args.runs,
        output_dir=args.output,
        datasets=datasets,
        models=models,
        groupings=groupings,
        verbose=not args.quiet,
    )

    # Run evaluation
    evaluator.run_evaluation()

    # Export results
    evaluator.export_results()

    # Print summary table
    evaluator.print_summary_table()

    print("\nStatistical evaluation complete!")


if __name__ == "__main__":
    main()
