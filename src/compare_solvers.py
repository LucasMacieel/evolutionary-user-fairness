"""
MILP vs GA Solver Comparison Script.

Runs both the MILP solver (PuLP/HiGHS) and GA optimizer on the same dataset
and compares their results in terms of:
- Solution quality (F1@10, NDCG@10)
- Fairness (UGF gap)
- Computational time
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd

from data_loader import DataLoader
from utils.tools import create_logger, evaluation_methods

# Import optimizers
from model import UGF as MILPOptimizer
from ga_optimizer import GAOptimizer


def run_comparison(
    dataset_name: str,
    model_name: str,
    group_name: str,
    dataset_folder: str = "../dataset",
    results_folder: str = "../results/comparison",
):
    """
    Run comparison between MILP and GA solvers.

    Args:
        dataset_name: Name of dataset (e.g., '5Beauty-rand')
        model_name: Name of model (e.g., 'NCF')
        group_name: Grouping method (e.g., '0.05_count')
        dataset_folder: Path to dataset folder
        results_folder: Path to save results
    """
    # Setup paths
    data_path = os.path.join(dataset_folder, dataset_name)
    rank_file = f"{model_name}_rank.csv"

    # Map group name to file names
    if group_name == "0.05_count":
        group_1_file = "0.05_count_active_test_ratings.txt"
        group_2_file = "0.05_count_inactive_test_ratings.txt"
    elif group_name == "sum_0.05":
        group_1_file = "sum_0.05_price_active_test_ratings.txt"
        group_2_file = "sum_0.05_price_inactive_test_ratings.txt"
    elif group_name == "max_0.05":
        group_1_file = "max_0.05_price_active_test_ratings.txt"
        group_2_file = "max_0.05_price_inactive_test_ratings.txt"
    else:
        raise ValueError(f"Unknown group_name: {group_name}")

    # Create results directory
    os.makedirs(results_folder, exist_ok=True)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        results_folder,
        f"comparison_{model_name}_{dataset_name}_{group_name}_{timestamp}.log"
    )
    logger = create_logger(name="comparison_logger", path=log_file)

    print("=" * 80)
    print(f"MILP vs GA Comparison")
    print(f"Dataset: {dataset_name} | Model: {model_name} | Grouping: {group_name}")
    print("=" * 80)
    logger.info(f"Comparison: {dataset_name} | {model_name} | {group_name}")

    # Load data
    print("\nLoading data...")
    dl = DataLoader(
        data_path,
        rank_file=rank_file,
        group_1_file=group_1_file,
        group_2_file=group_2_file,
    )
    print(f"  Total users: {len(dl.rank_df['uid'].unique())}")
    print(f"  Total records: {len(dl.rank_df)}")

    # Configuration
    metrics_list = ["ndcg@10", "f1@10"]
    k = 10

    # ========================================
    # Run MILP Optimizer
    # ========================================
    print("\n" + "-" * 40)
    print("Running MILP Optimizer (PuLP/HiGHS)...")
    print("-" * 40)
    logger.info("=" * 40)
    logger.info("MILP Optimizer")
    logger.info("=" * 40)

    milp_logger = create_logger(
        name="milp_logger",
        path=os.path.join(results_folder, f"milp_{model_name}_{dataset_name}_{group_name}_{timestamp}.log")
    )

    milp_start = time.time()
    milp_optimizer = MILPOptimizer(
        data_loader=dl,
        k=k,
        eval_metric_list=metrics_list,
        fairness_metric="f1",
        epsilon="auto",
        logger=milp_logger,
        model_name=model_name,
        group_name=group_name,
    )
    milp_optimizer.train()
    milp_time = time.time() - milp_start

    print(f"MILP completed in {milp_time:.2f} seconds")
    logger.info(f"MILP CPU time: {milp_time:.2f}s")

    # ========================================
    # Run GA Optimizer
    # ========================================
    print("\n" + "-" * 40)
    print("Running GA Optimizer...")
    print("-" * 40)
    logger.info("=" * 40)
    logger.info("GA Optimizer")
    logger.info("=" * 40)

    # Reload data for fresh start
    dl_ga = DataLoader(
        data_path,
        rank_file=rank_file,
        group_1_file=group_1_file,
        group_2_file=group_2_file,
    )

    ga_logger = create_logger(
        name="ga_logger",
        path=os.path.join(results_folder, f"ga_{model_name}_{dataset_name}_{group_name}_{timestamp}.log")
    )

    ga_start = time.time()
    ga_optimizer = GAOptimizer(
        data_loader=dl_ga,
        k=k,
        eval_metric_list=metrics_list,
        fairness_metric="f1",
        epsilon="auto",
        logger=ga_logger,
        model_name=model_name,
        group_name=group_name,
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_count=5,
        penalty_lambda=1000.0,
        seed=42,
    )
    ga_results = ga_optimizer.train()
    ga_time = time.time() - ga_start

    print(f"GA completed in {ga_time:.2f} seconds")
    logger.info(f"GA CPU time: {ga_time:.2f}s")

    # ========================================
    # Summary Comparison
    # ========================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Create comparison table
    comparison_data = {
        "Metric": ["CPU Time (s)", "Final UGF", "Constraint Satisfied"],
        "MILP": [f"{milp_time:.2f}", "See MILP log", "See MILP log"],
        "GA": [
            f"{ga_time:.2f}",
            f"{ga_results['final_ugf']:.4f}",
            "Yes" if ga_results["constraint_satisfied"] else "No",
        ],
    }

    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    logger.info("\nComparison Summary:")
    logger.info(df_comparison.to_string(index=False))

    # Speedup calculation
    if milp_time > 0:
        speedup = milp_time / ga_time if ga_time > 0 else float("inf")
        print(f"\nSpeedup (MILP time / GA time): {speedup:.2f}x")
        logger.info(f"Speedup: {speedup:.2f}x")

    print(f"\nDetailed logs saved to: {results_folder}")
    print(f"Comparison log: {log_file}")


if __name__ == "__main__":
    # Default configuration
    dataset_name = "5Beauty-rand"
    model_name = "NCF"
    group_name = "0.05_count"

    # Parse command line arguments if provided
    if len(sys.argv) >= 4:
        dataset_name = sys.argv[1]
        model_name = sys.argv[2]
        group_name = sys.argv[3]

    run_comparison(
        dataset_name=dataset_name,
        model_name=model_name,
        group_name=group_name,
    )
