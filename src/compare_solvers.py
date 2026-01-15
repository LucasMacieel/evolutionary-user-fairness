"""
MILP vs GA Solver Comparison Script.

Runs both the MILP solver (PuLP/SCIP) and GA optimizer on the same dataset
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
from utils.tools import create_logger

# Import optimizers
from model import UGF as MILPOptimizer
from ga_optimizer import GAOptimizer


def run_comparison(
    dataset_name: str,
    model_name: str,
    group_name: str,
    dataset_folder: str = "../dataset",
    results_folder: str = "../results/comparison",
    solver: str = "SCIP",
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
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Removed specific timestamp to overwrite
    log_file = os.path.join(
        results_folder, f"comparison_{model_name}_{dataset_name}_{group_name}.log"
    )
    logger = create_logger(name="comparison_logger", path=log_file)

    print("=" * 80)
    print("MILP vs GA Comparison")
    print(
        f"Dataset: {dataset_name} | Model: {model_name} | Grouping: {group_name} | Solver: {solver}"
    )
    print("=" * 80)
    logger.info(
        f"Comparison: {dataset_name} | {model_name} | {group_name} | Solver: {solver}"
    )

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
    print(f"Running MILP Optimizer (PuLP/{solver})...")
    print("-" * 40)
    logger.info("=" * 40)
    logger.info(f"MILP Optimizer ({solver})")
    logger.info("=" * 40)

    milp_logger = create_logger(
        name="milp_logger",
        path=os.path.join(
            results_folder, f"milp_{model_name}_{dataset_name}_{group_name}.log"
        ),
    )

    milp_start = time.perf_counter()
    milp_optimizer = MILPOptimizer(
        data_loader=dl,
        k=k,
        eval_metric_list=metrics_list,
        fairness_metric="f1",
        epsilon="auto",
        logger=milp_logger,
        model_name=model_name,
        group_name=group_name,
        solver=solver,
    )
    milp_results = milp_optimizer.train()
    milp_time = time.perf_counter() - milp_start

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
        path=os.path.join(
            results_folder, f"ga_{model_name}_{dataset_name}_{group_name}.log"
        ),
    )

    ga_start = time.perf_counter()
    ga_optimizer = GAOptimizer(
        data_loader=dl_ga,
        k=10,
        eval_metric_list=["ndcg@10", "f1@10"],
        fairness_metric="f1",
        epsilon="auto",
        logger=ga_logger,
        model_name=model_name,
        group_name=group_name,
        population_size=50,
        generations=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_count=5,
        penalty_lambda=None,
        seed=42,
    )
    ga_results = ga_optimizer.train()
    ga_time = time.perf_counter() - ga_start

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
        "Metric": [
            "CPU Time (s)",
            "Final UGF",
            "NDCG@10 (Overall)",
            "F1@10 (Overall)",
            "Constraint Satisfied",
        ],
        "MILP": [
            f"{milp_time:.2f}",
            f"{milp_results['final_ugf']:.4f}" if milp_results else "N/A",
            f"{milp_results['final_metrics'][0]:.4f}" if milp_results else "N/A",
            f"{milp_results['final_metrics'][1]:.4f}" if milp_results else "N/A",
            "Yes" if milp_results and milp_results["constraint_satisfied"] else "No",
        ],
        "GA": [
            f"{ga_time:.2f}",
            f"{ga_results['final_ugf']:.4f}",
            f"{ga_results['final_metrics'][0]:.4f}",
            f"{ga_results['final_metrics'][1]:.4f}",
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

    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "Group": group_name,
        "MILP_Time": milp_time,
        "MILP_UGF": milp_results["final_ugf"] if milp_results else None,
        "MILP_NDCG": milp_results["final_metrics"][0] if milp_results else None,
        "MILP_F1": milp_results["final_metrics"][1] if milp_results else None,
        "GA_Time": ga_time,
        "GA_UGF": ga_results["final_ugf"],
        "GA_NDCG": ga_results["final_metrics"][0],
        "GA_F1": ga_results["final_metrics"][1],
        "GA_Satisfied": ga_results["constraint_satisfied"],
    }


if __name__ == "__main__":
    # Configuration
    DATASETS = ["5Beauty-rand"]
    MODELS = ["NCF", "biasedMF"]
    GROUPS = ["0.05_count", "sum_0.05", "max_0.05"]

    # Check for command line args (single run mode)
    if len(sys.argv) >= 4:
        # Run single
        run_comparison(
            dataset_name=sys.argv[1],
            model_name=sys.argv[2],
            group_name=sys.argv[3],
            solver=sys.argv[4] if len(sys.argv) > 4 else "SCIP",
        )
    else:
        # Run batch over all combinations
        total_runs = len(DATASETS) * len(MODELS) * len(GROUPS)
        print("\n" + "=" * 80)
        print(f"Running Full Benchmark: {total_runs} experiments")
        print(f"Datasets: {DATASETS}")
        print(f"Models:   {MODELS}")
        print(f"Groups:   {GROUPS}")
        print("=" * 80 + "\n")

        all_results = []
        run_count = 0

        for dataset in DATASETS:
            for model in MODELS:
                for group in GROUPS:
                    run_count += 1
                    print(
                        f"\nProcessing [{run_count}/{total_runs}]: {dataset} | {model} | {group}..."
                    )
                    try:
                        res = run_comparison(
                            dataset_name=dataset, model_name=model, group_name=group
                        )
                        all_results.append(res)
                    except Exception as e:
                        print(f"Error processing {dataset}|{model}|{group}: {e}")
                        # Append error result to keep track
                        all_results.append(
                            {
                                "Dataset": dataset,
                                "Model": model,
                                "Group": group,
                                "MILP_Time": 0,
                                "GA_Time": 0,
                                "GA_Satisfied": f"Error: {str(e)}",
                            }
                        )

        # Create Master Summary
        if all_results:
            df_master = pd.DataFrame(all_results)

            # Reorder columns (ensure validation even if some columns missing due to errors)
            cols = [
                "Dataset",
                "Model",
                "Group",
                "MILP_Time",
                "GA_Time",
                "MILP_UGF",
                "GA_UGF",
                "MILP_NDCG",
                "GA_NDCG",
                "GA_Satisfied",
            ]

            # Filter columns that actually exist in the dataframe
            final_cols = [c for c in cols if c in df_master.columns]
            df_master = df_master[final_cols]

            print("\n" + "=" * 80)
            print("MASTER COMPARISON SUMMARY")
            print("=" * 80)
            print(df_master.to_string(index=False))

            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"../results/comparison/master_summary_{timestamp}.csv"
            df_master.to_csv(save_path, index=False)
            print(f"\nMaster summary saved to: {save_path}")
