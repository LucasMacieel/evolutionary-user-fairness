"""
Hyperparameter Tuning Module for GA Optimizer using Optuna.

This module provides automated hyperparameter optimization for the
Genetic Algorithm optimizer using Optuna's TPE sampler.
The optimization is 100% focused on minimizing CPU time, finding
hyperparameters that produce feasible solutions as quickly as possible.

Usage:
    python ga_hyperparameter_tuning.py --n_trials 50 --dataset 5Health-rand --model NCF
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import optuna
from optuna.samplers import TPESampler

from data_loader import DataLoader
from ga_optimizer import GAOptimizer
from utils.tools import create_logger


class GAHyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for the GA Optimizer.

    This class defines the search space, objective function, and manages
    the optimization study for finding optimal GA parameters.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        model_name: str = "",
        group_name: str = "",
        k: int = 10,
        eval_metric_list: list = None,
        fairness_metric: str = "f1",
        max_generations: int = 1000,  # Safety ceiling for GA (early stopping halts earlier)
        seed: int = 42,
        output_dir: str = "../results/tuning",
        logger=None,
    ):
        """
        Initialize the hyperparameter tuner.

        Optimization is 100% focused on minimizing CPU time, finding
        configurations that produce feasible solutions as quickly as possible.

        Args:
            data_loader: DataLoader instance with loaded data
            model_name: Name of the recommendation model
            group_name: Name of the grouping method
            k: Top-K for recommendation
            eval_metric_list: List of evaluation metrics
            fairness_metric: Metric used for fairness calculation
            max_generations: Safety ceiling for generations (early stopping halts earlier)
            seed: Random seed for reproducibility
            output_dir: Directory to save tuning results
            logger: Logger instance
        """
        self.data_loader = data_loader
        self.model_name = model_name
        self.group_name = group_name
        self.k = k
        self.eval_metric_list = eval_metric_list or ["ndcg@10", "f1@10"]
        self.fairness_metric = fairness_metric
        self.max_generations = max_generations
        self.seed = seed
        self.output_dir = output_dir
        self.logger = logger or create_logger()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Pre-build vectorized data with disk caching for faster subsequent runs
        # Cache is stored in src/cache/ folder
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        # Include dataset name in cache key to prevent cross-dataset cache collision
        dataset_name = os.path.basename(data_loader.path)
        cache_file = os.path.join(
            cache_dir,
            f"vectorized_cache_{dataset_name}_{model_name}_{group_name}_k{k}.pkl",
        )

        if os.path.exists(cache_file):
            # Load from disk cache (fast)
            self._prebuilt_data = GAOptimizer.load_vectorized_data(cache_file)
            self.logger.info(f"Loaded vectorized data from cache: {cache_file}")
        else:
            # Build and save to disk cache for future runs
            self._prebuilt_data = GAOptimizer.build_vectorized_data(data_loader, k)
            GAOptimizer.save_vectorized_data(self._prebuilt_data, cache_file)
            self.logger.info(f"Built and cached vectorized data: {cache_file}")

        self.logger.info(f"Tuner initialized for {model_name}/{group_name}")

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Optimizes 100% for speed (minimizing CPU time). The goal is to find
        hyperparameters that produce feasible solutions as quickly as possible.

        Args:
            trial: Optuna trial object

        Returns:
            Speed score to maximize: 1.0 / (1.0 + cpu_time)
        """
        # Sample hyperparameters (generations is a fixed ceiling since early stopping halts when feasible)
        population_size = trial.suggest_int("population_size", 10, 100, step=10)
        mutation_rate = trial.suggest_float("mutation_rate", 0.05, 0.5, log=True)
        crossover_rate = trial.suggest_float("crossover_rate", 0.5, 1.0)
        elitism_count = trial.suggest_int("elitism_count", 1, 20)

        # Adaptive penalty parameters (Bean & Hadj-Alouane method)
        penalty_beta1 = trial.suggest_float("penalty_beta1", 1.1, 3.0)
        penalty_beta2 = trial.suggest_float("penalty_beta2", 1.1, 3.0)
        penalty_history_k = trial.suggest_int("penalty_history_k", 3, 10)

        # Ensure elitism_count doesn't exceed population_size
        elitism_count = min(elitism_count, population_size // 2)

        try:
            # Create GA optimizer with trial parameters and pre-built data
            ga = GAOptimizer(
                data_loader=self.data_loader,
                k=self.k,
                eval_metric_list=self.eval_metric_list,
                fairness_metric=self.fairness_metric,
                model_name=self.model_name,
                group_name=self.group_name,
                population_size=population_size,
                generations=self.max_generations,  # Fixed ceiling (early stopping halts earlier)
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elitism_count=elitism_count,
                penalty_beta1=penalty_beta1,
                penalty_beta2=penalty_beta2,
                penalty_history_k=penalty_history_k,
                seed=self.seed + trial.number,  # Vary seed per trial
                prebuilt_data=self._prebuilt_data,  # Use pre-built data for fast init
                logger=self.logger,
            )

            # Run optimization
            results = ga.train()

            # Extract the sum of preference scores (actual GA objective)
            best_fitness = results["best_fitness"]
            final_f1 = (
                results["final_metrics"][1]
                if len(results["final_metrics"]) > 1
                else results["final_metrics"][0]
            )
            final_ugf = results["final_ugf"]
            cpu_time = results["cpu_time"]
            constraint_satisfied = results["constraint_satisfied"]

            # Log trial results (convert to native Python types for JSON serialization)
            trial.set_user_attr("best_fitness", float(best_fitness))
            trial.set_user_attr("final_f1", float(final_f1))
            trial.set_user_attr("final_ugf", float(final_ugf))
            trial.set_user_attr("cpu_time", float(cpu_time))
            trial.set_user_attr("constraint_satisfied", bool(constraint_satisfied))

            self.logger.info(
                f"Trial {trial.number}: fitness={best_fitness:.2f}, f1={final_f1:.4f}, "
                f"ugf={final_ugf:.4f}, time={cpu_time:.2f}s"
            )

            # Only consider feasible trials for selection
            # Infeasible trials (constraint not satisfied) return -inf so they are never selected as best
            if not constraint_satisfied:
                self.logger.info(
                    f"Trial {trial.number}: INFEASIBLE (UGF constraint not satisfied) - excluded from selection"
                )
                return float("-inf")

            # Compute objective: 100% focused on speed (minimizing CPU time)
            # Speed score: higher for faster runs (1/(1+time) gives value in (0,1])
            speed_score = 1.0 / (1.0 + cpu_time)

            trial.set_user_attr("speed_score", float(speed_score))

            self.logger.info(
                f"Trial {trial.number}: speed_score={speed_score:.4f} (cpu_time={cpu_time:.2f}s)"
            )

            return speed_score

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return float("-inf")  # Return worst possible score

    def run_study(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
    ) -> Dict:
        """
        Run the hyperparameter optimization study.

        Args:
            n_trials: Number of trials to run
            timeout: Maximum time in seconds for the study
            study_name: Name for the study (for persistence)

        Returns:
            Dictionary with best parameters and study results
        """
        study_name = study_name or f"ga_tuning_{self.model_name}_{self.group_name}"

        # Create study with TPE sampler
        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
        )

        # Run optimization
        self.logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        start_time = time.perf_counter()

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        elapsed_time = time.perf_counter() - start_time

        # Filter to only feasible trials (constraint_satisfied = True)
        feasible_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.user_attrs.get("constraint_satisfied", False)
        ]

        if not feasible_trials:
            self.logger.warning(
                "No feasible trials found! All trials violated constraints."
            )
            print("\n" + "=" * 70)
            print("HYPERPARAMETER TUNING FAILED")
            print("=" * 70)
            print("\nNo feasible solutions found across all trials.")
            print("Consider relaxing the epsilon constraint or adjusting search space.")
            return {
                "best_params": None,
                "best_score": None,
                "best_trial_number": None,
                "best_trial_user_attrs": None,
                "n_trials": len(study.trials),
                "n_feasible_trials": 0,
                "elapsed_time": elapsed_time,
                "study_name": study_name,
                "model_name": self.model_name,
                "group_name": self.group_name,
                "error": "No feasible trials found",
            }

        # Select best trial from feasible trials only
        best_trial = max(feasible_trials, key=lambda t: t.value)
        best_params = best_trial.params
        best_value = best_trial.value

        self.logger.info(
            f"Selected best from {len(feasible_trials)}/{len(study.trials)} feasible trials"
        )

        # Compile results
        results = {
            "best_params": best_params,
            "best_score": best_value,
            "best_trial_number": best_trial.number,
            "best_trial_user_attrs": best_trial.user_attrs,
            "n_trials": len(study.trials),
            "n_feasible_trials": len(feasible_trials),
            "elapsed_time": elapsed_time,
            "study_name": study_name,
            "model_name": self.model_name,
            "group_name": self.group_name,
        }

        # Save results to JSON
        results_file = os.path.join(
            self.output_dir, f"tuning_results_{self.model_name}_{self.group_name}.json"
        )
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to: {results_file}")
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best score: {best_value:.4f}")

        # Print summary
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING COMPLETE")
        print("=" * 70)
        print(f"\nFeasible trials: {len(feasible_trials)}/{len(study.trials)}")
        print("\nBest Parameters (from feasible trials only):")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\nBest Objective (100% speed): {best_value:.4f}")
        print(f"Trials completed: {len(study.trials)}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"\nResults saved to: {results_file}")

        return results

    def generate_optimized_config(self, results: Dict) -> str:
        """
        Generate a Python code snippet with optimized GA configuration.

        Args:
            results: Results dictionary from run_study

        Returns:
            Python code string with optimized parameters
        """
        params = results["best_params"]

        # Clamp elitism_count to valid range (must be < population_size)
        population_size = params.get("population_size", 50)
        elitism_count = params.get("elitism_count", 10)
        elitism_count = min(elitism_count, population_size - 1)

        config_code = f"""# Optimized GA Parameters (generated by hyperparameter tuning)
        # Model: {self.model_name}, Grouping: {self.group_name}
        # Best Objective (speed score): {results["best_score"]:.4f}
        # Optimization objective: 100% CPU time (speed)

        GA_OPTIMIZED_PARAMS = {{
            "population_size": {population_size},
            "mutation_rate": {params.get("mutation_rate", 0.24):.4f},
            "crossover_rate": {params.get("crossover_rate", 0.55):.4f},
            "elitism_count": {elitism_count},
            "penalty_beta1": {params.get("penalty_beta1", 1.5):.2f},
            "penalty_beta2": {params.get("penalty_beta2", 1.5):.2f},
            "penalty_history_k": {params.get("penalty_history_k", 5)},
        }}
        """

        # Save config to file
        config_file = os.path.join(
            self.output_dir, f"optimized_config_{self.model_name}_{self.group_name}.py"
        )
        with open(config_file, "w") as f:
            f.write(config_code)

        self.logger.info(f"Config saved to: {config_file}")

        return config_code


def main():
    """CLI entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for GA Optimizer using Optuna"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum time in seconds for tuning (default: None)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="5Health-rand",
        help="Dataset name (default: 5Health-rand)",
    )
    parser.add_argument(
        "--model", type=str, default="NCF", help="Model name (default: NCF)"
    )
    parser.add_argument(
        "--grouping",
        type=str,
        default="0.05_count",
        choices=["0.05_count", "sum_0.05", "max_0.05"],
        help="Grouping method (default: 0.05_count)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/tuning",
        help="Output directory for results (default: ../results/tuning)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Map grouping argument to file names
    grouping_map = {
        "0.05_count": (
            "0.05_count",
            "0.05_count_active_test_ratings.txt",
            "0.05_count_inactive_test_ratings.txt",
        ),
        "sum_0.05": (
            "sum_0.05",
            "sum_0.05_price_active_test_ratings.txt",
            "sum_0.05_price_inactive_test_ratings.txt",
        ),
        "max_0.05": (
            "max_0.05",
            "max_0.05_price_active_test_ratings.txt",
            "max_0.05_price_inactive_test_ratings.txt",
        ),
    }

    group_name, group_1_file, group_2_file = grouping_map[args.grouping]

    # Setup paths
    dataset_folder = "../dataset"
    data_path = os.path.join(dataset_folder, args.dataset)
    rank_file = f"{args.model}_rank.csv"

    # Check if files exist
    if not os.path.exists(os.path.join(data_path, rank_file)):
        print(f"Error: {rank_file} not found in {data_path}")
        sys.exit(1)

    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger = create_logger(
        name="ga_tuning",
        path=os.path.join(args.output_dir, f"tuning_{args.model}_{group_name}.log"),
    )

    # Load data
    print(f"\nLoading data: {args.dataset} / {args.model} / {group_name}")
    dl = DataLoader(
        data_path,
        rank_file=rank_file,
        group_1_file=group_1_file,
        group_2_file=group_2_file,
    )

    # Create tuner (optimizes 100% for CPU time)
    tuner = GAHyperparameterTuner(
        data_loader=dl,
        model_name=args.model,
        group_name=group_name,
        seed=args.seed,
        output_dir=args.output_dir,
        logger=logger,
    )

    # Run study
    results = tuner.run_study(
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # Generate config
    config = tuner.generate_optimized_config(results)
    print("\nOptimized Configuration:")
    print(config)


if __name__ == "__main__":
    main()
