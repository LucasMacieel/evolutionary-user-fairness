"""
Genetic Algorithm Optimizer with Penalty-Based Fitness Function
for User-oriented Fairness in Recommendation.

OPTIMIZED VERSION: Uses vectorized NumPy operations for 10-100x speedup.

This module implements a GA that solves the same problem as the MILP solver:
- maximize sum of preference scores (S_ij * W_ij)
- subject to fairness constraint: |UGF(group1, group2)| <= epsilon
- subject to selection constraint: exactly K items per user
"""

import os
import pickle
import numpy as np
import pandas as pd
import random
import time
from typing import List, Tuple, Dict

from data_loader import DataLoader
from utils.tools import create_logger, evaluation_methods


class GAOptimizer:
    """
    Genetic Algorithm optimizer for fairness-aware recommendation re-ranking.
    Uses vectorized NumPy operations for performance.
    """

    # Class constants
    FEASIBILITY_TOLERANCE = 1e-9
    START_EPSILON_FACTOR = 0.99
    INIT_MUTATION_RATE = 0.2

    def __init__(
        self,
        data_loader: DataLoader,
        k: int = 10,
        eval_metric_list: List[str] = None,
        fairness_metric: str = "f1",
        epsilon: float = None,
        logger=None,
        model_name: str = "",
        group_name: str = "",
        # GA parameters (optimized via Optuna hyperparameter tuning)
        population_size: int = 10,
        generations: int = 50,
        mutation_rate: float = 0.3030,
        crossover_rate: float = 0.9715,
        elitism_count: int = 9,
        penalty_lambda: float = None,
        seed: int = None,
        # Adaptive penalty parameters (Bean & Hadj-Alouane method)
        adaptive_penalty: bool = True,
        penalty_beta1: float = 2.54,  # Tightening factor when all feasible
        penalty_beta2: float = 3.00,  # Relaxation factor when all infeasible
        penalty_history_k: int = 9,  # Lookback window for feasibility history
        # Early stopping parameters
        early_stopping: bool = True,  # Halt immediately when feasible solution found
        # Pre-built data for faster initialization (use build_vectorized_data())
        prebuilt_data: Dict = None,
    ):
        """Initialize GA optimizer with vectorized data structures."""
        self.data_loader = data_loader
        self.dataset_name = data_loader.path.split("/")[-1]
        self.k = k
        self.eval_metric_list = eval_metric_list or ["ndcg@10", "f1@10"]
        self.fairness_metric = fairness_metric
        self._epsilon_input = epsilon
        self.epsilon = None
        self.original_ugf = None
        self.model_name = model_name
        self.group_name = group_name

        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self._penalty_lambda_input = penalty_lambda
        self.penalty_lambda = None

        # Adaptive penalty settings
        self.adaptive_penalty = adaptive_penalty
        self.penalty_beta1 = penalty_beta1
        self.penalty_beta2 = penalty_beta2
        self.penalty_history_k = penalty_history_k
        self.early_stopping = early_stopping

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if logger is None:
            self.logger = create_logger()
        else:
            self.logger = logger

        # Use pre-built vectorized data, or load/build with disk caching
        if prebuilt_data is not None:
            self._load_prebuilt_data(prebuilt_data)
        else:
            # Check for disk cache first
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
                prebuilt_data = GAOptimizer.load_vectorized_data(cache_file)
            else:
                # Build and save to disk cache for future runs
                prebuilt_data = GAOptimizer.build_vectorized_data(data_loader, k)
                GAOptimizer.save_vectorized_data(prebuilt_data, cache_file)

            self._load_prebuilt_data(prebuilt_data)

    @staticmethod
    def build_vectorized_data(data_loader: DataLoader, k: int = 10) -> Dict:
        """
        Build vectorized data structures once for reuse across multiple GA instances.

        This is useful for hyperparameter tuning where the same data is used
        across many trials, avoiding redundant data preparation.

        Args:
            data_loader: DataLoader instance with loaded data
            k: Top-K for recommendation (needed for F1 denominator)

        Returns:
            Dictionary containing all pre-built vectorized data structures
        """
        all_df = data_loader.rank_df.copy(deep=True)
        g1_df = data_loader.g1_df.copy(deep=True)
        g2_df = data_loader.g2_df.copy(deep=True)

        # Get unique users
        user_ids = all_df["uid"].unique()
        n_users = len(user_ids)
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

        # Determine max items per user
        user_counts = all_df.groupby("uid").size()
        n_items = user_counts.max()
        items_per_user = user_counts.to_dict()

        # Build score and label matrices
        import numpy as np

        scores_matrix = np.zeros((n_users, n_items), dtype=np.float32)
        labels_matrix = np.zeros((n_users, n_items), dtype=np.float32)

        for uid in user_ids:
            idx = user_id_to_idx[uid]
            user_df = all_df[all_df["uid"] == uid]
            n = len(user_df)
            scores_matrix[idx, :n] = user_df["score"].values
            labels_matrix[idx, :n] = user_df["label"].values

        # Pre-compute derived values
        total_relevant = labels_matrix.sum(axis=1)
        has_relevant = total_relevant > 0
        f1_denominator = total_relevant + k

        # Group masks
        g1_users = set(g1_df["uid"].unique())
        g2_users = set(g2_df["uid"].unique())
        g1_mask = np.array([uid in g1_users for uid in user_ids])
        g2_mask = np.array([uid in g2_users for uid in user_ids])

        print(f"Pre-built vectorized data: {n_users} users, {n_items} items per user")
        print(f"Group 1: {g1_mask.sum()} users, Group 2: {g2_mask.sum()} users")

        return {
            "all_df": all_df,
            "g1_df": g1_df,
            "g2_df": g2_df,
            "user_ids": user_ids,
            "n_users": n_users,
            "n_items": n_items,
            "user_id_to_idx": user_id_to_idx,
            "items_per_user": items_per_user,
            "scores_matrix": scores_matrix,
            "labels_matrix": labels_matrix,
            "total_relevant": total_relevant,
            "has_relevant": has_relevant,
            "f1_denominator": f1_denominator,
            "g1_mask": g1_mask,
            "g2_mask": g2_mask,
            "_g1_user_set": g1_users,
            "_g2_user_set": g2_users,
            "n_g1": g1_mask.sum(),
            "n_g2": g2_mask.sum(),
        }

    @staticmethod
    def save_vectorized_data(data: Dict, filepath: str) -> None:
        """
        Save pre-built vectorized data to disk using pickle.

        Args:
            data: Dictionary from build_vectorized_data()
            filepath: Path to save the pickle file (e.g., 'cache/vectorized_data.pkl')
        """
        # Create directory if needed
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Vectorized data saved to: {filepath}")

    @staticmethod
    def load_vectorized_data(filepath: str) -> Dict:
        """
        Load pre-built vectorized data from disk.

        Args:
            filepath: Path to the pickle file

        Returns:
            Dictionary containing vectorized data structures
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        print(f"Vectorized data loaded from: {filepath}")
        print(f"  {data['n_users']} users, {data['n_items']} items per user")

        return data

    def _load_prebuilt_data(self, data: Dict):
        """Load pre-built vectorized data structures."""
        self.all_df = data["all_df"]
        self.g1_df = data["g1_df"]
        self.g2_df = data["g2_df"]
        self.user_ids = data["user_ids"]
        self.n_users = data["n_users"]
        self.n_items = data["n_items"]
        self.user_id_to_idx = data["user_id_to_idx"]
        self.items_per_user = data["items_per_user"]
        self.scores_matrix = data["scores_matrix"]
        self.labels_matrix = data["labels_matrix"]
        self.total_relevant = data["total_relevant"]
        self.has_relevant = data["has_relevant"]
        self.f1_denominator = data["f1_denominator"]
        self.g1_mask = data["g1_mask"]
        self.g2_mask = data["g2_mask"]
        self._g1_user_set = data["_g1_user_set"]
        self._g2_user_set = data["_g2_user_set"]
        self.n_g1 = data["n_g1"]
        self.n_g2 = data["n_g2"]

    def _calculate_fitness_batch(
        self, population: np.ndarray, current_epsilon: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate objective (quality) and constraint violation for population.

        Args:
            population: shape (pop_size, n_users, n_items) binary selection matrices
            current_epsilon: current fairness constraint threshold

        Returns:
            objectives: shape (pop_size,) - The objective function value (sum of scores)
            violations: shape (pop_size,) - The constraint violation (max(0, ugf - epsilon))
        """
        pop_size = population.shape[0]

        # 1. Objective: Sum of preference scores
        # population: (pop_size, n_users, n_items)
        # scores_matrix: (n_users, n_items)
        objectives = (population * self.scores_matrix).sum(axis=(1, 2))  # (pop_size,)

        # 2. Constraint: Fairness (UGF <= epsilon)

        # Calculate selected labels per user per individual
        selected_labels = (
            population * self.labels_matrix
        )  # (pop_size, n_users, n_items)
        selected_relevant = selected_labels.sum(axis=2)  # (pop_size, n_users)

        # F1 metric per user: 2 * selected_relevant / (total_relevant + k)
        # Only for users with relevant items
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_per_user = (
                2 * selected_relevant / self.f1_denominator
            )  # (pop_size, n_users)
            f1_per_user = np.nan_to_num(f1_per_user, 0)

        # Mask users without relevant items
        f1_per_user[:, ~self.has_relevant] = 0

        # Group averages
        g1_has_relevant = self.g1_mask & self.has_relevant
        g2_has_relevant = self.g2_mask & self.has_relevant

        g1_avg = (
            f1_per_user[:, g1_has_relevant].mean(axis=1)
            if g1_has_relevant.sum() > 0
            else np.zeros(pop_size)
        )
        g2_avg = (
            f1_per_user[:, g2_has_relevant].mean(axis=1)
            if g2_has_relevant.sum() > 0
            else np.zeros(pop_size)
        )

        ugf_gaps = np.abs(g1_avg - g2_avg)  # (pop_size,)

        # Calculate violation
        # Violation = max(0, ugf - epsilon)
        # We normalize violation by epsilon to keep it roughly scale-independent, but raw works too.
        # Here we use raw deviation.
        violations = np.maximum(0, ugf_gaps - current_epsilon)

        # Calculate signed UGF (G1 - G2) to know direction of violation
        signed_ugf = g1_avg - g2_avg

        return objectives, violations, ugf_gaps, signed_ugf

    def _create_initial_population(self, size: int) -> np.ndarray:
        """
        Create initial population as 'Perturbed Greedy'.
        Instead of starting from zero quality (random), we start near the
        greedy baseline (high quality) and explore the neighborhood.
        """
        population = np.zeros((size, self.n_users, self.n_items), dtype=np.int8)

        # Start with greedy selection for everyone
        greedy_base = self._create_greedy_individual()

        # Apply random perturbations (simulated annealing style start)
        # We use the mutation logic to perturb
        for i in range(size):
            population[i] = greedy_base.copy()

        # Perturb the population (except the first one which stays pure greedy)
        # We apply mutations to diversify the population
        # Use a slightly higher rate for initialization diversity
        mutate_mask = np.random.random((size, self.n_users)) < self.INIT_MUTATION_RATE

        for i in range(1, size):  # Skip index 0 to keep one pure greedy
            for u in range(self.n_users):
                if mutate_mask[i, u]:
                    selected = np.where(population[i, u] == 1)[0]
                    unselected = np.where(population[i, u] == 0)[0]

                    if len(selected) > 0 and len(unselected) > 0:
                        # Random swap for initialization (unbiased exploration)
                        to_remove = np.random.choice(selected)
                        to_add = np.random.choice(unselected)
                        population[i, u, to_remove] = 0
                        population[i, u, to_add] = 1

        return population

    def _create_greedy_individual(self) -> np.ndarray:
        """Create greedy individual (top-K by score per user)."""
        individual = np.zeros((self.n_users, self.n_items), dtype=np.int8)

        for u in range(self.n_users):
            top_k = np.argsort(self.scores_matrix[u])[-self.k :]
            individual[u, top_k] = 1

        return individual

    def _two_parent_crossover(
        self,
        population: np.ndarray,
        objectives: np.ndarray,
        violations: np.ndarray,
        n_offspring: int,
    ) -> np.ndarray:
        """
        Two-parent uniform crossover.

        For each offspring, select 2 parents from the population via tournament selection.
        For each user position, randomly pick from one of the two parents.

        Returns:
            offspring: shape (n_offspring, n_users, n_items)
        """
        offspring = np.zeros((n_offspring, self.n_users, self.n_items), dtype=np.int8)

        for i in range(n_offspring):
            # Apply crossover with probability
            if np.random.random() >= self.crossover_rate:
                # No crossover - just copy a selected parent
                parent = self._tournament_selection(
                    population, objectives, violations, 1
                )[0]
                offspring[i] = parent
                continue

            # Select 2 parents for crossover
            parents = self._tournament_selection(population, objectives, violations, 2)

            # For each user, randomly pick from one parent (uniform crossover)
            parent_choices = np.random.randint(0, 2, size=self.n_users)

            for u in range(self.n_users):
                offspring[i, u] = parents[parent_choices[u], u]

        return offspring

    def _mutate_batch(
        self, population: np.ndarray, bias_dir: float = 0.0, current_rate: float = None
    ) -> np.ndarray:
        """
        Smart Swap Mutation with Repair Bias.

        Args:
            population: Batch of individuals
            bias_dir: Global bias direction (G1_avg - G2_avg).
                      positive -> G1 advantage (Need to suppress G1 / boost G2)
                      negative -> G2 advantage (Need to suppress G2 / boost G1)
        """
        pop_size = population.shape[0]
        mutated = population.copy()

        # Use provided rate or default instance rate
        rate = current_rate if current_rate is not None else self.mutation_rate

        # Decide which users to mutate for each individual
        mutate_mask = np.random.random((pop_size, self.n_users)) < rate

        # Repair logic: apply bias when violation exceeds epsilon
        apply_repair = abs(bias_dir) > self.epsilon
        # If bias is positive (G1 > G2), we want to lower G1 (Suppress) and raise G2 (Boost)
        # If bias is negative (G2 > G1), we want to lower G2 (Suppress) and raise G1 (Boost)

        # Pre-compute scores for fast access (though numpy indexing is fast enough)

        for i in range(pop_size):
            for u in range(self.n_users):
                if mutate_mask[i, u]:
                    selected = np.where(mutated[i, u] == 1)[0]
                    unselected = np.where(mutated[i, u] == 0)[0]

                    if len(selected) > 0 and len(unselected) > 0:
                        # Determine strategy for this user
                        strategy = "random"
                        if apply_repair:
                            is_g1 = self.g1_mask[u]
                            is_g2 = self.g2_mask[u]

                            # Logic:
                            # If G1 > G2 (bias > 0):
                            #   - G1 user: Suppress (Sacrifice quality)
                            #   - G2 user: Boost (Improve quality)
                            # If G2 > G1 (bias < 0):
                            #   - G1 user: Boost
                            #   - G2 user: Suppress

                            if bias_dir > 0:
                                if is_g1:
                                    strategy = "suppress"
                                elif is_g2:
                                    strategy = "boost"
                            else:
                                if is_g1:
                                    strategy = "boost"
                                elif is_g2:
                                    strategy = "suppress"

                        # Fallback for balanced/random if not repairing or neutral group
                        if strategy == "random" and np.random.random() < 0.5:
                            strategy = "boost"  # Default Smart Swap is a boost

                        if strategy == "boost":
                            # --- Boost (Greedy) ---
                            # Remove Worst Selected -> Add Best Unselected
                            cand_remove = np.random.choice(
                                selected, size=min(3, len(selected)), replace=False
                            )
                            scores_remove = self.scores_matrix[u, cand_remove]
                            to_remove = cand_remove[np.argmin(scores_remove)]

                            cand_add = np.random.choice(
                                unselected, size=min(5, len(unselected)), replace=False
                            )
                            scores_add = self.scores_matrix[u, cand_add]
                            to_add = cand_add[np.argmax(scores_add)]

                        elif strategy == "suppress":
                            # --- Suppress (Altruistic) ---
                            # Remove Best Selected -> Add Worst Unselected
                            # (Sacrifice own score to lower group average)
                            cand_remove = np.random.choice(
                                selected, size=min(3, len(selected)), replace=False
                            )
                            scores_remove = self.scores_matrix[u, cand_remove]
                            to_remove = cand_remove[
                                np.argmax(scores_remove)
                            ]  # Remove BEST

                            cand_add = np.random.choice(
                                unselected, size=min(5, len(unselected)), replace=False
                            )
                            scores_add = self.scores_matrix[u, cand_add]
                            to_add = cand_add[np.argmin(scores_add)]  # Add WORST
                        else:
                            # --- Random Swap ---
                            to_remove = np.random.choice(selected)
                            to_add = np.random.choice(unselected)

                        # Perform swap
                        mutated[i, u, to_remove] = 0
                        mutated[i, u, to_add] = 1

        return mutated

    def _tournament_selection(
        self,
        population: np.ndarray,
        objectives: np.ndarray,
        violations: np.ndarray,
        n_select: int,
        tournament_size: int = 5,
    ) -> np.ndarray:
        """
        Tournament selection using Deb's Feasibility Rules.

        Rules for comparing two solutions A and B:
        1. If A is feasible and B is infeasible, A wins.
        2. If A and B are both feasible, the one with better objective wins.
        3. If A and B are both infeasible, the one with lower constraint violation wins.
        """
        pop_size = population.shape[0]
        selected = np.zeros((n_select, self.n_users, self.n_items), dtype=np.int8)

        # Vectorized tournament
        # We perform 'n_select' tournaments independently

        for i in range(n_select):
            # Select random candidates
            candidates_idx = np.random.choice(
                pop_size, size=tournament_size, replace=False
            )

            # Extract their metrics
            cand_obj = objectives[candidates_idx]
            cand_viol = violations[candidates_idx]

            # Determine best candidate using Deb's rules
            # We'll find the best index within the candidates array
            best_local_idx = 0

            for j in range(1, tournament_size):
                # Compare current best (A) with candidate (B)
                curr_idx = best_local_idx
                challenger_idx = j

                # Metrics for A
                obj_A = cand_obj[curr_idx]
                viol_A = cand_viol[curr_idx]
                is_feas_A = (
                    viol_A <= self.FEASIBILITY_TOLERANCE
                )  # Treat effectively 0 as feasible to handle float precision

                # Metrics for B
                obj_B = cand_obj[challenger_idx]
                viol_B = cand_viol[challenger_idx]
                is_feas_B = viol_B <= self.FEASIBILITY_TOLERANCE

                if is_feas_A and is_feas_B:
                    # Both feasible: better objective wins
                    if obj_B > obj_A:
                        best_local_idx = challenger_idx
                elif is_feas_A and not is_feas_B:
                    # A feasible, B infeasible: A stays
                    pass
                elif not is_feas_A and is_feas_B:
                    # A infeasible, B feasible: B wins
                    best_local_idx = challenger_idx
                else:
                    # Both infeasible: lower violation wins
                    if viol_B < viol_A:
                        best_local_idx = challenger_idx

            # Retrieve winner info
            winner_global_idx = candidates_idx[best_local_idx]
            selected[i] = population[winner_global_idx]

        return selected

    def _solution_to_dataframe(self, solution: np.ndarray) -> pd.DataFrame:
        """Convert solution matrix to dataframe with 'q' column (vectorized)."""
        df = self.all_df.copy()

        # Vectorized: map each row to (user_idx, item_idx) and lookup in solution
        user_indices = df["uid"].map(self.user_id_to_idx).values
        item_indices = df.groupby("uid").cumcount().values

        # Direct vectorized lookup
        df["q"] = solution[user_indices, item_indices]

        return df

    def _evaluate_groups(self, solution_df: pd.DataFrame, metrics: List[str]) -> Dict:
        """
        Evaluate metrics for overall, group 1 (advantaged), and group 2 (disadvantaged).
        Returns dict with 'overall', 'g1', 'g2' keys.
        """
        # Use pre-computed group sets (cached in __init__)
        g1_mask = solution_df["uid"].isin(self._g1_user_set)
        g2_mask = solution_df["uid"].isin(self._g2_user_set)

        g1_df = solution_df[g1_mask]
        g2_df = solution_df[g2_mask]

        return {
            "overall": evaluation_methods(solution_df, metrics),
            "g1": evaluation_methods(g1_df, metrics),
            "g2": evaluation_methods(g2_df, metrics),
        }

    def _format_metrics(self, eval_result: List[float]) -> str:
        """Format evaluation metrics as 'metric=value' string."""
        return " ".join(
            [f"{m}={eval_result[i]:.4f}" for i, m in enumerate(self.eval_metric_list)]
        )

    @staticmethod
    def _get_best_idx(objectives: np.ndarray, violations: np.ndarray) -> int:
        """
        Get index of best individual using Deb's feasibility rules.
        Primary sort: ascending violation, secondary sort: descending objective.
        """
        indices = np.lexsort((-objectives, violations))
        return indices[0]

    def _log_baseline_metrics(self) -> Tuple[np.ndarray, Dict]:
        """
        Evaluate and log baseline (greedy) metrics, calculate UGF and set epsilon.

        Returns:
            Tuple of (baseline_solution, baseline_eval)
        """
        print("=" * 70)
        print("Before optimization (baseline - top-K by score):")
        baseline_solution = self._create_greedy_individual()
        baseline_df = self._solution_to_dataframe(baseline_solution)

        # Evaluate overall and per-group metrics
        baseline_eval = self._evaluate_groups(baseline_df, self.eval_metric_list)

        # Format and print metrics
        metric_str = self._format_metrics(baseline_eval["overall"])
        print(f"  Overall: {metric_str}")
        self.logger.info(f"Before optimization overall scores           : {metric_str}")

        metric_str = self._format_metrics(baseline_eval["g1"])
        print(f"  Group 1 (advantaged): {metric_str}")
        self.logger.info(f"Before optimization group 1 (active) scores  : {metric_str}")

        metric_str = self._format_metrics(baseline_eval["g2"])
        print(f"  Group 2 (disadvantaged): {metric_str}")
        self.logger.info(f"Before optimization group 2 (inactive) scores: {metric_str}")

        # Calculate original UGF using batch method with single individual
        baseline_pop = baseline_solution[np.newaxis, :, :]  # Add batch dimension
        _, _, ugf_gaps, _ = self._calculate_fitness_batch(
            baseline_pop, current_epsilon=1.0
        )
        self.original_ugf = ugf_gaps[0]

        print(f"  UGF gap: {self.original_ugf:.4f} ({self.original_ugf * 100:.2f}%)")
        self.logger.info(
            f"Before optimization UGF ({self.fairness_metric}@{self.k}): {self.original_ugf:.4f} ({self.original_ugf * 100:.2f}%)"
        )

        # Calculate epsilon if auto
        if self._epsilon_input == "auto" or self._epsilon_input is None:
            self.epsilon = self.original_ugf / 2
            print(f"\nDynamic epsilon (1/2 of original gap): {self.epsilon:.4f}")
        else:
            self.epsilon = self._epsilon_input

        self.logger.info(f"Epsilon: {self.epsilon:.4f}")

        print("\nProgressive constraint tightening:")
        print(f"  Start epsilon: {self.original_ugf:.4f} (baseline feasible)")
        print(f"  Target epsilon: {self.epsilon:.4f}")

        return baseline_solution, baseline_eval

    def _log_final_results(
        self,
        final_solution: np.ndarray,
        target_epsilon: float,
        baseline_eval: Dict,
        best_fitness: float,
        cpu_time: float,
    ) -> Dict:
        """
        Evaluate and log final optimization results.

        Returns:
            Results dictionary with all metrics.
        """
        print("\n" + "=" * 70)
        print("After optimization (GA solution):")
        final_df = self._solution_to_dataframe(final_solution)

        # Evaluate overall and per-group metrics
        final_eval = self._evaluate_groups(final_df, self.eval_metric_list)

        metric_str = self._format_metrics(final_eval["overall"])
        print(f"  Overall: {metric_str}")
        self.logger.info(f"After optimization overall metric scores     : {metric_str}")

        metric_str = self._format_metrics(final_eval["g1"])
        print(f"  Group 1 (advantaged): {metric_str}")
        self.logger.info(f"After optimization group 1 (active) scores   : {metric_str}")

        metric_str = self._format_metrics(final_eval["g2"])
        print(f"  Group 2 (disadvantaged): {metric_str}")
        self.logger.info(f"After optimization group 2 (inactive) scores : {metric_str}")

        # Final UGF
        final_pop = final_solution[np.newaxis, :, :]
        _, _, final_ugf_arr, _ = self._calculate_fitness_batch(
            final_pop, target_epsilon
        )
        final_ugf = final_ugf_arr[0]

        print(f"  Final UGF gap: {final_ugf:.4f} ({final_ugf * 100:.2f}%)")
        self.logger.info(f"After optimization UGF: {final_ugf:.4f}")

        # UGF improvement
        ugf_reduction = self.original_ugf - final_ugf
        ugf_reduction_pct = (
            (ugf_reduction / self.original_ugf) * 100 if self.original_ugf > 0 else 0
        )
        print(
            f"\nUGF reduction: {ugf_reduction:.4f} ({ugf_reduction_pct:.1f}% improvement)"
        )
        self.logger.info(
            f"UGF reduction: {ugf_reduction:.4f} ({ugf_reduction_pct:.1f}%)"
        )

        # Check constraint satisfaction
        constraint_satisfied = final_ugf <= self.epsilon
        print(
            f"Fairness constraint (UGF <= {self.epsilon:.4f}): {'SATISFIED' if constraint_satisfied else 'VIOLATED'}"
        )
        self.logger.info(f"Constraint satisfied: {constraint_satisfied}")

        return {
            "baseline_metrics": baseline_eval["overall"],
            "final_metrics": final_eval["overall"],
            "original_ugf": self.original_ugf,
            "final_ugf": final_ugf,
            "epsilon": self.epsilon,
            "constraint_satisfied": constraint_satisfied,
            "cpu_time": cpu_time,
            "best_fitness": best_fitness,
        }

    def train(self) -> Dict:
        """Run GA optimization with vectorized operations."""
        self.logger.info(
            f"GA Optimizer | Model:{self.model_name} | Dataset:{self.dataset_name} | "
            f"Group:{self.group_name} | K={self.k} | Fairness_metric={self.fairness_metric}"
        )
        self.logger.info(
            f"GA Parameters | Pop:{self.population_size} | Gen:{self.generations} | "
            f"Mut:{self.mutation_rate} | Cross:{self.crossover_rate}"
        )
        if self.adaptive_penalty:
            self.logger.info(
                f"Adaptive Penalty | beta1:{self.penalty_beta1} | beta2:{self.penalty_beta2} | k:{self.penalty_history_k}"
            )

        # Evaluate baseline and set epsilon
        baseline_solution, baseline_eval = self._log_baseline_metrics()

        # Initialize population
        print("\nStarting GA optimization (vectorized)...")
        start_time = time.perf_counter()

        # Create initial population: greedy + perturbed greedy
        population = self._create_initial_population(self.population_size)

        # Initial evaluation
        # Start slightly tighter than original to force immediate movement
        start_epsilon = self.original_ugf * self.START_EPSILON_FACTOR
        target_epsilon = self.epsilon
        objectives, violations, ugf_gaps, signed_ugf = self._calculate_fitness_batch(
            population, start_epsilon
        )

        best_idx = self._get_best_idx(objectives, violations)
        best_fitness = objectives[best_idx]
        best_solution = population[best_idx].copy()
        best_ugf = ugf_gaps[best_idx]
        best_viol = violations[best_idx]

        print(
            f"\nInitial population: best_obj={best_fitness:.2f}, best_viol={best_viol:.4f}, UGF={best_ugf:.4f}"
        )

        # Track best feasible solution
        best_feasible_solution = None
        best_feasible_fitness = float("-inf")

        # Adaptive penalty: track feasibility history of best individual per generation
        feasibility_history = []  # True if best individual was feasible, False otherwise

        # Pure Adaptive Penalty (Bean & Hadj-Alouane method)
        # Start at the loose constraint and adapt purely based on feasibility feedback
        current_epsilon = start_epsilon

        # Evolution loop
        for gen in range(self.generations):
            # Adaptive penalty adjustment (Bean & Hadj-Alouane method)
            # No generation-based schedule - purely feedback-driven
            if (
                self.adaptive_penalty
                and len(feasibility_history) >= self.penalty_history_k
            ):
                recent_history = feasibility_history[-self.penalty_history_k :]
                if all(recent_history):  # All recent best were feasible
                    # Tighten epsilon (make constraint harder, push toward target)
                    current_epsilon = current_epsilon / self.penalty_beta1
                elif not any(recent_history):  # All recent best were infeasible
                    # Relax epsilon (make constraint easier to satisfy)
                    current_epsilon = current_epsilon * self.penalty_beta2

                # Bound epsilon to valid range
                current_epsilon = max(
                    target_epsilon, min(current_epsilon, start_epsilon)
                )

            # Elitism: keep top individuals
            # Use same Deb's sort logic
            sorted_indices = np.lexsort((-objectives, violations))
            elite_indices = sorted_indices[: self.elitism_count]
            elites = population[elite_indices].copy()

            # Two-parent uniform crossover
            n_offspring = self.population_size - self.elitism_count
            offspring = self._two_parent_crossover(
                population, objectives, violations, n_offspring
            )

            # Mutation with Repair Bias
            # Determine current bias direction from population average
            # (Use previous generation's evaluation to guide mutation)
            # A positive mean means G1 is generally advantaged -> Suppress G1, Boost G2
            avg_bias = np.mean(signed_ugf)

            # Apply mutation with fixed rate
            offspring = self._mutate_batch(
                offspring, bias_dir=avg_bias, current_rate=self.mutation_rate
            )

            # New population
            population = np.concatenate([elites, offspring], axis=0)

            # Evaluate
            objectives, violations, ugf_gaps, signed_ugf = (
                self._calculate_fitness_batch(population, current_epsilon)
            )

            # Track best
            gen_best_idx = self._get_best_idx(objectives, violations)
            gen_best_fitness = objectives[gen_best_idx]
            gen_best_ugf = ugf_gaps[gen_best_idx]
            gen_best_viol = violations[gen_best_idx]

            # Update feasibility history for adaptive penalty
            gen_best_is_feasible = gen_best_viol <= self.FEASIBILITY_TOLERANCE
            feasibility_history.append(gen_best_is_feasible)

            if gen_best_fitness > best_fitness:
                best_solution = population[gen_best_idx].copy()
                best_fitness = gen_best_fitness
                best_ugf = gen_best_ugf

            target_violations = np.maximum(0, ugf_gaps - target_epsilon)
            feasible_mask = target_violations <= 1e-6

            if feasible_mask.any():
                feasible_objs = objectives.copy()
                feasible_objs[~feasible_mask] = float("-inf")
                current_best_feasible_idx = np.argmax(feasible_objs)

                if feasible_objs[current_best_feasible_idx] > best_feasible_fitness:
                    best_feasible_fitness = feasible_objs[current_best_feasible_idx]
                    best_feasible_solution = population[
                        current_best_feasible_idx
                    ].copy()

            # Progress logging
            if (gen + 1) % 10 == 0:
                adapt_status = ""
                if (
                    self.adaptive_penalty
                    and len(feasibility_history) >= self.penalty_history_k
                ):
                    recent = feasibility_history[-self.penalty_history_k :]
                    if all(recent):
                        adapt_status = " [RELAX]"
                    elif not any(recent):
                        adapt_status = " [TIGHT]"
                print(
                    f"  Gen {gen + 1}: eps={current_epsilon:.4f}{adapt_status}, mut={self.mutation_rate:.3f}, "
                    f"best_obj={gen_best_fitness:.2f}, UGF={gen_best_ugf:.4f}, viol={gen_best_viol:.4f}"
                )

            # Early stopping check
            if self.early_stopping and best_feasible_solution is not None:
                print(
                    f"\n  *** Early stop at gen {gen + 1}: feasible solution found "
                    f"(UGF={gen_best_ugf:.4f} <= target={target_epsilon:.4f}) ***"
                )
                self.logger.info(
                    f"Early stop at generation {gen + 1}: first feasible solution found"
                )
                break

        cpu_time = time.perf_counter() - start_time
        print(f"\nGA optimization completed in {cpu_time:.2f} seconds")
        self.logger.info(f"CPU time: {cpu_time:.2f} seconds")

        # Use feasible solution if available (Priority 1)
        if best_feasible_solution is not None:
            final_solution = best_feasible_solution
            print("Using best FEASIBLE solution (constraint satisfied)")
        else:
            # If no feasible solution found, use the best available (min violation)
            print(
                "Using best solution from FINAL generation (constraint violated, minimization violation)"
            )
            final_solution = best_solution

        # Log final results and return metrics
        return self._log_final_results(
            final_solution, target_epsilon, baseline_eval, best_fitness, cpu_time
        )


if __name__ == "__main__":
    """
    Comprehensive evaluation: runs all dataset/model/grouping combinations
    to replicate the paper's experimental setup.
    """
    import os

    ############### Configuration ###########
    epsilon = "auto"  # Dynamic epsilon (paper methodology)
    dataset_folder = "../dataset"

    # All available datasets
    datasets = ["5Beauty-rand", "5Grocery-rand", "5Health-rand"]

    # All available models (must have corresponding *_rank.csv files)
    models = ["NCF", "biasedMF"]

    # Grouping methods: (group_name, group_1_suffix, group_2_suffix)
    grouping_methods = [
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
        )
    ]

    ga_seed = 42

    metrics = ["ndcg", "f1"]
    topK = ["10"]
    metrics_list = [metric + "@" + k for metric in metrics for k in topK]

    # Results collection
    all_results = []

    # Create results directory
    results_base_dir = "../results"
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)

    ############### Run all combinations ###########
    total_runs = len(datasets) * len(models) * len(grouping_methods)
    current_run = 0

    for dataset_name in datasets:
        for model_name in models:
            for group_name, group_1_file, group_2_file in grouping_methods:
                current_run += 1
                print("\n" + "=" * 80)
                print(
                    f"RUN {current_run}/{total_runs}: {dataset_name} | {model_name} | {group_name}"
                )
                print("=" * 80)

                try:
                    # Setup paths
                    data_path = os.path.join(dataset_folder, dataset_name)
                    rank_file = model_name + "_rank.csv"

                    # Check if rank file exists
                    if not os.path.exists(os.path.join(data_path, rank_file)):
                        print(f"  SKIPPED: {rank_file} not found")
                        continue

                    # Setup logging
                    logger_dir = os.path.join(results_base_dir, model_name, "ga")
                    if not os.path.exists(logger_dir):
                        os.makedirs(logger_dir)
                    logger_file = f"ga_{model_name}_{dataset_name}_{group_name}.log"
                    logger_path = os.path.join(logger_dir, logger_file)

                    # Load data
                    dl = DataLoader(
                        data_path,
                        rank_file=rank_file,
                        group_1_file=group_1_file,
                        group_2_file=group_2_file,
                    )

                    logger = create_logger(
                        name=f"ga_logger_{dataset_name}_{model_name}_{group_name}",
                        path=logger_path,
                    )

                    # Run GA optimizer
                    ga = GAOptimizer(
                        data_loader=dl,
                        k=10,
                        eval_metric_list=metrics_list,
                        fairness_metric="f1",
                        epsilon=epsilon,
                        logger=logger,
                        model_name=model_name,
                        group_name=group_name,
                        seed=ga_seed,
                    )

                    # Get results
                    results = ga.train()

                    # Store results for summary
                    all_results.append(
                        {
                            "Dataset": dataset_name.replace("5", "").replace(
                                "-rand", ""
                            ),
                            "Model": model_name,
                            "Grouping": group_name,
                            "Epsilon": ga.epsilon,
                            "Final_UGF": results["final_ugf"],
                            "CPU_Time": results["cpu_time"],
                            "Status": "Completed",
                        }
                    )

                    print("  ✓ Completed successfully")

                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
                    all_results.append(
                        {
                            "Dataset": dataset_name.replace("5", "").replace(
                                "-rand", ""
                            ),
                            "Model": model_name,
                            "Grouping": group_name,
                            "Epsilon": "N/A",
                            "Final_UGF": "N/A",
                            "CPU_Time": "N/A",
                            "Status": f"Error: {str(e)}",
                        }
                    )

    ############### Summary ###########
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)

    # Print summary table
    print(
        f"\n{'Dataset':<12} {'Model':<12} {'Grouping':<12} {'Epsilon':<10} {'Final_UGF':<12} {'Time(s)':<10} {'Status':<20}"
    )
    print("-" * 90)
    for r in all_results:
        eps_str = (
            f"{r['Epsilon']:.4f}" if isinstance(r["Epsilon"], float) else r["Epsilon"]
        )
        ugf_str = (
            f"{r['Final_UGF']:.4f}"
            if isinstance(r["Final_UGF"], float)
            else r["Final_UGF"]
        )
        time_str = (
            f"{r['CPU_Time']:.2f}"
            if isinstance(r["CPU_Time"], float)
            else r["CPU_Time"]
        )
        print(
            f"{r['Dataset']:<12} {r['Model']:<12} {r['Grouping']:<12} {eps_str:<10} {ugf_str:<12} {time_str:<10} {r['Status']:<20}"
        )

    print(f"\nIndividual logs saved to: {results_base_dir}/<model_name>/ga/")
    print("\nAll experiments completed!")
