"""
Genetic Algorithm Optimizer with Penalty-Based Fitness Function
for User-oriented Fairness in Recommendation.

OPTIMIZED VERSION: Uses vectorized NumPy operations for 10-100x speedup.

This module implements a GA that solves the same problem as the MILP solver:
- maximize sum of preference scores (S_ij * W_ij)
- subject to fairness constraint: |UGF(group1, group2)| <= epsilon
- subject to selection constraint: exactly K items per user
"""

import numpy as np
import pandas as pd
import time
import random
from typing import List, Tuple, Dict, Optional
from data_loader import DataLoader
from utils.tools import create_logger, evaluation_methods


class GAOptimizer:
    """
    Genetic Algorithm optimizer for fairness-aware recommendation re-ranking.
    Uses vectorized NumPy operations for performance.
    """

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
        # GA parameters
        population_size: int = 100,
        generations: int = 200,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_count: int = 5,
        penalty_lambda: float = None,
        seed: int = None,
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

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if logger is None:
            self.logger = create_logger()
        else:
            self.logger = logger

        # Prepare vectorized data structures
        self._prepare_vectorized_data()

    def _prepare_vectorized_data(self):
        """Convert data to vectorized format for fast computation."""
        self.all_df = self.data_loader.rank_df.copy(deep=True)
        self.g1_df = self.data_loader.g1_df.copy(deep=True)
        self.g2_df = self.data_loader.g2_df.copy(deep=True)

        # Get unique users
        self.user_ids = self.all_df["uid"].unique()
        self.n_users = len(self.user_ids)
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}

        # Determine max items per user (handle variable-length candidate lists)
        user_counts = self.all_df.groupby("uid").size()
        self.n_items = user_counts.max()  # Use max to handle variable lengths
        self.items_per_user = user_counts.to_dict()  # Store actual count per user
        
        # Build score and label matrices: shape (n_users, n_items)
        self.scores_matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        self.labels_matrix = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        
        for uid in self.user_ids:
            idx = self.user_id_to_idx[uid]
            user_df = self.all_df[self.all_df["uid"] == uid]
            n = len(user_df)
            self.scores_matrix[idx, :n] = user_df["score"].values
            self.labels_matrix[idx, :n] = user_df["label"].values

        # Pre-compute total relevant items per user
        self.total_relevant = self.labels_matrix.sum(axis=1)  # shape (n_users,)
        
        # Group masks: boolean arrays indicating group membership
        g1_users = set(self.g1_df["uid"].unique())
        g2_users = set(self.g2_df["uid"].unique())
        
        self.g1_mask = np.array([uid in g1_users for uid in self.user_ids])
        self.g2_mask = np.array([uid in g2_users for uid in self.user_ids])
        
        # Users with relevant items (for metric calculation)
        self.has_relevant = self.total_relevant > 0

        # Pre-compute denominators for F1
        self.f1_denominator = self.total_relevant + self.k

        self.n_g1 = self.g1_mask.sum()
        self.n_g2 = self.g2_mask.sum()
        
        print(f"Vectorized data: {self.n_users} users, {self.n_items} items per user")
        print(f"Group 1: {self.n_g1} users, Group 2: {self.n_g2} users")

    def _calculate_fitness_batch(self, population: np.ndarray, current_epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
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
        selected_labels = population * self.labels_matrix  # (pop_size, n_users, n_items)
        selected_relevant = selected_labels.sum(axis=2)  # (pop_size, n_users)
        
        # F1 metric per user: 2 * selected_relevant / (total_relevant + k)
        # Only for users with relevant items
        with np.errstate(divide='ignore', invalid='ignore'):
            f1_per_user = 2 * selected_relevant / self.f1_denominator  # (pop_size, n_users)
            f1_per_user = np.nan_to_num(f1_per_user, 0)
        
        # Mask users without relevant items
        f1_per_user[:, ~self.has_relevant] = 0
        
        # Group averages
        g1_has_relevant = self.g1_mask & self.has_relevant
        g2_has_relevant = self.g2_mask & self.has_relevant
        
        g1_avg = f1_per_user[:, g1_has_relevant].mean(axis=1) if g1_has_relevant.sum() > 0 else np.zeros(pop_size)
        g2_avg = f1_per_user[:, g2_has_relevant].mean(axis=1) if g2_has_relevant.sum() > 0 else np.zeros(pop_size)
        
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
        init_mutation_rate = 0.2
        mutate_mask = np.random.random((size, self.n_users)) < init_mutation_rate
        
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
            top_k = np.argsort(self.scores_matrix[u])[-self.k:]
            individual[u, top_k] = 1
            
        return individual

    def _crossover_batch(self, parents1: np.ndarray, parents2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform crossover at user level for batch of parents.
        
        Args:
            parents1, parents2: shape (batch_size, n_users, n_items)
            
        Returns:
            children1, children2: same shape
        """
        batch_size = parents1.shape[0]
        
        # Decide which users come from which parent
        crossover_mask = np.random.random((batch_size, self.n_users, 1)) < 0.5
        
        # Apply crossover with probability
        do_crossover = np.random.random(batch_size) < self.crossover_rate
        do_crossover = do_crossover[:, np.newaxis, np.newaxis]
        
        children1 = np.where(do_crossover & crossover_mask, parents2, parents1)
        children2 = np.where(do_crossover & crossover_mask, parents1, parents2)
        
        return children1, children2

    def _mutate_batch(self, population: np.ndarray, bias_dir: float = 0.0) -> np.ndarray:
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
        
        # Decide which users to mutate for each individual
        mutate_mask = np.random.random((pop_size, self.n_users)) < self.mutation_rate
        
        # Repair logic thresholds
        apply_repair = abs(bias_dir) > (self.epsilon if self.epsilon else 0.05)
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
                                if is_g1: strategy = "suppress"
                                elif is_g2: strategy = "boost"
                            else:
                                if is_g1: strategy = "boost"
                                elif is_g2: strategy = "suppress"
                        
                        # Fallback for balanced/random if not repairing or neutral group
                        if strategy == "random" and np.random.random() < 0.5:
                            strategy = "boost" # Default Smart Swap is a boost

                        if strategy == "boost":
                            # --- Boost (Greedy) ---
                            # Remove Worst Selected -> Add Best Unselected
                            cand_remove = np.random.choice(selected, size=min(3, len(selected)), replace=False)
                            scores_remove = self.scores_matrix[u, cand_remove]
                            to_remove = cand_remove[np.argmin(scores_remove)]
                            
                            cand_add = np.random.choice(unselected, size=min(5, len(unselected)), replace=False)
                            scores_add = self.scores_matrix[u, cand_add]
                            to_add = cand_add[np.argmax(scores_add)]
                            
                        elif strategy == "suppress":
                             # --- Suppress (Altruistic) ---
                             # Remove Best Selected -> Add Worst Unselected
                             # (Sacrifice own score to lower group average)
                            cand_remove = np.random.choice(selected, size=min(3, len(selected)), replace=False)
                            scores_remove = self.scores_matrix[u, cand_remove]
                            to_remove = cand_remove[np.argmax(scores_remove)] # Remove BEST
                            
                            cand_add = np.random.choice(unselected, size=min(5, len(unselected)), replace=False)
                            scores_add = self.scores_matrix[u, cand_add]
                            to_add = cand_add[np.argmin(scores_add)] # Add WORST
                        else:
                            # --- Random Swap ---
                            to_remove = np.random.choice(selected)
                            to_add = np.random.choice(unselected)
                        
                        # Perform swap
                        mutated[i, u, to_remove] = 0
                        mutated[i, u, to_add] = 1
                        
        return mutated

    def _tournament_selection(self, population: np.ndarray, objectives: np.ndarray, 
                              violations: np.ndarray, n_select: int, tournament_size: int = 5) -> np.ndarray:
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
            candidates_idx = np.random.choice(pop_size, size=tournament_size, replace=False)
            
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
                is_feas_A = (viol_A <= 1e-9) # Treat effectively 0 as feasible to handle float precision
                
                # Metrics for B
                obj_B = cand_obj[challenger_idx]
                viol_B = cand_viol[challenger_idx]
                is_feas_B = (viol_B <= 1e-9)
                
                # Apply Rules
                scaler = 1.0 # Maximization
                
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
        """Convert solution matrix to dataframe with 'q' column."""
        df = self.all_df.copy()
        df["q"] = 0
        
        for uid in self.user_ids:
            u_idx = self.user_id_to_idx[uid]
            user_mask = df["uid"] == uid
            df.loc[user_mask, "q"] = solution[u_idx, :user_mask.sum()]
            
        return df

    def _evaluate_groups(self, solution_df: pd.DataFrame, metrics: List[str]) -> Dict:
        """
        Evaluate metrics for overall, group 1 (advantaged), and group 2 (disadvantaged).
        Returns dict with 'overall', 'g1', 'g2' keys.
        """
        # Get group user sets
        g1_users = set(self.g1_df["uid"].unique())
        g2_users = set(self.g2_df["uid"].unique())
        
        # Split solution_df by group
        g1_df = solution_df[solution_df["uid"].isin(g1_users)]
        g2_df = solution_df[solution_df["uid"].isin(g2_users)]
        
        return {
            "overall": evaluation_methods(solution_df, metrics),
            "g1": evaluation_methods(g1_df, metrics),
            "g2": evaluation_methods(g2_df, metrics),
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

        # Print original metrics (overall and per group)
        print("=" * 70)
        print("Before optimization (baseline - top-K by score):")
        baseline_solution = self._create_greedy_individual()
        baseline_df = self._solution_to_dataframe(baseline_solution)

        # Evaluate overall and per-group metrics
        baseline_eval = self._evaluate_groups(baseline_df, self.eval_metric_list)
        
        # Format and print metrics
        metric_str = " ".join([f"{m}={baseline_eval['overall'][i]:.4f}" for i, m in enumerate(self.eval_metric_list)])
        print(f"  Overall: {metric_str}")
        self.logger.info(f"Before optimization overall scores           : {metric_str}")
        
        metric_str = " ".join([f"{m}={baseline_eval['g1'][i]:.4f}" for i, m in enumerate(self.eval_metric_list)])
        print(f"  Group 1 (advantaged): {metric_str}")
        self.logger.info(f"Before optimization group 1 (active) scores  : {metric_str}")
        
        metric_str = " ".join([f"{m}={baseline_eval['g2'][i]:.4f}" for i, m in enumerate(self.eval_metric_list)])
        print(f"  Group 2 (disadvantaged): {metric_str}")
        self.logger.info(f"Before optimization group 2 (inactive) scores: {metric_str}")

        # Calculate original UGF using batch method with single individual
        baseline_pop = baseline_solution[np.newaxis, :, :]  # Add batch dimension
        _, _, ugf_gaps, _ = self._calculate_fitness_batch(baseline_pop, current_epsilon=1.0)
        self.original_ugf = ugf_gaps[0]
        
        print(f"  UGF gap: {self.original_ugf:.4f} ({self.original_ugf * 100:.2f}%)")
        self.logger.info(f"Before optimization UGF ({self.fairness_metric}@{self.k}): {self.original_ugf:.4f} ({self.original_ugf * 100:.2f}%)")

        # Calculate epsilon if auto
        if self._epsilon_input == "auto" or self._epsilon_input is None:
            self.epsilon = self.original_ugf / 4
            print(f"\nDynamic epsilon (1/4 of original gap): {self.epsilon:.4f}")
        else:
            self.epsilon = self._epsilon_input

        self.logger.info(f"Epsilon: {self.epsilon:.4f}")
        
        print(f"\nProgressive constraint tightening:")
        print(f"  Start epsilon: {self.original_ugf:.4f} (baseline feasible)")
        print(f"  Target epsilon: {self.epsilon:.4f}")

        # Initialize population
        print(f"\nStarting GA optimization (vectorized)...")
        start_time = time.time()

        # Create initial population: greedy + perturbed greedy
        population = self._create_initial_population(self.population_size)

        # Initial evaluation
        # Start slightly tighter than original to force immediate movement
        start_epsilon = self.original_ugf * 0.99
        target_epsilon = self.epsilon
        # Initial evaluation
        # Start slightly tighter than original to force immediate movement
        start_epsilon = self.original_ugf * 0.99
        target_epsilon = self.epsilon
        objectives, violations, ugf_gaps, signed_ugf = self._calculate_fitness_batch(population, start_epsilon)

        def get_best_idx(objs, viols):
            keys = (-objs, viols)
            indices = np.lexsort((-objs, viols))
            return indices[0]

        best_idx = get_best_idx(objectives, violations)
        best_fitness = objectives[best_idx]
        best_solution = population[best_idx].copy()
        best_ugf = ugf_gaps[best_idx]
        best_viol = violations[best_idx]

        print(f"\nInitial population: best_obj={best_fitness:.2f}, best_viol={best_viol:.4f}, UGF={best_ugf:.4f}")

        # Track best feasible solution
        best_feasible_solution = None
        best_feasible_fitness = float('-inf')

        # Evolution loop
        for gen in range(self.generations):
            # Progressive constraint tightening
            progress = gen / max(1, self.generations - 1)
            current_epsilon = start_epsilon + (target_epsilon - start_epsilon) * progress

            # Elitism: keep top individuals
            # Use same Deb's sort logic
            sorted_indices = np.lexsort((-objectives, violations))
            elite_indices = sorted_indices[:self.elitism_count]
            elites = population[elite_indices].copy()

            # Selection
            n_offspring = self.population_size - self.elitism_count
            n_pairs = (n_offspring + 1) // 2
            
            parents1 = self._tournament_selection(population, objectives, violations, n_pairs)
            parents2 = self._tournament_selection(population, objectives, violations, n_pairs)

            # Crossover
            children1, children2 = self._crossover_batch(parents1, parents2)
            offspring = np.concatenate([children1, children2], axis=0)[:n_offspring]

            # Mutation
            # Determine current bias direction from population average
            # (Use previous generation's evaluation to guide mutation)
            # A positive mean means G1 is generally advantaged -> Suppress G1, Boost G2
            avg_bias = np.mean(signed_ugf)
            
            # Mutation with Repair Bias
            offspring = self._mutate_batch(offspring, bias_dir=avg_bias)

            # New population
            population = np.concatenate([elites, offspring], axis=0)

            # Evaluate
            objectives, violations, ugf_gaps, signed_ugf = self._calculate_fitness_batch(population, current_epsilon)

            # Track best
            gen_best_idx = get_best_idx(objectives, violations)
            gen_best_fitness = objectives[gen_best_idx]
            gen_best_ugf = ugf_gaps[gen_best_idx]
            gen_best_viol = violations[gen_best_idx]

            if gen_best_fitness > best_fitness:
                 best_solution = population[gen_best_idx].copy()
                 best_fitness = gen_best_fitness
                 best_ugf = gen_best_ugf

            target_violations = np.maximum(0, ugf_gaps - target_epsilon)
            feasible_mask = target_violations <= 1e-6
            
            if feasible_mask.any():
                feasible_objs = objectives.copy()
                feasible_objs[~feasible_mask] = float('-inf')
                current_best_feasible_idx = np.argmax(feasible_objs)
                
                if feasible_objs[current_best_feasible_idx] > best_feasible_fitness:
                    best_feasible_fitness = feasible_objs[current_best_feasible_idx]
                    best_feasible_solution = population[current_best_feasible_idx].copy()

            # Progress logging
            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"  Gen {gen + 1}: eps={current_epsilon:.4f}, best_obj={gen_best_fitness:.2f}, "
                      f"UGF={gen_best_ugf:.4f}, viol={gen_best_viol:.4f}")

        cpu_time = time.time() - start_time
        print(f"\nGA optimization completed in {cpu_time:.2f} seconds")
        self.logger.info(f"CPU time: {cpu_time:.2f} seconds")

        # Use feasible solution if available (Priority 1)
        if best_feasible_solution is not None:
            final_solution = best_feasible_solution
            print("Using best FEASIBLE solution (constraint satisfied)")
        else:
            # If no feasible solution found, use the best available (min violation)
            print("Using best solution from FINAL generation (constraint violated, minimization violation)")
            final_solution = best_solution

        # Final results with group breakdown
        print("\n" + "=" * 70)
        print("After optimization (GA solution):")
        final_df = self._solution_to_dataframe(final_solution)

        # Evaluate overall and per-group metrics
        final_eval = self._evaluate_groups(final_df, self.eval_metric_list)
        
        metric_str = " ".join([f"{m}={final_eval['overall'][i]:.4f}" for i, m in enumerate(self.eval_metric_list)])
        print(f"  Overall: {metric_str}")
        self.logger.info(f"After optimization overall metric scores     : {metric_str}")
        
        metric_str = " ".join([f"{m}={final_eval['g1'][i]:.4f}" for i, m in enumerate(self.eval_metric_list)])
        print(f"  Group 1 (advantaged): {metric_str}")
        self.logger.info(f"After optimization group 1 (active) scores   : {metric_str}")
        
        metric_str = " ".join([f"{m}={final_eval['g2'][i]:.4f}" for i, m in enumerate(self.eval_metric_list)])
        print(f"  Group 2 (disadvantaged): {metric_str}")
        self.logger.info(f"After optimization group 2 (inactive) scores : {metric_str}")

        # Final UGF
        final_pop = final_solution[np.newaxis, :, :]
        _, _, final_ugf_arr, _ = self._calculate_fitness_batch(final_pop, target_epsilon)
        final_ugf = final_ugf_arr[0]
        
        print(f"  Final UGF gap: {final_ugf:.4f} ({final_ugf * 100:.2f}%)")
        self.logger.info(f"After optimization UGF: {final_ugf:.4f}")

        # UGF improvement
        ugf_reduction = self.original_ugf - final_ugf
        ugf_reduction_pct = (ugf_reduction / self.original_ugf) * 100 if self.original_ugf > 0 else 0
        print(f"\nUGF reduction: {ugf_reduction:.4f} ({ugf_reduction_pct:.1f}% improvement)")
        self.logger.info(f"UGF reduction: {ugf_reduction:.4f} ({ugf_reduction_pct:.1f}%)")

        # Check constraint satisfaction
        constraint_satisfied = final_ugf <= self.epsilon
        print(f"Fairness constraint (UGF <= {self.epsilon:.4f}): {'SATISFIED' if constraint_satisfied else 'VIOLATED'}")
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


if __name__ == "__main__":
    """Test GA optimizer on a sample dataset."""
    import os

    # Configuration
    dataset_folder = "../dataset"
    dataset_name = "5Beauty-rand"
    model_name = "NCF"
    group_name = "0.05_count"

    data_path = os.path.join(dataset_folder, dataset_name)
    rank_file = f"{model_name}_rank.csv"
    group_1_file = f"{group_name}_active_test_ratings.txt"
    group_2_file = f"{group_name}_inactive_test_ratings.txt"

    # Results directory
    results_dir = "../results/ga"
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(
        results_dir,
        f"GA_{model_name}_{dataset_name}_{group_name}.log"
    )

    # Load data
    print(f"Loading data from {data_path}...")
    dl = DataLoader(
        data_path,
        rank_file=rank_file,
        group_1_file=group_1_file,
        group_2_file=group_2_file,
    )

    logger = create_logger(name="ga_logger", path=log_file)

    # Run GA optimizer
    ga = GAOptimizer(
        data_loader=dl,
        k=10,
        eval_metric_list=["ndcg@10", "f1@10"],
        fairness_metric="f1",
        epsilon="auto",
        logger=logger,
        model_name=model_name,
        group_name=group_name,
        population_size=50,
        generations=200,
        mutation_rate=0.1,  # Lower mutation rate to preserve quality
        crossover_rate=0.8,
        elitism_count=5,
        penalty_lambda=None,
        seed=42,
    )

    results = ga.train()
    print(f"\nResults logged to: {log_file}")
