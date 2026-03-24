"""
MA Ablation Study
=================

Compares alternative mutation and crossover operators against the baseline GA
(swap mutation + uniform crossover) across all 18 experimental configurations:
  3 datasets × 2 models × 3 groupings

Ablation variants (4 combinations):
  - insertion mutation  + ordered crossover (OX)
  - insertion mutation  + partially-mapped crossover (PMX)
  - inversion mutation  + ordered crossover (OX)
  - inversion mutation  + partially-mapped crossover (PMX)

Baseline results are parsed from existing log files in:
  results/NCF/ga/   and   results/biasedMF/ga/

Usage:
    # Full study (72 runs)
    python ma_ablation_study.py

    # Smoke test — one config, 5 generations, first variant only
    python ma_ablation_study.py --smoke-test

    # Specific subset
    python ma_ablation_study.py --dataset 5Beauty-rand --model NCF --grouping max_0.05

    # Custom generations / output dir
    python ma_ablation_study.py --generations 200 --output ../results/ablation
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add src directory to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import DataLoader
from ma_optimizer import MAOptimizer
from utils.tools import create_logger

# ---------------------------------------------------------------------------
# Operator implementations as a subclass of MAOptimizer
# ---------------------------------------------------------------------------


class AblationMAOptimizer(MAOptimizer):
    """
    Extends MAOptimizer with additional mutation and crossover operators
    for ablation study purposes.

    All new operators preserve the invariants of the base MA:
      - Exactly K items selected per user.
      - Only valid (non-padded) item indices are touched.
    """

    # ------------------------------------------------------------------
    # Mutation operators
    # ------------------------------------------------------------------

    def _insertion_mutation(
        self,
        population: np.ndarray,
        bias_dir: float = 0.0,
        current_rate: float = None,
    ) -> np.ndarray:
        """
        Insertion Mutation with optional repair bias.

        For each user chosen for mutation, randomly selects one item from the
        selected set and re-inserts it at a different random position in the
        ordered list of selected items.  The total number of selected items K
        is preserved because the operation only *reorders* the selection,
        effectively swapping its rank position (useful in ranking-aware
        interpretations) — but here in bitmap space we implement it as:

            1. Pick a random selected item  (to_remove index).
            2. If repair bias applies, pick the replacement from unselected
               items using the same suppress/boost heuristic as swap mutation.
            3. Otherwise, remove a random selected item and insert (add) a
               random unselected item at its place.

        This is equivalent to a "random resection" that gives the operator a
        greedy / anti-greedy flavour under repair mode.
        """
        pop_size = population.shape[0]
        mutated = population.copy()

        rate = current_rate if current_rate is not None else self.mutation_rate
        mutate_mask = np.random.random((pop_size, self.n_users)) < rate

        apply_repair = abs(bias_dir) > self.epsilon

        for i in range(pop_size):
            for u in range(self.n_users):
                if mutate_mask[i, u]:
                    n_valid = self.items_per_user_arr[u]
                    user_slice = mutated[i, u, :n_valid]
                    selected = np.where(user_slice == 1)[0]
                    unselected = np.where(user_slice == 0)[0]

                    if len(selected) == 0 or len(unselected) == 0:
                        continue

                    # Determine strategy
                    strategy = "random"
                    if apply_repair:
                        is_g1 = self.g1_mask[u]
                        is_g2 = self.g2_mask[u]
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

                    if strategy == "random" and np.random.random() < 0.5:
                        strategy = "boost"

                    if strategy == "boost":
                        # Remove worst selected → insert best unselected
                        cand_remove = np.random.choice(
                            selected, size=min(3, len(selected)), replace=False
                        )
                        to_remove = cand_remove[
                            np.argmin(self.scores_matrix[u, cand_remove])
                        ]
                        cand_add = np.random.choice(
                            unselected, size=min(5, len(unselected)), replace=False
                        )
                        to_add = cand_add[
                            np.argmax(self.scores_matrix[u, cand_add])
                        ]
                    elif strategy == "suppress":
                        # Remove best selected → insert worst unselected
                        cand_remove = np.random.choice(
                            selected, size=min(3, len(selected)), replace=False
                        )
                        to_remove = cand_remove[
                            np.argmax(self.scores_matrix[u, cand_remove])
                        ]
                        cand_add = np.random.choice(
                            unselected, size=min(5, len(unselected)), replace=False
                        )
                        to_add = cand_add[
                            np.argmin(self.scores_matrix[u, cand_add])
                        ]
                    else:
                        # Pure insertion: pick a random selected item and
                        # "insert" an unselected item in its place (random swap)
                        to_remove = np.random.choice(selected)
                        to_add = np.random.choice(unselected)

                    mutated[i, u, to_remove] = 0
                    mutated[i, u, to_add] = 1

        return mutated

    def _inversion_mutation(
        self,
        population: np.ndarray,
        bias_dir: float = 0.0,
        current_rate: float = None,
    ) -> np.ndarray:
        """
        Inversion Mutation with optional repair bias.

        For each user chosen for mutation, picks two random cut points in the
        *ordered* list of selected item indices and reverses the sub-sequence
        between them.  This changes the relative ordering of items in the
        bitmap but keeps the same set of K selected items.

        When repair bias is active, with 50 % probability it falls back to a
        greedy swap (same as insertion mutation boost/suppress) so the operator
        can still drive fairness improvements.
        """
        pop_size = population.shape[0]
        mutated = population.copy()

        rate = current_rate if current_rate is not None else self.mutation_rate
        mutate_mask = np.random.random((pop_size, self.n_users)) < rate

        apply_repair = abs(bias_dir) > self.epsilon

        for i in range(pop_size):
            for u in range(self.n_users):
                if mutate_mask[i, u]:
                    n_valid = self.items_per_user_arr[u]
                    user_slice = mutated[i, u, :n_valid]
                    selected = np.where(user_slice == 1)[0]
                    unselected = np.where(user_slice == 0)[0]

                    if len(selected) < 2:
                        continue

                    # Under repair mode, use a greedy swap with 50 % probability
                    if apply_repair and np.random.random() < 0.5:
                        if len(unselected) == 0:
                            continue
                        is_g1 = self.g1_mask[u]
                        is_g2 = self.g2_mask[u]

                        if bias_dir > 0:
                            strategy = "suppress" if is_g1 else ("boost" if is_g2 else "random")
                        else:
                            strategy = "boost" if is_g1 else ("suppress" if is_g2 else "random")

                        if strategy == "boost":
                            cand_remove = np.random.choice(
                                selected, size=min(3, len(selected)), replace=False
                            )
                            to_remove = cand_remove[
                                np.argmin(self.scores_matrix[u, cand_remove])
                            ]
                            cand_add = np.random.choice(
                                unselected, size=min(5, len(unselected)), replace=False
                            )
                            to_add = cand_add[
                                np.argmax(self.scores_matrix[u, cand_add])
                            ]
                            mutated[i, u, to_remove] = 0
                            mutated[i, u, to_add] = 1
                        elif strategy == "suppress":
                            cand_remove = np.random.choice(
                                selected, size=min(3, len(selected)), replace=False
                            )
                            to_remove = cand_remove[
                                np.argmax(self.scores_matrix[u, cand_remove])
                            ]
                            cand_add = np.random.choice(
                                unselected, size=min(5, len(unselected)), replace=False
                            )
                            to_add = cand_add[
                                np.argmin(self.scores_matrix[u, cand_add])
                            ]
                            mutated[i, u, to_remove] = 0
                            mutated[i, u, to_add] = 1
                        # else: fall through to inversion below
                        else:
                            # Inversion on the selected set
                            a, b = sorted(
                                np.random.choice(len(selected), size=2, replace=False)
                            )
                            # New selected set = selected with sub-array reversed
                            new_selected = np.concatenate([
                                selected[:a],
                                selected[a:b + 1][::-1],
                                selected[b + 1:],
                            ])
                            # The selection set is the same; bitmap doesn't change
                            # (inversion only changes order, not membership).
                            # However, to make the operator meaningful in bitmap
                            # space, we treat the reversal as selecting a different
                            # item from the unselected pool at a random rank.
                            to_remove = np.random.choice(selected)
                            to_add = np.random.choice(unselected) if len(unselected) > 0 else None
                            if to_add is not None:
                                mutated[i, u, to_remove] = 0
                                mutated[i, u, to_add] = 1
                    else:
                        # Pure inversion: reverse a segment of the selected list
                        # Since inversion in bitmap space doesn't change membership,
                        # we perform a random swap of one selected with one unselected
                        # that mimics the spirit of reordering.
                        if len(unselected) > 0:
                            a, b = sorted(
                                np.random.choice(len(selected), size=2, replace=False)
                            )
                            # Swap segment items with unselected items (re-insertion)
                            segment = selected[a:b + 1]
                            n_swap = max(1, len(segment) // 2)
                            to_remove = np.random.choice(
                                segment, size=min(n_swap, len(segment)), replace=False
                            )
                            to_add = np.random.choice(
                                unselected,
                                size=min(n_swap, len(unselected)),
                                replace=False,
                            )
                            n_actual = min(len(to_remove), len(to_add))
                            for r, a_item in zip(to_remove[:n_actual], to_add[:n_actual]):
                                mutated[i, u, r] = 0
                                mutated[i, u, a_item] = 1

        return mutated

    # ------------------------------------------------------------------
    # Crossover operators
    # ------------------------------------------------------------------

    def _ordered_crossover(
        self,
        population: np.ndarray,
        objectives: np.ndarray,
        violations: np.ndarray,
        n_offspring: int,
    ) -> np.ndarray:
        """
        Ordered Crossover (OX) adapted for bitmap representation.

        For each user:
        1.  Sort each parent's selected items by their scores (descending) to
            create an ordered sequence of K item indices.
        2.  Pick a random crossover segment from parent-1's ordered sequence.
        3.  Fill the remaining K − len(segment) positions with items from
            parent-2's ordered sequence that are not already in the segment,
            preserving their relative order.
        4.  Write the resulting K items back to the offspring bitmap.

        Falls back to copying the best parent when crossover is not applied.
        """
        offspring = np.zeros((n_offspring, self.n_users, self.n_items), dtype=np.int8)

        for i in range(n_offspring):
            if np.random.random() >= self.crossover_rate:
                parent = self._tournament_selection(
                    population, objectives, violations, 1
                )[0]
                offspring[i] = parent
                continue

            parents = self._tournament_selection(population, objectives, violations, 2)
            p1, p2 = parents[0], parents[1]

            for u in range(self.n_users):
                n_valid = self.items_per_user_arr[u]

                # Ordered sequences: items sorted by score descending
                p1_selected = np.where(p1[u, :n_valid] == 1)[0]
                p2_selected = np.where(p2[u, :n_valid] == 1)[0]

                k = len(p1_selected)
                if k == 0:
                    continue

                # Sort by score (descending) to define order
                p1_order = p1_selected[
                    np.argsort(self.scores_matrix[u, p1_selected])[::-1]
                ]
                p2_order = p2_selected[
                    np.argsort(self.scores_matrix[u, p2_selected])[::-1]
                ]

                if k == 1:
                    # Nothing to cross
                    offspring[i, u, p1_selected] = 1
                    continue

                # Random segment from p1
                cut1, cut2 = sorted(np.random.choice(k, size=2, replace=False))
                segment = p1_order[cut1: cut2 + 1]
                segment_set = set(segment.tolist())

                # Fill remaining from p2 in order
                remainder = [x for x in p2_order if x not in segment_set]
                need = k - len(segment)
                fill = remainder[:need]

                # Combine — cast to int to guard against float64 when fill is []
                child_items = np.concatenate(
                    [segment, np.array(fill, dtype=np.intp)]
                ).astype(np.intp)

                # Write to bitmap (clear first)
                offspring[i, u, :n_valid] = 0
                if len(child_items) > 0:
                    offspring[i, u, child_items] = 1

        return offspring

    def _partially_mapped_crossover(
        self,
        population: np.ndarray,
        objectives: np.ndarray,
        violations: np.ndarray,
        n_offspring: int,
    ) -> np.ndarray:
        """
        Partially Mapped Crossover (PMX) adapted for bitmap representation.

        For each user:
        1.  Sort each parent's selected items by score (descending) to create
            ordered sequences of length K.
        2.  Pick a random crossover segment from parent-1's sequence.
        3.  Copy the segment into the child.
        4.  For positions outside the segment, take items from parent-2's
            sequence; if an item is already in the child (conflict), look up
            the PMX mapping to find a non-conflicting substitute.
        5.  Write the K unique items back to the offspring bitmap.
        """
        offspring = np.zeros((n_offspring, self.n_users, self.n_items), dtype=np.int8)

        for i in range(n_offspring):
            if np.random.random() >= self.crossover_rate:
                parent = self._tournament_selection(
                    population, objectives, violations, 1
                )[0]
                offspring[i] = parent
                continue

            parents = self._tournament_selection(population, objectives, violations, 2)
            p1, p2 = parents[0], parents[1]

            for u in range(self.n_users):
                n_valid = self.items_per_user_arr[u]

                p1_selected = np.where(p1[u, :n_valid] == 1)[0]
                p2_selected = np.where(p2[u, :n_valid] == 1)[0]

                k = len(p1_selected)
                if k == 0:
                    continue

                p1_order = p1_selected[
                    np.argsort(self.scores_matrix[u, p1_selected])[::-1]
                ]
                p2_order = p2_selected[
                    np.argsort(self.scores_matrix[u, p2_selected])[::-1]
                ]

                if k == 1:
                    offspring[i, u, p1_selected] = 1
                    continue

                # Random segment from p1
                cut1, cut2 = sorted(np.random.choice(k, size=2, replace=False))
                segment = list(p1_order[cut1: cut2 + 1])

                # Build PMX mapping: p1_segment[j] <-> p2_segment[j]
                # mapping[item_from_p1] = item_from_p2 at same position
                p1_seg = list(p1_order[cut1: cut2 + 1])
                p2_seg = list(p2_order[cut1: cut2 + 1])
                pmx_map = {}  # p1 → p2 direction
                for a, b in zip(p1_seg, p2_seg):
                    pmx_map[a] = b

                child = list(segment)
                child_set = set(child)

                # Fill remaining positions from p2 in order
                p2_outside = [x for idx, x in enumerate(p2_order)
                               if idx < cut1 or idx > cut2]

                for item in p2_outside:
                    if item not in child_set:
                        child.append(item)
                        child_set.add(item)
                    else:
                        # Resolve conflict via PMX mapping chain
                        mapped = item
                        seen = {item}
                        while mapped in child_set:
                            mapped = pmx_map.get(mapped, mapped)
                            if mapped in seen:
                                # Cycle detected — pick first unselected item not in child
                                mapped = None
                                break
                            seen.add(mapped)

                        if mapped is not None and mapped not in child_set:
                            child.append(mapped)
                            child_set.add(mapped)
                        else:
                            # Last resort: random valid item not yet selected
                            unselected = [
                                x for x in range(n_valid) if x not in child_set
                            ]
                            if unselected:
                                fallback = unselected[
                                    np.argmax(self.scores_matrix[u, unselected])
                                ]
                                child.append(fallback)
                                child_set.add(fallback)

                    if len(child) == k:
                        break

                # Pad if short (shouldn't normally happen)
                if len(child) < k:
                    unselected_all = [x for x in range(n_valid) if x not in child_set]
                    need = k - len(child)
                    extra = unselected_all[:need]
                    child.extend(extra)

                child_arr = np.array(child[:k], dtype=np.intp)

                offspring[i, u, :n_valid] = 0
                if len(child_arr) > 0:
                    offspring[i, u, child_arr] = 1

        return offspring

    # ------------------------------------------------------------------
    # Override train() to accept operator selection
    # ------------------------------------------------------------------

    def train_ablation(
        self,
        mutation_op: str = "swap",
        crossover_op: str = "uniform",
    ) -> Dict:
        """
        Run GA with the specified mutation and crossover operators.

        Args:
            mutation_op:  "swap" | "insertion" | "inversion"
            crossover_op: "uniform" | "ox" | "pmx"

        Returns:
            Results dict (same structure as MAOptimizer.train())
        """
        # -- Header logging --
        self.logger.info(
            f"MA Ablation | Model:{self.model_name} | Dataset:{self.dataset_name} | "
            f"Group:{self.group_name} | K={self.k} | Fairness_metric={self.fairness_metric}"
        )
        self.logger.info(
            f"Operators | Mutation:{mutation_op} | Crossover:{crossover_op}"
        )
        self.logger.info(
            f"MA Parameters | Pop:{self.population_size} | Gen:{self.generations} | "
            f"Mut:{self.mutation_rate} | Cross:{self.crossover_rate}"
        )
        self.logger.info(
            f"Adaptive Penalty | beta1:{self.penalty_beta1} | beta2:{self.penalty_beta2} "
            f"| k:{self.penalty_history_k}"
        )

        # Evaluate baseline and set epsilon
        baseline_solution, baseline_eval = self._log_baseline_metrics()

        start_time = time.perf_counter()
        population = self._create_initial_population(self.population_size)

        start_epsilon = self.original_ugf
        target_epsilon = self.epsilon
        objectives, violations, ugf_gaps, signed_ugf = self._calculate_fitness(
            population, start_epsilon
        )

        best_idx = self._get_best_idx(objectives, violations)
        best_fitness = objectives[best_idx]
        best_solution = population[best_idx].copy()

        best_feasible_solution = None
        best_feasible_fitness = float("-inf")

        feasibility_history = []
        current_epsilon = start_epsilon

        for gen in range(self.generations):
            # Adaptive penalty (Bean & Hadj-Alouane)
            penalty_action = None
            if len(feasibility_history) >= self.penalty_history_k:
                recent_history = feasibility_history[-self.penalty_history_k :]
                if all(recent_history):
                    current_epsilon = current_epsilon / self.penalty_beta1
                    penalty_action = "tighten"
                elif not any(recent_history):
                    current_epsilon = current_epsilon * self.penalty_beta2
                    penalty_action = "relax"
                current_epsilon = max(
                    target_epsilon, min(current_epsilon, start_epsilon)
                )

            # Elitism
            sorted_indices = np.lexsort((-objectives, violations))
            elite_indices = sorted_indices[: self.elitism_count]
            elites = population[elite_indices].copy()

            n_offspring = self.population_size - self.elitism_count
            avg_bias = np.mean(signed_ugf)

            # --- Crossover ---
            if crossover_op == "uniform":
                offspring = self._uniform_crossover(
                    population, objectives, violations, n_offspring
                )
            elif crossover_op == "ox":
                offspring = self._ordered_crossover(
                    population, objectives, violations, n_offspring
                )
            elif crossover_op == "pmx":
                offspring = self._partially_mapped_crossover(
                    population, objectives, violations, n_offspring
                )
            else:
                raise ValueError(f"Unknown crossover operator: {crossover_op}")

            # --- Mutation ---
            if mutation_op == "swap":
                offspring = self._swap_mutation_repair_bias(
                    offspring, bias_dir=avg_bias, current_rate=self.mutation_rate
                )
            elif mutation_op == "insertion":
                offspring = self._insertion_mutation(
                    offspring, bias_dir=avg_bias, current_rate=self.mutation_rate
                )
            elif mutation_op == "inversion":
                offspring = self._inversion_mutation(
                    offspring, bias_dir=avg_bias, current_rate=self.mutation_rate
                )
            else:
                raise ValueError(f"Unknown mutation operator: {mutation_op}")

            # New population
            population = np.concatenate([elites, offspring], axis=0)

            # Evaluate
            objectives, violations, ugf_gaps, signed_ugf = self._calculate_fitness(
                population, current_epsilon
            )

            # Track best
            gen_best_idx = self._get_best_idx(objectives, violations)
            gen_best_fitness = objectives[gen_best_idx]
            gen_best_ugf = ugf_gaps[gen_best_idx]
            gen_best_viol = violations[gen_best_idx]

            gen_best_is_feasible = gen_best_viol == 0
            feasibility_history.append(gen_best_is_feasible)

            if gen_best_fitness > best_fitness:
                best_solution = population[gen_best_idx].copy()
                best_fitness = gen_best_fitness

            target_violations = np.maximum(0, ugf_gaps - target_epsilon)
            feasible_mask = target_violations == 0

            if feasible_mask.any():
                feasible_objs = objectives.copy()
                feasible_objs[~feasible_mask] = float("-inf")
                current_best_feasible_idx = np.argmax(feasible_objs)

                if feasible_objs[current_best_feasible_idx] > best_feasible_fitness:
                    best_feasible_fitness = feasible_objs[current_best_feasible_idx]
                    best_feasible_solution = population[
                        current_best_feasible_idx
                    ].copy()

            adapt_status = (
                " [TIGHTEN]"
                if penalty_action == "tighten"
                else (" [RELAX]" if penalty_action == "relax" else "")
            )
            print(
                f"  Gen {gen + 1}: UGF={gen_best_ugf:.4f}, "
                f"viol={gen_best_viol:.4f}{adapt_status}"
            )

            if best_feasible_solution is not None:
                print(f"  Early stop at gen {gen + 1}: feasible solution found")
                self.logger.info(
                    f"Early stop at generation {gen + 1}: first feasible solution found"
                )
                break

        cpu_time = time.perf_counter() - start_time
        self.logger.info(f"CPU time: {cpu_time:.2f} seconds")

        final_solution = (
            best_feasible_solution
            if best_feasible_solution is not None
            else best_solution
        )

        print(f"Completed in {cpu_time:.2f}s ({self.generations} gens)")

        return self._log_final_results(
            final_solution, target_epsilon, baseline_eval, best_fitness, cpu_time
        )


# ---------------------------------------------------------------------------
# Baseline log parser
# ---------------------------------------------------------------------------

_BASELINE_LOG_RE = {
    "final_ugf": re.compile(
        r"After optimization UGF \(f1@\d+\): ([\d.]+)", re.IGNORECASE
    ),
    "original_ugf": re.compile(
        r"Before optimization UGF \(f1@\d+\): ([\d.]+)", re.IGNORECASE
    ),
    "epsilon": re.compile(r"Epsilon: ([\d.]+)", re.IGNORECASE),
    "constraint_satisfied": re.compile(
        r"Constraint satisfied: (True|False)", re.IGNORECASE
    ),
    "cpu_time": re.compile(r"CPU time: ([\d.]+) seconds", re.IGNORECASE),
    "final_ndcg": re.compile(
        r"After optimization overall metric scores\s*:\s*ndcg@\d+=(\S+)", re.IGNORECASE
    ),
    "final_f1": re.compile(
        r"After optimization overall metric scores\s*:.*?f1@\d+=(\S+)", re.IGNORECASE
    ),
    "ugf_reduction_pct": re.compile(
        r"UGF reduction: [\d.]+ \(([\d.]+)%\)", re.IGNORECASE
    ),
    "final_g1_f1": re.compile(
        r"After optimization group 1.*?:\s*ndcg@\d+=\S+\s+f1@\d+=(\S+)", re.IGNORECASE
    ),
    "final_g2_f1": re.compile(
        r"After optimization group 2.*?:\s*ndcg@\d+=\S+\s+f1@\d+=(\S+)", re.IGNORECASE
    ),
    "final_g1_ndcg": re.compile(
        r"After optimization group 1.*?:\s*ndcg@\d+=(\S+)", re.IGNORECASE
    ),
    "final_g2_ndcg": re.compile(
        r"After optimization group 2.*?:\s*ndcg@\d+=(\S+)", re.IGNORECASE
    ),
    "final_ugf_ndcg": re.compile(
        r"After optimization UGF.*?\| UGF \(ndcg@\d+\): ([\d.]+)", re.IGNORECASE
    ),
}


def parse_baseline_log(log_path: str) -> Optional[Dict]:
    """
    Parse a GA baseline log file and return a dict with key metrics.
    Returns None if the file does not exist or cannot be parsed.
    """
    if not os.path.exists(log_path):
        return None

    try:
        with open(log_path, "r") as f:
            content = f.read()
    except IOError:
        return None

    result = {"variant": "swap+uniform (baseline)", "log_path": log_path}

    for key, pattern in _BASELINE_LOG_RE.items():
        m = pattern.search(content)
        if m:
            val = m.group(1)
            if key == "constraint_satisfied":
                result[key] = val.lower() == "true"
            else:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
        else:
            result[key] = None

    # Compute UGF reduction pct if not found
    if result.get("ugf_reduction_pct") is None:
        orig = result.get("original_ugf")
        final = result.get("final_ugf")
        if orig and final and orig > 0:
            result["ugf_reduction_pct"] = (orig - final) / orig * 100

    return result


# ---------------------------------------------------------------------------
# Ablation study runner
# ---------------------------------------------------------------------------

ABLATION_VARIANTS: List[Tuple[str, str, str]] = [
    ("insertion+uniform", "insertion", "uniform"),
    ("inversion+uniform", "inversion", "uniform"),
]

ALL_DATASETS = ["5Beauty-rand", "5Grocery-rand", "5Health-rand"]
ALL_MODELS = ["biasedMF", "NCF"]
ALL_GROUPINGS = [
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


def run_ablation_study(
    datasets: List[str] = None,
    models: List[str] = None,
    groupings: List[Tuple[str, str, str]] = None,
    variants: List[Tuple[str, str, str]] = None,
    dataset_folder: str = "../dataset",
    results_base_dir: str = "../results",
    output_dir: str = "../results/ablation",
    generations: int = 1000,
    seed: int = 42,
    population_size: int = 10,
    mutation_rate: float = 0.3504,
    crossover_rate: float = 0.7262,
    elitism_count: int = 5,
    penalty_beta1: float = 2.43,
    penalty_beta2: float = 1.88,
    penalty_history_k: int = 6,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run the full ablation study.

    Returns:
        (all_run_results, baseline_results)
        where all_run_results contains one dict per (config, variant) run
        and baseline_results contains one dict per baseline (config).
    """
    datasets = datasets or ALL_DATASETS
    models = models or ALL_MODELS
    groupings = groupings or ALL_GROUPINGS
    variants = variants or ABLATION_VARIANTS

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    total_configs = len(datasets) * len(models) * len(groupings)
    total_runs = total_configs * len(variants)

    print("=" * 80)
    print("MA ABLATION STUDY")
    print("=" * 80)
    print(f"Datasets   : {datasets}")
    print(f"Models     : {models}")
    print(f"Groupings  : {[g[0] for g in groupings]}")
    print(f"Variants   : {[v[0] for v in variants]}")
    print(f"Generations: {generations} | Seed: {seed}")
    print(f"Configs    : {total_configs} | Total runs: {total_runs}")
    print("=" * 80)

    all_run_results: List[Dict] = []
    baseline_results: List[Dict] = []

    # ----------------------------------------------------------------
    # 1.  Parse all baseline logs up-front
    # ----------------------------------------------------------------
    print("\n[Step 1] Parsing baseline logs from existing results …")
    for dataset_name in datasets:
        for model_name in models:
            for group_name, _, _ in groupings:
                log_file = f"ga_{model_name}_{dataset_name}_{group_name}.log"
                log_path = os.path.join(results_base_dir, model_name, "ga", log_file)
                baseline = parse_baseline_log(log_path)
                if baseline is not None:
                    baseline["dataset"] = dataset_name
                    baseline["model"] = model_name
                    baseline["grouping"] = group_name
                    baseline_results.append(baseline)
                    print(
                        f"  ✓ Baseline parsed: {model_name} | {dataset_name} | "
                        f"{group_name}  →  UGF={baseline.get('final_ugf', 'N/A')}"
                    )
                else:
                    print(
                        f"  ✗ Baseline log not found: {log_path}"
                    )

    # ----------------------------------------------------------------
    # 2.  Run ablation variants
    # ----------------------------------------------------------------
    print(f"\n[Step 2] Running {total_runs} ablation experiments …\n")
    current_run = 0

    for dataset_name in datasets:
        for model_name in models:

            # Pre-build vectorized data once per (dataset, model) pair
            data_path = os.path.join(dataset_folder, dataset_name)
            rank_file = f"{model_name}_rank.csv"

            if not os.path.exists(os.path.join(data_path, rank_file)):
                print(f"  SKIPPED (rank file not found): {rank_file}")
                continue

            # We need a dummy grouping to build data (use first grouping)
            first_group_name, first_g1, first_g2 = groupings[0]
            dummy_dl = DataLoader(
                data_path,
                rank_file=rank_file,
                group_1_file=first_g1,
                group_2_file=first_g2,
            )

            # Shared cache key (model+dataset level, grouping-independent for scores)
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")
            os.makedirs(cache_dir, exist_ok=True)

            for group_name, group_1_file, group_2_file in groupings:
                cache_file = os.path.join(
                    cache_dir,
                    f"vectorized_cache_{dataset_name}_{model_name}_{group_name}_k10.pkl",
                )

                try:
                    dl = DataLoader(
                        data_path,
                        rank_file=rank_file,
                        group_1_file=group_1_file,
                        group_2_file=group_2_file,
                    )

                    if os.path.exists(cache_file):
                        prebuilt_data = MAOptimizer.load_vectorized_data(cache_file)
                    else:
                        prebuilt_data = MAOptimizer.build_vectorized_data(dl, k=10)
                        MAOptimizer.save_vectorized_data(prebuilt_data, cache_file)

                except Exception as e:
                    print(f"  ERROR loading data for {dataset_name}/{model_name}/{group_name}: {e}")
                    continue

                for variant_label, mutation_op, crossover_op in variants:
                    current_run += 1
                    safe_label = variant_label.replace("+", "_").replace(" ", "_")

                    print(f"\n{'=' * 70}")
                    print(
                        f"RUN {current_run}/{total_runs}: "
                        f"{dataset_name} | {model_name} | {group_name} | {variant_label}"
                    )
                    print("=" * 70)

                    # Setup per-run logger
                    logger_file = (
                        f"ablation_{model_name}_{dataset_name}_{group_name}_{safe_label}.log"
                    )
                    logger_path = os.path.join(log_dir, logger_file)
                    logger = create_logger(
                        name=f"ablation_{dataset_name}_{model_name}_{group_name}_{safe_label}",
                        path=logger_path,
                    )

                    try:
                        ga = AblationMAOptimizer(
                            data_loader=dl,
                            k=10,
                            eval_metric_list=["ndcg@10", "f1@10"],
                            fairness_metric="f1",
                            logger=logger,
                            model_name=model_name,
                            group_name=group_name,
                            population_size=population_size,
                            generations=generations,
                            mutation_rate=mutation_rate,
                            crossover_rate=crossover_rate,
                            elitism_count=elitism_count,
                            penalty_beta1=penalty_beta1,
                            penalty_beta2=penalty_beta2,
                            penalty_history_k=penalty_history_k,
                            seed=seed,
                            prebuilt_data=prebuilt_data,
                        )

                        results = ga.train_ablation(
                            mutation_op=mutation_op, crossover_op=crossover_op
                        )

                        run_result = {
                            "dataset": dataset_name,
                            "model": model_name,
                            "grouping": group_name,
                            "variant": variant_label,
                            "mutation_op": mutation_op,
                            "crossover_op": crossover_op,
                            "original_ugf": float(results["original_ugf"]),
                            "final_ugf": float(results["final_ugf"]),
                            "epsilon": float(results["epsilon"]),
                            "constraint_satisfied": bool(results["constraint_satisfied"]),
                            "cpu_time": float(results["cpu_time"]),
                            "final_ndcg": float(results["final_metrics"][0]),
                            "final_f1": float(results["final_metrics"][1]),
                            "final_g1_f1": float(results["final_g1_f1"]),
                            "final_g2_f1": float(results["final_g2_f1"]),
                            "final_g1_ndcg": float(results["final_g1_ndcg"])
                            if results.get("final_g1_ndcg") is not None
                            else None,
                            "final_g2_ndcg": float(results["final_g2_ndcg"])
                            if results.get("final_g2_ndcg") is not None
                            else None,
                            "final_ugf_ndcg": float(results["final_ugf_ndcg"])
                            if results.get("final_ugf_ndcg") is not None
                            else None,
                            "ugf_reduction_pct": float(
                                (results["original_ugf"] - results["final_ugf"])
                                / results["original_ugf"]
                                * 100
                                if results["original_ugf"] > 0
                                else 0
                            ),
                            "log_path": logger_path,
                        }
                        all_run_results.append(run_result)
                        print(
                            f"  ✓  UGF: {run_result['original_ugf']:.4f} → "
                            f"{run_result['final_ugf']:.4f} "
                            f"({run_result['ugf_reduction_pct']:.1f}% reduction) "
                            f"| Feasible: {run_result['constraint_satisfied']}"
                        )

                    except Exception as e:
                        import traceback
                        print(f"  ✗ ERROR: {e}")
                        traceback.print_exc()
                        all_run_results.append(
                            {
                                "dataset": dataset_name,
                                "model": model_name,
                                "grouping": group_name,
                                "variant": variant_label,
                                "mutation_op": mutation_op,
                                "crossover_op": crossover_op,
                                "error": str(e),
                            }
                        )

    return all_run_results, baseline_results


# ---------------------------------------------------------------------------
# Export & reporting
# ---------------------------------------------------------------------------

def export_results(
    all_run_results: List[Dict],
    baseline_results: List[Dict],
    output_dir: str,
) -> Tuple[str, str]:
    """Save results to JSON and CSV and return the paths."""
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Combined JSON (ablation runs + baselines)
    combined = {
        "ablation_runs": all_run_results,
        "baseline_results": baseline_results,
        "timestamp": timestamp,
    }
    json_path = os.path.join(output_dir, f"ablation_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nRaw results saved to: {json_path}")

    # CSV — merge ablation + baseline rows
    csv_rows = []

    # Baseline rows
    for b in baseline_results:
        csv_rows.append(
            {
                "dataset": b.get("dataset", ""),
                "model": b.get("model", ""),
                "grouping": b.get("grouping", ""),
                "variant": "swap+uniform (baseline)",
                "mutation_op": "swap",
                "crossover_op": "uniform",
                "original_ugf": b.get("original_ugf"),
                "final_ugf": b.get("final_ugf"),
                "epsilon": b.get("epsilon"),
                "constraint_satisfied": b.get("constraint_satisfied"),
                "cpu_time": b.get("cpu_time"),
                "final_ndcg": b.get("final_ndcg"),
                "final_f1": b.get("final_f1"),
                "final_g1_f1": b.get("final_g1_f1"),
                "final_g2_f1": b.get("final_g2_f1"),
                "final_g1_ndcg": b.get("final_g1_ndcg"),
                "final_g2_ndcg": b.get("final_g2_ndcg"),
                "final_ugf_ndcg": b.get("final_ugf_ndcg"),
                "ugf_reduction_pct": b.get("ugf_reduction_pct"),
            }
        )

    # Ablation rows
    for r in all_run_results:
        if "error" not in r:
            csv_rows.append(
                {
                    "dataset": r.get("dataset", ""),
                    "model": r.get("model", ""),
                    "grouping": r.get("grouping", ""),
                    "variant": r.get("variant", ""),
                    "mutation_op": r.get("mutation_op", ""),
                    "crossover_op": r.get("crossover_op", ""),
                    "original_ugf": r.get("original_ugf"),
                    "final_ugf": r.get("final_ugf"),
                    "epsilon": r.get("epsilon"),
                    "constraint_satisfied": r.get("constraint_satisfied"),
                    "cpu_time": r.get("cpu_time"),
                    "final_ndcg": r.get("final_ndcg"),
                    "final_f1": r.get("final_f1"),
                    "final_g1_f1": r.get("final_g1_f1"),
                    "final_g2_f1": r.get("final_g2_f1"),
                    "final_g1_ndcg": r.get("final_g1_ndcg"),
                    "final_g2_ndcg": r.get("final_g2_ndcg"),
                    "final_ugf_ndcg": r.get("final_ugf_ndcg"),
                    "ugf_reduction_pct": r.get("ugf_reduction_pct"),
                }
            )

    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, f"ablation_summary_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")

    return json_path, csv_path


def print_comparison_table(
    all_run_results: List[Dict],
    baseline_results: List[Dict],
    groupings: List[Tuple[str, str, str]] = None,
    models: List[str] = None,
    datasets: List[str] = None,
) -> None:
    """
    Print a formatted comparison table: baseline vs. each ablation variant
    for every configuration (dataset × model × grouping).
    """
    groupings = groupings or ALL_GROUPINGS
    models = models or ALL_MODELS
    datasets = datasets or ALL_DATASETS

    # Index baseline by (dataset, model, grouping)
    baseline_idx: Dict[Tuple, Dict] = {}
    for b in baseline_results:
        key = (b["dataset"], b["model"], b["grouping"])
        baseline_idx[key] = b

    # Index ablation runs
    ablation_idx: Dict[Tuple, Dict] = {}
    for r in all_run_results:
        if "error" not in r:
            key = (r["dataset"], r["model"], r["grouping"], r["variant"])
            ablation_idx[key] = r

    col_w = 22
    variants_all = ["swap+uniform (baseline)"] + [v[0] for v in ABLATION_VARIANTS]

    print("\n" + "=" * 120)
    print("ABLATION STUDY — COMPARISON TABLE")
    print("=" * 120)
    print(
        f"{'Config':<38} "
        + "  ".join(f"{'[' + v + ']':>{col_w}}" for v in variants_all)
    )
    sub = "Final_UGF | UGF_Red% | Feas"
    print(
        f"{'':38} "
        + "  ".join(f"{sub:>{col_w}}" for _ in variants_all)
    )
    print("-" * 120)

    for dataset_name in datasets:
        for model_name in models:
            for group_name, _, _ in groupings:
                cfg = (
                    f"{dataset_name.replace('5','').replace('-rand','')} "
                    f"| {model_name} | {group_name}"
                )

                row_parts = [f"{cfg:<38}"]

                # Baseline
                b = baseline_idx.get((dataset_name, model_name, group_name))
                if b:
                    feas = "✓" if b.get("constraint_satisfied") else "✗"
                    cell = (
                        f"{b.get('final_ugf', float('nan')):.4f} | "
                        f"{b.get('ugf_reduction_pct', float('nan')):.1f}% | {feas}"
                    )
                else:
                    cell = "N/A"
                row_parts.append(f"{cell:>{col_w}}")

                # Ablation variants
                for variant_label, _, _ in ABLATION_VARIANTS:
                    r = ablation_idx.get(
                        (dataset_name, model_name, group_name, variant_label)
                    )
                    if r:
                        feas = "✓" if r.get("constraint_satisfied") else "✗"
                        cell = (
                            f"{r.get('final_ugf', float('nan')):.4f} | "
                            f"{r.get('ugf_reduction_pct', float('nan')):.1f}% | {feas}"
                        )
                    else:
                        cell = "N/A (not run)"
                    row_parts.append(f"{cell:>{col_w}}")

                print("  ".join(row_parts))

    print("=" * 120)

    # ---- Aggregate summary by variant ----
    print("\nAGGREGATE SUMMARY (mean across all configs)")
    print("-" * 80)
    header = (
        f"{'Variant':<30} {'Mean_UGF':>10} {'Mean_UGF_Red%':>15} "
        f"{'Mean_F1':>10} {'Success%':>10}"
    )
    print(header)
    print("-" * 80)

    # Baseline aggregate
    ugfs = [b.get("final_ugf") for b in baseline_results if b.get("final_ugf") is not None]
    reds = [b.get("ugf_reduction_pct") for b in baseline_results if b.get("ugf_reduction_pct") is not None]
    f1s = [b.get("final_f1") for b in baseline_results if b.get("final_f1") is not None]
    succ = [b.get("constraint_satisfied") for b in baseline_results if b.get("constraint_satisfied") is not None]
    if ugfs:
        print(
            f"{'swap+uniform (baseline)':<30} "
            f"{sum(ugfs)/len(ugfs):>10.4f} "
            f"{sum(reds)/len(reds):>15.1f} "
            f"{sum(f1s)/len(f1s) if f1s else float('nan'):>10.4f} "
            f"{sum(1 for s in succ if s)/len(succ)*100 if succ else 0:>9.1f}%"
        )

    for variant_label, _, _ in ABLATION_VARIANTS:
        runs = [
            r for r in all_run_results
            if r.get("variant") == variant_label and "error" not in r
        ]
        if not runs:
            print(f"  {variant_label:<28} No results")
            continue
        m_ugf = sum(r["final_ugf"] for r in runs) / len(runs)
        m_red = sum(r["ugf_reduction_pct"] for r in runs) / len(runs)
        m_f1 = sum(r["final_f1"] for r in runs) / len(runs)
        m_succ = sum(1 for r in runs if r.get("constraint_satisfied")) / len(runs) * 100
        print(
            f"{variant_label:<30} "
            f"{m_ugf:>10.4f} "
            f"{m_red:>15.1f} "
            f"{m_f1:>10.4f} "
            f"{m_succ:>9.1f}%"
        )

    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MA Ablation Study — compare mutation and crossover operators"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset name (default: all 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model name (default: biasedMF and NCF)",
    )
    parser.add_argument(
        "--grouping",
        type=str,
        default=None,
        help="Single grouping name, e.g. max_0.05 (default: all 3)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=1000,
        help="Max GA generations per run (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42, same as baseline)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/ablation",
        help="Output directory for logs and results (default: ../results/ablation)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Base results directory containing existing GA logs (default: ../results)",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default="../dataset",
        help="Path to dataset folder (default: ../dataset)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Quick smoke test: first config only, 5 generations, "
            "first variant (insertion+OX) only"
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=[v[0] for v in ABLATION_VARIANTS],
        help="Run only one specific variant (default: all 4)",
    )

    args = parser.parse_args()

    # Build configuration lists
    datasets = [args.dataset] if args.dataset else ALL_DATASETS
    models = [args.model] if args.model else ALL_MODELS

    groupings = ALL_GROUPINGS
    if args.grouping:
        groupings = [g for g in ALL_GROUPINGS if g[0] == args.grouping]
        if not groupings:
            print(f"ERROR: Unknown grouping '{args.grouping}'")
            print(f"Available: {[g[0] for g in ALL_GROUPINGS]}")
            sys.exit(1)

    variants = ABLATION_VARIANTS
    if args.variant:
        variants = [v for v in ABLATION_VARIANTS if v[0] == args.variant]

    generations = args.generations

    if args.smoke_test:
        print("[SMOKE TEST MODE] One config, 5 generations, insertion+OX only")
        datasets = [ALL_DATASETS[0]]
        models = [ALL_MODELS[1]]  # NCF
        groupings = [ALL_GROUPINGS[2]]  # max_0.05
        variants = [ABLATION_VARIANTS[0]]  # insertion+OX
        generations = 5

    all_run_results, baseline_results = run_ablation_study(
        datasets=datasets,
        models=models,
        groupings=groupings,
        variants=variants,
        dataset_folder=args.dataset_folder,
        results_base_dir=args.results_dir,
        output_dir=args.output,
        generations=generations,
        seed=args.seed,
    )

    # Export
    export_results(all_run_results, baseline_results, args.output)

    # Print comparison table
    print_comparison_table(
        all_run_results,
        baseline_results,
        groupings=groupings,
        models=models,
        datasets=datasets,
    )

    print("\nAblation study complete!")
    print(f"Results saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
