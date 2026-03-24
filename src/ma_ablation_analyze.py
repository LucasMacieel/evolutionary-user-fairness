"""
Ablation Study Results Analyzer
================================

Reads all ablation JSON result files from results/ablation/ and prints
a summary comparing mean CPU time and fairness metrics across variants.

Usage:
    python ma_ablation_analyze.py
    python ma_ablation_analyze.py --results-dir ../results/ablation
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def load_all_runs(results_dir: str):
    """Load and merge ablation runs from all JSON files in results_dir."""
    pattern = os.path.join(results_dir, "ablation_results_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No ablation result files found in: {results_dir}")
        return [], []

    all_runs = []
    all_baselines = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        all_runs.extend(data.get("ablation_runs", []))
        all_baselines.extend(data.get("baseline_results", []))
        print(f"  Loaded: {os.path.basename(f)}")

    return all_runs, all_baselines


def summarize(runs):
    """Return per-variant summary dict."""
    groups = defaultdict(list)
    for r in runs:
        if "error" not in r and r.get("cpu_time") is not None:
            groups[r["variant"]].append(r)

    summary = {}
    for variant, rs in groups.items():
        summary[variant] = {
            "n": len(rs),
            "mean_cpu": sum(r["cpu_time"] for r in rs) / len(rs),
            "mean_ugf": sum(r["final_ugf"] for r in rs) / len(rs),
            "mean_ugf_red": sum(r["ugf_reduction_pct"] for r in rs) / len(rs),
            "mean_f1": sum(r["final_f1"] for r in rs) / len(rs),
            "success_pct": sum(1 for r in rs if r.get("constraint_satisfied")) / len(rs) * 100,
        }
    return summary


def baseline_summary(baselines):
    """Summarize baseline entries, deduplicating by (dataset, model, grouping)."""
    seen = set()
    unique = []
    for b in baselines:
        key = (b.get("dataset"), b.get("model"), b.get("grouping"))
        if key not in seen and b.get("cpu_time") is not None:
            seen.add(key)
            unique.append(b)

    if not unique:
        return None

    return {
        "n": len(unique),
        "mean_cpu": sum(b["cpu_time"] for b in unique) / len(unique),
        "mean_ugf": sum(b["final_ugf"] for b in unique if b.get("final_ugf")) / len(unique),
        "mean_ugf_red": sum(b["ugf_reduction_pct"] for b in unique if b.get("ugf_reduction_pct")) / len(unique),
        "mean_f1": sum(b["final_f1"] for b in unique if b.get("final_f1") or 0) / len(unique),
        "success_pct": sum(1 for b in unique if b.get("constraint_satisfied")) / len(unique) * 100,
    }


def print_table(variant_summary, baseline):
    print("\n" + "=" * 90)
    print("ABLATION RESULTS — FULL COMPARISON (including CPU time)")
    print("=" * 90)
    print(
        f"{'Variant':<30} {'Mean_UGF':>10} {'UGF_Red%':>10} "
        f"{'Mean_F1':>10} {'Success%':>10} {'Mean_CPU(s)':>12} {'N':>5}"
    )
    print("-" * 90)

    # Baseline row
    if baseline:
        b = baseline
        print(
            f"{'swap+uniform (baseline)':<30} {b['mean_ugf']:>10.4f} {b['mean_ugf_red']:>10.1f} "
            f"{b['mean_f1']:>10.4f} {b['success_pct']:>9.1f}% {b['mean_cpu']:>12.2f} {b['n']:>5}"
        )

    for variant, s in sorted(variant_summary.items()):
        # Compute CPU overhead vs baseline
        overhead = ""
        if baseline:
            delta = s["mean_cpu"] - baseline["mean_cpu"]
            overhead = f" ({'+' if delta >= 0 else ''}{delta:.1f}s)"
        print(
            f"{variant:<30} {s['mean_ugf']:>10.4f} {s['mean_ugf_red']:>10.1f} "
            f"{s['mean_f1']:>10.4f} {s['success_pct']:>9.1f}% "
            f"{s['mean_cpu']:>10.2f}{overhead:<8} {s['n']:>5}"
        )

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Analyze MA ablation study results")
    parser.add_argument(
        "--results-dir",
        default="../results/ablation",
        help="Directory containing ablation_results_*.json files",
    )
    args = parser.parse_args()

    print(f"Loading results from: {os.path.abspath(args.results_dir)}")
    all_runs, all_baselines = load_all_runs(args.results_dir)

    if not all_runs and not all_baselines:
        return

    print(f"\nTotal ablation runs loaded: {len(all_runs)}")

    variant_summary = summarize(all_runs)
    baseline = baseline_summary(all_baselines)

    print_table(variant_summary, baseline)


if __name__ == "__main__":
    main()
