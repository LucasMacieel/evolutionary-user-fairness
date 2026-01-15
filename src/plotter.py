"""
Results Plotter Module for MILP vs GA Comparison.

Generates visualizations and insights from comparison CSV files:
- Time comparison bar charts
- Quality (NDCG) comparison
- UGF fairness comparison
- Speedup analysis
- Summary statistics
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


# Set seaborn style
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("paper", font_scale=1.2)


def load_all_results(results_folder: str = "../results/comparison") -> pd.DataFrame:
    """Load and combine all summary CSV files from the results folder."""
    csv_files = glob.glob(os.path.join(results_folder, "*_summary.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No summary CSV files found in {results_folder}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Add computed columns
    combined["Speedup"] = combined["MILP_Time"] / combined["GA_Time"]
    combined["NDCG_Gap"] = (
        (combined["GA_NDCG"] - combined["MILP_NDCG"]) / combined["MILP_NDCG"] * 100
    )
    combined["UGF_Diff"] = combined["GA_UGF"] - combined["MILP_UGF"]
    combined["GA_Quality_Retention"] = combined["GA_NDCG"] / combined["MILP_NDCG"] * 100

    return combined


def _prepare_long_format(
    df: pd.DataFrame, value_cols: dict, id_vars: list
) -> pd.DataFrame:
    """Convert wide format to long format for seaborn plotting."""
    records = []
    for _, row in df.iterrows():
        for solver, col in value_cols.items():
            record = {var: row[var] for var in id_vars}
            record["Solver"] = solver
            record["Value"] = row[col]
            records.append(record)
    return pd.DataFrame(records)


def plot_time_comparison(df: pd.DataFrame, output_dir: str, figsize=(12, 6)):
    """Generate bar chart comparing MILP vs GA execution times."""
    fig, ax = plt.subplots(figsize=figsize)

    # Create label column
    df["Config"] = (
        df["Dataset"].str.replace("-rand", "")
        + " | "
        + df["Model"]
        + " | "
        + df["Group"]
    )

    # Prepare long format for seaborn
    df_long = _prepare_long_format(
        df,
        value_cols={"MILP": "MILP_Time", "GA": "GA_Time"},
        id_vars=["Config", "Dataset", "Model", "Group"],
    )

    # Create grouped bar plot
    sns.barplot(
        data=df_long,
        x="Config",
        y="Value",
        hue="Solver",
        palette={"MILP": "#2E86AB", "GA": "#A23B72"},
        ax=ax,
        edgecolor="black",
    )

    ax.set_xlabel("Experiment Configuration", fontsize=12)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax.set_title(
        "MILP vs GA: Execution Time Comparison", fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Solver")

    # Add speedup annotations
    for i, (_, row) in enumerate(df.iterrows()):
        speedup = row["MILP_Time"] / row["GA_Time"]
        ax.annotate(
            f"{speedup:.1f}x",
            xy=(i, row["MILP_Time"] + 20),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#2E86AB",
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "time_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_quality_comparison(df: pd.DataFrame, output_dir: str, figsize=(12, 6)):
    """Generate bar chart comparing MILP vs GA NDCG quality."""
    fig, ax = plt.subplots(figsize=figsize)

    df["Config"] = (
        df["Dataset"].str.replace("-rand", "")
        + " | "
        + df["Model"]
        + " | "
        + df["Group"]
    )

    df_long = _prepare_long_format(
        df,
        value_cols={"MILP (Optimal)": "MILP_NDCG", "GA": "GA_NDCG"},
        id_vars=["Config", "Dataset", "Model", "Group"],
    )

    sns.barplot(
        data=df_long,
        x="Config",
        y="Value",
        hue="Solver",
        palette={"MILP (Optimal)": "#2E86AB", "GA": "#A23B72"},
        ax=ax,
        edgecolor="black",
    )

    ax.set_xlabel("Experiment Configuration", fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_title(
        "MILP vs GA: Solution Quality (NDCG@10)", fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Solver")

    # Set y-axis to start from reasonable baseline
    min_val = min(df["MILP_NDCG"].min(), df["GA_NDCG"].min())
    ax.set_ylim(bottom=min_val * 0.9)

    # Add quality retention annotations
    for i, (_, row) in enumerate(df.iterrows()):
        retention = row["GA_NDCG"] / row["MILP_NDCG"] * 100
        ax.annotate(
            f"{retention:.1f}%",
            xy=(i + 0.2, row["GA_NDCG"] + 0.003),
            ha="center",
            fontsize=8,
            color="#A23B72",
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "quality_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_ugf_comparison(df: pd.DataFrame, output_dir: str, figsize=(12, 6)):
    """Generate bar chart comparing MILP vs GA UGF (fairness)."""
    fig, ax = plt.subplots(figsize=figsize)

    df["Config"] = (
        df["Dataset"].str.replace("-rand", "")
        + " | "
        + df["Model"]
        + " | "
        + df["Group"]
    )

    df_long = _prepare_long_format(
        df,
        value_cols={"MILP": "MILP_UGF", "GA": "GA_UGF"},
        id_vars=["Config", "Dataset", "Model", "Group"],
    )

    sns.barplot(
        data=df_long,
        x="Config",
        y="Value",
        hue="Solver",
        palette={"MILP": "#2E86AB", "GA": "#A23B72"},
        ax=ax,
        edgecolor="black",
    )

    ax.set_xlabel("Experiment Configuration", fontsize=12)
    ax.set_ylabel("User-Gap-Fairness (UGF)", fontsize=12)
    ax.set_title(
        "MILP vs GA: Fairness Constraint (UGF)", fontsize=14, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Solver")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "ugf_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_speedup_chart(df: pd.DataFrame, output_dir: str, figsize=(10, 6)):
    """Generate horizontal bar chart showing speedup with quality retention."""
    fig, ax = plt.subplots(figsize=figsize)

    df["Config"] = (
        df["Dataset"].str.replace("-rand", "")
        + " | "
        + df["Model"]
        + " | "
        + df["Group"]
    )
    df_sorted = df.sort_values("Speedup", ascending=True).reset_index(drop=True)

    # Create color gradient based on quality retention
    colors = sns.color_palette("RdYlGn", n_colors=len(df_sorted))
    retention_order = df_sorted["GA_Quality_Retention"].argsort().argsort()
    bar_colors = [colors[i] for i in retention_order]

    sns.barplot(
        data=df_sorted,
        y="Config",
        x="Speedup",
        palette=bar_colors,
        ax=ax,
        edgecolor="black",
        orient="h",
    )

    ax.axvline(x=1, color="red", linestyle="--", linewidth=2, label="Break-even (1x)")
    ax.set_xlabel("Speedup (MILP Time / GA Time)", fontsize=12)
    ax.set_ylabel("Experiment Configuration", fontsize=12)
    ax.set_title("GA Speedup over MILP Solver", fontsize=14, fontweight="bold")

    # Annotate with speedup and quality
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.annotate(
            f"{row['Speedup']:.2f}x ({row['GA_Quality_Retention']:.1f}% qual.)",
            xy=(row["Speedup"] + 0.1, i),
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "speedup_chart.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_tradeoff_scatter(df: pd.DataFrame, output_dir: str, figsize=(10, 8)):
    """Generate scatter plot showing speedup vs quality tradeoff."""
    fig, ax = plt.subplots(figsize=figsize)

    # Seaborn scatter with hue and style
    sns.scatterplot(
        data=df,
        x="Speedup",
        y="GA_Quality_Retention",
        hue="Model",
        style="Dataset",
        s=200,
        ax=ax,
        palette={"NCF": "#2E86AB", "biasedMF": "#A23B72"},
        edgecolor="black",
        linewidth=0.8,
    )

    # Reference lines
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.6, label="100% Quality")
    ax.axhline(y=95, color="orange", linestyle="--", alpha=0.6, label="95% Quality")
    ax.axvline(x=1, color="red", linestyle="--", alpha=0.6, label="1x Speedup")

    ax.set_xlabel("Speedup (MILP Time / GA Time)", fontsize=12)
    ax.set_ylabel("GA Quality Retention (%)", fontsize=12)
    ax.set_title("Speedup vs Quality Tradeoff", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "tradeoff_scatter.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_grouped_comparison(df: pd.DataFrame, output_dir: str, figsize=(16, 5)):
    """Generate a 3-panel comparison showing time, quality, and fairness."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    df["Config"] = df["Model"] + "\n" + df["Group"]

    # Time comparison
    df_time = _prepare_long_format(
        df, {"MILP": "MILP_Time", "GA": "GA_Time"}, ["Config"]
    )
    sns.barplot(
        data=df_time,
        x="Config",
        y="Value",
        hue="Solver",
        palette={"MILP": "#2E86AB", "GA": "#A23B72"},
        ax=axes[0],
        edgecolor="black",
    )
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_xlabel("")
    axes[0].set_title("Execution Time", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(title="")

    # Quality comparison
    df_qual = _prepare_long_format(
        df, {"MILP": "MILP_NDCG", "GA": "GA_NDCG"}, ["Config"]
    )
    sns.barplot(
        data=df_qual,
        x="Config",
        y="Value",
        hue="Solver",
        palette={"MILP": "#2E86AB", "GA": "#A23B72"},
        ax=axes[1],
        edgecolor="black",
    )
    axes[1].set_ylabel("NDCG@10")
    axes[1].set_xlabel("")
    axes[1].set_title("Solution Quality", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    min_ndcg = min(df["MILP_NDCG"].min(), df["GA_NDCG"].min())
    axes[1].set_ylim(bottom=min_ndcg * 0.9)
    axes[1].get_legend().remove()

    # Fairness comparison
    df_ugf = _prepare_long_format(df, {"MILP": "MILP_UGF", "GA": "GA_UGF"}, ["Config"])
    sns.barplot(
        data=df_ugf,
        x="Config",
        y="Value",
        hue="Solver",
        palette={"MILP": "#2E86AB", "GA": "#A23B72"},
        ax=axes[2],
        edgecolor="black",
    )
    axes[2].set_ylabel("UGF")
    axes[2].set_xlabel("")
    axes[2].set_title("Fairness (UGF)", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].get_legend().remove()

    plt.suptitle(
        "MILP vs GA: Comprehensive Comparison", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, "grouped_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def plot_heatmap_summary(df: pd.DataFrame, output_dir: str, figsize=(10, 8)):
    """Generate a heatmap showing key metrics across experiments."""
    fig, ax = plt.subplots(figsize=figsize)

    df["Config"] = (
        df["Dataset"].str.replace("-rand", "")
        + " | "
        + df["Model"]
        + " | "
        + df["Group"]
    )

    # Create summary matrix
    summary_df = df[
        ["Config", "Speedup", "GA_Quality_Retention", "GA_Satisfied"]
    ].copy()
    summary_df["Constraint"] = summary_df["GA_Satisfied"].apply(
        lambda x: 100 if x else 0
    )
    summary_df = summary_df.drop(columns=["GA_Satisfied"])
    summary_df = summary_df.set_index("Config")
    summary_df.columns = ["Speedup (x)", "Quality Retention (%)", "Constraint Met (%)"]

    # Normalize for heatmap (min-max scaling per column)
    normalized = summary_df.copy()
    for col in normalized.columns:
        normalized[col] = (normalized[col] - normalized[col].min()) / (
            normalized[col].max() - normalized[col].min() + 1e-9
        )

    sns.heatmap(
        summary_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Value"},
    )

    ax.set_title("GA Performance Summary Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "heatmap_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def generate_summary_stats(df: pd.DataFrame) -> str:
    """Generate summary statistics as formatted text."""
    summary = []
    summary.append("=" * 60)
    summary.append("SUMMARY STATISTICS")
    summary.append("=" * 60)

    summary.append(f"\nTotal experiments: {len(df)}")
    summary.append(f"Datasets: {df['Dataset'].unique().tolist()}")
    summary.append(f"Models: {df['Model'].unique().tolist()}")
    summary.append(f"Groups: {df['Group'].unique().tolist()}")

    summary.append("\n--- Timing ---")
    summary.append(
        f"MILP Avg Time: {df['MILP_Time'].mean():.2f}s (std: {df['MILP_Time'].std():.2f})"
    )
    summary.append(
        f"GA Avg Time: {df['GA_Time'].mean():.2f}s (std: {df['GA_Time'].std():.2f})"
    )
    summary.append(
        f"Avg Speedup: {df['Speedup'].mean():.2f}x (range: {df['Speedup'].min():.2f}x - {df['Speedup'].max():.2f}x)"
    )

    summary.append("\n--- Quality ---")
    summary.append(f"MILP Avg NDCG: {df['MILP_NDCG'].mean():.4f}")
    summary.append(f"GA Avg NDCG: {df['GA_NDCG'].mean():.4f}")
    summary.append(
        f"Avg Quality Gap: {df['NDCG_Gap'].mean():.2f}% (range: {df['NDCG_Gap'].min():.2f}% to {df['NDCG_Gap'].max():.2f}%)"
    )
    summary.append(f"Avg Quality Retention: {df['GA_Quality_Retention'].mean():.2f}%")

    summary.append("\n--- Fairness ---")
    summary.append(f"MILP Avg UGF: {df['MILP_UGF'].mean():.6f}")
    summary.append(f"GA Avg UGF: {df['GA_UGF'].mean():.6f}")
    summary.append(f"Avg UGF Difference: {df['UGF_Diff'].mean():.6f}")
    summary.append(f"All GA constraints satisfied: {df['GA_Satisfied'].all()}")

    summary.append("\n" + "=" * 60)

    return "\n".join(summary)


def generate_latex_table(df: pd.DataFrame, output_dir: str) -> str:
    """Generate a LaTeX-formatted table for academic papers."""
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{MILP vs GA Comparison Results}")
    latex.append(r"\label{tab:comparison}")
    latex.append(r"\begin{tabular}{llccccc}")
    latex.append(r"\hline")
    latex.append(
        r"Model & Group & MILP Time & GA Time & Speedup & MILP NDCG & GA NDCG \\"
    )
    latex.append(r"\hline")

    for _, row in df.iterrows():
        latex.append(
            f"{row['Model']} & {row['Group']} & "
            f"{row['MILP_Time']:.1f}s & {row['GA_Time']:.1f}s & "
            f"{row['Speedup']:.2f}x & "
            f"{row['MILP_NDCG']:.4f} & {row['GA_NDCG']:.4f} \\\\"
        )

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)

    save_path = os.path.join(output_dir, "comparison_table.tex")
    with open(save_path, "w") as f:
        f.write(latex_str)
    print(f"Saved: {save_path}")

    return latex_str


def generate_all_plots(
    results_folder: str = "../results/comparison", output_dir: Optional[str] = None
):
    """Generate all plots and save to output directory."""

    if output_dir is None:
        output_dir = os.path.join(results_folder, "plots")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    df = load_all_results(results_folder)
    print(f"Loaded {len(df)} experiment results\n")

    print("Generating plots...")
    plot_time_comparison(df, output_dir)
    plot_quality_comparison(df, output_dir)
    plot_ugf_comparison(df, output_dir)
    plot_speedup_chart(df, output_dir)
    plot_tradeoff_scatter(df, output_dir)
    plot_grouped_comparison(df, output_dir)
    plot_heatmap_summary(df, output_dir)

    print("\nGenerating summary statistics...")
    stats = generate_summary_stats(df)
    print(stats)

    stats_path = os.path.join(output_dir, "summary_stats.txt")
    with open(stats_path, "w") as f:
        f.write(stats)
    print(f"\nSaved: {stats_path}")

    print("\nGenerating LaTeX table...")
    generate_latex_table(df, output_dir)

    print(f"\n✓ All plots saved to: {output_dir}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots from MILP vs GA comparison results"
    )
    parser.add_argument(
        "--results",
        "-r",
        default="../results/comparison",
        help="Path to results folder containing CSV files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for plots (default: results/comparison/plots)",
    )

    args = parser.parse_args()

    generate_all_plots(args.results, args.output)
