#!/usr/bin/env python3
"""
Generate publication-quality figures.

Creates:
- Fig 1: O/E by feature type (extension vs truncation)
- Fig 2: O/E by length quartile
- Fig 3: O/E by start codon
- Fig 4: Top constrained features (isoform importance)
- Fig 5: O/E by LOEUF decile
- Fig 6: Within-gene ratio distribution
- Fig 7: Feature O/E vs Canonical O/E scatter
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_project_root, oe_poisson_ci


# Publication style settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def fig1_by_feature_type(df: pd.DataFrame, output_dir: Path) -> None:
    """O/E by feature type with confidence intervals."""
    fig, ax = plt.subplots(figsize=(4, 4))

    results = []
    for ft in ["extension", "truncation"]:
        subset = df[df["feature_type"] == ft]
        obs = subset["observed_missense"].sum()
        exp = subset["expected_missense"].sum()
        oe, lower, upper = oe_poisson_ci(int(obs), exp)
        results.append({
            "type": ft.title(),
            "n": len(subset),
            "oe": oe,
            "lower": lower,
            "upper": upper
        })

    results = pd.DataFrame(results)

    colors = ["#4C72B0", "#DD8452"]
    x = range(len(results))

    bars = ax.bar(
        x, results["oe"],
        yerr=[results["oe"] - results["lower"], results["upper"] - results["oe"]],
        capsize=5, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85
    )

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['type']}\n(n={r['n']:,})" for _, r in results.iterrows()])
    ax.set_ylabel("Missense O/E")
    ax.set_ylim(0.8, 1.05)
    ax.set_title("Constraint by Feature Type")

    # Add values on bars
    for i, (_, r) in enumerate(results.iterrows()):
        ax.text(i, r["oe"] + 0.02, f"{r['oe']:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_by_type.pdf")
    fig.savefig(output_dir / "fig1_by_type.png")
    plt.close()
    print("  Created fig1_by_type")


def fig2_by_length(df: pd.DataFrame, output_dir: Path) -> None:
    """O/E by length quartile, grouped by feature type."""
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = {"extension": "#4C72B0", "truncation": "#DD8452"}
    width = 0.35

    for i, ft in enumerate(["extension", "truncation"]):
        subset = df[df["feature_type"] == ft].copy()
        subset["length_q"] = pd.qcut(
            subset["feature_length_aa"], q=4,
            labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        )

        results = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            q_data = subset[subset["length_q"] == q]
            if len(q_data) == 0:
                continue
            obs = q_data["observed_missense"].sum()
            exp = q_data["expected_missense"].sum()
            oe, lower, upper = oe_poisson_ci(int(obs), exp)
            results.append({"q": q, "oe": oe, "lower": lower, "upper": upper})

        results = pd.DataFrame(results)
        x = np.arange(len(results)) + i * width

        ax.bar(
            x, results["oe"],
            width=width,
            yerr=[results["oe"] - results["lower"], results["upper"] - results["oe"]],
            capsize=3, color=colors[ft], edgecolor="black", linewidth=0.5,
            alpha=0.85, label=ft.title()
        )

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(np.arange(4) + width / 2)
    ax.set_xticklabels(["Q1\n(shortest)", "Q2", "Q3", "Q4\n(longest)"])
    ax.set_ylabel("Missense O/E")
    ax.set_xlabel("Length Quartile")
    ax.set_ylim(0.8, 1.05)
    ax.set_title("Constraint by Length")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_by_length.pdf")
    fig.savefig(output_dir / "fig2_by_length.png")
    plt.close()
    print("  Created fig2_by_length")


def fig3_by_codon(df: pd.DataFrame, output_dir: Path) -> None:
    """O/E by start codon type."""
    fig, ax = plt.subplots(figsize=(7, 4))

    results = []
    for codon in df["alternative_start_codon"].dropna().unique():
        subset = df[df["alternative_start_codon"] == codon]
        obs = subset["observed_missense"].sum()
        exp = subset["expected_missense"].sum()
        if exp > 0:
            oe, lower, upper = oe_poisson_ci(int(obs), exp)
            results.append({
                "codon": codon,
                "n": len(subset),
                "oe": oe,
                "lower": lower,
                "upper": upper
            })

    results = pd.DataFrame(results).sort_values("oe")
    x = range(len(results))

    # Color by constraint level
    colors = ["#C44E52" if oe < 0.9 else "#4C72B0" if oe < 0.95 else "#55A868"
              for oe in results["oe"]]

    ax.barh(
        x, results["oe"],
        xerr=[results["oe"] - results["lower"], results["upper"] - results["oe"]],
        capsize=3, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85
    )

    ax.axvline(x=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{r['codon']} (n={r['n']})" for _, r in results.iterrows()])
    ax.set_xlabel("Missense O/E")
    ax.set_title("Constraint by Start Codon")
    ax.set_xlim(0.8, 1.1)

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_by_codon.pdf")
    fig.savefig(output_dir / "fig3_by_codon.png")
    plt.close()
    print("  Created fig3_by_codon")


def fig4_isoform_importance(df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    """
    4-panel figure showing biological interpretation of constraint.

    - Extensions low O/E: Extended form functional
    - Extensions high O/E: Canonical form functional
    - Truncations low O/E: Canonical form functional
    - Truncations high O/E: Truncated form functional
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    panels = [
        # (ax position, feature_type, sort_ascending, title, interpretation)
        ((0, 0), "extension", True, "Extensions: Extended Form Functional",
         "Low O/E → extension is constrained"),
        ((0, 1), "extension", False, "Extensions: Canonical Form Functional",
         "High O/E → extension is dispensable"),
        ((1, 0), "truncation", True, "Truncations: Canonical Form Functional",
         "Low O/E → N-terminus is essential"),
        ((1, 1), "truncation", False, "Truncations: Truncated Form Functional",
         "High O/E → N-terminus is dispensable"),
    ]

    for (row, col), ft, ascending, title, interpretation in panels:
        ax = axes[row, col]
        subset = df[(df["feature_type"] == ft) & (df["oe_missense"].notna())]

        if ascending:
            top = subset.nsmallest(top_n, "oe_missense")
            color = "#4C72B0"  # Blue for constrained
        else:
            top = subset.nlargest(top_n, "oe_missense")
            color = "#DD8452"  # Orange for relaxed

        y = range(len(top))

        ax.barh(
            y, top["oe_missense"],
            xerr=[
                top["oe_missense"] - top["oe_missense_lower"],
                top["oe_missense_upper"] - top["oe_missense"]
            ],
            capsize=2, color=color, edgecolor="black", linewidth=0.3, alpha=0.85
        )

        ax.axvline(x=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(top["gene_name"], fontsize=8)
        ax.set_xlabel("Missense O/E")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.invert_yaxis()

        # Set appropriate x-axis limits
        if ascending:
            ax.set_xlim(0, 1.3)
        else:
            ax.set_xlim(0.5, max(top["oe_missense"].max() * 1.1, 1.5))

        # Add interpretation as subtitle
        ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
                fontsize=8, fontstyle="italic", va="top", color="gray")

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_isoform_importance.pdf")
    fig.savefig(output_dir / "fig4_isoform_importance.png")
    plt.close()
    print("  Created fig4_isoform_importance")


def fig5_by_loeuf(df: pd.DataFrame, output_dir: Path) -> None:
    """O/E by LOEUF decile ."""
    if "loeuf" not in df.columns:
        print("  Skipping fig5_by_loeuf: no LOEUF data")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    # Create LOEUF deciles
    df_valid = df[df["loeuf"].notna()].copy()
    if len(df_valid) < 100:
        print("  Skipping fig5_by_loeuf: insufficient data")
        return

    df_valid["loeuf_decile"] = pd.qcut(df_valid["loeuf"], q=10, labels=False, duplicates="drop") + 1

    results = []
    for decile in sorted(df_valid["loeuf_decile"].unique()):
        subset = df_valid[df_valid["loeuf_decile"] == decile]
        obs = subset["observed_missense"].sum()
        exp = subset["expected_missense"].sum()
        if exp > 0:
            oe, lower, upper = oe_poisson_ci(int(obs), exp)
            loeuf_median = subset["loeuf"].median()
            results.append({
                "decile": int(decile),
                "loeuf_median": loeuf_median,
                "oe": oe,
                "lower": lower,
                "upper": upper,
                "n": len(subset)
            })

    if not results:
        return

    results = pd.DataFrame(results)
    x = results["decile"]

    # Color by LOEUF (low = constrained gene = darker)
    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(results)))

    ax.bar(
        x, results["oe"],
        yerr=[results["oe"] - results["lower"], results["upper"] - results["oe"]],
        capsize=3, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85
    )

    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in x])
    ax.set_xlabel("LOEUF Decile (D1=most constrained genes, D10=least)")
    ax.set_ylabel("Feature Missense O/E")
    ax.set_title("Feature Constraint by Gene-Level Constraint (LOEUF)")

    # Add trend line
    from scipy import stats
    x_arr = results["decile"].values
    slope, intercept, r, p, se = stats.linregress(x_arr, results["oe"].values)
    ax.plot(x_arr, slope * x_arr + intercept, "r--", linewidth=1.5, alpha=0.7,
            label=f"Trend: r={r:.2f}, p={p:.2e}")
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_by_loeuf.pdf")
    fig.savefig(output_dir / "fig5_by_loeuf.png")
    plt.close()
    print("  Created fig5_by_loeuf")


def fig6_within_gene_ratio(df: pd.DataFrame, output_dir: Path) -> None:
    """Distribution of within-gene O/E ratios ."""
    if "canonical_oe_mis" not in df.columns:
        print("  Skipping fig6_within_gene_ratio: no canonical O/E data")
        return

    # Calculate ratio if not present
    df_valid = df.copy()
    if "oe_ratio" not in df_valid.columns:
        df_valid["feature_oe"] = df_valid["observed_missense"] / df_valid["expected_missense"]
        df_valid["oe_ratio"] = df_valid["feature_oe"] / df_valid["canonical_oe_mis"]

    df_valid = df_valid[df_valid["oe_ratio"].notna() & (df_valid["oe_ratio"] < 10)]

    if len(df_valid) < 50:
        print("  Skipping fig6_within_gene_ratio: insufficient data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Histogram
    ax = axes[0]
    for ft, color in [("extension", "#4C72B0"), ("truncation", "#DD8452")]:
        subset = df_valid[df_valid["feature_type"] == ft]["oe_ratio"]
        if len(subset) > 0:
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=ft.title(), density=True)

    ax.axvline(x=1, color="red", linestyle="--", linewidth=2, label="Ratio = 1")
    ax.set_xlabel("O/E Ratio (Feature / Canonical)")
    ax.set_ylabel("Density")
    ax.set_title("Within-Gene Constraint Comparison")
    ax.legend()
    ax.set_xlim(0, 3)

    # Panel B: By feature type (violin plot - more robust)
    ax = axes[1]
    colors = {"extension": "#4C72B0", "truncation": "#DD8452"}

    for i, ft in enumerate(["extension", "truncation"]):
        data = df_valid[df_valid["feature_type"] == ft]["oe_ratio"].dropna().values
        if len(data) > 0:
            parts = ax.violinplot([data], positions=[i+1], showmeans=False, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[ft])
                pc.set_alpha(0.6)
            parts["cmedians"].set_color("black")

            median = np.median(data)
            ax.annotate(f"median={median:.2f}", xy=(i+1.1, median),
                       fontsize=8, va="center")

    ax.axhline(y=1, color="red", linestyle="--", linewidth=1.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Extension", "Truncation"])
    ax.set_ylabel("O/E Ratio (Feature / Canonical)")
    ax.set_title("Ratio Distribution by Type")
    ax.set_ylim(0, 3)

    plt.tight_layout()
    fig.savefig(output_dir / "fig6_within_gene_ratio.pdf")
    fig.savefig(output_dir / "fig6_within_gene_ratio.png")
    plt.close()
    print("  Created fig6_within_gene_ratio")


def fig7_scatter_feature_vs_canonical(df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of feature O/E vs canonical O/E ."""
    if "canonical_oe_mis" not in df.columns:
        print("  Skipping fig7_scatter: no canonical O/E data")
        return

    df_valid = df.copy()
    if "feature_oe" not in df_valid.columns:
        df_valid["feature_oe"] = df_valid["observed_missense"] / df_valid["expected_missense"]

    # Filter to valid values
    df_valid = df_valid[
        df_valid["feature_oe"].notna() &
        df_valid["canonical_oe_mis"].notna() &
        (df_valid["feature_oe"] < 5) &
        (df_valid["canonical_oe_mis"] < 2)
    ]

    if len(df_valid) < 50:
        print("  Skipping fig7_scatter: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    colors = {"extension": "#4C72B0", "truncation": "#DD8452"}
    for ft in ["extension", "truncation"]:
        subset = df_valid[df_valid["feature_type"] == ft]
        ax.scatter(
            subset["canonical_oe_mis"], subset["feature_oe"],
            c=colors[ft], alpha=0.4, s=15, label=ft.title()
        )

    # Diagonal line (equal constraint)
    lims = [0, min(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Equal constraint")

    ax.set_xlabel("Canonical CDS O/E (gnomAD)")
    ax.set_ylabel("Feature O/E (extension/truncation)")
    ax.set_title("Feature vs Canonical Constraint")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 3)

    # Add annotation
    n_below = (df_valid["feature_oe"] < df_valid["canonical_oe_mis"]).sum()
    n_above = (df_valid["feature_oe"] > df_valid["canonical_oe_mis"]).sum()
    pct_below = 100 * n_below / len(df_valid)
    ax.text(0.95, 0.05, f"{pct_below:.0f}% below diagonal\n(more constrained than canonical)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "fig7_scatter_oe.pdf")
    fig.savefig(output_dir / "fig7_scatter_oe.png")
    plt.close()
    print("  Created fig7_scatter_oe")


def main(config_path: str = "config.yaml"):
    """Generate all figures."""
    config = load_config(config_path)
    project_root = get_project_root()

    # Try to load merged data (with gnomAD) first
    merged_path = project_root / "results" / "features_with_gnomad.csv"
    oe_path = project_root / "results" / "oe_per_feature.csv"

    if merged_path.exists():
        df = pd.read_csv(merged_path)
        print(f"Loaded {len(df)} features (with gnomAD data)")
    elif oe_path.exists():
        df = pd.read_csv(oe_path)
        print(f"Loaded {len(df)} features")
    else:
        print(f"ERROR: No input data found. Run earlier steps first.")
        return

    # Output directory
    output_dir = project_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating figures...")

    # Original figures
    fig1_by_feature_type(df, output_dir)
    fig2_by_length(df, output_dir)
    if "alternative_start_codon" in df.columns:
        fig3_by_codon(df, output_dir)
    if "oe_missense" in df.columns:
        fig4_isoform_importance(df, output_dir, top_n=15)

    # New V2 figures
    fig5_by_loeuf(df, output_dir)
    fig6_within_gene_ratio(df, output_dir)
    fig7_scatter_feature_vs_canonical(df, output_dir)

    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    args = parser.parse_args()
    main(args.config)
