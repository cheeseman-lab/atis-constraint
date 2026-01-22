#!/usr/bin/env python3
"""
Visualize basic constraint metrics for aTIS regions vs canonical CDS.

Creates exploratory plots to understand the data before statistical testing.

Output: results/figures/
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Set style
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def plot_oe_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot O/E ratio distributions for aTIS vs canonical.
    """
    print("\nPlotting O/E distributions...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ("oe_mis", "Missense O/E"),
        ("oe_syn", "Synonymous O/E"),
        ("oe_lof", "LoF O/E"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        atis_col = f"aTIS_{metric}"
        canonical_col = f"canonical_{metric}"

        # Get data
        atis_vals = df[atis_col].dropna()
        canonical_vals = df[canonical_col].dropna()

        # Plot histograms
        ax.hist(
            canonical_vals,
            bins=50,
            alpha=0.6,
            label="Canonical",
            color="blue",
            density=True,
        )
        ax.hist(atis_vals, bins=50, alpha=0.6, label="aTIS", color="red", density=True)

        # Add median lines
        ax.axvline(
            canonical_vals.median(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Canonical median: {canonical_vals.median():.2f}",
        )
        ax.axvline(
            atis_vals.median(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"aTIS median: {atis_vals.median():.2f}",
        )

        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=8)

        # Limit x-axis for better visualization
        if metric == "oe_lof":
            ax.set_xlim(0, 10)
        else:
            ax.set_xlim(0, 5)

    plt.tight_layout()
    plt.savefig(output_dir / "01_oe_distributions.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 01_oe_distributions.png")


def plot_density_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot variant density distributions for aTIS vs canonical.
    """
    print("\nPlotting density distributions...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ("density_mis", "Missense density (per AA)"),
        ("density_syn", "Synonymous density (per AA)"),
        ("density_lof", "LoF density (per AA)"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        atis_col = f"aTIS_{metric}"
        canonical_col = f"canonical_{metric}"

        # Get data
        atis_vals = df[atis_col].dropna()
        canonical_vals = df[canonical_col].dropna()

        # Plot histograms
        ax.hist(
            canonical_vals,
            bins=50,
            alpha=0.6,
            label="Canonical",
            color="blue",
            density=True,
        )
        ax.hist(atis_vals, bins=50, alpha=0.6, label="aTIS", color="red", density=True)

        # Add median lines
        ax.axvline(
            canonical_vals.median(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Canonical median: {canonical_vals.median():.2f}",
        )
        ax.axvline(
            atis_vals.median(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"aTIS median: {atis_vals.median():.2f}",
        )

        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=8)

        # Limit x-axis for better visualization
        if metric == "density_lof":
            ax.set_xlim(0, 0.5)
        else:
            ax.set_xlim(0, 3)

    plt.tight_layout()
    plt.savefig(output_dir / "02_density_distributions.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 02_density_distributions.png")


def plot_scatter_oe(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Scatter plots of aTIS vs canonical O/E ratios.
    """
    print("\nPlotting O/E scatter plots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("oe_mis", "Missense O/E"),
        ("oe_syn", "Synonymous O/E"),
        ("oe_lof", "LoF O/E"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        atis_col = f"aTIS_{metric}"
        canonical_col = f"canonical_{metric}"

        x = df[canonical_col]
        y = df[atis_col]

        # Scatter plot with alpha for overlapping points
        ax.scatter(x, y, alpha=0.3, s=10, color="black")

        # Add diagonal line (y=x)
        max_val = max(x.max(), y.max())
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="y=x")

        ax.set_xlabel(f"Canonical {label}")
        ax.set_ylabel(f"aTIS {label}")
        ax.set_title(f"{label}: aTIS vs Canonical")
        ax.legend()

        # Set equal limits for both axes
        if metric == "oe_lof":
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 15)
        else:
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 8)

    plt.tight_layout()
    plt.savefig(output_dir / "03_oe_scatter.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 03_oe_scatter.png")


def plot_delta_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot distributions of paired differences (aTIS - canonical).
    """
    print("\nPlotting delta distributions...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # O/E deltas
    oe_metrics = [
        ("delta_oe_mis", "Δ Missense O/E\n(aTIS - canonical)"),
        ("delta_oe_syn", "Δ Synonymous O/E\n(aTIS - canonical)"),
        ("delta_oe_lof", "Δ LoF O/E\n(aTIS - canonical)"),
    ]

    # Density deltas
    density_metrics = [
        ("delta_density_mis", "Δ Missense density\n(aTIS - canonical)"),
        ("delta_density_syn", "Δ Synonymous density\n(aTIS - canonical)"),
        ("delta_density_lof", "Δ LoF density\n(aTIS - canonical)"),
    ]

    for ax, (metric, label) in zip(axes[0], oe_metrics):
        vals = df[metric].dropna()

        ax.hist(vals, bins=50, alpha=0.7, color="purple", edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No difference")
        ax.axvline(
            vals.median(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Median: {vals.median():.2f}",
        )

        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=8)

        # Show percentage < 0
        pct_negative = 100 * (vals < 0).mean()
        ax.text(
            0.05,
            0.95,
            f"{pct_negative:.1f}% < 0",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    for ax, (metric, label) in zip(axes[1], density_metrics):
        vals = df[metric].dropna()

        ax.hist(vals, bins=50, alpha=0.7, color="green", edgecolor="black")
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No difference")
        ax.axvline(
            vals.median(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Median: {vals.median():.2f}",
        )

        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend(fontsize=8)

        # Show percentage < 0
        pct_negative = 100 * (vals < 0).mean()
        ax.text(
            0.05,
            0.95,
            f"{pct_negative:.1f}% < 0",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_dir / "04_delta_distributions.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 04_delta_distributions.png")


def plot_feature_type_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Compare extensions vs truncations.
    """
    print("\nPlotting feature type comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("delta_oe_mis", "Δ Missense O/E"),
        ("delta_oe_syn", "Δ Synonymous O/E"),
        ("delta_oe_lof", "Δ LoF O/E"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        # Prepare data for box plot
        extensions = df[df["feature_type"] == "extension"][metric].dropna().values
        truncations = df[df["feature_type"] == "truncation"][metric].dropna().values

        # Box plot
        bp = ax.boxplot(
            [list(extensions), list(truncations)],
            labels=[
                "Extensions\n(n={})".format(len(extensions)),
                "Truncations\n(n={})".format(len(truncations)),
            ],
            patch_artist=True,
            showmeans=True,
        )

        # Color the boxes
        colors = ["lightblue", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(label)
        ax.set_title(label)
        ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "05_feature_type_comparison.png", bbox_inches="tight")
    plt.close()
    print("  Saved: 05_feature_type_comparison.png")


def print_summary_statistics(df: pd.DataFrame) -> None:
    """
    Print summary statistics for the metrics.
    """
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print("\naTIS vs Canonical O/E ratios:")
    for metric_type in ["mis", "syn", "lof"]:
        atis = df[f"aTIS_oe_{metric_type}"]
        canonical = df[f"canonical_oe_{metric_type}"]
        delta = df[f"delta_oe_{metric_type}"]

        print(f"\n{metric_type.upper()}:")
        print(f"  aTIS median: {atis.median():.3f}")
        print(f"  Canonical median: {canonical.median():.3f}")
        print(f"  Delta median: {delta.median():.3f}")
        print(
            f"  aTIS < canonical: {(delta < 0).sum()} / {len(delta)} ({100 * (delta < 0).mean():.1f}%)"
        )

    print("\naTIS vs Canonical densities:")
    for metric_type in ["mis", "syn", "lof"]:
        atis = df[f"aTIS_density_{metric_type}"]
        canonical = df[f"canonical_density_{metric_type}"]
        delta = df[f"delta_density_{metric_type}"]

        print(f"\n{metric_type.upper()}:")
        print(f"  aTIS median: {atis.median():.3f}")
        print(f"  Canonical median: {canonical.median():.3f}")
        print(f"  Delta median: {delta.median():.3f}")
        print(
            f"  aTIS < canonical: {(delta < 0).sum()} / {len(delta)} ({100 * (delta < 0).mean():.1f}%)"
        )


def generate_plots_for_subset(
    df: pd.DataFrame, output_dir: Path, subset_name: str
) -> None:
    """
    Generate all plots for a data subset.

    Parameters
    ----------
    df : pd.DataFrame
        Data to plot
    output_dir : Path
        Output directory
    subset_name : str
        Name of subset (e.g., "all", "extensions", "truncations")
    """
    print(f"\n{'=' * 60}")
    print(f"PLOTTING: {subset_name.upper()} (n={len(df)})")
    print(f"{'=' * 60}")

    # Create subdirectory for this subset
    subset_dir = output_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    plot_oe_distributions(df, subset_dir)
    plot_density_distributions(df, subset_dir)
    plot_scatter_oe(df, subset_dir)
    plot_delta_distributions(df, subset_dir)

    print(f"  ✓ Plots saved to: {subset_dir}")


def main():
    """Main pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "features_with_metrics.csv"
    output_dir = project_root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} features")
    print(f"  Extensions: {(df['feature_type'] == 'extension').sum()}")
    print(f"  Truncations: {(df['feature_type'] == 'truncation').sum()}")

    # Print summary statistics for all features
    print_summary_statistics(df)

    # Generate plots for all features
    generate_plots_for_subset(df, output_dir, "all")

    # Generate plots for extensions only
    extensions = df[df["feature_type"] == "extension"].copy()
    generate_plots_for_subset(extensions, output_dir, "extensions")

    # Generate plots for truncations only
    truncations = df[df["feature_type"] == "truncation"].copy()
    generate_plots_for_subset(truncations, output_dir, "truncations")

    print(f"\n{'=' * 60}")
    print("✓ ALL PLOTS COMPLETE")
    print(f"{'=' * 60}")
    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── all/         - All features (n={len(df)})")
    print(f"  ├── extensions/  - Extensions only (n={len(extensions)})")
    print(f"  └── truncations/ - Truncations only (n={len(truncations)})")
    print("\nPlots per directory:")
    print("  01_oe_distributions.png")
    print("  02_density_distributions.png")
    print("  03_oe_scatter.png")
    print("  04_delta_distributions.png")


if __name__ == "__main__":
    main()
