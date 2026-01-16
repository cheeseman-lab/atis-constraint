#!/usr/bin/env python3
"""
LOEUF stratification analysis.

Tests whether aTIS constraint correlates with gene-level constraint (LOEUF).
Prediction: Extensions in highly constrained genes (low LOEUF) should show
stronger depletion than extensions in unconstrained genes (high LOEUF).

Reference: Wieder et al. 2024 found 5'UTR complexity correlates with LOEUF.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_project_root, oe_poisson_ci


def load_merged_data(config: dict) -> pd.DataFrame:
    """Load feature data with gnomAD constraint metrics."""
    project_root = get_project_root()
    input_file = project_root / "results" / "features_with_gnomad.csv"

    if not input_file.exists():
        print(f"ERROR: {input_file} not found")
        print("Please run 04_merge_gnomad_constraint.py first")
        sys.exit(1)

    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Total features: {len(df)}")

    # Filter to features with LOEUF
    df_with_loeuf = df[df["loeuf"].notna()].copy()
    print(f"  Features with LOEUF: {len(df_with_loeuf)}")

    return df_with_loeuf


def create_loeuf_bins(
    df: pd.DataFrame,
    n_bins: int = 10,
    method: str = "decile"
) -> pd.DataFrame:
    """
    Bin genes by LOEUF score.

    Parameters
    ----------
    df : pd.DataFrame
        Data with LOEUF column
    n_bins : int
        Number of bins (default 10 for deciles)
    method : str
        "decile" for quantile-based, "fixed" for fixed thresholds

    Returns
    -------
    pd.DataFrame
        Data with loeuf_bin column added
    """
    result = df.copy()

    if method == "decile":
        # Quantile-based bins
        result["loeuf_bin"] = pd.qcut(
            result["loeuf"],
            q=n_bins,
            labels=[f"D{i+1}" for i in range(n_bins)],
            duplicates="drop"
        )
        result["loeuf_decile"] = pd.qcut(
            result["loeuf"],
            q=n_bins,
            labels=False,
            duplicates="drop"
        ) + 1

    elif method == "fixed":
        # Fixed thresholds based on gnomAD recommendations
        bins = [0, 0.35, 0.6, 1.0, float("inf")]
        labels = ["Highly constrained", "Constrained", "Intermediate", "Unconstrained"]
        result["loeuf_bin"] = pd.cut(result["loeuf"], bins=bins, labels=labels)

    return result


def calculate_oe_by_loeuf_bin(
    df: pd.DataFrame,
    ci_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate aggregate O/E for each LOEUF bin.

    Parameters
    ----------
    df : pd.DataFrame
        Data with loeuf_bin and variant counts
    ci_level : float
        Confidence level for CIs

    Returns
    -------
    pd.DataFrame
        O/E statistics per LOEUF bin
    """
    results = []

    # Get variant count columns
    mis_col = "observed_missense" if "observed_missense" in df.columns else "count_gnomad_missense_variant"
    syn_col = "observed_synonymous" if "observed_synonymous" in df.columns else "count_gnomad_synonymous_variant"
    exp_col = "expected_missense"

    for bin_label in sorted(df["loeuf_bin"].dropna().unique()):
        subset = df[df["loeuf_bin"] == bin_label]

        # Aggregate counts
        total_mis = subset[mis_col].sum()
        total_syn = subset[syn_col].sum()

        # Calculate expected if not present
        if exp_col in subset.columns:
            total_exp = subset[exp_col].sum()
        else:
            total_exp = total_syn * 2.5  # genome-wide mis/syn ratio

        # O/E with CI
        oe, lower, upper = oe_poisson_ci(int(total_mis), total_exp, ci_level)

        # LOEUF range for this bin
        loeuf_min = subset["loeuf"].min()
        loeuf_max = subset["loeuf"].max()
        loeuf_median = subset["loeuf"].median()

        results.append({
            "loeuf_bin": bin_label,
            "loeuf_range": f"{loeuf_min:.2f}-{loeuf_max:.2f}",
            "loeuf_median": round(loeuf_median, 3),
            "n_features": len(subset),
            "n_genes": subset["gene_name"].nunique() if "gene_name" in subset.columns else len(subset),
            "total_missense": int(total_mis),
            "total_synonymous": int(total_syn),
            "expected_missense": round(total_exp, 1),
            "oe_missense": round(oe, 4) if not np.isnan(oe) else np.nan,
            "oe_lower": round(lower, 4) if not np.isnan(lower) else np.nan,
            "oe_upper": round(upper, 4) if not np.isnan(upper) else np.nan,
        })

    return pd.DataFrame(results)


def test_loeuf_trend(
    df: pd.DataFrame,
    oe_by_bin: pd.DataFrame
) -> dict:
    """
    Test for trend between LOEUF and extension constraint.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-level data with LOEUF
    oe_by_bin : pd.DataFrame
        Bin-level O/E results

    Returns
    -------
    dict
        Trend test results
    """
    results = {}

    # 1. Spearman correlation at feature level (LOEUF vs extension O/E)
    if "oe_missense" in df.columns:
        valid = df[["loeuf", "oe_missense"]].dropna()
        if len(valid) > 10:
            rho, p_val = stats.spearmanr(valid["loeuf"], valid["oe_missense"])
            results["feature_level"] = {
                "n": len(valid),
                "spearman_rho": round(rho, 4),
                "p_value": p_val,
                "interpretation": "Higher LOEUF (less constrained genes) → higher extension O/E" if rho > 0 else "Higher LOEUF → lower extension O/E"
            }

    # 2. Bin-level correlation (LOEUF median vs bin O/E)
    valid_bins = oe_by_bin[oe_by_bin["oe_missense"].notna()]
    if len(valid_bins) >= 5:
        rho, p_val = stats.spearmanr(valid_bins["loeuf_median"], valid_bins["oe_missense"])
        results["bin_level"] = {
            "n_bins": len(valid_bins),
            "spearman_rho": round(rho, 4),
            "p_value": p_val,
        }

    # 3. Jonckheere-Terpstra test for ordered alternative
    # Tests if O/E increases monotonically with LOEUF
    if "loeuf_decile" in df.columns and "oe_missense" in df.columns:
        valid = df[["loeuf_decile", "oe_missense"]].dropna()
        if len(valid) > 20:
            # Group O/E values by decile
            groups = [valid[valid["loeuf_decile"] == d]["oe_missense"].values
                      for d in sorted(valid["loeuf_decile"].unique())]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) >= 3:
                # Use Kruskal-Wallis as approximation (scipy doesn't have J-T)
                stat, p_val = stats.kruskal(*groups)
                results["kruskal_wallis"] = {
                    "statistic": round(stat, 4),
                    "p_value": p_val,
                    "interpretation": "Significant difference in O/E across LOEUF deciles" if p_val < 0.05 else "No significant difference"
                }

    return results


def analyze_by_feature_type(
    df: pd.DataFrame,
    ci_level: float = 0.95
) -> dict:
    """
    Run LOEUF stratification separately for extensions and truncations.

    Parameters
    ----------
    df : pd.DataFrame
        Data with feature_type column
    ci_level : float
        Confidence level

    Returns
    -------
    dict
        Results by feature type
    """
    results = {}

    if "feature_type" not in df.columns:
        return results

    for feature_type in df["feature_type"].unique():
        subset = df[df["feature_type"] == feature_type]
        if len(subset) < 50:
            continue

        # Bin by LOEUF
        binned = create_loeuf_bins(subset, n_bins=5, method="decile")

        # Calculate O/E by bin
        oe_by_bin = calculate_oe_by_loeuf_bin(binned, ci_level)

        # Test trend
        trend = test_loeuf_trend(binned, oe_by_bin)

        results[feature_type] = {
            "n_features": len(subset),
            "oe_by_bin": oe_by_bin,
            "trend_tests": trend
        }

    return results


def print_results(
    oe_by_bin: pd.DataFrame,
    trend_tests: dict,
    by_type: dict
) -> None:
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("LOEUF STRATIFICATION RESULTS")
    print("=" * 70)

    print("\n## Overall O/E by LOEUF Decile\n")
    print(oe_by_bin.to_string(index=False))

    print("\n## Trend Tests\n")
    if "feature_level" in trend_tests:
        t = trend_tests["feature_level"]
        print(f"Feature-level correlation (LOEUF vs extension O/E):")
        print(f"  N: {t['n']}")
        print(f"  Spearman rho: {t['spearman_rho']:.4f}")
        print(f"  P-value: {t['p_value']:.2e}")
        print(f"  Interpretation: {t['interpretation']}")

    if "bin_level" in trend_tests:
        t = trend_tests["bin_level"]
        print(f"\nBin-level correlation:")
        print(f"  N bins: {t['n_bins']}")
        print(f"  Spearman rho: {t['spearman_rho']:.4f}")
        print(f"  P-value: {t['p_value']:.2e}")

    if "kruskal_wallis" in trend_tests:
        t = trend_tests["kruskal_wallis"]
        print(f"\nKruskal-Wallis test:")
        print(f"  Statistic: {t['statistic']:.4f}")
        print(f"  P-value: {t['p_value']:.2e}")
        print(f"  {t['interpretation']}")

    # By feature type
    if by_type:
        print("\n" + "=" * 70)
        print("## Results by Feature Type")
        print("=" * 70)

        for ft, data in by_type.items():
            print(f"\n### {ft.upper()} (n={data['n_features']})\n")
            print(data["oe_by_bin"].to_string(index=False))

            if "feature_level" in data["trend_tests"]:
                t = data["trend_tests"]["feature_level"]
                print(f"\nTrend: rho={t['spearman_rho']:.3f}, p={t['p_value']:.2e}")


def save_results(
    oe_by_bin: pd.DataFrame,
    trend_tests: dict,
    by_type: dict,
    output_dir: Path
) -> None:
    """Save results to files."""

    # Save O/E by LOEUF bin
    oe_by_bin.to_csv(output_dir / "loeuf_oe_by_decile.csv", index=False)

    # Save summary to markdown
    summary_path = output_dir / "loeuf_stratification_summary.md"
    with open(summary_path, "w") as f:
        f.write("# LOEUF Stratification Results\n\n")

        f.write("## Hypothesis\n\n")
        f.write("If aTIS extensions are functional, extensions in highly constrained genes ")
        f.write("(low LOEUF) should show stronger constraint than extensions in ")
        f.write("unconstrained genes (high LOEUF).\n\n")

        f.write("## O/E by LOEUF Decile\n\n")
        f.write("| Decile | LOEUF Range | N | O/E | 95% CI |\n")
        f.write("|--------|-------------|---|-----|--------|\n")
        for _, row in oe_by_bin.iterrows():
            f.write(f"| {row['loeuf_bin']} | {row['loeuf_range']} | ")
            f.write(f"{row['n_features']} | {row['oe_missense']:.3f} | ")
            f.write(f"{row['oe_lower']:.3f}-{row['oe_upper']:.3f} |\n")
        f.write("\n")

        f.write("## Trend Analysis\n\n")
        if "feature_level" in trend_tests:
            t = trend_tests["feature_level"]
            f.write(f"- **Spearman correlation**: rho = {t['spearman_rho']:.4f}, ")
            f.write(f"p = {t['p_value']:.2e}\n")
            f.write(f"- **Interpretation**: {t['interpretation']}\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        if "feature_level" in trend_tests:
            rho = trend_tests["feature_level"]["spearman_rho"]
            p = trend_tests["feature_level"]["p_value"]
            if p < 0.05 and rho > 0:
                f.write("**Positive correlation detected**: Extensions in constrained genes ")
                f.write("(low LOEUF) show lower O/E (stronger constraint) than extensions ")
                f.write("in unconstrained genes. This supports functional importance of aTIS extensions.\n")
            elif p < 0.05 and rho < 0:
                f.write("**Negative correlation detected**: Extensions in constrained genes ")
                f.write("show higher O/E than extensions in unconstrained genes. ")
                f.write("This is unexpected and warrants further investigation.\n")
            else:
                f.write("**No significant correlation**: Extension constraint does not ")
                f.write("correlate with gene-level constraint (LOEUF).\n")

    print(f"\nSaved results to: {output_dir}")


def main(config_path: str = "config.yaml"):
    """Main function for LOEUF stratification analysis."""
    config = load_config(config_path)
    project_root = get_project_root()
    ci_level = config["analysis"].get("ci_level", 0.95)

    # Load data
    df = load_merged_data(config)

    # Create LOEUF bins (deciles)
    print("\nBinning by LOEUF decile...")
    df_binned = create_loeuf_bins(df, n_bins=10, method="decile")

    # Calculate O/E by bin
    print("\nCalculating O/E by LOEUF bin...")
    oe_by_bin = calculate_oe_by_loeuf_bin(df_binned, ci_level)

    # Test for trend
    print("\nTesting for LOEUF-constraint trend...")
    trend_tests = test_loeuf_trend(df_binned, oe_by_bin)

    # Analyze by feature type
    print("\nAnalyzing by feature type...")
    by_type = analyze_by_feature_type(df_binned, ci_level)

    # Print results
    print_results(oe_by_bin, trend_tests, by_type)

    # Save results
    output_dir = project_root / "results"
    save_results(oe_by_bin, trend_tests, by_type, output_dir)

    return oe_by_bin, trend_tests, by_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LOEUF stratification analysis"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    main(args.config)
