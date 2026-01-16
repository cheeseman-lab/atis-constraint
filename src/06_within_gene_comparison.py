#!/usr/bin/env python3
"""
Within-gene paired comparison (PRIMARY ANALYSIS).

Compares constraint between extension/truncation regions and canonical CDS
within the same gene. This controls for gene-level constraint.

For each gene:
  extension_oe = extension_missense / expected_missense
  canonical_oe = canonical_obs_mis / canonical_exp_mis  (from gnomAD)
  ratio = extension_oe / canonical_oe

ratio < 1 → extension is MORE constrained than canonical
ratio = 1 → similar constraint
ratio > 1 → extension is LESS constrained (possibly non-functional)

Statistical test: Wilcoxon signed-rank on paired ratios; test if median != 1.0
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

    return df


def calculate_feature_oe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate O/E for each feature if not already present.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data

    Returns
    -------
    pd.DataFrame
        Data with feature_oe column
    """
    result = df.copy()

    # Get variant columns
    mis_col = "observed_missense" if "observed_missense" in df.columns else "count_gnomad_missense_variant"
    syn_col = "observed_synonymous" if "observed_synonymous" in df.columns else "count_gnomad_synonymous_variant"

    # Calculate expected if not present
    if "expected_missense" not in result.columns:
        result["expected_missense"] = result[syn_col] * 2.5

    # Calculate feature O/E
    result["feature_oe"] = result[mis_col] / result["expected_missense"]
    result["feature_oe"] = result["feature_oe"].replace([np.inf, -np.inf], np.nan)

    return result


def calculate_paired_ratios(df: pd.DataFrame, min_variants: int = 5) -> pd.DataFrame:
    """
    Calculate the ratio of feature O/E to canonical O/E for each gene.

    Parameters
    ----------
    df : pd.DataFrame
        Data with feature_oe and canonical_oe_mis
    min_variants : int
        Minimum variants required for reliable O/E

    Returns
    -------
    pd.DataFrame
        Data with oe_ratio column
    """
    result = df.copy()

    # Get variant columns for filtering
    syn_col = "observed_synonymous" if "observed_synonymous" in df.columns else "count_gnomad_synonymous_variant"

    # Filter to features with enough variants
    has_enough_variants = result[syn_col] >= min_variants

    # Filter to features with canonical O/E
    has_canonical = result["canonical_oe_mis"].notna()

    # Filter to features with valid feature O/E
    has_feature_oe = result["feature_oe"].notna() & (result["feature_oe"] > 0)

    valid_mask = has_enough_variants & has_canonical & has_feature_oe

    print(f"\nFiltering for paired comparison:")
    print(f"  Has >= {min_variants} synonymous: {has_enough_variants.sum()}")
    print(f"  Has canonical O/E: {has_canonical.sum()}")
    print(f"  Has valid feature O/E: {has_feature_oe.sum()}")
    print(f"  Valid for comparison: {valid_mask.sum()}")

    # Calculate ratio
    result["oe_ratio"] = np.nan
    result.loc[valid_mask, "oe_ratio"] = (
        result.loc[valid_mask, "feature_oe"] /
        result.loc[valid_mask, "canonical_oe_mis"]
    )

    # Handle infinite/invalid ratios
    result["oe_ratio"] = result["oe_ratio"].replace([np.inf, -np.inf], np.nan)

    return result


def test_ratio_vs_one(ratios: np.ndarray) -> dict:
    """
    Test if the ratio distribution differs from 1.0.

    Parameters
    ----------
    ratios : np.ndarray
        Array of O/E ratios (feature/canonical)

    Returns
    -------
    dict
        Test results
    """
    valid_ratios = ratios[~np.isnan(ratios)]

    if len(valid_ratios) < 10:
        return {"error": "Too few valid ratios for testing"}

    results = {}

    # Descriptive statistics
    results["n"] = len(valid_ratios)
    results["mean"] = round(np.mean(valid_ratios), 4)
    results["median"] = round(np.median(valid_ratios), 4)
    results["std"] = round(np.std(valid_ratios), 4)
    results["q25"] = round(np.percentile(valid_ratios, 25), 4)
    results["q75"] = round(np.percentile(valid_ratios, 75), 4)

    # Proportion below 1 (more constrained than canonical)
    results["pct_below_1"] = round(100 * (valid_ratios < 1).mean(), 1)
    results["pct_above_1"] = round(100 * (valid_ratios > 1).mean(), 1)

    # Wilcoxon signed-rank test (paired test against 1.0)
    # Tests if median ratio differs from 1.0
    try:
        stat, p_val = stats.wilcoxon(valid_ratios - 1.0, alternative="two-sided")
        results["wilcoxon_statistic"] = round(stat, 4)
        results["wilcoxon_pvalue"] = p_val
    except Exception as e:
        results["wilcoxon_error"] = str(e)

    # One-sample t-test (assumes normality, use with caution)
    try:
        stat, p_val = stats.ttest_1samp(valid_ratios, 1.0)
        results["ttest_statistic"] = round(stat, 4)
        results["ttest_pvalue"] = p_val
    except Exception as e:
        results["ttest_error"] = str(e)

    # Sign test (non-parametric, tests if median = 1)
    n_below = (valid_ratios < 1).sum()
    n_above = (valid_ratios > 1).sum()
    n_total = n_below + n_above
    if n_total > 0:
        # Binomial test
        p_val = stats.binom_test(n_below, n_total, 0.5, alternative="two-sided")
        results["sign_test_pvalue"] = p_val

    # Interpretation
    if "wilcoxon_pvalue" in results:
        p = results["wilcoxon_pvalue"]
        median = results["median"]
        if p < 0.05:
            if median < 1:
                results["interpretation"] = (
                    f"Extensions are MORE constrained than canonical (median ratio = {median:.3f}, p = {p:.2e})"
                )
            else:
                results["interpretation"] = (
                    f"Extensions are LESS constrained than canonical (median ratio = {median:.3f}, p = {p:.2e})"
                )
        else:
            results["interpretation"] = (
                f"No significant difference from canonical (median ratio = {median:.3f}, p = {p:.2e})"
            )

    return results


def analyze_by_feature_type(df: pd.DataFrame) -> dict:
    """
    Run within-gene comparison separately for extensions and truncations.

    Parameters
    ----------
    df : pd.DataFrame
        Data with oe_ratio column

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
        ratios = subset["oe_ratio"].dropna().values

        if len(ratios) >= 10:
            results[feature_type] = test_ratio_vs_one(ratios)

    return results


def analyze_by_length(df: pd.DataFrame, length_col: str = "feature_length_aa") -> dict:
    """
    Stratify within-gene comparison by feature length.

    Parameters
    ----------
    df : pd.DataFrame
        Data with oe_ratio column
    length_col : str
        Column with feature length

    Returns
    -------
    dict
        Results by length quartile
    """
    results = {}

    if length_col not in df.columns:
        return results

    # Create length quartiles
    df_temp = df[df["oe_ratio"].notna()].copy()
    if len(df_temp) < 40:
        return results

    try:
        df_temp["length_quartile"] = pd.qcut(
            df_temp[length_col],
            q=4,
            labels=["Q1 (short)", "Q2", "Q3", "Q4 (long)"]
        )
    except ValueError:
        return results

    for quartile in df_temp["length_quartile"].unique():
        subset = df_temp[df_temp["length_quartile"] == quartile]
        ratios = subset["oe_ratio"].values

        if len(ratios) >= 10:
            length_range = f"{subset[length_col].min():.0f}-{subset[length_col].max():.0f} aa"
            test_results = test_ratio_vs_one(ratios)
            test_results["length_range"] = length_range
            results[str(quartile)] = test_results

    return results


def print_results(
    overall: dict,
    by_type: dict,
    by_length: dict
) -> None:
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("WITHIN-GENE PAIRED COMPARISON (PRIMARY ANALYSIS)")
    print("=" * 70)

    print("\n## Overall Results\n")
    print(f"  N features: {overall.get('n', 'N/A')}")
    print(f"  Median ratio: {overall.get('median', 'N/A')}")
    print(f"  Mean ratio: {overall.get('mean', 'N/A')}")
    print(f"  IQR: {overall.get('q25', 'N/A')} - {overall.get('q75', 'N/A')}")
    print(f"  % below 1 (more constrained): {overall.get('pct_below_1', 'N/A')}%")
    print(f"  % above 1 (less constrained): {overall.get('pct_above_1', 'N/A')}%")

    if "wilcoxon_pvalue" in overall:
        print(f"\n  Wilcoxon signed-rank test (H0: median = 1):")
        print(f"    Statistic: {overall['wilcoxon_statistic']}")
        print(f"    P-value: {overall['wilcoxon_pvalue']:.2e}")

    if "interpretation" in overall:
        print(f"\n  >>> {overall['interpretation']}")

    # By feature type
    if by_type:
        print("\n" + "-" * 70)
        print("## By Feature Type\n")
        for ft, data in by_type.items():
            print(f"  {ft.upper()}:")
            print(f"    N: {data.get('n', 'N/A')}")
            print(f"    Median ratio: {data.get('median', 'N/A')}")
            print(f"    % below 1: {data.get('pct_below_1', 'N/A')}%")
            if "wilcoxon_pvalue" in data:
                print(f"    P-value: {data['wilcoxon_pvalue']:.2e}")
            if "interpretation" in data:
                print(f"    {data['interpretation']}")
            print()

    # By length
    if by_length:
        print("-" * 70)
        print("## By Feature Length\n")
        for quartile, data in sorted(by_length.items()):
            print(f"  {quartile} ({data.get('length_range', '')}):")
            print(f"    N: {data.get('n', 'N/A')}, Median ratio: {data.get('median', 'N/A')}")


def save_results(
    df: pd.DataFrame,
    overall: dict,
    by_type: dict,
    by_length: dict,
    output_dir: Path
) -> None:
    """Save results to files."""

    # Save feature-level data with ratios
    output_cols = [
        "gene_name", "feature_type", "feature_length_aa",
        "feature_oe", "canonical_oe_mis", "oe_ratio", "loeuf"
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(output_dir / "within_gene_ratios.csv", index=False)

    # Save summary to markdown
    summary_path = output_dir / "within_gene_comparison_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Within-Gene Paired Comparison\n\n")

        f.write("## Method\n\n")
        f.write("For each gene, we compare the O/E ratio of the extension/truncation ")
        f.write("region to the O/E ratio of the canonical CDS (from gnomAD).\n\n")
        f.write("```\n")
        f.write("ratio = feature_O/E / canonical_O/E\n")
        f.write("```\n\n")
        f.write("- ratio < 1 → feature is MORE constrained than canonical\n")
        f.write("- ratio = 1 → similar constraint\n")
        f.write("- ratio > 1 → feature is LESS constrained\n\n")

        f.write("## Overall Results\n\n")
        f.write(f"- **N features**: {overall.get('n', 'N/A')}\n")
        f.write(f"- **Median ratio**: {overall.get('median', 'N/A')}\n")
        f.write(f"- **IQR**: {overall.get('q25', 'N/A')} - {overall.get('q75', 'N/A')}\n")
        f.write(f"- **% more constrained** (ratio < 1): {overall.get('pct_below_1', 'N/A')}%\n")
        f.write(f"- **% less constrained** (ratio > 1): {overall.get('pct_above_1', 'N/A')}%\n\n")

        if "wilcoxon_pvalue" in overall:
            f.write("### Statistical Test\n\n")
            f.write("Wilcoxon signed-rank test (H0: median ratio = 1)\n\n")
            f.write(f"- Statistic: {overall['wilcoxon_statistic']}\n")
            f.write(f"- P-value: {overall['wilcoxon_pvalue']:.2e}\n\n")

        if "interpretation" in overall:
            f.write(f"### Conclusion\n\n**{overall['interpretation']}**\n\n")

        # By feature type
        if by_type:
            f.write("## By Feature Type\n\n")
            f.write("| Type | N | Median Ratio | % < 1 | P-value |\n")
            f.write("|------|---|--------------|-------|--------|\n")
            for ft, data in by_type.items():
                p = data.get('wilcoxon_pvalue', np.nan)
                p_str = f"{p:.2e}" if not np.isnan(p) else "N/A"
                f.write(f"| {ft} | {data.get('n', 'N/A')} | ")
                f.write(f"{data.get('median', 'N/A')} | ")
                f.write(f"{data.get('pct_below_1', 'N/A')}% | {p_str} |\n")
            f.write("\n")

        # By length
        if by_length:
            f.write("## By Feature Length\n\n")
            f.write("| Quartile | Length | N | Median Ratio | P-value |\n")
            f.write("|----------|--------|---|--------------|--------|\n")
            for quartile, data in sorted(by_length.items()):
                p = data.get('wilcoxon_pvalue', np.nan)
                p_str = f"{p:.2e}" if not np.isnan(p) else "N/A"
                f.write(f"| {quartile} | {data.get('length_range', '')} | ")
                f.write(f"{data.get('n', 'N/A')} | {data.get('median', 'N/A')} | {p_str} |\n")
            f.write("\n")

    print(f"\nSaved results to: {output_dir}")


def main(config_path: str = "config.yaml"):
    """Main function for within-gene paired comparison."""
    config = load_config(config_path)
    project_root = get_project_root()
    min_variants = config["analysis"].get("min_variants", 5)

    # Load data
    df = load_merged_data(config)

    # Calculate feature O/E
    print("\nCalculating feature O/E...")
    df = calculate_feature_oe(df)

    # Calculate paired ratios
    print("\nCalculating paired ratios (feature O/E / canonical O/E)...")
    df = calculate_paired_ratios(df, min_variants=min_variants)

    # Overall test
    print("\nTesting if ratio differs from 1.0...")
    ratios = df["oe_ratio"].dropna().values
    overall = test_ratio_vs_one(ratios)

    # By feature type
    print("\nAnalyzing by feature type...")
    by_type = analyze_by_feature_type(df)

    # By length
    print("\nAnalyzing by feature length...")
    cols = config.get("columns", {})
    length_col = cols.get("feature_length_aa", "feature_length_aa")
    by_length = analyze_by_length(df, length_col=length_col)

    # Print results
    print_results(overall, by_type, by_length)

    # Save results
    output_dir = project_root / "results"
    save_results(df, overall, by_type, by_length, output_dir)

    return df, overall, by_type, by_length


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Within-gene paired comparison"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    main(args.config)
