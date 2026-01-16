#!/usr/bin/env python3
"""
Sequence composition check (sensitivity analysis).

Validates that missense/synonymous ratios are comparable between regions.
Under neutrality, mis/syn ratio should be ~2.5 (genome-wide average).

If ratios differ wildly between extensions and canonical regions,
sequence composition may confound our constraint analysis.

This serves as a sanity check for the assumptions in the constraint analyses.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_project_root

# Genome-wide expected missense/synonymous ratio
EXPECTED_MIS_SYN_RATIO = 2.5


def load_data(config: dict) -> pd.DataFrame:
    """Load feature data."""
    project_root = get_project_root()

    # Try merged data first
    merged_file = project_root / "results" / "features_with_gnomad.csv"
    if merged_file.exists():
        print(f"Loading from: {merged_file}")
        return pd.read_csv(merged_file)

    # Fall back to raw data
    data_dir = project_root / config["data"]["swissisoform_dir"]
    input_file = data_dir / config["data"]["isoform_results"]
    print(f"Loading from: {input_file}")
    return pd.read_csv(input_file)


def calculate_mis_syn_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missense/synonymous ratio for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data with variant counts

    Returns
    -------
    pd.DataFrame
        Data with mis_syn_ratio column
    """
    result = df.copy()

    # Get variant columns
    mis_col = "observed_missense" if "observed_missense" in df.columns else "count_gnomad_missense_variant"
    syn_col = "observed_synonymous" if "observed_synonymous" in df.columns else "count_gnomad_synonymous_variant"

    # Calculate ratio (only where syn > 0)
    result["mis_syn_ratio"] = np.nan
    valid = result[syn_col] > 0
    result.loc[valid, "mis_syn_ratio"] = result.loc[valid, mis_col] / result.loc[valid, syn_col]

    return result


def aggregate_by_group(
    df: pd.DataFrame,
    group_col: str
) -> pd.DataFrame:
    """
    Calculate aggregate mis/syn ratio by group.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data
    group_col : str
        Column to group by

    Returns
    -------
    pd.DataFrame
        Aggregate statistics per group
    """
    mis_col = "observed_missense" if "observed_missense" in df.columns else "count_gnomad_missense_variant"
    syn_col = "observed_synonymous" if "observed_synonymous" in df.columns else "count_gnomad_synonymous_variant"

    results = []
    for group in df[group_col].unique():
        subset = df[df[group_col] == group]

        total_mis = subset[mis_col].sum()
        total_syn = subset[syn_col].sum()

        ratio = total_mis / total_syn if total_syn > 0 else np.nan

        # Compare to expected
        deviation = (ratio - EXPECTED_MIS_SYN_RATIO) / EXPECTED_MIS_SYN_RATIO * 100

        results.append({
            "group": group,
            "n_features": len(subset),
            "total_missense": int(total_mis),
            "total_synonymous": int(total_syn),
            "mis_syn_ratio": round(ratio, 3) if not np.isnan(ratio) else np.nan,
            "expected_ratio": EXPECTED_MIS_SYN_RATIO,
            "pct_deviation": round(deviation, 1) if not np.isnan(deviation) else np.nan,
        })

    return pd.DataFrame(results)


def compare_ratios_between_groups(
    df: pd.DataFrame,
    group_col: str
) -> dict:
    """
    Test if mis/syn ratios differ between groups.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data with mis_syn_ratio
    group_col : str
        Column to group by

    Returns
    -------
    dict
        Statistical test results
    """
    groups = df[group_col].unique()

    if len(groups) < 2:
        return {"error": "Need at least 2 groups for comparison"}

    # Get ratio distributions for each group
    group_ratios = {}
    for g in groups:
        ratios = df[df[group_col] == g]["mis_syn_ratio"].dropna().values
        if len(ratios) >= 10:
            group_ratios[g] = ratios

    if len(group_ratios) < 2:
        return {"error": "Not enough groups with sufficient data"}

    results = {}

    # Mann-Whitney U test between each pair
    group_list = list(group_ratios.keys())
    if len(group_list) == 2:
        g1, g2 = group_list
        stat, p_val = stats.mannwhitneyu(
            group_ratios[g1], group_ratios[g2], alternative="two-sided"
        )
        results["mann_whitney"] = {
            "group1": g1,
            "group2": g2,
            "statistic": round(stat, 4),
            "p_value": p_val,
        }

    # Kruskal-Wallis for multiple groups
    if len(group_ratios) >= 2:
        stat, p_val = stats.kruskal(*[group_ratios[g] for g in group_ratios])
        results["kruskal_wallis"] = {
            "statistic": round(stat, 4),
            "p_value": p_val,
            "n_groups": len(group_ratios),
        }

    # Interpretation
    if "kruskal_wallis" in results:
        p = results["kruskal_wallis"]["p_value"]
        if p < 0.05:
            results["interpretation"] = (
                f"Significant difference in mis/syn ratios between groups (p={p:.2e}). "
                "Sequence composition may differ between region types."
            )
        else:
            results["interpretation"] = (
                f"No significant difference in mis/syn ratios (p={p:.2e}). "
                "Sequence composition is comparable between region types."
            )

    return results


def test_ratio_vs_expected(df: pd.DataFrame, expected: float = 2.5) -> dict:
    """
    Test if observed mis/syn ratios differ from expected genome-wide ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data with mis_syn_ratio
    expected : float
        Expected ratio (genome-wide ~2.5)

    Returns
    -------
    dict
        Test results
    """
    ratios = df["mis_syn_ratio"].dropna().values

    if len(ratios) < 10:
        return {"error": "Too few features for testing"}

    results = {}

    # Descriptive stats
    results["n"] = len(ratios)
    results["mean"] = round(np.mean(ratios), 3)
    results["median"] = round(np.median(ratios), 3)
    results["std"] = round(np.std(ratios), 3)

    # One-sample Wilcoxon test against expected
    try:
        stat, p_val = stats.wilcoxon(ratios - expected, alternative="two-sided")
        results["wilcoxon_statistic"] = round(stat, 4)
        results["wilcoxon_pvalue"] = p_val
    except Exception as e:
        results["wilcoxon_error"] = str(e)

    # Aggregate ratio (more stable)
    mis_col = "observed_missense" if "observed_missense" in df.columns else "count_gnomad_missense_variant"
    syn_col = "observed_synonymous" if "observed_synonymous" in df.columns else "count_gnomad_synonymous_variant"
    total_mis = df[mis_col].sum()
    total_syn = df[syn_col].sum()
    aggregate_ratio = total_mis / total_syn if total_syn > 0 else np.nan
    results["aggregate_ratio"] = round(aggregate_ratio, 3)

    # Interpretation
    if not np.isnan(aggregate_ratio):
        deviation = (aggregate_ratio - expected) / expected * 100
        results["pct_deviation_from_expected"] = round(deviation, 1)

        if abs(deviation) < 10:
            results["interpretation"] = (
                f"Aggregate ratio ({aggregate_ratio:.2f}) is within 10% of expected ({expected}). "
                "Sequence composition is consistent with genome-wide expectations."
            )
        elif deviation < 0:
            results["interpretation"] = (
                f"Aggregate ratio ({aggregate_ratio:.2f}) is {abs(deviation):.0f}% BELOW expected ({expected}). "
                "This suggests constraint (fewer missense than expected) or unusual sequence composition."
            )
        else:
            results["interpretation"] = (
                f"Aggregate ratio ({aggregate_ratio:.2f}) is {deviation:.0f}% ABOVE expected ({expected}). "
                "This could indicate relaxed selection or unusual sequence composition."
            )

    return results


def print_results(
    by_type: pd.DataFrame,
    comparison: dict,
    vs_expected: dict
) -> None:
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("SEQUENCE COMPOSITION CHECK")
    print("=" * 70)

    print("\n## Missense/Synonymous Ratio by Feature Type\n")
    print(by_type.to_string(index=False))

    print("\n## Comparison Between Groups\n")
    if "interpretation" in comparison:
        print(f"  {comparison['interpretation']}")
    if "kruskal_wallis" in comparison:
        kw = comparison["kruskal_wallis"]
        print(f"\n  Kruskal-Wallis test:")
        print(f"    Statistic: {kw['statistic']}")
        print(f"    P-value: {kw['p_value']:.2e}")

    print("\n## Comparison to Genome-Wide Expected (2.5)\n")
    print(f"  N features: {vs_expected.get('n', 'N/A')}")
    print(f"  Aggregate ratio: {vs_expected.get('aggregate_ratio', 'N/A')}")
    print(f"  Deviation from expected: {vs_expected.get('pct_deviation_from_expected', 'N/A')}%")
    if "interpretation" in vs_expected:
        print(f"\n  {vs_expected['interpretation']}")


def save_results(
    by_type: pd.DataFrame,
    comparison: dict,
    vs_expected: dict,
    output_dir: Path
) -> None:
    """Save results to files."""

    # Save summary to markdown
    summary_path = output_dir / "composition_check_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Sequence Composition Check\n\n")

        f.write("## Purpose\n\n")
        f.write("This sensitivity analysis validates that missense/synonymous ratios ")
        f.write("are comparable between region types. Under neutrality, the ratio ")
        f.write(f"should be approximately {EXPECTED_MIS_SYN_RATIO} (genome-wide average).\n\n")
        f.write("If ratios differ significantly between extensions and canonical regions, ")
        f.write("sequence composition may confound our constraint analysis.\n\n")

        f.write("## Results by Feature Type\n\n")
        f.write("| Type | N | Total Mis | Total Syn | Ratio | Expected | Deviation |\n")
        f.write("|------|---|-----------|-----------|-------|----------|----------|\n")
        for _, row in by_type.iterrows():
            f.write(f"| {row['group']} | {row['n_features']} | ")
            f.write(f"{row['total_missense']} | {row['total_synonymous']} | ")
            f.write(f"{row['mis_syn_ratio']} | {row['expected_ratio']} | ")
            f.write(f"{row['pct_deviation']}% |\n")
        f.write("\n")

        f.write("## Statistical Tests\n\n")
        if "kruskal_wallis" in comparison:
            kw = comparison["kruskal_wallis"]
            f.write(f"**Kruskal-Wallis test** (difference between groups):\n")
            f.write(f"- Statistic: {kw['statistic']}\n")
            f.write(f"- P-value: {kw['p_value']:.2e}\n\n")

        if "interpretation" in comparison:
            f.write(f"**Interpretation**: {comparison['interpretation']}\n\n")

        f.write("## Comparison to Expected\n\n")
        f.write(f"- Aggregate mis/syn ratio: {vs_expected.get('aggregate_ratio', 'N/A')}\n")
        f.write(f"- Deviation from expected: {vs_expected.get('pct_deviation_from_expected', 'N/A')}%\n\n")

        if "interpretation" in vs_expected:
            f.write(f"**Interpretation**: {vs_expected['interpretation']}\n\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        if "kruskal_wallis" in comparison:
            if comparison["kruskal_wallis"]["p_value"] >= 0.05:
                f.write("Sequence composition is comparable between region types. ")
                f.write("The constraint analysis in the constraint analyses is not confounded by ")
                f.write("systematic differences in mis/syn ratios.\n")
            else:
                f.write("**Caution**: Significant differences in mis/syn ratios detected. ")
                f.write("Consider this when interpreting constraint results.\n")

    print(f"\nSaved results to: {output_dir}")


def main(config_path: str = "config.yaml"):
    """Main function for composition check."""
    config = load_config(config_path)
    project_root = get_project_root()

    # Load data
    df = load_data(config)
    print(f"Loaded {len(df)} features")

    # Calculate mis/syn ratios
    print("\nCalculating mis/syn ratios...")
    df = calculate_mis_syn_ratios(df)

    # Aggregate by feature type
    print("\nAggregating by feature type...")
    by_type = aggregate_by_group(df, "feature_type")

    # Compare ratios between groups
    print("\nComparing ratios between groups...")
    comparison = compare_ratios_between_groups(df, "feature_type")

    # Test vs expected
    print("\nTesting against expected genome-wide ratio...")
    vs_expected = test_ratio_vs_expected(df)

    # Print results
    print_results(by_type, comparison, vs_expected)

    # Save results
    output_dir = project_root / "results"
    save_results(by_type, comparison, vs_expected, output_dir)

    return by_type, comparison, vs_expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence composition check"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    main(args.config)
