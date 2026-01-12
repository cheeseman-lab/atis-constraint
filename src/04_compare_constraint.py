#!/usr/bin/env python3
"""
Step 4: Statistical comparison of constraint within aTIS extensions.

Compares o/e between different extension subgroups:
- By start codon type (CTG, TTG, GTG, etc.)
- By extension length
- Between extensions and expected neutral baseline
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_project_root, permutation_test, oe_poisson_ci


def compare_to_neutral(
    df: pd.DataFrame,
    neutral_oe: float = 1.0
) -> dict:
    """
    Test if aggregate o/e differs significantly from neutral (O/E=1).

    Uses a one-sample test based on Poisson model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with counts
    neutral_oe : float
        Expected O/E under neutrality (default 1.0)

    Returns
    -------
    dict
        Test results
    """
    total_obs = df["observed_missense"].sum()
    total_exp = df["expected_missense"].sum()

    # Under null (neutral), expected count would be total_exp * neutral_oe
    expected_under_null = total_exp * neutral_oe

    # One-sided Poisson test: is observed significantly less than expected?
    # Using exact Poisson test
    p_value_lower = stats.poisson.cdf(total_obs, expected_under_null)
    p_value_upper = 1 - stats.poisson.cdf(total_obs - 1, expected_under_null)
    p_value_two_sided = 2 * min(p_value_lower, p_value_upper)

    observed_oe = total_obs / total_exp if total_exp > 0 else np.nan

    return {
        "observed_missense": int(total_obs),
        "expected_missense": round(total_exp, 1),
        "observed_oe": round(observed_oe, 4),
        "null_oe": neutral_oe,
        "p_value_lower": round(p_value_lower, 6),
        "p_value_upper": round(p_value_upper, 6),
        "p_value_two_sided": round(p_value_two_sided, 6),
        "significant_constraint": p_value_lower < 0.05,
    }


def compare_by_codon(
    df: pd.DataFrame,
    codon_col: str = "alternative_start_codon",
    ci_level: float = 0.95
) -> pd.DataFrame:
    """
    Compare o/e between different start codon types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with counts
    codon_col : str
        Column with start codon
    ci_level : float
        Confidence level

    Returns
    -------
    pd.DataFrame
        O/E by codon type
    """
    if codon_col not in df.columns:
        print(f"WARNING: {codon_col} column not found")
        return pd.DataFrame()

    results = []
    for codon in sorted(df[codon_col].dropna().unique()):
        mask = df[codon_col] == codon
        subset = df[mask]

        total_obs = subset["observed_missense"].sum()
        total_exp = subset["expected_missense"].sum()

        oe, lower, upper = oe_poisson_ci(int(total_obs), total_exp, ci_level)

        results.append({
            "start_codon": codon,
            "n_extensions": len(subset),
            "observed_missense": int(total_obs),
            "expected_missense": round(total_exp, 1),
            "oe_missense": round(oe, 4) if not np.isnan(oe) else np.nan,
            "oe_lower": round(lower, 4) if not np.isnan(lower) else np.nan,
            "oe_upper": round(upper, 4) if not np.isnan(upper) else np.nan,
        })

    return pd.DataFrame(results)


def compare_short_vs_long(
    df: pd.DataFrame,
    length_col: str = "feature_length_aa",
    threshold: int = 20,
    n_permutations: int = 10000
) -> dict:
    """
    Compare constraint between short and long extensions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with counts
    length_col : str
        Column with extension length
    threshold : int
        Length threshold to split groups
    n_permutations : int
        Number of permutations for test

    Returns
    -------
    dict
        Comparison results
    """
    if length_col not in df.columns:
        print(f"WARNING: {length_col} column not found")
        return {}

    short_mask = df[length_col] <= threshold
    long_mask = df[length_col] > threshold

    short_df = df[short_mask]
    long_df = df[long_mask]

    if len(short_df) == 0 or len(long_df) == 0:
        print("WARNING: One group is empty")
        return {}

    # Calculate o/e for each group
    short_oe = short_df["observed_missense"].sum() / short_df["expected_missense"].sum()
    long_oe = long_df["observed_missense"].sum() / long_df["expected_missense"].sum()

    # Permutation test
    diff, p_value = permutation_test(
        short_df["observed_missense"].values,
        short_df["expected_missense"].values,
        long_df["observed_missense"].values,
        long_df["expected_missense"].values,
        n_permutations=n_permutations
    )

    return {
        "threshold_aa": threshold,
        "n_short": len(short_df),
        "n_long": len(long_df),
        "oe_short": round(short_oe, 4),
        "oe_long": round(long_oe, 4),
        "delta_oe": round(diff, 4),
        "p_value": round(p_value, 6),
    }


def correlation_length_constraint(
    df: pd.DataFrame,
    length_col: str = "feature_length_aa"
) -> dict:
    """
    Test correlation between extension length and constraint.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with per-extension o/e
    length_col : str
        Column with extension length

    Returns
    -------
    dict
        Correlation results
    """
    if length_col not in df.columns or "oe_missense" not in df.columns:
        return {}

    # Filter to valid o/e values
    valid = df[[length_col, "oe_missense"]].dropna()

    if len(valid) < 10:
        print("WARNING: Too few data points for correlation")
        return {}

    # Spearman correlation (more robust to outliers)
    rho, p_value = stats.spearmanr(valid[length_col], valid["oe_missense"])

    return {
        "n_valid": len(valid),
        "spearman_rho": round(rho, 4),
        "p_value": round(p_value, 6),
        "interpretation": "Longer extensions more constrained" if rho < 0 else "Longer extensions less constrained"
    }


def print_results(neutral_test: dict, codon_df: pd.DataFrame,
                  short_long: dict, correlation: dict) -> None:
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("1. TEST AGAINST NEUTRAL (O/E = 1)")
    print("=" * 70)
    print(f"  Observed missense:   {neutral_test['observed_missense']}")
    print(f"  Expected missense:   {neutral_test['expected_missense']}")
    print(f"  Observed O/E:        {neutral_test['observed_oe']:.3f}")
    print(f"  P-value (constraint): {neutral_test['p_value_lower']:.2e}")
    if neutral_test['significant_constraint']:
        print("  --> SIGNIFICANT constraint (O/E < 1, p < 0.05)")
    else:
        print("  --> Not significantly constrained")

    if not codon_df.empty:
        print("\n" + "=" * 70)
        print("2. O/E BY START CODON TYPE")
        print("=" * 70)
        print(codon_df.to_string(index=False))

    if short_long:
        print("\n" + "=" * 70)
        print("3. SHORT vs LONG EXTENSIONS")
        print("=" * 70)
        print(f"  Threshold:           {short_long['threshold_aa']} aa")
        print(f"  Short (n={short_long['n_short']}): O/E = {short_long['oe_short']:.3f}")
        print(f"  Long (n={short_long['n_long']}):  O/E = {short_long['oe_long']:.3f}")
        print(f"  Delta O/E:           {short_long['delta_oe']:.3f}")
        print(f"  P-value (permutation): {short_long['p_value']:.4f}")

    if correlation:
        print("\n" + "=" * 70)
        print("4. CORRELATION: LENGTH vs CONSTRAINT")
        print("=" * 70)
        print(f"  N extensions:        {correlation['n_valid']}")
        print(f"  Spearman rho:        {correlation['spearman_rho']:.3f}")
        print(f"  P-value:             {correlation['p_value']:.4f}")
        print(f"  Interpretation:      {correlation['interpretation']}")


def main(config_path: str = "config.yaml", input_path: str = None):
    """Main function for statistical comparisons."""
    config = load_config(config_path)
    project_root = get_project_root()

    # Load data with o/e
    if input_path is None:
        input_path = project_root / "results" / "oe_per_feature.csv"

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Please run 03_calculate_oe.py first")
        return None

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} extensions")

    # 1. Test against neutral
    print("\n" + "=" * 60)
    print("Testing against neutral baseline...")
    print("=" * 60)
    neutral_test = compare_to_neutral(df)

    # 2. Compare by codon type
    print("\n" + "=" * 60)
    print("Comparing by start codon...")
    print("=" * 60)
    codon_df = compare_by_codon(df)

    # 3. Short vs long
    print("\n" + "=" * 60)
    print("Comparing short vs long extensions...")
    print("=" * 60)
    cols = config.get("columns", {})
    length_col = cols.get("feature_length_aa", "feature_length_aa")

    # Find median length for threshold
    if length_col in df.columns:
        median_len = int(df[length_col].median())
        short_long = compare_short_vs_long(df, length_col=length_col, threshold=median_len)
    else:
        short_long = {}

    # 4. Correlation
    print("\n" + "=" * 60)
    print("Testing correlation between length and constraint...")
    print("=" * 60)
    correlation = correlation_length_constraint(df, length_col=length_col)

    # Print all results
    print_results(neutral_test, codon_df, short_long, correlation)

    # Append to summary.md
    results_dir = project_root / "results"
    summary_path = results_dir / "summary.md"

    with open(summary_path, "a") as f:
        # Statistical test
        f.write("## Statistical Tests\n\n")
        f.write("### Test vs Neutral (O/E = 1)\n\n")
        f.write(f"- **Observed O/E**: {neutral_test['observed_oe']:.3f}\n")
        f.write(f"- **P-value**: {neutral_test['p_value_lower']:.2e}\n")
        f.write(f"- **Result**: {'Significant constraint' if neutral_test['significant_constraint'] else 'Not significant'}\n\n")

        # By codon
        if not codon_df.empty:
            f.write("### O/E by Start Codon\n\n")
            f.write("| Codon | N | O/E | 95% CI |\n")
            f.write("|-------|---|-----|--------|\n")
            for _, row in codon_df.sort_values("oe_missense").iterrows():
                if pd.notna(row["oe_missense"]):
                    f.write(f"| {row['start_codon']} | {row['n_extensions']:,} | ")
                    f.write(f"{row['oe_missense']:.3f} | {row['oe_lower']:.3f}-{row['oe_upper']:.3f} |\n")
            f.write("\n")

        # Short vs long
        if short_long:
            f.write("### Short vs Long Features\n\n")
            f.write(f"- **Threshold**: {short_long['threshold_aa']} aa\n")
            f.write(f"- **Short O/E**: {short_long['oe_short']:.3f} (n={short_long['n_short']})\n")
            f.write(f"- **Long O/E**: {short_long['oe_long']:.3f} (n={short_long['n_long']})\n")
            f.write(f"- **P-value**: {short_long['p_value']:.4f}\n\n")

        # Correlation
        if correlation:
            f.write("### Length vs Constraint Correlation\n\n")
            f.write(f"- **Spearman rho**: {correlation['spearman_rho']:.3f}\n")
            f.write(f"- **P-value**: {correlation['p_value']:.4f}\n")
            f.write(f"- **Interpretation**: {correlation['interpretation']}\n")

    print(f"\nAppended statistical tests to: {summary_path}")

    return neutral_test, codon_df, short_long, correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistical comparison of constraint"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input file (output from step 3)"
    )
    args = parser.parse_args()

    main(args.config, args.input)
