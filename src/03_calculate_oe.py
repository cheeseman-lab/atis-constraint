#!/usr/bin/env python3
"""
Step 3: Calculate observed/expected (o/e) ratios with confidence intervals.

Computes o/e ratios for aTIS extensions with Poisson confidence intervals.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config,
    get_project_root,
    oe_poisson_ci,
    bootstrap_oe_ci,
)


def calculate_per_extension_oe(
    df: pd.DataFrame,
    min_variants: int = 5,
    ci_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate o/e ratio for each individual extension.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with observed and expected counts
    min_variants : int
        Minimum synonymous variants required
    ci_level : float
        Confidence level

    Returns
    -------
    pd.DataFrame
        DataFrame with o/e ratios added
    """
    result = df.copy()

    oe_mis = []
    oe_mis_lower = []
    oe_mis_upper = []

    for _, row in df.iterrows():
        obs = row["observed_missense"]
        exp = row["expected_missense"]
        syn = row["observed_synonymous"]

        # Only calculate if enough synonymous variants for reliable expectation
        if syn >= min_variants and exp > 0:
            oe, lower, upper = oe_poisson_ci(int(obs), exp, ci_level)
        else:
            oe, lower, upper = np.nan, np.nan, np.nan

        oe_mis.append(oe)
        oe_mis_lower.append(lower)
        oe_mis_upper.append(upper)

    result["oe_missense"] = oe_mis
    result["oe_missense_lower"] = oe_mis_lower
    result["oe_missense_upper"] = oe_mis_upper

    # LoF o/e if available
    if "observed_lof" in df.columns and "expected_lof" in df.columns:
        oe_lof = []
        oe_lof_lower = []
        oe_lof_upper = []

        for _, row in df.iterrows():
            obs = row["observed_lof"]
            exp = row["expected_lof"]
            syn = row["observed_synonymous"]

            if syn >= min_variants and exp > 0:
                oe, lower, upper = oe_poisson_ci(int(obs), exp, ci_level)
            else:
                oe, lower, upper = np.nan, np.nan, np.nan

            oe_lof.append(oe)
            oe_lof_lower.append(lower)
            oe_lof_upper.append(upper)

        result["oe_lof"] = oe_lof
        result["oe_lof_lower"] = oe_lof_lower
        result["oe_lof_upper"] = oe_lof_upper

    return result


def calculate_aggregate_oe(
    df: pd.DataFrame,
    ci_level: float = 0.95
) -> dict:
    """
    Calculate aggregate o/e for all extensions combined.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with counts
    ci_level : float
        Confidence level

    Returns
    -------
    dict
        Aggregate o/e with CI
    """
    total_obs_mis = df["observed_missense"].sum()
    total_exp_mis = df["expected_missense"].sum()
    total_obs_syn = df["observed_synonymous"].sum()

    oe, lower, upper = oe_poisson_ci(int(total_obs_mis), total_exp_mis, ci_level)

    result = {
        "n_extensions": len(df),
        "total_synonymous": int(total_obs_syn),
        "total_missense": int(total_obs_mis),
        "total_expected_missense": round(total_exp_mis, 1),
        "oe_missense": round(oe, 4),
        "oe_missense_lower": round(lower, 4),
        "oe_missense_upper": round(upper, 4),
    }

    if "observed_lof" in df.columns and "expected_lof" in df.columns:
        total_obs_lof = df["observed_lof"].sum()
        total_exp_lof = df["expected_lof"].sum()
        oe_lof, lower_lof, upper_lof = oe_poisson_ci(int(total_obs_lof), total_exp_lof, ci_level)

        result["total_lof"] = int(total_obs_lof)
        result["total_expected_lof"] = round(total_exp_lof, 1)
        result["oe_lof"] = round(oe_lof, 4)
        result["oe_lof_lower"] = round(lower_lof, 4)
        result["oe_lof_upper"] = round(upper_lof, 4)

    return result


def calculate_length_normalized_metrics(
    df: pd.DataFrame,
    length_col: str = "feature_length_aa"
) -> dict:
    """
    Calculate length-normalized constraint metrics.

    This verifies that O/E is independent of length by computing:
    - Variants per AA
    - Expected per AA
    - Correlation between length and O/E

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with counts
    length_col : str
        Column with extension length

    Returns
    -------
    dict
        Length-normalized metrics
    """
    from scipy import stats

    if length_col not in df.columns:
        return {}

    total_length = df[length_col].sum()
    total_missense = df["observed_missense"].sum()
    total_synonymous = df["observed_synonymous"].sum()
    total_expected = df["expected_missense"].sum()

    result = {
        "total_aa": int(total_length),
        "missense_per_aa": round(total_missense / total_length, 4),
        "synonymous_per_aa": round(total_synonymous / total_length, 4),
        "expected_missense_per_aa": round(total_expected / total_length, 4),
    }

    # O/E should be the same whether calculated from totals or per-AA
    result["oe_from_totals"] = round(total_missense / total_expected, 4)
    result["oe_from_per_aa"] = round(
        result["missense_per_aa"] / result["expected_missense_per_aa"], 4
    )

    # Check if O/E correlates with length (it shouldn't if properly normalized)
    valid = df[[length_col, "observed_missense", "expected_missense"]].dropna()
    valid = valid[valid["expected_missense"] > 0]
    valid["oe"] = valid["observed_missense"] / valid["expected_missense"]

    if len(valid) > 10:
        rho, p_value = stats.spearmanr(valid[length_col], valid["oe"])
        result["length_oe_correlation"] = round(rho, 4)
        result["length_oe_pvalue"] = round(p_value, 4)

    return result


def stratify_by_length(
    df: pd.DataFrame,
    length_col: str = "feature_length_aa",
    n_bins: int = 4,
    ci_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate o/e stratified by extension length.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with counts
    length_col : str
        Column with extension length
    n_bins : int
        Number of length bins
    ci_level : float
        Confidence level

    Returns
    -------
    pd.DataFrame
        O/E by length stratum
    """
    if length_col not in df.columns:
        print(f"WARNING: {length_col} column not found")
        return pd.DataFrame()

    # Create length bins
    df_temp = df.copy()
    try:
        df_temp["length_bin"] = pd.qcut(
            df_temp[length_col],
            q=n_bins,
            labels=[f"Q{i+1}" for i in range(n_bins)],
            duplicates="drop"
        )
    except ValueError:
        # Fall back to fixed bins if qcut fails
        df_temp["length_bin"] = pd.cut(
            df_temp[length_col],
            bins=n_bins,
            labels=[f"Bin{i+1}" for i in range(n_bins)]
        )

    results = []
    for bin_label in sorted(df_temp["length_bin"].dropna().unique()):
        mask = df_temp["length_bin"] == bin_label
        subset = df_temp[mask]

        total_obs = subset["observed_missense"].sum()
        total_exp = subset["expected_missense"].sum()

        oe, lower, upper = oe_poisson_ci(int(total_obs), total_exp, ci_level)

        # Get length range for this bin
        min_len = subset[length_col].min()
        max_len = subset[length_col].max()

        # Per-AA metrics for fair comparison
        total_aa = subset[length_col].sum()
        missense_per_aa = total_obs / total_aa if total_aa > 0 else 0
        expected_per_aa = total_exp / total_aa if total_aa > 0 else 0

        results.append({
            "length_bin": bin_label,
            "length_range": f"{min_len:.0f}-{max_len:.0f} aa",
            "n_extensions": len(subset),
            "total_aa": int(total_aa),
            "observed_missense": int(total_obs),
            "expected_missense": round(total_exp, 1),
            "missense_per_aa": round(missense_per_aa, 4),
            "expected_per_aa": round(expected_per_aa, 4),
            "oe_missense": round(oe, 4),
            "oe_missense_lower": round(lower, 4),
            "oe_missense_upper": round(upper, 4),
        })

    return pd.DataFrame(results)


def calculate_oe_by_feature_type(
    df: pd.DataFrame,
    ci_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate O/E for each feature type (extension, truncation).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with counts
    ci_level : float
        Confidence level

    Returns
    -------
    pd.DataFrame
        O/E by feature type
    """
    if "feature_type" not in df.columns:
        return pd.DataFrame()

    results = []
    for feature_type in sorted(df["feature_type"].unique()):
        subset = df[df["feature_type"] == feature_type]

        total_obs = subset["observed_missense"].sum()
        total_exp = subset["expected_missense"].sum()
        total_syn = subset["observed_synonymous"].sum()

        oe, lower, upper = oe_poisson_ci(int(total_obs), total_exp, ci_level)

        row = {
            "feature_type": feature_type,
            "n_features": len(subset),
            "total_synonymous": int(total_syn),
            "total_missense": int(total_obs),
            "expected_missense": round(total_exp, 1),
            "oe_missense": round(oe, 4),
            "oe_lower": round(lower, 4),
            "oe_upper": round(upper, 4),
        }

        # LoF if available
        if "observed_lof" in subset.columns and "expected_lof" in subset.columns:
            total_lof = subset["observed_lof"].sum()
            total_exp_lof = subset["expected_lof"].sum()
            oe_lof, lower_lof, upper_lof = oe_poisson_ci(int(total_lof), total_exp_lof, ci_level)
            row["total_lof"] = int(total_lof)
            row["oe_lof"] = round(oe_lof, 4)

        results.append(row)

    return pd.DataFrame(results)


def print_results(aggregate: dict, stratified: pd.DataFrame,
                  length_norm: dict = None, by_type: pd.DataFrame = None) -> None:
    """Print formatted results."""

    # By feature type first
    if by_type is not None and not by_type.empty:
        print("\n" + "=" * 70)
        print("O/E BY FEATURE TYPE")
        print("=" * 70)
        print(f"{'Type':<12} {'N':>8} {'Obs Mis':>10} {'Exp Mis':>10} {'O/E':>8} {'95% CI':>18}")
        print("-" * 70)
        for _, row in by_type.iterrows():
            ci = f"({row['oe_lower']:.3f}-{row['oe_upper']:.3f})"
            print(f"{row['feature_type']:<12} {row['n_features']:>8} {row['total_missense']:>10} "
                  f"{row['expected_missense']:>10.0f} {row['oe_missense']:>8.3f} {ci:>18}")

    print("\n" + "=" * 70)
    print("AGGREGATE O/E (ALL FEATURES)")
    print("=" * 70)
    print(f"  N features:          {aggregate['n_extensions']}")
    print(f"  Total synonymous:    {aggregate['total_synonymous']}")
    print(f"  Total missense:      {aggregate['total_missense']}")
    print(f"  Expected missense:   {aggregate['total_expected_missense']}")
    print(f"  O/E missense:        {aggregate['oe_missense']:.3f} "
          f"({aggregate['oe_missense_lower']:.3f} - {aggregate['oe_missense_upper']:.3f})")

    # Length-normalized metrics
    if length_norm:
        print("\n" + "=" * 70)
        print("LENGTH-NORMALIZED METRICS (per amino acid)")
        print("=" * 70)
        print(f"  Total amino acids:   {length_norm['total_aa']:,}")
        print(f"  Missense per AA:     {length_norm['missense_per_aa']:.4f}")
        print(f"  Synonymous per AA:   {length_norm['synonymous_per_aa']:.4f}")
        print(f"  Expected mis per AA: {length_norm['expected_missense_per_aa']:.4f}")
        print(f"  O/E (from totals):   {length_norm['oe_from_totals']:.4f}")
        print(f"  O/E (from per-AA):   {length_norm['oe_from_per_aa']:.4f}")
        if "length_oe_correlation" in length_norm:
            print(f"\n  Length vs O/E correlation:")
            print(f"    Spearman rho:      {length_norm['length_oe_correlation']:.4f}")
            print(f"    P-value:           {length_norm['length_oe_pvalue']:.4f}")
            if abs(length_norm['length_oe_correlation']) < 0.1:
                print("    -> O/E is effectively independent of length (good!)")

    if "oe_lof" in aggregate:
        print(f"\n  Total LoF:           {aggregate['total_lof']}")
        print(f"  Expected LoF:        {aggregate['total_expected_lof']}")
        print(f"  O/E LoF:             {aggregate['oe_lof']:.3f} "
              f"({aggregate['oe_lof_lower']:.3f} - {aggregate['oe_lof_upper']:.3f})")

    # Interpretation
    print("\n  Interpretation:")
    if aggregate['oe_missense'] < 1:
        print(f"    O/E < 1 suggests evolutionary CONSTRAINT (selection against missense)")
    else:
        print(f"    O/E >= 1 suggests neutral evolution or positive selection")

    if not stratified.empty:
        print("\n" + "=" * 70)
        print("O/E BY EXTENSION LENGTH")
        print("=" * 70)
        print(stratified.to_string(index=False))


def main(config_path: str = "config.yaml", input_path: str = None):
    """Main function to calculate o/e ratios."""
    config = load_config(config_path)
    project_root = get_project_root()

    # Analysis parameters
    ci_level = config["analysis"].get("ci_level", 0.95)
    min_variants = config["analysis"].get("min_variants", 5)

    # Load data with expected counts
    if input_path is None:
        input_path = project_root / "results" / "features_with_expected.csv"

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        print("Please run 02_get_expected_counts.py first")
        return None, None

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    # Report by feature type
    if "feature_type" in df.columns:
        for ft in df["feature_type"].unique():
            print(f"  {ft}: {(df['feature_type'] == ft).sum()}")
    print(f"Total: {len(df)} features")

    # Calculate per-extension o/e
    print("\n" + "=" * 60)
    print("Calculating per-extension o/e ratios...")
    print("=" * 60)
    extension_df = calculate_per_extension_oe(
        df,
        min_variants=min_variants,
        ci_level=ci_level
    )

    valid_oe = extension_df["oe_missense"].notna().sum()
    print(f"Extensions with valid o/e: {valid_oe} / {len(extension_df)}")

    # Calculate O/E by feature type
    print("\n" + "=" * 60)
    print("Calculating O/E by feature type...")
    print("=" * 60)
    by_type = calculate_oe_by_feature_type(df, ci_level=ci_level)

    # Calculate aggregate o/e
    print("\n" + "=" * 60)
    print("Calculating aggregate o/e...")
    print("=" * 60)
    aggregate = calculate_aggregate_oe(df, ci_level=ci_level)

    # Length-normalized metrics
    cols = config.get("columns", {})
    length_col = cols.get("feature_length_aa", "feature_length_aa")

    print("\n" + "=" * 60)
    print("Calculating length-normalized metrics...")
    print("=" * 60)
    length_norm = calculate_length_normalized_metrics(df, length_col=length_col)

    # Stratify by length
    stratified = stratify_by_length(df, length_col=length_col, n_bins=4, ci_level=ci_level)

    # Print results
    print_results(aggregate, stratified, length_norm, by_type)

    # Save outputs
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-feature O/E (main data file)
    per_feature_output = results_dir / "oe_per_feature.csv"
    extension_df.to_csv(per_feature_output, index=False)
    print(f"\nSaved per-feature O/E to: {per_feature_output}")

    # 2. Most constrained features (sorted by O/E ascending)
    cols = config.get("columns", {})
    gene_col = cols.get("gene", "gene_name")
    length_col = cols.get("feature_length_aa", "feature_length_aa")

    constrained = extension_df[extension_df["oe_missense"].notna()].copy()
    constrained = constrained.sort_values("oe_missense")
    constrained_cols = [gene_col, "feature_type", length_col,
                        "observed_missense", "expected_missense",
                        "oe_missense", "oe_missense_upper"]
    constrained_cols = [c for c in constrained_cols if c in constrained.columns]
    constrained_output = results_dir / "most_constrained.csv"
    constrained[constrained_cols].to_csv(constrained_output, index=False)
    print(f"Saved ranked constraints to: {constrained_output}")

    # 3. Generate summary.md
    summary_path = results_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("# aTIS Constraint Analysis Results\n\n")

        # Overall
        f.write("## Overall\n\n")
        f.write(f"- **Total features**: {aggregate['n_extensions']:,}\n")
        f.write(f"- **Aggregate O/E**: {aggregate['oe_missense']:.3f} ")
        f.write(f"({aggregate['oe_missense_lower']:.3f}-{aggregate['oe_missense_upper']:.3f})\n")
        f.write(f"- **Interpretation**: {'Constrained' if aggregate['oe_missense'] < 1 else 'Neutral/Relaxed'}\n\n")

        # By feature type
        if not by_type.empty:
            f.write("## O/E by Feature Type\n\n")
            f.write("| Type | N | Obs Mis | Exp Mis | O/E | 95% CI |\n")
            f.write("|------|---|---------|---------|-----|--------|\n")
            for _, row in by_type.iterrows():
                f.write(f"| {row['feature_type']} | {row['n_features']:,} | ")
                f.write(f"{row['total_missense']:,} | {row['expected_missense']:,.0f} | ")
                f.write(f"{row['oe_missense']:.3f} | {row['oe_lower']:.3f}-{row['oe_upper']:.3f} |\n")
            f.write("\n")

        # By length
        if not stratified.empty:
            f.write("## O/E by Length Quartile\n\n")
            f.write("| Length | N | O/E | 95% CI |\n")
            f.write("|--------|---|-----|--------|\n")
            for _, row in stratified.iterrows():
                f.write(f"| {row['length_range']} | {row['n_extensions']:,} | ")
                f.write(f"{row['oe_missense']:.3f} | {row['oe_missense_lower']:.3f}-{row['oe_missense_upper']:.3f} |\n")
            f.write("\n")

        # Biological interpretation tables (4 tables)
        f.write("## Isoform Functional Importance\n\n")
        f.write("**Interpretation guide:**\n")
        f.write("- Extensions: Low O/E = extended form functional | High O/E = canonical form functional\n")
        f.write("- Truncations: Low O/E = canonical form functional | High O/E = truncated form functional\n\n")

        top_n = 10

        # Extensions - low O/E (extended form matters)
        ext_low = constrained[constrained["feature_type"] == "extension"].head(top_n)
        if len(ext_low) > 0:
            f.write("### Extensions: Extended Form Functional (Low O/E)\n\n")
            f.write("*These extensions show constraint - the extended isoform is likely important.*\n\n")
            f.write("| Gene | Length | O/E |\n")
            f.write("|------|--------|-----|\n")
            for _, row in ext_low.iterrows():
                f.write(f"| {row[gene_col]} | {row[length_col]:.0f} aa | {row['oe_missense']:.3f} |\n")
            f.write("\n")

        # Extensions - high O/E (canonical form matters)
        ext_high = constrained[constrained["feature_type"] == "extension"].tail(top_n).iloc[::-1]
        if len(ext_high) > 0:
            f.write("### Extensions: Canonical Form Functional (High O/E)\n\n")
            f.write("*These extensions are not constrained - the canonical (non-extended) isoform is likely dominant.*\n\n")
            f.write("| Gene | Length | O/E |\n")
            f.write("|------|--------|-----|\n")
            for _, row in ext_high.iterrows():
                f.write(f"| {row[gene_col]} | {row[length_col]:.0f} aa | {row['oe_missense']:.3f} |\n")
            f.write("\n")

        # Truncations - low O/E (canonical form matters)
        trunc_low = constrained[constrained["feature_type"] == "truncation"].head(top_n)
        if len(trunc_low) > 0:
            f.write("### Truncations: Canonical Form Functional (Low O/E)\n\n")
            f.write("*These truncation regions are constrained - the canonical (full-length) isoform needs its N-terminus.*\n\n")
            f.write("| Gene | Length | O/E |\n")
            f.write("|------|--------|-----|\n")
            for _, row in trunc_low.iterrows():
                f.write(f"| {row[gene_col]} | {row[length_col]:.0f} aa | {row['oe_missense']:.3f} |\n")
            f.write("\n")

        # Truncations - high O/E (truncated form matters)
        trunc_high = constrained[constrained["feature_type"] == "truncation"].tail(top_n).iloc[::-1]
        if len(trunc_high) > 0:
            f.write("### Truncations: Truncated Form Functional (High O/E)\n\n")
            f.write("*These truncation regions are not constrained - the truncated isoform may be dominant.*\n\n")
            f.write("| Gene | Length | O/E |\n")
            f.write("|------|--------|-----|\n")
            for _, row in trunc_high.iterrows():
                f.write(f"| {row[gene_col]} | {row[length_col]:.0f} aa | {row['oe_missense']:.3f} |\n")
            f.write("\n")

    print(f"Saved summary to: {summary_path}")

    return extension_df, aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate o/e ratios"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input file (output from step 2)"
    )
    args = parser.parse_args()

    main(args.config, args.input)
