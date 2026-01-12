#!/usr/bin/env python3
"""
Step 2: Calculate expected variant counts.

Uses synonymous variants as internal control to estimate expected
missense counts, normalized by the genome-wide mis/syn ratio.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_project_root


def load_features(config: dict) -> pd.DataFrame:
    """Load feature data from SwissIsoform (extensions and/or truncations)."""
    project_root = get_project_root()
    data_dir = project_root / config["data"]["swissisoform_dir"]
    input_file = data_dir / config["data"]["isoform_results"]

    df = pd.read_csv(input_file)

    # Filter to configured feature types
    cols = config.get("columns", {})
    feature_col = cols.get("feature_type", "feature_type")
    feature_types = config.get("feature_types", ["extension", "truncation"])

    df = df[df[feature_col].isin(feature_types)].copy()

    # Report counts by type
    for ft in feature_types:
        n = (df[feature_col] == ft).sum()
        print(f"Loaded {n} {ft} features")

    return df


def calculate_expected_counts(
    df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Calculate expected missense counts based on synonymous.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with gnomAD variant counts
    config : dict
        Configuration with column mappings and ratios

    Returns
    -------
    pd.DataFrame
        DataFrame with added expected count columns
    """
    cols = config.get("columns", {})
    analysis = config.get("analysis", {})

    # Get column names
    mis_col = cols.get("gnomad_missense", "count_gnomad_missense_variant")
    syn_col = cols.get("gnomad_synonymous", "count_gnomad_synonymous_variant")
    nonsense_col = cols.get("gnomad_nonsense", "count_gnomad_nonsense_variant")
    frameshift_col = cols.get("gnomad_frameshift", "count_gnomad_frameshift_variant")

    # Get ratios
    mis_syn_ratio = analysis.get("genome_wide_mis_syn_ratio", 2.5)
    lof_syn_ratio = analysis.get("genome_wide_lof_syn_ratio", 0.15)

    result = df.copy()

    # Calculate expected missense from synonymous
    result["expected_missense"] = result[syn_col] * mis_syn_ratio

    # Calculate expected LoF (nonsense + frameshift) if available
    if nonsense_col in result.columns:
        result["expected_lof"] = result[syn_col] * lof_syn_ratio

        # Create combined LoF column
        if frameshift_col in result.columns:
            result["observed_lof"] = (
                result[nonsense_col].fillna(0) +
                result[frameshift_col].fillna(0)
            )
        else:
            result["observed_lof"] = result[nonsense_col].fillna(0)

    # Rename for clarity
    result["observed_missense"] = result[mis_col]
    result["observed_synonymous"] = result[syn_col]

    # Add length-normalized metrics (per AA)
    length_col = cols.get("feature_length_aa", "feature_length_aa")
    if length_col in result.columns:
        result["missense_per_aa"] = result["observed_missense"] / result[length_col]
        result["synonymous_per_aa"] = result["observed_synonymous"] / result[length_col]
        result["expected_missense_per_aa"] = result["expected_missense"] / result[length_col]

        if "observed_lof" in result.columns:
            result["lof_per_aa"] = result["observed_lof"] / result[length_col]

    return result


def calibrate_ratio_from_data(df: pd.DataFrame, config: dict) -> dict:
    """
    Calibrate mis/syn ratio from the data itself.

    Useful for checking if data matches genome-wide expectations.

    Parameters
    ----------
    df : pd.DataFrame
        Extension data
    config : dict
        Configuration

    Returns
    -------
    dict
        Observed ratios
    """
    cols = config.get("columns", {})
    mis_col = cols.get("gnomad_missense", "count_gnomad_missense_variant")
    syn_col = cols.get("gnomad_synonymous", "count_gnomad_synonymous_variant")
    nonsense_col = cols.get("gnomad_nonsense", "count_gnomad_nonsense_variant")

    total_mis = df[mis_col].sum()
    total_syn = df[syn_col].sum()

    ratios = {}

    if total_syn > 0:
        ratios["observed_mis_syn_ratio"] = total_mis / total_syn
        print(f"Observed mis/syn ratio in extensions: {ratios['observed_mis_syn_ratio']:.3f}")
    else:
        ratios["observed_mis_syn_ratio"] = np.nan

    if nonsense_col in df.columns:
        total_nonsense = df[nonsense_col].sum()
        if total_syn > 0:
            ratios["observed_nonsense_syn_ratio"] = total_nonsense / total_syn
            print(f"Observed nonsense/syn ratio: {ratios['observed_nonsense_syn_ratio']:.3f}")

    return ratios


def summarize_expected(df: pd.DataFrame) -> None:
    """Print summary of expected vs observed."""
    print("\n" + "=" * 60)
    print("Expected vs Observed Summary")
    print("=" * 60)

    total_obs_mis = df["observed_missense"].sum()
    total_exp_mis = df["expected_missense"].sum()
    total_obs_syn = df["observed_synonymous"].sum()

    print(f"  Total synonymous (obs): {total_obs_syn:.0f}")
    print(f"  Total missense (obs):   {total_obs_mis:.0f}")
    print(f"  Total missense (exp):   {total_exp_mis:.1f}")
    print(f"  Aggregate O/E missense: {total_obs_mis / total_exp_mis:.3f}" if total_exp_mis > 0 else "")

    if "observed_lof" in df.columns and "expected_lof" in df.columns:
        total_obs_lof = df["observed_lof"].sum()
        total_exp_lof = df["expected_lof"].sum()
        print(f"  Total LoF (obs):        {total_obs_lof:.0f}")
        print(f"  Total LoF (exp):        {total_exp_lof:.1f}")
        print(f"  Aggregate O/E LoF:      {total_obs_lof / total_exp_lof:.3f}" if total_exp_lof > 0 else "")

    # Extensions with enough variants for individual O/E
    min_variants = 5
    has_enough = (df["observed_synonymous"] >= min_variants).sum()
    print(f"\n  Extensions with >= {min_variants} syn variants: {has_enough}")


def main(config_path: str = "config.yaml", output_path: str = None):
    """Main function to calculate expected counts."""
    config = load_config(config_path)
    project_root = get_project_root()

    # Load data
    print("=" * 60)
    print("Loading feature data...")
    print("=" * 60)
    df = load_features(config)

    # Check observed ratios
    print("\n" + "=" * 60)
    print("Checking observed ratios...")
    print("=" * 60)
    observed_ratios = calibrate_ratio_from_data(df, config)

    # Report configured vs observed
    configured_ratio = config["analysis"].get("genome_wide_mis_syn_ratio", 2.5)
    print(f"Configured mis/syn ratio: {configured_ratio}")
    if "observed_mis_syn_ratio" in observed_ratios:
        obs_ratio = observed_ratios["observed_mis_syn_ratio"]
        if obs_ratio < configured_ratio:
            print(f"  -> Extensions show LOWER mis/syn ({obs_ratio:.2f}) than genome-wide ({configured_ratio})")
            print("     This suggests constraint against missense in extensions")
        else:
            print(f"  -> Extensions show similar or higher mis/syn ratio")

    # Calculate expected counts
    print("\n" + "=" * 60)
    print("Calculating expected counts...")
    print("=" * 60)
    result_df = calculate_expected_counts(df, config)

    # Summarize
    summarize_expected(result_df)

    # Save output
    if output_path is None:
        output_path = project_root / "results" / "features_with_expected.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate expected variant counts"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path"
    )
    args = parser.parse_args()

    main(args.config, args.output)
