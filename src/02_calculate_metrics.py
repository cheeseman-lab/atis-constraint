#!/usr/bin/env python3
"""
Calculate constraint metrics for aTIS regions and canonical CDS.

For each feature, computes:
- O/E ratios (missense, synonymous, LoF)
- Variant densities (variants per amino acid)
- Paired comparisons (aTIS vs canonical)

Output: data/features_with_metrics.csv
"""

from pathlib import Path
import pandas as pd


def calculate_atis_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate O/E and density metrics for aTIS regions.

    Uses gnomAD's expected counts scaled by region length.

    Parameters
    ----------
    df : pd.DataFrame
        Features with variant counts and gnomAD constraint

    Returns
    -------
    pd.DataFrame
        Data with aTIS metrics added
    """
    print("\nCalculating aTIS region metrics...")

    # Calculate amino acid lengths
    df["aTIS_length_aa"] = df["feature_length_aa"]
    df["canonical_length_aa"] = df["canonical_cds_length"] / 3

    # Length ratio for scaling expected counts
    df["length_ratio"] = df["aTIS_length_aa"] / df["canonical_length_aa"]

    # --- Expected counts (scaled from canonical) ---
    df["aTIS_exp_mis"] = df["canonical_exp_mis"] * df["length_ratio"]
    df["aTIS_exp_syn"] = df["canonical_exp_syn"] * df["length_ratio"]
    df["aTIS_exp_lof"] = df["canonical_exp_lof"] * df["length_ratio"]

    # --- Observed counts (from SwissIsoform gnomAD variants) ---
    df["aTIS_obs_mis"] = df["count_gnomad_missense_variant"]
    df["aTIS_obs_syn"] = df["count_gnomad_synonymous_variant"]
    df["aTIS_obs_lof"] = df["count_gnomad_lof"]

    # --- O/E ratios ---
    # Add pseudocount to avoid division by zero
    pseudocount = 0.001

    df["aTIS_oe_mis"] = df["aTIS_obs_mis"] / (df["aTIS_exp_mis"] + pseudocount)
    df["aTIS_oe_syn"] = df["aTIS_obs_syn"] / (df["aTIS_exp_syn"] + pseudocount)
    df["aTIS_oe_lof"] = df["aTIS_obs_lof"] / (df["aTIS_exp_lof"] + pseudocount)

    # --- Variant densities (per amino acid) ---
    df["aTIS_density_mis"] = df["aTIS_obs_mis"] / df["aTIS_length_aa"]
    df["aTIS_density_syn"] = df["aTIS_obs_syn"] / df["aTIS_length_aa"]
    df["aTIS_density_lof"] = df["aTIS_obs_lof"] / df["aTIS_length_aa"]

    print(f"  Calculated metrics for {len(df)} aTIS regions")

    return df


def calculate_canonical_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate density metrics for canonical CDS.

    O/E ratios already exist in gnomAD data.

    Parameters
    ----------
    df : pd.DataFrame
        Features with gnomAD constraint

    Returns
    -------
    pd.DataFrame
        Data with canonical metrics added
    """
    print("\nCalculating canonical CDS metrics...")

    # O/E ratios (already in gnomAD data - no calculation needed)
    # Already named correctly with canonical_ prefix

    # Variant densities (per amino acid)
    df["canonical_density_mis"] = df["canonical_obs_mis"] / df["canonical_length_aa"]
    df["canonical_density_syn"] = df["canonical_obs_syn"] / df["canonical_length_aa"]
    df["canonical_density_lof"] = df["canonical_obs_lof"] / df["canonical_length_aa"]

    print(f"  Calculated metrics for {len(df)} canonical CDS")

    return df


def calculate_paired_differences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate paired differences and ratios (aTIS vs canonical).

    Parameters
    ----------
    df : pd.DataFrame
        Features with both aTIS and canonical metrics

    Returns
    -------
    pd.DataFrame
        Data with difference/ratio columns added
    """
    print("\nCalculating paired differences...")

    # --- O/E differences (aTIS - canonical) ---
    df["delta_oe_mis"] = df["aTIS_oe_mis"] - df["canonical_oe_mis"]
    df["delta_oe_syn"] = df["aTIS_oe_syn"] - df["canonical_oe_syn"]
    df["delta_oe_lof"] = df["aTIS_oe_lof"] - df["canonical_oe_lof"]

    # --- O/E ratios (aTIS / canonical) ---
    pseudocount = 0.001
    df["ratio_oe_mis"] = df["aTIS_oe_mis"] / (df["canonical_oe_mis"] + pseudocount)
    df["ratio_oe_syn"] = df["aTIS_oe_syn"] / (df["canonical_oe_syn"] + pseudocount)
    df["ratio_oe_lof"] = df["aTIS_oe_lof"] / (df["canonical_oe_lof"] + pseudocount)

    # --- Density differences (aTIS - canonical) ---
    df["delta_density_mis"] = df["aTIS_density_mis"] - df["canonical_density_mis"]
    df["delta_density_syn"] = df["aTIS_density_syn"] - df["canonical_density_syn"]
    df["delta_density_lof"] = df["aTIS_density_lof"] - df["canonical_density_lof"]

    # --- Density ratios (aTIS / canonical) ---
    df["ratio_density_mis"] = df["aTIS_density_mis"] / (
        df["canonical_density_mis"] + pseudocount
    )
    df["ratio_density_syn"] = df["aTIS_density_syn"] / (
        df["canonical_density_syn"] + pseudocount
    )
    df["ratio_density_lof"] = df["aTIS_density_lof"] / (
        df["canonical_density_lof"] + pseudocount
    )

    print(f"  Calculated paired comparisons for {len(df)} features")

    return df


def add_loeuf_deciles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin genes into LOEUF deciles (Whiffin approach).

    Decile 1 = most constrained (lowest LOEUF)
    Decile 10 = most tolerant (highest LOEUF)

    Parameters
    ----------
    df : pd.DataFrame
        Features with LOEUF scores

    Returns
    -------
    pd.DataFrame
        Data with loeuf_decile column
    """
    print("\nCalculating LOEUF deciles...")

    # Remove any NaN LOEUF values
    valid_loeuf = df["loeuf"].notna()
    n_valid = valid_loeuf.sum()

    df["loeuf_decile"] = pd.NA

    if n_valid > 0:
        df.loc[valid_loeuf, "loeuf_decile"] = pd.qcut(
            df.loc[valid_loeuf, "loeuf"], q=10, labels=range(1, 11), duplicates="drop"
        )

        print(f"  Assigned {n_valid} features to LOEUF deciles")
        print("\n  LOEUF ranges by decile:")
        for decile in range(1, 11):
            mask = df["loeuf_decile"] == decile
            if mask.sum() > 0:
                loeuf_vals = df.loc[mask, "loeuf"]
                print(
                    f"    Decile {decile:2d}: [{loeuf_vals.min():.3f}, {loeuf_vals.max():.3f}] (n={mask.sum()})"
                )

    return df


def summarize_metrics(df: pd.DataFrame) -> None:
    """Print summary of calculated metrics."""
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)

    metrics = {
        "aTIS O/E missense": "aTIS_oe_mis",
        "aTIS O/E synonymous": "aTIS_oe_syn",
        "aTIS O/E LoF": "aTIS_oe_lof",
        "Canonical O/E missense": "canonical_oe_mis",
        "Canonical O/E synonymous": "canonical_oe_syn",
        "Canonical O/E LoF": "canonical_oe_lof",
    }

    for label, col in metrics.items():
        if col in df.columns:
            vals = df[col].dropna()
            print(f"\n{label}:")
            print(f"  Mean: {vals.mean():.3f}")
            print(f"  Median: {vals.median():.3f}")
            print(f"  Range: [{vals.min():.3f}, {vals.max():.3f}]")

    # Paired differences
    print("\n" + "-" * 60)
    print("Paired differences (aTIS - canonical):")
    print("-" * 60)

    for metric_type in ["oe_mis", "oe_syn", "oe_lof"]:
        delta_col = f"delta_{metric_type}"
        if delta_col in df.columns:
            vals = df[delta_col].dropna()
            n_lower = (vals < 0).sum()
            pct_lower = 100 * n_lower / len(vals)
            print(f"\n{metric_type.upper()}:")
            print(f"  Mean delta: {vals.mean():.3f}")
            print(f"  Median delta: {vals.median():.3f}")
            print(f"  aTIS < canonical: {n_lower}/{len(vals)} ({pct_lower:.1f}%)")


def main():
    """Main pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "merged_features.csv"
    output_file = project_root / "data" / "features_with_metrics.csv"

    # Load data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} features")

    # Calculate all metrics
    df = calculate_atis_metrics(df)
    df = calculate_canonical_metrics(df)
    df = calculate_paired_differences(df)

    # Note: LOEUF deciles calculated separately in step 04 if needed

    # Summarize
    summarize_metrics(df)

    # Save
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    print(f"  Final dataset: {len(df)} features with all metrics")

    return df


if __name__ == "__main__":
    main()
