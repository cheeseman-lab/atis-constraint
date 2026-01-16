#!/usr/bin/env python3
"""
Merge gnomAD constraint metrics with feature data.

Downloads gnomAD v2.1.1 constraint file (if needed) and merges:
- LOEUF scores for stratification
- Canonical transcript O/E for within-gene comparison
"""

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_project_root

# gnomAD constraint file URL
GNOMAD_CONSTRAINT_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/"
    "constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
)


def download_gnomad_constraint(output_path: Path, force: bool = False) -> Path:
    """
    Download gnomAD v2.1.1 constraint file if not present.

    Parameters
    ----------
    output_path : Path
        Where to save the file
    force : bool
        Re-download even if file exists

    Returns
    -------
    Path
        Path to the downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"gnomAD constraint file already exists: {output_path}")
        return output_path

    print(f"Downloading gnomAD v2.1.1 constraint file...")
    print(f"  URL: {GNOMAD_CONSTRAINT_URL}")
    print(f"  Destination: {output_path}")

    try:
        urllib.request.urlretrieve(GNOMAD_CONSTRAINT_URL, output_path)
        print(f"  Download complete: {output_path.stat().st_size / 1e6:.1f} MB")
    except Exception as e:
        print(f"ERROR: Failed to download gnomAD constraint file: {e}")
        raise

    return output_path


def load_gnomad_constraint(file_path: Path) -> pd.DataFrame:
    """
    Load and parse gnomAD constraint file.

    Parameters
    ----------
    file_path : Path
        Path to bgzipped constraint file

    Returns
    -------
    pd.DataFrame
        Constraint metrics per gene
    """
    print(f"Loading gnomAD constraint data from: {file_path}")

    # Read bgzipped TSV
    with gzip.open(file_path, "rt") as f:
        df = pd.read_csv(f, sep="\t")

    print(f"  Loaded {len(df)} transcripts")

    # Key columns we need
    key_cols = [
        "gene", "transcript", "canonical",
        "obs_mis", "exp_mis", "oe_mis", "oe_mis_lower", "oe_mis_upper",
        "obs_syn", "exp_syn", "oe_syn",
        "obs_lof", "exp_lof", "oe_lof", "oe_lof_lower", "oe_lof_upper",
        "pLI", "cds_length"
    ]

    # Check which columns exist
    available_cols = [c for c in key_cols if c in df.columns]
    missing_cols = [c for c in key_cols if c not in df.columns]
    if missing_cols:
        print(f"  Note: Missing columns: {missing_cols}")

    # Filter to canonical transcripts only
    if "canonical" in df.columns:
        df_canonical = df[df["canonical"] == True].copy()
        print(f"  Filtered to {len(df_canonical)} canonical transcripts")
    else:
        df_canonical = df.copy()
        print("  Warning: No 'canonical' column, using all transcripts")

    # Calculate LOEUF (upper bound of LoF O/E CI) - this is the standard constraint metric
    # It's already in the file as oe_lof_upper
    if "oe_lof_upper" in df_canonical.columns:
        df_canonical["loeuf"] = df_canonical["oe_lof_upper"]
    else:
        print("  Warning: oe_lof_upper not found, cannot compute LOEUF")

    # Rename for clarity
    df_canonical = df_canonical.rename(columns={
        "gene": "gnomad_gene",
        "transcript": "gnomad_transcript",
        "obs_mis": "canonical_obs_mis",
        "exp_mis": "canonical_exp_mis",
        "oe_mis": "canonical_oe_mis",
        "obs_syn": "canonical_obs_syn",
        "exp_syn": "canonical_exp_syn",
        "cds_length": "canonical_cds_length",
    })

    return df_canonical


def load_features(config: dict) -> pd.DataFrame:
    """Load feature data with O/E calculations."""
    project_root = get_project_root()

    # Try to load from results (after running previous steps)
    oe_file = project_root / "results" / "oe_per_feature.csv"
    if oe_file.exists():
        print(f"Loading features with O/E from: {oe_file}")
        return pd.read_csv(oe_file)

    # Fall back to features_with_expected.csv
    expected_file = project_root / "results" / "features_with_expected.csv"
    if expected_file.exists():
        print(f"Loading features from: {expected_file}")
        return pd.read_csv(expected_file)

    # Fall back to raw data
    data_dir = project_root / config["data"]["swissisoform_dir"]
    input_file = data_dir / config["data"]["isoform_results"]
    print(f"Loading raw features from: {input_file}")
    return pd.read_csv(input_file)


def merge_gnomad_with_features(
    features_df: pd.DataFrame,
    gnomad_df: pd.DataFrame,
    gene_col: str = "gene_name"
) -> pd.DataFrame:
    """
    Merge gnomAD constraint data with feature data.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature data (extensions/truncations)
    gnomad_df : pd.DataFrame
        gnomAD constraint data
    gene_col : str
        Column name for gene in features_df

    Returns
    -------
    pd.DataFrame
        Merged data
    """
    print(f"\nMerging gnomAD data with features...")
    print(f"  Features: {len(features_df)}")
    print(f"  gnomAD genes: {len(gnomad_df)}")

    # Merge on gene name
    merged = features_df.merge(
        gnomad_df,
        left_on=gene_col,
        right_on="gnomad_gene",
        how="left"
    )

    # Report merge stats
    n_matched = merged["gnomad_gene"].notna().sum()
    n_unmatched = merged["gnomad_gene"].isna().sum()
    print(f"  Matched: {n_matched} ({100*n_matched/len(merged):.1f}%)")
    print(f"  Unmatched: {n_unmatched} ({100*n_unmatched/len(merged):.1f}%)")

    # Report LOEUF coverage
    if "loeuf" in merged.columns:
        n_loeuf = merged["loeuf"].notna().sum()
        print(f"  Features with LOEUF: {n_loeuf} ({100*n_loeuf/len(merged):.1f}%)")

    return merged


def summarize_merged_data(df: pd.DataFrame) -> None:
    """Print summary of merged data."""
    print("\n" + "=" * 60)
    print("MERGED DATA SUMMARY")
    print("=" * 60)

    # By feature type
    if "feature_type" in df.columns:
        print("\nBy feature type:")
        for ft in df["feature_type"].unique():
            subset = df[df["feature_type"] == ft]
            n_with_loeuf = subset["loeuf"].notna().sum() if "loeuf" in df.columns else 0
            print(f"  {ft}: {len(subset)} features, {n_with_loeuf} with LOEUF")

    # LOEUF distribution
    if "loeuf" in df.columns:
        loeuf_valid = df["loeuf"].dropna()
        print(f"\nLOEUF distribution (n={len(loeuf_valid)}):")
        print(f"  Mean: {loeuf_valid.mean():.3f}")
        print(f"  Median: {loeuf_valid.median():.3f}")
        print(f"  Range: {loeuf_valid.min():.3f} - {loeuf_valid.max():.3f}")

        # Count by constraint level
        n_constrained = (loeuf_valid < 0.35).sum()
        n_intermediate = ((loeuf_valid >= 0.35) & (loeuf_valid < 1.0)).sum()
        n_unconstrained = (loeuf_valid >= 1.0).sum()
        print(f"\n  Highly constrained (LOEUF < 0.35): {n_constrained}")
        print(f"  Intermediate (0.35 <= LOEUF < 1.0): {n_intermediate}")
        print(f"  Unconstrained (LOEUF >= 1.0): {n_unconstrained}")

    # Canonical O/E coverage
    if "canonical_oe_mis" in df.columns:
        n_with_oe = df["canonical_oe_mis"].notna().sum()
        print(f"\nFeatures with canonical O/E: {n_with_oe}")


def main(config_path: str = "config.yaml", force_download: bool = False):
    """Main function to merge gnomAD constraint data."""
    config = load_config(config_path)
    project_root = get_project_root()

    # Set up paths
    gnomad_dir = project_root / config["data"]["gnomad_dir"]
    gnomad_file = gnomad_dir / "gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"

    # Download gnomAD constraint file if needed
    download_gnomad_constraint(gnomad_file, force=force_download)

    # Load gnomAD constraint data
    gnomad_df = load_gnomad_constraint(gnomad_file)

    # Load feature data
    features_df = load_features(config)

    # Merge
    cols = config.get("columns", {})
    gene_col = cols.get("gene", "gene_name")
    merged_df = merge_gnomad_with_features(features_df, gnomad_df, gene_col=gene_col)

    # Summarize
    summarize_merged_data(merged_df)

    # Save
    output_path = project_root / "results" / "features_with_gnomad.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved merged data to: {output_path}")

    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge gnomAD constraint data with features"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download gnomAD file even if it exists"
    )
    args = parser.parse_args()

    main(args.config, args.force_download)
