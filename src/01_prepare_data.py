#!/usr/bin/env python3
"""
Load SwissIsoform aTIS data and merge with gnomAD v4.1 constraint metrics.

Downloads gnomAD v4.1 if not present, then merges with SwissIsoform features.

Output: data/merged_features.csv
"""

from pathlib import Path
import pandas as pd
import urllib.request
import sys


def download_gnomad_v4(output_path: Path) -> None:
    """
    Download gnomAD v4.1 constraint metrics if not present.

    Parameters
    ----------
    output_path : Path
        Where to save the downloaded file
    """
    if output_path.exists():
        print(f"gnomAD v4.1 file already exists: {output_path}")
        return

    print("Downloading gnomAD v4.1 constraint metrics...")
    print("  This is a ~150MB file and may take a few minutes")

    url = "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv"

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        sys.stdout.write(
            f"\r  Progress: {percent}% ({downloaded // 1_000_000}MB / {total_size // 1_000_000}MB)"
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, progress_hook)
    print(f"\n✓ Downloaded to: {output_path}")


def load_swissisoform_raw(swissisoform_file: Path) -> pd.DataFrame:
    """
    Load raw SwissIsoform data and subset to essential columns.

    Returns
    -------
    pd.DataFrame
        Features with only essential columns
    """
    print(f"\nLoading raw SwissIsoform data from: {swissisoform_file}")
    df = pd.read_csv(swissisoform_file)
    print(f"  Loaded {len(df)} features")
    print(f"  Total columns: {len(df.columns)}")

    # Essential columns only
    essential_cols = [
        # Identifiers
        "gene_name",
        "transcript_id",
        "feature_id",
        "feature_type",
        # Feature info
        "feature_start",
        "feature_end",
        "feature_length_aa",
        # gnomAD variant counts (observed in aTIS region)
        "count_gnomad_missense_variant",
        "count_gnomad_synonymous_variant",
        "count_gnomad_nonsense_variant",
        "count_gnomad_frameshift_variant",
    ]

    # Check which columns exist
    available = [c for c in essential_cols if c in df.columns]
    missing = [c for c in essential_cols if c not in df.columns]

    if missing:
        print(f"  Warning: Missing columns: {missing}")

    # Subset to essential columns
    df_subset = df[available].copy()
    print(f"  Subset to {len(df_subset.columns)} essential columns")

    # Strip version from transcript_id for gnomAD matching
    # SwissIsoform has versions (e.g., ENST00000209873.8)
    # gnomAD v4.1 doesn't (e.g., ENST00000209873)
    df_subset["transcript_id_clean"] = df_subset["transcript_id"].str.split(".").str[0]
    print("  Created transcript_id_clean (version stripped)")

    return df_subset


def load_gnomad_constraint(gnomad_file: Path) -> pd.DataFrame:
    """
    Load gnomAD v4.1 constraint metrics.

    Returns
    -------
    pd.DataFrame
        Constraint metrics per transcript
    """
    print(f"\nLoading gnomAD v4.1 constraint from: {gnomad_file}")

    df = pd.read_csv(gnomad_file, sep="\t")
    print(f"  Loaded {len(df)} transcript entries")

    # Map v4.1 column names to our standard names
    # gnomAD v4.1 uses dotted names: lof.obs, mis.obs, syn.obs
    column_mapping = {
        "gene": "gnomad_gene",
        "transcript": "gnomad_transcript",
        "chromosome": "chromosome",
        # Missense
        "mis.obs": "canonical_obs_mis",
        "mis.exp": "canonical_exp_mis",
        "mis.oe": "canonical_oe_mis",
        # Synonymous
        "syn.obs": "canonical_obs_syn",
        "syn.exp": "canonical_exp_syn",
        "syn.oe": "canonical_oe_syn",
        # LoF
        "lof.obs": "canonical_obs_lof",
        "lof.exp": "canonical_exp_lof",
        "lof.oe": "canonical_oe_lof",
        "lof.oe_ci.upper": "loeuf",  # LOEUF (LoF O/E Upper bound)
        # Gene info
        "cds_length": "canonical_cds_length",
        "mane_select": "mane_select",
        "canonical": "is_canonical",
    }

    # Check which columns exist
    available_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    missing = [k for k in column_mapping.keys() if k not in df.columns]

    if missing:
        print(f"  Warning: Missing expected columns: {missing}")

    # Rename columns
    df = df.rename(columns=available_mapping)

    # Subset to renamed columns
    keep_cols = list(available_mapping.values())
    df = df[keep_cols].copy()

    print(f"  Subset to {len(df.columns)} essential columns")

    # Show MANE and canonical counts
    if "mane_select" in df.columns:
        n_mane = df["mane_select"].sum()
        print(f"  MANE Select transcripts: {n_mane}")

    if "is_canonical" in df.columns:
        n_canonical = df["is_canonical"].sum()
        print(f"  Canonical transcripts: {n_canonical}")

    return df


def merge_data(features_df: pd.DataFrame, gnomad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SwissIsoform features with gnomAD constraint.

    Parameters
    ----------
    features_df : pd.DataFrame
        aTIS features from SwissIsoform
    gnomad_df : pd.DataFrame
        gnomAD constraint metrics

    Returns
    -------
    pd.DataFrame
        Merged data
    """
    print("\nMerging SwissIsoform features with gnomAD v4.1 constraint...")
    print(f"  Features: {len(features_df)}")
    print(f"  gnomAD transcripts: {len(gnomad_df)}")

    # Merge on transcript ID (version already stripped in SwissIsoform)
    merged = features_df.merge(
        gnomad_df,
        left_on="transcript_id_clean",
        right_on="gnomad_transcript",
        how="left",
    )

    # Report stats
    n_matched = merged["gnomad_transcript"].notna().sum()
    n_unmatched = merged["gnomad_transcript"].isna().sum()

    print(f"\n  Matched: {n_matched} ({100 * n_matched / len(merged):.1f}%)")
    print(f"  Unmatched: {n_unmatched} ({100 * n_unmatched / len(merged):.1f}%)")

    # Check for duplicate feature_ids (should be none - multiple features per gene is OK!)
    n_duplicates = len(merged) - merged["feature_id"].nunique()
    if n_duplicates > 0:
        print(f"  WARNING: {n_duplicates} duplicate feature_ids found!")
    else:
        print("  ✓ No duplicate feature_ids (multiple features per gene is expected)")

    if "loeuf" in merged.columns:
        n_loeuf = merged["loeuf"].notna().sum()
        print(f"  Features with LOEUF: {n_loeuf} ({100 * n_loeuf / len(merged):.1f}%)")

    return merged


def clean_variant_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure variant count columns are clean integers.

    Parameters
    ----------
    df : pd.DataFrame
        Merged data

    Returns
    -------
    pd.DataFrame
        Data with clean variant counts
    """
    print("\nCleaning variant count columns...")

    # aTIS observed counts (from gnomAD variants in SwissIsoform regions)
    variant_cols = [
        "count_gnomad_missense_variant",
        "count_gnomad_synonymous_variant",
        "count_gnomad_nonsense_variant",
        "count_gnomad_frameshift_variant",
    ]

    for col in variant_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Create combined LoF count
    if all(
        c in df.columns
        for c in ["count_gnomad_nonsense_variant", "count_gnomad_frameshift_variant"]
    ):
        df["count_gnomad_lof"] = (
            df["count_gnomad_nonsense_variant"] + df["count_gnomad_frameshift_variant"]
        )

    return df


def filter_to_valid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to features with valid data for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Merged data

    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    print("\nFiltering to valid features...")
    print(f"  Starting: {len(df)} features")

    # Must have gnomAD match (on transcript)
    df = df[df["gnomad_transcript"].notna()].copy()
    print(f"  After gnomAD match: {len(df)}")

    # Must have LOEUF
    df = df[df["loeuf"].notna()].copy()
    print(f"  After LOEUF filter: {len(df)}")

    # Must have canonical constraint metrics
    df = df[df["canonical_exp_mis"].notna()].copy()
    df = df[df["canonical_exp_syn"].notna()].copy()
    df = df[df["canonical_exp_lof"].notna()].copy()
    print(f"  After canonical metrics filter: {len(df)}")

    # Must have feature length
    df = df[df["feature_length_aa"].notna()].copy()
    df = df[df["feature_length_aa"] > 0].copy()
    print(f"  After length filter: {len(df)}")

    # Must have canonical CDS length
    df = df[df["canonical_cds_length"].notna()].copy()
    df = df[df["canonical_cds_length"] > 0].copy()
    print(f"  After canonical CDS length filter: {len(df)}")

    return df


def summarize_data(df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)

    print(f"\nTotal features: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    if "gene_name" in df.columns:
        print(f"Unique genes: {df['gene_name'].nunique()}")
    if "transcript_id" in df.columns:
        print(f"Unique transcripts: {df['transcript_id'].nunique()}")

    # By feature type
    if "feature_type" in df.columns:
        print("\nFeatures by type:")
        for ft in sorted(df["feature_type"].unique()):
            n = (df["feature_type"] == ft).sum()
            print(f"  {ft}: {n}")

    # LOEUF distribution
    if "loeuf" in df.columns:
        loeuf = df["loeuf"].dropna()
        print(f"\nLOEUF distribution (n={len(loeuf)}):")
        print(f"  Mean: {loeuf.mean():.3f}")
        print(f"  Median: {loeuf.median():.3f}")
        print(f"  Range: [{loeuf.min():.3f}, {loeuf.max():.3f}]")

    # Variant counts
    if "count_gnomad_missense_variant" in df.columns:
        mis = df["count_gnomad_missense_variant"]
        print("\nMissense variants per aTIS feature:")
        print(f"  Mean: {mis.mean():.1f}")
        print(f"  Median: {mis.median():.1f}")
        print(
            f"  Features with ≥1 variant: {(mis > 0).sum()} ({100 * (mis > 0).mean():.1f}%)"
        )

    # Feature length
    if "feature_length_aa" in df.columns:
        length = df["feature_length_aa"]
        print("\naTIS feature length (amino acids):")
        print(f"  Mean: {length.mean():.1f}")
        print(f"  Median: {length.median():.1f}")
        print(f"  Range: [{length.min():.0f}, {length.max():.0f}]")


def main():
    """Main pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    # Download gnomAD v4.1 if needed
    gnomad_file = data_dir / "gnomad" / "gnomad.v4.1.constraint_metrics.tsv"
    download_gnomad_v4(gnomad_file)

    # Load raw SwissIsoform data
    swissisoform_file = data_dir / "swissisoform" / "isoform_level_results_mane.csv"
    features_df = load_swissisoform_raw(swissisoform_file)

    # Clean variant counts
    variant_cols = [
        "count_gnomad_missense_variant",
        "count_gnomad_synonymous_variant",
        "count_gnomad_nonsense_variant",
        "count_gnomad_frameshift_variant",
    ]

    for col in variant_cols:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(0).astype(int)

    # Create combined LoF count
    if all(
        c in features_df.columns
        for c in ["count_gnomad_nonsense_variant", "count_gnomad_frameshift_variant"]
    ):
        features_df["count_gnomad_lof"] = (
            features_df["count_gnomad_nonsense_variant"]
            + features_df["count_gnomad_frameshift_variant"]
        )

    # Load gnomAD v4.1 constraint
    gnomad_df = load_gnomad_constraint(gnomad_file)

    # Merge
    merged_df = merge_data(features_df, gnomad_df)

    # Filter to valid features
    merged_df = filter_to_valid_features(merged_df)

    # Summarize
    summarize_data(merged_df)

    # Save
    output_file = output_dir / "merged_features.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    print(
        f"  Final dataset: {len(merged_df)} features, {len(merged_df.columns)} columns"
    )

    return merged_df


if __name__ == "__main__":
    main()
