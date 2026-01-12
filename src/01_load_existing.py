#!/usr/bin/env python3
"""
Step 1: Load existing data from SwissIsoform.

Loads isoform-level results containing aTIS extension regions
and their gnomAD variant counts.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_project_root


def load_isoform_results(filepath: Path, config: dict) -> pd.DataFrame:
    """
    Load isoform-level results CSV.

    Parameters
    ----------
    filepath : Path
        Path to isoform_level_results_mane.csv
    config : dict
        Configuration with column mappings

    Returns
    -------
    pd.DataFrame
        Loaded and validated DataFrame
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filepath.name}")
    print(f"Columns: {len(df.columns)}")

    # Get column mappings
    cols = config.get("columns", {})

    # Validate required columns exist
    required = [
        cols.get("gene", "gene_name"),
        cols.get("transcript", "transcript_id"),
        cols.get("feature_type", "feature_type"),
        cols.get("gnomad_missense", "count_gnomad_missense_variant"),
        cols.get("gnomad_synonymous", "count_gnomad_synonymous_variant"),
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"WARNING: Missing required columns: {missing}")

    return df


def filter_extensions(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Filter to extension features only.

    Parameters
    ----------
    df : pd.DataFrame
        Full isoform results
    config : dict
        Configuration

    Returns
    -------
    pd.DataFrame
        Filtered to extensions only
    """
    cols = config.get("columns", {})
    feature_col = cols.get("feature_type", "feature_type")
    feature_types = config.get("feature_types", ["extension"])

    mask = df[feature_col].isin(feature_types)
    filtered = df[mask].copy()

    print(f"Filtered to {len(filtered)} extension features")
    return filtered


def summarize_data(df: pd.DataFrame, config: dict) -> dict:
    """
    Generate summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Extension data
    config : dict
        Configuration with column mappings

    Returns
    -------
    dict
        Summary statistics
    """
    cols = config.get("columns", {})

    mis_col = cols.get("gnomad_missense", "count_gnomad_missense_variant")
    syn_col = cols.get("gnomad_synonymous", "count_gnomad_synonymous_variant")
    nonsense_col = cols.get("gnomad_nonsense", "count_gnomad_nonsense_variant")
    length_col = cols.get("feature_length_aa", "feature_length_aa")
    gene_col = cols.get("gene", "gene_name")

    summary = {
        "n_extensions": len(df),
        "n_genes": df[gene_col].nunique(),
        "total_gnomad_missense": int(df[mis_col].sum()),
        "total_gnomad_synonymous": int(df[syn_col].sum()),
    }

    if nonsense_col in df.columns:
        summary["total_gnomad_nonsense"] = int(df[nonsense_col].sum())

    # Extension length stats
    if length_col in df.columns:
        summary["mean_extension_length_aa"] = round(df[length_col].mean(), 1)
        summary["median_extension_length_aa"] = round(df[length_col].median(), 1)
        summary["max_extension_length_aa"] = int(df[length_col].max())

    # Regions with variants
    summary["extensions_with_missense"] = int((df[mis_col] > 0).sum())
    summary["extensions_with_synonymous"] = int((df[syn_col] > 0).sum())

    return summary


def print_summary(summary: dict) -> None:
    """Print formatted summary."""
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)

    for key, value in summary.items():
        label = key.replace("_", " ").title()
        print(f"  {label}: {value}")


def main(config_path: str = "config.yaml"):
    """Main function to load and summarize data."""
    config = load_config(config_path)
    project_root = get_project_root()

    # Paths
    data_dir = project_root / config["data"]["swissisoform_dir"]
    input_file = data_dir / config["data"]["isoform_results"]

    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Input file: {input_file}")
    print()

    # Check paths
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please create symlink:")
        print(f"  ln -s /path/to/swissisoform/mutations {data_dir}")
        return None

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return None

    # Load data
    print("=" * 60)
    print("Loading isoform results...")
    print("=" * 60)
    df = load_isoform_results(input_file, config)

    # Filter to extensions
    print("\n" + "=" * 60)
    print("Filtering to extensions...")
    print("=" * 60)
    extensions_df = filter_extensions(df, config)

    # Summarize
    summary = summarize_data(extensions_df, config)
    print_summary(summary)

    # Preview
    cols = config.get("columns", {})
    preview_cols = [
        cols.get("gene", "gene_name"),
        cols.get("transcript", "transcript_id"),
        cols.get("feature_type", "feature_type"),
        cols.get("feature_length_aa", "feature_length_aa"),
        cols.get("gnomad_missense", "count_gnomad_missense_variant"),
        cols.get("gnomad_synonymous", "count_gnomad_synonymous_variant"),
    ]
    preview_cols = [c for c in preview_cols if c in extensions_df.columns]

    print("\n" + "=" * 60)
    print("Preview (first 10 rows):")
    print("=" * 60)
    print(extensions_df[preview_cols].head(10).to_string(index=False))

    return extensions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load SwissIsoform extension data"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    df = main(args.config)
