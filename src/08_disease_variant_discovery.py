#!/usr/bin/env python3
"""
Stage 2: Disease Variant Discovery

Identifies variants in constrained aTIS regions that exist in disease databases
(ClinVar, COSMIC) but are absent from population databases (gnomAD).

Logic:
1. Filter to constrained features (within-gene ratio < 1)
2. Read mutation-level data for those features
3. Match variants by position (chromosome-position-ref-alt)
4. Flag variants in ClinVar/COSMIC but NOT in gnomAD
5. Focus on deleterious variants (missense, nonsense, frameshift)
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_project_root


# Variant types to include (skip synonymous)
DELETERIOUS_TYPES = {
    "missense variant",
    "nonsense variant",
    "frameshift variant",
    "inframe deletion",
    "inframe insertion",
}

# Disease databases vs population databases
DISEASE_DBS = {"ClinVar", "COSMIC"}
POPULATION_DBS = {"gnomAD"}


def load_constrained_features(project_root: Path, ratio_threshold: float = 1.0) -> pd.DataFrame:
    """
    Load features that showed constraint in Stage 1 (ratio < threshold).

    Parameters
    ----------
    project_root : Path
        Project root directory
    ratio_threshold : float
        Maximum ratio to consider constrained (default < 1.0)

    Returns
    -------
    pd.DataFrame
        Constrained features with their metadata
    """
    ratios_file = project_root / "results" / "within_gene_ratios.csv"

    if not ratios_file.exists():
        print(f"ERROR: {ratios_file} not found")
        print("Please run Stage 1 analysis first (06_within_gene_comparison.py)")
        sys.exit(1)

    df = pd.read_csv(ratios_file)
    print(f"Loaded {len(df)} features with within-gene ratios")

    # Filter to constrained features
    constrained = df[df["oe_ratio"] < ratio_threshold].copy()
    print(f"Constrained features (oe_ratio < {ratio_threshold}): {len(constrained)}")

    return constrained


def find_mutation_files(feature_row: pd.Series, swissisoform_dir: Path) -> list:
    """
    Find mutation CSV files for a given feature.

    Parameters
    ----------
    feature_row : pd.Series
        Row from features dataframe
    swissisoform_dir : Path
        Path to SwissIsoform data directory

    Returns
    -------
    list
        List of mutation file paths
    """
    gene_name = feature_row["gene_name"]
    gene_dir = swissisoform_dir / gene_name

    if not gene_dir.exists():
        return []

    # Find all mutation files for this gene
    mutation_files = list(gene_dir.glob("**/*mutations.csv"))

    # Filter to files matching this feature's transcript if available
    if "transcript_id" in feature_row and pd.notna(feature_row.get("transcript_id")):
        transcript = feature_row["transcript_id"]
        # Extract base transcript ID (without version)
        transcript_base = transcript.split(".")[0] if "." in str(transcript) else transcript
        mutation_files = [f for f in mutation_files if transcript_base in f.name]

    return mutation_files


def read_mutations_for_feature(mutation_files: list) -> pd.DataFrame:
    """
    Read and combine mutations from multiple files.

    Parameters
    ----------
    mutation_files : list
        List of mutation file paths

    Returns
    -------
    pd.DataFrame
        Combined mutations
    """
    if not mutation_files:
        return pd.DataFrame()

    dfs = []
    for f in mutation_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def create_variant_key(row: pd.Series) -> str:
    """
    Create a unique variant key from chromosome-position-ref-alt.

    Parameters
    ----------
    row : pd.Series
        Mutation row

    Returns
    -------
    str
        Variant key
    """
    return f"{row['chromosome']}-{row['position']}-{row['reference']}-{row['alternate']}"


def analyze_variants(mutations: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze which variants are in disease vs population databases.

    Parameters
    ----------
    mutations : pd.DataFrame
        Mutation data with source column

    Returns
    -------
    pd.DataFrame
        Variants with database presence flags
    """
    if mutations.empty:
        return pd.DataFrame()

    # Filter to deleterious variants
    mutations = mutations[mutations["impact_validated"].isin(DELETERIOUS_TYPES)].copy()

    if mutations.empty:
        return pd.DataFrame()

    # Create variant key
    mutations["variant_key"] = mutations.apply(create_variant_key, axis=1)

    # Group by variant key and collect sources
    variant_sources = mutations.groupby("variant_key").agg({
        "source": lambda x: set(x),
        "chromosome": "first",
        "position": "first",
        "reference": "first",
        "alternate": "first",
        "gene_name": "first",
        "impact_validated": "first",
        "hgvsp": "first",
        "clinical_significance": lambda x: [v for v in x if pd.notna(v) and v != "Unknown"],
    }).reset_index()

    # Flag database presence
    variant_sources["in_gnomad"] = variant_sources["source"].apply(
        lambda x: bool(x & POPULATION_DBS)
    )
    variant_sources["in_disease_db"] = variant_sources["source"].apply(
        lambda x: bool(x & DISEASE_DBS)
    )
    variant_sources["sources"] = variant_sources["source"].apply(
        lambda x: ",".join(sorted(x))
    )

    # Flag disease-only variants (in ClinVar/COSMIC but NOT in gnomAD)
    variant_sources["disease_only"] = (
        variant_sources["in_disease_db"] & ~variant_sources["in_gnomad"]
    )

    # Get clinical significance (take first non-empty)
    variant_sources["clinical_sig"] = variant_sources["clinical_significance"].apply(
        lambda x: x[0] if x else "Unknown"
    )

    # Clean up
    variant_sources = variant_sources.drop(columns=["source", "clinical_significance"])

    return variant_sources


def process_all_features(
    constrained: pd.DataFrame,
    swissisoform_dir: Path
) -> tuple:
    """
    Process all constrained features and find disease-only variants.

    Parameters
    ----------
    constrained : pd.DataFrame
        Constrained features
    swissisoform_dir : Path
        Path to SwissIsoform data

    Returns
    -------
    tuple
        (all_variants DataFrame, disease_only_variants DataFrame)
    """
    all_variants = []
    features_processed = 0
    features_with_data = 0

    print(f"\nProcessing {len(constrained)} constrained features...")

    for idx, row in constrained.iterrows():
        # Find mutation files
        mutation_files = find_mutation_files(row, swissisoform_dir)

        if not mutation_files:
            continue

        features_processed += 1

        # Read mutations
        mutations = read_mutations_for_feature(mutation_files)

        if mutations.empty:
            continue

        # Analyze variants
        variants = analyze_variants(mutations)

        if variants.empty:
            continue

        features_with_data += 1

        # Add feature metadata
        variants["feature_gene"] = row["gene_name"]
        variants["feature_type"] = row.get("feature_type", "unknown")
        variants["constraint_ratio"] = row["oe_ratio"]

        all_variants.append(variants)

        if features_processed % 100 == 0:
            print(f"  Processed {features_processed} features...")

    print(f"\nProcessed {features_processed} features with mutation files")
    print(f"Features with deleterious variants: {features_with_data}")

    if not all_variants:
        return pd.DataFrame(), pd.DataFrame()

    # Combine all variants
    all_df = pd.concat(all_variants, ignore_index=True)

    # Get disease-only variants
    disease_only_df = all_df[all_df["disease_only"]].copy()

    return all_df, disease_only_df


def summarize_results(all_variants: pd.DataFrame, disease_only: pd.DataFrame) -> dict:
    """
    Generate summary statistics.

    Parameters
    ----------
    all_variants : pd.DataFrame
        All analyzed variants
    disease_only : pd.DataFrame
        Disease-only variants

    Returns
    -------
    dict
        Summary statistics
    """
    if all_variants.empty:
        return {"error": "No variants analyzed"}

    summary = {
        "total_variants": len(all_variants),
        "variants_in_gnomad": all_variants["in_gnomad"].sum(),
        "variants_in_disease_db": all_variants["in_disease_db"].sum(),
        "disease_only_variants": len(disease_only),
        "pct_disease_only": round(len(disease_only) / len(all_variants) * 100, 1),
    }

    # By variant type
    summary["by_impact"] = all_variants.groupby("impact_validated").agg({
        "variant_key": "count",
        "disease_only": "sum"
    }).rename(columns={"variant_key": "total", "disease_only": "disease_only"}).to_dict()

    # Disease-only by clinical significance
    if not disease_only.empty:
        summary["disease_only_by_clinical_sig"] = disease_only["clinical_sig"].value_counts().to_dict()
        summary["disease_only_genes"] = disease_only["gene_name"].nunique()

    return summary


def print_results(summary: dict, disease_only: pd.DataFrame) -> None:
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("STAGE 2: DISEASE VARIANT DISCOVERY RESULTS")
    print("=" * 70)

    if "error" in summary:
        print(f"\nError: {summary['error']}")
        return

    print(f"\n## Overview")
    print(f"  Total deleterious variants analyzed: {summary['total_variants']}")
    print(f"  Variants in gnomAD: {summary['variants_in_gnomad']}")
    print(f"  Variants in disease databases: {summary['variants_in_disease_db']}")
    print(f"  Disease-only variants: {summary['disease_only_variants']} ({summary['pct_disease_only']}%)")

    if summary['disease_only_variants'] > 0:
        print(f"\n## Disease-Only Variants by Impact Type")
        for impact, counts in summary.get("by_impact", {}).get("disease_only", {}).items():
            if counts > 0:
                print(f"  {impact}: {int(counts)}")

        print(f"\n## Disease-Only Variants by Clinical Significance")
        for sig, count in summary.get("disease_only_by_clinical_sig", {}).items():
            print(f"  {sig}: {count}")

        print(f"\n## Top Disease-Only Variants")
        # Show top variants by pathogenicity
        if not disease_only.empty:
            top = disease_only.sort_values("clinical_sig", ascending=False).head(10)
            for _, row in top.iterrows():
                print(f"  {row['gene_name']}: {row['variant_key']} ({row['impact_validated']}) - {row['clinical_sig']}")


def save_results(
    all_variants: pd.DataFrame,
    disease_only: pd.DataFrame,
    summary: dict,
    output_dir: Path
) -> None:
    """Save results to files."""

    # Save disease-only variants (skip all_variants to save space)
    if not disease_only.empty:
        disease_only.to_csv(output_dir / "stage2_disease_only_variants.csv", index=False)

    # Save summary markdown
    summary_path = output_dir / "stage2_disease_variant_summary.md"
    with open(summary_path, "w") as f:
        f.write("# Stage 2: Disease Variant Discovery\n\n")

        f.write("## Overview\n\n")
        f.write("This analysis identifies variants in constrained aTIS regions that exist in disease databases ")
        f.write("(ClinVar, COSMIC) but are absent from population databases (gnomAD).\n\n")

        f.write("**Rationale**: Variants that cause disease should be:\n")
        f.write("1. Present in disease databases (ClinVar, COSMIC)\n")
        f.write("2. Absent or rare in population databases (gnomAD)\n")
        f.write("3. Located in evolutionarily constrained regions\n\n")

        if "error" in summary:
            f.write(f"**Error**: {summary['error']}\n")
            return

        f.write("## Results\n\n")
        f.write(f"| Metric | Count |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Total deleterious variants | {summary['total_variants']} |\n")
        f.write(f"| Variants in gnomAD | {summary['variants_in_gnomad']} |\n")
        f.write(f"| Variants in disease DBs | {summary['variants_in_disease_db']} |\n")
        f.write(f"| **Disease-only variants** | **{summary['disease_only_variants']}** ({summary['pct_disease_only']}%) |\n")
        f.write(f"| Genes with disease-only variants | {summary.get('disease_only_genes', 0)} |\n\n")

        if summary['disease_only_variants'] > 0:
            f.write("## Disease-Only Variants by Clinical Significance\n\n")
            f.write("| Clinical Significance | Count |\n")
            f.write("|----------------------|-------|\n")
            for sig, count in sorted(summary.get("disease_only_by_clinical_sig", {}).items(),
                                     key=lambda x: -x[1]):
                f.write(f"| {sig} | {count} |\n")
            f.write("\n")

            f.write("## Interpretation\n\n")
            pathogenic_count = sum(
                v for k, v in summary.get("disease_only_by_clinical_sig", {}).items()
                if "athogenic" in k
            )
            if pathogenic_count > 0:
                f.write(f"**{pathogenic_count} pathogenic/likely pathogenic variants** were found in constrained ")
                f.write("aTIS regions that are absent from gnomAD. These represent high-confidence ")
                f.write("disease variants in alternative translation initiation regions.\n\n")

        f.write("## Output Files\n\n")
        f.write("- `stage2_disease_only_variants.csv`: Variants in disease DBs but not gnomAD\n")

    print(f"\nSaved results to: {output_dir}")


def main(config_path: str = "config.yaml", ratio_threshold: float = 1.0):
    """Main function for disease variant discovery."""

    config = load_config(config_path)
    project_root = get_project_root()

    # Get SwissIsoform directory
    swissisoform_dir = project_root / config["data"]["swissisoform_dir"]
    if not swissisoform_dir.exists():
        print(f"ERROR: SwissIsoform directory not found: {swissisoform_dir}")
        sys.exit(1)

    print(f"SwissIsoform data: {swissisoform_dir}")

    # Load constrained features from Stage 1
    constrained = load_constrained_features(project_root, ratio_threshold)

    # Process all features
    all_variants, disease_only = process_all_features(constrained, swissisoform_dir)

    # Generate summary
    summary = summarize_results(all_variants, disease_only)

    # Print results
    print_results(summary, disease_only)

    # Save results
    output_dir = project_root / "results"
    save_results(all_variants, disease_only, summary, output_dir)

    return all_variants, disease_only, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Disease Variant Discovery"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=1.0,
        help="Maximum within-gene ratio to consider constrained (default: 1.0)"
    )
    args = parser.parse_args()

    main(args.config, args.ratio_threshold)
