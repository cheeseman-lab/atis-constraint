#!/usr/bin/env python3
"""
Statistical comparison of aTIS regions vs canonical CDS.

Performs:
1. Overall paired tests (aTIS vs canonical)
2. Stratification by LOEUF deciles (Whiffin approach)
3. Comparison by feature type (extensions vs truncations)

Output: results/comparison_summary.txt
        results/loeuf_stratification.csv
"""

from pathlib import Path
import pandas as pd
from scipy import stats


def paired_comparison(df: pd.DataFrame, metric_type: str = "oe_mis") -> dict:
    """
    Perform paired comparison between aTIS and canonical for a metric.

    Parameters
    ----------
    df : pd.DataFrame
        Features with metrics
    metric_type : str
        Type of metric (e.g., 'oe_mis', 'density_mis')

    Returns
    -------
    dict
        Test results
    """
    atis_col = f"aTIS_{metric_type}"
    canonical_col = f"canonical_{metric_type}"

    # Get valid pairs
    mask = df[atis_col].notna() & df[canonical_col].notna()
    atis_vals = df.loc[mask, atis_col]
    canonical_vals = df.loc[mask, canonical_col]

    n = len(atis_vals)

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(atis_vals, canonical_vals)

    # Wilcoxon signed-rank test (non-parametric)
    w_stat, w_pval = stats.wilcoxon(atis_vals, canonical_vals)

    # Summary statistics
    delta = atis_vals - canonical_vals

    results = {
        "metric": metric_type,
        "n": n,
        "aTIS_mean": atis_vals.mean(),
        "aTIS_median": atis_vals.median(),
        "canonical_mean": canonical_vals.mean(),
        "canonical_median": canonical_vals.median(),
        "delta_mean": delta.mean(),
        "delta_median": delta.median(),
        "n_aTIS_lower": (delta < 0).sum(),
        "pct_aTIS_lower": 100 * (delta < 0).mean(),
        "t_statistic": t_stat,
        "t_pvalue": t_pval,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_pval,
    }

    return results


def overall_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all paired comparisons (aTIS vs canonical).

    Returns
    -------
    pd.DataFrame
        Summary of all comparisons
    """
    print("\n" + "=" * 60)
    print("OVERALL PAIRED COMPARISONS (aTIS vs Canonical)")
    print("=" * 60)

    metrics_to_test = [
        "oe_mis",
        "oe_syn",
        "oe_lof",
        "density_mis",
        "density_syn",
        "density_lof",
    ]

    results = []
    for metric in metrics_to_test:
        res = paired_comparison(df, metric)
        results.append(res)

        print(f"\n{metric.upper()}:")
        print(f"  aTIS: mean={res['aTIS_mean']:.3f}, median={res['aTIS_median']:.3f}")
        print(
            f"  Canonical: mean={res['canonical_mean']:.3f}, median={res['canonical_median']:.3f}"
        )
        print(
            f"  Delta: mean={res['delta_mean']:.3f}, median={res['delta_median']:.3f}"
        )
        print(
            f"  aTIS < canonical: {res['n_aTIS_lower']}/{res['n']} ({res['pct_aTIS_lower']:.1f}%)"
        )
        print(f"  Paired t-test: t={res['t_statistic']:.3f}, p={res['t_pvalue']:.2e}")
        print(
            f"  Wilcoxon test: W={res['wilcoxon_statistic']:.0f}, p={res['wilcoxon_pvalue']:.2e}"
        )

    return pd.DataFrame(results)


def comparison_by_feature_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare extensions vs truncations.

    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    print("\n" + "=" * 60)
    print("COMPARISON BY FEATURE TYPE")
    print("=" * 60)

    metrics = [
        "delta_oe_mis",
        "delta_oe_syn",
        "delta_oe_lof",
        "delta_density_mis",
        "delta_density_syn",
        "delta_density_lof",
    ]

    results = []

    for metric in metrics:
        ext = df[df["feature_type"] == "extension"][metric].dropna()
        trunc = df[df["feature_type"] == "truncation"][metric].dropna()

        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(ext, trunc, alternative="two-sided")

        results.append(
            {
                "metric": metric,
                "extension_mean": ext.mean(),
                "extension_median": ext.median(),
                "extension_n": len(ext),
                "truncation_mean": trunc.mean(),
                "truncation_median": trunc.median(),
                "truncation_n": len(trunc),
                "mannwhitneyu_statistic": u_stat,
                "mannwhitneyu_pvalue": u_pval,
            }
        )

        print(f"\n{metric}:")
        print(
            f"  Extensions: mean={ext.mean():.3f}, median={ext.median():.3f} (n={len(ext)})"
        )
        print(
            f"  Truncations: mean={trunc.mean():.3f}, median={trunc.median():.3f} (n={len(trunc)})"
        )
        print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_pval:.2e}")

    return pd.DataFrame(results)


def loeuf_stratification(
    df: pd.DataFrame, metric: str = "delta_oe_mis"
) -> pd.DataFrame:
    """
    Stratify metric by LOEUF deciles (Whiffin approach).

    Parameters
    ----------
    df : pd.DataFrame
        Features with metrics and LOEUF deciles
    metric : str
        Column name to stratify

    Returns
    -------
    pd.DataFrame
        Summary by decile
    """
    results = []

    for decile in range(1, 11):
        subset = df[df["loeuf_decile"] == decile][metric].dropna()

        if len(subset) > 0:
            results.append(
                {
                    "decile": decile,
                    "n": len(subset),
                    "mean": subset.mean(),
                    "median": subset.median(),
                    "std": subset.std(),
                    "q25": subset.quantile(0.25),
                    "q75": subset.quantile(0.75),
                    "n_negative": (subset < 0).sum(),
                    "pct_negative": 100 * (subset < 0).mean(),
                }
            )

    return pd.DataFrame(results)


def all_loeuf_stratifications(df: pd.DataFrame) -> dict:
    """
    Run LOEUF stratification for all metrics.

    Returns
    -------
    dict
        Dictionary of DataFrames, one per metric
    """
    print("\n" + "=" * 60)
    print("LOEUF STRATIFICATION (Whiffin Approach)")
    print("=" * 60)

    metrics = {
        "Delta O/E Missense": "delta_oe_mis",
        "Delta O/E Synonymous": "delta_oe_syn",
        "Delta O/E LoF": "delta_oe_lof",
        "Ratio Density Missense": "ratio_density_mis",
        "Ratio Density Synonymous": "ratio_density_syn",
        "Ratio Density LoF": "ratio_density_lof",
    }

    all_results = {}

    for label, metric in metrics.items():
        strat = loeuf_stratification(df, metric)
        all_results[metric] = strat

        print(f"\n{label} ({metric}):")
        print(
            f"  Decile 1 (constrained): mean={strat.iloc[0]['mean']:.3f}, median={strat.iloc[0]['median']:.3f}"
        )
        print(
            f"  Decile 10 (tolerant):   mean={strat.iloc[-1]['mean']:.3f}, median={strat.iloc[-1]['median']:.3f}"
        )

        # Test for trend across deciles using Spearman correlation
        valid = df[df[metric].notna() & df["loeuf_decile"].notna()]
        if len(valid) > 0:
            rho, p = stats.spearmanr(valid["loeuf_decile"], valid[metric])
            print(f"  Spearman correlation with decile: rho={rho:.3f}, p={p:.2e}")

    return all_results


def main():
    """Main analysis pipeline."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "features_with_metrics.csv"
    output_dir = project_root / "results_v2"
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} features")
    print(f"  Extensions: {(df['feature_type'] == 'extension').sum()}")
    print(f"  Truncations: {(df['feature_type'] == 'truncation').sum()}")

    # Run analyses
    overall_results = overall_comparisons(df)
    feature_type_results = comparison_by_feature_type(df)
    loeuf_results = all_loeuf_stratifications(df)

    # Save results
    overall_results.to_csv(output_dir / "overall_comparisons.csv", index=False)
    feature_type_results.to_csv(output_dir / "feature_type_comparison.csv", index=False)

    for metric, strat_df in loeuf_results.items():
        filename = f"loeuf_stratification_{metric}.csv"
        strat_df.to_csv(output_dir / filename, index=False)

    print(f"\n✓ Saved results to: {output_dir}")
    print("  - overall_comparisons.csv")
    print("  - feature_type_comparison.csv")
    print("  - loeuf_stratification_*.csv")

    return overall_results, feature_type_results, loeuf_results


if __name__ == "__main__":
    main()
