# aTIS Constraint Analysis

Evolutionary constraint analysis of alternative translation initiation site (aTIS) regions using gnomAD variants.

## Overview

This analysis compares constraint metrics between:
- **aTIS regions**: 5'UTR extensions and N-terminal truncations
- **Canonical CDS**: Standard protein-coding sequences

Following the approach from **Whiffin et al. 2024** (Genome Biology), we use:
1. Simple paired comparisons (aTIS vs canonical within same gene)
2. LOEUF stratification to test correlation with gene-level constraint

## Dataset

**Source**: SwissIsoform MANE isoform results
- 4,817 aTIS features (3,006 extensions + 1,811 truncations)
- gnomAD v2.1.1 variant counts and constraint metrics

**Only essential columns retained**:
- Identifiers: gene_name, transcript_id, feature_id, feature_type
- Feature info: feature_start, feature_end, feature_length_aa
- gnomAD variant counts: missense, synonymous, nonsense, frameshift

## Methods

See **[METHODS.md](METHODS.md)** for detailed methodology.

### Key Approach

**Constraint metrics calculated for both aTIS and canonical regions:**

1. **O/E Ratios** (Observed/Expected)
   - Missense, Synonymous, LoF O/E
   - aTIS expected counts scaled from gnomAD canonical using length ratio: `exp_aTIS = exp_canonical × (AA_aTIS / AA_canonical)`
   - **O/E < 1** = constrained (fewer variants than expected)
   - **O/E > 1** = tolerant (more variants than expected)

2. **Variant Densities** (variants per amino acid)
   - Missense, Synonymous, LoF densities

3. **Paired Comparisons** (within-gene)
   - Delta: aTIS - canonical
   - Ratio: aTIS / canonical

### Data Integration

- **Transcript-based merge**: SwissIsoform features matched to gnomAD constraint metrics via Ensembl transcript IDs (version-stripped)
- **Quality filters**: Requires gnomAD match, LOEUF score, valid lengths, complete constraint metrics
- **Result**: 4,817 features from 2,828 genes with paired aTIS/canonical metrics

## Pipeline

```bash
# 1. Prepare data (merge SwissIsoform aTIS features with gnomAD constraint)
python src/01_prepare_data.py
# Output: data/merged_features.csv (4,817 features, 27 columns)

# 2. Calculate constraint metrics (O/E ratios, densities, paired comparisons)
python src/02_calculate_metrics.py
# Output: data/features_with_metrics.csv (4,817 features with all metrics)

# 3. Visualize distributions (exploratory plots)
python src/03_plot_metrics.py
# Output: results/figures/{all,extensions,truncations}/
```

**Status**: Exploratory analysis complete. Statistical testing (step 04) pending.

## Preliminary Observations (Exploratory Plots)

From visual inspection of distributions (`results/figures/`):

1. **aTIS regions show higher O/E ratios than canonical CDS**
   - aTIS O/E distributions shifted right (higher O/E = less constrained)
   - Most aTIS regions have O/E > 1 (more variants than expected)

2. **Paired comparisons show consistent patterns**
   - Median delta O/E > 0 for most features (aTIS less constrained)
   - Small fraction (~10-15%) show negative delta (aTIS more constrained than canonical)

3. **Extensions vs Truncations**
   - Both feature types show similar constraint patterns
   - Separate plots enable visual comparison

**Note**: These are descriptive observations. Formal statistical testing (paired tests, LOEUF stratification) pending in step 04.

## Outputs

```
data/
├── merged_features.csv          # SwissIsoform + gnomAD merged (4,817 features, 27 columns)
└── features_with_metrics.csv    # All calculated constraint metrics (4,817 features)

results/figures/
├── all/                         # Plots for all 4,817 features
│   ├── 01_oe_distributions.png
│   ├── 02_density_distributions.png
│   ├── 03_oe_scatter.png
│   └── 04_delta_distributions.png
├── extensions/                  # Plots for 3,006 extensions
│   └── [same 4 plots]
└── truncations/                 # Plots for 1,811 truncations
    └── [same 4 plots]
```

## References

- **Methodology**: Inspired by Whiffin et al. 2024. "Differences in 5'untranslated regions highlight the importance of translational regulation of dosage sensitive genes." *Genome Biology* 25:111. [DOI: 10.1186/s13059-024-03248-0](https://doi.org/10.1186/s13059-024-03248-0)

- **Data sources**:
  - Karczewski KJ, Francioli LC, Tiao G, et al. (2020) The mutational constraint spectrum quantified from variation in 141,456 humans. *Nature* 581:434-443. [gnomAD v2.1.1]
  - SwissIsoform MANE alternative isoform database

- **Detailed methods**: See [METHODS.md](METHODS.md)
