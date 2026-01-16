# atis-constraint

Evolutionary constraint analysis of alternative translation initiation site (aTIS) regions using gnomAD variants.

## Method Overview

Two-stage analysis following validated approaches from Karczewski et al. 2020 and Wieder et al. 2024:

**Stage 1: Evolutionary Constraint**
- Step C: Composition check (mis/syn ratio validation)
- Step D: LOEUF stratification (gene-level constraint correlation)
- Step E: Within-gene paired comparison (primary analysis)

**Stage 2: Disease Variant Discovery**
- ClinVar pathogenic variants in constrained aTIS regions
- Cross-reference with gnomAD absence

## Usage

```bash
# Stage 1 Pipeline
python src/01_load_existing.py           # Load SwissIsoform data
python src/02_get_expected_counts.py     # Calculate expected counts
python src/03_calculate_oe.py            # O/E ratios with CIs
python src/04_merge_gnomad_constraint.py # Merge LOEUF + canonical O/E (downloads gnomAD)
python src/05_loeuf_stratification.py    # LOEUF stratification analysis
python src/06_within_gene_comparison.py  # Within-gene comparison (PRIMARY)
python src/07_composition_check.py       # Composition check (sensitivity)

# Stage 2 Pipeline
python src/08_disease_variant_discovery.py  # Disease variant identification
python src/09_visualize.py               # Publication figures
```

## Key Analyses

### Within-Gene Comparison (Primary)
Compares constraint between aTIS regions and canonical CDS within the same gene:
```
ratio = feature_O/E / canonical_O/E
```
- ratio < 1 → feature MORE constrained than canonical
- ratio = 1 → similar constraint
- ratio > 1 → feature LESS constrained

### LOEUF Stratification
Tests if aTIS constraint correlates with gene-level constraint (LOEUF).
Prediction: Extensions in constrained genes (low LOEUF) should show stronger depletion.

### Composition Check
Validates that mis/syn ratios are comparable between region types (~2.5 genome-wide).

## Output

```
results/
├── features_with_gnomad.csv       # Features with LOEUF and canonical O/E
├── loeuf_stratification_summary.md # LOEUF stratification results
├── loeuf_oe_by_decile.csv         # O/E by LOEUF decile
├── within_gene_comparison_summary.md # Within-gene comparison results
├── within_gene_ratios.csv         # Feature/canonical O/E ratios
├── composition_check_summary.md   # Composition check results
├── disease_variants_summary.md    # Disease variant discovery results (Stage 2)
├── figures/
│   ├── fig1_by_type.pdf           # O/E by feature type
│   ├── fig2_by_length.pdf         # O/E by length
│   ├── fig5_by_loeuf.pdf          # O/E by LOEUF decile
│   ├── fig6_within_gene_ratio.pdf # Ratio distribution
│   └── fig7_scatter_oe.pdf        # Feature vs canonical O/E
└── [legacy outputs...]
```

## Source Code

| File | Description |
|------|-------------|
| `utils.py` | O/E calculation, Poisson CIs, statistical tests |
| `01_load_existing.py` | Load features from SwissIsoform |
| `02_get_expected_counts.py` | Expected missense from synonymous × 2.5 |
| `03_calculate_oe.py` | Per-feature O/E with CIs |
| `04_merge_gnomad_constraint.py` | Download + merge gnomAD LOEUF/canonical O/E |
| `05_loeuf_stratification.py` | LOEUF stratification analysis |
| `06_within_gene_comparison.py` | Within-gene paired comparison |
| `07_composition_check.py` | Mis/syn ratio validation |
| `08_disease_variant_discovery.py` | Disease variant identification (Stage 2) |
| `09_visualize.py` | Publication figures |

## Data

**Input:** `data/swissisoform/isoform_level_results_mane.csv`
- 3,228 extensions + 1,924 truncations
- gnomAD variant counts per feature

**External data (auto-downloaded):**
- gnomAD v2.1.1 constraint metrics (LOEUF, canonical O/E)

## References

- Karczewski et al. 2020 - gnomAD constraint framework
- Wieder et al. 2024 - 5'UTR features correlate with LOEUF
