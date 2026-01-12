# atis-constraint

Evolutionary constraint analysis of aTIS regions using gnomAD variants.

## Method

```
O/E = observed missense / expected missense
Expected = synonymous × 2.5 (genome-wide mis/syn ratio)
```

- O/E < 1 → constrained (selection against missense)
- O/E = 1 → neutral

## Usage

```bash
python src/01_load_existing.py       # Load SwissIsoform data
python src/02_get_expected_counts.py # Calculate expected counts
python src/03_calculate_oe.py        # O/E ratios with CIs
python src/04_compare_constraint.py  # Statistical tests
python src/05_visualize.py           # Publication figures
```

## Output

```
results/
├── summary.md           # Main results (tables, stats)
├── most_constrained.csv # All features ranked by O/E
├── figures/             # Publication-quality figures
└── oe_per_feature.csv   # Per-feature O/E data
```

## Source Code

| File | Description |
|------|-------------|
| `utils.py` | O/E calculation, Poisson CIs |
| `01_load_existing.py` | Load and filter features from SwissIsoform |
| `02_get_expected_counts.py` | Expected missense from synonymous |
| `03_calculate_oe.py` | O/E with CIs, ranked output, summary |
| `04_compare_constraint.py` | Tests vs neutral, by codon, by length |
| `05_visualize.py` | Publication figures |

## Data

Input: `data/swissisoform/isoform_level_results_mane.csv`
- 3,228 extensions + 1,924 truncations
- gnomAD variant counts per feature
