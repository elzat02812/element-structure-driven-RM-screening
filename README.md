# Element-Structure-Driven Redox Mediator Screening

Code and data accompanying the manuscript on an element-structure-driven
random-forest model for screening organic redox mediators in lithium metal
batteries.

> **Status.** The associated manuscript is currently under peer review.
> A permanent Zenodo DOI will be issued upon acceptance.

## Overview

This repository reproduces the random-forest regression analysis described
in the Methods section and Supplementary Figure 1 of the manuscript. The
purpose of the model is **descriptor identification**, not high-precision
point prediction:

1. Read SMILES strings, molecular formulae, and experimental oxidation
   potentials of 10 N-containing redox mediators from
   `data/redox_mediators.csv` (Supplementary Table 1, No. 1-10).
2. Compute 10 molecular descriptors per molecule — two elemental
   descriptors (C/N ratio and DBE / degree of unsaturation) derived directly
   from the formula, and eight RDKit descriptors derived from the SMILES.
3. Train a random forest regressor (200 trees, `random_state=42`) and
   evaluate predictive accuracy by leave-one-out cross-validation (LOOCV).
4. Rank descriptors by mean decrease in impurity (MDI) and cross-check
   against an independent Spearman rank correlation analysis.

Running the workflow on the pinned dependency versions reproduces the
values reported in Supplementary Figure 1:

```
LOOCV  R^2 = 0.538   MAE = 0.129 V   (n = 10)
```

The top two descriptors ranked by MDI are **C/N** and **DBE**, in agreement
with the independent Spearman rank correlation analysis (Fig. 2b of the
manuscript). These two descriptors subsequently served as the basis for the
two-parameter (C/N vs. DBE) screening space (Fig. 2c) used for rational
design of new mediator candidates.

## Repository layout

```
.
├── data/
│   └── redox_mediators.csv      # 10 N-containing mediators (Supp. Table 1, No. 1-10)
├── src/
│   ├── train_rf_loocv.py        # Descriptors + RF + LOOCV + MDI + Spearman
│   └── plot_results.py          # Reproduces Supp. Fig. 1a/b
├── results/                     # (generated) descriptors, predictions, metrics
├── figures/                     # (generated) Supp. Fig. 1 panels
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Python 3.10 or newer is recommended. Using a fresh virtual environment:

```bash
git clone https://github.com/elzat02812/element-structure-driven-RM-screening.git
cd element-structure-driven-RM-screening

python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The dependency versions in `requirements.txt` are pinned because random
forest predictions and feature-importance rankings can vary across
scikit-learn / RDKit releases at the very small sample size used here
(n = 10). The pinned versions deterministically reproduce R² = 0.538 and
MAE = 0.129 V.

## Reproducing the results

Run the two scripts in order from the repository root:

```bash
python src/train_rf_loocv.py
python src/plot_results.py
```

Expected outputs:

| File | Description |
| --- | --- |
| `results/descriptors.csv` | 10 descriptors computed for all 10 molecules |
| `results/loocv_predictions.csv` | Per-molecule LOOCV predictions and absolute errors |
| `results/metrics.json` | R² and MAE on LOOCV |
| `results/feature_importance.csv` | MDI feature importance ranking |
| `results/spearman_correlation.csv` | Spearman rank correlation baseline |
| `figures/supp_fig1a_parity.png` | Predicted vs. experimental potential |
| `figures/supp_fig1b_importance.png` | MDI importance bar chart |

Total runtime is under 30 seconds on a standard laptop CPU; no GPU is
required.

## Molecular descriptors

Ten descriptors are used, matching Supplementary Discussion 1 of the
manuscript:

| # | Descriptor | Source | Description |
| --- | --- | --- | --- |
| 1 | `C/N` | formula | Carbon-to-nitrogen atom ratio (0 if N = 0) |
| 2 | `DBE` | formula | Double-bond equivalents: C + 1 − (H + Cl)/2 + N/2 |
| 3 | `FractionCSP3` | RDKit | Fraction of sp³ carbons |
| 4 | `TPSA` | RDKit | Topological polar surface area |
| 5 | `MaxAbsCharge` | RDKit | Max |Gasteiger partial charge| |
| 6 | `MolLogP` | RDKit (Crippen) | Octanol–water partition coefficient |
| 7 | `MolWt` | RDKit | Relative molecular weight |
| 8 | `NumRotatableBonds` | RDKit | Number of rotatable bonds |
| 9 | `NumHDonors` | RDKit | Number of hydrogen-bond donors |
| 10 | `NumHeteroatoms` | RDKit | Number of heteroatoms |

`DBE` and the manuscript's `DOU` are mathematically equivalent:
`C + 1 − (H + Cl)/2 + N/2 = (2C + N − H − X + 2)/2`.

## Data

`data/redox_mediators.csv` contains the 10 N-containing redox mediators
used for model training (Supplementary Table 1, No. 1–10). Columns:

| Column | Description |
| --- | --- |
| `number` | Mediator index (1–10) in Supplementary Table 1 |
| `name`, `abbreviation`, `molecular_formula`, `smiles` | Identifiers |
| `E_ox_V_vs_Li` | Oxidation potential referenced to Li⁺/Li |
| `reference` | Literature source |

All oxidation potentials are referenced to Li⁺/Li to ensure thermodynamic
consistency, as described in the Methods.

## Citation

The associated manuscript is currently under peer review. Citation
information will be added here once a preprint or accepted version is
available.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Molecular data are derived from the original literature sources listed in
`data/redox_mediators.csv`; please cite those primary references when
reusing specific data points.
