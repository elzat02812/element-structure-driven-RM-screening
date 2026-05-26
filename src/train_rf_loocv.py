"""
Element-Structure-Driven Redox Mediator Screening Model.

Reproduces the random forest regression analysis described in the Methods
section and Supplementary Figure 1 of the manuscript:

    - Reads SMILES strings, molecular formulae, and experimental oxidation
      potentials for 10 N-containing redox mediators (Supplementary Table 1,
      No. 1-10) from data/redox_mediators.csv.
    - Computes 10 molecular descriptors per molecule: two elemental
      descriptors derived from the formula (C/N ratio and DBE), plus eight
      RDKit descriptors derived from the SMILES.
    - Trains a random forest regressor (200 trees, random_state=42) and
      evaluates predictive accuracy by leave-one-out cross-validation
      (LOOCV).
    - Ranks descriptors by mean decrease in impurity (MDI) and computes a
      Spearman rank correlation baseline.

Expected output (deterministic with the pinned dependencies in
requirements.txt):

    LOOCV  R^2 = 0.538   MAE = 0.129 V

The top two descriptors ranked by MDI are C/N ratio and DBE (degree of
unsaturation), in agreement with the independent Spearman rank correlation
analysis (Fig. 2b of the manuscript). These descriptors were subsequently
used as rational-design guides to identify high-C/N, high-DBE candidates
such as 10-phenylphenothiazine (PTH).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "C/N",
    "DBE",
    "FractionCSP3",
    "TPSA",
    "MaxAbsCharge",
    "MolLogP",
    "MolWt",
    "NumRotatableBonds",
    "NumHDonors",
    "NumHeteroatoms",
]

RF_KWARGS = dict(n_estimators=200, random_state=42)


# -----------------------------------------------------------------------------
# Descriptor computation
# -----------------------------------------------------------------------------

def atom_counts_from_formula(formula: str) -> dict:
    """Parse a molecular formula string into an atom-count dictionary.

    Supports the elements relevant to the redox-mediator library
    (C, H, N, O, S, Cl). Unsupported elements are silently ignored.
    """
    counts = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0, "Cl": 0}
    for element, count in re.findall(r"([A-Z][a-z]*)(\d*)", formula):
        if element in counts:
            counts[element] += int(count) if count else 1
    return counts


def elemental_descriptors(formula: str) -> dict:
    """Compute the two elemental descriptors directly from the formula.

    Returns
    -------
    DBE : double bond equivalent, computed as
              C + 1 - (H + Cl)/2 + N/2
          (rings + degrees of unsaturation, halogens treated as monovalent).
          Mathematically equivalent to the DOU expression in
          Supplementary Table 1 of the manuscript:
              (2C + N - H - X + 2) / 2
    C/N : ratio of carbon to nitrogen atoms; 0 if the molecule has no N.
    """
    counts = atom_counts_from_formula(formula)
    dbe = counts["C"] + 1 - (counts["H"] + counts["Cl"]) / 2 + counts["N"] / 2
    c_n = counts["C"] / counts["N"] if counts["N"] != 0 else 0.0
    return {"DBE": dbe, "C/N": c_n}


def rdkit_descriptors(smiles: str) -> Optional[dict]:
    """Compute eight RDKit-derived descriptors from a SMILES string.

    Returns ``None`` if RDKit fails to parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    AllChem.ComputeGasteigerCharges(mol)
    charges = [
        float(atom.GetProp("_GasteigerCharge"))
        for atom in mol.GetAtoms()
        if atom.HasProp("_GasteigerCharge")
    ]
    max_abs_charge = max(abs(c) for c in charges if np.isfinite(c)) if charges else 0.0

    return {
        "TPSA": Descriptors.TPSA(mol),
        "MaxAbsCharge": max_abs_charge,
        "MolLogP": Crippen.MolLogP(mol),
        "MolWt": Descriptors.MolWt(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
    }


def build_descriptor_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 10 descriptors for every row of the input dataframe."""
    records = []
    for _, row in df.iterrows():
        rdkit_feats = rdkit_descriptors(row["smiles"])
        if rdkit_feats is None:
            raise ValueError(f"RDKit could not parse SMILES: {row['smiles']}")
        records.append(
            {
                "number": row["number"],
                "name": row["name"],
                "abbreviation": row["abbreviation"],
                "smiles": row["smiles"],
                "E_ox_V_vs_Li": row["E_ox_V_vs_Li"],
                **elemental_descriptors(row["molecular_formula"]),
                **rdkit_feats,
            }
        )
    return pd.DataFrame.from_records(records)


# -----------------------------------------------------------------------------
# Model training and evaluation
# -----------------------------------------------------------------------------

def loocv_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Generate leave-one-out predictions for every sample in ``y``."""
    predictions = np.empty_like(y, dtype=float)
    for train_idx, test_idx in LeaveOneOut().split(X):
        model = RandomForestRegressor(**RF_KWARGS)
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])
    return predictions


def spearman_baseline(df: pd.DataFrame, target: str = "E_ox_V_vs_Li") -> pd.DataFrame:
    """Compute Spearman rank correlation of each descriptor with the target."""
    rows = []
    for feature in FEATURE_COLUMNS:
        rho, p_value = spearmanr(df[feature], df[target])
        rows.append({"feature": feature, "spearman_rho": rho, "p_value": p_value})
    return (
        pd.DataFrame(rows)
        .assign(abs_rho=lambda d: d["spearman_rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .drop(columns="abs_rho")
        .reset_index(drop=True)
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main(input_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_csv)
    descriptors = build_descriptor_table(raw)
    descriptors.to_csv(out_dir / "descriptors.csv", index=False)

    X = descriptors[FEATURE_COLUMNS].to_numpy()
    y = descriptors["E_ox_V_vs_Li"].to_numpy()

    # ---------- LOOCV evaluation ----------
    y_pred = loocv_predict(X, y)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"LOOCV  R^2 = {r2:.3f}   MAE = {mae:.3f} V   (n = {len(y)})")

    pd.DataFrame(
        {
            "number": descriptors["number"],
            "abbreviation": descriptors["abbreviation"],
            "E_ox_true": y,
            "E_ox_pred": y_pred,
            "abs_error": np.abs(y - y_pred),
        }
    ).to_csv(out_dir / "loocv_predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump({"R2": float(r2), "MAE": float(mae), "n": int(len(y))}, fh, indent=2)

    # ---------- MDI feature importance (full-data fit) ----------
    full_model = RandomForestRegressor(**RF_KWARGS)
    full_model.fit(X, y)
    importance_df = (
        pd.DataFrame(
            {"feature": FEATURE_COLUMNS, "importance_MDI": full_model.feature_importances_}
        )
        .sort_values("importance_MDI", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
    print("\nMDI feature importance:")
    print(importance_df.to_string(index=False))

    # ---------- Spearman rank correlation baseline ----------
    spearman_df = spearman_baseline(descriptors)
    spearman_df.to_csv(out_dir / "spearman_correlation.csv", index=False)
    print("\nSpearman rank correlation with E_ox:")
    print(spearman_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/redox_mediators.csv"),
        help="Input CSV with SMILES, formula, and oxidation potentials.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for descriptors, predictions, and metrics.",
    )
    args = parser.parse_args()
    main(args.input, args.out_dir)
