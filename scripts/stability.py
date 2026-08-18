"""What happens to the loadings when you add data, and why it matters for trading them.

Three different questions get three different answers, and conflating them is how
people end up saying "PCA loadings are unstable" without meaning anything:

  1. Extend an 11-year sample by a month. Nothing moves. A month is noise on 2,900 days.
  2. Roll a 250-day window forward by a month, which is what a desk actually re-fits.
     PC1 moves under a degree, PC3 moves several, occasionally 20 plus.
  3. Compare a 250-day window to the full sample. PC1 6 deg, PC3 14 deg median.

So: PC1 is stable enough to trade. PC3 is not a stable object at one-year estimation
length, and the ordering PC1 < PC2 < PC3 is monotone in every measure below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ycpca import data, pca

WINDOW: int = 250   # about one trading year
STEP: int = 21      # about one trading month
NAMES: list[str] = ["PC1 level", "PC2 slope", "PC3 curvature"]


def vector_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two directions in degrees, ignoring sign."""
    cosine: float = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def main() -> None:
    levels = data.load()[data.COUPON]
    changes = data.daily_changes(levels)
    matrix: np.ndarray = changes.to_numpy()
    full = pca.fit_eig(matrix, standardise=False, n_components=3)

    print("--- 1. extend the full sample by one month ---")
    before = pca.fit_eig(matrix[:-STEP], standardise=False, n_components=3)
    for i, name in enumerate(NAMES):
        print(f"  {name:<16} moves {vector_angle(before.loadings[:, i], full.loadings[:, i]):>5.2f} deg")
    print("  Nothing moves. 21 new days against 2,885 old ones cannot move the covariance.")

    print(f"\n--- 2. roll a {WINDOW}-day window forward by one month (what a desk re-fits) ---")
    rolled: list[list[float]] = []
    for start in range(0, len(matrix) - WINDOW - STEP, STEP):
        old = pca.fit_eig(matrix[start:start + WINDOW], standardise=False, n_components=3)
        new = pca.fit_eig(matrix[start + STEP:start + WINDOW + STEP], standardise=False, n_components=3)
        rolled.append([vector_angle(old.loadings[:, i], new.loadings[:, i]) for i in range(3)]
                      + [pca.subspace_angle(old.loadings, new.loadings)])
    roll = pd.DataFrame(rolled, columns=["PC1", "PC2", "PC3", "subspace"])
    print(f"  {'':<12}{'median':>9}{'90th':>8}{'worst':>8}   (degrees)")
    for column in roll.columns:
        print(f"  {column:<12}{roll[column].median():>9.2f}{roll[column].quantile(0.9):>8.2f}"
              f"{roll[column].max():>8.2f}")

    print(f"\n--- 3. each {WINDOW}-day window against the full sample ---")
    rows: list[dict[str, float]] = []
    for start in range(0, len(matrix) - WINDOW, STEP):
        result = pca.fit_eig(matrix[start:start + WINDOW], standardise=False, n_components=3)
        rows.append({
            "date": changes.index[start + WINDOW - 1],
            "pc1_share": result.explained_variance_ratio[0],
            "PC1": vector_angle(result.loadings[:, 0], full.loadings[:, 0]),
            "PC2": vector_angle(result.loadings[:, 1], full.loadings[:, 1]),
            "PC3": vector_angle(result.loadings[:, 2], full.loadings[:, 2]),
            "subspace": pca.subspace_angle(result.loadings, full.loadings),
            "eig_gap_23": result.eigenvalues[1] / result.eigenvalues[2],
        })
    table = pd.DataFrame(rows).set_index("date")
    print(f"  {len(table)} windows, {table.index.min().date()} to {table.index.max().date()}")
    print(f"  {'':<12}{'median':>9}{'90th':>8}{'worst':>8}   (degrees)")
    for column in ["PC1", "PC2", "PC3", "subspace"]:
        print(f"  {column:<12}{table[column].median():>9.2f}{table[column].quantile(0.9):>8.2f}"
              f"{table[column].max():>8.2f}")
    print(f"  PC1 share: min {table['pc1_share'].min():.1%}, median {table['pc1_share'].median():.1%}, "
          f"max {table['pc1_share'].max():.1%}")

    print("\n--- does eigenvalue separation explain it ---")
    gap: pd.Series = table["eig_gap_23"]
    close = table[gap < gap.median()]
    apart = table[gap >= gap.median()]
    print(f"  lambda2/lambda3 ranges {gap.min():.1f} to {gap.max():.1f}, median {gap.median():.1f}")
    print(f"  narrower half (gap < {gap.median():.1f}): median PC2 angle {close['PC2'].median():.2f} deg")
    print(f"  wider half    (gap >= {gap.median():.1f}): median PC2 angle {apart['PC2'].median():.2f} deg")
    print(f"  correlation(gap, PC2 angle) = {gap.corr(table['PC2']):+.3f}")
    print("\n  CAVEAT, stated because it limits the claim: on this curve lambda2/lambda3\n"
          "  never falls below 4, so the near-degenerate case that leaves two eigenvectors\n"
          "  genuinely unidentified does NOT occur in this sample. What is shown here is a\n"
          "  tendency in the right direction, not a demonstration of eigenvalue crossing.\n"
          "  The theory says a small gap destroys identification. This data cannot confirm it.")


if __name__ == "__main__":
    main()
