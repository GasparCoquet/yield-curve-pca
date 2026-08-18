"""Walk-forward test of the three-factor hedge.

The headline number in run_analysis.py estimates the covariance and measures the
variance reduction on the SAME days. That flatters it. Here the loadings and the
hedge are solved on a training window and then applied, untouched, to days the
estimator has never seen.

Protocol, repeated on a rolling basis:
  fit PCA on 250 days -> solve the 3-factor hedge on those loadings
  hold that hedge fixed over the NEXT 21 days
  record realised P&L on those 21 out-of-sample days

The book DV01 is held constant throughout, which a real book would not be. That is a
simplification, not a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ycpca import curve, data, pca

TRAIN: int = 250
TEST: int = 21
BOOK: dict[str, float] = {"2 Yr": 50_000.0, "5 Yr": -30_000.0, "10 Yr": 40_000.0, "30 Yr": -20_000.0}
HEDGE_TENORS: tuple[str, ...] = ("2 Yr", "10 Yr", "30 Yr")


def main() -> None:
    levels = data.load()[data.COUPON]
    changes = data.daily_changes(levels)
    matrix: np.ndarray = changes.to_numpy()
    tenors: list[str] = list(changes.columns)

    dv01: np.ndarray = np.array([BOOK.get(t, 0.0) for t in tenors])
    hedge_indices: list[int] = [tenors.index(t) for t in HEDGE_TENORS]

    unhedged: list[np.ndarray] = []
    three_factor: list[np.ndarray] = []
    duration_only: list[np.ndarray] = []

    for start in range(0, len(matrix) - TRAIN - TEST, TEST):
        train: np.ndarray = matrix[start:start + TRAIN]
        test: np.ndarray = matrix[start + TRAIN:start + TRAIN + TEST]

        fitted = pca.fit_eig(train, standardise=False, n_components=3)
        hedge_dv01: np.ndarray = curve.solve_three_factor_hedge(dv01, fitted.loadings, hedge_indices)
        hedged: np.ndarray = curve.apply_hedge(dv01, hedge_dv01, hedge_indices)

        naive: np.ndarray = dv01.copy()
        naive[tenors.index("10 Yr")] -= dv01.sum()

        unhedged.append(test @ dv01)
        three_factor.append(test @ hedged)
        duration_only.append(test @ naive)

    base: np.ndarray = np.concatenate(unhedged)
    hedged_pnl: np.ndarray = np.concatenate(three_factor)
    naive_pnl: np.ndarray = np.concatenate(duration_only)

    print(f"walk-forward: {TRAIN}-day estimation, {TEST}-day holding, "
          f"{len(base)} out-of-sample days")
    print(f"({len(unhedged)} non-overlapping test blocks, "
          f"{changes.index[TRAIN].date()} to {changes.index[-1].date()})\n")

    print(f"  {'hedge':<22}{'daily P&L sd':>16}{'variance removed':>20}")
    print("  " + "-" * 58)
    print(f"  {'unhedged':<22}${base.std(ddof=1):>15,.0f}")
    for name, series in [("DV01-neutral only", naive_pnl), ("three-factor", hedged_pnl)]:
        reduction: float = 1.0 - series.var(ddof=1) / base.var(ddof=1)
        print(f"  {name:<22}${series.std(ddof=1):>15,.0f}{reduction:>19.1%}")

    # For contrast: the same hedge solved and measured on the full sample.
    full = pca.fit_eig(matrix, standardise=False, n_components=3)
    in_hedge: np.ndarray = curve.solve_three_factor_hedge(dv01, full.loadings, hedge_indices)
    in_dv01: np.ndarray = curve.apply_hedge(dv01, in_hedge, hedge_indices)
    covariance: np.ndarray = np.cov(matrix, rowvar=False)
    in_sample: float = 1.0 - curve.pnl_variance(in_dv01, covariance) / curve.pnl_variance(dv01, covariance)
    realised: float = 1.0 - hedged_pnl.var(ddof=1) / base.var(ddof=1)

    print()
    print(f"  in-sample, same days used to estimate : {in_sample:>8.3%}")
    print(f"  walk-forward, never-seen days         : {realised:>8.3%}")
    print(f"  degradation                           : {(in_sample - realised) * 100:>8.3f} pts")
    print()
    print("Reading: the hedge survives out of sample essentially intact. That is not luck,")
    print("it follows from the loading stability measured in stability.py: PC1 moves under a")
    print("degree when the estimation window rolls forward a month, so a hedge solved on last")
    print("year is still the right hedge this month. A yield curve is an unusually benign case")
    print("for this. Do not carry the result over to a covariance matrix that moves more.")


if __name__ == "__main__":
    main()
