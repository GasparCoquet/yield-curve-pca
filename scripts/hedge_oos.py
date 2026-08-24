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
    min_variance: list[np.ndarray] = []

    selector: np.ndarray = np.zeros((len(tenors), len(hedge_indices)))
    for column, index in enumerate(hedge_indices):
        selector[index, column] = 1.0

    for start in range(0, len(matrix) - TRAIN - TEST, TEST):
        train: np.ndarray = matrix[start:start + TRAIN]
        test: np.ndarray = matrix[start + TRAIN:start + TRAIN + TEST]

        fitted = pca.fit_eig(train, standardise=False, n_components=3)
        hedge_dv01: np.ndarray = curve.solve_three_factor_hedge(dv01, fitted.loadings, hedge_indices)
        hedged: np.ndarray = curve.apply_hedge(dv01, hedge_dv01, hedge_indices)

        naive: np.ndarray = dv01.copy()
        naive[tenors.index("10 Yr")] -= dv01.sum()

        # The null model this repo has to beat, and the first thing an
        # interviewer asks about a PCA hedge. Minimise h'(P'SP)h + 2h'P'S*dv01
        # directly on the training covariance, no eigendecomposition anywhere:
        #     h = -(P'SP)^-1 P'S dv01
        train_cov: np.ndarray = np.cov(train, rowvar=False)
        mv_h: np.ndarray = -np.linalg.solve(
            selector.T @ train_cov @ selector, selector.T @ train_cov @ dv01)
        mv: np.ndarray = dv01 + selector @ mv_h

        unhedged.append(test @ dv01)
        three_factor.append(test @ hedged)
        duration_only.append(test @ naive)
        min_variance.append(test @ mv)

    base: np.ndarray = np.concatenate(unhedged)
    hedged_pnl: np.ndarray = np.concatenate(three_factor)
    naive_pnl: np.ndarray = np.concatenate(duration_only)
    mv_pnl: np.ndarray = np.concatenate(min_variance)

    print(f"walk-forward: {TRAIN}-day estimation, {TEST}-day holding, "
          f"{len(base)} out-of-sample days")
    print(f"({len(unhedged)} non-overlapping test blocks, "
          f"{changes.index[TRAIN].date()} to {changes.index[-1].date()})\n")

    print(f"  {'hedge':<22}{'daily P&L sd':>16}{'variance removed':>20}")
    print("  " + "-" * 58)
    print(f"  {'unhedged':<22}${base.std(ddof=1):>15,.0f}")
    for name, series in [("DV01-neutral only", naive_pnl), ("three-factor PCA", hedged_pnl),
                         ("min-variance (no PCA)", mv_pnl)]:
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
    mv_reduction: float = 1.0 - mv_pnl.var(ddof=1) / base.var(ddof=1)
    print(f"  min-variance, same protocol           : {mv_reduction:>8.3%}")
    print(f"  PCA minus min-variance                : {(realised - mv_reduction) * 100:>8.3f} pts")
    print()
    print("Reading: the hedge survives out of sample essentially intact. That is not luck,")
    print("it follows from the loading stability measured in stability.py: PC1 moves under a")
    print("degree when the estimation window rolls forward a month, so a hedge solved on last")
    print("year is still the right hedge this month. A yield curve is an unusually benign case")
    print("for this. Do not carry the result over to a covariance matrix that moves more.")
    print()
    print("And the null model wins on this metric. Solving the same problem straight off the")
    print("sample covariance, with no eigendecomposition at all, removes more variance than")
    print("the three-factor hedge does. That is not a bug in the PCA hedge, it is what the")
    print("two are optimising: min-variance minimises exactly the quantity being scored here,")
    print("so on a stable covariance and a book inside the estimation panel it must win or")
    print("tie. Three reasons to still want the factor hedge, none of which this test scores:")
    print("  1. The factor hedge is expressed in level/slope/curvature exposures, which are")
    print("     the units a desk sets limits in. Min-variance returns three numbers with no")
    print("     interpretation and no way to attribute a P&L miss to a cause.")
    print("  2. It transfers. Loadings fitted here price an instrument that was never in the")
    print("     panel; an inverse sample covariance does not extend past its own columns.")
    print("  3. Min-variance inverts the sample covariance, which is where estimation error")
    print("     concentrates. It wins on 250 clean days of a benign curve. Shorten the window")
    print("     or widen the panel and the ranking is not safe. The honest version of that")
    print("     claim needs a second experiment this repo has not run.")


def random_books(seed: int = 7, draws: int = 6) -> None:
    """One book is an anecdote. Redraw the DV01 profile and check the ranking holds."""
    levels = data.load()[data.COUPON]
    changes = data.daily_changes(levels)
    matrix: np.ndarray = changes.to_numpy()
    tenors: list[str] = list(changes.columns)
    hedge_indices: list[int] = [tenors.index(t) for t in HEDGE_TENORS]

    selector: np.ndarray = np.zeros((len(tenors), len(hedge_indices)))
    for column, index in enumerate(hedge_indices):
        selector[index, column] = 1.0

    def score(dv01: np.ndarray) -> tuple[float, float]:
        base, pca_pnl, mv_pnl = [], [], []
        for start in range(0, len(matrix) - TRAIN - TEST, TEST):
            train, test = matrix[start:start + TRAIN], matrix[start + TRAIN:start + TRAIN + TEST]
            fitted = pca.fit_eig(train, standardise=False, n_components=3)
            hedge = curve.solve_three_factor_hedge(dv01, fitted.loadings, hedge_indices)
            train_cov = np.cov(train, rowvar=False)
            mv_h = -np.linalg.solve(selector.T @ train_cov @ selector, selector.T @ train_cov @ dv01)
            base.append(test @ dv01)
            pca_pnl.append(test @ curve.apply_hedge(dv01, hedge, hedge_indices))
            mv_pnl.append(test @ (dv01 + selector @ mv_h))
        variance = np.concatenate(base).var(ddof=1)
        return (1.0 - np.concatenate(pca_pnl).var(ddof=1) / variance,
                1.0 - np.concatenate(mv_pnl).var(ddof=1) / variance)

    rng = np.random.default_rng(seed)
    print()
    print(f"--- does the ranking survive a different book? (seed {seed}) ---")
    print(f"  {'book':<10}{'three-factor':>14}{'min-variance':>14}{'PCA - MV':>11}")
    losses = 0
    books: list[tuple[str, np.ndarray]] = [
        ("default", np.array([BOOK.get(t, 0.0) for t in tenors]))]
    for draw in range(draws):
        dv01 = np.zeros(len(tenors))
        for tenor in BOOK:
            dv01[tenors.index(tenor)] = rng.uniform(-60_000.0, 60_000.0)
        books.append((f"random {draw + 1}", dv01))
    for name, dv01 in books:
        pca_r, mv_r = score(dv01)
        losses += pca_r < mv_r
        print(f"  {name:<10}{pca_r:>13.3%}{mv_r:>14.3%}{(pca_r - mv_r) * 100:>10.3f}")
    print()
    print(f"  PCA loses on {losses} of {len(books)} books.")


if __name__ == "__main__":
    main()
    random_books()
