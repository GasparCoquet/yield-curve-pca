"""Main analysis: what the components are, how much they explain, and what they hedge."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ycpca import curve, data, pca

FIGURES: Path = Path(__file__).resolve().parent.parent / "figures"
CURVE: list[str] = data.COUPON
BOOK: dict[str, float] = {"2 Yr": 50_000.0, "5 Yr": -30_000.0, "10 Yr": 40_000.0, "30 Yr": -20_000.0}


def variance_table(result: pca.PCAResult, tenors: list[str]) -> None:
    print("component   sd (bp/day)   share     cumulative   shape")
    print("-" * 62)
    for i in range(min(5, len(result.eigenvalues))):
        shape: str = curve.label(result.loadings[:, i]) if i < 3 else ""
        print(f"PC{i + 1:<9} {np.sqrt(result.eigenvalues[i]):>10.3f}   "
              f"{result.explained_variance_ratio[i]:>6.2%}    "
              f"{result.cumulative_variance_ratio[i]:>7.2%}      {shape}")


def loadings_table(result: pca.PCAResult, tenors: list[str]) -> None:
    print(f"\n{'tenor':<8}" + "".join(f"{f'PC{i + 1}':>9}" for i in range(3)))
    print("-" * 35)
    for row, tenor in enumerate(tenors):
        print(f"{tenor:<8}" + "".join(f"{result.loadings[row, i]:>9.3f}" for i in range(3)))


def hedge_demo(changes: np.ndarray, result: pca.PCAResult, tenors: list[str]) -> None:
    dv01: np.ndarray = np.array([BOOK.get(t, 0.0) for t in tenors])
    covariance: np.ndarray = np.cov(changes, rowvar=False)

    three: pca.PCAResult = pca.fit_eig(changes, standardise=False, n_components=3)
    three = curve.orient_economically(three)

    unhedged: float = curve.pnl_variance(dv01, covariance)
    exposures: np.ndarray = curve.pc_exposures(dv01, three.loadings)

    print("\nbook DV01 ($/bp): " + ", ".join(f"{t} {v:+,.0f}" for t, v in BOOK.items()))
    print(f"daily P&L sd unhedged            : ${np.sqrt(unhedged):>10,.0f}")
    print("factor exposures ($ per 1 sd move):")
    for i, name in enumerate(["level", "slope", "curvature"]):
        print(f"  {name:<10} {exposures[i] * np.sqrt(three.eigenvalues[i]):>12,.0f}")

    # Naive alternative: one instrument, total DV01 to zero, ignores curve shape.
    naive: np.ndarray = dv01.copy()
    naive[tenors.index("10 Yr")] -= dv01.sum()
    naive_var: float = curve.pnl_variance(naive, covariance)

    hedge_indices: list[int] = [tenors.index(t) for t in ("2 Yr", "10 Yr", "30 Yr")]
    hedge_dv01: np.ndarray = curve.solve_three_factor_hedge(dv01, three.loadings, hedge_indices)
    hedged: np.ndarray = curve.apply_hedge(dv01, hedge_dv01, hedge_indices)
    hedged_var: float = curve.pnl_variance(hedged, covariance)

    print(f"\ndaily P&L sd, DV01-neutral only  : ${np.sqrt(naive_var):>10,.0f}"
          f"   ({1 - naive_var / unhedged:.1%} of variance removed)")
    print("three-factor hedge DV01 ($/bp)   : " + ", ".join(
        f"{tenors[j]} {hedge_dv01[i]:+,.0f}" for i, j in enumerate(hedge_indices)))
    print(f"daily P&L sd, 3-factor hedged    : ${np.sqrt(hedged_var):>10,.0f}"
          f"   ({1 - hedged_var / unhedged:.1%} of variance removed)")
    print(f"residual is PC4 and beyond, which carry "
          f"{1 - three.eigenvalues.sum() / np.cov(changes, rowvar=False).trace():.2%} of curve variance")


def standardise_comparison(tenors: list[str]) -> None:
    """Standardising only matters when the columns have genuinely different scales.

    On the coupon curve every tenor moves 5 to 6 bp a day, so covariance and correlation
    PCA agree. Add the bills, whose vols differ by a factor of two, and they diverge.
    Quoting "always standardise" as a rule misses that it is a statement about units.
    """
    full = data.daily_changes(data.load())
    for name, panel in [("coupon curve only", full[tenors]), ("with bills included", full)]:
        raw_sd: np.ndarray = panel.std(axis=0, ddof=1).to_numpy()
        print()
        print(f"  {name}: daily sd {raw_sd.min():.1f} to {raw_sd.max():.1f} bp "
              f"(ratio {raw_sd.max() / raw_sd.min():.1f}x)")
        for standardise in (False, True):
            result = curve.orient_economically(pca.fit_eig(panel.to_numpy(), standardise=standardise))
            tag: str = "correlation" if standardise else "covariance "
            print(f"    {tag} PCA: PC1 {result.explained_variance_ratio[0]:>6.2%}, "
                  f"PC1+2+3 {result.cumulative_variance_ratio[2]:>6.2%}")

def make_figures(levels, changes: np.ndarray, result: pca.PCAResult, tenors: list[str]) -> None:
    years: list[float] = [data.TENOR_YEARS[t] for t in tenors]
    FIGURES.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, name in enumerate(["PC1 level", "PC2 slope", "PC3 curvature"]):
        axes[0].plot(years, result.loadings[:, i], marker="o", label=name)
    axes[0].axhline(0.0, color="grey", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xticks(years)
    axes[0].set_xticklabels(tenors, rotation=45, fontsize=7)
    axes[0].set_title("Loadings: the shape of each factor")
    axes[0].set_ylabel("loading")
    axes[0].legend(fontsize=8)

    share: np.ndarray = result.explained_variance_ratio
    axes[1].bar(range(1, len(share) + 1), share * 100, color="steelblue")
    axes[1].plot(range(1, len(share) + 1), result.cumulative_variance_ratio * 100,
                 color="firebrick", marker="o", lw=1.2, label="cumulative")
    axes[1].set_title("Variance explained")
    axes[1].set_xlabel("component")
    axes[1].set_ylabel("percent")
    axes[1].legend(fontsize=8)

    # Sanity check that PC1 IS the level, not just that it looks flat.
    # Two traps, both of which bit on the first attempt:
    #   scores come from CENTRED changes, so cumulating them drops the drift and the
    #     series walks away by a linear trend. Project the raw changes instead.
    #   the loading vector is unit norm, not averaging, so the projection is in units
    #     of sum(loadings) times an average move. Divide it back out to compare.
    raw_changes = data.daily_changes(levels).to_numpy()
    pc1_raw = (raw_changes @ result.loadings[:, 0]).cumsum() / result.loadings[:, 0].sum()
    actual = (levels.mean(axis=1) - levels.mean(axis=1).iloc[0]).to_numpy()[1:] * 100.0
    correlation = float(np.corrcoef(pc1_raw, actual)[0, 1])
    axes[2].plot(levels.index[1:], pc1_raw, lw=1.1, label="PC1 cumulated, rescaled")
    axes[2].plot(levels.index[1:], actual, lw=1.1, ls="--", color="black",
                 label="mean curve level, actual")
    axes[2].axhline(0.0, color="grey", lw=0.8)
    axes[2].set_title(f"PC1 is the level (corr {correlation:.4f})")
    axes[2].set_ylabel("bp from start")
    axes[2].legend(fontsize=8)
    axes[2].tick_params(axis="x", labelrotation=30, labelsize=7)
    print(f"PC1 cumulated vs actual mean curve level: correlation {correlation:.4f}, "
          f"max gap {np.abs(pc1_raw - actual).max():.1f} bp")

    fig.suptitle("US Treasury curve, daily changes, covariance PCA", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "components.png", dpi=140)
    plt.close(fig)
    print(f"\nfigure written: {FIGURES / 'components.png'}")


def main() -> None:
    levels = data.load()[CURVE]
    changes_df = data.daily_changes(levels)
    changes: np.ndarray = changes_df.to_numpy()
    tenors: list[str] = list(levels.columns)

    print(f"US Treasury par curve, {levels.index.min().date()} to {levels.index.max().date()}")
    print(f"{changes.shape[0]} daily changes in bp, {changes.shape[1]} tenors\n")

    result = curve.orient_economically(pca.fit_eig(changes, standardise=False))
    variance_table(result, tenors)
    loadings_table(result, tenors)
    print("\n--- standardise or not ---")
    standardise_comparison(tenors)
    print("\n--- three-factor hedge ---")
    hedge_demo(changes, result, tenors)
    make_figures(levels, changes, result, tenors)


if __name__ == "__main__":
    main()
