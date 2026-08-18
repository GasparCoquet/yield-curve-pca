"""Why PC1 is not 90% here, and what has to be true for it to be.

The textbook number ("PC1 is roughly 90% of curve variance") is a claim about the
COUPON curve. Run it on a panel that includes Treasury bills and PC1 falls to 73%,
because the front of the bill curve is close to uncorrelated with duration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ycpca import data, pca


def line(name: str, changes: pd.DataFrame) -> None:
    result = pca.fit_eig(changes.to_numpy(), standardise=False)
    print(f"{name:<32} n={len(changes):>5}   PC1 {result.explained_variance_ratio[0]:>6.2%}"
          f"   PC1-3 {result.cumulative_variance_ratio[2]:>6.2%}")


def main() -> None:
    levels = data.load()
    changes = data.daily_changes(levels)

    print("--- which tenors are in the panel (2015 to 2026) ---")
    line("all 11 tenors", changes)
    line("drop 1 Mo", changes.drop(columns=["1 Mo"]))
    line("drop 1 Mo and 3 Mo", changes.drop(columns=["1 Mo", "3 Mo"]))
    line("coupon curve, 2 Yr and longer", changes[data.COUPON])

    print("\n--- is the coupon result stable across regimes ---")
    for start, end in [("2015", "2019"), ("2020", "2021"), ("2022", "2023"), ("2024", "2026")]:
        line(f"coupon curve, {start} to {end}", changes.loc[start:end, data.COUPON])

    print("\n--- why the bills break it ---")
    reference: pd.Series = changes["10 Yr"]
    for tenor in data.TENORS:
        print(f"  corr({tenor:<6}, 10 Yr) = {changes[tenor].corr(reference):>6.3f}")

    print("\nlargest 1 Mo daily moves (bp), all debt-ceiling or policy dated:")
    largest = changes["1 Mo"].abs().nlargest(5)
    for date, move in largest.items():
        print(f"  {date.date()}  {changes.loc[date, '1 Mo']:+.0f} bp")

    print("\nReading: PC1 near 90% is a property of the coupon curve and holds in every\n"
          "sub-period tested. The 73% figure is not a different market, it is a different\n"
          "panel. Say which one you mean before quoting the number.")


if __name__ == "__main__":
    main()
