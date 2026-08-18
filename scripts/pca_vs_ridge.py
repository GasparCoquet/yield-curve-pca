"""PCA and ridge both attack collinearity. The difference is one line of arithmetic.

Write the regression in the SVD basis. In direction j with singular value d_j, the
coefficient the estimator puts on that direction is scaled by a FILTER FACTOR:

    OLS    f_j = 1                 every direction kept, including the noisy ones
    PCR    f_j = 1 if j <= k else 0     hard truncation, a step function
    ridge  f_j = d_j^2 / (d_j^2 + lam)  smooth shrinkage, never exactly zero

Same basis, same directions, different weighting. PCR makes a discrete decision about
how many directions exist. Ridge makes a continuous decision about how much to believe
each one. That is the whole answer, and everything else follows from it.

This is a DESCRIPTIVE illustration of collinearity on real curve data. It is not a
predictive model and there is no performance claim anywhere in this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ycpca import data

TARGET: str = "10 Yr"


def standardise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean: np.ndarray = matrix.mean(axis=0)
    scale: np.ndarray = matrix.std(axis=0, ddof=1)
    return (matrix - mean) / scale, mean, scale


def fit_all(design: np.ndarray, target: np.ndarray, k: int, lam: float) -> dict[str, np.ndarray]:
    """OLS, PCR and ridge from one SVD, so the only difference is the filter factor."""
    left, singular, right_t = np.linalg.svd(design, full_matrices=False)
    projected: np.ndarray = left.T @ target

    filters: dict[str, np.ndarray] = {
        "OLS": np.ones_like(singular),
        f"PCR k={k}": (np.arange(len(singular)) < k).astype(float),
        f"ridge lam={lam:g}": singular**2 / (singular**2 + lam),
    }
    return {
        name: right_t.T @ ((f / singular) * projected) for name, f in filters.items()
    } | {"_filters": filters, "_singular": singular}


def main() -> None:
    levels = data.load()[data.COUPON]
    changes = data.daily_changes(levels)
    predictors: list[str] = [t for t in data.COUPON if t != TARGET]

    design_raw: np.ndarray = changes[predictors].to_numpy()
    target_raw: np.ndarray = changes[TARGET].to_numpy()
    design, _, _ = standardise(design_raw)
    target: np.ndarray = target_raw - target_raw.mean()

    singular: np.ndarray = np.linalg.svd(design, compute_uv=False)
    print(f"regressing the {TARGET} daily change on the other {len(predictors)} coupon tenors")
    print(f"n = {len(design)}, condition number of the design = {singular.max() / singular.min():.1f}")
    print(f"singular values: {np.array2string(singular, precision=1)}")

    lam: float = 50.0
    k: int = 3
    fits = fit_all(design, target, k=k, lam=lam)

    print(f"\n--- filter factor applied to each SVD direction ---")
    print(f"  {'direction':<12}" + "".join(f"{f'd{i + 1}':>9}" for i in range(len(singular))))
    for name, values in fits["_filters"].items():
        print(f"  {name:<12}" + "".join(f"{v:>9.3f}" for v in values))
    print("  Ridge never reaches zero. PCR is only ever zero or one. That is the difference.")

    print(f"\n--- resulting coefficients (standardised predictors) ---")
    print(f"  {'tenor':<10}" + "".join(f"{n:>16}" for n in ["OLS", f"PCR k={k}", f"ridge lam={lam:g}"]))
    for row, tenor in enumerate(predictors):
        print(f"  {tenor:<10}" + "".join(
            f"{fits[n][row]:>16.3f}" for n in ["OLS", f"PCR k={k}", f"ridge lam={lam:g}"]))
    names: list[str] = ["OLS", f"PCR k={k}", f"ridge lam={lam:g}"]
    print(f"  {'sum |beta|':<10}" + "".join(f"{np.abs(fits[n]).sum():>16.3f}" for n in names))

    print(f"\n--- coefficient stability: refit on the first and second half ---")
    half: int = len(design) // 2
    for name in names:
        first = fit_all(design[:half], target[:half], k=k, lam=lam)[name]
        second = fit_all(design[half:], target[half:], k=k, lam=lam)[name]
        print(f"  {name:<16} max coefficient swing between halves {np.abs(first - second).max():>7.3f}")

    print("\nReading: with a condition number this size OLS is not wrong, it is just\n"
          "estimating directions the data barely contains. Ridge downweights those\n"
          "directions in proportion to how little variance they carry. PCR deletes them.\n"
          "Neither is 'more accurate' in general: PCR assumes the small directions are\n"
          "pure noise, ridge assumes they are weak signal.")


if __name__ == "__main__":
    main()
