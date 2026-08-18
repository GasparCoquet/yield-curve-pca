"""Prove the from-scratch PCA is correct.

Five independent checks. Any failure raises, so this script is the gate that stops a
wrong implementation from being believed:
  1. eigendecomposition and SVD routes agree
  2. loadings are orthonormal
  3. eigenvalues reproduce the covariance matrix (spectral decomposition)
  4. total variance is conserved, and full reconstruction is exact
  5. results match sklearn, which was never used to compute them
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ycpca import data, pca

TOL: float = 1e-8


def report(name: str, error: float, tol: float = TOL) -> bool:
    ok: bool = error < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} max abs error {error:.3e}")
    return ok


def main() -> None:
    levels = data.load()
    changes = data.daily_changes(levels).to_numpy()
    print(f"data: {changes.shape[0]} daily changes, {changes.shape[1]} tenors\n")

    passed: list[bool] = []

    for standardise in (False, True):
        tag: str = "correlation" if standardise else "covariance"
        print(f"--- {tag} PCA ---")
        by_eig = pca.fit_eig(changes, standardise=standardise)
        by_svd = pca.fit_svd(changes, standardise=standardise)

        # 1. the two routes are the same decomposition
        passed.append(report("eig vs svd: eigenvalues",
                             np.abs(by_eig.eigenvalues - by_svd.eigenvalues).max()))
        aligned = pca.align_signs(by_eig.loadings, by_svd.loadings)
        passed.append(report("eig vs svd: loadings", np.abs(by_eig.loadings - aligned).max(), 1e-6))

        # 2. eigenvectors of a symmetric matrix are orthonormal
        gram = by_eig.loadings.T @ by_eig.loadings
        passed.append(report("loadings orthonormal (L'L = I)",
                             np.abs(gram - np.eye(gram.shape[0])).max()))

        # 3. the decomposition rebuilds the matrix it came from
        matrix = (changes - changes.mean(axis=0)) / by_eig.scale
        covariance = (matrix.T @ matrix) / (matrix.shape[0] - 1)
        rebuilt = by_eig.loadings @ np.diag(by_eig.eigenvalues) @ by_eig.loadings.T
        passed.append(report("spectral decomposition (S = L D L')",
                             np.abs(covariance - rebuilt).max()))

        # 4. rotation moves variance around, it does not create or destroy it
        passed.append(report("total variance conserved",
                             abs(by_eig.eigenvalues.sum() - np.trace(covariance))))
        passed.append(report("full reconstruction is exact",
                             np.abs(by_eig.reconstruct() - changes).max(), 1e-6))

        # 5. independent reference. sklearn appears nowhere in ycpca/.
        from sklearn.decomposition import PCA as SklearnPCA

        reference = SklearnPCA().fit(matrix)
        passed.append(report("eigenvalues match sklearn",
                             np.abs(by_eig.eigenvalues - reference.explained_variance_).max(), 1e-6))
        ref_loadings = pca.align_signs(by_eig.loadings, reference.components_.T)
        passed.append(report("loadings match sklearn",
                             np.abs(by_eig.loadings - ref_loadings).max(), 1e-6))
        print()

    print(f"{sum(passed)}/{len(passed)} checks passed")
    if not all(passed):
        raise SystemExit("validation FAILED")


if __name__ == "__main__":
    main()
