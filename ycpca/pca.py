"""PCA from scratch.

Two independent routes to the same answer:
  1. eigendecomposition of the sample covariance (or correlation) matrix
  2. singular value decomposition of the centred data matrix

Route 2 is what production code uses (never forms the covariance matrix, so it is
better conditioned), but route 1 is the definition. `validate.py` checks they agree.

No sklearn here on purpose: the decomposition is the part worth writing yourself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCAResult:
    """Everything a fitted PCA produces, in the orientation used throughout."""

    eigenvalues: np.ndarray   # (k,) variance carried by each component, descending
    loadings: np.ndarray      # (n_vars, k) columns are unit-norm eigenvectors
    scores: np.ndarray        # (n_obs, k) the principal components themselves
    mean: np.ndarray          # (n_vars,) column means removed before fitting
    scale: np.ndarray         # (n_vars,) column scales divided out (ones if not standardised)
    standardised: bool

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum()

    @property
    def cumulative_variance_ratio(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    def reconstruct(self, n_components: int | None = None) -> np.ndarray:
        """Rebuild the original data from the first k components."""
        k: int = self.loadings.shape[1] if n_components is None else n_components
        approx: np.ndarray = self.scores[:, :k] @ self.loadings[:, :k].T
        return approx * self.scale + self.mean


def _preprocess(
    data: np.ndarray, standardise: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre always, scale only if asked. Returns the transformed matrix, mean, scale."""
    mean: np.ndarray = data.mean(axis=0)
    centred: np.ndarray = data - mean
    if standardise:
        scale: np.ndarray = data.std(axis=0, ddof=1)
        if np.any(scale == 0.0):
            raise ValueError("cannot standardise a column with zero variance")
    else:
        scale = np.ones(data.shape[1], dtype=float)
    return centred / scale, mean, scale


def _fix_signs(loadings: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvectors are sign-ambiguous. Pin each so its largest entry is positive.

    Without this the sign flips arbitrarily between runs and between windows, which
    looks like instability but is not.
    """
    dominant: np.ndarray = np.argmax(np.abs(loadings), axis=0)
    signs: np.ndarray = np.sign(loadings[dominant, np.arange(loadings.shape[1])])
    signs[signs == 0.0] = 1.0
    return loadings * signs, scores * signs


def fit_eig(data: np.ndarray, standardise: bool = False, n_components: int | None = None) -> PCAResult:
    """PCA by eigendecomposition of the sample covariance matrix. The definition."""
    matrix: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    matrix, mean, scale = _preprocess(np.asarray(data, dtype=float), standardise)
    n_obs: int = matrix.shape[0]

    # If standardise=True this is the correlation matrix of the raw data.
    covariance: np.ndarray = (matrix.T @ matrix) / (n_obs - 1)

    # eigh, not eig: the covariance matrix is symmetric, so eigenvalues are real
    # and the eigenvectors come back orthonormal. It returns them ascending.
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order: np.ndarray = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Tiny negatives are floating point noise on a PSD matrix, not real.
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    scores: np.ndarray = matrix @ eigenvectors
    eigenvectors, scores = _fix_signs(eigenvectors, scores)

    k: int = eigenvectors.shape[1] if n_components is None else n_components
    return PCAResult(
        eigenvalues=eigenvalues[:k], loadings=eigenvectors[:, :k], scores=scores[:, :k],
        mean=mean, scale=scale, standardised=standardise,
    )


def fit_svd(data: np.ndarray, standardise: bool = False, n_components: int | None = None) -> PCAResult:
    """PCA by SVD of the centred data matrix. What production code actually does."""
    matrix: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    matrix, mean, scale = _preprocess(np.asarray(data, dtype=float), standardise)
    n_obs: int = matrix.shape[0]

    # X = U S V'. Columns of V are the eigenvectors of X'X, so of the covariance,
    # and the singular values are the square roots of its eigenvalues scaled by n-1.
    left: np.ndarray
    singular: np.ndarray
    right_t: np.ndarray
    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)

    eigenvalues: np.ndarray = singular**2 / (n_obs - 1)
    loadings: np.ndarray = right_t.T
    scores: np.ndarray = left * singular
    loadings, scores = _fix_signs(loadings, scores)

    k: int = loadings.shape[1] if n_components is None else n_components
    return PCAResult(
        eigenvalues=eigenvalues[:k], loadings=loadings[:, :k], scores=scores[:, :k],
        mean=mean, scale=scale, standardised=standardise,
    )


def align_signs(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Flip columns of `target` to match the sign of `reference`, for comparison only."""
    signs: np.ndarray = np.sign((reference * target).sum(axis=0))
    signs[signs == 0.0] = 1.0
    return target * signs


def subspace_angle(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Largest principal angle between two subspaces, in degrees.

    Sign-blind and rotation-blind, so it measures whether the same *space* is spanned
    rather than whether the individual vectors happen to line up.
    """
    q_a: np.ndarray = np.linalg.qr(basis_a)[0]
    q_b: np.ndarray = np.linalg.qr(basis_b)[0]
    singular: np.ndarray = np.linalg.svd(q_a.T @ q_b, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(singular.min(), -1.0, 1.0))))
