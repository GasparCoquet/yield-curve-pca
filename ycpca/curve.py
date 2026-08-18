"""Curve-specific reading of the components, and the three-factor hedge."""
from __future__ import annotations

import numpy as np

from .pca import PCAResult


def count_sign_changes(loading: np.ndarray) -> int:
    """How many times a loading vector crosses zero as tenor increases."""
    signs: np.ndarray = np.sign(loading[np.abs(loading) > 1e-10])
    return int(np.sum(signs[1:] != signs[:-1]))


def label(loading: np.ndarray) -> str:
    """Name a component from the shape of its loading vector.

    0 crossings, every tenor moves together: level.
    1 crossing, short and long ends move opposite: slope.
    2 crossings, wings move against the belly: curvature.
    """
    crossings: int = count_sign_changes(loading)
    return {0: "level", 1: "slope", 2: "curvature"}.get(crossings, f"higher order ({crossings} crossings)")


def orient_economically(result: PCAResult) -> PCAResult:
    """Re-sign the first three components so they read the market way round.

    The maths does not care about sign, but a desk does: PC1 up must mean yields up,
    PC2 up must mean steeper, PC3 up must mean more humped.
    """
    loadings: np.ndarray = result.loadings.copy()
    scores: np.ndarray = result.scores.copy()
    n_components: int = loadings.shape[1]
    middle: int = loadings.shape[0] // 2

    targets: list[float] = []
    if n_components > 0:
        targets.append(loadings[:, 0].sum())                                    # level: yields up
    if n_components > 1:
        targets.append(loadings[-1, 1] - loadings[0, 1])                        # slope: long end up
    if n_components > 2:
        wings: float = 0.5 * (loadings[0, 2] + loadings[-1, 2])
        targets.append(wings - loadings[middle, 2])                             # curvature: wings up

    for index, target in enumerate(targets):
        if target < 0:
            loadings[:, index] *= -1.0
            scores[:, index] *= -1.0

    return PCAResult(
        eigenvalues=result.eigenvalues, loadings=loadings, scores=scores,
        mean=result.mean, scale=result.scale, standardised=result.standardised,
    )


def pc_exposures(dv01: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Book sensitivity per unit move of each component.

    dv01 is money per basis point at each tenor. Projecting it on the loadings turns
    N tenor exposures into k factor exposures.
    """
    return loadings.T @ dv01


def solve_three_factor_hedge(
    dv01: np.ndarray, loadings: np.ndarray, hedge_indices: list[int]
) -> np.ndarray:
    """DV01 to add at the hedge tenors so the first k factor exposures go to zero.

    Solves (L' P) h = -L' v, a k by k system. Needs exactly one hedge instrument per
    factor, and they must not be collinear in factor space.
    """
    n_factors: int = loadings.shape[1]
    if len(hedge_indices) != n_factors:
        raise ValueError(f"need exactly {n_factors} hedge instruments, got {len(hedge_indices)}")

    selector: np.ndarray = np.zeros((dv01.shape[0], n_factors))
    for column, tenor_index in enumerate(hedge_indices):
        selector[tenor_index, column] = 1.0

    system: np.ndarray = loadings.T @ selector
    if abs(np.linalg.det(system)) < 1e-12:
        raise ValueError("hedge instruments are collinear in factor space")

    return np.linalg.solve(system, -loadings.T @ dv01)


def apply_hedge(dv01: np.ndarray, hedge_dv01: np.ndarray, hedge_indices: list[int]) -> np.ndarray:
    """Book DV01 after adding the hedge legs."""
    hedged: np.ndarray = dv01.copy()
    for column, tenor_index in enumerate(hedge_indices):
        hedged[tenor_index] += hedge_dv01[column]
    return hedged


def pnl_variance(dv01: np.ndarray, covariance: np.ndarray) -> float:
    """Daily P&L variance of a book, in money squared. covariance is in bp squared."""
    return float(dv01 @ covariance @ dv01)
