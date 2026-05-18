"""
gaussian.py

Gaussian primitives for the Coulomb / plane-wave project.

Goal
----
This module defines Cartesian primitive Gaussian functions of the form

    G_abc(r; alpha, A)
    = N * (x - Ax)^a (y - Ay)^b (z - Az)^c
      exp(-alpha |r - A|^2)

where

    alpha > 0
    A = (Ax, Ay, Az)
    powers = (a, b, c)

The normalization factor N is optional. For debugging Coulomb integrals, it is
often useful to start with unnormalized primitives, because the algebra is more
transparent. Later, for physical matrix elements, normalized primitives are more
appropriate.

Why this module now?
--------------------
The plane-wave expansion now gives us terms of the form

    radial_factor(r) * x^a y^b z^c.

To test Coulomb-type integrals, we need another factor in the integrand. The
simplest orbital-like object is a Cartesian Gaussian primitive.

A first numerical Coulomb prototype will use integrals like

    integral d r  G(r) * [1 / |r - C|] * exp(i k.r)

and compare direct evaluation of exp(i k.r) against the solid-harmonic expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np


# =============================================================================
# Small math helpers
# =============================================================================

def double_factorial_odd(n: int) -> int:
    """
    Return n!! for odd n >= -1.

    For Gaussian normalization, we need values like

        (2a - 1)!!

    with the convention

        (-1)!! = 1
         0!!  = 1

    Examples
    --------
    >>> double_factorial_odd(-1)
    1
    >>> double_factorial_odd(1)
    1
    >>> double_factorial_odd(5)
    15
    """
    if n in (-1, 0):
        return 1
    if n < -1:
        raise ValueError("double factorial input must be >= -1")

    result = 1
    for k in range(n, 0, -2):
        result *= k
    return result


def gaussian_norm_cartesian(alpha: float, powers: tuple[int, int, int]) -> float:
    """
    Normalization constant for a Cartesian primitive Gaussian.

    The primitive is

        N x^a y^b z^c exp(-alpha r^2)

    centered at the origin. Translation of the center does not change the
    normalization.

    The normalization is chosen so that

        integral |G(r)|^2 d r = 1.

    Formula
    -------
    For powers (a,b,c), with L = a+b+c,

        N = [ (2 alpha / pi)^(3/2)
              * (4 alpha)^L
              / ((2a-1)!! (2b-1)!! (2c-1)!!) ]^(1/2)

    This is the standard Cartesian Gaussian primitive normalization.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    a, b, c = powers
    if min(a, b, c) < 0:
        raise ValueError("Gaussian powers must be nonnegative")

    L = a + b + c
    denom = (
        double_factorial_odd(2 * a - 1)
        * double_factorial_odd(2 * b - 1)
        * double_factorial_odd(2 * c - 1)
    )

    value = ((2.0 * alpha / np.pi) ** 1.5) * ((4.0 * alpha) ** L) / denom
    return float(np.sqrt(value))


# =============================================================================
# Cartesian primitive Gaussian object
# =============================================================================

@dataclass(frozen=True)
class PrimitiveGaussian:
    """
    Cartesian primitive Gaussian.

    Represents

        G(r) = N * (x-Ax)^a (y-Ay)^b (z-Az)^c
               exp(-alpha |r-A|^2)

    Parameters
    ----------
    alpha
        Gaussian exponent. Must be positive.

    center
        Center A = (Ax, Ay, Az).

    powers
        Cartesian powers (a,b,c).

    normalized
        If True, include the Cartesian primitive normalization constant.
        If False, use N = 1.

    Notes
    -----
    The powers define Cartesian angular momentum. For example:

        powers=(0,0,0)  s-type
        powers=(1,0,0)  px-like Cartesian primitive
        powers=(0,1,0)  py-like Cartesian primitive
        powers=(0,0,1)  pz-like Cartesian primitive
        powers=(2,0,0)  dxx-like Cartesian primitive
    """

    alpha: float
    center: np.ndarray
    powers: tuple[int, int, int] = (0, 0, 0)
    normalized: bool = False

    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        center = np.asarray(self.center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must be a length-3 array")

        if len(self.powers) != 3:
            raise ValueError("powers must be a tuple of length 3")

        if min(self.powers) < 0:
            raise ValueError("powers must be nonnegative")

        # Because the dataclass is frozen, use object.__setattr__ to store the
        # sanitized NumPy array.
        object.__setattr__(self, "center", center)

    @property
    def angular_momentum(self) -> int:
        """Return total Cartesian angular momentum a+b+c."""
        return int(sum(self.powers))

    @property
    def norm(self) -> float:
        """Return the normalization factor N."""
        if not self.normalized:
            return 1.0
        return gaussian_norm_cartesian(self.alpha, self.powers)

    def describe(self) -> str:
        """Return a compact human-readable description."""
        return (
            f"PrimitiveGaussian(alpha={self.alpha}, "
            f"center={self.center.tolist()}, "
            f"powers={self.powers}, "
            f"normalized={self.normalized})"
        )


# =============================================================================
# Gaussian evaluation
# =============================================================================

def eval_primitive_gaussian(
    gaussian: PrimitiveGaussian,
    X,
    Y,
    Z,
) -> np.ndarray:
    """
    Evaluate a primitive Gaussian on scalar coordinates or NumPy arrays.

    Parameters
    ----------
    gaussian
        PrimitiveGaussian object.
    X, Y, Z
        Scalars or broadcast-compatible NumPy arrays.

    Returns
    -------
    numpy.ndarray
        Values of the Gaussian on the grid.
    """
    Ax, Ay, Az = gaussian.center
    a, b, c = gaussian.powers

    dx = X - Ax
    dy = Y - Ay
    dz = Z - Az

    r2 = dx**2 + dy**2 + dz**2

    polynomial = (dx**a) * (dy**b) * (dz**c)
    exponential = np.exp(-gaussian.alpha * r2)

    return gaussian.norm * polynomial * exponential


def eval_gaussian_product(
    gaussians: list[PrimitiveGaussian],
    X,
    Y,
    Z,
) -> np.ndarray:
    """
    Evaluate a product of primitive Gaussians on a grid.

    This is useful for later Coulomb tests involving products like

        G_a(r) G_b(r)

    For now it is just a convenience function.
    """
    if not gaussians:
        raise ValueError("gaussians list cannot be empty")

    result = np.ones_like(np.asarray(X, dtype=float), dtype=float)
    for g in gaussians:
        result = result * eval_primitive_gaussian(g, X, Y, Z)
    return result


# =============================================================================
# Simple 3D grid helper for numerical normalization checks
# =============================================================================

def make_3d_grid(n: int = 101, extent: float = 5.0):
    """
    Build a cubic 3D grid for rough numerical integration tests.

    Returns
    -------
    X, Y, Z, dV

    where dV is the volume element.

    Warning
    -------
    This is not meant to be a high-quality quadrature rule. It is only for
    debugging and sanity checks.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if extent <= 0:
        raise ValueError("extent must be positive")

    axis = np.linspace(-extent, extent, n)
    dx = axis[1] - axis[0]
    X, Y, Z = np.meshgrid(axis, axis, axis, indexing="ij")
    dV = dx**3
    return X, Y, Z, dV


def numerical_norm_check(
    gaussian: PrimitiveGaussian,
    n: int = 101,
    extent: float = 5.0,
) -> float:
    """
    Numerically approximate integral |G(r)|^2 d r on a cubic grid.

    If gaussian.normalized=True and the grid is large/fine enough, this should
    be close to 1.
    """
    X, Y, Z, dV = make_3d_grid(n=n, extent=extent)
    values = eval_primitive_gaussian(gaussian, X, Y, Z)
    return float(np.sum(np.abs(values) ** 2) * dV)


# =============================================================================
# Command-line demo
# =============================================================================

if __name__ == "__main__":
    print("\n=== Gaussian primitive demo ===")

    g_s = PrimitiveGaussian(
        alpha=1.0,
        center=np.array([0.0, 0.0, 0.0]),
        powers=(0, 0, 0),
        normalized=True,
    )

    g_px = PrimitiveGaussian(
        alpha=1.0,
        center=np.array([0.0, 0.0, 0.0]),
        powers=(1, 0, 0),
        normalized=True,
    )

    for g in [g_s, g_px]:
        print("\n", g.describe())
        print(f"normalization factor N = {g.norm:.10f}")

        # This is only a rough grid check. Increase n/extent for better accuracy,
        # but note that a large 3D grid can become expensive.
        approx_norm = numerical_norm_check(g, n=81, extent=5.0)
        print(f"rough numerical integral |G|^2 = {approx_norm:.8f}")

    print("\nDone.")
