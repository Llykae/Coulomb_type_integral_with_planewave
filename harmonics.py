"""
harmonics.py

First module of the Coulomb / plane-wave project.

Goal
----
This module handles the angular objects

    S_lm(x,y,z) = r^l Y_lm(rhat)

called regular solid harmonics.

For each pair (l,m), S_lm is a homogeneous Cartesian polynomial of degree l.
For example,

    S_1,0 = sqrt(3/(4*pi)) z

and

    S_2,0 = sqrt(5/(16*pi)) (2 z^2 - x^2 - y^2)

The important practical task is to rewrite each S_lm as a list of Cartesian
monomials:

    S_lm(x,y,z) = sum_{a,b,c} C_{abc}^{lm} x^a y^b z^c

This is the format we will later feed into the plane-wave expansion and then
into Coulomb-type integrals.

Current scope
-------------
This first version uses explicit formulas for l <= 4.
That is enough to validate the algebra and build the project structure.
Later, if we need high l, we can replace this table with a recurrence-based
solid-harmonic generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import sympy as sp


# =============================================================================
# SciPy spherical-harmonic compatibility wrapper
# =============================================================================

# Different SciPy versions expose spherical harmonics with different APIs.
#
# Newer SciPy:
#
#     sph_harm_y(l, m, theta, phi)
#
# Older SciPy:
#
#     sph_harm(m, l, phi, theta)
#
# Note the change in both function name and argument order.
#
# In this project we use one stable wrapper:
#
#     complex_sph_harm(l, m, theta, phi)
#
# where
#
#     theta = polar angle from the +z axis
#     phi   = azimuthal angle in the xy plane

try:
    from scipy.special import sph_harm_y as _scipy_sph_harm_y

    def complex_sph_harm(l: int, m: int, theta, phi):
        """Return complex spherical harmonic Y_lm(theta, phi)."""
        return _scipy_sph_harm_y(l, m, theta, phi)

except ImportError:
    from scipy.special import sph_harm as _scipy_sph_harm

    def complex_sph_harm(l: int, m: int, theta, phi):
        """Return complex spherical harmonic Y_lm(theta, phi), old SciPy API."""
        return _scipy_sph_harm(m, l, phi, theta)


# =============================================================================
# Symbolic variables
# =============================================================================

# These are the Cartesian variables used in the polynomial expressions.
x, y, z = sp.symbols("x y z", real=True)

# Short aliases used in the formulas.
pi = sp.pi
I = sp.I


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class MonomialTerm:
    """
    One term in a Cartesian polynomial.

    Represents

        coefficient * x^a y^b z^c

    where

        powers = (a, b, c)

    Example
    -------
    The term

        -sqrt(5/(16*pi)) * x^2

    is represented as

        MonomialTerm(
            powers=(2, 0, 0),
            coefficient=-sqrt(5/(16*pi)),
        )
    """

    powers: tuple[int, int, int]
    coefficient: sp.Expr


# =============================================================================
# Solid harmonics: explicit low-l table
# =============================================================================

@lru_cache(None)
def solid_harmonic_expr(l: int, m: int) -> sp.Expr:
    """
    Return the regular solid harmonic

        S_lm(x,y,z) = r^l Y_lm(rhat)

    as a SymPy expression.

    Parameters
    ----------
    l
        Angular momentum quantum number. Must be >= 0.
    m
        Magnetic quantum number. Must satisfy -l <= m <= l.

    Returns
    -------
    sympy.Expr
        Homogeneous polynomial of total degree l in x, y, z.

    Convention
    ----------
    We use the complex Condon-Shortley spherical harmonics, consistent with
    SciPy's spherical harmonic convention.

    Notes
    -----
    This first version is an explicit table for l <= 4. That is deliberate:
    it makes the algebra transparent and easy to validate before we introduce
    recurrence relations.
    """

    if l < 0:
        raise ValueError(f"l must be nonnegative, got l={l}")

    if abs(m) > l:
        raise ValueError(f"Need |m| <= l, got l={l}, m={m}")

    # Common complex Cartesian combinations.
    #
    # Many complex spherical harmonics contain powers of
    #
    #     x + i y
    #     x - i y
    xp = x + I * y
    xm = x - I * y

    # -------------------------------------------------------------------------
    # l = 0, s-type
    # -------------------------------------------------------------------------
    if l == 0:
        if m == 0:
            return sp.sqrt(1 / (4 * pi))

    # -------------------------------------------------------------------------
    # l = 1, p-type
    # -------------------------------------------------------------------------
    if l == 1:
        if m == -1:
            return sp.sqrt(3 / (8 * pi)) * xm
        if m == 0:
            return sp.sqrt(3 / (4 * pi)) * z
        if m == 1:
            return -sp.sqrt(3 / (8 * pi)) * xp

    # -------------------------------------------------------------------------
    # l = 2, d-type
    # -------------------------------------------------------------------------
    if l == 2:
        if m == -2:
            return sp.sqrt(15 / (32 * pi)) * xm**2
        if m == -1:
            return sp.sqrt(15 / (8 * pi)) * z * xm
        if m == 0:
            return sp.sqrt(5 / (16 * pi)) * (2 * z**2 - x**2 - y**2)
        if m == 1:
            return -sp.sqrt(15 / (8 * pi)) * z * xp
        if m == 2:
            return sp.sqrt(15 / (32 * pi)) * xp**2

    # -------------------------------------------------------------------------
    # l = 3, f-type
    # -------------------------------------------------------------------------
    if l == 3:
        if m == -3:
            return sp.sqrt(35 / (64 * pi)) * xm**3
        if m == -2:
            return sp.sqrt(105 / (32 * pi)) * z * xm**2
        if m == -1:
            return sp.sqrt(21 / (64 * pi)) * xm * (4 * z**2 - x**2 - y**2)
        if m == 0:
            return sp.sqrt(7 / (16 * pi)) * z * (2 * z**2 - 3 * x**2 - 3 * y**2)
        if m == 1:
            return -sp.sqrt(21 / (64 * pi)) * xp * (4 * z**2 - x**2 - y**2)
        if m == 2:
            return sp.sqrt(105 / (32 * pi)) * z * xp**2
        if m == 3:
            return -sp.sqrt(35 / (64 * pi)) * xp**3

    # -------------------------------------------------------------------------
    # l = 4, g-type
    # -------------------------------------------------------------------------
    if l == 4:
        r2 = x**2 + y**2 + z**2

        if m == -4:
            return sp.Rational(3, 16) * sp.sqrt(35 / (2 * pi)) * xm**4
        if m == -3:
            return sp.Rational(3, 8) * sp.sqrt(35 / pi) * z * xm**3
        if m == -2:
            return sp.Rational(3, 8) * sp.sqrt(5 / (2 * pi)) * xm**2 * (7 * z**2 - r2)
        if m == -1:
            return sp.Rational(3, 8) * sp.sqrt(5 / pi) * xm * z * (7 * z**2 - 3 * r2)
        if m == 0:
            return sp.Rational(3, 16) * sp.sqrt(1 / pi) * (
                35 * z**4 - 30 * z**2 * r2 + 3 * r2**2
            )
        if m == 1:
            return -sp.Rational(3, 8) * sp.sqrt(5 / pi) * xp * z * (7 * z**2 - 3 * r2)
        if m == 2:
            return sp.Rational(3, 8) * sp.sqrt(5 / (2 * pi)) * xp**2 * (7 * z**2 - r2)
        if m == 3:
            return -sp.Rational(3, 8) * sp.sqrt(35 / pi) * z * xp**3
        if m == 4:
            return sp.Rational(3, 16) * sp.sqrt(35 / (2 * pi)) * xp**4

    raise NotImplementedError(
        "solid_harmonic_expr currently supports only l <= 4. "
        "Later we can replace this table by a recurrence generator."
    )


# =============================================================================
# Polynomial expansion into monomials
# =============================================================================

def solid_harmonic_polynomial(l: int, m: int, simplify: bool = True) -> sp.Expr:
    """
    Return S_lm as an expanded Cartesian polynomial.

    This simply wraps solid_harmonic_expr and calls SymPy expansion.
    """

    expr = sp.expand(solid_harmonic_expr(l, m))
    return sp.simplify(expr) if simplify else expr


def solid_harmonic_terms(l: int, m: int) -> list[MonomialTerm]:
    """
    Expand S_lm into Cartesian monomial terms.

    Returns
    -------
    list[MonomialTerm]
        A list of terms representing

            S_lm(x,y,z) = sum coefficient * x^a y^b z^c

    Example
    -------
    S_2,0 is

        sqrt(5/(16*pi)) * (2 z^2 - x^2 - y^2)

    so this function returns terms with powers

        (2,0,0), (0,2,0), (0,0,2)

    and their corresponding coefficients.
    """

    expr = solid_harmonic_polynomial(l, m, simplify=False)

    # Build a SymPy polynomial object in x,y,z.
    # This lets us ask for the coefficient of each monomial cleanly.
    poly = sp.Poly(expr, x, y, z)

    terms: list[MonomialTerm] = []
    for powers, coeff in poly.terms():
        terms.append(MonomialTerm(powers=powers, coefficient=sp.simplify(coeff)))

    return terms


# =============================================================================
# Numerical evaluation
# =============================================================================

@lru_cache(None)
def lambdified_solid_harmonic(l: int, m: int) -> Callable:
    """
    Convert S_lm(x,y,z) into a fast NumPy function.

    SymPy expressions are useful for algebra, but slow for grid evaluation.
    lambdify creates a function that can accept NumPy arrays.
    """

    expr = solid_harmonic_polynomial(l, m)
    return sp.lambdify((x, y, z), expr, modules="numpy")


def eval_solid_harmonic(l: int, m: int, X, Y, Z) -> np.ndarray:
    """
    Evaluate S_lm on scalar coordinates or NumPy arrays.

    Parameters
    ----------
    X, Y, Z
        Scalars or broadcast-compatible NumPy arrays.

    Returns
    -------
    numpy.ndarray
        Complex values of S_lm(X,Y,Z).
    """

    f = lambdified_solid_harmonic(l, m)
    return np.asarray(f(X, Y, Z), dtype=np.complex128)


# =============================================================================
# Coordinate conversion and validation
# =============================================================================

def cartesian_to_spherical_angles(X, Y, Z):
    """
    Convert Cartesian coordinates to spherical angles.

    Returns
    -------
    R, theta, phi

    where

        R     = sqrt(X^2 + Y^2 + Z^2)
        theta = polar angle from +z
        phi   = atan2(Y, X)

    At R=0, theta and phi are undefined. We set theta=0 and phi=0 there.
    This is fine for validation because we avoid the origin in random tests.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    R = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.zeros_like(R, dtype=float)
    phi = np.zeros_like(R, dtype=float)

    mask = R > 1e-14
    theta[mask] = np.arccos(Z[mask] / R[mask])
    phi[mask] = np.arctan2(Y[mask], X[mask])

    return R, theta, phi


def check_against_scipy(lmax: int = 4, ntest: int = 200, seed: int = 1234) -> None:
    """
    Verify the identity

        S_lm(x,y,z) / r^l = Y_lm(theta, phi)

    against SciPy's spherical harmonics.

    This is the most important unit test for this module.
    """

    if lmax > 4:
        raise ValueError("This table currently supports only lmax <= 4.")

    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(ntest, 3))

    # Avoid points too close to the origin because we divide by r^l.
    R_raw = np.linalg.norm(pts, axis=1)
    pts = pts[R_raw > 1e-10]

    Xp = pts[:, 0]
    Yp = pts[:, 1]
    Zp = pts[:, 2]

    R, theta, phi = cartesian_to_spherical_angles(Xp, Yp, Zp)

    print("\n=== Checking S_lm / r^l against SciPy Y_lm ===")
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            lhs = eval_solid_harmonic(l, m, Xp, Yp, Zp) / (R**l)
            rhs = complex_sph_harm(l, m, theta, phi)
            err = np.max(np.abs(lhs - rhs))
            print(f"l={l:2d}, m={m:3d}, max error = {err:.3e}")


# =============================================================================
# Small command-line demo
# =============================================================================

if __name__ == "__main__":
    print("\nPrinting S_lm polynomials up to l=2")
    for l in range(3):
        for m in range(-l, l + 1):
            print(f"\nl={l}, m={m}")
            print(solid_harmonic_polynomial(l, m))
            print("Monomial terms:")
            for term in solid_harmonic_terms(l, m):
                print(f"  powers={term.powers}, coeff={term.coefficient}")

    check_against_scipy(lmax=4)
