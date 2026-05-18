"""
radial_coulomb_2d.py

Two-dimensional radial Coulomb integrals for the atom-centered Coulomb /
plane-wave project.

Purpose
-------
The symbolic atom-centered terms contain radial placeholders of the form

    R(...)

This module gives the first numerical definition of those radial objects.

For one angular coupling, the radial integral has the generic structure

    R = ∫_0^∞ dr ∫_0^∞ ds
        r^(2 + n_r) s^(2 + n_s)
        exp(-alpha_r r^2)
        exp(-alpha_s s^2)
        j_lp(k r)
        r_<^L / r_>^(L+1)

where

    r_< = min(r,s)
    r_> = max(r,s)

and

    n_r = radial power from the r-side Cartesian/solid-harmonic channel
    n_s = radial power from the s-side density channel
    lp  = plane-wave angular momentum
    L   = Coulomb multipole rank

Why 2D radial integrals?
------------------------
The brute-force prototype integrated over 6 Cartesian dimensions. For
atom-centered orbitals, angular algebra reduces the problem to angular
coefficients times radial integrals. This module is the first step toward that
reduction.

Current scope
-------------
This is still numerical. It uses scipy.integrate.quad nested integration after
splitting the Coulomb radial kernel into two domains:

    s < r
    s > r

For fixed r,

    ∫_0^∞ ds ... r_<^L / r_>^(L+1)

becomes

    ∫_0^r   ds ... s^L / r^(L+1)
  + ∫_r^∞   ds ... r^L / s^(L+1)

This avoids min/max inside the inner integrand and is closer to analytic radial
formulas.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import sqrt, log

import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn


# =============================================================================
# Result object
# =============================================================================

@dataclass(frozen=True)
class RadialCoulomb2DResult:
    """
    Result for one 2D radial Coulomb integral.
    """

    value: float
    estimated_error_outer: float
    lp: int
    L: int
    k: float
    alpha_r: float
    alpha_s: float
    n_r: int
    n_s: int
    rmax: float
    smax: float

    def print(self) -> None:
        print("\n=== Radial Coulomb 2D result ===")
        print(f"lp       = {self.lp}")
        print(f"L        = {self.L}")
        print(f"k        = {self.k}")
        print(f"alpha_r  = {self.alpha_r}")
        print(f"alpha_s  = {self.alpha_s}")
        print(f"n_r      = {self.n_r}")
        print(f"n_s      = {self.n_s}")
        print(f"rmax     = {self.rmax}")
        print(f"smax     = {self.smax}")
        print(f"value    = {self.value:.16e}")
        print(f"outer err= {self.estimated_error_outer:.3e}")


# =============================================================================
# Helpers
# =============================================================================

def validate_parameters(
    lp: int,
    L: int,
    k: float,
    alpha_r: float,
    alpha_s: float,
    n_r: int,
    n_s: int,
) -> None:
    """Validate radial Coulomb integral parameters."""
    if lp < 0:
        raise ValueError("lp must be nonnegative")
    if L < 0:
        raise ValueError("L must be nonnegative")
    if k < 0:
        raise ValueError("k must be nonnegative")
    if alpha_r <= 0 or alpha_s <= 0:
        raise ValueError("alpha_r and alpha_s must be positive")
    if n_r < 0 or n_s < 0:
        raise ValueError("n_r and n_s must be nonnegative")


def gaussian_cutoff(alpha: float, tail_tol: float = 1e-14, minimum: float = 8.0) -> float:
    """
    Choose a finite cutoff such that exp(-alpha*rmax^2) is tiny.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not (0 < tail_tol < 1):
        raise ValueError("tail_tol must be in (0,1)")
    return max(float(sqrt(-log(tail_tol) / alpha)), float(minimum))


def s_weight(s: float, alpha_s: float, n_s: int) -> float:
    """
    s-side radial weight, excluding Coulomb kernel.

        s^(2+n_s) exp(-alpha_s s^2)
    """
    return (s ** (2 + n_s)) * np.exp(-alpha_s * s * s)


def r_weight_with_bessel(r: float, lp: int, k: float, alpha_r: float, n_r: int) -> float:
    """
    r-side radial weight, including spherical Bessel j_lp(k r):

        r^(2+n_r) exp(-alpha_r r^2) j_lp(k r)
    """
    return (r ** (2 + n_r)) * np.exp(-alpha_r * r * r) * spherical_jn(lp, k * r)


# =============================================================================
# Inner s integrals
# =============================================================================

def inner_s_integral_split(
    r: float,
    L: int,
    alpha_s: float,
    n_s: int,
    smax: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float]:
    """
    Compute the inner s integral for fixed r:

        ∫_0^∞ ds s^(2+n_s) exp(-alpha_s s^2) r_<^L / r_>^(L+1)

    by splitting at s=r:

        ∫_0^r   ds weight_s(s) s^L / r^(L+1)
      + ∫_r^∞   ds weight_s(s) r^L / s^(L+1)

    The upper infinity is approximated by smax.

    Returns
    -------
    value, estimated_error
    """
    if r == 0.0:
        # At r=0, for L>0 the kernel factor in the s>r part has r^L=0.
        # For L=0, the kernel is 1/s. The s<r region is zero length.
        if L == 0:
            val, err = quad(
                lambda s: s_weight(s, alpha_s, n_s) / s if s != 0.0 else 0.0,
                0.0,
                smax,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
            )
            return float(val), float(err)
        return 0.0, 0.0

    # Region 1: 0 <= s <= r
    upper1 = min(r, smax)
    if upper1 > 0.0:
        val1, err1 = quad(
            lambda s: s_weight(s, alpha_s, n_s) * (s ** L) / (r ** (L + 1)),
            0.0,
            upper1,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
    else:
        val1, err1 = 0.0, 0.0

    # Region 2: r <= s <= smax
    if r < smax:
        val2, err2 = quad(
            lambda s: s_weight(s, alpha_s, n_s) * (r ** L) / (s ** (L + 1)),
            r,
            smax,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
    else:
        val2, err2 = 0.0, 0.0

    return float(val1 + val2), float(err1 + err2)


# =============================================================================
# Main 2D radial integral
# =============================================================================

def radial_coulomb_2d_integral(
    lp: int,
    L: int,
    k: float,
    alpha_r: float,
    alpha_s: float,
    n_r: int,
    n_s: int,
    rmax: float | None = None,
    smax: float | None = None,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 300,
) -> RadialCoulomb2DResult:
    """
    Compute the 2D radial Coulomb integral

        ∫_0^∞ dr ∫_0^∞ ds
        r^(2+n_r) s^(2+n_s)
        exp(-alpha_r r^2) exp(-alpha_s s^2)
        j_lp(k r)
        r_<^L / r_>^(L+1).

    Parameters
    ----------
    lp
        Plane-wave angular momentum.
    L
        Coulomb multipole rank.
    k
        Plane-wave magnitude.
    alpha_r
        Gaussian exponent on the r side.
    alpha_s
        Combined Gaussian exponent on the s-density side. If
        phi_s1 has exponent alpha_s1 and phi_s2 has exponent alpha_s2, then
        alpha_s = alpha_s1 + alpha_s2.
    n_r
        Radial power from the r-side Cartesian/solid-harmonic polynomial.
    n_s
        Radial power from the s-side density polynomial.
    """
    validate_parameters(lp, L, k, alpha_r, alpha_s, n_r, n_s)

    if rmax is None:
        rmax = gaussian_cutoff(alpha_r)
    if smax is None:
        smax = gaussian_cutoff(alpha_s)

    def outer_integrand(r: float) -> float:
        inner, _inner_err = inner_s_integral_split(
            r=r,
            L=L,
            alpha_s=alpha_s,
            n_s=n_s,
            smax=smax,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        return r_weight_with_bessel(r, lp, k, alpha_r, n_r) * inner

    value, outer_err = quad(
        outer_integrand,
        0.0,
        rmax,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )

    return RadialCoulomb2DResult(
        value=float(value),
        estimated_error_outer=float(outer_err),
        lp=lp,
        L=L,
        k=float(k),
        alpha_r=float(alpha_r),
        alpha_s=float(alpha_s),
        n_r=n_r,
        n_s=n_s,
        rmax=float(rmax),
        smax=float(smax),
    )


# =============================================================================
# Basic sanity checks
# =============================================================================

def radial_coulomb_2d_grid_reference(
    lp: int,
    L: int,
    k: float,
    alpha_r: float,
    alpha_s: float,
    n_r: int,
    n_s: int,
    rmax: float,
    smax: float,
    ngrid: int = 500,
) -> float:
    """
    Crude tensor-grid reference for the same radial integral.

    This is only for sanity checks. The split quad integral should be preferred.
    """
    r = np.linspace(0.0, rmax, ngrid)
    s = np.linspace(0.0, smax, ngrid)
    dr = r[1] - r[0]
    ds = s[1] - s[0]

    R, S = np.meshgrid(r, s, indexing="ij")
    r_less = np.minimum(R, S)
    r_greater = np.maximum(R, S)

    kernel = np.zeros_like(R)
    mask = r_greater > 0.0
    kernel[mask] = (r_less[mask] ** L) / (r_greater[mask] ** (L + 1))

    integrand = (
        (R ** (2 + n_r))
        * (S ** (2 + n_s))
        * np.exp(-alpha_r * R * R)
        * np.exp(-alpha_s * S * S)
        * spherical_jn(lp, k * R)
        * kernel
    )

    return float(np.sum(integrand) * dr * ds)


def run_grid_sanity_check(
    lp: int,
    L: int,
    k: float,
    alpha_r: float,
    alpha_s: float,
    n_r: int,
    n_s: int,
    rmax: float,
    smax: float,
) -> None:
    """Compare split quad with a crude 2D grid reference."""
    print("\n=== Grid sanity check ===")
    quad_res = radial_coulomb_2d_integral(
        lp=lp,
        L=L,
        k=k,
        alpha_r=alpha_r,
        alpha_s=alpha_s,
        n_r=n_r,
        n_s=n_s,
        rmax=rmax,
        smax=smax,
    )
    grid_val = radial_coulomb_2d_grid_reference(
        lp=lp,
        L=L,
        k=k,
        alpha_r=alpha_r,
        alpha_s=alpha_s,
        n_r=n_r,
        n_s=n_s,
        rmax=rmax,
        smax=smax,
        ngrid=600,
    )
    print(f"quad value = {quad_res.value:.16e}")
    print(f"grid value = {grid_val:.16e}")
    print(f"abs diff   = {abs(quad_res.value - grid_val):.3e}")


# =============================================================================
# Convergence helpers
# =============================================================================

def convergence_with_cutoff(
    lp: int,
    L: int,
    k: float,
    alpha_r: float,
    alpha_s: float,
    n_r: int,
    n_s: int,
) -> list[RadialCoulomb2DResult]:
    """Scan common rmax=smax cutoff values."""
    base = max(gaussian_cutoff(alpha_r), gaussian_cutoff(alpha_s))
    cutoffs = [0.5 * base, 0.75 * base, base, 1.25 * base]

    results = []
    print("\n=== cutoff convergence ===")
    print(" cutoff              value                 outer err")
    print("---------------------------------------------------------")
    for cutoff in cutoffs:
        res = radial_coulomb_2d_integral(
            lp=lp,
            L=L,
            k=k,
            alpha_r=alpha_r,
            alpha_s=alpha_s,
            n_r=n_r,
            n_s=n_s,
            rmax=cutoff,
            smax=cutoff,
        )
        results.append(res)
        print(f" {cutoff:8.3f}   {res.value: .16e}   {res.estimated_error_outer:.3e}")
    return results


# =============================================================================
# Command-line demo
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="2D radial Coulomb integral prototype.")
    parser.add_argument("--lp", type=int, default=0, help="plane-wave angular momentum")
    parser.add_argument("--L", type=int, default=0, help="Coulomb multipole rank")
    parser.add_argument("--k", type=float, default=2.0, help="plane-wave magnitude")
    parser.add_argument("--alpha-r", type=float, default=1.0, help="r-side Gaussian exponent")
    parser.add_argument("--alpha-s", type=float, default=2.0, help="s-density Gaussian exponent alpha_s1+alpha_s2")
    parser.add_argument("--n-r", type=int, default=0, help="r-side polynomial radial power")
    parser.add_argument("--n-s", type=int, default=0, help="s-side density polynomial radial power")
    parser.add_argument("--rmax", type=float, default=None)
    parser.add_argument("--smax", type=float, default=None)
    parser.add_argument("--epsabs", type=float, default=1e-10)
    parser.add_argument("--epsrel", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--cutoff-scan", action="store_true")
    parser.add_argument("--grid-check", action="store_true")

    args = parser.parse_args()

    rmax = args.rmax if args.rmax is not None else gaussian_cutoff(args.alpha_r)
    smax = args.smax if args.smax is not None else gaussian_cutoff(args.alpha_s)

    print("\n=== Radial Coulomb 2D setup ===")
    print(f"lp      = {args.lp}")
    print(f"L       = {args.L}")
    print(f"k       = {args.k}")
    print(f"alpha_r = {args.alpha_r}")
    print(f"alpha_s = {args.alpha_s}")
    print(f"n_r     = {args.n_r}")
    print(f"n_s     = {args.n_s}")
    print(f"rmax    = {rmax}")
    print(f"smax    = {smax}")

    result = radial_coulomb_2d_integral(
        lp=args.lp,
        L=args.L,
        k=args.k,
        alpha_r=args.alpha_r,
        alpha_s=args.alpha_s,
        n_r=args.n_r,
        n_s=args.n_s,
        rmax=args.rmax,
        smax=args.smax,
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        limit=args.limit,
    )

    result.print()

    if args.cutoff_scan:
        convergence_with_cutoff(
            lp=args.lp,
            L=args.L,
            k=args.k,
            alpha_r=args.alpha_r,
            alpha_s=args.alpha_s,
            n_r=args.n_r,
            n_s=args.n_s,
        )

    if args.grid_check:
        run_grid_sanity_check(
            lp=args.lp,
            L=args.L,
            k=args.k,
            alpha_r=args.alpha_r,
            alpha_s=args.alpha_s,
            n_r=args.n_r,
            n_s=args.n_s,
            rmax=rmax,
            smax=smax,
        )


if __name__ == "__main__":
    main()
