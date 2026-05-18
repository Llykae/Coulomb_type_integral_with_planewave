"""
radial_integrals.py

One-dimensional radial integrals for the atom-centered Coulomb / plane-wave
project.

Why this module?
----------------
For atom-centered Gaussian orbitals, angular and radial structure can often be
separated. The plane-wave expansion introduces spherical Bessel functions

    j_l(k r)

and the Cartesian solid-harmonic form introduces factors like

    j_l(k r) / r^l.

When this is multiplied by Cartesian Gaussian orbitals, the radial part reduces
to integrals of the general type

    ∫_0^∞ r^n exp(-alpha r^2) j_l(k r) dr

or, with the solid-harmonic radial factor,

    ∫_0^∞ r^n exp(-alpha r^2) [j_l(k r) / r^l] dr.

This module implements numerical 1D quadrature for these building blocks.

Current scope
-------------
This is a numerical module, not yet the final analytic formula engine.

It provides:

1. radial_bessel_gaussian_integral

       ∫_0^∞ r^power exp(-alpha r^2) j_l(k r) dr

2. radial_solid_bessel_gaussian_integral

       ∫_0^∞ r^power exp(-alpha r^2) j_l(k r)/r^l dr

3. convergence helpers to compare finite cutoffs and tolerances.

Notes for k ~ 72
----------------
For large k, the integrand oscillates. Adaptive quadrature can still work, but
it may need tighter settings and/or a larger subdivision limit.

The Gaussian exp(-alpha r^2) localizes the integral. A useful radial cutoff is
roughly

    rmax ~ sqrt(-log(tol) / alpha)

because exp(-alpha rmax^2) ~ tol.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil, sqrt, log

import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class RadialIntegralResult:
    """
    Result returned by radial integral routines.

    value
        Numerical integral value.

    estimated_error
        Error estimate returned by scipy.integrate.quad.

    l, k, alpha, power
        Parameters of the integral.

    rmax
        Upper integration cutoff used to approximate infinity.

    kind
        Human-readable label for the type of integral.
    """

    value: float
    estimated_error: float
    l: int
    k: float
    alpha: float
    power: int
    rmax: float
    kind: str

    def print(self) -> None:
        """Pretty-print the result."""
        print(f"{self.kind}")
        print(f"  l        = {self.l}")
        print(f"  k        = {self.k}")
        print(f"  alpha    = {self.alpha}")
        print(f"  power    = {self.power}")
        print(f"  rmax     = {self.rmax}")
        print(f"  value    = {self.value:.16e}")
        print(f"  quad err = {self.estimated_error:.3e}")


# =============================================================================
# Validation and cutoff helpers
# =============================================================================

def validate_radial_parameters(l: int, k: float, alpha: float, power: int) -> None:
    """
    Validate common radial integral parameters.
    """
    if l < 0:
        raise ValueError("l must be nonnegative")
    if k < 0:
        raise ValueError("k must be nonnegative")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if power < 0:
        raise ValueError("power must be nonnegative")


def gaussian_cutoff(alpha: float, tail_tol: float = 1e-14, minimum: float = 8.0) -> float:
    """
    Choose a finite rmax for Gaussian radial integration.

    We choose rmax so that

        exp(-alpha rmax^2) ~ tail_tol.

    That gives

        rmax = sqrt(-log(tail_tol) / alpha).

    A minimum cutoff is also enforced, because for diffuse Gaussians or small
    powers we still want a reasonably large integration domain.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not (0 < tail_tol < 1):
        raise ValueError("tail_tol must be between 0 and 1")

    r = sqrt(-log(tail_tol) / alpha)
    return max(float(r), float(minimum))


def double_factorial_odd(n: int) -> int:
    """
    Odd double factorial n!! with (-1)!! = 1.

    Used in small analytic checks.
    """
    if n in (-1, 0):
        return 1
    if n < -1:
        raise ValueError("n must be >= -1")
    result = 1
    for v in range(n, 0, -2):
        result *= v
    return result


# =============================================================================
# Core radial integrands
# =============================================================================

def bessel_gaussian_integrand(r: float, l: int, k: float, alpha: float, power: int) -> float:
    """
    Integrand for

        r^power exp(-alpha r^2) j_l(k r).
    """
    return (r**power) * np.exp(-alpha * r * r) * spherical_jn(l, k * r)


def solid_bessel_gaussian_integrand(r: float, l: int, k: float, alpha: float, power: int) -> float:
    """
    Integrand for

        r^power exp(-alpha r^2) j_l(k r) / r^l.

    Near r=0, j_l(k r)/r^l has the finite limit

        k^l / (2l+1)!!.

    This removes the apparent singularity.
    """
    if r == 0.0:
        denom = double_factorial_odd(2 * l + 1)
        jl_over_rl = (k**l) / denom
    else:
        jl_over_rl = spherical_jn(l, k * r) / (r**l)

    return (r**power) * np.exp(-alpha * r * r) * jl_over_rl


# =============================================================================
# Core radial integrals
# =============================================================================

def radial_bessel_gaussian_integral(
    l: int,
    k: float,
    alpha: float,
    power: int,
    rmax: float | None = None,
    epsabs: float = 1e-11,
    epsrel: float = 1e-11,
    limit: int = 400,
) -> RadialIntegralResult:
    """
    Numerically compute

        ∫_0^∞ r^power exp(-alpha r^2) j_l(k r) dr.

    The infinity limit is approximated by a finite Gaussian cutoff rmax.
    """
    validate_radial_parameters(l, k, alpha, power)
    if rmax is None:
        rmax = gaussian_cutoff(alpha)

    value, err = quad(
        lambda rr: bessel_gaussian_integrand(rr, l, k, alpha, power),
        0.0,
        rmax,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )

    return RadialIntegralResult(
        value=float(value),
        estimated_error=float(err),
        l=l,
        k=float(k),
        alpha=float(alpha),
        power=power,
        rmax=float(rmax),
        kind="Integral ∫ r^power exp(-alpha r^2) j_l(k r) dr",
    )


def radial_solid_bessel_gaussian_integral(
    l: int,
    k: float,
    alpha: float,
    power: int,
    rmax: float | None = None,
    epsabs: float = 1e-11,
    epsrel: float = 1e-11,
    limit: int = 400,
) -> RadialIntegralResult:
    """
    Numerically compute

        ∫_0^∞ r^power exp(-alpha r^2) [j_l(k r) / r^l] dr.

    This is the radial factor that appears naturally after writing

        Y_lm(rhat) = S_lm(x,y,z) / r^l.
    """
    validate_radial_parameters(l, k, alpha, power)
    if rmax is None:
        rmax = gaussian_cutoff(alpha)

    value, err = quad(
        lambda rr: solid_bessel_gaussian_integrand(rr, l, k, alpha, power),
        0.0,
        rmax,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )

    return RadialIntegralResult(
        value=float(value),
        estimated_error=float(err),
        l=l,
        k=float(k),
        alpha=float(alpha),
        power=power,
        rmax=float(rmax),
        kind="Integral ∫ r^power exp(-alpha r^2) j_l(k r)/r^l dr",
    )


# =============================================================================
# Analytic checks for simple cases
# =============================================================================

def analytic_j0_power2(alpha: float, k: float) -> float:
    """
    Analytic check for

        ∫_0^∞ r^2 exp(-alpha r^2) j_0(k r) dr.

    Since j_0(kr) = sin(kr)/(kr), this integral is

        sqrt(pi) / (4 alpha^(3/2)) * exp(-k^2/(4 alpha)).

    This is the 3D Gaussian Fourier-transform radial core.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return float(np.sqrt(np.pi) / (4.0 * alpha**1.5) * np.exp(-(k * k) / (4.0 * alpha)))


def run_basic_analytic_checks(alpha: float = 1.0, k: float = 2.0) -> None:
    """
    Run a simple numerical-vs-analytic check.
    """
    print("\n=== Basic analytic check ===")
    numerical = radial_bessel_gaussian_integral(l=0, k=k, alpha=alpha, power=2)
    exact = analytic_j0_power2(alpha=alpha, k=k)
    abs_err = abs(numerical.value - exact)
    rel_err = abs_err / max(abs(exact), 1e-300)

    print("Integral: ∫ r^2 exp(-alpha r^2) j_0(k r) dr")
    print(f"alpha     = {alpha}")
    print(f"k         = {k}")
    print(f"numerical = {numerical.value:.16e}")
    print(f"analytic  = {exact:.16e}")
    print(f"abs err   = {abs_err:.3e}")
    print(f"rel err   = {rel_err:.3e}")


# =============================================================================
# Convergence helpers
# =============================================================================

def convergence_with_rmax(
    l: int,
    k: float,
    alpha: float,
    power: int,
    solid: bool = False,
    rmax_values: list[float] | None = None,
) -> list[RadialIntegralResult]:
    """
    Compute the same integral with several radial cutoffs.

    This is useful for checking whether the Gaussian tail cutoff is large enough.
    """
    if rmax_values is None:
        base = gaussian_cutoff(alpha)
        rmax_values = [0.5 * base, 0.75 * base, base, 1.25 * base]

    results = []
    print("\n=== rmax convergence ===")
    print(" rmax              value                 quad err")
    print("--------------------------------------------------------")

    for rmax in rmax_values:
        if solid:
            res = radial_solid_bessel_gaussian_integral(l, k, alpha, power, rmax=rmax)
        else:
            res = radial_bessel_gaussian_integral(l, k, alpha, power, rmax=rmax)
        results.append(res)
        print(f" {rmax:8.3f}   {res.value: .16e}   {res.estimated_error:.3e}")

    return results


def convergence_with_l(
    lmax: int,
    k: float,
    alpha: float,
    power_offset: int = 2,
    solid: bool = False,
) -> list[RadialIntegralResult]:
    """
    Print sample radial integrals for l=0..lmax.

    The power is chosen as

        power = l + power_offset

    by default. This avoids too-singular-looking low-power examples for the
    solid form and gives a simple scan over angular momentum.
    """
    if lmax < 0:
        raise ValueError("lmax must be nonnegative")

    results = []
    print("\n=== l scan ===")
    print(" l   power              value                 quad err")
    print("------------------------------------------------------------")

    for l in range(lmax + 1):
        power = l + power_offset
        if solid:
            res = radial_solid_bessel_gaussian_integral(l, k, alpha, power)
        else:
            res = radial_bessel_gaussian_integral(l, k, alpha, power)
        results.append(res)
        print(f" {l:2d}   {power:5d}   {res.value: .16e}   {res.estimated_error:.3e}")

    return results


# =============================================================================
# Command-line demo
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numerical radial Gaussian-Bessel integrals."
    )

    parser.add_argument("--l", type=int, default=0, help="spherical Bessel angular momentum l")
    parser.add_argument("--k", type=float, default=2.0, help="wave-vector magnitude k")
    parser.add_argument("--alpha", type=float, default=1.0, help="Gaussian exponent alpha")
    parser.add_argument("--power", type=int, default=2, help="radial power r^power")
    parser.add_argument("--rmax", type=float, default=None, help="finite radial cutoff")
    parser.add_argument("--solid", action="store_true", help="integrate j_l(k r)/r^l instead of j_l(k r)")
    parser.add_argument("--epsabs", type=float, default=1e-11, help="absolute quadrature tolerance")
    parser.add_argument("--epsrel", type=float, default=1e-11, help="relative quadrature tolerance")
    parser.add_argument("--limit", type=int, default=400, help="quad subdivision limit")
    parser.add_argument("--check", action="store_true", help="run analytic j0 check")
    parser.add_argument("--rmax-scan", action="store_true", help="scan radial cutoff convergence")
    parser.add_argument("--l-scan", type=int, default=None, help="scan l from 0 to LSCAN")

    args = parser.parse_args()

    print("\n=== Radial integral setup ===")
    print(f"l      = {args.l}")
    print(f"k      = {args.k}")
    print(f"alpha  = {args.alpha}")
    print(f"power  = {args.power}")
    print(f"solid  = {args.solid}")
    print(f"rmax   = {args.rmax if args.rmax is not None else gaussian_cutoff(args.alpha)}")

    if args.solid:
        result = radial_solid_bessel_gaussian_integral(
            l=args.l,
            k=args.k,
            alpha=args.alpha,
            power=args.power,
            rmax=args.rmax,
            epsabs=args.epsabs,
            epsrel=args.epsrel,
            limit=args.limit,
        )
    else:
        result = radial_bessel_gaussian_integral(
            l=args.l,
            k=args.k,
            alpha=args.alpha,
            power=args.power,
            rmax=args.rmax,
            epsabs=args.epsabs,
            epsrel=args.epsrel,
            limit=args.limit,
        )

    print("\n=== Result ===")
    result.print()

    if args.check:
        run_basic_analytic_checks(alpha=args.alpha, k=args.k)

    if args.rmax_scan:
        convergence_with_rmax(
            l=args.l,
            k=args.k,
            alpha=args.alpha,
            power=args.power,
            solid=args.solid,
        )

    if args.l_scan is not None:
        convergence_with_l(
            lmax=args.l_scan,
            k=args.k,
            alpha=args.alpha,
            power_offset=max(args.power - args.l, 0),
            solid=args.solid,
        )


if __name__ == "__main__":
    main()
