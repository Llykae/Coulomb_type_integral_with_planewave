"""
analytic_sss_check.py

Analytic validation for the atom-centered Coulomb / plane-wave integral in the
simplest nontrivial case:

    phi_r  = s Gaussian
    phi_s1 = s Gaussian
    phi_s2 = s Gaussian
    lmax_pw = 0

Target integral
---------------

    I(k) = ∫∫ d^3r d^3s
           exp(i k.r)
           exp(-alpha_r r^2)
           1/|r-s|
           exp(-alpha_s1 s^2)
           exp(-alpha_s2 s^2)

For s/s/s, only the plane-wave lp=0 and Coulomb L=0 channel survives, so
lmax_pw=0 is exact.

Let

    a = alpha_r
    b = alpha_s1 + alpha_s2
    k = |k|

For unnormalized primitives with unit coefficients, the analytic result is

    I(k) = 2*pi^(5/2) / (a*b*sqrt(a+b))                         if k = 0

and for k > 0,

    I(k) = 2*pi^3 / (sqrt(a)*b^(3/2)*k)
           exp(-k^2/(4a))
           erfi( k/2 * sqrt( b/(a*(a+b)) ) )

For numerical stability, the implementation uses Dawson's integral:

    exp(-k^2/(4a)) erfi(x)
      = exp(-k^2/(4(a+b))) * 2/sqrt(pi) * dawson(x)

where

    x = k/2 * sqrt( b/(a*(a+b)) ).

If --normalized is passed, the analytic result is multiplied by the three
Cartesian s normalization factors, matching the project primitive convention.

Examples
--------

Unnormalized, k from E=2 Ha:

    python3 analytic_sss_check.py --energy 2.0

Normalized:

    python3 analytic_sss_check.py --energy 2.0 --normalized

Different exponents:

    python3 analytic_sss_check.py \
      --alpha-r 1.2 \
      --alpha-s1 0.7 \
      --alpha-s2 1.5 \
      --energy-ev 2600 \
      --direction 1 0 0
"""

from __future__ import annotations

import argparse
import math

try:
    from scipy.special import dawsn
except Exception:  # pragma: no cover
    dawsn = None

try:
    from scipy.special import erfi
except Exception:  # pragma: no cover
    erfi = None

from gaussian import gaussian_norm_cartesian
from full_coulomb_integral import ContractedGaussian
from density_contraction import build_contracted_density
from full_coulomb_integral_density import full_coulomb_plane_wave_integral_density
from plane_wave_parameters import add_plane_wave_cli_arguments, plane_wave_from_cli_args


# =============================================================================
# Analytic formula
# =============================================================================

def analytic_sss_unnormalized(alpha_r: float, alpha_s1: float, alpha_s2: float, k: float) -> float:
    """
    Analytic s/s/s Coulomb / plane-wave integral for unnormalized primitives.

    Parameters
    ----------
    alpha_r
        Exponent of phi_r.

    alpha_s1, alpha_s2
        Exponents of the two s-side Gaussians. Only their sum enters the
        s-density exponent.

    k
        Magnitude of the plane-wave vector.
    """
    if alpha_r <= 0 or alpha_s1 <= 0 or alpha_s2 <= 0:
        raise ValueError("All exponents must be positive")

    a = float(alpha_r)
    b = float(alpha_s1) + float(alpha_s2)
    k = abs(float(k))

    if k < 1e-14:
        return 2.0 * math.pi ** 2.5 / (a * b * math.sqrt(a + b))

    x = 0.5 * k * math.sqrt(b / (a * (a + b)))

    # Stable form using Dawson's integral.
    if dawsn is not None:
        return (
            4.0
            * math.pi ** 2.5
            / (math.sqrt(a) * b ** 1.5 * k)
            * math.exp(-k * k / (4.0 * (a + b)))
            * float(dawsn(x))
        )

    # Fallback direct erfi form. Less stable for very large x.
    if erfi is None:
        raise RuntimeError("Need scipy.special.dawsn or scipy.special.erfi for k > 0")

    return (
        2.0
        * math.pi ** 3
        / (math.sqrt(a) * b ** 1.5 * k)
        * math.exp(-k * k / (4.0 * a))
        * float(erfi(x))
    )


def analytic_sss_full(
    alpha_r: float,
    alpha_s1: float,
    alpha_s2: float,
    k: float,
    coef_r: float = 1.0,
    coef_s1: float = 1.0,
    coef_s2: float = 1.0,
    normalized: bool = False,
) -> float:
    """
    Analytic result including coefficients and optional primitive normalization.
    """
    value = analytic_sss_unnormalized(alpha_r, alpha_s1, alpha_s2, k)

    factor = float(coef_r) * float(coef_s1) * float(coef_s2)

    if normalized:
        factor *= gaussian_norm_cartesian(alpha_r, (0, 0, 0))
        factor *= gaussian_norm_cartesian(alpha_s1, (0, 0, 0))
        factor *= gaussian_norm_cartesian(alpha_s2, (0, 0, 0))

    return factor * value


# =============================================================================
# Project numerical path
# =============================================================================

def project_sss_value(
    alpha_r: float,
    alpha_s1: float,
    alpha_s2: float,
    coef_r: float,
    coef_s1: float,
    coef_s2: float,
    normalized: bool,
    plane_wave,
    epsabs: float,
    epsrel: float,
    limit: int,
):
    """
    Evaluate the same s/s/s integral using the project density path.
    """
    phi_r = ContractedGaussian.single(
        alpha=alpha_r,
        powers=(0, 0, 0),
        coefficient=coef_r,
        normalized=normalized,
        label="phi_r_s",
    )
    phi_s1 = ContractedGaussian.single(
        alpha=alpha_s1,
        powers=(0, 0, 0),
        coefficient=coef_s1,
        normalized=normalized,
        label="phi_s1_s",
    )
    phi_s2 = ContractedGaussian.single(
        alpha=alpha_s2,
        powers=(0, 0, 0),
        coefficient=coef_s2,
        normalized=normalized,
        label="phi_s2_s",
    )

    density, density_report = build_contracted_density(phi_s1, phi_s2)

    result = full_coulomb_plane_wave_integral_density(
        phi_r=phi_r,
        density=density,
        plane_wave=plane_wave,
        lmax_pw=0,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        use_radial_table=True,
        precompute_radial=False,
        print_radial_report=False,
        print_hard_keys=0,
        max_contributions_store=5,
    )

    return result, density_report


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analytic s/s/s validation for lmax_pw=0.")

    add_plane_wave_cli_arguments(parser)

    parser.add_argument("--alpha-r", type=float, default=1.0)
    parser.add_argument("--alpha-s1", type=float, default=1.0)
    parser.add_argument("--alpha-s2", type=float, default=1.0)

    parser.add_argument("--coef-r", type=float, default=1.0)
    parser.add_argument("--coef-s1", type=float, default=1.0)
    parser.add_argument("--coef-s2", type=float, default=1.0)

    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--epsabs", type=float, default=1e-10)
    parser.add_argument("--epsrel", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-8)

    args = parser.parse_args()

    plane_wave = plane_wave_from_cli_args(args)

    print("\n=== Analytic s/s/s validation setup ===")
    print(plane_wave.describe())
    print("lmax_pw = 0  # exact for s/s/s")
    print(f"alpha_r  = {args.alpha_r}")
    print(f"alpha_s1 = {args.alpha_s1}")
    print(f"alpha_s2 = {args.alpha_s2}")
    print(f"alpha_density = {args.alpha_s1 + args.alpha_s2}")
    print(f"normalized = {args.normalized}")

    analytic = analytic_sss_full(
        alpha_r=args.alpha_r,
        alpha_s1=args.alpha_s1,
        alpha_s2=args.alpha_s2,
        k=plane_wave.k_abs,
        coef_r=args.coef_r,
        coef_s1=args.coef_s1,
        coef_s2=args.coef_s2,
        normalized=args.normalized,
    )

    numerical, density_report = project_sss_value(
        alpha_r=args.alpha_r,
        alpha_s1=args.alpha_s1,
        alpha_s2=args.alpha_s2,
        coef_r=args.coef_r,
        coef_s1=args.coef_s1,
        coef_s2=args.coef_s2,
        normalized=args.normalized,
        plane_wave=plane_wave,
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        limit=args.limit,
    )

    print("\n=== Density contraction ===")
    density_report.print()

    print("\n=== Results ===")
    print(f"analytic  = {analytic:.16e}")
    print(f"project   = {numerical.value.real:.16e} + {numerical.value.imag:.16e} i")

    abs_err = abs(numerical.value - analytic)
    rel_err = abs_err / max(1.0, abs(analytic))

    print(f"abs error = {abs_err:.6e}")
    print(f"rel error = {rel_err:.6e}")
    print(f"tol       = {args.tol:.6e}")

    if abs_err <= args.tol or rel_err <= args.tol:
        print("status    = PASS")
    else:
        print("status    = FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
