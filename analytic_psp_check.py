"""
analytic_psp_check.py

Analytic validation for a less trivial atom-centered Coulomb / plane-wave case:

    phi_r  = p_x Gaussian
    phi_s1 = s Gaussian
    phi_s2 = p_x Gaussian

so the s-side density is p_x:

    rho_s(s) = phi_s1(s) phi_s2(s) ~ x_s exp(-B s^2)

with

    B = alpha_s1 + alpha_s2.

Target full integral
--------------------

    I(k) = ∫∫ d^3r d^3s
           exp(i k x_r)
           x_r exp(-a r^2)
           1/|r-s|
           x_s exp(-B s^2)

for k along x.

Selection rule note
-------------------
For p_x / (s p_x), the angular channels are:

    r side      : l_r = 1
    density     : l_s = 1
    Coulomb     : L = 1

The plane-wave channel l_p must satisfy the Gaunt selection rules for

    ∫ Y_1 Y_l_p Y_1* dΩ.

The nonzero plane-wave channels are

    l_p = 0 and l_p = 2.

Therefore:

    lmax_pw = 1 gives only the l_p=0 partial result;
    lmax_pw = 2 is the exact full result for this p/s/p case.

This script compares the project value to the analytic full value. For a true
analytic comparison, use lmax_pw >= 2.

Analytic reference
------------------
Using the Fourier representation of the Coulomb kernel,

    1/|r-s| = ∫ d^3q/(2π)^3 4π/q^2 exp(i q·(r-s)),

and Gaussian Fourier transforms,

    ∫ x exp(i P·r) exp(-a r^2) dr
      = i P_x/(2a) (π/a)^(3/2) exp(-P^2/(4a)),

one obtains a one-dimensional analytic reduction:

    I_pxspx(k) = K/(4 a B) ∫_0^∞ dt π^(3/2) S^(-3/2)
                 exp[-A D k^2 / S]
                 [1/(2S) - k^2 A D / S^2]

where

    A = 1/(4a)
    C = 1/(4B)
    D = C + t
    S = A + D
    K = 4π/(2π)^3 (π/a)^(3/2) (π/B)^(3/2)

This is independent of the project's angular/radial decomposition and is used
as the analytic reference here.

Examples
--------

Exact full comparison, default exponents:

    python3 analytic_psp_check.py --energy 2.0 --lmax-pw 2

Show that lmax=1 is partial:

    python3 analytic_psp_check.py --energy 2.0 --lmax-pw 1

Normalized primitives:

    python3 analytic_psp_check.py --energy 2.0 --lmax-pw 2 --normalized
"""

from __future__ import annotations

import argparse
import math

from scipy.integrate import quad

from gaussian import gaussian_norm_cartesian
from full_coulomb_integral import ContractedGaussian
from density_contraction import build_contracted_density
from full_coulomb_integral_density import full_coulomb_plane_wave_integral_density
from plane_wave_parameters import add_plane_wave_cli_arguments, plane_wave_from_cli_args
from angular_pipeline import angular_pipeline


# =============================================================================
# Analytic formula
# =============================================================================

def analytic_px_s_px_unnormalized(
    alpha_r: float,
    alpha_s1: float,
    alpha_s2: float,
    k: float,
    epsabs: float = 1e-12,
    epsrel: float = 1e-12,
    limit: int = 300,
) -> float:
    """
    Analytic p_x / (s p_x) full integral for unnormalized primitives.

    This uses the one-dimensional Fourier/Laplace reduced expression described
    in the module docstring.
    """
    if alpha_r <= 0 or alpha_s1 <= 0 or alpha_s2 <= 0:
        raise ValueError("All Gaussian exponents must be positive")

    a = float(alpha_r)
    B = float(alpha_s1) + float(alpha_s2)
    k = abs(float(k))

    A = 1.0 / (4.0 * a)
    C = 1.0 / (4.0 * B)

    K = (
        4.0
        * math.pi
        / (2.0 * math.pi) ** 3
        * (math.pi / a) ** 1.5
        * (math.pi / B) ** 1.5
    )

    def integrand(t: float) -> float:
        D = C + t
        S = A + D
        exp_part = math.exp(-A * D * k * k / S)
        bracket = 1.0 / (2.0 * S) - (k * k * A * D) / (S * S)
        return math.pi ** 1.5 * S ** (-1.5) * exp_part * bracket

    integral, err = quad(
        integrand,
        0.0,
        math.inf,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )

    return K * integral / (4.0 * a * B)


def analytic_px_s_px_full(
    alpha_r: float,
    alpha_s1: float,
    alpha_s2: float,
    k: float,
    coef_r: float = 1.0,
    coef_s1: float = 1.0,
    coef_s2: float = 1.0,
    normalized: bool = False,
    epsabs: float = 1e-12,
    epsrel: float = 1e-12,
    limit: int = 300,
) -> float:
    """
    Analytic p_x/(s p_x) result including coefficients and optional primitive
    normalization factors.
    """
    value = analytic_px_s_px_unnormalized(
        alpha_r=alpha_r,
        alpha_s1=alpha_s1,
        alpha_s2=alpha_s2,
        k=k,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )

    factor = float(coef_r) * float(coef_s1) * float(coef_s2)

    if normalized:
        factor *= gaussian_norm_cartesian(alpha_r, (1, 0, 0))
        factor *= gaussian_norm_cartesian(alpha_s1, (0, 0, 0))
        factor *= gaussian_norm_cartesian(alpha_s2, (1, 0, 0))

    return factor * value


# =============================================================================
# Project numerical path
# =============================================================================

def project_px_s_px_value(
    alpha_r: float,
    alpha_s1: float,
    alpha_s2: float,
    coef_r: float,
    coef_s1: float,
    coef_s2: float,
    normalized: bool,
    plane_wave,
    lmax_pw: int,
    epsabs: float,
    epsrel: float,
    limit: int,
):
    """
    Evaluate p_x/(s p_x) with the project's density path.
    """
    phi_r = ContractedGaussian.single(
        alpha=alpha_r,
        powers=(1, 0, 0),
        coefficient=coef_r,
        normalized=normalized,
        label="phi_r_px",
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
        powers=(1, 0, 0),
        coefficient=coef_s2,
        normalized=normalized,
        label="phi_s2_px",
    )

    density, density_report = build_contracted_density(phi_s1, phi_s2)

    result = full_coulomb_plane_wave_integral_density(
        phi_r=phi_r,
        density=density,
        plane_wave=plane_wave,
        lmax_pw=lmax_pw,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        use_radial_table=True,
        precompute_radial=False,
        print_radial_report=False,
        print_hard_keys=0,
        max_contributions_store=10,
    )

    return result, density_report


def print_angular_summary(lmax_pw: int) -> tuple[int, ...]:
    """
    Print allowed lp values for p_x/(s p_x).
    """
    _r_channels, _s_channels, _pw_channels, couplings = angular_pipeline(
        powers_r=(1, 0, 0),
        powers_s1=(0, 0, 0),
        powers_s2=(1, 0, 0),
        lmax_pw=lmax_pw,
    )
    allowed_lp = tuple(sorted({c.pw_channel.lp for c in couplings}))
    print(f"allowed lp at lmax_pw={lmax_pw}: {allowed_lp}")
    return allowed_lp


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analytic p_x/(s p_x) validation.")

    add_plane_wave_cli_arguments(parser)

    parser.add_argument("--alpha-r", type=float, default=1.0)
    parser.add_argument("--alpha-s1", type=float, default=1.0)
    parser.add_argument("--alpha-s2", type=float, default=1.0)

    parser.add_argument("--coef-r", type=float, default=1.0)
    parser.add_argument("--coef-s1", type=float, default=1.0)
    parser.add_argument("--coef-s2", type=float, default=1.0)

    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--lmax-pw", type=int, default=2)
    parser.add_argument("--epsabs", type=float, default=1e-10)
    parser.add_argument("--epsrel", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="do not fail when lmax_pw < 2; print the partial value instead",
    )

    args = parser.parse_args()

    plane_wave = plane_wave_from_cli_args(args)

    print("\n=== Analytic p_x / (s p_x) validation setup ===")
    print(plane_wave.describe())
    print(f"lmax_pw = {args.lmax_pw}")
    print(f"alpha_r  = {args.alpha_r}")
    print(f"alpha_s1 = {args.alpha_s1}")
    print(f"alpha_s2 = {args.alpha_s2}")
    print(f"alpha_density = {args.alpha_s1 + args.alpha_s2}")
    print(f"normalized = {args.normalized}")

    allowed_lp = print_angular_summary(args.lmax_pw)

    if args.lmax_pw < 2:
        print("\nWARNING:")
        print("  For p_x/(s p_x), the exact full plane-wave result needs lp=0 and lp=2.")
        print("  lmax_pw < 2 gives only a partial angular expansion.")
        print("  Use --lmax-pw 2 for analytic full comparison.")

    analytic = analytic_px_s_px_full(
        alpha_r=args.alpha_r,
        alpha_s1=args.alpha_s1,
        alpha_s2=args.alpha_s2,
        k=plane_wave.k_abs,
        coef_r=args.coef_r,
        coef_s1=args.coef_s1,
        coef_s2=args.coef_s2,
        normalized=args.normalized,
        epsabs=args.epsabs * 0.01,
        epsrel=args.epsrel * 0.01,
        limit=args.limit,
    )

    numerical, density_report = project_px_s_px_value(
        alpha_r=args.alpha_r,
        alpha_s1=args.alpha_s1,
        alpha_s2=args.alpha_s2,
        coef_r=args.coef_r,
        coef_s1=args.coef_s1,
        coef_s2=args.coef_s2,
        normalized=args.normalized,
        plane_wave=plane_wave,
        lmax_pw=args.lmax_pw,
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        limit=args.limit,
    )

    print("\n=== Density contraction ===")
    density_report.print()

    print("\n=== Results ===")
    print(f"analytic full = {analytic:.16e}")
    print(f"project       = {numerical.value.real:.16e} + {numerical.value.imag:.16e} i")

    abs_err = abs(numerical.value - analytic)
    rel_err = abs_err / max(1.0, abs(analytic))

    print(f"abs error     = {abs_err:.6e}")
    print(f"rel error     = {rel_err:.6e}")
    print(f"tol           = {args.tol:.6e}")

    if args.lmax_pw < 2 and args.allow_partial:
        print("status        = PARTIAL angular result; comparison to full analytic value not enforced")
        return

    if args.lmax_pw < 2:
        print("status        = FAIL because lmax_pw < 2 is partial for this case")
        raise SystemExit(1)

    if abs_err <= args.tol or rel_err <= args.tol:
        print("status        = PASS")
    else:
        print("status        = FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
