"""
atom_centered_convergence.py

Convergence driver for the atom-centered angular/radial evaluator.

Purpose
-------
This script scans the maximum plane-wave angular momentum

    lmax_pw

for the atom-centered Coulomb / plane-wave integral

    I(k) = ∫∫ exp(i k.r) phi_r(r) 1/|r-s| phi_s1(s) phi_s2(s) dr ds.

Plane-wave input
----------------
The preferred interface is now energy + direction:

    --energy 2.0 --direction 1.0 0.0 0.0

Atomic units are used:

    E = k^2 / 2
    k = sqrt(2E)

You may also use:

    --energy-ev 100.0

or old-style direct magnitude:

    --k 2.0

The direction vector is real-valued and normalized internally. The integral code
keeps the k-direction dependence in Y_lm*(khat), rather than expanding into
kx, ky, kz polynomials.
"""

from __future__ import annotations

import argparse

import numpy as np

from atom_centered_evaluator import evaluate_atom_centered_terms
from plane_wave_parameters import add_plane_wave_cli_arguments, plane_wave_from_cli_args

try:
    from parity import parity_report
except ImportError:
    parity_report = None


Powers = tuple[int, int, int]


# =============================================================================
# Helpers
# =============================================================================

def parse_lmax_values(text: str) -> list[int]:
    """
    Parse comma-separated lmax values.

    Examples
    --------
    "0,1,2,4,6" -> [0,1,2,4,6]
    "0:8"       -> [0,1,2,3,4,5,6,7,8]
    "0:12:2"    -> [0,2,4,6,8,10,12]
    """
    text = text.strip()

    if ":" in text:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError("range format must be start:stop or start:stop:step")
        return list(range(start, stop + 1, step))

    return [int(part.strip()) for part in text.split(",") if part.strip()]


def effective_radius_estimate(alpha: float) -> float:
    """
    Very rough Gaussian length scale estimate.

    For exp(-alpha r^2), a typical radial size is about 1/sqrt(alpha).
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return 1.0 / np.sqrt(alpha)


def print_setup(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    alpha_r: float,
    alpha_s1: float,
    alpha_s2: float,
    kvec: np.ndarray,
    lmax_values: list[int],
) -> None:
    """Print the convergence setup."""
    k_abs = float(np.linalg.norm(kvec))
    alpha_s_total = alpha_s1 + alpha_s2

    r_eff = effective_radius_estimate(alpha_r)
    s_eff = effective_radius_estimate(alpha_s_total)

    print("\n=== Atom-centered convergence setup ===")
    print(f"powers_r       = {powers_r}")
    print(f"powers_s1      = {powers_s1}")
    print(f"powers_s2      = {powers_s2}")
    print(f"alpha_r        = {alpha_r}")
    print(f"alpha_s1       = {alpha_s1}")
    print(f"alpha_s2       = {alpha_s2}")
    print(f"alpha_s_total  = {alpha_s_total}")
    print(f"kvec           = {kvec}")
    print(f"|k|            = {k_abs:.8g}")
    print(f"r_eff ~ 1/sqrt(alpha_r)       = {r_eff:.8g}")
    print(f"s_eff ~ 1/sqrt(alpha_s_total) = {s_eff:.8g}")
    print(f"k*r_eff        = {k_abs * r_eff:.8g}")
    print(f"lmax values    = {lmax_values}")

    if parity_report is not None:
        parity_report(
            powers_r=powers_r,
            powers_s1=powers_s1,
            powers_s2=powers_s2,
            kvec=kvec,
        ).print()


# =============================================================================
# Convergence calculation
# =============================================================================

def convergence_scan(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    alpha_r: float,
    alpha_s1: float,
    alpha_s2: float,
    kvec: np.ndarray,
    lmax_values: list[int],
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 300,
) -> list[tuple[int, complex, float | None]]:
    """
    Evaluate the atom-centered integral for each lmax value.

    Returns
    -------
    list of (lmax, value, delta_from_previous)
    """
    rows: list[tuple[int, complex, float | None]] = []
    previous: complex | None = None

    print("\n=== lmax_pw convergence ===")
    print(" lmax              Re(I)                  Im(I)             |delta prev|")
    print("--------------------------------------------------------------------------------")

    for lmax in lmax_values:
        result = evaluate_atom_centered_terms(
            powers_r=powers_r,
            powers_s1=powers_s1,
            powers_s2=powers_s2,
            alpha_r=alpha_r,
            alpha_s1=alpha_s1,
            alpha_s2=alpha_s2,
            kvec=kvec,
            lmax_pw=lmax,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )

        value = result.value
        delta = None if previous is None else abs(value - previous)
        rows.append((lmax, value, delta))

        delta_text = "---" if delta is None else f"{delta:14.6e}"
        print(f" {lmax:4d}   {value.real:20.12e}   {value.imag:20.12e}   {delta_text}")

        previous = value

    return rows


# =============================================================================
# Command-line interface
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan plane-wave angular momentum convergence for atom-centered evaluator."
    )

    parser.add_argument("--powers-r", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s1", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s2", nargs=3, type=int, default=(0, 0, 0))

    parser.add_argument("--alpha-r", type=float, default=1.0)
    parser.add_argument("--alpha-s1", type=float, default=1.0)
    parser.add_argument("--alpha-s2", type=float, default=1.0)

    add_plane_wave_cli_arguments(parser)

    parser.add_argument(
        "--lmax-values",
        type=str,
        default="0:8",
        help="comma list or range, e.g. '0,1,2,4,6' or '0:12:2'",
    )

    parser.add_argument("--epsabs", type=float, default=1e-10)
    parser.add_argument("--epsrel", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=300)

    args = parser.parse_args()

    powers_r = tuple(args.powers_r)
    powers_s1 = tuple(args.powers_s1)
    powers_s2 = tuple(args.powers_s2)

    plane_wave = plane_wave_from_cli_args(args)
    kvec = plane_wave.kvec

    lmax_values = parse_lmax_values(args.lmax_values)

    print("\n" + plane_wave.describe())

    print_setup(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        alpha_r=args.alpha_r,
        alpha_s1=args.alpha_s1,
        alpha_s2=args.alpha_s2,
        kvec=kvec,
        lmax_values=lmax_values,
    )

    convergence_scan(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        alpha_r=args.alpha_r,
        alpha_s1=args.alpha_s1,
        alpha_s2=args.alpha_s2,
        kvec=kvec,
        lmax_values=lmax_values,
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
