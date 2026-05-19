"""
plane_wave_lmax_scan.py

Scan convergence with respect to the maximum plane-wave angular momentum lmax_pw.

This script answers the practical question:

    How many plane-wave angular channels l_p do I need?

It prints, for each lmax_pw:

    lmax_pw
    allowed lp values included by the angular pipeline
    number of angular couplings
    Re(I), Im(I)
    |delta from previous lmax|

Important interpretation
------------------------
For the current atom-centered Cartesian Gaussian problem, the set of exactly
allowed l_p values is determined by angular momentum selection rules. It does
not grow with k for a fixed polynomial angular case.

The value of k changes the radial weights through spherical Bessel functions
j_l(k r), but it does not create new allowed l_p channels in the atom-centered
finite-polynomial setting.

Examples
--------

s / (p_x s): exact at lmax_pw = 1

    python3 plane_wave_lmax_scan.py \
      --powers-r 0 0 0 \
      --powers-s1 1 0 0 \
      --powers-s2 0 0 0 \
      --energy 2.0 \
      --direction 1 0 0 \
      --lmax-max 8

p_x / (s p_x): exact at lmax_pw = 2

    python3 plane_wave_lmax_scan.py \
      --powers-r 1 0 0 \
      --powers-s1 0 0 0 \
      --powers-s2 1 0 0 \
      --energy 2.0 \
      --direction 1 0 0 \
      --lmax-max 8

High-k s / (s s): still exact at lmax_pw = 0 because only lp=0 is allowed

    python3 plane_wave_lmax_scan.py \
      --powers-r 0 0 0 \
      --powers-s1 0 0 0 \
      --powers-s2 0 0 0 \
      --k 72 \
      --direction 1 0 0 \
      --lmax-max 8
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from angular_pipeline import angular_pipeline
from full_coulomb_integral import ContractedGaussian
from density_contraction import build_contracted_density
from full_coulomb_integral_density import full_coulomb_plane_wave_integral_density
from plane_wave_parameters import add_plane_wave_cli_arguments, plane_wave_from_cli_args
from parity import product_powers


Powers = tuple[int, int, int]


@dataclass(frozen=True)
class LmaxScanRow:
    lmax_pw: int
    allowed_lp: tuple[int, ...]
    n_couplings: int
    value: complex
    delta: float | None


def tuple3(values: list[int] | tuple[int, int, int], name: str) -> Powers:
    if len(values) != 3:
        raise ValueError(f"{name} must have length 3")
    out = tuple(int(v) for v in values)
    if any(v < 0 for v in out):
        raise ValueError(f"{name} must be nonnegative")
    return out  # type: ignore[return-value]


def make_single_gaussian(label: str, powers: Powers, alpha: float, coefficient: float, normalized: bool):
    return ContractedGaussian.single(
        alpha=alpha,
        powers=powers,
        coefficient=coefficient,
        normalized=normalized,
        label=label,
    )


def allowed_lp_for_lmax(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    lmax_pw: int,
) -> tuple[tuple[int, ...], int]:
    _r_channels, _s_channels, _pw_channels, couplings = angular_pipeline(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        lmax_pw=lmax_pw,
    )
    allowed_lp = tuple(sorted({c.pw_channel.lp for c in couplings}))
    return allowed_lp, len(couplings)


def evaluate_for_lmax(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
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
    use_radial_table: bool,
) -> complex:
    phi_r = make_single_gaussian(
        label="phi_r",
        powers=powers_r,
        alpha=alpha_r,
        coefficient=coef_r,
        normalized=normalized,
    )
    phi_s1 = make_single_gaussian(
        label="phi_s1",
        powers=powers_s1,
        alpha=alpha_s1,
        coefficient=coef_s1,
        normalized=normalized,
    )
    phi_s2 = make_single_gaussian(
        label="phi_s2",
        powers=powers_s2,
        alpha=alpha_s2,
        coefficient=coef_s2,
        normalized=normalized,
    )

    density, _density_report = build_contracted_density(phi_s1, phi_s2)

    result = full_coulomb_plane_wave_integral_density(
        phi_r=phi_r,
        density=density,
        plane_wave=plane_wave,
        lmax_pw=lmax_pw,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        use_radial_table=use_radial_table,
        precompute_radial=False,
        print_radial_report=False,
        print_hard_keys=0,
        max_contributions_store=0,
    )

    return result.value


def run_scan(args) -> list[LmaxScanRow]:
    powers_r = tuple3(args.powers_r, "powers_r")
    powers_s1 = tuple3(args.powers_s1, "powers_s1")
    powers_s2 = tuple3(args.powers_s2, "powers_s2")
    powers_s_total = product_powers(powers_s1, powers_s2)

    plane_wave = plane_wave_from_cli_args(args)

    print("\n=== Plane-wave lmax scan setup ===")
    print(plane_wave.describe())
    print(f"powers_r       = {powers_r}")
    print(f"powers_s1      = {powers_s1}")
    print(f"powers_s2      = {powers_s2}")
    print(f"powers_s_total = {powers_s_total}")
    print(f"alpha_r        = {args.alpha_r}")
    print(f"alpha_s1       = {args.alpha_s1}")
    print(f"alpha_s2       = {args.alpha_s2}")
    print(f"alpha_density  = {args.alpha_s1 + args.alpha_s2}")
    print(f"normalized     = {args.normalized}")
    print(f"lmax range     = {args.lmax_min} ... {args.lmax_max}")

    if args.alpha_r > 0:
        r_eff = 1.0 / math.sqrt(args.alpha_r)
        print(f"r_eff ~ 1/sqrt(alpha_r) = {r_eff:.8g}")
        print(f"k*r_eff                 = {plane_wave.k_abs * r_eff:.8g}")

    full_allowed_lp, full_n_couplings = allowed_lp_for_lmax(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        lmax_pw=args.lmax_max,
    )
    print("\n=== Angular selection at largest lmax ===")
    print(f"allowed lp up to lmax={args.lmax_max}: {full_allowed_lp}")
    print(f"number of couplings at lmax={args.lmax_max}: {full_n_couplings}")
    if full_allowed_lp:
        print(f"minimum exact lmax suggested by angular selection = {max(full_allowed_lp)}")
    else:
        print("no allowed angular couplings found")

    rows: list[LmaxScanRow] = []
    previous_value: complex | None = None

    print("\n=== lmax_pw convergence ===")
    print(
        f"{'lmax':>5}  {'allowed lp':>18}  {'n_cpl':>6}  "
        f"{'Re(I)':>22}  {'Im(I)':>22}  {'|delta prev|':>14}"
    )
    print("-" * 98)

    for lmax_pw in range(args.lmax_min, args.lmax_max + 1):
        allowed_lp, n_couplings = allowed_lp_for_lmax(
            powers_r=powers_r,
            powers_s1=powers_s1,
            powers_s2=powers_s2,
            lmax_pw=lmax_pw,
        )

        value = evaluate_for_lmax(
            powers_r=powers_r,
            powers_s1=powers_s1,
            powers_s2=powers_s2,
            alpha_r=args.alpha_r,
            alpha_s1=args.alpha_s1,
            alpha_s2=args.alpha_s2,
            coef_r=args.coef_r,
            coef_s1=args.coef_s1,
            coef_s2=args.coef_s2,
            normalized=args.normalized,
            plane_wave=plane_wave,
            lmax_pw=lmax_pw,
            epsabs=args.epsabs,
            epsrel=args.epsrel,
            limit=args.limit,
            use_radial_table=not args.no_radial_table,
        )

        if previous_value is None:
            delta = None
            delta_text = "---"
        else:
            delta = abs(value - previous_value)
            delta_text = f"{delta:.6e}"

        allowed_text = str(allowed_lp)
        print(
            f"{lmax_pw:5d}  {allowed_text:>18}  {n_couplings:6d}  "
            f"{value.real:22.12e}  {value.imag:22.12e}  {delta_text:>14}"
        )

        rows.append(
            LmaxScanRow(
                lmax_pw=lmax_pw,
                allowed_lp=allowed_lp,
                n_couplings=n_couplings,
                value=value,
                delta=delta,
            )
        )
        previous_value = value

    if rows:
        final = rows[-1]
        print("\n=== Final value at largest lmax ===")
        print(f"lmax_pw = {final.lmax_pw}")
        print(f"value   = {final.value.real:.16e} + {final.value.imag:.16e} i")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan plane-wave lmax_pw convergence for one atom-centered primitive integral."
    )

    parser.add_argument("--powers-r", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s1", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s2", nargs=3, type=int, default=(0, 0, 0))

    parser.add_argument("--alpha-r", type=float, default=1.0)
    parser.add_argument("--alpha-s1", type=float, default=1.0)
    parser.add_argument("--alpha-s2", type=float, default=1.0)

    parser.add_argument("--coef-r", type=float, default=1.0)
    parser.add_argument("--coef-s1", type=float, default=1.0)
    parser.add_argument("--coef-s2", type=float, default=1.0)
    parser.add_argument("--normalized", action="store_true")

    add_plane_wave_cli_arguments(parser)

    parser.add_argument("--lmax-min", type=int, default=0)
    parser.add_argument("--lmax-max", type=int, default=8)

    parser.add_argument("--epsabs", type=float, default=1e-10)
    parser.add_argument("--epsrel", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--no-radial-table", action="store_true")

    args = parser.parse_args()

    if args.lmax_min < 0 or args.lmax_max < args.lmax_min:
        raise ValueError("Require 0 <= lmax_min <= lmax_max")

    run_scan(args)


if __name__ == "__main__":
    main()
