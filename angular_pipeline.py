"""
angular_pipeline.py

Angular-coupling diagnostic for the atom-centered Coulomb / plane-wave project.

Purpose
-------
This module connects:

1. atom_centered_decomposition.py

   Decomposes Cartesian monomials into solid-harmonic channels:

       x^a y^b z^c = sum coeff * r^(2q) * S_lm(x,y,z)

2. gaunt.py

   Computes angular integrals of three spherical harmonics:

       ∫ Y_l1,m1 Y_l2,m2 Y_l3,m3 dΩ

3. the plane-wave expansion

   exp(i k.r) contributes an angular channel Y_lp,mp(rhat) on the r side.

Why the plane-wave channel matters
----------------------------------
The target integral is

    I(k) = ∫∫ exp(i k.r) phi_r(r) 1/|r-s| rho_s(s) dr ds

where

    rho_s(s) = phi_s1(s) phi_s2(s).

After angular decomposition:

    phi_r angular channel      -> Y_lr,mr(rhat)
    plane-wave channel         -> Y_lp,mp(rhat)
    Coulomb multipole channel  -> Y_L,M(rhat), Y_L,M(shat)
    s-density angular channel  -> Y_ls,ms(shat)

So the r-side angular integral is a three-harmonic Gaunt-like coupling:

    ∫ Y_lr,mr(rhat) Y_lp,mp(rhat) Y_L,M*(rhat) dΩ_r

and the s-side angular integral is a two-harmonic overlap:

    ∫ Y_ls,ms(shat) Y_L,M(shat) dΩ_s.

This module is diagnostic. It lists angular channels and allowed couplings.
It does not compute the final radial Coulomb integral.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from atom_centered_decomposition import (
    CartesianDecomposition,
    decompose_cartesian_monomial,
)
from gaunt import gaunt_conj_third
from parity import product_powers, validate_powers


Powers = tuple[int, int, int]


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class AngularChannel:
    """
    One angular channel from a Cartesian monomial decomposition.

    Represents

        coefficient * r^(2q) * S_lm(x,y,z)

    Since

        S_lm = r^l Y_lm,

    the polynomial radial power carried by this component is

        l + 2q.
    """

    side: str
    l: int
    m: int
    r2_power: int
    coefficient: sp.Expr

    @property
    def radial_power_from_polynomial(self) -> int:
        return self.l + 2 * self.r2_power

    def print(self) -> None:
        print(
            f"{self.side}: l={self.l:2d}, m={self.m:3d}, "
            f"r2_power={self.r2_power}, "
            f"radial_power={self.radial_power_from_polynomial}, "
            f"coeff={sp.sstr(self.coefficient)}"
        )


@dataclass(frozen=True)
class PlaneWaveChannel:
    """
    One angular channel from the plane-wave expansion.

    The full plane-wave coefficient also contains

        4*pi*i^lp*j_lp(kr)*Y_lp,mp*(khat),

    but this diagnostic only cares about angular momentum selection rules.
    """

    lp: int
    mp: int

    def print(self) -> None:
        print(f"plane wave: lp={self.lp:2d}, mp={self.mp:3d}")


@dataclass(frozen=True)
class AngularCoupling:
    """
    One allowed angular coupling candidate.

    r_channel
        Angular channel from phi_r(r).

    pw_channel
        Angular channel from exp(i k.r).

    s_channel
        Angular channel from rho_s(s) = phi_s1(s) phi_s2(s).

    L, M
        Coulomb multipole channel.

    r_gaunt
        r-side angular integral:

            ∫ Y_lr,mr Y_lp,mp Y_L,M* dΩ_r

    s_overlap
        s-side angular overlap:

            ∫ Y_ls,ms Y_L,M dΩ_s

        This is nonzero only if L=ls and M=-ms, up to a phase.
    """

    r_channel: AngularChannel
    pw_channel: PlaneWaveChannel
    s_channel: AngularChannel
    L: int
    M: int
    r_gaunt: sp.Expr
    s_overlap: sp.Expr

    def print(self) -> None:
        print("\n--- Allowed angular coupling ---")
        print(
            f"r channel     : l={self.r_channel.l}, m={self.r_channel.m}, "
            f"coeff={sp.sstr(self.r_channel.coefficient)}"
        )
        print(f"plane channel : lp={self.pw_channel.lp}, mp={self.pw_channel.mp}")
        print(
            f"s channel     : l={self.s_channel.l}, m={self.s_channel.m}, "
            f"coeff={sp.sstr(self.s_channel.coefficient)}"
        )
        print(f"Coulomb       : L={self.L}, M={self.M}")
        print(f"r Gaunt       : {sp.sstr(self.r_gaunt)}")
        print(f"s overlap     : {sp.sstr(self.s_overlap)}")


# =============================================================================
# Channel construction
# =============================================================================

def channels_from_decomposition(
    decomp: CartesianDecomposition,
    side: str,
) -> list[AngularChannel]:
    """
    Convert a CartesianDecomposition into AngularChannel objects.
    """
    return [
        AngularChannel(
            side=side,
            l=component.l,
            m=component.m,
            r2_power=component.r2_power,
            coefficient=component.coefficient,
        )
        for component in decomp.components
    ]


def channels_for_powers(powers: Powers, side: str) -> list[AngularChannel]:
    """
    Decompose Cartesian powers and return angular channels.
    """
    powers = validate_powers(powers, f"powers_{side}")
    decomp = decompose_cartesian_monomial(powers)
    return channels_from_decomposition(decomp, side=side)


def plane_wave_channels(lmax_pw: int) -> list[PlaneWaveChannel]:
    """
    Return plane-wave angular channels up to lp=lmax_pw.
    """
    if lmax_pw < 0:
        raise ValueError("lmax_pw must be nonnegative")

    channels: list[PlaneWaveChannel] = []

    for lp in range(lmax_pw + 1):
        for mp in range(-lp, lp + 1):
            channels.append(PlaneWaveChannel(lp=lp, mp=mp))

    return channels


def print_channels(title: str, channels) -> None:
    """
    Pretty-print channels.
    """
    print(f"\n=== {title} ===")

    if not channels:
        print("No channels.")
        return

    for channel in channels:
        channel.print()


# =============================================================================
# Angular algebra helpers
# =============================================================================

def two_harmonic_overlap_no_conjugate(
    l1: int,
    m1: int,
    l2: int,
    m2: int,
) -> sp.Expr:
    """
    Compute

        ∫ Y_l1,m1(Ω) Y_l2,m2(Ω) dΩ.

    For complex spherical harmonics,

        ∫ Y_l1,m1 Y_l2,m2 dΩ = (-1)^m1 δ_l1,l2 δ_m2,-m1.

    This is the s-side angular integral with the Coulomb channel convention used
    here.
    """
    if l1 == l2 and m2 == -m1:
        return sp.Integer(-1 if (m1 % 2) else 1)

    return sp.Integer(0)


def r_side_gaunt_with_coulomb_conjugate(
    lr: int,
    mr: int,
    lp: int,
    mp: int,
    L: int,
    M: int,
) -> sp.Expr:
    """
    Compute

        ∫ Y_lr,mr Y_lp,mp Y_L,M* dΩ.

    This is handled by gaunt_conj_third.
    """
    return sp.simplify(gaunt_conj_third(lr, mr, lp, mp, L, M))


# =============================================================================
# Coupling construction
# =============================================================================

def build_allowed_couplings(
    r_channels: list[AngularChannel],
    s_channels: list[AngularChannel],
    pw_channels: list[PlaneWaveChannel],
) -> list[AngularCoupling]:
    """
    Build all nonzero angular couplings.

    For each s-channel (ls,ms), the s-side overlap with the Coulomb channel is
    nonzero only for

        L = ls
        M = -ms.

    Then we test the r-side Gaunt coefficient

        ∫ Y_lr,mr Y_lp,mp Y_L,M* dΩ_r.

    If both are nonzero, we keep the coupling.
    """
    couplings: list[AngularCoupling] = []

    for rch in r_channels:
        for sch in s_channels:
            L = sch.l
            M = -sch.m

            s_overlap = two_harmonic_overlap_no_conjugate(
                sch.l,
                sch.m,
                L,
                M,
            )

            if s_overlap == 0:
                continue

            for pw in pw_channels:
                r_gaunt = r_side_gaunt_with_coulomb_conjugate(
                    lr=rch.l,
                    mr=rch.m,
                    lp=pw.lp,
                    mp=pw.mp,
                    L=L,
                    M=M,
                )

                if r_gaunt != 0:
                    couplings.append(
                        AngularCoupling(
                            r_channel=rch,
                            pw_channel=pw,
                            s_channel=sch,
                            L=L,
                            M=M,
                            r_gaunt=sp.simplify(r_gaunt),
                            s_overlap=sp.simplify(s_overlap),
                        )
                    )

    return couplings


# =============================================================================
# Full diagnostic pipeline
# =============================================================================

def angular_pipeline(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    lmax_pw: int = 4,
) -> tuple[
    list[AngularChannel],
    list[AngularChannel],
    list[PlaneWaveChannel],
    list[AngularCoupling],
]:
    """
    Run the angular diagnostic pipeline.

    Returns
    -------
    r_channels, s_channels, pw_channels, couplings
    """
    powers_r = validate_powers(powers_r, "powers_r")
    powers_s1 = validate_powers(powers_s1, "powers_s1")
    powers_s2 = validate_powers(powers_s2, "powers_s2")

    powers_s_total = product_powers(powers_s1, powers_s2)

    r_channels = channels_for_powers(powers_r, side="r")
    s_channels = channels_for_powers(powers_s_total, side="s")
    pw_channels = plane_wave_channels(lmax_pw)
    couplings = build_allowed_couplings(r_channels, s_channels, pw_channels)

    return r_channels, s_channels, pw_channels, couplings


def print_allowed_couplings(
    couplings: list[AngularCoupling],
    max_print: int | None = None,
) -> None:
    """
    Print allowed couplings.
    """
    print("\n=== Allowed angular couplings ===")
    print(f"number of allowed couplings = {len(couplings)}")

    if not couplings:
        print("No allowed couplings found.")
        return

    to_print = couplings if max_print is None else couplings[:max_print]

    for coupling in to_print:
        coupling.print()

    if max_print is not None and len(couplings) > max_print:
        print(f"\n... skipped {len(couplings) - max_print} additional couplings")


# =============================================================================
# Command-line demo
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Atom-centered angular diagnostic pipeline."
    )

    parser.add_argument("--powers-r", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s1", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s2", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument(
        "--lmax-pw",
        type=int,
        default=4,
        help="maximum plane-wave angular channel",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=30,
        help="maximum couplings to print",
    )
    parser.add_argument(
        "--hide-plane",
        action="store_true",
        help="do not print plane-wave channels",
    )

    args = parser.parse_args()

    powers_r = tuple(args.powers_r)
    powers_s1 = tuple(args.powers_s1)
    powers_s2 = tuple(args.powers_s2)
    powers_s_total = product_powers(powers_s1, powers_s2)

    print("\n=== Angular pipeline setup ===")
    print(f"powers_r       = {powers_r}")
    print(f"powers_s1      = {powers_s1}")
    print(f"powers_s2      = {powers_s2}")
    print(f"powers_s_total = {powers_s_total}")
    print(f"lmax_pw        = {args.lmax_pw}")

    r_channels, s_channels, pw_channels, couplings = angular_pipeline(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        lmax_pw=args.lmax_pw,
    )

    print_channels("r-side orbital channels", r_channels)
    print_channels("s-side density channels", s_channels)

    if not args.hide_plane:
        print_channels("plane-wave angular channels", pw_channels)

    print_allowed_couplings(couplings, max_print=args.max_print)


if __name__ == "__main__":
    main()