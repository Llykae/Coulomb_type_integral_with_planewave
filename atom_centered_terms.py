"""
atom_centered_terms.py

Symbolic term builder for the atom-centered Coulomb / plane-wave integral.

Goal
----
This module turns the angular-coupling diagnostics from angular_pipeline.py into
structured symbolic terms for the atom-centered two-electron integral

    I(k) = ∫∫ d^3r d^3s
           exp(i k.r)
           phi_r(r)
           1/|r-s|
           phi_s1(s) phi_s2(s).

We are not yet evaluating the final analytic radial integral here. Instead, each
term contains a symbolic placeholder for the radial part.

Why this module?
----------------
The previous module, angular_pipeline.py, tells us which angular couplings are
allowed. This module adds the remaining symbolic prefactors:

1. Cartesian-to-solid-harmonic coefficients from phi_r and rho_s.
2. Plane-wave angular coefficient:

       4*pi*i^lp * Y_lp,mp*(khat)

3. Coulomb multipole coefficient:

       4*pi / (2L + 1)

4. Angular integrals:

       r-side Gaunt coefficient
       s-side two-harmonic overlap

5. A symbolic radial-integral placeholder.

The result is a list of AtomCenteredTerm objects.

Important
---------
This is still a symbolic bookkeeping module. The next step will be to define and
evaluate the radial integral associated with each term.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from angular_pipeline import (
    AngularCoupling,
    angular_pipeline,
)
from parity import product_powers, validate_powers


Powers = tuple[int, int, int]


# =============================================================================
# Symbol helpers
# =============================================================================

def ylm_k_conj_symbol(l: int, m: int) -> sp.Symbol:
    """
    Symbolic placeholder for Y_lm*(khat).
    """
    return sp.Symbol(f"Y_{l}_{m}_k_conj")


def radial_integral_symbol(
    lr: int,
    mr: int,
    lp: int,
    mp: int,
    ls: int,
    ms: int,
    L: int,
    powers_r: Powers,
    powers_s_total: Powers,
) -> sp.Symbol:
    """
    Create a readable symbolic placeholder for the radial integral.

    The exact radial integral will depend on:

    - r-side Gaussian exponent alpha_r,
    - s-side combined Gaussian exponent alpha_s1 + alpha_s2,
    - plane-wave k,
    - radial powers from Cartesian decomposition,
    - Coulomb multipole rank L,
    - plane-wave angular momentum lp.

    For now we encode the angular and power metadata in the symbol name.
    """
    ar, br, cr = powers_r
    as_, bs, cs = powers_s_total

    name = (
        f"R_lr{lr}_mr{mr}_lp{lp}_mp{mp}_"
        f"ls{ls}_ms{ms}_L{L}_"
        f"pr{ar}{br}{cr}_ps{as_}{bs}{cs}"
    )

    # Replace minus signs to keep the symbol name readable.
    name = name.replace("-", "m")
    return sp.Symbol(name)


# =============================================================================
# Data structure
# =============================================================================

@dataclass(frozen=True)
class AtomCenteredTerm:
    """
    One symbolic contribution to the atom-centered integral.

    Fields
    ------
    coupling
        The angular coupling object from angular_pipeline.py.

    powers_r
        Cartesian powers of phi_r.

    powers_s_total
        Cartesian powers of rho_s = phi_s1 * phi_s2.

    plane_prefactor
        4*pi*i^lp * Y_lp,mp*(khat)

    coulomb_prefactor
        4*pi / (2L + 1)

    angular_prefactor
        Product of:

            r-channel coefficient
            s-channel coefficient
            r-side Gaunt coefficient
            s-side overlap

    radial_symbol
        Placeholder for the radial integral.

    full_symbolic
        Product of all prefactors and the radial placeholder.
    """

    coupling: AngularCoupling
    powers_r: Powers
    powers_s_total: Powers
    plane_prefactor: sp.Expr
    coulomb_prefactor: sp.Expr
    angular_prefactor: sp.Expr
    radial_symbol: sp.Symbol
    full_symbolic: sp.Expr

    def print(self) -> None:
        c = self.coupling
        print("\n--- Atom-centered symbolic term ---")
        print(
            f"r channel     : lr={c.r_channel.l}, mr={c.r_channel.m}, "
            f"coeff={sp.sstr(c.r_channel.coefficient)}"
        )
        print(f"plane channel : lp={c.pw_channel.lp}, mp={c.pw_channel.mp}")
        print(
            f"s channel     : ls={c.s_channel.l}, ms={c.s_channel.m}, "
            f"coeff={sp.sstr(c.s_channel.coefficient)}"
        )
        print(f"Coulomb       : L={c.L}, M={c.M}")
        print(f"plane pref    : {sp.sstr(self.plane_prefactor)}")
        print(f"Coulomb pref  : {sp.sstr(self.coulomb_prefactor)}")
        print(f"angular pref  : {sp.sstr(self.angular_prefactor)}")
        print(f"radial symbol : {self.radial_symbol}")
        print(f"full term     : {sp.sstr(self.full_symbolic)}")


# =============================================================================
# Term construction
# =============================================================================

def build_atom_centered_terms(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    lmax_pw: int = 4,
) -> list[AtomCenteredTerm]:
    """
    Build symbolic atom-centered integral terms.

    Parameters
    ----------
    powers_r
        Cartesian powers of phi_r(r).

    powers_s1, powers_s2
        Cartesian powers of the two s-side Gaussian factors.

    lmax_pw
        Maximum plane-wave angular momentum channel to include.

    Returns
    -------
    list[AtomCenteredTerm]
    """
    powers_r = validate_powers(powers_r, "powers_r")
    powers_s1 = validate_powers(powers_s1, "powers_s1")
    powers_s2 = validate_powers(powers_s2, "powers_s2")
    powers_s_total = product_powers(powers_s1, powers_s2)

    _, _, _, couplings = angular_pipeline(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        lmax_pw=lmax_pw,
    )

    terms: list[AtomCenteredTerm] = []

    for coupling in couplings:
        lp = coupling.pw_channel.lp
        mp = coupling.pw_channel.mp
        L = coupling.L

        plane_prefactor = 4 * sp.pi * (sp.I ** lp) * ylm_k_conj_symbol(lp, mp)
        coulomb_prefactor = sp.Rational(4, 1) * sp.pi / (2 * L + 1)

        angular_prefactor = sp.simplify(
            coupling.r_channel.coefficient
            * coupling.s_channel.coefficient
            * coupling.r_gaunt
            * coupling.s_overlap
        )

        radial_symbol = radial_integral_symbol(
            lr=coupling.r_channel.l,
            mr=coupling.r_channel.m,
            lp=lp,
            mp=mp,
            ls=coupling.s_channel.l,
            ms=coupling.s_channel.m,
            L=L,
            powers_r=powers_r,
            powers_s_total=powers_s_total,
        )

        full = sp.simplify(
            plane_prefactor
            * coulomb_prefactor
            * angular_prefactor
            * radial_symbol
        )

        terms.append(
            AtomCenteredTerm(
                coupling=coupling,
                powers_r=powers_r,
                powers_s_total=powers_s_total,
                plane_prefactor=sp.simplify(plane_prefactor),
                coulomb_prefactor=sp.simplify(coulomb_prefactor),
                angular_prefactor=sp.simplify(angular_prefactor),
                radial_symbol=radial_symbol,
                full_symbolic=sp.simplify(full),
            )
        )

    return terms


def symbolic_atom_centered_expression(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    lmax_pw: int = 4,
) -> sp.Expr:
    """
    Sum all symbolic atom-centered terms.
    """
    terms = build_atom_centered_terms(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        lmax_pw=lmax_pw,
    )
    return sp.simplify(sum(term.full_symbolic for term in terms))


def print_atom_centered_terms(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    lmax_pw: int = 4,
    max_print: int | None = 30,
) -> None:
    """
    Print symbolic atom-centered terms.
    """
    terms = build_atom_centered_terms(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        lmax_pw=lmax_pw,
    )

    print("\n=== Atom-centered symbolic terms ===")
    print(f"number of terms = {len(terms)}")

    to_print = terms if max_print is None else terms[:max_print]
    for term in to_print:
        term.print()

    if max_print is not None and len(terms) > max_print:
        print(f"\n... skipped {len(terms) - max_print} additional terms")

    print("\n=== Summed symbolic expression ===")
    print(sp.sstr(sum(term.full_symbolic for term in terms)))


# =============================================================================
# Command-line demo
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build symbolic atom-centered Coulomb / plane-wave terms."
    )
    parser.add_argument("--powers-r", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s1", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s2", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--lmax-pw", type=int, default=4)
    parser.add_argument("--max-print", type=int, default=30)

    args = parser.parse_args()

    powers_r = tuple(args.powers_r)
    powers_s1 = tuple(args.powers_s1)
    powers_s2 = tuple(args.powers_s2)
    powers_s_total = product_powers(powers_s1, powers_s2)

    print("\n=== Atom-centered term setup ===")
    print(f"powers_r       = {powers_r}")
    print(f"powers_s1      = {powers_s1}")
    print(f"powers_s2      = {powers_s2}")
    print(f"powers_s_total = {powers_s_total}")
    print(f"lmax_pw        = {args.lmax_pw}")

    print_atom_centered_terms(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        lmax_pw=args.lmax_pw,
        max_print=args.max_print,
    )


if __name__ == "__main__":
    main()
