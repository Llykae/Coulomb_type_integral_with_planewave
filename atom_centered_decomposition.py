"""
atom_centered_decomposition.py

Bridge module between Cartesian atom-centered Gaussians and spherical/solid
harmonic angular channels.

Why this module?
----------------
Our Gaussian primitives are Cartesian:

    x^a y^b z^c exp(-alpha r^2).

But the atom-centered Coulomb machinery wants angular channels expressed in
spherical harmonics / solid harmonics, because the Coulomb kernel and plane wave
naturally couple angular momenta through Gaunt coefficients.

This module begins that bridge.

Core idea
---------
For total Cartesian degree

    n = a + b + c,

the monomial

    x^a y^b z^c

can be decomposed into regular solid harmonics multiplied by radial powers:

    x^a y^b z^c = sum_{lambda,m,q} A_{lambda,m,q} r^{2q} S_{lambda,m}(x,y,z)

where

    lambda + 2q = n.

For example:

    z = sqrt(4*pi/3) S_1,0

and

    z^2 = (1/3) r^2 + angular d-part.

Because the low-l solid harmonics in harmonics.py are validated, this module
uses them as the angular basis and solves for the coefficients algebraically.

Current scope
-------------
This first version supports total Cartesian degree n <= 4, because harmonics.py
currently contains S_lm up to l=4.

The coefficients may be complex, because we are using complex spherical
harmonics. Later we can add real tesseral harmonics if desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from harmonics import x, y, z, solid_harmonic_polynomial


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class SolidHarmonicComponent:
    """
    One component in the decomposition of a Cartesian monomial.

    Represents

        coefficient * r2_power * S_lm(x,y,z)

    where

        r2_power = q

    means a factor

        (x^2 + y^2 + z^2)^q.

    Since S_lm has degree l, the total degree is

        l + 2*q.
    """

    l: int
    m: int
    r2_power: int
    coefficient: sp.Expr

    def expression(self) -> sp.Expr:
        """Return this component as a SymPy expression."""
        r2 = x**2 + y**2 + z**2
        return sp.expand(self.coefficient * (r2 ** self.r2_power) * solid_harmonic_polynomial(self.l, self.m))


@dataclass(frozen=True)
class CartesianDecomposition:
    """
    Decomposition of one Cartesian monomial into solid-harmonic components.
    """

    powers: tuple[int, int, int]
    components: tuple[SolidHarmonicComponent, ...]

    def original_expression(self) -> sp.Expr:
        """Return x^a y^b z^c."""
        a, b, c = self.powers
        return x**a * y**b * z**c

    def reconstructed_expression(self) -> sp.Expr:
        """Return the sum of all solid-harmonic components."""
        return sp.expand(sum(component.expression() for component in self.components))

    def residual(self) -> sp.Expr:
        """Return original - reconstructed, simplified."""
        return sp.simplify(self.original_expression() - self.reconstructed_expression())

    def print(self) -> None:
        """Pretty-print the decomposition."""
        print("\n=== Cartesian monomial decomposition ===")
        print(f"powers = {self.powers}")
        print(f"monomial = {sp.sstr(self.original_expression())}")
        print("components:")
        for comp in self.components:
            print(
                f"  coeff={sp.sstr(comp.coefficient)}  "
                f"r2_power={comp.r2_power}  "
                f"S_{{{comp.l},{comp.m}}}"
            )
        print(f"residual = {sp.sstr(self.residual())}")


# =============================================================================
# Basis construction
# =============================================================================

def validate_powers(powers: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate Cartesian powers."""
    if len(powers) != 3:
        raise ValueError("powers must have length 3")
    if any(p < 0 for p in powers):
        raise ValueError("powers must be nonnegative")
    return tuple(int(p) for p in powers)


def allowed_l_values_for_degree(n: int) -> list[int]:
    """
    Allowed solid-harmonic angular momenta for total polynomial degree n.

    Since each factor r^2 lowers angular degree by 2, possible l values are

        l = n, n-2, n-4, ..., 0 or 1.

    Examples
    --------
    n=0 -> [0]
    n=1 -> [1]
    n=2 -> [2,0]
    n=3 -> [3,1]
    n=4 -> [4,2,0]
    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    return list(range(n, -1, -2))


def solid_harmonic_basis_for_degree(n: int) -> list[SolidHarmonicComponent]:
    """
    Build basis components r^(2q) S_lm for homogeneous degree n.

    The basis contains all pairs satisfying

        l + 2q = n.
    """
    if n > 4:
        raise ValueError("Current low-l harmonic table supports only degree <= 4")

    basis: list[SolidHarmonicComponent] = []
    for l in allowed_l_values_for_degree(n):
        q = (n - l) // 2
        for m in range(-l, l + 1):
            basis.append(
                SolidHarmonicComponent(
                    l=l,
                    m=m,
                    r2_power=q,
                    coefficient=sp.Integer(1),
                )
            )
    return basis


def homogeneous_monomials_of_degree(n: int) -> list[tuple[int, int, int]]:
    """
    Return all Cartesian monomial powers with total degree n.

    Example
    -------
    n=2 gives

        (2,0,0), (1,1,0), (1,0,1), ...
    """
    powers = []
    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - a - b
            powers.append((a, b, c))
    return powers


# =============================================================================
# Decomposition solver
# =============================================================================
@lru_cache(None)
def decompose_cartesian_monomial(powers: tuple[int, int, int]) -> CartesianDecomposition:
    """
    Decompose x^a y^b z^c into r^(2q) S_lm components.

    This version uses SymPy Poly dictionaries instead of expr.coeff(mon),
    which correctly handles the constant monomial powers=(0,0,0).
    """
    powers = validate_powers(powers)
    n = sum(powers)

    if n > 4:
        raise ValueError("This first decomposition module supports only total degree <= 4")

    target = x**powers[0] * y**powers[1] * z**powers[2]
    basis = solid_harmonic_basis_for_degree(n)

    coeff_symbols = sp.symbols(f"c0:{len(basis)}")

    expr = sp.expand(
        sum(
            csym * basis_component.expression()
            for csym, basis_component in zip(coeff_symbols, basis)
        )
    )

    # Robust coefficient matching, including the constant term.
    target_poly = sp.Poly(target, x, y, z)
    expr_poly = sp.Poly(expr, x, y, z)

    target_dict = target_poly.as_dict()
    expr_dict = expr_poly.as_dict()

    all_keys = sorted(set(target_dict.keys()) | set(expr_dict.keys()))

    equations = []
    for key in all_keys:
        equations.append(
            sp.Eq(
                expr_dict.get(key, sp.Integer(0)),
                target_dict.get(key, sp.Integer(0)),
            )
        )

    solution = sp.solve(equations, coeff_symbols, dict=True, simplify=True)

    if not solution:
        raise RuntimeError(f"Could not decompose monomial powers={powers}")

    sol = solution[0]

    components: list[SolidHarmonicComponent] = []

    for csym, basis_component in zip(coeff_symbols, basis):
        coeff = sp.simplify(sol.get(csym, 0))

        if coeff != 0:
            components.append(
                SolidHarmonicComponent(
                    l=basis_component.l,
                    m=basis_component.m,
                    r2_power=basis_component.r2_power,
                    coefficient=coeff,
                )
            )

    result = CartesianDecomposition(
        powers=powers,
        components=tuple(components),
    )

    residual = result.residual()

    if residual != 0:
        raise RuntimeError(
            f"Nonzero decomposition residual for powers={powers}: {residual}"
        )

    return result

# =============================================================================
# Bulk helpers
# =============================================================================

def decompose_all_monomials_up_to_degree(nmax: int = 4) -> list[CartesianDecomposition]:
    """
    Decompose all Cartesian monomials up to degree nmax.
    """
    if nmax > 4:
        raise ValueError("Current decomposition supports only nmax <= 4")

    results = []
    for n in range(nmax + 1):
        for powers in homogeneous_monomials_of_degree(n):
            results.append(decompose_cartesian_monomial(powers))
    return results


def print_all_decompositions(nmax: int = 2) -> None:
    """
    Print decompositions for all monomials up to nmax.
    """
    for decomp in decompose_all_monomials_up_to_degree(nmax):
        decomp.print()


# =============================================================================
# Command-line demo
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Decompose Cartesian monomials into solid-harmonic components."
    )
    parser.add_argument("--powers", nargs=3, type=int, default=(0, 0, 2), metavar=("A", "B", "C"))
    parser.add_argument("--all", action="store_true", help="print all decompositions up to --nmax")
    parser.add_argument("--nmax", type=int, default=2, help="maximum degree for --all")

    args = parser.parse_args()

    if args.all:
        print_all_decompositions(args.nmax)
    else:
        decomp = decompose_cartesian_monomial(tuple(args.powers))
        decomp.print()


if __name__ == "__main__":
    main()
