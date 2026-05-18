"""
gaunt.py

Angular Gaunt coefficients for the atom-centered Coulomb / plane-wave project.

Why this module?
----------------
Once all orbitals are atom-centered, we can separate many integrals into angular
and radial parts. The angular parts involve integrals of three spherical
harmonics, called Gaunt coefficients.

The basic complex Gaunt coefficient is

    G(l1,m1; l2,m2; l3,m3)
      = ∫ dΩ Y_l1,m1(Ω) Y_l2,m2(Ω) Y_l3,m3(Ω).

This module computes those coefficients using Wigner 3j symbols.

Formula
-------
For complex spherical harmonics with the usual Condon-Shortley convention,

    ∫ Y_l1,m1 Y_l2,m2 Y_l3,m3 dΩ

is

    sqrt((2l1+1)(2l2+1)(2l3+1)/(4π))
    * Wigner3j(l1,l2,l3; 0,0,0)
    * Wigner3j(l1,l2,l3; m1,m2,m3).

Important selection rules
-------------------------
The coefficient is zero unless:

1. m1 + m2 + m3 = 0
2. |l1-l2| <= l3 <= l1+l2
3. l1+l2+l3 is even, due to the (0,0,0) Wigner 3j factor

These rules are extremely useful for pruning angular sums.

Complex conjugates
------------------
Many physical formulas involve conjugated spherical harmonics. We use

    Y_lm*(Ω) = (-1)^m Y_l,-m(Ω)

so conjugated forms can be reduced to the basic non-conjugated Gaunt integral.

Current role in project
-----------------------
This module does not yet compute the full Coulomb integral. It provides the
angular coefficients needed for the next analytic atom-centered layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp
from sympy.physics.wigner import wigner_3j


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class GauntIndex:
    """
    One spherical harmonic index pair (l,m).
    """

    l: int
    m: int

    def validate(self) -> None:
        if self.l < 0:
            raise ValueError("l must be nonnegative")
        if abs(self.m) > self.l:
            raise ValueError(f"Need |m| <= l, got l={self.l}, m={self.m}")


@dataclass(frozen=True)
class GauntReport:
    """
    Report for one Gaunt coefficient.
    """

    i1: GauntIndex
    i2: GauntIndex
    i3: GauntIndex
    allowed_by_m: bool
    allowed_by_triangle: bool
    allowed_by_parity: bool
    value: sp.Expr

    @property
    def allowed(self) -> bool:
        return self.allowed_by_m and self.allowed_by_triangle and self.allowed_by_parity

    def print(self) -> None:
        print("\n=== Gaunt coefficient report ===")
        print(f"(l1,m1) = ({self.i1.l},{self.i1.m})")
        print(f"(l2,m2) = ({self.i2.l},{self.i2.m})")
        print(f"(l3,m3) = ({self.i3.l},{self.i3.m})")
        print(f"m rule       : {self.allowed_by_m}")
        print(f"triangle rule: {self.allowed_by_triangle}")
        print(f"parity rule  : {self.allowed_by_parity}")
        print(f"allowed      : {self.allowed}")
        print(f"value        : {sp.sstr(self.value)}")
        print(f"numeric      : {complex(sp.N(self.value))}")


# =============================================================================
# Selection rules
# =============================================================================

def validate_lm(l: int, m: int) -> None:
    """Validate one (l,m) pair."""
    if l < 0:
        raise ValueError("l must be nonnegative")
    if abs(m) > l:
        raise ValueError(f"Need |m| <= l, got l={l}, m={m}")


def m_selection_rule(m1: int, m2: int, m3: int) -> bool:
    """
    Gaunt m selection rule.

    Nonzero only if

        m1 + m2 + m3 = 0.
    """
    return (m1 + m2 + m3) == 0


def triangle_rule(l1: int, l2: int, l3: int) -> bool:
    """
    Angular momentum triangle rule.

    Nonzero only if

        |l1-l2| <= l3 <= l1+l2.
    """
    return abs(l1 - l2) <= l3 <= (l1 + l2)


def parity_rule(l1: int, l2: int, l3: int) -> bool:
    """
    Parity rule from Wigner3j(l1,l2,l3;0,0,0).

    Nonzero only if

        l1 + l2 + l3

    is even.
    """
    return ((l1 + l2 + l3) % 2) == 0


def gaunt_allowed(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> bool:
    """
    Return True if the basic selection rules allow a nonzero coefficient.
    """
    validate_lm(l1, m1)
    validate_lm(l2, m2)
    validate_lm(l3, m3)
    return (
        m_selection_rule(m1, m2, m3)
        and triangle_rule(l1, l2, l3)
        and parity_rule(l1, l2, l3)
    )


# =============================================================================
# Basic Gaunt coefficient
# =============================================================================

@lru_cache(None)
def gaunt_complex(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> sp.Expr:
    """
    Compute the complex Gaunt coefficient

        ∫ Y_l1,m1 Y_l2,m2 Y_l3,m3 dΩ.

    Returns a SymPy exact expression when possible.
    """
    validate_lm(l1, m1)
    validate_lm(l2, m2)
    validate_lm(l3, m3)

    if not gaunt_allowed(l1, m1, l2, m2, l3, m3):
        return sp.Integer(0)

    prefactor = sp.sqrt(
        sp.Rational((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1), 1)
        / (4 * sp.pi)
    )

    value = (
        prefactor
        * wigner_3j(l1, l2, l3, 0, 0, 0)
        * wigner_3j(l1, l2, l3, m1, m2, m3)
    )

    return sp.simplify(value)


def gaunt_report(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> GauntReport:
    """
    Build a report for one Gaunt coefficient.
    """
    i1 = GauntIndex(l1, m1)
    i2 = GauntIndex(l2, m2)
    i3 = GauntIndex(l3, m3)
    i1.validate()
    i2.validate()
    i3.validate()

    return GauntReport(
        i1=i1,
        i2=i2,
        i3=i3,
        allowed_by_m=m_selection_rule(m1, m2, m3),
        allowed_by_triangle=triangle_rule(l1, l2, l3),
        allowed_by_parity=parity_rule(l1, l2, l3),
        value=gaunt_complex(l1, m1, l2, m2, l3, m3),
    )


# =============================================================================
# Conjugated variants
# =============================================================================

def conjugate_phase(m: int) -> int:
    """
    Phase in

        Y_lm* = (-1)^m Y_l,-m.

    For integer m, (-1)^m is always ±1.
    """
    return -1 if (m % 2) else 1


@lru_cache(None)
def gaunt_conj_first(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> sp.Expr:
    """
    Compute

        ∫ Y_l1,m1* Y_l2,m2 Y_l3,m3 dΩ.

    Uses

        Y_lm* = (-1)^m Y_l,-m.
    """
    return sp.simplify(conjugate_phase(m1) * gaunt_complex(l1, -m1, l2, m2, l3, m3))


@lru_cache(None)
def gaunt_conj_second(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> sp.Expr:
    """
    Compute

        ∫ Y_l1,m1 Y_l2,m2* Y_l3,m3 dΩ.
    """
    return sp.simplify(conjugate_phase(m2) * gaunt_complex(l1, m1, l2, -m2, l3, m3))


@lru_cache(None)
def gaunt_conj_third(l1: int, m1: int, l2: int, m2: int, l3: int, m3: int) -> sp.Expr:
    """
    Compute

        ∫ Y_l1,m1 Y_l2,m2 Y_l3,m3* dΩ.
    """
    return sp.simplify(conjugate_phase(m3) * gaunt_complex(l1, m1, l2, m2, l3, -m3))


# =============================================================================
# Lists of allowed couplings
# =============================================================================

def allowed_l3_values(l1: int, l2: int) -> list[int]:
    """
    Return l3 values allowed by triangle and parity rules for given l1,l2.
    """
    if l1 < 0 or l2 < 0:
        raise ValueError("l1 and l2 must be nonnegative")

    values = []
    for l3 in range(abs(l1 - l2), l1 + l2 + 1):
        if parity_rule(l1, l2, l3):
            values.append(l3)
    return values


def allowed_gaunt_terms(l1: int, l2: int, l3: int) -> list[tuple[int, int, int]]:
    """
    Return all (m1,m2,m3) combinations allowed for fixed l1,l2,l3.

    This applies only the m selection rule. Triangle/parity are checked first;
    if they fail, an empty list is returned.
    """
    if not triangle_rule(l1, l2, l3) or not parity_rule(l1, l2, l3):
        return []

    terms = []
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            m3 = -(m1 + m2)
            if -l3 <= m3 <= l3:
                terms.append((m1, m2, m3))
    return terms


# =============================================================================
# Numerical angular validation
# =============================================================================

def numeric_gaunt_grid(
    l1: int,
    m1: int,
    l2: int,
    m2: int,
    l3: int,
    m3: int,
    n_theta: int = 200,
    n_phi: int = 400,
) -> complex:
    """
    Crude numerical angular integration for validation.

    This computes

        ∫ Y1 Y2 Y3 dΩ

    on a tensor-product theta/phi grid.

    It is not intended for production; it is only a sanity check for low l.
    """
    from harmonics import complex_sph_harm

    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    integrand = (
        complex_sph_harm(l1, m1, TH, PH)
        * complex_sph_harm(l2, m2, TH, PH)
        * complex_sph_harm(l3, m3, TH, PH)
        * np.sin(TH)
    )

    return complex(np.sum(integrand) * dtheta * dphi)


# =============================================================================
# Command-line demo
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gaunt coefficient helper.")
    parser.add_argument("--l1", type=int, default=1)
    parser.add_argument("--m1", type=int, default=0)
    parser.add_argument("--l2", type=int, default=1)
    parser.add_argument("--m2", type=int, default=0)
    parser.add_argument("--l3", type=int, default=0)
    parser.add_argument("--m3", type=int, default=0)
    parser.add_argument("--numeric", action="store_true", help="also do crude numerical angular integration")
    parser.add_argument("--list", action="store_true", help="list allowed m terms for l1,l2,l3")
    parser.add_argument("--l3-values", action="store_true", help="list allowed l3 values for l1,l2")

    args = parser.parse_args()

    if args.l3_values:
        print(f"Allowed l3 values for l1={args.l1}, l2={args.l2}:")
        print(allowed_l3_values(args.l1, args.l2))

    if args.list:
        print(f"Allowed (m1,m2,m3) terms for l1={args.l1}, l2={args.l2}, l3={args.l3}:")
        for item in allowed_gaunt_terms(args.l1, args.l2, args.l3):
            print(item)

    report = gaunt_report(args.l1, args.m1, args.l2, args.m2, args.l3, args.m3)
    report.print()

    if args.numeric:
        numeric = numeric_gaunt_grid(args.l1, args.m1, args.l2, args.m2, args.l3, args.m3)
        exact = complex(sp.N(report.value))
        print("\n=== Crude numerical angular check ===")
        print(f"numeric = {numeric}")
        print(f"exact   = {exact}")
        print(f"abs err = {abs(numeric - exact):.3e}")


if __name__ == "__main__":
    main()
