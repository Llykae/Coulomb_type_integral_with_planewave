"""
basis.py

AO/MO bookkeeping layer for the atom-centered Coulomb / plane-wave project.

Purpose
-------
Until now, the main integral script has used manual orbital definitions such as

    --powers-r 1 0 0
    --alpha-r 1.0
    --coef-r 1.0

That is useful for testing, but the long-term goal is to define orbitals from
AO and MO basis data.

This module introduces:

1. Primitive Gaussian data
2. Contracted AO data
3. AO basis containers
4. MO coefficient containers
5. Helpers to build MO orbitals as linear combinations of AOs

Atom-centered assumption
------------------------
For the current project stage, all AOs are centered at the origin. Therefore an
AO has the form

    chi_mu(r) = sum_p d_{mu,p} N_{mu,p}
                x^a y^b z^c exp(-alpha_{mu,p} r^2)

where all primitives in a contracted AO share the same Cartesian powers.

MO definition
-------------
A molecular orbital is represented as

    psi_i(r) = sum_mu C_{mu,i} chi_mu(r).

Since each AO is itself a contraction of primitives, an MO is a larger linear
combination of primitives. This module can flatten an MO into a generic
contracted orbital compatible with full_coulomb_integral.py.

Important
---------
This module is bookkeeping only. It does not evaluate integrals directly.
Integral evaluation still lives in full_coulomb_integral.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from full_coulomb_integral import AtomCenteredPrimitive, ContractedGaussian


Powers = tuple[int, int, int]


# =============================================================================
# AO primitive and contracted AO definitions
# =============================================================================

@dataclass(frozen=True)
class AOPrimitive:
    """
    One primitive inside an atom-centered contracted AO.

    Represents the radial Gaussian primitive coefficient and exponent.

    The Cartesian powers are stored at the AO level, not here, because all
    primitives in one contracted Cartesian AO normally share the same powers.
    """

    alpha: float
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("AOPrimitive alpha must be positive")

    def describe(self) -> str:
        return f"AOPrimitive(alpha={self.alpha}, coefficient={self.coefficient})"


@dataclass(frozen=True)
class ContractedAO:
    """
    One atom-centered contracted Cartesian AO.

    Represents

        chi_mu(r) = sum_p d_p N_p x^a y^b z^c exp(-alpha_p r^2)

    Parameters
    ----------
    label
        Human-readable AO label, e.g. '1s', '2px', '3dxy'.

    powers
        Cartesian powers (a,b,c).

    primitives
        Tuple of AOPrimitive objects.

    normalized
        If True, each primitive uses the Cartesian primitive normalization from
        gaussian.py through AtomCenteredPrimitive.
    """

    label: str
    powers: Powers
    primitives: tuple[AOPrimitive, ...]
    normalized: bool = True

    def __post_init__(self) -> None:
        if len(self.powers) != 3:
            raise ValueError("AO powers must have length 3")
        if any(p < 0 for p in self.powers):
            raise ValueError("AO powers must be nonnegative")
        if not self.primitives:
            raise ValueError("ContractedAO must contain at least one primitive")

    @classmethod
    def primitive(
        cls,
        label: str,
        alpha: float,
        powers: Powers = (0, 0, 0),
        coefficient: float = 1.0,
        normalized: bool = True,
    ) -> "ContractedAO":
        """Build a one-primitive AO."""
        return cls(
            label=label,
            powers=powers,
            primitives=(AOPrimitive(alpha=alpha, coefficient=coefficient),),
            normalized=normalized,
        )

    def to_contracted_gaussian(self, coefficient_scale: float = 1.0) -> ContractedGaussian:
        """
        Convert this AO into the ContractedGaussian format used by the integral engine.

        coefficient_scale is useful when this AO appears inside an MO with
        coefficient C_mu,i.
        """
        prims = []
        for prim in self.primitives:
            prims.append(
                AtomCenteredPrimitive(
                    alpha=prim.alpha,
                    coefficient=coefficient_scale * prim.coefficient,
                    powers=self.powers,
                    normalized=self.normalized,
                )
            )

        return ContractedGaussian(
            primitives=tuple(prims),
            label=self.label,
        )

    def describe(self) -> str:
        lines = [
            f"ContractedAO(label={self.label}, powers={self.powers}, "
            f"nprim={len(self.primitives)}, normalized={self.normalized})"
        ]
        for i, prim in enumerate(self.primitives):
            lines.append(f"  [{i}] {prim.describe()}")
        return "\n".join(lines)


# =============================================================================
# AO basis container
# =============================================================================

@dataclass(frozen=True)
class AOBasis:
    """
    Atom-centered AO basis.

    Stores an ordered list of ContractedAO objects. The order matters because MO
    coefficient matrices use the same AO ordering.
    """

    aos: tuple[ContractedAO, ...]
    label: str = "AO basis"

    def __post_init__(self) -> None:
        if not self.aos:
            raise ValueError("AOBasis must contain at least one AO")

    @property
    def size(self) -> int:
        return len(self.aos)

    def labels(self) -> list[str]:
        return [ao.label for ao in self.aos]

    def ao(self, index: int) -> ContractedAO:
        return self.aos[index]

    def describe(self) -> str:
        lines = [f"AOBasis(label={self.label}, size={self.size})"]
        for i, ao in enumerate(self.aos):
            lines.append(f"AO {i}: {ao.label}, powers={ao.powers}, nprim={len(ao.primitives)}")
        return "\n".join(lines)


# =============================================================================
# MO coefficient container
# =============================================================================

@dataclass(frozen=True)
class MOBasis:
    """
    Molecular orbital coefficient matrix over an AO basis.

    coefficients[mu, i] = C_{mu,i}

    where mu indexes AOs and i indexes MOs.
    """

    ao_basis: AOBasis
    coefficients: np.ndarray
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        coeff = np.asarray(self.coefficients, dtype=float)
        if coeff.ndim != 2:
            raise ValueError("MO coefficients must be a 2D array")
        if coeff.shape[0] != self.ao_basis.size:
            raise ValueError(
                f"MO coefficient row count {coeff.shape[0]} does not match AO basis size {self.ao_basis.size}"
            )
        if self.labels is not None and len(self.labels) != coeff.shape[1]:
            raise ValueError("MO labels length must match number of MO columns")

    @property
    def n_ao(self) -> int:
        return self.coefficients.shape[0]

    @property
    def n_mo(self) -> int:
        return self.coefficients.shape[1]

    def mo_label(self, index: int) -> str:
        if self.labels is None:
            return f"MO{index}"
        return self.labels[index]

    def describe(self) -> str:
        lines = [f"MOBasis(n_ao={self.n_ao}, n_mo={self.n_mo})"]
        lines.append("AO order: " + ", ".join(self.ao_basis.labels()))
        if self.labels is not None:
            lines.append("MO labels: " + ", ".join(self.labels))
        return "\n".join(lines)


# =============================================================================
# Flatten AO/MO orbitals into ContractedGaussian objects
# =============================================================================

def combine_contracted_gaussians(
    orbitals: Iterable[ContractedGaussian],
    label: str = "combined",
) -> ContractedGaussian:
    """
    Combine several ContractedGaussian objects into one larger contraction.

    This is how an MO, which is a linear combination of AOs, becomes a single
    ContractedGaussian object for the integral wrapper.
    """
    primitives: list[AtomCenteredPrimitive] = []
    for orbital in orbitals:
        primitives.extend(orbital.primitives)

    if not primitives:
        raise ValueError("Cannot combine an empty list of orbitals")

    return ContractedGaussian(
        primitives=tuple(primitives),
        label=label,
    )


def ao_orbital(ao_basis: AOBasis, ao_index: int) -> ContractedGaussian:
    """
    Return one AO as a ContractedGaussian compatible with the integral engine.
    """
    ao = ao_basis.ao(ao_index)
    return ao.to_contracted_gaussian(coefficient_scale=1.0)


def mo_orbital(mo_basis: MOBasis, mo_index: int, drop_tol: float = 0.0) -> ContractedGaussian:
    """
    Build one MO as a ContractedGaussian by flattening its AO expansion.

    Parameters
    ----------
    mo_basis
        MO coefficient container.

    mo_index
        Which MO column to build.

    drop_tol
        AO coefficients with |C_mu,i| <= drop_tol are skipped.
    """
    if not (0 <= mo_index < mo_basis.n_mo):
        raise IndexError("mo_index out of range")

    pieces: list[ContractedGaussian] = []

    for mu, ao in enumerate(mo_basis.ao_basis.aos):
        c_mu_i = float(mo_basis.coefficients[mu, mo_index])
        if abs(c_mu_i) <= drop_tol:
            continue
        pieces.append(ao.to_contracted_gaussian(coefficient_scale=c_mu_i))

    if not pieces:
        raise ValueError(f"MO {mo_index} is empty after applying drop_tol={drop_tol}")

    return combine_contracted_gaussians(
        pieces,
        label=mo_basis.mo_label(mo_index),
    )


# =============================================================================
# Small example bases
# =============================================================================

def minimal_sp_basis(alpha_s: float = 1.0, alpha_p: float = 1.0, normalized: bool = True) -> AOBasis:
    """
    Build a tiny atom-centered s + p basis for testing.

    AO order:

        0: s
        1: px
        2: py
        3: pz
    """
    return AOBasis(
        aos=(
            ContractedAO.primitive("s", alpha=alpha_s, powers=(0, 0, 0), normalized=normalized),
            ContractedAO.primitive("px", alpha=alpha_p, powers=(1, 0, 0), normalized=normalized),
            ContractedAO.primitive("py", alpha=alpha_p, powers=(0, 1, 0), normalized=normalized),
            ContractedAO.primitive("pz", alpha=alpha_p, powers=(0, 0, 1), normalized=normalized),
        ),
        label="minimal_sp_basis",
    )


def identity_mo_basis(ao_basis: AOBasis) -> MOBasis:
    """
    Build an MO basis where each MO equals one AO.

    Useful for testing the AO/MO machinery without changing the physics.
    """
    coeff = np.eye(ao_basis.size)
    return MOBasis(
        ao_basis=ao_basis,
        coefficients=coeff,
        labels=tuple(ao_basis.labels()),
    )


def demo_mixed_sp_mo(ao_basis: AOBasis) -> MOBasis:
    """
    Build a small example MO basis with mixed s/px character.

    Requires the minimal s,px,py,pz ordering.
    """
    if ao_basis.size < 4:
        raise ValueError("demo_mixed_sp_mo expects at least 4 AOs")

    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    coeff = np.zeros((ao_basis.size, 4))

    # MO0 = s
    coeff[0, 0] = 1.0

    # MO1 = px
    coeff[1, 1] = 1.0

    # MO2 = (s + px)/sqrt(2)
    coeff[0, 2] = inv_sqrt2
    coeff[1, 2] = inv_sqrt2

    # MO3 = (s - px)/sqrt(2)
    coeff[0, 3] = inv_sqrt2
    coeff[1, 3] = -inv_sqrt2

    return MOBasis(
        ao_basis=ao_basis,
        coefficients=coeff,
        labels=("s", "px", "s_plus_px", "s_minus_px"),
    )


# =============================================================================
# Command-line demo
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AO/MO basis bookkeeping demo.")
    parser.add_argument("--alpha-s", type=float, default=1.0)
    parser.add_argument("--alpha-p", type=float, default=1.0)
    parser.add_argument("--unnormalized", action="store_true")
    parser.add_argument("--mo-index", type=int, default=2)
    parser.add_argument("--drop-tol", type=float, default=0.0)

    args = parser.parse_args()

    ao_basis = minimal_sp_basis(
        alpha_s=args.alpha_s,
        alpha_p=args.alpha_p,
        normalized=not args.unnormalized,
    )
    mo_basis = demo_mixed_sp_mo(ao_basis)

    print("\n=== AO basis ===")
    print(ao_basis.describe())

    print("\n=== MO basis ===")
    print(mo_basis.describe())
    print("Coefficient matrix C[AO,MO]:")
    print(mo_basis.coefficients)

    mo = mo_orbital(mo_basis, args.mo_index, drop_tol=args.drop_tol)

    print(f"\n=== Flattened MO {args.mo_index}: {mo.label} ===")
    print(mo.describe())


if __name__ == "__main__":
    main()
