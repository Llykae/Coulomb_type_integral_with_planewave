"""
integral_optimization.py

Cost-reduction helpers for contracted Gaussian Coulomb / plane-wave integrals.

Why this module matters
-----------------------
QP molecular orbitals are expanded as linear combinations of contracted AOs.
When we flatten an MO into primitives, we can get many duplicate primitive
entries with the same:

    alpha
    Cartesian powers
    normalized flag

For example, several contracted s AOs often share the same exponent list. A
flattened MO can therefore contain repeated primitives that can be merged by
summing coefficients.

This is mathematically exact because the primitive functions are identical:

    c1 * g(alpha,powers) + c2 * g(alpha,powers)
    = (c1+c2) * g(alpha,powers).

This can reduce cost dramatically. If an MO goes from 37 primitives to 12, then
an MO/MO/MO triple contraction goes from

    37^3 = 50653

to

    12^3 = 1728.

That is a ~29x reduction before any radial cache is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

from full_coulomb_integral import AtomCenteredPrimitive, ContractedGaussian


@dataclass(frozen=True)
class CompressionReport:
    """Report for one contracted Gaussian compression."""

    label: str
    n_before: int
    n_after: int
    dropped_small: int

    @property
    def reduction_factor(self) -> float:
        if self.n_after == 0:
            return float("inf")
        return self.n_before / self.n_after

    def print(self) -> None:
        print(
            f"{self.label}: nprim {self.n_before} -> {self.n_after} "
            f"(x{self.reduction_factor:.2f}), dropped={self.dropped_small}"
        )


def primitive_merge_key(
    primitive: AtomCenteredPrimitive,
    alpha_round_digits: int = 12,
) -> tuple[float, tuple[int, int, int], bool]:
    """
    Key used to merge identical primitives.

    We round alpha to avoid tiny text/binary representation differences.
    For QP text arrays, 12 digits is usually conservative.
    """
    return (
        round(float(primitive.alpha), alpha_round_digits),
        tuple(int(p) for p in primitive.powers),
        bool(primitive.normalized),
    )


def compress_contracted_gaussian(
    orbital: ContractedGaussian,
    coefficient_tol: float = 1e-14,
    alpha_round_digits: int = 12,
    label_suffix: str = "_compressed",
) -> tuple[ContractedGaussian, CompressionReport]:
    """
    Merge duplicate primitives in a ContractedGaussian.

    Parameters
    ----------
    orbital
        ContractedGaussian to compress.

    coefficient_tol
        After summing coefficients, primitives with |coefficient| <= this value
        are dropped.

    alpha_round_digits
        Number of decimal digits used for exponent matching.

    Returns
    -------
    compressed_orbital, report
    """
    grouped: dict[tuple[float, tuple[int, int, int], bool], float] = defaultdict(float)

    for primitive in orbital.primitives:
        key = primitive_merge_key(primitive, alpha_round_digits=alpha_round_digits)
        grouped[key] += float(primitive.coefficient)

    compressed_primitives: list[AtomCenteredPrimitive] = []
    dropped_small = 0

    for (alpha, powers, normalized), coefficient in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        if abs(coefficient) <= coefficient_tol:
            dropped_small += 1
            continue

        compressed_primitives.append(
            AtomCenteredPrimitive(
                alpha=float(alpha),
                coefficient=float(coefficient),
                powers=powers,
                normalized=normalized,
            )
        )

    if not compressed_primitives:
        raise ValueError(f"All primitives vanished while compressing {orbital.label}")

    compressed = ContractedGaussian(
        primitives=tuple(compressed_primitives),
        label=orbital.label + label_suffix,
    )

    report = CompressionReport(
        label=orbital.label,
        n_before=len(orbital.primitives),
        n_after=len(compressed.primitives),
        dropped_small=dropped_small,
    )

    return compressed, report


def compress_three_orbitals(
    phi_r: ContractedGaussian,
    phi_s1: ContractedGaussian,
    phi_s2: ContractedGaussian,
    coefficient_tol: float = 1e-14,
    alpha_round_digits: int = 12,
) -> tuple[ContractedGaussian, ContractedGaussian, ContractedGaussian, tuple[CompressionReport, CompressionReport, CompressionReport]]:
    """
    Compress the three orbitals used in the full integral.
    """
    phi_r_c, report_r = compress_contracted_gaussian(
        phi_r,
        coefficient_tol=coefficient_tol,
        alpha_round_digits=alpha_round_digits,
    )
    phi_s1_c, report_s1 = compress_contracted_gaussian(
        phi_s1,
        coefficient_tol=coefficient_tol,
        alpha_round_digits=alpha_round_digits,
    )
    phi_s2_c, report_s2 = compress_contracted_gaussian(
        phi_s2,
        coefficient_tol=coefficient_tol,
        alpha_round_digits=alpha_round_digits,
    )

    return phi_r_c, phi_s1_c, phi_s2_c, (report_r, report_s1, report_s2)


def primitive_triple_count(
    phi_r: ContractedGaussian,
    phi_s1: ContractedGaussian,
    phi_s2: ContractedGaussian,
) -> int:
    """Return nprim_r * nprim_s1 * nprim_s2."""
    return len(phi_r.primitives) * len(phi_s1.primitives) * len(phi_s2.primitives)


def print_primitive_cost_report(
    phi_r: ContractedGaussian,
    phi_s1: ContractedGaussian,
    phi_s2: ContractedGaussian,
    title: str = "Primitive contraction cost",
) -> None:
    """Print primitive counts and triple count."""
    nr = len(phi_r.primitives)
    ns1 = len(phi_s1.primitives)
    ns2 = len(phi_s2.primitives)
    print(f"\n=== {title} ===")
    print(f"nprim_r  = {nr}")
    print(f"nprim_s1 = {ns1}")
    print(f"nprim_s2 = {ns2}")
    print(f"primitive triples = {nr} * {ns1} * {ns2} = {nr * ns1 * ns2}")
