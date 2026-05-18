"""
density_contraction.py

Build and compress the s-side density contraction

    rho_s(s) = phi_s1(s) * phi_s2(s)

for general atom-centered Cartesian Gaussian contractions.

This is not restricted to s orbitals. For any two Cartesian Gaussian primitives,

    x^a1 y^b1 z^c1 exp(-alpha1 r^2)
    x^a2 y^b2 z^c2 exp(-alpha2 r^2)

we get the density primitive

    x^(a1+a2) y^(b1+b2) z^(c1+c2)
    exp(-(alpha1+alpha2) r^2).

The coefficient is the product of the two primitive total factors, including
contraction coefficients and optional normalization constants.

Why this matters
----------------
The old full integral loop was

    r primitive × s1 primitive × s2 primitive

with cost

    N_r * N_s1 * N_s2.

After building the density first, the loop becomes

    r primitive × density primitive

with cost

    N_r * N_density.

Duplicate density primitives with the same exponent and powers are merged by
summing coefficients.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from full_coulomb_integral import AtomCenteredPrimitive, ContractedGaussian


Powers = tuple[int, int, int]


@dataclass(frozen=True)
class DensityPrimitive:
    """
    One primitive in the contracted density rho_s.

    Represents

        coefficient * x^a y^b z^c exp(-alpha r^2)

    The coefficient already includes:

        primitive coefficient from s1
        primitive coefficient from s2
        optional normalization factor from s1 primitive
        optional normalization factor from s2 primitive

    Therefore DensityPrimitive does not need a normalized flag.
    """

    alpha: float
    coefficient: float
    powers: Powers

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("DensityPrimitive alpha must be positive")
        if len(self.powers) != 3:
            raise ValueError("DensityPrimitive powers must have length 3")
        if any(p < 0 for p in self.powers):
            raise ValueError("DensityPrimitive powers must be nonnegative")

    def describe(self) -> str:
        return (
            f"DensityPrimitive(alpha={self.alpha}, coefficient={self.coefficient}, "
            f"powers={self.powers})"
        )


@dataclass(frozen=True)
class ContractedDensity:
    """
    Contracted Cartesian density rho_s = phi_s1 * phi_s2.
    """

    primitives: tuple[DensityPrimitive, ...]
    label: str = "density"

    def __post_init__(self) -> None:
        if not self.primitives:
            raise ValueError("ContractedDensity must contain at least one primitive")

    def describe(self, max_primitives: int | None = None) -> str:
        lines = [f"ContractedDensity(label={self.label}, nprim={len(self.primitives)})"]
        prims = self.primitives if max_primitives is None else self.primitives[:max_primitives]
        for i, primitive in enumerate(prims):
            lines.append(f"  [{i}] {primitive.describe()}")
        if max_primitives is not None and len(self.primitives) > max_primitives:
            lines.append(f"  ... skipped {len(self.primitives) - max_primitives} density primitives")
        return "\n".join(lines)


@dataclass(frozen=True)
class DensityBuildReport:
    """Report for density contraction construction."""

    label: str
    n_s1: int
    n_s2: int
    n_pair_raw: int
    n_density: int
    dropped_small: int

    @property
    def reduction_factor(self) -> float:
        if self.n_density == 0:
            return float("inf")
        return self.n_pair_raw / self.n_density

    def print(self) -> None:
        print("\n=== Density contraction report ===")
        print(f"label       = {self.label}")
        print(f"nprim_s1    = {self.n_s1}")
        print(f"nprim_s2    = {self.n_s2}")
        print(f"raw pairs   = {self.n_s1} * {self.n_s2} = {self.n_pair_raw}")
        print(f"density n   = {self.n_density}")
        print(f"reduction   = x{self.reduction_factor:.2f}")
        print(f"dropped     = {self.dropped_small}")


def add_powers(powers_a: Powers, powers_b: Powers) -> Powers:
    """Add Cartesian powers."""
    return tuple(int(a) + int(b) for a, b in zip(powers_a, powers_b))


def density_merge_key(
    alpha: float,
    powers: Powers,
    alpha_round_digits: int = 12,
) -> tuple[float, Powers]:
    """Hashable key for merging density primitives."""
    return (round(float(alpha), alpha_round_digits), tuple(int(p) for p in powers))


def build_contracted_density(
    phi_s1: ContractedGaussian,
    phi_s2: ContractedGaussian,
    coefficient_tol: float = 1e-14,
    alpha_round_digits: int = 12,
    label: str | None = None,
) -> tuple[ContractedDensity, DensityBuildReport]:
    """
    Build rho_s = phi_s1 * phi_s2 as a compressed ContractedDensity.

    This is exact up to coefficient_tol and alpha rounding used for merging.
    """
    grouped: dict[tuple[float, Powers], float] = defaultdict(float)

    n_raw = len(phi_s1.primitives) * len(phi_s2.primitives)

    for p1 in phi_s1.primitives:
        for p2 in phi_s2.primitives:
            alpha = float(p1.alpha) + float(p2.alpha)
            powers = add_powers(p1.powers, p2.powers)
            coefficient = float(p1.total_factor) * float(p2.total_factor)

            key = density_merge_key(alpha, powers, alpha_round_digits=alpha_round_digits)
            grouped[key] += coefficient

    density_primitives: list[DensityPrimitive] = []
    dropped_small = 0

    for (alpha, powers), coefficient in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        if abs(coefficient) <= coefficient_tol:
            dropped_small += 1
            continue
        density_primitives.append(
            DensityPrimitive(
                alpha=float(alpha),
                coefficient=float(coefficient),
                powers=powers,
            )
        )

    if not density_primitives:
        raise ValueError("Density contraction vanished after compression")

    if label is None:
        label = f"rho({phi_s1.label}*{phi_s2.label})"

    density = ContractedDensity(
        primitives=tuple(density_primitives),
        label=label,
    )

    report = DensityBuildReport(
        label=label,
        n_s1=len(phi_s1.primitives),
        n_s2=len(phi_s2.primitives),
        n_pair_raw=n_raw,
        n_density=len(density_primitives),
        dropped_small=dropped_small,
    )

    return density, report


def print_density_cost_report(phi_r: ContractedGaussian, density: ContractedDensity) -> None:
    """Print r × density cost."""
    nr = len(phi_r.primitives)
    nd = len(density.primitives)
    print("\n=== r × density primitive cost ===")
    print(f"nprim_r       = {nr}")
    print(f"nprim_density = {nd}")
    print(f"primitive pairs = {nr} * {nd} = {nr * nd}")
