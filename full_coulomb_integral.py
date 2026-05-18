"""
full_coulomb_integral.py

Optimized user-facing full atom-centered Coulomb / plane-wave integral evaluator.

This version adds two important cost reductions:

1. Optional contraction compression

   Duplicate primitives with the same exponent, powers, and normalization flag
   can be merged before the integral is evaluated.

2. Primitive integral cache

   If the same primitive triple appears repeatedly, the primitive angular/radial
   value is computed once and reused.

The target integral remains

    I(k) = ∫∫ d^3r d^3s
           exp(i k.r)
           phi_r(r)
           1/|r-s|
           phi_s1(s) phi_s2(s).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product

import numpy as np

from atom_centered_evaluator import evaluate_atom_centered_terms
from gaussian import gaussian_norm_cartesian
from plane_wave_parameters import (
    PlaneWaveParameters,
    add_plane_wave_cli_arguments,
    plane_wave_from_cli_args,
)


Powers = tuple[int, int, int]


# =============================================================================
# Gaussian orbital data structures
# =============================================================================

@dataclass(frozen=True)
class AtomCenteredPrimitive:
    """
    One atom-centered Cartesian Gaussian primitive.

    Represents

        coefficient * N * x^a y^b z^c exp(-alpha r^2)

    where N is included only if normalized=True.
    """

    alpha: float
    coefficient: float = 1.0
    powers: Powers = (0, 0, 0)
    normalized: bool = False

    def __post_init__(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if len(self.powers) != 3:
            raise ValueError("powers must have length 3")
        if any(p < 0 for p in self.powers):
            raise ValueError("powers must be nonnegative")

    @property
    def norm(self) -> float:
        """Return Cartesian primitive normalization factor."""
        if not self.normalized:
            return 1.0
        return gaussian_norm_cartesian(self.alpha, self.powers)

    @property
    def total_factor(self) -> float:
        """Return contraction coefficient times optional primitive normalization."""
        return self.coefficient * self.norm

    def describe(self) -> str:
        return (
            f"AtomCenteredPrimitive(alpha={self.alpha}, "
            f"coefficient={self.coefficient}, "
            f"powers={self.powers}, "
            f"normalized={self.normalized}, "
            f"total_factor={self.total_factor})"
        )


@dataclass(frozen=True)
class ContractedGaussian:
    """
    Atom-centered contracted Cartesian Gaussian orbital.
    """

    primitives: tuple[AtomCenteredPrimitive, ...]
    label: str = "orbital"

    def __post_init__(self) -> None:
        if not self.primitives:
            raise ValueError("ContractedGaussian must contain at least one primitive")

    @classmethod
    def single(
        cls,
        alpha: float,
        powers: Powers = (0, 0, 0),
        coefficient: float = 1.0,
        normalized: bool = False,
        label: str = "primitive",
    ) -> "ContractedGaussian":
        return cls(
            primitives=(
                AtomCenteredPrimitive(
                    alpha=alpha,
                    coefficient=coefficient,
                    powers=powers,
                    normalized=normalized,
                ),
            ),
            label=label,
        )

    def describe(self, max_primitives: int | None = None) -> str:
        lines = [f"ContractedGaussian(label={self.label}, nprim={len(self.primitives)})"]
        prims = self.primitives if max_primitives is None else self.primitives[:max_primitives]
        for i, primitive in enumerate(prims):
            lines.append(f"  [{i}] {primitive.describe()}")
        if max_primitives is not None and len(self.primitives) > max_primitives:
            lines.append(f"  ... skipped {len(self.primitives) - max_primitives} primitives")
        return "\n".join(lines)


# =============================================================================
# Result structures
# =============================================================================

@dataclass(frozen=True)
class PrimitiveTripleContribution:
    index_r: int
    index_s1: int
    index_s2: int
    factor: float
    primitive_value: complex
    contribution: complex
    from_cache: bool = False

    def print(self) -> None:
        cache_tag = " cache" if self.from_cache else " eval "
        print(
            f"  triple ({self.index_r},{self.index_s1},{self.index_s2})[{cache_tag}] "
            f"factor={self.factor:.8e} "
            f"primitive={self.primitive_value.real:.12e}+{self.primitive_value.imag:.12e}i "
            f"contribution={self.contribution.real:.12e}+{self.contribution.imag:.12e}i"
        )


@dataclass(frozen=True)
class FullCoulombIntegralResult:
    value: complex
    contributions: tuple[PrimitiveTripleContribution, ...]
    plane_wave: PlaneWaveParameters
    cache_hits: int = 0
    cache_misses: int = 0

    def print(self, max_contributions: int | None = 20) -> None:
        print("\n=== Full contracted Coulomb / plane-wave integral ===")
        print(self.plane_wave.describe())
        print(f"number of primitive triples = {len(self.contributions)}")
        print(f"primitive cache hits        = {self.cache_hits}")
        print(f"primitive cache misses      = {self.cache_misses}")
        print(f"value = {self.value.real:.16e} + {self.value.imag:.16e} i")

        if max_contributions is None:
            shown = self.contributions
        else:
            shown = self.contributions[:max_contributions]

        print("\nPrimitive triple contributions:")
        for contribution in shown:
            contribution.print()

        if max_contributions is not None and len(self.contributions) > max_contributions:
            print(f"  ... skipped {len(self.contributions) - max_contributions} additional contributions")


# =============================================================================
# Cache helpers
# =============================================================================

def primitive_key(primitive: AtomCenteredPrimitive, alpha_round_digits: int = 12) -> tuple:
    """Hashable key for a primitive integral cache."""
    return (
        round(float(primitive.alpha), alpha_round_digits),
        tuple(int(p) for p in primitive.powers),
        bool(primitive.normalized),
    )


def primitive_triple_key(
    pr: AtomCenteredPrimitive,
    ps1: AtomCenteredPrimitive,
    ps2: AtomCenteredPrimitive,
    k_abs: float,
    direction: np.ndarray,
    lmax_pw: int,
    alpha_round_digits: int = 12,
    direction_round_digits: int = 12,
) -> tuple:
    """Hashable key for the primitive value without contraction coefficients."""
    return (
        primitive_key(pr, alpha_round_digits),
        primitive_key(ps1, alpha_round_digits),
        primitive_key(ps2, alpha_round_digits),
        round(float(k_abs), alpha_round_digits),
        tuple(round(float(x), direction_round_digits) for x in direction),
        int(lmax_pw),
    )


# =============================================================================
# Cost reporting
# =============================================================================

def primitive_triple_count(
    phi_r: ContractedGaussian,
    phi_s1: ContractedGaussian,
    phi_s2: ContractedGaussian,
) -> int:
    return len(phi_r.primitives) * len(phi_s1.primitives) * len(phi_s2.primitives)


def print_primitive_cost_report(
    phi_r: ContractedGaussian,
    phi_s1: ContractedGaussian,
    phi_s2: ContractedGaussian,
    title: str = "Primitive contraction cost",
) -> None:
    nr = len(phi_r.primitives)
    ns1 = len(phi_s1.primitives)
    ns2 = len(phi_s2.primitives)
    print(f"\n=== {title} ===")
    print(f"nprim_r  = {nr}")
    print(f"nprim_s1 = {ns1}")
    print(f"nprim_s2 = {ns2}")
    print(f"primitive triples = {nr} * {ns1} * {ns2} = {nr * ns1 * ns2}")


# =============================================================================
# Full contracted integral
# =============================================================================

def full_coulomb_plane_wave_integral(
    phi_r: ContractedGaussian,
    phi_s1: ContractedGaussian,
    phi_s2: ContractedGaussian,
    plane_wave: PlaneWaveParameters,
    lmax_pw: int = 8,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 300,
    use_cache: bool = True,
    alpha_round_digits: int = 12,
    max_contributions_store: int | None = None,
) -> FullCoulombIntegralResult:
    """
    Evaluate the full contracted atom-centered integral.

    Parameters
    ----------
    use_cache
        Cache primitive integral values. Strongly recommended for QP MO tests.

    max_contributions_store
        Store only the first N primitive contributions in the result object.
        The total value still includes all triples. This saves memory for large
        QP contractions.
    """
    total = 0.0 + 0.0j
    contributions: list[PrimitiveTripleContribution] = []

    cache: dict[tuple, complex] = {}
    cache_hits = 0
    cache_misses = 0

    k_abs = plane_wave.k_abs
    direction = plane_wave.direction

    for (ir, pr), (is1, ps1), (is2, ps2) in product(
        enumerate(phi_r.primitives),
        enumerate(phi_s1.primitives),
        enumerate(phi_s2.primitives),
    ):
        factor = pr.total_factor * ps1.total_factor * ps2.total_factor

        key = primitive_triple_key(
            pr=pr,
            ps1=ps1,
            ps2=ps2,
            k_abs=k_abs,
            direction=direction,
            lmax_pw=lmax_pw,
            alpha_round_digits=alpha_round_digits,
        )

        from_cache = False
        if use_cache and key in cache:
            primitive_value = cache[key]
            cache_hits += 1
            from_cache = True
        else:
            primitive_eval = evaluate_atom_centered_terms(
                powers_r=pr.powers,
                powers_s1=ps1.powers,
                powers_s2=ps2.powers,
                alpha_r=pr.alpha,
                alpha_s1=ps1.alpha,
                alpha_s2=ps2.alpha,
                kvec=plane_wave.kvec,
                lmax_pw=lmax_pw,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
            )
            primitive_value = primitive_eval.value
            if use_cache:
                cache[key] = primitive_value
            cache_misses += 1

        contribution = factor * primitive_value
        total += contribution

        if max_contributions_store is None or len(contributions) < max_contributions_store:
            contributions.append(
                PrimitiveTripleContribution(
                    index_r=ir,
                    index_s1=is1,
                    index_s2=is2,
                    factor=float(factor),
                    primitive_value=primitive_value,
                    contribution=contribution,
                    from_cache=from_cache,
                )
            )

    return FullCoulombIntegralResult(
        value=complex(total),
        contributions=tuple(contributions),
        plane_wave=plane_wave,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


# =============================================================================
# Command-line convenience
# =============================================================================

def parse_powers(values: list[int]) -> Powers:
    if len(values) != 3:
        raise ValueError("Need exactly three powers")
    powers = tuple(int(v) for v in values)
    if any(p < 0 for p in powers):
        raise ValueError("Powers must be nonnegative")
    return powers


def make_single_orbital_from_args(
    alpha: float,
    powers: Powers,
    coefficient: float,
    normalized: bool,
    label: str,
) -> ContractedGaussian:
    return ContractedGaussian.single(
        alpha=alpha,
        powers=powers,
        coefficient=coefficient,
        normalized=normalized,
        label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full atom-centered Coulomb / plane-wave integral for Gaussian orbitals."
    )

    add_plane_wave_cli_arguments(parser)

    parser.add_argument("--alpha-r", type=float, default=1.0)
    parser.add_argument("--alpha-s1", type=float, default=1.0)
    parser.add_argument("--alpha-s2", type=float, default=1.0)

    parser.add_argument("--coef-r", type=float, default=1.0)
    parser.add_argument("--coef-s1", type=float, default=1.0)
    parser.add_argument("--coef-s2", type=float, default=1.0)

    parser.add_argument("--powers-r", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s1", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s2", nargs=3, type=int, default=(0, 0, 0))

    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--lmax-pw", type=int, default=8)
    parser.add_argument("--epsabs", type=float, default=1e-10)
    parser.add_argument("--epsrel", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-contributions", type=int, default=20)
    parser.add_argument("--no-cache", action="store_true")

    args = parser.parse_args()

    plane_wave = plane_wave_from_cli_args(args)

    phi_r = make_single_orbital_from_args(
        alpha=args.alpha_r,
        powers=parse_powers(args.powers_r),
        coefficient=args.coef_r,
        normalized=args.normalized,
        label="phi_r",
    )
    phi_s1 = make_single_orbital_from_args(
        alpha=args.alpha_s1,
        powers=parse_powers(args.powers_s1),
        coefficient=args.coef_s1,
        normalized=args.normalized,
        label="phi_s1",
    )
    phi_s2 = make_single_orbital_from_args(
        alpha=args.alpha_s2,
        powers=parse_powers(args.powers_s2),
        coefficient=args.coef_s2,
        normalized=args.normalized,
        label="phi_s2",
    )

    print("\n=== Full Coulomb integral setup ===")
    print(plane_wave.describe())
    print(f"lmax_pw = {args.lmax_pw}")
    print(phi_r.describe())
    print(phi_s1.describe())
    print(phi_s2.describe())
    print_primitive_cost_report(phi_r, phi_s1, phi_s2)

    result = full_coulomb_plane_wave_integral(
        phi_r=phi_r,
        phi_s1=phi_s1,
        phi_s2=phi_s2,
        plane_wave=plane_wave,
        lmax_pw=args.lmax_pw,
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        limit=args.limit,
        use_cache=not args.no_cache,
        max_contributions_store=args.max_contributions,
    )

    result.print(max_contributions=args.max_contributions)


if __name__ == "__main__":
    main()
