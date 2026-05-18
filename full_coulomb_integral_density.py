"""
full_coulomb_integral_density.py

Contracted-density full integral wrapper with radial-table support.

This evaluates

    I(k) = ∫∫ exp(i k.r) phi_r(r) 1/|r-s| rho_s(s) dr ds

where rho_s = phi_s1 * phi_s2 has already been contracted.

Compared to the previous density wrapper, this version:

1. builds the list of radial keys needed by all r-density primitive pairs;
2. prints a radial table report;
3. optionally precomputes radial values;
4. reuses the radial table during contraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from density_contraction import ContractedDensity, DensityPrimitive
from full_coulomb_integral import ContractedGaussian, PlaneWaveParameters
from radial_table import RadialIntegralTable, RadialKey, print_hardest_radial_keys
from atom_centered_evaluator_density import (
    evaluate_atom_centered_terms_density,
    required_radial_keys_for_density_primitive,
)


@dataclass(frozen=True)
class RDensityPairContribution:
    index_r: int
    index_density: int
    factor: float
    primitive_value: complex
    contribution: complex

    def print(self) -> None:
        print(
            f"  pair ({self.index_r},{self.index_density}) "
            f"factor={self.factor:.8e} "
            f"primitive={self.primitive_value.real:.12e}+{self.primitive_value.imag:.12e}i "
            f"contribution={self.contribution.real:.12e}+{self.contribution.imag:.12e}i"
        )


@dataclass(frozen=True)
class DensityIntegralResult:
    value: complex
    contributions: tuple[RDensityPairContribution, ...]
    plane_wave: PlaneWaveParameters
    n_pairs_total: int
    radial_table: RadialIntegralTable | None

    def print(self, max_contributions: int | None = 20) -> None:
        print("\n=== Full Coulomb / plane-wave integral via contracted density ===")
        print(self.plane_wave.describe())
        print(f"number of r-density pairs = {self.n_pairs_total}")
        print(f"stored contributions      = {len(self.contributions)}")
        if self.radial_table is not None:
            print(f"radial table values       = {self.radial_table.n_values}")
            print(f"radial table hits         = {self.radial_table.hits}")
            print(f"radial table misses       = {self.radial_table.misses}")
        print(f"value = {self.value.real:.16e} + {self.value.imag:.16e} i")

        shown = self.contributions if max_contributions is None else self.contributions[:max_contributions]
        print("\nPrimitive r-density pair contributions:")
        for contribution in shown:
            contribution.print()
        if max_contributions is not None and self.n_pairs_total > len(shown):
            print(f"  ... skipped {self.n_pairs_total - len(shown)} additional pairs")


def collect_radial_keys_for_density_integral(
    phi_r: ContractedGaussian,
    density: ContractedDensity,
    plane_wave: PlaneWaveParameters,
    lmax_pw: int,
) -> list[RadialKey]:
    """Collect all radial keys needed by an r × density contraction."""
    keys: list[RadialKey] = []
    for pr in phi_r.primitives:
        for pd in density.primitives:
            keys.extend(
                required_radial_keys_for_density_primitive(
                    powers_r=pr.powers,
                    powers_density=pd.powers,
                    alpha_r=pr.alpha,
                    alpha_density=pd.alpha,
                    k_abs=plane_wave.k_abs,
                    lmax_pw=lmax_pw,
                )
            )
    return keys


def full_coulomb_plane_wave_integral_density(
    phi_r: ContractedGaussian,
    density: ContractedDensity,
    plane_wave: PlaneWaveParameters,
    lmax_pw: int = 8,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 300,
    use_radial_table: bool = True,
    precompute_radial: bool = False,
    print_radial_report: bool = True,
    print_hard_keys: int = 0,
    alpha_round_digits: int = 12,
    max_contributions_store: int | None = 20,
) -> DensityIntegralResult:
    """
    Evaluate the full integral using r primitives and density primitives.
    """
    radial_table: RadialIntegralTable | None = None

    if use_radial_table:
        radial_table = RadialIntegralTable(
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            key_round_digits=alpha_round_digits,
        )
        keys = collect_radial_keys_for_density_integral(
            phi_r=phi_r,
            density=density,
            plane_wave=plane_wave,
            lmax_pw=lmax_pw,
        )
        if print_radial_report:
            radial_table.report_from_keys(keys).print()
        if print_hard_keys > 0:
            print_hardest_radial_keys(keys, n=print_hard_keys)
        if precompute_radial:
            radial_table.precompute(keys, verbose=True, progress_every=50)

    total = 0.0 + 0.0j
    contributions: list[RDensityPairContribution] = []
    n_pairs_total = len(phi_r.primitives) * len(density.primitives)

    for (ir, pr), (idens, pd) in product(enumerate(phi_r.primitives), enumerate(density.primitives)):
        factor = float(pr.total_factor) * float(pd.coefficient)

        primitive_eval = evaluate_atom_centered_terms_density(
            powers_r=pr.powers,
            powers_density=pd.powers,
            alpha_r=pr.alpha,
            alpha_density=pd.alpha,
            kvec=plane_wave.kvec,
            lmax_pw=lmax_pw,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            radial_table=radial_table,
        )
        primitive_value = primitive_eval.value
        contribution = factor * primitive_value
        total += contribution

        if max_contributions_store is None or len(contributions) < max_contributions_store:
            contributions.append(
                RDensityPairContribution(
                    index_r=ir,
                    index_density=idens,
                    factor=factor,
                    primitive_value=primitive_value,
                    contribution=contribution,
                )
            )

    return DensityIntegralResult(
        value=complex(total),
        contributions=tuple(contributions),
        plane_wave=plane_wave,
        n_pairs_total=n_pairs_total,
        radial_table=radial_table,
    )
