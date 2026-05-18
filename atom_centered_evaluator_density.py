"""
atom_centered_evaluator_density.py

Density-aware primitive evaluator for the atom-centered Coulomb / plane-wave
integral.

This version supports the contracted-density path:

    rho_s(s) = phi_s1(s) phi_s2(s)

so the primitive integral is parameterized by

    r-side primitive:
        powers_r, alpha_r

    density primitive:
        powers_density, alpha_density

Target primitive integral
-------------------------

    I = ∫∫ d^3r d^3s
        exp(i k.r)
        x_r^a y_r^b z_r^c exp(-alpha_r r^2)
        1/|r-s|
        x_s^d y_s^e z_s^f exp(-alpha_density s^2)

Important k=0 behavior
----------------------
At k=0, the direction khat is undefined. However, all plane-wave channels with
lp > 0 vanish because j_lp(0)=0. The lp=0 channel is independent of khat because
Y_00 is constant. Therefore, for k=0 we use a dummy direction [1,0,0] only for
angular prefactor bookkeeping and skip all lp>0 channels explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from angular_pipeline import (
    AngularCoupling,
    channels_for_powers,
    plane_wave_channels,
    build_allowed_couplings,
)
from atom_centered_evaluator import (
    numeric_prefactor_without_radial,
    EvaluatedAtomCenteredTerm,
    AtomCenteredEvaluation,
)
from radial_coulomb_2d import radial_coulomb_2d_integral
from radial_table import RadialIntegralTable, RadialKey
from parity import validate_powers


Powers = tuple[int, int, int]


@dataclass(frozen=True)
class DensitySymbolicTerm:
    """
    Minimal symbolic-term object compatible with numeric_prefactor_without_radial.

    It carries the same fields used by numeric_prefactor_without_radial:

        coupling
        angular_prefactor
    """

    coupling: AngularCoupling
    angular_prefactor: sp.Expr


def build_density_symbolic_terms(
    powers_r: Powers,
    powers_density: Powers,
    lmax_pw: int,
) -> list[DensitySymbolicTerm]:
    """
    Build allowed angular terms for one r primitive and one density primitive.
    """
    powers_r = validate_powers(powers_r, "powers_r")
    powers_density = validate_powers(powers_density, "powers_density")

    r_channels = channels_for_powers(powers_r, side="r")
    density_channels = channels_for_powers(powers_density, side="s")
    pw_channels = plane_wave_channels(lmax_pw)

    couplings = build_allowed_couplings(
        r_channels=r_channels,
        s_channels=density_channels,
        pw_channels=pw_channels,
    )

    terms: list[DensitySymbolicTerm] = []

    for coupling in couplings:
        angular_prefactor = sp.simplify(
            coupling.r_channel.coefficient
            * coupling.s_channel.coefficient
            * coupling.r_gaunt
            * coupling.s_overlap
        )
        terms.append(
            DensitySymbolicTerm(
                coupling=coupling,
                angular_prefactor=angular_prefactor,
            )
        )

    return terms


def radial_key_for_density_term(
    term: DensitySymbolicTerm,
    k_abs: float,
    alpha_r: float,
    alpha_density: float,
) -> RadialKey:
    """
    Build the RadialKey for one density symbolic term.
    """
    c = term.coupling
    return RadialKey(
        lp=c.pw_channel.lp,
        L=c.L,
        k=float(k_abs),
        alpha_r=float(alpha_r),
        alpha_s=float(alpha_density),
        n_r=c.r_channel.radial_power_from_polynomial,
        n_s=c.s_channel.radial_power_from_polynomial,
    )


def required_radial_keys_for_density_primitive(
    powers_r: Powers,
    powers_density: Powers,
    alpha_r: float,
    alpha_density: float,
    k_abs: float,
    lmax_pw: int,
) -> list[RadialKey]:
    """
    Return all radial keys needed by one r primitive / density primitive pair.

    At k=0, lp>0 channels are skipped because j_lp(0)=0.
    """
    terms = build_density_symbolic_terms(
        powers_r=powers_r,
        powers_density=powers_density,
        lmax_pw=lmax_pw,
    )

    keys: list[RadialKey] = []
    for term in terms:
        if abs(k_abs) < 1e-14 and term.coupling.pw_channel.lp > 0:
            continue
        keys.append(
            radial_key_for_density_term(
                term=term,
                k_abs=k_abs,
                alpha_r=alpha_r,
                alpha_density=alpha_density,
            )
        )

    return keys


def evaluate_atom_centered_terms_density(
    powers_r: Powers,
    powers_density: Powers,
    alpha_r: float,
    alpha_density: float,
    kvec: np.ndarray,
    lmax_pw: int = 4,
    rmax: float | None = None,
    smax: float | None = None,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 300,
    radial_table: RadialIntegralTable | None = None,
) -> AtomCenteredEvaluation:
    """
    Evaluate primitive atom-centered integral using a prebuilt density primitive.

    If radial_table is provided, radial integrals are requested from the table.
    Otherwise the direct radial_coulomb_2d_integral backend is used.
    """
    if alpha_r <= 0 or alpha_density <= 0:
        raise ValueError("Gaussian exponents must be positive")

    kvec = np.asarray(kvec, dtype=float)
    if kvec.shape != (3,):
        raise ValueError("kvec must have shape (3,)")

    k_abs = float(np.linalg.norm(kvec))

    # At k=0, khat is undefined. For lp=0 this is harmless because Y_00 is
    # direction-independent. Use a dummy nonzero vector only for angular
    # prefactors. All lp>0 channels are skipped below.
    if k_abs < 1e-14:
        kvec_for_angles = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        kvec_for_angles = kvec

    symbolic_terms = build_density_symbolic_terms(
        powers_r=powers_r,
        powers_density=powers_density,
        lmax_pw=lmax_pw,
    )

    evaluated_terms: list[EvaluatedAtomCenteredTerm] = []
    total = 0.0 + 0.0j

    for term in symbolic_terms:
        c = term.coupling

        # At k=0, j_lp(0)=0 for lp>0. Avoid evaluating meaningless angular
        # directions and radial channels that must vanish.
        if k_abs < 1e-14 and c.pw_channel.lp > 0:
            continue

        n_r = c.r_channel.radial_power_from_polynomial
        n_s = c.s_channel.radial_power_from_polynomial

        if radial_table is not None:
            radial_value = radial_table.get(
                RadialKey(
                    lp=c.pw_channel.lp,
                    L=c.L,
                    k=float(k_abs),
                    alpha_r=float(alpha_r),
                    alpha_s=float(alpha_density),
                    n_r=n_r,
                    n_s=n_s,
                )
            )
        else:
            radial_result = radial_coulomb_2d_integral(
                lp=c.pw_channel.lp,
                L=c.L,
                k=k_abs,
                alpha_r=alpha_r,
                alpha_s=alpha_density,
                n_r=n_r,
                n_s=n_s,
                rmax=rmax,
                smax=smax,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
            )
            radial_value = radial_result.value

        prefactor, ylm_k_conj = numeric_prefactor_without_radial(
            term,  # type: ignore[arg-type]
            kvec_for_angles,
        )
        numeric_value = prefactor * radial_value
        total += numeric_value

        evaluated_terms.append(
            EvaluatedAtomCenteredTerm(
                symbolic_term=term,  # type: ignore[arg-type]
                radial_value=radial_value,
                ylm_k_conj_value=ylm_k_conj,
                numeric_prefactor_without_radial=prefactor,
                numeric_value=numeric_value,
            )
        )

    return AtomCenteredEvaluation(
        value=complex(total),
        terms=tuple(evaluated_terms),
    )
