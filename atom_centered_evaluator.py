"""
atom_centered_evaluator.py

First full atom-centered angular/radial evaluator for the Coulomb / plane-wave
integral.

Target integral
---------------
We evaluate

    I(k) = ∫∫ d^3r d^3s
           exp(i k.r)
           phi_r(r)
           1/|r-s|
           phi_s1(s) phi_s2(s)

for atom-centered Cartesian Gaussian primitives.

The command-line interface now uses plane-wave kinetic energy plus direction:

    --energy 2.0 --direction 1.0 0.0 0.0

Atomic units:

    E = k^2 / 2

The direction is normalized internally. Internally the evaluator still receives
kvec, because low-level numerical routines are clearer with an explicit vector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import sympy as sp

from atom_centered_terms import AtomCenteredTerm, build_atom_centered_terms
from harmonics import complex_sph_harm
from radial_coulomb_2d import radial_coulomb_2d_integral
from plane_wave_parameters import add_plane_wave_cli_arguments, plane_wave_from_cli_args


Powers = tuple[int, int, int]


# =============================================================================
# k-vector helpers
# =============================================================================

def cartesian_to_spherical_angles_vector(kvec: np.ndarray) -> tuple[float, float, float]:
    """
    Convert k vector into |k|, theta, phi.

    theta = polar angle from +z.
    phi   = azimuthal angle from +x toward +y.
    """
    kvec = np.asarray(kvec, dtype=float)
    if kvec.shape != (3,):
        raise ValueError("kvec must be a length-3 vector")

    k_abs = float(np.linalg.norm(kvec))
    if k_abs == 0.0:
        raise ValueError("kvec must be nonzero")

    theta = float(np.arccos(kvec[2] / k_abs))
    phi = float(np.arctan2(kvec[1], kvec[0]))
    return k_abs, theta, phi


def build_kvec(k_abs: float, direction: tuple[float, float, float]) -> np.ndarray:
    """
    Backward-compatible helper for code that still wants direct |k|.

    New public scripts should prefer plane_wave_parameters.py.
    """
    direction_array = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction_array)
    if norm == 0.0:
        raise ValueError("k direction cannot be zero")
    return k_abs * direction_array / norm


# =============================================================================
# Evaluation result data structures
# =============================================================================

@dataclass(frozen=True)
class EvaluatedAtomCenteredTerm:
    """
    One numerically evaluated atom-centered term.
    """

    symbolic_term: AtomCenteredTerm
    radial_value: float
    ylm_k_conj_value: complex
    numeric_prefactor_without_radial: complex
    numeric_value: complex

    def print(self) -> None:
        c = self.symbolic_term.coupling
        print("\n--- Evaluated atom-centered term ---")
        print(f"r channel     : lr={c.r_channel.l}, mr={c.r_channel.m}")
        print(f"plane channel : lp={c.pw_channel.lp}, mp={c.pw_channel.mp}")
        print(f"s channel     : ls={c.s_channel.l}, ms={c.s_channel.m}")
        print(f"Coulomb       : L={c.L}, M={c.M}")
        print(f"radial value  : {self.radial_value:.16e}")
        print(f"Y*_k          : {self.ylm_k_conj_value}")
        print(f"prefactor     : {self.numeric_prefactor_without_radial}")
        print(f"term value    : {self.numeric_value}")


@dataclass(frozen=True)
class AtomCenteredEvaluation:
    """
    Full evaluated atom-centered result.
    """

    value: complex
    terms: tuple[EvaluatedAtomCenteredTerm, ...]

    def print(self, max_terms: int | None = 20) -> None:
        print("\n=== Atom-centered angular/radial evaluation ===")
        print(f"number of terms = {len(self.terms)}")
        print(f"value = {self.value.real:.16e} + {self.value.imag:.16e} i")

        if max_terms is None:
            to_print = self.terms
        else:
            to_print = self.terms[:max_terms]

        for term in to_print:
            term.print()

        if max_terms is not None and len(self.terms) > max_terms:
            print(f"\n... skipped {len(self.terms) - max_terms} additional terms")


# =============================================================================
# Core evaluator
# =============================================================================

def numeric_prefactor_without_radial(
    term: AtomCenteredTerm,
    kvec: np.ndarray,
) -> tuple[complex, complex]:
    """
    Evaluate all factors in a symbolic term except the radial integral.

    Returns
    -------
    prefactor, ylm_k_conj
    """
    _k_abs, theta_k, phi_k = cartesian_to_spherical_angles_vector(kvec)

    lp = term.coupling.pw_channel.lp
    mp = term.coupling.pw_channel.mp
    ylm_k_conj = complex(np.conjugate(complex_sph_harm(lp, mp, theta_k, phi_k)))

    plane_prefactor = 4.0 * np.pi * (1j ** lp) * ylm_k_conj
    coulomb_prefactor = 4.0 * np.pi / (2 * term.coupling.L + 1)
    angular_prefactor = complex(sp.N(term.angular_prefactor))

    prefactor = plane_prefactor * coulomb_prefactor * angular_prefactor
    return prefactor, ylm_k_conj


def evaluate_atom_centered_terms(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    alpha_r: float,
    alpha_s1: float,
    alpha_s2: float,
    kvec: np.ndarray,
    lmax_pw: int = 4,
    rmax: float | None = None,
    smax: float | None = None,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 300,
) -> AtomCenteredEvaluation:
    """
    Evaluate the atom-centered angular/radial expression.
    """
    if alpha_r <= 0 or alpha_s1 <= 0 or alpha_s2 <= 0:
        raise ValueError("Gaussian exponents must be positive")

    k_abs, _theta_k, _phi_k = cartesian_to_spherical_angles_vector(kvec)
    alpha_s_total = alpha_s1 + alpha_s2

    symbolic_terms = build_atom_centered_terms(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        lmax_pw=lmax_pw,
    )

    evaluated_terms: list[EvaluatedAtomCenteredTerm] = []
    total = 0.0 + 0.0j

    for term in symbolic_terms:
        c = term.coupling

        n_r = c.r_channel.radial_power_from_polynomial
        n_s = c.s_channel.radial_power_from_polynomial

        radial_result = radial_coulomb_2d_integral(
            lp=c.pw_channel.lp,
            L=c.L,
            k=k_abs,
            alpha_r=alpha_r,
            alpha_s=alpha_s_total,
            n_r=n_r,
            n_s=n_s,
            rmax=rmax,
            smax=smax,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )

        prefactor, ylm_k_conj = numeric_prefactor_without_radial(term, kvec)
        numeric_value = prefactor * radial_result.value
        total += numeric_value

        evaluated_terms.append(
            EvaluatedAtomCenteredTerm(
                symbolic_term=term,
                radial_value=radial_result.value,
                ylm_k_conj_value=ylm_k_conj,
                numeric_prefactor_without_radial=prefactor,
                numeric_value=numeric_value,
            )
        )

    return AtomCenteredEvaluation(
        value=complex(total),
        terms=tuple(evaluated_terms),
    )


# =============================================================================
# Optional comparison with brute-force prototype
# =============================================================================

def compare_with_cartesian_grid_prototype(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    alpha_r: float,
    alpha_s1: float,
    alpha_s2: float,
    kvec: np.ndarray,
    lmax_pw: int,
    n_grid: int = 13,
    extent: float = 3.0,
    epsilon: float = 0.2,
    normalized: bool = False,
) -> None:
    """
    Compare angular/radial evaluator with the old 3D grid prototype.

    Warning
    -------
    The grid prototype uses a regularized Coulomb kernel, while this evaluator
    uses the exact Coulomb radial kernel. Exact agreement is not expected.
    """
    try:
        from gaussian import PrimitiveGaussian
        from coulomb_prototype_full import (
            make_3d_grid,
            two_electron_integral_direct,
        )
    except ImportError:
        print("\nSkipping comparison: could not import coulomb_prototype_full.py")
        return

    grid = make_3d_grid(n=n_grid, extent=extent)

    phi_r = PrimitiveGaussian(
        alpha=alpha_r,
        center=np.array([0.0, 0.0, 0.0]),
        powers=powers_r,
        normalized=normalized,
    )
    phi_s1 = PrimitiveGaussian(
        alpha=alpha_s1,
        center=np.array([0.0, 0.0, 0.0]),
        powers=powers_s1,
        normalized=normalized,
    )
    phi_s2 = PrimitiveGaussian(
        alpha=alpha_s2,
        center=np.array([0.0, 0.0, 0.0]),
        powers=powers_s2,
        normalized=normalized,
    )

    brute = two_electron_integral_direct(
        phi_r=phi_r,
        phi_s1=phi_s1,
        phi_s2=phi_s2,
        kvec=kvec,
        grid_r=grid,
        grid_s=grid,
        epsilon=epsilon,
        batch_size=256,
        verbose=False,
    )

    analytic_like = evaluate_atom_centered_terms(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        alpha_r=alpha_r,
        alpha_s1=alpha_s1,
        alpha_s2=alpha_s2,
        kvec=kvec,
        lmax_pw=lmax_pw,
    ).value

    print("\n=== Comparison with Cartesian grid prototype ===")
    print("Warning: grid prototype uses regularized Coulomb; angular/radial uses exact Coulomb.")
    print(f"grid direct       = {brute.real:.16e} + {brute.imag:.16e} i")
    print(f"angular/radial    = {analytic_like.real:.16e} + {analytic_like.imag:.16e} i")
    print(f"absolute diff     = {abs(brute - analytic_like):.6e}")


# =============================================================================
# Command-line interface
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate atom-centered Coulomb / plane-wave integral using angular/radial terms."
    )

    parser.add_argument("--powers-r", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s1", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--powers-s2", nargs=3, type=int, default=(0, 0, 0))

    parser.add_argument("--alpha-r", type=float, default=1.0)
    parser.add_argument("--alpha-s1", type=float, default=1.0)
    parser.add_argument("--alpha-s2", type=float, default=1.0)

    add_plane_wave_cli_arguments(parser)

    parser.add_argument("--lmax-pw", type=int, default=4)
    parser.add_argument("--rmax", type=float, default=None)
    parser.add_argument("--smax", type=float, default=None)
    parser.add_argument("--epsabs", type=float, default=1e-10)
    parser.add_argument("--epsrel", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=300)

    parser.add_argument("--max-terms", type=int, default=20)
    parser.add_argument("--compare-grid", action="store_true")
    parser.add_argument("--grid-n", type=int, default=13)
    parser.add_argument("--grid-extent", type=float, default=3.0)
    parser.add_argument("--grid-epsilon", type=float, default=0.2)

    args = parser.parse_args()

    powers_r = tuple(args.powers_r)
    powers_s1 = tuple(args.powers_s1)
    powers_s2 = tuple(args.powers_s2)

    plane_wave = plane_wave_from_cli_args(args)
    kvec = plane_wave.kvec

    print("\n=== Atom-centered evaluator setup ===")
    print(plane_wave.describe())
    print(f"powers_r  = {powers_r}")
    print(f"powers_s1 = {powers_s1}")
    print(f"powers_s2 = {powers_s2}")
    print(f"alpha_r   = {args.alpha_r}")
    print(f"alpha_s1  = {args.alpha_s1}")
    print(f"alpha_s2  = {args.alpha_s2}")
    print(f"kvec      = {kvec}")
    print(f"lmax_pw   = {args.lmax_pw}")

    result = evaluate_atom_centered_terms(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        alpha_r=args.alpha_r,
        alpha_s1=args.alpha_s1,
        alpha_s2=args.alpha_s2,
        kvec=kvec,
        lmax_pw=args.lmax_pw,
        rmax=args.rmax,
        smax=args.smax,
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        limit=args.limit,
    )

    result.print(max_terms=args.max_terms)

    if args.compare_grid:
        compare_with_cartesian_grid_prototype(
            powers_r=powers_r,
            powers_s1=powers_s1,
            powers_s2=powers_s2,
            alpha_r=args.alpha_r,
            alpha_s1=args.alpha_s1,
            alpha_s2=args.alpha_s2,
            kvec=kvec,
            lmax_pw=args.lmax_pw,
            n_grid=args.grid_n,
            extent=args.grid_extent,
            epsilon=args.grid_epsilon,
        )


if __name__ == "__main__":
    main()
