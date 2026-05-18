"""
run_validation_suite.py

Small validation suite for the atom-centered Coulomb / plane-wave project.

Purpose
-------
This script runs a few physically meaningful checks through the current full
integral stack:

    full_coulomb_integral.py
    plane_wave_parameters.py
    atom_centered_evaluator.py
    angular_pipeline.py
    radial_coulomb_2d.py

The goal is to catch broken conventions early.

Validated cases
---------------
1. s / (s s), k along x

   Expected:
       real result
       imaginary part ~ 0
       only lp = 0 contributes

2. s / (p_x s), k along x

   Expected:
       imaginary result
       real part ~ 0
       first nonzero plane-wave channel is lp = 1

3. p_x / (p_x s), k along x

   Expected:
       real result
       imaginary part ~ 0
       lp = 0 and lp = 2 contribute

4. high-energy s / (s s), k = 72 bohr^-1

   Expected:
       small real value
       imaginary part ~ 0

This is not a formal unit test framework yet. It is a practical project-level
sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from full_coulomb_integral import (
    ContractedGaussian,
    full_coulomb_plane_wave_integral,
)
from plane_wave_parameters import plane_wave_from_energy_hartree, plane_wave_from_k_abs
from angular_pipeline import angular_pipeline


Powers = tuple[int, int, int]


@dataclass(frozen=True)
class ValidationCase:
    """One validation case."""

    name: str
    powers_r: Powers
    powers_s1: Powers
    powers_s2: Powers
    alpha_r: float
    alpha_s1: float
    alpha_s2: float
    plane_wave_kind: str
    plane_wave_value: float
    direction: tuple[float, float, float]
    lmax_pw: int
    expected_kind: str
    expected_allowed_lp: tuple[int, ...]


@dataclass(frozen=True)
class ValidationResult:
    """Result of one validation case."""

    case: ValidationCase
    value: complex
    allowed_lp: tuple[int, ...]
    passed: bool
    messages: tuple[str, ...]

    def print(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"\n=== {self.case.name} ===")
        print(f"status = {status}")
        print(f"value  = {self.value.real:.16e} + {self.value.imag:.16e} i")
        print(f"allowed lp from angular pipeline = {self.allowed_lp}")
        print(f"expected lp                      = {self.case.expected_allowed_lp}")
        for msg in self.messages:
            print(f"  - {msg}")


def make_single_orbital(alpha: float, powers: Powers, normalized: bool = False) -> ContractedGaussian:
    """Build a one-primitive atom-centered orbital."""
    return ContractedGaussian.single(
        alpha=alpha,
        powers=powers,
        coefficient=1.0,
        normalized=normalized,
        label=f"powers={powers}",
    )


def make_plane_wave(case: ValidationCase):
    """Build plane-wave parameters for a validation case."""
    if case.plane_wave_kind == "energy":
        return plane_wave_from_energy_hartree(case.plane_wave_value, direction=case.direction)
    if case.plane_wave_kind == "k":
        return plane_wave_from_k_abs(case.plane_wave_value, direction=case.direction)
    raise ValueError(f"Unknown plane_wave_kind={case.plane_wave_kind!r}")


def allowed_lp_for_case(case: ValidationCase) -> tuple[int, ...]:
    """Return allowed plane-wave lp values from angular_pipeline.py."""
    _r_channels, _s_channels, _pw_channels, couplings = angular_pipeline(
        powers_r=case.powers_r,
        powers_s1=case.powers_s1,
        powers_s2=case.powers_s2,
        lmax_pw=case.lmax_pw,
    )
    values = sorted({c.pw_channel.lp for c in couplings})
    return tuple(values)


def check_expected_kind(value: complex, expected_kind: str, tol: float) -> tuple[bool, str]:
    """Check real/imaginary expectation."""
    if expected_kind == "real":
        ok = abs(value.imag) <= tol
        return ok, f"imaginary part should vanish: |Im|={abs(value.imag):.3e}, tol={tol:.3e}"

    if expected_kind == "imaginary":
        ok = abs(value.real) <= tol
        return ok, f"real part should vanish: |Re|={abs(value.real):.3e}, tol={tol:.3e}"

    if expected_kind == "small_real":
        ok = abs(value.imag) <= tol and abs(value.real) < 1e-6
        return ok, f"expected small real value: Re={value.real:.3e}, Im={value.imag:.3e}"

    raise ValueError(f"Unknown expected_kind={expected_kind!r}")


def run_case(case: ValidationCase, tol: float = 1e-9) -> ValidationResult:
    """Run one validation case."""
    plane_wave = make_plane_wave(case)

    phi_r = make_single_orbital(case.alpha_r, case.powers_r)
    phi_s1 = make_single_orbital(case.alpha_s1, case.powers_s1)
    phi_s2 = make_single_orbital(case.alpha_s2, case.powers_s2)

    result = full_coulomb_plane_wave_integral(
        phi_r=phi_r,
        phi_s1=phi_s1,
        phi_s2=phi_s2,
        plane_wave=plane_wave,
        lmax_pw=case.lmax_pw,
    )

    allowed_lp = allowed_lp_for_case(case)

    messages: list[str] = []
    passed = True

    ok_kind, msg_kind = check_expected_kind(result.value, case.expected_kind, tol)
    messages.append(msg_kind)
    passed = passed and ok_kind

    ok_lp = allowed_lp == case.expected_allowed_lp
    messages.append(f"allowed lp check: got {allowed_lp}, expected {case.expected_allowed_lp}")
    passed = passed and ok_lp

    return ValidationResult(
        case=case,
        value=result.value,
        allowed_lp=allowed_lp,
        passed=passed,
        messages=tuple(messages),
    )


def default_cases() -> list[ValidationCase]:
    """Return default validation cases."""
    return [
        ValidationCase(
            name="s / (s s), energy=2 Ha, k along x",
            powers_r=(0, 0, 0),
            powers_s1=(0, 0, 0),
            powers_s2=(0, 0, 0),
            alpha_r=1.0,
            alpha_s1=1.0,
            alpha_s2=1.0,
            plane_wave_kind="energy",
            plane_wave_value=2.0,
            direction=(1.0, 0.0, 0.0),
            lmax_pw=8,
            expected_kind="real",
            expected_allowed_lp=(0,),
        ),
        ValidationCase(
            name="s / (px s), energy=2 Ha, k along x",
            powers_r=(0, 0, 0),
            powers_s1=(1, 0, 0),
            powers_s2=(0, 0, 0),
            alpha_r=1.0,
            alpha_s1=1.0,
            alpha_s2=1.0,
            plane_wave_kind="energy",
            plane_wave_value=2.0,
            direction=(1.0, 0.0, 0.0),
            lmax_pw=8,
            expected_kind="imaginary",
            expected_allowed_lp=(1,),
        ),
        ValidationCase(
            name="px / (px s), energy=2 Ha, k along x",
            powers_r=(1, 0, 0),
            powers_s1=(1, 0, 0),
            powers_s2=(0, 0, 0),
            alpha_r=1.0,
            alpha_s1=1.0,
            alpha_s2=1.0,
            plane_wave_kind="energy",
            plane_wave_value=2.0,
            direction=(1.0, 0.0, 0.0),
            lmax_pw=8,
            expected_kind="real",
            expected_allowed_lp=(0, 2),
        ),
        ValidationCase(
            name="high-k s / (s s), k=72 bohr^-1, k along x",
            powers_r=(0, 0, 0),
            powers_s1=(0, 0, 0),
            powers_s2=(0, 0, 0),
            alpha_r=1.0,
            alpha_s1=1.0,
            alpha_s2=1.0,
            plane_wave_kind="k",
            plane_wave_value=72.0,
            direction=(1.0, 0.0, 0.0),
            lmax_pw=8,
            expected_kind="small_real",
            expected_allowed_lp=(0,),
        ),
    ]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run project validation suite.")
    parser.add_argument("--tol", type=float, default=1e-9, help="zero tolerance for real/imag checks")
    args = parser.parse_args()

    print("\n=== Running atom-centered Coulomb / plane-wave validation suite ===")

    results = [run_case(case, tol=args.tol) for case in default_cases()]

    n_pass = sum(result.passed for result in results)
    n_total = len(results)

    for result in results:
        result.print()

    print("\n=== Summary ===")
    print(f"passed {n_pass} / {n_total}")

    if n_pass != n_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
