"""
run.py

Unified dynamic front-end for the atom-centered Coulomb / plane-wave project.

This script supports two styles:

1. Interactive dynamic mode

       python3 run.py
       python3 run.py interactive

   The script asks the user what to run step by step.

2. Reproducible CLI mode

       python3 run.py angular ...
       python3 run.py manual ...
       python3 run.py qp-mo ...

Target integral
---------------

    I(k) = ∫∫ d^3r d^3s
           exp(i k.r)
           phi_r(r)
           1/|r-s|
           phi_s1(s) phi_s2(s)

Current assumptions
-------------------
All orbitals are atom-centered. Off-center AOs are not supported yet.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from angular_pipeline import angular_pipeline, print_channels, print_allowed_couplings
from basis import AOPrimitive, AOBasis, ContractedAO, MOBasis, mo_orbital
from density_contraction import build_contracted_density, print_density_cost_report
from full_coulomb_integral import (
    ContractedGaussian,
    full_coulomb_plane_wave_integral,
    print_primitive_cost_report,
)
from full_coulomb_integral_density import full_coulomb_plane_wave_integral_density
from integral_optimization import compress_contracted_gaussian
from plane_wave_parameters import add_plane_wave_cli_arguments, plane_wave_from_cli_args
from parity import product_powers


Powers = tuple[int, int, int]


# =============================================================================
# Cartesian shell helpers
# =============================================================================

SHELL_TO_POWERS: dict[str, Powers] = {
    "s": (0, 0, 0),
    "px": (1, 0, 0),
    "py": (0, 1, 0),
    "pz": (0, 0, 1),
    "dxx": (2, 0, 0),
    "dyy": (0, 2, 0),
    "dzz": (0, 0, 2),
    "dxy": (1, 1, 0),
    "dxz": (1, 0, 1),
    "dyz": (0, 1, 1),
    "fxxx": (3, 0, 0),
    "fyyy": (0, 3, 0),
    "fzzz": (0, 0, 3),
    "fxxy": (2, 1, 0),
    "fxxz": (2, 0, 1),
    "fxyy": (1, 2, 0),
    "fyyz": (0, 2, 1),
    "fxzz": (1, 0, 2),
    "fyzz": (0, 1, 2),
    "fxyz": (1, 1, 1),
    "gxxxx": (4, 0, 0),
    "gyyyy": (0, 4, 0),
    "gzzzz": (0, 0, 4),
    "gxxxy": (3, 1, 0),
    "gxxxz": (3, 0, 1),
    "gxyyy": (1, 3, 0),
    "gyyyz": (0, 3, 1),
    "gxzzz": (1, 0, 3),
    "gyzzz": (0, 1, 3),
    "gxxyy": (2, 2, 0),
    "gxxzz": (2, 0, 2),
    "gyyzz": (0, 2, 2),
    "gxxyz": (2, 1, 1),
    "gxyyz": (1, 2, 1),
    "gxyzz": (1, 1, 2),
}


def parse_powers(values: list[int] | tuple[int, int, int] | None, fallback_shell: str, name: str) -> Powers:
    if values is not None:
        if len(values) != 3:
            raise ValueError(f"{name} powers must have length 3")
        powers = tuple(int(v) for v in values)
    else:
        shell = fallback_shell.lower()
        if shell not in SHELL_TO_POWERS:
            allowed = ", ".join(sorted(SHELL_TO_POWERS))
            raise ValueError(f"Unknown {name} shell {fallback_shell!r}. Allowed: {allowed}")
        powers = SHELL_TO_POWERS[shell]

    if any(p < 0 for p in powers):
        raise ValueError(f"{name} powers must be nonnegative")

    return powers


def powers_to_label(powers: Powers) -> str:
    for label, p in SHELL_TO_POWERS.items():
        if p == powers:
            return label
    return f"x{powers[0]}y{powers[1]}z{powers[2]}"


# =============================================================================
# EZFIO readers and QP basis loading
# =============================================================================

def read_ezfio_gz_numeric_array(path: str | Path, dtype=float) -> np.ndarray:
    path = Path(path)
    with gzip.open(path, "rt") as handle:
        tokens = handle.read().split()

    if not tokens:
        raise ValueError(f"Empty EZFIO file: {path}")

    rank = int(tokens[0])
    dims = tuple(int(tok) for tok in tokens[1 : 1 + rank])
    raw_values = tokens[1 + rank :]
    expected = int(np.prod(dims))

    if len(raw_values) != expected:
        raise ValueError(
            f"File {path} contains {len(raw_values)} values, expected {expected} for dims={dims}"
        )

    values = np.array([dtype(v) for v in raw_values])
    return values.reshape(dims, order="F")


def read_ezfio_gz_string_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    with gzip.open(path, "rt") as handle:
        tokens = handle.read().split()

    if not tokens:
        raise ValueError(f"Empty EZFIO file: {path}")

    rank = int(tokens[0])
    dims = tuple(int(tok) for tok in tokens[1 : 1 + rank])
    raw_values = tokens[1 + rank :]
    expected = int(np.prod(dims))

    if len(raw_values) != expected:
        raise ValueError(
            f"File {path} contains {len(raw_values)} strings, expected {expected} for dims={dims}"
        )

    values = np.array(raw_values, dtype=object)
    return values.reshape(dims, order="F")


def make_ao_label(index: int, powers: Powers) -> str:
    return f"AO_{index:03d}_{powers_to_label(powers)}"


def load_qp_ao_basis(
    ao_coef_path: str | Path,
    ao_expo_path: str | Path,
    ao_power_path: str | Path,
    normalized: bool = True,
    zero_tol: float = 1e-14,
) -> AOBasis:
    coef = read_ezfio_gz_numeric_array(ao_coef_path, dtype=float)
    expo = read_ezfio_gz_numeric_array(ao_expo_path, dtype=float)
    power = read_ezfio_gz_numeric_array(ao_power_path, dtype=int)

    if coef.shape != expo.shape:
        raise ValueError(f"ao_coef shape {coef.shape} != ao_expo shape {expo.shape}")
    if power.ndim != 2 or power.shape[1] != 3:
        raise ValueError(f"ao_power should have shape (ao_num, 3), got {power.shape}")
    if power.shape[0] != coef.shape[0]:
        raise ValueError(f"ao_power AO count {power.shape[0]} != ao_coef AO count {coef.shape[0]}")

    ao_num, max_prim = coef.shape
    aos: list[ContractedAO] = []

    for iao in range(ao_num):
        powers = tuple(int(v) for v in power[iao, :])
        primitives: list[AOPrimitive] = []

        for iprim in range(max_prim):
            c = float(coef[iao, iprim])
            a = float(expo[iao, iprim])
            if abs(c) <= zero_tol or abs(a) <= zero_tol:
                continue
            primitives.append(AOPrimitive(alpha=a, coefficient=c))

        if not primitives:
            raise ValueError(f"AO {iao} has no nonzero primitives")

        aos.append(
            ContractedAO(
                label=make_ao_label(iao, powers),
                powers=powers,
                primitives=tuple(primitives),
                normalized=normalized,
            )
        )

    return AOBasis(aos=tuple(aos), label="QP_AO_basis")


def load_qp_mo_basis(
    ao_basis: AOBasis,
    mo_coef_path: str | Path,
    mo_occ_path: str | Path | None = None,
    mo_class_path: str | Path | None = None,
) -> tuple[MOBasis, np.ndarray | None, np.ndarray | None]:
    mo_coef = read_ezfio_gz_numeric_array(mo_coef_path, dtype=float)

    if mo_coef.ndim != 2:
        raise ValueError(f"mo_coef must be rank 2, got shape {mo_coef.shape}")
    if mo_coef.shape[0] != ao_basis.size:
        raise ValueError(
            f"mo_coef first dimension {mo_coef.shape[0]} does not match AO basis size {ao_basis.size}. "
            f"Read shape is {mo_coef.shape}; expected (ao_num, mo_num)."
        )

    n_mo = mo_coef.shape[1]

    mo_occ = None
    if mo_occ_path is not None:
        mo_occ = read_ezfio_gz_numeric_array(mo_occ_path, dtype=float)
        if mo_occ.shape != (n_mo,):
            raise ValueError(f"mo_occ shape {mo_occ.shape} does not match n_mo={n_mo}")

    mo_class = None
    if mo_class_path is not None:
        mo_class = read_ezfio_gz_string_array(mo_class_path)
        if mo_class.shape != (n_mo,):
            raise ValueError(f"mo_class shape {mo_class.shape} does not match n_mo={n_mo}")

    labels = []
    for imo in range(n_mo):
        parts = [f"MO_{imo:03d}"]
        if mo_occ is not None:
            parts.append(f"occ={mo_occ[imo]:.3g}")
        if mo_class is not None:
            parts.append(str(mo_class[imo]))
        labels.append("_".join(parts))

    return MOBasis(ao_basis=ao_basis, coefficients=mo_coef, labels=tuple(labels)), mo_occ, mo_class


# =============================================================================
# Interactive helpers
# =============================================================================

def ask_text(prompt: str, default: str | None = None) -> str:
    if default is None:
        raw = input(f"{prompt}: ").strip()
    else:
        raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw else ("" if default is None else default)


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = ask_text(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print("Please enter an integer.")


def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = ask_text(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print("Please enter a number.")


def ask_yes_no(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "1", "true"):
            return True
        if raw in ("n", "no", "0", "false"):
            return False
        print("Please answer y or n.")


def ask_choice(prompt: str, choices: list[tuple[str, str]], default_key: str) -> str:
    print(f"\n{prompt}")
    keys = {key for key, _desc in choices}
    for key, desc in choices:
        default_marker = "  <default>" if key == default_key else ""
        print(f"  {key}) {desc}{default_marker}")

    while True:
        raw = ask_text("Choice", default_key).lower()
        if raw in keys:
            return raw
        print(f"Please choose one of: {', '.join(sorted(keys))}")


def ask_vector3(prompt: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    default_text = f"{default[0]} {default[1]} {default[2]}"
    while True:
        raw = ask_text(prompt, default_text)
        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            print("Please enter three numbers, e.g. 1 0 0")
            continue
        try:
            return tuple(float(x) for x in parts)  # type: ignore[return-value]
        except ValueError:
            print("Please enter three valid numbers.")


def ask_shell_or_powers(name: str, default_shell: str = "s") -> Powers:
    mode = ask_choice(
        f"Choose angular momentum for {name}",
        choices=[("s", "shell label"), ("p", "explicit powers a b c")],
        default_key="s",
    )
    if mode == "s":
        allowed_hint = "s, px, py, pz, dxy, dxz, dyz, dxx, dyy, dzz, ..."
        while True:
            shell = ask_text(f"{name} shell ({allowed_hint})", default_shell).lower()
            if shell in SHELL_TO_POWERS:
                return SHELL_TO_POWERS[shell]
            print(f"Unknown shell {shell!r}.")
    else:
        while True:
            raw = ask_text(f"{name} powers a b c", "0 0 0")
            parts = raw.replace(",", " ").split()
            if len(parts) != 3:
                print("Please enter exactly three integers.")
                continue
            try:
                powers = tuple(int(x) for x in parts)
            except ValueError:
                print("Please enter integers.")
                continue
            if any(p < 0 for p in powers):
                print("Powers must be nonnegative.")
                continue
            return powers  # type: ignore[return-value]


def ask_precision_preset() -> tuple[float, float]:
    choice = ask_choice(
        "Precision preset",
        choices=[
            ("q", "quick      epsabs=epsrel=1e-7"),
            ("n", "normal     epsabs=epsrel=1e-10"),
            ("s", "strict     epsabs=epsrel=1e-12"),
            ("c", "custom"),
        ],
        default_key="q",
    )
    if choice == "q":
        return 1e-7, 1e-7
    if choice == "n":
        return 1e-10, 1e-10
    if choice == "s":
        return 1e-12, 1e-12
    epsabs = ask_float("epsabs", 1e-10)
    epsrel = ask_float("epsrel", 1e-10)
    return epsabs, epsrel


def make_plane_wave_args_interactive():
    unit = ask_choice(
        "Plane-wave energy unit",
        choices=[("ha", "Hartree"), ("ev", "electron-volts")],
        default_key="ha",
    )
    if unit == "ha":
        energy = ask_float("Energy in Hartree", 2.0)
        energy_ev = None
    else:
        energy = None
        energy_ev = ask_float("Energy in eV", 54.422772492)

    direction = ask_vector3("Direction vector", (1.0, 0.0, 0.0))

    return SimpleNamespace(
        energy=energy,
        energy_ev=energy_ev,
        k=None,
        kvec=None,
        direction=direction,
    )


def make_common_args_interactive() -> SimpleNamespace:
    pw_args = make_plane_wave_args_interactive()
    epsabs, epsrel = ask_precision_preset()

    lmax_pw = ask_int("Plane-wave lmax_pw", 8)
    use_density = ask_yes_no("Use contracted density path", True)
    use_compress = ask_yes_no("Compress duplicate primitives", True)
    use_radial_table = ask_yes_no("Use radial table", True)
    show_radial_report = ask_yes_no("Show radial table report", True)
    hard_keys = ask_int("Number of hardest radial keys to print", 0)
    precompute = ask_yes_no("Precompute radial table before contraction", False)

    return SimpleNamespace(
        energy=pw_args.energy,
        energy_ev=pw_args.energy_ev,
        k=pw_args.k,
        kvec=pw_args.kvec,
        direction=pw_args.direction,
        lmax_pw=lmax_pw,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=300,
        max_contributions=5,
        no_density=not use_density,
        no_compress=not use_compress,
        compress_tol=1e-14,
        density_tol=1e-14,
        alpha_round_digits=12,
        no_radial_table=not use_radial_table,
        precompute_radial=precompute,
        hide_radial_report=not show_radial_report,
        hard_radial_keys=hard_keys,
    )


# =============================================================================
# Common execution helpers
# =============================================================================

def maybe_compress_orbital(orbital: ContractedGaussian, args) -> ContractedGaussian:
    if args.no_compress:
        return orbital
    compressed, report = compress_contracted_gaussian(
        orbital,
        coefficient_tol=args.compress_tol,
        alpha_round_digits=args.alpha_round_digits,
    )
    report.print()
    return compressed


def run_density_or_triple_path(
    phi_r: ContractedGaussian,
    phi_s1: ContractedGaussian,
    phi_s2: ContractedGaussian,
    plane_wave,
    args,
):
    print_primitive_cost_report(phi_r, phi_s1, phi_s2, title="Before optional compression")

    if not args.no_compress:
        print("\n=== Primitive compression reports ===")
        phi_r = maybe_compress_orbital(phi_r, args)
        phi_s1 = maybe_compress_orbital(phi_s1, args)
        phi_s2 = maybe_compress_orbital(phi_s2, args)
        print_primitive_cost_report(phi_r, phi_s1, phi_s2, title="After compression")

    if args.no_density:
        print("\n=== Evaluation path: old triple loop r × s1 × s2 ===")
        result = full_coulomb_plane_wave_integral(
            phi_r=phi_r,
            phi_s1=phi_s1,
            phi_s2=phi_s2,
            plane_wave=plane_wave,
            lmax_pw=args.lmax_pw,
            epsabs=args.epsabs,
            epsrel=args.epsrel,
            limit=args.limit,
            use_cache=True,
            max_contributions_store=args.max_contributions,
        )
        result.print(max_contributions=args.max_contributions)
        return result

    print("\n=== Evaluation path: contracted density r × rho(s) ===")
    density, density_report = build_contracted_density(
        phi_s1=phi_s1,
        phi_s2=phi_s2,
        coefficient_tol=args.density_tol,
        alpha_round_digits=args.alpha_round_digits,
    )
    density_report.print()
    print_density_cost_report(phi_r, density)

    result = full_coulomb_plane_wave_integral_density(
        phi_r=phi_r,
        density=density,
        plane_wave=plane_wave,
        lmax_pw=args.lmax_pw,
        epsabs=args.epsabs,
        epsrel=args.epsrel,
        limit=args.limit,
        use_radial_table=not args.no_radial_table,
        precompute_radial=args.precompute_radial,
        print_radial_report=not args.hide_radial_report,
        print_hard_keys=args.hard_radial_keys,
        alpha_round_digits=args.alpha_round_digits,
        max_contributions_store=args.max_contributions,
    )
    result.print(max_contributions=args.max_contributions)
    return result


# =============================================================================
# Mode implementations
# =============================================================================

def run_manual(args) -> None:
    plane_wave = plane_wave_from_cli_args(args)

    powers_r = parse_powers(args.powers_r, args.r_shell, "r")
    powers_s1 = parse_powers(args.powers_s1, args.s1_shell, "s1")
    powers_s2 = parse_powers(args.powers_s2, args.s2_shell, "s2")

    phi_r = ContractedGaussian.single(
        alpha=args.alpha_r,
        powers=powers_r,
        coefficient=args.coef_r,
        normalized=args.normalized,
        label=f"phi_r_{powers_to_label(powers_r)}",
    )
    phi_s1 = ContractedGaussian.single(
        alpha=args.alpha_s1,
        powers=powers_s1,
        coefficient=args.coef_s1,
        normalized=args.normalized,
        label=f"phi_s1_{powers_to_label(powers_s1)}",
    )
    phi_s2 = ContractedGaussian.single(
        alpha=args.alpha_s2,
        powers=powers_s2,
        coefficient=args.coef_s2,
        normalized=args.normalized,
        label=f"phi_s2_{powers_to_label(powers_s2)}",
    )

    print("\n=== Manual run setup ===")
    print(plane_wave.describe())
    print(f"lmax_pw = {args.lmax_pw}")
    print(f"powers_r  = {powers_r} ({powers_to_label(powers_r)})")
    print(f"powers_s1 = {powers_s1} ({powers_to_label(powers_s1)})")
    print(f"powers_s2 = {powers_s2} ({powers_to_label(powers_s2)})")
    print(phi_r.describe())
    print(phi_s1.describe())
    print(phi_s2.describe())

    run_density_or_triple_path(phi_r, phi_s1, phi_s2, plane_wave, args)


def run_qp_mo(args) -> None:
    plane_wave = plane_wave_from_cli_args(args)

    mo_occ_path = args.mo_occ if args.mo_occ and Path(args.mo_occ).exists() else None
    mo_class_path = args.mo_class if args.mo_class and Path(args.mo_class).exists() else None

    ao_basis = load_qp_ao_basis(
        ao_coef_path=args.ao_coef,
        ao_expo_path=args.ao_expo,
        ao_power_path=args.ao_power,
        normalized=not args.unnormalized,
        zero_tol=args.zero_tol,
    )
    mo_basis, mo_occ, mo_class = load_qp_mo_basis(
        ao_basis=ao_basis,
        mo_coef_path=args.mo_coef,
        mo_occ_path=mo_occ_path,
        mo_class_path=mo_class_path,
    )

    print("\n=== QP-MO run setup ===")
    print(plane_wave.describe())
    print(f"lmax_pw = {args.lmax_pw}")
    print(f"n_ao = {ao_basis.size}")
    print(f"n_mo = {mo_basis.n_mo}")
    print(f"mo_coef shape = {mo_basis.coefficients.shape}")

    if getattr(args, "list_mos", False):
        print("\n=== MO list ===")
        for i in range(mo_basis.n_mo):
            print(f"{i:4d}: {mo_basis.mo_label(i)}")

    phi_r = mo_orbital(mo_basis, args.r_mo, drop_tol=args.drop_tol)
    phi_s1 = mo_orbital(mo_basis, args.s1_mo, drop_tol=args.drop_tol)
    phi_s2 = mo_orbital(mo_basis, args.s2_mo, drop_tol=args.drop_tol)

    print("\n=== Selected QP MOs ===")
    print(f"r  MO index = {args.r_mo}, label={mo_basis.mo_label(args.r_mo)}")
    print(f"s1 MO index = {args.s1_mo}, label={mo_basis.mo_label(args.s1_mo)}")
    print(f"s2 MO index = {args.s2_mo}, label={mo_basis.mo_label(args.s2_mo)}")

    if args.print_orbitals:
        print("\n--- phi_r ---")
        print(phi_r.describe(max_primitives=20))
        print("\n--- phi_s1 ---")
        print(phi_s1.describe(max_primitives=20))
        print("\n--- phi_s2 ---")
        print(phi_s2.describe(max_primitives=20))

    run_density_or_triple_path(phi_r, phi_s1, phi_s2, plane_wave, args)


def run_angular(args) -> None:
    powers_r = parse_powers(args.powers_r, args.r_shell, "r")
    powers_s1 = parse_powers(args.powers_s1, args.s1_shell, "s1")
    powers_s2 = parse_powers(args.powers_s2, args.s2_shell, "s2")
    powers_s_total = product_powers(powers_s1, powers_s2)

    print("\n=== Angular-only setup ===")
    print(f"powers_r       = {powers_r} ({powers_to_label(powers_r)})")
    print(f"powers_s1      = {powers_s1} ({powers_to_label(powers_s1)})")
    print(f"powers_s2      = {powers_s2} ({powers_to_label(powers_s2)})")
    print(f"powers_s_total = {powers_s_total} ({powers_to_label(powers_s_total)})")
    print(f"lmax_pw        = {args.lmax_pw}")

    r_channels, s_channels, pw_channels, couplings = angular_pipeline(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        lmax_pw=args.lmax_pw,
    )

    print_channels("r-side orbital channels", r_channels)
    print_channels("s-side density channels", s_channels)

    if not args.hide_plane:
        print_channels("plane-wave angular channels", pw_channels)

    print_allowed_couplings(couplings, max_print=args.max_print)


# =============================================================================
# Interactive mode implementations
# =============================================================================

def interactive_manual() -> None:
    common = make_common_args_interactive()
    powers_r = ask_shell_or_powers("phi_r", "s")
    powers_s1 = ask_shell_or_powers("phi_s1", "s")
    powers_s2 = ask_shell_or_powers("phi_s2", "s")

    args = SimpleNamespace(
        **common.__dict__,
        r_shell=powers_to_label(powers_r),
        s1_shell=powers_to_label(powers_s1),
        s2_shell=powers_to_label(powers_s2),
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        alpha_r=ask_float("alpha_r", 1.0),
        alpha_s1=ask_float("alpha_s1", 1.0),
        alpha_s2=ask_float("alpha_s2", 1.0),
        coef_r=ask_float("coefficient r", 1.0),
        coef_s1=ask_float("coefficient s1", 1.0),
        coef_s2=ask_float("coefficient s2", 1.0),
        normalized=ask_yes_no("Normalize primitives", False),
    )
    run_manual(args)


def interactive_angular() -> None:
    powers_r = ask_shell_or_powers("phi_r", "s")
    powers_s1 = ask_shell_or_powers("phi_s1", "px")
    powers_s2 = ask_shell_or_powers("phi_s2", "s")
    args = SimpleNamespace(
        powers_r=powers_r,
        powers_s1=powers_s1,
        powers_s2=powers_s2,
        r_shell=powers_to_label(powers_r),
        s1_shell=powers_to_label(powers_s1),
        s2_shell=powers_to_label(powers_s2),
        lmax_pw=ask_int("Plane-wave lmax_pw", 4),
        max_print=ask_int("Max couplings to print", 30),
        hide_plane=not ask_yes_no("Print plane-wave channels", True),
    )
    run_angular(args)


def interactive_qp_mo() -> None:
    common = make_common_args_interactive()

    ao_coef = ask_text("Path to ao_coef.gz", "ao_coef.gz")
    ao_expo = ask_text("Path to ao_expo.gz", "ao_expo.gz")
    ao_power = ask_text("Path to ao_power.gz", "ao_power.gz")
    mo_coef = ask_text("Path to mo_coef.gz", "mo_coef.gz")
    mo_occ = ask_text("Path to mo_occ.gz", "mo_occ.gz")
    mo_class = ask_text("Path to mo_class.gz", "mo_class.gz")

    print("\nLoading QP basis to show MO labels...")
    ao_basis = load_qp_ao_basis(ao_coef, ao_expo, ao_power, normalized=True)
    mo_occ_path = mo_occ if Path(mo_occ).exists() else None
    mo_class_path = mo_class if Path(mo_class).exists() else None
    mo_basis, occ, klass = load_qp_mo_basis(ao_basis, mo_coef, mo_occ_path, mo_class_path)

    print(f"n_ao = {ao_basis.size}")
    print(f"n_mo = {mo_basis.n_mo}")
    print("\nAvailable MOs:")
    for i in range(mo_basis.n_mo):
        print(f"  {i:4d}: {mo_basis.mo_label(i)}")

    args = SimpleNamespace(
        **common.__dict__,
        ao_coef=ao_coef,
        ao_expo=ao_expo,
        ao_power=ao_power,
        mo_coef=mo_coef,
        mo_occ=mo_occ,
        mo_class=mo_class,
        r_mo=ask_int("r MO index", 0),
        s1_mo=ask_int("s1 MO index", 0),
        s2_mo=ask_int("s2 MO index", 0),
        unnormalized=not ask_yes_no("Normalize AO primitives", True),
        zero_tol=1e-14,
        drop_tol=ask_float("Drop MO AO coefficients below", 1e-12),
        print_orbitals=ask_yes_no("Print flattened orbitals", False),
        list_mos=False,
    )
    run_qp_mo(args)


def run_interactive(_args=None) -> None:
    choice = ask_choice(
        "Choose run mode",
        choices=[
            ("m", "manual primitive integral"),
            ("q", "QP MO integral"),
            ("a", "angular diagnostic only"),
        ],
        default_key="q",
    )
    if choice == "m":
        interactive_manual()
    elif choice == "q":
        interactive_qp_mo()
    elif choice == "a":
        interactive_angular()
    else:
        raise RuntimeError("unreachable")


# =============================================================================
# CLI construction
# =============================================================================

def add_common_numeric_args(parser: argparse.ArgumentParser) -> None:
    add_plane_wave_cli_arguments(parser)
    parser.add_argument("--lmax-pw", type=int, default=8)
    parser.add_argument("--epsabs", type=float, default=1e-10)
    parser.add_argument("--epsrel", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-contributions", type=int, default=10)

    parser.add_argument("--no-density", action="store_true", help="use old r*s1*s2 loop")
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--compress-tol", type=float, default=1e-14)
    parser.add_argument("--density-tol", type=float, default=1e-14)
    parser.add_argument("--alpha-round-digits", type=int, default=12)

    parser.add_argument("--no-radial-table", action="store_true")
    parser.add_argument("--precompute-radial", action="store_true")
    parser.add_argument("--hide-radial-report", action="store_true")
    parser.add_argument("--hard-radial-keys", type=int, default=0)


def add_shell_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--r-shell", default="s")
    parser.add_argument("--s1-shell", default="s")
    parser.add_argument("--s2-shell", default="s")
    parser.add_argument("--powers-r", nargs=3, type=int, default=None)
    parser.add_argument("--powers-s1", nargs=3, type=int, default=None)
    parser.add_argument("--powers-s2", nargs=3, type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic runner for atom-centered Coulomb / plane-wave integrals.")
    subparsers = parser.add_subparsers(dest="mode")

    interactive = subparsers.add_parser("interactive", help="ask questions dynamically")
    interactive.set_defaults(func=run_interactive)

    manual = subparsers.add_parser("manual", help="manual Cartesian Gaussian primitive integral")
    add_shell_args(manual)
    add_common_numeric_args(manual)
    manual.add_argument("--alpha-r", type=float, default=1.0)
    manual.add_argument("--alpha-s1", type=float, default=1.0)
    manual.add_argument("--alpha-s2", type=float, default=1.0)
    manual.add_argument("--coef-r", type=float, default=1.0)
    manual.add_argument("--coef-s1", type=float, default=1.0)
    manual.add_argument("--coef-s2", type=float, default=1.0)
    manual.add_argument("--normalized", action="store_true")
    manual.set_defaults(func=run_manual)

    qp = subparsers.add_parser("qp-mo", help="QP MO integral")
    add_common_numeric_args(qp)
    qp.add_argument("--ao-coef", default="ao_coef.gz")
    qp.add_argument("--ao-expo", default="ao_expo.gz")
    qp.add_argument("--ao-power", default="ao_power.gz")
    qp.add_argument("--mo-coef", default="mo_coef.gz")
    qp.add_argument("--mo-occ", default="mo_occ.gz")
    qp.add_argument("--mo-class", default="mo_class.gz")
    qp.add_argument("--r-mo", type=int, required=True)
    qp.add_argument("--s1-mo", type=int, required=True)
    qp.add_argument("--s2-mo", type=int, required=True)
    qp.add_argument("--unnormalized", action="store_true")
    qp.add_argument("--zero-tol", type=float, default=1e-14)
    qp.add_argument("--drop-tol", type=float, default=1e-12)
    qp.add_argument("--print-orbitals", action="store_true")
    qp.add_argument("--list-mos", action="store_true")
    qp.set_defaults(func=run_qp_mo)

    angular = subparsers.add_parser("angular", help="angular coupling diagnostic only")
    add_shell_args(angular)
    angular.add_argument("--lmax-pw", type=int, default=4)
    angular.add_argument("--max-print", type=int, default=30)
    angular.add_argument("--hide-plane", action="store_true")
    angular.set_defaults(func=run_angular)

    return parser


def main() -> None:
    parser = build_parser()

    # No arguments means dynamic interactive mode.
    if len(sys.argv) == 1:
        run_interactive()
        return

    args = parser.parse_args()

    if not hasattr(args, "func"):
        run_interactive()
        return

    args.func(args)


if __name__ == "__main__":
    main()
