"""
integral_from_qp_mo.py

QP-MO driver using the optimized contracted-density path by default.

Target integral
---------------

    I_ijk(k) = ∫∫ d^3r d^3s
               exp(i k.r)
               phi_i(r)
               1/|r-s|
               phi_j(s) phi_k(s)

Each phi is a Quantum Package molecular orbital:

    phi_i(r) = sum_mu C_{mu,i} chi_mu(r).

Input files
-----------
This script reads QP / EZFIO compressed array files:

    ao_coef.gz
    ao_expo.gz
    ao_power.gz
    mo_coef.gz
    mo_occ.gz      optional
    mo_class.gz    optional

Array convention
----------------
The files are EZFIO-style arrays:

    rank
    dim1 dim2 ... dim_rank
    values...

Values are stored in Fortran order, so arrays are reshaped with order="F".

For the example data:

    ao_coef.gz  : shape (19, 12)
    ao_expo.gz  : shape (19, 12)
    ao_power.gz : shape (19, 3)
    mo_coef.gz  : shape (19, 18)
    mo_occ.gz   : shape (18,)
    mo_class.gz : shape (18,)

The MO coefficient convention is:

    mo_coef[ao_index, mo_index]

The number of AOs and MOs do not need to be equal.

Cost reductions
---------------
The original primitive loop is:

    r primitive × s1 primitive × s2 primitive.

This script reduces cost by:

1. flattening selected MOs into primitive contractions;
2. compressing duplicate primitives inside each MO;
3. building the s-side contracted density

       rho_jk(s) = phi_j(s) phi_k(s)

   for general Cartesian powers;
4. evaluating only

       r primitive × density primitive;

5. optionally using a radial integral table to diagnose/precompute radial jobs.

Useful commands
---------------
Fast density-path test:

    python3 integral_from_qp_mo.py \
      --ao-coef ao_coef.gz \
      --ao-expo ao_expo.gz \
      --ao-power ao_power.gz \
      --mo-coef mo_coef.gz \
      --mo-occ mo_occ.gz \
      --mo-class mo_class.gz \
      --r-mo 0 \
      --s1-mo 0 \
      --s2-mo 0 \
      --energy 2.0 \
      --direction 1.0 0.0 0.0 \
      --quiet-basis \
      --epsabs 1e-7 \
      --epsrel 1e-7 \
      --max-contributions 5 \
      --hard-radial-keys 10

Compare to old triple loop:

    add --no-density

Precompute radial table first:

    add --precompute-radial
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

from basis import AOPrimitive, AOBasis, ContractedAO, MOBasis, mo_orbital
from density_contraction import (
    build_contracted_density,
    print_density_cost_report,
)
from full_coulomb_integral import (
    full_coulomb_plane_wave_integral,
    print_primitive_cost_report,
)
from full_coulomb_integral_density import full_coulomb_plane_wave_integral_density
from integral_optimization import compress_contracted_gaussian
from plane_wave_parameters import add_plane_wave_cli_arguments, plane_wave_from_cli_args


# =============================================================================
# EZFIO compressed array readers
# =============================================================================

def read_ezfio_gz_numeric_array(path: str | Path, dtype=float) -> np.ndarray:
    """
    Read a numeric compressed EZFIO-style array.

    File format:

        rank
        dim1 dim2 ... dim_rank
        values...

    Values are reshaped in Fortran order.
    """
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
            f"File {path} contains {len(raw_values)} values, "
            f"expected {expected} for dims={dims}"
        )

    values = np.array([dtype(v) for v in raw_values])
    return values.reshape(dims, order="F")


def read_ezfio_gz_string_array(path: str | Path) -> np.ndarray:
    """
    Read a string compressed EZFIO-style array.

    Used for mo_class.gz.
    """
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
            f"File {path} contains {len(raw_values)} strings, "
            f"expected {expected} for dims={dims}"
        )

    values = np.array(raw_values, dtype=object)
    return values.reshape(dims, order="F")


# =============================================================================
# AO / MO basis loading
# =============================================================================

def angular_label_from_powers(powers: tuple[int, int, int]) -> str:
    """
    Return a compact Cartesian angular label from powers.

    Examples
    --------
    (0,0,0) -> s
    (1,0,0) -> px
    (0,1,0) -> py
    (0,0,1) -> pz
    (2,0,0) -> dxx
    (1,1,0) -> dxy
    """
    px, py, pz = powers
    total = px + py + pz

    if total == 0:
        return "s"

    prefix = {
        1: "p",
        2: "d",
        3: "f",
        4: "g",
    }.get(total, f"l{total}")

    return prefix + "x" * px + "y" * py + "z" * pz


def make_ao_label(index: int, powers: tuple[int, int, int]) -> str:
    """Build a stable AO label."""
    return f"AO_{index:03d}_{angular_label_from_powers(powers)}"


def load_qp_ao_basis(
    ao_coef_path: str | Path,
    ao_expo_path: str | Path,
    ao_power_path: str | Path,
    normalized: bool = True,
    zero_tol: float = 1e-14,
) -> AOBasis:
    """
    Load QP AO arrays into AOBasis.
    """
    coef = read_ezfio_gz_numeric_array(ao_coef_path, dtype=float)
    expo = read_ezfio_gz_numeric_array(ao_expo_path, dtype=float)
    power = read_ezfio_gz_numeric_array(ao_power_path, dtype=int)

    if coef.shape != expo.shape:
        raise ValueError(f"ao_coef shape {coef.shape} != ao_expo shape {expo.shape}")

    if power.ndim != 2 or power.shape[1] != 3:
        raise ValueError(f"ao_power should have shape (ao_num, 3), got {power.shape}")

    if power.shape[0] != coef.shape[0]:
        raise ValueError(
            f"ao_power AO count {power.shape[0]} != ao_coef AO count {coef.shape[0]}"
        )

    ao_num, max_prim = coef.shape
    aos: list[ContractedAO] = []

    for iao in range(ao_num):
        powers = tuple(int(v) for v in power[iao, :])
        primitives: list[AOPrimitive] = []

        for iprim in range(max_prim):
            c = float(coef[iao, iprim])
            a = float(expo[iao, iprim])

            # QP arrays are padded with zeros.
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
    """
    Load QP MO coefficient, occupation, and class arrays.

    The MO coefficient array must have shape:

        (n_ao, n_mo)

    Only the first dimension must match the AO basis size. n_ao and n_mo may be
    different.
    """
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

    labels: list[str] = []
    for imo in range(n_mo):
        parts = [f"MO_{imo:03d}"]
        if mo_occ is not None:
            parts.append(f"occ={mo_occ[imo]:.3g}")
        if mo_class is not None:
            parts.append(str(mo_class[imo]))
        labels.append("_".join(parts))

    mo_basis = MOBasis(
        ao_basis=ao_basis,
        coefficients=mo_coef,
        labels=tuple(labels),
    )

    return mo_basis, mo_occ, mo_class


# =============================================================================
# Diagnostics
# =============================================================================

def print_qp_basis_summary(
    ao_basis: AOBasis,
    mo_basis: MOBasis,
    mo_occ: np.ndarray | None,
    mo_class: np.ndarray | None,
    max_mos: int = 8,
) -> None:
    """
    Print compact AO/MO summary.
    """
    print("\n=== QP AO/MO summary ===")
    print(f"n_ao = {ao_basis.size}")
    print(f"n_mo = {mo_basis.n_mo}")
    print(f"MO coefficient shape = {mo_basis.coefficients.shape}")

    if mo_occ is not None:
        print(f"MO occupations = {mo_occ}")

    if mo_class is not None:
        unique = sorted(set(str(x) for x in mo_class))
        print(f"Unique MO classes = {unique}")

    print("\nFirst MO coefficient previews:")
    for imo in range(min(max_mos, mo_basis.n_mo)):
        coeff = mo_basis.coefficients[:, imo]
        nonzero = np.where(np.abs(coeff) > 1e-8)[0]
        preview = ", ".join(f"AO{mu}:{coeff[mu]:+.4e}" for mu in nonzero[:8])
        if len(nonzero) > 8:
            preview += ", ..."

        occ_text = "" if mo_occ is None else f" occ={mo_occ[imo]:.3g}"
        class_text = "" if mo_class is None else f" class={mo_class[imo]}"
        print(f"  MO {imo:3d}:{occ_text}{class_text}  {preview if preview else 'all ~ 0'}")


# =============================================================================
# Main driver
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QP MO Coulomb / plane-wave integral using contracted density and radial table."
    )

    # QP/EZFIO files.
    parser.add_argument("--ao-coef", default="ao_coef.gz")
    parser.add_argument("--ao-expo", default="ao_expo.gz")
    parser.add_argument("--ao-power", default="ao_power.gz")
    parser.add_argument("--mo-coef", default="mo_coef.gz")
    parser.add_argument("--mo-occ", default="mo_occ.gz")
    parser.add_argument("--mo-class", default="mo_class.gz")

    # MO selectors.
    parser.add_argument("--r-mo", type=int, required=True)
    parser.add_argument("--s1-mo", type=int, required=True)
    parser.add_argument("--s2-mo", type=int, required=True)

    # Basis / contraction controls.
    parser.add_argument("--unnormalized", action="store_true")
    parser.add_argument("--zero-tol", type=float, default=1e-14)
    parser.add_argument("--drop-tol", type=float, default=1e-12)

    # Primitive compression controls.
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--compress-tol", type=float, default=1e-14)
    parser.add_argument("--density-tol", type=float, default=1e-14)
    parser.add_argument("--alpha-round-digits", type=int, default=12)

    # Evaluation path controls.
    parser.add_argument(
        "--no-density",
        action="store_true",
        help="fall back to old r*s1*s2 triple loop",
    )

    # Radial-table controls. Only used by density path.
    parser.add_argument(
        "--no-radial-table",
        action="store_true",
        help="disable radial table in the density path",
    )
    parser.add_argument(
        "--precompute-radial",
        action="store_true",
        help="precompute all radial integrals before contraction",
    )
    parser.add_argument(
        "--hide-radial-report",
        action="store_true",
        help="do not print radial table report",
    )
    parser.add_argument(
        "--hard-radial-keys",
        type=int,
        default=0,
        help="print the N hardest-looking radial keys before evaluation",
    )

    # Plane wave controls.
    add_plane_wave_cli_arguments(parser)

    # Numerical controls.
    parser.add_argument("--lmax-pw", type=int, default=8)
    parser.add_argument("--epsabs", type=float, default=1e-10)
    parser.add_argument("--epsrel", type=float, default=1e-10)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-contributions", type=int, default=10)

    # Printing controls.
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--quiet-basis", action="store_true")
    parser.add_argument("--print-orbitals", action="store_true")

    args = parser.parse_args()

    plane_wave = plane_wave_from_cli_args(args)

    # Treat optional metadata files gracefully.
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

    if not args.quiet_basis:
        print_qp_basis_summary(ao_basis, mo_basis, mo_occ, mo_class)

    if args.summary_only:
        return

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

    print_primitive_cost_report(
        phi_r,
        phi_s1,
        phi_s2,
        title="Before MO primitive compression",
    )

    if not args.no_compress:
        phi_r, report_r = compress_contracted_gaussian(
            phi_r,
            coefficient_tol=args.compress_tol,
            alpha_round_digits=args.alpha_round_digits,
        )
        phi_s1, report_s1 = compress_contracted_gaussian(
            phi_s1,
            coefficient_tol=args.compress_tol,
            alpha_round_digits=args.alpha_round_digits,
        )
        phi_s2, report_s2 = compress_contracted_gaussian(
            phi_s2,
            coefficient_tol=args.compress_tol,
            alpha_round_digits=args.alpha_round_digits,
        )

        print("\n=== MO primitive compression reports ===")
        for report in (report_r, report_s1, report_s2):
            report.print()

        print_primitive_cost_report(
            phi_r,
            phi_s1,
            phi_s2,
            title="After MO primitive compression",
        )

    print("\n=== Plane wave ===")
    print(plane_wave.describe())

    # -------------------------------------------------------------------------
    # Old reference path: r × s1 × s2
    # -------------------------------------------------------------------------
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
        return

    # -------------------------------------------------------------------------
    # New default path: r × contracted density
    # -------------------------------------------------------------------------
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


if __name__ == "__main__":
    main()
