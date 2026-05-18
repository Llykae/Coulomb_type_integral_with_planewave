"""
basis_io_qp.py

Read Quantum Package / EZFIO AO basis arrays into the project's AOBasis format.

Context
-------
Quantum Package stores AO basis data in compressed EZFIO-style text arrays such
as:

    ao_coef.gz
    ao_expo.gz
    ao_power.gz

The files uploaded in this project have the structure:

    ao_coef.gz   rank 2, shape (ao_num, ao_prim_num)
    ao_expo.gz   rank 2, shape (ao_num, ao_prim_num)
    ao_power.gz  rank 2, shape (ao_num, 3)

For the uploaded example:

    ao_num      = 19
    ao_prim_num = 12

Important storage convention
----------------------------
The EZFIO text arrays are stored in Fortran order: the first index varies
fastest. Therefore, after reading all numbers, we reshape using

    order="F"

so that

    coef[ao_index, prim_index]
    expo[ao_index, prim_index]
    power[ao_index, xyz]

have the expected meaning.

Mapping to our project
----------------------
Each AO becomes a ContractedAO:

    ContractedAO(
        label="AO_000_s",
        powers=(0,0,0),
        primitives=(AOPrimitive(alpha=..., coefficient=...), ...),
        normalized=True,
    )

Zero coefficient/exponent padding is skipped.

This module only reads atom-centered AO basis data. It does not yet read MO
coefficients.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

from basis import AOPrimitive, AOBasis, ContractedAO


# =============================================================================
# Low-level EZFIO array reader
# =============================================================================

def read_ezfio_gz_array(path: str | Path, dtype=float) -> np.ndarray:
    """
    Read a compressed EZFIO-style text array.

    File format observed:

        rank
        dim1 dim2 ... dim_rank
        values...

    Values are reshaped using Fortran order because EZFIO/Fortran stores the
    first index fastest.
    """
    path = Path(path)

    with gzip.open(path, "rt") as handle:
        tokens = handle.read().split()

    if not tokens:
        raise ValueError(f"Empty EZFIO array file: {path}")

    rank = int(tokens[0])
    dims = tuple(int(tok) for tok in tokens[1 : 1 + rank])
    raw_values = tokens[1 + rank :]

    expected = int(np.prod(dims))
    if len(raw_values) != expected:
        raise ValueError(
            f"File {path} has {len(raw_values)} values, expected {expected} for dims={dims}"
        )

    values = np.array([dtype(v) for v in raw_values])
    return values.reshape(dims, order="F")


# =============================================================================
# AO labels
# =============================================================================

def angular_label_from_powers(powers: tuple[int, int, int]) -> str:
    """
    Build a simple Cartesian AO angular label from powers.

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

    xyz = "x" * px + "y" * py + "z" * pz
    return prefix + xyz


def make_ao_label(index: int, powers: tuple[int, int, int]) -> str:
    """Create a stable AO label."""
    return f"AO_{index:03d}_{angular_label_from_powers(powers)}"


# =============================================================================
# QP AO basis reader
# =============================================================================

def load_qp_ao_basis_from_arrays(
    ao_coef_path: str | Path,
    ao_expo_path: str | Path,
    ao_power_path: str | Path,
    normalized: bool = True,
    zero_tol: float = 1e-14,
    label: str = "QP_AO_basis",
) -> AOBasis:
    """
    Load AO basis from Quantum Package EZFIO array files.

    Parameters
    ----------
    ao_coef_path
        Path to ao_coef.gz.

    ao_expo_path
        Path to ao_expo.gz.

    ao_power_path
        Path to ao_power.gz.

    normalized
        Whether primitives should be normalized by the integral engine.

    zero_tol
        Skip primitives for which both coefficient and exponent are effectively
        zero, or coefficient alone is effectively zero.

    Returns
    -------
    AOBasis
    """
    coef = read_ezfio_gz_array(ao_coef_path, dtype=float)
    expo = read_ezfio_gz_array(ao_expo_path, dtype=float)
    power = read_ezfio_gz_array(ao_power_path, dtype=int)

    if coef.shape != expo.shape:
        raise ValueError(f"coef shape {coef.shape} != expo shape {expo.shape}")

    if power.ndim != 2 or power.shape[1] != 3:
        raise ValueError(f"power array should have shape (ao_num, 3), got {power.shape}")

    ao_num, prim_num = coef.shape
    if power.shape[0] != ao_num:
        raise ValueError(f"power has {power.shape[0]} AOs but coef/expo have {ao_num}")

    aos: list[ContractedAO] = []

    for iao in range(ao_num):
        powers = tuple(int(v) for v in power[iao, :])

        primitives: list[AOPrimitive] = []
        for iprim in range(prim_num):
            c = float(coef[iao, iprim])
            a = float(expo[iao, iprim])

            # QP arrays are padded with zeros up to max primitive count.
            if abs(c) <= zero_tol:
                continue
            if abs(a) <= zero_tol:
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

    return AOBasis(aos=tuple(aos), label=label)


# =============================================================================
# Diagnostics
# =============================================================================

def print_qp_array_summary(
    ao_coef_path: str | Path,
    ao_expo_path: str | Path,
    ao_power_path: str | Path,
) -> None:
    """Print a short summary of the raw QP arrays."""
    coef