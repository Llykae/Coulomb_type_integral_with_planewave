"""
validate_rectangular_mo_basis.py

Validation script for rectangular AO/MO coefficient matrices.

Why this matters
----------------
In real Quantum Package data, the number of AOs and MOs are not necessarily the
same.

The correct MO coefficient shape is

    C[ao_index, mo_index]

with shape

    (n_ao, n_mo)

and only the first dimension must match the AO basis size.

For example, the uploaded QP data has

    n_ao = 19
    n_mo = 18

so

    mo_coef.shape = (19, 18)

This is valid.

This script validates that our bookkeeping layer handles rectangular MO matrices
correctly.
"""

from __future__ import annotations

import numpy as np

from basis import AOBasis, AOPrimitive, ContractedAO, MOBasis, mo_orbital


# =============================================================================
# Build a fake rectangular test basis
# =============================================================================

def build_fake_ao_basis(n_ao: int) -> AOBasis:
    """
    Build a simple fake AO basis with n_ao atom-centered s-like primitives.

    The powers are all (0,0,0). This is enough to test bookkeeping.
    """
    aos = []
    for iao in range(n_ao):
        aos.append(
            ContractedAO(
                label=f"AO_{iao:03d}",
                powers=(0, 0, 0),
                primitives=(AOPrimitive(alpha=1.0 + 0.1 * iao, coefficient=1.0),),
                normalized=True,
            )
        )
    return AOBasis(aos=tuple(aos), label=f"fake_{n_ao}_AO_basis")


def build_fake_rectangular_mo_basis(n_ao: int, n_mo: int) -> MOBasis:
    """
    Build a fake rectangular MO basis with shape (n_ao, n_mo).
    """
    ao_basis = build_fake_ao_basis(n_ao)

    coeff = np.zeros((n_ao, n_mo))

    # Fill with a deterministic pattern that is clearly rectangular-safe.
    # Each MO gets at least one nonzero AO coefficient.
    for imo in range(n_mo):
        primary_ao = imo % n_ao
        coeff[primary_ao, imo] = 1.0

        secondary_ao = (imo + 1) % n_ao
        coeff[secondary_ao, imo] = 0.1 * (imo + 1)

    labels = tuple(f"MO_{imo:03d}" for imo in range(n_mo))

    return MOBasis(
        ao_basis=ao_basis,
        coefficients=coeff,
        labels=labels,
    )


# =============================================================================
# Tests
# =============================================================================

def assert_rectangular_basis_works(n_ao: int, n_mo: int) -> None:
    """
    Build a rectangular MO basis and check basic behavior.
    """
    print(f"\n=== Testing rectangular MO basis: n_ao={n_ao}, n_mo={n_mo} ===")

    mo_basis = build_fake_rectangular_mo_basis(n_ao, n_mo)

    assert mo_basis.n_ao == n_ao
    assert mo_basis.n_mo == n_mo
    assert mo_basis.coefficients.shape == (n_ao, n_mo)

    print(mo_basis.describe())
    print(f"coefficient shape = {mo_basis.coefficients.shape}")

    # Make sure the last MO index is valid even if n_mo != n_ao.
    last_mo_index = n_mo - 1
    mo = mo_orbital(mo_basis, last_mo_index, drop_tol=0.0)

    print(f"Flattened last MO index {last_mo_index}:")
    print(mo.describe())

    # Make sure out-of-range MO index fails against n_mo, not n_ao.
    try:
        mo_orbital(mo_basis, n_mo, drop_tol=0.0)
    except IndexError:
        print("Correctly rejected mo_index == n_mo")
    else:
        raise AssertionError("mo_orbital should reject mo_index == n_mo")


def main() -> None:
    # Case similar to uploaded QP data: more AOs than MOs.
    assert_rectangular_basis_works(n_ao=19, n_mo=18)

    # Opposite case: more MOs than AOs. This can happen in other contexts or
    # generated test data, and the bookkeeping should still be valid.
    assert_rectangular_basis_works(n_ao=4, n_mo=6)

    print("\n=== Summary ===")
    print("Rectangular AO/MO coefficient handling is OK.")


if __name__ == "__main__":
    main()
