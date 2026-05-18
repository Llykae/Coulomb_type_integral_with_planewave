"""
parity.py

Parity helpers for atom-centered Coulomb / plane-wave integrals.

All Gaussian orbitals are atom-centered. A Cartesian primitive has the form

    phi(r) = x^a y^b z^c exp(-alpha r^2).

The target integral is

    I(k) = ∫∫ d^3r d^3s
           exp(i k.r)
           phi_r(r)
           1/|r-s|
           phi_s1(s) phi_s2(s).

Because all orbitals are centered at the origin, parity gives useful selection
rules and debugging information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Powers = tuple[int, int, int]
AXES = ("x", "y", "z")


@dataclass(frozen=True)
class ParityReport:
    """Human-readable parity information for one atom-centered integral."""

    powers_r: Powers
    powers_s_total: Powers
    k_axis: str | None
    direct_static_parity: tuple[str, str, str]
    likely_real_part: str
    likely_imag_part: str
    notes: list[str]

    def print(self) -> None:
        print("\n=== Atom-centered parity report ===")
        print(f"powers_r        = {self.powers_r}")
        print(f"powers_s_total  = {self.powers_s_total}")
        print(f"k_axis          = {self.k_axis}")
        print(
            "static parity    = "
            f"x:{self.direct_static_parity[0]}, "
            f"y:{self.direct_static_parity[1]}, "
            f"z:{self.direct_static_parity[2]}"
        )
        print(f"likely Re(I)    = {self.likely_real_part}")
        print(f"likely Im(I)    = {self.likely_imag_part}")
        if self.notes:
            print("notes:")
            for note in self.notes:
                print(f"  - {note}")


def validate_powers(powers: Powers, name: str = "powers") -> Powers:
    """Validate a Cartesian power tuple."""
    if len(powers) != 3:
        raise ValueError(f"{name} must have length 3")
    if any(p < 0 for p in powers):
        raise ValueError(f"{name} must be nonnegative")
    return tuple(int(p) for p in powers)


def parity_label(power: int) -> str:
    """Return 'even' or 'odd' for an integer power."""
    return "even" if power % 2 == 0 else "odd"


def product_powers(powers_a: Powers, powers_b: Powers) -> Powers:
    """
    Powers of the product of two Cartesian monomials.

    If

        phi_a ~ x^a y^b z^c
        phi_b ~ x^d y^e z^f

    then

        phi_a phi_b ~ x^(a+d) y^(b+e) z^(c+f).
    """
    powers_a = validate_powers(powers_a, "powers_a")
    powers_b = validate_powers(powers_b, "powers_b")
    return tuple(p + q for p, q in zip(powers_a, powers_b))


def static_density_parity(powers_r: Powers, powers_s_total: Powers) -> tuple[str, str, str]:
    """
    Parity of the non-plane-wave polynomial part along each axis.

    Under simultaneous inversion of one Cartesian axis on r and s,

        (r_axis, s_axis) -> (-r_axis, -s_axis),

    the Coulomb kernel is even. The static parity is therefore controlled by

        powers_r[axis] + powers_s_total[axis].
    """
    powers_r = validate_powers(powers_r, "powers_r")
    powers_s_total = validate_powers(powers_s_total, "powers_s_total")

    total = tuple(pr + ps for pr, ps in zip(powers_r, powers_s_total))
    return tuple(parity_label(p) for p in total)


def detect_cartesian_k_axis(kvec: np.ndarray, tol: float = 1e-12) -> str | None:
    """
    Detect whether kvec lies along x, y, or z.

    Returns 'x', 'y', 'z', or None. Sign does not matter for parity.
    """
    kvec = np.asarray(kvec, dtype=float)

    if kvec.shape != (3,):
        raise ValueError("kvec must be length 3")

    nonzero = np.where(np.abs(kvec) > tol)[0]

    if len(nonzero) != 1:
        return None

    return AXES[int(nonzero[0])]


def predict_real_imag_for_axis_k(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    k_axis: str,
) -> tuple[str, str, list[str]]:
    """
    Predict whether real/imaginary parts are symmetry-allowed for k along one axis.

    If k is along x,

        exp(i k x_r) = cos(k x_r) + i sin(k x_r).

    The real part contributes even parity along x.
    The imaginary part contributes odd parity along x.
    """
    powers_r = validate_powers(powers_r, "powers_r")
    powers_s1 = validate_powers(powers_s1, "powers_s1")
    powers_s2 = validate_powers(powers_s2, "powers_s2")

    if k_axis not in AXES:
        raise ValueError("k_axis must be 'x', 'y', or 'z'")

    axis_index = AXES.index(k_axis)
    powers_s_total = product_powers(powers_s1, powers_s2)

    combined = [powers_r[i] + powers_s_total[i] for i in range(3)]

    real_combined = list(combined)

    imag_combined = list(combined)
    imag_combined[axis_index] += 1

    real_allowed = all(p % 2 == 0 for p in real_combined)
    imag_allowed = all(p % 2 == 0 for p in imag_combined)

    notes: list[str] = []
    notes.append(f"For k along {k_axis}, Re uses cos(k {k_axis}_r), which is even.")
    notes.append(f"For k along {k_axis}, Im uses sin(k {k_axis}_r), which is odd.")

    if real_allowed and not imag_allowed:
        notes.append("Symmetry predicts a mostly real integral.")
    elif imag_allowed and not real_allowed:
        notes.append("Symmetry predicts a mostly imaginary integral.")
    elif real_allowed and imag_allowed:
        notes.append("Both real and imaginary parts are symmetry-allowed.")
    else:
        notes.append("Both real and imaginary parts are symmetry-forbidden by this simple parity rule.")

    real_text = "allowed" if real_allowed else "forbidden / should vanish"
    imag_text = "allowed" if imag_allowed else "forbidden / should vanish"

    return real_text, imag_text, notes


def parity_report(
    powers_r: Powers,
    powers_s1: Powers,
    powers_s2: Powers,
    kvec: np.ndarray,
) -> ParityReport:
    """Build a parity report for an atom-centered integral."""
    powers_r = validate_powers(powers_r, "powers_r")
    powers_s1 = validate_powers(powers_s1, "powers_s1")
    powers_s2 = validate_powers(powers_s2, "powers_s2")

    powers_s_total = product_powers(powers_s1, powers_s2)
    static = static_density_parity(powers_r, powers_s_total)
    k_axis = detect_cartesian_k_axis(kvec)

    notes: list[str] = []

    if k_axis is None:
        likely_real = "unknown from simple axis rule"
        likely_imag = "unknown from simple axis rule"
        notes.append("k is not aligned with a single Cartesian axis.")
        notes.append("Simple cos/sin parity prediction is skipped.")
    else:
        likely_real, likely_imag, notes_axis = predict_real_imag_for_axis_k(
            powers_r=powers_r,
            powers_s1=powers_s1,
            powers_s2=powers_s2,
            k_axis=k_axis,
        )
        notes.extend(notes_axis)

    return ParityReport(
        powers_r=powers_r,
        powers_s_total=powers_s_total,
        k_axis=k_axis,
        direct_static_parity=static,
        likely_real_part=likely_real,
        likely_imag_part=likely_imag,
        notes=notes,
    )


if __name__ == "__main__":
    print("Example 1: all s orbitals, k along x")
    parity_report(
        powers_r=(0, 0, 0),
        powers_s1=(0, 0, 0),
        powers_s2=(0, 0, 0),
        kvec=np.array([2.0, 0.0, 0.0]),
    ).print()

    print("\nExample 2: s-side px times s, k along x")
    parity_report(
        powers_r=(0, 0, 0),
        powers_s1=(1, 0, 0),
        powers_s2=(0, 0, 0),
        kvec=np.array([2.0, 0.0, 0.0]),
    ).print()

    print("\nExample 3: r-side px, s-side s times s, k along x")
    parity_report(
        powers_r=(1, 0, 0),
        powers_s1=(0, 0, 0),
        powers_s2=(0, 0, 0),
        kvec=np.array([2.0, 0.0, 0.0]),
    ).print()