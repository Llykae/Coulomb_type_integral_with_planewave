"""
plane_wave_parameters.py

Define a plane wave from kinetic energy and propagation direction.

Atomic units
------------
In atomic units, for a free particle,

    E = k^2 / 2

so

    k = sqrt(2E)

where

    E is in Hartree
    k is in bohr^-1

This module lets the public scripts use

    --energy E --direction dx dy dz

instead of directly passing kx, ky, kz.

Important design point
----------------------
The direction components dx, dy, dz are arbitrary real numbers.
They are normalized internally.

We do NOT expand the final integral into explicit kx, ky, kz polynomials.

The r-dependence is expanded through solid harmonics into Cartesian
polynomials, while the k-direction dependence remains inside

    Y_lm*(khat).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# CODATA value, enough precision for our use.
HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class PlaneWaveParameters:
    """
    Plane-wave parameters.

    Attributes
    ----------
    energy_hartree
        Plane-wave kinetic energy in Hartree.

    k_abs
        Magnitude |k| in bohr^-1.

    direction
        Unit vector khat.

    kvec
        Full wave vector k = |k| khat.
    """

    energy_hartree: float
    k_abs: float
    direction: np.ndarray
    kvec: np.ndarray

    @property
    def energy_ev(self) -> float:
        """Return the kinetic energy in electron-volts."""
        return self.energy_hartree * HARTREE_TO_EV

    def describe(self) -> str:
        """Return a compact human-readable description."""
        return (
            "PlaneWaveParameters("
            f"E={self.energy_hartree:.12g} Ha, "
            f"E={self.energy_ev:.12g} eV, "
            f"|k|={self.k_abs:.12g} bohr^-1, "
            f"direction={self.direction.tolist()}, "
            f"kvec={self.kvec.tolist()}"
            ")"
        )


# =============================================================================
# Energy / k conversion
# =============================================================================

def energy_hartree_to_k_abs(energy_hartree: float) -> float:
    """
    Convert kinetic energy in Hartree to |k| in bohr^-1.

    Atomic-unit relation:

        E = k^2 / 2

    therefore

        k = sqrt(2E).
    """
    if energy_hartree < 0:
        raise ValueError("Plane-wave kinetic energy must be nonnegative")

    return float(np.sqrt(2.0 * energy_hartree))


def k_abs_to_energy_hartree(k_abs: float) -> float:
    """
    Convert |k| in bohr^-1 to kinetic energy in Hartree.

        E = k^2 / 2
    """
    if k_abs < 0:
        raise ValueError("|k| must be nonnegative")

    return float(0.5 * k_abs * k_abs)


def energy_ev_to_hartree(energy_ev: float) -> float:
    """Convert electron-volts to Hartree."""
    if energy_ev < 0:
        raise ValueError("Plane-wave kinetic energy must be nonnegative")

    return float(energy_ev / HARTREE_TO_EV)


def energy_hartree_to_ev(energy_hartree: float) -> float:
    """Convert Hartree to electron-volts."""
    if energy_hartree < 0:
        raise ValueError("Plane-wave kinetic energy must be nonnegative")

    return float(energy_hartree * HARTREE_TO_EV)


# =============================================================================
# Direction handling
# =============================================================================

def normalize_direction(
    direction: tuple[float, float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    """
    Normalize a real-valued 3D direction vector.

    The input does not need to have unit norm.

    Examples
    --------
    These are equivalent directions:

        (1, 0, 0)
        (2, 0, 0)
        (10, 0, 0)

    A non-axis direction is also allowed:

        (0.3, 0.4, 0.8660254)
    """
    vec = np.asarray(direction, dtype=float)

    if vec.shape != (3,):
        raise ValueError("direction must have length 3")

    norm = np.linalg.norm(vec)

    if norm == 0.0:
        raise ValueError("direction cannot be the zero vector")

    return vec / norm


# =============================================================================
# Plane-wave constructors
# =============================================================================

def plane_wave_from_energy_hartree(
    energy_hartree: float,
    direction: tuple[float, float, float] | list[float] | np.ndarray = (1.0, 0.0, 0.0),
) -> PlaneWaveParameters:
    """
    Build PlaneWaveParameters from kinetic energy in Hartree.

    Parameters
    ----------
    energy_hartree
        Kinetic energy in Hartree.

    direction
        Real-valued propagation direction. It is normalized internally.
    """
    unit_direction = normalize_direction(direction)
    k_abs = energy_hartree_to_k_abs(energy_hartree)
    kvec = k_abs * unit_direction

    return PlaneWaveParameters(
        energy_hartree=float(energy_hartree),
        k_abs=float(k_abs),
        direction=unit_direction,
        kvec=kvec,
    )


def plane_wave_from_energy_ev(
    energy_ev: float,
    direction: tuple[float, float, float] | list[float] | np.ndarray = (1.0, 0.0, 0.0),
) -> PlaneWaveParameters:
    """
    Build PlaneWaveParameters from kinetic energy in electron-volts.
    """
    energy_hartree = energy_ev_to_hartree(energy_ev)

    return plane_wave_from_energy_hartree(
        energy_hartree=energy_hartree,
        direction=direction,
    )


def plane_wave_from_k_abs(
    k_abs: float,
    direction: tuple[float, float, float] | list[float] | np.ndarray = (1.0, 0.0, 0.0),
) -> PlaneWaveParameters:
    """
    Build PlaneWaveParameters from |k| directly.

    This is mostly for backward compatibility with older scripts.
    Preferred public input is energy.
    """
    if k_abs < 0:
        raise ValueError("|k| must be nonnegative")

    unit_direction = normalize_direction(direction)
    energy_hartree = k_abs_to_energy_hartree(k_abs)
    kvec = k_abs * unit_direction

    return PlaneWaveParameters(
        energy_hartree=float(energy_hartree),
        k_abs=float(k_abs),
        direction=unit_direction,
        kvec=kvec,
    )


# =============================================================================
# CLI helpers
# =============================================================================

def add_plane_wave_cli_arguments(parser) -> None:
    """
    Add common plane-wave command-line arguments to an argparse parser.

    Preferred usage:

        --energy E --direction dx dy dz

    or

        --energy-ev Eev --direction dx dy dz

    Backward compatibility:

        --k K --direction dx dy dz
    """
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--energy",
        type=float,
        default=None,
        help="plane-wave kinetic energy in Hartree",
    )

    group.add_argument(
        "--energy-ev",
        type=float,
        default=None,
        help="plane-wave kinetic energy in electron-volts",
    )

    group.add_argument(
        "--k",
        type=float,
        default=None,
        help="direct |k| in bohr^-1; kept for backward compatibility",
    )

    parser.add_argument(
        "--direction",
        nargs=3,
        type=float,
        default=(1.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="real-valued propagation direction; normalized internally",
    )


def plane_wave_from_cli_args(args) -> PlaneWaveParameters:
    """
    Build PlaneWaveParameters from command-line arguments.

    Expected possible attributes
    ----------------------------
    args.energy
        Energy in Hartree.

    args.energy_ev
        Energy in electron-volts.

    args.k
        Direct |k| in bohr^-1, backward compatibility.

    args.direction
        Direction vector, three real numbers.

    Priority
    --------
    1. --energy
    2. --energy-ev
    3. --k
    4. default |k| = 2
    """
    direction = getattr(args, "direction", (1.0, 0.0, 0.0))

    energy = getattr(args, "energy", None)
    energy_ev = getattr(args, "energy_ev", None)
    k_abs = getattr(args, "k", None)

    if energy is not None:
        return plane_wave_from_energy_hartree(
            energy_hartree=energy,
            direction=direction,
        )

    if energy_ev is not None:
        return plane_wave_from_energy_ev(
            energy_ev=energy_ev,
            direction=direction,
        )

    if k_abs is not None:
        return plane_wave_from_k_abs(
            k_abs=k_abs,
            direction=direction,
        )

    # Backward-compatible default:
    # old scripts used |k| = 2, which corresponds to E = 2 Hartree.
    return plane_wave_from_k_abs(
        k_abs=2.0,
        direction=direction,
    )


# =============================================================================
# Command-line test
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Plane-wave energy / direction helper."
    )

    add_plane_wave_cli_arguments(parser)

    args = parser.parse_args()

    plane_wave = plane_wave_from_cli_args(args)

    print(plane_wave.describe())


if __name__ == "__main__":
    main()