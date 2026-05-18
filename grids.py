"""
grids.py

Grid utilities for the Coulomb / plane-wave project.

Goal
----
This module provides small, reusable helpers for building Cartesian grids.

For now we focus on 2D slices through 3D space because they are ideal for
plotting and debugging:

    xy plane: z = constant
    xz plane: y = constant
    yz plane: x = constant

The plane wave and solid-harmonic expansion are still 3D functions. A 2D grid is
just a slice through the 3D coordinates.

Why separate this from plane_wave_expansion.py?
-----------------------------------------------
The expansion module should only know about mathematics. Grid generation and
plotting are support utilities. Keeping them separate makes the project easier
to test and extend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    """
    A 2D slice through 3D Cartesian space.

    Attributes
    ----------
    X, Y, Z
        NumPy arrays with the same shape. Together they represent points

            r = (X[i,j], Y[i,j], Z[i,j])

        on a 2D slice.

    horizontal_name
        Name of the coordinate shown on the horizontal plotting axis.

    vertical_name
        Name of the coordinate shown on the vertical plotting axis.

    fixed_name
        Name of the coordinate held fixed.

    fixed_value
        Value of the fixed coordinate.

    Example
    -------
    An xz grid at y=0 has

        horizontal_name = "x"
        vertical_name   = "z"
        fixed_name      = "y"
        fixed_value     = 0
    """

    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    horizontal_name: str
    vertical_name: str
    fixed_name: str
    fixed_value: float

    @property
    def shape(self) -> tuple[int, int]:
        """Return the 2D grid shape."""
        return self.X.shape

    @property
    def R(self) -> np.ndarray:
        """Return radius r = sqrt(x^2 + y^2 + z^2) on the grid."""
        return np.sqrt(self.X**2 + self.Y**2 + self.Z**2)

    @property
    def rmax(self) -> float:
        """Largest radius present on the grid."""
        return float(np.max(self.R))

    @property
    def extent_for_imshow(self) -> list[float]:
        """
        Return matplotlib imshow extent.

        imshow expects

            [horizontal_min, horizontal_max, vertical_min, vertical_max]
        """
        H = self.horizontal_values
        V = self.vertical_values
        return [float(H.min()), float(H.max()), float(V.min()), float(V.max())]

    @property
    def horizontal_values(self) -> np.ndarray:
        """Return the coordinate array used as the horizontal plotting axis."""
        if self.horizontal_name == "x":
            return self.X
        if self.horizontal_name == "y":
            return self.Y
        if self.horizontal_name == "z":
            return self.Z
        raise ValueError(f"Unknown horizontal coordinate {self.horizontal_name!r}")

    @property
    def vertical_values(self) -> np.ndarray:
        """Return the coordinate array used as the vertical plotting axis."""
        if self.vertical_name == "x":
            return self.X
        if self.vertical_name == "y":
            return self.Y
        if self.vertical_name == "z":
            return self.Z
        raise ValueError(f"Unknown vertical coordinate {self.vertical_name!r}")

    def describe(self) -> str:
        """Return a short human-readable description of the grid."""
        return (
            f"{self.horizontal_name}{self.vertical_name}-grid, "
            f"{self.fixed_name}={self.fixed_value}, "
            f"shape={self.shape}, rmax={self.rmax:.6g}"
        )


def _axis(n: int, extent: float) -> np.ndarray:
    """
    Build one coordinate axis from -extent to +extent.

    Parameters
    ----------
    n
        Number of grid points.
    extent
        Half-width of the grid.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if extent <= 0:
        raise ValueError("extent must be positive")
    return np.linspace(-extent, extent, n)


def make_xz_grid(n: int = 201, extent: float = 0.1, y_value: float = 0.0) -> Grid2D:
    """
    Build an xz-plane grid at fixed y.

    This is useful when the plane wave has a z component, or when we want to see
    radial/angular structure involving z.
    """
    axis = _axis(n, extent)
    X, Z = np.meshgrid(axis, axis, indexing="xy")
    Y = np.full_like(X, y_value, dtype=float)
    return Grid2D(
        X=X,
        Y=Y,
        Z=Z,
        horizontal_name="x",
        vertical_name="z",
        fixed_name="y",
        fixed_value=float(y_value),
    )


def make_xy_grid(n: int = 201, extent: float = 0.1, z_value: float = 0.0) -> Grid2D:
    """
    Build an xy-plane grid at fixed z.
    """
    axis = _axis(n, extent)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    Z = np.full_like(X, z_value, dtype=float)
    return Grid2D(
        X=X,
        Y=Y,
        Z=Z,
        horizontal_name="x",
        vertical_name="y",
        fixed_name="z",
        fixed_value=float(z_value),
    )


def make_yz_grid(n: int = 201, extent: float = 0.1, x_value: float = 0.0) -> Grid2D:
    """
    Build a yz-plane grid at fixed x.
    """
    axis = _axis(n, extent)
    Y, Z = np.meshgrid(axis, axis, indexing="xy")
    X = np.full_like(Y, x_value, dtype=float)
    return Grid2D(
        X=X,
        Y=Y,
        Z=Z,
        horizontal_name="y",
        vertical_name="z",
        fixed_name="x",
        fixed_value=float(x_value),
    )


def make_grid(
    plane: str = "xz",
    n: int = 201,
    extent: float = 0.1,
    fixed_value: float = 0.0,
) -> Grid2D:
    """
    Generic grid factory.

    Parameters
    ----------
    plane
        One of "xy", "xz", or "yz".
    n
        Number of points per axis.
    extent
        Coordinates run from -extent to +extent along each plotted axis.
    fixed_value
        Value of the coordinate not shown on the plane.
    """
    plane = plane.lower()
    if plane == "xy":
        return make_xy_grid(n=n, extent=extent, z_value=fixed_value)
    if plane == "xz":
        return make_xz_grid(n=n, extent=extent, y_value=fixed_value)
    if plane == "yz":
        return make_yz_grid(n=n, extent=extent, x_value=fixed_value)
    raise ValueError("plane must be one of 'xy', 'xz', or 'yz'")


def estimate_lmax(k_abs: float, rmax: float, safety: int = 20) -> int:
    """
    Roughly estimate the angular momentum cutoff needed for a plane wave.

    The convergence parameter is approximately k*r.

    A practical first guess is

        lmax ≈ k*rmax + safety

    This is not a rigorous error bound. It is only a warning/starting point.
    Always check convergence by increasing lmax.
    """
    if k_abs < 0:
        raise ValueError("k_abs must be nonnegative")
    if rmax < 0:
        raise ValueError("rmax must be nonnegative")
    return int(np.ceil(k_abs * rmax + safety))


if __name__ == "__main__":
    grid = make_grid(plane="xz", n=101, extent=0.1, fixed_value=0.0)
    print(grid.describe())

    k_abs = 72.0
    print(f"For k={k_abs}, k*rmax={k_abs * grid.rmax:.6g}")
    print(f"Rough suggested lmax: {estimate_lmax(k_abs, grid.rmax)}")
