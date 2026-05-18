"""
plotting.py

Plotting utilities for the Coulomb / plane-wave project.

Goal
----
This module saves and optionally displays 2D maps for:

1. the exact plane wave,
2. the Cartesian solid-harmonic expansion approximation,
3. the absolute error between them.

It is deliberately independent from the math modules except for the expected
input arrays. The usual workflow is:

    grid = make_grid(...)
    exact = eval_exact_plane_wave(kvec, grid.X, grid.Y, grid.Z)
    approx = eval_plane_wave_cartesian_expansion(kvec, grid.X, grid.Y, grid.Z, lmax)
    plot_plane_wave_comparison(grid, exact, approx, ...)

Complex-valued functions cannot be shown directly as a single scalar image, so
we typically plot one of:

    real part
    imaginary part
    absolute value
    phase angle

For a pure exact plane wave, abs(exact) is always 1, so real/imag/phase are more
interesting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grids import Grid2D


# =============================================================================
# Array component selection
# =============================================================================

def select_component(values: np.ndarray, component: str) -> np.ndarray:
    """
    Convert a complex array into a real array suitable for plotting.

    Parameters
    ----------
    values
        Real or complex NumPy array.
    component
        One of:

            "real"   -> real part
            "imag"   -> imaginary part
            "abs"    -> magnitude
            "phase"  -> complex phase angle

    Returns
    -------
    numpy.ndarray
        Real-valued array.
    """
    component = component.lower()

    if component == "real":
        return np.real(values)
    if component == "imag":
        return np.imag(values)
    if component == "abs":
        return np.abs(values)
    if component == "phase":
        return np.angle(values)

    raise ValueError("component must be one of 'real', 'imag', 'abs', or 'phase'")


def component_label(component: str, name: str) -> str:
    """
    Human-readable colorbar label.
    """
    component = component.lower()
    if component == "real":
        return f"Re {name}"
    if component == "imag":
        return f"Im {name}"
    if component == "abs":
        return f"|{name}|"
    if component == "phase":
        return f"arg {name}"
    return name


# =============================================================================
# Basic image saving
# =============================================================================

def save_scalar_field(
    grid: Grid2D,
    array: np.ndarray,
    filename: str | Path,
    title: str,
    colorbar_label: str,
    show: bool = False,
) -> None:
    """
    Save one 2D scalar field as a PNG image.

    Parameters
    ----------
    grid
        Grid2D object defining coordinates and plotting extent.
    array
        Real-valued 2D data to plot.
    filename
        Output PNG path.
    title
        Plot title.
    colorbar_label
        Label for the colorbar.
    show
        If True, call plt.show() after saving.
    """
    import matplotlib.pyplot as plt

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        array,
        origin="lower",
        extent=grid.extent_for_imshow,
        aspect="equal",
    )
    plt.colorbar(label=colorbar_label)
    plt.xlabel(grid.horizontal_name)
    plt.ylabel(grid.vertical_name)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    print(f"Saved {filename}")

    if show:
        plt.show()
    else:
        plt.close()


# =============================================================================
# Plane-wave comparison plots
# =============================================================================

def plot_plane_wave_comparison(
    grid: Grid2D,
    exact: np.ndarray,
    approx: np.ndarray,
    lmax: int,
    component: str = "real",
    output_dir: str | Path = ".",
    prefix: str = "plane_wave",
    show: bool = False,
) -> None:
    """
    Save comparison plots for exact and approximate plane waves.

    The function writes three PNG files:

        {prefix}_exact_{component}.png
        {prefix}_approx_{component}_lmax{lmax}.png
        {prefix}_abs_error_lmax{lmax}.png

    The error plot always uses absolute error:

        |approx - exact|
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_component = select_component(exact, component)
    approx_component = select_component(approx, component)
    abs_error = np.abs(approx - exact)

    plane_desc = f"{grid.horizontal_name}{grid.vertical_name} plane, {grid.fixed_name}={grid.fixed_value}"

    save_scalar_field(
        grid=grid,
        array=exact_component,
        filename=output_dir / f"{prefix}_exact_{component}.png",
        title=f"Exact plane wave ({component}), {plane_desc}",
        colorbar_label=component_label(component, "exact"),
        show=False,
    )

    save_scalar_field(
        grid=grid,
        array=approx_component,
        filename=output_dir / f"{prefix}_approx_{component}_lmax{lmax}.png",
        title=f"Expansion ({component}), lmax={lmax}, {plane_desc}",
        colorbar_label=component_label(component, "approx"),
        show=False,
    )

    save_scalar_field(
        grid=grid,
        array=abs_error,
        filename=output_dir / f"{prefix}_abs_error_lmax{lmax}.png",
        title=f"Absolute error, lmax={lmax}, {plane_desc}",
        colorbar_label="|approx - exact|",
        show=show,
    )


# =============================================================================
# Save numerical arrays
# =============================================================================

def save_plane_wave_npz(
    grid: Grid2D,
    exact: np.ndarray,
    approx: np.ndarray,
    kvec: np.ndarray,
    lmax: int,
    filename: str | Path = "plane_wave_grid_data.npz",
) -> None:
    """
    Save grid and complex field data to an NPZ file.

    This is useful if you want to make custom plots later without recomputing
    the expansion.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        filename,
        X=grid.X,
        Y=grid.Y,
        Z=grid.Z,
        exact_real=exact.real,
        exact_imag=exact.imag,
        approx_real=approx.real,
        approx_imag=approx.imag,
        abs_error=np.abs(approx - exact),
        kvec=np.asarray(kvec, dtype=float),
        lmax=int(lmax),
        horizontal_name=grid.horizontal_name,
        vertical_name=grid.vertical_name,
        fixed_name=grid.fixed_name,
        fixed_value=grid.fixed_value,
    )
    print(f"Saved {filename}")


if __name__ == "__main__":
    # Minimal smoke test using fake data.
    from grids import make_grid

    grid = make_grid("xz", n=101, extent=0.1)
    exact = np.exp(1j * 72.0 * grid.X)
    approx = exact * (1.0 + 0.05 * grid.R)

    plot_plane_wave_comparison(
        grid=grid,
        exact=exact,
        approx=approx,
        lmax=4,
        component="real",
        output_dir="plots_test",
        show=False,
    )

    save_plane_wave_npz(
        grid=grid,
        exact=exact,
        approx=approx,
        kvec=np.array([72.0, 0.0, 0.0]),
        lmax=4,
        filename="plots_test/test_data.npz",
    )
