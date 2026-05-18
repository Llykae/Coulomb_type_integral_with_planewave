"""
example_plot_plane_wave.py

Small example that connects the project modules:

    grids.py
    plane_wave_expansion.py
    plotting.py

It does the following:

1. Builds a 2D grid.
2. Evaluates the exact plane wave exp(i k.r).
3. Evaluates the low-l Cartesian solid-harmonic expansion.
4. Prints a convergence report for lmax = 0..4.
5. Saves PNG plots and an NPZ data file.

Run example
-----------

    python3 example_plot_plane_wave.py

With custom parameters:

    python3 example_plot_plane_wave.py --k 10 --extent 0.05 --lmax 4 --plane xz --show

For a high-k case close to the origin:

    python3 example_plot_plane_wave.py --k 72 --extent 0.02 --lmax 4 --show

Important
---------
This example uses the Cartesian solid-harmonic expansion from
plane_wave_expansion.py, which currently supports only lmax <= 4.

For k ~ 72, lmax=4 is only meaningful very close to the origin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from grids import make_grid, estimate_lmax
from plane_wave_expansion import (
    convergence_report,
    eval_exact_plane_wave,
    eval_plane_wave_cartesian_expansion,
)
from plotting import plot_plane_wave_comparison, save_plane_wave_npz


def build_kvec(k_abs: float, direction: tuple[float, float, float]) -> np.ndarray:
    """
    Build a k-vector from a magnitude and direction.

    The direction is normalized internally.
    """
    direction_array = np.array(direction, dtype=float)
    norm = np.linalg.norm(direction_array)

    if norm == 0.0:
        raise ValueError("k direction cannot be the zero vector")

    return k_abs * direction_array / norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot exact plane wave and low-l Cartesian solid-harmonic expansion."
    )

    parser.add_argument("--k", type=float, default=10.0, help="magnitude |k|")
    parser.add_argument("--kx-dir", type=float, default=1.0, help="x component of k direction")
    parser.add_argument("--ky-dir", type=float, default=0.0, help="y component of k direction")
    parser.add_argument("--kz-dir", type=float, default=0.0, help="z component of k direction")

    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xz", help="2D plotting plane")
    parser.add_argument("--n", type=int, default=201, help="grid points per axis")
    parser.add_argument("--extent", type=float, default=0.05, help="grid half-width")
    parser.add_argument("--fixed-value", type=float, default=0.0, help="value of the coordinate held fixed")

    parser.add_argument("--lmax", type=int, default=4, help="maximum angular momentum, currently <= 4")
    parser.add_argument(
        "--component",
        choices=["real", "imag", "abs", "phase"],
        default="real",
        help="component to plot",
    )

    parser.add_argument("--output-dir", type=str, default="plots", help="directory for PNG and NPZ outputs")
    parser.add_argument("--show", action="store_true", help="open matplotlib window after saving plots")
    parser.add_argument("--no-convergence", action="store_true", help="skip convergence report")

    args = parser.parse_args()

    if args.lmax > 4:
        raise ValueError(
            "This example uses the current Cartesian solid-harmonic table, "
            "which supports only lmax <= 4."
        )

    kvec = build_kvec(args.k, (args.kx_dir, args.ky_dir, args.kz_dir))
    grid = make_grid(
        plane=args.plane,
        n=args.n,
        extent=args.extent,
        fixed_value=args.fixed_value,
    )

    k_abs = float(np.linalg.norm(kvec))
    krmax = k_abs * grid.rmax
    suggested = estimate_lmax(k_abs, grid.rmax, safety=20)

    print("\n=== Example setup ===")
    print(f"kvec          = {kvec}")
    print(f"|k|           = {k_abs:.6g}")
    print(f"grid          = {grid.describe()}")
    print(f"k*rmax        = {krmax:.6g}")
    print(f"lmax          = {args.lmax}")
    print(f"rough lmax estimate for full convergence = {suggested}")

    if args.lmax < krmax:
        print("\nWARNING:")
        print("  lmax is smaller than k*rmax.")
        print("  The expansion may be visibly truncated on this grid.")
        print("  For the current low-l Cartesian prototype, reduce --extent if needed.")

    print("\n=== Evaluating fields ===")
    exact = eval_exact_plane_wave(kvec, grid.X, grid.Y, grid.Z)
    approx = eval_plane_wave_cartesian_expansion(kvec, grid.X, grid.Y, grid.Z, args.lmax)

    err = np.abs(approx - exact)
    print(f"max error = {np.max(err):.6e}")
    print(f"rms error = {np.sqrt(np.mean(err**2)):.6e}")

    if not args.no_convergence:
        convergence_report(
            kvec,
            grid.X,
            grid.Y,
            grid.Z,
            lmax_values=tuple(range(args.lmax + 1)),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Saving plots and data ===")
    plot_plane_wave_comparison(
        grid=grid,
        exact=exact,
        approx=approx,
        lmax=args.lmax,
        component=args.component,
        output_dir=output_dir,
        prefix="plane_wave_cartesian",
        show=args.show,
    )

    save_plane_wave_npz(
        grid=grid,
        exact=exact,
        approx=approx,
        kvec=kvec,
        lmax=args.lmax,
        filename=output_dir / "plane_wave_cartesian_data.npz",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
