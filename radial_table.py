"""
radial_table.py

Radial integral table / precomputation layer for the atom-centered
Coulomb / plane-wave project.

Purpose
-------
After MO compression and density contraction, the expensive part is the radial
integral evaluation. Each r-density primitive pair requires one or more radial
Coulomb integrals of the form

    R(lp, L, k, alpha_r, alpha_s, n_r, n_s)

where

    lp       = plane-wave angular momentum channel
    L        = Coulomb multipole channel
    k        = |k|
    alpha_r  = r-side Gaussian exponent
    alpha_s  = density-side Gaussian exponent
    n_r      = radial polynomial power from r-side solid-harmonic decomposition
    n_s      = radial polynomial power from density-side solid-harmonic decomposition

This module makes those radial jobs explicit and caches their values.

Why this helps
--------------
Even when every r-density pair is unique, this table gives us:

1. a count of unique radial jobs before integration;
2. diagnostics for hard keys, e.g. very tight exponents;
3. one place to add parallelism later;
4. one place to replace numerical quadrature by analytic/recurrence formulas.

Current backend
---------------
For now, each table entry calls radial_coulomb_2d_integral from
radial_coulomb_2d.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import numpy as np

from radial_coulomb_2d import radial_coulomb_2d_integral


# =============================================================================
# Radial key / value objects
# =============================================================================

@dataclass(frozen=True, order=True)
class RadialKey:
    """
    Hashable key for one radial Coulomb integral.
    """

    lp: int
    L: int
    k: float
    alpha_r: float
    alpha_s: float
    n_r: int
    n_s: int

    def rounded(self, digits: int = 12) -> "RadialKey":
        """Return a rounded copy useful for stable dictionary keys."""
        return RadialKey(
            lp=int(self.lp),
            L=int(self.L),
            k=round(float(self.k), digits),
            alpha_r=round(float(self.alpha_r), digits),
            alpha_s=round(float(self.alpha_s), digits),
            n_r=int(self.n_r),
            n_s=int(self.n_s),
        )

    def short(self) -> str:
        return (
            f"lp={self.lp}, L={self.L}, k={self.k:.6g}, "
            f"ar={self.alpha_r:.6g}, as={self.alpha_s:.6g}, "
            f"nr={self.n_r}, ns={self.n_s}"
        )


@dataclass(frozen=True)
class RadialValue:
    """
    Value and optional error estimate for one radial integral.
    """

    key: RadialKey
    value: float
    estimated_error: float | None = None


@dataclass(frozen=True)
class RadialTableReport:
    """Summary of a radial table."""

    n_unique: int
    lp_counts: dict[int, int]
    L_counts: dict[int, int]
    alpha_r_min: float
    alpha_r_max: float
    alpha_s_min: float
    alpha_s_max: float
    n_r_values: tuple[int, ...]
    n_s_values: tuple[int, ...]

    def print(self) -> None:
        print("\n=== Radial table report ===")
        print(f"unique radial integrals = {self.n_unique}")
        print(f"lp counts = {self.lp_counts}")
        print(f"L counts  = {self.L_counts}")
        print(f"alpha_r range = [{self.alpha_r_min:.6g}, {self.alpha_r_max:.6g}]")
        print(f"alpha_s range = [{self.alpha_s_min:.6g}, {self.alpha_s_max:.6g}]")
        print(f"n_r values = {self.n_r_values}")
        print(f"n_s values = {self.n_s_values}")


# =============================================================================
# Table class
# =============================================================================

class RadialIntegralTable:
    """
    Cache and precomputation table for radial integrals.
    """

    def __init__(
        self,
        epsabs: float = 1e-10,
        epsrel: float = 1e-10,
        limit: int = 300,
        key_round_digits: int = 12,
    ) -> None:
        self.epsabs = epsabs
        self.epsrel = epsrel
        self.limit = limit
        self.key_round_digits = key_round_digits
        self._values: dict[RadialKey, RadialValue] = {}
        self.hits = 0
        self.misses = 0

    def normalize_key(self, key: RadialKey) -> RadialKey:
        return key.rounded(self.key_round_digits)

    def has(self, key: RadialKey) -> bool:
        return self.normalize_key(key) in self._values

    def get(self, key: RadialKey) -> float:
        """
        Return a radial integral value, computing it if needed.
        """
        key = self.normalize_key(key)
        if key in self._values:
            self.hits += 1
            return self._values[key].value

        result = radial_coulomb_2d_integral(
            lp=key.lp,
            L=key.L,
            k=key.k,
            alpha_r=key.alpha_r,
            alpha_s=key.alpha_s,
            n_r=key.n_r,
            n_s=key.n_s,
            epsabs=self.epsabs,
            epsrel=self.epsrel,
            limit=self.limit,
        )

        err = getattr(result, "outer_error", None)
        if err is None:
            err = getattr(result, "outer_err", None)

        self._values[key] = RadialValue(
            key=key,
            value=float(result.value),
            estimated_error=None if err is None else float(err),
        )
        self.misses += 1
        return self._values[key].value

    def precompute(
        self,
        keys: list[RadialKey],
        verbose: bool = True,
        progress_every: int = 50,
    ) -> None:
        """
        Precompute all unique keys.

        progress_every controls how often progress is printed.
        Set progress_every <= 0 to print only the header and final line.
        """
        unique = sorted({self.normalize_key(key) for key in keys})

        if verbose:
            print("\n=== Precomputing radial table ===")
            print(f"requested keys = {len(keys)}")
            print(f"unique keys    = {len(unique)}")

        for i, key in enumerate(unique, start=1):
            if verbose and progress_every > 0:
                if i == 1 or i == len(unique) or i % progress_every == 0:
                    print(f"  [{i:5d}/{len(unique):5d}] {key.short()}")

            self.get(key)

        if verbose:
            print(f"done precomputing {len(unique)} radial integrals")

    @property
    def n_values(self) -> int:
        return len(self._values)

    def report_from_keys(self, keys: list[RadialKey] | None = None) -> RadialTableReport:
        """
        Build a report from given keys or from stored values.
        """
        if keys is None:
            keys = list(self._values.keys())
        keys = [self.normalize_key(key) for key in keys]
        unique = sorted(set(keys))

        if not unique:
            raise ValueError("Cannot report an empty radial table")

        lp_counts = Counter(key.lp for key in unique)
        L_counts = Counter(key.L for key in unique)

        return RadialTableReport(
            n_unique=len(unique),
            lp_counts=dict(sorted(lp_counts.items())),
            L_counts=dict(sorted(L_counts.items())),
            alpha_r_min=min(key.alpha_r for key in unique),
            alpha_r_max=max(key.alpha_r for key in unique),
            alpha_s_min=min(key.alpha_s for key in unique),
            alpha_s_max=max(key.alpha_s for key in unique),
            n_r_values=tuple(sorted({key.n_r for key in unique})),
            n_s_values=tuple(sorted({key.n_s for key in unique})),
        )

    def print_cache_stats(self) -> None:
        print("\n=== Radial table cache stats ===")
        print(f"stored values = {self.n_values}")
        print(f"hits          = {self.hits}")
        print(f"misses        = {self.misses}")


# =============================================================================
# Helper for sorting hard/easy keys
# =============================================================================

def radial_key_difficulty_score(key: RadialKey) -> float:
    """
    Heuristic score for sorting radial keys.

    Larger exponents and larger angular/power values may be harder for generic
    quadrature. This is only diagnostic.
    """
    return (
        np.log10(1.0 + key.alpha_r)
        + np.log10(1.0 + key.alpha_s)
        + 0.25 * (key.lp + key.L)
        + 0.1 * (key.n_r + key.n_s)
    )


def print_hardest_radial_keys(keys: list[RadialKey], n: int = 10) -> None:
    """Print the hardest-looking radial keys according to a simple heuristic."""
    unique = sorted(set(keys), key=radial_key_difficulty_score, reverse=True)
    print(f"\n=== Hardest-looking radial keys, top {min(n, len(unique))} ===")
    for key in unique[:n]:
        print(f"  score={radial_key_difficulty_score(key):.3f}  {key.short()}")
