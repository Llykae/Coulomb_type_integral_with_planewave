"""
run_fast_tests.py

Fast project-level health check for the atom-centered Coulomb / plane-wave code.

Run:

    python3 run_fast_tests.py

This script checks:

1. analytic s/s/s validation cases, including k=0;
2. analytic p_x/(s p_x) validation against an independent Fourier/Laplace formula;
3. angular pipeline selection rules;
4. rectangular AO/MO coefficient handling;
5. existing validation suite;
6. QP AO/MO file loading, if QP files are present;
7. one small QP-MO density-path integral, if QP files are present.

The QP tests are skipped automatically if the required .gz files are missing.
"""

from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np


# =============================================================================
# Small test framework
# =============================================================================

class TestFailure(RuntimeError):
    pass


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_test(name: str, func) -> bool:
    print_header(name)
    try:
        func()
    except Exception as exc:
        print("status = FAIL")
        print(f"error  = {exc}")
        traceback.print_exc()
        return False
    else:
        print("status = PASS")
        return True


def assert_close(value: complex | float, expected: complex | float, tol: float, label: str) -> None:
    err = abs(value - expected)
    rel = err / max(1.0, abs(expected))
    print(
        f"{label}: value={value}, expected={expected}, "
        f"abs_err={err:.3e}, rel_err={rel:.3e}, tol={tol:.3e}"
    )
    if not (err <= tol or rel <= tol):
        raise TestFailure(f"{label} mismatch: abs_err={err}, rel_err={rel}, tol={tol}")


# =============================================================================
# Analytic s/s/s tests
# =============================================================================

def run_one_analytic_sss_case(
    label: str,
    alpha_r: float = 1.0,
    alpha_s1: float = 1.0,
    alpha_s2: float = 1.0,
    k: float = 2.0,
    coef_r: float = 1.0,
    coef_s1: float = 1.0,
    coef_s2: float = 1.0,
    normalized: bool = False,
    tol: float = 1e-8,
) -> None:
    from analytic_sss_check import analytic_sss_full, project_sss_value
    from plane_wave_parameters import plane_wave_from_k_abs

    plane_wave = plane_wave_from_k_abs(k_abs=k, direction=(1.0, 0.0, 0.0))

    analytic = analytic_sss_full(
        alpha_r=alpha_r,
        alpha_s1=alpha_s1,
        alpha_s2=alpha_s2,
        k=plane_wave.k_abs,
        coef_r=coef_r,
        coef_s1=coef_s1,
        coef_s2=coef_s2,
        normalized=normalized,
    )

    numerical, _density_report = project_sss_value(
        alpha_r=alpha_r,
        alpha_s1=alpha_s1,
        alpha_s2=alpha_s2,
        coef_r=coef_r,
        coef_s1=coef_s1,
        coef_s2=coef_s2,
        normalized=normalized,
        plane_wave=plane_wave,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=300,
    )

    assert_close(numerical.value, analytic, tol=tol, label=label)


def test_analytic_sss_cases() -> None:
    """Permanent analytic regression tests for s/s/s, lmax_pw=0."""
    run_one_analytic_sss_case(
        label="analytic s/s/s k=2 unnormalized",
        k=2.0,
        normalized=False,
        tol=1e-9,
    )
    run_one_analytic_sss_case(
        label="analytic s/s/s k=0 unnormalized",
        k=0.0,
        normalized=False,
        tol=1e-9,
    )
    run_one_analytic_sss_case(
        label="analytic s/s/s k=2 normalized",
        k=2.0,
        normalized=True,
        tol=1e-9,
    )
    run_one_analytic_sss_case(
        label="analytic s/s/s different exponents",
        alpha_r=1.2,
        alpha_s1=0.7,
        alpha_s2=1.5,
        k=2.0,
        normalized=False,
        tol=1e-9,
    )
    run_one_analytic_sss_case(
        label="analytic s/s/s coefficient scaling",
        k=2.0,
        coef_r=2.0,
        coef_s1=-0.5,
        coef_s2=3.0,
        normalized=False,
        tol=1e-9,
    )


# =============================================================================
# Analytic p/s/p tests
# =============================================================================

def run_one_analytic_psp_case(
    label: str,
    alpha_r: float = 1.0,
    alpha_s1: float = 1.0,
    alpha_s2: float = 1.0,
    k: float = 2.0,
    coef_r: float = 1.0,
    coef_s1: float = 1.0,
    coef_s2: float = 1.0,
    normalized: bool = False,
    tol: float = 1e-8,
) -> None:
    from analytic_psp_check import analytic_px_s_px_full, project_px_s_px_value
    from plane_wave_parameters import plane_wave_from_k_abs

    plane_wave = plane_wave_from_k_abs(k_abs=k, direction=(1.0, 0.0, 0.0))

    analytic = analytic_px_s_px_full(
        alpha_r=alpha_r,
        alpha_s1=alpha_s1,
        alpha_s2=alpha_s2,
        k=plane_wave.k_abs,
        coef_r=coef_r,
        coef_s1=coef_s1,
        coef_s2=coef_s2,
        normalized=normalized,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=300,
    )

    numerical, _density_report = project_px_s_px_value(
        alpha_r=alpha_r,
        alpha_s1=alpha_s1,
        alpha_s2=alpha_s2,
        coef_r=coef_r,
        coef_s1=coef_s1,
        coef_s2=coef_s2,
        normalized=normalized,
        plane_wave=plane_wave,
        lmax_pw=2,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=300,
    )

    assert_close(numerical.value, analytic, tol=tol, label=label)


def test_analytic_psp_cases() -> None:
    """Analytic regression tests for p_x/(s p_x), requiring lmax_pw=2."""
    run_one_analytic_psp_case(
        label="analytic p_x/(s p_x) k=2 unnormalized",
        k=2.0,
        normalized=False,
        tol=1e-9,
    )
    run_one_analytic_psp_case(
        label="analytic p_x/(s p_x) k=0 unnormalized",
        k=0.0,
        normalized=False,
        tol=1e-9,
    )
    run_one_analytic_psp_case(
        label="analytic p_x/(s p_x) k=2 normalized",
        k=2.0,
        normalized=True,
        tol=1e-9,
    )
    run_one_analytic_psp_case(
        label="analytic p_x/(s p_x) different exponents",
        alpha_r=1.2,
        alpha_s1=0.7,
        alpha_s2=1.5,
        k=2.0,
        normalized=False,
        tol=1e-9,
    )
    run_one_analytic_psp_case(
        label="analytic p_x/(s p_x) coefficient scaling",
        k=2.0,
        coef_r=2.0,
        coef_s1=-0.5,
        coef_s2=3.0,
        normalized=False,
        tol=1e-9,
    )


# =============================================================================
# Angular / bookkeeping tests
# =============================================================================

def test_angular_pipeline_spx() -> None:
    """Check the known s / (px s) angular selection rule."""
    from angular_pipeline import angular_pipeline

    _r_channels, _s_channels, _pw_channels, couplings = angular_pipeline(
        powers_r=(0, 0, 0),
        powers_s1=(1, 0, 0),
        powers_s2=(0, 0, 0),
        lmax_pw=4,
    )

    allowed_lp = tuple(sorted({c.pw_channel.lp for c in couplings}))
    print(f"allowed lp = {allowed_lp}")
    if allowed_lp != (1,):
        raise TestFailure(f"Expected allowed lp=(1,), got {allowed_lp}")

    if len(couplings) != 2:
        raise TestFailure(f"Expected 2 couplings, got {len(couplings)}")


def test_angular_pipeline_psp() -> None:
    """Check p_x/(s p_x) angular selection rule: lp = 0 and 2."""
    from angular_pipeline import angular_pipeline

    _r_channels, _s_channels, _pw_channels, couplings_l1 = angular_pipeline(
        powers_r=(1, 0, 0),
        powers_s1=(0, 0, 0),
        powers_s2=(1, 0, 0),
        lmax_pw=1,
    )
    allowed_l1 = tuple(sorted({c.pw_channel.lp for c in couplings_l1}))
    print(f"allowed lp at lmax=1 = {allowed_l1}")
    if allowed_l1 != (0,):
        raise TestFailure(f"Expected partial allowed lp=(0,) at lmax=1, got {allowed_l1}")

    _r_channels, _s_channels, _pw_channels, couplings_l2 = angular_pipeline(
        powers_r=(1, 0, 0),
        powers_s1=(0, 0, 0),
        powers_s2=(1, 0, 0),
        lmax_pw=2,
    )
    allowed_l2 = tuple(sorted({c.pw_channel.lp for c in couplings_l2}))
    print(f"allowed lp at lmax=2 = {allowed_l2}")
    if allowed_l2 != (0, 2):
        raise TestFailure(f"Expected full allowed lp=(0,2) at lmax=2, got {allowed_l2}")


def test_rectangular_mo_basis() -> None:
    """Check n_ao != n_mo support without relying on external test files."""
    from basis import AOPrimitive, AOBasis, ContractedAO, MOBasis, mo_orbital

    def build_fake_ao_basis(n_ao: int) -> AOBasis:
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

    def check_case(n_ao: int, n_mo: int) -> None:
        ao_basis = build_fake_ao_basis(n_ao)

        coeff = np.zeros((n_ao, n_mo))
        for imo in range(n_mo):
            coeff[imo % n_ao, imo] = 1.0
            coeff[(imo + 1) % n_ao, imo] = 0.1 * (imo + 1)

        mo_basis = MOBasis(
            ao_basis=ao_basis,
            coefficients=coeff,
            labels=tuple(f"MO_{imo:03d}" for imo in range(n_mo)),
        )

        print(f"checking n_ao={n_ao}, n_mo={n_mo}, shape={mo_basis.coefficients.shape}")

        if mo_basis.coefficients.shape != (n_ao, n_mo):
            raise TestFailure("Rectangular MO coefficient shape mismatch")

        last_mo = mo_orbital(mo_basis, n_mo - 1)
        print(f"last MO nprim = {len(last_mo.primitives)}")

        try:
            mo_orbital(mo_basis, n_mo)
        except IndexError:
            print("correctly rejected mo_index == n_mo")
        else:
            raise TestFailure("mo_orbital should reject mo_index == n_mo")

    check_case(n_ao=19, n_mo=18)
    check_case(n_ao=4, n_mo=6)


def test_reference_validation_suite() -> None:
    """
    Run the existing validation suite if available.
    """
    try:
        import run_validation_suite
    except ImportError:
        print("run_validation_suite.py not found; skipping.")
        return

    if hasattr(run_validation_suite, "main"):
        run_validation_suite.main()
    else:
        print("run_validation_suite.py has no main(); skipping direct call.")


# =============================================================================
# QP tests
# =============================================================================

def qp_files_available() -> bool:
    required = [
        "ao_coef.gz",
        "ao_expo.gz",
        "ao_power.gz",
        "mo_coef.gz",
    ]
    missing = [name for name in required if not Path(name).exists()]
    if missing:
        print(f"QP files missing: {missing}; skipping QP tests.")
        return False
    return True


def test_qp_load_summary() -> None:
    """Check that QP AO/MO files load and have compatible dimensions."""
    if not qp_files_available():
        return

    from integral_from_qp_mo import load_qp_ao_basis, load_qp_mo_basis

    mo_occ = "mo_occ.gz" if Path("mo_occ.gz").exists() else None
    mo_class = "mo_class.gz" if Path("mo_class.gz").exists() else None

    ao_basis = load_qp_ao_basis(
        ao_coef_path="ao_coef.gz",
        ao_expo_path="ao_expo.gz",
        ao_power_path="ao_power.gz",
        normalized=True,
    )
    mo_basis, occ, klass = load_qp_mo_basis(
        ao_basis=ao_basis,
        mo_coef_path="mo_coef.gz",
        mo_occ_path=mo_occ,
        mo_class_path=mo_class,
    )

    print(f"n_ao = {ao_basis.size}")
    print(f"n_mo = {mo_basis.n_mo}")
    print(f"mo_coef shape = {mo_basis.coefficients.shape}")

    if mo_basis.coefficients.shape[0] != ao_basis.size:
        raise TestFailure("MO coefficient first dimension does not match AO size")

    if occ is not None:
        print(f"mo_occ shape = {occ.shape}")
        if occ.shape != (mo_basis.n_mo,):
            raise TestFailure("mo_occ shape mismatch")

    if klass is not None:
        print(f"mo_class shape = {klass.shape}")
        if klass.shape != (mo_basis.n_mo,):
            raise TestFailure("mo_class shape mismatch")


def test_qp_mo0_density_integral_fast() -> None:
    """
    Run a fast QP MO0/MO0/MO0 density-path integral if QP files exist.
    """
    if not qp_files_available():
        return

    from integral_from_qp_mo import load_qp_ao_basis, load_qp_mo_basis
    from basis import mo_orbital
    from integral_optimization import compress_contracted_gaussian
    from density_contraction import build_contracted_density
    from plane_wave_parameters import plane_wave_from_energy_hartree
    from full_coulomb_integral_density import full_coulomb_plane_wave_integral_density

    mo_occ = "mo_occ.gz" if Path("mo_occ.gz").exists() else None
    mo_class = "mo_class.gz" if Path("mo_class.gz").exists() else None

    ao_basis = load_qp_ao_basis(
        ao_coef_path="ao_coef.gz",
        ao_expo_path="ao_expo.gz",
        ao_power_path="ao_power.gz",
        normalized=True,
    )
    mo_basis, _occ, _klass = load_qp_mo_basis(
        ao_basis=ao_basis,
        mo_coef_path="mo_coef.gz",
        mo_occ_path=mo_occ,
        mo_class_path=mo_class,
    )

    phi_r = mo_orbital(mo_basis, 0, drop_tol=1e-12)
    phi_s1 = mo_orbital(mo_basis, 0, drop_tol=1e-12)
    phi_s2 = mo_orbital(mo_basis, 0, drop_tol=1e-12)

    phi_r, report_r = compress_contracted_gaussian(phi_r)
    phi_s1, report_s1 = compress_contracted_gaussian(phi_s1)
    phi_s2, report_s2 = compress_contracted_gaussian(phi_s2)

    print("compression:")
    report_r.print()
    report_s1.print()
    report_s2.print()

    density, density_report = build_contracted_density(phi_s1, phi_s2)
    density_report.print()

    plane_wave = plane_wave_from_energy_hartree(2.0, direction=(1.0, 0.0, 0.0))

    result = full_coulomb_plane_wave_integral_density(
        phi_r=phi_r,
        density=density,
        plane_wave=plane_wave,
        lmax_pw=8,
        epsabs=1e-7,
        epsrel=1e-7,
        limit=300,
        use_radial_table=True,
        precompute_radial=False,
        print_radial_report=True,
        print_hard_keys=0,
        max_contributions_store=0,
    )

    print(f"value = {result.value.real:.16e} + {result.value.imag:.16e} i")

    expected = complex(-2.0336133539667030, 0.0)
    assert_close(result.value, expected, tol=1e-6, label="QP MO0/MO0/MO0 density integral")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    tests = [
        ("Analytic s/s/s cases", test_analytic_sss_cases),
        ("Analytic p_x/(s p_x) cases", test_analytic_psp_cases),
        ("Angular pipeline: s / (px s)", test_angular_pipeline_spx),
        ("Angular pipeline: p_x / (s p_x)", test_angular_pipeline_psp),
        ("Rectangular MO basis", test_rectangular_mo_basis),
        ("Existing validation suite", test_reference_validation_suite),
        ("QP AO/MO load summary", test_qp_load_summary),
        ("QP MO0 density integral fast", test_qp_mo0_density_integral_fast),
    ]

    passed = 0
    for name, func in tests:
        ok = run_test(name, func)
        passed += int(ok)

    print_header("Summary")
    print(f"passed {passed} / {len(tests)}")

    if passed != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
