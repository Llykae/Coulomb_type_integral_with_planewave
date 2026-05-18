# Coulomb-type Integrals with Plane Waves

This project computes atom-centered Coulomb-type integrals between Gaussian orbitals and a plane wave.

The central quantity is:

$$
I(\mathbf{k}) =
\int_{\mathbb{R}^3} d^3\mathbf{r}
\int_{\mathbb{R}^3} d^3\mathbf{s}\;
e^{i\mathbf{k}\cdot\mathbf{r}}\,
\phi_r(\mathbf{r})\,
\frac{1}{|\mathbf{r}-\mathbf{s}|}\,
\phi_{s1}(\mathbf{s})\,
\phi_{s2}(\mathbf{s}).
$$

In plain text:

```text
I(k) = ∫ d³r ∫ d³s
       exp(i k·r)
       phi_r(r)
       1/|r-s|
       phi_s1(s) phi_s2(s)
```

where:

- `phi_r(r)` is the orbital on the plane-wave side,
- `phi_s1(s)` and `phi_s2(s)` form the density on the Coulomb side,
- `exp(i k·r)` is the plane wave,
- `1/|r-s|` is the Coulomb kernel.

All orbitals are currently assumed to be atom-centered at the origin.

---

## 1. Main features

### Atom-centered Gaussian orbitals

A Cartesian Gaussian primitive is represented as:

$$
\phi(\mathbf{r}) =
x^a y^b z^c e^{-\alpha r^2}.
$$

The code supports:

- Cartesian Gaussian primitives;
- contracted Gaussian orbitals;
- optional primitive normalization;
- molecular orbitals expanded in an AO basis;
- Quantum Package / EZFIO-style AO and MO data.

Supported angular labels include:

```text
s
px py pz
dxx dyy dzz dxy dxz dyz
fxxx fyyy fzzz fxyz ...
gxxxx gyyyy gzzzz ...
```

Explicit Cartesian powers can also be used:

```text
(1, 0, 0) -> px
(0, 1, 0) -> py
(0, 0, 1) -> pz
(1, 1, 0) -> dxy
```

---

### Plane wave

The plane wave is:

$$
e^{i\mathbf{k}\cdot\mathbf{r}}.
$$

The plane wave can be specified by:

- kinetic energy in Hartree;
- kinetic energy in eV;
- wave-vector magnitude `|k|`;
- full vector `(kx, ky, kz)`;
- direction vector plus energy.

In atomic units:

$$
E = \frac{k^2}{2}.
$$

Example:

$$
|\mathbf{k}| = 2\ \mathrm{bohr}^{-1}
\quad\Rightarrow\quad
E = 2\ \mathrm{Ha}
= 54.422772492\ \mathrm{eV}.
$$

---

### Angular decomposition

Cartesian monomials are decomposed into solid-harmonic channels:

$$
x^a y^b z^c
=
\sum_i c_i r^{2q_i} S_{l_i m_i}(x,y,z),
$$

where:

$$
S_{lm}(x,y,z) = r^l Y_{lm}(\hat{\mathbf{r}}).
$$

The angular pipeline combines:

1. the angular channel from `phi_r`;
2. the angular channel from the plane wave;
3. the angular channel from the Coulomb expansion;
4. the angular channel from the density `rho_s = phi_s1 * phi_s2`.

The plane-wave angular channel is essential. For example:

```text
phi_r = s
rho_s = px
```

is not forbidden: the plane wave can provide the missing p-like angular momentum through its `lp = 1` channel.

---

### Density contraction

The product on the `s` side is contracted before the full integral is evaluated:

$$
\rho_s(\mathbf{s}) =
\phi_{s1}(\mathbf{s})\phi_{s2}(\mathbf{s}).
$$

For two Cartesian Gaussian primitives:

$$
x^{a_1}y^{b_1}z^{c_1}e^{-\alpha_1s^2}
\;
x^{a_2}y^{b_2}z^{c_2}e^{-\alpha_2s^2}
=
x^{a_1+a_2}
y^{b_1+b_2}
z^{c_1+c_2}
e^{-(\alpha_1+\alpha_2)s^2}.
$$

So the old primitive loop:

```text
r primitive × s1 primitive × s2 primitive
```

is replaced by:

```text
r primitive × density primitive
```

This is exact for atom-centered Cartesian Gaussians, up to explicit compression and drop tolerances.

---

### Quantum Package AO/MO support

The code reads Quantum Package / EZFIO-style compressed files:

```text
ao_coef.gz
ao_expo.gz
ao_power.gz
mo_coef.gz
mo_occ.gz
mo_class.gz
```

The MO coefficient matrix is expected as:

```text
mo_coef[ao_index, mo_index]
```

The number of AOs and MOs does not need to be the same.

For example, this is supported:

```text
n_ao = 19
n_mo = 18
mo_coef shape = (19, 18)
```

---

## 2. Repository layout

Important files:

```text
run.py
    Main interactive and CLI front-end.

run_fast_tests.py
    Project-level smoke/regression test suite.

analytic_sss_check.py
    Analytic s/s/s validation.

analytic_psp_check.py
    Analytic p_x/(s p_x) validation.

angular_pipeline.py
    Angular coupling diagnostic.

atom_centered_decomposition.py
    Cartesian monomial to solid-harmonic decomposition.

atom_centered_evaluator.py
    Primitive evaluator for separate s1/s2 form.

atom_centered_evaluator_density.py
    Primitive evaluator for contracted-density form.

basis.py
    AO/MO data structures and MO flattening.

integral_from_qp_mo.py
    QP-MO integral driver.

full_coulomb_integral.py
    Contracted integral using old triple loop.

full_coulomb_integral_density.py
    Contracted integral using density path.

density_contraction.py
    Builds rho_s = phi_s1 * phi_s2.

integral_optimization.py
    Primitive compression helpers.

radial_coulomb_2d.py
    2D radial Coulomb integration backend.

radial_table.py
    Radial-key table/cache/diagnostics.

radial_integrals.py
    Simpler radial Bessel-integral checks.

plane_wave_parameters.py
    Plane-wave energy/k-vector helpers.

gaussian.py
    Cartesian Gaussian normalization.

gaunt.py
    Gaunt coefficients.

harmonics.py
    Spherical/solid-harmonic helpers.

parity.py
    Parity diagnostics.

run_validation_suite.py
    Original selection-rule validation suite.
```

---

## 3. Installation

A minimal Python environment needs:

```bash
pip install numpy scipy sympy
```

No package installation is currently required. The project runs as plain Python scripts.

---

## 4. Quick start

### Run the full fast test suite

```bash
python3 run_fast_tests.py
```

A healthy repository should end with:

```text
passed 8 / 8
```

The current validated state passes all 8 test groups.

---

### Use the dynamic interactive runner

```bash
python3 run.py
```

or:

```bash
python3 run.py interactive
```

The script asks for:

- run mode: manual primitive, QP-MO, or angular-only;
- orbital angular momenta or MO indices;
- plane-wave energy and direction;
- `lmax_pw`;
- precision preset;
- density-contraction and radial-table options.

---

### Run a manual primitive integral

```bash
python3 run.py manual \
  --r-shell s \
  --s1-shell px \
  --s2-shell s \
  --alpha-r 1.0 \
  --alpha-s1 1.0 \
  --alpha-s2 1.0 \
  --energy 2.0 \
  --direction 1 0 0 \
  --lmax-pw 8
```

---

### Run a QP-MO integral

```bash
python3 run.py qp-mo \
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
  --direction 1 0 0 \
  --lmax-pw 8 \
  --epsabs 1e-7 \
  --epsrel 1e-7
```

---

### Angular-only diagnostic

```bash
python3 run.py angular \
  --r-shell s \
  --s1-shell px \
  --s2-shell s \
  --lmax-pw 4
```

This prints angular channels and allowed couplings without evaluating the radial integral.

---

## 5. Shell labels and Cartesian powers

The runner accepts shell labels:

```text
s
px py pz
dxx dyy dzz dxy dxz dyz
fxxx fyyy fzzz fxyz ...
gxxxx gyyyy gzzzz ...
```

Example:

```bash
python3 run.py angular \
  --r-shell dxy \
  --s1-shell px \
  --s2-shell pz \
  --lmax-pw 4
```

Explicit powers can also be used:

```bash
python3 run.py angular \
  --powers-r 1 1 0 \
  --powers-s1 1 0 0 \
  --powers-s2 0 0 1 \
  --lmax-pw 4
```

Explicit powers override shell labels.

---

## 6. Validation suite

Run:

```bash
python3 run_fast_tests.py
```

The suite contains 8 groups.

---

### 6.1 Analytic `s/s/s` cases

Target integral:

$$
I(k)=
\int_{\mathbb{R}^3}d^3\mathbf{r}
\int_{\mathbb{R}^3}d^3\mathbf{s}\;
e^{i\mathbf{k}\cdot\mathbf{r}}
e^{-\alpha_r r^2}
\frac{1}{|\mathbf{r}-\mathbf{s}|}
e^{-\alpha_{s1}s^2}
e^{-\alpha_{s2}s^2}.
$$

Let:

$$
a=\alpha_r,
\qquad
b=\alpha_{s1}+\alpha_{s2},
\qquad
k=|\mathbf{k}|.
$$

For unnormalized primitives:

$$
I(0)=
\frac{2\pi^{5/2}}{ab\sqrt{a+b}}.
$$

For `k > 0`:

$$
I(k)=
\frac{2\pi^3}{\sqrt a\,b^{3/2}k}
e^{-k^2/(4a)}
\operatorname{erfi}
\left[
\frac{k}{2}
\sqrt{\frac{b}{a(a+b)}}
\right].
$$

The test suite checks:

```text
s/s/s, k=2, unnormalized
s/s/s, k=0, unnormalized
s/s/s, k=2, normalized
s/s/s, different exponents
s/s/s, coefficient scaling
```

These validate:

- Coulomb prefactors;
- plane-wave `lp = 0` channel;
- `k = 0` handling;
- primitive normalization;
- coefficient scaling;
- density contraction for `s*s`.

---

### 6.2 Analytic `p_x/(s p_x)` cases

Target integral:

$$
I(k)=
\int_{\mathbb{R}^3}d^3\mathbf{r}
\int_{\mathbb{R}^3}d^3\mathbf{s}\;
e^{ikx_r}
x_r e^{-\alpha_r r^2}
\frac{1}{|\mathbf{r}-\mathbf{s}|}
e^{-\alpha_{s1}s^2}
x_s e^{-\alpha_{s2}s^2}.
$$

The density is p-like:

$$
\rho_s(\mathbf{s})
=
x_s e^{-(\alpha_{s1}+\alpha_{s2})s^2}.
$$

The nonzero full plane-wave channels are:

```text
lp = 0 and lp = 2
```

Therefore, the exact analytic comparison requires:

```text
lmax_pw >= 2
```

The test suite checks:

```text
p_x/(s p_x), k=2, unnormalized
p_x/(s p_x), k=0, unnormalized
p_x/(s p_x), k=2, normalized
p_x/(s p_x), different exponents
p_x/(s p_x), coefficient scaling
```

These validate:

- p-type Cartesian powers;
- p-type density contraction;
- nontrivial angular coupling;
- Coulomb `L = 1` channel;
- plane-wave `lp = 0` and `lp = 2` channels;
- even-parity real-valued result.

---

### 6.3 Angular pipeline: `s/(px s)`

This checks:

```text
phi_r = s
rho_s = px
```

The expected allowed plane-wave channel is:

```text
lp = 1
```

This validates that the plane wave supplies the missing p-like angular momentum.

---

### 6.4 Angular pipeline: `px/(s px)`

This checks:

```text
lmax_pw = 1 -> allowed lp = (0,)       # partial
lmax_pw = 2 -> allowed lp = (0, 2)    # full
```

This guards against accidentally dropping the `lp = 2` plane-wave channel.

---

### 6.5 Rectangular MO basis

This verifies support for:

```text
n_ao != n_mo
```

Tested cases:

```text
n_ao = 19, n_mo = 18
n_ao = 4,  n_mo = 6
```

The MO coefficient matrix shape is:

```text
(n_ao, n_mo)
```

---

### 6.6 Existing validation suite

The original validation suite checks:

```text
s/(s s), energy=2 Ha, k along x
s/(px s), energy=2 Ha, k along x
px/(px s), energy=2 Ha, k along x
high-k s/(s s), k=72 bohr^-1, k along x
```

It verifies real/imaginary parity expectations and allowed `lp` channels.

---

### 6.7 QP AO/MO loading

This checks that QP/EZFIO files load correctly:

```text
n_ao = 19
n_mo = 18
mo_coef shape = (19, 18)
mo_occ shape = (18,)
mo_class shape = (18,)
```

---

### 6.8 QP MO0 density integral

This validates the full QP-MO density path for:

```text
r  = MO 0
s1 = MO 0
s2 = MO 0
```

The current test compresses the MO primitives:

```text
37 -> 12
```

for each selected MO, then contracts the density:

```text
raw pairs = 12 * 12 = 144
density n = 78
```

The radial table reports:

```text
unique radial integrals = 936
lp counts = {0: 936}
L counts  = {0: 936}
alpha_r range = [0.2337, 145700]
alpha_s range = [0.4674, 291400]
n_r values = (0,)
n_s values = (0,)
```

The expected value is:

```text
-2.0336133539667030 + 0.0 i
```

The test passes at tolerance `1e-6`.

The generic radial quadrature may emit `IntegrationWarning` messages for very tight Gaussian exponents. In the current validation suite this is not a correctness failure because the final QP-MO value matches the reference at the requested tolerance.

---

## 7. Analytic checker scripts

### `analytic_sss_check.py`

Run:

```bash
python3 analytic_sss_check.py --energy 2.0
```

or:

```bash
python3 analytic_sss_check.py --k 0.0
```

or normalized:

```bash
python3 analytic_sss_check.py --energy 2.0 --normalized
```

This compares the project density-path value to the exact closed form for `s/s/s`.

---

### `analytic_psp_check.py`

Run:

```bash
python3 analytic_psp_check.py --energy 2.0 --lmax-pw 2
```

For `p_x/(s p_x)`, `lmax_pw = 2` is required for the full analytic comparison.

To show the partial result with `lmax_pw = 1`:

```bash
python3 analytic_psp_check.py --energy 2.0 --lmax-pw 1 --allow-partial
```

---

## 8. Cost reduction strategy

The project uses three main cost reductions.

### 8.1 MO primitive compression

Flattened MOs may contain repeated primitive exponents and Cartesian powers. These are merged exactly by summing coefficients.

Example:

```text
MO_000_occ=2_Active: nprim 37 -> 12
```

---

### 8.2 Contracted density

Instead of looping over:

```text
r primitive × s1 primitive × s2 primitive
```

the code first builds:

$$
\rho_s = \phi_{s1}\phi_{s2}
$$

and loops over:

```text
r primitive × density primitive
```

Example:

```text
12 * 12 * 12 = 1728
```

becomes:

```text
12 * 78 = 936
```

---

### 8.3 Radial table

Radial integrals are keyed by:

```text
(lp, L, k, alpha_r, alpha_s, n_r, n_s)
```

The radial table allows:

- counting unique radial jobs;
- printing exponent ranges;
- identifying hard radial keys;
- future precomputation/parallelization;
- future replacement of selected quadrature cases by analytic or recurrence formulas.

---

## 9. Common commands

### Smoke test

```bash
python3 run_fast_tests.py
```

---

### Interactive mode

```bash
python3 run.py
```

---

### Manual `s/(px s)` test

```bash
python3 run.py manual \
  --r-shell s \
  --s1-shell px \
  --s2-shell s \
  --energy 2.0 \
  --direction 1 0 0 \
  --lmax-pw 8
```

---

### Manual `px/(s px)` setup

```bash
python3 run.py manual \
  --r-shell px \
  --s1-shell s \
  --s2-shell px \
  --energy 2.0 \
  --direction 1 0 0 \
  --lmax-pw 2
```

---

### QP MO0 test through runner

```bash
python3 run.py qp-mo \
  --r-mo 0 \
  --s1-mo 0 \
  --s2-mo 0 \
  --energy 2.0 \
  --direction 1 0 0 \
  --epsabs 1e-7 \
  --epsrel 1e-7
```

---

### Print hardest radial keys

```bash
python3 run.py qp-mo \
  --r-mo 0 \
  --s1-mo 0 \
  --s2-mo 0 \
  --energy 2.0 \
  --direction 1 0 0 \
  --hard-radial-keys 10
```

---

## 10. Data files and `.gitignore`

QP/EZFIO files such as:

```text
ao_coef.gz
ao_expo.gz
ao_power.gz
mo_coef.gz
mo_occ.gz
mo_class.gz
```

are input data files. They may be large or system-specific, so they are usually not committed.

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
*.pyo
*.pyd

.ipynb_checkpoints/
.pytest_cache/

archive/
*.log

# QP/EZFIO data files
*.gz

.env
.venv/
venv/
```

If you want to publish a tiny example dataset, place it in a dedicated directory such as:

```text
examples/data/
```

and explicitly unignore only that dataset.

---

## 11. Current limitations

- All orbitals are assumed atom-centered at the origin.
- Off-center Gaussian products are not implemented yet.
- The radial backend still relies on numerical quadrature for general keys.
- Very tight Gaussian exponents may produce SciPy `IntegrationWarning` messages.
- The solid-harmonic decomposition must support the Cartesian degree being requested.
- The code is currently a research/prototyping codebase rather than a packaged library.

---

## 12. Suggested next steps

Possible future improvements:

- Add special analytic radial formulas for common cases such as `lp=0, L=0, n_r=0, n_s=0`.
- Add radial recurrence relations for higher angular momenta.
- Add parallel radial-table precomputation.
- Add off-center Gaussian support using Gaussian product theorem translations.
- Add automated QP/EZFIO folder detection.
- Add a small example dataset and tutorial notebook.
- Convert the project into an installable Python package.

---

## 13. Recommended development workflow

Before committing changes:

```bash
python3 run_fast_tests.py
```

Only commit when the suite ends with:

```text
passed 8 / 8
```

Then:

```bash
git add .
git commit -m "Describe your change"
git push
```

---

## 14. Status

Current status:

```text
run_fast_tests.py: passed 8 / 8
```

The project currently validates both trivial and nontrivial analytic cases and a QP-MO density-path integral.
