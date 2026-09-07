# Prompt 05 — `TkSourceFunctions`: a two-region representation of $T_k$ for consumers (A2, part 1)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §0.2 A2; `TK-report.md` §1 rows R23/R24/R26 and TK-8; `QS-report.md` QS-5;
`docs/resonance-scaffolding/sigw-resonance-reconciliation.md` §1.1 ("Modes in amplitude–phase form"), §3.4
**Depends on:** nothing hard (soft: 01, 02). Prompts 07 and 08 consume what this defines.
**Recommended model:** Opus
**Files you may touch:** new `ComputeTargets/TkSourceFunctions.py`, `ComputeTargets/__init__.py`
(export), new `ComputeTargets/tests/test_tk_source_functions.py` (and `tests/__init__.py` if 03 has
not created it), plus the log and the status board. **Do not** touch `TkNumericIntegration.py`,
`TkWKBIntegration.py`, `QuadSource.py` or `QuadSourceIntegral.py`.

Read first: README §2 (a) and (b) — they state the two facts this prompt is built on; spec 01
§3.4–3.5 (NUM 08/09, the LG form); `TkWKBIntegration.store()` at `TkWKBIntegration.py:470-535`;
`GkSourcePolicyData._create_functions()` at `GkSourcePolicyData.py:573-697`, which is the pattern
to follow; `LiouvilleGreen/phase_spline.py` (constructor and `raw_theta`, `theta_mod_2pi`,
`theta_deriv`); `ComputeTargets/spline_wrappers.py`.

---

## 1. Character of this commit

A new, **non-persisted** helper class that gives downstream consumers a single object describing
$T_k(z)$ over the whole source-redshift range, in two regions:

- the **numeric region**, from the start of the `TkNumericIntegration` grid down to the
  numeric→WKB hand-over redshift, where $T$ and $dT/dz$ are smooth and are provided as splines;
- the **WKB (LG) region**, from the hand-over down to `z_end`, where
  $T = M(z)\sin\theta(z)$ and the consumer is given $M$, $d\ln M/dz$, $\theta$ (as a
  `phase_spline`) and $d\theta/dz = \omega_{\rm eff}$.

Above the numeric region the transfer function is exactly $T = 1$, $dT/dz = 0$ (spec 01 §2.8; this
is the default `compute_quad_source` already substitutes at `QuadSource.py:116-121`).

This is the transfer-function analogue of `GkSourceFunctions`, with one simplification (README §2
(a)): there is no overlap and no policy, because the hand-over redshift is the `mode="stop"` point
and is already fixed.

**Why not a compute target.** Nothing here is a new computed result; every number is a re-reading
of `TkNumericIntegration` and `TkWKBIntegration` values plus closed-form model functions. Persisting
it would only duplicate rows. If, while implementing, you find a genuine reason it must be
persisted, stop and open a §3 issue rather than adding a Datastore factory.

## 2. Inputs and protocol

Constructor: `TkSourceFunctions(model: BackgroundModel, k: wavenumber, Tk_numeric, Tk_WKB)`.

Define the **duck-typed protocol** the two inputs must satisfy, in the module docstring, and
program against it — the tests will pass synthetic objects, not `TkNumericIntegration`/
`TkWKBIntegration` instances:

- `Tk_numeric`: `.z_sample` (descending `redshift_array`), `.values` (list, each with `.z.z`,
  `.T`, `.Tprime` $= dT/dz$), `.z_init` (redshift), `.stop_deltaz_subh`.
- `Tk_WKB`: `.z_sample`, `.values` (each with `.z.z`, `.theta_div_2pi`, `.theta_mod_2pi`,
  `.friction`, `.H_ratio`, `.omega_WKB_sq`), `.sin_coeff`, `.cos_coeff` (assert it is 0 —
  `TkWKBIntegration.store()` sets it so at `:458`), `.z_init` (float).

Check these attribute names against the two classes (`TkNumericIntegration.py:467-545`,
`TkWKBIntegration.py:536-650`) and correct the list if any differ; record any correction as
STRUCTURALLY REQUIRED.

## 3. What the object exposes

A `namedtuple`/dataclass `TkSourceFunctions` (mirror `GkSourceFunctions`'s style) with:

| field | meaning |
|---|---|
| `numeric_region` | `(z_max, z_min)` = (first numeric sample, hand-over redshift) |
| `WKB_region` | `(z_max, z_min)` = (hand-over redshift, last WKB sample) |
| `crossover_z` | the hand-over redshift (float). Must equal `Tk_WKB.z_init`; assert it also equals `k.z_exit − Tk_numeric.stop_deltaz_subh` to `DEFAULT_FLOAT_PRECISION` relative |
| `T(z, z_is_log=False)` | numeric-region $T$ (spline in $\log(1+z)$); returns exactly `1.0` above `numeric_region[0]`; raises below `crossover_z` |
| `dT_dz(z, ...)` | numeric-region $dT/dz$ — spline the *stored* `Tprime`, do not differentiate the $T$ spline; returns `0.0` above the region |
| `M(z, ...)` | LG amplitude, WKB region only |
| `dlnM_dz(z, ...)` | $d\ln M/dz$, WKB region only, **closed form** (§4) |
| `phase` | the `phase_spline` of $\theta$ over the WKB region |
| `omega(z, ...)` | $d\theta/dz = \sqrt{\texttt{Tk\_omegaEff\_sq}(model,k,z)}$ — closed form, not the spline derivative |
| `T_WKB(z, ...)` | convenience: $M\sin(\theta \bmod 2\pi)$, for tests and diagnostics |

Use `ZSplineWrapper` for the splines so out-of-range behaviour matches the rest of the code, and
give each a correct label. Splines in $\log(1+z)$, as everywhere else.

## 4. The amplitude and its derivative — use the closed forms

From `TkWKBIntegration.store()` (`:494-506`), with `cos_coeff = 0`:

$$M(z) = \texttt{sin\_coeff}\cdot\sqrt{\frac{H_{\rm init}}{H(z)}}\;\omega_{\rm eff}(z)^{-1/2}\,e^{F(z)},$$

where `H_ratio = H_init/H` and `friction` $=F$ are stored per value, and
$\omega_{\rm eff}^2 =$ `WKB_Tk.Tk_omegaEff_sq(model, k, z)`. Build `M` as: a spline of the stored
$F$ samples (smooth, monotone; spline it in $\log(1+z)$) combined with `model.functions.Hubble`,
$H_{\rm init} = H(\texttt{crossover\_z})$ and the closed-form $\omega_{\rm eff}$. Do **not** spline
the product; the product's components are exact and the spline is only needed for $F$.

The derivative, from spec 01 R23/R24 (audit TK §1 rows R23, R24 verified these against the code):

$$\frac{d\ln M}{dz} = -\frac{\epsilon(z)}{2(1+z)} \;-\; \frac12\,\frac{d\ln\omega_{\rm eff}}{dz} \;+\; \frac32\,\frac{1+c_s^2(z)}{1+z},$$

with $\epsilon =$ `model.functions.epsilon(z)`, $c_s^2 =$ `model.functions.wPerturbations(z)`,
$d\ln\omega_{\rm eff}/dz =$ `WKB_Tk.Tk_d_ln_omegaEff_dz(model, k, z)`. The first term is
$d\ln\sqrt{H_{\rm init}/H}/dz$, the last is $dF/dz$ (`TkWKBIntegration.friction_RHS`, `:25-50` —
read it and confirm the sign and the factor, then cite the line in the module docstring).

**Verify this identity numerically in the test** (§6): compare `dlnM_dz` with a centred finite
difference of $\ln M$ built from the stored samples, to $10^{-6}$ relative away from the region ends.

## 5. Phase

Build `phase` exactly as `GkSourcePolicyData._create_functions` does at `:657-671`
(`phase_spline(log_x, theta_div_2pi, theta_mod_2pi, x_is_log=True, x_is_redshift=True,
chunk_step=None, chunk_logstep=125, increasing=False)`), over the WKB region. Note the convention:
$\theta(z_{\rm init}) = 0$ and $\theta$ **decreases** (goes negative) as $z$ falls (README board
§5 note 3). Also expose the stored `sin_coeff` — a consumer that wants $T$'s sign convention needs
it (`M` already includes it, so `M` may be negative; do not take an absolute value).

## 6. Tests — exact constant-$w$ fixture

`ComputeTargets/tests/test_tk_source_functions.py`. Build the inputs synthetically from the exact
solution, so the test is independent of the pipeline (README §2 (c)):

1. A constant-$w$ stand-in model as in `docs/spec-code-audit/scripts/TK_03_numeric_vs_analytic.py`
   (`FakeModel` with `Hubble`, `tau`, `epsilon`, `wPerturbations`, `wBackground`, derivatives; copy
   what you need into the test, do not import from `docs/`).
2. The numeric-region values from `ComputeTargets/analytic_Tk.compute_analytic_T/Tprime` on a
   100-per-decade grid from 5 e-folds super-horizon to a hand-over point 3–4 e-folds sub-horizon.
3. The WKB-region values from `LiouvilleGreen.bessel_phase.bessel_phase(1.5 + b, ...)`: with
   $x = k c_s a_0\eta(z)$, $J_{3/2+b}(x) = m(x)\sin\vartheta(x)$, so
   $T = 2^{3/2+b}\Gamma(\tfrac52+b)\,x^{-3/2-b}\,m(x)\,\sin\vartheta(x)$ and the fixture's
   `theta` is $\vartheta(x(z)) - \vartheta(x(z_{\rm init}))$ (rebased to 0 at the hand-over, matching
   the code's convention), split into `div_2pi`/`mod_2pi` with `LiouvilleGreen.WKBtools.wrap_theta`
   (check its sign convention against `WKB_mod_2pi`). Set `friction`, `H_ratio`, `omega_WKB_sq`
   from the model so that the *code's* $M$ formula reproduces the *exact*
   $2^{3/2+b}\Gamma x^{-3/2-b}m$ — i.e. choose `sin_coeff` from the matching at the hand-over as
   `TkWKBIntegration.store()` does (`:438-459`), or simply back it out from the exact $M$ at
   $z_{\rm init}$. Say which in the log.
   Read `bessel_phase.py:270-290` for the exact `(m, θ)` convention before using it.
4. Assertions:
   - `T`, `dT_dz` reproduce the exact solution in the numeric region to $10^{-8}$ (spline of exact
     samples);
   - `T_WKB` reproduces the exact $T$ in the WKB region to $10^{-8}$ relative-to-envelope (this is
     an *exact* LG fixture, so the residual is spline error only);
   - `M` matches the exact amplitude to $10^{-8}$; `dlnM_dz` matches a finite difference to
     $10^{-6}$;
   - `omega` matches `phase.theta_deriv(z)` (spline derivative of $\theta$ w.r.t. $z$) to $10^{-6}$
     away from the region ends — this cross-checks the closed-form frequency against the stored
     phase;
   - `crossover_z` consistency assertion fires when `stop_deltaz_subh` is inconsistent;
   - `T` above the numeric region is exactly `1.0` and `dT_dz` exactly `0.0`.
   Run for $w = 1/3$ and $w = 0.2$ ($b = 0$ and $b = 0.25$).

## 7. Verification

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` passes, under
  ~30 s.
- Quote in the log the measured maxima for each assertion.

## 8. Log and commit

Log to `logs/05-tk-source-functions.md`. In **State handed to the next prompt**, list the final
field names and signatures verbatim — prompts 07 and 08 program against them. Board: row 05, item
A2 (1/3). One commit; body explains why the object is not persisted and states the closed-form
$d\ln M/dz$ used.
