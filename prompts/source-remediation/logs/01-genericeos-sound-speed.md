# Log 01 — Fix the `GenericEOS` perturbation sound speed (A1)

**Prompt:** prompts/source-remediation/01-genericeos-sound-speed.md
**Commit:** *(SHA intentionally not embedded — see Deviation 4)* — "Exclude Lambda from the
GenericEOS perturbation sound speed", the single commit that adds this log
**Model:** Claude Opus 5
**Date:** 2026-09-08
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

**`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:290`** (one line, inside
`wPerturbations`):

```
-        denominator = self.rho(z)                          # rho_m + rho_r + rho_Lambda
+        denominator = rho["matter"] + rho["radiation"]     # Lambda unperturbed
```

`rho` is the dict already fetched from `_rho_fluid(z)` two lines above; its keys were checked
(`_rho_fluid`, `:220-225`) and are `"T"`, `"matter"`, `"radiation"`, `"lambda"`. No other line
changed: the numerator, the comment block and `wBackground` (`:269-279`, which correctly divides
by the total density and carries `p_Lambda = -rho_Lambda` in the numerator) are untouched.
`QCD_Cosmology` subclasses `LambdaCDM_GenericEOS` and does not override `wPerturbations`, so it
inherits the fix.

**New `CosmologyModels/tests/__init__.py`** (empty) and **new
`CosmologyModels/tests/test_wPerturbations.py`** — a `unittest` module, no Ray, no datastore,
5 test methods (11 subtests):

- `PureRadiationEOS`, a stub `GenericEOSBase` in the test file with constant, equal `G` and `Gs`,
  so that `GenericEOSBase.w()` returns `4 g_S/(3 g) - 1 = 1/3` identically and `T(z) = T_CMB (1+z)`
  exactly. Its `g` is `2 + 2*(7/8)*Neff*(4/11)^(4/3)`, i.e. the same photon+neutrino budget
  `LambdaCDM.__init__` hardwires into `rho_r0`, so the two classes have identical `rho_r0` and
  `omega_r`. No production-code helper was added.
- `_Params`, a minimal parameter block so `omega_cc` can be varied at fixed `omega_m`, `h`,
  `T_CMB_Kelvin` (`Planck2018` ties `omega_m = 1 - omega_cc`, which would have confounded the
  Lambda-independence test).
- `test_omega_r_matches` — guards the premise of the agreement test (ratio 1 to 1e-14; measured
  exactly 0.0 difference).
- `test_agrees_with_LambdaCDM` — `wPerturbations` of the two classes at
  `z in {0, 0.5, 1, 2, 10, 1e3}`, relative tolerance `1e-8` (see Deviation 1).
- `test_independent_of_omega_lambda` — same z grid, `omega_cc = 0.6889` vs `0.4` at fixed
  `omega_m, omega_r`; relative tolerance `1e-12` (the two models share one T(z) spline
  construction path, so this comparison is not limited by the spline).
- `test_wBackground_does_depend_on_omega_lambda` — the companion assertion, so that a
  `wPerturbations` insensitive to the density composition altogether could not pass.
- `test_limits` — `wPerturbations(1e8) -> 1/3` within `1e-4`; `0 < wPerturbations(0) < 1e-4`;
  `wPerturbations/wBackground -> 1` within `1e-3` at `z = 1e3` and `1e4`.

Models are built once in `setUpClass` because each `LambdaCDM_GenericEOS` construction runs 500
root solves for its T(z) spline.

## Deviations from the prompt

### 1. Agreement tolerance is 1e-8 relative, not 1e-10 — STRUCTURALLY REQUIRED

The prompt asks for agreement with `LambdaCDM.wPerturbations` "to 1e-10 relative". That is not
reachable, for a reason that has nothing to do with A1 and cannot be fixed without touching
production code the prompt does not permit.

`LambdaCDM_GenericEOS` never evaluates `T(z)` directly: `_rho_fluid` reads `self._T_z_spline(z)`,
a 500-point interpolating spline in `log(1+z)` built by `_build_T_z_spline` over
`[0.95 * (-0.2), 1.05 * max_z]`. With a constant-`g_S` equation of state the exact answer is
`T = T_CMB (1+z)`, i.e. an exponential in the spline's abscissa, and a cubic interpolant of an
exponential on a uniform grid of spacing `h` has relative error `~h^4/384`. `rho_r ∝ T^4`, so the
error is amplified fourfold into `wPerturbations`. Measured relative difference from `LambdaCDM`
*after* the fix, worst over `z in {0, 0.5, 1, 2, 10, 1e3}`:

| `max_z` | grid spacing `h` in ln(1+z) | worst relative difference |
|---|---|---|
| 1e20 (class default) | 0.092 | 6.4e-7 |
| 1e6 | 0.028 | 6.6e-9 |
| 1e4 (chosen) | 0.019 | 1.3e-9 |
| 4e3 | 0.017 | 8.0e-10 |

`max_z` cannot be pushed lower than about 3500: `LambdaCDM_GenericEOS.__init__` solves for
matter–radiation equality at `z = 3403` and the T(z) spline must cover it (with `max_z = 1e3` the
constructor raises `RuntimeError: ... evaluated T(z) out of bounds @ z=3403.1`). `samples` is a
default argument of a private method, not a constructor parameter, so the grid cannot be refined
from a test. So the floor is ~8e-10 and no choice of `max_z` reaches 1e-10.

I chose `max_z = 1e4` (floor 1.3e-9) rather than the marginally better `4e3`, because 1e4 leaves
headroom above the equality redshift and lets `test_limits` probe `z = 1e4`; and I set the
threshold at `1e-8`, roughly an order of magnitude above the measured floor, so the test is not
fragile against a scipy spline-evaluation change. 1e-8 is still seven orders of magnitude below
the discrepancy A1 produced (0.69 relative at z=0), so the test's discriminating power against the
defect is unaffected — confirmed by the revert-and-run below, where every one of the six
agreement subtests fails.

The alternatives considered and rejected: (a) comparing `wPerturbations` against a closed form
recomputed from the model's own `_rho_fluid` — tautological, it would pass with the bug present if
the same denominator were used, and if the correct denominator were used it would merely restate
the source line; (b) adding a `samples` or exact-`T(z)` hook to `LambdaCDM_GenericEOS` — production
change the prompt forbids and scope creep (README §5.5).

The measured floor is recorded as an active issue on the board so that a later prompt does not
mistake it for a physics discrepancy.

### 2. The parenthetical comment was kept — IMPLEMENTATION CHOICE

Step 2 of the prompt leaves it to judgement whether to delete "(Possibly we shouldn't do that, but
instead allow the cosmological constant to cluster with c_s=1?)". I kept it verbatim. It is not
misleading about what the code now does — the sentence before it states the convention the code
implements, and the parenthetical is explicitly flagged as a question about the convention itself,
not a description of the implementation. It is also the author's own note recording an open
physical choice (the same one README §1.1 defers as the `csSquared(z)` separation), and deleting an
author's open question is not something a remediation commit should do silently. Someone who
disagrees can delete one line.

### 3. Two test methods beyond the three the prompt lists — IMPLEMENTATION CHOICE

The prompt says "at least" three. `test_omega_r_matches` was added because the agreement test is
meaningless if the two models disagree on `Omega_r`, and a failure there should be reported as
its own cause rather than as a `wPerturbations` mismatch.
`test_wBackground_does_depend_on_omega_lambda` is the second half of the prompt's own
Lambda-independence bullet ("`wBackground` must *change*"), split into its own method so the two
assertions fail independently.

### 4. The commit SHA is not embedded in the log or the board — IMPLEMENTATION CHOICE

README §5.1's template has a **Commit:** `<sha>` field and `IMPLEMENTATION_STATE.md` has a Commit
column, but both files are *inside* the commit they would name, so no value written into them can
be correct: writing the SHA and amending changes the SHA again, and the fixed point does not
exist. The previous campaign hit exactly this and recorded it as
`prompts/backport-modules/IMPLEMENTATION_STATE.md` §4 `[commit-sha-links-stale]`, where logs 01–03
were left pointing one amend behind the tip and had to be corrected by a later housekeeping pass;
from its prompt 04 onward the convention adopted was to omit the SHA rather than embed a wrong
one. I followed that precedent: the log and the board identify the commit by its subject line, and
`git log --oneline -- prompts/source-remediation/logs/01-genericeos-sound-speed.md` resolves it in
one command. The alternative — recording a knowingly stale SHA and asking the orchestrator to fix
it up afterwards — trades a correct record for a cosmetic one.

## Verification performed

**Ran, passed.** `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .`
→ `Ran 5 tests in 0.015s / OK`.

**Ran, failed on the unfixed code, as required.** The one-line change was reverted locally, the
suite re-run, and the change restored (verified with `git diff` that only that line differs from
`e9a43a2`). Result: `Ran 5 tests ... FAILED (failures=11)` — every agreement subtest and every
Lambda-independence subtest failed; `test_omega_r_matches`, `test_wBackground_does_depend_on_omega_lambda`
and `test_limits` still passed (`test_limits`'s thresholds are loose enough that the defect does
not break them — it is a sanity test, not the regression guard). Printed values, unfixed code:

```
z=0     LambdaCDM=9.789220566828e-05  GenericEOS=3.046042774312e-05  ratio 0.31116  (1/3.2138)
z=0.5   LambdaCDM=1.468167501792e-04  GenericEOS=8.866657692238e-05  ratio 0.60393
z=1     LambdaCDM=1.957269309136e-04  GenericEOS=1.533144104607e-04  ratio 0.78331
z=2     LambdaCDM=2.935042263483e-04  GenericEOS=2.712752240816e-04  ratio 0.92426
z=10    LambdaCDM=1.073661170547e-03  GenericEOS=1.071883607626e-03  ratio 0.99834
z=1e3   LambdaCDM=7.574543482189e-02  GenericEOS=7.574543467247e-02  ratio 1 - 2.0e-9
Lambda-independence, unfixed: omega_cc 0.6889 -> 0.4 moves wPerturbations by
  +40.6% at z=0, +19.9% at z=0.5, +10.0% at z=1, +3.3% at z=2, +0.069% at z=10.
```

The ratios reproduce the audit's TK-1 measurement table exactly (3.2138 at z=0, 1.6558 at 0.5,
1.2766 at 1, 1.0819 at 2, 1.0017 at 10).

**Ran, same values with fixed code:**

```
z=0     LambdaCDM=9.789220566828e-05  GenericEOS=9.789220564955e-05  rel -1.9e-10
z=0.5   LambdaCDM=1.468167501792e-04  GenericEOS=1.468167499822e-04  rel -1.3e-09
z=1     LambdaCDM=1.957269309136e-04  GenericEOS=1.957269306837e-04  rel -1.2e-09
z=2     LambdaCDM=2.935042263483e-04  GenericEOS=2.935042263481e-04  rel -7.3e-13
z=10    LambdaCDM=1.073661170547e-03  GenericEOS=1.073661169122e-03  rel -1.3e-09
z=1e3   LambdaCDM=7.574543482189e-02  GenericEOS=7.574543480170e-02  rel -2.7e-10
```

**Ran `docs/spec-code-audit/scripts/TK_05_background_derivatives.py`, section (d), before and
after: byte-identical output, z=0 ratio 3.2138 both times.** This is not a null result — it is
because section (d) does not call `LambdaCDM_GenericEOS` at all. Lines 124-130 of the script
reconstruct the defective expression inline from a `LambdaCDM` object
(`wp_eos = (1/3) * cosmo.rho_r0*(1+z)**4 / cosmo.rho(z)`) so that the discrepancy can be exhibited
without paying for a GenericEOS construction. The script is therefore a *description* of the
defect that is unaffected by fixing it; the live before/after measurement above, on the actual
class, is the evidence. Full section (d) output, unchanged:

```
       z    w_P (LambdaCDM)  w_P (GenericEOS form)    ratio
       0      9.7892206e-05          3.0460428e-05   3.2138
     0.5      0.00014681675          8.8666577e-05   1.6558
       1      0.00019572693          0.00015331441   1.2766
       2      0.00029350423          0.00027127522   1.0819
       5      0.00058649204          0.00058055079   1.0102
      10       0.0010736612           0.0010718836   1.0017
     100      0.0096050353           0.0096050153   1.0000
```

**Reasoned, not run.** The consequences for stored data. `wPerturbations` feeds
`TkNumericIntegration.py:62,74-78` (the ODE), `WKB_Tk.py:9,17-23` (`omegaEff_sq`),
`TkWKBIntegration.py:47-49` (the LG friction integrand), `compute_analytic_T` — and, because
`LambdaCDM_GenericEOS` supplies no analytic `d_wPerturbations_dz`/`d2_wPerturbations_dz2`, the
`BackgroundModel` spline stack for `w'` and `w''` as well. So every stored `BackgroundModel`,
`TkNumericIntegration`, `TkWKBIntegration` and downstream row computed with a `GenericEOS`/QCD
model in an existing datastore is stale. No migration was attempted; recorded in
`IMPLEMENTATION_STATE.md` §5. Plain `LambdaCDM` datastores are unaffected — that class was already
correct.

## Observations not acted on

- **`docs/spec-code-audit/scripts/TK_05_background_derivatives.py` section (d) will keep printing
  a 3.2138 "discrepancy" after this fix**, because it recomputes the defective form inline rather
  than calling `LambdaCDM_GenericEOS`. It is an audit artefact, correct as a record of what was
  found; but a reader re-running the audit scripts post-campaign will be misled. Prompt 11 (spec
  and audit annotations) or prompt 12 (verification) may want to add a one-line note to the script
  or to the audit. Not touched here: the audit documents are prompt 11's territory and the
  scripts are a frozen record.
- **`LambdaCDM_GenericEOS` has no analytic `d_wPerturbations_dz` / `d2_wPerturbations_dz2`**, so
  those go through `BackgroundModel._build_derivative` — which is exactly the code path prompt 03
  (A7) addresses. Adding closed forms is possible in principle (`dT/dz` would be needed from the
  T(z) spline, so it would not be fully analytic) but is out of this prompt's scope and would
  overlap prompt 03's measurement.
- **`GenericEOSBase.w()` carries a TODO** noting the `4 g_S/(3 g) - 1` formula is invalid after
  e+e- annihilation, when the neutrino and photon temperatures separate. Untouched; it is a
  physics-modelling question in the EOS layer, not part of A1.
- **`_solve_T_z` uses `root_scalar(..., xtol=1e-6, rtol=1e-4)`** — the measured T(z) accuracy is
  ~3e-10 relative, so brentq is converging far past its stated tolerance and nothing is wrong
  today, but the stated tolerances do not describe the accuracy the rest of the code relies on.

## State handed to the next prompt

- **New test package `CosmologyModels/tests/`**, discoverable with
  `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .`
- **`PureRadiationEOS` and `lambdaCDM_gstar(Neff)`** live in
  `CosmologyModels/tests/test_wPerturbations.py` and are importable if another test needs a
  `LambdaCDM_GenericEOS` that is analytically equal to a `LambdaCDM`. They are test-only; do not
  promote them into production code.
- **Accuracy floor of any `LambdaCDM_GenericEOS` quantity against a closed form is ~1.3e-9
  relative at `max_z = 1e4` and ~6e-7 at the default `max_z = 1e20`**, set by the 500-point T(z)
  spline, not by the physics. Prompts 05 and 12 should not set tolerances below this when a
  GenericEOS/QCD model is in play, and should not read a residual of that size as a defect.
  Prompt 05's tests use a constant-`w` stand-in model rather than `GenericEOS`, so they are not
  affected.
- **`wPerturbations` is now Lambda-free in both cosmology classes**, so the spec 01 Tier 3 /
  spec 03 §0.5 convention holds uniformly. Prompt 11's spec 01 R16/Q4 annotation can state that
  the literal-R16 transcription was removed from the code at this commit.
