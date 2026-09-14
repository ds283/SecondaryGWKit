# Log 01 — The background-against-background harness, and the guard that would have caught T1

**Prompt:** prompts/qcd-background-audit/01-background-reference-harness.md
**Commit:** *(this commit)* — Add a background-against-background test for the QCD temperature
**Model:** Claude Opus 5
**Date:** 2026-09-14
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

**No production file was touched.** `git diff --stat` against `2a5e0fa` shows two new test modules,
this log, the campaign board and `docs/OPEN_ISSUES.md`. `T_Z_REPRESENTATION_VERSION` does not exist
at this commit — it is introduced by prompt 03 — so its value before and after is *(not yet
defined)*.

### `CosmologyModels/tests/T_z_reference.py` (new, 411 lines)

The measurement infrastructure. Its module docstring states the independence rule in the same
terms `ComputeTargets/tests/wkb_reference.py` states its own: nothing in the reference path calls
`_solve_T_z`, `_build_T_z_spline`, `T_photon`, `_T_z_spline`, `_T_z_spline_knots_log1pz` or
`integration_break_points`.

Public surface, with the exact signatures prompt 05 and prompt 06 will import:

| Symbol | Signature | What it is |
|---|---|---|
| `accurate_T` | `accurate_T(cosmology, z: float, rtol: float = 1.0e-14) -> float` | The defining equation, bracketed `[0.95 T_CMB (1+z)/g*^(1/3), 1.05 T_CMB (1+z)]`, solved with `root_scalar(..., xtol=1e-300, rtol=rtol)`. **The reference for everything in this campaign.** |
| `entropy_factor` | `entropy_factor(cosmology, u: float, rtol: float = 1.0e-14) -> float` | `F(u) = log(T / (T_CMB (1+z)))` = `-(1/3) log(Gs(T)/Gs_CMB)` |
| `tabulated_u_range` | `tabulated_u_range(cosmology) -> tuple` | `(u_lo, u_hi)`, the interval `_build_T_z_spline` covers, reconstructed from the model's declared `max_z` and `DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT` with the 5% buffer in `1+z` |
| `jump_locations` | `jump_locations(cosmology, rtol: float = 1.0e-14) -> list` | Ascending `u` at which `T(z)` crosses each `break_temperatures_GeV`, by **geometric bisection of the monotone `T(z)`** to a relative `1e-15` in `1+z` |
| `production_source_z_values` | `production_source_z_values(z_init=PRODUCTION_Z_INIT, z_end=PRODUCTION_Z_END, samples_per_log10z=PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z) -> np.ndarray` | `populate_z_sample`'s geometry; 1,732 descending samples on the production defaults |
| `probe_set` | `probe_set(z_values=None, z_min=PROBE_Z_MIN, z_max=PROBE_Z_MAX, decimation=PROBE_DECIMATION) -> np.ndarray` | The audit's 640-point probe geometry: grid nodes **and** the midpoints between them in `u`, restricted to `z in (1, 1e16)`, decimated by 5 |
| `reference_temperatures` | `reference_temperatures(cosmology, probe_z, rtol=1.0e-14) -> np.ndarray` | `accurate_T` over a probe set; ~5 ms for the production 640 |
| `relative` | `relative(candidate, reference) -> np.ndarray` | elementwise `abs(c - r)/abs(r)` |
| `Stats` | `NamedTuple(max, p90, median)`, with `Stats.of(rel) -> Stats` and `Stats.format(label="") -> str` | the campaign's three statistics, **always in that order** |
| `temperature_override` | `@contextmanager temperature_override(cosmology, T_of_z)` | assigns `cosmology._T_z_spline` for the duration; the measurement hook |
| `Hubble_with` | `Hubble_with(cosmology, T_of_z, z_values) -> np.ndarray` | `H(z)` under a substituted temperature (prompt 06 needs this for README §6.2) |
| `inverse_Hubble_integral` | `inverse_Hubble_integral(cosmology, T_of_z, u_a, u_b, points=None, epsrel=1.0e-11, limit=400) -> float` | `int dz/H` in `u`, with `IntegrationWarning` suppressed and the audit's convergence justification in the docstring |

Module constants: `PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z = 100`, `PRODUCTION_Z_END = 0.1`,
`PRODUCTION_Z_INIT = 2.0636395964161516e16`, `PROBE_Z_MIN = 1.0`, `PROBE_Z_MAX = 1.0e16`,
`PROBE_DECIMATION = 5`, `PRODUCTION_SPANS_RAD`, `PRODUCTION_FLOOR_RAD`.

### `CosmologyModels/tests/test_T_z_representation.py` (new, 483 lines)

Seven cases in two classes. Every threshold is a named module constant carrying a comment that
says which prompt tightens it, so the characterisation→guard transition is a one-line edit.

Numbered as the prompt numbers them (§2.2 cases 1–6, §2.3 case 7), which is the numbering used
throughout this log.

`TestQCDTemperatureRepresentation` (cases 1–5 and 7; builds `QCD_Cosmology(store_id=0,
Mpc_units(), Planck2018(), max_z=1e20)`, the probe set, the reference and the jump locations once
in `setUpClass`):

1. `test_T_z_matches_the_defining_equation`
2. `test_the_node_solve_converges`
3. `test_conformal_time_matches_the_exact_background` — **the T1 guard**
4. `test_the_branch_joins_are_where_the_fixture_puts_them`
5. `test_T_z_is_a_step_at_the_lowest_crossing`
7. `test_a_segment_edge_bisected_and_one_root_found_disagree`

`TestConstantEntropyEquationOfState` (case 6; `LambdaCDM_GenericEOS` on `PureRadiationEOS` at the
same `max_z = 1e20`):

6. `test_a_constant_gs_equation_of_state_is_an_exact_ramp`

---

## Deviations from the prompt

### 1. The probe grid is reproduced locally, not imported from `ComputeTargets.tests.wkb_reference` — IMPLEMENTATION CHOICE

The prompt left this open: *"The grid itself may come from
`ComputeTargets.tests.wkb_reference.production_source_grid`; if importing across test packages is
awkward … reproduce the geometry locally from the documented constants and **say which you did**."*

It is not awkward — the import works under `discover -s CosmologyModels/tests -t .` and costs
1.7 s — but it was rejected for two reasons:

* `wkb_reference` imports `ComputeTargets.BackgroundModel`, hence Ray, into a package that
  otherwise needs neither. `CosmologyModels/tests` runs in 0.11 s today and that is worth keeping.
* `production_source_grid`'s `z_init` comes from `wkb_reference_data.json`, which **prompt 02
  makes regenerable and prompts 04–06 regenerate**. `z_init` is a horizon-exit solve against
  `H(z)`, so it moves with the background. Importing it would shift the probe set out from under
  README §6.1's acceptance table at exactly the moment the table is being read.

`PRODUCTION_Z_INIT = 2.0636395964161516e16` is therefore pinned as a module constant, with a
comment saying where it came from and why it is frozen. **Verified identical:** the locally built
grid is `np.array_equal` to `production_source_grid(refs["models"]["LambdaCDMModel"]["grid"]
["z_init"]).as_float_list()` — 1,732 samples, bit for bit — and the probe set is the audit's
640 points, reproducing every §6.1 figure to the digit (see Verification).

### 2. Case 5 asserts 7.622923e-04, not 7.614e-04 — STRUCTURALLY REQUIRED

The prompt asks for *"a step of **7.614e-04** between them"*. The step measured in `F(u)` is
**7.6229229003969e-04**, and the implied relative jump in `T` is **7.625829e-04**. Neither is
7.614e-04 in the third significant figure.

All three numbers are in the audit and all three are right; they are three different quantities:

* audit §1's `forced dT/T` column is the **linearised** `-(1/3) dgs/gs` = **7.614213e-04**;
* audit §2's own table prints `F` as `0.000762292290`, which is `-(1/3) log(1 + dgs/gs)` and is
  what a step in `F` measures — **7.622923e-04**, matching to all twelve printed decimals;
* the relative jump in `T` is `expm1` of that — **7.625829e-04**.

The prose in audit §2 ("jumps by a relative 7.614e-04") quotes the §1 linearisation rather than its
own table. The test pins **both**: the step in `F` to `7.622923e-04 ± 1e-09`, and `expm1(step)`
against the audit's quoted `7.614e-04` to `± 2e-06`, with the linearisation named in a comment and
printed at run time. Nothing in the audit is rewritten (`CLAUDE.md`: additive only).

**Prompt 06 must use 7.6229229003969e-04 in `F`, not 7.614e-04**, if it asserts a jump height.
This is carried in "State handed to the next prompt" rather than opened as a board issue: it is a
clarification of which of three correct figures applies where, not a defect in the tree.

### 3. Case 7 (§2.3) takes the prompt's *second* branch: the solver returns a non-root — STRUCTURALLY REQUIRED

The prompt allows either outcome: *"Assert that the two **do not** agree to better than a grid
interval, or — if the bracketing solver raises or returns a non-root — that it fails."*

Measured, the first branch is false and the second is true. `root_scalar` bracketed on
`accurate_T(z) - T_break` **converges**, and returns a `u` between 4 and 317 ulp from the bisected
edge depending on its tolerances — far below the production grid spacing of 2.3032e-02. So a
grid-interval comparison would find them equal.

But it is not a root. `(T - T_break)/T_break` is **+8.844019e-06** at the bisected edge and
**−7.531645e-04** one ulp of `u` below it: the sign changes inside a single ulp **without passing
through zero**, and the residual at every point the solver returns is +8.84e-06 — 1.2 % of the jump
height. Audit §2's central claim therefore holds on this tree, and the test asserts exactly that:
no root exists, and the solver reports one anyway.

**This is not the stop condition.** The stop was "a test that finds them equal", meaning a genuine
root; what is measured is a reported convergence onto a point that is not a root. The test asserts
`converged is True` *and* `abs(residual) > 1e-08` for three bracketings, which is a stronger
statement than the prompt's first branch would have been.

The practical consequence is quantified rather than asserted, because it is prompt 06's to act on:
a naive implementation bracketing over the **whole tabulated range** with `root_scalar`'s default
tolerances returns `u_edge + 1.126e-12`, which is **more than** the `pad = 1e-12` the audit's
`build_segmented` uses. The segment below the jump would then take its topmost node from
`u_edge + 1.26e-13` — above the jump — and interpolate across the discontinuity after all. With
tight tolerances (`xtol = rtol = 1e-15`) the offset falls to `+1.421e-14` (4 ulp) and the same
naive code would be safe, which is precisely why the correctness must not rest on a tolerance
setting. The measured offsets are printed by the test.

### 4. Two classes rather than one — IMPLEMENTATION CHOICE

Case 6 (`PureRadiationEOS`) needs a different cosmology from the other six, and
`setUpClass` is the campaign's stated way of paying for a model once. Splitting it into
`TestConstantEntropyEquationOfState` avoids building both models for every test. The alternative —
one class with a lazily built second model — was rejected as more machinery for no benefit. The
module still contains exactly the seven named cases.

### 5. `Hubble_with` is shipped but not called by any test — IMPLEMENTATION CHOICE

README §6.2 makes `H(z)` relative error on the production grid an acceptance quantity for prompt
06, and audit §5 measures it. The helper is four lines on top of `temperature_override`, which the
T1 guard does use, and shipping it now means prompt 06 does not have to add to this module. The
alternative — leaving prompt 06 to write it — was rejected because the harness is meant to be the
thing later prompts are scored against, not a thing they extend.

---

## Verification performed

All figures below were **run**, not reasoned about. Commands were executed from the repository root
with `PYTHONPATH=.`.

### Suite counts

| Suite | Before (`2a5e0fa`) | After | Wall |
|---|---|---|---|
| `CosmologyModels/tests` | 11 OK | **18 OK** | 0.11 s |
| `ComputeTargets/tests` | 339 OK | **339 OK** | 157.0 s |
| `LiouvilleGreen/tests` | 148 OK | **148 OK** | 1090 s |

The `CosmologyModels` count rises by seven, as the prompt requires. Nothing else moved.

### Every threshold, with its measured value

| # | Test | Asserted | Measured | Audit / README |
|---|---|---|---|---|
| 1 | `test_T_z_matches_the_defining_equation` | max ≤ 7.2e-04 | **7.177180e-04** | 7.177e-04 ✓ |
| 1 | | p90 ≤ 1.4e-05 | **1.322901e-05** | 1.323e-05 ✓ |
| 1 | | median ≤ 2.0e-07 | **1.890443e-07** | 1.890e-07 ✓ |
| 2 | `test_the_node_solve_converges` | max ≤ 2.5e-05 | **2.496322e-05** | 2.496e-05 ✓ |
| 3 | `test_conformal_time_matches_the_exact_background` | rel ≤ 4e-08 | **3.4605051e-08** | 3.461e-08 / 3.4605e-08 ✓ |
| 4 | `test_the_branch_joins_are_where_the_fixture_puts_them` | see below | see below | audit §1 ✓ |
| 5 | `test_T_z_is_a_step_at_the_lowest_crossing` | step in `F` = 7.622923e-04 ± 1e-09 | **7.6229229003969e-04** | audit §2 table ✓ |
| 5 | | each side flat to ≤ 1e-12 | **0.0** below, **2.218e-16** above | "twelve decimals" ✓ |
| 5 | | `expm1(step)` = 7.614e-04 ± 2e-06 | **7.625829e-04** | deviation 2 |
| 6 | `test_a_constant_gs_..._exact_ramp` | `accurate_T` max ≤ 1e-15 | **2.218198e-16** | — |
| 6 | | `_solve_T_z` max ≤ 1e-15 | **2.218198e-16** | — |
| 6 | | shipped `T_photon` max ≤ 2.0e-07 | **1.940083e-07** | — |
| 7 | `test_a_segment_edge_...disagree` | `abs(residual)` > 1e-08 | **8.844019e-06** | deviation 3 |

The audit was taken on `b3e3769`, this tree is `2a5e0fa`, and no production file changed in
between. **Every figure agrees to every digit the audit quotes.** Nothing differs even in the last
place.

### Case 3 in full — the T1 guard

```
int dz/H over z in [1e+02, 1e+12], 3 interior jumps given to the integrator
  shipped background = 1.3320002968728165e+03
  exact   background = 1.3320002507788795e+03
  relative error in tau = 3.4605e-08
    k =     1e+05 /Mpc:   4.751e+01 rad   against a 1-ulp floor of 3.050e-07 rad
    k =     1e+07 /Mpc:   4.751e+03 rad   against a 1-ulp floor of 3.050e-05 rad
    k =     3e+08 /Mpc:   1.425e+05 rad   against a 1-ulp floor of 9.150e-04 rad
```

The audit's §5 row is 3.461e-08 / 4.75e+01 / 4.75e+03 / 1.43e+05 rad. Reproduced.

### Case 4 in full — the equation-of-state branch joins

Evaluated at `T*(1 ± 1e-9)`:

| `T_break` [GeV] | `dg/g` measured | audit §1 | `dgs/gs` measured | audit §1 | forced `dT/T` |
|---|---|---|---|---|---|
| 1e+16 | +1.454185e-02 | +1.454e-02 | +1.394721e-02 | +1.395e-02 | −4.649069e-03 |
| 0.12 | −2.075036e-04 | −2.075e-04 | −3.743558e-04 | −3.744e-04 | +1.247853e-04 |
| 0.002 | +1.751081e-11 | +1.751e-11 | +1.754690e-11 | +1.755e-11 | −5.848965e-12 |
| 1e-05 | +8.875740e-04 | +8.876e-04 | −2.284264e-03 | −2.284e-03 | +7.614213e-04 |

The three discontinuous joins are asserted to a relative 1e-3 of the audit's four-significant-figure
values (worst agreement 2.0e-4, at `dgs/gs` for 1e16 GeV); the 0.002 GeV join is asserted `|·| ≤
1e-10` and measures 1.75e-11. The test also asserts that `break_temperatures_GeV` is still exactly
these four, so that a change to the fixture's *set* of joins fails loudly rather than silently
skipping a row.

### Case 7 in full — the segment-edge trap

```
bisected edge u = 17.565806941870026  (z = 4.253369e+07)
  (T - T_break)/T_break at the edge     = +8.844019e-06
  (T - T_break)/T_break one ulp below   = -7.531645e-04
  naive, full range, defaults      converged=True  u - u_bisect = +1.126e-12 (+317.0 ulp)  residual = +8.844020e-06
  naive, full range, tight         converged=True  u - u_bisect = +1.421e-14   (+4.0 ulp)  residual = +8.844019e-06
  naive, local bracket, defaults   converged=True  u - u_bisect = +3.304e-13  (+93.0 ulp)  residual = +8.844020e-06
  segmented build pads by 1e-12 in u; production grid spacing 2.3032e-02
```

### Wall time

The module runs in **1.30 s** measured from `loadTestsFromName` to the end of the run, of which
~1.2 s is importing `numpy`/`scipy` and the cosmology package. The seven test bodies together take
**0.07 s**, and `setUpClass` for the QCD class — cosmology, probe set, 640-point reference, jump
locations — takes **0.039 s**.

`test_conformal_time_matches_the_exact_background` takes **0.013 s**, not the ~20 s the prompt
budgeted. The exact integrand root-solves on every call at 8 µs, but `quad` with the three interior
jumps supplied needs only a few hundred evaluations over the ten decades. No range reduction was
needed and none was made: the guard runs over the full `z in [1e2, 1e12]`.

### `black`

`./venv/bin/python -m black --check CosmologyModels/` — 17 files unchanged, clean.

---

## Observations not acted on

1. **The audit quotes the jump height three ways.** Deviation 2 above. Not opened as a board issue:
   all three figures are correct for their own quantity, the audit is additive-only, and the
   clarification is carried to prompt 06 in the handover below.

2. **`_solve_T_z` is exact on a constant-`g_s` equation of state despite `rtol = 1e-4`.** Measured
   2.218e-16 max over the 640 probes, i.e. round-off. The mechanism is *reasoned*, not measured:
   with `g_s` constant the defining equation is linear in `T`, so Brent's interpolation step lands
   on the root exactly and the `rtol` never binds. What is measured is that the shipped node solve
   carries no error at all on
   `PureRadiationEOS` — and the shipped *spline* over those exact nodes is still wrong by
   1.940e-07. That is finding **T3 in isolation**, with T2 and T4 both switched off, and it is a
   cleaner demonstration of T3 than anything in audit §3. Recorded here because prompt 05's
   acceptance test is on this model and should expect a few ulp, not 1.9e-07.

3. **`[01-genericeos-tz-spline-floor]`'s quoted floor and this prompt's max are different
   quantities, and a later reader could take them for the same one.** `docs/OPEN_ISSUES.md` §1.7
   quotes "1.3e-9 at `max_z=1e4`, 6.4e-7 at the default 1e20"; the comment at
   `CosmologyModels/tests/test_wPerturbations.py:34` quotes "~1.3e-9 … and ~4e-7", and that one is
   an agreement between `LambdaCDM.wPerturbations` and `LambdaCDM_GenericEOS.wPerturbations` on a
   `PureRadiationEOS` model, not a `T(z)` error. What this prompt measures, on the 640-point probe
   set at `max_z = 1e20`, is: QCD, shipped spline, **max 7.177e-04 / median 1.890e-07**; and
   `PureRadiationEOS`, shipped spline against the closed form, **max 1.940e-07 / median
   1.098e-07**. None of these contradicts the issue, but none of them *is* the issue's number
   either. Prompts 05 and 06 own `[01-genericeos-tz-spline-floor]` and should restate it against
   one definition; nothing was changed here.

4. **`entropy_factor` is exactly `0.0` below the lowest crossing, not merely small.** All three
   probes at `z_c × (0.95, 0.99, 0.999)` return `0.0` bit-exactly, because `Gs` is the literal
   constant 3.940 there and equals `Gs(T_CMB)`. Worth knowing for prompt 05: on that branch the
   entropy-factor representation has nothing to approximate, so a spline of `F` is exact by
   construction and the only error left is the `exp(F) = exp(0)` round trip.

---

## State handed to the next prompt

**Module paths.** `CosmologyModels/tests/T_z_reference.py` (the harness) and
`CosmologyModels/tests/test_T_z_representation.py` (the seven cases). Run them with

```bash
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_T_z_representation -v
```

**The reference.** `accurate_T(cosmology, z, rtol=1.0e-14)`. Prompts 04–06 must score against this
and nothing else; `_solve_T_z` is a subject, never a reference (README §2 (a)). The full public
surface and its signatures are the table in "What shipped".

**The probe set.** `probe_set()` — 640 points, `z ∈ [1.0006165874356605,
9539907715036264]`, fixed for the whole campaign by `PRODUCTION_Z_INIT = 2.0636395964161516e16`.
**Do not re-derive it from `wkb_reference_data.json`**, which prompts 04–06 regenerate; that would
move the acceptance table's own abscissae. Every figure in README §6.1 is measured on this set.

**The three `jump_locations()` values, to 17 digits** (prompt 06's segment edges):

| # | `u = log(1+z)` | hex float | `z` |
|---|---|---|---|
| 1 | `17.565806941870026` | `0x1.190d8b9472e79p+4` | `42533685.432205893` |
| 2 | `23.197460552819649` | `0x1.7328cc6589c48p+4` | `11872142813.468176` |
| 3 | `27.485391822044257` | `0x1.b7c42a3716d0ap+4` | `864478111114.07019` |

These are the crossings of `T_LO = 1e-5`, `EOS_T_LO = 0.002` and `T_120_MEV = 0.12` GeV, in that
order — i.e. **`break_temperatures_GeV`, four of them, three in range** (README §7 D4 keeps this
choice). `T_HI = 1e16` GeV is crossed at `z ~ 1e28`, far above the tabulated range, and
`jump_locations` drops it because it cannot be bracketed inside `z ∈ [1e-6, 1e19]`.

The hex forms are given because these are the one place in the campaign where the last bit matters:
`u_1` sits on the **upper** branch (`(T − T_break)/T_break = +8.844019e-06`) and `u_1 − 1 ulp` sits
on the lower one (`−7.531645e-04`).

**The tabulated range** at the production `max_z = 1e20`: `u ∈ [-0.2744368457017603,
46.100492024050347]`, from `tabulated_u_range(cosmology)`. Build any candidate over this interval
so that it is scored over the same interval the shipped spline covers.

**The jump height, and which figure to use.** Prompt 06 should assert the step in `F` as
**7.6229229003969e-04** (`= -(1/3) log(1 + dgs/gs)`), or the relative jump in `T` as
**7.625829086481684e-04** (`= expm1` of it). The **7.614e-04** in audit §2's prose and README §2
(b) is §1's linearised `-(1/3) dgs/gs = 7.614213e-04` — correct, but a different quantity, and
0.15 % away. See deviation 2.

**The segment-edge trap, quantified.** `jump_locations` bisects. A `root_scalar` bracket on
`T(z) − T_break` reports `converged=True` and returns a non-root whose position depends on the
tolerance: `+1.126e-12` in `u` above the bisected edge with defaults over the full range,
`+3.304e-13` with defaults over a local bracket, `+1.421e-14` with `xtol = rtol = 1e-15`. The
audit's `build_segmented` uses `pad = 1e-12`, so the **first** of those three misplaces the lower
segment's topmost node onto the upper branch. Prompt 06 must not root-find, whatever tolerance it
would have used.

**Achieved accuracies on this tree** (all on the 640-point probe set, `max_z = 1e20`, QCD):

| | max | p90 | median |
|---|---|---|---|
| shipped `T_photon` | 7.177180e-04 | 1.322901e-05 | 1.890443e-07 |
| shipped `_solve_T_z` | 2.496322e-05 | 1.245711e-05 | 1.221913e-08 |

and, on `PureRadiationEOS` at the same `max_z`, against the closed form `T = T_CMB (1+z)`:

| | max | p90 | median |
|---|---|---|---|
| `accurate_T` (the reference) | 2.218198e-16 | 1.664952e-16 | 0.0 |
| shipped `_solve_T_z` | 2.218198e-16 | 1.664952e-16 | 0.0 |
| shipped `T_photon` | 1.940083e-07 | 1.901873e-07 | 1.097960e-07 |

**The T1 guard's numbers**: shipped `1.3320002968728165e+03`, exact `1.3320002507788795e+03`,
relative `3.4605051325764105e-08`, over `z ∈ [1e2, 1e12]` with `epsrel = 1e-11`, `limit = 400` and
the three interior jumps passed as `points`. Prompt 06 tightens `CONFORMAL_TIME_REL` from `4.0e-08`
to `1e-15`.

**Costs.** The module is 1.30 s wall including imports; the seven test bodies are 0.07 s; the
QCD `setUpClass` (cosmology + probe set + 640-point reference + jump locations) is 0.039 s; the T1
guard alone is 0.013 s. There is room for prompt 06 to add a second, tighter integral without
troubling the suite.

**The QCD reference fixture.** Not touched here and **not yet regenerable in one command** — that
is prompt 02's deliverable, and this log cannot quote the command because it does not exist at this
commit. Nothing in this prompt moved a number, so `ComputeTargets/tests/wkb_reference_data.json` is
untouched and all 339 `ComputeTargets` tests pass against it unchanged.

**`T_Z_REPRESENTATION_VERSION`**: does not exist at this commit. Prompt 03 introduces it at 1.

**Thresholds to tighten, and where.** All are module constants at the top of
`test_T_z_representation.py`, each with a comment naming its prompt:

| Constant | Now | Prompt 04 | Prompt 05 | Prompt 06 |
|---|---|---|---|---|
| `T_PHOTON_MAX` | 7.2e-04 | — | — | 1e-10 |
| `T_PHOTON_P90` | 1.4e-05 | 2.0e-07 | 9.0e-08 | 1e-14 |
| `T_PHOTON_MEDIAN` | 2.0e-07 | 1.1e-07 | 3.0e-10 | 1e-15 |
| `NODE_SOLVE_MAX` | 2.5e-05 | 1e-14 | — | — |
| `CONFORMAL_TIME_REL` | 4.0e-08 | — | — | 1e-15 |
| `EXACT_RAMP_MAX` | 2.0e-07 | — | a few ulp | — |

`BRANCH_JOINS`, `CONTINUOUS_JOIN_BOUND`, `ENTROPY_FACTOR_STEP`, `LINEARISED_JUMP`,
`BRANCH_FLATNESS`, `NON_ROOT_RESIDUAL` and `SEGMENT_PAD` are **characterisations, not accuracy
targets**: they do not tighten, and a failure in one of them means the fixture or the arithmetic
changed underneath, not that the representation regressed.
