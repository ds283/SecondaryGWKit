# Log 05 — Spline the entropy factor, not the temperature: fix the median (T3)

**Prompt:** prompts/qcd-background-audit/05-entropy-factor-representation.md
**Commit:** *(this commit)* — Spline the entropy factor rather than the temperature itself
**Model:** Claude Opus 5
**Date:** 2026-09-14
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

**`T_Z_REPRESENTATION_VERSION` before this commit: `2`. After: `3`.**

### `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`

- **New class `TemperatureRepresentation`** (`:52`), the object `_build_T_z_spline` now returns.
  Signature: `TemperatureRepresentation(spline, T_CMB: float, label: str, min_z: float,
  max_z: float)`; `__call__(self, z: float, z_is_log: bool = False) -> float`. It holds the
  SciPy `BSpline` of `F(u)` and evaluates `T_CMB * (1 + z) * exp(F(u))`. This is README §7 **D2**,
  shape (ii); see deviation-free discussion under *Choices* below.
- **Module constants** `DEFAULT_T_Z_SPLINE_SAMPLES = 500` and `DEFAULT_T_Z_SPLINE_ORDER = 3`
  (`:48-49`), with the measured accuracy of five candidate (nodes, order) pairs tabulated in the
  comment above them.
- **`_build_T_z_spline`** (`:351`): signature gains `samples` / `order` defaults from those
  constants and the return type becomes `TemperatureRepresentation`. Body: the node loop that was
  `T_values = [self._solve_T_z(exp(logz) - 1.0) for logz in log_z_values]` is now
  `z_values = [exp(logz) - 1.0 ...]` followed by
  `F_values = [log(self._solve_T_z(z) / (self._T_CMB * (1.0 + z))) for z in z_values]` — **the
  same single `_solve_T_z` call per node**, no second solve. `make_interp_spline(log_z_values,
  F_values, k=order)`. The 5 % buffer in `1+z` and its comment are byte-identical. The
  `_T_z_spline_knots_log1pz` assignment is unchanged and still reads `spline.t`.
- **Import** `from ComputeTargets.spline_wrappers import ZSplineWrapper` →
  `... import _outward`. `ComputeTargets/spline_wrappers.py` itself is **not touched**.
- **`T_Z_REPRESENTATION_VERSION`** (`:174`): `2` → `3`, with a new row in the version table
  (`3 | 05 | the entropy factor F(u) is splined, not T itself; T = T_CMB (1+z) e^F`) and the
  "prompts 05, 06 and 07 each append a row" line narrowed to 06 and 07.

### `CosmologyModels/tests/test_T_z_representation.py`

| constant | before | after | achieved |
|---|---|---|---|
| `T_PHOTON_MAX` | `7.27e-04` | `7.24e-04` | 7.236e-04 |
| `T_PHOTON_P90` | `2.0e-07` | `9.0e-08` | 8.912e-08 |
| `T_PHOTON_MEDIAN` | `1.1e-07` | `3.0e-10` | 2.599e-10 |
| `EXACT_RAMP_MAX` | `2.0e-07` | `1.0e-15` | 2.928e-16 |
| `CONFORMAL_TIME_REL` | `4.0e-08` | `4.0e-08` (unchanged) | 5.4264e-10 |

Comments above each rewritten. `CONFORMAL_TIME_REL`'s is the only one that is *not* a threshold
change: prompt 06 owns it, so the 64× improvement is recorded in the comment instead. The
docstring of `test_a_constant_gs_equation_of_state_is_an_exact_ramp` is rewritten to say that the
representation is now exact on a constant-`g_s` model rather than merely accurate.

### `CosmologyModels/tests/test_temperature_spline.py`

`INTERPOLATION_FLOOR` `1.3e-9` → `1.0e-15` (achieved: buffer effect max 5.646e-16, median exactly
0; worst departure from `T_CMB (1+z)` 1.110e-16), with the comment rewritten to say it is a
round-off floor on this model now, not an interpolation floor. The module docstring's
`[01-genericeos-tz-spline-floor]` paragraph narrowed. **The four range tests are untouched.**

### `ComputeTargets/tests/wkb_reference_data.json`

Regenerated via `docs/qcd-background-audit/generate_qcd_references.py` (no hand editing), 142.2 s,
no Ray, no datastore. Largest relative move per key:

| key | max relative move |
|---|---|
| `rho_G` | 7.475097e-03 |
| `rho_T` | 7.169622e-06 |
| `rho_anchor_z` | 2.913898e-07 |
| `cs_tau_minus_top` | 2.372032e-07 |
| `tau_minus_top` | 2.372042e-07 |
| `primitives_at_rho_anchor` | 2.019482e-07 |
| `friction_F_minus_top` | 3.308367e-09 |
| `reference_floor` | up to 1.2e+01, `short_baseline` up to 5.0e-01 — self-agreement figures already at the 1e-15/1e-16 floor; the absolute moves are ~1e-15 |
| `grid`, `z_top`, `checkpoints` | 0 (unchanged) |

### `ComputeTargets/tests/` — three tolerances and one assertion

See **Deviations** 2 and 3. `QCD_BREAK_POINT_ALIGNMENT_TOL` 1.4e-05 → 3.1e-05;
`QCD_FLOOR_FACTOR` 3.0 → 3.2 (`test_background_tau.py`) and 3.0 → 8.3
(`test_background_cs_tau_friction.py`); `test_the_branch_crossing_is_inside_the_range`'s literal
`8.64366999e11` → `8.6438180e11` at unchanged `places=6`; new class constant
`TestQCDReferenceConvergence.UNSPLIT_PENALTY_FACTOR = 5.0` replacing the `assertGreater(unsplit,
1e-6)` premise, plus a printed line and a rewritten docstring carrying both measurements.

### Campaign and board files

- `prompts/qcd-background-audit/IMPLEMENTATION_STATE.md`: prompt 05's row, the progress counter,
  the header date, the version table, the T1 paragraph, the T3 mechanism row, the escalation of
  `[01-convergence-block-has-a-separate-generator]`, and the new §3 issue
  `[04-unsplit-tk-run-now-meets-the-criterion]`.
- `prompts/source-remediation/IMPLEMENTATION_STATE.md`: `[01-genericeos-tz-spline-floor]` narrowed
  in §3 with the measured figures (prompt §5).
- `docs/OPEN_ISSUES.md`: count `58` → `59`; the new issue's row added under §1.7;
  `[01-genericeos-tz-spline-floor]`'s hook narrowed; §1.7's "1 executed" corrected to "5 executed"
  (drift flagged in log 04's observations).

## Choices

**README §7 D2 — a `TemperatureRepresentation` class, not a composed callable in `ZSplineWrapper`.**
The alternatives were (i) keep `ZSplineWrapper` and hand it a callable in `u` that already
contains the ramp and the `exp`, and (ii) a small class honouring the same contract. **(ii) was
taken, for three reasons and one blocker.** The blocker: under (i) the wrapper's soft clamp acts
only on what it passes to the spline, so above `max_z` it would clamp `F` while the `(1+z)` factor
inside the composed callable kept climbing — the clamp would stop clamping. Under (ii) both are
clamped together (`raw_z` is set to `self._max_z` alongside `log_z`), so the value returned at a
softly-clamped bound is `T(max_z)`, exactly what the old wrapper returned. The three reasons: the
wrapper's `log_z=True` / `deriv=True` branches would have come to mean something the docstring
does not say (the prompt names this); the range logic and the representation now sit in one
readable place, which is the thing prompt 06 has to segment; and a class can carry the node/order
provenance in its own docstring. **Cost of (ii):** ~20 lines of `ZSplineWrapper`'s bounds logic
are reproduced. That is mitigated by importing `_outward` rather than re-deriving it — one
definition of "outward" in the repository — and by keeping the `RuntimeError` text byte-identical,
prefix included, so nothing that reads or greps it changes. The internal attribute is deliberately
named `_spline` because `docs/gktk-remedial/residual_convergence.py:268` reads
`cosmology._T_z_spline._spline.t` for the knot vector, and that script is prompt 08's tool.

**README §7 D3 — 500 nodes at `k = 3`, unchanged.** Measured on the audit's 640-point probe set:

| nodes | order | max | p90 | median | build | interior knots |
|---|---|---|---|---|---|---|
| **500** | **3** | **7.236e-04** | **8.912e-08** | **2.599e-10** | **12.4 ms** | **498** |
| 500 | 5 | 7.226e-04 | 2.022e-08 | 8.099e-13 | 11.9 ms | 496 |
| 1000 | 3 | 1.569e-04 | 4.513e-09 | 8.850e-12 | 24.3 ms | 998 |
| 2000 | 3 | 5.066e-05 | 2.508e-10 | 1.476e-13 | 50.5 ms | 1998 |
| 2000 | 5 | 5.580e-05 | 9.002e-14 | 2.320e-16 | 48.7 ms | 1996 |
| 3000 | 5 | 3.730e-04 | 7.062e-15 | 1.943e-16 | 71.9 ms | 2996 |

Every one of the audit §4 rows it overlaps reproduces to the digit. The count was held fixed
because **the max is pinned at the jump height at every density** — it is set by one spline running
straight across a genuine discontinuity, which only segmentation touches — so extra nodes buy the
p90 and the median only, and they are paid for one-for-one in declared break points: every interior
knot is returned by `integration_break_points`, and every quadrature and every ODE in the tree
splits a panel at each of them. 2,000 nodes would have quadrupled `BREAK_POINT_ALL` from 407 to
~1,600 in range, changed how `BackgroundModel` splits every interval, and made
`test_background_tau.py::test_qcd_break_points`'s count assertion **false** rather than merely
inaccurate — which is `[02-fixture-tests-pinned-to-todays-break-point-artefact]`, and prompt 07's,
not this prompt's. Holding it fixed also keeps the campaign's separation intact: prompt 05 moves
exactly one thing, the quantity splined. **Note for prompt 06:** 3,000 / `k=5` *unsegmented* is
worse in the max than 2,000 / `k=5` (3.730e-04 against 5.580e-05); the audit's recommendation of
3,000 / `k=5` is a *segmented* figure and should not be read as an unsegmented one.

## Deviations from the prompt

### 1. The prompt stopped and asked before landing — STRUCTURALLY REQUIRED

Prompt §2 item 6 makes more than one loosened tolerance a stop, and three had to loosen. Work was
completed and verified, then paused with the tree uncommitted and the question put to the user
through the orchestrator, with all four failures measured on both trees. **The user chose option
(a)**: loosen the three with comments attributing them to the stale block, re-score the literal,
edit the falsified assertion, and land as `COMPLETE WITH DEVIATIONS`. The user explicitly withheld
authorisation to run `docs/gktk-remedial/residual_convergence.py`, which stays prompt 08's. No
number changed between the stop and the landing except the four test edits below; the JSON
regeneration was not repeated.

### 2. Three tolerances loosened — IMPLEMENTATION CHOICE (user-authorised), single root cause

All three are `[01-convergence-block-has-a-separate-generator]` and nothing else: the JSON's QCD
block was regenerated here, the top-level `convergence` block was not, so a figure measured on the
entropy-factor background is being scored against a floor recorded on the `T`-against-`u` one.

| constant | module | was | now | measured |
|---|---|---|---|---|
| `QCD_BREAK_POINT_ALIGNMENT_TOL` | `test_background_tau.py` | 1.4e-05 | 3.1e-05 | 3.046858e-05 |
| `QCD_FLOOR_FACTOR` | `test_background_tau.py` | 3.0 | 3.2 | 5.8348e-14 / 1.879e-14 = 3.106 |
| `QCD_FLOOR_FACTOR` | `test_background_cs_tau_friction.py` | 3.0 | 8.3 | 1.5501e-13 / 1.887e-14 = 8.213 |

The first is prompt 04's loosening moved again for prompt 04's reason:
`_temperature_crossing_log1pz` root-solves `T_photon(z) - T_break` on whatever spline is in the
tree, so the located crossing moves when the spline does. The other two are new here. Both
multiply `json_vs_reference_max_rel`; the quantity scored against it is the same kind of thing one
level down (model-against-JSON, where the floor is JSON-against-adaptive-reference), so this is a
floor-against-floor comparison at the 1e-13 level — a few hundred ulp of a cumulative quadrature
over twenty decades — and not an accuracy claim. Measured on both trees:
tau 2.194e-14 → 5.8348e-14, cs_tau 2.186e-14 → 1.5501e-13, worst point moving from
$z = 1.005\times10^7$ to $z = 1.007\times10^{11}$. The companion `friction_F` assertion is scored
against `FRICTION_REL_TOL`, not against the stale floor, and passes untouched (6.525e-14 against
1e-13). Each constant's comment names the issue, says what it is scored against and why that floor
is stale, and says prompt 08 should take it back. **Whether they will go back to 1e-9 / 3.0 / 3.0
once the block is regenerated is not established here and must not be assumed.**

Not counted among the three, on prompt 04's precedent: `test_the_branch_crossing_is_inside_the_range`'s
hardcoded `8.64366999e11` → `8.6438180e11` is a **re-scored golden value**, not a loosened
tolerance — the same `T_120_MEV` branch located against the new spline, a further 1.712e-05
relative, at unchanged `places=6`.

### 3. `test_split_converges_where_unsplit_does_not`'s premise was falsified — UNINTENDED DRIFT in the *measurement*, deliberate in the *edit*

The test asserted `unsplit > 1e-6`: that a $T_k$ numeric run which does not split at the declared
`T_120_MEV` discontinuity fails the 3.4e-08 criterion at $k = 4.972\times10^7$/Mpc. Driving the
test class's own `_drift` on both trees:

| tree | split | unsplit | criterion |
|---|---|---|---|
| prompt 04 (`71b842a`) | 2.2136e-09 | 1.0213e-06 — fails, clearing the `1e-6` bound by 2 % | 3.4e-08 |
| prompt 05 (this commit) | 2.9753e-09 | 2.2767e-08 — passes | 3.4e-08 |

This was not sought and is not a defect introduced here: the unsplit run improved 45× while the
split run barely moved, which says most of what the split rescued was the old representation's
$10^{-7}$-level interpolation noise, not the jump in $H(z)$. It is kept, not reverted — there is
nothing to revert. The test now asserts what is still true (`split < ACCEPTANCE_DRIFT`, and
`unsplit > UNSPLIT_PENALTY_FACTOR * split` with the factor at 5.0 against a measured 7.65), prints
both drifts, and carries both trees' measurements in its docstring together with a statement that
**whether `BREAK_POINT_ALL` is still load-bearing is README §7 D5's question, reserved for prompt
08 and not decided here.** The method name is left alone deliberately: renaming it would ripple
into `docs/qcd-background-audit/REFERENCE-FIXTURE.md`'s map, which prompt 02 owns; the docstring
opens by saying the name records a measurement that no longer holds. Opened as
`[04-unsplit-tk-run-now-meets-the-criterion]`.

### 4. `CONFORMAL_TIME_REL` left at 4.0e-08 although the tree now achieves 5.4264e-10 — IMPLEMENTATION CHOICE

Prompt §3 lists five tests and this is not among them; the constant's own comment names prompt 06
as its owner, which takes it to 1e-15. Tightening it here would have pre-empted that prompt's
acceptance measurement for a threshold that is about to move three further orders of magnitude.
The measurement is recorded in the constant's comment and on the board instead. The alternative —
tighten to ~6e-10 now — would have made the guard real one prompt earlier; a reader who disagrees
should note only that prompt 06 is the immediately following prompt.

### None else

`integration_break_points`' logic, `QCD_EOS.py`, `phase_residual.py`, every `ComputeTargets/`
compute target, `Quadrature/`, `Datastore/`, `main.py` and `ComputeTargets/spline_wrappers.py`
were not touched (`git diff --stat` confirms). `RESIDUAL_WKB_REGION_MARGIN` is untouched.

## Verification performed

Everything below was **run**, not reasoned about. No Ray, no datastore anywhere.

**The representation, on the audit's 640-point probe set** (`T_z_reference.probe_set()`), against
the defining equation at `rtol=1e-14`:

| | before (prompt 04) | after | prompt §4 target |
|---|---|---|---|
| max | 7.2615e-04 | **7.236e-04** | unchanged — prompt 06's |
| p90 | 1.936e-07 | **8.912e-08** | ≤ 9.0e-08 |
| median | 1.071e-07 | **2.599e-10** | ≤ 3.0e-10 |

A factor of **412** in the median and **2.2** in the p90. Reproduces audit §4's "entropy factor,
500 pts" row (7.236e-04 / 8.912e-08 / 2.599e-10) exactly.

**Constant-`g_s` exactness** (prompt §3 test 2), `PureRadiationEOS`, same probe set:
`accurate_T` 2.218e-16, `_solve_T_z` 2.218e-16, **`T_photon` 2.928e-16** — about **1.3 ulp**,
against 1.940e-07 on the prompt-04 tree. `F` is identically zero there and the interpolating
spline of a constant is that constant, so the representation returns `T_CMB (1+z)` in closed form.

**Costs** (prompt §4): `T_photon` **2.240 µs/call**, best of 5 over 200 probe points — *below* the
2.26–2.44 µs baseline, so no regression (README §2 (c) confirmed, not merely assumed);
`_build_T_z_spline` **13.3 ms** at 500 nodes, against 13.4 ms before and a ≤100 ms target.

**Break points** (prompt §2 item 4, §4), from the reproduction script §5 on the 1,732-sample
production grid: `BREAK_POINT_ALL` **407** (median spacing 9.2936e-02 in `u`, 4.04× the grid),
of which **404** are knots of the `T(z)` interpolant; `BREAK_POINT_DISCONTINUITY` **2**. **All
three identical to the prompt-04 tree** — tabulating `F` instead of `T` moves no knot, because the
nodes and the order are unchanged. Handed to prompt 07 unchanged; the QCD `BackgroundModel` build
is 0.578 s (from the regeneration run), so there is no build-time effect to quantify either.

**Downstream** (reproduction script §4): `H(z)` on the probe set max **1.280e-03**, p90
**1.923e-07**, median **5.430e-10** (README §6.2's "now" row was 1.278e-03 / 2.895e-05 /
3.424e-07). $\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$: **5.426e-10** relative against the exact
background, from 3.4509e-08 — a factor of **64**, worth 0.74 / 74.5 / 2.24e3 rad at
$k = 10^5 / 10^7 / 3\times10^8$ against floors of 3.05e-07 / 3.05e-05 / 9.15e-04 rad. **T1 is not
closed**; prompt 06 closes it.

**The reproduction script runs unedited** (prompt §3 test 5):
`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py`, 1.0 s,
exit 0. Its §3 table now shows the entropy-factor figures on the "shipped" row, and its
monkey-patch of `cosmology._T_z_spline` with a bare `lambda z: ...` still works — which is the
check that `TemperatureRepresentation` did not make the hook un-replaceable.

**LambdaCDM and `RadiationModel` bit-identity** (prompt §3 test 4): `T_photon`, `Hubble`, `rho`,
`wBackground`, `wPerturbations` for `LambdaCDM(Planck2018)` over 4,001 redshifts spanning
$z\in[0,10^{19}]$, and `RadiationModel`'s `Hubble`, `tau`, `wBackground`, `wPerturbations`,
`T_photon` over 2,000, as exact `float.hex()`: **6,008 lines byte-identical** to `71b842a`,
MD5 `cb93a1a382c822a8077d025572faf5b1` on both. (`RadiationModel` holds no cosmology at all, so
this is a control; `LambdaCDM` computes $T=T_{\rm CMB}(1+z)$ in closed form and has no spline.)

**The tightened thresholds fail on `HEAD~1`** (prompt §3 test 1, by prompt 04 §3's procedure):
`git checkout HEAD -- CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` with the new thresholds
in place gives **4 failures** — `test_T_z_matches_the_defining_equation` (7.2615e-04 > 7.24e-04,
and the printed p90 1.936e-07 > 9.0e-08), `test_a_constant_gs_equation_of_state_is_an_exact_ramp`
(1.940e-07 > 1e-15), and both `test_temperature_spline` floors (3.469e-10 and 2.960e-10 > 1e-15).
The file was restored from a copy and `git diff --stat` confirmed afterwards.

**Suite counts, before → after** (from the repository root):

| suite | before | after |
|---|---|---|
| `CosmologyModels` | 18 | **18**, OK |
| `ComputeTargets` | 354 | **354**, OK |
| `LiouvilleGreen` (fast set — every module except `test_3bessel_analytic`) | 143 | **143**, OK |

The full `LiouvilleGreen` suite (148 with `test_3bessel_analytic`, ~1,400 s) was not run; that
module exercises the three-Bessel analytic path and touches no cosmology. `black --check` clean on
all seven modified files.

## Observations not acted on

- **The `ω²/ω₀²` scatter near $z\sim4\times10^{15}$ was not re-measured here.** Log 04 established
  the null result (span $[-0.0676,+0.2376]$, unchanged by node accuracy to 0.1 %) and suggested
  prompts 05/06 observe whether it narrows. Prompt 05's §3 does not ask for it and the measurement
  needs a `BackgroundModel` build plus the `phase_residual` path; prompt 06, which removes the
  remaining 7.2e-04, is better placed to take it, and log 04's figures are the baseline.
- **`test_qcd_checkpoints`'s worst point moved from $z=1.005\times10^7$ to $z=1.007\times10^{11}$**
  and the `friction_F` figure with it (4.263e-16 → 6.525e-14, absolute 1.421e-14 → 1.585e-12,
  still inside `FRICTION_REL_TOL = 1e-13`). Not investigated: all of it is at the 1e-13 level
  against a stale floor, so there is nothing to conclude until the `convergence` block is
  regenerated. Recorded under `[01-convergence-block-has-a-separate-generator]` so prompt 08 has
  the before/after.
- **`test_split_converges_where_unsplit_does_not` is now misnamed.** Left alone deliberately
  (deviation 3): a rename would ripple into prompt 02's `REFERENCE-FIXTURE.md` map. Whoever
  resolves `[04-unsplit-tk-run-now-meets-the-criterion]` should rename it and the map together.

## State handed to the next prompt

- **`T_Z_REPRESENTATION_VERSION` is now `3`.** Prompt 06 bumps it to `4`, on
  `LambdaCDM_GenericEOS` (`:174`), and adds a row to the table in the comment block above it.
- **The object prompt 06 segments is `TemperatureRepresentation`**
  (`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:52`), constructed as
  `TemperatureRepresentation(spline, T_CMB=..., label="T(z)", min_z=..., max_z=...)` and called as
  `__call__(z, z_is_log=False)`. It holds one `BSpline` of `F(u)` in `self._spline`. To segment,
  give it a list of `(upper_u, spline)` pairs and dispatch inside `__call__` *after* the bounds
  check and the clamp — the clamp must keep setting `raw_z` alongside `log_z` or the ramp stops
  being clamped. **`self._spline` must go on existing, or at least `self._spline.t` must**:
  `docs/gktk-remedial/residual_convergence.py:268` reads `cosmology._T_z_spline._spline.t`, and
  that script is prompt 08's tool.
- **Node count and order are `DEFAULT_T_Z_SPLINE_SAMPLES = 500` / `DEFAULT_T_Z_SPLINE_ORDER = 3`**
  (`:48-49`), and `_build_T_z_spline(min_z, max_z, samples, order)` takes both as parameters, so
  prompt 06 can measure two node counts without touching the body. The measured table for five
  candidates is in the comment above the constants and in **Choices** above. **Prompt 06 should
  not read the audit's "3,000 / k=5" as an unsegmented recommendation**: unsegmented, 3,000 / k=5
  has max 3.730e-04, *worse* than 2,000 / k=5's 5.580e-05.
- **Segment edges.** Use `T_z_reference.jump_locations(cosmology)` and **bisect**, never root-find
  on `T(z) - T_break` (README §2 (b); log 01's handover carries the three values to 17 digits, and
  `test_a_segment_edge_bisected_and_one_root_found_disagree` is the standing demonstration).
  Nothing in this prompt touched that.
- **Achieved figures prompt 06 starts from** (audit probe set, 640 points): max **7.236e-04**,
  p90 **8.912e-08**, median **2.599e-10**. Conformal-time guard **5.4264e-10** (threshold still
  4.0e-08; prompt 06 takes it to 1e-15). `H(z)` max/p90/median **1.280e-03 / 1.923e-07 /
  5.430e-10**. Constant-`g_s` exactness **2.928e-16**.
- **Costs:** `T_photon` **2.240 µs/call**; `_build_T_z_spline` **13.3 ms** at 500 nodes, i.e.
  ~26.6 µs per node including the `rtol=1e-14` solve. A segmented build at 3,000 nodes should land
  near 80 ms, inside the 100 ms stop.
- **Break points, for prompt 07:** `BREAK_POINT_ALL` **407** on the production source grid (404
  knots + 3 crossings), `BREAK_POINT_DISCONTINUITY` **2** — **unchanged by this prompt and
  unchanged by prompt 04**, so prompt 07 inherits exactly the set the audit measured.
- **Regenerate the QCD reference fixture with**
  `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py`
  (no `--dry-run`); 142.2 s on this machine, no Ray, no datastore. Largest move this commit:
  `rho_G`, 7.475097e-03 relative.
- **Three tolerances are loosened and are owed back to prompt 08:**
  `QCD_BREAK_POINT_ALIGNMENT_TOL` 3.1e-05 (`test_background_tau.py`), `QCD_FLOOR_FACTOR` 3.2
  (`test_background_tau.py`) and 8.3 (`test_background_cs_tau_friction.py`). All three are
  `[01-convergence-block-has-a-separate-generator]`. **Prompt 06 will regenerate the QCD block a
  third time and should expect to move them again**, and should say so rather than treating a
  further move as new.
- **`[04-unsplit-tk-run-now-meets-the-criterion]` is the most important thing this prompt hands
  forward, and it is workstream B's.** At $k = 4.972\times10^7$/Mpc the unsplit $T_k$ numeric run
  now **meets** the 3.4e-08 criterion (2.2767e-08, from 1.0213e-06) while the split run barely
  moved (2.9753e-09, from 2.2136e-09). Prompt 08 re-takes `GkTk-remedial` prompt 19's measurement
  across all fifty wavenumbers; this is one of them, and it points the same way. It does **not**
  settle README §7 D5, and prompt 05 did not treat it as settled.
- **Tests whose tolerances, thresholds or literals moved:**
  `CosmologyModels/tests/test_T_z_representation.py` (`T_PHOTON_MAX`, `T_PHOTON_P90`,
  `T_PHOTON_MEDIAN`, `EXACT_RAMP_MAX` — all tightened),
  `CosmologyModels/tests/test_temperature_spline.py` (`INTERPOLATION_FLOOR`, tightened),
  `ComputeTargets/tests/test_background_tau.py` (`QCD_BREAK_POINT_ALIGNMENT_TOL`,
  `QCD_FLOOR_FACTOR` — loosened), `ComputeTargets/tests/test_background_cs_tau_friction.py`
  (`QCD_FLOOR_FACTOR` — loosened), `ComputeTargets/tests/test_numeric_break_points.py` (the branch
  crossing literal, re-scored; `UNSPLIT_PENALTY_FACTOR`, new). No other test file changed.
