# Log 07 — Re-derive `integration_break_points`: the knot lattice is not cosmology (G1)

**Prompt:** prompts/qcd-background-audit/07-rederive-break-points.md
**Commit:** (this commit) — *Declare only the cosmology's own break points, not the spline's knots*
**Model:** Claude Opus 5
**Date:** 2026-09-14
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

`T_Z_REPRESENTATION_VERSION` **4 → 5** (`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:401`),
with its row added to the table in the comment block above the declaration. The bump is the only
signal a datastore gets, and it is needed here because the break-point set changes what
`BackgroundModel` computes even though it changes no value the cosmology returns.

### Production

**`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`**

1. **`integration_break_points` (`:863`) no longer returns the tabulation's knots.** Both kinds are
   now crossings of an equation-of-state temperature and nothing else:
   `BREAK_POINT_ALL` is the crossings of `break_temperatures_GeV`, `BREAK_POINT_DISCONTINUITY` the
   crossings of `discontinuity_temperatures_GeV`. On the production source grid that is **3** and
   **2**, from **2,414** and 2.
2. **The crossings are read, not re-solved.** New
   `LambdaCDM_GenericEOS._build_break_point_crossings_log1pz(self) -> Mapping[float, float]`
   (`:703`), called from `__init__` (`:466`) *before* `_build_T_z_spline`, bisects
   `_bisect_temperature_crossing_log1pz` once per break temperature and caches the result on
   `self._break_point_crossings_log1pz`, keyed by the temperature in GeV. Both consumers read that
   one cache: `_entropy_segment_edges_log1pz` (`:673`), which now filters it instead of bisecting
   again, and `integration_break_points`, which used to root-solve. Prompt §2 item 3's preferred
   resolution, taken.
3. **`_temperature_crossing_log1pz` (`:810`) is off the production path**, kept as a measurement
   probe with a docstring that says so in terms, and its `RuntimeError` text corrected (it named
   `integration_break_points`, which no longer calls it).
4. **`_T_z_spline_knots_log1pz` (`:647`) is kept**, no longer as a declaration but as the record
   that makes the claim checkable — see "Deviations" 2.
5. Docstrings and comment blocks rewritten with this prompt's own measurements: the
   `DEFAULT_T_Z_SPLINE_SAMPLES` comment block, `SegmentedEntropyFactor`'s class docstring,
   `_build_T_z_spline`'s knot-recording comment, `_entropy_segment_edges_log1pz`,
   `_temperature_crossing_log1pz` and `integration_break_points`.

**`CosmologyModels/GenericEOS/GenericEOS.py`** — the three texts prompt §2 item 4 names: the
`BREAK_POINT_*` comment block (`:20-40`), `break_temperatures_GeV` (`:78-90`) and
`discontinuity_temperatures_GeV` (`:92-108`). The last said an adaptive ODE solver "only has to be
split at a jump" (refuted by `GkTk-remedial` prompt 19's measurement) and that the two sets "differ
by three orders of magnitude on the production range (404 spline knots against 3 temperature
crossings)" (refuted here). This closes
`[19-cosmologymodels-docstrings-predate-per-sector-policy]`.

**`ComputeTargets/BackgroundModel.py`** — `_cosmology_break_points`'s docstring (`:203-226`), which
named the knots as part of what the QCD cosmology declares.

**`Quadrature/integrators/numeric_with_phase_cut.py`** — **docstrings and comments only.** The
module docstring's "404 of those (the `T(z)` spline knots) against 3 jumps";
`declared_discontinuities_in_z`'s "the C2 points (the `T(z)` spline knots)";
`_separated_boundaries`'s "~125 boundaries inside one production numeric range" and "the closest a
declared temperature crossing comes to a knot anywhere in (z = 0.1, 1e14) is 3.4e-03"; and the
`break_point_kind` parameter paragraph. **No logic was touched** — not `_segmented_solve`, not
`_separated_boundaries`' body, not `BREAK_POINT_STANDOFF`, and neither
`TkNumericIntegration.BREAK_POINT_KIND` nor `GkNumericIntegration.BREAK_POINT_KIND`.

### Tests

**`ComputeTargets/tests/test_numeric_break_points.py`**

- `TestDeclaration.test_kind_selects_knots_or_jumps` → `test_kind_selects_the_kink_or_only_the_jumps`:
  `assertGreater(len(every), 100)` → `assertEqual(len(every), 3)`. This is the second half of
  `[02-fixture-tests-pinned-to-todays-break-point-artefact]`.
- New `TestDeclaration.test_no_declared_break_point_is_a_knot` — finding G1 as a standing assertion:
  on both ranges, under both kinds, `np.intersect1d(declared, knots_in_range)` is empty while the
  tabulation still carries 2,411 interior knots in range.
- New `TestDeclaration.test_the_declared_points_are_prompt_06s_segment_edges` — the declared points
  equal `_T_z_spline.segment_edges` **and** prompt 06's 17-digit handover, transcribed as the module
  constant `PROMPT_06_SEGMENT_EDGES_LOG1PZ` rather than re-derived (prompt §2 item 3's "assert
  agreement with prompt 06's 17-digit edges in a test").
- New class `TestConsumerKnotVectorConstructs` (3 cases) — prompt §3 item 5, below.
- Module docstring item 1 and item 5, and `_ManyBreakCosmology`'s docstring, corrected.

**`ComputeTargets/tests/test_background_tau.py`** — `test_qcd_break_points` rewritten: expected
count is `len(branch_boundaries)` = 3, plus a new direct assertion that the intersection with the
tabulation's knots is empty. `QCD_BREAK_POINT_ALIGNMENT_TOL` **unchanged at 1.5e-04**, with a new
paragraph recording that it was re-measured here and did not move.

**`ComputeTargets/tests/test_phase_residual.py`** — `COST_BREAK_POINT_FACTOR` **2.40 → 1.01**
(measured 1.002), with the three-step history in its comment. The strict `> baseline` assertion is
unchanged and is what still checks that the crossings *are* declared.

**`ComputeTargets/tests/wkb_reference_data.json` is untouched** — see "Deviations" 1.

---

## Deviations from the prompt

### 1. `STRUCTURALLY REQUIRED` — the QCD reference fixture did not move, and could not have

Prompt §2 item 6 says "Regenerate the QCD reference fixture, in this commit, and quote the largest
relative move per key. **Expect the `tau`/`cs_tau`/`friction_F` references to move**: the panel
structure changes even though the integrand does not."

The generator was run (`PYTHONPATH=. ./venv/bin/python
docs/qcd-background-audit/generate_qcd_references.py`, 193.3 s) and reported:

> No change: the QCD block's science content (12 keys) is bit-identical to the shipped file.
> Nothing written (not even the provenance fields).

That is correct and is a property of the fixture's design, not an omission. Prompt 02's generator
computes every QCD reference by **converged adaptive quadrature** of the double-precision integrand
(`generate_qcd_references.py:118`, `_adaptive_reference`); it never builds a fixed-order cumulative
table and never consults `integration_break_points`. The panel structure is therefore not an input
to the reference. What the break-point set changes is the *model* side of every comparison, which is
rebuilt from the tree on every test run.

The stronger statement, measured, is that **this commit moves no background value at all**:
`T_photon`, `Hubble` and `rho` as exact `float.hex()` over 2,001 points spanning `z ∈ [0, 1e19]`,
for `LambdaCDM`, `RadiationModel`, `LambdaCDM_GenericEOS(PureRadiationEOS)` **and `QCD_Cosmology`**,
are byte-identical to `a1d667a` — 18,017 lines, MD5 `e9799e4299250b2ececb5b00a2a82514`. Prompts 04,
05 and 06 each had to regenerate the fixture because they moved `T(z)`; this one changes only which
points a quadrature splits at.

Prompts 04–06 had `--dry-run` report a delta; here it reports none, so there is nothing to quote per
key. The largest relative move is **0.0 in all twelve keys**.

### 2. `IMPLEMENTATION CHOICE` — `_T_z_spline_knots_log1pz` is kept

Prompt §2 item 2: "Remove `_T_z_spline_knots_log1pz` and the code that populates it, unless prompt
06's representation needs it for something else — in which case say what."

It is kept, and what needs it is **the measurement this prompt exists to make**. The acceptance row
is "of those, knots of an interpolant: 404 → **0**", and "none of the declared break points is a
knot" is only checkable from outside the class if the knots are still knowable from outside the
class. Two things in the tree do exactly that check and would break without it:

- `docs/qcd-background-audit/measure_T_z_representation.py:401`, which prompt §3 item 6 requires to
  run **unedited**, and which is not on this prompt's list of files that may be touched;
- `ComputeTargets/tests/test_background_tau.py::test_qcd_break_points` and the new
  `test_no_declared_break_point_is_a_knot`.

Removing it would also not hide the knots: `docs/gktk-remedial/residual_convergence.py:268` recovers
them from `cosmology._T_z_spline._spline.t`, which is public-by-convention and which prompt 06
deliberately preserved. The alternative considered — delete the attribute and have each consumer
reach into `_T_z_spline._spline.t` itself — moves the same information behind a longer path and
costs an edit to a file this prompt may not touch, for no reduction in coupling.

Its comment block was rewritten so that a later reader cannot mistake it for a declaration: it opens
with "It is **NOT** a break-point declaration and has not been one since prompt 07."

### 3. `IMPLEMENTATION CHOICE` — `_temperature_crossing_log1pz` is kept, off the production path

Prompt §2 item 3 offers two resolutions and prefers returning prompt 06's bisected edges. That was
taken, which leaves `_temperature_crossing_log1pz` with no production caller. Deleting it was
considered and rejected:

- `ComputeTargets/tests/test_numeric_break_points.py::test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink`
  uses it to find a *neighbourhood* of a crossing, which its ~1e-12 offset does not disturb;
- `CosmologyModels/tests/T_z_reference.py:45` cites it by name for the redshift-arithmetic rule, and
  that file is not on this prompt's list;
- it is the documented illustration of the trap README §2 (b) is about, next to the bisector that
  replaced it.

Its docstring now opens "**Nothing in production calls this, and nothing may put it back on the
break-point path**", states why (since prompt 06 `T_photon` genuinely jumps at exactly these
temperatures, so `log T_photon(z) − log T` need not have a root), and names the standing
demonstration. A follow-up to convert it into a test helper is opened as
`[08-temperature-crossing-solver-is-test-only]`.

### 4. `IMPLEMENTATION CHOICE` — the six bands of prompt §3 item 5 were reconstructed, and the
reconstruction is asserted

`prompts/phase-representation` log 02 §1 measures six (sector, k) geometries but its harness was a
scratch script and is not in the tree. The bands were recovered from that log's own sample counts —
1016 / 1040 / 1218 / 1242 / 1377 / 1401 — as the lowest that many nodes of the production source
grid, and the reconstruction is **checked rather than assumed**: a separate case asserts that the
`BREAK_POINT_DISCONTINUITY` count on each band reproduces that log's DISC column (1, 1, 1, 1, 2, 2).
Nothing about the consumer spline, the phase or the reference is reproduced; construction only, as
the prompt requires.

### 5. `IMPLEMENTATION CHOICE` — the test asserts the *old* set is singular as well

Prompt §3 item 5 asks only that the vector now constructs. A third case restores the tabulation's
knots to the declared set in the same process and requires `LinAlgError` on all six, so the test
reads as a measurement of what changed rather than as an assertion that nothing went wrong. It also
gives prompt 10 a working reproduction of the blocker it inherits.

Nothing is tagged `UNINTENDED DRIFT`. No README §2 design fact was touched: `QCD_EOS.py` was not
opened, the shipped `_solve_T_z` was not used as a reference, no segment edge was located by
root-finding on `T(z) − T_break` (the opposite — the one place that still did was taken off the
path), no threshold was loosened, and neither `BREAK_POINT_KIND` was approached.

---

## Verification performed

### §2 item 1 — measure before you remove

*Ran*, in one process on the production `QCD_Cosmology`. Two measurements, as the prompt demands
both.

**(a) The discontinuity in the highest derivative any consumer uses.** The deepest derivative
anything in the tree builds is `d3_lnH_dz3` (`ComputeTargets/BackgroundModel.py:377`; the stack is
`d_lnH_dz` → `d2_lnH_dz2` → `d3_lnH_dz3`, plus `d2_wPerturbations_dz2`). The one-sided derivative
jumps of `F` at its interior knots were read exactly off the per-segment `PPoly`, over all 2,405
simple interior knots inside the production grid:

| derivative | worst absolute jump | where | relative |
|---|---|---|---|
| `d1F/du1` | 1.286748e-13 | u = 27.531814 | 5.452792e-13 |
| `d2F/du2` | 7.067125e-12 | u = 27.531814 | 7.804940e-12 |
| `d3F/du3` | 4.950318e-10 | u = 36.676888 | 2.098371e-07 |
| `d4F/du4` | 6.179229e-08 | u = 36.676888 | 1.252007e-05 |
| **`d5F/du5`** | **8.184279e+01** | u = 27.531814 | **2.629780e-01** |

The first four are the floating-point noise of a one-sided Horner evaluation, not jumps: an order-`k`
interpolating B-spline through **simple** interior knots is `C(k-1)` there by construction, so at
`k = 5` the first genuinely discontinuous derivative is the fifth, and the table shows exactly that
break. **The consumer's deepest derivative is three levels below the first discontinuous one.**

The control is the lattice that was declared. The same measurement on a 500-node `k = 3`
unsegmented tabulation of the same `F` — prompt 05's representation, rebuilt in process — gives 404
interior knots in range and:

| derivative | worst absolute jump | relative | median relative |
|---|---|---|---|
| `d1F/du1` | 3.331536e-15 | 6.782434e-13 | 2.415589e-15 |
| `d2F/du2` | 5.880713e-14 | 3.011671e-12 | 1.753803e-15 |
| **`d3F/du3`** | **1.361730e+01** | **1.536226e+00** | **7.426483e-01** |

A cubic is `C2`, so its **third** derivative jumps — by a median 74 % relative — and that is
precisely the derivative `d3_lnH_dz3` reads. That is why splitting a panel at every one of those
knots was buying something, and why it is not any more.

**(b) The observable.** 60-odd knots sampled uniformly across the production range, evaluated
through the production `Hubble` and `wPerturbations` with the temperature representation swapped
(`T_z_reference.temperature_override`), scored against the exact background
(`accurate_T`, `rtol = 1e-14`):

| across a knot | NEW (3,000, k=5, segmented) | OLD (500, k=3, unsegmented) |
|---|---|---|
| step in `H` | max 4.009e-12, median 3.953e-12 | max 4.008e-12, median 3.956e-12 |
| step in `c_s^2` | max 2.000e-12, median 4.502e-14 | max 1.999e-12, median 4.603e-14 |
| **step in `d lnH/du`** | **max 8.337e-11**, median 2.464e-12 | **max 6.294e-07**, median 9.966e-12 |
| **`|H/H_exact − 1|` near a knot** | **max 6.319e-12**, median 0.0 | **max 2.097e-04**, median 0.0 |

The `H` and `c_s^2` rows are the probe's own resolution, not a measurement of either representation:
the step is read at `u ± 1e-12`, and `H` moves by ~2·2e-12 = 4e-12 across that interval on its own.
Both representations are at least `C2`, so `H` is continuous at a knot in both, and the row says so.
The two rows that discriminate are the last two: **7,550× and 3.3e7×**. The 2.097e-04 is the
"1e-4-level defect the old lattice carried" the prompt asks this to be quoted against.

Reproduction: the script is in this log's commit message trail only (a scratch measurement); every
number above is also written into the docstrings at
`LambdaCDM_GenericEOS.integration_break_points` and the `_T_z_spline_knots_log1pz` comment, which
is where prompt §2 item 4 requires them.

### §3 item 1 — the set is what §2 item 2 says

*Ran.* On `QCD_Cosmology` over the production source grid (0.1 to 2.064e16):

```
all                3 points in range, median spacing 4.9598e+00 in u = 215.34 x the grid spacing
discontinuity      2 points in range, median spacing 9.9196e+00 in u = 430.69 x the grid spacing
```

The three, to 17 digits, are `17.565806941870026`, `23.197460552819653`, `27.485391822044257` —
**bit-identical** to prompt 06's segment edges and to `_T_z_spline.segment_edges`, asserted both
ways in `test_the_declared_points_are_prompt_06s_segment_edges`. None of the three is a knot;
`np.intersect1d(declared, knots_in_range)` is empty under both kinds, on both ranges, while
`knots_in_range` is 2,411 (`test_no_declared_break_point_is_a_knot`).

**The count fails on `HEAD~1`.** `HEAD~1` is `a1d667a`, prompt 06's commit, where the figure is
**2,414**, not the 407 the prompt's text expects — prompt 06 raised the tabulation from 500 nodes
to 3,000 and every interior knot was declared (`[05-break-point-set-grew-with-the-node-count]`).
Measured in a worktree at `a1d667a`: `[tau] QCD break points in (0.1, 2.06e+16): 2414 (2411 knots +
3 temperature crossings)`, against this tree's `3 (= 3 temperature crossings, 0 of the tabulation's
2411 interior knots in range)`.

### §3 item 2 — the cumulative tables do not lose accuracy

*Ran*, in a worktree at `a1d667a` and in this tree, the same two modules, same machine, against the
**same** (unchanged) references:

| QCD, scored against the reference | before (`a1d667a`) | after | floor |
|---|---|---|---|
| `tau` checkpoints, max rel err | 2.104e-15 @ z = 1.005e7 | **2.254e-15** @ z = 1.005e7 | 1.879e-14 |
| `cs_tau` checkpoints, max rel err | 2.212e-15 @ z = 1.005e7 | **2.212e-15** @ z = 1.005e7 | 1.887e-14 |
| `friction_F` checkpoints, max rel err | 3.340e-16 @ z = 1.005e7 (abs 1.421e-14) | **3.340e-16** @ z = 1.005e7 (abs 1.421e-14) | — |
| `tau` short baseline, z = 1.004e6 | 4.038e-15 / 3.386e-14 | 4.200e-15 / 3.386e-14 | — |
| `tau` short baseline, z = 100.2 | 9.155e-15 / 1.971e-14 | 9.354e-15 / 1.984e-14 | — |
| `tau` short baseline, z = 1.001 | 1.746e-15 / 9.275e-15 | 1.746e-15 / 9.275e-15 | — |

**`cs_tau` and `friction_F` are unchanged to every digit printed. `tau` is 7 % larger** — 2.254e-15
against 2.104e-15, a move of 1.5e-16 absolute — and that is a narrow miss of the prompt's "no
worse", recorded rather than argued away. What it is: both figures are **8.3× below the reference
floor** the JSON records for the reference they are scored against (1.879e-14), and the quantity is
a cumulative quadrature over ~1,700 panels, so a 1.5e-16 move in the last figure is the panel
structure being different rather than worse. The test's own threshold (`QCD_FLOOR_FACTOR = 3.0`,
5.636e-14) is met with a factor of 25 to spare and was **not** touched. **The 404 splits were
buying nothing**, which is what this row was written to establish: removing 2,411 of 2,414 declared
break points leaves two of the three tables bit-unchanged and the third at 12 % of the floor.

### §3 item 3 — the build gets cheaper

**The row to read is integrand evaluations**, which is exact, load-independent, and the measure
this campaign has used elsewhere. Wall times are quoted beside it, each with a control taken in the
same process, and this machine was heavily loaded throughout (see the note at the end of this
section), so they carry the weight of an order of magnitude and not of a third digit.

*Ran.* The suite's own counters, which report each cumulative table's integrand evaluations,
measured in a worktree at `a1d667a` and in this tree:

| | before (`a1d667a`) | after | ratio |
|---|---|---|---|
| **`[tables] QCD build`, tau / cs_tau / friction_F** | **16,580** each | **6,936** each | **2.39×** |
| `[tables] LambdaCDM build` (control, declares nothing) | 6,924 each | 6,924 each | 1.00× |
| `[tables] QCD build`, all three tables | 0.781 s | 0.418 s | 1.87× |
| `[tau] QCD`, whole `compute_background` | 0.959 s | 0.599 s | 1.60× |
| `[tau] LambdaCDM`, table (control) | 0.179 s | 0.173 s | 1.03× |
| `[tau] QCD delta` throughput, on/off | 39.48 µs | 29.23 µs | 1.35× |
| `[tau] QCD delta` throughput, off/off | 77.97 µs | 57.66 µs | 1.35× |

**2.39× fewer integrand evaluations**, and QCD's 6,936 now sits **0.17 %** above LambdaCDM's
break-free 6,924 — three extra Gauss panels in 1,731 intervals, which is exactly what "this
cosmology has three break points" costs. The LambdaCDM control moves by 1.03× across the same pair
of runs, which is the scale of the machine noise on the wall-clock rows.

A second, tighter control, taken **in one process against one cosmology object** so that nothing
but the declared set differs — the "before" run wraps `integration_break_points` to return the
tabulation's knots again, as the tree did at `a1d667a` — counting *every* `Hubble` call rather than
one table's integrand:

| | `Hubble` evaluations | wall (two runs) |
|---|---|---|
| QCD, knots declared (2,414 points) | **40,110** | 0.972 / 0.970 s |
| QCD, 3 crossings | **20,822** | 0.608 / 0.602 s |
| LambdaCDM control (declares nothing) | 15,580 | 0.209 / 0.210 s |

**1.93× fewer evaluations**, 1.61× in wall time, at load average 16.8 falling to 16.1 across the
measurement.

The `rho` residual table is the same story at a different call site, and it is an evaluation count
too: **1.002×** the order×intervals baseline at worst over the six production (model, sector, k)
builds, against 2.337× with the knots and 1.24× with the old 404. `COST_BREAK_POINT_FACTOR`
accordingly comes back **2.40 → 1.01**, below its original 1.30 rather than merely to it — exactly
what prompt 06 predicted, and `[05-break-point-set-grew-with-the-node-count]` is closed on it.

### §3 item 4 — a cosmology declaring nothing is bit-identical

*Ran.* `T_photon`, `Hubble` and `rho` as exact `float.hex()` over 2,001 points spanning
`z ∈ [0, 1e19]` for `LambdaCDM`, `RadiationModel` and
`LambdaCDM_GenericEOS(PureRadiationEOS)`: **12,012 lines, MD5 `249882626ba52a553ae2df0b74e99643`,
byte-identical** between a worktree at `a1d667a` and this tree. With `QCD_Cosmology` added to the
same dump: **18,017 lines, MD5 `e9799e4299250b2ececb5b00a2a82514`, also byte-identical** — this
commit moves no background value at all, on any model.

`_ManyBreakCosmology` and the eight `TestPerSectorPolicy` cases keep their meaning and needed no
edit; only the class docstring changed, to say that no production cosmology has that geometry any
more and that the fixture is retained because it is the only thing in the tree that exercises the
`_separated_boundaries` standoff guard.

### §3 item 5 — prompt 02's blocker is gone

*Ran*, `TestConsumerKnotVectorConstructs`, 3 cases, 2.3 s:

| sector | k | samples | declared ALL | DISC (log 02's column) | multiplicity-3 vector |
|---|---|---|---|---|---|
| `Gk` | 1e5 | 1016 | 1 | 1 ✓ | **constructs** |
| `Tk` | 1e5 | 1040 | 1 | 1 ✓ | **constructs** |
| `Gk` | 1e7 | 1218 | 2 | 1 ✓ | **constructs** |
| `Tk` | 1e7 | 1242 | 2 | 1 ✓ | **constructs** |
| `Gk` | 3e8 | 1377 | 3 | 2 ✓ | **constructs** |
| `Tk` | 3e8 | 1401 | 3 | 2 ✓ | **constructs** |

With the tabulation's knots restored to the declared set (1,354 to 1,921 points per band), the same
construction raises `LinAlgError: Colocation matrix is singular` on **all six**, which is prompt 02's
finding reproduced in the tree. Construction only: no consumer spline is built and no phase is
scored — that is prompt 10's.

### §3 item 6 — reproduction

*Ran*, unedited: `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py`.
Its §5 reads `all 3 points in range` / `discontinuity 2 points in range`. Its prose line beneath the
table still says "Of the BREAK_POINT_ALL points, 2411 are knots of the T(z) spline itself", which is
now a **mis-statement of its own variable** — it prints the number of knots in range, never the
intersection with the declared set — and the script is not on this prompt's list of files that may
be touched. Opened as `[09-audit-script-section-5-prose-counts-the-wrong-set]`.

### Suites

| suite | before (`a1d667a`) | after |
|---|---|---|
| `CosmologyModels/tests` | 30 OK | **30 OK** (0.58 s) |
| `ComputeTargets/tests` | 354 OK | **359 OK** (178.8 s; +5 — two new `TestDeclaration` cases and the three of `TestConsumerKnotVectorConstructs`; the renamed case does not change the count) |
| `LiouvilleGreen/tests`, fast set | 143 OK | **143 OK** (14.3 s) |

The LiouvilleGreen fast set is every module except `test_3bessel_analytic`, which is excluded for
wall time (~1,400 s for the full 148); it is untouched by anything in this commit, which reaches no
`LiouvilleGreen` file and moves no background value.

**A note on wall-clock numbers in this log.** This machine carried heavy external load
throughout, peaking at a load average of **143**. Two intermediate `ComputeTargets` runs showed
`test_tk_wkb_phase.TestCost.test_wall_time_per_object` — a **LambdaCDM** wall-time case, on a model
this commit leaves byte-identical and which takes the unchanged no-declaration code path — reading
0.0617 s against its 0.06 s bound. It did not reproduce once the load settled: the orchestrator
re-ran `test_background_tau` three times consecutively (14 OK each) and the full suite (359 OK) at
a load average of ~7. **It is a machine-load artefact, not a finding, no threshold was touched and
no issue is opened for it.** Every wall-clock figure in this log is nevertheless paired with a
control and labelled with the load, and §3 item 3's load-independent row — integrand evaluations —
is the one to read.

**The three files outside `LambdaCDM_GenericEOS.py` and the tests were verified to be
docstring-and-comment-only**, not by inspection but by parsing: `ast.dump` of each file with every
docstring node stripped is byte-identical to the same dump taken in the `a1d667a` worktree, for
`Quadrature/integrators/numeric_with_phase_cut.py`, `ComputeTargets/BackgroundModel.py` and
`CosmologyModels/GenericEOS/GenericEOS.py`. Prompt §1's "docstrings and comments only" is therefore
a measured statement about `numeric_with_phase_cut.py`, not a claim.

`black` — the seven changed files are clean under `--check`. The repository-wide `--check` reports
54 files it would reformat, identically at `a1d667a`: a pre-existing condition of the installed
`black` version, not introduced here.

---

## Observations not acted on

1. **`docs/qcd-background-audit/measure_T_z_representation.py` §5's prose now counts the wrong
   set** — see §3 item 6 above. `[09-audit-script-section-5-prose-counts-the-wrong-set]`.
2. **`_temperature_crossing_log1pz` has no production caller.** Kept for the reasons in deviation 3,
   but a private method on a production class whose only callers are tests is a trap for a later
   reader even with the docstring it now carries.
   `[08-temperature-crossing-solver-is-test-only]`.
3. **`QCD_BREAK_POINT_ALIGNMENT_TOL` could not be tightened, and the reason is not this prompt's.**
   Re-measured here: 1.418851e-04 at `T_120_MEV`, 1.728034e-05 at `T_LO`, 1.060594e-06 at
   `EOS_T_LO` — to every digit prompt 06's figures, because the bisected crossing and the root-found
   one differ by ~1e-14 in `u`. The whole of it is the age of the JSON's `convergence` block
   (`[01-convergence-block-has-a-separate-generator]`) and prompt 08 still owns it.
4. **The `tau` checkpoint figure moved 7 %** (2.104e-15 → 2.254e-15), at 12 % of the floor it is
   scored against. Not an issue in its own right — recorded in §3 item 2 above and in the Result
   line — but it is the one row of prompt §4 that is not strictly "no worse", and prompt 09, which
   scores the consumers, should know it moved.
5. **`docs/gktk-remedial/residual_convergence.py`'s `branch+knots` scheme now has no production
   counterpart.** Its `SCHEME_ORDER = ("plain", "branch", "branch+knots")` measures three
   segmentations of the QCD panels, and the third is what the tree used to do. Prompt 08 re-runs
   that script and owns what the schemes should be; not touched here, and deliberately not
   pre-empted (`docs/gktk-remedial/residual_convergence.py` is prompt 08's and running it is a
   campaign stop condition out of order).

---

## State handed to the next prompt

- **`T_Z_REPRESENTATION_VERSION` is now `5`** (`LambdaCDM_GenericEOS.py:401`). The table in the
  comment block above it carries rows 1–5; prompt 08 appends a row only if it moves a per-sector
  break-point policy.
- **`BREAK_POINT_ALL` is 3 on the production source grid and `BREAK_POINT_DISCONTINUITY` is 2.**
  The three points, to 17 digits, are `17.565806941870026`, `23.197460552819653`,
  `27.485391822044257` — `T_LO`, `EOS_T_LO`, `T_120_MEV` — and they are **bit-identical** to prompt
  06's segment edges, because both now read the same cache. `BREAK_POINT_DISCONTINUITY` drops
  `EOS_T_LO`, the join at which `w` kinks but `g_s` does not step.
- **Where the crossings live:** `LambdaCDM_GenericEOS._break_point_crossings_log1pz`, a
  `{T_break_GeV: u}` mapping built once in `__init__` by
  `_build_break_point_crossings_log1pz()` from `_bisect_temperature_crossing_log1pz`. It is the
  single definition; `integration_break_points` and `_entropy_segment_edges_log1pz` both filter it.
  **Do not reintroduce a root solve on `T_photon(z) − T_break`** —
  `_temperature_crossing_log1pz` survives only as a test probe and its docstring says so.
- **`_T_z_spline_knots_log1pz` still exists** (2,411 interior knots in the production range) and is
  *not* a declaration. It is the record that makes "0 of the declared points is a knot" checkable;
  `residual_convergence.py:268`'s `cosmology._T_z_spline._spline.t` also still works.
- **Costs, for prompt 08's baseline. Quote the evaluation counts, not the wall times:** this
  machine peaked at a load average of **143** while these were taken. QCD `BackgroundModel`:
  tau/cs_tau/friction_F **6,936** integrand evaluations each (was 16,580, a factor of 2.39),
  against LambdaCDM's unchanged break-free **6,924** — 0.17 % apart. The `rho` residual table is
  **1.002×** the order×intervals baseline (was 2.337×). Wall times, each with a same-process
  control and taken at load average ~16: all three tables 0.781 → **0.418 s**, whole
  `compute_background` 0.959 → **0.599 s** with LambdaCDM's control moving 1.03× over the same
  pair of runs; `delta` throughput on/off 39.48 → **29.23 µs**, off/off 77.97 → **57.66 µs**.
- **Accuracy, for prompt 08's baseline.** QCD against the (unchanged) references: `tau` 2.254e-15,
  `cs_tau` 2.212e-15, `friction_F` 3.340e-16 relative, against floors 1.879e-14 and 1.887e-14.
- **The reference fixture was regenerated and did not move**: 12 of 12 science keys bit-identical,
  nothing written, 193.3 s. The command is unchanged:
  `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py`. **Prompt 08
  should not expect to have to regenerate it either** unless it moves a background value; the
  references are converged adaptive quadratures and do not see the panel structure.
- **One tolerance is still owed back to prompt 08**, unchanged in value and now re-measured twice:
  `QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04` in `test_background_tau.py`, measured 1.418851e-04 at
  `T_120_MEV`. Both `QCD_FLOOR_FACTOR`s are at 3.0 and `COST_BREAK_POINT_FACTOR` is now **1.01**.
- **For prompt 08 specifically.** `[04-unsplit-tk-run-now-meets-the-criterion]` is unchanged by this
  commit — that measurement is about `BREAK_POINT_DISCONTINUITY` versus `BREAK_POINT_ALL` at
  `T_120_MEV`, and both sets still contain it. What *has* changed is that `BREAK_POINT_ALL` and
  `BREAK_POINT_DISCONTINUITY` now differ by **one kink**, not by 2,411 knots, so re-taking
  `GkTk-remedial` prompt 19's measurement is now a question about `EOS_T_LO` alone. Neither
  `TkNumericIntegration.BREAK_POINT_KIND` nor `GkNumericIntegration.BREAK_POINT_KIND` was touched.
- **For prompt 10.** The blocker is gone and there is a working reproduction of it in the tree:
  `ComputeTargets/tests/test_numeric_break_points.py::TestConsumerKnotVectorConstructs`, with
  `PROMPT_02_BANDS` (the six geometries, recovered from log 02's sample counts and validated against
  its DISC column) and `_repeated_knot_vector(sites, breaks, order)`, which pays for each
  multiplicity-`order` knot locally — the only placement log 02 found that can satisfy
  Schoenberg–Whitney. `[13-consumer-spline-crosses-eos-break-points]` is **narrowed, not closed**.
