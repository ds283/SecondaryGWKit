# Log 13 — Segment every background spline at the cosmology's break points

**Prompt:** prompts/qcd-background-audit/13-segment-background-derivative-splines.md
**Commit:** *(this commit)* — Segment the background derivative splines at the declared break points
**Model:** Claude Opus 5
**Date:** 2026-09-15
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

**`T_Z_REPRESENTATION_VERSION` before: 5. After: 6.**

### `ComputeTargets/BackgroundModel.py`

| what | before | after |
|---|---|---|
| `STORED_SAMPLE_SPLINE_ORDER` | — | new, `:65-70`, `= 3` (the order `make_interp_spline` already defaulted to at the stored-sample site; named so that the degeneracy refusal can quote the node count it needs) |
| `SegmentedSpline` | — | new class, `:131-181`. `__slots__ = ("_edges", "_splines")`; `__init__(edges: Sequence[float], splines: Sequence)`, properties `segment_edges -> tuple` and `splines -> tuple`, `__call__(u)` dispatching `self._splines[bisect_right(self._edges, float(u))](u)` |
| `_segment_slices` | — | new, `:184-200`. `_segment_slices(x: np.ndarray, edges: Sequence[float]) -> List[slice]`; `np.searchsorted(..., "left")`, so a datum exactly on an edge opens the segment **above** it, matching `bisect_right` |
| `_refuse_degenerate_segment` | — | new, `:203-222`. `(site, index, count, lo, hi, nodes, order)`; raises `RuntimeError` naming the site, the branch, its range in `u`, its node count and the `order + 1` it needed |
| `build_stored_sample_spline` | — | new **public** module-level function, `:225-278`. `(attr, x_data, y_data, min_z, max_z, break_points=()) -> ZSplineWrapper`. The body is what `_create_functions._build_func` used to hold inline |
| `compute_background` derivative fit | one `make_interp_spline(fit_x, y_data, k=fit_k)` for the whole grid (`:367` on `HEAD~1`) | `fit_break_points = _cosmology_break_points(cosmology, float(fit_z[0]), float(fit_z[-1]))`, `fit_segments = _segment_slices(fit_x, fit_break_points)` computed once (`:505-541`), every branch checked for `fit_k + 1` nodes, and `_build_derivative` looping one spline per branch (`:558-566`) |
| `BackgroundModel._create_functions._build_func` | `make_interp_spline(x_data, y_data)` + `ZSplineWrapper` inline | `stored_break_points = _cosmology_break_points(self._cosmology, self.z_sample.min.z, self.z_sample.max.z)` computed once (`:784-796`); `_build_func` now defers to `build_stored_sample_spline` (`:798-812`) |
| imports | — | `bisect.bisect_right`, `typing.Sequence` |

`_build_derivative_fit_grid` is **unchanged** — no node was added, moved or removed (§2 item 4; see
"The padding decision" below).

### `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`

`T_Z_REPRESENTATION_VERSION: int = 5` → `6` (`:414`), with a row added to the table in the comment
block above it and a paragraph saying why a change to `ComputeTargets/BackgroundModel.py` belongs on
a constant that lives on the cosmology: the constant keys *the background a cosmology yields*, and
"the lattice the background's derivative fields are splined on, and whether it is split where this
cosmology says it is not smooth" is on that list for the same reason the break-point set is.

### `ComputeTargets/tests/wkb_reference.py`

`_model_functions_from_background._build` carried its own copy of the stored-sample spline, and its
docstring undertakes to reproduce `_create_functions`. It now calls `build_stored_sample_spline`
with `_cosmology_break_points(cosmology, min_z, max_z)`. See deviation 1 — this is the prompt's own
measurement path and without it the acceptance test scores the production change as a partial
failure.

### `ComputeTargets/tests/test_background_segmentation.py` (new, 12 tests, 0.21 s)

`TestDeclaredCrossingsAreNotSplinedAcross` (6 tests) re-takes `docs/qcd-background-verification.md`
§10.4's measurement on the production grid with each of the two sites independently on or off;
`TestSegmentSlices` (5) pins the partition against the dispatch, the refusal, and the literal
identity of the no-edges path; `TestCosmologyThatDeclaresNothing` (1) pins that LambdaCDM declares
none. Public helpers: `production_source_grid(cosmology)`,
`epsilon_accessor(cosmology, z_sample, segment_fit: bool, segment_stored: bool)`,
`epsilon_from_cosmology(cosmology, u)`.

### `prompts/qcd-background-audit/README.md`

§7 D5 gains a `> Settled, 2026-09-15` block recording the user's decision to keep both
`BREAK_POINT_KIND` values as they are, which prompt §5 directs this prompt to record there. **No
value was changed anywhere**; the block is prose. That file is not on §"Files you may touch", but
§5 names §7 D5's entry explicitly, so the edit is directed rather than discretionary.

### `ComputeTargets/tests/wkb_reference_data.json`

Regenerated in this commit by `docs/qcd-background-audit/generate_qcd_references.py` (197.6 s, no
Ray, no datastore). QCD block only; `LambdaCDMModel`, `RadiationModel` and the top-level
`convergence` block untouched.

---

## Deviations from the prompt

### 1. `ComputeTargets/tests/wkb_reference.py` had to be changed — STRUCTURALLY REQUIRED

**What the prompt assumed.** Its "Files you may touch" list names `BackgroundModel.py` and three
test modules, and §2 names two spline sites: `_build_derivative` and
`_create_functions._build_func`.

**What was actually there.** A **third** copy of the second site.
`ComputeTargets/tests/wkb_reference.py:273-330`'s `_model_functions_from_background` builds a
`ModelFunctions` from a `compute_background` payload and its own docstring says it is
*"reproducing `BackgroundModel._create_functions`"* — by re-implementing the three lines rather
than calling them. Every `QCDModel` in the tree is built through it, including the one
`docs/qcd-background-audit/grid_density_criterion.py` uses, which is the tool prompt §3 item 2
tells this prompt to score its acceptance table with. So with the production site segmented and the
harness copy not, the harness fits one cubic straight across the declared crossings and reports the
repaired background as a partial failure: measured, `T_120_MEV` reads **1.91e-04** relative in
`epsilon` with the fit segmented and the harness copy unsegmented, against **4.07e-09** when both
are segmented — four orders, entirely inside the harness.

**What was done instead.** The three lines were lifted out of `_create_functions` into a public
module-level `build_stored_sample_spline` in `BackgroundModel.py` — a file the prompt does allow —
and both callers now call it. There is one implementation and the duplication that made this
possible is gone. Nothing else in `wkb_reference.py` changed: the diff is the import block, six
lines of docstring, one new local and the call. Its `make_interp_spline` and `ZSplineWrapper`
imports are now referenced only from a docstring and were **left in place** — nothing else in the
tree re-imports them from this module, so removing them is safe but is a change this prompt has no
reason to make.

**Why this is not scope creep.** It is not a different fix; it is the same fix, in the second place
the same three lines live. Leaving it would have meant either landing a change whose own acceptance
measurement says it did not work, or measuring the acceptance somewhere other than where the prompt
says to measure it. The alternative — stop and ask — was weighed and rejected because the edit is
mechanical, is confined to the function whose documented contract is "reproduce the production
site", and *reduces* the surface it touches rather than enlarging it.

**Nothing else in the tree duplicates either site.** Every other `ZSplineWrapper` construction was
checked: `TkSourceFunctions.py:340,349`, `QuadSource.py:426`, `GkSourcePolicyData.py:677,736`,
`test_quadsource_integral.py:181` and `test_background_tau.py:550,558` spline different quantities
(transfer functions, sources, Green's functions, a deliberately-old `tau` accessor), not background
derivative fields over the source grid, and all of them are compute targets this prompt may not
touch.

### 2. The break points for the derivative fit are taken over the *padded* range, not `z_sample`'s — IMPLEMENTATION CHOICE

§2 item 1 says "at `_cosmology_break_points(cosmology, z_lo, z_hi)`" without fixing `z_lo`/`z_hi`.
`compute_background` already holds `break_points` over the sample range (for the cumulative tables),
and reusing it would have been one word shorter. The fit grid is padded 12 points beyond each end,
so a crossing can in principle lie inside the padding and outside the sample range; splining across
it there would bias the end samples, which is the one thing the padding exists to prevent. The fit
therefore computes its own list over `(fit_z[0], fit_z[-1])`. On the production grid the two lists
are the same three points, so this costs nothing and is measured to change nothing; the alternative
(reuse) is a one-line revert if a later prompt prefers it. The stored-sample site keeps the sample
range, because its data *is* the sample grid.

### 3. "Show it fails on `HEAD~1`" was done by suppressing the declaration, not by checking the file out — IMPLEMENTATION CHOICE

§3 item 3 says to check out the previous `BackgroundModel.py` and confirm the ringing test fails.
The new test module imports `_segment_slices`, `SegmentedSpline` and `build_stored_sample_spline`,
none of which exists on `HEAD~1`, so against that file it fails with an `ImportError` and measures
nothing. `HEAD~1` consulted `_cosmology_break_points` at **neither** spline site, so a cosmology
that declares nothing reproduces it exactly; the run therefore patches
`_cosmology_break_points` to return an empty array in both modules and runs the same tests. The
numbers it prints (2.036e-02 at `T_LO`, 1.034e-03 at `T_120_MEV`, 1.609e-09 at the control) are
§10.4's shipped-grid figures to three digits, which is the evidence that the substitution is
faithful. The same comparison is a **standing** test in the tree rather than a one-off:
`test_an_unsegmented_fit_still_rings` asserts the unsegmented construction is worse than 1e-03, so
it cannot silently start passing.

### 4. `test_away_from_a_crossing_nothing_moves` became `..._nothing_of_consequence_moves` — UNINTENDED DRIFT, kept

The test as first written asserted bit-identity away from a crossing, on the assumption that
segmentation is local. It is not: an interpolating spline is global over the interval it is fitted
on, so every value on both branches moves. Measured, the move is **≤ 1.2e-14 relative** (a few tens
of ulp) at `u` = 5, 10, 14, 20, 25, 31, 34, five orders below the 3.9e-09 the same comparison
against the cosmology itself reads there. The test now asserts `< 1e-12` and its docstring says
plainly that there is no locality to appeal to. Noticed only when the test failed; the assertion
was wrong, not the code.

---

## Verification performed

All figures measured on this machine on 2026-09-15, on the tree at this commit unless marked
`HEAD~1`. "Production grid" is prompt 11's cosmology-aware 1,773-sample QCD grid; "base grid" is the
uniform 1,732-sample one.

### 1. The ringing, all three crossings — prompt §3 item 1, §4 rows 1–3

`epsilon` from the model against a central difference (step 1e-6 in `u`) of the cosmology's own
pointwise `Hubble`, worst over `du ∈ {±0.20, ±0.05, ±0.02, ±0.005}` — §10.4's probe, unchanged.
Reproduction: `PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_background_segmentation`,
or §10.4's own tool,
`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py --models QCDModel --sectors Tk Gk --k 1e5 --sections A B` (6.5 s).

| crossing | `u` | base grid, `HEAD~1` | base grid, here | **production grid, `HEAD~1`** | **production grid, here** | target |
|---|---|---|---|---|---|---|
| `T_LO` | 17.565806941870026 | 3.663e-02 | 4.608e-08 | **2.036e-02** | **1.931e-09** | ≤ 1e-08 ✓ |
| `EOS_T_LO` *(control)* | 23.197460552819653 | 1.889e-09 | 8.190e-08 | **1.609e-09** | **1.609e-09** | unchanged, no worse ✓ |
| `T_120_MEV` | 27.485391822044257 | 6.476e-03 | 4.334e-06 | **1.034e-03** | **4.070e-09** | ≤ 1e-08 ✓ |

The two genuine steps are inside the 3.907e-09 max / 8.091e-10 median regime that holds away from
any crossing. The control is **bit-for-bit the same number** on the production grid, 1.609e-09
before and after. `HEAD~1`'s two figures reproduce §10.4's 2.04e-02 and 1.03e-03 to three digits,
which is what says this is the same measurement.

**On the base grid the control gets 43× worse — recorded, not argued away.** 1.889e-09 → 8.190e-08
at `EOS_T_LO`. That grid has no sample near the crossing (prompt 11's straddling pair and ±5×2
refinement are what put one there), so cutting the branch leaves the two new interior ends up to a
full grid interval from the nearest data, and a cubic's not-a-knot end bias shows. It is not the
production configuration and it is not what §4 scores, but it is the honest statement of what
segmentation costs on a grid that does not resolve the crossing it is asked to segment at. Opened
below as `[13-segmenting-costs-accuracy-on-a-grid-that-does-not-resolve-the-crossing]`, because
prompt 15 is about to make the grid coarser.

### 2. What each of the two sites bought, separately — prompt §5

Same probe, production grid. "fit" is `compute_background._build_derivative`; "stored" is
`_create_functions` / `build_stored_sample_spline`.

| crossing | neither (= `HEAD~1`) | fit only | **both (shipped)** |
|---|---|---|---|
| `T_LO` | 2.036e-02 | 5.02e-09 | **1.931e-09** |
| `EOS_T_LO` | 1.609e-09 | 1.61e-09 | **1.609e-09** |
| `T_120_MEV` | 1.034e-03 | 1.91e-04 | **4.070e-09** |

**Neither site is redundant, and the answer is not symmetric.** Segmenting the fit alone carries
`T_LO` the whole way (2.0e-02 → 5.0e-09, because `epsilon` is flat at 2.0 there and the cubic
through accurate samples has almost nothing to do) but leaves **five sixths of nothing** at
`T_120_MEV`: 1.03e-03 → 1.91e-04, a factor of 5.4, where the crossing sits in the QCD crossover and
`epsilon` runs from 1.9460 to 1.9101 over 0.1 in `u`. There the residue is the *stored* cubic
crossing the step, and only segmenting it too closes it (1.91e-04 → 4.07e-09, a further 47,000×).
Prompt §2 item 2's instruction to segment both, and to say so with a measurement if one does not
need it, is answered: both need it.

### 3. Which `kind` — and why `BREAK_POINT_ALL` rather than the two jumps

Not asked for, but the control row forced the question: `EOS_T_LO` is where `g_s` is continuous to
1.8e-11 and only `w` kinks, `H` does not step there, and segmenting there is what costs the base
grid 43×. Scored on the production grid, with the fit and the stored spline both segmented at the
**2** points `BREAK_POINT_DISCONTINUITY` declares and at the **3** `BREAK_POINT_ALL` declares:

| crossing | quantity | none | `DISC` (2) | **`ALL` (3), shipped** |
|---|---|---|---|---|
| `T_LO` | `epsilon` | 2.036e-02 | 1.931e-09 | **1.931e-09** |
| | `d_wPerturbations_dz` | 8.179e-02 | 1.093e-06 | **1.091e-06** |
| `EOS_T_LO` | `epsilon` | 1.609e-09 | 1.609e-09 | **1.609e-09** |
| | `d_wPerturbations_dz` | **3.459** | **3.459** | **7.752e-04** |
| `T_120_MEV` | `epsilon` | 1.034e-03 | 4.070e-09 | **4.070e-09** |
| | `d_wPerturbations_dz` | 3.670e-02 | 3.515e-07 | **3.515e-07** |

**`BREAK_POINT_ALL` is load-bearing and the row that says so is the control's own.**
`d_wPerturbations_dz` at `EOS_T_LO` is **346 % wrong** — 3.459 relative — under both `none` and
`DISC`, and **7.752e-04** under `ALL`: a factor of **4,462**. `c_s^2` is `wPerturbations` in the
transfer-function sector (`CLAUDE.md`), `EOS_T_LO` is exactly where the equation of state clamps
`w` (README §7 D4), and its derivative enters the transfer sector's `omega_eff`. So the third
crossing earns its place for the quantity it was declared for, at no cost at all in `epsilon` on the
production grid. The prompt's prescription — plain `_cosmology_break_points`, i.e. the default
`BREAK_POINT_ALL` — is what shipped, and no `BREAK_POINT_KIND` constant was touched anywhere.

`d_wPerturbations_dz` was also **8.2 %** and **3.7 %** wrong at the two genuine steps on the
production grid and is now 1.1e-06 and 3.5e-07. That is a second defect this commit closes and it
was not in the prompt's tables.

### 4. The padding decision — prompt §2 item 4

**No padding is added at a segment edge, at either site, and `_build_derivative_fit_grid` is
untouched.** The argument is different at the two sites and both halves are needed:

* **The fit grid.** The outer pad exists because a not-a-knot end has no data beyond it. A segment
  edge is not an outer end: what lies beyond it is the *other branch*, whose values are precisely
  what must not enter this fit, and the cosmology exposes no analytic continuation of one branch
  past the crossing — `LambdaCDM_GenericEOS`'s own `SegmentedEntropyFactor` dispatches on `u` and
  would hand back the other branch's value, which is the failure prompt 06's
  `SEGMENT_EDGE_PAD_LOG1PZ` exists to prevent. The only padding that *is* constructible is extra
  nodes inside the branch approaching the edge; it was not built, because the measurement says the
  fit-grid branch ends are not where the residue is. Segmenting the fit alone leaves ≤ 5.0e-09 at
  `T_LO` and ≤ 1.61e-09 at `EOS_T_LO` on the production grid, both already inside the
  away-from-a-crossing regime, and the 1.91e-04 left at `T_120_MEV` is entirely the *other* site
  (§2 above).
* **The stored-sample spline.** It **cannot** be padded, by construction: its node set *is* the
  stored sample grid, and the quantity it splines is one the cosmology does not supply as a method
  — that is why it is splined at all — so there is no value to evaluate at an extra node. The only
  lever there is the sample grid itself, which is prompt 11's and prompt 15's, not this prompt's.

The residue this leaves is therefore a statement about grid density and not about padding, and the
base-grid control row above is its size.

### 5. The four-way table — prompt §3 item 2, §4 row 4

`docs/qcd-background-audit/grid_density_criterion.py`, **reused unedited** (§3 item 2's preference;
nothing had to be extended). `--models QCDModel --sectors Tk Gk --k 1e5 1e7 --sections B`, 13.8 s on
`HEAD~1` and 15.2 s here. Max |spline(φ) − φ| in rad against the phase-residual oracle.

| sector, k | background / samples | `HEAD~1` | here | factor |
|---|---|---|---|---|
| **Tk, 1e5** | base / base | 1.4055e-05 | 5.9053e-08 | 238× |
| | base / shipped | 3.9744e-07 | 5.8444e-08 | 6.8× |
| | shipped / base | 1.1851e-04 | 5.9048e-08 | 2,007× |
| | **shipped / shipped — production** | **2.4859e-05** | **5.8437e-08** | **425×** |
| **Gk, 1e5** | **shipped / shipped** | **1.4356e-05** | **1.6782e-08** | **855×** |
| **Tk, 1e7** | **shipped / shipped** | **4.4783e-04** | **5.6892e-05** | **7.9×** |
| **Gk, 1e7** | **shipped / shipped** | **2.1574e-04** | **2.4203e-05** | **8.9×** |

**The acceptance row is met with 6.8× to spare**: the production configuration at QCD `Tk`,
`k = 10^5` reads **5.8437e-08 rad** against the **≤ 3.9744e-07 rad** §4 asks for, and against the
2.4859e-05 it read before. The production row improved at **every** (sector, `k`) scored.

Two things the table now says that it did not before:

* **The background's grid has stopped mattering.** At `Tk`, `k = 10^5` the four cells were
  1.41e-05 / 3.97e-07 / 1.19e-04 / 2.49e-05 — a spread of 300× set by which grid the background was
  built on. They are now 5.905e-08 / 5.844e-08 / 5.905e-08 / 5.844e-08: the two "background on"
  rows agree to four digits and only the *sample* set moves anything. That is the defect being
  gone, stated as an identity rather than as a ratio.
* **The max has left the crossing.** At `Tk`, `k = 10^5` the production row's maximum was at the
  crossing (near-crossing 2.4859e-05 = the max); it is now 5.8437e-08 **away** from one, at
  `z = 1.44e10`, with 1.0024e-08 near the crossing. What is left is
  `[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]`'s 7.84-ulp row,
  which is prompt 15's and not this one's.

**One non-production cell got worse and it is recorded.** At `Tk`, `k = 10^7`, base background /
shipped samples goes 4.9524e-06 → 5.6892e-05, 11.5×; the same at `Gk` (2.2338e-06 → 2.4204e-05).
That combination — a background built on one grid and sampled on another — is a harness
configuration and is not what `main.py` does. The mechanism is the one prompt 09 measured from the
other side (`docs/qcd-background-verification.md` §3.3): a background that no longer smears the
step delivers it undiluted, so a consumer's cubic across it has more to bridge. At `k = 10^7` the
neighbourhood refinement prompt 11 sized at `k = 10^5` is not wide enough to bridge it, which is
`[13-crossing-neighbourhood-refinement-was-sized-at-k-1e5]` below. Away from a crossing every "away"
column is unchanged or better (`Tk` 1e5 5.8437e-08 both; `Gk` 1e7 9.1102e-10 both).

### 6. A cosmology that declares nothing is bit-identical — prompt §3 item 4, §4 row 5

`compute_background`'s **whole payload** (every sample array, all eighteen keys) on an 801-point
grid over `z ∈ [0.1, 10^16]`, plus `epsilon`, `d_epsilon_dz`, `d2_epsilon_dz2`, `d_lnH_dz`,
`d2_lnH_dz2`, `d3_lnH_dz3`, `d_wPerturbations_dz`, `d2_wPerturbations_dz2`, `T_photon`, `Hubble`,
`rho`, `wBackground`, `wPerturbations` at 1,301 off-grid probes to `z = 9×10^15` — every value as
exact `float.hex()`, 130,043 lines, dumped on `HEAD~1` in a worktree and here and compared with
`cmp`.

| model | lines | `HEAD~1` md5 | here md5 | |
|---|---|---|---|---|
| `LambdaCDM(Planck2018)` | 28,933 | `a6efda7436e591dc6e8ad09601345759` | same | **byte-identical** |
| `LambdaCDM` **with its five analytic derivatives hidden** | 28,933 | `a769ff0714bb03c6a57edf1013196ab3` | same | **byte-identical** |
| `LambdaCDM_GenericEOS(PureRadiationEOS)` | 28,933 | `2cab0ee89304913aa5ea2cd29331c199` | same | **byte-identical** |
| `RadiationModel` | 14,311 | `74f29814aea076370de395e4ef734936` | same | **byte-identical** |
| `QCD_Cosmology` | 28,933 | `7b1d01d67410e387bb277845bb2ec7d7` | `19f62ed620f3a41ecebbbebc325fc559` | **differs, 21,466 lines** — the point of the commit |

The second row is the one that matters: the `_HideAnalyticDerivatives` stand-in of
`test_background_derivatives.py` drives the **spline branch** with no declared break points, so it
is the direct test that `_segment_slices` with an empty edge list reproduces the unsegmented spline
to the last bit rather than merely to a tolerance. `test_an_unsegmented_build_is_the_spline_it_always_was`
asserts the same identity for the stored-sample site against a hand-written `ZSplineWrapper`.

### 7. Cost — prompt §3 item 5, §4 row 7

Best of 3, same machine, same process shape.

| | samples | `tau` / `cs_tau` / `friction_F` integrand evaluations | `compute_background` wall |
|---|---|---|---|
| QCD, `HEAD~1` | 1,773 | 7,100 / 7,100 / 7,100 | 0.5369 s |
| QCD, here | 1,773 | **7,100 / 7,100 / 7,100** | 0.5301 s |
| LambdaCDM, `HEAD~1` | 1,732 | 6,924 / 6,924 / 6,924 | 0.1742 s |
| LambdaCDM, here | 1,732 | **6,924 / 6,924 / 6,924** | 0.1734 s |

**The evaluation counts are exactly unchanged**, as they must be: segmenting adds splines, not
quadrature. §4's "6,936 per table" is prompt 07's figure on the 1,732-sample *base* grid; on prompt
11's 1,773-sample production grid the figure is 7,100, before and after this commit alike. Wall time
moved by −1.3 % and −0.5 %, which is noise.

### 8. The reference fixture — prompt §2 item 6

`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py` — dry run
197 s, real run 197.6 s, `compute_background` + splines 0.624 s of it. QCD block only.

| key | values | moved | largest relative move |
|---|---|---|---|
| `rho_G` | 90 | 27 | **7.2781e-02** at `k = 3×10^8`, −1.3723414976e-03 → −1.2724607173e-03 |
| `rho_T` | 90 | 27 | **2.0856e-03** at `k = 3×10^8`, −9.2314456375e-02 → −9.2121925294e-02 |
| `reference_floor` | 63 | 24 | an absolute 0.0 → **1.5651484227e-16** (`rho_T[1e5]`, `quad_epsrel_1e-12`); relative is undefined because the floor was exactly zero |
| `tau_minus_top`, `cs_tau_minus_top`, `friction_F_minus_top` | 13 each | **0** | unchanged |
| `short_baseline`, `primitives_at_rho_anchor`, `checkpoints`, `grid`, `z_top`, `rho_anchor_z` | 78 | **0** | unchanged |

That the three cumulative-table keys did not move by a bit is the correctness check on the change's
*scope*: this commit touches the derivative fields and nothing else, so `tau`, `cs_tau` and
`friction_F` — which are Gauss quadratures of `1/H`, `c_s/H` and `(1+c_s^2)/(1+z)` and never see a
derivative spline — must be identical, and are. `rho` is the phase residual, which is built from
`epsilon` and its derivatives, so it is exactly what should move.

**No tolerance was loosened.** Prompt 04 §2 item 5's rule allows one loosening as a finding and
makes two a stop; the count is **zero**. Every one of the 45 QCD-dependent test methods
`docs/qcd-background-audit/REFERENCE-FIXTURE.md` maps re-scored green against the regenerated
fixture with no threshold touched, `test_phase_residual.TestAgainstReferences.test_rho_at_the_reference_checkpoints`
included (which failed at 1.9253e-04 rad against a 1e-07 threshold before regeneration, and passes
after).

### 9. "Show it fails on `HEAD~1`" — prompt §3 item 3

`HEAD~1`'s numerics, reproduced by suppressing the cosmology's declaration (deviation 3), run
against the new module:

```
FAIL: test_both_sites_segmented_removes_the_ringing (crossing=0)
  epsilon rings by 2.036e-02 at the crossing u = 17.565806941870026
FAIL: test_both_sites_segmented_removes_the_ringing (crossing=2)
  epsilon rings by 1.034e-03 at the crossing u = 27.485391822044257
FAIL: test_the_stored_sample_spline_is_needed_as_well_as_the_fit
  0.001034193961635732 not less than 1e-08
Ran 3 tests ... FAILED (failures=3)
```

`test_the_control_crossing_is_not_made_worse` passes on both trees, which is the control behaving
as a control.

### 10. Suites

Run from the repository root. The **full** LiouvilleGreen set, not the fast set prompts 02–08 used.

| suite | before (`HEAD~1`) | after |
|---|---|---|
| `CosmologyModels/tests` | 30, OK | **30, OK** |
| `ComputeTargets/tests` | 380, OK | **392, OK** |
| `LiouvilleGreen/tests` | 148, OK | **148, OK** |

None falls. `black` reports the four changed files and the new one unchanged under `--check`.

---

## Observations not acted on

1. **On a grid that does not resolve a declared crossing, segmenting costs accuracy at that
   crossing.** The base-grid control row: `epsilon` at `EOS_T_LO` goes 1.889e-09 → 8.190e-08, 43×
   worse, because the branch cut creates two interior ends up to a whole grid interval from the
   nearest sample and a cubic's not-a-knot end bias shows there. It does not bind on the production
   grid (1.609e-09, unchanged) because prompt 11 put a straddling pair and a ±5×2 refinement at
   every crossing. **Prompt 15 is about to make the grid coarser**, and
   `[12-source-grid-density-…]`'s cap ladder does not mention this constraint. Opened as
   `[13-segmenting-costs-accuracy-on-a-grid-that-does-not-resolve-the-crossing]`.

2. **The crossing neighbourhood was sized at `k = 10^5` and does not hold at `k = 10^7`.** After
   this commit the production row at QCD `Tk`, `k = 10^7` reads 5.6892e-05 rad near a crossing —
   **59.7 ulp** of that band's stored phase, against 0.008 ulp away from one. It improved 7.9× here
   and is in the same regime as the 65.78 ulp prompt 11 shipped at `k = 10^5` and called a success,
   so it is not a regression; but prompt 10's conclusion that "the crossing is a `k = 10^5`
   phenomenon" was taken against φ recovered from a *stored* θ, whose floor at `k = 10^7` is
   3.05e-05 rad and hid it. Against the residual oracle it is visible. Opened as
   `[13-crossing-neighbourhood-refinement-was-sized-at-k-1e5]`.

3. **`d_wPerturbations_dz` was 8.2 % / 346 % / 3.7 % wrong at the three crossings and nobody was
   measuring it.** It is now 1.1e-06 / 7.8e-04 / 3.5e-07. The campaign's whole attention was on
   `epsilon`, because that is what `omega_eff`'s leading term uses; `c_s^2`'s derivative enters the
   transfer sector too and had a larger relative defect than `epsilon` did. Not a new issue — this
   commit fixes it — but the *absence of any measurement of it* until now is why it is recorded
   here. `d2_wPerturbations_dz2` was not scored at all in this prompt.

4. **`fit_k` still falls globally to 3 on a short grid.**
   `fit_k = DERIVATIVE_SPLINE_ORDER if len(fit_x) >= DERIVATIVE_SPLINE_ORDER + 1 else 3` predates
   this prompt and is unchanged: it is a silent order drop of exactly the kind §2 item 4 forbids at
   a segment edge, applied at the whole-grid level. It never fires in production (the fit grid has
   5,341 points) and changing it was out of scope. The new per-branch check is strict — a branch
   that cannot hold `fit_k + 1` nodes raises rather than dropping — so the two rules now disagree in
   spirit. Left alone deliberately; it is a one-line question for whoever owns
   `_build_derivative_fit_grid` next.

5. **`epsilon`'s `d_epsilon_dz` at `T_LO` cannot be scored relatively.** The reference — a second
   difference of `ln H` — is ~0 there because `epsilon` is flat at 2.0, so the relative figures
   (1.810e+05 → 1.194e-01 on the production grid) are a ratio of two small numbers and are quoted
   nowhere in the tables above. The `T_120_MEV` figures are meaningful (1.071e+01 → 1.358e-04,
   79,000×) and the `EOS_T_LO` control is 2.616e-03 before and after.

---

## State handed to the next prompt

**`T_Z_REPRESENTATION_VERSION` is 6** (`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:414`),
raised from 5 by this prompt. A pre-prompt-13 datastore's QCD cosmology row no longer matches and
`sqla_QCDCosmology_factory.build()` raises; there is no migration and none is wanted, because every
stored QCD `omega_eff` and therefore every stored QCD phase has moved.

**New public symbols in `ComputeTargets/BackgroundModel.py`:**

* `STORED_SAMPLE_SPLINE_ORDER = 3`
* `class SegmentedSpline(edges: Sequence[float], splines: Sequence)` — `.segment_edges -> tuple`,
  `.splines -> tuple`, `__call__(u)` dispatching by `bisect_right` on `u`
* `_segment_slices(x: np.ndarray, edges: Sequence[float]) -> List[slice]` — `searchsorted(..., "left")`,
  so a datum on an edge opens the segment above it
* `_refuse_degenerate_segment(site, index, count, lo, hi, nodes, order)`
* `build_stored_sample_spline(attr, x_data, y_data, min_z, max_z, break_points=()) -> ZSplineWrapper`
  — **the single implementation of the stored-sample site**; `ComputeTargets/tests/wkb_reference.py`
  now calls it, so a future change to that site reaches the reference harness automatically. Do not
  re-inline it.

**New test module:** `ComputeTargets/tests/test_background_segmentation.py`, 12 tests, 0.21 s, no
Ray and no datastore. `epsilon_accessor(cosmology, z_sample, segment_fit, segment_stored)` gives any
of the four site combinations for one pass of `Hubble` over the fit grid and is the cheapest way to
re-take §10.4's measurement; `production_source_grid(cosmology)` returns
`(redshift_array, break_z)` for prompt 11's grid.

**The segment edges, to 17 digits**, unchanged from prompts 06 and 07 and read from the same cache
(`_break_point_crossings_log1pz`), in `u = log(1+z)`:

| name | `u` | `z` | `H` steps by |
|---|---|---|---|
| `T_LO` | `17.565806941870026` | 4.253369e+07 | 1.970e-03 relative |
| `EOS_T_LO` | `23.197460552819653` | 1.187214e+10 | 3.990e-12 (the control; only `w` kinks) |
| `T_120_MEV` | `27.485391822044257` | 8.644781e+11 | 1.377e-04 relative |

**Both `BREAK_POINT_KIND` constants are untouched and README §7 D5 is settled** — the user's
decision is to keep `TkNumericIntegration.BREAK_POINT_ALL` and
`GkNumericIntegration.BREAK_POINT_DISCONTINUITY` as they are, which is prompt 08's recommendation.
The derivative splines segment at `BREAK_POINT_ALL` (all three crossings), and §3 above is the
measurement that says the third one is load-bearing for `d_wPerturbations_dz`.

**Prompt 12's density recommendation is still open and unimplemented** —
`[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]`. Nothing here acts on
it and no grid was changed; the grid-adjacent measurements above are diagnostics of the background,
not of the density.

**What prompt 15 needs from this prompt.** The background no longer depends on which grid it is
built on (§5: the two "background on" rows now agree to four digits), so a density change is now
measurable in isolation — which is why 13 runs before 15. But **segmenting is not free on a coarse
grid**: observation 1's base-grid control row is 43× worse than the uniform fit was, at a crossing
the base grid does not resolve, and `[12-…]`'s cap-2× grid is coarser than the base grid
everywhere. A cap ladder that coarsens the crossing neighbourhoods will pay for it there.

**Reproduction, one command each:**

```bash
# §1 and §5 -- the ringing at all three crossings and the four-way table (15 s; no Ray, no datastore)
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py \
    --models QCDModel --sectors Tk Gk --k 1e5 1e7 --sections A B

# §1, §2, §4 and §6 as standing tests (0.21 s)
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_background_segmentation

# §8 -- the QCD reference fixture (~198 s; --dry-run reports without writing)
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py

# §10
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .
```
