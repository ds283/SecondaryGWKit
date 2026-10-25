# B1 — the $T_k$ seam gap removed by overlap: measurements

This document is the `handover` campaign's record for board item **B1**,
[prompt 12](../../prompts/handover/12-remove-the-tk-gap.md). §1–§4 are the code part, measured on the
laptop, offline (no Ray, no datastore), on branch `handover-12-b1` cut at `98e68d3`. §5 is
rehearsal R8, to be run on the Mac Studio and appended. Sections are additive (`CLAUDE.md`, campaign
conventions, rule 6).

**Measured:** 2026-10-25, Apple laptop, load average 8–15 (other sessions were running).
**Log:** [`prompts/handover/logs/12-remove-the-tk-gap.md`](../../prompts/handover/logs/12-remove-the-tk-gap.md).

**What changed, in one paragraph.** Three redshifts are now held apart
([`ComputeTargets/handover.py`](../../ComputeTargets/handover.py)). $z_{\rm init}$ is the
Liouville–Green initialisation point, the phase extremum, unchanged. The **switch node** $z_{\rm sw}$
is the largest source-grid node strictly below $z_{\rm init}$, the Liouville–Green row's first
sample, and the point where `TkSourceFunctions`, `QuadSource` and `QuadSourceIntegral` change
representation. The **overlap** is `HANDOVER_OVERLAP_NODES = 2` numeric nodes kept below
$z_{\rm sw}$, in the $T_k$ numeric spline and in the stored source. Both representations are stored
at $z_{\rm sw}$ and at the overlap nodes, so every representation covers its sub-interval and the
clamp is gone. No producer, no grid and no schema changed.

---

## 1. Continuity at the switch node and the overlap nodes (test (d))

`TkSourceFunctions.handover_continuity` compares **stored** samples of both representations at
$z_{\rm sw}$ and at each overlap node:
$\Delta T_{\rm rel} = (T_{\rm num} - T_{\rm LG})/|M|$ and
$\Delta T'_{\rm rel} = (T'_{\rm num} - T'_{\rm LG})/(|M|\,\omega)$, with $T_{\rm LG} = M\sin\theta$ and
$T'_{\rm LG} = M(\mathrm{d}\ln M/\mathrm{d}z\,\sin\theta + \omega\cos\theta)$.

On the **exact** fixture of `ComputeTargets/tests/test_tk_source_functions.py` in production's shape
(`Fixture(w, drop_first_WKB_sample=True)`, $k = 10^4$, hand-over 3.5 e-folds sub-horizon, numeric
samples continuing five nodes past $z_{\rm init}$):

| $w$ | node | $z$ | $x = k c_s a_0\eta$ | $\Delta T_{\rm rel}$ | $\Delta T'_{\rm rel}$ |
|---|---|---|---|---|---|
| 1/3 | $z_{\rm sw}$ | 294.10587814408103 | 19.56 | −3.426e-15 | −6.350e-06 |
| 1/3 | overlap 1 | 287.3941237465756 | 20.02 | −1.215e-15 | −8.462e-06 |
| 1/3 | overlap 2 | 280.83501845038865 | 20.49 | −1.894e-15 | −8.571e-06 |
| 0.2 | $z_{\rm sw}$ | 1229.2165004582325 | 18.86 | −2.229e-15 | −6.450e-05 |
| 0.2 | overlap 1 | 1201.2684593585168 | 19.21 | +1.704e-15 | −2.058e-06 |
| 0.2 | overlap 2 | 1173.9553414621726 | 19.56 | −3.180e-14 | +5.759e-05 |

($z_{\rm init}$ = 300.973834223185 at $w = 1/3$ and 1257.8142242434012 at $w = 0.2$.)

**The bounds asserted, and where they come from** (the test module's own tolerance comments):
- $|\Delta T_{\rm rel}| < 10^{-11}$. At a stored node both sides are the exact $T$. What is left is
  `bessel_phase`'s declared phase error, 5e-12 rad (module docstring, "the Bessel oracle"), and the
  amplitude path, asserted below 1e-12 in `test_amplitude_and_reconstruction`. Measured at most
  3.2e-14: the round-off of the reconstruction.
- $|\Delta T'_{\rm rel}| < 2.2\times10^{-5}$ ($w = 1/3$) and $3.8\times10^{-4}$ ($w = 0.2$): twice the
  Liouville–Green truncation of $\omega_{\rm eff}$ against the exact phase derivative, 1.1e-5 and
  1.9e-4 (`Fixture.theta_exact`'s docstring), because $\mathrm{d}\ln M/\mathrm{d}z$ carries a
  truncation of the same order and enters additively. Measured at most 8.6e-6 and 6.5e-5.

So on exact ingredients the two representations agree in $T$ to round-off and in $T'$ to the
Liouville–Green truncation at $x_T\approx19$, which is
`[07-lg-derivative-truncation-at-handover]`'s term. The size these residuals *should* have on
production rows is B2's question (prompt §6); they are recorded, not judged.

For contrast, the same record on the fixture's default shape, whose Liouville–Green row stores a
sample *at* $z_{\rm init}$ (the switch is then $z_{\rm init}$ itself): $\Delta T'_{\rm rel}$ =
−2.438e-06 there at $w = 1/3$ and −1.222e-04 at $w = 0.2$, the truncation one node shallower.

---

## 2. End to end on the realistic fixture, against A2 (test (g))

`N + 9/8` is $-\tfrac98\,\text{total}/\text{predicted} + \tfrac98$, scored against Kohri & Terada
eq. (22) with the head subtracted (`kohri_terada.total_from_I_RD`, double precision, with its own
returned error). The configuration is A2's: `realistic_large_x.py`'s `run_cell` at $\lambda = 1$,
$b = 0$, $x_{\rm resp} = 980$, the fixtures' `x_max` from its `fixture_x_max` rule, tolerances
(1e-45, 1e-12).

### 2.1 The test's configuration

| | N + 9/8 | its own error |
|---|---|---|
| **B1, production seam** (both LG rows start below $z_{\rm init}$, both numeric rows run past it), T-first | **−6.0375e-05** | declared 3.5e-09; eq. (22) reference 1.7e-14 |
| A2 realistic, seam closed (REALISTIC-LARGE-X.md Table 2) | +3.387e-04 | 9.2e-09 |
| A2 realistic, seam open (clamped) | −3.574e-03 | 9.1e-09 |
| A2 clamp term, realistic (Table 3) | −3.913e-03 | 2.1e-08 |

The test asserts $|N_{\rm B1}+9/8| < |N_{\rm A2,closed}+9/8| = 3.387\times10^{-4}$, and that it is
more than half a clamp term away from the open cell. The references' errors (1.7e-14, 9.2e-09) are
four orders and more below that tolerance and do not enter it (README §5 rule 8). Wall time of the
integral: 9–11 s on the laptop under load.

**Why the tolerance is the representation term's size and not its value.** B1 is not A2's closed
cell bit for bit, and should not be: it switches one grid node lower, and it moves the end intervals
of the $T$ numeric spline and of the source spline off the seam. §2.2 measures what each of those
does.

### 2.2 All three shapes, and what moved

Measured with the same harness on two trees: A2's closed cell re-run at `98e68d3` (an export of that
commit, offline), and three B1 variants at the B1 tree. The A2 column reproduces
REALISTIC-LARGE-X.md Table 2 to every printed digit.

| shape | A2 closed, at `98e68d3` | B1, LG sample at $z_{\rm init}$, numeric $T$ overlap only | + source overlap | **B1, production seam** | A2 open (clamped) |
|---|---|---|---|---|---|
| T-first | +3.387071e-04 | +3.419463e-04 | −1.729588e-05 | **−6.037469e-05** | −3.574e-03 |
| q-smooth | −4.113025e-04 | −7.785901e-05 | −7.785901e-05 | **−5.405051e-05** | +2.580e-01 |
| together | −2.677469e-03 | −2.436645e-03 | +2.652599e-03 | **+3.470031e-03** | +4.567e-01 |

Declared errors of the B1 cells: T-first 3.5e-09–9.4e-09, q-smooth 1.0e-10, together 2.1e-08–2.5e-08.
The double-precision eq. (22) reference is good to 1.7e-14 (T-first), 1.5e-08 (q-smooth, its
$u = 0.01$ cancellation) and 1.3e-13 (together).

Read across a row, each column changes one thing:
- **numeric $T$ overlap** (the $T_k$ spline no longer ends at the seam): moves `q-smooth` by
  +3.3e-4, `together` by +2.4e-4, `T-first` by +3e-6;
- **source overlap** (the $f$ spline no longer ends at the seam; `[06-source-spline-residual-vs-handover]`'s
  end interval): moves `together` by +5.1e-3 and `T-first` by −3.6e-4, `q-smooth` not at all (its
  all-smooth sub-interval ends at $G$'s hand-over, above the $T$ seam);
- **the switch one node lower** (production's seam): −4.3e-5, +2.4e-5, +8.2e-4.

So A2's "representation term" at $x_{\rm resp}=980$ was, in two of three shapes, mostly the two
splines' end intervals at the seam. B1 removes them. What is left after B1 is 6.0e-5 (T-first) and
5.4e-5 (q-smooth), five to eight times below A2's representation term and 65 to 4800 times below
its clamp term. **`together` is the exception: 3.5e-3, larger than A2's 2.7e-3**, though still 130
times below its clamp term. It is not attributed here. It is B2's to attribute, with the caveat that
the realistic Green's function is still a raw `phase_spline`
(`[02-realistic-fixture-Gk-phase-is-the-superseded-construction]`).

---

## 3. Deliberate breakage (prompt §4.3)

Each was made in the working tree, the four affected modules run
(`test_tk_source_functions`, `test_quadsource`, `test_quadsource_integral`, `test_phase_groups`,
88 tests), and the change reverted. `git status` showed only B1's own files afterwards.

**`HANDOVER_OVERLAP_NODES = 0`.** 13 failures, 1 error:
- **(c)** `test_c_numeric_spline_reaches_exactly_the_overlap` fails at both $w$: the spline ends at
  $z_{\rm sw}$ (294.106) where the decided overlap puts it two nodes lower (280.835).
- **(d), the continuity record** `test_d_handover_continuity_is_at_the_representation_floor` fails
  at both $w$: the record holds $z_{\rm sw}$ only, not the switch node and two overlap nodes. Its
  residual *values* cannot fail on this breakage — at $z_{\rm sw}$ they compare stored samples, which
  do not depend on the overlap — so (d) catches it through the record's coverage.
- **(e)** `test_e_region_reaches_the_overlap_below_the_earlier_switch_node`, and three more
  `TestQuadSourceRegion` tests (`test_A3_regression_region_and_defaults`,
  `test_store_truncates_z_sample_and_the_spline_range`, `test_datastore_round_trip_shape`): the region
  ends at the switch node.
- **(f)** `test_f_partition_records_no_gap_and_a_shortened_representation_raises`: the continuity
  record's node list.
- **(g)** errors: with no overlap in `QuadSource`'s stand-in grid the strict check refuses the source
  spline ("sub-interval bottom z=353.12 is out-of-bounds for the QuadSource spline of f").
- Also `test_reproduces_exact_solution` (both $w$), `test_WKB_grid_starting_below_the_hand_over` and
  `test_LG_row_that_skips_the_switch_node_raises` (the fixtures then store too few numeric nodes for
  the skipped node to be found, so a different message is raised).

**The switch put back at $z_{\rm init}$** (`TkSourceFunctions.crossover_z` returning `z_init`). 7
failures, 2 errors:
- **(b)** `test_b_crossover_is_the_first_LG_sample_and_the_switch_node` fails at both $w$
  (300.97 ≠ 294.11; 1257.81 ≠ 1229.22).
- **(f)** errors: `build_partition` now breaks $T_r$ at $z_{\rm init}$, which its Liouville–Green
  representation does not reach, and the strict check raises: "sub-interval top z=361.37 is
  out-of-bounds for the Liouville-Green representation of Tr (switch node z=361.3686,
  z_init=361.3686) region (353.13, 5.9282)". This is the gap the clamp used to bridge, now refused.
- **(g)** errors the same way.
- Also (a) (both $w$), (d) (both $w$) and `test_WKB_grid_starting_below_the_hand_over`.

---

## 4. Existing tests whose numbers moved

Measured on the same laptop: the `ComputeTargets` root at `98e68d3` (from an export of the commit)
against the B1 tree. Timing lines are left out. Every assertion that held before holds after.

| test | before | after | why |
|---|---|---|---|
| `test_tk_source_functions` `TestNumericRegion.test_reproduces_exact_solution`, midpoint error, $w=1/3$ | T 7.400e-06, dT/dz 2.720e-04 | T 8.554e-07, dT/dz 6.495e-05 | the numeric spline continues past the seam, so its last interval is no longer an end interval |
| the same, $w = 0.2$ | T 1.322e-06, dT/dz 2.701e-04 | T 1.593e-07, dT/dz 3.178e-05 | likewise |
| `test_phase_groups` seam boundary, exact fixture, $w=1/3$, midpoints (G smooth / oscillatory) | 1.031e-03 / 8.342e-04 | 4.424e-04 / 4.422e-04 | likewise; its probe range also extends ($x_r$ 27.9 → 28.5) |
| the same, $w = 0.2$ | 7.226e-04 / 6.991e-04 | 2.710e-04 / 2.622e-04 | likewise |
| the same, matched LG fixture, $w=1/3$ / $0.2$ | 1.065e-03 / 1.064e-03 (G smooth) | 4.732e-04 / 1.032e-03 | likewise |
| `test_phase_groups` G-only regime, f_sin vs $A_G f/H^2$ | 2.532e-16, 2.288e-16 | 4.970e-16, 3.484e-16 | round-off |
| `test_quadsource_integral` `TestAnalyticOracle.test_realistic_fixtures`, worst over 18 cases | rel 2.380e-03; smooth 4.553e-04, Levin 1.303e-04, total 4.912e-04 | rel 2.358e-03; smooth 4.782e-04, Levin 1.044e-04, total 4.865e-04 | numeric $T$ and source splines extended past the seam (thresholds 1e-3 / 5e-4 / 1.5e-3) |
| `TestErrorBound`, worst $\lvert$total−analytic$\rvert$/abserr, realistic | 4.468e+04 | 4.659e+04 | likewise |
| `TestSeams.test_integrand_continuity_at_the_hand_over`, realistic $b=0$ | 5.619e-06 | 5.613e-06 | likewise |
| `TestDirectQuadOracle`, $b=0.2$ q-smooth; `TestPartition` G-type WKB | 3.422e-13; 1.706e-14 | 3.336e-13; 1.765e-14 | round-off: the accessors are no longer called through `_ClampedTk`, which moved abscissae by an ulp |
| `test_quadsource` `TestQuadSourceRegion` (four tests) | region ends at `grid[i_stop_r]` | ends two nodes below `grid[i_stop_r + 1]` | the region now reaches the overlap; the fixture's $z_{\rm init}$ lies between nodes, as production's does |

The removed `TestHandOverClamp.test_gap_is_recorded_and_bridged` printed, at `98e68d3`: gap
2.301e-02 in $\log(1+z)$ (1.00 grid step), total with gap against without: rel 5.140e-03. That is
the clamp's effect, which B1 removes.

---

## 5. Rehearsal R8 — to come

On the Mac Studio, after the code commit is pushed: prompt §6. To be appended by the orchestrator.
