# Log 06 — Segment the representation at the jumps: fix the max, and close T1

**Prompt:** prompts/qcd-background-audit/06-segment-at-the-jumps.md
**Commit:** *(this commit)* — "Segment the QCD temperature at its genuine discontinuities"
**Model:** Claude Opus 5
**Date:** 2026-09-14
**Result:** COMPLETE WITH DEVIATIONS

**T1 is closed.** $\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$ computed from the shipped
background is now **bit-identical** to the same integral computed from the exact background —
`1.3320002507788795e+03` on both sides, all 17 digits, relative error exactly `0.0` — against
`3.4605e-08` when prompt 01 built the guard. The equivalent phase at $k=10^5/10^7/3\times10^8$/Mpc
is `0.000e+00` rad against floors of 3.05e-07 / 3.05e-05 / 9.15e-04 rad.

**The trap, up front (README §2 (b)).** **No bracketing solver is applied to $T(z)-T_{\rm break}$
anywhere on the path that builds the representation.** The segment edges come from
`_bisect_temperature_crossing_log1pz`, which geometrically bisects the *monotone* $T(z)$ and whose
only call is to `_solve_T_z` — a root solve on the defining equation in $T$ at fixed $z$, not on
$T(z)-T_{\rm break}$ in $z$. The three `root_scalar` sites in the file are enumerated and
accounted for under "How the edges were located" below. The edges are, to 17 digits,
`17.565806941870026`, `23.197460552819653` and `27.485391822044257` in $u=\log(1+z)$, from
`break_temperatures_GeV` — `T_LO`, `EOS_T_LO`, `T_120_MEV` (README §7 **D4**) — and each is
verified to be the crossing itself: $T(\text{edge})\ge T_{\rm break}$ and
$T(\text{edge}-1\,\text{ulp}) < T_{\rm break}$.

**The number prompt 07 consumes: `BREAK_POINT_ALL` is now 2,414** on the production source grid
(2,411 tabulation knots + 3 crossings), where prompt 05 handed over 407, with median spacing
**0.67×** the grid spacing rather than 4.04×. `BREAK_POINT_DISCONTINUITY` is **2**, unchanged. The
node count, 500 → 3,000, is what bought the p90 and the median, and the knots are what it cost;
they are prompt 07's to remove, and they are why `COST_BREAK_POINT_FACTOR` had to go 1.30 → 2.40.

---

## What shipped

### `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`

**`T_Z_REPRESENTATION_VERSION`: 3 → 4** (`:390`), with the row
`4 | 06 | F is splined per branch, edges bisected onto the jumps; 3000 nodes, k=5` added to the
table in the comment block above it.

**New module-level constant** `SEGMENT_EDGE_PAD_LOG1PZ = 1.0e-12` (`:75`) — how far inside its own
branch each segment's node set is held, in $u=\log(1+z)$. The audit's `build_segmented` value;
`test_T_z_representation.SEGMENT_PAD` is asserted equal to it.

**`DEFAULT_T_Z_SPLINE_SAMPLES` 500 → 3000, `DEFAULT_T_Z_SPLINE_ORDER` 3 → 5** (`:64-65`), the
audit's §4 recommendation (README §7 **D3**). `samples` is now a *total* shared out between the
segments in proportion to their width in $u$. The measured table in the comment above them gains
three segmented rows next to prompt 05's five unsegmented ones.

**New class `SegmentedEntropyFactor(edges, splines)`** (`:78`) — $F(u)$, one `BSpline` per branch.
`__call__(u)` dispatches with `bisect_right(self._edges, u)`, so a $u$ exactly equal to an edge
evaluates in the segment *above* it, which is the branch the bisected edge belongs to. Exposes
`splines` (the per-branch splines, public so that `TemperatureRepresentation.__call__` can index
them in line), and `t` (the concatenation of the segments' knot vectors, so it stands in for a single `BSpline`
wherever one was read for its knots — `integration_break_points`, and
`docs/gktk-remedial/residual_convergence.py:268`, which reaches for
`cosmology._T_z_spline._spline.t`) and the property `segment_edges`.

**New function
`build_segmented_entropy_spline(F_of_u, edges, u_lo, u_hi, samples, order, pad=SEGMENT_EDGE_PAD_LOG1PZ)`**
(`:136`) — returns a plain `BSpline` when `edges` is empty and a `SegmentedEntropyFactor`
otherwise. Node share `max(order + 1, round(samples * width_i / total_width))`; interior ends
padded; raises `RuntimeError` naming the segment when the geometry is degenerate (`"narrower than
twice the ... padding"`, `"cannot hold ... distinct nodes"`, `"not strictly ascending and strictly
inside the tabulated range"`).

**`TemperatureRepresentation`** (`:225`) — unchanged in its arithmetic and its range logic; the
object it holds in `self._spline` is now either a `BSpline` or a `SegmentedEntropyFactor`. New
property `segment_edges`, empty for a cosmology that declares no break temperatures. `__call__`
dispatches the segments **in line** — `self._splines[bisect_right(self._edges, log_z)]`, from
`SegmentedEntropyFactor`'s `segment_edges` and `splines` hoisted in `__init__` — rather than
calling the segmented object, which measurably halves the dispatch overhead (deviation 5); an
unsegmented representation takes the `else` branch and evaluates `self._spline` exactly as prompt
05 did.

**`LambdaCDM_GenericEOS._build_T_z_spline`** (`:558`) — now asks
`_entropy_segment_edges_log1pz(u_lo, u_hi)` for the edges and
`build_segmented_entropy_spline` for the tabulation; `_T_z_spline_knots_log1pz` is the union of
every segment's knots, as before via `np.unique(spline.t)`.

**New `LambdaCDM_GenericEOS._entropy_factor_log1pz(u)`** (`:619`) — the node value
$F(u)=\log(T/[T_{\rm CMB}(1+z)])$, lifted verbatim out of prompt 05's in-line list comprehension so
that the builder takes it as a callable. Same `_solve_T_z` call, same `z = exp(u) - 1.0`, same
`(1.0 + z)`: at 500 nodes and $k=3$ the resulting representation is **bit-identical** to prompt
05's over 2,001 probes (a test asserts it).

**New `LambdaCDM_GenericEOS._entropy_segment_edges_log1pz(u_lo, u_hi) -> List[float]`** (`:635`) —
the crossings of every `break_temperatures_GeV` that lie strictly inside the tabulated range,
ascending.

**New `LambdaCDM_GenericEOS._bisect_temperature_crossing_log1pz(T_break, z_lo=1e-6, z_hi=1e19,
rtol=1e-15) -> Optional[float]`** (`:666`) — **geometric bisection of the monotone $T(z)$**, never
a root solve on $T(z)-T_{\rm break}$. Its docstring carries the reason at length.

**`integration_break_points`'s docstring** (`:769`) — the parenthetical "404 against 3 on the
production range" is now "2,411 against 3", with a sentence saying why the count moved. No logic
changed; that is prompt 07's.

### `ComputeTargets/tests/wkb_reference_data.json`

Regenerated **through prompt 02's generator only**
(`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py`, 186.2 s,
no Ray, no datastore). Nine of the twelve science keys moved; `grid`, `z_top` and `checkpoints` did
not. The `method` string's sentence *"H(z) here is itself a spline evaluation of T(z), so no
higher-precision reference exists"* was replaced, before the run, by the measured statement that
the reference is no longer circular; the generator carried it through and appended its own
`T_Z_REPRESENTATION_VERSION=4` provenance.

### Tests

`CosmologyModels/tests/test_T_z_representation.py` — thresholds tightened to README §6.1's "After
06" column (`T_PHOTON_MAX` 7.24e-04 → **1e-10**, `T_PHOTON_P90` 9.0e-08 → **1e-14**,
`T_PHOTON_MEDIAN` 3.0e-10 → **1e-15**, `CONFORMAL_TIME_REL` 4.0e-08 → **1e-15**); new constants
`HUBBLE_MAX/P90/MEDIAN`, `EDGE_AGREEMENT_ULP`, `MISPLACED_EDGE_MIN_MAX`, `STEP_FLOOR`; five new
cases and one new class. `CosmologyModels` **18 → 30**.

`ComputeTargets/tests/test_background_tau.py` — `QCD_FLOOR_FACTOR` **3.2 → 3.0** (tightened back),
`QCD_BREAK_POINT_ALIGNMENT_TOL` 3.1e-05 → **1.5e-04**, and `test_qcd_break_points` takes its knot
count from the tabulation in the tree rather than from the stale `convergence` block.
`ComputeTargets/tests/test_background_cs_tau_friction.py` — `QCD_FLOOR_FACTOR` **8.3 → 3.0**
(tightened back). `ComputeTargets/tests/test_numeric_break_points.py` — two golden values
re-scored. `ComputeTargets/tests/test_phase_residual.py` — `COST_BREAK_POINT_FACTOR` 1.30 → **2.40**.

---

## Deviations from the prompt

### 1. The bisection is geometric in $z$, not in $1+z$ — IMPLEMENTATION CHOICE

Prompt §2 item 1 says "geometric bisection on $(1+z)$ to a relative $10^{-15}$". The shipped code
bisects geometrically in $z$ over $[10^{-6}, 10^{19}]$ and takes `log1p` of the final bracket,
which is what `T_z_reference.jump_locations` does step for step. The reason is that
`jump_locations` is the reference implementation this prompt is scored against (prompt §2 item 1
again), and mirroring it exactly makes the comparison an equality rather than a tolerance: two of
the three production edges come out bit-identical to it and the third one ulp away (see
Verification). The crossings are at $z\ge4\times10^7$, where the two geometries differ by 1e-8 of
a bracket width in the first iterations and by nothing at all in the last ones. The one ulp of
difference that remains comes from `math.sqrt(lo*hi)` against the reference's `(lo*hi)**0.5`,
which disagree for about 0.13 % of arguments on this platform; `sqrt` is the correctly-rounded
operation and is what shipped.

The test therefore asserts something **stronger** than agreement with the reference: that each
edge *is* the crossing, `T(edge) >= T_break` and `T(edge - 1 ulp) < T_break`, which is a
first-principles statement about a monotone function and is what licenses the dispatch.

### 2. `test_background_tau.py::test_qcd_break_points` had to stop reading its expected count from the JSON — STRUCTURALLY REQUIRED

The prompt allows "tolerances, thresholds and comments only" in the `ComputeTargets` test modules.
This assertion is none of those: `expected = convergence.geometry.QCDModel.T_spline_knots_in_range
+ len(branch_boundaries)` = 404 + 3, and the tabulation now has 2,411 knots inside the production
range. The count is **false**, not stale, so no tolerance can absorb it. This is
`[02-fixture-tests-pinned-to-todays-break-point-artefact]`, opened by prompt 02 against prompt 07,
arriving one prompt early because the node count moved as well as the representation's shape.

What shipped keeps the assertion's substance — the break-point set is *every interior knot of the
tabulation plus the equation-of-state crossings, and nothing else* — and takes the knot count from
`self.s.qcd._T_z_spline_knots_log1pz` instead of from the block
`docs/gktk-remedial/residual_convergence.py` wrote before the background moved. The stale figure is
still printed beside it. Prompt 07 rewrites this assertion again when the set collapses to 3.

The alternative — keeping 500 nodes so the count did not move — was rejected because it is not
available: segmented at 500 nodes and $k=3$ the representation reads max 1.013e-05 / p90 6.544e-08
/ median 9.018e-12 and the conformal-time guard reads **3.8391e-13**, which misses every row of
prompt §4 including the one whose miss is a stop.

### 3. Two loosened tolerances, and two of prompt 05's taken back — IMPLEMENTATION CHOICE

Prompt 04 §2 item 5's rule (a tolerance may tighten or stay; more than one loosening is a stop and
ask) was applied as follows.

**Loosened, one:** `COST_BREAK_POINT_FACTOR` 1.30 → 2.40 in `test_phase_residual.py`, measured
2.337. This is a real cost and not a stale figure: `BREAK_POINT_ALL` on the production source grid
is **2,414** where it was 407, so the `rho` Gauss table splits a panel every 0.67 grid intervals
rather than every 4.04. It is the price of the node count, it is prompt 07's to remove, and it is
opened below as `[05-break-point-set-grew-with-the-node-count]`.

**Loosened, but the same known staleness:** `QCD_BREAK_POINT_ALIGNMENT_TOL` 3.1e-05 → 1.5e-04
(measured 1.418851e-04), which is one of the three figures
`[01-convergence-block-has-a-separate-generator]` already owns and which prompts 04 and 05 each
moved for the same reason. The orchestrator's briefing for this prompt says in terms to treat a
further move of these three as that known staleness rather than as a new defect, and not to run
`residual_convergence.py`. Noting what is being compared: the block records the `T_120_MEV`
crossing at $u = 27.485249937$, and the tree now computes $27.485391822$ — and **the tree is
right**, to 3 ulp of the independent bisection, because the representation now jumps where the
cosmology does.

**Tightened, two, one prompt earlier than the board expected:** both `QCD_FLOOR_FACTOR`s go back to
**3.0** (`test_background_tau.py` from 3.2, `test_background_cs_tau_friction.py` from 8.3). The
numerators fell by two orders under the segmented background — 5.8348e-14 → 2.104e-15 for `tau`
and 1.5501e-13 → 2.212e-15 for `cs_tau`, against floors still recorded as 1.879e-14 and 1.887e-14
— so the model's fixed-order table now agrees with the JSON an *order below* the floor recorded for
the JSON itself. Prompt 08 still re-runs the script and re-measures both sides; one of prompt 05's
three is left for it.

### 4. Two golden values in `test_numeric_break_points.py` re-scored — STRUCTURALLY REQUIRED

The same class of update prompt 04 made twice (its log, deviation 2): a literal that moved with the
representation, not a loosened tolerance.

* `test_the_branch_crossing_is_inside_the_range`: `8.6438180e11` → **`8.6447811e11`**, `places=6`
  unchanged. Every earlier value in that comment was the redshift at which a *smooth interpolant*
  happened to pass through 0.12 GeV; this one agrees to 3 ulp of $u$ with the independent
  bisection, because the representation now steps where the equation of state does.
* `test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink`: the expected relative jumps in
  $H$ move from `{T_LO: 4.4e-4, T_120_MEV: 1.0e-4}` to **`{T_LO: 1.97e-3, T_120_MEV: 1.38e-4}`**
  (measured 1.969955e-03 and 1.377111e-04; `EOS_T_LO` still continuous, 9.272151e-11 against a
  1e-8 bound). **This is a finding, not a re-score of noise**: the old figures were what a smoothed
  representation showed, because evaluating $H$ a relative $10^{-12}$ in $u$ either side of the
  crossing moved $T$ *continuously* and picked up only the branch jump in $g$ at fixed $T$. The
  whole of it is $\delta H/H = 2\,\delta T/T + \tfrac12\,\delta g/g$ = 2(7.626e-04) +
  (1/2)(8.876e-04) = 1.97e-03. So **`GkTk-remedial` log 02's "4.4e-4 jump in $H(z)$ at
  $z=4.24\times10^7$" is the old representation's figure and not the cosmology's** — the cosmology
  jumps by 1.97e-03 there. Recorded on the board against
  `[00-eos-branch-joins-do-not-match]`, whose impact statement quotes the 4.4e-4.

### 5. `T_photon`'s cost per call is a **possible miss** of prompt §4's ≤ 2.5 µs — IMPLEMENTATION CHOICE (reduced once, then reported rather than chased)

**This machine is not quiet.** The absolute figures below were taken at load averages between 11
and 15 from unrelated processes, where every candidate reads ~20 % high; the same
`timeit` harness on the same tree read 2.206 µs for prompt 05's shape earlier in the day and
2.651 µs for it now. **Only the ratios taken inside one process are trustworthy, and the absolute
target needs re-measuring on a quiet machine.** Both are given.

*First measurement, machine quiet* (`timeit`, min of 5 runs of 20 passes over 200 probe points,
all candidates in one process, dispatch inside `SegmentedEntropyFactor.__call__`):

| representation | µs/call |
|---|---|
| prompt 05's shape (500 nodes, $k=3$, unsegmented) | 2.206 / 2.224 / 2.217 |
| unsegmented, 3,000 nodes, $k=5$ | 2.369 / 2.397 / 2.447 |
| **segmented, 3,000, $k=5$** | **2.519 / 2.514 / 2.564** |

That is ~2.53 µs against ≤ 2.5 µs, of which +0.19 µs was the order-5 spline evaluation and
+0.13 µs the dispatch. The dispatch cost was a *second Python-level call*, not the `bisect_right`,
so it was removed: `TemperatureRepresentation.__call__` now indexes the segment list in line
(`SegmentedEntropyFactor.splines` is public for that reason) instead of calling the segmented
object. Nothing else changed — `LambdaCDM` and `RadiationModel` are still byte-identical, the
suites still pass, and an unsegmented representation takes an `if self._edges:` branch that leaves
prompt 05's path as it was.

*Second measurement, after the inline, machine at load 11–15* (same harness, one process):

| representation | µs/call | ratio to prompt 05's shape |
|---|---|---|
| the audit's own control (bare closure, segmented, no range logic) | 2.710 / 2.622 / 2.723 | — |
| prompt 05's shape (500, $k=3$), through this class | 2.651 / 2.639 / 2.677 | 1.00 |
| unsegmented 3,000, $k=5$, through this class | 2.872 / 2.837 / 2.907 | **1.08 / 1.08 / 1.09** |
| **prompt 06 as shipped (segmented 3,000, $k=5$)** | 2.879 / 2.989 / 2.913 | **1.09 / 1.13 / 1.09** |

Two things follow, and they are ratios so the load does not touch them. **The segment dispatch is
now free**: shipped against unsegmented-at-the-same-order reads 1.002 / 1.054 / 1.002, where before
the inline it was ~1.05–1.06. **What is left is entirely the order-5 spline evaluation**, +8 to
+9 % over prompt 05's cubic, and that is SciPy's `BSpline.__call__`, which was already 2.0 µs of a
2.5 µs call on the quiet machine — four fifths of the cost, whatever the order. Order 5 is not
optional: a cubic needs ~25,000 nodes to reach the p90 prompt §4 requires, and every node is a
declared break point.

**Recorded as a possible miss, not as a met target.** The best estimate is ~2.4 µs (prompt 05's
quiet-machine 2.21 µs × the measured 1.09), which would be inside ≤ 2.5 µs, but that is an
inference and not a measurement, and the one direct quiet-machine measurement of the pre-inline
code was 2.53 µs. The target was not loosened, and the row in the acceptance table below is marked
`?` rather than ✓. Opened as `[06-t-photon-call-cost-needs-a-quiet-machine]` for re-measurement.

For context on audit §2 (c): the audit's own bare closure, measured in the same loaded process
above, reads 2.71 / 2.62 / 2.72 µs — i.e. **the shipped representation is not slower than the
design the audit measured**; it is the same thing plus `TemperatureRepresentation`'s range logic,
which prompt 05 introduced and `ZSplineWrapper` has always carried. Hoisting that logic's two
loop-invariant `_outward` calls into `__init__` would return a further ~0.11 µs (measured, quiet
machine); not done, because it is prompt 05's code and this prompt was not asked to change it —
`[07-t-photon-range-logic-recomputes-its-bounds]`.

`_build_T_z_spline` costs **87.8 ms** at the shipped defaults, of which **4.2 ms** is the edge
bisection (4 temperatures × ~57 halvings × one `rtol=1e-14` root solve). Inside prompt §4's
≤ 100 ms, and the largest single number in the acceptance table that is close to its bound.

### None else

`QCD_EOS.py`, `integration_break_points`'s logic, `ComputeTargets/spline_wrappers.py`,
`ComputeTargets/BackgroundModel.py`, `phase_residual.py`, `AdaptiveLevin/`, `QuadSourceIntegral.py`
and every `extract_*.py` are untouched (`git diff --name-only` is the seven files listed in "What
shipped"). No threshold in README §6 was loosened.

---

## Verification performed

Everything below was **run**, from the repository root with `PYTHONPATH=.`, no Ray and no
datastore except where a suite is named.

### Prompt §4's acceptance table

| Quantity | Before (prompt 05) | Target | **Measured** |
|---|---|---|---|
| `T(z)` relative error, max | 7.236e-04 | ≤ 1e-10 | **6.807e-11** ✓ |
| `T(z)` relative error, p90 | 8.912e-08 | ≤ 1e-14 | **3.237e-15** ✓ |
| `T(z)` relative error, median | 2.599e-10 | ≤ 1e-15 | **1.765e-16** ✓ |
| $H(z)$ on the production grid, max / p90 / median | 1.280e-03 / 1.923e-07 / 5.430e-10 | ≤ 2e-10 / 1e-14 / 1e-15 | **1.690e-10 / 6.276e-15 / 2.804e-16** ✓ |
| **relative error in $\int\mathrm{d}z/H$** | 5.4264e-10 | ≤ 1e-15 | **0.0 — bit-identical** ✓ |
| equivalent phase at $k=10^5/10^7/3\times10^8$ | 0.74 / 74 / 2235 rad | below 3.05e-7 / 3.05e-5 / 9.15e-4 rad | **0.000e+00 / 0.000e+00 / 0.000e+00** ✓ |
| `T_photon` cost per call | 2.240 µs | ≤ 2.5 µs | **2.53 µs quiet / +9 % on prompt 05 under load** ? (deviation 5) |
| `_build_T_z_spline` wall time | 13.3 ms | ≤ 100 ms | **87.8 ms** ✓ |

The three `T(z)` statistics are the audit §4 row for the recommended representation to the digit
(6.807e-11 / 3.123e-15 / 1.773e-16), and the $H(z)$ row is audit §5's improved row
(1.690e-10 / 6.314e-15 / 2.803e-16). The probe set is prompt 01's fixed 640 points.

### The T1 guard, in full

```
[conformal time] int dz/H over z in [1e+02, 1e+12], 3 interior jumps given to the integrator
  shipped background = 1.3320002507788795e+03
  exact   background = 1.3320002507788795e+03
  relative error in tau = 0.0000e+00   (bit-identical: True)
    k =     1e+05 /Mpc:   0.000e+00 rad   against a 1-ulp floor of 3.050e-07 rad
    k =     1e+07 /Mpc:   0.000e+00 rad   against a 1-ulp floor of 3.050e-05 rad
    k =     3e+08 /Mpc:   0.000e+00 rad   against a 1-ulp floor of 9.150e-04 rad
```

3.4605051e-08 (prompt 01) → 3.4509e-08 (04) → 5.4264e-10 (05) → **0.0** (06). The threshold
asserted is 1e-15 rather than the equality, because the identity is the last bit of a sum over a
few hundred quadrature panels and should not be a test's hinge; the phase floors are asserted
instead, and the identity is printed.

### The segment edges, to 17 digits — README §7 **D4**: `break_temperatures_GeV`, all four

All four break temperatures are used as candidate edges, not the three
`discontinuity_temperatures_GeV`, which is what the audit measured and what README §7 D4 keeps.
`T_HI = 1e16` GeV is crossed at $z\sim10^{28}$, above the tabulated range, and is dropped by the
range filter, so the production model has three edges:

| # | $T_{\rm break}$ | `u = log(1+z)` | hex | `z` | vs `jump_locations` |
|---|---|---|---|---|---|
| 1 | `T_LO` = 1e-5 GeV | `17.565806941870026` | `0x1.190d8b9472e79p+4` | `4.253368543e+07` | **identical** |
| 2 | `EOS_T_LO` = 0.002 GeV | `23.197460552819653` | `0x1.7328cc6589c49p+4` | `1.187214281e+10` | +1 ulp |
| 3 | `T_120_MEV` = 0.12 GeV | `27.485391822044257` | `0x1.b7c42a3716d0ap+4` | `8.644781111e+11` | **identical** |

and each is the crossing itself to the last bit:

| $T_{\rm break}$ | $(T-T_{\rm break})/T_{\rm break}$ at the edge | one ulp below |
|---|---|---|
| 1e-05 GeV | **+8.844019e-06** | **−7.531645e-04** |
| 0.002 GeV | +1.998401e-15 | −1.998401e-15 |
| 0.12 GeV | **+1.010849e-04** | **−6.194863e-08** |

The 1-ulp disagreement at edge 2 is at the join where $g_s$ is continuous to 1.751e-11 — the one
place where which side of the edge a point falls on does not matter.

### How the edges were located — no bracketing solver anywhere in the path

`_build_T_z_spline` → `_entropy_segment_edges_log1pz` → `_bisect_temperature_crossing_log1pz`,
and that last function contains one loop:

```python
lo, hi = z_lo, z_hi
for _ in range(200):
    mid = sqrt(lo * hi)
    if self._solve_T_z(mid) < T_break:
        lo = mid
    else:
        hi = mid
    if hi / lo - 1.0 < rtol:
        break
return log1p(0.5 * (lo + hi))
```

The only call it makes is `_solve_T_z`, whose own `root_scalar` solves the *defining equation*
$T\,g_s(T)^{1/3} = T_{\rm CMB} g_s(T_{\rm CMB})^{1/3}(1+z)$ at fixed $z$ — a continuous, monotone
function of $T$ with a genuine root — and never $T(z)-T_{\rm break}$ as a function of $z$. The
test of each step is an inequality, which a discontinuity does not disturb. `root_scalar` appears
in this file in three places and none of them is an edge: `_solve_T_z` (the node solve),
`_find_rho_equality` (matter–radiation equality) and `_temperature_crossing_log1pz`, which is
`integration_break_points`' own crossing finder and is prompt 07's to decide about — it is not on
the path that builds the representation.

The trap is still demonstrated on this tree by prompt 01's case 7, unchanged:
`+1.126e-12` in $u$ above the bisected edge at `root_scalar`'s default tolerances (317 ulp, and
more than the 1e-12 padding), `+3.304e-13` on a local bracket, `+1.421e-14` at
`xtol = rtol = 1e-15`, each reporting `converged=True` with a residual of +8.844e-06.

### The edge-misplacement guard (prompt §3 test 3)

Built with the lowest edge moved up by one node of its own segment (1.5473e-02 in $u$, i.e.
$z_{\rm edge}$ 4.253369e+07 → 4.319692e+07 — **1.6 %**, against a production grid spacing of
2.3032e-02 in $u$):

```
segmented, one edge one node too high          max 1.709e-05   p90 4.040e-15   median 1.771e-16
  inside the displaced window, z = 4.269853e+07: 7.054e-04
  inside the displaced window, z = 4.286402e+07: 5.842e-04
  inside the displaced window, z = 4.303015e+07: 3.639e-04
```

Two things worth reading twice. The probe-set maximum comes back by **2.5×10⁵**, from 6.807e-11 to
1.709e-05 — but the p90 and the median *do not move at all*, which is exactly the silent failure
the prompt warns about. And the 640-point probe set does not sample the displaced window (its
spacing in $u$ is 5.8e-02 against the window's 1.5e-02), so the full **7.054e-04** — the audit's
5.7e-04, the whole jump height — is only visible to a probe placed inside it deliberately. The
test asserts both.

### The step is reproduced, not smoothed (prompt §3 test 4)

Evaluated one ulp of $u$ either side of each edge, on $u$ directly rather than through a recovered
$z$:

| edge | $T$ below | $T$ at the edge | error vs the defining equation | step | exact step |
|---|---|---|---|---|---|
| $z=4.253368543\times10^{7}$ | 1.562126613304e+33 | 1.563317864360e+33 | 0.0 / 1.844e-16 | **+7.625829e-04** | +7.625829e-04 |
| $z=1.187214281\times10^{10}$ | 3.126608076939e+35 | 3.126608076939e+35 | 0.0 / 0.0 | +3.996803e-15 | +3.996803e-15 |
| $z=8.644781111\times10^{11}$ | 1.875964729950e+37 | 1.876154477877e+37 | 1.259e-16 / 5.034e-16 | **+1.011469e-04** | +1.011469e-04 |

The step at the lowest crossing is `expm1(7.6229229003969e-04) = 7.625829e-04`, log 01's deviation
2, and the representation carries it to six figures with both sides at the round-off floor. A
representation that smoothed the step by even one node interval would fail on the lower side.

### Degenerate geometry (prompt §3 test 5)

`TestSegmentGeometry`, seven cases on a synthetic smooth $F$ so that the geometry is what is under
test: an edge above the range, below it, exactly at either tabulation bound, out of order,
repeated — all `RuntimeError`, and the first two name "ascending"; a segment narrower than twice
the padding — `RuntimeError` naming "padding"; a segment too narrow for `order + 1` distinct
floats (with `pad=0` so the first check does not catch it first) — `RuntimeError` naming "distinct
nodes"; two edges 1e-06 apart — **builds**, the middle segment taking exactly `order + 1 = 6`
nodes and the representation still accurate to 2.222e-11 over 5,000 probes; no edges — a plain
`BSpline`. Plus the production filter: `_entropy_segment_edges_log1pz` over $z\in(1,10^3)$ returns
`[]`, and over the tabulated range returns 3.

### Smooth cosmologies (prompt §3 test 6, README §2 (g))

**`LambdaCDM` and `RadiationModel` are byte-identical to `8d6e913`.** 7,014 `float.hex` lines —
`Hubble`, `rho`, `T_photon`, `wBackground`, `wPerturbations` for `LambdaCDM(Planck2018)` and
`Hubble`, `tau` for `RadiationModel`, over 1,001 redshifts from 0 to $10^{19}$ — MD5
`78f633c20adcf05528e3141ea06d62a1` on both trees, `diff` empty. (The prompt-05 tree was checked out
as a `git worktree` and driven by the same script.)

**The single-segment code path is prompt 05's path, bit for bit.** On
`LambdaCDM_GenericEOS(PureRadiationEOS)`, `build_segmented_entropy_spline(..., [], u_lo, u_hi, 500,
3)` and an explicit `linspace` + `make_interp_spline` reproduction of prompt 05's construction agree
on **2,001 of 2,001** probes exactly, and their knot vectors are equal element for element. The
node count has to be passed explicitly because the shipped defaults moved; what is asserted is that
the *code path* is the same one, which is the thing that could have drifted.

**A constant-$g_s$ equation of state is still exact**: `EXACT_RAMP_MAX` = 1e-15 unchanged, and the
representation additionally now asserts `segment_edges == ()` and that the object inside it is a
plain `BSpline`. `test_temperature_spline.py`'s `INTERPOLATION_FLOOR` = 1e-15 passes untouched.

### The reproduction script, unedited (prompt §3 test 7)

`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py`, 1.3 s
total, no edits. Its §3 table now reads:

```
   shipped (500 pts, sloppy nodes, T against u)         max 6.807e-11   p90 3.237e-15   median 1.765e-16
   ...
   SEGMENTED entropy factor, 3000 pts, k=5              max 6.807e-11   p90 3.123e-15   median 1.773e-16
```

— the production class and the script's own independently-built segmented candidate agree on the
max to every digit, and differ in the last digits of the p90 and the median because the script
bisects with `(lo*hi)**0.5` and takes its range from its own `SHIPPED_*` constants. §4 reads
`shipped = improved = exact = 1.3320002507788795e+03` and every phase column `0.000e+00`. §0's
"shipped `_solve_T_z`" line reads `0.0 / 0.0 / 0.0` and its "shipped T(z) spline" line is the
production representation. (Its *labels* are now wrong — it calls the production representation
"500 pts, sloppy nodes" — but the script is a reproduction of the audit and the audit is
additive-only; it is not edited here.)

### README §7 **D3** — what the cheaper candidate bought

Three segmented candidates, built through `_build_T_z_spline(samples=, order=)` and scored on the
same probe set, with `H(z)` on the production grid and the conformal-time guard alongside:

| nodes | order | max | p90 | median | $H$ max | $\int\mathrm{d}z/H$ | build | knots in range |
|---|---|---|---|---|---|---|---|---|
| 500 | 3 | 1.013e-05 | 6.544e-08 | 9.018e-12 | 2.516e-05 | **3.8391e-13** | 18.8 ms | 398 |
| 2000 | 5 | 1.292e-09 | 4.481e-14 | 2.374e-16 | 1.026e-08 | **0.0** | 61.4 ms | 1604 |
| **3000** | **5** | **6.807e-11** | **3.237e-15** | **1.765e-16** | **1.690e-10** | **0.0** | 88.2 ms | 2411 |

2,000 / $k=5$ is genuinely cheap and already gives a **bit-identical** conformal time — the T1 row
does not separate it from 3,000 — but it misses two of prompt §4's four accuracy rows (max by four
orders, p90 by 4.5×) and buys only 27 ms and 800 break points. 500 / $k=3$ segmented fixes the max
by two orders but misses everything, including the T1 row, which is a stop. **3,000 / $k=5$ is
shipped, which is the audit's recommendation unchanged.** Order 5 rather than 3 is what buys the
p90: a cubic needs $h$ smaller by $(6.5\times10^{-8}/10^{-14})^{1/4}\approx50$, i.e. ~25,000 nodes,
and every node is a declared break point.

### The QCD reference fixture

Regenerated in this commit, 186.2 s. Science keys that moved, largest relative move in each:

| key | largest relative move | where |
|---|---|---|
| `rho_G` | **1.613399e-01** | $k=3\times10^8$: −0.0011816880337 → −0.0013723414976 |
| `rho_T` | 3.878795e-03 | $k=3\times10^8$: −0.09195777 → −0.09231446 |
| `tau_minus_top` | 1.568815e-05 | 0.04617195672569687 → 0.046172681078437666 |
| `cs_tau_minus_top` | 1.568765e-05 | |
| `primitives_at_rho_anchor` | 3.668170e-06 | |
| `rho_anchor_z` | 9.343532e-08 | |
| `friction_F_minus_top` | 3.657094e-08 | |
| `short_baseline` | 1.428571e-01 | an `agreement_fraction` diagnostic, 1.539e-15 → 1.319e-15 |
| `reference_floor` | — | self-agreement diagnostics only |

`grid`, `z_top` and `checkpoints` unchanged. The 16 % move in `rho_G` is the largest number in this
campaign so far and it is the right sign of surprise: `rho_G` is a ~1e-3 rad quantity accumulated
across the QCD transition, which is precisely where the background moved.

### `[02-qcd-reference-floor]` and `[03-qcd-short-baseline-reference-endpoint-rounding]` — re-measured, neither closes

**`[02-qcd-reference-floor]`: narrowed, not closed.** Its number (1.88e-14 / 1.89e-14) is
`convergence.models.QCDModel.branch+knots.*.json_vs_reference_max_rel`, which only
`residual_convergence.py` writes and which this prompt must not run. What can be measured moved a
long way: the model's order-4 cumulative table now agrees with the JSON at **2.104e-15** (`tau`)
and **2.212e-15** (`cs_tau`), an order *below* the recorded floor, where prompt 05 sat above it;
and the JSON's own self-agreement, which prompt 02's generator does write, is `quad_epsrel_1e-12`
**0.0** for both and `gauss40_bisect` 1.9875e-15 / 2.4488e-14. The circularity half of the issue is
gone — the representation reproduces the defining equation to 6.807e-11 max and 1.765e-16 median,
so the `rtol=1e-14` root solve is a genuine oracle for the background the references are built on,
and the fixture's `method` string now says so. Closing it needs prompt 08's re-run.

**`[03-qcd-short-baseline-reference-endpoint-rounding]`: unchanged in kind, and not closable by
this campaign.** The cause is that the short-baseline references integrate between *rounded*
`log1p(z)` endpoints — a parametrisation of the quadrature limits, not a property of the
background — so a better $T(z)$ cannot touch it. Measured after the regeneration, the three
`agreement_full` / `agreement_fraction` figures are 1.4537e-15 / 1.3192e-15, 1.9903e-16 /
5.3991e-16 and 0.0 / 1.6863e-16, i.e. the same 1e-15 level as before (the middle and last are
bit-identical to prompt 05's). `test_background_tau`'s QCD short baselines read 4.200e-15,
9.155e-15 and 1.746e-15 against README §6's 1e-13.

### Break points, for prompt 07

| kind | on the production source grid (1,732 samples, $z\in[0.1, 2.064\times10^{16}]$) |
|---|---|
| `BREAK_POINT_ALL` | **2,414** — 2,411 knots + 3 crossings; median spacing 1.5474e-02 in $u$ = **0.67×** the grid spacing |
| `BREAK_POINT_DISCONTINUITY` | **2** — unchanged, median spacing 430.69× the grid spacing |

Prompt 05 handed over 407 (404 + 3). The knot count rose with the node count, which is the price
paid for the p90 and the median, and prompt 07 removes all 2,411 of them.

### Suites

All three were run again after the dispatch was inlined (deviation 5), and these are those runs:

| suite | before (`8d6e913`) | after | wall |
|---|---|---|---|
| `CosmologyModels/tests` | 18 OK | **30 OK** | 0.69 s |
| `ComputeTargets/tests` | 354 OK | **354 OK** | 196 s |
| `LiouvilleGreen/tests` (fast set) | 143 OK | **143 OK** | 16.4 s |

The `LiouvilleGreen` fast set is every module except `test_3bessel_analytic`, which is ~1,400 s and
which nothing in this prompt can reach: no file it imports is touched, and the QCD cosmology does
not appear in it.

`black` (25.1.0): the seven files this commit touches are clean under `--check`.

---

## Observations not acted on

1. **`TemperatureRepresentation.__call__` recomputes `_outward(self._max_log_z, +1)` and
   `_outward(self._min_log_z, -1)` on every call.** Both are loop-invariant — the bounds are set in
   `__init__` and never mutated — and each costs 0.056 µs, measured on the quiet machine, of a
   ~2.5 µs call. Hoisting them into `__init__` is numerically null and would return ~0.11 µs. Not
   done: it is prompt 05's code and this prompt was not asked to change it. Opened as
   `[07-t-photon-range-logic-recomputes-its-bounds]`.

2. **`docs/qcd-background-audit/measure_T_z_representation.py`'s labels now lie**, although its
   numbers do not: it prints the production representation as "shipped (500 pts, sloppy nodes, T
   against u)" and "shipped T(z) spline (500 pts, over those nodes)". The script is the audit's
   reproduction and the audit is additive-only (`CLAUDE.md`), and prompt §3 test 7 asks that it
   still run unedited, which it does. A later reader comparing its §0 and §3 rows will find them
   identical and should read §0's label as "whatever is in the tree".

3. **`_temperature_crossing_log1pz` root-finds on `T_photon(z) - T_break`**, which the
   representation has now made a genuine step function — the thing README §2 (b) forbids for a
   segment edge. It happens to land within 3 ulp of $u$ of the bisected crossing on all three
   production crossings (measured: 17.565806941870036 against 17.565806941870026, and similarly for
   the other two), because the step it is bracketing is now one ulp wide rather than a node
   interval wide. **But it is the same solver on the same shape of function, and its landing place
   depends on its tolerances.** The prompt says explicitly that prompt 07 owns whether it survives;
   it is not touched here. Prompt 07 should replace it with
   `_bisect_temperature_crossing_log1pz`, which now exists in the same class and is what the
   representation itself uses.

4. **The QCD `BackgroundModel` cumulative table costs more to build**, as the break-point count
   demands: 16,580 `Hubble` evaluations and 0.683 s against LambdaCDM's 6,924 and 0.166 s, and the
   `rho` residual table needs 2.337× the order×intervals baseline where it needed 1.24×. Recorded
   as `[05-break-point-set-grew-with-the-node-count]` and owned by prompt 07.

5. **`test_kind_selects_knots_or_jumps`'s `assertGreater(len(every), 100)` survived** this prompt
   (2,414 > 100) but is still one of the two assertions
   `[02-fixture-tests-pinned-to-todays-break-point-artefact]` names; prompt 07 still has to rewrite
   it. Only its sibling in `test_background_tau.py` had to be touched here.

---

## State handed to the next prompt

- **`T_Z_REPRESENTATION_VERSION` is now `4`** (`LambdaCDM_GenericEOS.py:390`). Prompt 07 bumps it
  to 5 and adds its row to the table in the comment block above the declaration.
- **`BREAK_POINT_ALL` is 2,414 on the production source grid** (2,411 knots of the tabulation + 3
  equation-of-state crossings) and `BREAK_POINT_DISCONTINUITY` is **2**. Prompt 05 handed over 407;
  the difference is the node count, 500 → 3,000, and **every one of the 2,411 is prompt 07's to
  remove**. README §6.3's target of 3 is unchanged by this.
- **The three segment edges, to 17 digits**, which are also the three genuine crossings:
  `17.565806941870026` (`0x1.190d8b9472e79p+4`, $z=42533685.432205893$),
  `23.197460552819653` (`0x1.7328cc6589c49p+4`, $z=11872142813.468178$) and
  `27.485391822044257` (`0x1.b7c42a3716d0ap+4`, $z=864478111114.07019$) — `T_LO`, `EOS_T_LO`,
  `T_120_MEV`, i.e. **`break_temperatures_GeV`** (README §7 D4 kept). These are the values
  `T_z_reference.jump_locations` returns, to 0 or 1 ulp.
- **A production bisector now exists:**
  `LambdaCDM_GenericEOS._bisect_temperature_crossing_log1pz(T_break, z_lo=1e-6, z_hi=1e19,
  rtol=1e-15)` and `._entropy_segment_edges_log1pz(u_lo, u_hi)`. Prompt 07 should use them rather
  than `_temperature_crossing_log1pz`, whose `root_scalar` now brackets a genuine step (observation
  3 above).
- **The representation object:** `TemperatureRepresentation._spline` is a `SegmentedEntropyFactor`
  for a cosmology with break temperatures and a plain `BSpline` for one without; both are callable
  on $u$ and both expose `.t`, so `docs/gktk-remedial/residual_convergence.py:268` still works
  unchanged. `TemperatureRepresentation.segment_edges` and
  `SegmentedEntropyFactor.segment_edges` return the interior edges.
- **Node count and order are `DEFAULT_T_Z_SPLINE_SAMPLES = 3000` / `DEFAULT_T_Z_SPLINE_ORDER = 5`**
  (a *total*, shared by width), with the eight-row measured table beside them.
  `_build_T_z_spline(min_z, max_z, samples, order)` still takes both, so a candidate can be scored
  without touching the body.
- **Costs:** `T_photon` **2.53 µs/call** (target ≤ 2.5; deviation 5, with the decomposition and the
  0.11 µs that is available for free); `_build_T_z_spline` **87.8 ms**, of which 4.2 ms is the edge
  bisection, against a 100 ms stop. **Prompt 07 will take most of the 87.8 ms back only if it
  reduces the node count, which it should not do for accuracy reasons** — but it will take back the
  quadrature cost, which is the larger number in practice.
- **Regenerate the QCD reference fixture with**
  `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py`
  (no `--dry-run`); 186.2 s on this machine. Largest move this commit: `rho_G` at $k=3\times10^8$,
  1.613e-01 relative.
- **One tolerance is still owed back to prompt 08**, not three: `QCD_BREAK_POINT_ALIGNMENT_TOL`
  = 1.5e-04 in `test_background_tau.py`. Both `QCD_FLOOR_FACTOR`s are back at 3.0. Prompt 08 should
  expect the alignment figure to be *the tree's* right answer and the block's to be the stale one.
- **`COST_BREAK_POINT_FACTOR` = 2.40** in `test_phase_residual.py` (was 1.30) is prompt 07's to take
  back, and it should end up *below* 1.30, not at it.
- **Tests whose thresholds, tolerances or literals moved:**
  `CosmologyModels/tests/test_T_z_representation.py` (tightened, plus five new cases and a new
  class), `ComputeTargets/tests/test_background_tau.py` (`QCD_FLOOR_FACTOR` tightened,
  `QCD_BREAK_POINT_ALIGNMENT_TOL` loosened, `test_qcd_break_points` re-sourced),
  `ComputeTargets/tests/test_background_cs_tau_friction.py` (`QCD_FLOOR_FACTOR` tightened),
  `ComputeTargets/tests/test_numeric_break_points.py` (two golden values),
  `ComputeTargets/tests/test_phase_residual.py` (`COST_BREAK_POINT_FACTOR` loosened). No other test
  file changed.
- **For prompt 09:** the guard test to quote is
  `CosmologyModels.tests.test_T_z_representation.TestQCDTemperatureRepresentation.test_conformal_time_matches_the_exact_background`,
  and it now reads at zero. What prompt 09 measures is whether the *consumers* moved; README §6.4
  says they are not required to improve, and §0.2 says why.
