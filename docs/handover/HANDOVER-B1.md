# B1 — the $T_k$ seam gap removed by overlap: measurements

This document is the `handover` campaign's record for board item **B1**,
[prompt 12](../../prompts/handover/12-remove-the-tk-gap.md). §1–§4 are the code part, measured on the
laptop, offline (no Ray, no datastore), on branch `handover-12-b1` cut at `98e68d3`. §5 is
rehearsal R8, run on the Mac Studio and appended. Sections are additive (`CLAUDE.md`, campaign
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

## 5. Rehearsal R8 — the Mac Studio, 2026-10-25

**Host:** mps026298 (Mac Studio, arm64, Darwin 27.0.0), checkout
`/Users/ds283/Documents/Code/SecondaryGWKit` at `8a02f0c` (B1's code `049c7c8`; the commits between
touch only markdown), `git status --short` empty throughout. Python 3.13.16, `ray` 2.59.0,
`SQLAlchemy` 2.0.55, `datastorekit` 0.5.0 (`f47b9a2`), `raypoolkit` 0.3.0 (`67a75b9`),
`runregistrykit` 0.2.0 (`0bae3a3`). Load average about 4.6 on 10 cores at the start.
**Records:** [`b1/r8/`](b1/r8/): `out/`, `scripts/` and `state/`, the agent's `log.md`,
`verification.md` (this section as first written) and `issues.md`, and the orchestrator's
`review.md`. The readers' `extract/` output, its CSVs and the store were not copied back.
**Log:** [`12-remove-the-tk-gap.md`](../../prompts/handover/logs/12-remove-the-tk-gap.md) §6.

### 5.1 The Gate and the roots (prompt §6.1)

Re-checked read-only before anything was written, as §B.5 of the orchestrator's prompt gives it:
`HEAD` `8a02f0c`; 11 runs in the registry, all finished, none `running` or stale; `~/.raypoolkit/`
empty and `~/ScienceData/.rehearsal-lock` absent; no `handover-12` store and no `var/handover-12/`;
R3's two recorded runs at `var/handover-06/recorded/`; `~/.ray/auth_token` mode `-rw-------`;
117 GiB free.

| root | laptop, Part A review | Mac Studio, `8a02f0c` |
|---|---|---|
| ComputeTargets | 742 OK | **742 OK** (137.8 s) |
| ObjectFactories | 280 OK | **280 OK** (12.0 s) |

[`b1/r8/out/roots_ComputeTargets.txt`](b1/r8/out/roots_ComputeTargets.txt), [`b1/r8/out/roots_ObjectFactories.txt`](b1/r8/out/roots_ObjectFactories.txt). No `FAIL:` or `ERROR:` line; the
wall-time flake did not fire.

### 5.2 The lock and the build (prompt §6.2, §6.3)

The lock transcript, [`b1/r8/state/lock.log`](b1/r8/state/lock.log), each command its own shell invocation, output by `>>`:

| # | command | output | exit |
|---|---|---|---|
| 1 | `acquire --project SecondaryGWKit --purpose "handover prompt 12, rehearsal R8 on the Mac Studio (mps026298), tree 8a02f0c (B1's code 049c7c8)" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 2 | `set-pid --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk 21718` | recorded pid 21718 | 0 |
| 3 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after the readers) | (nothing listed) | 0 |
| 4 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 5 | `status --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | usage error: `status` takes no `--project` | 2 |
| 6 | `status --ray-temp-dir /tmp/ray-sgk` | `lock: free …`; `Ray processes of /tmp/ray-sgk: none` | 0 |

Row 5 is a mistyped look, not a lock operation ([log](../../prompts/handover/logs/12-remove-the-tk-gap.md) §6, deviation 5).

**The launch** is §B.1's command, verbatim, with `--register r8-reference --campaign handover
--prompt 12`, `--atol 1e-32 --rtol 1e-7 --cpus 6`, into
`var/datastores/handover-12-reference.sqlite`, output to `var/runs/handover-12-reference.launch.log`.
The `sh` wrapper was pid 21715 (`$!`); the driver, its child by `pgrep -P 21715`, pid **21718**, the
pid `status.json` records.

**The run** `handover-12-r8-reference-20261025T210551`:
- created 21:05:51 UTC, fingerprint taken 21:08:41, driver gone by 21:09:24; launch log ends
  `** driver exit status 0`; no `Traceback` in the launch log or the run's `stdout.log`;
  `stderr.log` empty;
- `status.json`: `done`, `exit_code` 0, pid 21718, host and boot recorded;
- manifest: campaign `handover`, prompt `12`, `git_head` `8a02f0c4…`, `git_dirty` false, Python
  3.13.16, script `RunDrivers/quadsource_atol_sweep.py`, SHA-256 `5b984c75…` (R7's launcher);
- stages: Tk numeric 0.99 s, Tk WKB 0.29 s, QuadSource 0.92 s, Gk numeric 23.1 s, Gk WKB 63.0 s,
  GkSource 23 s, policies 8.8 s, QuadSourceIntegral 40.5 s (864 of 1728 triples after the triangle
  filter, as R3);
- `fingerprint_sidecar`: "not written … there is none". The store has no sidecar, by design of the
  launcher's `--build`; the fingerprint is in `status.json`.

### 5.3 The fingerprint (prompt §6.4)

**R8's fingerprint, the rehearsal reference from here on:**
`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` (format 6, encoding 1, 21
classes, problems none), taken by `handover-12-r8-reference-20261025T210551` at `8a02f0c`,
`git_dirty` false.

`compare_with_recorded.py var/datastores/handover-12-reference.sqlite --against
var/handover-06/recorded/rrk-v020-adoption-03-reference-20261018T232838`
([`b1/r8/out/fingerprint_against_R3.txt`](b1/r8/out/fingerprint_against_R3.txt)), exit **1**, verdict `DIFFERENT`:

```
compare_fingerprints(…rrk-v020-adoption-03-reference-20261018T232838/status.json, store): 1 difference(s)
  QuadSource: tag set [LargestSourceRedshift_6.8788e+13, OutsideHorizonEfolds_e3, Run_default,
  SmallestSourceRedshift_250, SourceGridConstruction_2, SourceRedshiftGrid_1159_7139cb9c,
  TkOneLoopDensity] differs: 6 recorded, 6 now
```

**One class differs, `QuadSource`; no other class differs. `QuadSourceIntegral` does not differ.**
Every class other than `QuadSource` has R3's digest.

Why `QuadSourceIntegral` cannot differ in the fingerprint: format 6's `inventory_spec()` for
`QuadSourceIntegral` (`ObjectFactories/QuadSourceIntegral.py`) is its identity only (parents and
tags, "No validated flag and no value table"), so it does not see `total`, `metadata` or the
partition. Its `source` parent resolves cross-shard to the `QuadSource` row's reference digest,
which is `reference_digest(key, tags)` (`datastorekit/store_inventory.py:167`), and does not cover
that row's value count. The `QuadSource` rows' identities are unchanged; only their
`QuadSourceValue` count moved, which `QuadSource`'s own spec covers (`values=("QuadSourceValue",
…)`). So the fingerprint records B1's `QuadSource` change and cannot record its `QuadSourceIntegral`
change. §5.4 shows the integrals did change.

**The user's ruling (2026-10-25): this meets the review's check 3 and prompt §6.4 / §7.3.** The
prompt, the orchestrator's note and the launch's `--purpose` expected `QuadSourceIntegral` to
differ as well, because they assumed the fingerprint could see its values; that was their
misreading. What §6.4 guards, a difference in any other class, did not occur. The fingerprint's
blindness to `QuadSourceIntegral` values is opened as
`[12-the-fingerprint-cannot-see-a-quadsourceintegral-change]` on the `handover` board.

### 5.4 R8's rows, read-only (prompt §6.5)

[`b1/r8/scripts/r8_rows.py`](b1/r8/scripts/r8_rows.py), every store file opened `mode=ro` ([`b1/r8/out/r8_rows.txt`](b1/r8/out/r8_rows.txt), exit 0, problems 0).
R3's store is retired. The `QuadSource` comparison is therefore against the R4 store's
`extract_tensor_source_data` CSVs (`var/handover-06/extract/…`, handover 06's resumed store, whose
fingerprint equalled R3's `39809dca`).

**`QuadSourceIntegral`.** 864 rows; **864** carry `partition.handover` with `Tq` and `Tr`, each with
`z_init`, `z_switch` and `handover_continuity`. **0** rows have any metadata key containing `clamp`,
so none has `clamp_gaps_log1pz` or `max_clamp_gap_log1pz`. Each $k$'s continuity record is the same
in every row that carries it.

**The switch node.** For each $k$, `z_switch` is the first source-grid node below `z_init`, with no
source-grid node between them:

| $k$ / Mpc | $z_{\rm init}$ | $z_{\rm sw}$ |
|---|---|---|
| 1e5 | 1724800979.6868515 | 1712616453.6720948 |
| 3.1623e5 | 5454303327.635345 | 5420993175.39622 |
| 1e6 | 17248025285.92749 | 17159222629.609928 |

**`handover_continuity`**, recorded, not judged. Node 0 is $z_{\rm sw}$; nodes 1 and 2 are the
overlap nodes:

| $k$ / Mpc | node | $z$ | `dT_rel` | `dTprime_rel` |
|---|---|---|---|---|
| 1e5 | 0 | 1712616453.6720948 | -3.153510e-07 | 5.674049e-06 |
| 1e5 | 1 | 1673600269.6012518 | -5.545501e-06 | 2.165710e-05 |
| 1e5 | 2 | 1635472937.5650687 | -1.603827e-05 | 3.079759e-05 |
| 3.1623e5 | 0 | 5420993175.39622 | -2.354245e-07 | 4.915088e-06 |
| 3.1623e5 | 1 | 5297494147.272005 | -5.207091e-06 | 2.110052e-05 |
| 3.1623e5 | 2 | 5176808627.568508 | -1.552583e-05 | 3.059522e-05 |
| 1e6 | 0 | 17159222629.609928 | -1.672076e-07 | 4.151410e-06 |
| 1e6 | 1 | 16768307671.85978 | -4.878320e-06 | 2.053187e-05 |
| 1e6 | 2 | 16386298391.685555 | -1.501807e-05 | 3.037574e-05 |

By node, over $k$: node 0, $|\Delta T_{\rm rel}|$ 1.67e-07 to 3.15e-07 and $|\Delta T'_{\rm rel}|$
4.15e-06 to 5.67e-06; node 1, 4.88e-06 to 5.55e-06 and 2.05e-05 to 2.17e-05; node 2, 1.50e-05 to
1.60e-05 and 3.04e-05 to 3.08e-05. Every `dT_rel` is negative and every `dTprime_rel` positive, and
both grow monotonically away from the switch node.

**`QuadSource`**, R8 against R4:

| (q, r) / Mpc | R8 nodes (`z_samples`) | R4 nodes | added | removed | common nodes differing | R8 lowest $z$ | R4 lowest $z$ |
|---|---|---|---|---|---|---|---|
| (1e5, 1e5) | 478 (478) | 475 | 3 | 0 | 0 of 475 | 1635472937.5650687 | 1752542211.3413053 |
| (1e5, 3.1623e5) | 428 (428) | 425 | 3 | 0 | 0 of 425 | 5176808627.568508 | 5547371302.491309 |
| (1e5, 1e6) | 371 (371) | 368 | 3 | 0 | 0 of 368 | 16386298391.685555 | 17559250880.554787 |
| (3.1623e5, 3.1623e5) | 428 (428) | 425 | 3 | 0 | 0 of 425 | 5176808627.568508 | 5547371302.491309 |
| (3.1623e5, 1e6) | 371 (371) | 368 | 3 | 0 | 0 of 368 | 16386298391.685555 | 17559250880.554787 |
| (1e6, 1e6) | 371 (371) | 368 | 3 | 0 | 0 of 368 | 16386298391.685555 | 17559250880.554787 |

The top of every row is unchanged, at $z$ = 68787986538016.445. Every row gained exactly
`HANDOVER_OVERLAP_NODES` + 1 = 3 nodes, 18 in all. They are the larger-$k$ mode's $z_{\rm sw}$ and
its two overlap nodes. At every node R4 already had, all nine value columns are bit-identical. All
six rows are `validated`.

**Supplement: `QuadSourceIntegral` values against R4** ([`b1/r8/scripts/compare_qsi_extract.py`](b1/r8/scripts/compare_qsi_extract.py), reading
only the readers' CSVs; [`b1/r8/out/qsi_values_R8_vs_R4.txt`](b1/r8/out/qsi_values_R8_vs_R4.txt)). The readers draw their triples unseeded, so
only one triple has an `integral_data.csv` in both extracts: $(k,q,r) = (3.1623{\rm e}5, 1{\rm e}5,
3.1623{\rm e}5)$, 96 `z_response` rows.

| column | bit-identical | largest $\lvert\Delta\rvert$ / max$\lvert$value$\rvert$ over z_response | where |
|---|---|---|---|
| `total` | 33/96 | 1.82e-03 | $z_{\rm resp}$ = 5297494147.27 (an overlap node) |
| `numeric_quad` | 93/96 | 1.89e-03 | the same |
| `WKB_Levin` | 34/96 | 1.83e-02 | $z_{\rm resp}$ = 1329131612.30 |
| `analytic_rad` | 96/96 | 0 | |

The largest pointwise relative difference, 0.80, is in `total` and `WKB_Levin` at $z_{\rm resp}$ =
47846.4, where the integral is small. The 33 bit-identical `total`s are exactly the rows with
$z_{\rm resp} \geq$ 9210251401.997; every row with $z_{\rm resp} \leq$ 6985073578.495 differs.
That includes the one row above both $z_{\rm init}$ (5.45e9) and below 9.2e9, $z_{\rm resp}$ =
6.985e9, where `numeric_quad` moves by 3.0e-13 of max|value|. That is consistent with the source
spline's coefficients changing a little above the added nodes, but it was not traced. So B1 changed
the integrals, as intended, and the fingerprint (§5.3) does not show it. These are recorded, not
judged.

### 5.5 The readers (prompt §6.6)

`bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh var/datastores/handover-12-reference.sqlite
var/handover-12/extract`, with the lock held ([`b1/r8/out/readers.txt`](b1/r8/out/readers.txt)), 21:12:47 to about 21:15:02 UTC.
Overall exit **0**.

| reader | exit | wall | lookups | files written (R7; R3) | Traceback | store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 26 s | 116 | 531 (441; 468) | none | identical |
| `extract_GkWKB_data.py` | 0 | 44 s | 348 | 2436 (2436; 2436) | none | identical |
| `extract_GkSource_data.py` | 0 | 18 s | 10 | 160 (180; 156) | none | identical |
| `extract_TkWKB_data.py` | 0 | 15 s | 3 | 27 (27; 27) | none | identical |
| `extract_tensor_source_data.py` | 0 | 13 s | 6 | 27 (27; 27) | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 16 s | 24 | 76 (73; 70) | none | identical |

The lookups equal R3's and R7's. File counts differ only where a reader draws an unseeded subsample
(`[10-the-readers-draw-their-work-lists-unseeded]`). After each reader, 0 Ray processes of this
checkout's venv were left. No log has an authentication or token line.

**The store, before every read and after every read** ([`b1/r8/state/store_sha256_before_reads.txt`](b1/r8/state/store_sha256_before_reads.txt),
[`b1/r8/state/store_sha256_after_reads.txt`](b1/r8/state/store_sha256_after_reads.txt); `diff` empty):

```
531073a9940233ee260f7fb0bb9289fa1ceea15e0bbfcaa8c142646ebd5d3759  handover-12-reference.sqlite        36864
c8cb8d7c07936bf384474d68f470256fa9e6ae3fc50ab454ec28b99f7ba3c8d5  handover-12-reference-shard0000.sqlite  24096768
836df59f6067688fba9f37e4444fdd8722605e809279a244ce6a0957507d5b78  handover-12-reference-shard0001.sqlite  24207360
82119003fb3d2b834c1660b2899442697e884fc21282cde2d6766aed4df7ea74  handover-12-reference-shard0002.sqlite   1077248
fdf375f5c674d21056dac8cb6cd018fb83f7e4f03e4d37403c120a5326ff6d59  handover-12-reference-shard0003.sqlite  23633920
```

No `-journal`, `-wal` or `-shm` file at any point. "Before" was taken before
`compare_with_recorded.py`, so it covers §6.4, §6.5 and §6.6.

### 5.6 The lock released (prompt §6.7)

`ray-processes` listed nothing (exit 0). `release` exited 0. `status` then said
`lock: free (/Users/ds283/.raypoolkit/rehearsal-lock does not exist)` and
`Ray processes of /tmp/ray-sgk: none` (exit 0). The registry ends with 12 runs, all finished;
R8 is `done`.

### 5.7 Against prompt §7.3

- R8 ends `done`: **yes**.
- It differs from R3 in `QuadSource` and `QuadSourceIntegral` only: **yes**, in the sense that it
  differs in `QuadSource` alone, which is within that set. The dispatch expected `QuadSourceIntegral`
  to differ as well (§5.3; [log](../../prompts/handover/logs/12-remove-the-tk-gap.md) §6, deviation 7). **Accepted as met by the user's
  ruling of 2026-10-25** (§5.3).
- No row records a clamp gap: **yes**, 0 of 864.
- The readers exit 0: **yes**, and the store is byte-identical.
- The lock is released: **yes**.

### 5.8 The review

The orchestrator's review on the Mac Studio, against `orchestrator/prompt-12.md` §B.3, passed
**7 of 7, check 3 on the user's ruling** (§5.3): [`b1/r8/review.md`](b1/r8/review.md). It re-ran the
fingerprint and the comparison itself, read the rows with its own `mode=ro` reader (864 of 864
with `partition.handover`, none with a clamp key; the `QuadSource` node counts 371, 371, 371,
428, 428, 478 against R4's 368, 368, 368, 425, 425, 475), and found the store's hashes unchanged
by its own reads.

**R8's store was retired** after the records commit, on the user's decision, at 21:33 GMT on
2026-10-25 ([`b1/r8/out/retire_r8.txt`](b1/r8/out/retire_r8.txt)). The run
directory `handover-12-r8-reference-20261025T210551` on the Mac Studio keeps R8's fingerprint,
`1b332ffa…`, as the rehearsal reference from here on.
