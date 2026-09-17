# Log 02a — Make the source grid buildable at every production anchor

**Prompt:** `prompts/tolerance-convergence/02a-source-grid-density-guard.md`
**Commit:** *(this prompt's own commit)* — "Guard the source grid density criterion off-node"
**Model:** Opus 5
**Date:** 2026-09-17
**Result:** COMPLETE WITH DEVIATIONS — the version-2 source grid now builds at **every** production
anchor on **every** production cosmology. QCD at its own anchor builds at **2034 samples** with
**53 guarded nodes**; both published grids are **bit-identical** (1996 / `4849552b` and 1778 /
`60a3205a`, **zero** guarded), so the guard is inert on every figure in the record.
`[01-v2-density-raises-at-the-qcd-production-anchor]` is **closed**;
`[01-density-criterion-imposed-outside-the-wkb-region]` is **narrowed** and handed its evidence.

---

## What shipped

Four files outside `prompts/` and `docs/`. **Two are production code** — `main.py` and
`CosmologyConcepts/wavenumber.py` — and two are the test tree.

### `main.py` — `source_grid_spacing_profile` only

1. **The guard.** The five-point stencil's four `dphi_du` calls are wrapped in
   `try: … except ValueError:`, which leaves `d4[i] = 0.0` and `usable[i] = False` and continues.
   `ValueError` and nothing wider: that is what `phase_residual_integrand` raises when
   $\omega^2 \le 0$, and a `TypeError` or `KeyError` there is a defect rather than a region
   boundary. A guarded node is then filled by the log-interpolation the criterion already applies
   to a declared crossing's neighbourhood — the machinery was there, and a node with no fourth
   derivative to equidistribute is exactly what it is for.
2. **The census.** `guarded` per `(k, sector)`, accumulated into `guarded_total` / `band_total`
   with a per-case `guarded_detail` list, surfaced two ways: an optional `report` dict argument,
   and a printed line when the count is non-zero.
3. **The refusal.** `guarded > SOURCE_GRID_MAX_GUARDED_FRACTION * band_size` raises `ValueError`
   naming the count, the band size, the realised fraction, $k$, the sector and the constant.
4. A docstring block recording the off-node mechanism, the bit-identity claim, and the boundary
   with item T7.

The return value is unchanged in every case, because every caller passes it straight to
`build_z_sample` as its `spacing`.

### `CosmologyConcepts/wavenumber.py` — one appended constant

`SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05`, with the measurement table and the margin argument in
its comment, appended after `SOURCE_GRID_SPLINE_EDGE_FACTOR`. **No existing line in that file was
touched** — see deviation 1, which records the scope amendment that authorises it.

### `ComputeTargets/tests/wkb_reference.py` — the production anchors

`PRODUCTION_Z_INIT_LAMBDACDM = 2.0636395964161516e16` and
`PRODUCTION_Z_INIT_QCD = 3.30033444460513e16`, with `PRODUCTION_Z_INIT` an alias of the first so
that all 28 existing call sites stay bit-identical. The old comment claimed one constant was "the
top of the universal source grid on **both** production cosmologies"; it is corrected, and says why
the distinction is load-bearing.

### `ComputeTargets/tests/test_source_grid.py` — seven new tests, 38 → 45

`TestTheCriterionBuildsAtEveryProductionAnchor`: the two anchors are distinct; the published grids
are bit-identical and guard nothing; QCD builds at its own anchor and is descending, distinct and
resolvable; the guarded nodes are counted and attributed per sector; the worst production band sits
far below the refusal; a misplaced band is refused; only `ValueError` is guarded. The existing 38
tests are untouched, and the four assertions the module docstring calls the ones that must not be
weakened are untouched.

### Board, index, log

`prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` — row 02a, item **T13**, the two `[01-…]`
entries updated; `docs/OPEN_ISSUES.md` — one row removed, one narrowed, count and date corrected.
Same commit (`CLAUDE.md`).

---

## The measurements

### 1. The `z_init` sensitivity scan — what closes `[01-v2-density-raises-at-the-qcd-production-anchor]`

`_solve_horizon_exit(QCD_Cosmology, k = 3e8/Mpc, -5)` returns **3.30033444460513e+16** on this
tree, reproducing the board's figure to all 15 digits. At production geometry — fifty wavenumbers,
both Liouville–Green sectors, `z_end = 0.1`, 100 samples per decade — the version-2 construction
**raised** there before this prompt and the version-1 construction built (1793 samples,
`80d459cf`).

Perturbing `z_init` relatively, **with the guard in place** so the scan can be run at all:

| relative offset | guarded | band-node evaluations | overall | worst single band | samples | digest |
|---|---|---|---|---|---|---|
| 0 | 53 | 136453 | 3.8841e-04 | 7.7160e-04 | 2034 | `21ffc126` |
| 1e-16 | 53 | 136453 | 3.8841e-04 | 7.7160e-04 | 2034 | `21ffc126` |
| 1e-14 | 53 | 136453 | 3.8841e-04 | 7.7160e-04 | 2032 | `aa7529ae` |
| 1e-12 | 53 | 136453 | 3.8841e-04 | 7.7160e-04 | 2016 | `bc07d160` |
| 1e-10 | 53 | 136453 | 3.8841e-04 | 7.7160e-04 | 2022 | `64dd7be6` |
| 1e-8 | 53 | 136453 | 3.8841e-04 | 7.7160e-04 | 2013 | `1385aae6` |
| 1e-6 | **0** | 136453 | 0 | 0 | 2025 | `a7e6babf` |
| 1e-4 | **0** | 136453 | 0 | 0 | 2026 | `e3562e91` |

Every offset from 1e-16 to 1e-8 guards — i.e. raised, before the guard — and the first that does
not is 1e-6. **The anchor's own solve is nowhere near that loose**, so no re-solve of it escapes the
band: this was never the accident of one float, and a QCD production run genuinely could not build
its source grid. That is the measurement the issue was waiting for, and it is why this was a blocker
rather than a curiosity.

The guarded count is a **stable 53** across the whole trip band, which is the stronger statement:
it is not a quantity that varies wildly with where the lattice falls, so 7.716e-04 is a
characteristic figure and not a lucky low.

### 2. The guarded-node census, per cosmology and per sector — the form item T7 can consume

All figures at production geometry, **source-grid generation: version 2** (README §5 rule 6), fifty
production wavenumbers, both sectors.

| cosmology | anchor | guarded | of band-node evaluations | fraction | Gk | Tk | cases | worst single band | samples | digest |
|---|---|---|---|---|---|---|---|---|---|---|
| QCD | LambdaCDM's, 2.0636395964161516e+16 | **0** | 136492 | 0 | 0 | 0 | 0 / 100 | 0 | **1996** | **`4849552b`** |
| LambdaCDM | own, 2.0636395964161516e+16 | **0** | 150932 | 0 | 0 | 0 | 0 / 100 | 0 | **1778** | **`60a3205a`** |
| QCD | own, 3.30033444460513e+16 | **53** | 136453 | 3.8841e-04 | **34** | **19** | **53 / 100** | **7.7160e-04** | **2034** | `21ffc126` |

**Every affected case guards exactly one node** — 53 cases, 53 nodes, and the per-case maximum is
therefore one node of the smallest band (1304). That shape matters for T7: it says the band reaches
*just* past where the expansion exists, at one lattice node, rather than running for a stretch
through a region of breakdown. The Gk/Tk split is 34/19, and the affected wavenumbers are the small
ones — the census begins at $k = 2.665\times10^5$/Mpc and the smallest three production wavenumbers
are unaffected in Gk because their bands do not reach the third declared crossing at all.

### 3. Bit-identity — the load-bearing result

Re-run **after** the refusal of §3.3 landed, not only after the guard:

- QCD at LambdaCDM's anchor, **version 2**: 1996 samples, digest **`4849552b`**, 0 guarded.
- LambdaCDM at its own anchor, **version 2**: 1778 samples, digest **`60a3205a`**, 0 guarded.

Both match the published values exactly. The argument is stronger than a digest comparison: the
guarded count is **zero** on both, so the `except` branch is never entered and the code executed is
the code that was executed before. No figure in the record moves.

### 4. Why `SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05`, and the margin

The prompt fixes the constant's existence and leaves its value to measurement. The value is scored
**per `(k, sector)` band**, which is the unit the log-interpolation fill operates over and the unit
the existing `usable.sum() < 2` check already uses.

- The **only** production case that guards anything reaches **7.716e-04** of a band.
- That figure is **stable** at every relative perturbation of `z_init` from 1e-16 to 1e-8 (§1).
- `0.05 / 7.716e-04` = **64.8x**. That is the stated margin.

The margin is deliberately large in one direction and small in the other. Large, because the
production figure must never be near the ceiling — a guard that fires on a healthy grid is worse
than no guard. Small, because at 5% of a band **95% of its nodes remain** to fit the
log-interpolation from, so anything that trips the ceiling is a misplaced band rather than a
boundary effect, which is exactly the distinction the constant exists to draw. A ceiling at, say,
0.5 would let a band half of which lies outside the region be absorbed silently, and that is the
failure mode §3.3 names: *a guard that can silently absorb an arbitrarily wrong band is a worse
defect than the raise, because the raise at least stops.*

---

## Deviations from the prompt

### 1. `CosmologyConcepts/wavenumber.py` is edited — **STRUCTURALLY REQUIRED, and authorised**

Prompt 02a §3.3 requires a new constant "beside the other `SOURCE_GRID_*` values". All fifteen of
those live in `CosmologyConcepts/wavenumber.py`, which the prompt's own "Do not touch" list and the
dispatch both forbid; `config/defaults.py` and `ComputeTargets/phase_residual.py` are likewise
forbidden, `main.py` module scope is outside D6's carve-out, and a constant defined inside
`source_grid_spacing_profile` would be the one `SOURCE_GRID_*` value no importer can see, which
defeats acceptance test 3's premise.

So the prompt's acceptance could not pass inside its file list. Per `CLAUDE.md` rule 4 the agent
**stopped and asked** rather than editing, and did not commit. **The user settled it on 2026-09-17:
option (a), the append-only definition in `CosmologyConcepts/wavenumber.py`, touching no existing
line in that file.** `build_z_sample`, `_solve_horizon_exit` and every existing `SOURCE_GRID_*`
value in that file remain out of scope and are untouched, as are `config/defaults.py`,
`ComputeTargets/phase_residual.py`, every `Datastore/` file and every function in `main.py` other
than `source_grid_spacing_profile`.

**This is a scope amendment, not drift.** It is recorded here as such because the file appeared on
a "do not touch" list in the prompt as written.

### 2. `main.py`'s import block is edited, outside `source_grid_spacing_profile` — **STRUCTURALLY REQUIRED**

The function cannot reference `SOURCE_GRID_MAX_GUARDED_FRACTION` unless `main.py` imports it. One
line changed: `from CosmologyConcepts.wavenumber import SOURCE_GRID_CONSTRUCTION_VERSION` became a
parenthesised two-name import. That import path — rather than adding the name to
`CosmologyConcepts/__init__.py`'s re-export list, where the other `SOURCE_GRID_*` names come from —
was chosen deliberately, because it keeps a third production file out of the diff and follows the
precedent already on that very line. No other statement in `main.py` is touched.

### 3. The guarded count is not surfaced at `main.py:963` — **STRUCTURALLY REQUIRED**

§3.2 says "`main.py:963` already prints a line about the criterion, and that is the natural place".
It is, and it is **out of scope**: that line is inside `run_pipeline`, and the carve-out is
`source_grid_spacing_profile` alone. Surfacing it there would also mean changing the function's
return signature, which every caller — production and the whole test tree — passes straight into
`build_z_sample` as its `spacing`.

Instead the count is surfaced from **inside** the function, two ways: a `print` in the same
`   @@ ` house style as the caller's line, emitted only when the count is non-zero, and an optional
`report` dict parameter that the tests assert against. The optional dict is what makes the census
testable at all — a print is not — and it leaves the return type alone, so no caller changes.

Printing only when non-zero keeps the two published paths' console output bit-identical too, which
is a small extra guarantee in the same direction as the digests.

### 4. The refusal test misplaces the band rather than widening the stencil — **IMPLEMENTATION CHOICE**

§4.3 says to build the synthetic case "by narrowing the band or widening the stencil in the test".
**Widening the stencil does not work**, and the reason is worth recording because it is a property
of the code rather than of the test: the stencil centre is clamped,
`u = min(max(u_profile[i], u_lo + 2 delta), u_hi - 2 delta)`, so the arms are confined to
`[u_lo, u_hi]` however large `delta` is. At `delta = 1.0` every node collapses onto `u_hi - 2` and
the four arms land inside the band, which is valid everywhere by construction — nothing is guarded
and nothing is refused.

The other option §4.3 offers was taken: the test lifts `source_grid_spacing_profile` a second time
with a stand-in `residual_node_range` that hands back the whole grid instead of the band the margin
test establishes. Most of those nodes lie far outside the Liouville–Green region, the guarded
fraction goes well past 0.05, and the refusal fires with the count in the message. This is also the
better test of the two, because a *misplaced band* is the failure the constant exists to catch and
is the live question behind item T7; a wide stencil is not.

The production constant was not edited, as §4.3 requires.

### 5. `_solve_horizon_exit` re-solves LambdaCDM's anchor 3.6e-13 away from `PRODUCTION_Z_INIT` — **observation, not acted on**

Running the solve directly returns `2.0636395964154036e+16` against the recorded
`2.0636395964161516e+16`. That is `[02a-grid-digest-not-reproducible]` visible in the data, it is
**T6**'s and prompt 05's, and this prompt is explicitly forbidden to touch the anchor solve. The new
`PRODUCTION_Z_INIT_LAMBDACDM` therefore carries the **recorded** digits, not a fresh solve, which is
what keeps the published digests reproducible. `PRODUCTION_Z_INIT_QCD` carries the value the board
records, which does reproduce exactly.

---

## Observations not acted on

1. **The crossing mask's ordering is still wrong, and reordering it is still not the fix.** The
   mask `usable &= |u_profile - u_break| > SOURCE_GRID_CROSSING_MASK_U` is applied *after* the
   stencil loop, so a node the mask would discard is still evaluated. That is what makes QCD's
   anchor differ from LambdaCDM's, and it is **not** the mechanism: the band is established
   node-wise and the stencil is evaluated off-node, so the `z = 3.61e15` case in
   `RESIDUAL_WKB_REGION_MARGIN`'s own comment — away from any declared crossing — would survive any
   reordering. Not acted on, per §2 of the prompt: reordering changes which nodes are evaluated,
   hence `usable`, hence the fit, hence the profile, so it could move a published grid and nothing
   has measured whether it does. No new issue: this is already inside
   `[01-v2-density-raises-at-the-qcd-production-anchor]`'s narrowing block and inside
   `[01-density-criterion-imposed-outside-the-wkb-region]`.

2. **The grid digest turns over on every `z_init` perturbation while the sample count barely
   moves.** §1's table shows six distinct digests across offsets of 1e-14 and smaller, with sample
   counts between 2013 and 2034. That is `[02a-grid-digest-not-reproducible]` in the data, already
   open and already assigned to **T6** / prompt 03 and prompt 05. Not acted on; explicitly not this
   prompt's (README §4, §7 D6).

3. **`wkb_reference.production_source_grid` still exists with its historic version-0 behaviour**,
   imported at 28 call sites in 20 files. This is the open half of
   `[00-three-production-grid-reproductions]` and it is prompt 01's narrowing, unchanged by this
   prompt. Not acted on: those twenty files are not in this prompt's scope.

4. **No `SOURCE_GRID_*` constant records the anchor it was measured at.** The new
   `SOURCE_GRID_MAX_GUARDED_FRACTION` does, in its comment table. The others do not, and after this
   prompt the anchor is known to matter. Not acted on — it is comment churn across a file this
   prompt may only append to, and prompt 06's provenance note is the right place for it.

---

## Verification performed

Every command run from the repository root at the commit this log describes.

| Prompt §4 acceptance | Result |
|---|---|
| 1. QCD at its own anchor builds, strictly descending, duplicate-free, no closer than `SOURCE_GRID_MIN_SEPARATION` | **Yes.** **2034 samples**, digest `21ffc126`, **53 guarded nodes** — reconciling with the probe's 2034 / 53 exactly. Asserted in `test_qcd_builds_at_its_own_anchor` |
| 2. Both published grids bit-identical, zero guarded | **Yes.** QCD at LambdaCDM's anchor **1996 / `4849552b`**, LambdaCDM at its own **1778 / `60a3205a`**, both **0 guarded**, re-run after the refusal landed. No digest moved |
| 3. The refusal fires on a synthetic case, with the count in the message | **Yes** — `test_a_band_the_expansion_is_not_defined_on_is_refused`, built by misplacing the band (deviation 4), asserting the message carries the count, the constant's name and the sector |
| 4. `test_source_grid.py`'s existing 38 tests still pass, unchanged | **Yes.** 38 → **45** (seven added); the existing 38 are untouched, as are the four assertions the module docstring protects |
| 5. Both suites green at the board's counts | **`ComputeTargets` 484 → 491, OK** (seven added by this prompt; 484 was prompt 01's figure, not `bc6dc97`'s 452, and it must not fall below 452). **`CosmologyModels` 39 → 39, OK.** The `[computetargets-suite-flake]` wall-clock assertion did not fire on this run |
| `black --check` | **Clean.** 4 files would be left unchanged |
| Production files in the diff | **Two** — `main.py` (one import line plus `source_grid_spacing_profile`) and `CosmologyConcepts/wavenumber.py` (one appended constant). Four files outside `prompts/` and `docs/`, counting the two test modules |

---

## State handed to the next prompt

**The anchors, by name.** Prompts 03, 04 and 06 must say which anchor a figure was taken at, the
same way README §2 (b) makes them say which grid generation:

- `ComputeTargets.tests.wkb_reference.PRODUCTION_Z_INIT_LAMBDACDM = 2.0636395964161516e16`
- `ComputeTargets.tests.wkb_reference.PRODUCTION_Z_INIT_QCD = 3.30033444460513e16`
- `PRODUCTION_Z_INIT` is an alias of the first. Every pre-existing call site means LambdaCDM's and
  is unchanged. **Every version-2 QCD figure previously in the record was taken at LambdaCDM's
  anchor**, including `test_source_grid.py`'s 1996 / `4849552b`, `RECONCILIATION.md` §2.7's "1,996
  samples on QCD" and `docs/qcd-background-verification.md` §10's density measurements. Those
  figures are not wrong; they are at the other anchor, and they now have a name to say so with.

**The QCD production grid, version 2, at QCD's own anchor:** 2034 samples, digest `21ffc126`, 53
guarded nodes. This is the grid prompts 03, 04 and 06 measure QCD on. **It is anchor-sensitive**:
`[02a-grid-digest-not-reproducible]` is open and the digest turns over under perturbations of
`z_init` far below the anchor solve's own 3.8e-7 convergence, so quote the anchor with the digest.

**For item T7 / `[01-density-criterion-imposed-outside-the-wkb-region]`, the evidence it was
waiting for** — a direct measure of how far `residual_node_range`'s band overreaches, and where:

| cosmology / anchor | guarded | band-node evaluations | fraction | Gk | Tk | cases affected | worst single band |
|---|---|---|---|---|---|---|---|
| QCD at LambdaCDM's | 0 | 136492 | 0 | 0 | 0 | 0 / 100 | 0 |
| LambdaCDM at its own | 0 | 150932 | 0 | 0 | 0 | 0 / 100 | 0 |
| **QCD at its own** | **53** | **136453** | **3.884e-04** | **34** | **19** | **53 / 100** | **7.716e-04** |

The shape is as important as the total: **every affected case guards exactly one node**, so the band
reaches just past the edge of the region at a single lattice node rather than running through a
stretch of breakdown. Prompt 04 decides whether the criterion should run over a horizon-based band
of its own; 02a does not pre-empt it, and the band is exactly as `residual_node_range` returns it.

**The new parameter, in README §1.2's five fields**, for `docs/TOLERANCE-PROVENANCE.md`:

- **Value and what it keys:** `SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05`
  (`CosmologyConcepts/wavenumber.py`). It keys **no object type** and is in no lookup key: it is an
  acceptance ceiling on the source-grid density criterion, consulted per `(k, sector)` band inside
  `main.source_grid_spacing_profile`. It cannot change any grid that builds — it can only convert a
  build into a refusal — so it does not invalidate any datastore row.
- **What measurement chose it:** the guarded-node census above, at production geometry (fifty
  wavenumbers, both Liouville–Green sectors, `z_end = 0.1`, 100 samples per decade, **source-grid
  generation version 2**) on `QCD_Cosmology` and `LambdaCDM` at both production anchors. The worst
  production band reaches 7.716e-04; 0.05 is **64.8x** above it. There is no converged reference and
  therefore no reference drift: the quantity is an exact integer count of nodes at which
  `phase_residual_integrand` raises, not an estimate.
- **What it competes against:** nothing numerical. The opposing consideration is structural — at 5%
  of a band, 95% of its nodes remain to fit the log-interpolation from, so the ceiling separates "a
  boundary effect the fill can absorb" from "a band the expansion is not defined on".
- **Cost:** zero. It is one comparison per `(k, sector)` case — 100 per grid build, one build per
  model per run — against the ~136,000 integrand evaluations the criterion already performs.
- **Campaign, prompt, log, date:** `prompts/tolerance-convergence`, prompt 02a,
  `logs/02a-source-grid-density-guard.md`, 2026-09-17.

**What remains explicitly not settled here**, and whose it is: the band itself (T7, prompt 04); the
anchor solve's tolerance (T6, prompt 03); one design tolerance applied to both the redshift row
match and the grid digest (prompt 05); the crossing mask's ordering (nobody yet — observation 1
above, inside the two `[01-…]` issues).
