# Log 10 — Supply the transfer-function objects to `QuadSourceIntegral`, and settle `QuadSourcePolicy` (A4, wiring)

**Prompt:** prompts/source-remediation/10-qsi-main-plumbing.md
**Commit:** *(this commit)* — "Supply the transfer functions to the source integral stage"
(the SHA cannot be embedded in the commit that contains this file; see log 01 deviation 4)
**Model:** Claude Opus 5 (1M context)
**Date:** 2026-09-09
**Result:** COMPLETE WITH DEVIATIONS

The deviations are three files touched that the prompt's own §3 and §4 direct it to touch but its
"Files you may touch" header omits, one choice about where the dry test lives, and one formatting
side effect. None touches a formula, a sign, a normalisation or the phase-group table.

The `QuadSourcePolicy` decision is the user's, given to this prompt directly: *"Accept §6 as it
stands. The fix here is to make the Levin integrator more intelligent about choosing a fallback,
so that it routes the integrand [to] the Clenshaw–Curtis wholesale at the outset, rather than
finding that adaptive Levin needs many bisections which end up in Clenshaw–Curtis anyway."*
That is prompt 10 §3's **first** branch in substance — no threshold is reinstated — even though
the measured ratio (2.65–3.56×, above 3× on the three $b=0$ rows) is nominally the second
branch's trigger. See "The `QuadSourcePolicy` decision" below.

## What shipped

### `main.py`

- **New module-level `build_QuadSourceIntegral_payload(z_response, k, q, r, Gk_cache,
  source_cache, Tk_numeric_cache, Tk_WKB_cache, b_value, Bessel_0pt5_proxy, Bessel_2pt5_proxy)`**
  (`:261-334`, immediately after prompt 04's `closes_triangle` and before `run_pipeline`). It does
  the six cache lookups, the availability checks (printing the stage's existing
  `!! MISSING DATA WARNING` lines, verbatim for `GkPolicy` and `source`, in the same shape for the
  four new objects, then raising the existing `RuntimeError` message unchanged), and returns the
  nine-key payload dict:

  | key | object |
  |---|---|
  | `GkPolicy` | `Gk_cache[(k.store_id, z_response.store_id)]` |
  | `source` | `source_cache[(q.store_id, r.store_id)]` |
  | `b` | `b_value` |
  | `Bessel_0pt5`, `Bessel_2pt5` | the two `BesselPhaseProxy` objects |
  | `Tq_numeric`, `Tr_numeric` | `Tk_numeric_cache[q.store_id]`, `[r.store_id]` |
  | `Tq_WKB`, `Tr_WKB` | `Tk_WKB_cache[q.store_id]`, `[r.store_id]` |

  Module level (not nested in `run_pipeline`) for the same reason prompt 04 put `closes_triangle`
  there: it is the only way a test can reach it, since `main.py` cannot be imported. The three
  values that used to be closed over (`b_value`, the two proxies) are therefore parameters.

- **Two new lookup queues in `build_QuadSourceIntegral_batch`** (`:2496-2588`), after the existing
  `QuadSource` lookup and before the cache assembly. The distinct modes are collected as
  `missing_Tk_set = set(q for ... in missing_labels)` updated with the `r`s and frozen into a list
  (the `missing_set`/`missing_Tk` idiom of the QuadSource stage, `:921-923`), then two
  `RayWorkPool`s with `task_builder=lambda x: pool.object_get("TkNumericIntegration", **x)` and
  `...("TkWKBIntegration", **x)`, `store_results=True`, `create_batch_size=process_batch_size=20`
  — the QuadSource stage's `Tk_lookup_queue` (`:925-960`) copied, with the six production tags,
  `z_sample=None`, `z_init=None`, `atol`, `rtol`, `model=model_proxy`, plus `solver_labels=[]` for
  `TkWKBIntegration` (its factory reads `payload["solver_labels"]` unconditionally,
  `Datastore/SQL/ObjectFactories/TkWKBIntegration.py:180`).
  **Neither lookup sets `_do_not_populate`**, unlike the Tk-WKB stage's queries at `:670-763`,
  which only need `stop_T`/`stop_Tprime`/`stop_deltaz_subh`: `TkSourceFunctions` reads `.values` on
  both objects and both `.values` properties raise when the flag is set
  (`TkNumericIntegration.py:290-298`, `TkWKBIntegration.py:317-325`). A comment at the call site
  records this so it is not "optimised" later.

- **`Tk_numeric_cache` / `Tk_WKB_cache`** (`:2601-2607`), keyed by `wavenumber_exit_time.store_id`,
  alongside the existing `Gk_cache`/`source_cache`.

- **The work-item loop** (`:2612-2663`) now calls `build_QuadSourceIntegral_payload(...)` and puts
  the result in `"compute_payload"`; the 24 lines of inline lookup, checking and dict-building it
  replaces are gone. One clause is appended to the existing object-store `TODO` noting that the
  four transfer-function objects are the ones shared across the most work items, so they are what
  the object-store change would help most (per §2 item 4 — the TODO itself is untouched).

### `ComputeTargets/tests/test_main_plumbing.py` (new, 7 tests, 0.001 s)

`main.py` is a script — importing it builds a `ShardedPool`, contacts Ray and runs the pipeline —
so the test extracts `build_QuadSourceIntegral_payload` with `ast`, compiles that one
`FunctionDef` on its own and executes it in a namespace holding `datetime` plus stubs for the
names in its signature annotations (`load_main_py_functions`, reusable for any other top-level
`main.py` helper; it raises if the name is not module-level, so a future re-nesting fails the test
rather than silently skipping it). Stub ingredients carry only `store_id`, `available` and
`k.k_inv_Mpc`/`z`. Each cache is loaded with decoy entries under neighbouring keys, so a
transposed or mis-keyed lookup fails. Tests: payload keys **equal
`QuadSourceIntegral.REQUIRED_PAYLOAD_KEYS`**; each slot holds the object for the right
`store_id`s and `Tq_*`/`Tr_*` are not transposed; `b` and the two proxies pass through by
identity; $q=r$ gives the same object in both slots; each of the six ingredients in turn
unavailable → `RuntimeError` *and* a printed `!! MISSING DATA WARNING`; all-available prints
nothing; a cache miss is a `KeyError`, not a silent `None`.

### `ComputeTargets/QuadSourceIntegral.py` (comments and two deleted constants only)

- `LEVIN_MIN_2PI_CYCLES = 10` and `LEVIN_MIN_PHASE_DIFF` **deleted** (`:39-40`). The comment block
  above them is kept and rewritten: it still records what the retired Green's-function-only gate
  was and why it was wrong (audit QI-6), and now records prompt 10's decision and its reason (the
  cost is accepted; the improvement belongs to the Levin driver's choice of fallback, in
  `AdaptiveLevin/`, not to a second threshold here). Two stale cross-references to the deleted
  names fixed: the `CHEBYSHEV_ORDER` rationale (`:47`) now says "the then-current
  `LEVIN_MIN_PHASE_DIFF` gate", and `build_partition`'s `Levin_z_unused` comment (`:549`) points at
  the renamed comment block. No executable line changed; `pi` is still used elsewhere (`:1590`).

### `MetadataConcepts/QuadSourcePolicy.py` (docstring only)

A class docstring saying the object is persisted and threaded but **read by nothing**, and why
that is deliberate for each field: `numeric_policy` has no crossover to choose (the transfer
function's numeric→LG hand-over is a single determined redshift, README §2(a)); `Levin_threshold`
would duplicate `adaptive_levin_sincos`'s own total-variation gate with less information (it
cannot see $\theta_G\pm\theta_q\pm\theta_r$), and the prompt-08 §6 cost of relying on that gate
was measured and accepted. Ends with the reason it stays: schema and signature stability, and
somewhere for a future policy to live.

### `ComputeTargets/GkSourcePolicyData.py` (comment only)

A seven-line comment at `apply_GkSource_policy`'s `payload["Levin_z"] = ...` (`:50-56`) recording
that `Levin_z` is diagnostic only from this commit: still computed and persisted (dropping it is a
schema change, and the extract scripts plot it), but no consumer routes on it, and
`GkSourcePolicy.Levin_threshold` that sets it is likewise diagnostic. Prompt §4.

### Not touched

`ComputeTargets/QuadSourceIntegral.py`'s executable code (the payload contract needed no
adjustment: `REQUIRED_PAYLOAD_KEYS` and `compute()`'s `store_id` cross-checks are exactly what
`main.py` now satisfies), `Datastore/`, `AdaptiveLevin/`, `LiouvilleGreen/`,
`ComputeTargets/phase_groups.py`, `TkSourceFunctions.py`, `QuadSource.py`, `docs/`, every
`extract_*.py`.

## The `QuadSourcePolicy` decision

**Branch taken: no threshold** (prompt §3's first branch), **on the ratio 2.65–3.56×** — nominally
the second branch's trigger, and the reason the orchestrator stopped on prompt 08's board issue
`[08-levin-fallback-cost-ratio]`. The user's instruction was to accept prompt 08 §6 as measured
and to fix the *driver's* fallback choice instead: route a weakly oscillatory integrand to
Clenshaw–Curtis wholesale at the outset rather than discovering it after many bisections that each
end in Clenshaw–Curtis anyway.

That instruction is consistent with what prompt 08 measured, which is why it is implementable
here as "do nothing but delete the dead constants": at 1.00–1.44× the integrand evaluations for
2.65–3.56× the wall-clock, the excess is *not* wasted integrand work but the driver's per-region
overhead across the 20–28 Clenshaw–Curtis regions its own gate produced. A `Levin_threshold` in
`QuadSourceIntegral` would avoid that overhead by re-deciding, from a cycle count of $\theta_G$
alone, something the driver already decides correctly from the composed phase — reintroducing in
weaker form exactly the defect A4 names. The lever the user asked for is in
`AdaptiveLevin/levin_quadrature.py` (log 08 observation 3 identifies the same lever: a coarser
first bisection, or a plain adaptive quadrature when the whole call's phase span is below a few
$2\pi$), which README §1.1 and §5 item 8 place outside this campaign. Recorded as the new board
issue `[10-levin-wholesale-cc-fallback]`; `[08-levin-fallback-cost-ratio]` is closed into §4 with
this decision.

Consequences shipped: `QuadSourcePolicy` stays persisted, stays threaded through `run_pipeline`,
gains the docstring; the two dead constants are gone; nothing in the payload carries a policy
object for the source integral. `GkSourcePolicyData.Levin_z` and both `Levin_threshold` fields
survive as diagnostics.

## Deviations from the prompt

### 1. Three files edited that the "Files you may touch" header does not list — STRUCTURALLY REQUIRED

The header allows `main.py`, `ComputeTargets/QuadSourceIntegral.py` "only if the payload contract
needs a final adjustment", `MetadataConcepts/QuadSourcePolicy.py` (docstring/comment), the log and
the board. But §3 instructs "delete the dead `LEVIN_MIN_2PI_CYCLES`/`LEVIN_MIN_PHASE_DIFF`
constants in `QuadSourceIntegral.py`" — not a payload-contract adjustment — and §4 instructs "add
a comment at the persistence site" of `Levin_z`, which is
`ComputeTargets/GkSourcePolicyData.py:50-56`, a file the header does not mention at all. Both
instructions were followed, because they are the prompt's own specific directions and the header's
list is the general one. Both edits are comments plus the deletion of two unreferenced constants:
no executable behaviour changes in either file. `git diff HEAD~1 --stat` therefore shows
`GkSourcePolicyData.py` (+7, comment only) and `QuadSourceIntegral.py` (comments, −2 constants),
which an orchestrator checking the file list against the header should expect.

### 2. The dry test lives in `ComputeTargets/tests/`, not in a new root `tests/` — IMPLEMENTATION CHOICE

§5 offers either. Chosen: `ComputeTargets/tests/test_main_plumbing.py`. Reasons: (i) the campaign
runs one discover root per package (README §5 item 7) and there is no root `tests/` directory in
the tree, so a new one would need a discover command nothing else runs and the orchestrator's
check would silently pass with zero tests if it were forgotten; (ii) the contract under test is
this package's — the assertion is literally `set(payload.keys()) ==
set(QuadSourceIntegral.REQUIRED_PAYLOAD_KEYS)`, imported from
`ComputeTargets.QuadSourceIntegral`. Cost of the choice: a test in `ComputeTargets/tests` reaches
up two directories for `main.py` (`Path(__file__).parents[2]`), which is a little impolite. The
alternative (root `tests/`, run as `discover -s tests -t .`) was rejected on (i).

### 3. The extraction mechanism, and what it does not check — IMPLEMENTATION CHOICE

`main.py` cannot be imported (it opens a `ShardedPool` and a Ray connection at module scope), so
the test compiles the single `FunctionDef` extracted by `ast` and supplies `datetime` plus
annotation stubs as its globals. This checks the *shipped source text* of the function, which is
the point, but deliberately does not check that the enclosing stage passes the right caches: that
call site is inside the nested `build_QuadSourceIntegral_batch` and is covered only by the `ast`
parse and by prompt 12's live run. Alternative considered: move the helper into a new module
(say `config/` or a new `Pipeline/` package) so it could be imported directly. Rejected — it would
move pipeline wiring out of `main.py` for the sake of a test, and prompt 04 set the module-level
precedent inside `main.py`.

### 4. `black` also reformatted prompt 04's `closes_triangle` docstring — IMPLEMENTATION CHOICE (kept)

Running the repository's formatter over `main.py` moved the closing `"""` of `closes_triangle`'s
docstring onto its own line (two lines changed, no code). Prompt 04's commit had left `main.py`
not `black`-clean. Kept rather than reverted, so that `black --check main.py` passes after this
commit; it is in the diff but is not this prompt's work.

### 5. §2 item 4's payload-size measurement is per-value, not a whole serialised payload — IMPLEMENTATION CHOICE

"Measure the serialised size of one payload if you can do so cheaply" — a real payload needs a
populated datastore, so instead the *dominant* term was measured directly: `pickle.dumps` of 900
`TkWKBValue` objects (each with its `redshift`) is **174.7 kB** (194 B/value) and of 900
`TkNumericValue` objects **106.1 kB** (118 B/value); 900 bare `redshift` objects are 28.0 kB.
Arithmetic and conclusion under "Verification performed".

## Verification performed

All from the repository root with `PYTHONPATH=.` and `./venv/bin/python`.

- **`python -c "import ast; ast.parse(open('main.py').read())"` → `PARSE_OK`** (§5 item 1).
- **`python -m unittest discover -s ComputeTargets/tests -t .`: Ran 81 tests in 221.1 s — OK**
  (7 new, 74 pre-existing). The 74 include prompts 03, 05, 06, 07, 08 and 09's suites, so the
  comment-only edits to `QuadSourceIntegral.py` and `GkSourcePolicyData.py` and the deletion of the
  two constants break nothing. `python -m unittest ComputeTargets.tests.test_main_plumbing -v`:
  7 tests, all pass, 0.001 s.
- **`black --check`** on all five touched files: clean (after the reformat of deviation 4).
- **Dead constants**: `grep -rn "LEVIN_MIN_2PI_CYCLES\|LEVIN_MIN_PHASE_DIFF" --include="*.py"`
  over the tree returns **only** the historical comment in `QuadSourceIntegral.py:32` (which
  quotes the retired name deliberately). No `.py` reader existed before the deletion either; the
  remaining hits are in `docs/` and `prompts/`, which record the pre-fix state.
- **`QuadSourcePolicy` is still persisted and threaded**: `grep -n QuadSourcePolicy main.py`
  shows the import (`:42`), the two `run_pipeline` parameters (`:346-347`), the two
  `pool.object_get("QuadSourcePolicy", ...)` constructions (`:2843-2858`) and the two arguments
  at the call site (`:2901-2902`) — unchanged by this commit.
- **Payload size** (deviation 5): per mode the two objects together hold at most one full source
  grid of samples (the numeric object covers `z_exit_suph_e5` → hand-over, the WKB object
  hand-over → `z_end`), so at 118–194 B per value the pair costs ~0.15 MB per 1000 grid points, and
  a work item carries two modes: **~0.3 MB per work item per 1000 grid points**, i.e. roughly
  0.4–0.6 MB on the production grid (`main.py`'s own note at `:2699-2704` puts the grid at "about
  2k z-sample points" and the `GkSource` + `QuadSource` pair already in the payload at ~2 MB).
  So the added payload is **+15–30 %** in size — not the pressing part. What *is* more pressing is
  the sharing pattern: a `GkSource` is specific to one $(k,z_{\rm resp})$ and a `QuadSource` to one
  $(q,r)$, but each transfer-function pair is re-serialised into **every** work item containing its
  mode — on the post-filter grid, thousands of times per batch. That is exactly what the
  `:2618-2623` TODO proposes to fix, and this commit makes it a better investment than it was; it
  is not acted on here (§2 item 4 says not to).
- **Not verified, and needing prompt 12's live run:** that the two lookups actually find validated
  rows with the production tag set and return populated objects (no datastore was available here);
  that `compute()`'s four `store_id` cross-checks pass on real objects; that
  `TkSourceFunctions` accepts real `TkNumericIntegration`/`TkWKBIntegration` rows (log 08's own
  caveat); the wall-clock and object-store cost of the enlarged payload under
  `max_task_queue=1000`. **This commit has not been exercised end to end.** It makes the
  `--quad-source-integral-queue` stage *constructible* again, which is all a static change can do.

## Observations not acted on

1. **`_classify_Levin` can raise `KeyError`** (`ComputeTargets/GkSourcePolicyData.py:135-199`):
   `payload["Levin_z"]` is only set inside the `for z_source` loop, so if no sampled redshift
   exceeds `policy.Levin_threshold` the function returns `{"metadata": ...}` and
   `apply_GkSource_policy:58` (`Levin_data["Levin_z"]`) raises `KeyError: 'Levin_z'` rather than
   storing `None` — the early-return path at `:154` does supply `None`. Pre-existing, unrelated to
   this prompt, and now that `Levin_z` is diagnostic the fix is trivial
   (`payload.setdefault("Levin_z", None)`), but it is a behaviour change in a file this prompt may
   only comment on. Whether it is reachable in production is unknown (it needs a Green's function
   whose $|d\theta_G/d\log(1+z)|$ never exceeds 1.5 anywhere in its WKB range).
2. **`missing_Tk` is ordered by set iteration**, i.e. by object hash, so the lookup order varies
   between runs. Harmless (the same list indexes the results), and it is the existing idiom of the
   QuadSource stage, but it means the two new queues do not hit the shards in a reproducible order.
3. **The Tk lookups are not vectorised by shard key**, unlike the `GkSource`/`GkSourcePolicyData`/
   `QuadSource` lookups in the same function: they use plain `pool.object_get` inside a
   `RayWorkPool`, copying the QuadSource stage (`:925-960`). At ≤50 distinct modes per batch this
   is at most 100 gets against 750 work items, so it was not worth diverging from the proven
   pattern; if it ever shows up in a profile, `object_get_vectorized` with `{"k": k_exit}` as the
   shard key is the change.
4. **`docs/spec-code-audit/scripts/QI_03_measure.py` sections 2–4 remain broken** (log 08
   observation 2) and `extract_QuadSourceIntegral_data.py` still reads the vestigial `WKB_quad`
   columns (`[09-WKB_quad-columns-are-vestigial]`). Both are out of scope (README §5 item 8).
5. **`GkSourcePolicy.Levin_threshold` = 1.5/5.0 still generates two policy objects** and
   `GkSource_policy_5pt0`/`QuadSource_policy_1pt5`/`QuadSource_policy_5pt0` are built but never
   used by any stage (`GkSource_policy_1pt5` is the only one consumed). Now that both
   `Levin_threshold`s are diagnostic, the 5.0 variants distinguish nothing downstream except the
   stored `Levin_z`. Left alone: they are cheap, and removing them is a schema/label question.

## State handed to the next prompt

Prompts 11 and 12 program against the following.

1. **The pipeline is constructible end to end again.** `main.py`'s QuadSourceIntegral stage
   supplies all nine keys of `QuadSourceIntegral.REQUIRED_PAYLOAD_KEYS`;
   `[08-pipeline-non-runnable-until-10]` is closed (board §4). It has **not** been run:
   prompt 12's live run is the first execution of this path, and the first check that the two new
   lookups find their rows.
2. **`build_QuadSourceIntegral_payload(z_response, k, q, r, Gk_cache, source_cache,
   Tk_numeric_cache, Tk_WKB_cache, b_value, Bessel_0pt5_proxy, Bessel_2pt5_proxy)`** is a
   module-level function of `main.py`, next to prompt 04's `closes_triangle`. Anything that needs
   to build a payload — a diagnostic script, prompt 12's harness — should call it rather than
   rebuild the dict, and can reach it with
   `ComputeTargets/tests/test_main_plumbing.load_main_py_functions([...])`, which extracts a
   named top-level `main.py` function without importing the script.
3. **The two new lookups require validated, fully populated `TkNumericIntegration` and
   `TkWKBIntegration` rows** for every $q$ and $r$ in the batch, carrying the six production tags
   (`TkProductionTag`, `SourceZGridSizeTag`, `OutsideHorizonEfoldsTag`, `LargestSourceZTag`,
   `SmallestSourceZTag`, `SourceSamplesPerLog10ZTag`) and the run's `atol`/`rtol`. If prompt 12
   scopes its run to a few $k$ modes it must ensure the Tk-numeric **and** Tk-WKB stages have been
   completed for every mode appearing as $q$ or $r$, or the stage raises
   `QuadSourceIntegral builder: missing or incomplete source data ...` after a
   `!! MISSING DATA WARNING` naming the object and the mode.
4. **No `QuadSourcePolicy` reaches the integrator, by decision** (see "The `QuadSourcePolicy`
   decision"). Prompt 11 should not record a `Levin_threshold` anywhere as active policy; prompt 12
   should not expect one. `GkSourcePolicyData.Levin_z`, `GkSourcePolicy.Levin_threshold` and
   `QuadSourcePolicy.*` are all diagnostic.
5. **The board's error floors are unchanged by this commit** — nothing numerical happened here.
   The dominant term prompt 12 must expect is still the hand-over clamp,
   `[08-handover-clamp-error]` (~5e-3 of `total` per one-grid-step gap), then
   `[06-source-spline-residual-vs-handover]` (4.5e-4) and
   `[07-lg-derivative-truncation-at-handover]`; `total_abserr` is a quadrature bound only
   (`[09-abserr-is-a-quadrature-bound]`).
6. **Payload size grew by ~15–30 %** and the four Tk objects are re-serialised per work item
   (verification, item "Payload size"). If prompt 12's scoped run hits object-store pressure, the
   `max_task_queue=1000` (`main.py:2719`) and the `n=750` work-batch size (`:2706`) are the dials,
   and the `:2618-2623` TODO is the real fix.
7. **New board issue `[10-levin-wholesale-cc-fallback]`** carries the user's requested improvement
   to `AdaptiveLevin`'s fallback choice. It is not a defect in this campaign's output and does not
   block prompt 12; it is a performance item for a campaign that is allowed to edit
   `AdaptiveLevin/`.
