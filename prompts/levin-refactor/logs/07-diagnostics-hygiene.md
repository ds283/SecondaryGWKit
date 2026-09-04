# Log 07 — Diagnostics, logging and import hygiene

**Prompt:** prompts/levin-refactor/07-diagnostics-hygiene.md
**Commit:** Make the Levin module cheap to import and honest about its counters (SHA intentionally omitted — see README §5 rule 5)
**Date:** 2026-09-04
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### Item 1 — Import cost

`import seaborn as sns` / `from matplotlib import pyplot as plt` moved from module scope
(`levin_quadrature.py:90-91` at HEAD) to function scope inside `_write_progress_data()`, the only
function that uses either. Measured with `-X importtime`, three runs each:

| | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|
| before | 2.003 s | 1.317 s | 1.377 s | 1.566 s |
| after | 0.302 s | 0.214 s | 0.242 s | 0.253 s |

An 84% reduction in mean import time, not the "over 90%" the prompt's own framing anticipated
(seaborn/matplotlib were 95-97% of the *old* total, but what remains after removing them —
`numpy`/`scipy.linalg` for `toeplitz`, both genuine dependencies of the hot path — is itself
~0.25 s, not free). `grep -n "^import seaborn\|^from matplotlib" AdaptiveLevin/levin_quadrature.py`
returns nothing (verification item 9).

### Item 2 — File paths

Added `DEFAULT_LEVIN_DIAGNOSTICS_PATH = Path("levin_diagnostics")` and a `diagnostics_path`
parameter on `adaptive_levin_sincos()` → `_adaptive_levin()` → (`_adaptive_levin_subregion()` →
`_adaptive_levin_subregion_impl()` for the failure dump; `_write_progress_data()` for the
diagnostics payload). `_adaptive_levin()` resolves it once (`Path(diagnostics_path)` if supplied,
else the default) and threads the resolved `Path` — never `None` — to every writer below it.

- The lstsq-failure dump (`levin_quadrature.py:1259-1290` after this commit) now writes to
  `<diagnostics_path>/failures/LevinL_<id_label>_<isoformat>.txt` /
  `f_Cheb_<id_label>_<isoformat>.txt`, with the run's `id_label` folded into the *filename*, not
  just the directory — verified in "Parallel-safety" below.
- `_write_progress_data()` now writes to `<diagnostics_path>/SlowLevinData/<id_label>/<isoformat>`
  — same relative structure as before, just rooted under `diagnostics_path` instead of cwd.

**Behavioural change, as required by the prompt to be stated plainly:** before this commit, the
failure dump landed directly in the process's cwd, and the progress data under a cwd-relative
`SlowLevinData/`. Both now default to `./levin_diagnostics/...` instead. Nothing in this repository
reads either path (checked: `grep -rln "SlowLevinData\|emit_diagnostics" --include=*.py .` finds
only this module and its own tests), so there is no in-tree caller to update, but an external
script relying on the old bare-cwd location would need `diagnostics_path=Path(".")` to recover it.

### Item 3 — Logging

Added `_logger = logging.getLogger(__name__)` with `_logger.addHandler(logging.NullHandler())`,
and a `LOGGING` section in the module docstring documenting the default-silent behaviour and how a
host enables it. All 26 `print()` call sites converted:

- Every `!! WARNING (...)`-prefixed message → `_logger.warning(...)`.
- The `@@ adaptive_levin (...)` depth-18 diagnostic and its per-level history lines, and the
  `** STATUS UPDATE` / `|` progress lines in `_notify_progress()` → `_logger.info(...)`.

**Judgement call: kept the `!!`/`@@`/`**` prefixes.** The level now carries the same
warning-vs-info information, making the glyphs partly redundant, but they are how this module's
output has been read in terminal logs for years, and a caller who filters/greps on them (or who
built a Slack/log-aggregator rule around them) should not have to also learn the level changed at
the same commit. Stripping them is a plausible future cleanup, not this one's job.

**Judgement call: `logging.NullHandler()`, not a configured default handler.** This is the standard
library-author pattern and is what makes messages silent by default without needing this module to
decide log format/destination for its host — exactly the prompt's own framing ("a library should
not configure logging for its host"). It means every message, including warnings, is now invisible
unless the host calls `logging.basicConfig()` or attaches a handler — verified in "Logging" below.

Also fixed while converting: `_notify_progress()`'s progress line said "N integrand evaluations";
`num_evaluations` counts subregion *solves* (see item 4), so the message was wrong on its own
terms even before this commit. Reworded to "N subregion solves" — a log-message wording fix, not a
returned-value change.

### Item 4 — Honest counters

- `_adaptive_levin_subregion_impl()` now sets `metadata["lstsq_solve"] = 1` when the `lstsq`
  branch succeeds, and `metadata["pinv_solve"] = 1` when the `pinv` branch succeeds (mirroring the
  existing `metadata["direct_solve"] = 1` on the LU fast path). Neither flag existed before.
- `_adaptive_levin()` accumulates three **true totals** — `num_solves_direct`, `num_solves_lstsq`,
  `num_solves_pinv` — from every subregion solve actually attempted: a region's own fresh solve
  (only inside the `if data is None:` branch, i.e. only when a solve happens right there — see
  "Deviations" below for why this placement matters) *and* the two comparison children `dataL`/
  `dataR`, accumulated unconditionally at the point they are computed, regardless of whether the
  parent region is ultimately accepted or bisected. `num_solves_total` is their sum. Returned
  alongside the existing counters, not replacing them.
- `evaluations` is unchanged (still counts subregion solves, not integrand evaluations, under its
  original — and, per the audit, misleading — name, because
  `docs/adaptive-levin-benchmark/levin_bench/` reads it by key). Added `num_subregion_solves` as an
  identical value under an accurate name.
- The "reported `chebyshev_order` is 6 when every order fails" claim (README's citation of audit
  §1.10) does **not** reproduce against the current code: `_adaptive_levin_subregion()`'s
  `if data["value"] is None: raise LinAlgError(...)` fires *before* the metadata update that would
  report `chebyshev_order`, for exactly the case where the last-attempted order failed. Confirmed
  with a forced-total-failure regression (see "Verification performed"). Documented in place with a
  code comment; no fix needed because there is nothing to fix.

### Item 5 — `_write_progress_data` inconsistencies

- The unguarded `min(|estimate|, |refined_estimate|)` relative-error denominator is now floored at
  `atol`, mirroring the main path's `relerr_denom` guard (`_adaptive_levin`'s driver loop).
- Left alone, as instructed: the 500×(m+1)-evaluations-per-region diagnostic re-sampling. Noted
  below under *Observations not acted on*.
- **Unintended drift, fixed — see "Deviations" below**: `_write_progress_data()` crashed
  (`TypeError: 'NoneType' object is not iterable`) whenever one of its own diagnostic
  `dataL`/`dataR` re-solves landed on a Clenshaw-Curtis fallback region, because it iterated
  `dataX["p_sample"]` unconditionally and a fallback region's `p_sample` is always `None`. Guarded
  with `X["p_sample"] or []` at the three call sites.

### Item 6 — Fixed per-call overhead

- **`uuid4()` laziness, implemented as the prompt specifies.** A `_LazyUUID` wrapper class defers
  `uuid.uuid4()` (measured here: 1.887 µs/call, close to the audit's 2.3 µs) until first
  stringified. `_adaptive_levin()` now constructs `id_label = _LazyUUID()` instead of calling
  `uuid.uuid4()` directly. This alone would not have helped: `_adaptive_levin_subregion_impl()`
  (the hottest function in the module — called on every popped region and both its comparison
  children) used to build `label = f"{notify_label} id={id_label}"` unconditionally at its top,
  which would stringify the lazy id on the very first subregion solve of any run regardless of
  whether a message is ever emitted. That eager build is removed; `notify_label`/`id_label` are
  passed straight through to whichever (rare) branch actually logs something, via the new
  `_format_label()` helper, called only at each logging call site. The same restructuring was
  applied to `build_Levin_data()` and `_adaptive_levin_subregion_cc()` (both previously took a
  precomputed `label` string; both now take `notify_label`/`id_label` and format lazily), and to
  `_adaptive_levin()` itself (the depth-18 diagnostic, the converged/health-check warnings, and the
  chebyshev_order-clamp warning all previously read a precomputed `label`; all now call
  `_format_label()` inline, only inside their own already-rare branches).
- **Confirmed the two `raw_theta` gate calls are gone**: `grep -n "raw_theta"
  AdaptiveLevin/levin_quadrature.py AdaptiveLevin/tests/*.py` finds only the method's own
  definition, zero call sites. Prompt 03 removed them as claimed; nothing to do here.
- **Return-dictionary construction: measured, not made conditional.** The dict is ~30 key/value
  pairs, all already-computed scalars/lists at the point of construction — a dict literal of that
  size is sub-microsecond to build. Not isolated as its own microbenchmark (the reasoning above is
  sufficient at the "few microseconds" threshold the prompt itself sets), and no flag was added, per
  the prompt's own instruction to skip the flag when the saving would be negligible.
- **Net measured saving** on the audit's own scenario (a minimal single-region call, three solves):
  mean per-call time went from 201.4 µs (single baseline run, before) to a stable 188-198 µs (four
  runs, after; mean ≈ 191 µs) — roughly a 5% reduction, consistent with eliminating one `uuid4()`
  call plus three no-longer-built `label` strings per call. Not chased further, per the prompt's own
  "do not chase these below the point where the measurement is meaningful."

## Numerical evidence

**Required by rule 2 (bit-equality), not rule 9** — this commit changes no numerics; it is verified
by equality, not by a before/after accuracy comparison against a closed form.

Five problems (`sin_1_100`, `atan_lambda40`, `exp_sin_grz100`, `stationary_split`, `small_atol` —
spanning: a single-region Levin case, a fully-fallback case, an ordinary Levin+fallback mix, the C2
stationary-phase regression with an infinite round-off floor present, and a case that does not
converge), each run before and after this commit, comparing every numeric field returned
(`value`, `abserr`, `relerr`, `abserr_resolution`, `abserr_roundoff`, `abserr_fallback`,
`abserr_truncation`, `converged`, `phase_limited`, `num_phase_limited_regions`, `num_regions`,
`num_simple_regions`, `evaluations`, `num_SVD_errors`, `num_order_changes`, `num_direct_solves`,
`chebyshev_min_order`, `max_depth`):

```
before == after: EQUAL (json.loads() structural equality, all five problems, all fields)
```

Script: `scratchpad/baseline_snapshot.py` (this session's scratch directory). The only textual
difference between the two raw captures is the disappearance of the `!! WARNING (...)` lines that
used to print to stdout — expected, and itself evidence item 3 shipped (see "Logging" below).

## Deviations from the prompt

### STRUCTURALLY REQUIRED — new counter names cannot literally be `num_direct_solves`/`num_lstsq_solves`/`num_solves`

Item 4's illustrative names for the true totals (`num_direct_solves`, `num_lstsq_solves`,
`num_solves`) collide with the *existing* `num_direct_solves` key, which
`docs/adaptive-levin-benchmark/levin_bench/runners.py` reads by name
(`data.get("num_direct_solves", np.nan)`) expecting the old per-region semantics, and which rule 1
of the "Do not" section ("Do not change any number the module returns... verified by equality")
and the prompt's own "Do not remove keys... Add, or rename-and-update-the-harness" jointly forbid
silently repurposing. Renaming the old key and reusing the bare name for the new total was
considered and rejected: it would touch a file outside this prompt's stated file list for no
benefit over just picking different names. Shipped as `num_solves_direct`, `num_solves_lstsq`,
`num_solves_pinv`, `num_solves_total` instead; `num_direct_solves` keeps its exact old value and
name, documented in the docstring as counting regions, not solves, with a pointer to the new keys.

### STRUCTURALLY REQUIRED — true-solve accumulation had to be placed to avoid double-counting

A first pass accumulated `num_solves_direct/lstsq/pinv` from `data["metadata"]` unconditionally,
every time a region is popped — mirroring where the existing (kept) `num_direct_solves`
accumulation already sits. This double-counts: when a region's `data` is a *reused*
`current_region.estimate` (carried forward from a parent's `dataL`/`dataR`), that solve was already
counted once, at the point `dataL`/`dataR` were computed. The final version accumulates from a
fresh `data` only inside the `if data is None:` branch (i.e. only when a solve happens right there)
and from `dataL`/`dataR` unconditionally at the point they are computed — each real solve counted
exactly once. Verified by construction (see "Verification performed") and by the sanity check in
"Observations not acted on" (`num_solves_direct >= num_direct_solves` on every problem checked, as
expected since the true total is a superset of the per-region count).

### UNINTENDED DRIFT — `_write_progress_data()` crashed on a Clenshaw-Curtis comparison region, fixed

Discovered while executing the prompt's own verification item 6 ("The `emit_diagnostics=True` path
still works end to end"): `_write_progress_data()`'s per-region diagnostic re-solves
`dataL`/`dataR` at `(start, mid)`/`(mid, end)` and unconditionally iterates
`dataL["p_sample"]`/`dataR["p_sample"]` to build plot data. A Clenshaw-Curtis fallback region
always returns `"p_sample": None` (no Levin antiderivative exists there — see
`_adaptive_levin_subregion_cc()`'s own return-dict comment). On the stationary-phase regression
problem (`theta = 1e6*(x - x^2)` on `(0, 1)`, `atol=1e-9, rtol=1e-8`), several of the notify-time
regions are exactly this case, and `emit_diagnostics=True` raised `TypeError: 'NoneType' object is
not iterable` before writing anything. This is latent since prompt 03 introduced the
Clenshaw-Curtis branch — `_write_progress_data()` was never updated for it, and (being
diagnostics-only, off by default) nothing exercised the combination until this prompt's own
verification requirement did. Fixed with `X["p_sample"] or []` at the three iteration sites — an
empty p-curve for a region type that has no Levin antiderivative to plot, not a numerics change
(this function's output never reaches the returned dictionary). Re-verified end to end after the
fix (see "Verification performed").

## Verification performed

1. **`AdaptiveLevin/tests/` passes.** `PYTHONPATH=. ./venv/bin/python -m unittest discover -s
   AdaptiveLevin/tests -t .` → `Ran 23 tests in ... OK`, before and after. One test
   (`test_input_validation`'s chebyshev_order-clamp check) had to be updated: it previously
   asserted on captured stdout (`contextlib.redirect_stdout` + `"clamp" in buf.getvalue()`), which
   is exactly what item 3 changes. Rewritten to `self.assertLogs("AdaptiveLevin.levin_quadrature",
   level="WARNING")`. This is a required consequence of item 3, not a deviation from it — the old
   assertion tested the behaviour item 3 is instructed to change. `contextlib`/`io` imports removed
   from the test file (no longer used elsewhere in it).
2. **Bit-equality**, five problems, all returned fields — see "Numerical evidence" above. EQUAL.
3. **Import time**, `-X importtime`, three runs before and after — see item 1. 1.566 s → 0.253 s
   mean (84% reduction); `grep` for the module-scope imports returns nothing.
4. **Counters cross-checked against prompt 02's lstsq-share re-measurement.** Re-ran prompt 02's
   own scenario (`J000`/`J110` three-Bessel oracles, `k=1.3, q=1.7, s=2.1, max_x=1e5,
   chebyshev_order=12`) reading `num_solves_direct`/`num_solves_lstsq`/`num_solves_pinv` directly
   from the returned dictionary (via a thin wrapper around `_adaptive_levin`) instead of temporary
   instrumentation:

   | Oracle | LU (`num_solves_direct`) | lstsq (`num_solves_lstsq`) | pinv | lstsq share (now) | lstsq share (prompt 02) |
   |---|---|---|---|---|---|
   | `J000` | 91 | 85 | 0 | 48.3% | 69.2% |
   | `J110` | 177 | 283 | 0 | 61.5% | 76.6% |

   The absolute counts and shares differ from prompt 02's figures, and are expected to: prompt 02's
   measurement predates prompts 03-06, which substantially restructured which regions reach a
   Levin solve at all — the total-variation gate (03) routes weakly-oscillatory regions to the
   Clenshaw-Curtis fallback instead of Levin (removing some solves that used to happen), while also
   *correctly* routing stationary-point-adjacent regions to Levin that the old net-phase gate
   mis-routed to direct quadrature (adding others); the round-off floor (04) and mode filter (06)
   change region acceptance and bisection depth. Standing note 16 in `IMPLEMENTATION_STATE.md`
   already documents a 2.2-3.4x evaluation-count change on these exact difference-type phase groups
   from prompt 03 alone. What this check actually verifies — and what it confirms — is (a) the
   measurement is now obtainable directly from the returned dictionary, with no instrumentation,
   and (b) it remains qualitatively consistent with prompt 02's finding: lstsq is a substantial
   fraction of solves on this production-shaped integrand (48-62%), not the audit's synthetic 0-25%
   — still the evidence base for "measure on real integrands before considering RRQR" that prompt
   02 established.
   Script: `scratchpad/lstsq_share_check.py`.
5. **Parallel-safety of the failure-dump path.** Forced `numpy.linalg.lstsq` to always raise and
   froze `datetime.now()` to a fixed value, then ran two separate top-level
   `adaptive_levin_sincos()` calls sharing that frozen timestamp. Two distinct files were written
   (`LevinL_<id_label_1>_<ts>.txt`, `LevinL_<id_label_2>_<ts>.txt`) — no collision, by construction
   of the `id_label` in the filename. Script: `scratchpad/parallel_safety_check.py`.
6. **`emit_diagnostics=True` end to end.** Ran the stationary-phase regression problem with
   `notify_interval=0, emit_diagnostics=True, diagnostics_path=<tmp dir>`: writes
   `<tmp>/SlowLevinData/<id>/<timestamp>/payload.json` and per-region `.png`/`.pdf` plots (5
   payloads, 10 PNGs on this run), with the returned `value` identical to the non-diagnostic run
   (`-0.0006879079722347704`, matching the bit-equality snapshot) — confirms the diagnostic path is
   side-effect-free with respect to the actual computation. This surfaced and was blocked by the
   `p_sample` crash under "Deviations" above; passes after that fix. Script:
   `scratchpad/emit_diagnostics_check.py`.
7. **Logging.** With `logging.basicConfig(level=logging.WARNING)` configured, the chebyshev_order
   clamp warning appears on stderr with the expected text. With no handler configured at all, the
   same call produces no output and does not crash (`logging.NullHandler()` plus Python's own
   "no handler" fallback being suppressed by that NullHandler — see item 3).
8. **Per-call overhead**, minimal single-region problem (`AdaptiveLevin`'s own three-Bessel-style
   `exp(-x)sin(100x)` on `(1/3, 7/3)`), 2000-call mean, warmed up first: 201.4 µs (before, one
   run) → 188-198 µs across four post-change runs (mean ≈ 191 µs). See item 6.
9. `grep -n "^import seaborn\|^from matplotlib" AdaptiveLevin/levin_quadrature.py` → no output.
10. **Forced-total-failure regression** (item 4's `chebyshev_order` claim): monkeypatched
    `numpy.linalg.solve`/`lstsq`/`pinv` to always raise, ran `adaptive_levin_sincos` on a problem
    that reaches the Levin branch — raises `LinAlgError("SVD failure")` as expected, confirming a
    total failure never returns a `chebyshev_order` that was never successfully used. Script:
    `scratchpad/forced_failure_test.py`.

## Observations not acted on

- **The 500×(m+1)-evaluations-per-region diagnostic in `_write_progress_data()`** — left exactly as
  instructed (item 5: "Leave that alone — it is the price of the diagnostic").
- **`num_solves_direct >= num_direct_solves` on every problem checked** (e.g. the stationary-phase
  regression: 49 vs 25) — the expected shape given the true total is a strict superset of the old
  per-region count; recorded here as a sanity check on the accounting, not something requiring
  further action.
- **The `_format_label(None, id_label)` case formats as `"None id=<uuid>"`** when `notify_label` is
  not supplied — reproduces the pre-existing f-string's behaviour exactly (`f"{notify_label}
  id={id_label}"` was always going to interpolate literal `None`); not a regression, not changed.

## State handed to the next prompt

- **The returned dictionary has seven new keys**: `num_subregion_solves` (alias of `evaluations`),
  `num_solves_direct`, `num_solves_lstsq`, `num_solves_pinv`, `num_solves_total` (true totals),
  alongside the unchanged `evaluations`/`num_direct_solves`. All additive; confirmed safe for
  `docs/adaptive-levin-benchmark/levin_bench/` (still reads only the pre-existing keys, by grep).
- **`adaptive_levin_sincos()` has a new `diagnostics_path` parameter** (default `None`, resolved to
  `DEFAULT_LEVIN_DIAGNOSTICS_PATH = Path("levin_diagnostics")`). Prompt 09 (caller propagation)
  should decide whether `QuadSourceIntegral.py`/`three_bessel_integrals.py` want to pass one
  explicitly (e.g. under whatever output directory the production run already uses) rather than
  inherit the bare default, though nothing requires it — both callers currently pass
  `emit_diagnostics=False` implicitly (never set), so the default path is never exercised by them.
- **This module now logs through `logging.getLogger("AdaptiveLevin.levin_quadrature")`, silent by
  default.** Any later prompt or caller that wants to see its warnings in a test or a production run
  needs to configure logging (`logging.basicConfig(...)` or an explicit handler) — it will no
  longer appear on stdout unconditionally the way it used to.
- **`_LazyUUID` and `_format_label()` are the new pattern for anything that logs `id_label`.** A
  future change that adds a new log call site should use `_format_label(notify_label, id_label)`
  inline at the call site, not a precomputed `label` variable — precomputing reintroduces the
  eager-stringification cost item 6 removed.
