# Prompt 07 — Diagnostics, logging and import hygiene

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §1.10 (C10), §1.11 (C11), §3.4, §4.3; §5 recommendations 11 and 12
**Depends on:** prompt 06 for ordering only — it is independent of every numerical change, but it
rewrites `_write_progress_data`, whose subregion API prompts 02 and 03 both reshape.
**Files you may touch:** `AdaptiveLevin/levin_quadrature.py`,
`docs/adaptive-levin-benchmark/levin_bench/{runners,sweeps}.py` (only if you rename a returned key),
plus the log and the status board.

---

## Character of this commit

**No numerics.** Every value the module returns must be unchanged, bit for bit, on every problem.
This is an operational-hygiene commit: it makes the module cheap to import, safe to run in parallel,
and honest about what its counters count.

If any number moves, you have made a mistake.

---

## Item 1 — Import cost (C11, recommendation 12)

`seaborn` and `matplotlib.pyplot` are imported at module scope (`levin_quadrature.py:66-67`) for the
sole benefit of `_write_progress_data`, which only runs under `emit_diagnostics=True`.

**Measured at HEAD**, three runs of `python -X importtime -c "import AdaptiveLevin.levin_quadrature"`:

```
run 1:  seaborn 2.00 s (cumulative)  |  module total 2.10 s   -> 95%
run 2:  seaborn 1.20 s               |  module total 1.26 s   -> 95%
run 3:  seaborn 1.46 s               |  module total 1.51 s   -> 97%
```

This is **worse than the audit reports** (it says 1.13–1.36 s of 1.57 s). Under a Ray driver it is
paid per worker.

Move both imports inside `_write_progress_data`. Keep them at function scope rather than behind a
module-level lazy-import dance; the function is called at most once every three notification
intervals, so a repeated import lookup is free, and function-scope imports are the clearest
expression of "this is a diagnostics-only dependency".

**Verify with `-X importtime`**, before and after, and report both numbers.

## Item 2 — File paths (C11, recommendation 12)

Two cwd-relative write paths, both hazardous under Ray:

- **The `lstsq` failure dump** (`:722-729`) writes `LevinL_<isoformat>.txt` and
  `f_Cheb_<isoformat>.txt` into the **current working directory** at one-second timestamp
  resolution. Under parallel workers this collides, and it is unbounded in disk use and arbitrary in
  location.
- **`_write_progress_data`** (`:1297-1301`) writes to a cwd-relative `SlowLevinData/<id>/<isoformat>`.

Parameterise both. Suggested shape: an optional `diagnostics_path` argument on
`adaptive_levin_sincos`, defaulting to something explicit rather than to `Path.cwd()`, threaded down
to both writers. Include the `uuid4` `id_label` in the failure-dump filename so parallel workers
cannot collide even at the same timestamp.

**Do not silently change where diagnostics land for an existing user** without saying so in the log
and the commit message — someone may have a script that reads `SlowLevinData/`.

## Item 3 — Route messages through `logging` (C11, recommendation 12)

All warnings and progress notices go to `print`. Convert them to a module-level
`logging.getLogger(__name__)`, choosing levels deliberately:

- the two health checks at `:1197` and `:1213`, and prompt 01's `converged` warning — `WARNING`;
- the SVD-failure order step-down at `:591` and the `lstsq`/`pinv` failures at `:719`/`:741` —
  `WARNING` (they precede a raise or a degradation);
- `_notify_progress` and the depth-18 history dump at `:891-925` — `INFO`.

**This is a behavioural change for existing users**, who currently see these on stdout with no
logging configuration. A library should not configure logging for its host, so the messages will
*disappear* for anyone who has not set up a handler. Say so in the commit message and in the
docstring, and consider whether a `logging.NullHandler()` plus a one-line note in the module
docstring is enough guidance. This is a judgement call — record it.

Note the two existing message prefixes (`!! WARNING (…)` and `** STATUS UPDATE #n`) are part of how
the module is currently read in terminal output. Decide whether to keep them once the level carries
the same information, and say why.

## Item 4 — Honest counters (C10, recommendation 11)

`num_direct_solves`, `num_SVD_errors` and `num_order_changes` are accumulated only from
`data["metadata"]` (`:1005-1009`) — the *parent* solve. The two comparison solves `dataL`, `dataR`
are never inspected. Audit §1.10 measured the gap:

```
lorentz peak, k=12:  true solves 95, true LU 71, true lstsq 24  |  reported num_direct_solves 39
gauss peak,  k=12:   true solves 43, true LU 43, true lstsq  0  |  reported num_direct_solves 21
```

> **Correction to the audit, and it changes the shape of the fix.** §1.10 frames this as a counting
> bug. The source comment at `:977-980` states the per-region accounting is **deliberate**: "the
> metadata of the comparison regions `dataL`/`dataR` has never been accumulated, so reusing them
> preserves the existing accounting exactly" — it was written to explain why the child-estimate
> caching added in `b28d3c1` did not perturb the counters.
>
> Both readings are defensible. What is not defensible is a counter **named** `num_direct_solves`
> that counts regions. So the fix is *make the names match the semantics, and add true solve counts
> alongside* — not "fix an under-count".

Concretely:

- Add counters that accumulate from `dataL` and `dataR` as well, named for what they count
  (`num_direct_solves`, `num_lstsq_solves`, `num_solves`).
- Keep or rename the per-region ones explicitly (`num_regions_direct_solve`, …) so the distinction
  is visible in the returned dictionary rather than implied.
- **`evaluations` counts subregion solves, not integrand evaluations.** The prior review misread it
  as "linear solves ÷ 3". Add a correctly-named key. **Do not remove `evaluations`** —
  `docs/adaptive-levin-benchmark/levin_bench/{runners,sweeps}.py` read it (standing note 8). If you
  rename rather than add, update the harness in this commit and say so.
- **When every order down to the floor fails, the reported `chebyshev_order` is 6** — an order that
  was never used, because the step-down loop at `:566-585` decrements before re-testing. Report the
  lowest order actually attempted.

The immediate payoff: audit §4.4's claim that the LU fast path covers 75–100% of solves becomes
checkable from the returned dictionary, instead of needing external instrumentation. Prompt 02 was
asked to measure the post-complexification `lstsq` share by temporary instrumentation; **after this
commit that measurement should be reproducible from the API**. Re-run it and confirm the two agree —
that is a direct check that the new counters are right.

## Item 5 — `_write_progress_data` inconsistencies (C11)

Diagnostics-only, but inconsistent with the main path:

- It uses the **unguarded** `min(|estimate|, |refined_estimate|)` denominator at `:1393-1395`, the
  one that was fixed in the main path at `:1059` with a floor at `atol`. Apply the same guard.
- It re-solves three subregions per region and evaluates the integrand 500×(m+1) times per region.
  **Leave that alone** — it is the price of the diagnostic and it only runs under
  `emit_diagnostics=True`. Note it in the log as an observation.

## Item 6 — Fixed per-call overhead (§4.3)

Audit §4.3 measures **≈80 µs of fixed per-call driver overhead**: a minimal single-region call takes
248 µs for three solves accounting for ~168 µs. That is **32% of a small call**, and
`three_bessel_integrals.py` makes four calls per integral while `QuadSourceIntegral.py` makes up to
nine.

The itemised cheap fixes:

- **`uuid4()` is 2.3 µs.** Generate it lazily, only when a message is actually emitted.
- **The two `raw_theta` gate calls per region** — already removed by prompt 03. Confirm they are
  gone; if they are not, that is a prompt 03 regression and belongs in a §3 issue, not a silent fix
  here.
- **Build the diagnostics portion of the return dictionary only when asked.** Note the benchmark
  harness reads several of those keys, so "only when asked" must default to *on* for anything the
  harness reads, or the harness must be updated. Prefer keeping the dictionary complete and cheap
  over making it conditional — measure before adding a flag, and if the saving is under a few
  microseconds, do not add the flag at all and say so.
- The rest is dict/list/object construction, `time.time()`, sorting and the two health-check loops.
  **Do not chase these below the point where the measurement is meaningful.** Report what you
  actually saved.

---

## Do not

- Do not change any number the module returns. This commit is verified by *equality*.
- Do not rewrite `_write_progress_data`'s diagnostic strategy (item 5).
- Do not remove `emit_diagnostics` or the diagnostics path.
- Do not remove keys from the returned dictionary. Add, or rename-and-update-the-harness.
- Do not touch the numerics of prompts 02–06.

---

## Verification

1. `AdaptiveLevin/tests/` passes.
2. **Bit-equality (required).** Run at least five problems before and after and confirm `value`,
   `abserr` and every component are **exactly equal**. Not "close" — equal. Any difference is a bug
   in this commit.
3. **Import time**, `-X importtime`, before and after, three runs each. Expect the module total to
   drop by >90%.
4. **The counters are right.** Re-derive prompt 02's `lstsq`-share measurement from the returned
   dictionary alone and confirm it matches what prompt 02 measured by instrumentation. Report both.
5. **Parallel-safety of the dump path.** Show that two concurrent calls cannot collide — either by
   construction (the filename carries the `uuid4`) or by an actual concurrent run.
6. **The `emit_diagnostics=True` path still works end to end**, writing to the parameterised
   location, with the guarded denominator.
7. **Logging.** Confirm the messages appear with a handler configured and do not crash without one.
8. **Per-call overhead**, before and after, on a minimal single-region call. Report the actual
   saving, including if it is negligible.
9. `grep -n "^import seaborn\|^from matplotlib" AdaptiveLevin/levin_quadrature.py` returns nothing.

---

## Finish

1. Write `prompts/levin-refactor/logs/07-diagnostics-hygiene.md`. Under *Numerical evidence*, the
   bit-equality check (item 2) — that is what makes this commit safe to revert or keep independently.
   Judgement calls to record: the logging levels and whether the `!!`/`**` prefixes were kept; the
   default diagnostics path and whether it changed for existing users; whether counters were renamed
   or added alongside; and whether the return-dictionary construction was made conditional.
2. Update `IMPLEMENTATION_STATE.md`. **C10 and C11 close here**, and the rest of rec 11.
3. Commit in one commit. Suggested message:

```
Make the Levin module cheap to import and honest about its counters

None of this changes a returned value; all of it changes what the module
costs to load and what its diagnostics mean.

seaborn and matplotlib.pyplot were imported at module scope for the sole
benefit of _write_progress_data, which runs only under emit_diagnostics.
Measured with -X importtime, they accounted for 95-97% of the module's
1.3-2.1 s import -- paid per worker under a Ray driver. Both are now imported
inside the function that uses them.

The lstsq failure path wrote LevinL_<isoformat>.txt and f_Cheb_<isoformat>.txt
into the current working directory at one-second timestamp resolution, and
_write_progress_data wrote to a cwd-relative SlowLevinData/ tree. Both are
now parameterised, and the failure dump carries the run's uuid so parallel
workers cannot collide.

Warnings and progress notices go through logging rather than print.

The solve counters were accumulated only from the parent solve, never from
the two comparison solves, so num_direct_solves reported roughly a third of
the real figure: 39 against 95 true solves on one measured problem. The
existing behaviour was deliberate -- it counts regions, and was written that
way so that caching child estimates would not perturb it -- but a counter
named for solves that counts regions is a trap. Both quantities are now
reported under names that say which is which, and the returned dictionary
gains a true solve count so that the LU fast path's coverage is measurable
from the API rather than only by instrumentation.

Two smaller diagnostics fixes: when every Chebyshev order down to the floor
failed, the reported order was one step below the lowest order actually
tried; and _write_progress_data used the unguarded relative-error denominator
that the main path had already been fixed to floor at atol.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
