# Log 13 — Compare the region-coverage guard in log(1+z)

**Prompt:** prompts/source-remediation/13-region-guard-tolerance.md
**Commit:** *(SHA not embedded, per prompt 01 log deviation 4)* — "Compare the source-integral region guards in log(1+z)"
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

**1. `ComputeTargets/QuadSourceIntegral.py:561-570` → `:563-610` — `_check_region_covers`
compares in log(1+z).** Signature was `(label, region, z_max, z_min)`; it is now
`(label, region, log_z_max, log_z_min, z_max, z_min)`. The two decisions were

```python
if z_max > region_max_z + DEFAULT_FLOAT_PRECISION:      # 1e-7, absolute, in z
if z_min < region_min_z - DEFAULT_FLOAT_PRECISION:
```

and are now

```python
if log_z_max > log(1.0 + region_max_z) + MIN_SUBINTERVAL_LOG_WIDTH:
if log_z_min < log(1.0 + region_min_z) - MIN_SUBINTERVAL_LOG_WIDTH:
```

— the same expression shape the `Tq`/`Tr` branch of the same loop already used (`:475`, now
`:477`). Only the safe conversion direction (z → log(1+z), ~1 ulp, non-amplifying) appears in the
decision path. `z_max`/`z_min` are still parameters and the two error messages are unchanged, so
they still print z at 5 significant figures. A domain guard rejects a region bound with z ≤ −1,
which has no log(1+z) (unreachable at `DEFAULT_ZEND = 0.1` with `z_response ≥ 0`, but the bounds
arrive from the factor objects as plain floats). A docstring records why the comparison must not
be made in z, and that the identical `z_hi`/`z_lo` are harmless as *quadrature limits* — so that a
later reader does not "finish the job" on `build_partition:439-440`.

**2. `ComputeTargets/QuadSourceIntegral.py:462, :468` — the two call sites** pass
`log_hi, log_lo` alongside `z_hi, z_lo`. The logs are the loop variables already in scope; nothing
is recomputed on the decision path.

**3. `ComputeTargets/QuadSourceIntegral.py:1715-1723` → `:1715-1741` — `numeric_quad_integral`'s
copy of the same guard,** which the prompt did not name. See deviation 1: the same units fix,
against the same `Gk_f.numeric_region`, using the `log_min_z`/`log_max_z` the function already
computed for the quadrature (moved above the guard). Both messages unchanged.

**4. `ComputeTargets/tests/test_quadsource_integral.py` — new class
`TestRegionCoverageTolerance`, four tests** (81 → 85 in `ComputeTargets/tests`).

## Deviations from the prompt

### 1. The same defect exists in a second guard, and it was fixed too — STRUCTURALLY REQUIRED

**What the prompt assumed.** Its "Files you may touch" line allows `_check_region_covers` "and its
two call sites **only**", and its §4 states that the round trip "is only wrong for an
*equality-like comparison*, which is what the guard does and **the quadrature does not**".

**What was actually there.** `numeric_quad_integral` (`:1715-1723` before this commit) re-checks
`Gk_f.numeric_region` against its `max_z`/`min_z` arguments — which are exactly the `z_hi`/`z_lo`
`build_partition` produced by `exp(log_z) - 1` — in z, with `DEFAULT_FLOAT_PRECISION`. It is
therefore the same equality-like comparison on the same region over the same sub-interval as the
guard the prompt names: a duplicate check, with the retired units. It is reached only on
`method == "quad"` sub-intervals, and `build_partition` raised first on every work item that would
have tripped it, which is why verification document §5.1 could record that "no other failure mode
occurred".

**What was measured.** With only the prompt's guard fixed, the live acceptance re-run
(§ Verification below, first re-run) went from 1813/3185 to **2414/3185 completing**, with **zero**
failures in `_check_region_covers` and **771 new failures in `numeric_quad_integral`**, over 22
distinct response redshifts from z = 1.9151e+09 to 4.8682e+14, e.g.

```
RuntimeError: compute_QuadSource_integral: attempting to evaluate numerical quadrature, but
min_z=4.8682e+14 is out-of-bounds for the region (6.8788e+14, 4.8682e+14) where a numerical
solution is available [domain=5.4628e+14, 4.8682e+14]
```

**What was done instead.** Stopped, reported the measurement and the two options to the user — fix
the sibling guard in this commit and exceed the prompt's file scope, or stop at the prompt's
boundary with `PARTIAL`, a new §3 issue and the stage still blocked — and **the user chose to
extend the fix**. The sibling guard now makes the same log(1+z) comparison with the same tolerance
and the same domain check. Nothing else in `numeric_quad_integral` changed: the quadrature limits
`log_min_z`/`log_max_z` are the values it already computed, merely computed before the guard rather
than after it.

**How to disagree.** The alternative was to treat "a second guard with the same defect" as the
"different failure mode" the prompt's Verification section says to report rather than fix, schedule
it as prompt 14, and accept that `[12-region-check-absolute-tolerance]` stays open. That keeps the
one-commit-per-defect-site property; it was rejected because the two guards check the same region
over the same sub-interval, the fix is the same two lines, and splitting them would ship a commit
whose stated purpose — unblocking `--quad-source-integral-queue` — is not achieved and cannot be
tested live.

### 2. The domain check raises rather than asserting — IMPLEMENTATION CHOICE

The prompt allows "an assertion or a comment". Both guards raise a `RuntimeError` naming the
region, in the wording of their neighbours, because `python -O` strips `assert` and because every
other data-shape failure in this module raises. The alternatives were a bare `assert` (stripped
under `-O`, and it would be the only one in the module) and a comment alone (silent if it is ever
violated). It costs two floating-point comparisons per sub-interval per work item.

## Verification performed

### Offline

- **`PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` — `Ran 85
  tests ... OK`** (81 before this prompt, 84 after the first three tests, 85 with the sibling
  guard's). Ran in 215 s.
- **`test_region_ending_exactly_at_z_response_does_not_raise`** — `build_partition` over
  `[z_response, z_source_max] = [4.8682e14, 6.8e14]` with the Green's-function numeric region
  ending exactly at `z_response`, which is what production hands it. The two ends were chosen so
  that **both** directions of the retired comparison are exercised: the round trip moves the top
  **+1.125** and the bottom **−1.188** in z, against a retired tolerance of 1e-7. It now yields one
  all-smooth sub-interval. On the pre-commit tree this test **errors**, with the production
  message: `sub-interval top z=6.8e+14 is out-of-bounds for the Gk numeric region (6.8e+14,
  4.8682e+14)`.
- **`test_a_region_short_of_the_subinterval_still_raises`** — the guard still fires. A region whose
  bottom is raised **0.1114 in log(1+z) (5 source-grid steps of 0.02228)** to z = 5.44188e+14
  raises, and the message names both the region and the sub-interval:
  `sub-interval bottom z=4.8682e+14 is out-of-bounds for the Gk numeric region (6.8e+14,
  5.4419e+14) [domain=6.8e+14, 4.8682e+14]`. The same is asserted at the top end.
- **`test_the_quadrature_helper_makes_the_same_comparison`** — the same pair for
  `numeric_quad_integral`: it integrates over the round-tripped limits against a region ending
  exactly at the bottom (value 1.59767e-45 on the constant stand-in), and still raises
  `min_z=4.8682e+14 is out-of-bounds for the region (6.8e+14, 5.4419e+14)` five grid steps short.
- **`test_round_trip_reproduction_selects_nothing_the_log_comparison_rejects`** — verification
  document §5.1's pure-arithmetic reproduction, on the grid the pipeline actually builds
  (`logspace` in z, as `wavenumber_exit_time.populate_z_sample` does, not in log(1+z)) over run A's
  own endpoints. Of **784** redshifts, `exp(log(1+z)) − 1 < z − 1e-7` selects **329 (42 %)** and the
  upward counterpart **313 (40 %)** — reproducing §5.1's 331/784 (42 %) to within the datastore
  float round trip that table went through. `_check_region_covers` accepts **all 784** with a region
  ending exactly at each.

### Live — the acceptance test

Prompt 12's run A datastore was still on disk and was reused read-only, so the work list is
identical (7 modes over 1e5…1e7 /Mpc, 784 source and 66 response redshifts, 3185 triangle-passing
work items). Ray was bootstrapped locally by the harness, `num_cpus=10`; the volume was 97 % full
with 36 GB free and no run failed for space.

**Re-run 1, prompt's guard only** (`run_quadsource_integrals.py`, same flags as §4.1):
**2414 ok, 771 failed**, every failure in `numeric_quad_integral` — deviation 1.

**Re-run 2, both guards** (same flags, `--z-stride 1 --include-blocked`):

```
** 7 source k, 7 response k, 784 source z, 66 response z
** response redshifts below z_source_max: 65; 28 trip the region guard, 37 do not
   cumulative: 3185 ok, 0 failed
```

**3185 of 3185 complete, zero failures, in any guard.** The 1372 newly completing items span the
same **28 distinct response redshifts from 6.9312e+07 to 4.8682e+14** the board reported, and their
sub-intervals reach six regimes: all-smooth 1113, G only 497, G+r 256, all three 415, r only 8,
q+r 96.

**Nothing stored changed.** Against `qsi_full.jsonl`, the 1813 items that already completed are
**bit-identical**: worst relative difference on `total` is **0.000e+00**, and likewise on
`numeric_quad`, `WKB_Levin`, `total_abserr` and `analytic_rad`. This commit touches no arithmetic
and the measurement says so exactly, not merely to round-off.

**`main.py`'s own stage — what `[12-region-check-absolute-tolerance]` blocked.**
`scoped_pipeline_run.py` on a **fresh** datastore path (run A's command verbatim except the
database and job name), so every stage, tag, tolerance and queue parameter is `main.py`'s:

```
   @@ QuadSourceIntegral triangle filter: kept 3185 of 12740 (k,q,r) triples (25.00%)

** CALCULATE QUADRATIC SOURCE INTEGRALS
   -- ALL WORK ITEMS COMPLETE in time 1m 19.7s
      Queue summary: 3185 lookup (avg 39.98/sec), 3185 compute (avg 39.98/sec), 3185 store (avg 39.98/sec)
```

**The stage completes and stores.** `select count(*) from QuadSourceIntegral` over the four shards
gives **3185** rows (910/910/455/910), every one with a non-null, finite `total` — against zero
rows on the pre-commit tree, where the stage aborted on its first batch. Whole pipeline 5 minutes,
79 MB. As sorted multisets the 3185 stored `total` values are **bit-identical** to the harness
run's, so `main.py`'s payload path and
`docs/source-remediation-verification/run_quadsource_integrals.py` agree exactly.

## Observations not acted on

1. **`evaluate_QuadSource_integral`'s all-smooth branch checks the QuadSource spline range with
   `DEFAULT_FLOAT_PRECISION` too** (`:504`, `log(1.0 + source_max) < log_hi - DEFAULT_FLOAT_PRECISION`).
   That comparison is *already* in log(1+z) — only its tolerance is spelled with the raw constant
   rather than `MIN_SUBINTERVAL_LOG_WIDTH`, and the two are equal by definition (`:118`). No
   behaviour is at stake; renaming it was left alone as cosmetic scope creep.
2. **The `Gk numeric` coverage check is now made twice** per all-smooth sub-interval — once in
   `build_partition`, once in `numeric_quad_integral`. The second is defensive: it is the only
   guard if `numeric_quad_integral` is ever called from outside `evaluate_QuadSource_integral`
   (it is not today). Deduplicating it would mean deciding which caller owns the invariant, which
   is a design question this commit should not settle.
3. **`analytic_integral`'s `eta_cut` comparisons** (`:1118`, `:1145` — unchanged by this commit) use `DEFAULT_FLOAT_PRECISION`
   as a *ratio* tolerance, not an absolute one in z, so they do not have this defect.
4. `[12-atol-too-loose-for-the-source-integral]` and `[12-handover-clamp-error-in-production]` are
   untouched, as the prompt requires. Now that the whole work list completes, both are measurable
   on 3185 rather than 1813 items.

## State handed to the next prompt

1. **`_check_region_covers(label, region, log_z_max, log_z_min, z_max, z_min)`** — logs decide,
   z prints. Any new caller must pass the logs it already holds, never `log(1 + exp(log_z) - 1)`
   recomputed from a value that came from somewhere else.
2. **`MIN_SUBINTERVAL_LOG_WIDTH` is now the tolerance of every region-coverage comparison in this
   module** (Green's function and both transfer functions, in `build_partition` and in
   `numeric_quad_integral`). Its justification comment at `:110-118` covers the value.
3. **The full production work list completes.** `qsi_full_after13b.jsonl` (3185 rows, all `ok`) is
   the new reference for anything that wants the whole distribution rather than the 1813-item
   subset prompt 12 measured; the campaign's five verification scripts all run unchanged.
4. **Prompt 12's run A datastore is reusable and unmodified.** The harness stores nothing.
