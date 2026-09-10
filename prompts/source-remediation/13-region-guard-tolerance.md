# Prompt 13 — Compare the region-coverage guard in $\log(1+z)$

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Board issue:** §3 `[12-region-check-absolute-tolerance]` (opened by prompt 12, run-blocking)
**Record:** [`docs/source-remediation-verification.md`](../../docs/source-remediation-verification.md) §5.1
**Depends on:** 12 (its harness and datastore are this prompt's acceptance test)
**Recommended model:** Opus
**Files you may touch:** `ComputeTargets/QuadSourceIntegral.py` (`_check_region_covers` and its two
call sites **only**), `ComputeTargets/tests/test_quadsource_integral.py`,
`docs/source-remediation-verification.md` (an additive re-run note), plus the log and the status
board. **Do not** touch any other production file, any schema, or the two other board issues
prompt 12 opened.

---

## Character of this commit

A units fix in one guard, plus a regression test and a re-run. No schema change, no change to any
stored number, no new behaviour. It unblocks the `--quad-source-integral-queue` stage, which
currently cannot complete at production redshifts.

## The defect

`_check_region_covers` (`QuadSourceIntegral.py:561-570`) compares a sub-interval end against a
factor's region boundary **in $z$, with an absolute tolerance**:

```python
if z_min < region_min_z - DEFAULT_FLOAT_PRECISION:   # DEFAULT_FLOAT_PRECISION = 1e-7
```

but `build_partition` carries its breakpoints in $\log(1+z)$ and recovers $z$ for the caller by a
round trip (`:438-440`):

```python
z_hi = exp(log_hi) - 1.0
z_lo = exp(log_lo) - 1.0
```

At $z\approx5\times10^{14}$, $\log(1+z)\approx32.3$ carries ~16 significant digits, so the
recoverable $1+z$ has a granularity of ~0.02 — five orders of magnitude above the tolerance. **The
information is already gone in the log representation**; no amount of `expm1` care recovers it. The
Green's-function region always ends exactly at `z_response`, so the last sub-interval's bottom
always coincides with a region boundary and the sign of the round-trip movement alone decides
whether the guard fires.

Prompt 12 measured the consequence (verification document §5.1): **1372 of 3185 production work
items (43 %) raise** — 875 on the `Gk numeric` region, 497 on the `Gk WKB` region, over 28 distinct
response redshifts — and no other failure mode occurred. `RayWorkPool` propagates the first
failure, so the stage aborts and stores nothing. Both comparisons are affected: 29 of the live
run's 66 response redshifts round-trip downward past the tolerance and 25 round-trip upward.

This is audit B11's defect in the guard *next to* the one prompt 09 closed. It fails loudly, so no
wrong number is at risk.

## What to do

1. **Compare in $\log(1+z)$.** Change `_check_region_covers` to take the sub-interval ends as the
   $\log(1+z)$ values the caller already holds, convert the region bounds in the *safe* direction,
   and use `MIN_SUBINTERVAL_LOG_WIDTH` as the tolerance:

   - $z \to \log(1+z)$ costs ~1 ulp of the log and does not amplify; $\log(1+z) \to z$ is
     irreducibly lossy at large $z$. Only the first direction may appear in the guard's decision
     path.
   - Use the same expression shape the transfer-function branch of the same loop already uses
     (`:475`, `log(1.0 + region_min) > log_lo + MIN_SUBINTERVAL_LOG_WIDTH`). **This is not a new
     convention** — it is the one the `Tq`/`Tr` branch of the same function already follows, and
     the Green's-function guard is the only place that did not.

2. **Pass the log values, do not recompute them.** `log_hi` and `log_lo` are in scope at both call
   sites (`:462`, `:468`). Keep `z_hi`/`z_lo` as parameters as well, or thread them separately, so
   that the **error messages still print $z$** — they are for humans and 5 significant figures is
   the right resolution there. Do not make the messages print logs.

3. **Guard the domain.** `log(1.0 + z)` requires $z > -1$. The pipeline never goes below
   `DEFAULT_ZEND = 0.1` and $z_{\rm response}\ge0$, so this is unreachable today, but the region
   bounds arrive from the factor objects as plain floats. Add an assertion or a comment rather than
   trusting it silently.

4. **Leave the quadrature limits alone.** `z_hi`/`z_lo` are also passed on as integration limits.
   Do **not** change how those are computed: a 0.02 absolute error at $z=5\times10^{14}$ is
   $4\times10^{-17}$ relative on an integration limit, far below every tolerance in the chain. The
   round trip is only wrong for an *equality-like comparison*, which is what the guard does and the
   quadrature does not. Say so in the log so a later reader does not "finish the job".

5. **Do not widen the tolerance instead.** A relative tolerance in $z$
   (`DEFAULT_FLOAT_PRECISION * max(1.0, abs(z))`) is numerically near-equivalent and was
   considered; it is rejected because it keeps the lossy conversion and swallows its error, and
   because it is ~$10^9\times$ wider than needed, so the guard would accept a genuine overshoot of
   ~$5\times10^7$ in $z$ at the top of the grid. If you disagree after measuring, say so in the log
   and stop — do not ship the other fix silently.

## Verification

**Offline.**

- A regression test in `ComputeTargets/tests/test_quadsource_integral.py` driving `build_partition`
  with a region whose bottom is exactly `z_response` at $z\approx5\times10^{14}$: it must not
  raise. Use prompt 09's `_MinimalGk`/`_MinimalTk`/`_MinimalSource` stand-ins — they need no
  `bessel_phase` and run in milliseconds.
- **A test that the guard still fires.** A region genuinely short of the sub-interval by several
  grid steps must still raise, with the region and the sub-interval in the message. A fix that
  merely stops the guard complaining is not the fix.
- The pure-arithmetic reproduction from verification document §5.1
  (`exp(log(1+z)) - 1 < z - 1e-7` over the run's own grids) must now select nothing that the new
  comparison rejects.
- Full `ComputeTargets/tests` suite green (81 tests before this prompt's additions).

**Live — this is the acceptance test.** Prompt 12's datastore and harness are committed and
reusable (board §5 note 12):

- Re-run `docs/source-remediation-verification/run_quadsource_integrals.py` over the same work
  list. Expect **3185 of 3185 to complete** and zero failures in this guard. If a *different*
  failure mode appears in the 1372 items that never ran before, **stop and open a §3 issue** — it
  is a new finding, not this prompt's to fix.
- The **1813 items that already completed must be unchanged**: this commit touches no arithmetic.
  Compare `total` against `qsi_full.jsonl` and quote the worst relative difference; anything above
  round-off means the change was not confined to the guard.
- Then run `main.py`'s own `--quad-source-integral-queue` stage through
  `scoped_pipeline_run.py` on a **fresh** datastore path and confirm the stage completes and stores
  rows. That is what `[12-region-check-absolute-tolerance]` blocks, and completing it is what
  closes the issue.

Quote the numbers, not just pass/fail. If a Ray cluster or a writable datastore is unavailable,
stop and say so; do not substitute stand-ins for the live step.

## Log and commit

Log to `logs/13-region-guard-tolerance.md` using the template in `README.md` §5.1. Board: row 13,
the **Progress** line, and move `[12-region-check-absolute-tolerance]` from §3 to §4 with what
closed it. Add an additive note to `docs/source-remediation-verification.md` recording the re-run
and the new completion count — do not rewrite prompt 12's measurements; they were correct for the
tree they were taken on.

`[12-atol-too-loose-for-the-source-integral]` and `[12-handover-clamp-error-in-production]` stay
open. Neither is this prompt's work.

One commit.
