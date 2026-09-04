# Log 10 — Test matrix and campaign verification

**Prompt:** prompts/levin-refactor/10-test-matrix.md
**Commit:** Complete the Levin test matrix and verify the refactor campaign; SHA intentionally
omitted — see README §5 rule 5
**Date:** 2026-09-04
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### Part 1 — the test matrix (`AdaptiveLevin/tests/test_levin_quadrature.py`)

Suite grew from 28 tests (post-prompt-08 baseline) to **32 tests**, all passing in
**0.043-0.059s** — well inside the "a few seconds" budget; no test needed a slow-run marker.

1. `test_theta_mod_2pi_path_only` — the branch production (`ComputeTargets/QuadSourceIntegral.py`)
   actually uses: `theta` + `theta_mod_2pi` supplied, `theta_deriv` deliberately omitted.
2. `test_theta_deriv_path_only` — `theta` + `theta_deriv` supplied, `theta_mod_2pi` omitted (the
   shape `three_bessel_integrals.py`'s four phase groups use).
3. `test_reversed_span` — `x_span=(b,a)` with `b > a`; asserts exact negation and that both
   directions converge and match the closed form.
4. `test_generic_m_component_basis_two_decoupled_phases` — the `m != 2` row. A new module-level
   class, `_TwoIndependentSinCosBasis`, implements the `build_Levin_data`/`eval_basis`/
   `theta_abserr_at` contract for an `m = 4` basis (two independent `(sin, cos)` pairs at
   independent linear phases, block-diagonal `A^T`), driven directly through `_adaptive_levin()`
   since there is no public `m != 2` entry point. Has a real closed form (elementary). No design
   problem found constructing this from outside the module — see the log's "Deviations" section.
5. `test_GRZIntegral` — `_GRZIntegral` now returns its `data` dict, and `test_GRZIntegral` asserts
   explicitly which path each lambda (10, 100, 1000) takes, via `num_simple_regions`, re-derived
   against the current total-variation gate rather than assumed unchanged from the audit.
6. The four original tests (`test_SinIntegral`, `test_CosIntegral`, `test_SincIntegral`,
   `_GRZIntegral`) had their `|value - literal| < 1e-10` assertions replaced with a full-precision
   closed form and a threshold set from a measured true error with a stated margin (1e-12 for
   Sin/Cos, 1e-11 for Sinc/GRZ — 35x-1000x above the measured true error in each case, never at the
   measured value itself). All four, plus `test_stationary_phase_gate_uses_total_variation`,
   `test_theta_abserr_declared_endpoint_term`, `test_roundoff_floor_independent_of_theta_
   presentation`, and the three new tests above, now assert `abserr >= true_err` against the closed
   form.

Every C12 row is now either covered by a named new test or verified via an existing one — see
`docs/adaptive-levin-verification.md` §3 for the full table.

### Part 2 — campaign verification (`docs/adaptive-levin-verification.md`)

Independently re-ran, this session, against the actual `c8a1918` baseline (loaded via
`importlib.util` from `git show c8a1918:...`, as prompts 02/05/06 did) and the actual `HEAD`
(`d384031`): C1 (5 cases, including 3 beyond the audit's own two), C2 (stationary-phase oracle),
C3 (18 problem/tolerance combinations), C4 (omega-ladder, both phase modes), the `lstsq` share
(48.3% at order 12, 55.2% at the module's actual default order 16), all seven three-Bessel oracles
at `c8a1918` vs `HEAD`, and the end-to-end speedup — the last of which is the most important
finding in this document; see "Numerical evidence" below.

## Numerical evidence

Full tables are in `docs/adaptive-levin-verification.md`; the single most consequential number this
prompt produced, not present in any prior log:

**The audit's/prompt 02's 1.4-1.8x complexification speedup does not describe the net effect of the
finished campaign.** On the production three-Bessel `J000` oracle (`k,q,s=1.3,1.7,2.1, max_x=1e5,
chebyshev_order=12`), through the real `quad_JJJ` caller chain, evaluations rise from 308 (old) to
1042 (new) — a 3.4x increase, matching prompt 03's own log for this exact oracle — and wall time
rises from 0.804s to 2.961s: a **3.7x slowdown**, not a speedup, despite the complexified solve
being real and confirmed faster in isolation. This is the necessary cost of fixing C2 (the
total-variation gate) on exactly the phase-group family the audit itself flagged as most exposed,
not a new defect — `IMPLEMENTATION_STATE.md`'s `[03-fallback-cost-on-difference-groups]` already
names the mechanism — but no prior log computed this composite wall-clock ratio against the
pre-campaign baseline, and it directly contradicts a naive reading of the audit's own headline
number as still true of the finished campaign.

Second most important: **all seven three-Bessel oracles agree between `c8a1918` and `HEAD` to
round-off or near-round-off** (6.2e-15 to 3.0e-12 absolute, against relative errors to the analytic
oracle of 1e-7 to 3e-6 in both versions) — the campaign changed reporting and cost, not the delivered
numbers, on this integrand family.

## Deviations from the prompt

### IMPLEMENTATION CHOICE — pre-campaign core driven via a small return-dict shim for the caller-chain comparisons

`LiouvilleGreen/three_bessel_integrals.py`'s current `quad_JJJ`/`_Levin_3bessel` (post-prompt-09)
reads `data["converged"]` off the returned dict, a key that does not exist in the `c8a1918` module.
Rather than reverting `three_bessel_integrals.py` too (which would mean comparing against a stale
*caller*, not just a stale *core* — the point of §4.5/§4.7's comparison is to isolate the core), a
thin shim (`d.setdefault("converged", True); d.setdefault("phase_limited", False)`) wraps the old
core's return value before the unmodified current caller code runs. This adds nothing to what is
timed (the setdefault calls are after the `adaptive_levin_sincos` call returns) and does not affect
the returned `value`. Considered reverting `three_bessel_integrals.py` to its own `c8a1918` state
instead; rejected because that would also revert prompt 09's own error-propagation changes, muddying
exactly the "core only" isolation the comparison needs.

### STRUCTURALLY REQUIRED — `test_JJJ`/`test_YJJ` (with plotting) could not be confirmed to completion in this session

See `docs/adaptive-levin-verification.md` §5/§5.1 for the full account: launched in the background
with a 30-minute budget (prompt 09's own log estimated 10-30 minutes); the outcome is recorded there,
not fabricated here.

### None beyond the above

Part 1's four new tests and the `_GRZIntegral`/four-original-test tightening were implemented
exactly as the prompt describes; no other deviation.

## Verification performed

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` — **32 tests,
  OK, 0.043-0.059s**, run repeatedly through this session (including after `black` reformatted the
  test file — clean both before and after).
- `./venv/bin/black --check AdaptiveLevin/tests/test_levin_quadrature.py` — clean (one reformat
  applied and re-verified during this prompt's own work).
- Every number in `docs/adaptive-levin-verification.md` was actually executed this session via
  `PYTHONPATH=. ./venv/bin/python`, not transcribed from a prior log — see that document's own
  "Method used throughout" note.
- `git merge-base --is-ancestor <sha> HEAD` for all nine prompt commits plus the planning commit —
  all confirmed ancestors of `HEAD`; subject lines read directly from `git log`.

## Observations not acted on

- **The composite-speedup finding (§4.5) does not itself warrant a new fix** — it is evidence for a
  decision (raising `SIX_PI`) that `IMPLEMENTATION_STATE.md`'s existing
  `[03-fallback-cost-on-difference-groups]` issue already names as the next step, with prompt 03's
  own measurements as the starting evidence. This prompt adds a sharper number to that evidence base
  (a wall-clock ratio against the true pre-campaign baseline) but does not act on it — that would be
  production-code scope, forbidden here.
- **`abserr = +inf` on the C2 regression case at `HEAD`** (docs/adaptive-levin-verification.md
  §4.2) is `IMPLEMENTATION_STATE.md`'s existing `[04-roundoff-floor-can-be-infinite]` issue,
  observed here for the first time on the campaign's own headline regression case (not just a
  hand-built reproducer). No new issue opened; the existing one already covers it and already names
  the right next step (a future diagnostics consumer must not assume finiteness).

## State handed to the next prompt

There is no next prompt — this is the tenth and final prompt of the campaign.
`IMPLEMENTATION_STATE.md` is updated to mark the campaign complete (10/10), fill in the commit SHAs
for prompts 01-09 (previously placeholders for 09; 01-08 were already filled in by earlier prompts'
own board edits), and close out C12/recommendation 16's remaining rows. §3's open issues are left
exactly as accurate as this pass found them: `[03-fallback-cost-on-difference-groups]`,
`[04-roundoff-floor-can-be-infinite]`, `[04-theta-abserr-cc-branch-proxy]`,
`[05-zero-width-span-raises]`, `[09-abserr-does-not-bound-phase-spline-floor]`, and
`[09-quadsource-total-error-incomplete]` all remain genuinely open — nothing in this prompt's file
list could close any of them (each needs either a production-code change this prompt is forbidden
from making, or a `LiouvilleGreen/phase_spline.py` accuracy API that is out of scope for this whole
campaign, README §6).
