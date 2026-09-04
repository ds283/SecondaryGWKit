# Log 06 — Fix the `p_use` mode filter

**Prompt:** prompts/levin-refactor/06-mode-filter.md
**Commit:** Make the Levin mode filter self-consistent with the estimate it feeds (SHA intentionally omitted — see README §5 rule 5)
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

- **Problem (a) — gate on the endpoint values, not the collocation-point mean.**
  `AdaptiveLevin/levin_quadrature.py:1198-1220` (`_adaptive_levin_subregion_impl`): the
  `p_ratios` computation changed from `p_means = np.fabs(P).mean(axis=1)` (mean of `|P[j, :]|`
  over all collocation points) to `p_endpoint = np.fabs(P[:, -1]) + np.fabs(P[:, 0])` (the sum of
  the two endpoint magnitudes, `|p_i(a)| + |p_i(b)|`) — the quantity `lower_limit`/`upper_limit`
  actually consume. **Normalisation kept as ratio-to-maximum** (`pe / p_endpoint_max`), not
  switched to fraction-of-total: this preserves the meaning of the existing `rtol` threshold and
  of the `p_ratios` values recorded in `used_interval` / printed by `__str__` and the depth-18
  diagnostic (README's explicit "conservative option"). The `p_use` gate itself
  (`np.isfinite(r) and r > rtol`) and the divide-by-zero fallback (all-ones when every endpoint
  value is identically zero) are unchanged in structure, just fed the new quantity.
- **Problem (b) — the discarded endpoint contribution is now a reported error component.**
  New line `abserr_truncation = float(sum(pe for pe, used in zip(p_endpoint, p_use) if not
  used))` (`:1246-1248`), added to the Levin branch's returned dict
  (`"abserr_truncation": abserr_truncation`, `:1276`). Chosen bound: `|w0[i]|, |wk[i]| <= 1`
  (sin/cos), so summing the *undamped* endpoint magnitude over discarded components is a valid
  upper bound on what they would have contributed to `lower_limit`/`upper_limit`, at the cost of
  no extra evaluations.
- **Where the term went — a fourth `used_interval` component, added into `total_err`.**
  `used_interval.__init__` gained `abserr_truncation: Optional[float] = None`
  (`:371`, `:400`), a matching `abserr_truncation` property (`:529-540`), and `total_err`
  (`:469-497`) now returns `max(abserr, phase_err) + abserr_truncation` (falling back gracefully
  when either operand is `None`) rather than just the max. The Clenshaw-Curtis fallback branch
  reports `"abserr_truncation": None` (`:1379-1382`) — no Levin antiderivative exists there, so
  there is nothing for the rtol gate to discard, matching the existing
  `abserr_resolution`/`abserr_fallback` type-disambiguation pattern. Chose **add a fourth
  component** over folding into `abserr_roundoff`: the audit's own §3.4 point is that a caller
  wants to see *which* source is limiting them, and a truncation term conflated into the
  round-off floor would look like it responds to `chebyshev_order` when it actually responds to
  `rtol`.
  - `_adaptive_levin`'s driver loop passes `abserr_truncation=data.get("abserr_truncation")` into
    both `used_interval(...)` call sites (`:1673`, Levin branch `:1826`), reading it off `data`
    (the *parent's own* solve, not `dataL`/`dataR`) so the reported term matches the value that
    was actually contributed — this is the same object the driver already reuses for `estimate`
    when a region is accepted, and the one it threads through `_levin_interval.estimate` when a
    region is bisected instead, so no additional plumbing was needed to carry it across
    subdivision.
  - Aggregation (`:1897-1913`) gained `abserr_truncation_total`, summed the same way as the other
    three components; the returned dict gained `"abserr_truncation": float(abserr_truncation_total)`
    (`:2004`).
  - `adaptive_levin_sincos`'s `:return:` docstring (`:2324-2334`) updated to describe all four
    components.
- **Two regression tests added** to `AdaptiveLevin/tests/test_levin_quadrature.py`:
  `test_mode_filter_gates_on_endpoint_not_mean` (problem (a), using
  `_adaptive_levin_subregion_impl` directly — new import) and
  `test_mode_filter_truncation_visible_in_abserr` (problem (b), via the public API). Both are
  described under *Numerical evidence* below since they encode measurements made while verifying
  this prompt.
- **The open question is answered: the filter still earns its keep at `m = 2`.** See its own
  section below.

## Numerical evidence

All runs `PYTHONPATH=. ./venv/bin/python`. "BEFORE" loads `AdaptiveLevin/levin_quadrature.py` at
`HEAD` (prompt 05's state, `88f6ab3`) from a scratch copy via `importlib.util`, exactly as prompt
05's log did; "AFTER" is the working tree. Scratch scripts:
`/private/tmp/.../scratchpad/verify_06.py` and `verify_06_bessel.py`/`verify_06_bessel_sweep.py`
(not committed, outside the repository).

### Item 2 — the `rtol`-jump is now visible in `abserr`

The audit's §1.5b case (one region, order 16, mode ratios around `[1e-6, 1]`) doesn't come with
exact parameters, so it was reconstructed: `f = [exp(-x), 0]`, `theta(x) = 1e6 * x`,
`x_span = (1.0, 1.0003)`, `chebyshev_order=16`, `atol=1e-20` (so acceptance is governed by
`rtol`). This lands on `p_ratios = [1.00e-6, 1.0]` at order 16 (checked directly against
`_adaptive_levin_subregion_impl`), matching the audit's qualitative description.

| `rtol` | BEFORE value | BEFORE `abserr` | AFTER value | AFTER `abserr` | AFTER `abserr_truncation` |
|---|---|---|---|---|---|
| 1e-2 | 4.809096874481584e-07 | 2.21e-16 | 4.809096874481584e-07 | **7.36e-13** | 7.357e-13 |
| 1e-4 | 4.809096874481584e-07 | 2.21e-16 | 4.809096874481584e-07 | **7.36e-13** | 7.357e-13 |
| 1e-6 | 4.809099003166193e-07 | 5.05e-15 | 4.809099003166193e-07 | 5.05e-15 | 0.0 |
| 1e-8 | 4.809099003166193e-07 | 2.07e-16 | 4.809099003166193e-07 | 2.07e-16 | 0.0 |

Value is **identical** BEFORE/AFTER at every `rtol` (problem (a) does not fire here — mode 2's
ratio, 1.00e-6, is far from the `rtol` boundary in either normalisation). The jump itself:
`|v(rtol=1e-2) - v(rtol=1e-8)| = 2.128685e-13` (relative `4.43e-7`, matching the mode ratio's
order of magnitude, as expected). **BEFORE:** this jump is invisible — `abserr` at `rtol=1e-2` is
2.21e-16, eight orders of magnitude too small to bound a 2.13e-13 jump. **AFTER:** `abserr` at
`rtol=1e-2` is 7.36e-13, entirely `abserr_truncation`, and it *does* bound the jump
(7.36e-13 ≥ 2.13e-13). This is `test_mode_filter_truncation_visible_in_abserr`.

### Item 3 — five-problem A/B, `atol=1e-12, rtol=1e-10, chebyshev_order=16`

Same five problems as prompt 05's log (README §5.2 closed forms): `sin_1_100`, `grz_100`,
`grz_1000`, `expsin_30000`, `expsin_1e6`.

| Problem | BEFORE value | AFTER value | `\|before-after\|` | AFTER `abserr_truncation` | regions |
|---|---|---|---|---|---|
| sin_1_100 | −3.220165664195495e-01 | −3.220165664195495e-01 | 0.0 | 6.13e-14 | 1 |
| grz_100 | −1.967176421757699e-15 | −1.967176421757699e-15 | 0.0 | 2.00e-15 | 4 |
| grz_1000 | −1.607083230269651e-16 | **4.584212488070839e-13** | **4.586e-13** | 0.0 | 2 |
| expsin_30000 | −2.457570673857313e-05 | −2.457570673857313e-05 | 0.0 | 0.0 | 1 |
| expsin_1e6 | −3.347268410464613e-07 | −3.347268410464613e-07 | 0.0 | 0.0 | 1 |

Four of five problems are bit-identical before/after, as the prompt's framing expects. **`grz_1000`
is not**, and this is a genuine, found, *live* instance of problem (a)'s mechanism — see the next
subsection; it is not a regression, and it is not covered by `abserr_truncation` because the mode
in question is *retained*, not discarded, by the corrected gate.

### A field case where problem (a) changes a result (contradicting the audit's "no case found")

The audit states (README §2.2 (a), problem statement) that a 400-problem randomised sweep found no
case where the mean-vs-endpoint distinction changed a result. `grz_1000` at `x_span=(-1,0)`
(the driver's own comparison sub-region for the full `(-1,1)` problem, `rtol=1e-10`,
`chebyshev_order=16`) is such a case, found while producing the table above, not by construction:

```
BEFORE (mean-based):     p_ratios = [1.0, 2.217e-11]   -> mode 2 DISCARDED (ratio < rtol)
AFTER  (endpoint-based): p_ratios = [1.0, 1.384e-10]   -> mode 2 KEPT      (ratio > rtol)
```

Both ratios are within a factor of ~6 of `rtol=1e-10` itself — an inherently threshold-sensitive
case, exactly the kind problem (a) says was possible in principle ("a mode with a small mean and a
large endpoint value could be discarded wrongly"). The two normalisations disagree about which
side of the threshold this mode falls on, and the endpoint-based one is the one that is actually
consistent with what `lower_limit`/`upper_limit` consume. This is `test_mode_filter_gates_on_
endpoint_not_mean`. Consequence for the aggregate value: the closed form for `grz_1000` is
`(2/1000)*sin(250*pi) = 0` **exactly** (not merely small); both the BEFORE value
(−1.607e-16) and the AFTER value (4.584e-13) are round-off/truncation noise around that exact
zero, and AFTER's reported `abserr` (4.586e-13) still comfortably bounds AFTER's distance from the
true value, well inside the requested `atol=1e-12`. So this is a real change in the returned
number, correctly reported, not a correctness regression — but it does mean the "no case found"
framing does not survive contact with this campaign's own standard problem set, and the log says so
plainly rather than repeating the audit's claim.

### Item 4 — filter-disabled experiment (problem-(a)-corrected gate, threshold removed entirely)

Filter-disabled = `_adaptive_levin_subregion_impl` with `p_use = [np.isfinite(r) for r in
p_ratios]` (keeps prompt 01's non-finite rejection, drops only the `rtol` threshold), patched in
via `inspect.getsource` + `exec` against the *working-tree* module (i.e. this is filter-ON vs
filter-OFF within the AFTER code, not a BEFORE/AFTER comparison).

| Problem | filter-ON value | filter-OFF value | `\|on-off\|` |
|---|---|---|---|
| sin_1_100 | −3.220165664195495e-01 | −3.220165664195637e-01 | **1.421e-14** |
| grz_100 | −1.967176421757699e-15 | −1.976717400875572e-15 | 9.54e-18 |
| grz_1000 | 4.584212488070839e-13 | 4.584212488070839e-13 | 0.0 |
| expsin_30000 | −2.457570673857313e-05 | −2.457570673857313e-05 | 0.0 |
| expsin_1e6 | −3.347268410464613e-07 | −3.347268410464613e-07 | 0.0 |
| sec1.5b-analog (`rtol=1e-2`) | 4.809096874481584e-07 | 4.809099003166193e-07 | 2.129e-13 |

`grz_1000` is identical filter-ON/OFF (unlike the BEFORE/AFTER row in item 3): once problem (a)'s
fix is applied, the borderline mode is *already* retained by the enabled filter, so disabling the
filter changes nothing further for this problem.

**A case where turning the filter off degrades a result**, closing the open question's first
branch: `sin_1_100` against its closed form `cos(1) - cos(100) = -0.3220165664195441`:

```
filter-ON  error vs closed form: 5.385e-15
filter-OFF error vs closed form: 1.960e-14   (3.64x larger)
```

Filter-ON is measurably *closer* to the true value here — the discarded mode (endpoint ratio
below `rtol=1e-10`, `abserr_truncation = 6.13e-14`) is genuine noise, and including it moves the
result away from the truth, not toward it. This is small in absolute terms (both errors are far
below any tolerance used in this campaign), but it is a real, reproducible instance of exactly what
the filter's original reasoning (commit `af85ef2`) claims: a mode below the noise floor pollutes
the result more than it informs it. `grz_100` shows the same direction (filter-ON error
1.957e-15 vs filter-OFF 1.967e-15 against a closed form that is itself exactly zero, so this
comparison is dominated by round-off and is reported for completeness, not as independent
evidence).

### Item 5 — three-Bessel oracles, before/after and filter-on/off

Following prompt 05's log precedent (full 7-oracle × `max_x=1e12` matrix not attempted here on
cost grounds — it is what makes `test_3bessel_analytic.py` slow, and this class of change is
orthogonal to it):

**Before/after** (`J000`, `k,q,s=1.3,1.7,2.1`, `max_x=1e6`, `phase_atol=1e-25, phase_rtol=5e-14,
quad_atol=1e-14, quad_rtol=1e-10`, `adaptive_levin_sincos` monkeypatched between modules as in
prompt 05's log, shared `bessel_phase` splines):

```
analytic J000 = 0.16923037349654133
BEFORE (prompt 05): value=1.692305e-01  abserr_vs_analytic=1.017e-07  relerr=6.010e-07
AFTER  (prompt 06): value=1.692305e-01  abserr_vs_analytic=1.017e-07  relerr=6.010e-07
```

Identical to the digits printed, matching prompt 05's own J000 baseline exactly.

**Filter-on/off sweep**, extended beyond J000 to `J231` (the difference-type phase group flagged
in `IMPLEMENTATION_STATE.md` §3 `[03-fallback-cost-on-difference-groups]` as most C2/C5-exposed),
3 triangle configurations × 2 `max_x` values × 2 oracles = 12 combinations, `quad_atol=1e-14,
quad_rtol=1e-10`:

```
max |on-off| across all 12 combinations = 0.000e+00
max relative difference                 = 0.000e+00
```

**Bit-identical in every case.** A direct check of one phase group's `p_ratios` with the
production `f2 ≡ 0` shape (`f = [gaussian bump, 0]`, `theta(x) = 5000*x`, span `(1, 9)`, order 12)
gives `p_ratios = [1.0, 0.0114]` — mode 2's endpoint ratio is ~1.1%, nowhere near
`rtol=1e-10`. This confirms the standing note's claim ("`f2 ≡ 0` does not make `p2 ≡ 0`... the
filter is live on the production path" — in the sense that the gate is evaluated and coupling is
real) while showing that at the coupling strengths and `rtol` values the three-Bessel oracles
actually exercise, the ratio never gets close enough to `rtol` for the gate to discard anything.

### Item 6 — prompt 01's C1 reproducers still raise

```
adaptive_levin_sincos((0,1), [lambda x: nan, lambda x: 0], {"theta": lambda x: 1000*x}, ...)
  -> ValueError: sampled amplitude f contains non-numeric values (np.nan, np.inf, or np.-inf)
```

Raised as expected — the non-finite rejection in `p_use` (`np.isfinite(r) and ...`) and the
upstream `f_Cheb`/`sol` finiteness checks prompt 01 added are untouched by this prompt's edits.

## The open question, answered

**Conclusion: the filter still earns its keep at `m = 2`, and it is left in place.**

What it protects against: a p-mode whose *endpoint* magnitude is at or below the numerical noise
floor relative to the dominant mode. At `m = 2` this is "how much of `Re q` (or `Im q`) is real
signal versus noise at the two endpoints" rather than the audit's original SVD-spectrum framing,
but the empirical behaviour is the same kind of protection: `sin_1_100`, an entirely ordinary
problem in this campaign's own standard set (not adversarially constructed), shows the filter
firing (`abserr_truncation = 6.13e-14` at `rtol=1e-10`) and *improving* accuracy against the closed
form by a measured factor of 3.64x when compared against the same run with the filter disabled
(item 4 above). `grz_100` shows the same direction, more weakly (dominated by round-off in its own
near-zero closed form).

Evidence against blanket removal: on the three-Bessel oracle sweep (12 combinations, `J000`/`J231`,
three triangle configurations, two `max_x` values, `rtol=1e-10` matching production
`QuadSourceIntegral.py`/`three_bessel_integrals.py` usage) the filter never once fires —
filter-on and filter-off are bit-identical in all 12 cases — so on the module's actual current
production callers it is currently a no-op that costs nothing and is available as a backstop
should a future caller land closer to the noise floor (e.g. a much larger `rtol`, or a phase
spline whose coupled component is smaller relative to the dominant one than anything tested here).
Combined with the demonstrated `sin_1_100` improvement, this is the "still earns its keep" branch
of the prompt's decision tree, not the "remove it" branch — no §3 issue opened.

Problem (a)'s fix is not moot even under this conclusion: `grz_1000` (item 3/its subsection above)
shows the *gate itself* changing behaviour (mean-based discards a mode the endpoint-based version
correctly keeps) independent of whether the filter is enabled or disabled, so the reasoning fix
matters on its own terms, contrary to the "no case found" premise carried over from the audit.

## Deviations from the prompt

### IMPLEMENTATION CHOICE — normalisation kept as ratio-to-maximum

The prompt asked for a deliberate choice between ratio-to-maximum (conservative, preserves
existing threshold meaning) and fraction-of-total. Ratio-to-maximum was chosen, exactly along the
lines the prompt itself suggested as the conservative option: it keeps the `p_ratios` values
recorded in `used_interval`/its `__str__`/the depth-18 diagnostic meaning what they meant before
(the audit's threshold discussion, the campaign's own `test_stationary_phase_gate...` and other
tests reading `p_ratios` off `used_interval`, and the printed diagnostics all assume this
normalisation). No caller inspected during this prompt reads `p_ratios` as a fraction of a total,
so there was no offsetting reason to switch.

### IMPLEMENTATION CHOICE — `abserr_truncation` as a fourth component, added (not maxed) into `total_err`

The prompt allowed either a new component or folding into `abserr_roundoff`. A new component was
chosen (§3.4's stated reason: a caller wants to know *which* source is limiting them, and
`abserr_roundoff` already means something specific — the eq. (151) floor plus declared
`theta_abserr` — that responds to `chebyshev_order`, not `rtol`; conflating the two would make
`abserr_roundoff` respond to a parameter it shouldn't). It is *added* to
`max(abserr, phase_err)` in `total_err`, not folded into the `max(...)`, because it is a distinct,
always-present bias (a value perturbation with a known sign relative to what full resolution would
give), not another estimate of the same underlying quantity the `max()` already chooses between —
maxing it in would let a large truncation term hide behind an already-large round-off floor instead
of being visible in the total.

### UNINTENDED DRIFT — none

The two regression tests added (`test_mode_filter_gates_on_endpoint_not_mean`,
`test_mode_filter_truncation_visible_in_abserr`) encode measurements made while verifying this
prompt, per rule 9's "every prompt that changes numerics must record a before/after comparison" —
not scope creep, since rule 3's own log requirement is what generated the need for a reproducible
case.

## Verification performed

1. `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` — **23 tests,
   OK**, 0.072s (21 pre-existing + 2 new). Actually run.
2. Items 2–6 above: all actually run via
   `/private/tmp/.../scratchpad/verify_06.py`,
   `verify_06_bessel.py`, `verify_06_bessel_sweep.py` (not committed). Every number in the tables
   is copy-pasted from actual output, not reasoned about.
3. The `_adaptive_levin_subregion_impl` finiteness/gate structure was read directly to confirm the
   `p_use` line's semantics and confirm no other site reads the pre-fix `p_means`/`p_mean_max`
   variables (grep for `p_means` — none remain outside the edited block).

## Observations not acted on

- **`p_ratios_history` and the depth-18 diagnostic (`:1503-1521`) now print endpoint-based ratios,
  not mean-based ones**, since they consume the same `p_ratios` list this prompt changed the
  meaning of. This is the intended, documented consequence of problem (a)'s fix (README: "If you
  change its meaning... either keep the meaning or update the label" — the meaning is
  deliberately kept the same *shape* — ratio-to-max, gated at `rtol` — just fed a different
  quantity), not a separate defect. No diagnostic label needed updating since neither prints a
  name for what `p_ratios` measures beyond "p-ratios".
- **The `grz_1000` field case (this log's own finding) is not itself a bug**, but it is a good
  illustration that a mode sitting within roughly an order of magnitude of `rtol` produces a
  result that depends on exactly which quantity is thresholded and exactly how it is computed
  (spectral solve round-off at the ~1e-11–1e-10 level). Nothing to fix here — this is the
  irreducible sensitivity `abserr_truncation` exists to make visible when it does occur, and it
  does not occur in this case (the mode was *retained*, not discarded) so there was nothing for
  the new component to report.

## State handed to the next prompt

- **C5 is closed.** The non-finite-rejection third of C5 closed with prompt 01; problems (a) and
  (b) close the rest.
- **`used_interval` now carries four abserr components** (`abserr_resolution`/`abserr_fallback`/
  `abserr_roundoff`/`abserr_truncation`), and `_adaptive_levin`'s returned dict carries a matching
  `"abserr_truncation"` key alongside the three prompt-04 keys. Prompt 07 (diagnostics hygiene)
  should be aware of this fourth key if it touches `_write_progress_data()`'s error reporting —
  the same caution prompt 04 recorded (`IMPLEMENTATION_STATE.md` §3
  `[04-roundoff-floor-can-be-infinite]`) does not apply to this component: `abserr_truncation` is
  always finite (it is a finite sum of finite `|p|` values, or `None` for a fallback region), so
  no additional `np.isfinite()` guard is needed where it is consumed.
- **`docs/adaptive-levin-benchmark/levin_bench/{runners,sweeps}.py` is unaffected** — it reads
  `num_direct_solves`, `evaluations`, etc. off `used_interval`/the top-level dict by name, and the
  new `abserr_truncation` key/property is additive, matching standing note 8's rule.
