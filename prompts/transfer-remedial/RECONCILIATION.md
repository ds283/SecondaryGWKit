# Reconciliation of `DRAFT-PLAN.md` against the working tree

**Date:** 2026-09-08
**Tree:** branch `transfer-remedial-plan`, at `95cc326` (measured at `c4c4905`; see §0)
**Plan reconciled:** [`DRAFT-PLAN.md`](DRAFT-PLAN.md) revision 2
**Environment:** Python 3.12.14, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0, Darwin 25.5.0 (arm64)

This document is the planning agent's record. It states which of the plan's checkable claims were
re-measured against the current tree, which reproduced, and which did not. **Where this document
and `DRAFT-PLAN.md` disagree, this document is the one the prompts are built on**, and each
correction is carried into the prompt that depends on it.

Every measurement below was made from the repository root with
`PYTHONPATH=. ./venv/bin/python`, against the tree as committed. No production code was modified.

---

## 0. The tree moved after these measurements were taken

Every measurement in this document was made against `c4c4905`. Six commits have since landed on
`transfer-remedial-plan`, and the campaign documents were re-pointed at `95cc326` before being
committed:

| commit | what it was |
|---|---|
| `39ed7fc` | `docs/gk-wkb-numerical-review-2026-09.md` — a numerical review of the tensor Green's-function WKB construction |
| `4afd531` | `source-remediation` prompt 08 — partition the source time integral, Levin-integrate its phase groups |
| `ffc50ae` | its prompt 09 — `b`, an error bound and honest tolerances on `QuadSourceIntegral` |
| `154126b` | its prompt 04 — triangle filter |
| `815217b` | its prompt 10 — supply the transfer functions to the source integral stage |
| `95cc326` | its prompt 11 — spec annotations |

**`LiouvilleGreen/` was not touched by any of them**, so §1, §2 and §3.1 stand unchanged — every
measurement of `bessel_phase`, `hankel1e`, `jv`/`yv`, the tail series and the split evaluation was
re-verified as still applying to the current tree. What moved:

1. **Line numbers.** `ComputeTargets/QuadSourceIntegral.py` was rewritten (1464 lines changed) and
   `main.py` grew by 248 lines. Every reference in this campaign was re-derived against `95cc326`:
   `main.py` Bessel stage 425-437 → **516-528**, `b_value` 419 → **510**; `BesselPhaseProxy` 52 →
   **121**; `_three_bessel_Levin` 505-670 → **1175-1442**, its four phase sums → **1226, 1267,
   1308, 1349**.
2. **§3.2's scheduling rationale**, rewritten below — the rewrite it was avoiding has happened.
3. **README §1.1's chunking deferral.** The cosmological `phase_spline` measurement this campaign
   deferred has since been made, by `39ed7fc`, with a harsher verdict and four further defects. See
   README §1.1, which now cites it. One of those defects — a hard chunk-switch discontinuity of
   1.08e-4 rad that a Levin consumer sees — is an *additional* argument for prompt 05's removal of
   `phase_spline` from `bessel_phase`.

Worth knowing: `docs/gk-wkb-numerical-review-2026-09.md:13` records that it read this campaign's
README and this document while they were still uncommitted, and its §7 endorses the scope split
("The Bessel campaign correctly leaves cosmological phase construction to separate work"). So the
two documents are consistent by construction, not by luck.

---

## 1. Claims that reproduced

| Plan § | Claim | Measured here |
|---|---|---|
| §3, §7.1 | \(S_\nu=\sqrt{\pi x/2}\,e^{i(\pi\nu/2+\pi/4)}\operatorname{hankel1e}(\nu,x)=a_\nu e^{ir_\nu}\) is **exact**, and \(\theta=x+c_\nu+r_\nu\), \(A=\sqrt{2/\pi x}\,a_\nu\) reproduce \(J_\nu,Y_\nu\) | \(\max\lvert A\sin\theta-J\rvert\le7.6\times10^{-15}\), \(\max\lvert-A\cos\theta-Y\rvert\le3.0\times10^{-15}\) for \(\nu\in\{1/2,3/2,5/2,20.5\}\) over \(x\le10^4\). Confirmed |
| §4.3 | `phi` is a spurious root-solve artefact, and \(E_\theta\approx\lvert\phi\rvert\) at tight tolerance | `phi` = \(-1.149353\times10^{-8}\) (\(\nu=3/2\)), \(-2.044386\times10^{-8}\) (\(7/4\)), \(-4.836537\times10^{-8}\) (\(5/2\)) — the plan's table to seven digits. \(E_\theta\) at `rtol=1e-12` = 1.169e-8, 2.093e-8, 4.873e-8. Confirmed |
| §4.4 | `hankel1e` returns exactly `-0j`, silently, and `isfinite` passes | `hankel1e(100.5, 1e9) == -0j`; `np.isfinite` → `True`; `np.log(np.abs(...))` → `-inf`; `np.angle` → `-0.0`. Boundaries: 2.247e15 for \(\nu\le20.5\), 7.13e8 for \(\nu\in\{100.5,1000.5\}\). Confirmed |
| §4.5 | Residual advance per interval at 250 samples/e-fold exceeds \(\pi\) at \(\nu=1000.5\) | max over the near region: 0.0019 rad (\(\nu=1.5\)), 0.0582 (20.5), 0.3333 (100.5), **3.6847 (1000.5)**. Confirmed; the plan's per-order figures are the values just above the turning point and agree |
| §4.6 | `chunk_logstep=125` is live, the comment is false, and the chunk count grows with \(x_{\max}\) | 2 chunks at \(x_{\max}=10^3\), 4 at \(10^7\), 5 at \(10^{10}\), 7 at \(10^{13}\), 8 at \(10^{15}\). Confirmed |
| §4.6 | `_build_log_chunks_positive` has no progress guard | `phase_spline.py:476` sets the next start to `round(0.75*chunk_end - 0.5)`. With `logstep=1.2`: start 1 → end 2 → next start 1. Non-terminating. Confirmed by inspection |
| §5.2 | The residual right-hand side \((2/\pi)/m-x\) loses all significance | \(x=10^6\): −3.00037e-6 vs −3.0e-6. \(x=10^8\): **−1.49012e-8** vs −3.0e-8. \(x=10^{12}\): −2.44141e-4 vs −3.0e-12. Confirmed |
| §5.3, §7.2 | DLMF 10.18.18 at two terms, and \(a=(1+r')^{-1/2}\) from the Wronskian | \(\lvert\delta r\rvert\): 1.77e-10 (\(\nu=5/2\), \(x=50\nu\)), 7.64e-10 (20.5), 4.01e-9 (100.5); \(\lvert\delta a/a\rvert\) 3.5e-12, 1.9e-12, 2.0e-12 at \(50\nu\) and 5.5e-14, 2.9e-14, 3.1e-14 at \(100\nu\). Confirmed, including that **one series governs both quantities** and that \(\nu=1/2\) is exact (\(r\equiv0\), \(a\equiv1\)) |
| §6.3 | Angle addition beats forming `x + d` | \(\nu=3/2\), against 70-digit `mpmath`: naive 3.475e-14 (\(10^3\)), 1.210e-10 (\(10^7\)), 2.723e-6 (\(10^{12}\)), **4.725e-2** (\(10^{15}\)); split \(\le1.11\times10^{-16}\) at every point. Confirmed |
| §6.3 | `np.sin` is correctly rounded to \(10^{16}\) | \(\lvert\texttt{np.sin}-\texttt{mp.sin}\rvert\le1.11\times10^{-16}\) at \(10^{12},10^{15},2.5\times10^{15},10^{16}\). Confirmed |
| §8.1 | `raw_theta` is never *evaluated* by Levin when both accessors are supplied | `levin_quadrature.py:1038` sets `need_theta_Cheb` false when `theta_mod_2pi` and `theta_deriv` are both present; `:1090` computes `phase_span` from `theta_prime_Cheb`. Confirmed |
| §8.1 | `theta_abserr` exists and nothing supplies it | `levin_quadrature.py:962` reads it; the comment at `:2360` states the intended use; no call site in `LiouvilleGreen/` or `ComputeTargets/` passes it. Confirmed |
| §9 Stage 1 | The existing tests cannot establish the improvement | `test_bessel_phase.py:34` `REL_DIFF = 0.5`; `:116` `relerr < 1e-3` with the "catch garbage" comment at `:113`. Confirmed — but see **C4** |
| §10 | Production builds only \(\nu=1/2\) and \(5/2\), at `atol=1e-25, rtol=5e-14` | `main.py:520-528`, with `b_value = 0.0` at `main.py:510`. Confirmed |

---

## 2. Corrections

### C1 — The scalability claim is wrong in magnitude and, more importantly, in mechanism

**Plan §4.7 says:** the ODE build takes 0.02–0.05 s to \(x_{\max}=10^{11}\) and **">600 s (did not
complete)"** at \(10^{13}\); §1 concludes that "the two-region construction is O(1) in
\(x_{\max}\)" and promotes this from a caveat to a motivation.

**Measured** (\(\nu=1/2\), `atol=1e-25, rtol=5e-14`, the production settings):

| \(x_{\max}\) | build | chunks |
|---|---:|---:|
| \(10^{13}\) | **0.09 s** | 7 |
| \(3\times10^{13}\) | 0.07 s | 7 |
| \(10^{14}\) | 0.07 s | 7 |
| \(3\times10^{14}\) | 0.07 s | 7 |
| \(10^{15}\) | 0.09 s | 8 |
| \(3\times10^{15}\) | **> 60 s, abandoned** | — |

So the \(10^{13}\) row does not reproduce: the build is two decades cheaper than the plan states,
and there is no cost curve at all across five decades. What exists is a **cliff between
\(10^{15}\) and \(3\times10^{15}\)**.

**The mechanism is not integration cost.** Counting right-hand-side evaluations of the same ODE
directly:

| \(x_{\max}\) | `nfev` | accepted steps | time |
|---|---:|---:|---:|
| \(10^{13}\) | 230 | 16 | <0.01 s |
| \(10^{15}\) | 242 | 17 | <0.01 s |
| \(2\times10^{15}\) | 254 | 18 | <0.01 s |
| \(3\times10^{15}\) | — | — | **>45 s, abandoned** |

Flat, then non-terminating. The cause is upstream of the integrator. The right-hand side
\((2/\pi)/(x\,m)\) must equal \(1+O(\nu^2/x^2)\); sampled at five nearby doubles:

| \(x\) | \((2/\pi)/(x\,m)\) at five adjacent arguments |
|---|---|
| \(10^{12}\) – \(2\times10^{15}\) | 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 |
| \(3\times10^{15}\) | 0.988677, 0.948322, 1.028585, 0.978680, 1.033255 |
| \(5\times10^{15}\) | 1.676372, 1.653327, 1.077128, 1.389636, 1.665364 |
| \(10^{16}\) | 0.750691, 1.094350, 1.035889, 0.734835, 1.060665 |

SciPy/Amos `jv`/`yv` lose argument-reduction accuracy above \(x\approx2.5\times10^{15}\), so
\(m=J^2+Y^2\) becomes **O(1)-relatively noisy**. DOP853 at `rtol=5e-14` cannot pass its error
test on noise, so it drives the step size to zero and stalls. This is the same Amos limitation as
§4.4, seen through a different symptom — not a `-0.0` return but O(1) noise — and it is why the
benchmark's \(\kappa=1000\) case (\(x_{\max}\approx8.6\times10^{15}\)) hangs.

**Consequences for the plan**, all of which the prompts carry:

1. **Do not sell the replacement on cost.** The ODE build is ~0.1 s over the entire production
   range. The 11–14× speedups in §4.7 are on a 0.02–0.09 s operation and are not a reason to do
   anything. The real benefit is that a **hard cliff at \(x\approx2.5\times10^{15}\) is removed**,
   which raises the supported \(x_{\max}\) by more than a decade and unblocks the benchmark tier.
   §1's "the existing construction is a documented blocker" survives; "O(1) in \(x_{\max}\)" as a
   *performance* argument does not.
2. **The replacement clears the cliff only because of the closed-form tail.** Above \(x_\star\sim100\nu\)
   nothing evaluates `hankel1e`, `jv` or `yv` at all, so the noise is never sampled. This makes
   the tail load-bearing for the *supported domain*, not merely for accuracy — a strictly stronger
   version of §1's argument.
3. **Above \(x\approx2.5\times10^{15}\) there is no double-precision reference.** `jv`/`yv` are as
   noisy as the ODE they were driving, so any test at the top of the domain must use `mpmath`.
   §9 Stage 1's "permanent `mpmath` references at the corners" is therefore not defence in depth;
   it is the only way to test there at all. Prompt 01 treats it as mandatory.
4. `np.sin`/`math.sin` are **not** implicated: they are correctly rounded to \(10^{16}\) (§1
   above). The degradation is confined to Amos.

### C2 — The residual is not \(O(1)\), and not sub-cycle, at high order

**Plan §1 says** \(r_\nu\) "never exceeds \(O(1)\)"; **§4.6 says** "the interpolated residual is
\(O(1)\) and never exceeds a cycle, so `simple_mod_2pi` is unnecessary, \(\operatorname{div}2\pi\)
is identically zero, and there is nothing to chunk."

**Measured**, from continuously tracked `hankel1e` over the near region \([x_0,\,100\nu]\):

| \(\nu\) | \(r(x_0)\) | \(r(x_\star)\) | span | in cycles |
|---|---:|---:|---:|---:|
| 1/2 | 0.0000 | 0.000000 | 0.0000 | 0.00 |
| 3/2 | 0.6155 | 0.006667 | 0.6088 | 0.10 |
| 5/2 | 1.1832 | 0.012000 | 1.1712 | 0.19 |
| 20.5 | 11.4437 | 0.102440 | 11.3412 | 1.81 |
| 100.5 | 57.1042 | 0.502492 | 56.6017 | 9.01 |
| 1000.5 | **570.8200** | 5.002540 | **565.8175** | **90.05** |

\(r(x_0)\approx\pi\nu/2-\sqrt{\nu^2-\tfrac14}\), which grows linearly in \(\nu\); and
\(r(x_\star)=(4\nu^2-1)/(800\nu)\approx\nu/200\), which exceeds \(\pi\) for \(\nu\gtrsim630\).

**The plan's conclusion survives; its justification does not.** Dropping `phase_spline`,
`simple_mod_2pi` and the `(div_2pi, mod_2pi)` representation from this module is still right,
because at \(\nu=1000.5\) the double-precision resolution of \(\lvert r\rvert\le571\) rad is
\(\varepsilon\cdot571\approx1.3\times10^{-13}\), an order below the \(10^{-11}\) low-order target
and seven below the \(10^{-6}\) high-order target. **That** is the argument to make. "Never exceeds
a cycle" is false above \(\nu\approx630\) and must not appear in a docstring or a commit message.

Two further consequences:

- **§4.5's branch-tracking requirement is bigger than stated.** At \(\nu=1000.5\) the tracker has
  ~90 genuine \(2\pi\) wraps to resolve across the near region, not a handful, and one interval
  advances 3.68 rad. Prompt 04's acceptance test must count wraps, not merely check continuity.
- The plan's remark that \(\nu=1/2\) is a degenerate case is sharper than it looks: \(\mu-1=0\)
  makes \(r\equiv0\) and \(a\equiv1\) **identically at every \(x\)**, so the whole domain is
  closed-form and the near-region sampler should never run. That is a free exactness test, and
  prompt 03 asserts it.

### C3 — `plot_besssel_phase.py` is already broken, so "migrate it" is a decision, not a port

**Plan §8.1** lists `plot_besssel_phase.py:22` as a diagnostic consumer of `Q` to be migrated.
It cannot currently run: `plot_besssel_phase.py:15` reads `data["x_min"]`, a key `bessel_phase`
has never returned (the dict has `min_x`), so the function raises `KeyError` on the first line of
its body. `:30` then calls `phase(x)`, and `phase_spline` defines no `__call__`. The script is
dead code that predates at least two interface changes.

Prompt 06 therefore has to *decide* — repair it against the new interface, or delete it — rather
than mechanically re-point one accessor. Deleting it is defensible (nothing imports it; the live
diagnostic is `ComputeTargets/QuadSourceIntegral_debug.py`), but it is the user's call, so the
prompt requires the agent to state the choice and the orchestrator to surface it.

### C4 — A third existing test already contracts on the derivative, at \(10^{-6}\)

**Plan §9 Stage 1** justifies the new harness by citing only `test_bessel_phase.py:34` (50 %) and
`:113` (\(10^{-3}\), "catch garbage"). It does not mention `test_phase_derivative`
(`test_bessel_phase.py:121-142`), which asserts

```
|theta_deriv(x) / [(2/pi)/(x (J^2+Y^2))] - 1| < 1e-6
```

at \(\nu\in\{2.5,20.5,100.5\}\) over \(x\in[2x_0+1,\,0.95x_{\max}]\). That is a real derivative
contract on the quantity §4.7 identifies as binding, and it is the existing test most likely to be
*disturbed* rather than trivially passed by the replacement: §4.7 measures the new \(\theta'=e^{-2\ell}\)
route at \(2.09\times10^{-8}\) (\(\nu=100.5\), quintic, 250 per e-fold), which clears \(10^{-6}\) by
under two orders. It is a regression gate from prompt 03 onward and prompts must not loosen it.

Note also that this test samples from \(2x_0+1\), i.e. it *excludes* the turning-point interval
where §4.7 locates every maximum. Prompt 01's harness must include that interval; prompt 08 decides
whether this test's lower bound moves.

---

## 3. Facts the plan does not record, that the prompts need

### 3.1 The amplitude plausibility band, with numbers

§4.4 and §7.1 require a plausibility band on \(a_\nu\) rather than `isfinite`, and say only that
\(a_\nu\gtrsim1\) above the turning point. Measured over \([x_0,\,100\nu]\):

| \(\nu\) | \(\min a\) | \(\max a=a(x_0)\) |
|---|---:|---:|
| 1/2 | 1.0000000000 | 1.000000 |
| 5/2 | 1.0000240009 | 1.32288 |
| 20.5 | 1.0000249867 | 1.85684 |
| 100.5 | 1.0000250009 | 2.41795 |
| 1000.5 | 1.0000250016 | 3.54597 |

\(a\) is monotone decreasing in \(x\), attains its maximum at the turning point, and that maximum
grows only like \(\nu^{1/6}\). So the band is genuinely **two-sided and \(O(1)\)**: something like
\(0.99\le a\le 8\) rejects the `-0j` failure (\(a=0\)) and also rejects a spuriously large value,
which a one-sided \(a\gtrsim1\) test would let through. Prompt 04 fixes the constants and tests
both sides.

### 3.2 `ComputeTargets/QuadSourceIntegral.py` is a fourth Bessel-phase consumer, and it is worse off than the ones the plan lists

`_three_bessel_Levin` (`QuadSourceIntegral.py:1175-1442`) makes **eight** `adaptive_levin_sincos`
calls whose phases are signed sums of three `bessel_phase` `raw_theta` values
(`:1226, :1267, :1308, :1349`) with `theta_mod_2pi` companions, and supplies **no `theta_deriv`** —
so unlike `three_bessel_integrals.py`, `need_theta_Cheb` is `True` there and Levin obtains
\(\theta'\) by spectral differentiation of the raw phase, exactly the lossy route
`three_bessel_integrals._phase_group`'s docstring was written to avoid. It has the §8.2 phase-group
cancellation problem too.

**The gap survived that file's own rewrite, which sharpens the finding.** When this was first
measured, the exclusion rationale was scheduling: `source-remediation`'s prompts 08–10 were about to
rewrite the file. They have now done so (`4afd531`, `ffc50ae`, `815217b`), and the eight calls still
pass `theta={"theta": ..., "theta_mod_2pi": ...}` and nothing else. Meanwhile the *new* phase-group
route in the same file **does** supply a derivative — `group.levin_theta(include_deriv=LEVIN_USE_THETA_DERIV)`
at `:1008`, with `LEVIN_USE_THETA_DERIV = True` at `:93`. So `_three_bessel_Levin` is now anomalous
within its own file, and the file's own comment at `:74` notes it is the odd one out. Checked: that
campaign's board records only B6 (`atol`/`rtol` forwarding) against these call sites, **not** the
missing derivative, so the hand-off in prompt 09 is still needed and still novel.

**It remains out of scope**, now for a better reason than scheduling: the file was just rewritten by
that campaign, whose prompt 12 (verification) has not yet run, so this is a fresh, documented state
belonging to that campaign's follow-up. Its rules forbid touching `LiouvilleGreen/`; this campaign
reciprocates. See README §1.1 and §4.2.

### 3.3 The consumer inventory, complete

| Consumer | Reads | Note |
|---|---|---|
| `main.py:520-528` | constructor, `atol`/`rtol` | production; migrate (prompt 06) |
| `ComputeTargets/QuadSourceIntegral.py:121` `BesselPhaseProxy` | whole dict through `ray.put` | serialization check (prompt 06) |
| `ComputeTargets/QuadSourceIntegral.py:1201-1205, 1175-1442` | `phase`, `mod`, `raw_theta`, `theta_mod_2pi` | **out of scope**, §3.2 |
| `ComputeTargets/QuadSourceIntegral_debug.py:55, 254-255` | `Q`, `mod`, `phase`, `bessel_j`, `bessel_y`, `min_x`, `max_x` | live diagnostic; migrate (prompt 06) |
| `plot_besssel_phase.py:14-31` | `x_min` (nonexistent), `phase(x)` (not callable), `Q` | already broken, §C3 |
| `LiouvilleGreen/three_bessel_integrals.py:163-208, 234-247` | `phase`, `mod`, `min_x` | phase groups (prompt 07) |
| `LiouvilleGreen/tests/test_bessel_phase.py` | everything | re-tighten (prompt 08) |
| `LiouvilleGreen/tests/test_three_bessel.py:51-67` | constructor, `mod` as `XSplineWrapper` | **imports `XSplineWrapper` by name** (prompt 06 must keep it importable or update this) |
| `LiouvilleGreen/tests/test_3bessel_analytic.py` | constructor | tolerances 1e-5/1e-6, 1e-2/1e-3 near singularities (prompt 08) |
| `ComputeTargets/tests/test_tk_source_functions.py:228-266` | `mod`, `phase.raw_theta`; defines \(\theta=\pi-\vartheta\) | fixture; re-check sign (prompt 08) |
| `ComputeTargets/tests/test_phase_groups.py:283-350` | `mod`, `phase` through a real `phase_spline` | fixture (prompt 08) |
| `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py:82` | constructor, `sample_points` | **passes `sample_points`**; benchmark (prompt 09) |
| `docs/spec-code-audit/scripts/QI_02_analytic_numeric.py:60-61` | constructor | audit script; leave, note in log |

Two interface details in that list that the plan does not flag and a prompt would otherwise break:
`XSplineWrapper` is imported by name from `LiouvilleGreen.bessel_phase` by
`test_three_bessel.py:10`, and `bessel_phase`'s `sample_points` argument is passed by
`bessel_tier.py:82`. Both are part of the public surface.

### 3.4 Baseline: the existing test suite passes

`LiouvilleGreen/tests/test_bessel_phase.py` passes as committed (4 tests, including
`test_phase_derivative` at \(10^{-6}\)).

The full `LiouvilleGreen/tests` discovery run is **much worse than "slow": it did not complete
within 50 minutes** on this machine and was abandoned. `test_3bessel_analytic.py` rebuilds
`bessel_phase` objects per case (`:47-56`, `:504-512`, `:572-604`) and there are many cases. This is
a pre-existing cost, not something the campaign introduces, but it has two consequences the prompts
carry: later prompts must not read a long run as a regression without comparing against a baseline,
and prompt 01 is told to record **per-module** times if the discovery run will not finish. Prefer
per-module runs while iterating.

---

## 4. Net effect on the plan

Nothing in §1's recommendation changes. The two-region construction, the closed-form tail as a
*required* part, the removal of `phi`, the \(\theta'=e^{-2\ell}\) route, the split evaluation and
the phase-group restructure all stand, and every load-bearing measurement behind them reproduced.

What changes is three framings and one scope note:

1. **Motivation**: "removes a hard cliff at \(x\approx2.5\times10^{15}\) caused by Amos noise
   destabilising the stepper", not "removes an O(\(x_{\max}\)) cost curve" (C1).
2. **Justification for dropping `phase_spline` here**: "\(\varepsilon\lvert r\rvert_{\max}\approx1.3\times10^{-13}\)
   at the largest supported order", not "\(r\) never exceeds a cycle" (C2).
3. **Reference policy**: `mpmath` is mandatory above \(x\approx2.5\times10^{15}\), because
   `jv`/`yv` are not a reference there (C1.3).
4. **Scope**: `ComputeTargets/QuadSourceIntegral.py` is excluded by the in-flight
   `source-remediation` campaign, not merely by preference (§3.2).

`DRAFT-PLAN.md` is left as the review record of revision 2 and is **not** edited by this campaign;
prompt 09 records the corrections in `docs/` instead, where the rest of the project's measured
facts live.

---

## 5. Reproduction

The scripts behind §1–§3 are short and are reproduced in full in the prompts that depend on them:
C1 in [`01-reference-harness.md`](01-reference-harness.md) §4 and
[`02-domain-boundary-tests.md`](02-domain-boundary-tests.md) §3; C2 and §3.1 in
[`04-near-region-sampler.md`](04-near-region-sampler.md) §3; the tail series check in
[`03-closed-form-tail.md`](03-closed-form-tail.md) §5. `DRAFT-PLAN.md` §12 supplies the rest.
