# The QCD background campaign — verification and close-out

**Campaign:** [`prompts/qcd-background-audit/`](../prompts/qcd-background-audit/README.md) ·
**Board:** [`IMPLEMENTATION_STATE.md`](../prompts/qcd-background-audit/IMPLEMENTATION_STATE.md)
**Source document:** [`qcd-background-audit-2026-09.md`](qcd-background-audit-2026-09.md)
**Tree:** `qcd-background-audit` at prompt 09's commit, against the campaign base **`e8f746d`**
(identical to `main` when the campaign was planned).
**Taken:** 2026-09-14, by prompt 09. **No production file is touched by this prompt.**

> **Additive.** This document follows `CLAUDE.md`'s rule for verification documents: a later
> re-measurement **appends a section**; nothing already written here is rewritten, because it was
> correct for the tree it was taken on.

---

## 0. What this document is, and the one thing to read first

### 0.1 The headline

**The campaign's result is §2, and it is a guard, not a consumer table.** The QCD background's
$\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$ is now **bit-identical** to the same integral taken
over an independently root-solved exact background — `1.3320002507788795e+03` at all 17 digits,
relative error `0.0` — where at the campaign base it carried **3.4605051e-08**, worth 47.5 / 4.75e3
/ **1.43e5 radians** at $k = 10^5 / 10^7 / 3\times10^8$ per Mpc against 1-ulp floors of 3.05e-07 /
3.05e-05 / 9.15e-04 rad.

### 0.2 Why the consumer tables are *not* the headline, and are not required to improve

`docs/gktk-remedial-verification.md` §3.5 and §3.6 score a **consumer** against a **producer**, and
both are built from the same `BackgroundModel` — hence the same $H$, hence the same $\tau$. An
error in the background is therefore **common mode between the two sides of that comparison and
cancels in it exactly**. That is how §3.5 could legitimately read 1.00 ulp of its span while the
background underneath both sides carried $10^5$ radians of systematic phase error at
$k=3\times10^8$/Mpc: self-consistency is blind to this entire class of defect.

**It follows that the error this campaign removed cancels in §3.5 and §3.6, and that the consumer
numbers are not required to improve.** They are not the acceptance test for T1 and never could
have been; the only test that can see T1 is one that scores the background against an *independent*
background, which is prompt 01's guard and which is §2 below. What §3 of this document establishes
is the weaker and different thing README §6.4 asks for: which consumer numbers moved, in which
direction, and why.

**Two of §3.5's twelve rows moved the wrong way**, by measured factors of 4.25 and 4.39 (and two of
§3.6's, by 2.0 and 2.8), and §3.3 says exactly why: the
corrected background presents the consumer with a *sharper* feature at the equation of state's
lowest branch crossing than the defective one did. That is recorded as a miss of README §6.4's
"nothing got worse", not argued away, and it is `[13-consumer-spline-crosses-eos-break-points]` —
prompt 10's, and untouched here.

### 0.3 What was run

| # | Command | Wall | Notes |
|---|---|---|---|
| 1 | `PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py` | 45.8 s at `HEAD` / 48.8 s at base | **unedited**; no Ray, no datastore |
| 2 | `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py` | 1.0 s at `HEAD` / 1.4 s at base | **unedited** |
| 3 | `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .` | 0.6 s | |
| 4 | `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` | 152.5 s | |
| 5 | `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` | 997.6 s | the **full** set, `test_3bessel_analytic` included |

**"`HEAD`" throughout this document means `4804aac`, prompt 08's commit**, which is the tree every
measurement here was taken on. Prompt 09 changes no code — its diff is this document, one script
under `docs/qcd-background-audit/`, a §9 appended to `docs/gktk-remedial-verification.md`, its log,
the campaign board, one other board's §3 entry and `docs/OPEN_ISSUES.md` — so every figure holds
unchanged at prompt 09's own commit.

**The base run.** Commands 1 and 2 were taken at **`2a5e0fa`** rather than at `e8f746d`.
`git diff --name-only e8f746d 2a5e0fa` is 21 files, **all of them under `docs/` or `prompts/`**:
`docs/OPEN_ISSUES.md`, the campaign's twelve prompt files, its README, board, log index and four
orchestrator prompts. No Python file, no test module, no fixture, and neither of the two scripts
above. The two trees are therefore **numerically identical** and the base run is a base run for
`e8f746d`.

**Machine load.** This machine was under erratic external load throughout the day (load average
peaked above 140 while prompt 07 was measuring). Every wall-clock figure here is labelled with the
load average at which it was taken and is paired, wherever a conclusion could rest on it, with an
integrand- or RHS-evaluation count — which is reproducible — or with a same-process control.
**No claim in this document rests on a wall-clock number.**

---

## 1. The representation

The audit §4 table, re-measured at `HEAD` on prompt 01's fixed 640-point probe set
(`CosmologyModels/tests/T_z_reference.probe_set()`, $z\in[1.0006,9.54\times10^{15}]$, grid nodes
*and* the midpoints between them, because a spline is worst between its knots).

| Quantity | Base (`e8f746d`) | `HEAD` | Factor |
|---|---|---|---|
| `T(z)` relative error, **max** | 7.177e-04 | **6.807e-11** | 1.1e7 |
| `T(z)` relative error, **p90** | 1.323e-05 | **3.237e-15** | 4.1e9 |
| `T(z)` relative error, **median** | 1.890e-07 | **1.765e-16** | 1.1e9 |
| `_solve_T_z` node error, **max** | 2.496e-05 | **0.000e+00** | exact |
| $H(z)$ on the production grid, max / p90 / median | 1.278e-03 / 2.895e-05 / 3.424e-07 | **1.690e-10 / 6.276e-15 / 2.804e-16** | 7.6e6 / 4.6e9 / 1.2e9 |

Every one of these is a line of `measure_T_z_representation.py`'s own §0, §3 and §4 output, run
unedited, and each reproduces the audit's recommended-representation row to the digit.

**Which defect bought which order of magnitude** — the campaign spent three prompts on the three
representation defects precisely so that this table could exist (README §8 item 1), and the audit
§4 row it reproduces separates them: accurate nodes fix the **p90**, the entropy factor fixes the
**median**, segmentation fixes the **max**.

| after | max | p90 | median | what changed |
|---|---|---|---|---|
| base | 7.177e-04 | 1.323e-05 | 1.890e-07 | — |
| **04** | 7.2615e-04 | **1.936e-07** | 1.071e-07 | `_solve_T_z` `rtol` 1e-4 → 1e-14: **the p90, 68×**; the max is pinned at the jump height and nudges slightly worse |
| **05** | 7.236e-04 | 8.912e-08 | **2.599e-10** | tabulate $F(u)=\log(T/[T_{\rm CMB}(1+z)])$, not $T$: **the median, 412×**, at the same 500 nodes and *lower* cost per call |
| **06** | **6.807e-11** | **3.237e-15** | **1.765e-16** | one spline per branch, edges *bisected* onto the jumps, 3,000 nodes at order 5: **the max, 1.1e7×**, and the p90 and median to round-off |
| 07 | — | — | — | no background value moves; only which points a quadrature splits at |

Prompt 06's own control is the one worth keeping: an edge deliberately misplaced by **one node**
restores **7.054e-04** inside the displaced window while leaving the p90 and the median untouched —
the silent failure mode, now a standing guard in the test suite.

**The equation of state was not touched, demonstrated rather than asserted.** Sections §1 and §2 of
the audit script — the branch joins in $g$ and $g_s$ at all four break temperatures, the forced
steps in $T(z)$, and the jump geometry at the lowest crossing — are **character-for-character
identical** between the base run and `HEAD`. That is README §0.5's boundary held, and it is the
reason a segmented representation could be built at all: it reproduces a discontinuous fixture
*exactly*, so nothing here waited on the answer to README §7 D6.

**The stand-in is now exact, not merely accurate.** `LambdaCDM_GenericEOS` on the constant-$g_s$
`PureRadiationEOS`, scored against the closed form $T = T_{\rm CMB}(1+z)$ over 2,001 points in
$z\in[10^{-3},10^{16}]$:

| | max | p90 | median |
|---|---|---|---|
| base (`2a5e0fa`) | 2.270323e-07 | 1.904049e-07 | 1.006789e-07 |
| **`HEAD`** | **3.330669e-16** | **2.220446e-16** | **0.000000e+00** |

This is the one place where a model that declares no break points is *expected* to move, and
README §2 (g) says so: on a constant-$g_s$ equation of state $F(u)$ is identically constant and the
new representation is exact. It was prompt 05's acceptance test. **The movement is towards the
closed form**, by 6.8e8 in the maximum and to bit-exactness at the median.

---

## 2. T1 closed

**The guard.**
`CosmologyModels/tests/test_T_z_representation.TestQCDTemperatureRepresentation.test_conformal_time_matches_the_exact_background`
computes $\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$ **twice from the same cosmology** — once as
shipped, once with the temperature replaced by the `rtol=1e-14` root solve of the defining equation
$T\,g_s(T)^{1/3} = T_{\rm CMB}\,g_s(T_{\rm CMB})^{1/3}(1+z)$ — so that nothing cancels. It is the
only measurement in the tree that scores the background against an *independent* background.

| | base (`e8f746d`) | `HEAD` |
|---|---|---|
| shipped background, $\int\mathrm{d}z/H$ | `1.3320002968728165e+03` | `1.3320002507788795e+03` |
| exact background, $\int\mathrm{d}z/H$ | `1.3320002507788795e+03` | `1.3320002507788795e+03` |
| relative error | **3.4605051e-08** | **0.0000e+00** (bit-identical: `True`) |
| equivalent phase, $k=10^5$/Mpc | 4.751e+01 rad | **0.000e+00 rad** (floor 3.050e-07) |
| equivalent phase, $k=10^7$/Mpc | 4.751e+03 rad | **0.000e+00 rad** (floor 3.050e-05) |
| equivalent phase, $k=3\times10^8$/Mpc | 1.425e+05 rad | **0.000e+00 rad** (floor 9.150e-04) |

The asserted threshold is `CONFORMAL_TIME_REL = 1e-15`, not the equality, because the identity is
the last bit of a sum over a few hundred quadrature panels and should not be a test's hinge; the
three phase floors are asserted alongside it and the identity is printed.

**How it fell, prompt by prompt** — the record matters, because it says which defect was carrying
the error:

| after prompt | guard | what moved it |
|---|---|---|
| 01 | 3.4605051e-08 | — (the guard is built) |
| 04 | 3.4509e-08 | node accuracy: almost nothing |
| 05 | **5.4264e-10** | the entropy factor: a factor of **64** |
| 06 | **0.0** | segmentation: the remaining 5.4e-10, i.e. 0.74 rad at $k=10^5$ |

The lesson the audit did not predict is prompt 05's: **most of the conformal-time error was never
the jump.** A jump is a set of measure zero inside an integral; what the integral saw was the
interpolation error carried across the whole range, which is what the median measures and what the
entropy factor removes. Segmentation was still required — it is what fixes the max, and the max is
what a *consumer* at a break point meets (§3.3).

**Producer-side consequences, for free.** §3.2 and §3.3 of `docs/gktk-remedial-verification.md`
score the producer's own fixed-order phase table against a converged quadrature on the same
background, so they too are common mode in the background *value* — but not in its *smoothness*, and
a smooth background is what a fixed-order Gauss rule wants. **All six QCD rows improved:**

| quantity | base | `HEAD` | the script's printed floor |
|---|---|---|---|
| $\theta_G$, QCD, $k=10^5$ | 4.7684e-07 rad | **2.3842e-07** | 3.048e-07 |
| $\theta_G$, QCD, $k=10^7$ | 4.5776e-05 | **3.0518e-05** | 3.048e-05 |
| $\theta_G$, QCD, $k=3\times10^8$ | 9.7656e-04 | **4.8828e-04** | 9.145e-04 |
| $\theta_T$, QCD, $k=10^5$ | 7.4506e-08 | **1.4901e-08** | 1.371e-08 |
| $\theta_T$, QCD, $k=10^7$ | 7.6294e-06 | **9.5367e-07** | 1.371e-06 |
| $\theta_T$, QCD, $k=3\times10^8$ | 2.4414e-04 | **4.5776e-05** | 4.113e-05 |

**All six were above the script's printed `eps*|theta|_max` floor at the base**, by 1.07× to
5.94×. At `HEAD` three are below it (0.53×, 0.70×, 0.78×) and the other three are within 11 % of it
— 1.5 to 2 ulp of the accumulated phase, which is a representation floor and not an approximation
error. The corresponding LambdaCDM rows are unchanged, line for line.

And in `gktk-remedial-verification` §3.4, the stored $F$ and $\rho_T$ against prompt 01's
references:

| quantity, QCD | base | `HEAD` |
|---|---|---|
| $\max\lvert F_{\rm stored}/F_{\rm ref}-1\rvert$, $k=10^5$ / $10^7$ | 1.345e-15 / 1.782e-15 | **6.442e-16 / 8.314e-16** |
| the same, $k=3\times10^8$ | 2.665e-14 | **6.252e-14** (2.3× worse) |
| $\max\lvert\rho_T-{\rm ref}\rvert$, the three $k$ | 1.012e-08 / 9.471e-07 / 3.650e-05 rad | **3.234e-09 / 6.926e-07 / 1.155e-05** (3.1× / 1.4× / 3.2× better) |

The one figure that rises, $F$ at $k=3\times10^8$, is 6.3e-14 relative on a friction integral
accumulated over thirteen decades of redshift — a few hundred ulp of a cumulative quadrature, and
its reference moved with the background. The LambdaCDM rows of that section are unchanged.

**And in the primitives** (`verify_production_path.py` §0, scored against the regenerated
references):

| quantity | base | `HEAD` |
|---|---|---|
| QCD `tau` at the production nodes, relative | 2.104e-14 | **2.254e-15** |
| QCD `cs_tau` | 2.108e-14 | **2.212e-15** |
| QCD `friction_F` | 3.340e-16 | 3.340e-16 (unchanged) |
| QCD one-interval $\Delta\tau$ | 9.354e-15 | 9.354e-15 (unchanged) |
| QCD 37 % fraction | 3.408e-14 | 3.386e-14 |
| QCD `rho` worst, absolute | 3.608e-16 rad at (`Tk`, 3e8, $z=1.007\times10^{11}$) | 2.248e-15 rad at (`Tk`, 1e5, $z=1.004\times10^{6}$) |

`tau` and `cs_tau` fall by 9.3× and 9.5×, from *above* the 1.879e-14 / 1.887e-14 floor recorded for
their own references to an order *below* it. The one row that rises is `rho`, whose reference was
itself regenerated in prompt 06 (`rho_G` moved by up to 1.613e-01 relative, `rho_T` by 3.879e-03,
both at $k=3\times10^8$, because $\rho$ is a ~1e-3 rad quantity accumulated straight through the
QCD transition); both figures are nine orders below README §6's $10^{-6}$ rad target and both are
at the round-off floor of the quantity, and the argmax simply relocated.

---

## 3. The consumers

Read §0.2 first. What follows is a record, not an acceptance test.

### 3.1 §3.5 — the consumers at production $x$, all twelve rows

Maximum $|\theta_{\rm consumer} - \theta_{\rm producer}|$ at ten points per production grid
interval, against the producer evaluated at the same points.

| model | $k$ [1/Mpc] | sector | base [rad] | `HEAD` [rad] | base ulp | `HEAD` ulp | at $z$ (`HEAD`) |
|---|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | $G_k$ | 2.3842e-07 | **2.3842e-07** | 1.00 | 1.00 | 1.14594e+06 |
| LambdaCDM | 1e7 | $G_k$ | 0.0000e+00 | **0.0000e+00** | 0.00 | 0.00 | 1.38784e+11 |
| LambdaCDM | 3e8 | $G_k$ | 0.0000e+00 | **0.0000e+00** | 0.00 | 0.00 | 4.09953e+12 |
| LambdaCDM | 1e5 | $T_k$ | 7.4506e-09 | **7.4506e-09** | 1.00 | 1.00 | 147.272 |
| LambdaCDM | 1e7 | $T_k$ | 9.5367e-07 | **9.5367e-07** | 1.00 | 1.00 | 12.1903 |
| LambdaCDM | 3e8 | $T_k$ | 3.0518e-05 | **3.0518e-05** | 1.00 | 1.00 | 8.81189 |
| QCD | 1e5 | $G_k$ | 1.9073e-06 | **8.1062e-06** ⚠ | 8.00 | 34.00 | **4.24388e+07** |
| QCD | 1e7 | $G_k$ | 1.5259e-05 | 1.5259e-05 | 1.00 | 1.00 | 1.19864e+11 |
| QCD | 3e8 | $G_k$ | 4.8828e-04 | 4.8828e-04 | 1.00 | 1.00 | 3.8742e+12 |
| QCD | 1e5 | $T_k$ | 3.1859e-06 | **1.3982e-05** ⚠ | 427.60 | 1876.61 | **4.24388e+07** |
| QCD | 1e7 | $T_k$ | 9.5367e-07 | 9.5367e-07 | 1.00 | 1.00 | 13.4791 |
| QCD | 3e8 | $T_k$ | 3.0518e-05 | 3.0518e-05 | 1.00 | 1.00 | 8.81189 |

**All six LambdaCDM rows are bit-identical**, to every digit the script prints, including the
`phi in [...]` ranges and the sample-point maxima. That is the acceptance test: LambdaCDM computes
$T = T_{\rm CMB}(1+z)$ in closed form and declares no break points, so no prompt in this campaign
may move it, and none did.

**Four of the six QCD rows are bit-identical in value** and moved only in the $z$ at which the
maximum was attained — they are pinned at 1.00 ulp of their span, which is a representation floor
and not an approximation error, so the argmax wanders freely among the many points that attain it.

**Two rose, by 4.25× and 4.39×**, both of them at $z = 4.24388\times10^{7}$ — which is inside the
single production grid interval containing `QCD_EOS`'s `T_LO` crossing. §3.3 measures why.

### 3.2 §3.6 — `theta_deriv` against $\omega$

Relative error of $|{\rm phase.theta\_deriv}(z)|$ against $\omega(z)$ from `*_omegaEff_sq`, over
the consumer's own sample set. `[3:-3]` and the deep interior are the shipped test's windows,
trimmed from the high-$z$ (hand-over) end.

| model | $k$ | sector | base: max / `[3:-3]` / deep | `HEAD`: max / `[3:-3]` / deep | at $z$ (`HEAD`) |
|---|---|---|---|---|---|
| LambdaCDM | 1e5 | $G_k$ | 5.5027e-10 / 5.1353e-10 / 4.3707e-10 | **identical** | 1.389e+09 |
| LambdaCDM | 1e7 | $G_k$ | 5.5095e-12 / 5.1417e-12 / 4.3760e-12 | **identical** | 1.3908e+11 |
| LambdaCDM | 3e8 | $G_k$ | 1.8060e-13 / 1.6856e-13 / 1.4363e-13 | **identical** | 4.1081e+12 |
| LambdaCDM | 1e5 | $T_k$ | 1.7635e-08 / 3.0306e-10 / 8.8279e-12 | **identical** | 2.3055e+09 |
| LambdaCDM | 1e7 | $T_k$ | 1.6810e-08 / 2.8897e-10 / 8.3889e-12 | **identical** | 2.2558e+11 |
| LambdaCDM | 3e8 | $T_k$ | 1.7091e-08 / 2.9376e-10 / 8.5379e-12 | **identical** | 6.8187e+12 |
| QCD | 1e5 | $G_k$ | 5.1283e-07 / 2.3123e-07 / 2.3123e-07 | **1.0232e-06 / 1.0232e-06 / 1.0232e-06** ⚠ | **4.2885e+07** |
| QCD | 1e5 | $T_k$ | 1.0998e-06 / 1.0998e-06 / 1.0998e-06 | **3.0630e-06 / 3.0630e-06 / 3.0630e-06** ⚠ | **4.2885e+07** |
| QCD | 1e7 | $G_k$ | 1.0984e-05 / 7.0431e-06 / 6.0842e-06 | 1.0969e-05 / 6.7435e-06 / 6.0229e-06 | 1.4902e+11 |
| QCD | 1e7 | $T_k$ | 4.8896e-07 / 4.8896e-07 / 4.3292e-07 | **1.8853e-08 / 3.4112e-10 / 3.0630e-10** ✦ | 2.5901e+11 |
| QCD | 3e8 | $G_k$ | 3.3092e-04 / 3.3092e-04 / 1.7475e-04 | 2.5418e-04 / 2.5418e-04 / **2.5418e-04** | 3.9232e+12 |
| QCD | 3e8 | $T_k$ | 8.4188e-06 / 8.4188e-06 / 8.4188e-06 | 7.3236e-06 / 7.3236e-06 / 7.3236e-06 | **8.5795e+11** |

All six LambdaCDM rows are bit-identical, including the per-sample decay ladders.

**✦ The clearest single improvement in the campaign is QCD $T_k$ at $k=10^7$**, and it is worth
reading closely because of *what* it became rather than by how much it fell. The maximum improves
26×, the `[3:-3]` window 1,430× and the deep interior **1,413×**. But the diagnostic is the shape of
the last five samples at the hand-over:

```
base   4.8896e-07 window;  last five (high z)  3.74e-07  4.89e-07  3.93e-07  3.01e-07  4.81e-07
HEAD   3.4112e-10 window;  last five (high z)  8.63e-11  3.41e-10  1.29e-09  4.94e-09  1.89e-08
```

At the base those five numbers are **flat and of the same size as the interior** — they are not an
end effect at all, they are background noise. At `HEAD` they are a clean geometric ladder rising by
a factor of ~3.8 per sample inwards from the hand-over, *which is exactly the LambdaCDM signature*
(`9.05e-11 2.89e-10 1.16e-09 4.38e-09 1.68e-08` at the same $k$) and is
`[10-residual-spline-end-condition]`'s decay, closed at the cubic in `gktk-remedial-verification`
§3.6. **The five hand-over samples now agree with LambdaCDM's to within 20 %** (ratios 0.95, 1.18,
1.11, 1.13, 1.13), so what dominates this row is the same end condition on both models; what used
to sit on top of it was the $T(z)$ representation, and it is gone. The deep interior is still 36×
LambdaCDM's (3.0630e-10 against 8.3889e-12) — the QCD background's own remaining structure, not the
representation. This row is the consumer-side counterpart of §2's guard, and it is the one place where a
common-mode-cancelling comparison could still see the defect — because the *derivative* of a
producer–consumer difference is not common mode in the background's **smoothness**.

**The two rows that rose** are the same $z = 4.2885\times10^{7}$ — the production grid node
immediately above the `T_LO` crossing — as §3.1's two, and §3.3 is their cause.

**QCD $G_k$ at $k=10^7$ and $3\times10^8$ barely moved**, and that is expected: `phase-representation`
prompt 02 established that those two rows are **neither** the knots **nor** the node tolerance, but
`[02-consumer-phi-below-the-storage-granularity]` — $\varphi$ is recovered as a difference of two
numbers of size $k\tau$, so its whole range spans 6.0 ulp of the stored phase at $k=10^7$ and 2.0
ulp (three distinct values over 1,377 samples) at $3\times10^8$, and differentiating that staircase
is worse than omitting $\varphi$ altogether. No amount of background accuracy touches that, audit
§9 says so, and README §0.5 puts it out of scope. The $k=3\times10^8$ row's **deep interior rose**
1.7475e-04 → 2.5418e-04 while its maximum fell 3.3092e-04 → 2.5418e-04: the three windows have
collapsed onto one number, which is what a quantity dominated by a three-valued staircase looks
like when the noise that used to differentiate the windows is removed.

### 3.3 Why the two $k=10^5$ QCD rows rose — measured, not inferred

Both rise at the `T_LO` branch crossing, $z_c = 4.253368543\times10^{7}$. The mechanism is that the
**corrected background presents a sharper feature there than the defective one did**, and a cubic
spline of $\varphi$ with default knots meets the whole of it inside one grid interval instead of
smeared across four.

$\mathrm{d}\ln H/\mathrm{d}u$ on production-grid spacing ($\Delta u = 2.3032\times10^{-2}$),
25 intervals centred on the crossing, deviation from the local median:

| | base (`2a5e0fa`) | `HEAD` (`4804aac`) |
|---|---|---|
| peak deviation, in the crossing's own interval | +3.071e-02 | **+8.545e-02** |
| total $\sum\lvert$deviation$\rvert$ over the 25 intervals | 1.202e-01 | **8.558e-02** |
| fraction of that total inside the crossing's own interval | **25.55 %** | **99.84 %** |
| intervals carrying > 10 % of the peak deviation | **8** | **1** |
| background away from the crossing | ±3e-03 scatter | smooth drift at ~1e-05 |

Two things happened at once and both push the consumer the same way. The old 500-node, order-3
$T$-against-$u$ spline had a **knot spacing of 9.2936e-02 in $u$, which is 4.04× the production
grid spacing** — so the discontinuity was smeared over about four grid intervals and only a quarter
of the step was ever presented to the consumer inside one of them. And the old representation
carried its own ±3e-03 scatter in $\mathrm{d}\ln H/\mathrm{d}u$ for several intervals either side,
which flattened the contrast further. The corrected background does neither: the step is the
genuine one, it is 2.78× taller, and **99.84 % of it is inside a single grid interval**.

§3.1's maximum for both sectors is at $z = 4.24388\times10^{7}$, which lies **inside that one
interval** (production nodes 869 and 868, $z = 4.190902\times10^{7}$ and $4.288548\times10^{7}$);
§3.2's is at node 868 itself, the first node above it. The measured consumer factors — 4.25× ($G_k$)
and 4.39× ($T_k$) — are consistent with the 2.78× in step height carrying a further ~1.5× from the
concentration, though that decomposition is a plausibility argument: what is measured is the two
halves of the table above and the two consumer factors.

**This is not a regression of the background.** The background is now right where it was wrong: the
step at $z_c$ is `QCD_EOS`'s own, forced by a $-2.284\times10^{-3}$ jump in $g_s$ at the
$10^{-5}$ GeV branch join, and the campaign's segmented representation reproduces it to six figures
with both sides at the round-off floor (log 06). What rose is the error of `PrimitivePhase`'s and
`TkSourceFunctions`' **cubic spline of $\varphi$ across a genuine discontinuity**, which is
`[13-consumer-spline-crosses-eos-break-points]` — opened by `GkTk-remedial` prompt 13, narrowed
twice, assigned to this campaign's **prompt 10**, and deliberately untouched here because prompt 09
is a verification prompt.

**Recorded as a miss.** README §6.4 asks prompt 09 to establish "that nothing got worse". Two of
the twelve §3.5 rows and two of the twelve §3.6 rows did get worse, by factors between 2.0 and 4.4.
No threshold was loosened and nothing was fixed; the measurement above is the cause, and the board
entry carries it forward.

**For scale, and not as an excuse.** The two risen §3.5 figures are 8.1e-06 and 1.4e-05 rad, against
the **~1e-03 rad Liouville–Green truncation floor that bounds any QCD phase claim**
(`gktk-remedial-verification` §3.5). They are 70–120× below it, where before they were 300–500×
below it. They are still above README §6's $10^{-6}$ rad consumer target, as they were at the base.

### 3.4 The rows this script cannot see — `[02-verify-script-builds-its-own-Gk-consumer]`

`docs/gktk-remedial/verify_production_path.py` constructs `PrimitivePhase(...)` **directly**, at
`:557` and `:1173`, instead of going through `GkSourcePolicyData._build_phase`. The consequence is
a standing limitation of every $G_k$ number in §3.1 and §3.2 above, and it is restated here rather
than fixed (it is `prompts/phase-representation`'s file and this is a verification prompt):

**Blind rows — six of §3.5's twelve, and all six $G_k$ `theta_deriv` column groups** (the
issue's own wording is "both `theta_deriv` $G_k$ columns", meaning both models):

| §3.1 rows | §3.2 rows |
|---|---|
| LambdaCDM $G_k$ at $k=10^5$, $10^7$, $3\times10^8$ | LambdaCDM $G_k$ at $k=10^5$, $10^7$, $3\times10^8$ |
| QCD $G_k$ at $k=10^5$, $10^7$, $3\times10^8$ | QCD $G_k$ at $k=10^5$, $10^7$, $3\times10^8$ |

What they are blind to is **anything the production $G_k$ call site passes to `PrimitivePhase`** —
knot vectors, spline order, break points, policy. They measure the consumer's *representation* of
$\varphi$ faithfully and its *construction by production code* not at all. A reader must not read
the six $G_k$ rows as evidence about the production $G_k$ path's configuration; in particular, if
prompt 10 gives `PrimitivePhase` a knot vector at the declared break points, **this script will not
show it on the $G_k$ side** until the issue is fixed. The six $T_k$ rows do go through the
production `TkSourceFunctions` constructor and are not affected.

### 3.5 `WKB_mod_2pi` self-consistency — unchanged, all zero

`gktk-remedial-verification` §3.7's defect was resolved by `prompts/phase-representation` prompt 01
before this campaign began. It stays resolved: **0 inconsistent samples in all twelve (model,
sector, $k$) rows** — 43,434 to 79,809 samples in the $G_k$ rectangles, 1,037 to 1,401 in $T_k$ —
and **0 of 400,000** in each of the three uniform controls, including $|\theta|\sim4\times10^{12}$
where half an ulp of $|\theta|/2\pi$ is 6.104e-05 cycles. Base and `HEAD` are character-identical
here.

---

## 4. The break-point set, and what it cost

### 4.1 The set

`integration_break_points` now returns the crossings of the equation of state's declared break
temperatures and **nothing else** (prompt 07). On the production source grid, 1,732 samples over
$z\in[0.1,2.064\times10^{16}]$:

| | base (`e8f746d`) | after prompt 06 | `HEAD` (after 07) |
|---|---|---|---|
| `BREAK_POINT_ALL` | **407** | 2,414 | **3** |
| median spacing, in grid intervals | 4.04× | 0.67× | **215.34×** |
| `BREAK_POINT_DISCONTINUITY` | 2 | 2 | **2** |
| of the declared points, knots of the `T(z)` interpolant | **404** | 2,411 | **0** |

The three, to 17 digits in $u=\log(1+z)$, **bit-identical to prompt 06's segment edges** because
both read one cache bisected once in `__init__`: `17.565806941870026` (`T_LO`, $10^{-5}$ GeV),
`23.197460552819653` (`EOS_T_LO`, 0.002 GeV), `27.485391822044257` (`T_120_MEV`, 0.12 GeV).
`BREAK_POINT_DISCONTINUITY` drops the middle one, where $w$ kinks but $g_s$ does not step (the join
that matches to 1.751e-11).

The count is the audit's G1 verdict discharged: the 404 were a uniform lattice of a 500-point
auxiliary interpolant, splitting a Gauss panel every 4.04 grid intervals throughout
`BackgroundModel` for an artefact, and they were the sole cause of `prompts/phase-representation`
prompt 02's Schoenberg–Whitney failure. A multiplicity-`spline_order` knot vector now **constructs
on all six** of the production geometries that log measured as singular, and the same construction
with the knot lattice restored still raises `LinAlgError` on all six — both asserted in the tree by
`ComputeTargets/tests/test_numeric_break_points.py::TestConsumerKnotVectorConstructs`.

> **Caveat on the audit script's own §5 prose.** `measure_T_z_representation.py:401-405` prints
> "Of the BREAK_POINT_ALL points, 2411 are knots of the T(z) spline itself". It never computes
> that intersection — the number is the tabulation's interior knots in range, and the true
> intersection is **0**. The table above the sentence is correct. This is
> `[09-audit-script-section-5-prose-counts-the-wrong-set]`; prompt 09 must run the script unedited
> and does not fix it.

### 4.2 What the build cost, in evaluations

Integrand-evaluation counts, which are exactly reproducible; the wall times beside them were taken
at load average ~16 with a same-process LambdaCDM control that moved 1.03× over the same pair of
runs (log 07).

| | with the 404-knot lattice | `HEAD` | factor |
|---|---|---|---|
| QCD `BackgroundModel`, `tau` / `cs_tau` / `friction_F`, each | 16,580 | **6,936** | 2.39× |
| LambdaCDM, same (break-free control) | 6,924 | 6,924 | 1.00× |
| all three tables, wall | 0.781 s | 0.418 s | 1.87× |
| whole `compute_background`, wall | 0.959 s | 0.599 s | 1.60× |
| `rho` residual table, × the order×intervals baseline | 2.337× | **1.002×** | — |

QCD's 6,936 is now **0.17 % above LambdaCDM's break-free 6,924** — three extra Gauss panels in
1,731 intervals, which is what declaring three genuine discontinuities ought to cost.
`COST_BREAK_POINT_FACTOR` in `test_phase_residual.py` came back 2.40 → **1.01**, below the 1.30 it
started at rather than merely to it.

`verify_production_path.py`'s own per-object costs, at `HEAD` against base, with the LambdaCDM rows
as the control:

| model | sector | build evals, base → `HEAD` | cached evals |
|---|---|---|---|
| LambdaCDM | $G_k$ | 7,392 → **7,392** | 468 → 468 |
| LambdaCDM | $T_k$ | 11,376 → **11,376** | 5,540 → 5,540 |
| QCD | $G_k$ | 8,380 → **6,892** (−17.8 %) | 472 → 472 |
| QCD | $T_k$ | 12,896 → **11,532** (−10.6 %) | 5,608 → 5,608 |

and in bulk through the Levin-side accessor, QCD `raw_theta` off-grid needs **4.00** integrand
evaluations per call where it needed **4.27** — exactly LambdaCDM's 4.00, i.e. the extra panel
splits the knot lattice used to force inside a partial interval are gone.
(28.14 µs against 34.06 µs per call at `HEAD` and base respectively, at load averages ~6 and
unknown; the evaluation count is the reproducible half of that row. Log 07's same-process pair for
the same accessor is 39.48 → 29.23 µs on/off and 77.97 → 57.66 µs off/off.)

### 4.3 Prompt 08's drift table — the policy question, re-taken

`GkTk-remedial` prompt 19 chose `TkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_ALL` **on
measurement**, against knots that carried a $10^{-4}$-level defect. Prompt 08 re-took that
measurement across 50 wavenumbers × 2 sectors × 2 policies × 3 models
(`docs/qcd-background-audit/per_sector_policy_remeasure.py` → `PER-SECTOR-POLICY.md`, 744 s):

| sector | model | policy | prompt 19 | worst drift now | median | $k$ above 3.4e-08 |
|---|---|---|---|---|---|---|
| $T_k$ | QCD | `all` | 8.72e-09 | **7.08e-09** | 1.96e-09 | **0** |
| $T_k$ | QCD | `discontinuity` | **1.97e-07** (3 offenders) | **8.85e-09** | 1.96e-09 | **0** |
| $G_k$ | QCD | `all` | — | 3.67e-09 | 2.16e-10 | 0 |
| $G_k$ | QCD | `discontinuity` | 8.41e-09 | **3.67e-09** | 2.16e-10 | 0 |
| $T_k$ | LambdaCDM / Radiation | either | 5.7e-11 / 4.21e-11 | **identical** | — | 0 |
| $G_k$ | LambdaCDM / Radiation | either | 2.1e-11 / 1.94e-11 | **identical** | — | 0 |

The state found is **(a)**: the knots were a proxy for the representation's own defect. The three
wavenumbers prompt 19's decision actually turned on read 8.89e-10 / 2.03e-09 / 5.60e-10 with the
jumps alone, against 6.1e-08 / 2.0e-07 / 3.5e-08 then — factors of 69, 98 and 63, **with no change
to the integrator**. The two models that declare nothing are bit-identical between the policies at
all 50 wavenumbers in both sectors and reproduce prompt 19's grid totals as exact integers
(401,677 / 429,178 / 637,138 / 640,213).

**What is still load-bearing is the jumps, not the kinds.** With the declaration suppressed
altogether, QCD $T_k$ is above the criterion at **19 of 50** wavenumbers, worst **9.61e-06** at
$k=4.223\times10^7$/Mpc. `GkTk-remedial` prompt 18's result stands undisturbed; what became
vestigial is only the `BREAK_POINT_ALL`-versus-`BREAK_POINT_DISCONTINUITY` distinction, now one
kink (`EOS_T_LO`) rather than 2,411 knots.

**Neither `BREAK_POINT_KIND` was changed.** Both are in a datastore lookup key; the wider policy
costs **+0.99 %** of the $T_k$ evaluations (8,897 → 8,986 per object, against +220 % when the
lattice was declared) and changing it would move every stored QCD $T_k$ value by up to **2.86e-04**
of the envelope and demand a regeneration of that sector and everything below it. That is README
§7 **D5**, reported to the user and not decided, with log 08's recommendation to leave both alone.

---

## 5. Cost

### 5.1 `T_photon` per call — **a confirmed miss** of README §6.2

README §6.2 sets `T_photon` at **≤ 2.5 µs/call** (audit §2 (c): shipped 2.26–2.44 µs, the audit's
own segmented candidate 2.19–2.21 µs) and README §2 (c) makes a regression a stop condition.

| candidate | audit / base | `HEAD` |
|---|---|---|
| shipped representation | 2.454 µs | **2.596 µs** ⚠ |
| entropy factor, 2,000 pts (script's own control) | 2.118 µs | 2.204 µs |
| segmented entropy factor (script's own control) | 2.239 µs | 2.259 µs |
| accurate root solve, build time only (control) | 8.464 µs | 8.753 µs |

**This is a miss, and it is now measured rather than inferred.** Re-taken on a quiet machine after
prompt 06 committed, the shipped representation reads **2.596 µs mean, range 2.505–2.671 over five
runs**, with the audit script's own internal controls back inside their baseline band; the single
run tabulated above was taken at load average ~5 and reads 2.596 µs with its controls +0.9 % to
+4.1 % of their base values. **Against ≤ 2.5 µs that is a miss of about 3.8 %.** Log 06's
deviation 5 quotes a **2.53 µs** quiet-machine figure for the *pre-inline* code; both measurement
sets are in that deviation and they do not contradict each other — the inline moved the segment
dispatch, not the spline evaluation.

**What the excess is.** The shipped-to-unsegmented ratio at the same order is 1.00–1.05 — the
segmentation is free. The whole excess over prompt 05's shape is the **order-5 spline evaluation**,
+8 to +9 %, and that is SciPy's `BSpline.__call__`, already four fifths of the call. **Order 5 is
not optional**: a cubic needs ~25,000 nodes to reach README §6.1's p90, and at 3,000 nodes of order
5 the representation reaches 6.807e-11 / 3.237e-15 / 1.765e-16 where 500 / $k=3$ segmented misses
every row including T1.

**Impact: none downstream.** `T_photon` costs ~2.6 µs inside a ~10 µs `Hubble` call, and every
aggregate cost in §4.2 fell. **What is available cheaply** is
`[07-t-photon-range-logic-recomputes-its-bounds]`: `TemperatureRepresentation.__call__` evaluates
two loop-invariant `_outward` calls on every call, 0.056 µs each, measured — 0.11 µs, which would
bring 2.596 µs to ~2.49 µs. That is a production edit and prompt 09 may not make it.

### 5.2 Build cost

| quantity | figure | stop condition |
|---|---|---|
| `_build_T_z_spline`, README §6.2's base row | ~4 ms (500 nodes @ 8.3 µs — the audit's *estimate*, not a measurement) | — |
| the same, measured after prompt 04 (500 nodes, order 3, `rtol=1e-14`) | 13.4 ms | — |
| the same, measured after prompt 05 | 13.3 ms | — |
| **the same at `HEAD`** (3,000 nodes, order 5) | **87.8 ms**, of which 4.2 ms is the edge bisection | **> 100 ms** |
| QCD `compute_background`, whole, with the 404-knot lattice | 0.959 s | — |
| **the same at `HEAD`** | **0.599 s** | — |

The build cost rose 6.6× over the measured 500-node figure and is the largest number in the
campaign's acceptance table that is close to its bound, at 88 % of it. It is paid **once per
cosmology object**; it bought the p90 and the median outright; and it is more than repaid by
§4.2's 2.39× fall in the cumulative tables' integrand evaluations, which is a cost paid per
compute target rather than per cosmology.

### 5.3 Suites

| suite | base (`e8f746d`) | `HEAD` | wall at `HEAD` |
|---|---|---|---|
| `CosmologyModels/tests` | 11 OK | **30 OK** | 0.6 s at load ~5 |
| `ComputeTargets/tests` | 339 OK | **359 OK** | 152.5 s at load ~5.8 |
| `LiouvilleGreen/tests` (full) | 148 OK | **148 OK** | 997.6 s at load ~5.8 → 11.5 |

No count falls. `CosmologyModels` rises by 19 (prompt 01's seven cases and prompt 06's twelve) and
`ComputeTargets` by 20 (prompt 03's fifteen and prompt 07's five).

---

## 6. What the campaign did not establish

In the shape of `docs/OPEN_ISSUES.md` §5: standing caveats that bound how the rest of this document
may be read.

1. **The verification geometry still does not reach production $x$.** `source-remediation`'s run A
   used `zend = 1e7` to stay inside radiation domination where `analytic_rad` is a valid oracle, so
   its largest accumulated phase was $x = 4.63\times10^5$ against $x\sim1.4\times10^7$ at the
   production `zend = 0.1` for the largest $k$. Every "verified live" claim in that campaign carries
   that ceiling, and nothing here lifts it. `docs/OPEN_ISSUES.md` §5.

2. **`[00-consumer-anchoring-floor]` and `[02-consumer-phi-below-the-storage-granularity]` are
   untouched, and they are what limit `theta_deriv` at $k\ge10^7$.** $\varphi$ is recovered as a
   difference of two numbers of size $k\tau$; on QCD $G_k$ its whole range is 6.0 ulp of the stored
   phase at $k=10^7$ and 2.0 ulp — three distinct values over 1,377 samples — at $3\times10^8$.
   Differentiating that staircase is 3× and 10× *worse* than omitting $\varphi$ entirely, which is
   what §3.2's two QCD $G_k$ rows at those wavenumbers actually are. No amount of background
   accuracy touches it (audit §9); per-region anchoring is a different campaign, and README §0.5
   puts it out of scope here.

3. **The equation of state's branch joins are unrepaired.** `QCD_EOS`'s joins at $10^{16}$, 0.12
   and $10^{-5}$ GeV jump by +1.395e-02, −3.744e-04 and −2.284e-03 in $g_s$, forcing steps in
   $T(z)$ at fixed $z$ of −4.649e-03, +1.248e-04 and +7.614e-04; the join at 0.002 GeV matches to
   **1.751e-11**, and that asymmetry is the evidence that the other three are a defect of the
   transcription rather than the parametrisation's intent. **This campaign deliberately did not
   repair it** — it is an upstream data fixture, and a segmented representation reproduces a
   discontinuous fixture exactly, so nothing here waited on it. Prompt 01 **pins all four joins**
   and the *set* of `break_temperatures_GeV` in
   `CosmologyModels/tests/test_T_z_representation.py::test_the_branch_joins_are_where_the_fixture_puts_them`,
   so a later correction announces itself as a test failure and not as a silent change of
   cosmology. The question for the fixture's authors is README §7 **D6**.
   `[00-eos-branch-joins-do-not-match]`. **The $10^{-5}$ GeV join is the origin of §3.3's entire
   finding**: it is the step the consumer's cubic now meets undiluted.

4. **A pre-campaign datastore is refused, not migrated.** Prompt 03 took README §7 D1 option (i): a
   `T_z_representation` integer column on the QCD cosmology table, filtered on as an equality in
   `build()` and fed by `LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION`, which is **5** at this
   commit. A datastore written before prompt 03 raises `RuntimeError` from
   `sqla_QCDCosmology_factory.build()` naming this campaign and demanding regeneration; the
   mismatch arrives as SQLite's `no such column`, because `Datastore._build_schema()` builds the
   `Table` from the code while `_ensure_tables()` never alters an existing one. **There is no
   migration and none is recommended.** A datastore written *between* prompts 03 and 07 is
   correctly rejected by the key. Note that the same protection does **not** extend to
   `BREAK_POINT_KIND`'s consumers in the way §4.3's decision would need — that is
   `GkTk-remedial` prompt 20's ground and README §7 D5's cost.

5. **The `convergence` block of `ComputeTargets/tests/wkb_reference_data.json` is stale, and this
   prompt could not take it.** It is written by `docs/gktk-remedial/residual_convergence.py`, not by
   the campaign's own generator, and five tests read it directly. One tolerance is owed on its
   account — `QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04` in `ComputeTargets/tests/test_background_tau.py`,
   measured **1.418851e-04** at `T_120_MEV`, the whole of which is the block's age: the tree's
   crossing is now the genuine jump at $u = 27.485391822$ while the block still records where a
   smooth interpolant passed through 0.12 GeV, at $u = 27.485249937$. Prompt 09's "files you may
   touch" includes neither that JSON nor that test module, and prompt 09 may not touch production
   code at all, so **it was not done**: `[01-convergence-block-has-a-separate-generator]` stays
   open and `[02-qcd-reference-floor]` on the `GkTk-remedial` board waits on the same run.

6. **Six of §3.5's twelve rows and all six $G_k$ `theta_deriv` column groups are blind to the
   production $G_k$ call site.** §3.4 above, `[02-verify-script-builds-its-own-Gk-consumer]`.

7. **§3.3's finding is not repaired.** The consumer's cubic spline of $\varphi$ across a genuine
   discontinuity is `[13-consumer-spline-crosses-eos-break-points]`, it is now 4.3× worse than the
   figure `gktk-remedial-verification` §3.5 records, and it belongs to **prompt 10**, which prompt
   07 unblocked and which is gated on README §7 **D7**. Until it runs, the two QCD $k=10^5$
   consumer rows are the campaign's one un-discharged consequence.

8. **The numeric→Liouville–Green hand-over, the Levin path and the source sample grid are
   untouched.** README §0.5. The grid is the campaign's own G2 and is prompts 11–12, also gated on
   D7: `populate_z_sample` is a bare `logspace` that never consults the cosmology, `winnow` is a
   blind stride, and the tag `SourceRedshiftGrid_{len}` labels size only, so two different grids of
   equal length collide in the datastore.

---

## 7. Reproduction

```bash
# the whole of §3 and §4.2, and the base comparison (~46 s; no Ray, no datastore)
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py

# §1, §2 and §5.1 -- the audit's own six sections (~1 s)
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py

# §2's guard on its own, with its numbers printed (0.4 s)
PYTHONPATH=. ./venv/bin/python -m unittest \
    CosmologyModels.tests.test_T_z_representation -v

# §4.3 (744 s)
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/per_sector_policy_remeasure.py

# the QCD reference fixture (~190 s; --dry-run reports without writing)
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py --dry-run

# §5.3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .
```

§3.3 has its own script, added by prompt 09 (~1 s, no Ray, no datastore). Run it on this tree and
then on a `git worktree` at the campaign base; the two runs differ only in
`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`:

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/consumer_break_point_profile.py

git worktree add --detach /tmp/qcd-base 2a5e0fa
ln -s "$PWD/venv" /tmp/qcd-base/venv        # a worktree has no ./venv of its own
PYTHONPATH=. /tmp/qcd-base/venv/bin/python \
    docs/qcd-background-audit/consumer_break_point_profile.py --root /tmp/qcd-base
```

It takes `--crossing {T_LO,EOS_T_LO,T_120_MEV}`; `T_LO` is the default and is the one §3.5 rises
at. **Read the concentration statistics only at `T_LO`**: there the smooth background is locally
flat ($\mathrm{d}\ln H/\mathrm{d}u = 1.99996$ with a ~1e-05 drift across the whole profile), so the
deviation from the local median is the step and nothing else. At the other two crossings the
profile sits inside the QCD transition, where $g_*(T)$ is changing fast enough that the smooth
variation across 25 grid intervals swamps the step; the script prints a caution there, and its
per-interval table remains correct at all three.

---

## 8. Prompt 10 — the consumer spline at the crossing, and what actually fixes it

**Added 2026-09-15 by prompt 10, additive.** Nothing above this line is edited. The tree is
`bc8c3e4` (prompt 09) plus this prompt's script, test class and documentation; **no production file
changed**, and `T_Z_REPRESENTATION_VERSION` is **5** before and after, so every figure in §0–§7
holds unchanged here.

§3.3 above established *why* the two QCD $k=10^5$ rows rose: the corrected background delivers the
equation of state's step undiluted where the old `T(z)` spline smeared three quarters of it over the
neighbouring grid intervals, and it charged the rise to
`[13-consumer-spline-crosses-eos-break-points]` and to prompt 10. This section is prompt 10's answer,
and it is not the one the issue predicted.

**Reproduction** (~120 s, no Ray, no datastore):

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/consumer_knot_scheme_scan.py
```

### 8.1 Nine knot schemes, twelve rows — nothing recovers the two that rose

`max |theta − ref|` at ten points per production interval, rad (ulp of the span). `base` is the
shipped `make_interp_spline` default and reproduces §3.1 to every printed digit. `ALLx3` / `DISCx3`
are the **repeated multiplicity-`spline_order` ($C^0$) knot vector** at the two break-point kinds —
the construction the issue's own "next step" named and prompt 10 §2 makes the default; `segALL` /
`segDISC` are **per-segment splines**, its fallback; `x1` and `x2` are controls, not proposals.

| model | $k$ | sector | base | segment | $C^0$ knot | mult. 1 | mult. 2 |
|---|---|---|---|---|---|---|---|
| LambdaCDM | all three | both | *(six rows)* | **identical** | **identical** | **identical** | **identical** |
| **QCD** | **1e5** | **$G_k$** | **8.1062e-06 (34.00)** | 4.0531e-05 (170.0) — **5.00× worse** | 1.6928e-05 (71.0) — **2.09× worse** | 6.6757e-06 (28.0) | 6.6757e-06 (28.0) |
| **QCD** | **1e5** | **$T_k$** | **1.3982e-05 (1876.6)** | 7.0770e-05 (9498.6) — **5.06× worse** | 2.9315e-05 (3934.6) — **2.10× worse** | 1.1631e-05 (1561.1) | 1.1528e-05 (1547.2) |
| QCD | 1e7 | $G_k$ | 1.5259e-05 (1.00) | identical | identical | identical | identical |
| QCD | 1e7 | $T_k$ | 9.5367e-07 (1.00) | identical | identical | identical | identical |
| QCD | 3e8 | $G_k$ | 4.8828e-04 (1.00) | identical | identical | identical | identical |
| QCD | 3e8 | $T_k$ | 3.0518e-05 (1.00) | identical | identical | identical | identical |

Three readings, of which only the first was expected:

- **All six LambdaCDM rows are identical under every scheme** — it declares no break points, so
  `_cosmology_break_points` returns an empty array and every scheme is the identity.
- **The ten rows at 1.00 ulp stay there under every scheme**, `ALLx1` included. When prompt 02 of
  `prompts/phase-representation` measured, `ALLx1` broke two of them to 3.00 ulp because
  `BREAK_POINT_ALL` was 226–325 points on those grids; prompt 07 made it 1–3, and no scheme now
  touches a row other than the two at $k=10^5$.
- **Nothing reaches the target and nothing returns to the campaign base.** Against 1.907e-06 /
  3.186e-06 rad at the base and a 1e-06 rad / 2 ulp target, the best of the nine is 6.6757e-06 rad
  (28.00 ulp) and 1.1528e-05 rad (1547.2 ulp) — **1.21×** better than shipped, against the 4.25×
  and 4.39× that would have to be recovered. The two constructions the prompt names are worse.

**Why, mechanically.** An interpolating knot vector has fixed length $n+k+1$, and
Schoenberg–Whitney permits the three knots a multiplicity-3 knot consumes to be removed only
*locally*, from the intervals adjacent to the break — so the $C^0$ freedom is bought by coarsening
the spline exactly where $\varphi$ is least smooth. Per-segment splines steal no knots, but every
declared break lies strictly inside a grid interval (fractional position **0.6424** at `T_LO` on
both $k=10^5$ grids), so each side must extrapolate up to 0.64 of an interval to reach it; both
sectors' maxima move onto exactly that point, $z=4.25278\times10^{7}$ against $4.24388\times10^{7}$
for `base`.

### 8.2 Prompt 02's kink fit, re-taken on the corrected background

The `GkTk-remedial` board required this before any design: prompt 02's item 3 fitted the kink on a
step the old representation had smeared. One-sided cubics on $\varphi$ from the dense reference,
either side of `T_LO`, over windows of 1, 2 and 3 production grid intervals:

| window | $[\varphi']$, $G_k$ $k=10^5$ | $[\varphi']$, $T_k$ $k=10^5$ |
|---|---|---|
| 1 grid interval | +7.5232e-05 | −2.1692e-04 |
| 2 grid intervals | −3.0026e-03 | +5.1925e-03 |
| 3 grid intervals | −6.1529e-03 | +1.0634e-02 |

**A genuine slope discontinuity gives a window-independent jump. This does not** — it moves by two
orders and changes sign — at any window the production grid supports, on a background whose $T(z)$
error has fallen from 7.18e-04 to 6.807e-11. That is prompt 02's signature of *smooth-but-unresolved
data*, reproduced with the representation defect it was once attributed to removed, and it is the
quantitative reason a $C^0$ knot at the break cannot help: there is no resolved corner for it to
turn.

### 8.3 `theta_deriv` against $\omega$ — split and attributed

Deep interior, relative. `phi_zero` is the control that removes the $\varphi$ spline's derivative
altogether, so `theta_deriv` is the closed-form leading term alone; `$\varphi$ range` is the dynamic
range of the recovered $\varphi$ in ulp of the stored phase.

| model | $k$ | sector | base (§3.2) | best break-point scheme | `phi_zero` | $\varphi$ range [ulp] |
|---|---|---|---|---|---|---|
| LambdaCDM | all three | both | *(six rows)* | **identical** | — | — |
| QCD | 1e5 | $G_k$ | 1.0232e-06 | **6.5730e-07** (1.56×) | 9.9272e-06 | 1.12e+03 |
| QCD | 1e5 | $T_k$ | 3.0630e-06 | **1.9771e-06** (1.55×) | 4.6918e-03 | 1.18e+07 |
| QCD | 1e7 | $G_k$ | 6.0229e-06 | 6.0229e-06 — **no scheme moves it** | **3.9489e-06** | **6.0** |
| QCD | 1e7 | $T_k$ | 3.0630e-10 | 1.9770e-10 (1.55×) | 5.0634e-03 | 9.31e+04 |
| QCD | 3e8 | $G_k$ | 2.5418e-04 | 2.5418e-04 — **no scheme moves it** | **1.5976e-05** | **2.0** |
| QCD | 3e8 | $T_k$ | 7.3236e-06 | 1.2456e-05 — **1.70× worse** | 4.8542e-03 | 3.02e+03 |

- **The break points' share is 35.8 % ($G_k$) and 35.5 % ($T_k$), at $k=10^5$ only**, and costs
  2.09×/2.10× of consumer phase to buy. Prompt 02 measured 38 % at the same price on the defective
  background: that half of its finding is **unchanged** by the corrected one.
- **`[02-consumer-phi-below-the-storage-granularity]`'s share is 100 % of the two rows that miss
  $10^{-6}$ away from $k=10^5$.** At QCD $G_k$ $10^7$ and $3\times10^8$ the recovered $\varphi$
  spans **6.0** and **2.0 ulp** of the stored phase, no scheme moves either figure by a printed
  digit, and removing $\varphi'$ altogether *improves* them by **1.53×** and **15.9×**. Prompt 02's
  item 4 is reproduced exactly on a background seven orders more accurate, which is the strongest
  available evidence that it is neither the knots' nor the representation's. README §0.5 keeps it
  out of this campaign.
- **QCD $T_k$ at $3\times10^8$ is a regression** under both $C^0$ schemes — the trap prompt 02's log
  named ("a one-number acceptance test would have shipped it"), firing in a row it did not.

### 8.4 What does fix it: the resolution ladder

$\varphi$ reconstructed from the dense reference and splined with **default knots and no
break-point treatment at all**, given different *samples*. The stored node values are kept exactly
where they exist, so the first row is the shipped consumer. $h = 2.3030\times10^{-2}$ in $u$.

| samples given to a plain cubic | extra samples | QCD $G_k$ $k=10^5$ | QCD $T_k$ $k=10^5$ |
|---|---|---|---|
| **the production grid (shipped)** | — | 8.1329e-06 rad, **34.11 ulp** | 1.3982e-05 rad, **1876.61 ulp** |
| +4 inside the break's own interval alone | 4 | 17.42 ulp | 960.11 ulp |
| refine ±1 interval by 5× | 8 | 15.80 | 848.39 |
| refine ±2 intervals by 5× | 16 | 7.93 | 447.57 |
| refine ±3 intervals by 5× | 24 | 2.29 | 170.86 |
| **refine ±5 intervals by 2×** | **10 (1.0 %)** | **1.64** (3.9121e-07 rad) | **74.60** (5.5584e-07 rad) |
| refine ±5 intervals by 5× | 40 (3.9 %) | 1.64 | **26.97** (2.0096e-07 rad) |
| uniform 2× over the whole range | 1,014 | 1.48 | 72.76 |
| uniform 5× over the whole range | 4,059 | 0.95 | 2.87 |

**Three statements.**

1. **The feature is real and resolvable.** Doubling the grid takes $G_k$ to 1.48 ulp — the floor the
   other ten rows sit at — where the best knot vector leaves it at 28.00.
2. **It is not confined to the break's own interval.** Four extra samples inside that interval buy
   1.96× and stall; the error collapses only once ±3 to ±5 intervals are refined. That is the
   "±3 grid interval" arch structure prompt 02 measured, re-measured with the `T(z)` knot lattice
   once blamed for it gone — so it is $\varphi$'s own, as prompt 02 originally said.
3. **±5 intervals at 2× is the cheap fix, and it is a *grid* fix.** Ten extra samples in 1,016
   put both sectors inside the 1e-06 rad consumer target and $G_k$ inside the 2-ulp one. Nothing in
   `PrimitivePhase` can do this: the information is not in the node values it is given.

At $k=10^7$ and $3\times10^8$ the ladder reads **0.00 ulp near the break in both sectors at every
density** — the crossing contributes nothing there, which is the independent confirmation that it is
a $k=10^5$ phenomenon and that the $k\ge10^7$ `theta_deriv` misses belong to the other issue.

### 8.5 The verdict, and where the defect goes

**`PrimitivePhase` keeps `make_interp_spline`'s default knots, on measurement.** The residue at the
`T_LO` crossing is not the consumer spline's knot placement; it is the production source grid's
failure to resolve a feature three to five grid intervals wide, which is
`prompts/phase-representation/IMPLEMENTATION_STATE.md` §5 note 6 — *a knot vector cannot resolve a
break the sample grid does not resolve* — established on the corrected background rather than
inherited. `[13-consumer-spline-crosses-eos-break-points]` is closed on that measurement and
`[10-consumer-phi-unresolved-at-the-eos-crossing]` opens in its place, **assigned to prompt 11**,
which owns the source grid (**G2**).

**Cost.** No production code changed, so nothing moved; measured for the record, best of 9 at load
average 12.28: `PrimitivePhase(...)` builds in **3.18e-04 s** (LambdaCDM, 1,361 samples) and
**3.38e-04 s** (QCD, 1,377), **0 integrand evaluations** — $\varphi$ is supplied, and the
constructor is an `argsort`, a `log1p` and one `make_interp_spline`.

**Suites** at this commit: `CosmologyModels` **30**, `ComputeTargets` 359 → **361** (the two new
tests), `LiouvilleGreen` **143/143** on the fast set (`test_3bessel_analytic` excluded; the full set
is 148 and no file it touches is in this diff).

---

## 9. Prompt 11 — a source grid that knows the cosmology it samples (2026-09-15)

**Added by prompt 11 of `prompts/qcd-background-audit/`; §§1–8 above are untouched** (`CLAUDE.md`:
verification documents are additive). Reproduction, one command, ~32 s, no Ray and no datastore:

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/source_grid_consumer_check.py
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/source_grid_consumer_check.py \
    --k 1e5 1e7 3e8 --no-standoff-scan          # the all-three-wavenumbers control, ~110 s
```

That script does **not** re-implement §8's measurement. It imports `build_cases` and
`resolution_ladder` from `consumer_knot_scheme_scan.py` unchanged, and asserts on every run that
its own scoring closure reproduces `resolution_ladder`'s shipped row bit for bit before printing
anything; what it adds is the ability to score an *arbitrary* extra sample set, which is what is
needed to score the grid `CosmologyConcepts.wavenumber.build_z_sample` actually produces rather
than a ladder rung.

### 9.1 The grid

| | LambdaCDM | `QCD_Cosmology` |
|---|---|---|
| declared break points in range (`BREAK_POINT_ALL`) | 0 | **3** |
| feature redshifts (matter–radiation, matter–$\Lambda$) | 0 — see below | **2** |
| source grid | **1,732, bit-identical to today's `logspace`** | 1,732 → **1,773** (+41, **+2.37 %**) |
| protected samples | 0 | **8** (a pair per break, plus each feature) |
| response grid (`winnow(12)`) | **145**, unchanged | 148 under a blind stride → **156** |
| source tag | `SourceRedshiftGrid_1732` → `SourceRedshiftGrid_1732_0960e169` | `SourceRedshiftGrid_1773_303f9ce7` |
| response tag | `ResponseRedshiftGrid_145` → `ResponseRedshiftGrid_145_69050b4c` | `ResponseRedshiftGrid_156_197b46de` |

The 41 extra QCD samples are $3\times(2+11)+2$: per break, the **pair straddling it** at a quarter
of a grid interval, and the **11 intervals from $-5$ to $+5$ around it refined by 2**. No base
sample is displaced (`set(base) ⊆ set(new)`), the grid is strictly descending, and its closest
approach between neighbouring samples is **1.028e-03** relative in $(1+z)$ — four orders above the
1e-07 at which `Datastore/SQL/ObjectFactories/redshift.py` would treat two samples as the same row.

**LambdaCDM gets no feature samples, deliberately.** It has equality redshifts; nothing asks for
them. `main.py`'s `cosmology_feature_redshifts` returns an empty protected set for any cosmology
that declares no break points, so every LambdaCDM model, `RadiationModel` and test stand-in takes
the unchanged code path and its grid is bit-identical — README §0.5 and §2 (g), which are a stop
condition rather than an expectation. Two equality samples in 1,732 would buy no measured accuracy,
because the background is perfectly smooth at either equality.

### 9.2 The consumer at the crossing — and what the prompt's letter would have bought alone

`max |\varphi_{\rm spline} - \varphi_{\rm ref}|` at ten points per production interval, scored by
§8's scorer, at QCD $k=10^5$ where the `T_LO` crossing lives. The 1e-06 rad target is prompt 10 §5's.

| samples given to the production consumer | $G_k$ | $T_k$ |
|---|---|---|
| the production grid (shipped) | 8.1329e-06 rad, **34.11 ulp** | 1.3982e-05 rad, **1876.61 ulp** |
| **the straddling pair alone** — prompt §2 item 2's letter | 4.1563e-06, **17.43** — 1.96× | 7.0724e-06, **949.24** — 1.98× |
| the ±5 × 2 neighbourhood alone — prompt 10's row | 3.9121e-07, **1.64** | 5.5584e-07, **74.60** |
| **what this prompt ships: pair + ±5 × 2** | **3.8296e-07 rad, 1.61 ulp** | **4.9012e-07 rad, 65.78 ulp** |

**Two samples straddling each break are necessary and nowhere near sufficient**, and the second row
is the number that says so: on its own the pair buys **1.96× / 1.98×** and leaves both rows outside
the target, which is the same factor prompt 10 measured for four extra samples inside the break's
own interval. The feature is three to five grid intervals wide; a grid design that protects only
the declared point does not work, and this is that statement re-measured on the grid actually
built rather than on a ladder rung.

Both shipped rows are **inside the 1e-06 rad target**, $G_k$ also inside its 2-ulp one, and both
are better than the campaign base (1.9073e-06 / 3.1859e-06 rad) as well as than `HEAD`.

### 9.3 The standoff, scored rather than assumed

Fraction of a base grid interval; the shipped value is **0.25**, i.e. 5.82e-03 relative in $(1+z)$
on the production grid.

| standoff | $G_k$ $k=10^5$ | $T_k$ $k=10^5$ |
|---|---|---|
| 1/2 | 1.83 ulp | 74.35 ulp |
| **1/4 (shipped)** | **1.61** | **65.78** |
| 1/8 | 1.86 | 88.64 |
| 1/16 | 1.61 | 76.16 |
| 1/32 | 1.67 | 81.77 |
| 1/64 | 1.98 | 96.38 |
| 1e-3 | 2.53 | 123.65 |
| 1e-4 | 2.52 | 123.34 |
| *(no pair at all)* | *1.64* | *74.60* |

**1/4 is the only value that beats the no-pair column on both rows.** Below ~1/32 the pair stops
helping and then hurts, saturating 1.5×–1.7× worse than no pair at all: a pair a distance $d$ apart
implies a slope with an uncertainty $\sim2\,\mathrm{ulp}/d$ from the consumer's storage
granularity (`[02-consumer-phi-below-the-storage-granularity]`), and a cubic through it propagates
that to the neighbours. **That is why
`Quadrature/integrators/numeric_with_phase_cut.BREAK_POINT_STANDOFF = 1e-12` must not be borrowed
here**: it places an ODE restart, where nothing is interpolated and nothing is stored, and the
plateau at the bottom of this table is what it would deliver. Tying the standoff to the grid
spacing, rather than fixing it absolutely, is what keeps it above that floor at any grid density —
which matters for prompt 12, whose question is the density.

### 9.4 All three wavenumbers, both sectors — the trap prompts 02 and 10 both warn about

`--k 1e5 1e7 3e8`. "near break" is the worst within three grid intervals of the crossing.

| model | $k$ | sector | shipped grid | cosmology-aware grid |
|---|---|---|---|---|
| QCD | 1e5 | $G_k$ | 8.1329e-06 (34.11 ulp; near 34.11) | **3.8296e-07 (1.61; near 1.61)** |
| QCD | 1e5 | $T_k$ | 1.3982e-05 (1876.61; near 1876.61) | **4.9012e-07 (65.78; near 65.78)** |
| QCD | 1e7 | $G_k$ | 1.4251e-05 (0.93; near 0.00) | 1.4251e-05 (0.93; near 0.00) — identical |
| QCD | 1e7 | $T_k$ | 1.2028e-06 (1.26; near 0.15) | 1.2028e-06 (1.26; **near 0.01**) |
| QCD | 3e8 | $G_k$ | 3.2200e-04 (0.66; near 0.00) | 3.2200e-04 (0.66; near 0.00) — identical |
| QCD | 3e8 | $T_k$ | 3.8674e-05 (1.27; near 0.00) | 3.8999e-05 (**1.28**; near 0.00) |
| LambdaCDM | all three | both | — | **bit-identical grid, so bit-identical rows** |

**One row moves the wrong way and it is recorded rather than argued away**: QCD $T_k$ at
$3\times10^8$ goes 3.8674e-05 → 3.8999e-05 rad, **+0.84 %**, 1.27 → 1.28 ulp of the span. It is not
at the crossing (`near break` is 0.00 both times, and the pair-alone column reproduces it exactly,
so it is the pair), and 1 ulp of the span is the floor ten of the twelve production rows already
sit at. For scale, the $C^0$ knot schemes prompt 10 scored moved this same row by **1.70×**.

### 9.5 `[03-derivative-pad-clamp-on-coarse-grids]` — measured, and it does not bind

Prompt §2 item 5. `_derivative_fit_grid`'s low-end padding is `h_lo = min(x[1]-x[0],
-log(0.9)/12, 0.05 (x[-1]-x[0])/12)` in $x=\log(1+z)$, and the clamp binds when the first term is
not the smallest.

| grid | first $x$ interval | floor cap | fraction cap | `h_lo` | binds |
|---|---|---|---|---|---|
| base (1,732) | 2.115878e-03 | 8.780043e-03 | 1.561272e-01 | 2.115878e-03 | no |
| cosmology-aware (1,773) | 2.115878e-03 | 8.780043e-03 | 1.561272e-01 | 2.115878e-03 | no |

The two are the same float. The lowest protected sample is matter–$\Lambda$ equality at
$z=0.3034$, **48.19 base intervals above** $z_{\rm end}=0.1$, and the bottom 40 samples of the
cosmology-aware grid are element-for-element the base grid's, so neither $z_{\rm min}$ nor the
spacing at either end moves. Nothing was changed, as the prompt requires.

### 9.6 The grid tag, and what changing it invalidates

`SourceRedshiftGrid_{len}` labelled size alone; it is now `SourceRedshiftGrid_{len}_{digest}`, the
digest being `blake2b` over the exact bits of the grid's own values (8 hex characters).
`ResponseRedshiftGrid_` likewise. Two grids of equal length that differ in any sample — including
in the protected set alone — now get different tags, which is the collision the audit §7 names.

**The bill.** Eight `pool.object_get` call sites in `main.py` filter on these tags, across **seven
stored object types**: `TkNumericIntegration`, `TkWKBIntegration` and `QuadSource` (source tag
only); `GkNumericIntegration`, `GkWKBIntegration` (two sites), `GkSource` and `QuadSourceIntegral`
(both tags). Their factories treat the sample grid as *"a target rather than a selection
criterion"* (`Datastore/SQL/ObjectFactories/GkNumericIntegration.py:200`), so **the tag is the only
thing that identifies which grid an object was computed on**, and every stored object of those
seven types under the old labels becomes unfindable. `GkSourcePolicyData` carries no grid tag but
is keyed on `GkSource`, so it goes with them. `BackgroundModel`, `wavenumber_exit_time`,
`redshift`, `bessel_phase`, `tolerance`, `IntegrationSolver` and `store_tag` are **not** filtered
on these tags and survive — which for `BackgroundModel` is a hazard in its own right, recorded as
`[11-background-model-not-keyed-on-the-source-grid]`.

What that costs, from the only measured production-shaped run in the tree
(`docs/gktk-remedial-verification.md` §4.2: LambdaCDM, 5 source and 5 response wavenumbers, a
1,584-node grid):

| type | objects | wall |
|---|---|---|
| `TkNumericIntegration` | 5 | 1.15 s |
| `TkWKBIntegration` | 5 | 0.669 s |
| `QuadSource` | 15 | 0.714 s |
| `GkNumericIntegration` | 2,455 | 54.9 s |
| `GkWKBIntegration` | 7,920 | 3 m 59.9 s |
| `GkSource` | 660 | 1 m 18.2 s |
| `GkSourcePolicyData` (transitive) | 660 | 19.2 s |
| `QuadSourceIntegral` | 3,300 | **stopped after ~3 h, incomplete** |
| **total** | **15,020** | **6 m 35 s plus an unfinished `QuadSourceIntegral` stage** |

Production is **50 source and 50 response wavenumbers**, ten times that run in each, and two
models. The per-$k$ stages scale ×10; `QuadSource` runs over unordered pairs, $\binom{51}{2}=1275$
against 15, so ×85; `QuadSourceIntegral` scales with the pairs *and* the response wavenumbers, of
order ×850 on a stage that already failed to finish 3,300 objects in three hours. **This is an
extrapolation from a measured run, not a measurement**; the honest summary is that a pre-prompt-11
datastore is regenerated from `TkNumericIntegration` downwards for both models, and that the
`QuadSourceIntegral` stage is what the bill is made of.

**One of the two halves of that bill is avoidable, and the reader should know it.** The QCD half
was invalidated anyway — its grid is 1,773 samples where it was 1,732, so the old tag would not
have matched on length alone even without the digest, and prompt 03's `T_z_representation` key had
already invalidated its cosmology row for a different reason. The **LambdaCDM half is invalidated
by the tag change and by nothing else**: its grid is bit-identical. Because `store_tag` is keyed on
its label and is a replicated table, a datastore holding **only** LambdaCDM objects can be carried
across by relabelling in place rather than recomputing:

```sql
UPDATE store_tag SET label='SourceRedshiftGrid_1732_0960e169'  WHERE label='SourceRedshiftGrid_1732';
UPDATE store_tag SET label='ResponseRedshiftGrid_145_69050b4c' WHERE label='ResponseRedshiftGrid_145';
```

**Do not run that on a datastore that also holds pre-prompt-11 QCD objects**: they share the
`SourceRedshiftGrid_1732` row, and relabelling it would re-validate QCD objects computed on a grid
that no longer exists — precisely the silent staleness the digest is for. Drop them first.

### 9.7 Suites

| suite | before | after | command |
|---|---|---|---|
| `CosmologyModels/tests` | 30 OK | **30 OK** (0.55 s) | `discover -s CosmologyModels/tests -t .` |
| `ComputeTargets/tests` | 361 OK | **380 OK** (151.3 s) | `discover -s ComputeTargets/tests -t .` |
| `LiouvilleGreen/tests` | 143 OK (fast set) | **143 OK** (14.5 s) | every module except `test_3bessel_analytic` |

The `LiouvilleGreen` figure is the fast set, as prompts 02–08 and 10 used; the full set is 148
(prompt 09 ran it) and this commit touches no `LiouvilleGreen` file. The +19 are
`ComputeTargets/tests/test_source_grid.py`. `black` is clean on every file in the diff.

---

## 10. Prompt 12 — what the grid's density buys, and what a criterion on $\varphi$'s curvature would buy (2026-09-15)

**Added by prompt 12 of `prompts/qcd-background-audit/`; §§1–9 above are untouched** (`CLAUDE.md`:
verification documents are additive). **No production file is in this prompt's diff**, and
`T_Z_REPRESENTATION_VERSION` is **5** before and after. Reproduction, one command, **88 s**, no Ray
and no datastore, at load average 4.6–7.3:

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py \
    --models QCDModel --k 1e5 --sections A B     # a narrower slice
```

Every figure below was taken twice on this tree and is **bit-identical between the two runs** apart
from wall clocks.

### 10.0 The recommendation, and the ceiling it carries

**Read this paragraph with `docs/OPEN_ISSUES.md` §5 in front of you.** *The verification runs never
reached production $x$*: `source-remediation`'s run A used `zend = 1e7` to stay inside radiation
domination, so its largest accumulated phase was $x = 4.63\times10^5$ against $x\sim1.4\times10^7$
at the production `zend = 0.1` for the largest $k$. **This section is measured at production
`zend = 0.1` and production `z_init`**, against an oracle that is a Gauss-quadrature table rather
than a pipeline run, so it is not itself limited by that ceiling — but **no pipeline has ever been
run on any grid discussed here, including the one that ships**, and the §5 caveat therefore applies
in full to any claim that a re-gridded run would behave as these numbers predict.

Three statements, in the order of their confidence:

1. **Away from the equation of state's declared crossings, the shipped grid is adequate at
   $k\ge10^7$ and misses by 7.9× at $k=10^5$ in the transfer-function sector.** The one place it is
   under-resolved is the **top decade of each Liouville–Green band**, just inside horizon entry;
   everywhere below that it is over-resolved, by up to **$10^{19}$** in error at the bottom of the
   range. A uniform density over a curvature that spans twenty orders is wrong in both directions
   at once, and §10.2 is that statement in numbers.
2. **The criterion that works is fourth-derivative equidistribution,
   $h(u)^4\,|\varphi''''(u)|/384 \le \varepsilon$, and it is computable before the grid exists**
   from $H$, $c_s^2$ and $k$ alone (§10.1, §10.3). At the **same** sample count it puts every
   production row inside its floor where the shipped grid misses two of them; at the **same**
   accuracy it needs **1.75×** fewer samples on QCD and **2.06×** fewer on LambdaCDM (§10.5).
3. **The saving and `[03-derivative-pad-clamp-on-coarse-grids]` are the same lever.** The whole of
   the saving is coarsening at low $z$, and the derivative-fit padding clamps the moment the lowest
   grid interval exceeds $-\log(0.9)/12 = 8.7800\times10^{-3}$ in $u$ — which is exactly when the
   coarsening begins. At the "never coarser than today" cap the clamp does not bind and the saving
   is 1 %; at every coarser cap the clamp binds. **A density change is therefore not free of that
   issue, it is the thing that triggers it.**

**What is not measured here, and bounds all three.** The source grid carries far more than
$\varphi$: the numeric ODE's sample points, the four `CumulativeTable`s' Gauss panels, and the
abscissae `QuadSource` and `QuadSourceIntegral` integrate the source over. **None of those
requirements is measured in this section**, and a criterion derived from $\varphi$ alone is a lower
bound on the density, never an upper one. That is why §10.5 reports a *cap ladder* rather than a
single number: the cap is where the unmeasured consumers would have to be let in.

**The decision is the user's**, and it costs a full datastore regeneration — eight stored object
types, quantified in §9.6 — because every one of them is keyed on the grid tag.

### 10.1 The oracle, and why it is not the stored phase

Prompts 10 and 11 scored $\varphi$ recovered from a **stored** $\theta$, which carries the
$\epsilon k\tau$ rounding of `[02-consumer-phi-below-the-storage-granularity]`. That floor is
precisely what a *density* measurement has to see past. So the reference here is the phase residual
itself,

$$\varphi(z) \;=\; \rho(z) + \text{const},\qquad \rho(z)=\int_z^{z_{\rm top}}\frac{C}{\omega+\omega_0}\,\mathrm{d}z',$$

read off `ComputeTargets.phase_residual.build_phase_residual`'s `CumulativeTable`, whose `delta`
does genuine Gauss quadrature between arbitrary endpoints rather than interpolating. `PrimitivePhase`'s
own module docstring is the warrant: $\varphi$ is $\rho$ plus the producer's constant offsets, and a
constant is invisible to an interpolation error. The storage floor then enters only where it
belongs — as the line below which no grid decision is visible.

**The criterion's inputs are pre-grid, which is the whole point.** `populate_z_sample` runs before
any `BackgroundModel` exists, so a criterion may use only what the *cosmology* supplies pointwise.
It supplies enough:

$$\frac{\mathrm{d}\varphi}{\mathrm{d}u} \;=\; -(1+z)\,\frac{C(z)}{\omega(z,k)+\omega_0(z,k)},\qquad u=\log(1+z)$$

with $\omega_0^2 = w\,(k/H)^2$ ($w=1$ in the $G_k$ sector, $w=c_s^2$ in $T_k$) and $C$ built from
$H$, $c_s^2$ and their $z$-derivatives. Two checks license it:

| check | result |
|---|---|
| the closed form against the table's own numerical derivative, mid-band, twelve (model, sector, $k$) | agrees to **2.4e-11 … 1.2e-05** relative, the larger figures being the central difference's own noise where $\varphi'\sim10^{-11}$ |
| $\epsilon$ as the **model** supplies it (a spline over the grid's own samples on QCD) against $\epsilon$ from a central difference of the cosmology's pointwise `Hubble` — what a pre-grid criterion would have to use | **max 2.5e-09 (LambdaCDM), 3.9e-09 (QCD)**, median 8.1e-10, away from a declared crossing |

So a criterion evaluated from finite differences of the cosmology's own $H$ and $c_s^2$ is, away
from a crossing, the same criterion measured here to nine digits. It costs $5N$ closed-form
evaluations for an $N$-point profile: **0.01–0.27 s** for the production profile, against the
**0.599 s** a QCD `compute_background` takes (§4.2).

### 10.2 Sub-question 1 — what the shipped grid delivers, per decade

Interpolation error of a cubic of $\varphi$ on the grid prompt 11 ships, scored at four points
inside every production interval. "floor" is 1 ulp of the $k\tau$ span; "away" excludes 0.15 in $u$
either side of a declared crossing, because those neighbourhoods are prompt 11's local mechanism and
not a density.

| model | sector | $k$ | band samples | max, all | max, away | floor (1 ulp) | away / floor |
|---|---|---|---|---|---|---|---|
| LambdaCDM | $G_k$ | 1e5 | 1,732 | 2.7680e-12 | 2.7680e-12 | 2.3842e-07 | 1.16e-05 |
| LambdaCDM | $G_k$ | 1e7 | 1,732 | 2.7458e-14 | 2.7458e-14 | 1.5259e-05 | 1.80e-09 |
| LambdaCDM | $G_k$ | 3e8 | 1,732 | 9.1527e-16 | 9.1527e-16 | 4.8828e-04 | 1.87e-12 |
| LambdaCDM | $T_k$ | 1e5 | 1,113 | 5.8576e-08 | 5.8576e-08 | 7.4506e-09 | **7.86** |
| LambdaCDM | $T_k$ | 1e7 | 1,313 | 5.9076e-08 | 5.9076e-08 | 9.5367e-07 | 6.19e-02 |
| LambdaCDM | $T_k$ | 3e8 | 1,460 | 5.3192e-08 | 5.3192e-08 | 3.0518e-05 | 1.74e-03 |
| QCD | $G_k$ | 1e5 | 1,271 | 1.4356e-05 | 1.6782e-08 | 2.3842e-07 | 7.04e-02 |
| QCD | $G_k$ | 1e7 | 1,509 | 2.1574e-04 | 9.1102e-10 | 1.5259e-05 | 5.97e-05 |
| QCD | $G_k$ | 3e8 | 1,644 | 7.1558e-06 | 5.1456e-10 | 4.8828e-04 | 1.05e-06 |
| QCD | $T_k$ | 1e5 | 1,145 | 2.4859e-05 | 5.8437e-08 | 7.4506e-09 | **7.84** |
| QCD | $T_k$ | 1e7 | 1,360 | 4.4783e-04 | 1.5453e-08 | 9.5367e-07 | 1.62e-02 |
| QCD | $T_k$ | 3e8 | 1,520 | 1.3272e-05 | 5.8675e-08 | 3.0518e-05 | 1.92e-03 |

**Two rows miss, and they are the same row on both models**: the transfer-function sector at
$k=10^5$, **7.86** and **7.84** ulp, and in both the maximum sits in the **top decade of the band**
($z=1.32\times10^{10}$ and $1.46\times10^{10}$), within a few grid intervals of horizon entry. Every
other row is inside its floor, most of them by many orders.

**Where the error lives, and why a uniform grid is wrong in both directions.** Per decade, QCD
$T_k$ at $k=10^5$ (the spacing $h_u$ is the grid's own, and "headroom" is floor / max-away):

| decade of $z$ | $h_u$ | max away | headroom |
|---|---|---|---|
| −1 | 5.4866e-03 | 3.3307e-16 | 2.24e+07 |
| 0 … 3 | 1.75e-02 … 2.30e-02 | 2.2e-16 … 5.6e-16 | 1.3e+07 … 3.4e+07 |
| 4 | 2.3031e-02 | 2.5535e-15 | 2.92e+06 |
| 5 | 2.3032e-02 | 2.5202e-14 | 2.96e+05 |
| 6 | 2.3032e-02 | 2.5180e-13 | 2.96e+04 |
| 7 | 2.3032e-02 | 4.6224e-10 | 1.61e+01 |
| 8 | 2.3032e-02 | 2.2879e-11 | 3.26e+02 |
| 9 | 2.3032e-02 | 9.1955e-10 | 8.10e+00 |
| 10 | 1.1516e-02 | **5.8437e-08** | **1.27e-01** |

The error falls by about one order per decade downwards and spans **eight orders within a single
band**, while the spacing is constant to four digits over fourteen of those decades. Below $z\sim1$
the grid is additionally *finer* in $u$ — 5.49e-03 against 2.30e-02, because `populate_z_sample` is
log-spaced in $z$ and not in $1+z$ — so the density is at its highest exactly where the curvature is
at its lowest. On LambdaCDM $G_k$ the whole band has headroom between **8.6e+04** and **2.1e+19**:
$\varphi$ there is the residual of an equation of state that is radiation to a part in $10^4$, and
**no sample density is doing any work at all**.

**Where the criterion stops meaning anything.** The floor grows like $k$ (it is
$\epsilon\,k\tau$) while $\varphi$ and all its derivatives fall like $1/k$ (the residual is
$\int C/(\omega+\omega_0)$ and the denominator is $\sim 2k/H$ at large $k$), so the requirement and
the floor move apart by $k^2$. That is visible in the table: the $T_k$ "away / floor" column reads
**7.84** at $k=10^5$, **1.62e-02** at $10^7$ and **1.92e-03** at $3\times10^8$. Prompt 10 measured
the other end of the same fact — the recovered $\varphi$ spans **6.0 ulp** of the stored phase at
$k=10^7$ and **2.0 ulp** (three distinct values over 1,377 samples) at $3\times10^8$. **A density
criterion derived at $k=3\times10^8$ would be fitting rounding.** Everything below is derived at
$k=10^5$, where $\varphi$ has dynamic range, and the honest statement of its reach is: **above
$k\approx10^7$ the grid's density is not what limits the consumer, and no grid decision is visible
there.**

### 10.3 Sub-question 2 — the criterion, and whether it predicts

$$\boxed{\;h(u)^4\,\frac{|\varphi''''(u)|}{384}\;\le\;\varepsilon\;}\qquad
\varphi'(u) = -(1+z)\frac{C(z)}{\omega+\omega_0},\quad \varphi''''=\left(\varphi'\right)'''$$

with the third derivative taken by a five-point central stencil of step $10^{-3}$ in $u$ on the
closed form — four evaluations of $H$, $c_s^2$ and their $z$-derivatives per point, no grid, no
stored phase, no `BackgroundModel`.

Realised error against predicted, per production interval, away from a declared crossing and above
the 1e-15 rad at which the oracle itself runs out:

| model | sector | $k$ | intervals | p10 | median | p90 | spread |
|---|---|---|---|---|---|---|---|
| LambdaCDM | $G_k$ | 1e5 | 313 | 0.003 | 0.010 | 0.046 | 15.3× |
| LambdaCDM | $G_k$ | 1e7 | 119 | 0.003 | 0.011 | 0.052 | 16.2× |
| LambdaCDM | $G_k$ | 3e8 | — | *fewer than ten usable intervals* | | | |
| LambdaCDM | $T_k$ | 1e5 | 551 | 0.921 | **0.923** | 0.934 | **1.0×** |
| LambdaCDM | $T_k$ | 1e7 | 551 | 0.922 | **0.923** | 0.932 | **1.0×** |
| LambdaCDM | $T_k$ | 3e8 | 549 | 0.922 | **0.923** | 0.933 | **1.0×** |
| QCD | $G_k$ | 1e5 | 365 | 0.952 | 1.247 | 9.457 | 9.9× |
| QCD | $G_k$ | 1e7 | 520 | 0.948 | 1.054 | 10.396 | 11.0× |
| QCD | $G_k$ | 3e8 | 442 | 0.949 | 1.016 | 3.241 | 3.4× |
| QCD | $T_k$ | 1e5 | 529 | 0.922 | **0.928** | 0.943 | **1.0×** |
| QCD | $T_k$ | 1e7 | 517 | 0.923 | **0.928** | 0.955 | **1.0×** |
| QCD | $T_k$ | 3e8 | 529 | 0.922 | **0.927** | 0.954 | **1.0×** |

**In the transfer-function sector the indicator predicts the realised error to ±2 % over 500-odd
intervals, on both models, at all three wavenumbers, with the textbook constant $1/384$ and no
fitting.** In the QCD Green's-function sector the median is right and the tail reaches 10×. On
LambdaCDM $G_k$ the indicator **over**-predicts by about 100× and that is the stencil, not the
criterion: $\varphi'\sim10^{-8}$ there and the third difference divides by $2\delta^3 = 2\times10^{-9}$,
so $|\varphi''''|$ is roundoff-dominated. It errs towards asking for *more* samples than are needed,
which is the safe direction; a larger stencil step or an analytic third derivative would remove it,
and neither is needed for a criterion whose answer on that row is "any density will do".

**A second candidate, and it is refuted.** `SLOPE` — equidistribute $\varphi$ itself,
$h(u)\,|\varphi'(u)| = \delta$, one derivative and no numerical differentiation at all — is the
cheap alternative, and it does not work. It places density where $\varphi$ *moves* rather than where
it *bends*: on the production envelope it needs **113,000–114,000** samples against the shipped
1,773 (§10.5) and still misses two rows, and per case it fails outright on QCD $G_k$ at $k=10^5$ and
$3\times10^8$ ("never meets the target at any scale") while needing 15,463 samples for QCD $T_k$ at
$k=10^5$ where `CURV` needs 295. It is recorded because "equidistribute the function" is the obvious
first thing to try and the measurement says not to.

### 10.4 The crossings are excluded, and why — a finding prompt 11 could not have seen

Every table above separates the 0.15 in $u$ either side of a declared crossing. That is not
bookkeeping. On `QCD_Cosmology`, `epsilon`, `d_epsilon_dz` and `d2_epsilon_dz2` are **not** cosmology
methods: `BackgroundModel._create_functions` finds no `d_lnH_dz` on the cosmology and builds one as a
quintic spline of $\log H$ over `_build_derivative_fit_grid(z_sample)` — a padded, 3× refined copy of
**the source grid**, which is **not** split at the equation of state's break points. So $\omega_{\rm eff}$
itself is a function of the source grid, and at a genuine step in $H$ that spline rings.

$\epsilon$ from the model against $\epsilon$ from a central difference of the cosmology's own
`Hubble`, at offsets from each declared crossing, on the background built on the **base** 1,732-point
grid and on the one built on the **shipped** 1,773-point grid:

| crossing | worst relative disagreement, base-grid background | shipped-grid background |
|---|---|---|
| `T_LO`, $u=17.565806942$ ($z=4.25\times10^7$) | **3.66e-02** at $\mathrm{d}u=-0.005$ | **2.04e-02** at $+0.005$ |
| `EOS_T_LO`, $u=23.197460553$ ($z=1.19\times10^{10}$) | 1.9e-09 | 1.6e-09 |
| `T_120_MEV`, $u=27.485391822$ ($z=8.64\times10^{11}$) | **6.48e-03** at $+0.005$ | **1.03e-03** at $-0.005$ |

The middle row is the control and it is the campaign's own physics: `EOS_T_LO` is where $g_s$ is
continuous to 1.8e-11 and only $w$ kinks (README §7 D4), so $H$ does not step and nothing rings.
At the two genuine steps the ringing is real, and prompt 11's refinement **improves its amplitude by
1.8× and 6.3×**.

**But the amplitude is not what the consumer's spline sees.** Four combinations of
(background built on which grid) × (samples taken from which grid), QCD $T_k$ at $k=10^5$, and the
same pattern in all six QCD cases:

| background built on | samples from | band $n$ | max | near a crossing | away from one |
|---|---|---|---|---|---|
| base | base | 1,117 | 1.4055e-05 | 1.4055e-05 | 1.0523e-07 |
| base | **shipped** | 1,145 | **3.9744e-07** | 3.9744e-07 | 5.8439e-08 |
| **shipped** | base | 1,117 | 1.1851e-04 | 1.1851e-04 | 5.9047e-08 |
| **shipped** | **shipped** | 1,145 | **2.4859e-05** | 2.4859e-05 | 5.8437e-08 |

Row 2 is **prompt 11's configuration** — that prompt added samples to the consumer's set while its
`qcd_model_with_tables` was built on `production_source_grid`, the base grid — and it reproduces
prompt 11's result on a completely different oracle: **1.4055e-05 → 3.9744e-07**, against §9.2's
1.3982e-05 → 4.9012e-07 (1876.61 → 65.78 ulp). The two starting values agree to **0.5 %** and the
two end values to **23 %**, a factor of 35.4 here against 28.5 there — close enough that the two
measurements are of the same thing, and different enough to say that they are independent.
**Row 4 is what production will do**, because `main.py` passes
`z_sample=z_source_sample` to `BackgroundModel` and prompt 11 changed that grid: 1.41e-05 →
**2.49e-05**, i.e. **1.77× worse, not 35× better**. The mechanism is that refining the lattice at a
step makes the ringing lower **and narrower**, and a cubic through samples half a base interval apart
resolves a narrower feature worse; the stacked `d2_lnH_dz2` and `d3_lnH_dz3` amplify what `d_lnH_dz`
leaves behind.

**What is unambiguous, and holds in every case, is the "away" column**: 1.05e-07 → 5.84e-08 here,
3.67e-07 → 9.11e-10 for QCD $G_k$ at $k=10^7$. **Away from a crossing prompt 11's grid is better on
every row.** This section makes no claim that prompt 11 should have done something else — its
neighbourhood refinement is what §10.5's criterion also asks for, and it was measured correctly for
the configuration it measured. It says that the residue at a crossing is **the background's own
derivative fields**, not the sample density, and that no density criterion can reach it. That is
`[12-background-derivative-fit-grid-rings-at-a-step]` below.

### 10.5 Sub-question 3 — what it would cost, and what it would save

`main.py` builds **one universal source grid**, so a criterion has to be the envelope over every
(sector, $k$) it serves: each case is tuned to its own target
$\min(1\ \text{ulp},\ 10^{-6}\ \text{rad})$ on its own band, and the grid takes the smallest spacing
any case asks for. Above the highest band no case constrains anything and the spacing is the cap.
The march is a forward Euler on $\mathrm{d}u/\mathrm{d}i = h(u)$ — what a practical
`populate_z_sample` would do, not an optimal grid — so the counts are the simple implementation's,
and the ratios below are non-monotone in the cap for that reason.

**QCD, `CURV`, all six cases scored on one grid:**

| cap on $h$ | grid | vs shipped 1,773 | worst row / target | which row | `[03-derivative-pad-…]` |
|---|---|---|---|---|---|
| **1× today** | **1,761** | 1.01× | **0.14** | $T_k$ 1e5 | **does not bind** |
| 2× | **1,015** | **1.75×** | **0.21** | $T_k$ 1e5 | BINDS |
| 4× | 703 | 2.52× | 1.65 ✗ | $T_k$ 1e5 | BINDS |
| 8× | 588 | 3.02× | 0.92 | $T_k$ 1e5 | BINDS |
| 32× | 521 | 3.40× | 4.17 ✗ | $T_k$ 1e5 | BINDS |

**LambdaCDM, `CURV`:**

| cap on $h$ | grid | vs shipped 1,732 | worst row / target | which row | `[03-derivative-pad-…]` |
|---|---|---|---|---|---|
| **1× today** | **1,634** | 1.06× | **0.34** | $T_k$ 1e5 | **does not bind** |
| 2× | **842** | **2.06×** | 1.09 ✗ (9 %) | $T_k$ 1e5 | BINDS |
| 4× | 471 | 3.68× | 5.09 ✗ | $T_k$ 1e7 | BINDS |
| 8× | 307 | 5.64× | 1.06 ✗ | $T_k$ 1e7 | BINDS |
| 32× | 218 | 7.94× | 1.35 ✗ | $T_k$ 1e7 | BINDS |

**`SLOPE` on the same envelope: 114,281 samples (QCD) and 4,229 (LambdaCDM) at the 1× cap**, 64×
and 2.4× *more* than the grids they replace, and above the 4× cap it misses by up to 1,723× (QCD
$G_k$, $3\times10^8$). Refuted.

**Accuracy at a fixed sample count**, per case, `CURV` tuned to the shipped band count:

| model | sector | $k$ | shipped $n$ | shipped max | `CURV` at the same $n$ | ratio |
|---|---|---|---|---|---|---|
| LambdaCDM | $T_k$ | 1e5 | 1,113 | 5.8576e-08 | 1,113 → **2.1427e-12** | 27,337× |
| LambdaCDM | $T_k$ | 1e7 | 1,313 | 5.9076e-08 | 1,313 → **2.0728e-13** | 285,009× |
| QCD | $G_k$ | 1e5 | 1,271 | 1.4356e-05 | 1,272 → **1.4881e-11** | 964,709× |
| QCD | $T_k$ | 1e5 | 1,145 | 2.4859e-05 | 1,145 → **1.0849e-11** | 2,291,320× |
| QCD | $T_k$ | 1e7 | 1,360 | 4.4783e-04 | 1,360 → **9.7801e-11** | 4,578,986× |

(The shipped-grid figures in that table include the crossing residue of §10.4, which `CURV` at the
same count does not have to carry because it is scored away from a crossing; the honest comparison
of the *density* alone is the envelope table above.)

**Samples at a fixed accuracy**, per case, against the two uniform controls:

| model | sector | $k$ | shipped $n$ | `CURV` | uniform in $u$ | uniform in $\log_{10} z$ |
|---|---|---|---|---|---|---|
| LambdaCDM | $G_k$ | all three | 1,732 | 164 (cap-limited) | 8 | 8 |
| LambdaCDM | $T_k$ | 1e5 | 1,113 | **204** | 1,720 | 1,898 |
| LambdaCDM | $T_k$ | 1e7 | 1,313 | **128** | 580 | 629 |
| LambdaCDM | $T_k$ | 3e8 | 1,460 | **144** | 625 | 673 |
| QCD | $G_k$ | 1e5 | 1,271 | **295** | 548 | 706 |
| QCD | $G_k$ | 1e7 | 1,509 | **221** | 841 | 838 |
| QCD | $G_k$ | 3e8 | 1,644 | **226** | 399 | 387 |
| QCD | $T_k$ | 1e5 | 1,145 | **295** | 1,730 | 1,908 |
| QCD | $T_k$ | 1e7 | 1,360 | **173** | 859 | 932 |
| QCD | $T_k$ | 3e8 | 1,520 | **182** | 650 | 699 |

The two $T_k$ rows at $k=10^5$ are the ones that matter: **uniform in $\log_{10} z$ needs 1,898 and
1,908 band samples to reach the floor where the grid has 1,113 and 1,145** — i.e. today's
`source_samples_per_log10z = 100` would have to become about **167** to meet it uniformly, against
**204 and 295** samples for the whole band under the criterion.

### 10.6 The two constraints the answer had to respect

**The response grid stays a decimation of the source grid.** It is built by a blind stride
`[::-n]`, which is a subset of *any* source grid by construction; it was checked on every candidate
grid in §10.5 and is `True` in all twenty. The counts scale with the source grid: QCD's response
grid goes 147 (cap 1×) → 85 (2×) → 44 (32×) against the 148 a blind stride gives on the shipped
grid (production is 156 with prompt 11's `protect=`). **A response grid of 44 samples over twenty
decades is almost certainly not usable**, which is a second reason the cap ladder stops well short
of its coarse end — and it is not a $\varphi$ question, so this section does not answer it.

**The large-$k$ floor.** §10.2's last paragraph, restated here because it bounds the criterion:
above $k\approx10^7$ the recovered $\varphi$ spans 6.0 ulp of the stored phase and at
$3\times10^8$ it spans 2.0 ulp, so the measured "error" there is quantisation. Every scale in §10.5
is set by the $k=10^5$ rows, and the $10^7$ and $3\times10^8$ rows are carried along, never binding.

### 10.7 The $k\tau$ oscillation, written down so it is not forgotten again

Audit §9's fourth bullet: *"The response grid and the $k\tau$ oscillation an $\Omega_{\rm GW}$
post-processing step would need to resolve in order to fit an RMS amplitude. Not yet designed;
noted so it is not forgotten."* Measured, on the shipped grids, response grid by a blind stride of
12:

| model | $k$ | $k\tau$ over the whole range | per response interval, median | per response interval, max | response samples for 4 per cycle | shortfall |
|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | 1.3728e+09 rad | 2.9562e+02 | 8.1634e+07 | 8.740e+08 | **6.03e+06×** |
| LambdaCDM | 1e7 | 1.3728e+11 | 2.9562e+04 | 8.1634e+09 | 8.740e+10 | **6.03e+08×** |
| LambdaCDM | 3e8 | 4.1184e+12 | 8.8687e+05 | 2.4490e+11 | 2.622e+12 | **1.81e+10×** |
| QCD | 1e5 | 1.3728e+09 | 1.8536e+02 | 8.1630e+07 | 8.740e+08 | **5.91e+06×** |
| QCD | 1e7 | 1.3728e+11 | 1.8536e+04 | 8.1630e+09 | 8.740e+10 | **5.91e+08×** |
| QCD | 3e8 | 4.1184e+12 | 5.5608e+05 | 2.4489e+11 | 2.622e+12 | **1.77e+10×** |

**The conclusion is not a grid size; it is that no grid size exists.** Even at the *smallest*
production wavenumber and the *median* response interval, the phase advances **185–296 radians**
— some 30 to 47 complete cycles — between neighbouring response samples, and resolving the
oscillation pointwise at four samples per cycle would need between $6\times10^6$ and
$1.8\times10^{10}$ times the response samples the grid carries. **An $\Omega_{\rm GW}$
post-processing step must therefore obtain its RMS amplitude analytically — from the phase object
and an oscillatory quadrature, which is what `AdaptiveLevin` already does upstream — and can never
obtain it by sampling the response grid more finely.** Any future grid decision should be taken with
that in view: the response grid's density is set by what the *envelope* of $\Omega_{\rm GW}$ needs,
never by its oscillation, and nothing in this campaign has measured the envelope's requirement.

### 10.8 Suites

| suite | before (`4f725c0`) | after | command |
|---|---|---|---|
| `CosmologyModels/tests` | 30 OK | **30 OK** (0.576 s) | `discover -s CosmologyModels/tests -t .` |
| `ComputeTargets/tests` | 380 OK | **380 OK** (153.568 s) | `discover -s ComputeTargets/tests -t .` |
| `LiouvilleGreen/tests` | 143 OK (fast set) | **143 OK** (13.923 s, at load average 9.7) | every module except `test_3bessel_analytic` |

No production file, no test file and no fixture is in this prompt's diff, so "before" and "after"
are the same tree for everything but `docs/`. `black` is clean on the one new file.
