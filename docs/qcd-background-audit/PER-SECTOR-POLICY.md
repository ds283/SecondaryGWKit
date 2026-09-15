# The per-sector break-point policy, re-measured on the corrected background

<!--
     Prompt 08 of prompts/qcd-background-audit. Sections 1-8 below are the verbatim stdout of

         PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/per_sector_policy_remeasure.py

     with the cosmology banners the model constructors print stripped from the top. The
     "one-paragraph answer" immediately below is written by hand, on the precedent of
     docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md, which is assembled the same way; re-running the
     script reproduces every table but not that paragraph. This document is additive: a later
     re-measurement appends a section rather than rewriting one, because each was correct for the
     tree it was taken on (CLAUDE.md).
-->

**The one-paragraph answer: state (a).** The transfer function's numeric sector no longer needs
`BREAK_POINT_ALL`. On the background prompts 04-07 corrected, QCD $T_k$ converges at **all 50**
production wavenumbers under *either* policy — worst reference-convergence drift
$7.08\times10^{-9}$ under `BREAK_POINT_ALL` and $8.85\times10^{-9}$ under
`BREAK_POINT_DISCONTINUITY`, median $1.96\times10^{-9}$ either way, against the
$3.4\times10^{-8}$ criterion — where `GkTk-remedial` prompt 19 measured $1.97\times10^{-7}$ and
three offenders with the jumps alone. The three wavenumbers that prompt 19's decision turned on
read $8.9\times10^{-10}$, $2.0\times10^{-9}$ and $5.6\times10^{-10}$ with the jumps alone, against
$6.1\times10^{-8}$, $2.0\times10^{-7}$ and $3.5\times10^{-8}$ then: **the 404 knots were standing
in for the representation's own defect, not for anything the integrator needed.** $G_k$ is
unchanged in kind and better in degree — worst $3.67\times10^{-9}$ on QCD under *both* policies,
against $8.41\times10^{-9}$ under its own policy before — so this is not state (c). The two models
that declare nothing are **bit-identical between the policies at all 50 wavenumbers in both
sectors** and reproduce prompt 19's grid totals as exact integers (401,677 / 429,178 / 637,138 /
640,213), prompt 17's two control figures included. The cost of the wider policy has collapsed
with the set it asks for: $T_k$ on QCD **8,897 → 8,986** right-hand-side evaluations per object
(**+0.99 %**, against +220 % when the knot lattice was declared) and $G_k$ **13,343 → 13,419**
(+0.57 %, against +155 %), differences too small to see in wall time against a same-process null
control. **Neither `BREAK_POINT_KIND` is changed**: the values are in a datastore lookup key
(`GkTk-remedial` prompt 20) and the two policies still give different numbers on QCD, by up to
$2.9\times10^{-4}$ of the envelope, so moving one has a regeneration attached and is
`prompts/qcd-background-audit/README.md` §7 **D5**, the user's.

**And one thing that is emphatically still needed** (§2b, which is beyond what prompt 08 §2 asked
for): with the cosmology's declaration suppressed *altogether*, QCD $T_k$ is above the criterion at
**19 of the 50** wavenumbers, worst $9.61\times10^{-6}$. Splitting at the equation of state's
genuine jumps is load-bearing and prompt 18's result stands undisturbed. What has become
vestigial is only the distinction between the two *kinds*.

---

<!-- generated 2026-09-14 by
     PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/per_sector_policy_remeasure.py
     in 744 s; Python 3.12.14, NumPy 2.2.4, SciPy 1.15.2;
     load average 4.6 / 8.4 / 10.2 at the start, 8.0 / 11.0 / 11.9 at the end -->

### 1. What the two policies now ask for

Prompt 07 took `integration_break_points` off the `T(z)` interpolant's knot lattice, so the two policies no longer differ by four hundred points. On the widest production geometry of either sector they differ by **one**: `EOS_T_LO = 0.002` GeV, the branch join at which `w` changes analytic form while `g_s` does not step.

| sector | wavenumber [1/Mpc] | z range of the run | points, `all` | points, `discontinuity` | in `all` and not in `discontinuity` |
|---|---|---|---|---|---|
| Tk | smallest k = 1e+05 | 9.653e+07 -- 1.017e+13 | 2 | 1 | 1.18721e+10 |
| Tk | largest k = 3e+08 | 3.282e+11 -- 3.3e+16 | 1 | 1 | -- |
| Gk | smallest k = 1e+05 | 7.667e+07 -- 8.457e+12 | 2 | 1 | 1.18721e+10 |
| Gk | largest k = 3e+08 | 3.063e+11 -- 2.562e+16 | 1 | 1 | -- |

### 2. The acceptance test, both sectors under both policies

Reference-convergence drift in the envelope-relative measure of `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §9.4, against the 3e-08 criterion of `prompts/GkTk-remedial` prompt 17 §2.1. Prompt 19's column is §10.1's, measured on the pre-campaign background.

| sector | model | policy | prompt 19 (§10.1) | worst drift now | at k [1/Mpc] | median drift | k above 3e-08 | criterion met? |
|---|---|---|---|---|---|---|---|---|
| Tk | RadiationModel | all | 4.21e-11 | 4.21e-11 | 3e+08 | 1.92e-11 | 0 | **yes** |
| Tk | RadiationModel | discontinuity | 4.21e-11 | 4.21e-11 | 3e+08 | 1.92e-11 | 0 | **yes** |
| Tk | LambdaCDMModel | all | 5.7e-11 | 5.7e-11 | 1.561e+08 | 3.76e-11 | 0 | **yes** |
| Tk | LambdaCDMModel | discontinuity | 5.7e-11 | 5.7e-11 | 1.561e+08 | 3.76e-11 | 0 | **yes** |
| Tk | QCDModel | all | 8.72e-09 | 7.08e-09 | 7.105e+05 | 1.96e-09 | 0 | **yes** |
| Tk | QCDModel | discontinuity | 1.97e-07 | 8.85e-09 | 1.387e+05 | 1.96e-09 | 0 | **yes** |
| Gk | RadiationModel | all | 1.94e-11 | 1.94e-11 | 1.561e+08 | 1.35e-11 | 0 | **yes** |
| Gk | RadiationModel | discontinuity | 1.94e-11 | 1.94e-11 | 1.561e+08 | 1.35e-11 | 0 | **yes** |
| Gk | LambdaCDMModel | all | 2.1e-11 | 2.1e-11 | 2.197e+07 | 1.39e-11 | 0 | **yes** |
| Gk | LambdaCDMModel | discontinuity | 2.1e-11 | 2.1e-11 | 2.197e+07 | 1.39e-11 | 0 | **yes** |
| Gk | QCDModel | all | -- | 3.67e-09 | 1.608e+06 | 2.16e-10 | 0 | **yes** |
| Gk | QCDModel | discontinuity | 8.41e-09 | 3.67e-09 | 1.608e+06 | 2.16e-10 | 0 | **yes** |

### 2b. And if the cosmology declared nothing at all

Neither policy, but the third column the board entry `[04-unsplit-tk-run-now-meets-the-criterion]` is about: the same runs with `integration_break_points` suppressed, so the integrator takes its historic single-`solve_ivp` path and never restarts at the jump in $H(z)$. That entry measured **1.0213e-06** on prompt 04's tree and **2.2767e-08** on prompt 05's at k = 4.972e7/Mpc; this is the same quantity across the whole grid, on prompt 07's.

| sector | model | worst drift, unsplit | at k [1/Mpc] | median drift | k above 3e-08 | at k = 4.972e+07, unsplit | at k = 4.972e+07, `discontinuity` | evals per object |
|---|---|---|---|---|---|---|---|---|
| Tk | QCDModel | 9.61e-06 | 4.223e+07 | 1.8e-08 | 19 | 2.91e-08 | 3.41e-09 | 8788 |
| Gk | QCDModel | 3.52e-09 | 1.608e+06 | 2.26e-10 | 0 | 4.33e-11 | 1.15e-10 | 13320 |

### 3. The three wavenumbers prompt 19's decision turned on

Under `BREAK_POINT_DISCONTINUITY`, three of the fifty QCD $T_k$ wavenumbers were above the criterion when prompt 19 measured them (§10.6, indices 13, 23 and 37). This is what they read now, on the corrected background.

| k [1/Mpc] | prompt 19, `discontinuity` | now, `discontinuity` | prompt 19, `all` | now, `all` | now <= 3e-08 under `discontinuity`? |
|---|---|---|---|---|---|
| 8.366e+05 | 6.1e-08 | 8.89e-10 | 6.9e-10 | 8.89e-10 | yes |
| 4.287e+06 | 2e-07 | 2.03e-09 | 4.7e-09 | 2.03e-09 | yes |
| 4.223e+07 | 3.5e-08 | 5.6e-10 | 2.3e-09 | 5.6e-10 | yes |

### 4. The control: the models that declare nothing

`RadiationModel` and `LambdaCDMModel` have no equation of state and declare no break points, so both policies must reach the same single-`solve_ivp` call and must give the same numbers -- and, since nothing in this campaign may move a LambdaCDM value (`README` §2 (g)), the same numbers prompt 19 measured. Grid totals are sums of the per-wavenumber right-hand-side evaluation counts: integers, machine-independent, and the sharpest available statement that nothing moved.

| sector | model | max segments | prompt 19 worst drift | worst drift now | prompt 19 grid evals | grid evals, `all` | grid evals, `discontinuity` | unchanged and bit-identical? |
|---|---|---|---|---|---|---|---|---|
| Tk | RadiationModel | 1 | 4.21e-11 | 4.21e-11 | 401677 | 401677 | 401677 | **yes** |
| Tk | LambdaCDMModel | 1 | 5.7e-11 | 5.7e-11 | 429178 | 429178 | 429178 | **yes** |
| Gk | RadiationModel | 1 | 1.94e-11 | 1.94e-11 | 637138 | 637138 | 637138 | **yes** |
| Gk | LambdaCDMModel | 1 | 2.1e-11 | 2.1e-11 | 640213 | 640213 | 640213 | **yes** |

And the module default is still `BREAK_POINT_DISCONTINUITY`: at every wavenumber of every (model, sector) the production run issued with `break_point_kind` omitted is bit-identical to the run issued with it named.

| sector | model | default matches `discontinuity` |
|---|---|---|
| Tk | RadiationModel | all 50 |
| Tk | LambdaCDMModel | all 50 |
| Tk | QCDModel | all 50 |
| Gk | RadiationModel | all 50 |
| Gk | LambdaCDMModel | all 50 |
| Gk | QCDModel | all 50 |

### 5. What each policy costs

Right-hand-side evaluations per object, averaged over the 50-wavenumber grid. **These counts are the measure** (`prompts/GkTk-remedial/README.md` §5 note 14): they are integers and do not depend on the machine, which is why this prompt quotes them first and wall time second.

| sector | model | prompt 19, `discontinuity` | prompt 19, `all` | now, `discontinuity` | now, `all` | `all` over `discontinuity` |
|---|---|---|---|---|---|---|
| Tk | RadiationModel | 8034 | 8034 | 8034 | 8034 | +0.00% |
| Tk | LambdaCDMModel | 8584 | 8584 | 8584 | 8584 | +0.00% |
| Tk | QCDModel | 9843 | 31521 | 8897 | 8986 | +0.99% |
| Gk | RadiationModel | 12743 | 12743 | 12743 | 12743 | +0.00% |
| Gk | LambdaCDMModel | 12804 | 12804 | 12804 | 12804 | +0.00% |
| Gk | QCDModel | 13320 | 13320 | 13343 | 13419 | +0.57% |

Wall time per object, best of 5 single-core runs at k = 4.972e+07/Mpc and the production tolerances, with the one-, five- and fifteen-minute load averages at which each pair was taken. Absolute seconds are a property of the machine and its load, so they are **not** comparable with prompt 19's, which were taken in a different session. What is comparable is the *ratio*: the two policies are timed back to back in the same process, and the four smooth-model rows -- where the two policies are literally the same computation -- are the null control that says how far from 1.000 a ratio has to be before it means anything.

| sector | model | s per object, `all` | s per object, `discontinuity` | ratio | evals, `all` / `discontinuity` | load average |
|---|---|---|---|---|---|---|
| Tk | RadiationModel | 0.0474 | 0.0461 | 1.028 | 8573 / 8573 | 8.4 / 11.2 / 12.0 |
| Gk | RadiationModel | 0.0673 | 0.0683 | 0.985 | 12695 / 12695 | 8.4 / 11.2 / 12.0 |
| Tk | LambdaCDMModel | 0.0626 | 0.0611 | 1.024 | 8687 / 8687 | 8.4 / 11.2 / 12.0 |
| Gk | LambdaCDMModel | 0.0789 | 0.0770 | 1.024 | 12740 / 12740 | 8.3 / 11.1 / 12.0 |
| Tk | QCDModel | 0.3060 | 0.3055 | 1.001 | 8674 / 8674 | 8.3 / 11.1 / 12.0 |
| Gk | QCDModel | 0.2119 | 0.2154 | 0.984 | 13231 / 13231 | 8.0 / 11.0 / 11.9 |

### 6. How far the answer itself moves between the policies

The production-tolerance run under `BREAK_POINT_DISCONTINUITY` scored against the production-tolerance run under `BREAK_POINT_ALL`, in the same envelope-relative measure -- i.e. what a stored object would move by if the policy were changed. On the models that declare nothing it is identically zero, which is the same statement as §4's bit-identity.

| sector | model | worst shift | at k [1/Mpc] | median shift |
|---|---|---|---|---|
| Tk | RadiationModel | 0 | 1e+05 | 0 |
| Tk | LambdaCDMModel | 0 | 1e+05 | 0 |
| Tk | QCDModel | 0.000286 | 3.139e+05 | 4.55e-08 |
| Gk | RadiationModel | 0 | 1e+05 | 0 |
| Gk | LambdaCDMModel | 0 | 1e+05 | 0 |
| Gk | QCDModel | 2.11e-08 | 3.092e+06 | 4.25e-10 |

### 7. Prompt 17's two control figures, under both policies

| control | prompt 12 | `all` | evals | `discontinuity` | evals | identical? |
|---|---|---|---|---|---|---|
| k=1e6 | 2.53e-06 | 2.53e-06 | 7403 | 2.53e-06 | 7403 | yes |
| k=3e8 | 0.000256 | 0.000256 | 8483 | 0.000256 | 8483 | yes |

### 8. Every wavenumber

#### Tk, RadiationModel

| k [1/Mpc] | segments, `all` / `disc` | drift, `all` | drift, `disc` | `disc` <= 3e-08? | shift between policies | evals, `all` / `disc` |
|---|---|---|---|---|---|---|
| 1e+05 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7262 / 7262 |
| 1.178e+05 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7184 / 7184 |
| 1.387e+05 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7169 / 7169 |
| 1.633e+05 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7094 / 7094 |
| 1.922e+05 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7151 / 7151 |
| 2.264e+05 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7136 / 7136 |
| 2.665e+05 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7106 / 7106 |
| 3.139e+05 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7175 / 7175 |
| 3.696e+05 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7058 / 7058 |
| 4.352e+05 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7031 / 7031 |
| 5.124e+05 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7049 / 7049 |
| 6.034e+05 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7103 / 7103 |
| 7.105e+05 | 1 / 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 7136 / 7136 |
| 8.366e+05 | 1 / 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 7175 / 7175 |
| 9.851e+05 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 7352 / 7352 |
| 1.16e+06 | 1 / 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 7670 / 7670 |
| 1.366e+06 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 7976 / 7976 |
| 1.608e+06 | 1 / 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 8138 / 8138 |
| 1.894e+06 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 8210 / 8210 |
| 2.23e+06 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 8234 / 8234 |
| 2.626e+06 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 8246 / 8246 |
| 3.092e+06 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 8321 / 8321 |
| 3.64e+06 | 1 / 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 8357 / 8357 |
| 4.287e+06 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 8396 / 8396 |
| 5.048e+06 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 8432 / 8432 |
| 5.943e+06 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 8444 / 8444 |
| 6.998e+06 | 1 / 1 | 2.1e-11 | 2.1e-11 | yes | 0 | 8369 / 8369 |
| 8.241e+06 | 1 / 1 | 2.4e-11 | 2.4e-11 | yes | 0 | 8390 / 8390 |
| 9.703e+06 | 1 / 1 | 2.7e-11 | 2.7e-11 | yes | 0 | 8387 / 8387 |
| 1.143e+07 | 1 / 1 | 2.8e-11 | 2.8e-11 | yes | 0 | 8363 / 8363 |
| 1.345e+07 | 1 / 1 | 2.9e-11 | 2.9e-11 | yes | 0 | 8366 / 8366 |
| 1.584e+07 | 1 / 1 | 2.7e-11 | 2.7e-11 | yes | 0 | 8405 / 8405 |
| 1.865e+07 | 1 / 1 | 2.8e-11 | 2.8e-11 | yes | 0 | 8519 / 8519 |
| 2.197e+07 | 1 / 1 | 2.9e-11 | 2.9e-11 | yes | 0 | 8444 / 8444 |
| 2.586e+07 | 1 / 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8456 / 8456 |
| 3.045e+07 | 1 / 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8459 / 8459 |
| 3.586e+07 | 1 / 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8645 / 8645 |
| 4.223e+07 | 1 / 1 | 3.2e-11 | 3.2e-11 | yes | 0 | 8534 / 8534 |
| 4.972e+07 | 1 / 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8573 / 8573 |
| 5.855e+07 | 1 / 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8561 / 8561 |
| 6.894e+07 | 1 / 1 | 3.3e-11 | 3.3e-11 | yes | 0 | 8570 / 8570 |
| 8.118e+07 | 1 / 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8630 / 8630 |
| 9.558e+07 | 1 / 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8657 / 8657 |
| 1.126e+08 | 1 / 1 | 3.5e-11 | 3.5e-11 | yes | 0 | 8516 / 8516 |
| 1.325e+08 | 1 / 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8528 / 8528 |
| 1.561e+08 | 1 / 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8495 / 8495 |
| 1.838e+08 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8552 / 8552 |
| 2.164e+08 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8660 / 8660 |
| 2.548e+08 | 1 / 1 | 3.5e-11 | 3.5e-11 | yes | 0 | 8486 / 8486 |
| 3e+08 | 1 / 1 | 4.2e-11 | 4.2e-11 | yes | 0 | 8507 / 8507 |

#### Tk, LambdaCDMModel

| k [1/Mpc] | segments, `all` / `disc` | drift, `all` | drift, `disc` | `disc` <= 3e-08? | shift between policies | evals, `all` / `disc` |
|---|---|---|---|---|---|---|
| 1e+05 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8567 / 8567 |
| 1.178e+05 | 1 / 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8528 / 8528 |
| 1.387e+05 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8462 / 8462 |
| 1.633e+05 | 1 / 1 | 4.4e-11 | 4.4e-11 | yes | 0 | 8639 / 8639 |
| 1.922e+05 | 1 / 1 | 3.5e-11 | 3.5e-11 | yes | 0 | 8738 / 8738 |
| 2.264e+05 | 1 / 1 | 4.5e-11 | 4.5e-11 | yes | 0 | 8714 / 8714 |
| 2.665e+05 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8435 / 8435 |
| 3.139e+05 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8648 / 8648 |
| 3.696e+05 | 1 / 1 | 4e-11 | 4e-11 | yes | 0 | 8654 / 8654 |
| 4.352e+05 | 1 / 1 | 4e-11 | 4e-11 | yes | 0 | 8468 / 8468 |
| 5.124e+05 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8429 / 8429 |
| 6.034e+05 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8564 / 8564 |
| 7.105e+05 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8585 / 8585 |
| 8.366e+05 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8633 / 8633 |
| 9.851e+05 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8690 / 8690 |
| 1.16e+06 | 1 / 1 | 4.9e-11 | 4.9e-11 | yes | 0 | 8597 / 8597 |
| 1.366e+06 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8501 / 8501 |
| 1.608e+06 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8702 / 8702 |
| 1.894e+06 | 1 / 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8480 / 8480 |
| 2.23e+06 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8549 / 8549 |
| 2.626e+06 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8651 / 8651 |
| 3.092e+06 | 1 / 1 | 4e-11 | 4e-11 | yes | 0 | 8648 / 8648 |
| 3.64e+06 | 1 / 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8624 / 8624 |
| 4.287e+06 | 1 / 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8531 / 8531 |
| 5.048e+06 | 1 / 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8612 / 8612 |
| 5.943e+06 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8624 / 8624 |
| 6.998e+06 | 1 / 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8507 / 8507 |
| 8.241e+06 | 1 / 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8621 / 8621 |
| 9.703e+06 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8549 / 8549 |
| 1.143e+07 | 1 / 1 | 3.9e-11 | 3.9e-11 | yes | 0 | 8588 / 8588 |
| 1.345e+07 | 1 / 1 | 3.9e-11 | 3.9e-11 | yes | 0 | 8408 / 8408 |
| 1.584e+07 | 1 / 1 | 4.1e-11 | 4.1e-11 | yes | 0 | 8651 / 8651 |
| 1.865e+07 | 1 / 1 | 4e-11 | 4e-11 | yes | 0 | 8588 / 8588 |
| 2.197e+07 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8612 / 8612 |
| 2.586e+07 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8627 / 8627 |
| 3.045e+07 | 1 / 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8609 / 8609 |
| 3.586e+07 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8504 / 8504 |
| 4.223e+07 | 1 / 1 | 4e-11 | 4e-11 | yes | 0 | 8612 / 8612 |
| 4.972e+07 | 1 / 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8687 / 8687 |
| 5.855e+07 | 1 / 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8732 / 8732 |
| 6.894e+07 | 1 / 1 | 3.9e-11 | 3.9e-11 | yes | 0 | 8432 / 8432 |
| 8.118e+07 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8561 / 8561 |
| 9.558e+07 | 1 / 1 | 4.2e-11 | 4.2e-11 | yes | 0 | 8558 / 8558 |
| 1.126e+08 | 1 / 1 | 3.9e-11 | 3.9e-11 | yes | 0 | 8588 / 8588 |
| 1.325e+08 | 1 / 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8444 / 8444 |
| 1.561e+08 | 1 / 1 | 5.7e-11 | 5.7e-11 | yes | 0 | 8564 / 8564 |
| 1.838e+08 | 1 / 1 | 3.5e-11 | 3.5e-11 | yes | 0 | 8603 / 8603 |
| 2.164e+08 | 1 / 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8501 / 8501 |
| 2.548e+08 | 1 / 1 | 5.3e-11 | 5.3e-11 | yes | 0 | 8711 / 8711 |
| 3e+08 | 1 / 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8648 / 8648 |

#### Tk, QCDModel

| k [1/Mpc] | segments, `all` / `disc` | drift, `all` | drift, `disc` | `disc` <= 3e-08? | shift between policies | evals, `all` / `disc` |
|---|---|---|---|---|---|---|
| 1e+05 | 3 / 2 | 6.5e-09 | 6.5e-09 | yes | 2.7e-07 | 9210 / 9043 |
| 1.178e+05 | 3 / 2 | 2.2e-09 | 2.5e-09 | yes | 1.4e-07 | 9429 / 9088 |
| 1.387e+05 | 3 / 2 | 2.9e-09 | 8.8e-09 | yes | 0.00025 | 9414 / 9145 |
| 1.633e+05 | 3 / 2 | 8.7e-10 | 9.2e-10 | yes | 0.00025 | 9207 / 9016 |
| 1.922e+05 | 3 / 2 | 2.1e-09 | 1.8e-09 | yes | 2.3e-07 | 9255 / 9214 |
| 2.264e+05 | 3 / 2 | 1.4e-09 | 1.4e-09 | yes | 3e-07 | 9423 / 9271 |
| 2.665e+05 | 3 / 2 | 4.4e-09 | 4.5e-09 | yes | 2.1e-07 | 9282 / 9010 |
| 3.139e+05 | 3 / 2 | 9.4e-11 | 9.2e-10 | yes | 0.00029 | 9432 / 9208 |
| 3.696e+05 | 3 / 2 | 1.5e-09 | 2.1e-09 | yes | 4.5e-07 | 9249 / 9214 |
| 4.352e+05 | 3 / 2 | 1.9e-10 | 1.9e-10 | yes | 1.4e-07 | 9324 / 9178 |
| 5.124e+05 | 3 / 2 | 1.2e-09 | 1.6e-09 | yes | 3.5e-07 | 9363 / 9217 |
| 6.034e+05 | 3 / 2 | 3.5e-09 | 3.7e-09 | yes | 3.3e-07 | 9372 / 9109 |
| 7.105e+05 | 3 / 2 | 7.1e-09 | 7.1e-09 | yes | 2.3e-07 | 9246 / 9130 |
| 8.366e+05 | 3 / 2 | 8.9e-10 | 8.9e-10 | yes | 1.6e-07 | 9321 / 9229 |
| 9.851e+05 | 3 / 2 | 1.5e-09 | 1.5e-09 | yes | 2e-07 | 9369 / 9019 |
| 1.16e+06 | 3 / 2 | 2.5e-09 | 2.8e-09 | yes | 2e-07 | 9243 / 9064 |
| 1.366e+06 | 3 / 2 | 2.3e-09 | 2.3e-09 | yes | 6.5e-08 | 9174 / 9085 |
| 1.608e+06 | 3 / 2 | 2.9e-09 | 2.9e-09 | yes | 9.5e-08 | 9129 / 9160 |
| 1.894e+06 | 3 / 2 | 1.2e-09 | 1.4e-09 | yes | 6.9e-08 | 9276 / 9076 |
| 2.23e+06 | 3 / 2 | 4.8e-09 | 4.8e-09 | yes | 5.7e-08 | 8976 / 8803 |
| 2.626e+06 | 3 / 2 | 1.3e-09 | 1.3e-09 | yes | 6.5e-08 | 8976 / 8914 |
| 3.092e+06 | 3 / 2 | 6.9e-09 | 6.9e-09 | yes | 4.7e-08 | 8856 / 8770 |
| 3.64e+06 | 3 / 2 | 2.1e-09 | 2.1e-09 | yes | 4.4e-08 | 8991 / 8881 |
| 4.287e+06 | 3 / 2 | 2e-09 | 2e-09 | yes | 4.8e-08 | 9081 / 8893 |
| 5.048e+06 | 3 / 2 | 3.2e-09 | 3.2e-09 | yes | 5.7e-08 | 8877 / 8662 |
| 5.943e+06 | 3 / 2 | 1.9e-09 | 1.9e-09 | yes | 4.8e-08 | 9126 / 9028 |
| 6.998e+06 | 3 / 2 | 1.1e-09 | 1.2e-09 | yes | 3.7e-08 | 8844 / 8746 |
| 8.241e+06 | 3 / 2 | 2.1e-09 | 2.1e-09 | yes | 4.3e-08 | 8928 / 8779 |
| 9.703e+06 | 3 / 2 | 1e-09 | 1e-09 | yes | 0 | 8800 / 8800 |
| 1.143e+07 | 2 / 2 | 2e-09 | 2e-09 | yes | 0 | 8839 / 8839 |
| 1.345e+07 | 2 / 2 | 5.7e-09 | 5.7e-09 | yes | 0 | 8911 / 8911 |
| 1.584e+07 | 2 / 2 | 4.9e-09 | 4.9e-09 | yes | 0 | 8749 / 8749 |
| 1.865e+07 | 2 / 2 | 6.1e-09 | 6.1e-09 | yes | 0 | 8716 / 8716 |
| 2.197e+07 | 2 / 2 | 1.1e-09 | 1.1e-09 | yes | 0 | 8863 / 8863 |
| 2.586e+07 | 2 / 2 | 2.4e-09 | 2.4e-09 | yes | 0 | 8725 / 8725 |
| 3.045e+07 | 2 / 2 | 3.8e-09 | 3.8e-09 | yes | 0 | 8701 / 8701 |
| 3.586e+07 | 2 / 2 | 3.5e-09 | 3.5e-09 | yes | 0 | 8710 / 8710 |
| 4.223e+07 | 2 / 2 | 5.6e-10 | 5.6e-10 | yes | 0 | 8584 / 8584 |
| 4.972e+07 | 2 / 2 | 3.4e-09 | 3.4e-09 | yes | 0 | 8674 / 8674 |
| 5.855e+07 | 2 / 2 | 2.6e-09 | 2.6e-09 | yes | 0 | 8650 / 8650 |
| 6.894e+07 | 2 / 2 | 5e-10 | 5e-10 | yes | 0 | 8608 / 8608 |
| 8.118e+07 | 2 / 2 | 4.5e-10 | 4.5e-10 | yes | 0 | 8674 / 8674 |
| 9.558e+07 | 2 / 2 | 3.1e-10 | 3.1e-10 | yes | 0 | 8761 / 8761 |
| 1.126e+08 | 2 / 2 | 1.5e-10 | 1.5e-10 | yes | 0 | 8626 / 8626 |
| 1.325e+08 | 2 / 2 | 9.7e-10 | 9.7e-10 | yes | 0 | 8722 / 8722 |
| 1.561e+08 | 2 / 2 | 6.7e-10 | 6.7e-10 | yes | 0 | 8734 / 8734 |
| 1.838e+08 | 2 / 2 | 6.4e-11 | 6.4e-11 | yes | 0 | 8761 / 8761 |
| 2.164e+08 | 2 / 2 | 9e-10 | 9e-10 | yes | 0 | 8698 / 8698 |
| 2.548e+08 | 2 / 2 | 1.3e-09 | 1.3e-09 | yes | 0 | 8680 / 8680 |
| 3e+08 | 2 / 2 | 1e-09 | 1e-09 | yes | 0 | 8716 / 8716 |

#### Gk, RadiationModel

| k [1/Mpc] | segments, `all` / `disc` | drift, `all` | drift, `disc` | `disc` <= 3e-08? | shift between policies | evals, `all` / `disc` |
|---|---|---|---|---|---|---|
| 1e+05 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12764 / 12764 |
| 1.178e+05 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12725 / 12725 |
| 1.387e+05 | 1 / 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12764 / 12764 |
| 1.633e+05 | 1 / 1 | 9.3e-12 | 9.3e-12 | yes | 0 | 12737 / 12737 |
| 1.922e+05 | 1 / 1 | 9.2e-12 | 9.2e-12 | yes | 0 | 12677 / 12677 |
| 2.264e+05 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12740 / 12740 |
| 2.665e+05 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12701 / 12701 |
| 3.139e+05 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12740 / 12740 |
| 3.696e+05 | 1 / 1 | 9.3e-12 | 9.3e-12 | yes | 0 | 12713 / 12713 |
| 4.352e+05 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12752 / 12752 |
| 5.124e+05 | 1 / 1 | 1e-11 | 1e-11 | yes | 0 | 12692 / 12692 |
| 6.034e+05 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12752 / 12752 |
| 7.105e+05 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12704 / 12704 |
| 8.366e+05 | 1 / 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12752 / 12752 |
| 9.851e+05 | 1 / 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12704 / 12704 |
| 1.16e+06 | 1 / 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12665 / 12665 |
| 1.366e+06 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12728 / 12728 |
| 1.608e+06 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12755 / 12755 |
| 1.894e+06 | 1 / 1 | 5e-12 | 5e-12 | yes | 0 | 12740 / 12740 |
| 2.23e+06 | 1 / 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12755 / 12755 |
| 2.626e+06 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12740 / 12740 |
| 3.092e+06 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12680 / 12680 |
| 3.64e+06 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12755 / 12755 |
| 4.287e+06 | 1 / 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12728 / 12728 |
| 5.048e+06 | 1 / 1 | 9.1e-12 | 9.1e-12 | yes | 0 | 12755 / 12755 |
| 5.943e+06 | 1 / 1 | 5.7e-12 | 5.7e-12 | yes | 0 | 12728 / 12728 |
| 6.998e+06 | 1 / 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12755 / 12755 |
| 8.241e+06 | 1 / 1 | 6.8e-12 | 6.8e-12 | yes | 0 | 12782 / 12782 |
| 9.703e+06 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12743 / 12743 |
| 1.143e+07 | 1 / 1 | 9.1e-12 | 9.1e-12 | yes | 0 | 12782 / 12782 |
| 1.345e+07 | 1 / 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12767 / 12767 |
| 1.584e+07 | 1 / 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 12719 / 12719 |
| 1.865e+07 | 1 / 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 12770 / 12770 |
| 2.197e+07 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12707 / 12707 |
| 2.586e+07 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12770 / 12770 |
| 3.045e+07 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12743 / 12743 |
| 3.586e+07 | 1 / 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12770 / 12770 |
| 4.223e+07 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12743 / 12743 |
| 4.972e+07 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12695 / 12695 |
| 5.855e+07 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12809 / 12809 |
| 6.894e+07 | 1 / 1 | 5.5e-12 | 5.5e-12 | yes | 0 | 12782 / 12782 |
| 8.118e+07 | 1 / 1 | 7e-12 | 7e-12 | yes | 0 | 12722 / 12722 |
| 9.558e+07 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12695 / 12695 |
| 1.126e+08 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12758 / 12758 |
| 1.325e+08 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12785 / 12785 |
| 1.561e+08 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12758 / 12758 |
| 1.838e+08 | 1 / 1 | 9.8e-12 | 9.8e-12 | yes | 0 | 12797 / 12797 |
| 2.164e+08 | 1 / 1 | 6.5e-12 | 6.5e-12 | yes | 0 | 12770 / 12770 |
| 2.548e+08 | 1 / 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12797 / 12797 |
| 3e+08 | 1 / 1 | 6.1e-12 | 6.1e-12 | yes | 0 | 12773 / 12773 |

#### Gk, LambdaCDMModel

| k [1/Mpc] | segments, `all` / `disc` | drift, `all` | drift, `disc` | `disc` <= 3e-08? | shift between policies | evals, `all` / `disc` |
|---|---|---|---|---|---|---|
| 1e+05 | 1 / 1 | 1e-11 | 1e-11 | yes | 0 | 12815 / 12815 |
| 1.178e+05 | 1 / 1 | 7.9e-12 | 7.9e-12 | yes | 0 | 12740 / 12740 |
| 1.387e+05 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12803 / 12803 |
| 1.633e+05 | 1 / 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12767 / 12767 |
| 1.922e+05 | 1 / 1 | 1e-11 | 1e-11 | yes | 0 | 12815 / 12815 |
| 2.264e+05 | 1 / 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12815 / 12815 |
| 2.665e+05 | 1 / 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 12764 / 12764 |
| 3.139e+05 | 1 / 1 | 4.5e-12 | 4.5e-12 | yes | 0 | 12815 / 12815 |
| 3.696e+05 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12842 / 12842 |
| 4.352e+05 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12815 / 12815 |
| 5.124e+05 | 1 / 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12854 / 12854 |
| 6.034e+05 | 1 / 1 | 2.1e-11 | 2.1e-11 | yes | 0 | 12827 / 12827 |
| 7.105e+05 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12815 / 12815 |
| 8.366e+05 | 1 / 1 | 8.5e-12 | 8.5e-12 | yes | 0 | 12854 / 12854 |
| 9.851e+05 | 1 / 1 | 6.6e-12 | 6.6e-12 | yes | 0 | 12827 / 12827 |
| 1.16e+06 | 1 / 1 | 7.3e-12 | 7.3e-12 | yes | 0 | 12815 / 12815 |
| 1.366e+06 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12788 / 12788 |
| 1.608e+06 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12803 / 12803 |
| 1.894e+06 | 1 / 1 | 7.9e-12 | 7.9e-12 | yes | 0 | 12788 / 12788 |
| 2.23e+06 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12815 / 12815 |
| 2.626e+06 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12791 / 12791 |
| 3.092e+06 | 1 / 1 | 2e-11 | 2e-11 | yes | 0 | 12752 / 12752 |
| 3.64e+06 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12779 / 12779 |
| 4.287e+06 | 1 / 1 | 7.3e-12 | 7.3e-12 | yes | 0 | 12827 / 12827 |
| 5.048e+06 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12827 / 12827 |
| 5.943e+06 | 1 / 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12767 / 12767 |
| 6.998e+06 | 1 / 1 | 1e-11 | 1e-11 | yes | 0 | 12740 / 12740 |
| 8.241e+06 | 1 / 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12854 / 12854 |
| 9.703e+06 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12815 / 12815 |
| 1.143e+07 | 1 / 1 | 4.2e-12 | 4.2e-12 | yes | 0 | 12815 / 12815 |
| 1.345e+07 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12842 / 12842 |
| 1.584e+07 | 1 / 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12815 / 12815 |
| 1.865e+07 | 1 / 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12854 / 12854 |
| 2.197e+07 | 1 / 1 | 2.1e-11 | 2.1e-11 | yes | 0 | 12788 / 12788 |
| 2.586e+07 | 1 / 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12815 / 12815 |
| 3.045e+07 | 1 / 1 | 9.9e-12 | 9.9e-12 | yes | 0 | 12842 / 12842 |
| 3.586e+07 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12803 / 12803 |
| 4.223e+07 | 1 / 1 | 7.7e-12 | 7.7e-12 | yes | 0 | 12791 / 12791 |
| 4.972e+07 | 1 / 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12740 / 12740 |
| 5.855e+07 | 1 / 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12815 / 12815 |
| 6.894e+07 | 1 / 1 | 8.4e-12 | 8.4e-12 | yes | 0 | 12788 / 12788 |
| 8.118e+07 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12815 / 12815 |
| 9.558e+07 | 1 / 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12779 / 12779 |
| 1.126e+08 | 1 / 1 | 2.1e-11 | 2.1e-11 | yes | 0 | 12827 / 12827 |
| 1.325e+08 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12767 / 12767 |
| 1.561e+08 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12752 / 12752 |
| 1.838e+08 | 1 / 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12815 / 12815 |
| 2.164e+08 | 1 / 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12779 / 12779 |
| 2.548e+08 | 1 / 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12827 / 12827 |
| 3e+08 | 1 / 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12815 / 12815 |

#### Gk, QCDModel

| k [1/Mpc] | segments, `all` / `disc` | drift, `all` | drift, `disc` | `disc` <= 3e-08? | shift between policies | evals, `all` / `disc` |
|---|---|---|---|---|---|---|
| 1e+05 | 3 / 2 | 1e-09 | 1e-09 | yes | 1.8e-09 | 13425 / 13324 |
| 1.178e+05 | 3 / 2 | 3e-10 | 3e-10 | yes | 4.4e-10 | 13377 / 13225 |
| 1.387e+05 | 3 / 2 | 1.9e-10 | 1.9e-10 | yes | 1.8e-09 | 13416 / 13264 |
| 1.633e+05 | 3 / 2 | 1.1e-10 | 1.1e-10 | yes | 2.1e-09 | 13458 / 13357 |
| 1.922e+05 | 3 / 2 | 2.2e-10 | 2.2e-10 | yes | 4.5e-10 | 13476 / 13351 |
| 2.264e+05 | 3 / 2 | 2.6e-10 | 2.6e-10 | yes | 3.1e-09 | 13467 / 13354 |
| 2.665e+05 | 3 / 2 | 6.5e-10 | 6.5e-10 | yes | 7.1e-09 | 13623 / 13435 |
| 3.139e+05 | 3 / 2 | 5.9e-10 | 5.9e-10 | yes | 2.6e-09 | 13638 / 13438 |
| 3.696e+05 | 3 / 2 | 3.3e-10 | 3.3e-10 | yes | 4.1e-10 | 13545 / 13393 |
| 4.352e+05 | 3 / 2 | 1.1e-09 | 1.1e-09 | yes | 2.6e-09 | 13644 / 13456 |
| 5.124e+05 | 3 / 2 | 4e-10 | 4e-10 | yes | 1.5e-09 | 13668 / 13468 |
| 6.034e+05 | 3 / 2 | 6e-11 | 6e-11 | yes | 9.6e-10 | 13695 / 13519 |
| 7.105e+05 | 3 / 2 | 2.4e-09 | 2.4e-09 | yes | 9.8e-10 | 13587 / 13423 |
| 8.366e+05 | 3 / 2 | 8.7e-11 | 8.7e-11 | yes | 8.9e-10 | 13509 / 13420 |
| 9.851e+05 | 3 / 2 | 2.3e-09 | 2.3e-09 | yes | 2e-10 | 13440 / 13327 |
| 1.16e+06 | 3 / 2 | 2.4e-09 | 2.4e-09 | yes | 1.8e-09 | 13458 / 13321 |
| 1.366e+06 | 3 / 2 | 2.8e-10 | 2.8e-10 | yes | 1.4e-08 | 13377 / 13303 |
| 1.608e+06 | 3 / 2 | 3.7e-09 | 3.7e-09 | yes | 9.6e-09 | 13296 / 13207 |
| 1.894e+06 | 3 / 2 | 1.2e-09 | 1.2e-09 | yes | 9.2e-09 | 13356 / 13207 |
| 2.23e+06 | 3 / 2 | 5.7e-10 | 5.7e-10 | yes | 7.5e-09 | 13509 / 13435 |
| 2.626e+06 | 3 / 2 | 4.6e-10 | 4.6e-10 | yes | 1.9e-08 | 13323 / 13138 |
| 3.092e+06 | 3 / 2 | 8.3e-10 | 8.3e-10 | yes | 2.1e-08 | 13275 / 13165 |
| 3.64e+06 | 3 / 2 | 1.7e-10 | 1.7e-10 | yes | 7.2e-09 | 13371 / 13222 |
| 4.287e+06 | 3 / 2 | 1.8e-10 | 1.8e-10 | yes | 6.5e-09 | 13305 / 13183 |
| 5.048e+06 | 3 / 2 | 2.6e-10 | 2.6e-10 | yes | 8.4e-09 | 13347 / 13210 |
| 5.943e+06 | 3 / 2 | 4.2e-10 | 4.2e-10 | yes | 1.7e-09 | 13188 / 13066 |
| 6.998e+06 | 3 / 2 | 1.4e-10 | 1.4e-10 | yes | 1.1e-09 | 13224 / 13102 |
| 8.241e+06 | 3 / 2 | 1.9e-10 | 1.9e-10 | yes | 1.6e-10 | 13266 / 13144 |
| 9.703e+06 | 3 / 2 | 2.1e-10 | 2.1e-10 | yes | 0 | 13192 / 13192 |
| 1.143e+07 | 3 / 2 | 1e-10 | 1e-10 | yes | 0 | 13177 / 13177 |
| 1.345e+07 | 3 / 2 | 6.5e-10 | 6.5e-10 | yes | 0 | 13165 / 13165 |
| 1.584e+07 | 2 / 2 | 3.5e-10 | 3.5e-10 | yes | 0 | 13264 / 13264 |
| 1.865e+07 | 2 / 2 | 7.8e-11 | 7.8e-11 | yes | 0 | 13102 / 13102 |
| 2.197e+07 | 2 / 2 | 1.6e-11 | 1.6e-11 | yes | 0 | 13108 / 13108 |
| 2.586e+07 | 2 / 2 | 3.5e-10 | 3.5e-10 | yes | 0 | 13198 / 13198 |
| 3.045e+07 | 2 / 2 | 5.7e-10 | 5.7e-10 | yes | 0 | 13234 / 13234 |
| 3.586e+07 | 2 / 2 | 5.2e-10 | 5.2e-10 | yes | 0 | 13270 / 13270 |
| 4.223e+07 | 2 / 2 | 1.3e-10 | 1.3e-10 | yes | 0 | 13186 / 13186 |
| 4.972e+07 | 2 / 2 | 1.2e-10 | 1.2e-10 | yes | 0 | 13231 / 13231 |
| 5.855e+07 | 2 / 2 | 6.9e-11 | 6.9e-11 | yes | 0 | 13315 / 13315 |
| 6.894e+07 | 2 / 2 | 1.9e-10 | 1.9e-10 | yes | 0 | 13399 / 13399 |
| 8.118e+07 | 2 / 2 | 2.2e-11 | 2.2e-11 | yes | 0 | 13534 / 13534 |
| 9.558e+07 | 2 / 2 | 1.3e-11 | 1.3e-11 | yes | 0 | 13591 / 13591 |
| 1.126e+08 | 2 / 2 | 9.3e-12 | 9.3e-12 | yes | 0 | 13525 / 13525 |
| 1.325e+08 | 2 / 2 | 1.9e-11 | 1.9e-11 | yes | 0 | 13669 / 13669 |
| 1.561e+08 | 2 / 2 | 2.9e-11 | 2.9e-11 | yes | 0 | 13648 / 13648 |
| 1.838e+08 | 2 / 2 | 2e-11 | 2e-11 | yes | 0 | 13720 / 13720 |
| 2.164e+08 | 2 / 2 | 1.5e-11 | 1.5e-11 | yes | 0 | 13660 / 13660 |
| 2.548e+08 | 2 / 2 | 1.7e-11 | 1.7e-11 | yes | 0 | 13732 / 13732 |
| 3e+08 | 2 / 2 | 1.4e-11 | 1.4e-11 | yes | 0 | 13750 / 13750 |

*Runtime 744 s (QCD stand-in build 0.5 s of it); 50 wavenumbers x 3 models x 2 sectors x 2 policies.*
