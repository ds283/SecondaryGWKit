# Log 01 — the Kohri–Terada radiation oracle

**Prompt:** `prompts/radiation-oracle/01-kohri-terada-radiation-oracle.md`
**Commit:** *(this prompt's own commit)* — "Land the Kohri-Terada radiation oracle with its tests"
**Model:** Opus 5
**Date:** 2026-09-18
**Result:** **DONE.** Kohri & Terada's eq. (22) is in the tree as
`ComputeTargets/tests/kohri_terada.py`, with eq. (25), eq. (19), eq. (16) at $w=1/3$, eq. (20)
for testing only, a quadrature of eq. (15), the `Cin`-regularised resonance and
`total_from_I_RD`, and `ComputeTargets/tests/test_kohri_terada_oracle.py` holds the six tests of
prompt §3 in nine methods. **The pipeline's `total` equals $-\tfrac98\,k_{\rm phys}^{-2}$ times
eq. (22) minus the head on all nine $b=0$ cases** at the reference pair `(1e-45, 1e-12)`: worst
$|N+9/8|$ **1.08e-10** (q-smooth, $x_{\rm resp}=980$), against that case's own bound of
**1.63e-08**, the sum of eq. (22)'s declared rounding estimate (1.4e-08 there, conservative: the
actual error at that point is 9.7e-11) and the pipeline's declared error (5.2e-12). On the six
cases where eq. (22) is well conditioned the worst is **6.9e-14** against bounds of 1.1e-12 to
9.9e-12, set by the pipeline. The resonance is finite and correct: $I_{\rm RD}$ at
$u=v=\sqrt3/2$, $x=15$ is **0.1391877548291586**, equal to the quadrature to the last bit
(quad's own error 8.5e-15). **Every one of the four traps is caught by a named failing test**
(§3.2 below); the inverted Ci/Si pairing by six of the nine methods, including tests 1, 2, 3, 4
and 6. **No production file, no existing `.py` file and nothing under `docs/` is in the diff**;
`config/defaults.py` is byte-identical. `ComputeTargets` **521 → 530** (nine methods added),
`CosmologyModels` **39**.

**One finding the prompt did not anticipate, and it changes how the audit's second figure should
be read.** The audit (§1, §7.1) and this campaign's README, board and prompt all attribute the
3.18e-10 of the eq.-(22) comparison to *the head subtraction*. It is not: the head's declared
quadrature error is 1e-20 to 1e-18 of $I$, and the whole deviation is **eq. (22)'s own
double-precision rounding at $u = q/k = 0.01$**, where the $1/(u^3v^3)$ prefactor multiplies
terms that nearly cancel. Measured against a 50-digit `mpmath` evaluation of eq. (22) at the three
q-smooth points, the module's double-precision value is off by **9.67e-11, 6.62e-11, 2.50e-11**,
and the three q-smooth deviations of $N$ are **1.08e-10, 7.45e-11, 2.82e-11** — the same numbers.
After removing it, the pipeline agrees with eq. (22) on q-smooth to ~1e-11. Opened as
`[01-the-eq22-figure-is-eq22s-own-rounding-not-the-head]`; the audit is a verification document
and is not edited here (CLAUDE.md invariant 6; `docs/radiation-oracle/` is outside this prompt).

---

## What shipped

| File | Status | What |
|---|---|---|
| `ComputeTargets/tests/kohri_terada.py` | **A** | The oracle module |
| `ComputeTargets/tests/test_kohri_terada_oracle.py` | **A** | Nine test methods: prompt §3's tests 1–6, plus the series-branch joins |
| `prompts/radiation-oracle/logs/01-kohri-terada-radiation-oracle.md` | **A** | This log |
| `prompts/radiation-oracle/IMPLEMENTATION_STATE.md` | M | Row 01, item R1, §3, §4 |
| `docs/OPEN_ISSUES.md` | M | One row added under §1.9; count 82 → 83 |

### The module and where it lives

**`ComputeTargets/tests/kohri_terada.py`**, beside `convergence_reference.py` and
`wkb_reference.py`, the two existing non-test helper modules in that directory. **Reason, in one
sentence: its only consumers are tests and it must never become a pipeline component, and
`ComputeTargets/tests/` is the one place that is both importable by the suite and outside the
package `main.py` and the Ray tasks import.** A top-level module would present a $b=0$-only
oracle as a library and invite a call from production code; `docs/radiation-oracle/` is a
directory the suite does not own and is the audit's, which CLAUDE.md invariant 6 makes additive.
Reaching it from elsewhere is not awkward in practice — `docs/radiation-oracle/kt_verification.py`
already imports `ComputeTargets.tests.test_quadsource_integral`, so `ComputeTargets.tests.*` is the
established way scripts reach test machinery. `grep -rn kohri_terada --include='*.py' . | grep -v
/tests/` returns nothing.

Public surface, each naming its KT equation in its docstring:

- `I_RD(v, u, x)` — eq. (22), regular at $u+v=\sqrt3$; `I_RD_with_rounding_scale` returns
  `(I_RD, scale)`: for every term, its magnitude plus the error its inputs' rounding induces (the
  trig arguments, $d$, the Ci/Si arguments), times the prefactor, so that $\epsilon\times$
  `scale` estimates eq. (22)'s own error.
- `I_RD_asymptotic(v, u, x)` — eq. (25).
- `Phi(x)`, `dPhi(x)` — eq. (19) and $d/dx$, series-summed below $|x|=1$.
- `f_RD(v, u, x)` — eq. (16) at $w=1/3$, **not** eq. (20).
- `f_RD_eq20(v, u, x)` and `f_RD_eq20_with_rounding_scale` — eq. (20), documented unusable below
  $x\approx0.1$ and used only by test 5.
- `Si`, `Ci`, `Cin` — `Cin` series-summed below $z=1$.
- `I_RD_quadrature(v, u, x, x_lo=0, x_hi=x)` — eq. (15) by `scipy.quad`, returning
  `(value, declared_error)`; the kernel carries the outer $x$, so the head is its own integral.
- `total_from_I_RD(k, q, r, tau_response, tau_source_max)` — the mapping, in the pipeline's own
  variables, returning `total`, its own `total_error` and the ingredients.
- `KT_NORM = -9/8`, with its factorisation and the sign convention in its comment.

### The errata, at the point of use

- **Erratum 1** (Ci/Si pairing): `I_RD`'s docstring writes eq. (22) out in full with the correct
  pairing, says the rendered PDF invites the inversion and why, gives the symptom of the inverted
  form ($O(1)$ miss, $-12.08$ as $x\to0$), and names the ar5iv `alttext` as the authority; the
  line that builds the brackets in `I_RD_with_rounding_scale` carries a one-line `ERRATUM 1`
  comment.
- **Erratum 2** (small $x$): `I_RD`'s docstring says $2x^2/9$, derives it, and says the paper's
  $x^2/2$ is wrong by $4/9$ and must not be an acceptance test. The module docstring repeats it.
- **Erratum 3** (eq. 20): `f_RD`'s docstring says it is eq. (16) and not eq. (20) and why, with the
  measured error growth; `f_RD_eq20`'s says it is unusable below $x\approx0.1$, diverges at $0$,
  and exists only for test 5.
- **The resonance**: `I_RD`'s docstring gives the `Cin` identity with $a,b,c,e$ defined, says the
  $\ln|c|$ cancels analytically, and says nothing raises, returns `nan` or nudges; the code
  comment above `sin_terms` repeats the identity where it is applied.

### The mapping keeps $a_0$ absorbed

`total_from_I_RD` takes `k`, `q`, `r` as the code's $k/a_0$ and $\tau$ as the code's $a_0\eta$,
forms $x = k\tau$, and multiplies by `KT_NORM / k**2`. Its docstring says the mapping is written in
$1/k_{\rm phys}^2$ **because $a_0$ is absorbed, not set to one** (spec 04 §0(1)), walks the
substitution of spec 04 §0(2) that produces $a_0^2/k^2 = 1/k_{\rm phys}^2$, and states the
$a_0\to\lambda a_0$ invariance.

## Deviations from the prompt

1. **The module exposes more than §1's list** — `I_RD_quadrature`, the two `*_with_rounding_scale`
   functions, and `Si`/`Ci`/`Cin`. **IMPLEMENTATION CHOICE.** §1 says "and nothing more". The
   quadrature of eq. (15) is needed by `total_from_I_RD` itself (the head $0\to\bar x_{\rm min}$
   has no closed form) and by tests 1, 2 and 4, so keeping it in the module is what puts the
   mapping "in one place". The rounding scales exist because prompt §3 asks for bounds "you can
   defend": a bound on eq. (22) that ignores its own cancellation is either vacuous or wrong at
   $u=0.01$ and at small $x$. The first version of the scale summed the terms' magnitudes only;
   the `mpmath` check found it **underestimating by 3.1×** at $(v,u,x)=(0.3,1.6,12)$, from the
   rounding of the trig arguments $ux/\sqrt3$, so it now carries the input-rounding terms too.
   Against 50-digit `mpmath` on 307 points the actual error is now **at most 0.33** of
   $\epsilon\times$ `scale` (§3.1).

2. **The series branches are wider than `kt_verification.py`'s.** `Phi`/`dPhi` switch at $|x|=1$
   with twelve terms, not at $10^{-2}$ with three; `Cin` at $z=1$ with twelve, not $10^{-2}$ with
   three. **IMPLEMENTATION CHOICE.** Against 50-digit `mpmath`, `kt_verification.py`'s `dPhi` is off
   by **1.1e-06** relative at $x=0.0101$ (3.9e-08 at 0.02, 5.4e-10 at 0.1), where its closed form
   cancels; the module's is ≤ 1.3e-16 on $[10^{-6}, 1)$ and ≤ 8.0e-15 at and just above 1.
   Harmless to the audit's figures — its small-$x$ quadrature column reproduces to all printed
   digits — but an oracle should not carry a 1e-6 error in an ingredient.

3. **Eq. (22) and eq. (20) sum their terms with `math.fsum`.** **IMPLEMENTATION CHOICE.** This
   changes the rounding, not the function, and moves two figures relative to the audit's, both
   inside eq. (22)'s own declared rounding: test 1's worst on the audit grid is **8.25e-15** at
   $(0.7, 1.1, x=1)$ against the audit's 3.5e-15 — at $x=1$ eq. (22)'s rounding estimate is
   **3.9e-13** relative and 50-digit `mpmath` puts the module at 8.25e-15 and the audit's script at
   3.38e-15 from the true value, with the quadrature exact, so both are rounding and neither is a
   transcription difference; and test 6's q-smooth worst is **1.08e-10** against the audit's
   3.18e-10, for the reason in the Result.

4. **Test 6 asserts a per-case bound, not one number.** **IMPLEMENTATION CHOICE.** Prompt §3.6 says
   "assert at the looser bound" — the eq.-(22) comparison's, not the quadrature's 3.09e-13 — and
   this does, with the bound built per case from the two sides' declared errors: the oracle's
   ($\epsilon\times$ eq. (22)'s rounding scale, plus the head's quad error) and the pipeline's
   (`total_abserr`, or `rtol`$\times(|$`numeric_quad`$|+|$`WKB_Levin`$|)$ where larger, because the
   two halves cancel). It is **3.7e-09 to 1.63e-08** relative on q-smooth — above 3.18e-10, as the
   prompt intends, because that is where eq. (22) limits — and **1.1e-12 to 9.9e-12** on the other
   six, where the pipeline's own error sets it. **The one bound above 1e-8, q-smooth
   $x_{\rm resp}=980$ at 1.63e-08, is argued from a named reference's own error, not from a
   margin:** it is eq. (22)'s declared rounding estimate at $u=0.01$, $x=1543$ (1.4e-08), which
   the `mpmath` check of §3.1 shows to be conservative. The price of not trusting a number the
   suite cannot check is that this one case is scored ~150× more loosely than its measured
   agreement (1.08e-10); a sign or factor-of-two error in the chain still misses it by eight
   orders. The test also asserts that no case's bound exceeds **1e-7**, an order above the
   loosest, so that a change to the rounding estimate that made test 6 vacuous would fail.
   A single uniform 1e-9 was considered and rejected: it sits below eq. (22)'s own declared error
   at all three q-smooth points, so it would depend on which libm rounds which term, and
   it would score the other six cases three orders more loosely than they deserve.

5. **Test 6 runs all nine cases; nothing was cut.** Not a deviation — recorded because prompt §3
   expected "minutes". The nine `evaluate_QuadSource_integral` calls at `(1e-45, 1e-12)` plus the
   oracle take **2.7 s** in total (0.04–0.6 s per case); the whole new module runs in ~3 s.

6. **Tests beyond §3's six.** Two methods check that `Phi`/`dPhi` and `Cin` meet across their
   series cut to rounding, with `Phi(0)=1`, `dPhi(0)=0`, $f\to4/3$ and `Cin`'s leading term; test 2
   also asserts the **next** order, $1-\alpha x^2$ with $\alpha = 1/20 + (u^2+v^2)/60$ derived in
   its docstring, not only $2x^2/9$; test 4 is two methods, on the resonance and approaching it.
   **IMPLEMENTATION CHOICE.** The series test is what catches deviation 2 being undone
   (row S1 of §3.2); $\alpha$ depends on $u,v$ where the leading term does not, so it checks the
   source's second-order term.

7. **The prompt header says this prompt "opens" `[01-general-w-normalisation-is-predicted-not-measured]`.
   It was already opened, on the board and at `docs/OPEN_ISSUES.md` §1.9, by the audit commit.**
   It is left exactly as it was — no second index row, no count change on its account.
   **STRUCTURALLY REQUIRED** (the index is one line per issue).

No **UNINTENDED DRIFT**.

## Verification performed

**Environment.** Branch `tolerance-convergence`, parent `f0dc880` (which changes only
`prompts/radiation-oracle/orchestrator/`; baseline `300d964`). `config/defaults.py` blob
`76bab78d9e9374263a91da87d86b509b2ed019d0` before and after. `black --check` clean on both new
files.

### Suites

| | parent `f0dc880` | this commit |
|---|---|---|
| `ComputeTargets` | **521**, `FAILED (failures=1)` — the known flake `test_tk_wkb_phase.TestCost.test_wall_time_per_object`; re-run alone three times: FAIL, FAIL, OK | **530**, `FAILED (failures=1)` — the same flake and nothing else; re-run alone three times: FAIL, FAIL, OK. An earlier full run on this tree, before the last test-only edits (the finiteness guards and the docstrings), was **530, OK** |
| `ComputeTargets` wall time | **3 min 50 s** (`time`; unittest reports 228 s) | **3 min 56 s** (`time`; unittest reports 233 s) — within run-to-run noise; two other runs on this tree took 3 min 36 s and 3 min 45 s |
| `CosmologyModels` | **39**, OK | **39**, OK |
| `test_kohri_terada_oracle` alone | — | **9**, OK, 2.9 s (4.9 s with interpreter start) |

### 3.1 The numbers, each beside its reference's own error

**Test 1 — eq. (22) against `quad` of eq. (15).** Bound: quad's declared error plus
$\epsilon\times$ eq. (22)'s rounding scale; a quad declaring worse than 1e-12 relative is refused.
The last column is |diff| as a fraction of the bound.

| v | u | x | eq. (22) | quadrature | rel | quad's own error | eq. (22) rounding | /bound |
|---|---|---|---|---|---|---|---|---|
| 0.7 | 1.1 | 1.0 | +2.053469779911e-01 | +2.053469779911e-01 | 8.25e-15 | 2.3e-15 | 8.1e-14 | 0.02 |
| 0.7 | 1.1 | 5.0 | +3.520250466749e-01 | +3.520250466749e-01 | 4.73e-16 | 8.7e-15 | 9.0e-15 | 0.01 |
| 0.7 | 1.1 | 20.0 | −2.023764954439e-01 | −2.023764954439e-01 | 4.11e-16 | 5.3e-15 | 4.1e-15 | 0.01 |
| 0.7 | 1.1 | 60.0 | +1.593532404202e-01 | +1.593532404202e-01 | 6.97e-16 | 3.0e-15 | 2.1e-15 | 0.02 |
| 1.0 | 1.0 | 10.0 | +2.416399669992e-01 | +2.416399669992e-01 | 6.89e-16 | 8.7e-15 | 2.4e-15 | 0.01 |
| 0.3 | 1.6 | 12.0 | −2.570546556294e-01 | −2.570546556294e-01 | 1.94e-15 | 4.1e-15 | 6.0e-15 | 0.05 |
| 1.2 | 0.5 | 25.0 | −1.770406660654e-01 | −1.770406660654e-01 | 9.41e-16 | 4.4e-15 | 4.6e-15 | 0.02 |
| 0.9 | 0.87 | 40.0 | +4.501650138327e-02 | +4.501650138327e-02 | 4.32e-15 | 4.3e-15 | 2.7e-15 | 0.03 |
| 0.86 | 0.87 | 15.0 | +1.368699956249e-01 | +1.368699956249e-01 | 1.01e-15 | 8.5e-15 | 4.6e-15 | 0.01 |
| 0.9 | 0.9 | 15.0 | +2.026653071225e-01 | +2.026653071225e-01 | 4.11e-16 | 8.2e-15 | 2.7e-15 | 0.01 |
| 1.2 | 1.0 | 14.43 | +1.164056489442e-01 | +1.164056489442e-01 | 0.00e+00 | 5.1e-15 | 7.6e-16 | 0.00 |
| 12.0 | 10.0 | 4.33 | −2.083240807992e-02 | −2.083240807992e-02 | 3.33e-16 | 2.5e-16 | 6.6e-17 | 0.02 |
| 1.1 | 0.01 | 47.24 | −6.645716679507e-02 | −6.645716679701e-02 | 2.93e-11 | 9.4e-15 | 2.2e-10 | 0.01 |

Worst on the audit's ten: **8.25e-15**, against the audit's 3.5e-15 (deviation 3). The last row is
the q-smooth shape: eq. (22) there is good to 3.3e-09 relative by its own estimate and 2.9e-11 in
fact.

**Test 2 — small $x$**, at $(v,u)=(0.7,1.1)$; $\alpha = 0.078\overline{3}$.

| x | eq. (22) / $\tfrac29x^2$ | quad / $\tfrac29x^2$ | $1-\alpha x^2$ | eq. (22)'s own rounding | quad's own error |
|---|---|---|---|---|---|
| 0.3 | 0.9929696981 | 0.9929696981071 | 0.9929500000000 | 3.2e-11 | 1.1e-14 |
| 0.1 | 0.9992169103 | 0.9992169101831 | 0.9992166666667 | 2.5e-09 | 1.1e-14 |
| 0.03 | 0.9999295875 | 0.9999295019728 | 0.9999295000000 | 3.0e-07 | 1.1e-14 |
| 0.01 | 0.9999867886 | 0.9999921666910 | 0.9999921666667 | 2.4e-05 | 1.1e-14 |

The quadrature follows $1-\alpha x^2$ to $O(x^4)$ with coefficient 1.9e-3 to 3.2e-3 over the five
pairs tested (bound 1e-2); eq. (22) follows it to within its own rounding. **The audit's
"0.99999 of $2x^2/9$ at $x=0.01$" is 0.9999861 there and 0.9999868 here: both are eq. (22)'s
rounding (5.4e-06 in fact, 2.4e-05 by its estimate) on top of the true 0.9999922.** Nothing asserts $x^2/2$.

**Test 3 — large $x$.** $D(X)$ = max $|$eq. (22) − eq. (25)$|$ over four periods from $X$,
divided by eq. (25)'s envelope. Per-decade factor, $X = 200\to2000\to2\times10^4\to2\times10^5$:

| (v, u) | factors | $X\cdot D$ |
|---|---|---|
| (0.7, 1.1) | 0.103, 0.101, 0.100 | 11.6, 11.9, 12.1, 12.0 |
| (1.2, 0.5) | 0.108, 0.101, 0.100 | 33.6, 36.4, 36.8, 36.8 |
| (0.3, 1.6) | 0.101, 0.101, 0.099 | 18.7, 18.9, 19.1, 18.8 |
| (1.0, 1.0) | 0.101, 0.100, 0.100 | 7.37, 7.46, 7.48, 7.48 |
| (0.9, 0.87) | 0.105, 0.099, 0.100 | 16.3, 17.1, 16.9, 17.0 |

Asserted $[0.08, 0.125]$ per decade. Eq. (22)'s rounding scale is asserted < 1e-12 of the envelope
at every sample, so it plays no part. The audit's single-point relative figures (1.03e-01 →
7.06e-04) are not used: they are dominated by where the zeros fall and are non-monotone for
(0.3, 1.6).

**Test 4 — the resonance**, $u=v=\sqrt3/2$ exactly ($1-(u+v)/\sqrt3 = 0.0$ is asserted).

| x | eq. (22), regularised | quadrature | rel | quad's own error | eq. (22) rounding |
|---|---|---|---|---|---|
| 0.5 | +5.452171051464197e-02 | +5.452171051463229e-02 | 1.8e-13 | 6.1e-16 | 2.7e-13 |
| 15 | **+1.391877548291586e-01** | +1.391877548291586e-01 | **0** | 8.5e-15 | 4.5e-15 |
| 150 | +2.719432985120082e-02 | +2.719432985120087e-02 | 1.8e-15 | 1.7e-15 | 2.8e-15 |

Against the audit's **0.1391877548292** (printed to 13 digits): |diff| **4.1e-14**, held to half a
unit in the last printed digit, 5e-14. At $x=0.5$ the difference is eq. (22)'s own small-$x$
rounding, inside its estimate. Approaching the resonance at $x=15$, the regularised form against
the quadrature (bound: quad's error plus eq. (22)'s rounding, ~1.3e-14 at every gap), and the
literal eq. (22) for comparison:

| $u+v-\sqrt3$ | regularised − quad | quad's own error | literal eq. (22) − quad |
|---|---|---|---|
| 1e-2 | 5.0e-16 | 8.4e-15 | 4.7e-15 |
| 1e-4 | 3.1e-16 | 8.5e-15 | **1.9e-13** |
| 1e-6 | 0 | 8.5e-15 | **1.8e-11** |
| 1e-8 | 1.9e-16 | 8.5e-15 | **1.7e-09** |
| 0 | 0 | 8.5e-15 | **inf** |

**Test 5 — eq. (16) against eq. (20)** on the audit's 25 points: worst **1.58e-12** relative, at
$x=0.5$, against eq. (20)'s own rounding estimate there of 2.3e-12 to 1.0e-11 (the bound is that
plus 16 ulp). Both formulas share the rounded arguments $ux/\sqrt3$, $vx/\sqrt3$, so they agree
with each other better than either agrees with `mpmath` (up to 1.1e-12 at $x=100$).

**Test 6 — the pipeline against eq. (22) with the head subtracted**, exact flavour, `(1e-45, 1e-12)`.
"Oracle err" is $\epsilon\times$ eq. (22)'s rounding scale plus the head's quad error; "pipe err" is
`max(total_abserr, rtol·(|numeric_quad|+|WKB_Levin|))`; both relative. The bound on $|N+9/8|$ is
$\tfrac98$(oracle + pipe).

| shape | $x_{\rm resp}$ | $x = k\tau$ | `total` | $N$ | $\lvert N+9/8\rvert$ | oracle err | pipe err | bound | $\lvert L\rvert/\lvert\text{total}\rvert$ |
|---|---|---|---|---|---|---|---|---|---|
| together | 980 | 1.5560e+03 | +9.2632298681e-13 | −1.1249999999999 | 6.91e-14 | 1.3e-13 | 8.7e-12 | 9.93e-12 | 3.846 |
| together | 100 | 1.5877e+02 | −1.4546904149e-10 | −1.1250000000000 | 2.00e-15 | 8.7e-15 | 1.6e-12 | 1.83e-12 | 0.309 |
| together | 30 | 4.7631e+01 | −1.9958705760e-10 | −1.1250000000000 | 8.88e-16 | 2.5e-14 | 1.0e-12 | 1.15e-12 | 0.386 |
| T-first | 980 | 1.4145e+02 | +7.4690421983e-11 | −1.1250000000000 | 2.42e-14 | 1.7e-14 | 4.8e-12 | 5.39e-12 | 2.885 |
| T-first | 100 | 1.4434e+01 | −8.4352152992e-09 | −1.1250000000000 | 2.98e-14 | 2.8e-15 | 1.0e-12 | 1.13e-12 | 0.099 |
| T-first | 30 | 4.3301e+00 | +2.3437794771e-08 | −1.1250000000000 | 1.11e-14 | 3.2e-15 | 1.0e-12 | 1.13e-12 | 0.135 |
| q-smooth | 980 | 1.5431e+03 | −8.5701377049e-12 | −1.1249999998915 | **1.08e-10** | 1.4e-08 | 5.2e-12 | **1.63e-08** | 2.099 |
| q-smooth | 100 | 1.5746e+02 | +1.5578803620e-10 | −1.1249999999255 | 7.45e-11 | 7.4e-09 | 2.2e-12 | 8.31e-09 | 0.579 |
| q-smooth | 30 | 4.7238e+01 | +7.4905262167e-10 | −1.1250000000282 | 2.82e-11 | 3.3e-09 | 2.9e-12 | 3.67e-09 | 1.943 |

The `total` column reproduces the audit's §7.1 to every printed digit. **$N=-9/8$ reproduces**,
so no stop condition fired. Eq. (22)'s actual error at the three q-smooth points, against
50-digit `mpmath`: **9.67e-11, 6.62e-11, 2.50e-11** — the whole of the q-smooth deviation. At the
other six points it is 0 to 2.9e-15. The head's declared quad error is 1e-20 to 1.4e-18 of
$I_{\rm RD}$ at all nine.

**The Levin half.** $|L|/|\text{total}|$ is 0.099 to 3.846 and exceeds 1 in four cases, as the
audit says. The agreement bounds a Levin-half error by $|\text{total}/\text{predicted}-1|\cdot
|\text{total}|/|L|$: **1.7e-15 to 2.7e-13** on the six cases where eq. (22) is well conditioned.
On q-smooth the same expression gives 1.3e-11 to 1.1e-10, but that is eq. (22)'s rounding, not a
statement about Levin. (The audit's 8.0e-14 to 3.1e-12 came from its comparison against the
quadrature, a different reference.) The three limits of prompt §3.6 are in the test's docstring:
it bounds the combination; it does **not** reach `analytic_rad`'s `_three_bessel_Levin`, so
`[06-analytic-rad-is-computed-at-the-callers-tolerance]` on `qsi-phase-groups` is untouched; and
it is the exact flavour only.

**Eq. (22)'s rounding estimate is conservative.** Against 50-digit `mpmath` evaluation of the
same formula, the actual error is **at most 0.33** of $\epsilon\times$ `scale` on two sets: 174
points (the 24 of this log — the audit grid, four small-$x$ points, the resonance at three $x$, the
pipeline shapes — plus 150 random) with median 0.04, and 307 points (7 fixed including the three
q-smooth, plus 300 random with $u\in[0.01,16]$, $v$ in the triangle $|1-u|\le v\le 1+u$,
$|v-u|<\sqrt3$, $x\in[0.03,3000]$). The largest ratios are at $u,v\approx13$, $x\approx1$ and at
$x=0.03$. It is loosest where it matters most for test 6: 0.007 to 0.009 at the three q-smooth
points.

### 3.2 Deliberate breakage — which trap each test catches

Each mutation was applied to an in-memory copy of the module (or, for T2, of the test file),
installed in `sys.modules` in place of the real one, and the whole test module run against it;
nothing in the tree was modified. "Fails" lists the methods with at least one failing subtest;
the rest passed.

| # | Trap | Mutation | Fails (methods) | Passes |
|---|---|---|---|---|
| **1** | Ci/Si pairing inverted | the PDF reading: `+Ci`, `−Si` on the $(v+u)$ pair, `−Ci`, `+Si` on the $(v-u)$ pair, literal log | **test 1** (13/13 subtests; e.g. $-12.22$ vs $+0.2053$ at $x=1$), **test 2** (18; −679 × $2x^2/9$ at $x=0.3$), **test 3** (12; the difference does not fall, factor 1.007/decade), **test 4 on** (−inf) and **approaching** (5/5), **test 6** (9/9; $N = -0.055$) | series ×2, test 5 |
| 1b | | only the Si pairing inverted | tests **1, 2, 3, 4 (both), 6** | series ×2, test 5 |
| 1c | | only the Ci pairing inverted | tests **1, 2, 3** (4 subtests), **4 (both), 6** ($N=-1.087$) | series ×2, test 5 |
| **2** | small $x$ | implementation scaled by 9/4 so that it meets the paper's $x^2/2$ | tests **1, 2** (ratio 2.234), **3, 4 (both), 6** ($N=-0.222$) | series ×2, test 5 |
| T2 | | **test** expects $x^2/2$, correct module | **test 2** (39 subtests: quadrature 0.4413 of $x^2/2$) — i.e. the paper's remark rejects a correct implementation | the other eight |
| **3** | eq. (20) as the source | `f_RD` returns eq. (20) | **test 1** (13; quad declares 3.8e-07, refused), **test 2** (1 subtest, quadrature at small $x$), **test 6** (9; the head's error blows the bound to 6.1e-05, over the 1e-7 cap), **series join** ($f(10^{-8}) = 5\times10^{18}$) | tests 3, 4 (both), 5 — test 5 compares eq. (20) with itself |
| **4** | resonance | literal $-\mathrm{Ci}(\lvert c\rvert x) + \log\lvert\ldots\rvert$ | **test 4 on** (4 subtests: **`I_RD = inf on the resonance`**), **approaching** (4: gaps 1e-4, 1e-6, 1e-8 miss by 1.9e-13, 1.8e-11, 1.7e-09 against a bound of ~1.3e-14, and gap 0 is `inf`), **test 2** (resonant pair: `eq. (22) = inf`) | tests 1, 3, 5, 6, series ×2 |
| 4b | | literal form with $c$ nudged to 1e-15 on the resonance | **test 4 on** (miss by 9.99 at $x=0.5$), **approaching** (4), **test 2** (4) | the rest |
| 4c | | guard returning `nan` on the resonance | **test 4 on** (`I_RD = nan`), **approaching** (1), **test 2** (4) | the rest |
| K1 | chain | `KT_NORM = -9/4` (the $h^{\rm us}=h^{\rm them}/2$ dropped) | **test 6 only** (9/9; $\lvert N+9/8\rvert = 1.125$) | the other eight |
| K2 | chain | `KT_NORM = +9/8` (orientation sign "fixed") | **test 6 only** (9/9) | the other eight |
| K3 | kernel | Green's function $\sin(x-\bar x)\to\sin(x+\bar x)$ in the quadrature only | tests **1, 2, 4 (both), 6** (via the head, $\lvert N+9/8\rvert = 9.8\times10^{-7}$) | test 3, 5, series ×2 |
| S1 | not a trap | `Phi`/`dPhi` series cut back to `kt_verification.py`'s $10^{-2}$ | **series join** (3.4e-14 > 1.4e-14) | the other eight |

**Reading the table.** Trap 1 is caught by six of nine methods, and by tests 1, 2, 3, 4 and 6
separately — the suite's coverage of the most dangerous erratum is not one assertion thick. Trap 4
is caught by test 4 on the resonance with an explicit `inf` (and `nan`, and a 9.99 miss for the
nudge). K1 and K2 are the prefactor-chain mutations board §5 note 1 says test 6 regression-guards;
only test 6 sees them, which is the audit §0 point made concrete.

**Two holes the breakage found and the commit closes, both the same shape.** Every closed-form
bound is built from eq. (22)'s own rounding scale, and an `inf` in eq. (22) makes the scale `inf`
too. On the first run trap 4 therefore passed test 2 — the branch that skips eq. (22) where its
rounding exceeds 1e-3 skipped the resonant pair — and passed the gap-0 subtest of test 4's
approach, where `inf <= inf` holds. Tests 1, 2 and 4 now assert that the closed form **and** its
scale are finite before comparing; the table above is the re-run on the final code.

## Observations not acted on

1. **The audit misattributes its 3.18e-10.** See the Result. The same attribution is in campaign
   README §1, board item R1, prompt §3.6 and §6, and the audit's
   §1 table ("Limited by: the head subtraction") and §7.1 preamble. None of those files is this
   prompt's to edit beyond the board, and the board's own wording is in item R1's history, which I
   have left as written and annotated in the status column. **Opened
   `[01-the-eq22-figure-is-eq22s-own-rounding-not-the-head]`.**

2. **`kt_verification.py`'s `dPhi` is off by up to 1.1e-06** just above its $10^{-2}$ series cut
   (deviation 2). No audit figure visibly depends on it — its §6 quadrature column reproduces to all
   printed digits — so this is recorded, not opened.

3. **Eq. (20)'s error grows like $\epsilon/x^4$, not "noise below $x\approx0.1$".** Measured: 1.2e-09
   of $f$ at $x=0.1$, 1.2e-07 at 0.03, 8.1e-06 at 0.01, 3.1e-04 at 0.003, 7.1e-02 at 0.001. The
   audit's "returns noise below $x\approx0.1$" overstates where it fails as a *function*; its
   nine-orders figure for the *integral* at $x=0.03$ is a statement about integrating from
   $\bar x=0$, where it diverges, and stands. The module's docstrings give the measured growth.

4. **Eq. (22) is ill-conditioned at small $u$** — at $u=0.01$ its error is ~1e-10 of $I$ in fact
   and 3e-9 to 1.4e-8 by its own conservative estimate. A better-conditioned form there (a small-$u$
   expansion) would tighten test 6 on q-smooth by two orders. Not attempted: the prompt lands
   eq. (22), and the regularisation it asks about (§6, fourth stop) is the resonance's, which is
   sufficient — the regularised form matches quadrature to 0 at $x=15$ and to within quad's own
   error at $x=0.5$ and 150.

5. **The quadrature's error estimate carries the tests' weight.** Tests 1, 2 and 4 are bounded by
   quad's declared error; the observed differences were 0.00–0.05 of the bound here, and 50-digit
   `mpmath` agrees with the quadrature to ≤ 1.7e-16 at the four points checked ($x = 0.3$ to 2). No action.

## State handed to the next prompt

There is no next prompt: this campaign has one. The state it leaves:

- **The oracle is in the tree** at `ComputeTargets/tests/kohri_terada.py`, $b=0$ only, with
  `total_from_I_RD` as the single statement of $\text{total} = -\tfrac98\,k_{\rm phys}^{-2}
  (I_{\rm RD} - \text{head})$. `test_kohri_terada_oracle` pins it against `quad`, the two limits,
  the resonance, eq. (20) and the pipeline; it runs in ~3 s and all nine $b=0$ pipeline cases are
  in it.
- **Board item R1 is done.** Campaign acceptance (README §5) is met: prompt 01 is ✅, the closed
  form is pinned against `scipy.quad` and against `total` at $N=-9/8$, the resonance is finite and
  correct at $u+v=\sqrt3$, and `ComputeTargets` (530) and `CosmologyModels` (39) pass at no lower
  count than the campaign started with.
- **Two issues are open on this board**, both indexed at `docs/OPEN_ISSUES.md` §1.9:
  `[01-general-w-normalisation-is-predicted-not-measured]` (unchanged — the $b=0.2$ cases still
  have no oracle) and `[01-the-eq22-figure-is-eq22s-own-rounding-not-the-head]` (new — the audit's
  attribution of 3.18e-10).
- **Nothing on another campaign's board moved.** In particular
  `[06-analytic-rad-is-computed-at-the-callers-tolerance]` stays open: the oracle scores `total`,
  and `analytic_rad`'s `_three_bessel_Levin` is a separate path it does not reach.
- **The spec is not edited.** `docs/spec/cross-spec-check.md` §3 item 9's Kohri–Terada half — the
  $h^{\rm us}_{ij}=h^{\rm them}_{ij}/2$ that `docs/spec/05-one-loop.md` §539 open question 11
  records as unchecked — is now checked by a test in the suite (test 6, with K1 of §3.2 showing it
  fails if the $\tfrac12$ is dropped). Recording that in the spec is `prompts/spec-transcription`'s
  and needs the author's sign-off. The Adshead half is untouched.
