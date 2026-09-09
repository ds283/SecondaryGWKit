# Prompt 07 — Preserve the leading term in three-Bessel phase groups

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design:** `DRAFT-PLAN.md` §8.2 (phase-group structure), §9 Stage 4
**Reconciliation items:** §3.2 (the analogous `QuadSourceIntegral` site is out of scope), §3.3
**Depends on:** 06 (hard — needs the residual accessors and `theta_abserr`)
**Recommended model:** Opus
**Files you may touch:** `LiouvilleGreen/three_bessel_integrals.py`,
`LiouvilleGreen/tests/test_three_bessel.py`, plus the log and the status board.
**Do not touch:** `LiouvilleGreen/bessel_phase.py`, `AdaptiveLevin/`,
`LiouvilleGreen/tests/test_3bessel_analytic.py` (prompt 08's), anything under `ComputeTargets/`.

Read first: README §2 (a) and §6 (the phase-group acceptance row); `RECONCILIATION.md` §3.2;
`DRAFT-PLAN.md` §8.2 and §9 Stage 4; then `LiouvilleGreen/three_bessel_integrals.py` in full,
especially `BesselIntegralResult`'s docstring (`:10-40`), `_phase_group` (`:163-208`) and
`_Levin_3bessel` (`:211-278`); and `logs/06-evaluation-and-compatibility.md`
§"State handed to the next prompt".

---

## 1. Character of this commit

Consumers can reintroduce cancellation and large-phase error *after* the individual Bessel functions
have been fixed. This prompt closes that gap for `three_bessel_integrals.py`, which is the only
in-scope consumer that sums three Bessel phases (README §1.1 and `RECONCILIATION.md` §3.2 explain
why `ComputeTargets/QuadSourceIntegral.py`'s analogous — and worse — call sites are excluded).

Current state: `_phase_group` (`:175-202`) builds each group by summing three independently
reconstructed **raw** phases,

```python
phase_mu.raw_theta(k * x) + e_nu * phase_nu.raw_theta(q * x) + e_sigma * phase_sigma.raw_theta(s * x)
```

and likewise sums three `theta_mod_2pi` values and three log-derivatives. Each `raw_theta` is
\(\varepsilon\theta\)-limited, so near resonance — where the group phase is a small difference of
large phases — the sum's *absolute* error is set by the largest constituent, not by the result.
`BesselIntegralResult`'s own docstring already identifies the shape of this problem ("four group
values of order one cancelling to 1e-6, each accurate to 1e-12, give a relative error of 1e-6") and
notes that nothing supplies `theta_abserr`. Prompt 06 changed that; this prompt uses it.

## 2. The restructure

`DRAFT-PLAN.md` §8.2. For a shared variable \(t\) (here \(t=x\), and the Levin integration variable
is \(\log x\)), assemble each group as

$$\theta_\mu(kt)+\epsilon_\nu\theta_\nu(qt)+\epsilon_\sigma\theta_\sigma(st)=Kt+C+R(t),$$
$$K=k+\epsilon_\nu q+\epsilon_\sigma s,\qquad
C=c_\mu+\epsilon_\nu c_\nu+\epsilon_\sigma c_\sigma,\qquad
R(t)=r_\mu(kt)+\epsilon_\nu r_\nu(qt)+\epsilon_\sigma r_\sigma(st).$$

Three requirements:

1. **Combine the leading coefficients before multiplying by \(t\).** \(K\) is formed once, from the
   three wavenumbers, at group-construction time. Near resonance \(K\) is a small difference of
   \(O(1)\) numbers — that cancellation is in the *inputs* and is unavoidable, but it must happen
   once in the coefficient rather than once per evaluation in a product of size \(\theta\). This is
   the entire point: \(\delta(Kt)=t\,\delta K\) with \(\delta K\sim\varepsilon\max(k,q,s)\), versus
   \(\delta(\sum\theta)\sim\varepsilon\max(\theta)\sim\varepsilon\max(k,q,s)\,t\) — the same size,
   *except* that \(\delta K\) is exact when \(K\) is exactly representable and is a single rounding
   otherwise, whereas the phase sum accumulates three interpolation errors and three
   \(\varepsilon\theta\) roundings. Say this in the docstring with the estimate, not just "for
   accuracy".
2. **Combine the residuals separately.** \(R(t)\) is a sum of three \(O(1)\)-to-\(O(\nu)\) smooth
   quantities (bounded by \(\sim571\) rad at the largest supported order —
   `RECONCILIATION.md` C2), so its sum is well conditioned and its error is the residual
   interpolation error, not \(\varepsilon\theta\).
3. **Form group derivatives from the same expression:**
   \(d\theta_{\rm group}/d\log x=x\bigl(K+R'(x)\bigr)\) where
   \(R'=k\,r_\mu'(kx)+\epsilon_\nu q\,r_\nu'(qx)+\epsilon_\sigma s\,r_\sigma'(sx)\); equivalently, sum
   the constituent log-derivatives, which is what the code does now. **The current derivative route
   is not obviously wrong** — it sums \(\theta'\) values, each of which prompt 05 supplies as
   \(e^{-2\ell}\), a well-conditioned primitive. Measure both and take the better, and say which and
   why. Do not assume the restructure improves the derivative just because it improves the value.

Use **appropriately accurate summation** for the three-term sums (§8.2). Three terms is small
enough that `math.fsum` or a two-sum compensated add is cheap; state what you used. Do not
over-engineer: the dominant error is the interpolants', not the summation's, and a comment saying so
is more useful than a Kahan loop nobody needs.

### 2.1 The bounded-angle path

`theta_mod_2pi` for a group should be built from the split: evaluate \(\sin\) and \(\cos\) of
\(Kx+C+R\) by angle addition on \((Kx)\) and \((C+R)\), then `atan2`. Do **not** sum three bounded
angles and hope — that is what the code does now (`:184-191`), and while the sum of three bounded
angles is bounded-ish, it is not the bounded angle of the sum, and the reduction error of each
constituent enters additively.

Note that \(Kx\) may itself be large, so `sin(K*x)`/`cos(K*x)` must be handed the unreduced product
and left to libm's Payne–Hanek reduction — the same argument as prompt 05 §2.4, and the same one
`LiouvilleGreen/range_reduce_mod_2pi.py`'s docstring makes.

### 2.2 `theta_abserr` pass-through

Every `adaptive_levin_sincos` call in this module must now pass a `theta_abserr` for its group,
combined from the three constituents' declared errors (`DRAFT-PLAN.md` §8.1: "its consumers should
pass it through"). Combine **linearly**, not in quadrature, for the reason `BesselIntegralResult`'s
docstring already gives for the group values: the three phases share a construction, so a
systematically inaccurate phase produces a common drift rather than independent noise.

Then **update `BesselIntegralResult`'s docstring**. As written (`:24-33`) it says the phase/modulus
fit error "is invisible from inside this module", quotes a measured ~2e-8 floor, and says
"nothing in `LiouvilleGreen/` supplies one yet". All three statements become false with this commit.
Rewrite that paragraph to say what `abserr` now includes, what it still does not, and what the new
floor measures at. Leave the *reasoning* about linear combination intact — it is correct and still
applies.

## 3. Tests

`LiouvilleGreen/tests/test_three_bessel.py`. It currently builds two `bessel_phase` objects and
integrates single Bessel functions (`:51-107`) — it does not exercise a three-phase group at all.
Extend it; do not remove what is there.

`DRAFT-PLAN.md` §9 Stage 4 and README §6's phase-group row require:

1. **Exact and near cancellation.** Triples with \(K\) exactly zero (e.g. \(k=q+s\) with
   \(\epsilon_\nu=\epsilon_\sigma=-1\)), \(K\) small (\(K/\max(k,q,s)\sim10^{-6},10^{-10}\)) and
   \(K\) generic. For each, compare the group phase and its derivative against an independent
   reference built from `bessel_reference` (prompt 01) — **not** from the objects under test.
2. **Absolute derivative checks scaled to constituent frequencies**, and **no division by a
   vanishing group derivative** (README §6). This is the trap: relative error in \(d\theta_{\rm group}\)
   is meaningless when the group derivative passes through zero, which it does at exact resonance.
   Score \(\lvert\delta\theta'_{\rm group}\rvert\) against \(\max(k,q,s)\), not against
   \(\theta'_{\rm group}\).
3. **Three-Bessel values and integrals.** `quad_JJJ` and `quad_YJJ` against the closed forms
   `test_3bessel_analytic.py` already encodes — but do **not** edit that file (prompt 08). Import or
   re-derive one or two closed forms here for a smoke check, or reference the existing test and
   assert only the new structural properties, and say which you chose.
4. **Amplitude and phase sign conventions** (§9 Stage 4): assert \(J=A\sin\theta\) and
   \(Y=-A\cos\theta\) hold for each constituent as reconstructed through the group machinery, so a
   sign slip in \(\epsilon_\nu,\epsilon_\sigma\) cannot hide.
5. **Quadrature error is not a certificate of Bessel accuracy** (§9 Stage 4, README §6). Do not
   read agreement under quadrature refinement as evidence the phases are accurate. Concretely:
   include one test that refines the quadrature tolerance and asserts the *result stops improving*
   at a floor, and that the floor is consistent with the declared `theta_abserr`. That is the
   positive statement; asserting "the integral matches to 1e-12 therefore the phase is good" is the
   error to avoid.

**A note on what may happen.** §9 Stage 4 warns that these tests' tolerances were set against the
old accuracy and that a six-to-eight-order improvement "can expose a different limiting error rather
than simply passing more easily". If a test that used to pass now fails, or a tolerance can be
tightened by six orders in one case and only one order in another, that asymmetry is a **finding**:
record the numbers and, if you cannot attribute it, stop. Prompt 08 owns re-tightening
`test_3bessel_analytic.py`; if your work shows its \(10^{-5}\)/\(10^{-6}\) tolerances are now
limited by something other than the Bessel phase, hand that over explicitly.

## 4. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_three_bessel -v` passes.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` passes,
  including `test_3bessel_analytic.py` **at its existing tolerances** (`ABS_TOLERANCE = 1e-6`,
  `REL_TOLERANCE = 1e-5`, and `1e-3`/`1e-2` near singularities, `test_3bessel_analytic.py:15-19`).
  Do not adjust them here.
- The near-resonant cases show a measurable improvement over the pre-commit code. Measure it: run
  the same triples against `HEAD~1`'s `_phase_group` (keep a copy, or measure before you edit) and
  quote both numbers. A restructure justified only by an argument, with no measurement, does not
  meet this prompt.
- Every `adaptive_levin_sincos` call in the module passes `theta_abserr`.
- `BesselIntegralResult`'s docstring no longer claims nothing supplies `theta_abserr`.

## 5. Log and commit

Follow README §5 and §5.1. Quote, as numbers:

- \(\lvert\delta\theta_{\rm group}\rvert\) and \(\lvert\delta\theta'_{\rm group}\rvert\) before and
  after, for exact-zero, \(10^{-6}\), \(10^{-10}\) and generic \(K\);
- which derivative route won (§2 item 3) and by how much;
- the summation scheme used;
- the new `abserr` floor for `quad_JJJ`/`quad_YJJ`, against the ~2e-8 the old docstring records;
- any tolerance in `test_3bessel_analytic.py` that now looks limited by something other than the
  Bessel phase, handed to prompt 08.

Commit subject, or something equally specific: `Preserve the leading term in Bessel phase groups`.
