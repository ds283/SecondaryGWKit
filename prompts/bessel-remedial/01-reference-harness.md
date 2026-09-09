# Prompt 01 — Independent references and the measurement harness

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design:** `DRAFT-PLAN.md` §6.1 (error definitions), §9 Stage 1, §10 (acceptance table), §12 (reproduction)
**Reconciliation items:** C1.3 (`mpmath` is mandatory above \(2.5\times10^{15}\)), C4 (a third existing derivative test)
**Depends on:** nothing. Everything else in the campaign depends on this.
**Recommended model:** Opus
**Files you may touch:** new `LiouvilleGreen/tests/bessel_reference.py`, new
`LiouvilleGreen/tests/test_bessel_reference.py`, new `LiouvilleGreen/tests/bessel_reference_data.json`,
new `docs/bessel-remedial/` (directory) containing `measure_bessel_phase.py` and its recorded
output, plus the log and the status board.
**Do not touch:** `LiouvilleGreen/bessel_phase.py` or any other production module. **This prompt
changes no production code at all.**

Read first: README §2 (the six design facts) and §6 (the acceptance table and the error
definitions); `RECONCILIATION.md` §1, C1 and C4; `DRAFT-PLAN.md` §6.1, §9 Stage 1 and §12.1;
then `LiouvilleGreen/bessel_phase.py` in full and `LiouvilleGreen/tests/test_bessel_phase.py`
in full.

---

## 1. Character of this commit

Measurement infrastructure, and a **recorded pre-change baseline**. Nothing in production changes.

The campaign replaces a construction whose accuracy the current test suite cannot see: the basic
value test allows 50 % relative error (`test_bessel_phase.py:34`) and the high-order test allows
\(10^{-3}\), with a comment saying so explicitly (`:111-113`). A campaign that claims a
six-to-eight-order improvement needs to be able to *measure* six to eight orders, against
references that do not share the error being measured.

Two independence requirements make this non-trivial, and they are the reason this prompt exists
separately:

- **References built from `bessel_phase` itself would conceal common error.** Several existing
  fixtures do exactly that (`ComputeTargets/tests/test_tk_source_functions.py:228-266` defines its
  "exact" phase as \(\pi-\vartheta(x)\) from `bessel_phase`).
- **References built only from `jv`/`yv` share the Amos library with `hankel1e`.** That is not
  merely a theoretical concern: above \(x\approx2.5\times10^{15}\), `jv`/`yv` are O(1)-relatively
  noisy (`RECONCILIATION.md` C1), so at the top of the domain they are not a reference at all.
  `mpmath` is the only usable reference there.

## 2. What to build

### 2.1 `LiouvilleGreen/tests/bessel_reference.py`

An importable helper module (not a test module — it must contain no `TestCase`). It is the single
place the campaign's metrics and references live; prompts 03–08 import it and must not redefine
any of it.

**Error metrics**, exactly as `DRAFT-PLAN.md` §6.1 defines them and README §6 repeats. Give each a
docstring stating why it has that form (a phase-pair error avoids dividing by a function near a
zero, and normalizes by the envelope):

```python
def phase_pair_error(sin_theta, minus_cos_theta, J, Y, amplitude): ...   # -> E_theta
def amplitude_error(A_ours, amplitude): ...                             # -> E_A
def derivative_error(theta_prime_ours, theta_prime_ref): ...            # -> relative
```

Each should accept scalars or arrays and return `(max_error, argmax_index)` — **the location of the
maximum is a required output, not a convenience.** `DRAFT-PLAN.md` §4.7 finds that every derivative
maximum falls in the interval adjacent to the turning point, and prompts 04 and 05 must be able to
confirm that on their own output.

**Reference tiers**, in increasing cost, with an explicit selector so a caller states which tier it
is using rather than getting one by accident:

1. `exact_half_integer(nu, x)` — closed forms for \(\nu=1/2,3/2,5/2\) (and \(-1/2\) if it falls out
   for free). \(J_{1/2}=\sqrt{2/\pi x}\sin x\), \(Y_{1/2}=-\sqrt{2/\pi x}\cos x\), and the
   recurrences upward. These are the only references with **no** shared library at all, so use them
   wherever the order allows. Build them from `math.sin`/`math.cos` on the supplied `x`, which
   README §2 and `RECONCILIATION.md` §1 confirm are correctly rounded to \(10^{16}\).
2. `scipy_reference(nu, x)` — `jv`/`yv` plus `hypot`. Must **refuse**, by raising, for
   \(x>\texttt{SCIPY\_REFERENCE\_MAX\_X}\), a module constant you set to \(2\times10^{15}\) with a
   comment pointing at `RECONCILIATION.md` C1 and at prompt 02's test. A silent wrong answer here
   is the exact failure this campaign is fixing; do not let a reference reproduce it.
3. `mpmath_reference(nu, x, dps=70)` — `besselj`/`bessely` at 70 digits. Convert the argument with
   `mpmath.mpf(float(x))`, **not** `mpf(repr(x))`: NumPy scalars stringify as
   `np.float64(...)` and `mpf` rejects that. The point of `mpf(float(x))` is that the reference is
   evaluated at *exactly the supplied double*, which §7.5 of the plan requires and which spot
   checks against a re-derived argument would violate.

For each tier also supply the exact derivative oracle
\(\theta'(x)=(2/\pi)/\bigl(x(J_\nu^2+Y_\nu^2)\bigr)\), which is an identity and not an
approximation, and the unwrapped reference phase \(\theta=\operatorname{atan2}(J,-Y)\) with the
branch convention documented (this is the anchor form `DRAFT-PLAN.md` §4.3 recommends in place of
the root solve).

### 2.2 Cached corner references, `bessel_reference_data.json`

`mpmath` at 70 digits over a large grid is too slow for a test that runs on every commit, but the
corners must be asserted permanently. So: a **generator** in `bessel_reference.py` and a **cached
table** committed alongside it.

Cover, at minimum, the \((\nu,x)\) corners `DRAFT-PLAN.md` §9 Stage 1 names:

- orders \(1/2,\,3/2,\,7/4,\,5/2,\,20.5,\,100.5,\,1000.5\);
- for each, \(x\in\{x_0,\;1.5x_0,\;10\nu,\;100\nu,\;10^3,\;10^7,\;10^{12},\;10^{15}\}\) restricted
  to those above that order's \(x_0=\sqrt{\nu^2-\tfrac14}\), plus \(2.5\times10^{15}\) and
  \(10^{16}\) for the orders whose supported domain reaches there;
- store \(J_\nu\), \(Y_\nu\), \(A=\operatorname{hypot}\), \(\theta=\operatorname{atan2}(J,-Y)\),
  \(a_\nu=\sqrt{\pi x/2}\,A\), \(r_\nu=\theta-x-c_\nu\) reduced to its correct branch, and
  \(\theta'\) — each as a decimal string with at least 30 significant digits, so the file is not
  itself limited to double precision.

**Record the environment in the file**: `scipy.__version__`, `numpy.__version__`,
`mpmath.__version__`, `platform.platform()` and the date. `DRAFT-PLAN.md` §9 Stage 1 requires this
because §4.4's boundaries are properties of the bundled Amos library, not guarantees.

Note when building \(r_\nu\): the naive fold of \(\theta-x-c_\nu\) into \((-\pi,\pi]\) is **wrong**
at high order — measured \(r(x_0)=570.82\) rad at \(\nu=1000.5\) (`RECONCILIATION.md` C2), so
\(r\) is many cycles from zero. Fix the branch by matching the two-term series
\(r\approx(\mu-1)/(8x)+(\mu-1)(\mu-25)/(384x^3)\), \(\mu=4\nu^2\), at the largest \(x\) for that
order and then tracking down. Getting this wrong produces a reference that is off by an exact
multiple of \(2\pi\), which is the single easiest way to poison every later prompt — assert in the
generator that the residual matches the series to \(10^{-8}\) at the largest \(x\) of each order.

### 2.3 `LiouvilleGreen/tests/test_bessel_reference.py`

Tests of the harness itself, because a harness nobody checks is worse than none:

1. The three tiers agree with each other where all three are valid: `exact_half_integer` vs
   `scipy_reference` vs the cached `mpmath` corners, to \(\le10^{-14}\) relative on \(J\), \(Y\)
   and \(A\), for \(\nu\in\{1/2,3/2,5/2\}\) at \(x\in\{10,10^3,10^7\}\).
2. `scipy_reference` **raises** for \(x>2\times10^{15}\), naming `RECONCILIATION.md` C1 in the
   message.
3. The exact derivative oracle satisfies the Wronskian against the cached corners:
   \(\lvert a^2\theta'-1\rvert\le10^{-13}\) at every corner.
4. The cached table's residuals reproduce the tail series at the largest \(x\) per order to
   \(10^{-8}\) — the branch check of §2.2, asserted rather than assumed.
5. `phase_pair_error` returns 0 for a perfect reconstruction and returns the index of the maximum
   for a deliberately perturbed one.

These must run in **under ~10 s** without touching `mpmath` (they read the cached JSON). Regenerating
the cache is a separate, explicitly invoked function, not part of the test run.

### 2.4 `docs/bessel-remedial/measure_bessel_phase.py`

A diagnostic script — not a test — that measures the **current** implementation and records the
baseline. `DRAFT-PLAN.md` §9 Stage 1 lists what it must report; all of it, per \((\nu,x_{\max})\)
case:

- construction wall-clock time, and the number of ODE right-hand-side evaluations if obtainable;
- the number of phase samples and the resulting `phase_spline.num_chunks`;
- evaluation cost per call for `theta_mod_2pi`, `raw_theta` and `theta_deriv`;
- \(E_\theta\), \(E_A\) and the derivative relative error, **each with the \(x\) at which it
  occurred**;
- the computed `phi`;
- distinguishing three point sets, which the plan requires be kept separate: **sample nodes**,
  **interior test points** (log-interval midpoints), and **endpoint intervals**.

Cases: \(\nu\in\{1/2,3/2,7/4,5/2,20.5,100.5\}\times x_{\max}\in\{10^3,10^7\}\) at both the
`config/defaults.py` fixture tolerances (`rtol=1e-8, atol=1e-10`) and the production tolerances
`main.py:520-528` uses (`rtol=5e-14, atol=1e-25`). Then, separately, the **cost and cliff sweep**
of §4 below.

Commit the script *and* its output, as `docs/bessel-remedial/baseline-2026-09.md`, with the
environment header of §2.2. Prompt 09 compares against this file; a baseline that lives only in a
transcript is not a baseline.

## 3. What this prompt must *not* do

- Do not modify `bessel_phase.py`, even to fix `phi`. That is prompt 05's, and measuring the
  unmodified object is the whole point of this commit.
- Do not modify `test_bessel_phase.py`. Its tolerances are prompt 08's to tighten. Note in your log
  that `test_phase_derivative` (`:121-142`) already contracts \(\theta'\) to \(10^{-6}\) relative
  at \(\nu\in\{2.5,20.5,100.5\}\) — `RECONCILIATION.md` C4 — and record its measured margin, since
  it is the campaign's standing derivative regression gate.
- Do not create a `bessel_phase`-derived reference anywhere. If you need a reference, it comes from
  §2.1 tier 1, 2 or 3.

## 4. The cost and cliff sweep

`DRAFT-PLAN.md` §4.7 reports the ODE build taking >600 s at \(x_{\max}=10^{13}\). **That does not
reproduce** (`RECONCILIATION.md` C1): it is 0.09 s there, and flat across five decades, with a
cliff between \(10^{15}\) and \(3\times10^{15}\) caused by Amos noise stalling the stepper rather
than by integration cost. Your baseline must record the true shape, because prompt 09 will claim
the cliff was removed and needs something honest to compare against.

Sweep \(x_{\max}\in\{10^{11},10^{13},10^{14},10^{15},2\times10^{15},3\times10^{15},8.6\times10^{15}\}\)
at \(\nu=1/2\) and \(\nu=5/2\), production tolerances, with a **per-case timeout of 60 s** so the
script terminates. Record `TIMEOUT` rather than a number for the cases that do not finish, and
report the last \(x_{\max}\) that completed.

Also record the mechanism directly, in three lines, so the diagnosis is in the repository and not
only in a planning document: sample \((2/\pi)/(x\,m(x))\) — which must equal
\(1+O(\nu^2/x^2)\) — at five adjacent doubles near \(x\in\{10^{15},3\times10^{15},10^{16}\}\), and
report the spread. Expect ~1.000000 at \(10^{15}\) and values between 0.73 and 1.68 above
\(3\times10^{15}\).

Finally, record the **wall-clock baseline of the existing test suite**:
`PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .`. It is slow —
`test_3bessel_analytic.py` rebuilds `bessel_phase` objects per case — and later prompts need to be
able to tell a regression from the pre-existing cost. If it exceeds ~20 minutes, say so and record
the per-module times instead of waiting for the whole discovery run.

## 5. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` passes,
  including the pre-existing four tests in `test_bessel_phase.py`. **Nothing may regress: this
  commit changes no production code, so any failure is yours.**
- `test_bessel_reference.py` passes in under ~10 s and does not import `mpmath` at test time.
- `docs/bessel-remedial/baseline-2026-09.md` exists and contains every quantity §2.4 and §4 list.
- The three reference tiers agree to \(10^{-14}\) where they overlap (§2.3 item 1) — quote the
  measured figure in the log.

## 6. Log and commit

Follow README §5 and §5.1. Your log must include, in "State handed to the next prompt":

- the exact import path and signature of every public function in `bessel_reference.py`;
- the tier-selection convention (how a caller says which reference it wants) and the value of
  `SCIPY_REFERENCE_MAX_X`;
- the JSON schema of `bessel_reference_data.json` and how to regenerate it;
- the baseline \(E_\theta\), \(E_A\) and derivative error per case, and the location of each
  maximum;
- the last \(x_{\max}\) at which the existing construction completes;
- `test_phase_derivative`'s measured margin against its \(10^{-6}\) contract.

Prompts 03 through 08 are programmed against these names. A log that says "as in the prompt" fails
review.

Commit subject, or something equally specific: `Add independent Bessel references and a
measurement harness`.
