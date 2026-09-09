# Prompt 03 — The closed-form tail and the remainder-tested crossover

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design:** `DRAFT-PLAN.md` §7.2 (the tail), §5.3 (the safe matching window), §1 (why the tail is required)
**Reconciliation items:** C2 (\(r\) is not sub-cycle at high order), §1 (the series measurements reproduced)
**Depends on:** 01 (references and metrics). Soft: 02.
**Recommended model:** Opus
**Files you may touch:** new `LiouvilleGreen/bessel_tail.py`, new
`LiouvilleGreen/tests/test_bessel_tail.py`, plus the log and the status board.
**Do not touch:** `LiouvilleGreen/bessel_phase.py`. Nothing consumes this module yet; prompt 05
wires it in.

Read first: README §2 (a), (b) and (c); `RECONCILIATION.md` §1 (the tail-series rows) and C2;
`DRAFT-PLAN.md` §7.2, §5.3 and §3 (the convention algebra); then
`logs/01-reference-harness.md` §"State handed to the next prompt".

---

## 1. Character of this commit

A small, pure, self-contained module: no state, no interpolation, no sampling, no SciPy Bessel
call. It is the easiest part of the campaign to get right and the easiest to verify to machine
precision, which is why it comes first.

It is also **mandatory, not an optimization** (`DRAFT-PLAN.md` §1). Two independent reasons, and
the prompt is written so that both are testable:

1. `hankel1e` fails silently above \(7.13\times10^8\) for \(\nu\gtrsim100\) and above
   \(2.247\times10^{15}\) for every order (README §2 (e)). A closed-form tail means the sampler is
   never asked for a value within three decades of that boundary.
2. `jv`/`yv` become O(1)-relatively noisy above \(x\approx2.5\times10^{15}\)
   (`RECONCILIATION.md` C1), which is what stalls the existing ODE build. A closed-form tail
   evaluates no Bessel routine there at all, so the supported \(x_{\max}\) is limited by
   \(\sin\)/\(\cos\) (correct to \(10^{16}\)) rather than by Amos.

## 2. The mathematics

With \(\mu=4\nu^2\), DLMF 10.18.18 in **this repository's** convention (which differs from DLMF's
by exactly \(+\pi/2\), absorbed into \(c_\nu=\pi/4-\pi\nu/2\); see `DRAFT-PLAN.md` §3):

$$r_\nu(x)=\frac{\mu-1}{8x}+\frac{(\mu-1)(\mu-25)}{384x^3}+\frac{(\mu-1)(\mu^2-114\mu+1073)}{15360x^5}+\cdots$$

Differentiate term by term for \(r_\nu'\). Then **the amplitude needs no second series**: the
Wronskian \(A^2\theta'=2/(\pi x)\) gives \(\theta'=a^{-2}\) exactly, and \(\theta=x+c_\nu+r_\nu\)
gives \(\theta'=1+r_\nu'\), hence

$$\boxed{\;a_\nu(x)=\bigl(1+r_\nu'(x)\bigr)^{-1/2}\;}$$

DLMF 10.18.17 is **not** required and must not be used: one series governing both quantities means
the crossover test on the phase automatically covers the amplitude, which is the property prompt 05
relies on. Note \(\nu=1/2\) gives \(\mu-1=0\), so \(r\equiv0\) and \(a\equiv1\) **identically at
every \(x\)** — a free exactness test, asserted in §4.

## 3. What to build

### 3.1 The series

```python
TAIL_SERIES_MAX_TERMS = 3          # terms implemented; see the note below

def tail_residual(nu, x, n_terms=...): ...          # r_nu(x)
def tail_residual_deriv(nu, x, n_terms=...): ...     # dr/dx
def tail_residual_log_deriv(nu, x, n_terms=...): ... # dr/d(log x) = x r'
def tail_amplitude(nu, x, n_terms=...): ...          # a_nu = (1 + r')^{-1/2}
def tail_first_omitted_term(nu, x, n_terms=...): ... # magnitude of term n_terms+1
```

Vectorize over `x` (accept scalars and arrays). Evaluate in a form that does not lose accuracy for
large \(x\): the terms fall like \(x^{-1},x^{-3},x^{-5}\), so summing smallest-first, or in terms of
\(1/x^2\) by Horner, is preferable to forming each power independently. State which you chose and
why in the module docstring.

Implement **three** terms plus a fourth for `tail_first_omitted_term`, so the remainder test of
§3.2 has something to test with at the maximum implemented order. Take the fourth coefficient from
DLMF 10.18.18 and cite the equation number; if you cannot obtain it with confidence, implement two
terms plus the third as the omitted term instead, and record that as an
`IMPLEMENTATION CHOICE` with the consequence stated (the crossover would then be sized by the
third term rather than the fourth, which is more conservative and therefore acceptable). **Do not
guess a coefficient.**

### 3.2 The remainder-tested crossover

`DRAFT-PLAN.md` §7.2 is explicit that \(x_\star\) must **not** be a fixed constant and **not** a
fixed multiple of \(\nu\):

> Choose \(x_\star(\nu)\) by a remainder test at the requested accuracy … evaluate the first omitted
> term, and require in addition that the near-region interpolant and the series agree to the
> accuracy budget at the crossover. Record \(x_\star\) and the agreement achieved.

This prompt owns the first half — the series side. Provide

```python
def tail_crossover(nu, phase_atol, amplitude_rtol, n_terms=..., x_max=None): ...
```

returning at least `(x_star, first_omitted_at_x_star, terms_used)` in a small named structure.
Semantics:

- \(x_\star\) is the **smallest** \(x\) at which the first omitted term is below `phase_atol`
  *and* the induced amplitude relative error is below `amplitude_rtol`. Smallest, not a safe large
  value: every e-fold below \(x_\star\) has to be sampled and branch-tracked by prompt 04, so
  making the tail as wide as the accuracy budget allows is the whole economy of the design.
- The amplitude budget is not the same test as the phase budget. \(a=(1+r')^{-1/2}\) gives
  \(\delta a/a\approx\tfrac12\lvert\delta r'\rvert\) and \(r'\sim r/x\), so the amplitude condition
  is *weaker* than the phase one by a factor \(\sim2x\) — measured, \(\lvert\delta a/a\rvert\) is
  \(10^{2}\)–\(10^{3}\) times smaller than \(\lvert\delta r\rvert\) at the same \(x\)
  (`RECONCILIATION.md` §1). Apply both anyway and let the phase condition bind; assert in a test
  that it does.
- **Clamp and report, do not silently extend.** If no \(x\) below `x_max` satisfies the test,
  raise, naming \(\nu\), the budgets, the best first-omitted term achieved and where. Prompt 05
  must be able to fail construction loudly rather than accept a tail it cannot certify.
- \(\nu=1/2\) must return \(x_\star=\) the domain lower bound (the series is exact everywhere), not
  a large number. Special-case it explicitly on \(\mu-1=0\) rather than relying on the numeric test,
  and comment why.

**Sanity anchor**, from `DRAFT-PLAN.md` §5.3 and reproduced in `RECONCILIATION.md` §1: two terms
reach machine precision for \(x\gtrsim50\)–\(100\nu\), and the cancellation floor of the *residual
right-hand side* stays below \(10^{-8}\) out to \(x\lesssim4800\nu\), so there is a wide safe
matching window \([100\nu,\,4800\nu]\) and no delicate crossover. Your \(x_\star\) at the campaign's
targets should land inside that window for every order in the supported set; assert it.

### 3.3 Do *not* build

- No sampling, no interpolation, no branch tracking, no cycle-count representation. This region has
  none of those by construction (`DRAFT-PLAN.md` §7.2) and adding any is a stop condition.
- No `hankel1e`, `jv` or `yv` call in the module itself. References belong in the tests, via
  `bessel_reference`.
- No knowledge of prompt 04's sampler or prompt 05's object. This module must be usable, and
  testable, entirely on its own.

## 4. Tests

`LiouvilleGreen/tests/test_bessel_tail.py`, scoring against `bessel_reference` (prompt 01) — the
cached `mpmath` corners for the high orders and large \(x\), `exact_half_integer` where the order
allows, `scipy_reference` in between and never above `SCIPY_REFERENCE_MAX_X`.

1. **\(\nu=1/2\) is exact.** \(r\equiv0\) and \(a\equiv1\) to \(0\) (not to a tolerance — assert
   exact equality, or `abs(...) == 0.0`) at \(x\in\{10^{-3},1,10,10^3,10^7,10^{12},10^{15}\}\).
   This is the one place in the campaign where an exact-equality assertion is right.
2. **Series accuracy against the references**, two and three terms, at \(x=50\nu\), \(100\nu\),
   \(200\nu\) for \(\nu\in\{5/2,20.5,100.5,1000.5\}\). Expected \(\lvert\delta r\rvert\), from
   `RECONCILIATION.md` §1 (two terms): 1.77e-10 (\(\nu=5/2\), \(50\nu\)), 7.64e-10 (20.5),
   4.01e-9 (100.5); and \(\lvert\delta a/a\rvert\) 3.5e-12, 1.9e-12, 2.0e-12 respectively, falling
   to \(\le5.5\times10^{-14}\) at \(100\nu\). Assert bounds a little looser than these and quote the
   measured values in the log.
3. **The Wronskian is satisfied by construction and against the reference.** \(a^2(1+r')=1\)
   exactly (a tautology — assert it anyway, as a guard against an algebra slip in the
   differentiation), and \(a^{-2}\) against the independent oracle
   \((2/\pi)/\bigl(x(J^2+Y^2)\bigr)\) to \(10^{-11}\) at \(x\ge100\nu\).
4. **The crossover is a real remainder test.** For each order and each of two budgets
   (\(10^{-11}\) and \(10^{-6}\), the two rows of README §6), assert: \(x_\star\) lies in
   \([50\nu,\,4800\nu]\); the first omitted term at \(x_\star\) is below the budget; the *measured*
   \(\lvert\delta r\rvert\) against the reference at \(x_\star\) is below the budget; and
   \(x_\star\) is monotone non-increasing as the budget loosens.
5. **The failure path.** `tail_crossover` raises, with \(\nu\) and the budgets in the message, when
   asked for a budget the implemented number of terms cannot reach below `x_max`.
6. **Branch sanity at high order.** \(r_\nu(x_\star)\) at \(\nu=1000.5\) is \(\approx5.0\) rad, not
   \(\approx5.0-2\pi\) — i.e. the series is *not* reduced mod \(2\pi\) anywhere
   (`RECONCILIATION.md` C2 measures \(r(x_\star)=5.002540\)). Assert \(r>\pi\) there, which fails
   if anyone ever wraps this quantity.

Runtime under ~20 s, reading cached references.

## 5. Reproduction to run first

```python
import numpy as np
from mpmath import mp, mpf, besselj, bessely, pi as mpi, sqrt as msqrt
mp.dps = 60

def series(nu, x, n):
    mu = 4.0 * nu * nu
    t1 = (mu - 1.0) / 8.0
    t3 = (mu - 1.0) * (mu - 25.0) / 384.0
    t5 = (mu - 1.0) * (mu * mu - 114.0 * mu + 1073.0) / 15360.0
    r, rp = t1 / x, -t1 / x**2
    if n >= 2:
        r += t3 / x**3;  rp += -3.0 * t3 / x**4
    if n >= 3:
        r += t5 / x**5;  rp += -5.0 * t5 / x**6
    return r, rp

for nu in (0.5, 2.5, 20.5, 100.5, 1000.5):
    for mult in (50.0, 100.0):
        x = mult * max(nu, 1.0)
        X, NU = mpf(float(x)), mpf(float(nu))
        a_true = msqrt((mpi * X / 2) * (besselj(NU, X) ** 2 + bessely(NU, X) ** 2))
        for n in (2, 3):
            r, rp = series(nu, x, n)
            a = (1.0 + rp) ** -0.5
            print(f"nu={nu:<7} x={x:<9.4g} n={n}: |da/a|={abs(a / float(a_true) - 1):.3e}")
```

Note this checks the amplitude only. Checking \(r\) against `mpmath` requires the branch fix of
prompt 01 §2.2 — the naive fold into \((-\pi,\pi]\) reports a spurious error of exactly \(2\pi k\)
at high order, and that is a reference bug, not a series error. Use `bessel_reference`'s corners,
which have the branch already resolved and asserted.

## 6. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_tail -v` passes.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` passes; in
  particular `test_bessel_phase.py`'s four tests still pass, since no production code changed.
- No import of `scipy.special` in `bessel_tail.py`.
- The \(x_\star\) table (order × budget) is in the log.

## 7. Log and commit

Follow README §5 and §5.1. In "State handed to the next prompt", give:

- the module's public signatures verbatim, including the returned structure of `tail_crossover` and
  the meaning of each field;
- how many series terms shipped, and which coefficient (third or fourth) is the omitted-term
  estimator;
- the **\(x_\star\) table**: for \(\nu\in\{1/2,3/2,7/4,5/2,20.5,100.5,1000.5\}\) at budgets
  \(10^{-11}\) and \(10^{-6}\), the value of \(x_\star\), the first omitted term there, and
  \(x_\star/\nu\). Prompt 04 sizes its sampled domain from exactly this table and prompt 05 asserts
  agreement at these points;
- the measured \(\lvert\delta r\rvert\) and \(\lvert\delta a/a\rvert\) at each \(x_\star\).

Commit subject, or something equally specific: `Add the closed-form Bessel tail and its crossover
test`.
