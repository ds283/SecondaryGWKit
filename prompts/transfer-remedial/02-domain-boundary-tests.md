# Prompt 02 — Pin the SciPy Bessel domain boundaries as tests

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design:** `DRAFT-PLAN.md` §4.4, §12.2; §9 Stage 1 ("record the SciPy version")
**Reconciliation items:** C1 (the `jv`/`yv` noise boundary and the stepper stall)
**Depends on:** 01 (uses `bessel_reference`'s cached corners and its environment header)
**Recommended model:** Sonnet
**Files you may touch:** new `LiouvilleGreen/tests/test_scipy_bessel_domain.py`, plus the log and
the status board. **No production code.**

Read first: README §2 (e); `RECONCILIATION.md` C1 and §1; `DRAFT-PLAN.md` §4.4 and §12.2; then
`logs/01-reference-harness.md` §"State handed to the next prompt" for the reference API and the
value of `SCIPY_REFERENCE_MAX_X`.

---

## 1. Character of this commit

Four facts about the bundled SciPy/Amos library, turned into tests that fail loudly if a SciPy
upgrade changes them. They are **not** facts about this repository's code, and that is precisely
why they need pinning: the campaign's supported \((\nu,x_{\max})\) domain is derived from them, so
a silent change to them silently invalidates the domain.

`DRAFT-PLAN.md` §9 Stage 1 asks only that the SciPy version be recorded. That is not enough. A
recorded version tells a later reader *which* library was measured; it does not tell them the
measurement no longer holds. These tests do.

Small, mechanical prompt. Do not over-build it.

## 2. The four facts

Each is measured in `RECONCILIATION.md` §1 and C1 on SciPy 1.15.2 / NumPy 2.2.4, and each is
reproduced by a short script in `DRAFT-PLAN.md` §12.2. Reproduce the numbers yourself before
writing the assertion; if any differs, that is a `STRUCTURALLY REQUIRED` deviation and you must
record both values and **stop** rather than silently relaxing the bound.

### Fact 1 — `hankel1e` returns exactly `-0j`, and it is finite

```
hankel1e(100.5, 1e9) == -0j
np.isfinite(hankel1e(100.5, 1e9))          is True      # so an isfinite() guard passes
np.log(np.abs(hankel1e(100.5, 1e9)))       is -inf
np.angle(hankel1e(100.5, 1e9))             is -0.0
```

Assert all four. The third and fourth are what would reach a log-amplitude interpolant and a phase
tracker. The point of the test is the *combination*: a finite value that produces `-inf`
downstream.

### Fact 2 — the `hankel1e` usable boundaries

Contiguous-from-below usable range of \(\lvert\sqrt{\pi x/2}\,\operatorname{hankel1e}(\nu,x)\rvert\),
requiring finiteness **and** a plausibility floor (use `> 1e-3`; the true value is \(\ge1\)):

| \(\nu\) | usable to |
|---|---|
| 1/2, 5/2, 20.5 | \(2.247\times10^{15}\) |
| 100.5, 1000.5 | \(7.13\times10^{8}\) |

Assert as inequalities with a margin, not as equalities — e.g. that the boundary is within a factor
of 2 of the tabulated value — so that a minor Amos change does not produce a spurious failure while
a decade-scale change does. State the margin and why in a comment.

Also assert the campaign-critical consequence directly: at \(x_\star=100\nu\), `hankel1e` is usable
for **every** order in the supported set, and the ratio (boundary / \(x_\star\)) is at least
\(10^3\) even at \(\nu=1000.5\) (\(7.13\times10^8\) vs \(10^5\)). That inequality is the
justification for README §2 (e) and it should be a test, not a remark.

### Fact 3 — `jv`/`yv` become O(1)-relatively noisy above \(x\approx2.5\times10^{15}\)

This is `RECONCILIATION.md` C1 and is **not** in `DRAFT-PLAN.md`. The identity
\((2/\pi)/\bigl(x\,(J_\nu^2+Y_\nu^2)\bigr)=1+O(\nu^2/x^2)\) is exact, so any departure from 1 at
large \(x\) is library error and nothing else. Measured at \(\nu=1/2\), five adjacent doubles:

| \(x\) | values |
|---|---|
| \(10^{12}\) … \(2\times10^{15}\) | 1.000000 (all five, to six places) |
| \(3\times10^{15}\) | 0.988677, 0.948322, 1.028585, 0.978680, 1.033255 |
| \(5\times10^{15}\) | 1.676372, 1.653327, 1.077128, 1.389636, 1.665364 |
| \(10^{16}\) | 0.750691, 1.094350, 1.035889, 0.734835, 1.060665 |

Assert two things:

- **Good below:** the quantity is within \(10^{-9}\) of 1 at \(x\in\{10^{12},10^{14},2\times10^{15}\}\).
- **Bad above:** it departs from 1 by more than \(10^{-2}\) somewhere in a small cluster of
  adjacent doubles near \(x=5\times10^{15}\).

The "bad above" assertion is deliberately an assertion that the library *is* broken. That is the
right shape: if a future SciPy fixes it, this test fails, and the correct response is to raise
`SCIPY_REFERENCE_MAX_X` and the supported domain — a decision, which a failing test forces someone
to make. Say that in the test's docstring so the next reader does not just delete it.

### Fact 4 — the failure is not monotonic in \(x\)

`DRAFT-PLAN.md` §4.4 states that `jv`/`yv` "recover at \(10^{16}\) for \(\nu=1000.5\) while failing
at \(10^{10}\)", so no simple ceiling can be certified for that route. Verify this and assert
whatever you actually measure. If it does not reproduce, record it as a deviation — the *claim* the
campaign relies on is only the weaker one, that a monotone ceiling must not be assumed, so an
assertion that the non-monotonicity exists at some specific \((\nu,x)\) pair is fine, and an
assertion that no ceiling is certifiable is not testable. Prefer the concrete pair.

## 3. Reproduction to run first

```python
import numpy as np
from scipy.special import hankel1e, jv, yv
np.seterr(all="ignore")

for nu in (0.5, 2.5, 20.5, 100.5, 1000.5):
    lo = np.sqrt(max(nu * nu - 0.25, 1e-10))
    xs = np.geomspace(max(1.5 * lo, 1.0), 1e16, 6000)
    aH = np.abs(np.sqrt(np.pi * xs / 2) * hankel1e(nu, xs))
    aJ = np.hypot(jv(nu, xs), yv(nu, xs)) * np.sqrt(np.pi * xs / 2)

    def contiguous(ok):
        i = np.argmax(~ok) if (~ok).any() else len(ok)
        return xs[i - 1] if i > 0 else float("nan")

    print(f"nu={nu}: hankel1e to {contiguous(np.isfinite(aH) & (aH > 1e-3)):.4g}, "
          f"jv/yv to {contiguous(np.isfinite(aJ) & (aJ > 1e-3)):.4g}")

# Fact 3: the RHS identity, at five adjacent doubles
for x in (1e12, 1e15, 2e15, 3e15, 5e15, 1e16):
    m = lambda xx: jv(0.5, xx) ** 2 + yv(0.5, xx) ** 2
    vals = [(2 / np.pi) / xx / m(xx) for xx in x * (1 + np.array([0, 1e-15, 2e-15, 3e-15, 1e-14]))]
    print(f"x={x:.1e}: {['%.6f' % v for v in vals]}")
```

## 4. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_scipy_bessel_domain -v`
  passes, in under ~30 s.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` still passes.
- Every assertion carries a comment giving the measured value, the SciPy version it was measured
  on, and a pointer to `RECONCILIATION.md` C1 or `DRAFT-PLAN.md` §4.4.
- The module docstring states plainly: these test SciPy, not this repository; a failure means the
  supported domain must be re-derived, not that a bound should be loosened.

## 5. Log and commit

Follow README §5 and §5.1. Quote every measured boundary. In "State handed to the next prompt",
give the assertion names and the numeric bounds, so prompt 04 can calibrate its plausibility band
and its declared domain against them rather than re-measuring.

Commit subject, or something equally specific: `Pin the SciPy Bessel domain boundaries as tests`.
