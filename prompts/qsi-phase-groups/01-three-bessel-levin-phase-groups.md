# Prompt 01 — Assemble `_three_bessel_Levin`'s phases as groups, and declare their error

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Model:** Opus
**Files you may touch:** `ComputeTargets/QuadSourceIntegral.py`,
`LiouvilleGreen/three_bessel_integrals.py` (the public-name change of §2.1 **only**),
`ComputeTargets/tests/test_quadsource_integral.py`,
`LiouvilleGreen/tests/test_three_bessel.py` (only if §2.1 forces it), plus the log and the board.
**Do not touch:** `ComputeTargets/phase_groups.py` (README §2 says why — it is a different object),
`ComputeTargets/QuadSource.py`, `ComputeTargets/TkSourceFunctions.py`, `AdaptiveLevin/`,
`Datastore/`, `LiouvilleGreen/bessel_phase.py`, `LiouvilleGreen/phase_spline.py`, `main.py`,
`thirdparty/`, any `extract_*.py`.

Read first: this campaign's `README.md` in full; then, in the code,
`ComputeTargets/QuadSourceIntegral.py:1210-1442` (`_three_bessel_Levin`) and `:990-1078`
(`phase_group_Levin_integral`, the sibling route in the same file that already supplies a
derivative); `LiouvilleGreen/three_bessel_integrals.py:219-440` (`_PhaseGroup`, the pattern);
`AdaptiveLevin/levin_quadrature.py:930-975` and `:1030-1095` (the phase-dict contract) and
`:2350-2370` (what `theta_abserr` is for); and
`ComputeTargets/tests/test_quadsource_integral.py:1230-1302` (`TestTolerancePlumbing`, whose spy
pattern you will reuse).

---

## 1. What is wrong

`_three_bessel_Levin` integrates over \(\log\eta\) with

$$x_1=k\eta,\qquad x_2=q c_s\eta,\qquad x_3=r c_s\eta,$$

and builds four phase groups with sign patterns \((+,+,+)\), \((+,+,-)\), \((+,-,+)\), \((+,-,-)\)
on \((\theta_{G_k},\theta_{T_k},\theta_{T_k})\) — each used by one `J` call (the \(\sin\) component)
and one `Y` call (the \(\cos\) component), eight calls in all.

Each group is currently assembled twice over, both times by summation:

```python
def phase1(log_eta):            # :1250
    ... return phase_Gk.raw_theta(x1) + phase_Tk.raw_theta(x2) + phase_Tk.raw_theta(x3)

def phase1_mod_2pi(log_eta):    # :1263
    ... return phase_Gk.theta_mod_2pi(x1) + phase_Tk.theta_mod_2pi(x2) + phase_Tk.theta_mod_2pi(x3)
```

Three defects, in decreasing order of size:

1. **Summing `raw_theta`.** Each is a double of magnitude \(\sim m\eta\) carrying absolute error
   \(\sim\varepsilon m\eta\), and those errors do not cancel with the group's signs. A group whose
   own phase is small — the near-resonant case the four-group decomposition exists to handle — is
   returned with the absolute error of its **largest constituent**.
2. **Summing three bounded angles.** The sum of three values in \((-\pi,\pi]\) is not the bounded
   angle of the sum, and each constituent's reduction error enters additively.
3. **No `theta_deriv` and no `theta_abserr`.** Without the former, Levin spectrally differentiates
   the raw phase, which inherits \(\varepsilon\theta/(\text{phase span})\); the derivative also
   decides subdivision, since `phase_span` comes from `theta_prime_Cheb`
   (`levin_quadrature.py:1090`). Without the latter, the reported `abserr` is silent about the
   phase construction — `levin_quadrature.py:2360` exists for exactly this case.

## 2. The change

### 2.1 Make the existing group class importable

`LiouvilleGreen/three_bessel_integrals.py` already holds `_PhaseGroup`, which does precisely this
job and is tested by `LiouvilleGreen/tests/test_three_bessel.py`'s `TestPhaseGroups`. **Do not
reimplement it**, and do not copy it into `ComputeTargets/`.

Give it a public name so `QuadSourceIntegral.py` can import it without reaching for a private
symbol. Either a public alias alongside the existing name, or rename the class and keep
`_PhaseGroup` bound to it — your choice, state which and why. `test_three_bessel.py` imports
`_PhaseGroup` by name in eight places and **must still pass**; you may update its import if you
rename, but change nothing else in that file.

Its behaviour must not change at all. This is a naming change.

### 2.2 Build the four groups

`_PhaseGroup(phases, coefficients, signs)` takes the three phase objects in order, the three
coefficients such that factor \(i\) is evaluated at `coefficients[i] * x`, and the three signs with
the first always \(+1\). For this call site that is

```python
phases       = (phase_Gk, phase_Tk, phase_Tk)
coefficients = (k.k, q.k * cs, r.k * cs)
signs        = (1.0, e_q, e_r)          # over (+1,+1), (+1,-1), (-1,+1), (-1,-1)
```

and its accessors already take the logarithm of the shared variable, which here is `log_eta`. The
same `phase_Tk` object appearing twice with different coefficients is expected and handled.

Replace the eight `phaseN` / `phaseN_mod_2pi` closures with four groups, and pass
`group.levin_theta()` — the four-key dict — to **both** the `J` and the `Y` call of that group.

**Change nothing else about the calls.** The `f` vectors (`[Levin_f, lambda x: 0.0]` for `J`,
`[lambda x: 0.0, lambda x: -Levin_f(x)]` for `Y`), `atol`, `rtol`, `chebyshev_order` and the
`notify_label` strings all stay exactly as they are. The `J`/`Y` split encodes the convention
\(J=A\sin\theta\), \(Y=-A\cos\theta\); preserving it is what keeps this a phase-assembly change and
nothing more.

Note that `theta_mod_2pi` changes representative — the old route's signed sum of three values in
\((-\pi,\pi]\) lies in \((-3\pi,3\pi]\), the new one is `atan2` of the split pair and lies in
\((-\pi,\pi]\). Both are valid: every consumer takes only \(\sin\)/\(\cos\) of it
(`levin_quadrature.py:1103-1105`, `:1120-1121`). Say so in a comment rather than leaving a reader
to wonder whether the range change matters.

### 2.3 Two stale comments in the same file

Both are about the quantity this prompt changes, so they are in scope. Neither may change any code.

- **`:80`** states that "`bessel_phase()` reconstructs `J_nu` to ~2e-8 of that envelope (audit
  QI-1)". That was the pre-`transfer-remedial` construction's floor. The object now declares
  \(\sim5\times10^{-12}\) rad of phase error at these orders and is measured well inside it.
  Correct the number and say what it is now, keeping the reasoning about the order guard intact.
- **`_check_bessel_order`'s docstring (`:675-713`)** says the returned dict "carries phase, mod, Q,
  phi, bessel_j, bessel_y, min_x, max_x and no `nu`", and explains that the order must therefore be
  checked numerically. Both halves are now false: `bessel_phase` returns a `"nu"` key, and `Q` was
  removed (reading `data["Q"]` raises `KeyError`). Correct the docstring. **Leave the numeric check
  itself alone** — replacing it with a direct comparison against `phase_data["nu"]` is a behaviour
  change and is not this prompt's. Record that option in the log instead.
  This closes `[05-quadsource-order-check-docstring-stale]`.

## 3. Tests

In `ComputeTargets/tests/test_quadsource_integral.py`. Extend; remove nothing.

1. **Every analytic Levin call carries all four keys.** Mirror
   `TestTolerancePlumbing.test_every_analytic_Levin_call_uses_the_same_tolerances` — the same
   `patch.object(qsi_module, "adaptive_levin_sincos", spy)` pattern, which already captures all
   sixteen calls (eight groups × two `nu_type`s) on `Case(0.0, SHAPES[0], 100.0, exact=True)`.
   Assert every captured `theta` dict has exactly
   `{"theta", "theta_mod_2pi", "theta_deriv", "theta_abserr"}`. This is the test that would have
   caught the gap in the first place.

2. **The improvement, measured.** A restructure justified only by an argument does not meet this
   prompt. Score the **group phase** \(\Theta(\log\eta)\) and its log-derivative from the old route
   and the new one against a high-precision reference — build the reference from the constituent
   Bessel functions directly (`mpmath`, or `LiouvilleGreen/tests/bessel_reference.py`'s tiers),
   **not** from the objects under test. Report both, before and after, for:
   - a **near-resonant** group, where \(K=k+e_q qc_s+e_r rc_s\) is zero or tiny relative to
     \(\max(k,qc_s,rc_s)\);
   - a **generic** group, where it is not.

   Quote \(\lvert\delta\Theta\rvert\) and \(\lvert\delta\,d\Theta/d\log\eta\rvert\) with the
   \(\eta\) at which each maximum occurred. Keep a copy of the old assembly inside the test to
   measure against — that is the only place it should survive.

   **Expect the generic group to show no improvement.** Both routes sit on one rounding of the
   product \(K\eta\) there; `prompts/transfer-remedial`'s `[07-generic-K-product-rounding]`
   measured 1.5e-5 rad for both at \(K=0.1\), \(x=10^{12}\). A test that claims a gain there is
   measuring something else.

3. **The derivative-cancellation trap.** Score \(\lvert\delta\,d\Theta/d\log\eta\rvert\) against
   \(\max(k,qc_s,rc_s)\,\eta\). **Never divide by the group derivative**, which passes through zero
   at exact resonance; a relative-error assertion on it is a bug in the test even if it currently
   passes.

4. **No regression in the stored value.** `analytic_rad` must still agree with its existing
   acceptance oracle: `TestAnalyticOracle` must pass **at its existing thresholds**, unmodified.
   Confirm with `git diff HEAD~1 -- ComputeTargets/tests/test_quadsource_integral.py` that you
   added tests and changed no existing threshold. Separately, report how much `analytic_rad` itself
   moved on those fixtures, and attribute it.

5. **The declared error reaches the caller.** Assert that supplying `theta_abserr` changes the
   reported `abserr` of at least one group — in the direction of *larger*, which is the point
   (`levin_quadrature.py:2360`: "so the caller sees an honest number instead of an artificially
   small one"). A test expecting `abserr` to shrink is expecting the wrong thing.

## 4. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` passes.
- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_three_bessel` passes.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` passes —
  nothing here touches it, so a failure means something strayed.
- `grep -c "theta_abserr" ComputeTargets/QuadSourceIntegral.py` is non-zero and every
  `adaptive_levin_sincos` call in the file supplies four keys.
- `./venv/bin/python -m black --check` clean on every file you touched.
- The full `LiouvilleGreen/tests` discovery run takes ~31 min and is dominated by
  `test_3bessel_analytic` drawing figures (`[08-3bessel-plot-cost-dominates-the-suite]`). Run it if
  you can; if you do not, say so explicitly rather than implying you did.

## 5. Log and commit

Follow the campaign README §4 and §4.1. The log must state:

- which naming option §2.1 took, and why;
- the before/after table of §3 item 2, as numbers, including the generic group that does not
  improve;
- how much `analytic_rad` moved on the existing fixtures, and the attribution;
- the `abserr` before and after declaring `theta_abserr`;
- whether the full `LiouvilleGreen/tests` run was completed.

Close `[transfer-remedial-qsi-phase-groups]` on `prompts/source-remediation`'s board — it is the
entry this prompt discharges, and it may be edited **only** to mark it resolved, with a pointer to
this campaign. Close `[05-quadsource-order-check-docstring-stale]` on
`prompts/transfer-remedial`'s board the same way. Update
[`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit: remove both rows and correct
the count and the date.

Commit subject, or something equally specific:
`Assemble the analytic three-Bessel phases as groups`.
