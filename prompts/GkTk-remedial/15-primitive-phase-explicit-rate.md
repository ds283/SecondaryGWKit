# Prompt 15 — Give `PrimitivePhase` an explicit leading-rate callable

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Covers:** §3 `[10-primitive-phase-leading-rate-is-hardcoded]`
**Review sections:** none new — this is a correctness/hygiene defect prompt 10 found in prompt 09's
design, not a review finding.
**Design facts:** README §2 (c) (the `delta` sign convention); M15 on the board.
**Depends on:** 09, 10.
**Recommended model:** Sonnet — a tightly specified signature change with a frozen call site to
preserve.
**Files you may touch:** `ComputeTargets/primitive_phase.py`, `ComputeTargets/TkSourceFunctions.py`,
`ComputeTargets/tests/test_primitive_phase.py`, `ComputeTargets/tests/test_tk_source_functions.py`,
plus the log, the status board and `docs/OPEN_ISSUES.md`.
**Do not touch:** `ComputeTargets/GkSourcePolicyData.py` (its `PrimitivePhase` call must need no
edit — see §2 item 3), `ComputeTargets/WKB_Gk.py`, `WKB_Tk.py`, `BackgroundModel.py`,
`cumulative_table.py`, `main.py`, the Datastore factories, and everything README §5 rule 8 lists
(`transfer-remedial`'s files, `AdaptiveLevin/`, `QuadSourceIntegral.py`, `QuadSource.py`,
`phase_groups.py`, `thirdparty/`, any `extract_*.py`).

**Out of sequence.** This prompt is numbered 15 because the campaign's numbers are append-only,
but there is no strict ordering requirement against 11–14: it touches only `primitive_phase.py`
and `TkSourceFunctions.py`, which 11–14 do not. It exists to close an issue prompt 10 opened
before Workstream E is dispatched, at the user's request. Note this in your log.

Read first: `ComputeTargets/primitive_phase.py` module docstring and `PrimitivePhase.__init__` /
`theta_deriv` (currently `:1-116`, `:119-198`, `:273-291`); `ComputeTargets/TkSourceFunctions.py`
module docstring `:89-103`, `_SoundHorizonRate` `:203-232`, the `PrimitivePhase(...)` call
`:393-402`; `ComputeTargets/GkSourcePolicyData.py:168-196`; `IMPLEMENTATION_STATE.md` M15 and the
`[10-primitive-phase-leading-rate-is-hardcoded]` entry in §3.

---

## 1. What is wrong

`PrimitivePhase.theta_deriv` (`primitive_phase.py:287`) computes the leading derivative in closed
form as

```python
dtheta_dz_leading = self._sign * self._k / self._Hubble(raw_x)
```

i.e. `sign * k / H(z)`. That is `d/dz[leading.delta(z, z_anchor)]` for `leading = tau`, where
`d(tau)/dz = -1/H`, correct for the Green's function. It is wrong by a factor `1/c_s` for
`leading = cs_tau` (the transfer function's sound horizon), where `d(cs_tau)/dz = -c_s/H`, so the
true rate is `c_s/H`, not `1/H`.

`primitive_phase.py` was outside prompt 10's file list (Workstream D scoping at the time), so
rather than generalising the closed form, prompt 10 worked around it: `TkSourceFunctions.py`
defines `_SoundHorizonRate` (`:203-232`), an adapter whose `.Hubble(z)` method actually returns
`H(z)/c_s(z)`, and passes `model_functions=_SoundHorizonRate(self._model.functions)` into
`PrimitivePhase`. Since `.Hubble` is the only attribute `PrimitivePhase` reads off
`model_functions`, the division `k / H_eff(z)` recovers `k * c_s(z) / H(z)` and the *value* of
`theta_deriv` is correct today. The problem is what the adapter costs:

- A `PrimitivePhase` built for `T_k` carries a `model_functions` whose `.Hubble` is not the
  Hubble rate — `phase._Hubble(z)` silently returns `H(z)/c_s(z)`. Anyone reading `model_functions`
  off a `PrimitivePhase` (directly, or through a future accessor) gets a rate-fudged function, not
  the real one.
- There is no clean extension point for a third leading primitive with yet another rate — each
  would need its own `_FooRate` adapter class duplicating this trick.
- A future caller who passes the model's genuine `ModelFunctions` directly, bypassing the adapter,
  reintroduces the factor-`1/c_s` (`~1.73` in radiation) error silently — `theta_deriv` gives no
  indication anything is wrong.

## 2. The change

1. **Add an explicit `rate` parameter to `PrimitivePhase.__init__`**, keyword-only, typed as
   `Optional[Callable[[float], float]]`, defaulting to `None`. `rate(z)` is
   `d/dz[leading.delta(z, z_anchor)]`'s magnitude — `1/H(z)` for `tau`, `c_s(z)/H(z)` for
   `cs_tau` — replacing the current `1/self._Hubble(z)` computed inline. When `rate is None`,
   construct the default `lambda z: 1.0 / self._Hubble(z)` from `model_functions.Hubble`, so that
   **every existing caller that does not pass `rate` needs no change and sees the same value**
   (verify this — see §3 item 1 — do not just assert it).

   Change `theta_deriv` (`:287`) from `self._sign * self._k / self._Hubble(raw_x)` to
   `self._sign * self._k * self._rate(raw_x)`. Update the class and module docstrings' "Derivative
   in closed form" section to describe `rate` directly (the module already writes
   `d theta / dz = sign * k / H(z) + phi'(z)` with `f(z) = 1/H(z)` for `tau` in its "Closed-form"
   discussion — generalise that paragraph to name `rate` rather than talking through `H_eff`).

   `model_functions` keeps its existing meaning and validation (`hasattr(..., "Hubble")`): it is
   still required, still used to build the default `rate`, and — now that no caller has to lie
   through it — `phase._Hubble` on every `PrimitivePhase`, `T_k` included, is genuinely `H(z)`.

2. **Drop the adapter in `TkSourceFunctions.py`.** Delete the `_SoundHorizonRate` class (`:203-232`)
   and rewrite the module docstring paragraph at `:98-101` to describe the `rate` callable directly
   rather than "`H_eff`" and the adapter. Change the `PrimitivePhase(...)` call (`:393-402`) to pass
   `model_functions=self._model.functions` (the real one, not an adapter) and
   `rate=<a callable computing c_s(z)/H(z)>`. Keep the existing positive-`c_s^2` `RuntimeError`
   check (`_SoundHorizonRate.Hubble`, `:224-230`) — it must still fire on the same condition with a
   message that still makes sense from wherever it now lives (a module-level function, a nested
   `def`, or a lambda plus a separate guarded helper — your choice, state it in the log).

3. **Confirm `GkSourcePolicyData.py` needs no edit.** It calls `PrimitivePhase(...)`
   (`:187-196`) without a `rate` argument, so it takes the new default. State in the log that you
   checked `git diff` shows no changes to that file, and that the Green's-function `theta_deriv` is
   unaffected (see §3 item 1).

## 3. Tests

### 3.1 `test_primitive_phase.py`

1. **Default-rate cases are unchanged.** Every existing test that builds a `PrimitivePhase`
   without `rate` (all four call sites: `:150`, `:478`, `:500`, `:531`) must pass with its
   thresholds unchanged — `git diff HEAD~1` on this file's existing assertions must show numbers,
   not edits to tolerances. This is the regression test that the default path is untouched.
2. **An explicit `rate` is honoured.** Add a case (e.g. a synthetic `leading` with a
   known-but-not-`1/H` derivative, or reuse the radiation control with `rate` set to something
   other than the implicit `1/H`) that asserts `theta_deriv` uses `rate(z)` and *not*
   `1/model_functions.Hubble(z)` when the two differ — this is what would have caught the original
   defect.
3. **`model_functions.Hubble` is not silently repurposed.** Build a `PrimitivePhase` with an
   explicit `rate` different from `1/Hubble`, and assert that `model_functions.Hubble(z)` (read
   directly off the object, however you choose to expose it — `phase._Hubble` or a public
   accessor) still returns the genuine Hubble rate, not the rate. This is the regression test for
   the impact statement in `[10-primitive-phase-leading-rate-is-hardcoded]`.

### 3.2 `test_tk_source_functions.py`

1. **`omega()` / `theta_deriv` for `T_k` are unchanged to measured precision.** The two
   expressions — old: `k / (H(z)/sqrt(c_s^2(z)))`; new: `k * (sqrt(c_s^2(z))/H(z))` — are
   mathematically identical but reorder the division and multiplication, so they need not be
   bit-identical. Measure the difference on the existing fixture(s) that exercise
   `test_omega_matches_phase_derivative` and quote it; it must be far below the 1e-9 / 1e-11
   window that test already asserts (`IMPLEMENTATION_STATE.md` M14/M15 note the 1e-10-scale
   numbers prompt 10 measured). Do not assume equality without measuring.
2. **`_SoundHorizonRate` is gone.** `git diff HEAD~1 --stat` on `TkSourceFunctions.py` must show
   no orphaned references — no import, no instantiation, nothing under that name.
3. Every other assertion in this file must pass unchanged (it must not need editing beyond
   whatever the removed class forces, if anything — most likely nothing, since no test names
   `_SoundHorizonRate` directly).

### 3.3 Full suite

`PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` passes; no
other package's tests are affected (this prompt does not touch anything outside `ComputeTargets/`).

## 4. Verification and acceptance

- All of §3 passes.
- `black --check` clean on both edited modules.
- `git diff HEAD~1 --stat` touches only the files in "Files you may touch" above.
- `[10-primitive-phase-leading-rate-is-hardcoded]` moves from §3 to §4 (Resolved) on the board,
  with what shipped and the measured §3.2 item 1 difference; its row is deleted from
  `docs/OPEN_ISSUES.md` and the header count corrected. M15's row gains a note that the adapter is
  gone.

## 5. Log and commit

"State handed to the next prompt", verbatim: the final `PrimitivePhase.__init__` signature; where
the positive-`c_s^2` check now lives; the measured §3.2 item 1 difference between the old and new
expression order.

Commit subject, or something equally specific: `Give PrimitivePhase an explicit leading-rate
callable`.
