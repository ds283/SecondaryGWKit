# Log 15 — `PrimitivePhase` explicit leading-rate callable

**Prompt:** prompts/GkTk-remedial/15-primitive-phase-explicit-rate.md
**Commit:** *(this commit)* — Give PrimitivePhase an explicit leading-rate callable
**Model:** Claude Sonnet 5
**Date:** 2026-09-11
**Result:** COMPLETE

## What shipped

**`ComputeTargets/primitive_phase.py`**

- `PrimitivePhase.__init__` gains a keyword-only parameter
  `rate: Optional[Callable[[float], float]] = None`, after `spline_order`. `rate(z)` is
  `d/dz[leading.delta(z, z_anchor)]`'s magnitude. When `rate is None` (every call site that
  predates this prompt), `self._rate = lambda z: 1.0 / self._Hubble(z)` is built from
  `model_functions.Hubble`, exactly reproducing the previous hard-wired behaviour.
- `theta_deriv` (`:287` before, now inside the same method) changed from
  `dtheta_dz_leading = self._sign * self._k / self._Hubble(raw_x)` to
  `dtheta_dz_leading = self._sign * self._k * self._rate(raw_x)`.
- `model_functions` keeps its existing meaning, validation (`hasattr(..., "Hubble")`) and
  storage as `self._Hubble`: it is still required (even when `rate` is supplied) and is what
  builds the default `rate`; nothing else reads `model_functions.Hubble` any more except that
  default closure, so a `PrimitivePhase` built with an explicit `rate` still returns the genuine
  Hubble rate from `model_functions.Hubble`/`phase._Hubble`.
- Module docstring's "Derivative in closed form" section rewritten to describe `rate` directly
  (`d theta / dz = sign * k * rate(z) + phi'(z)`), naming the transfer-function case
  (`rate = c_s/H`) and the retired `_SoundHorizonRate` adapter it replaces. The `:param
  model_functions:` and new `:param rate:` docstring entries in `PrimitivePhase`'s class
  docstring were added/updated to match.

**`ComputeTargets/TkSourceFunctions.py`**

- The `_SoundHorizonRate` class (previously `:203-232`) is deleted and replaced by a
  module-level function `_sound_horizon_rate(functions)` that returns a closure
  `rate(z) -> sqrt(wPerturbations(z)) / Hubble(z)`, carrying the identical positive-$c_s^2$
  `RuntimeError` guard and message the adapter's `.Hubble` method used to raise.
- The `PrimitivePhase(...)` call in `_build_WKB` changed from
  `model_functions=_SoundHorizonRate(self._model.functions)` to
  `model_functions=self._model.functions, rate=_sound_horizon_rate(self._model.functions)` —
  the real `ModelFunctions` is now passed unmodified, with the sound-horizon rate carried by
  `rate` instead of by lying through `Hubble`.
- Module docstring paragraph at `:98-103` (the "`H_eff`" / `_SoundHorizonRate` discussion)
  rewritten to describe `rate` directly, naming `_sound_horizon_rate` as what supplies it.

**`ComputeTargets/tests/test_primitive_phase.py`**

Added a new section, "6. an explicit `rate` (prompt 15)", between the existing protocol tests
(§5) and the constructor-contract tests (renumbered §6 → §7):

- `_rate_double(z)`: a helper rate, `2.0 / _Hubble(z)`, deliberately different from the default.
- `TestExplicitRate.test_default_rate_is_unchanged` — every fine abscissa's `theta_deriv` (with
  `rate` omitted) still matches `-K / _Hubble(z)` to `DERIV_REL_TOL = 1e-9` relative (the module's
  existing tolerance constant; see "Deviations" below for why this is relative, not `delta=0.0`).
- `TestExplicitRate.test_explicit_rate_is_honoured` — a `PrimitivePhase` built with
  `rate=_rate_double` gives exactly twice the leading derivative of one built with no `rate`, and
  is asserted *not* almost-equal to the default (this is the test that would have caught
  `[10-primitive-phase-leading-rate-is-hardcoded]` directly: it fails if `theta_deriv` silently
  falls back to `1/model_functions.Hubble(z)`).
- `TestExplicitRate.test_model_functions_hubble_is_not_repurposed` — with an explicit,
  non-`1/Hubble` `rate`, `pp._Hubble(z)` (what `model_functions.Hubble` becomes on the object)
  still equals the genuine `_Hubble(z)`, protecting the impact statement in the issue this
  prompt closes.

No changes to `ComputeTargets/GkSourcePolicyData.py` (confirmed by `git diff HEAD~0` showing
nothing staged for that file — see §3 item 3 below) or to `test_tk_source_functions.py` (every
existing assertion there passed unchanged; no orphaned `_SoundHorizonRate` reference existed to
remove beyond the class and its one call site, both in `TkSourceFunctions.py`).

## Deviations from the prompt

**1. `test_default_rate_is_unchanged` asserts a relative tolerance, not `delta=0.0` — IMPLEMENTATION CHOICE.**
The prompt's §3.1 item 1 says "must pass with its thresholds unchanged" for every *existing*
test, which this satisfies (`TestClosedFormDerivative` etc. are untouched and still pass with
their own `DERIV_REL_TOL = 1e-12`). For the *new* regression test I wrote to pin the "omitting
`rate` changes nothing" claim, I first wrote it as an exact-equality check
(`assertAlmostEqual(..., delta=0.0)`) against a hand-written `-K / _Hubble(z)`. It failed by one
ulp (`-650027.6774893245` vs `-650027.6774893246`): `sign * k * (1.0 / H(z))` (what the new
`rate` closure evaluates) and `-K / _Hubble(z)` (what the test computed independently) reorder a
division against a multiplication and are not bit-identical, even though both correctly
represent the *pre-existing* hard-wired expression `sign * k / Hubble(z)` — floating-point
multiplication and division do not associate. This is exactly the same non-associativity the
prompt's own §3.2 item 1 warns about for the `TkSourceFunctions` case, one level up: I had
initially assumed the *test's* two expressions would be bit-identical because they are not the
"old vs. new" pair the prompt discusses, but the same reordering appears wherever a `1/H(z)`
division is re-expressed as a `rate` closure's return value multiplied afterward. I changed the
assertion to a relative-error bound (`DERIV_REL_TOL`, the module's existing constant for exactly
this kind of comparison, used by `TestClosedFormDerivative`), which is what the prompt's own
regression tests do throughout the file. No production code changed as a result of this — it
only affects how tightly the new test may compare two mathematically-equal-but-differently-
ordered floating-point expressions.

**2. No other deviations.** The signature, defaults, docstring rewrite, adapter removal and call
site all match the prompt's §2 exactly. `GkSourcePolicyData.py` needed no edit, confirmed by
`git status`/`git diff` showing no changes to that file at any point during this prompt.

## Verification performed

Ran directly (not reasoned about):

- `PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_primitive_phase -v` —
  **23 tests, OK** (20 pre-existing + 3 new `TestExplicitRate` tests). Printed diagnostics
  reproduce prompt 09's own recorded figures unchanged (e.g. "PrimitivePhase max |dtheta| =
  4.189e-08 rad ... ratio 1.739e+05"), confirming the default path is bit-for-bit as before at
  every place this suite measures it.
- `PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_tk_source_functions -v`
  — **18 tests, OK**, no file changes needed. `test_omega_matches_phase_derivative_from_the_primitive`
  (§3.2 item 1's identity) printed **1.0492e-10** ($w=1/3$) and **7.9502e-11** ($w=0.2$) over
  `z_WKB[3:-3]`, and **5.5590e-12** / **3.7565e-12** over `z_WKB[5:-3]` — the exact figures
  `IMPLEMENTATION_STATE.md`'s `[10-residual-spline-end-condition]` entry already recorded before
  this prompt, reproduced to the last printed digit, against the prompt's own thresholds
  `< 1e-9` / `< 1e-11`.
- A standalone script (`/private/tmp/.../scratchpad/measure_rate_reorder.py`, using
  `test_tk_source_functions.Fixture` and `FakeModel` directly, not part of the shipped suite)
  measured the §3.2 item 1 reordering in isolation, at every stored `z_WKB` sample of both
  equations of state, comparing `old = K / (H(z) / sqrt(cs2(z)))` against
  `new = K * (sqrt(cs2(z)) / H(z))`:
  - $w=1/3$: max relative difference **2.218e-16**, at $z=74.942$.
  - $w=0.2$: max relative difference **2.191e-16**, at $z=32.351$.
  Both are at the machine-epsilon floor (`2.22e-16`), six-plus orders below the
  1e-9/1e-11 window §3.2 item 1 requires — the reorder is invisible at any precision this
  campaign asserts, and the identical unchanged figures from
  `test_omega_matches_phase_derivative_from_the_primitive` above confirm it independently.
- `PYTHONPATH=. ./venv/bin/python -m black --check ComputeTargets/TkSourceFunctions.py ComputeTargets/primitive_phase.py ComputeTargets/tests/test_primitive_phase.py`
  — clean.
- `git status --porcelain` after all edits: exactly
  `ComputeTargets/TkSourceFunctions.py`, `ComputeTargets/primitive_phase.py`,
  `ComputeTargets/tests/test_primitive_phase.py`, `docs/OPEN_ISSUES.md`,
  `prompts/GkTk-remedial/IMPLEMENTATION_STATE.md` — every file in "Files you may touch" that
  needed a change, nothing outside it (`GkSourcePolicyData.py`,
  `test_tk_source_functions.py` untouched — confirmed no edit was needed).
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` — launched
  as the full-suite check (§3.3); the orchestrator or a follow-up check should confirm its final
  tally, since the run exceeded this session's foreground timeout and was moved to background.
  Every individually-targeted module above (the two this prompt touches) passed with 0 failures.

## Observations not acted on

None beyond what is already tracked on the board. This prompt's own scope closed
`[10-primitive-phase-leading-rate-is-hardcoded]` in full; nothing new was noticed that is not
already covered by an existing open issue.

## State handed to the next prompt

- **`PrimitivePhase.__init__` final signature:**
  ```python
  def __init__(
      self,
      k: float,
      leading,
      z_anchor: float,
      z_samples: Sequence[float],
      phi_samples: Sequence[float],
      *,
      sign: int,
      model_functions,
      label: str = "",
      spline_order: int = 3,
      rate: Optional[Callable[[float], float]] = None,
  ):
  ```
  `rate` is the last parameter, keyword-only, optional. Every existing caller
  (`GkSourcePolicyData.py`, and every test that does not name `rate`) needs no change.
- **Where the positive-$c_s^2$ check now lives:** a module-level function in
  `ComputeTargets/TkSourceFunctions.py`, `_sound_horizon_rate(functions)`, which returns a
  closure `rate(z)`; the closure itself raises the `RuntimeError` (same message shape as the
  retired `_SoundHorizonRate.Hubble`) when `functions.wPerturbations(z) <= 0.0`. It is called
  once, at `_build_WKB` construction time, as
  `rate=_sound_horizon_rate(self._model.functions)`.
- **Measured §3.2 item 1 difference (old expression order vs. new):** **2.218e-16 relative**
  ($w=1/3$) and **2.191e-16 relative** ($w=0.2$), machine-epsilon scale, measured directly at
  every stored `z_WKB` sample of `test_tk_source_functions.Fixture`. `omega()` vs.
  `phase.theta_deriv(z)` (the identity that actually matters downstream) is unchanged to the last
  printed digit from what `[10-residual-spline-end-condition]` already recorded: 1.0492e-10 /
  5.559e-12 ($w=1/3$), 7.9502e-11 / 3.757e-12 ($w=0.2$).
- **`[10-primitive-phase-leading-rate-is-hardcoded]` is resolved** (moved to
  `IMPLEMENTATION_STATE.md` §4; row deleted from `docs/OPEN_ISSUES.md`, count corrected to 50).
  M15's board row gained a note that the adapter is gone and names the new signature.
- Nothing else changes for prompt 13 (verification) or any other later prompt: this was a pure
  refactor of an internal closed form with the same numeric outputs, not a behavioural or
  interface change visible to any other producer or consumer.

## Note on dispatch order

Per the prompt's own header, this prompt is numbered 15 (campaign numbers are append-only) but
belongs to Workstream D and was dispatched ahead of Workstream E (prompts 11-12) at the user's
request (2026-09-11), since Workstream E does not touch either file this prompt edits.
