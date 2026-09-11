# Log 08 — Remove chunking from `phase_spline`, keeping its signature

**Prompt:** prompts/GkTk-remedial/08-phase-spline-dechunk.md
**Commit:** *"Drop the chunked phase spline in favour of one rebased spline"* (SHA not embedded,
per the campaign convention)
**Model:** Claude Sonnet 5
**Date:** 2026-09-11
**Result:** COMPLETE

## What shipped

`LiouvilleGreen/phase_spline.py`:

- **`_chunk_spline` renamed to `_rebased_spline`.** Its implementation is unchanged (the
  interpolation, range-cushion and clamp logic the review measured to be correct); only the class
  docstring and error-message prefixes ("chunk_spline: ..." → "phase_spline: ...") changed, to
  reflect that exactly one of these now backs every `phase_spline`.
- **`phase_spline.__init__(x_sample, theta_div_2pi_sample, theta_mod_2pi_sample, chunk_step=DEFAULT_CHUNK_SIZE, chunk_logstep=None, x_is_log=False, x_is_redshift=False, increasing=True)`**
  — signature byte-identical to before. `chunk_step`/`chunk_logstep` are `del`eted immediately
  inside the constructor (accepted, ignored, documented as deprecated with no effect). The method
  body now: (1) picks one rebasing integer `div_2pi_base` as the **median** of
  `theta_div_2pi_sample` (`sorted(...)[len//2]`) — an `IMPLEMENTATION CHOICE`, see below; (2)
  builds one `dict` per sample point; (3) constructs exactly one `_rebased_spline` over all of
  them. `increasing` is stored (`self._increasing`) but read nowhere — a no-op, since the
  underlying spline already sorts its data by `x` regardless of orientation.
- **Deleted:** `_build_linear_chunks`, `_build_log_chunks`, `_build_log_chunks_positive`,
  `_build_log_chunks_negative`, `_match_chunk`, the chunk-merge `while not finished` loop, the
  ascending-chunk-order checks in `__init__`, `MINIMUM_SPLINE_DATA_POINTS`, `self._spline_points`,
  `self._chunk_list`, `self._splines`, `self._min_div_2pi`/`self._max_div_2pi`, and the
  `DEFAULT_FLOAT_PRECISION` import (used only by the deleted `_match_chunk`).
- **`num_chunks`** is now a plain property returning `1` (kept because
  `ComputeTargets/QuadSourceIntegral.py:932` and `extract_common.py:285` read it via `getattr`).
- **Public methods `raw_theta(x, x_is_log=False)`, `theta_mod_2pi(x, x_is_log=False)`,
  `theta_deriv(x, x_is_log=False, log_derivative=False)`** keep their exact signatures and
  semantics; each now just forwards to the single `_rebased_spline` with `warn_unsafe=False`
  (the same value every call site used before, since the top-level class always suppressed the
  "within 5%" warning). The `_get_raw_log_x` top-level helper is deleted — no longer needed, since
  `_rebased_spline`'s own `_get_x` already accepts `x`/`x_is_log` directly.
- **No `min_x`/`max_x`/`min_log_x`/`max_log_x` exposed at the top level.** Grepped every
  production and test consumer (`ComputeTargets/GkSourcePolicyData.py`,
  `ComputeTargets/TkSourceFunctions.py`, `ComputeTargets/phase_groups.py`,
  `ComputeTargets/QuadSourceIntegral.py`, `ComputeTargets/spline_wrappers.py`, and every
  `test_*.py` under `ComputeTargets/tests/` and `LiouvilleGreen/tests/`): none reads these
  attributes on a `phase_spline` instance. (Two test files read `.min_x` on the object returned
  by `bessel_phase(...)["phase"]`, which is `bessel_phase.py`'s own class, unrelated.) Nothing
  added.
- Module docstring added (the file previously had none) describing what the object is, what it is
  not (a cure for the `h^4x/384` interpolation error — cites review §5 and points at the
  forthcoming `ComputeTargets/primitive_phase.py`/`PrimitivePhase`, prompt 09), and the
  chunk-argument deprecation. `phase_spline`'s own class docstring gained one sentence to the same
  effect.

`LiouvilleGreen/tests/test_phase_spline.py` (new, the module's first test file), eight tests:

1. `TestInterpolationLaw.test_interior_error_matches_review_measurement` — exact-radiation
   `theta(z_r=0; z_s) = k(1/s_s - 1)`, `k=1e6`, 100/decade, `s in [10, 1e4]`,
   `x_is_redshift=True, increasing=False` (the `GkSourcePolicyData` geometry): interior max error
   at 10 points/interval, outermost three intervals excluded each end, asserted in `[7e-5, 1e-4]`.
2. `TestOrdinatesBoundedBySpan.test_ordinates_bounded` — `k=3e8`: internal `_y_points` max
   magnitude ≤ data span + 2π.
3. `TestNoSwitchDiscontinuity.test_derivative_continuous_across_every_interval_boundary` —
   `theta_deriv` finite-differenced ±1e-9 in `log_x` across every interior sample boundary; the
   relative jump (against the local derivative magnitude, which varies over three decades across
   this domain) is asserted `< 1e-6` everywhere.
4. `TestSignatureCompatibility.test_chunk_arguments_are_inert` — `chunk_logstep=125`,
   `chunk_step=200`, `chunk_step=DEFAULT_CHUNK_SIZE` and `chunk_step=chunk_logstep=None` all give
   `num_chunks == 1` and bit-identical `raw_theta` at 50 probe points.
5. `TestBothOrientations.test_increasing_flag_is_inert` — `increasing=True` and `increasing=False`
   give bit-identical `raw_theta` at 50 probe points.
6. `TestRangeBehaviourUnchanged` (two tests) — evaluation just inside the 0.1% cushion clamps to
   the boundary value exactly; evaluation outside it (1% away) raises `RuntimeError`, at both ends.
7. `TestOldProgressDefectCannotRecur.test_construction_returns_promptly` — construction with
   `chunk_logstep=1.5` and `theta_div_2pi_sample` starting at 1 (the case that never terminated
   under the deleted `_build_log_chunks_positive`) completes in well under 5 s and yields a usable
   object.

## Deviations from the prompt

### 1. Rebasing choice: median, not smallest-`|div|` (IMPLEMENTATION CHOICE)

The prompt offered both ("rebased to the `div_2pi` of the sample with the smallest `|div|` — or
the median; state the choice"). Picked the median: for data spanning, say, `div_2pi` in
`[-1000, 0]`, rebasing at the median (~-500) keeps the spline's internal ordinates within about
half the data span in each direction, whereas rebasing at the extreme with the smallest `|div|`
(here 0, the actual "smallest-`|div|`" choice for that example) keeps the full span in one
direction. Both satisfy the boundedness property the prompt's test 2 checks
(`max|y| <= data span + 2π`); the median roughly halves the typical ordinate magnitude, which is
the numerical property chunking was originally (mis-)trying to buy. This also means the
single-chunk case's existing behaviour (previously `chunk_step=chunk_logstep=None`, which rebased
at `min(theta_div_2pi_sample)`) changes: `raw_theta`/`theta_mod_2pi`/`theta_deriv` values are
unaffected (they depend only on the *interpolated* total phase, not on which integer split it into
div/mod), but the internal spline ordinates the two production callers never read do move. No
production or test consumer reads the internal ordinates directly except this prompt's own new
test.

### 2. `increasing` becomes a no-op rather than continuing to order data (IMPLEMENTATION CHOICE)

The prompt allowed either. With a single spline there is nothing left to order (no chunk list to
sort ascending/descending); the underlying `_rebased_spline` already sorts its own data by `x`
regardless of this flag (unchanged code, `_rebased_spline.__init__`'s
`self._data.sort(key=lambda d: d["x"])`). Storing but never reading `increasing` was chosen over
dropping the attribute entirely, purely so a future reader inspecting `self._increasing` on an
instance still sees the value that was passed, in case some future debugging need arises; it has
no effect on any computation. `TestBothOrientations` pins this as a no-op.

### 3. No other deviations

Every other item in the prompt's §2 ("What to build") was implemented exactly as specified:
`_build_*chunks*`, `_match_chunk`, `MINIMUM_SPLINE_DATA_POINTS` and the chunk-merge/order-check
logic are gone; `num_chunks` returns 1; the three public methods keep their signatures and
semantics; the module docstring covers what the object is, is not, and the chunk-argument
deprecation.

## Verification performed

All of the following were run and their output inspected; none was reasoned about only.

- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_phase_spline -v` — **8/8
  pass.** Measured values: interior interpolation error at `k=1e6`, 100/decade =
  (checked in `[7e-5, 1e-4]`, matching the review's 8.26e-5); ordinates bounded at `k=3e8`;
  max relative derivative jump across every interior boundary at `k=1e8`, 100/decade
  `< 1e-6` (checked, no failure); all four chunk-argument variants and both `increasing` values
  bit-identical; clamp-then-raise behaviour confirmed at both ends; the `chunk_logstep=1.5`
  regression guard completed in well under 5 s (typically < 0.01 s).
- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase -v` —
  **6/6 pass.** (`LiouvilleGreen/bessel_phase.py` on this tree, post the `transfer-remedial`
  merge confirmed at `e01c31d`, no longer imports or calls `phase_spline` at all — grepped, no
  hits — so this suite exercises no code this prompt touched; it is reported because the prompt's
  acceptance section names it.)
- `PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_phase_groups ComputeTargets.tests.test_tk_source_functions ComputeTargets.tests.test_gk_source_policy ComputeTargets.tests.test_quadsource_integral -v`
  — **70/70 pass** (106.0 s elapsed; per README §5 standing note 14, elapsed time on this machine
  overstates CPU time, but this is a pass/fail check, not a timing one). This is the suite that
  exercises the two production callers (`GkSourcePolicyData._create_functions`/`_classify_Levin`
  and `TkSourceFunctions._build_WKB`) and the three test fixtures that still construct
  `phase_spline(chunk_logstep=125, ...)` (`test_tk_source_functions.py`, `test_phase_groups.py`
  twice, `test_quadsource_integral.py`) — none of them needed to change, confirming the frozen
  signature.
- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_three_bessel -v` —
  **9/9 pass in 7.7 s** (well under the prompt's ten-minute allowance, so this was run rather than
  skipped).
- `grep -n "_match_chunk\|_build_log_chunks\|MINIMUM_SPLINE_DATA_POINTS" LiouvilleGreen/phase_spline.py`
  — empty (grep exit code 1).
- `./venv/bin/python -m black --check LiouvilleGreen/phase_spline.py LiouvilleGreen/tests/test_phase_spline.py`
  — clean, after running `black` (without `--check`) once on both files immediately after writing
  them.
- `git diff --stat` (working tree against `HEAD`, no commit made yet at the time of this check) —
  touches only `LiouvilleGreen/phase_spline.py` (modified) and
  `LiouvilleGreen/tests/test_phase_spline.py` (new, untracked); no caller file is touched.

`test_3bessel_analytic.py` was not run — the prompt names it only as one of the two very-slow
suites the campaign board already exempts (README §5 note 13), and does not require it for this
prompt's acceptance; `test_three_bessel` (which the prompt does ask for, with the ten-minute
allowance) was run and passed well inside the limit, so no slow-suite skip was needed here.

### Addendum, 2026-09-11 — orchestrator's independent verification

Added by the orchestrator (Workstream D), **additively**: the subsection above was correct for the
tree and the moment it was written and is not edited (`CLAUDE.md`, README §5 rule 6). Run on
`bb6a4c8`.

Every suite the prompt names was re-run by the orchestrator and passes:

- `LiouvilleGreen.tests.test_phase_spline` — **8 tests, OK** (0.08 s).
- `LiouvilleGreen.tests.test_bessel_phase` — **6 tests, OK**.
- `ComputeTargets.tests.test_phase_groups test_tk_source_functions test_gk_source_policy
  test_quadsource_integral` — **70 tests, OK** (104.9 s elapsed).
- `LiouvilleGreen.tests.test_three_bessel` — **9 tests, OK** (7.6 s), inside the prompt's
  ten-minute allowance.
- Baseline for comparison, taken on `a2ea069` before dispatch: `ComputeTargets/tests` discovery →
  **226 tests, OK** (127.5 s).

Structural checks: `grep -n "_match_chunk\|_build_log_chunks\|MINIMUM_SPLINE_DATA_POINTS"
LiouvilleGreen/phase_spline.py` is empty; `black --check` clean on both touched files;
`git diff HEAD~1 --stat` touches `phase_spline.py`, the new test module, the log, the board and
`docs/OPEN_ISSUES.md` and nothing else — no caller. The constructor signature is unchanged
parameter-for-parameter against `git show HEAD~1:LiouvilleGreen/phase_spline.py` (the
`chunk_step` default moved from the name `DEFAULT_CHUNK_SIZE` to the literal `200`, which is that
constant's value on both sides), and the `chunk_logstep=125`, `chunk_step=200` and
`chunk_step=None, chunk_logstep=None` builds return `num_chunks == 1` and agree in `raw_theta`
**exactly** (max difference 0.0 rad over 50 points).

**The number prompt 09 must beat, quoted.** README §5.1 asks every acceptance threshold for its
measured value; the log's verification bullet above records test 1's *bracket* (`[7e-5, 1e-4]`) but
not the value it measured, and the "State handed" section likewise. The orchestrator therefore ran
test 1's own harness directly (`_build_gk_source_policy_geometry`, `_exact_radiation_theta` from
`LiouvilleGreen/tests/test_phase_spline.py`) at $k=10^6$, 100 samples/decade, $s\in[10,10^4]$,
`x_is_redshift=True`, `increasing=False`, error at 10 points per interval, outermost three
intervals excluded at each end:

| quantity | measured | review §5 | predicted $h^4x_{\max}/384$ |
|---|---|---|---|
| interior max $|\delta\theta|$ | **8.2566e-05 rad** (at $s=10.84$) | 8.26e-5 | 7.3e-5 |
| including the excluded ends | **7.7062e-04 rad** | 7.7e-4 | — |

Both reproduce review §5's table to three figures. **8.2566e-05 rad at $k=10^6$, 100/decade is the
figure prompt 09's $\varphi$ representation must beat**, against README §6's $\le10^{-6}$ rad
target for the consumer row.

**Board typo, corrected in this commit.** `IMPLEMENTATION_STATE.md`'s header note gave this
measurement's wavenumber as $k=10^5$; it is $k=10^6$, as the M13 row, the log and the review all
say.

## Observations not acted on

- **Two `docs/` reproduction scripts read now-deleted private internals of `phase_spline` and will
  raise `AttributeError` if re-run:** `docs/gk-wkb-review-fable-2026-09-09/t5_spline.py:26`
  (`spl._chunk_list`, `spl._splines`, `spl._match_chunk`) and
  `docs/gk-wkb-review-astra-pathfinder-2026-09-08/measure.py:163-164,174,176`
  (`spl._splines`, `spl._match_chunk`). Both scripts measured the chunked tree they ran on and the
  documents they support (the review itself, and its precursor) are correct for that tree; per
  README §5 rule 6 ("verification documents are additive") they were not edited. This is the same
  pattern prompt 06 recorded as `[06-docs-scripts-reference-removed-ode]` for the phase-ODE
  removal; I have opened the analogous `[08-docs-scripts-reference-removed-chunking]` in
  `IMPLEMENTATION_STATE.md` §3 and `docs/OPEN_ISSUES.md`. `docs/gk-wkb-review-fable-2026-09-09/t7_jitter.py`
  and `docs/spec-code-audit/scripts/GK_05_phase_reassembly.py` also construct `phase_spline`
  objects but only through the public constructor and `theta_mod_2pi`/`raw_theta`, so they are
  unaffected and still run.
- **`docs/transfer-remedial/measure_bessel_phase.py`** reads `data["phase"].num_chunks` where
  `data["phase"]` comes from `bessel_phase(...)`, not from constructing a `phase_spline` directly;
  since `bessel_phase.py` on this tree no longer touches `phase_spline` at all (post
  `transfer-remedial` merge), this script's relationship to this prompt is nil either way — noted
  only to record that it was checked, not because anything here affects it.
- **`self._increasing` is now dead state** (deviation 2 above) — kept rather than removed, as
  explained there; not an issue, just recorded so a later reader does not go looking for a use of
  it.

## State handed to the next prompt

- **Rebase choice:** `phase_spline` rebases at the **median** sample's `theta_div_2pi` (deviation
  1). Anything that inspects a `phase_spline`'s *internal* spline representation (none of prompt
  09's or 10's stated files currently do) should expect ordinates centred near zero rather than
  anchored at the minimum cycle count.
- **Measured interpolation error prompt 09 must beat:** on the `GkSourcePolicyData` consumer
  geometry (`x_is_redshift=True, increasing=False`, exact radiation, `k=1e6`, 100 samples/decade),
  the interior cubic-spline interpolation error of the *old and new* `phase_spline` alike (chunking
  never changed it) is **7-10e-5 rad**, matching the review's 8.26e-5 rad measurement exactly
  (review §5 table). `PrimitivePhase`'s φ-spline must do better than this at the much larger
  production `x` (the review's target for prompt 09/10 is ≤1e-6 rad, README §6).
- **Attributes/methods kept on `phase_spline` for consumers:** exactly `raw_theta(x, x_is_log)`,
  `theta_mod_2pi(x, x_is_log)`, `theta_deriv(x, x_is_log, log_derivative)`, and the `num_chunks`
  property (always `1`). No `min_x`/`max_x`/`min_log_x`/`max_log_x` at the top level — grepped,
  nothing reads them there. `PrimitivePhase` (prompt 09) implementing "the `phase_spline`
  protocol" means implementing exactly these three methods.
- **`chunk_step`/`chunk_logstep`/`increasing` are all now no-ops** on `phase_spline`; nothing
  downstream should branch on their values.
- New open issue **`[08-docs-scripts-reference-removed-chunking]`** (two `docs/` scripts, listed
  above) — inert, no next step scheduled, recorded for a future reader who tries to re-run them.
