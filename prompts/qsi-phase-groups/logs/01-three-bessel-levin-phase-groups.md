# Log 01 — Assemble `_three_bessel_Levin`'s phases as groups, and declare their error

**Prompt:** prompts/qsi-phase-groups/01-three-bessel-levin-phase-groups.md
**Commit:** *(this commit; SHA not self-embedded)* — Assemble the analytic three-Bessel phases as groups
**Model:** Claude Opus 5
**Date:** 2026-09-11
**Result:** COMPLETE WITH DEVIATIONS — every item of the prompt shipped and every test named in
prompt §4 passes. The one acceptance sentence not met is README §5's "every
`adaptive_levin_sincos` call in `QuadSourceIntegral.py` supplies all four phase keys", which is
true of the eight analytic calls this campaign owns and false of the **ninth**, in
`phase_group_Levin_integral` — a cosmological-phase call site that both README §2 and the prompt's
own file list forbid touching, and which has nothing to declare (a `phase_spline` does not report
its fit accuracy). Deviation 1; issue `[01-cosmological-group-declares-no-phase-error]`.

## What shipped

### `LiouvilleGreen/three_bessel_integrals.py` — naming only, no behaviour change

* `_PhaseGroup` (`:220`) is renamed **`BesselPhaseGroup`**, and `_PhaseGroup = BesselPhaseGroup`
  is bound at `:452`, immediately above `_phase_group()`. Nothing else in the class changed: same
  `__init__(phases, coefficients, signs)`, same `K`/`C`/`C_reduced`, same `arguments`, `residual`,
  `theta`, `sin_cos`, `theta_mod_2pi`, `theta_deriv`, `theta_abserr`, `levin_theta`.
* `_phase_group()`'s return annotation and docstring name the new class; three prose mentions of
  `_PhaseGroup` (`:29`, `:45`, `:442`) follow the rename. The class docstring gains one paragraph
  saying why the name is spelled out and why `ComputeTargets/phase_groups.PhaseGroup` is a
  different object that must not be unified with it.
* `LiouvilleGreen/tests/test_three_bessel.py` is **not touched** — it imports `_PhaseGroup` in
  eight places and the alias keeps every one of them working. It passes unchanged (9 tests, 8.8 s).

### `ComputeTargets/QuadSourceIntegral.py`

* Import added: `from LiouvilleGreen.three_bessel_integrals import BesselPhaseGroup` (`:22`). No
  cycle: `three_bessel_integrals` imports only `AdaptiveLevin` and `Quadrature`.
* `_three_bessel_Levin`: the eight closures `phase1`/`phase1_mod_2pi` … `phase4`/`phase4_mod_2pi`
  (78 lines) are replaced by four objects,

  ```python
  group_phases = (phase_Gk, phase_Tk, phase_Tk)
  group_coefficients = (k.k, q.k * cs, r.k * cs)
  group1 = BesselPhaseGroup(group_phases, group_coefficients, (1.0,  1.0,  1.0))
  group2 = BesselPhaseGroup(group_phases, group_coefficients, (1.0,  1.0, -1.0))
  group3 = BesselPhaseGroup(group_phases, group_coefficients, (1.0, -1.0,  1.0))
  group4 = BesselPhaseGroup(group_phases, group_coefficients, (1.0, -1.0, -1.0))
  ```

  and each of the eight `adaptive_levin_sincos` calls now passes `theta=groupN.levin_theta()` —
  the four-key dict — to both the `J` and the `Y` call of its group. The same `phase_Tk` object
  appears twice with different coefficients, which the class handles. **Nothing else about the
  calls changed**: the `f` vectors, `atol`, `rtol`, `chebyshev_order` and the eight `notify_label`
  strings are byte-identical, so the `J = A sin theta`, `Y = -A cos theta` convention is preserved
  exactly.
* A comment block above the group construction states the three defects removed, quotes the
  sibling module's measured numbers and points at this repository's own re-measurement, records
  the `theta_mod_2pi` representative change from `(-3 pi, 3 pi]` to `(-pi, pi]` and why it is
  invisible (every consumer takes only sin/cos of it), and says why
  `ComputeTargets/phase_groups.py` is deliberately not used here.
* **`BESSEL_ORDER_CHECK_TOL`'s comment** (`:80`): the "`bessel_phase()` reconstructs `J_nu` to
  ~2e-8 of that envelope" claim is replaced by the measured current figures — the phase object
  declares 5e-12 rad (8.9e-16 at `nu = 1/2`, where the phase is closed-form) and the
  reconstruction tracks scipy's `J_nu` to at worst 2.1e-12 (`nu = 1/2 + b`) and 2.4e-12
  (`nu = 5/2 + b`) of the envelope over the whole domain at `max_x = 1e5`, and to 2e-16–1e-14 at
  the two abscissae the guard actually samples. The reasoning is unchanged in shape: 1e-3 now sits
  nine orders above the reconstruction floor instead of five, and two below the smallest defect it
  has to catch.
* **`_check_bessel_order`'s docstring** (`:675`): the two false halves are corrected. The dict does
  carry `"nu"` and does **not** carry `Q` (verified: the keys are `accuracy`, `accuracy_met`,
  `amplitude_relerr`, `bessel_j`, `bessel_y`, `crossover`, `max_x`, `min_x`, `mod`, `near_region`,
  `nu`, `phase`, `phi`, `theta_abserr`, `theta_deriv_relerr`, `x_star`). The docstring now says
  the numeric check is kept **deliberately** rather than for the obsolete reason, and why: a direct
  `phase_data["nu"]` comparison would accept a phase object whose declared order disagreed with its
  own splines, and reject a correct one built differently. **No code in the guard changed.**

### `ComputeTargets/tests/test_quadsource_integral.py` — added, nothing removed

A module-level reference block (`EPS`, `PHASE_GROUP_*`, `_phase_group_coefficients`,
`_phase_group_reference`, `_legacy_qsi_group`) and **`TestThreeBesselPhaseGroups`, 4 tests, 0.7 s**:

| test | what it pins |
|---|---|
| `test_every_analytic_Levin_call_supplies_all_four_phase_keys` | all 16 analytic Levin calls (8 groups × 2 `nu_type`s) carry exactly `{theta, theta_mod_2pi, theta_deriv, theta_abserr}` — the test that would have caught the gap |
| `test_the_group_phase_and_derivative_beat_the_summed_route_at_resonance` | `|delta Theta|` and `|delta dTheta/dlog eta|`, new vs old, against 60-digit `mpmath` at the exact products, for a near-resonant and a generic group; the declared `theta_abserr` plus one rounding of `K eta` bounds the measured error at every point; the derivative's acceptance metric is `|delta| / (max(k, q cs, r cs) eta)` |
| `test_the_group_derivative_has_no_relative_scale_at_resonance` | the trap, as a measurement: the group log-derivative is 1.21e-08 against a constituent scale of 1.0e+09, so a relative metric is meaningless; the old route gets it wrong by 100 % of itself |
| `test_declaring_theta_abserr_enlarges_the_reported_abserr` | every one of the 16 calls is run twice, once with the key and once with the same dict minus `theta_abserr`, and the reported `abserr` is compared |

The reference is built from the constituent Bessel functions with `mpmath` at 60 digits, at the
**exact** products `m_i eta` rather than at the doubles `fl(m_i eta)` — a route that forms
`fl(m_i eta)` commits an error `~eps m_i eta` with unit sensitivity, so a reference taken at the
doubles would hide exactly the error the restructure removes. `_legacy_qsi_group` keeps the
assembly this commit replaced, so the improvement is re-measured on every run rather than quoted
from this log.

One sentence was **deleted** from `TestBesselOrderGuard`'s docstring — "bessel_phase() does not
record its order, so the guard checks it numerically" — which is the same falsehood Q3 corrects in
`_check_bessel_order`. No test, assertion or threshold was removed or changed.

## Deviations from the prompt

### 1. The ninth Levin call in the file still supplies three keys — STRUCTURALLY REQUIRED

README §5 and prompt §4 both require "every `adaptive_levin_sincos` call in
`QuadSourceIntegral.py` supplies all four phase keys". There are nine. Eight are
`_three_bessel_Levin`'s and all eight now do. The ninth is `phase_group_Levin_integral`'s
(`:1062`), which passes `group.levin_theta(include_deriv=LEVIN_USE_THETA_DERIV)` from
`ComputeTargets/phase_groups.py`, whose `levin_theta` (`:194-202`) constructs
`{"theta", "theta_mod_2pi"}` plus optionally `"theta_deriv"` and has no `theta_abserr` branch at
all.

Closing that gap requires editing `ComputeTargets/phase_groups.py`, which is the first entry on
the prompt's "Do not touch" list, is out of scope on the board's §2, and is the subject of README
§2's explicit warning that it is a different object. It is also not a one-line fix: its phases are
cosmological `phase_spline` objects, which do not report their own fit accuracy, so there is no
number to declare — the root cause is `prompts/levin-refactor`'s
`[09-abserr-does-not-bound-phase-spline-floor]` seen from the other end.

I did not stop and ask (CLAUDE.md rule 4), because the prompt's own *executable* statement of the
requirement is narrower and is met: §3 item 1 asks for "Every **analytic** Levin call carries all
four keys", mirroring the 16-call spy, and that is what the shipped test asserts. §4's grep bullet
is a looser restatement of the same requirement that happens to over-reach by one call site. The
prompt row is marked ⚠️ rather than ✅ for this, and the gap is recorded as
`[01-cosmological-group-declares-no-phase-error]` on the board's §3 and in `docs/OPEN_ISSUES.md`
§2. **This is the item the orchestrator should surface.**

### 2. Renamed the class rather than adding an alias, and chose `BesselPhaseGroup` — IMPLEMENTATION CHOICE

Prompt §2.1 offers either. I renamed `_PhaseGroup` to `BesselPhaseGroup` and bound `_PhaseGroup`
to it, so `test_three_bessel.py` needed no change at all (the prompt permits updating its import;
not needing to is better, since the prompt also says to change nothing else in that file).

The name is not `PhaseGroup`, which would have been the obvious public spelling, because
`ComputeTargets/phase_groups.py` already owns a class of that name for the cosmological phases.
Two modules exporting `PhaseGroup` with the same method names and incompatible semantics is
precisely the confusion README §2 exists to prevent, and `ComputeTargets/QuadSourceIntegral.py`
imports from both modules. Alternatives considered: `PhaseGroup` (rejected, above);
`ThreeBesselPhaseGroup` (accurate but wrong — the class is not limited to three factors in
anything but its docstring's worked example, and the constructor takes sequences);
`LeadingTermPhaseGroup` (describes the mechanism rather than the domain, and the mechanism is what
might change). `BesselPhaseGroup` names what makes it different from the cosmological one: its
constituents are `BesselPhaseFunction` objects with an analytic leading term.

### 3. The measurement is taken at `x_max = 1e9`, not the sibling's `1e12` — IMPLEMENTATION CHOICE

Prompt §3 item 2 sets no range. `LiouvilleGreen/tests/test_three_bessel.py` measures to `x = 1e12`;
this call site's splines are built by `main.py` out to `x ~ 1e9` (the figure
`_check_bessel_order`'s abscissa comment already quotes), so that is where the measurement is
taken. It costs a factor ~3 in the absolute error of both routes and changes no conclusion; the
alternative would have been to report an improvement at an argument production never reaches.

### 4. The "generic" group improves by a factor 16 in the phase, where the prompt expected none — STRUCTURALLY REQUIRED (a measurement, not a change)

Prompt §3 item 2 says "**Expect the generic group to show no improvement**… A test that claims a
gain there is measuring something else", citing `[07-generic-K-product-rounding]`'s 1.5e-5 for both
routes at `K = 0.1, x = 1e12`. Measured here, the generic group's phase error falls from 1.1921e-07
to 7.4506e-09, a factor 16, while its **derivative** error is 2.9802e-08 for both routes, a factor
1.00.

This is not a contradiction of that issue and it is not a cancellation gain. Each route sits at one
rounding of its own leading scale — `eps |K| eta` for the group, `eps max(m_i) eta` for the sum of
three raw phases — and the ratio is therefore just `max / |K|` times the three-term summation. The
sibling's generic case has `K/max = 0.048` and the two floors happen to coincide there; this call
site's generic group has `|K| / max = 0.27`, so they differ by construction. The test asserts the
floor *model* for the generic case (`|delta Theta| <= 4 eps |K| eta_max`) and claims no gain, and
both the test's docstring and its inline comment say the 16 must not be read as the cancellation
improvement, which is 3.7e4.

### 5. The "old" derivative measured is better than the derivative the old code actually had — IMPLEMENTATION CHOICE

Prompt §3 item 2 asks for the log-derivative "from the old route". The old route supplied **no**
`theta_deriv` at all, so what Levin actually used was a spectral differentiation of the summed raw
phase over the Chebyshev grid, whose error is `~eps theta / (phase span across the subinterval)`
and depends on the subdivision. Reproducing that faithfully means reproducing the driver's grid
and is not a property of the phase assembly.

`_legacy_qsi_group.theta_deriv` therefore returns the signed sum of the three constituents' own
`theta_deriv(..., log_derivative=True)` values — the best the old route *could* have done had it
supplied one, which is also exactly what the sibling module's own legacy comparator does. The
measured 2368× improvement at resonance is therefore a **lower bound** on the improvement over
what shipped. The test's docstring says so.

### 6. Both the phase and the derivative are measured in one test rather than two — IMPLEMENTATION CHOICE

Prompt §3 items 2 and 3 read naturally as separate tests. The 60-digit `mpmath` reference is by far
the most expensive thing in the measurement and is the same for both quantities, so computing it
once per grid point and scoring both against it halves the cost. Item 3's substantive requirement —
that the derivative is scored against `max(k, q cs, r cs) eta` and **never** divided by the group
derivative — is an assertion inside that test, and the trap itself is stated as its own test
(`test_the_group_derivative_has_no_relative_scale_at_resonance`) because it is a claim about the
metric rather than about the assembly.

### 7. One sentence deleted from a test docstring — IMPLEMENTATION CHOICE

`TestBesselOrderGuard`'s docstring repeated `_check_bessel_order`'s false claim that
"bessel_phase() does not record its order". Q3 is scoped to "two comments in
`QuadSourceIntegral.py`", so this third copy is strictly outside it; leaving a corrected docstring
next to an uncorrected restatement of the same falsehood seemed worse than a one-line edit in a
file the prompt already puts in scope. It changes no assertion and no threshold.

## Verification performed

All runs from the repository root with `PYTHONPATH=. ./venv/bin/python`, on this commit unless
stated.

### The improvement, measured (prompt §3 item 2)

`b = 0` so `c_s = sqrt(1/3)`; orders `(1/2, 5/2, 5/2)` — the Green's function and the two transfer
factors of the `"2pt5"` call — with the **same** `phase_Tk` object at two coefficients, as
`_three_bessel_Levin` holds them; signs `(+, -, -)`, the group that can cancel; coefficients
`(k, q c_s, r c_s)`; 31 log-spaced points in `eta` up to `eta_max = 1e9 / max(coefficients)`.
Reference: 60-digit `mpmath` at the exact products. "old" is `_legacy_qsi_group` in the same file,
re-measured on every run.

The near-resonant triple takes `k = fl(q c_s + r c_s)` with `q = 1e4`, `r = 1.2e4`; for those two
coefficients that sum is exactly representable, so `K` is exactly `0.0` (asserted).

| case | `K` | `|K|/max` | `|delta Theta|` old | new | ratio |
|---|---|---|---|---|---|
| near-resonant | `0.0` | 0 | 5.1473e-08 (eta=78730) | **1.3758e-12** (eta=0.010133) | **3.741e4** |
| generic | `-2701.7059221717664` | 0.2702 | 1.1921e-07 (eta=52592) | 7.4506e-09 (eta=14547) | 16 |

| case | `|delta dTheta/dlog eta|` old | new | ratio | scaled by `max(k,q cs,r cs) eta`: old / new |
|---|---|---|---|---|
| near-resonant | 2.2824e-08 (eta=41737) | **9.6402e-12** (eta=0.010133) | **2368** | 7.4728e-14 / 7.4898e-14 |
| generic | 2.9802e-08 (eta=27660) | 2.9802e-08 (eta=52592) | 1.00 | 6.8996e-14 / 6.8962e-14 |

Worst measured `|delta Theta|` against the declared bound (`group.theta_abserr` plus one rounding
of `K eta`): **0.4877** near-resonant, **0.4895** generic — i.e. the declaration bounds the
measurement with a factor 2 to spare in both cases.

The scaled derivative metric is the acceptance threshold (asserted `< 1e-11`; measured 7.49e-14 and
6.90e-14) and, exactly as in the sibling module, **it does not distinguish the two routes**: both
attain their maximum at the bottom of the grid, where `eta` is small, there is no large leading
term to cancel and both sit on the residual interpolation floor. The improvement is in absolute
radians, which is what Levin's basis conditioning and subdivision consume (`phase_span`,
`levin_quadrature.py:1090`).

**The derivative-cancellation trap, as a measurement** (prompt §3 item 3): at `eta = 78730` in the
near-resonant case the true `dTheta/dlog eta` is **1.210000e-08** against a constituent scale
`max(k, q c_s, r c_s) eta = 1.000e+09`, a ratio of 1.2e-17. Relative error — the metric the test
refuses to use — is **0.000e+00** new and **1.000e+00** old. Nothing in the test divides by the
group derivative.

### All sixteen analytic Levin calls carry all four keys (prompt §3 item 1)

On `Case(0.0, SHAPES[0], 100.0, exact=True)` through `analytic_integral`, the spy captured **16**
calls, key sets `[('theta', 'theta_abserr', 'theta_deriv', 'theta_mod_2pi')]` — one distinct set,
all four keys, every call. `grep -c theta_abserr ComputeTargets/QuadSourceIntegral.py` = 3.

### The declared error reaches the caller (prompt §3 item 5)

Each of the same 16 calls run twice, once as shipped and once with `"theta_abserr"` removed from
the identical dict. Reported `abserr`, without → with:

| call | without | with | ratio |
|---|---|---|---|
| `analytic J4` (`nu_type="2pt5"`) | 1.057955e-18 | **2.496587e-17** | **×23.6** |
| the other fifteen | 3.19e-17 … 6.33e-16 | unchanged | ×1 |

So the declaration changes the reported number in the direction the prompt requires — larger — on
the group where the Levin rule's own estimate was smallest, which is the case
`levin_quadrature.py:2360` describes ("so the caller sees an honest number instead of an
artificially small one"). Fifteen of sixteen are unchanged because their quadrature estimate
already exceeds the declared phase error at this configuration's `rtol = 1e-8`.

### `analytic_rad` did not regress (prompt §3 item 4)

`TestAnalyticOracle` passes **at its existing thresholds, unmodified** (`EXACT_THRESHOLD = 1.0e-5`,
`REALISTIC_SMOOTH_THRESHOLD = 1.0e-3`, `REALISTIC_LEVIN_THRESHOLD = 5.0e-4`,
`REALISTIC_TOTAL_THRESHOLD = 1.5e-3`). `git diff -- ComputeTargets/tests/test_quadsource_integral.py`
shows **one** deleted line in the whole file — the stale docstring sentence of Deviation 7 — and no
changed threshold.

How much `analytic_rad` itself moved, measured by running `analytic_integral` over all eighteen
`TestAnalyticOracle` fixtures on the parent commit and on this one (`analytic_rad` is independent
of the exact/realistic flavour, so eighteen cases cover both):

| statistic | value |
|---|---|
| worst relative move | **1.019e-08**, on `b=0 together x_resp=980` |
| next three | 3.287e-11, 1.870e-11, 6.427e-11 (`b=0.2 together x_resp=980`, `b=0.2 together x_resp=100`, `b=0 together x_resp=100`) |
| twelve of eighteen | below 1e-12, three of them exactly 0 |
| declared `abserr` ratio, after/before | 0.80 to 23.40 |

**Attribution.** The moves are largest exactly where `|analytic_rad|` is smallest and the
four-group cancellation is therefore deepest: the worst case has `analytic_rad = 9.26e-13`, two to
five orders below every other fixture, so a 1e-8 *relative* move is an absolute move of 9.4e-21 —
below that case's own declared `abserr` of 3.8e-20. The three cases that did not move at all are
those where `_three_bessel_integrals` takes the pure-quadrature branch for at least one `nu_type`.
Every move is therefore inside the declared error bar of the quantity that moved, and three orders
inside `TestAnalyticOracle`'s 1e-5 exact threshold. The direction is the expected one: the group
phase is now accurate to `eps |K| eta` instead of `eps max(m_i) eta`, which changes the value most
where the cancellation is deepest.

### Test runs

| module | result | time |
|---|---|---|
| `ComputeTargets.tests.test_quadsource_integral.TestThreeBesselPhaseGroups` | **OK, 4 tests** | 0.7 s |
| `unittest discover -s ComputeTargets/tests -t .` | **OK, 101 tests** (97 before, + the 4 new) | 130.6 s |
| `LiouvilleGreen.tests.test_three_bessel` | **OK, 9 tests** (unchanged file) | 8.8 s |
| `unittest discover -s AdaptiveLevin/tests -t .` | **OK, 32 tests** | 0.07 s |
| `unittest discover -s LiouvilleGreen/tests -t .` | **OK, 133 tests** — the full run, completed | 1648.6 s (27.5 min) |

The full `LiouvilleGreen/tests` discovery run **was** completed, including `test_3bessel_analytic`,
whose figure-drawing sweeps are the 27 minutes (`[08-3bessel-plot-cost-dominates-the-suite]`).
`test_abserr_bounds_truth`, the `expectedFailure` that became an unexpected success in
`prompts/transfer-remedial`'s prompt 07 and was repaired by its prompt 08, passes as an ordinary
test on this tree; nothing in `LiouvilleGreen/` was changed here except the class name.

Acceptance-oracle headlines from the full `ComputeTargets` run, for comparison with the thresholds
quoted above: `[oracle 1, exact]` worst `total` vs `analytic_rad` **1.860e-08** against a 1e-5
threshold; `[oracle 1, realistic]` worst normalised deviation **4.912e-04** (smooth part 4.553e-04
against 1e-3, Levin part 1.303e-04 against 5e-4, total 4.912e-04 against 1.5e-3).

`black --check` clean on all four touched Python files (`QuadSourceIntegral.py`,
`test_quadsource_integral.py`, `three_bessel_integrals.py` — the last two reformatted by `black`
before committing).

The sibling module's own phase-group measurements are unchanged by the rename, as they must be:
`test_three_bessel` still prints `|delta Theta|` new 1.4211e-13 against old 7.6272e-05 at
`K/max ~ 1e-10` (ratio 5.367e8) and 1.5259e-05 for both routes at the generic `K = 0.1`.

### What was *not* verified

* **Anything above `x = 1e9`.** The phase-group measurement is taken over the range `main.py`
  builds these splines on; the floor model, not the measured constants, is what extrapolates.
  `docs/OPEN_ISSUES.md` §5's standing caveat applies unchanged.
* **The production `WKB_Levin` path's phase groups.** Untouched by this commit
  (`ComputeTargets/phase_groups.py` is forbidden), and Deviation 1 is the consequence.
* **Ray serialization of a `BesselPhaseGroup`.** It is constructed inside `_three_bessel_Levin`
  and never crosses a process boundary; what crosses is the `bessel_phase` dict through
  `BesselPhaseProxy`, which `prompts/transfer-remedial`'s prompt 06 verified and which this commit
  does not change.
* **`b` values other than 0 and 0.2** in the integral, and other than 0 in the phase-group
  measurement. The two `B_VALUES` are what the existing fixtures use.

## Observations not acted on

1. **`_check_bessel_order` could read `phase_data["nu"]` directly.** The prompt forbids it and the
   board lists it as out of scope, both because it is a behaviour change. Recorded here as the
   option it is: the direct comparison is one line and catches the mis-built-spline case earlier,
   but it trusts a declared field over the object's own reconstruction — it would accept a phase
   object whose `"nu"` disagreed with its splines, which is the failure mode the guard was written
   for, and reject a correctly reconstructing object built by some other route. A belt-and-braces
   version (assert `phase_data["nu"] == nu` *and* keep the numeric check) is the only variant I
   would argue for, and it is still a behaviour change.
2. **`Case.bessel_phase_data` still passes the deprecated `atol`/`rtol` to `bessel_phase`**
   (`test_quadsource_integral.py:482-483`), emitting two `DeprecationWarning`s per fixture — the
   same call-site staleness `prompts/transfer-remedial`'s prompt 06 recorded for
   `test_3bessel_analytic.py`. Not this prompt's; it would change what the fixtures measure.
3. **`CHEBYSHEV_ORDER = 24`'s comment is now questionable in the same way
   `DEFAULT_3BESSEL_CHEBYSHEV_ORDER`'s is.** It justifies 24 by a self-consistency sweep run
   against a synthetic problem built "with the same domain, amplitude form and phase contract
   (`theta + theta_mod_2pi`, **no `theta_deriv`**) as this file's own nine call sites". Eight of
   those nine now supply a `theta_deriv` *and* a `theta_abserr`, so the synthetic problem no longer
   matches the contract it was built to imitate. Explicitly out of scope
   (`[08-3bessel-chebyshev-order-is-now-the-limit]`, README §2), and no measurement here suggests
   24 is wrong — only that its stated justification has drifted from the code.
4. **`levin_quadrature._sample_vectorized()`'s docstring names
   `ComputeTargets/QuadSourceIntegral.py` among the callables that "do not [vectorize] … their
   phase and modulus splines branch on a scalar argument".** For these eight calls the phase side
   is now `BesselPhaseGroup`'s array-safe accessors, so the statement is stale here as well as in
   `three_bessel_integrals.py` — `[06-levin-theta-docstring-stale]` already carries the extension
   for the sibling module. `AdaptiveLevin/` is forbidden here; not opened as a second issue.
   (The `f` vectors in this function are still scalar-only: `Levin_f` uses `math.exp`/`math.pow`.
   That is unchanged by this commit.)

## State handed to the next prompt

This campaign has one prompt, so this section is for whoever picks up the file next.

### The interface now used at both three-Bessel call sites

```python
from LiouvilleGreen.three_bessel_integrals import BesselPhaseGroup   # _PhaseGroup is an alias

group = BesselPhaseGroup(phases, coefficients, signs)  # signs[0] is always +1
group.K, group.C, group.C_reduced   # floats, formed once by math.fsum at construction
group.theta(log_x)                  # K x + C + R(x)
group.sin_cos(log_x)                # (sin, cos) by angle addition on (K x), (C_reduced + R)
group.theta_mod_2pi(log_x)          # atan2 of that pair, in (-pi, pi]
group.theta_deriv(log_x)            # K x + sum_i e_i dr_i/dlog x
group.theta_abserr(log_x)           # linear sum of the constituents' theta_abserr_at
group.levin_theta()                 # the four-key dict for adaptive_levin_sincos
```

Every accessor takes the **logarithm of the shared variable** — `log(eta)` at this call site,
`log(x)` in `three_bessel_integrals.py`. All four callables in `levin_theta()` are array-safe and
bit-identical between the array and scalar paths; keep it that way, or
`levin_quadrature._detect_vectorized()` silently falls back to a Python loop
(`prompts/transfer-remedial/logs/07-bessel-phase-groups.md` deviation 3).

### What is still open in this file

1. **`[01-cosmological-group-declares-no-phase-error]`** — the ninth `adaptive_levin_sincos` call,
   `phase_group_Levin_integral`'s, supplies three keys. Root cause: `phase_spline` reports no fit
   accuracy, so `ComputeTargets/phase_groups.PhaseGroup` has nothing to declare. That is a
   `phase_spline` change before it is a `phase_groups` change, and it belongs with the hand-over
   campaign's representation work (`docs/OPEN_ISSUES.md` §1.1), not with a second phase-groups
   prompt. **This is the one acceptance sentence this campaign does not satisfy.**
2. **`CHEBYSHEV_ORDER = 24` was tuned against a phase contract that no longer holds** for eight of
   the nine call sites (observation 3). Whoever reopens
   `[08-3bessel-chebyshev-order-is-now-the-limit]` should re-run that sweep with the four-key
   contract rather than bump the constant.
3. **The measurement above stops at `x = 1e9`.** That is where `main.py` builds these splines, but
   `docs/OPEN_ISSUES.md` §5's standing caveat applies: the production `zend = 0.1` reaches
   `x ~ 1.4e7` in accumulated phase for the largest `k`, and nothing here was measured above 1e9.
   The floor model `eps (|K| eta + |C + R|)` is what extrapolates, not the measured constants.

### Numbers a later reader will want

* Group phase at exact resonance, this call site's orders and coefficients: **5.1473e-08 → 1.3758e-12** rad.
* Group log-derivative at exact resonance: **2.2824e-08 → 9.6402e-12** per unit `log eta`.
* Generic group (`|K|/max = 0.27`): phase 1.1921e-07 → 7.4506e-09 (a floor ratio, not a gain);
  derivative 2.9802e-08 → 2.9802e-08.
* Worst measured error / declared bound: **0.49** in both cases.
* `analytic_rad` moved by at most **1.019e-08** relative, on the fixture whose value is 9.26e-13;
  in absolute terms 9.4e-21, inside that fixture's own declared `abserr` of 3.8e-20.
* Declaring `theta_abserr` raised one of sixteen reported `abserr`s by **×23.6** and left fifteen
  unchanged.
