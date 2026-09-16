# Log 08 — Refresh the stale agreement threshold in `test_wPerturbations.py`

**Prompt:** prompts/background-solver-robustness/08-refresh-agreement-threshold.md
**Commit:** *(this commit)* — Refresh the stale agreement threshold in test_wPerturbations.py
**Model:** Claude Sonnet 5
**Date:** 2026-09-16
**Result:** COMPLETE

## What shipped

**One file touched, and only the module comment plus the constant it justifies.**

- `CosmologyModels/tests/test_wPerturbations.py:35-51` (was `:35-42`) — the comment above
  `AGREEMENT_RTOL` rewritten to describe the representation that has been in the tree since
  `qcd-background-audit` prompts 05 and 06: a segmented entropy-factor interpolant (the $(1+z)$
  ramp closed form, only the entropy factor $F$ tabulated, on `DEFAULT_T_Z_SPLINE_SAMPLES = 3,000`
  nodes of `DEFAULT_T_Z_SPLINE_ORDER = 5`, segmented at the equation of state's branch
  temperatures), rather than the "500-point spline in T" it used to describe. The comment names
  this prompt and the commit the figures were re-taken at, states both re-measured numbers, and
  keeps the sentence explaining what the constant is *for* (the representation, not the physics,
  is the floor; the threshold is still many orders below the 0.69 relative discrepancy the A1
  defect produced).
- `CosmologyModels/tests/test_wPerturbations.py:52` — `AGREEMENT_RTOL` tightened from `1.0e-8` to
  **`1.0e-14`**.
- No other line of this file changed. `git diff` (quoted in full below) touches only the comment
  block and the one constant; every assertion body, the `PureRadiationEOS` stand-in,
  `lambdaCDM_gstar`, `AGREEMENT_MAX_Z` and `HIGH_Z_MAX_Z` are untouched.
- `IMPLEMENTATION_STATE.md` — board row 08 marked complete; item table unchanged (this prompt
  closes an issue, it does not add a lettered item); §3's
  `[01-agreement-threshold-comment-predates-the-representation]` entry deleted and a resolved
  entry added to §4.
- `docs/OPEN_ISSUES.md` — the `[01-agreement-threshold-comment-predates-the-representation]` row
  deleted from §1.8 (the issue is closed, and `CLAUDE.md`'s rule is "delete its row, do not keep a
  resolved section here"); count and date corrected.

`T_Z_REPRESENTATION_VERSION` is **6** before and **6** after
(`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:428`, untouched — this prompt may not touch
production code at all, and did not).

## The two equality redshifts

Not applicable to this prompt in the sense workstreams A and B mean it — this prompt is workstream
D and never calls `_find_rho_equality` or reads its result. For completeness, and because
`test_wPerturbations.py`'s own `setUpClass` prints them as a side effect of construction: both
banner lines are unchanged by this prompt (it touches no production code), matching the values
recorded at prompt 09's commit — `QCD_Cosmology` is not built by this test module at all; the two
`LambdaCDM_GenericEOS` instances it constructs (the pure-radiation stand-in) print
`matter-radiation equality at z = 3403` and `matter-Lambda equality at z = 0.3034` at `:.4g`,
identical before and after this commit.

## Deviations from the prompt

None.

The prompt left the exact tightened value and its margin as a judgement call (§2 item 3, "tighten
`AGREEMENT_RTOL` to what the measurement supports, with margin. State the margin and why you
chose it"). That is not a deviation — the prompt asked for a choice, not a specific number — but
it is recorded here as the one open decision this prompt made:

**The tightened value, and why.** Measured worst departure is `8.8818e-16` (4 ulp of 1.0) at
`max_z = 1e4`, the value the assertion actually runs at. `AGREEMENT_RTOL = 1.0e-14` was chosen:
about eleven times the measured worst case, in the same style as the comment it replaces (the
original `1e-8` carried a margin of about 7.7× over its own claimed `1.3e-9`). A tighter round
number (`1e-15`) would leave less than a factor of two above the measured value and no headroom
for a few ulp of run-to-run libm/BLAS variation; a looser one (`1e-12` or `1e-13`) would leave
three to four orders of unused margin given the measurement is stable to the same order of
magnitude at both `max_z` values. `1.0e-14` was the smallest round number clearing the measured
worst case by an order of magnitude.

## Verification performed

Everything below was **run**.

### 1. Re-measurement (prompt §2 item 1)

Reproduction script (constructs the same models `test_wPerturbations.setUpClass` does, at
`max_z = 1e4` and `max_z = 1e20`, over `test_agrees_with_LambdaCDM`'s own probe set):

```bash
PYTHONPATH=. ./venv/bin/python <<'EOF'
from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import LambdaCDM_GenericEOS
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units
from CosmologyModels.tests.test_wPerturbations import PureRadiationEOS, lambdaCDM_gstar

units = Mpc_units(); params = Planck2018(); g = lambdaCDM_gstar(params.Neff)
lcdm = LambdaCDM(store_id=1, units=units, params=params)
for max_z in (1.0e4, 1.0e20):
    generic = LambdaCDM_GenericEOS(store_id=2, eos=PureRadiationEOS(units, g),
                                    units=units, params=params, max_z=max_z)
    worst = max(abs(generic.wPerturbations(z) / lcdm.wPerturbations(z) - 1.0)
                for z in [0.0, 0.5, 1.0, 2.0, 10.0, 1.0e3])
    print(max_z, worst)
EOF
```

Output, at this commit's parent (`662bba5`, unchanged by this prompt since it is read-only
measurement):

| `max_z` | Comment claimed | **Measured, worst over the probe set** | At | ulp of 1.0 |
|---|---|---|---|---|
| `1e4` (`AGREEMENT_MAX_Z`) | ~1.3e-9 | **8.8818e-16** | $z = 1$ | 4 |
| `1e20` (the default) | ~4e-7 | **6.6613e-16** | $z = 0.5$ | 3 |

Per-probe at `max_z = 1e4`: `2.2204e-16, 3.3307e-16, 8.8818e-16, 5.5511e-16, 4.4409e-16, 0.0` (at
$z = 10^3$ the two agree bit for bit). At `max_z = 1e20`:
`2.2204e-16, 6.6613e-16, 6.6613e-16, 4.4409e-16, 5.5511e-16, 6.6613e-16`.

**These are identical to the digit to `logs/01-equality-solve-characterisation.md` §4.3**, which
measured the same quantity at `3e820eb`. Re-taking it rather than inheriting it (as the prompt
requires) confirms nothing moved between that commit and this one.

### 2. The regime (prompt §2 item 4)

**This is the double-precision rounding floor, not an interpolation error.** Three pieces of
evidence, all above: (i) the worst departure is the *same order of magnitude* at `max_z = 1e4` and
`max_z = 1e20` — 4 ulp and 3 ulp of 1.0 respectively — where the old comment claimed a factor of
~300 difference (`1.3e-9` vs `4e-7`) between the same two cases, which is what an interpolation
error that scales with the tabulated range would look like and a rounding floor would not; (ii) on
`PureRadiationEOS`, $g_* = g_{S,*}$ is constant by construction (the class's own docstring), so the
entropy factor $F$ the representation now tabulates is identically constant, and an order-5
B-spline through 3,000 equal node values has nothing to interpolate — any residual is arithmetic
noise in the chain of floating-point operations from node evaluation through to `wPerturbations`,
not a representation error; (iii) the worst-case figures themselves, 3–4 ulp of 1.0, are the size
of noise accumulated over a handful of floating-point operations, not the size of any interpolation
error this project has measured elsewhere (`qcd-background-audit` measured genuine segmented-spline
interpolation error at $10^{-12}$–$10^{-9}$ depending on order and node count). **The shape of the
comment therefore changed, not just its numbers**, per the prompt's own instruction.

### 3. The tightened constant

```bash
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_wPerturbations -v
```
→ `Ran 5 tests in 0.083s / OK`. `test_agrees_with_LambdaCDM` (the one assertion that reads
`AGREEMENT_RTOL`) passes at `delta=1.0e-14` with six subtests, all six of them the measurements in
§1 above, all comfortably inside the tightened delta.

### 4. `git diff` — scope of the change

```
$ git diff --stat
 CosmologyModels/tests/test_wPerturbations.py | 26 ++++++++++++++++++--------
 1 file changed, 18 insertions(+), 8 deletions(-)
```

One file, one hunk: the fourteen-line comment (was seven lines) and the one constant. No assertion
body, no import, no other constant.

### 5. Suites

| Suite | Before (`662bba5`) | After |
|---|---|---|
| `CosmologyModels/tests` | **39**, OK, 0.682 s | **39**, OK, 0.689 s |
| `ComputeTargets/tests` | **452**, OK | **452**, OK (re-run; this prompt touches no file that
  suite imports) |

Both counts unchanged, as the prompt requires (this prompt adds no test and touches no file
`ComputeTargets` depends on).

### 6. Formatting

```
./venv/bin/python -m black CosmologyModels/tests/test_wPerturbations.py
./venv/bin/python -m black --check CosmologyModels/tests/test_wPerturbations.py
```
→ reformatted once (the new comment's line wrapping), then clean under `--check`.

## Observations not acted on

None beyond what prompt 01 already recorded and this prompt closes. Nothing new was found while
re-measuring.

## State handed to the next prompt

This is the last prompt on the board (workstream D, prompt 08 of 2). Nothing depends on it.

- `AGREEMENT_RTOL = 1.0e-14`, `AGREEMENT_MAX_Z = 1.0e4` (unchanged), the re-measured worst-case
  figures (`8.8818e-16` at `max_z=1e4`, `6.6613e-16` at `max_z=1e20`), and the reproduction script
  in §1 above are the reference for any later prompt that touches this comment again.
- `docs/OPEN_ISSUES.md` §1.8 no longer carries a row for
  `[01-agreement-threshold-comment-predates-the-representation]`; it is recorded resolved on this
  board's §4.
- Suite counts to carry forward: `CosmologyModels` **39**, `ComputeTargets` **452**.
  `T_Z_REPRESENTATION_VERSION` = **6**.

## Reproduction commands

```bash
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_wPerturbations -v
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
./venv/bin/python -m black --check CosmologyModels/tests/test_wPerturbations.py
```
