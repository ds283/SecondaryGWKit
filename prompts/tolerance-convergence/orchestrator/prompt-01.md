# Orchestrator — prompt 01, the convergence harness and one production grid

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../01-convergence-harness-and-grid.md`](../01-convergence-harness-and-grid.md)
**Model:** Opus. **Production code changed:** none.

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

---

## 1. Before you dispatch — take these baselines

They cannot be reconstructed after the commit lands, and two of the three reviews below are diffs
against them.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . > /tmp/tc-01-before-ct.log 2>&1; grep -E '^(Ran|OK|FAILED)' /tmp/tc-01-before-ct.log
```

Expect **452, OK**. Also record `CosmologyModels` (expect **39, OK**) and `git rev-parse --short HEAD`.

**The grid baseline.** Prompt 01's central claim is that hoisting the version-2 grid does not change
it. Capture what it produces *before* the hoist, so you can check the claim rather than take it:

```bash
PYTHONPATH=. ./venv/bin/python -c "
from ComputeTargets.tests.test_source_grid import _production_grid, _production_base_grid
print('base grid samples:', len(_production_base_grid()))
"
```

If that import fails because the names are private or the module does work at import time, **do not
work around it** — note what happened and rely on §3's bit-identity check instead, which the agent
must perform and quote. Record the sample counts the prompt expects to see (**1,996** on QCD,
**1,778** on LambdaCDM) so you can compare.

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `01-convergence-harness-and-grid`, model
**Opus**.

## 3. The review — five checks, in this order

**Check 1 is the campaign's construction check and it outranks the rest.** If it fails, stop; do
not look at the others first and do not let a green suite reassure you.

1. **Did prompt 17's published figures survive the fold?** The log's Verification section must
   carry, measured on the **named version-0 grid**:

   | model | worst reference drift | at $k$ | median drift |
   |---|---|---|---|
   | `RadiationModel` | 4.21e-11 | 3e+08 | 1.92e-11 |
   | `LambdaCDMModel` | 5.7e-11 | 1.561e+08 | 3.76e-11 |

   These must reproduce **to the digits published**. They are single-segment runs, so the numeric-ODE
   split of `GkTk-remedial` prompts 18–20 does not touch them, and `TK-NUMERIC-ATOL-SWEEP.md` §9.1
   re-published them unchanged after that split — there is no legitimate reason for them to move.

   **`QCDModel` is not a reproduction target.** Its published values are 6.17e-06 (§4, unsplit) and
   1.97e-07 (§9.1, split), and the board quotes 7.08e-09 on the current tree. The agent must
   *report* what it measures beside all three and say which tree each came from. **An agent that
   reports a QCD figure matching a published one exactly has probably tuned something** — ask how.

2. **Is the hoist bit-identical?** The log must quote, for both models, the sample count and
   `redshift_grid_digest` of the hoisted version-2 grid against `_production_grid`'s before the
   move, and the same for the named v0 and v1 against their originals. Compare the v2 counts with
   §1's 1,996 / 1,778. **"The tests still pass" is not this check** — `test_source_grid.py`'s
   assertions were written against the construction being moved, so they pass either way.

3. **Does the facility serve prompt 04?** Read the public API in "State handed to the next prompt".
   A facility that can only step a tolerance by a decade cannot express "one order tighter" for
   `TAU_GAUSS_ORDER`, and prompt 04 audits four integer orders with no tolerance to move. If the
   step is hard-coded as a float multiplication, that is a stop — prompt 04 would have to rewrite
   it, and half the campaign runs through it.

4. **The suites and the diff.**
   - `ComputeTargets`: 452 → 452 + *n*, OK. *n* is the new test module's tests; the log must say
     what *n* is. A **fall** is a stop.
   - `CosmologyModels`: **39**, unchanged. Prompt 01 touches nothing in that package.
   - `git diff --name-only HEAD~1 HEAD` must lie inside the prompt's allowed list: the two new
     modules, `wkb_reference.py`, `test_source_grid.py`, `test_background_segmentation.py`,
     `tk_numeric_atol_sweep.py`, the log, the board, `docs/OPEN_ISSUES.md`. **Zero production
     files.**
   - `black --check` clean on every `.py` touched.

5. **The bookkeeping.** Board row 01, item rows **T1** and **T2**, the narrowing of
   `[00-three-production-grid-reproductions]` in board §3, and `docs/OPEN_ISSUES.md` **in the same
   commit** with its count and date correct (`CLAUDE.md`). Every deviation classified.

## 4. What a good outcome looks like

- §4.1's two rows reproduce exactly; the QCD row is reported with all three published values beside
  it and an honest statement that it moved and why it was expected to.
- The version-2 figures sit beside the version-0 ones in the log, and the difference is *recorded*
  rather than explained away. That table is the first measurement in this campaign's record to
  carry its grid generation under §5 rule 6.
- Three named grid generations, none of them reachable by an unqualified name.
- `PARTIAL` with Part A complete and Part B handed over is an **acceptable** result and the prompt
  says so. Prefer it to a half-finished hoist.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **§4.1's two rows do not reproduce.** Report both measured values beside both published ones.
  Do not authorise "adjusting the facility until it matches" — that is the tuning this check exists
  to catch.
- **A bit-identity check fails**, or the agent argues that a difference is acceptable because it is
  small. A hoist that changes the grid is not a hoist, and "small" has no meaning here: the two
  constructions are either the same arithmetic or they are not.
- **The agent proposes to touch `wkb_reference_data.json`.** That is prompt 04's, under §7 D5.
- **The agent proposes any production change**, including adding a fallback to
  `cosmology_feature_redshifts` so that a simpler stand-in works. That fallback is exactly what
  `background-solver-robustness` prompt 09 removed on purpose.
- **The agent reports that `main.py`'s grid construction has moved again** since the re-anchor.
- **The `ComputeTargets` suite passes ~240 s.** Not a failure; a budget question, and this campaign
  has five more prompts to add to it.
