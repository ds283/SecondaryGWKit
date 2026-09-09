# Orchestrator prompt — Workstream C (prompts 06, 07)

You are orchestrating Workstream C of the transfer-function remedial campaign — its Bessel
amplitude-and-phase phase (README §0) — in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `transfer-remedial-plan`). You do not write code
yourself. You dispatch one fresh-context subagent per prompt, review what it produced against fixed
criteria, and either continue or stop and report to the user.

Workstream C makes the improvement reach its consumers. A better construction is useless if
evaluation immediately rounds its correction away, or if it reports an artificially small error to
the quadrature that consumes it, or if a consumer reintroduces the cancellation after the individual
Bessel functions have been fixed.

Mostly plumbing — but it contains **three judgement calls**, and two of them are stop conditions by
design. Your job on those is to surface the decision to the user with the agent's reasoning, not to
ratify it.

| Call | Prompt | Your action |
|---|---|---|
| What replaces `Q` (the pre-offset ODE state, which no longer exists) | 06 §2.2 | Report the decision and the alternatives weighed. Continue if the agent removed it cleanly with a `KeyError` naming the replacement; **stop** if it kept the key mapped to something that is not \(\theta/x\) |
| Repair or delete `plot_besssel_phase.py` (already broken) | 06 §2.4 | **Stop if the agent chose to delete.** That is the user's call |
| Which derivative route the phase groups use | 07 §2 item 3 | Report which won and by how much. Continue either way if it was *measured* |

## What to read

Read these in full before dispatching:

- `prompts/transfer-remedial/README.md` — **§0.1 (the layer split against
  `prompts/source-remediation`; this workstream is the one that comes closest to its files)**,
  §1.1 and **§4.2 (the scope boundary against the in-flight
  `source-remediation` campaign, and the `main.py` hunk discipline)**, §4.1, §4.3, §5, §5.1, §6.
- `prompts/transfer-remedial/RECONCILIATION.md` — **C3 (`plot_besssel_phase.py` is already broken),
  §1 (the Levin contract facts), §3.2 (why `QuadSourceIntegral.py` is excluded) and §3.3 (the
  complete consumer inventory)**.
- `prompts/transfer-remedial/IMPLEMENTATION_STATE.md` — the board, §3 (both open issues are relevant
  here), §5 notes 5, 6 and 11.
- `prompts/transfer-remedial/orchestrator/README.md` — the campaign-wide stop conditions.
- `logs/05-two-region-construction.md` — the object surface prompt 06 adapts. If its "State handed
  to the next prompt" does not give verbatim accessor signatures, **stop**: prompt 06 cannot write
  an adapter against a paraphrase.

In the code, read `AdaptiveLevin/levin_quadrature.py:930-975`, `:1030-1095` and `:2350-2370` so you
can check prompt 06's `theta_abserr` work without taking its word for the contract.

Read `06-evaluation-and-compatibility.md` only when about to dispatch it, and likewise 07. **Do not
read prompts 08–09.**

## Preconditions

`git status` clean; board rows 01–05 ✅ or ⚠️, rows 06 and 07 ⬜. Confirm the tree is where
Workstream B left it:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase LiouvilleGreen.tests.test_bessel_two_region -v
PYTHONPATH=. ./venv/bin/python -c "import main" && echo "main imports"
PYTHONPATH=. ./venv/bin/python -c "import ComputeTargets.QuadSourceIntegral_debug" && echo "debug imports"
```

The last two are the pre-edit baseline for prompt 06 — you need to know they worked *before* it
touched them.

**Check `source-remediation`'s state.** As of `95cc326` it is at **11 of 12** — its `main.py` edits
(prompts 04, 06, 10) have landed, so the stage boundaries are settled and prompt 06's hunks are
disjoint from them. Confirm this is still true (`grep -n "Progress:"
prompts/source-remediation/IMPLEMENTATION_STATE.md`). If its **prompt 12 (verification)** is still
⬜, say so in your report: it runs the pipeline, so running it after this campaign has changed the
Bessel oracle means it verifies a moved tree (`README.md` §4.2). That is the user's call, not yours.
If any of its prompts is dispatched concurrently, commits must be serialised by rebase, and **a
rebase conflict is a stop condition, not something to resolve**.

## Dispatching

For prompt NN, launch a subagent with **exactly** this context:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/transfer-remedial/README.md`, `prompts/transfer-remedial/RECONCILIATION.md`,
> `prompts/transfer-remedial/IMPLEMENTATION_STATE.md`, then your prompt
> `prompts/transfer-remedial/NN-<name>.md` and the `DRAFT-PLAN.md` sections it cites. Execute the
> prompt exactly. Do not read any other prompt under `prompts/transfer-remedial/`. You **may** read
> the "State handed to the next prompt" sections of the logs your prompt names, and only those.
> Follow README §5 for the commit, the log and the board update. When you finish, reply with: the
> commit SHA, the **Result** line from your log, the "State handed to the next prompt" section
> verbatim, and a list of every deviation with its classification tag.

Model per prompt: **06 → Opus**, **07 → Opus**. Do not substitute.

Run 06 → 07. Not independent: 07's phase groups consume 06's residual accessors, and
`theta_abserr` must exist before 07 can pass it through.

## Reviewing prompt 06 — the adapter

Structural checks 1–3 as in `README.md` §4.3. Then:

4. **Tests pass when you run them:**
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -10
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -10
   PYTHONPATH=. ./venv/bin/python -c "import main" && echo OK
   PYTHONPATH=. ./venv/bin/python -c "import ComputeTargets.QuadSourceIntegral_debug" && echo OK
   ```
   The `ComputeTargets/tests` run is the canary: `test_tk_source_functions.py` and
   `test_phase_groups.py` consume `mod` and `raw_theta`, so an adapter slip shows up there before
   anywhere else.

   Also, if the agent repaired `plot_besssel_phase.py`, run it and confirm it produces figures:
   `PYTHONPATH=. ./venv/bin/python plot_besssel_phase.py`.

5. **Allowed files, and the `main.py` hunk discipline.**
   ```bash
   git diff HEAD~1 --stat
   git diff HEAD~1 -- main.py
   ```
   `main.py` hunks **only** in the Bessel construction stage (around `main.py:516-528`
   pre-commit). A hunk in the Tk, QuadSource or QuadSourceIntegral stages fails this check outright.
   Verify **untouched**: `ComputeTargets/QuadSourceIntegral.py`, `ComputeTargets/QuadSource.py`,
   `ComputeTargets/phase_groups.py`, `ComputeTargets/TkSourceFunctions.py`, everything under
   `Datastore/` and `AdaptiveLevin/`, `LiouvilleGreen/three_bessel_integrals.py`,
   `LiouvilleGreen/tests/test_bessel_phase.py`, `LiouvilleGreen/tests/test_3bessel_analytic.py`.

6. **`theta_abserr` is declared and honest.** Read the test. It must assert the declared value is
   **≥** the measured \(E_\theta\) at the same \(x\), across both regions. Then read the log's
   ratio: an estimator that under-reports defeats the entire purpose
   (`levin_quadrature.py:2360`: "so the caller sees an honest number instead of an artificially
   small one"). Under-reporting anywhere is a stop.

7. **The real Levin call test exists** (prompt 06 §5 item 3) and asserts three things: convergence;
   that the reported `abserr` is now dominated by the declared phase error rather than implausibly
   small; and that `need_theta_Cheb` did not fire — most simply by showing the result is unchanged
   with a deliberately broken `theta` callable. That last assertion is the campaign's only direct
   evidence for `RECONCILIATION.md` §1's reading of `levin_quadrature.py:1038`; if it is missing,
   ask for it.

   Read the log's before/after `abserr` numbers. This is the concrete demonstration that the
   campaign's accuracy claim reaches the consumer, and it belongs in your report.

8. **The Ray round trip is bit-identical.** Read the test: `ray.cloudpickle` (or `ray.put`/`get`)
   round trip, then every accessor compared on a grid, **including** `theta_abserr` if it is a
   callable. A closure over a local that survives cloudpickle but not plain pickle is the usual
   hazard; if anything needed a workaround *in the test* rather than a design fix (a module-level
   carrier class), that is a stop.

9. **`Q` and `phi`.** `phi` must be exactly `0.0` for every \(\nu>1/2\), or documented as removed.
   `Q` must be gone cleanly — a `KeyError` naming the replacement. **If the key `Q` still exists and
   maps to anything that is not \(\theta/x\), stop**: `DRAFT-PLAN.md` §8.1 forbids silently
   replacing it "under the same undocumented meaning", and that is the specific harm.

10. **`XSplineWrapper` still imports:**
    `PYTHONPATH=. ./venv/bin/python -c "from LiouvilleGreen.bessel_phase import XSplineWrapper"`.
    `test_three_bessel.py:10` needs it and is not prompt 06's to edit
    (`IMPLEMENTATION_STATE.md` §5 note 6).

11. **The deprecation shim is documented and tested**, and **`main.py` does not rely on it** —
    prompt 06 §3 says to update the caller, not lean on the shim. Read the `main.py` diff and
    confirm it passes the new explicit accuracy arguments, and that the old comment about "keeping
    \(Q\) very accurately close to 1" and the Dormand–Prince stepper is gone. That comment is now
    wrong in every particular; leaving it is `UNINTENDED DRIFT`.

12. **The stale `AdaptiveLevin` docstring** at `levin_quadrature.py:2750` ("theta is always used to
    decide subdivision") must appear in the log's "Observations not acted on", **not** in a diff.
    `AdaptiveLevin/` is prohibited.

## Reviewing prompt 07 — the phase groups

Structural checks, plus:

4. **Tests pass when you run them:**
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_three_bessel -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -10
   ```
   `test_3bessel_analytic.py` must pass **at its existing tolerances** (`ABS_TOLERANCE = 1e-6`,
   `REL_TOLERANCE = 1e-5`, `1e-3`/`1e-2` near singularities, `test_3bessel_analytic.py:15-19`).
   Confirm with `git diff HEAD~1 -- LiouvilleGreen/tests/test_3bessel_analytic.py` returning empty:
   those tolerances are prompt 08's, not 07's.

5. **Allowed files only:** `LiouvilleGreen/three_bessel_integrals.py`,
   `LiouvilleGreen/tests/test_three_bessel.py`, the log, the board. `bessel_phase.py`,
   `AdaptiveLevin/` and everything under `ComputeTargets/` untouched.

6. **The improvement was measured, not merely argued.** Prompt 07 §4 requires before/after numbers
   on the same triples for exact-zero, \(10^{-6}\), \(10^{-10}\) and generic \(K\). **A restructure
   justified only by an argument does not meet this prompt.** If the log has no before/after table,
   stop.

7. **Every `adaptive_levin_sincos` call passes `theta_abserr`:**
   ```bash
   grep -n "adaptive_levin_sincos" -A 8 LiouvilleGreen/three_bessel_integrals.py | grep -c "theta_abserr"
   ```
   should match the number of call sites. Combined **linearly**, not in quadrature — the three
   phases share a construction, so an inaccurate phase drifts them together
   (`BesselIntegralResult`'s docstring already argues this for the group values).

8. **`BesselIntegralResult`'s docstring was updated.** As shipped before this prompt it says the
   phase fit error "is invisible from inside this module", quotes a ~2e-8 floor, and says "nothing
   in `LiouvilleGreen/` supplies one yet" (`:24-33`). All three are now false. Confirm the paragraph
   was rewritten and that the correct reasoning about linear combination survived.

9. **The derivative-cancellation trap was avoided.** Read the tests: \(\delta\theta'_{\rm group}\)
   must be scored against \(\max(k,q,s)\), **never divided by the group derivative**, which passes
   through zero at exact resonance (README §6). A relative-error assertion on the group derivative
   is a bug in the test even if it currently passes.

10. **The "quadrature error is not a certificate" test exists** (prompt 07 §3 item 5): refine the
    quadrature tolerance and assert the result *stops improving* at a floor consistent with the
    declared `theta_abserr`. This is the positive form of `DRAFT-PLAN.md` §9 Stage 4's warning; an
    "integral matches to 1e-12 therefore the phase is good" test is the error it exists to prevent.

11. **The bounded-angle path is built from the split**, not from summing three bounded angles
    (prompt 07 §2.1). Read the code. Summing three `theta_mod_2pi` values is what the module did
    before and is what this prompt replaces.

## Continue or stop

**Continue** when all checks pass and the Result is `COMPLETE`, or `COMPLETE WITH DEVIATIONS` where
every deviation is tagged `IMPLEMENTATION CHOICE` with reasoning stated.

**Stop and report** on any campaign-wide stop condition or:

- **06:** the agent chose to **delete** `plot_besssel_phase.py` (§2.4 — the user's call); the `Q`
  key survives mapped to something other than \(\theta/x\); `theta_abserr` under-reports against
  the measured error; the Ray round trip needed a test workaround rather than a design fix; a
  `main.py` hunk outside the Bessel stage; `AdaptiveLevin/` or any prohibited path touched;
  `XSplineWrapper` no longer importable; or a `source-remediation` rebase conflict.
- **07:** no before/after measurement; a `adaptive_levin_sincos` call without `theta_abserr`; a
  relative-error assertion on a group derivative that can vanish;
  `test_3bessel_analytic.py` modified or failing; `BesselIntegralResult`'s docstring still claiming
  nothing supplies `theta_abserr`.

**Report but do not stop:** if prompt 07's log hands over a candidate limiting factor for
`test_3bessel_analytic.py`'s tolerances — `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` is the likely one,
chosen when "accuracy is set by the phase and modulus splines, not by the spectral order"
(`three_bessel_integrals.py:53-57`), a statement this campaign has just invalidated — put it in
your report's first paragraph. Prompt 08 needs it, and it is **not** this campaign's to fix.

## Completion criterion

Rows 06 and 07 ✅ or ⚠️; both logs `COMPLETE`-class; `main.py` imports and its Bessel stage uses the
new accuracy arguments; the full `LiouvilleGreen/tests` and `ComputeTargets/tests` runs pass.

Then report: "Workstream C complete; the tree is at `<SHA>`. Production now runs on the new
construction and the phase groups preserve their leading term. This is a usable stopping point
(`README.md` §4.1). Ready for Workstream D (prompt 08)." Include: the three judgement-call
decisions with the agents' reasoning; the Levin `abserr` before and after declaring `theta_abserr`;
the phase-group before/after table; and any handed-over candidate for prompt 08.
