# Orchestrator prompt — Workstream D (prompts 08, 09)

You are orchestrating Workstream D, the close-out of the Bessel amplitude-and-phase campaign in the
repository at `/Users/ds283/Documents/Code/SecondaryGWKit` (branch `bessel-remedial-plan`). You do not
write code yourself. You dispatch one fresh-context subagent per prompt, review what it produced
against fixed criteria, and either continue or stop and report to the user.

Workstream D turns the campaign's numerical claims into assertions, re-reads the downstream fixtures
whose *interpretation* the improvement changes, re-runs the capped benchmark tier, and records the
outcome in `docs/`. Neither prompt changes production code.

**The characteristic risk here is not breakage; it is over-claiming.** Prompt 08's whole purpose is
to separate the Bessel-oracle improvement from three floors this campaign did **not** move — the
consumer re-spline error, the physical LG truncation, and the quadrature — and a tightened tolerance
whose real limit is one of those is a false claim in the test suite. Prompt 09's is to record an
honest performance story rather than the one `DRAFT-PLAN.md` §4.7 tells.

## What to read

Read in full before dispatching:

- `prompts/bessel-remedial/README.md` — §1.1, **§4.2 (the `ComputeTargets/tests/` hunk
  discipline)**, §4.3, §5, §5.1, **§6**, §7 (deferred work).
- `prompts/bessel-remedial/RECONCILIATION.md` — **C1 (the honest performance story) and §3.2 (the
  hand-off)**.
- `prompts/bessel-remedial/IMPLEMENTATION_STATE.md` — the board, §3 (both planning-pass issues close
  or hand over here), §5 notes 3, 4 and 12.
- `prompts/bessel-remedial/orchestrator/README.md` — the campaign-wide stop conditions.
- **Every log** in `prompts/bessel-remedial/logs/`. Unlike the earlier workstreams, you need all of
  them: prompt 09 assembles a verification document from all eight, and your job is to check the
  numbers it cites actually appear in them.
- `docs/bessel-remedial/baseline-2026-09.md` — prompt 01's pre-change baseline, which prompt 09
  compares against.
- `docs/lg-phase-and-handover-followup-2026-09.md` §2.4 and §2.5 — what prompt 09 must supersede.

Read `08-fixture-revalidation.md` only when about to dispatch it, and likewise 09.

## Preconditions

`git status` clean; board rows 01–07 ✅ or ⚠️, rows 08 and 09 ⬜. Confirm and **time** the two suites,
because prompt 08 will make them slower and you need the before number:

```bash
time PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -5
time PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -5
```

The first did not complete within 50 minutes before the campaign began
(`IMPLEMENTATION_STATE.md` §5 note 12), so budget for it: run it in the background, get on with
reading, and fall back to per-module runs if it will not finish.

Check `prompts/source-remediation/IMPLEMENTATION_STATE.md` and note its current state: prompt 09
adds **one entry** to its §3 and must change nothing else there.

## Dispatching

For prompt NN, launch a subagent with **exactly** this context:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/bessel-remedial/README.md`, `prompts/bessel-remedial/RECONCILIATION.md`,
> `prompts/bessel-remedial/IMPLEMENTATION_STATE.md`, then your prompt
> `prompts/bessel-remedial/NN-<name>.md` and the `DRAFT-PLAN.md` sections it cites. Execute the
> prompt exactly. Do not read any other prompt under `prompts/bessel-remedial/`. You **may** read
> the logs your prompt names — prompt 09 may read all of them. Follow README §5 for the commit, the
> log and the board update. When you finish, reply with: the commit SHA, the **Result** line from
> your log, the "State handed to the next prompt" section verbatim, and a list of every deviation
> with its classification tag.

Model per prompt: **08 → Opus**, **09 → Sonnet**. Do not substitute.

Run 08 → 09. Prompt 09 records prompt 08's attribution table and cannot run first.

## Reviewing prompt 08 — the tests

Structural checks 1–3 as in `README.md` §4.3. Then:

4. **Tests pass when you run them, all four suites:**
   ```bash
   time PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -10
   time PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -10
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t . 2>&1 | tail -5
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -5
   ```
   The last two must pass; nothing in prompt 08 touches them, so a failure means something strayed.

5. **Allowed files only**, and this is the check most likely to catch a problem:
   ```bash
   git diff HEAD~1 --stat
   git diff HEAD~1 -- ComputeTargets/tests/
   ```
   The second must show **only comment lines and numeric constants**. Fixture logic, protocols,
   class definitions and imports in `test_tk_source_functions.py` and `test_phase_groups.py` belong
   to the in-flight `source-remediation` campaign (`README.md` §4.2); a change to any of them is a
   stop. Verify no production module was touched anywhere:
   `git diff HEAD~1 --stat -- LiouvilleGreen/*.py ComputeTargets/*.py main.py` must be empty.

6. **The attribution table exists and is complete.** Prompt 08 §5 requires one row per fixture
   comparison with error-before, error-after and an attribution from a fixed set (**Bessel oracle**,
   **consumer re-spline**, **LG truncation**, **quadrature**, **reference value**). Read it, and
   apply the prompt's own consistency rule: **a row whose "after" barely moved and whose attribution
   is "Bessel oracle" is a contradiction.** If you find one, the attribution is wrong — stop and
   ask.

   This table is the campaign's substantive claim about what it did and did not improve. A log
   without it fails review even if every test passes.

7. **No threshold was loosened.** For each of the four files, read the diff's numeric changes and
   confirm every one moves in the tightening direction, or is unchanged with a comment saying what
   limits it. In particular:
   ```bash
   git diff HEAD~1 -- LiouvilleGreen/tests/test_bessel_phase.py
   ```
   `test_phase_derivative`'s threshold must be **tighter** than \(10^{-6}\) at low order (README §6
   asks \(10^{-9}\)) and may stay \(10^{-6}\) at \(\nu\ge20.5\). Loosening it, or any other
   threshold in the campaign, is a stop condition
   (`orchestrator/README.md`, `IMPLEMENTATION_STATE.md` §5 note 4).

8. **The measure changed, not just the number,** in `_test_bessel_value` and `test_high_order`. Both
   currently divide by `their_j`/`theirs`, which is meaningless near a zero of \(J_\nu\) — README §6
   requires zeros and extrema be covered, which needs the envelope-normalized phase-pair measure
   from `bessel_reference`. A test that keeps `fabs((ours - theirs)/theirs)` and merely tightens the
   bound has not done the prompt: it will now fail or pass essentially at random near a zero.

9. **Every un-tightened threshold has an explanation.** Read them. "Left as-is" without a stated
   limiting mechanism is incomplete work; the point of the exercise is knowing *what* limits each
   number now.

10. **The convention re-check happened.** `test_tk_source_functions.py:261-266` defines
    `theta_exact = pi - raw_theta(x)` with a four-clause justification. Prompt 08 §4.1 requires
    confirming each clause against the new zero-point. The log must say so **clause by clause**, not
    "checked, still fine" — this is the fixture that inherits the object's convention directly, and
    `DRAFT-PLAN.md` §8.3 asks for exactly this re-check.

11. **`DEFAULT_3BESSEL_CHEBYSHEV_ORDER`.** If prompt 07 or 08 found it limiting, the log must say so
    with a number and the board must carry a new §3 issue. **It must not have been changed** — that
    is not this campaign's to fix (prompt 08 §3 item 2 says so explicitly). Verify with
    `git diff HEAD~1 -- LiouvilleGreen/three_bessel_integrals.py` returning empty.

12. **Runtime.** Compare against your precondition timing and prompt 01's baseline. Tighter
    tolerances plus \(\nu=400.5\) and \(\nu=1000.5\) may make the suite substantially slower. If it
    exceeds ~30 minutes the prompt asks the agent to **propose, not implement**, what to skip by
    default; confirm nothing was skipped unilaterally, and put the proposal in your report.

## Reviewing prompt 09 — the record

Structural checks, plus:

4. **The suites still pass** (`LiouvilleGreen/tests`, `ComputeTargets/tests`). Nothing here touches
   code, so a failure means something strayed.

5. **Allowed files only:** `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`,
   `docs/lg-phase-and-handover-followup-2026-09.md`, `docs/bessel-remedial-verification.md`, the log,
   the board, **plus exactly one entry** in `prompts/source-remediation/IMPLEMENTATION_STATE.md`.
   Check that last one carefully:
   ```bash
   git diff HEAD~1 -- prompts/source-remediation/
   ```
   It must add one §3 issue and change **nothing else** — no status row, no item-level row, no
   standing note. Altering that campaign's board state is a stop.

6. **The benchmark actually ran**, and its result is recorded with times, split between
   `build_phases` and the Levin evaluation (`bessel_tier.py:76-90` returns `phase_time`, so the
   split is free).
   - If \(\kappa=1000\) completed, `B1_KAPPA` should now include it and the note should be rewritten
     with the measured times.
   - **If it did not complete, that is a stop condition** (prompt 09 §2). Confirm the agent stopped
     rather than raising the cap hopefully or extending the timeout past ~30 minutes. Check
     `git diff HEAD~1 -- docs/adaptive-levin-benchmark/` to see what it actually did.

7. **The benchmark note's diagnosis is corrected.** The old comment reads as a cost statement ("the
   phase layer, not the Levin core, is the scalability limit"). The replacement must state the
   **mechanism**: the build is ~0.1 s across five decades of \(x_{\max}\) and then *stalls* above
   \(x\approx2.5\times10^{15}\), because SciPy/Amos `jv`/`yv` lose argument-reduction accuracy, the
   ODE right-hand side stops being \(1+O(\nu^2/x^2)\), and DOP853 at `rtol=5e-14` cannot pass its
   error test on noise (`RECONCILIATION.md` C1). A rewrite that keeps the cost framing has recorded
   the wrong finding permanently — stop.

8. **The follow-up document's five updates all landed** (prompt 09 §3): the measured replacement
   accuracy; the offset finding; the fixture-versus-production tolerance distinction; the chunking
   measurement; and **removal of the stale blanket \(x\times10^{-8}\) claims while retaining the
   historical measurements as historical evidence**. Read the diff: numbers marked superseded and
   dated is right; numbers deleted is wrong. Also confirm §2.5's "`bessel_phase` already keeps `Q`
   as a spline" bullet was updated — that option no longer exists.

9. **The verification document is self-contained.** Prompt 09 §4 requires it to be checkable
   without reading any log. Test that: pick three numbers from it at random and confirm each appears
   in a log or in the baseline file. Then check the acceptance table has an **achieved** value
   beside every target, or an explicit "not met" with a reason — a target quietly omitted is the
   failure mode here.

10. **The performance claim is honest.** Grep the verification document and the commit message for
    speed-up ratios. The correct statement is that a hard cliff at \(x\approx2.5\times10^{15}\) was
    removed and the supported \(x_{\max}\) rose by more than a decade; a "13× faster" claim on a
    0.1 s operation is noise, and `IMPLEMENTATION_STATE.md` §5 note 3 forbids it.

11. **Every cited SHA resolves:** for each SHA in the document, `git cat-file -e <sha>`.

12. **The board is closed out properly:** row 09 filled, campaign marked complete, resolved §3
    entries **moved to §4** rather than deleted, and §3 containing only what is genuinely still open
    — which should include `[00-qsi-three-bessel-levin-excluded]`, since prompt 09 hands it over
    rather than closing it.

## Continue or stop

**Stop and report** on any campaign-wide stop condition or:

- **08:** any threshold loosened; a `ComputeTargets/tests/` change beyond comments and constants;
  any production module touched; the attribution table missing, incomplete, or self-contradictory
  per §6 above; the measure left as a bare relative error in `_test_bessel_value` or
  `test_high_order`; an un-tightened threshold with no stated limiting mechanism;
  `DEFAULT_3BESSEL_CHEBYSHEV_ORDER` changed; the convention re-check not done clause by clause; or
  a test skipped unilaterally to control runtime.
- **09:** the benchmark did not complete at \(\kappa=1000\); the cap raised without a completing
  run; the note's diagnosis still framed as cost; historical measurements deleted rather than marked
  superseded; an acceptance row with no achieved value and no "not met"; a speed-up claim; a SHA
  that does not resolve; or any change to `prompts/source-remediation/` beyond the single §3 entry.

## Completion criterion

Rows 08 and 09 ✅ or ⚠️; the campaign marked complete; `docs/bessel-remedial-verification.md`
existing and self-contained; the benchmark result recorded; the follow-up document superseded in
place; and the hand-off entry present in `prompts/source-remediation/IMPLEMENTATION_STATE.md` §3.

Then report to the user, as the campaign's final report:

- the **achieved acceptance table** against README §6, with the \((\nu,x)\) of each maximum, and any
  row not met;
- the **attribution table** — what this campaign improved, and what floors remain and why;
- the **domain result**: the largest \(x_{\max}\) at which construction completes, before and after,
  and the benchmark tier outcome;
- every item still open in `IMPLEMENTATION_STATE.md` §3, and the one handed to
  `source-remediation`;
- the deferred list from `README.md` §7, so the user can see what was deliberately not done.

Do not summarise the code changes in your own words. Point at the logs and at
`docs/bessel-remedial-verification.md`.
