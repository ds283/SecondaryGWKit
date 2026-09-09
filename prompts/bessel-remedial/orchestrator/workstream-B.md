# Orchestrator prompt — Workstream B (prompts 03, 04, 05)

You are orchestrating Workstream B of the Bessel amplitude-and-phase campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `bessel-remedial-plan`). You do not write code
yourself. You dispatch one fresh-context subagent per prompt, review what it produced against fixed
criteria, and either continue or stop and report to the user.

**This is the campaign's substance, and the workstream to review most closely.** Prompt 03 is a
small, exactly verifiable series module. Prompts 04 and 05 are the two hardest prompts in the
campaign: 04 must track a phase branch through ~90 wraps where the naive method demonstrably fails,
and 05 replaces what production computes. A wrong choice in either propagates into every later
prompt, and — unlike an accuracy shortfall, which the tests catch — a wrong *justification* written
into a docstring survives indefinitely.

Prompt 05 is the campaign's point of no return: after it, production runs on the new construction.

## What to read

Read these in full before dispatching anything:

- `prompts/bessel-remedial/README.md` — §1, §1.1, **§2 (all six design facts — these are what you
  check deviations against)**, §4, §4.1 (stopping points), §4.3, §5, §5.1, **§6 (the acceptance
  table)**.
- `prompts/bessel-remedial/RECONCILIATION.md` — all of it, and **C1 and C2 twice**. C2 in
  particular: `DRAFT-PLAN.md` §4.6 says the residual "never exceeds a cycle", which is false above
  \(\nu\approx630\), and an agent working from the plan alone will write that false claim into the
  code.
- `prompts/bessel-remedial/IMPLEMENTATION_STATE.md` — the board, §3, and **§5 notes 1, 2, 3, 4, 5,
  6, 7, 8 and 9**.
- `prompts/bessel-remedial/orchestrator/README.md` — the campaign-wide stop conditions.
- `logs/01-reference-harness.md` and `logs/02-domain-boundary-tests.md` — the reference API,
  `SCIPY_REFERENCE_MAX_X`, the baseline errors and the pinned boundaries. You need all of these to
  review 03, 04 and 05.

Read `prompts/bessel-remedial/03-closed-form-tail.md` only when you are about to dispatch it, and
likewise 04 and 05. **Do not read prompts 06–09.**

## Preconditions

`git status` clean; `IMPLEMENTATION_STATE.md` shows rows 01 and 02 as ✅ or ⚠️ and rows 03, 04, 05
as ⬜. Confirm the harness the whole workstream is scored against actually works:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_reference LiouvilleGreen.tests.test_scipy_bessel_domain -v
PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase -v
```

All must pass, the last with 4 tests. Record `test_bessel_phase`'s runtime — after prompt 05 it will
be exercising a different construction and you want to notice if it changes character.

If Workstream A's logs do not contain the verbatim signatures prompt 03 and 04 need, **stop**: an
agent that has to guess the reference API will build its own oracle, which is exactly what
Workstream A existed to prevent.

## Dispatching a prompt

For prompt NN, launch a subagent with **exactly** this context and nothing more:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/bessel-remedial/README.md`, `prompts/bessel-remedial/RECONCILIATION.md`,
> `prompts/bessel-remedial/IMPLEMENTATION_STATE.md`, then your prompt
> `prompts/bessel-remedial/NN-<name>.md` and the `DRAFT-PLAN.md` sections it cites. Execute the
> prompt exactly. Do not read any other prompt under `prompts/bessel-remedial/`. You **may** read
> the "State handed to the next prompt" sections of the logs your prompt names, and only those.
> Follow README §5 for the commit, the log and the board update. When you finish, reply with: the
> commit SHA, the **Result** line from your log, the "State handed to the next prompt" section
> verbatim, and a list of every deviation with its classification tag.

Model per prompt (README §3): **03 → Opus**, **04 → Opus**, **05 → Opus**. Prompts 04 and 05 are
marked as the campaign's hardest; if a model stronger than Opus is available, use it for those two.
Do not substitute anything weaker.

Run strictly 03 → 04 → 05. They are not independent: 04's sampled domain terminates at the
\(x_\star\) that 03 computes, and 05 has no content without both.

## Reviewing prompt 03 — the tail

Structural checks 1–3 and 5 as in `README.md` §4.3. Then:

4. **Tests pass when you run them:**
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_tail -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -5
   ```
5. **Allowed files only:** `LiouvilleGreen/bessel_tail.py`,
   `LiouvilleGreen/tests/test_bessel_tail.py`, the log, the board. Verify
   `git diff HEAD~1 -- LiouvilleGreen/bessel_phase.py` is empty.
6. **`grep -n "scipy" LiouvilleGreen/bessel_tail.py` must be empty.** The module is pure series
   arithmetic; a SciPy Bessel call in it would reintroduce the Amos dependency the tail exists to
   escape.
7. **Read the assertion lines** and confirm the tests assert what the prompt asked:
   - \(\nu=1/2\) is asserted **exactly** (`== 0.0`), not to a tolerance. This is the one place in
     the campaign where exact equality is right, and it is a genuine consequence of \(\mu-1=0\).
   - The crossover test asserts all four of: \(x_\star\in[50\nu,4800\nu]\); the first omitted term
     below the budget; the **measured** \(\lvert\delta r\rvert\) against the reference below the
     budget; and monotonicity of \(x_\star\) in the budget. A crossover "test" that only checks the
     first omitted term is not a remainder test — it never compares against a reference.
   - The failure path is tested: `tail_crossover` raises for an unreachable budget.
   - The high-order branch sanity test asserts \(r_\nu(x_\star)>\pi\) at \(\nu=1000.5\)
     (measured 5.0025 — `RECONCILIATION.md` C2). This is the test that fails if anyone ever wraps
     this quantity.
8. **Check the \(x_\star\) table in the log** against arithmetic. At the \(10^{-11}\) budget,
   \(x_\star/\nu\) should land in the low hundreds for \(\nu\gtrsim20\) (two terms give
   \(\lvert\delta r\rvert\approx4\times10^{-9}\) at \(50\nu\) and \(1.3\times10^{-10}\) at
   \(100\nu\) for \(\nu=100.5\), so \(10^{-11}\) needs a few hundred \(\nu\) at two terms, less at
   three). A table showing \(x_\star/\nu\) in the thousands, or below 50, wants a question.
9. **The coefficient question.** The prompt permits either three terms plus a fourth as the omitted
   estimator, or two plus the third, and **forbids guessing a coefficient**. Confirm the log says
   which shipped and cites DLMF for every coefficient used. A coefficient with no citation is a
   stop.

## Reviewing prompt 04 — the near region

This is the review that matters most. Structural checks as above, plus:

4. **Tests pass when you run them:**
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_near_region -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -5
   ```
   Time the first. The \(\nu=1000.5\) case is the expensive one; if the module takes more than
   ~2 minutes, note it — prompt 08 inherits the cost.
5. **Allowed files only:** `LiouvilleGreen/bessel_near_region.py`, its test, the log, the board.
   `bessel_phase.py` and `bessel_tail.py` untouched;
   `grep -n "import.*bessel_phase" LiouvilleGreen/bessel_near_region.py` empty.
6. **The branch-tracking test is present and has both halves.** This is the acceptance criterion of
   README §6's high-order structural row and the single most important check in the workstream.
   Read the test:
   - a **negative control** asserting a fixed-density 250-per-e-fold `np.unwrap` at \(\nu=1000.5\)
     gives \(E_\theta>0.1\); and
   - a **positive result** asserting the shipped tracker gives \(E_\theta\le10^{-6}\) **and**
     `wraps_tracked == 90 ± 1`.

   A test with only the positive half fails this check. Without the negative control the test does
   not document the failure the design exists to avoid, and a later reader cannot tell whether the
   tracker is doing anything.
7. **The shipped tracker is not `np.unwrap`.**
   `grep -n "unwrap" LiouvilleGreen/bessel_near_region.py` — occurrences are permitted **only** in
   the negative-control test file, never in the module. `DRAFT-PLAN.md` §12 says the fixed-grid
   `unwrap` prototype "must not be copied as the high-order branch algorithm". Also confirm there is
   **no `root_scalar`** anywhere: prompt 04 §3.3 forbids adding a root solve, and the whole `phi`
   finding is what a loose root solve costs.
8. **The plausibility band is two-sided and both sides are tested.** Read the band constants and
   confirm they are justified against `RECONCILIATION.md` §3.1's table in a comment. Then confirm
   the test feeds it *both* a fabricated `-0j` and a fabricated large value, and that it also
   asserts `isfinite` alone would have passed the `-0j` case. A one-sided \(a\gtrsim1\) check passes
   this prompt's accuracy tests and still fails the design.
9. **Two-sided adaptivity is asserted, not just claimed.** The test must assert the node spacing in
   the top decade is **coarser** than at the turning point, with a measured factor. Revision 1 of
   the plan discussed only refining; a construction that only ever refines passes every accuracy
   test and still fails README §2 (c).
10. **The derivative consistency test exists** (README §2 (f)): \(e^{-2\ell}\) vs \(1+r_u/x\) vs the
    reference \(\theta'\), all three. It is mandatory precisely because the Wronskian check is a
    tautology under this design; if the test is absent, the design has no independent check at all.
11. **No `phase_spline`, no `simple_mod_2pi`** in the module, and the docstring's justification for
    interpolating \(r\) directly is the \(\varepsilon\cdot571\approx1.3\times10^{-13}\) argument —
    **not** "never exceeds a cycle". Grep for the false phrase; finding it is a stop condition
    (`orchestrator/README.md`).
12. **Read the per-order table in the log.** Expect \(E_\theta\lesssim10^{-14}\) at low order,
    \(\lesssim10^{-9}\) at \(\nu=100.5\), \(\lesssim10^{-6}\) at \(\nu=1000.5\); derivative errors
    an order or two worse than the values; and — this is the interesting one — **every maximum in
    the interval adjacent to the turning point**. A maximum located elsewhere is not a failure but
    it contradicts `DRAFT-PLAN.md` §4.7, so ask about it.
13. **Compare `achieved_*` against measured.** The log must state the estimator method and how the
    estimates compare to the measured errors. An estimator that **under-reports** by an order is a
    real problem — prompt 06 turns these numbers into `theta_abserr`, whose entire purpose is that
    "the caller sees an honest number instead of an artificially small one". Under-reporting is
    worth a stop; over-reporting is fine and should be noted.

## Reviewing prompt 05 — the construction

Structural checks, plus:

4. **Tests pass when you run them, and these three commands are the review:**
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_two_region -v
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -20
   ```
   The second is the campaign's standing regression gate: 4 tests, OK, with
   `test_phase_derivative`'s \(10^{-6}\) contract **unchanged**. Read the file's diff (there should
   be none — it is not in prompt 05's allowed list) and confirm no threshold moved.

   The third is where a surprise is most likely. `test_three_bessel.py` and
   `test_3bessel_analytic.py` had their tolerances set against the old accuracy, and
   `DRAFT-PLAN.md` §9 Stage 4 warns that a six-to-eight-order improvement "can expose a different
   limiting error rather than simply passing more easily". **If either fails, that is a finding, not
   a nuisance: stop and report the numbers.** Do not let the agent loosen them — they are prompt
   08's.
5. **Allowed files only:** `LiouvilleGreen/bessel_phase.py`,
   `LiouvilleGreen/tests/test_bessel_two_region.py`, the log, the board. Verify **untouched**:
   `main.py`, `LiouvilleGreen/three_bessel_integrals.py`,
   `LiouvilleGreen/tests/test_bessel_phase.py`, `bessel_tail.py`, `bessel_near_region.py`,
   everything under `ComputeTargets/`.
6. **The four removals happened:**
   ```bash
   grep -n "solve_ivp\|root_scalar\|phase_spline\|simple_mod_2pi" LiouvilleGreen/bessel_phase.py
   ```
   must be empty. Each removal must have its measured justification in the commit body (prompt 05
   §1 tabulates all four).
7. **The commit message does not claim a speed improvement** as the motivation
   (`IMPLEMENTATION_STATE.md` §5 note 3) and does not contain the false cycle claim (note 2).
8. **The crossover is a hard stitch, not a blend.** Read the region-selection code. A taper or
   weighted average across \(x_\star\) is an `IMPLEMENTATION CHOICE` that prompt 05 §2.2 permits
   only with explicit justification and a seam continuity test; if it appears without both, stop.
9. **The split-evaluation test asserts the *gain*, not just the accuracy.** Prompt 05 §4 item 4
   requires a comparison against a deliberately naive `sin(x + d)`, expecting ~4.7e-2 versus
   ~1.1e-16 at \(x=10^{15}\). Without it, a later refactor can silently undo the design.
10. **`phi` is 0.0** for every \(\nu>1/2\) tested, or the key is documented as removed. Not a small
    number — exactly zero.
11. **The cliff test exists.** Prompt 05 §4 item 8 requires building at
    \(x_{\max}=3\times10^{15}\) and \(8.6\times10^{15}\) and asserting **completion**. Those are the
    cases the old construction stalls on and clearing them is the real scalability result. Confirm
    the test does **not** assert a speed-up ratio against the old build — a ratio on a 0.1 s
    operation is noise, and asserting it would bake `DRAFT-PLAN.md` §4.7's incorrect framing into
    the suite.
12. **Domain discipline.** Confirm the supported \((\nu,x_{\max})\) domain is *stated* and that
    out-of-domain requests raise. Widening it is a stop condition even though the tail makes it
    tempting.

## Continue or stop

**Continue** when all checks pass and the log's Result is `COMPLETE`, or `COMPLETE WITH DEVIATIONS`
where every deviation is tagged `IMPLEMENTATION CHOICE` with its reasoning stated.

**Stop and report** on any campaign-wide stop condition (`orchestrator/README.md`) or:

- **03:** a series coefficient without a DLMF citation; the crossover not compared against a
  reference; `scipy` imported in `bessel_tail.py`; \(\nu=1/2\) not exact.
- **04:** the branch-tracking test missing either half; `wraps_tracked` not asserted; `np.unwrap` or
  `root_scalar` in the module; a one-sided plausibility band; no two-sided-adaptivity assertion; no
  derivative-consistency test; an `achieved_*` estimator that under-reports; the \(\nu=1000.5\)
  criterion unmet inside the refinement cap; a proposal to lower the supported order ceiling; or a
  conclusion that \(10^{-11}\) is unreachable at low order. **Any of the last three is a design
  question for the user, not an implementation detail** — prompt 04 §7 tells the agent to stop, so
  if it worked around one instead, that is a stop for you.
- **05:** any of the four removals incomplete; `test_phase_derivative` failing or altered;
  `test_three_bessel.py` or `test_3bessel_analytic.py` failing; a blend across the crossover without
  justification and a seam test; `phi` non-zero; the cliff test absent or asserting a speed ratio;
  the domain widened; or the false cycle claim / a speed-based motivation anywhere in the file or
  the commit message.

**Report but do not stop:** the \(\nu=1000.5\) build cost from 04, and the
`test_bessel_two_region.py` runtime from 05. Both are inputs to prompt 08's cost, not acceptance
criteria — put them in your report's first paragraph.

## Completion criterion

Rows 03, 04 and 05 ✅ or ⚠️; three logs with `COMPLETE`-class results;
`test_bessel_phase.py` passing unchanged; and the full `LiouvilleGreen/tests` discovery run passing.

Then report: "Workstream B complete; the tree is at `<SHA>`. `bessel_phase` is replaced and
accurate, but **do not stop here** — `main.py` has not been migrated, so the production tolerance
arguments have no referent (`README.md` §4.1). Ready for Workstream C (prompt 06)." Include the
achieved acceptance table per order with the \(x\) of each maximum, the \(x_\star\) actually used,
the crossover agreement, `test_phase_derivative`'s measured margin, and the largest \(x_{\max}\) at
which construction now completes.
