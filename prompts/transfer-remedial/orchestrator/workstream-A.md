# Orchestrator prompt — Workstream A (prompts 01, 02)

You are orchestrating Workstream A of the transfer-function remedial campaign — its Bessel
amplitude-and-phase phase (README §0) — in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `transfer-remedial-plan`). You do not write code
yourself. You dispatch one fresh-context subagent per prompt, review what it produced against fixed
criteria, and either continue or stop and report to the user.

Workstream A changes **no production code at all**. It builds the independent references and the
error metrics that every later prompt is scored against, records a pre-change baseline of the
current construction, and pins two SciPy/Amos boundaries as tests. So the review here is mostly
about **whether the measurement infrastructure is independent and whether the baseline is complete**
— a metric defined loosely, or a reference built from the object under test, silently invalidates
every acceptance threshold in prompts 03 through 08.

## What to read

Read these in full before dispatching anything:

- `prompts/transfer-remedial/README.md` — **§0 and §0.1 (what this campaign is called and why,
  and how it divides from `prompts/source-remediation` — read these first; the folder was renamed
  because the old name misled)**, §1 (what the campaign does), §1.1 (out of scope),
  **§2 (the six design facts)**, §4 (dependencies), §4.3 (your procedure and stop conditions), §5,
  §5.1 (the log template), **§6 (the acceptance table and the error definitions)**.
- `prompts/transfer-remedial/RECONCILIATION.md` — all of it. §1 is the list of plan claims that
  reproduced; §2 is the four corrections the prompts are built on; §3 is what the plan omits.
- `prompts/transfer-remedial/IMPLEMENTATION_STATE.md` — the board, §3 (two issues already open before
  any prompt runs) and §5 (standing notes; notes 7, 8 and 9 govern how references must be built).
- `prompts/transfer-remedial/orchestrator/README.md` — the campaign-wide stop conditions.

Read `prompts/transfer-remedial/01-reference-harness.md` only when you are about to dispatch it, and
likewise 02. **Do not read prompts 03–09.**

## Preconditions

Before the first dispatch, confirm: `git status` is clean; `git log -1` is at `95cc326` or a later
commit on `transfer-remedial-plan`; `IMPLEMENTATION_STATE.md` shows rows 01 and 02 as ⬜. Confirm the
environment works:

```bash
PYTHONPATH=. ./venv/bin/python -c "import scipy, numpy, mpmath, platform; print(scipy.__version__, numpy.__version__, mpmath.__version__, platform.platform())"
```

Expect `1.15.2 2.2.4 1.3.0` and a Darwin platform. **If SciPy differs, stop and report before
dispatching**: `RECONCILIATION.md` C1 and prompt 02's four facts are measurements of the bundled
Amos library, and prompt 02 will assert numbers that may no longer hold. The right response is a
user decision, not a loosened bound.

Confirm the baseline suite runs at all, and time it — you will need this to review 01's claim about
it:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase -v
```

Expect 4 tests, OK. Note the elapsed time. The **full** `LiouvilleGreen/tests` discovery run did
not complete within 50 minutes on the planning machine and was abandoned, dominated by
`test_3bessel_analytic.py` (`IMPLEMENTATION_STATE.md` §5 note 12); start it in the background if you
want your own number, but do not block on it, and prefer per-module runs.

One thing to raise with the user before Workstream B, not before A (Workstream A changes no
production code, so it is safe either way): `prompts/source-remediation` is at 11 of 12, and its
remaining prompt 12 **runs the pipeline to verify that campaign**. If it has not run by the time
Workstream B starts, it will be verifying a tree whose Bessel oracle changed under it. Check
`grep -n "Progress:" prompts/source-remediation/IMPLEMENTATION_STATE.md` and put the answer in your
completion report (`README.md` §4.2).

## Dispatching a prompt

For prompt NN, launch a subagent with **exactly** this context and nothing more:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/transfer-remedial/README.md`, `prompts/transfer-remedial/RECONCILIATION.md`,
> `prompts/transfer-remedial/IMPLEMENTATION_STATE.md`, then your prompt
> `prompts/transfer-remedial/NN-<name>.md` and the `DRAFT-PLAN.md` sections it cites. Execute the
> prompt exactly. Do not read any other file under `prompts/transfer-remedial/`, except that if you
> are running prompt 02 you may read `logs/01-reference-harness.md` §"State handed to the next
> prompt" for the reference API. Follow README §5 for the commit, the log and the board update.
> When you finish, reply with: the commit SHA, the **Result** line from your log, the "State handed
> to the next prompt" section verbatim, and a list of every deviation with its classification tag.

Model per prompt (README §3): **01 → Opus**, **02 → Sonnet**. Do not substitute.

Run them in order 01 → 02. They are not independent: 02 calibrates its assertions against 01's
`SCIPY_REFERENCE_MAX_X` and uses its cached corners.

## Reviewing prompt 01

Check these six things yourself. Do not take the subagent's word for any of them.

1. **One new commit** on top of the previous HEAD, message per README §5 item 2 (capitalised
   imperative subject under ~72 chars, prose body, `Co-Authored-By: Claude Opus 5
   <noreply@anthropic.com>`).
2. **`logs/01-reference-harness.md`** exists, is in that commit, follows §5.1, has a **Result**
   line, tags every deviation, and — specifically — its "State handed to the next prompt" section
   lists **verbatim signatures**, the tier-selection convention, the value of
   `SCIPY_REFERENCE_MAX_X`, the JSON schema, the per-case baseline errors with the location of each
   maximum, the last \(x_{\max}\) at which the old construction completes, and
   `test_phase_derivative`'s measured margin. A log that says "as in the prompt" fails this check —
   prompts 03 through 08 are programmed against these names.
3. **`IMPLEMENTATION_STATE.md`** in that commit has row 01 filled in and M18 updated.
4. **The tests pass when you run them:**
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_reference -v
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase -v
   ```
   The first must complete in **under ~10 s** and must not import `mpmath` at test time (check by
   reading the imports, and by the runtime). The second must still be 4 tests, OK — this commit
   changes no production code, so any failure there is the agent's.
5. **`git diff HEAD~1 --stat`** touches only: `LiouvilleGreen/tests/bessel_reference.py`,
   `LiouvilleGreen/tests/test_bessel_reference.py`,
   `LiouvilleGreen/tests/bessel_reference_data.json`, files under `docs/transfer-remedial/`, the log
   and the board. **`LiouvilleGreen/bessel_phase.py` must be untouched** — verify with
   `git diff HEAD~1 -- LiouvilleGreen/bessel_phase.py` returning empty. So must
   `LiouvilleGreen/tests/test_bessel_phase.py`.
6. **Independence, which is this prompt's whole point.** Three greps:
   - `grep -n "bessel_phase" LiouvilleGreen/tests/bessel_reference.py` — **must be empty.** A
     reference built from the object under test conceals common error, which is the failure mode
     README §9 Stage 1 exists to prevent.
   - `grep -n "SCIPY_REFERENCE_MAX_X" LiouvilleGreen/tests/bessel_reference.py` — must exist, and
     the constant must be \(2\times10^{15}\) or smaller.
   - Confirm `scipy_reference` actually **raises** above it, by running it:
     `PYTHONPATH=. ./venv/bin/python -c "from LiouvilleGreen.tests.bessel_reference import scipy_reference; scipy_reference(0.5, 1e16)"`
     must fail with an informative error, not return a number.

Then read three things in the log and sanity-check them against arithmetic, not against the code:

- **The baseline \(E_\theta\) figures** should be near 2.0e-6 (\(\nu=3/2\), \(x\le10^3\), fixture
  tolerances), ~1.2e-8 (same, `rtol=1e-12`) and ~5.9e-6 (\(x\le10^7\), tight). If they are wildly
  different, something in the harness is wrong; ask.
- **The `phi` figures** should be \(-1.149353\times10^{-8}\) (\(\nu=3/2\)),
  \(-2.044386\times10^{-8}\) (\(7/4\)), \(-4.836537\times10^{-8}\) (\(5/2\)). These reproduced twice
  independently, so a mismatch is a harness bug.
- **The cliff sweep** should show ~0.1 s builds through \(x_{\max}=10^{15}\) and `TIMEOUT` at
  \(3\times10^{15}\). **If it instead reproduces `DRAFT-PLAN.md` §4.7's ">600 s at \(10^{13}\)",
  stop and report** — that would mean the reconciliation was wrong about this machine, and prompts
  05 and 09 are built on the corrected version.

## Reviewing prompt 02

The same five structural checks, plus:

4′. **The tests pass when you run them:**
```bash
PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_scipy_bessel_domain -v
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -5
```
Under ~30 s for the first.

5′. **`git diff HEAD~1 --stat`** touches only
`LiouvilleGreen/tests/test_scipy_bessel_domain.py`, the log and the board.

6′. **The four facts are asserted, and asserted the right way.** Read the assertion lines:
- Fact 1 must assert all four of `== -0j`, `isfinite is True`, `log(abs(·)) == -inf`,
  `angle == -0.0`. The *combination* is the finding; three of four is not enough.
- Fact 2 must be an inequality with a stated margin, not an equality, and must include the
  campaign-critical consequence — that the boundary is at least \(10^3\times x_\star\) for every
  supported order.
- Fact 3 must assert **both** directions: good below (within \(10^{-9}\) of 1 at
  \(2\times10^{15}\)) and **bad above** (departing by more than \(10^{-2}\) near
  \(5\times10^{15}\)). The "bad above" assertion is deliberately an assertion that the library is
  broken; its docstring must say that a future SciPy fixing it should *fail* this test and prompt a
  decision, so that the next reader does not simply delete it.
- Fact 4 should assert a concrete non-monotonic \((\nu,x)\) pair, not an untestable general claim.

7′. **Every assertion carries the measured value, the SciPy version, and a pointer** to
`RECONCILIATION.md` C1 or `DRAFT-PLAN.md` §4.4 in a comment.

## Continue or stop

**Continue** when all checks pass and the log's Result is `COMPLETE`, or `COMPLETE WITH DEVIATIONS`
where every deviation is tagged `IMPLEMENTATION CHOICE` with its reasoning stated (you are judging
that a reason is *given*, not that you agree with it).

**Stop and report to the user** — do not dispatch the next prompt, do not amend or revert anything
— on any campaign-wide stop condition (`orchestrator/README.md`) or any of these:

- Prompt 01's reference module imports `bessel_phase`, or its metrics do not return the location of
  each maximum (README §6 and prompt 01 §2.1 both require it; prompts 04 and 05 need it to confirm
  the maximum falls adjacent to the turning point).
- The three reference tiers do not agree to \(10^{-14}\) where they overlap. That is either a
  harness bug or a genuine and surprising fact about the environment; either way the user decides.
- `bessel_reference_data.json` has no environment header, or its residuals were built by folding
  \(\theta-x-c_\nu\) into \((-\pi,\pi]\) rather than by matching the tail series
  (`IMPLEMENTATION_STATE.md` §5 note 9). A reference off by an exact \(2\pi k\) at high order
  poisons prompts 03–08 and is very hard to detect later.
- The cliff sweep contradicts `RECONCILIATION.md` C1 (see above).
- Either prompt modified any production module, or `test_bessel_phase.py`.
- Prompt 02 relaxed any bound because it did not reproduce, instead of recording the discrepancy as
  `STRUCTURALLY REQUIRED` and stopping. Its own §2 tells it to stop; if it did not, that is a stop
  for you.

**Report but do not stop:** the full-suite wall-clock, whatever it is. It was already over 25
minutes before this campaign began; if prompt 01 reports it as substantially worse, put the number
in the first paragraph of your report so prompt 08 inherits an honest baseline, but do not treat it
as a failure.

## Completion criterion

Workstream A is complete when rows 01 and 02 on the board are ✅ or ⚠️, both logs exist with
`COMPLETE`-class results, and `docs/transfer-remedial/baseline-2026-09.md` exists and contains the
baseline errors, the `phi` values, the cost-and-cliff sweep and the environment header.

Then report to the user: "Workstream A complete; the tree is at `<SHA>`, ready for Workstream B
(prompt 03). No production code has changed, so this is a natural stopping point
(`README.md` §4.1) — the tree now measures its own accuracy and pins two SciPy boundaries." Include
the baseline error table, `test_phase_derivative`'s margin, and the last \(x_{\max}\) at which the
current construction completes; prompt 09 will compare against all three.
