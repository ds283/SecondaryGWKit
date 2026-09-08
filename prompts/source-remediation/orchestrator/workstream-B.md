# Orchestrator prompt — Workstream B (prompts 05, 06)

You are orchestrating Workstream B of the source-remediation campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `main`). You do not write code yourself. You
dispatch one fresh-context subagent per prompt, review what it produced against fixed criteria,
and either continue or stop and report to the user.

Workstream B builds the transfer-function Liouville–Green representation (`TkSourceFunctions`,
prompt 05) and restricts `QuadSource` to the region where it is valid (prompt 06). Everything in
Workstream C is programmed against what these two prompts hand over, so the review here is mostly
about *what was promised to the next prompt*, not just whether tests pass.

## What to read

Read these, in full, before dispatching anything:

- `prompts/source-remediation/README.md` — §1.1 (out of scope), **§2 (a), (b), (c)** (the three
  facts 05 and 06 are built on: a single hand-over redshift per $k$, a closed-form LG amplitude,
  and an exact constant-$w$ fixture), §4 (dependencies), §4.1 (your procedure and stop
  conditions), §5, §5.1.
- `prompts/source-remediation/IMPLEMENTATION_STATE.md` — the status board, including §3 (active
  issues) and §5 (standing notes; note 3 states the phase sign convention 05 must follow).
- `docs/spec-code-audit-2026-09.md` §0.2 A2, A3 and §3 (author conventions); `docs/spec-code-audit/
  TK-report.md` §1 rows R23/R24/R26 and TK-8; `docs/spec-code-audit/QS-report.md` QS-5, QS-6, QS-9.

Read `prompts/source-remediation/05-tk-source-functions.md` only when you are about to dispatch
it, and likewise for 06. **Do not read prompts 07–12.** Do not read any file under
`prompts/source-remediation/logs/` other than the logs 05 and 06 produce (you will need 05's log to
review 06).

## Preconditions

Before the first dispatch, confirm: `git status` is clean; `git log -1` is at or after `f8c75f5`
(prompt 03's commit); `IMPLEMENTATION_STATE.md` shows rows 01 and 02 as ✅ or ⚠️ (soft
dependencies of 05) and rows 05 and 06 as ⬜. Note whether `ComputeTargets/tests/__init__.py`
already exists (prompt 03 may have created it); you need this to read 05's diff. If any check
fails, stop and report.

If Workstream D is running concurrently, its prompt 04 edits the *QuadSourceIntegral* stage of
`main.py`; prompt 06 edits the *QuadSource* stage. They do not conflict, but whichever commit lands
second must be rebased onto the first so that each prompt is still exactly one commit; if a rebase
conflicts, stop and report rather than resolving it.

## Dispatching a prompt

For prompt NN, launch a subagent with **exactly** this context and nothing more:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/source-remediation/README.md`, `prompts/source-remediation/IMPLEMENTATION_STATE.md`,
> `docs/spec-code-audit-2026-09.md` §0–§3, then your prompt
> `prompts/source-remediation/NN-<name>.md` and the audit report sections it cites. Execute the
> prompt exactly. Do not read any other file under `prompts/source-remediation/`, except that if
> you are running prompt 06 you may read `logs/05-tk-source-functions.md` §"State handed to the next
> prompt" for the hand-over definition and field names 05 settled. Follow README §5 for the commit,
> the log and the board update. When you finish, reply with: the commit SHA, the **Result** line
> from your log, the "State handed to the next prompt" section verbatim, and a list of every
> deviation with its classification tag.

Model per prompt (README §3): **05 → Opus**, **06 → Opus**. Do not substitute.

Run them strictly in order 05 → 06. They are **not** independent: 06 uses the hand-over definition
05 fixes, and 08 will assert the two agree. Do not run them in parallel.

## Reviewing a prompt's result

When the subagent replies, check these five things yourself. Do not take the subagent's word for
any of them, and do not re-derive the physics — that is what the prompt's tests are for.

1. **One new commit**, on top of the previous HEAD, whose message follows README §5 item 2 (a
   capitalised imperative subject under ~72 characters, a prose body, a `Co-Authored-By: Claude
   Opus 5 <noreply@anthropic.com>` trailer). For 05 the body must say why the object is not
   persisted and state the closed-form $d\ln M/dz$ used; for 06 it must say what region the object
   now covers and that the schema is unchanged.
2. **`prompts/source-remediation/logs/NN-<name>.md` exists**, is in that commit, follows the §5.1
   template, has a **Result** line, tags every deviation, and quotes numbers. Specifically:
   - **05:** the measured maximum for *each* assertion in the prompt's §6 item 4, for both
     $w = 1/3$ and $w = 0.2$; which method was used to fix `sin_coeff` (matching at the hand-over
     or backing out from the exact $M$); and a **"State handed to the next prompt"** section that
     lists the final field names and call signatures verbatim (prompts 07 and 08 program against
     these — a log that says "as in the prompt" fails this check).
   - **06:** the statement that the A3 `IndexError` was reproduced on the pre-commit code; the
     §3.4 spline residual as a number; and a "State handed to the next prompt" section naming the
     hand-over properties (`crossover_z_q`, `crossover_z_r`, `numeric_region` or whatever was
     shipped), the `QuadSourceFunctions` fields, and the residual.
3. **`IMPLEMENTATION_STATE.md`** in that commit has row NN filled in, the item-level rows updated
   (05: A2 1/3; 06: A3 and A2 2/3), and: for 06, the §3.4 residual entered in §3 as an open
   observation and a §5 note that existing `QuadSource` rows are stale.
4. **The prompt's tests pass when you run them.** Both prompts:
   `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` (time it; 05
   says under ~30 s — a large overshoot is worth a sentence in your report, not a stop), and also
   `-s CosmologyModels/tests` and `-s AdaptiveLevin/tests` to confirm nothing regressed.
   Then confirm the tests assert what the prompt asked for, by reading the assertion lines (not the
   whole file): for 05, tolerances of $10^{-8}$ (T, dT_dz, T_WKB, M), $10^{-6}$ (dlnM_dz, omega vs
   `theta_deriv`), exact `1.0`/`0.0` above the numeric region, the `crossover_z` consistency
   assertion, and both $w$ values; for 06, the $10^{-14}$ kernel comparison, the interior-gap
   `RuntimeError` naming the factor, and the A3 regression. A test whose tolerance is looser than
   the prompt states fails this check even if it passes.
   For 06 additionally run `docs/spec-code-audit/scripts/QS_04_coverage_and_jacobian.py` (or the
   adapted copy the log names, if the mocks needed the new attributes) and confirm part (a) no
   longer raises `IndexError`.
5. **`git diff HEAD~1 --stat` touches only** the allowed files plus the log and the board.
   - **05:** new `ComputeTargets/TkSourceFunctions.py`, `ComputeTargets/__init__.py`, new
     `ComputeTargets/tests/test_tk_source_functions.py` (and `tests/__init__.py` if it did not
     exist). `TkNumericIntegration.py`, `TkWKBIntegration.py`, `QuadSource.py`,
     `QuadSourceIntegral.py` and everything under `Datastore/` must be untouched. Also grep the new
     module for `ray` and `Datastore` imports; there must be none.
   - **06:** `ComputeTargets/QuadSource.py`, `main.py`, new `ComputeTargets/tests/test_quadsource.py`.
     `Datastore/SQL/ObjectFactories/QuadSource.py` must be untouched. Within `main.py`, hunks only
     in the QuadSource stage (`build_tensor_source_work` and neighbours, roughly `main.py:860-1000`
     pre-commit); a hunk in the Tk or QuadSourceIntegral stages fails. Within `QuadSource.py`,
     `source_function` (pre-commit lines 23–51) must have no hunk at all — the kernel is not this
     prompt's to change.

## Continue or stop

**Continue to the next prompt** when all five checks pass and the log's Result is `COMPLETE`, or
`COMPLETE WITH DEVIATIONS` where every deviation is tagged `IMPLEMENTATION CHOICE` and its
justification is stated (you are judging that a reason is *given*, not that you agree with it).

**Stop and report to the user** — do not dispatch the next prompt, do not amend or revert
anything — when any of the following holds:

- Any of the five checks fails.
- Result is `PARTIAL` or `BLOCKED`.
- A deviation is tagged `STRUCTURALLY REQUIRED` and concerns a formula, a sign or a convention: for
  05, the $M(z)$ product, any term of $d\ln M/dz$, the sign or zero-point of $\theta$, the handling
  of `sin_coeff` (it must not be absolute-valued), or the hand-over definition
  $z^{\rm X} = z_{\rm exit} - \texttt{stop\_deltaz\_subh}$; for 06, the both-numeric region
  ($z' \ge \max(z^{\rm X}_q, z^{\rm X}_r)$), the super-horizon default, or the kernel. The
  *permitted* kind of `STRUCTURALLY REQUIRED` in 05 is an attribute in the §2 duck-typed protocol
  that turned out to be named differently on the real classes; in 06, a differently named
  attribute or a factory read path that reconstructs `z_sample` differently from what the prompt
  assumed but still round-trips.
- 05 persisted `TkSourceFunctions` (any `Datastore/` change, or a new factory).
- 06 removed the interior-gap `RuntimeError`, interpolates across a gap, or changed the schema.
- 06 reports that the datastore round-trip (its §2.4) does not produce a consistent `z_sample` or
  `available == True`; the prompt tells the agent to open a §3 issue and stop in that case, so the
  Result should already be `PARTIAL`/`BLOCKED` — if the agent instead worked around it, that is a
  stop too.
- Any tolerance in the prompt's tests was loosened, or a numerical acceptance figure the prompt
  states was missed, even narrowly.
- A deviation is tagged `UNINTENDED DRIFT` and was kept.
- The subagent touched `AdaptiveLevin/`, `LiouvilleGreen/`, `Datastore/`, `QuadSourceIntegral.py`,
  or any file outside its allowed list.
- The subagent asks a question. Relay it verbatim; do not answer it yourself.

**Report but do not stop:** 06's §3.4 spline residual is a *reported* number, not an acceptance
threshold. If it is worse than ~$10^{-3}$, continue, but put the number in the first paragraph of
your report — it bounds the accuracy of the all-smooth region of the source integral, and the
prompt says the knob is the hand-over window in `main.py`, not anything in this workstream.

Your report to the user, in either case, is: for each prompt run — its SHA, Result, the deviations
with tags, the quoted verification numbers, the "State handed to the next prompt" section verbatim,
and which of the five checks you ran and their outcome. Then either "Workstream B complete; the
tree is at <SHA>, ready for Workstream C (prompt 07); `QuadSource` rows in existing datastores are
stale, and `QuadSourceIntegral` still uses the old integrand until prompt 08" or the specific stop
condition that fired and what the user needs to decide. Do not summarise the code changes in your
own words; point at the logs.
