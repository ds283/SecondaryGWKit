# Orchestrator prompt — Workstream A (prompts 01, 02, 03)

You are orchestrating Workstream A of the source-remediation campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `main`). You do not write code yourself. You
dispatch one fresh-context subagent per prompt, review what it produced against fixed criteria,
and either continue or stop and report to the user.

## What to read

Read these, in full, before dispatching anything:

- `prompts/source-remediation/README.md` — the campaign: §1.1 (out of scope), §4 (dependencies),
  §4.1 (your procedure and stop conditions), §5 (rules every prompt follows), §5.1 (log template).
- `prompts/source-remediation/IMPLEMENTATION_STATE.md` — the status board.
- `docs/spec-code-audit-2026-09.md` §0.2, §0.3 and §3 — the findings this workstream fixes (A1,
  A6, A7, B1–B4, B9, B10) and the author conventions no prompt may "correct".

Read `prompts/source-remediation/01-genericeos-sound-speed.md` only when you are about to dispatch
it, and likewise for 02 and 03. **Do not read prompts 04–12.** They are not yours to run, and
reading them will bias your review of 01–03 towards what later prompts would find convenient. Do
not read any file under `prompts/source-remediation/logs/` other than the logs 01–03 produce.

## Preconditions

Before the first dispatch, confirm: `git status` is clean; `git log -1` is at or after `d79f792`;
`IMPLEMENTATION_STATE.md` shows rows 01–03 as ⬜. If any fails, stop and report.

## Dispatching a prompt

For prompt NN, launch a subagent with **exactly** this context and nothing more:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/source-remediation/README.md`, `prompts/source-remediation/IMPLEMENTATION_STATE.md`,
> `docs/spec-code-audit-2026-09.md` §0–§3, then your prompt
> `prompts/source-remediation/NN-<name>.md` and the audit report sections it cites. Execute the
> prompt exactly. Do not read any other file under `prompts/source-remediation/`. Follow README §5
> for the commit, the log and the board update. When you finish, reply with: the commit SHA, the
> **Result** line from your log, and a list of every deviation with its classification tag.

Model per prompt (README §3): **01 → Opus**, **02 → Sonnet**, **03 → Opus**. Do not substitute.

Run them in order 01 → 02 → 03. They are independent, so you may run them in parallel if you
prefer, but then you must serialise the commits yourself (rebase the later onto the earlier) so
that each prompt still corresponds to exactly one commit; if a rebase conflicts, stop and report
rather than resolving it.

## Reviewing a prompt's result

When the subagent replies, check these five things yourself. Do not take the subagent's word for
any of them, and do not re-derive the physics — that is what the prompt's tests are for.

1. **One new commit**, on top of the previous HEAD, whose message follows README §5 item 2 (a
   capitalised imperative subject under ~72 characters, a prose body, a `Co-Authored-By: Claude
   <model> <noreply@anthropic.com>` trailer naming the model you dispatched).
2. **`prompts/source-remediation/logs/NN-<name>.md` exists**, is in that commit, follows the §5.1
   template, has a **Result** line, and tags every deviation `STRUCTURALLY REQUIRED`,
   `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`. A "Verification performed" section that says
   only "tests pass" without quoting numbers fails this check.
3. **`IMPLEMENTATION_STATE.md`** in that commit has row NN filled in (status, SHA, model, log
   link), the item-level rows for that prompt's items updated, and any unresolved matter entered
   in §3.
4. **The prompt's tests pass when you run them.** For 01:
   `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .`
   For 02 and 03 additionally: `... -s ComputeTargets/tests -t .` and
   `... -s AdaptiveLevin/tests -t .`. Also run the audit script the prompt names
   (`docs/spec-code-audit/scripts/TK_05_background_derivatives.py` for 01; `TK_08_criterion_sign.py`
   for 02; `TK_06_spline_end_bias.py` and `TK_07_omegaEff_spline_impact.py` for 03) and confirm the
   printed numbers match what the log quotes.
5. **`git diff HEAD~1 --stat` touches only** the files the prompt's "Files you may touch" line
   allows, plus the log and the board.

## Continue or stop

**Continue to the next prompt** when all five checks pass and the log's Result is `COMPLETE`, or
`COMPLETE WITH DEVIATIONS` where every deviation is tagged `IMPLEMENTATION CHOICE` and its
justification is stated (you are judging that a reason is *given*, not that you agree with it).

**Stop and report to the user** — do not dispatch the next prompt, do not amend or revert
anything — when any of the following holds:

- Any of the five checks fails.
- Result is `PARTIAL` or `BLOCKED`.
- A deviation is tagged `STRUCTURALLY REQUIRED` and concerns a formula, a coefficient, a sign, a
  denominator, or which `w` is used where (for 01 that means anything beyond "the dict key was
  named differently"; for 03 anything that changes the *analytic-derivative branch* of
  `_build_derivative`).
- A deviation is tagged `UNINTENDED DRIFT` and was kept.
- Prompt 03 reports that no remedy brought every end-point error within 10× the interior median
  (its §2 threshold), or that `BackgroundModel.compute()` became more than ~10× slower.
- Prompt 02 reports it changed anything in `phase_spline`, or that item B9 needed more than a
  comment or a one-line consistency change.
- The subagent touched `AdaptiveLevin/`, `LiouvilleGreen/`, `Datastore/`, `main.py`, or any file
  outside its allowed list.
- The subagent asks a question. Relay it verbatim; do not answer it yourself.

Your report to the user, in either case, is: for each prompt run — its SHA, Result, the deviations
with tags, the quoted verification numbers, and which of the five checks you ran and their outcome.
Then either "Workstream A complete; the tree is at <SHA>, ready for Workstream D or B" or the
specific stop condition that fired and what the user needs to decide. Do not summarise the code
changes in your own words; point at the logs.
