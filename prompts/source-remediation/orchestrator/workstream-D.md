# Orchestrator prompt — Workstream D (prompt 04)

You are orchestrating Workstream D of the source-remediation campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `main`). It is a single prompt. You do not
write code yourself. You dispatch one fresh-context subagent, review what it produced against fixed
criteria, and report to the user.

## What to read

Read these, in full, before dispatching:

- `prompts/source-remediation/README.md` — §1.1 (out of scope), §4 (04 must land before 10; 04, 06
  and 10 edit disjoint stages of `main.py`), §4.1 (your procedure and stop conditions), §5 (rules
  every prompt follows), §5.1 (log template).
- `prompts/source-remediation/IMPLEMENTATION_STATE.md` — the status board.
- `docs/spec-code-audit-2026-09.md` §0.2 A5 and `docs/spec-code-audit/QI-report.md` QI-12 — the
  finding this prompt fixes.
- `docs/spec-code-audit/scripts/QI_04_triangle.py`. **Run it once before dispatching** and record
  what it prints; the expected figures on the shipped grid are 63,750 triples and 5,133 triangles
  (8.05 %). Those are the numbers you will check the subagent's work against.

Read `prompts/source-remediation/04-triangle-filter.md` only when you are about to dispatch it.
**Do not read prompts 05–12.** Do not read any file under `prompts/source-remediation/logs/`
other than the log 04 produces.

## Preconditions

Before dispatching, confirm: `git status` is clean; `IMPLEMENTATION_STATE.md` shows row 04 as ⬜
and row 10 as ⬜ (if 10 has already landed, this prompt has been overtaken — stop and report).
Workstream D depends on nothing else, so the status of rows 01–03 and 05–09 does not matter; note
the current HEAD SHA for your report. If Workstream B is running concurrently, its prompt 06 edits
the *QuadSource* stage of `main.py`; 04 edits the *QuadSourceIntegral* stage. They do not conflict,
but whichever commit lands second must be rebased onto the first so that each prompt is still
exactly one commit; if a rebase conflicts, stop and report rather than resolving it.

## Dispatching the prompt

Launch a subagent with **exactly** this context and nothing more:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/source-remediation/README.md`, `prompts/source-remediation/IMPLEMENTATION_STATE.md`,
> `docs/spec-code-audit-2026-09.md` §0–§3, then your prompt
> `prompts/source-remediation/04-triangle-filter.md` and the audit report sections it cites.
> Execute the prompt exactly. Do not read any other file under `prompts/source-remediation/`.
> Follow README §5 for the commit, the log and the board update. When you finish, reply with: the
> commit SHA, the **Result** line from your log, the before/after triple counts, and a list of every
> deviation with its classification tag.

Model (README §3): **Sonnet**. Do not substitute.

## Reviewing the result

When the subagent replies, check these five things yourself. Do not take the subagent's word for
any of them.

1. **One new commit**, on top of the previous HEAD, whose message follows README §5 item 2 (a
   capitalised imperative subject under ~72 characters, a prose body stating the before/after
   counts, a `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` trailer).
2. **`prompts/source-remediation/logs/04-triangle-filter.md` exists**, is in that commit, follows
   the §5.1 template, has a **Result** line, tags every deviation `STRUCTURALLY REQUIRED`,
   `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`, and quotes the counts (63,750 → 5,133 on the
   shipped grid) rather than saying "matches".
3. **`IMPLEMENTATION_STATE.md`** in that commit has row 04 filled in (status, SHA, model, log
   link) and item A5 marked complete.
4. **The verification reproduces.** Run
   `PYTHONPATH=. ./venv/bin/python -c "import ast; ast.parse(open('main.py').read())"`.
   Then reproduce the count independently of the subagent: `main.py` has import-time side effects
   (check with a grep for top-level statements before trying), so extract the `closes_triangle`
   function's source with `ast`, `exec` it in a scratch namespace, apply it to the wavenumber grid
   `QI_04_triangle.py` builds, and confirm 5,133 kept of 63,750. Also construct by hand a triple
   sitting exactly on a boundary (`k = q + r` and `k = |q − r|` with values that are exactly
   representable, and one perturbed by a relative `1e-13`) and confirm the predicate **keeps** it —
   the prompt's §"What to do" item 1 requires degenerate triangles to be kept. Finally run the
   existing unit-test trees (`CosmologyModels/tests`, `ComputeTargets/tests`, `AdaptiveLevin/tests`
   with `PYTHONPATH=. ./venv/bin/python -m unittest discover -s <dir> -t .`) to confirm nothing else
   moved.
5. **`git diff HEAD~1 --stat` touches only** `main.py`, the log and the board. Within `main.py`,
   `git diff HEAD~1 -- main.py` must show hunks only in (a) a small helper near the other helpers at
   the top of the file and (b) the `qsi_work_items` construction and its diagnostic print. A hunk in
   the QuadSource stage, the response-redshift grid, or either wavenumber grid is a failure of this
   check.

## Continue or stop

**Report success** when all five checks pass and the log's Result is `COMPLETE`, or `COMPLETE WITH
DEVIATIONS` where every deviation is tagged `IMPLEMENTATION CHOICE` and its justification is stated
(you are judging that a reason is *given*, not that you agree with it).

**Stop and report to the user** — do not amend or revert anything — when any of the following
holds:

- Any of the five checks fails.
- Result is `PARTIAL` or `BLOCKED`.
- A deviation is tagged `STRUCTURALLY REQUIRED` and concerns the inequality itself
  ($|q-r| \le k \le q+r$), which quantity is compared (it must be the physical wavenumber `.k.k`),
  or the direction of the boundary tolerance (boundary triples must be kept, not dropped).
- The kept count differs from 5,133 and the log's explanation does not survive your own recount
  in check 4.
- A deviation is tagged `UNINTENDED DRIFT` and was kept.
- The subagent changed the $(q,r)$ pair grid, the response grid, or `QuadSource` scheduling, or
  touched any file outside `main.py`, the log and the board.
- The subagent asks a question. Relay it verbatim; do not answer it yourself.

Your report to the user, in either case, is: the SHA, the Result, the deviations with tags, the
quoted counts from the log and from your own recount, and which of the five checks you ran and
their outcome. Then either "Workstream D complete; the tree is at <SHA>; prompt 10 may be
dispatched once Workstream B and prompts 07–09 have landed" or the specific stop condition that
fired and what the user needs to decide. Do not summarise the code change in your own words; point
at the log.
