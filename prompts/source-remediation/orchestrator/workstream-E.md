# Orchestrator prompt — Workstream E (prompts 11, 12)

You are orchestrating Workstream E, the close-out of the source-remediation campaign in the
repository at `/Users/ds283/Documents/Code/SecondaryGWKit` (branch `main`). You do not write code
yourself. You dispatch one fresh-context subagent per prompt, review what it produced against
fixed criteria, and either continue or stop and report to the user.

Prompt 11 is docs only. Prompt 12 is the campaign's one live pipeline run and needs a Ray cluster
and a writable datastore; it may run for one to two hours and it is the prompt most likely to end
with a finding rather than a fix. A `PARTIAL` result from 12 that opens a §3 issue with a
reproduction is the campaign working as designed, not a failure of the agent — your job is to
make sure it is reported accurately.

## What to read

Read these, in full, before dispatching anything:

- `prompts/source-remediation/README.md` — all of it; §1.1, §4, §4.1, §5, §5.1 and §7 matter most.
- `prompts/source-remediation/IMPLEMENTATION_STATE.md` — the status board, including every §3
  active issue (12 must close or narrow them) and the §5 standing notes (which datastores are
  stale or unreadable).
- `docs/spec-code-audit-2026-09.md` §4 (the four UNVERIFIED items 12 measures), §6 (the three spec
  edits 11 makes), §7.
- **Every log** `prompts/source-remediation/logs/01-*.md` through `10-*.md`. Prompt 11 needs their
  commit SHAs; prompt 12 re-checks their "Verification performed" sections. You do not need to read
  prompts 01–10 themselves; the logs are the record.
- `docs/backport-modules-verification.md` and `prompts/backport-modules/10-verification.md` — the
  format and the scoped-driver approach 12 follows; you will judge 12's document against them.

Read `prompts/source-remediation/11-spec-annotations.md` only when you are about to dispatch it,
and likewise for 12.

## Preconditions

Before dispatching **11**, confirm: `git status` is clean; `IMPLEMENTATION_STATE.md` shows rows
01–10 as ✅ or ⚠️ and rows 11 and 12 as ⬜. Build the SHA table 11 needs yourself, from
`git log --oneline`, and keep it for check 4: the board rows for 01 and 03 record the commit
*subject* but not the SHA (prompt 01's log, deviation 4, explains why), so those must be recovered
by subject. At the time this orchestrator prompt was written they were `0f50782` (01), `3199c7b`
(02), `f8c75f5` (03); verify against the current log rather than trusting this.

Before dispatching **12**, additionally confirm:
- `PYTHONPATH=. ./venv/bin/python -c "import ray; print(ray.__version__)"` succeeds. If it does
  not, stop and report (README §4.1: 12 cannot obtain a Ray cluster).
- There is a location for a **fresh** datastore with enough free space (`df -h .`), and that
  location is either outside the repository or covered by `.gitignore` (check). Existing
  `*.sqlite` files in the repository root are stale or unreadable after prompts 06 and 09 (board §5)
  and are read-only reference material for 12's §2 item 1; they must not be overwritten or deleted.
- 11's commit is in place and check 5 for 11 passed.

## Dispatching a prompt

For prompt NN, launch a subagent with **exactly** this context and nothing more:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/source-remediation/README.md`, `prompts/source-remediation/IMPLEMENTATION_STATE.md`,
> `docs/spec-code-audit-2026-09.md`, every log under `prompts/source-remediation/logs/`, then your
> prompt `prompts/source-remediation/NN-<name>.md` and the documents it cites. Execute the prompt
> exactly. Do not read prompts 01–10 under `prompts/source-remediation/`; the logs are the record of
> what they did. Follow README §5 for the commit, the log and the board update. When you finish,
> reply with: the commit SHA, the **Result** line from your log, and a list of every deviation with
> its classification tag.

For **11**, append: "Where the board records a commit subject but no SHA, recover the SHA from
`git log --oneline` by subject and cite the SHA."

For **12**, append: "Create the fresh datastore at a path that does not already exist, under
<the location you confirmed above>; never overwrite, migrate or delete an existing datastore.
Record the exact path and the exact driver command in your log and in the verification document.
Do not commit the datastore. If a Ray cluster cannot be started or the datastore location is not
writable, stop and say so; do not substitute stand-in objects for Layer 2."

Model per prompt (README §3): **11 → Sonnet**, **12 → Opus**. Do not substitute.

Run them in order 11 → 12. 12's live run may take one to two hours; let it run in the background
and do not interrupt it. If it has not finished after ~4 hours, stop and report the state you can
observe (the datastore's size and modification time, any partial output) rather than killing it.

## Reviewing a prompt's result

When the subagent replies, check these five things yourself. Do not take the subagent's word for
any of them.

1. **One new commit**, on top of the previous HEAD, whose message follows README §5 item 2, with
   the trailer `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` for 11 and
   `Claude Opus 5` for 12. 11's body lists the three spec notes in one line each; 12's body gives
   the headline `total` vs `analytic_rad` agreement and anything that failed.
2. **`prompts/source-remediation/logs/NN-<name>.md` exists**, is in that commit, follows the §5.1
   template, has a **Result** line, tags every deviation. For 12, the log and
   `docs/source-remediation-verification.md` together must contain: the exact commands run; the
   Layer 1 table (script → expectation → observed) covering all 29 audit scripts and every unit-test
   tree; the Layer 2 measurements for each of the eight items in the prompt's §2, or an explicit
   "not reached" with a cost estimate for each one missed; the datastore path and the driver
   command; and the list of what remains unverified. "All tests pass" without numbers fails this
   check.
3. **`IMPLEMENTATION_STATE.md`** in that commit: for 11, row 11 and item §6 filled in; for 12,
   row 12, items §4.1–§4.4 closed or narrowed with a pointer into the verification document, the
   **Progress** line updated, resolved §3 issues moved to §4 (each with what closed it), and either
   the campaign-complete note or the list of what did not pass.
4. **The prompt's verification reproduces when you run it.**
   - **11:** for every SHA cited in the four edited documents, `git cat-file -e <sha>` succeeds and
     the SHA matches your own table from the preconditions; every file path cited exists; in each
     edited spec file the count of `$$` is even (`grep -c '\$\$'`); every added note carries the
     marker "Audit note (2026-09), not author sign-off"; the audit document's new §8 table covers
     A1–A7 and B1–B11, agrees with the board, and marks the one-loop-layer items "not in scope".
   - **12:** run Layer 1 yourself in full — `PYTHONPATH=. ./venv/bin/python -m unittest discover
     -s <dir> -t .` for `CosmologyModels/tests`, `ComputeTargets/tests`, `AdaptiveLevin/tests` and
     any test directory prompt 10 added — and run at least the audit scripts the document says now
     show a defect *fixed* (`QS_03_spline_error.py`, `QS_04_coverage_and_jacobian.py`,
     `QI_04_triangle.py`, `TK_05`, `TK_06`, `TK_07`), confirming the "observed" column. **Do not
     repeat the Layer 2 pipeline run.** Instead confirm: the driver script exists and parses; the
     datastore path the document names exists, is non-empty, and is not in the commit; and, if the
     driver separates its read-only analysis step from the run, re-run that step against the
     datastore and confirm the tables match the document. Then read the driver for stand-in or
     fake model classes: Layer 2 must use the real pipeline queues (background → Tk numeric → Tk
     WKB → QuadSource → Gk numeric → Gk WKB → GkSource/policy → QuadSourceIntegral) on a plain
     `LambdaCDM` model, and the document must say so.
5. **`git diff HEAD~1 --stat` touches only** the allowed files plus the log and the board.
   - **11:** `docs/spec/01-transfer-function.md`, `docs/spec/02-greens-function.md`,
     `docs/spec/04-source-integral.md`, `docs/spec-code-audit-2026-09.md`. No `-B.md` duplicate, no
     `REVIEW-QUEUE.md`. Within the spec files, hunks only in the §0 sign-off blocks (specs 02, 04)
     or the head block of author notes (spec 01); within the audit document, only an appended §8.
     Inspect the removed (`-`) lines of the diff: there should be none beyond whitespace. Any
     removed or altered line carrying an `R##` formula fails.
   - **12:** new `docs/source-remediation-verification.md`, new scripts under
     `docs/source-remediation-verification/` or `tools/`. **No production code:** any hunk under
     `ComputeTargets/`, `CosmologyModels/`, `Datastore/`, `MetadataConcepts/`, `LiouvilleGreen/`,
     `AdaptiveLevin/`, or in `main.py` fails. No `*.sqlite` or other large binary in the commit.

## Continue or stop

**Continue from 11 to 12** when all five checks pass and 11's Result is `COMPLETE`, or `COMPLETE
WITH DEVIATIONS` where every deviation is tagged `IMPLEMENTATION CHOICE` with a stated reason.

**Stop and report to the user** — do not amend or revert anything — when any of the following
holds:

- Any of the five checks fails.
- Result is `PARTIAL` or `BLOCKED`. For 12 this is the *expected* form of a real finding: report
  the §3 issue it opened, with its reproduction, in the first paragraph.
- 11 cites a SHA that does not exist or does not match your table, edits an `R##` formula, or adds
  a note without the "not author sign-off" marker.
- 12 touched production code — even to fix a defect it found. The prompt forbids it; the fix is a
  new prompt the user decides on.
- 12 faked Layer 2 (stand-in models, a stubbed datastore, or a `main.py` run described as live
  without a datastore path you can inspect).
- 12 could not obtain a Ray cluster or a writable datastore (the subagent should have asked; relay
  it).
- 12's radiation-era `total` vs `analytic_rad` agreement is worse than the $10^{-5}$ prompt 08
  achieved offline, or the ratio `|total − analytic_rad| / total_abserr` exceeds ~1 for more than
  a few rows without an explanation in the document. This contradicts the offline oracle and the
  user should see it before the campaign is declared complete.
- 12's §2 item 3 shows a `mixed`-type Green's function discontinuous at `crossover_z` beyond the
  document's own stated expectation, or item 2 shows the "$G$ smooth, both $T$ oscillatory" row
  never occurred for the $q \approx r \gg k$ shape (the driver's wavenumber choice failed the
  prompt's own requirement).
- A deviation is tagged `UNINTENDED DRIFT` and was kept.
- The subagent asks a question. Relay it verbatim; do not answer it yourself.

**Report but do not stop:** 12 skipped the `LambdaCDM_GenericEOS` run (the prompt makes it
conditional on time) — but it must then appear in the unverified list with a cost estimate, and
audit §4.4 must be left open on the board, not closed. 12's §4.3 count of `has_WKB_violation`
modes and its §4.2 count of `"minimal"`-quality policies are report-only items; quote them.

Your report to the user, in either case, is: for each prompt run — its SHA, Result, the deviations
with tags, the quoted verification numbers (for 12 the headline oracle agreement, the regime mix,
and the four audit §4 outcomes), and which of the five checks you ran and their outcome. Then
either "Campaign complete; the tree is at <SHA>; the verification record is
`docs/source-remediation-verification.md`; still unverified: <the document's list>; every
datastore built before prompt 09 must be rebuilt" or the specific stop condition that fired and
what the user needs to decide. Do not summarise the changes in your own words; point at the logs
and the verification document.
