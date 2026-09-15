# Orchestrator prompt — workstream C, close-out (prompt 09)

You are orchestrating workstream C of the QCD background campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `qcd-background-audit`). **You do not write
code**, and neither does prompt 09 — it is a verification prompt, and a verification prompt may not
touch production code. That rule is what created `prompts/phase-representation` in the first place,
and enforcing it is most of this review.

One prompt. It answers the question the audit §9 leaves open: **what did the corrected background
do to the consumers?**

## What to read

`../README.md` §0.2 and §6.4 — **the frame, and the thing most likely to be got wrong**;
`../IMPLEMENTATION_STATE.md`; the logs of prompts 01–08; `docs/gktk-remedial-verification.md` §3.5,
§3.6, §3.7, §5, §8; `docs/qcd-background-audit-2026-09.md` §6 and §9; `orchestrator/README.md`;
`CLAUDE.md` — in particular **verification documents are additive**.

## Preconditions

Workstreams A and B complete; `git status` clean. You need **the campaign's base run**, not a
recent one:

```bash
git stash list                        # must be empty
git log --oneline e8f746d..HEAD       # the campaign's commits, in order

git checkout e8f746d
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py \
  > /tmp/qcdbg-verify-base.txt 2>&1
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py \
  > /tmp/qcdbg-audit-base.txt 2>&1
git checkout qcd-background-audit
git status                            # must be clean
```

If you took these at the start of workstream A, reuse them and say so; do not take two different
baselines and compare against whichever is convenient.

## Reviewing prompt 09

1. One commit; log per §5.1; the board updated for rows 01–09, §2, §3 and §4;
   `docs/OPEN_ISSUES.md` in the same commit with its count, date, `Boards` line and §1.7 correct.
2. **No production file in the diff.** `git diff HEAD~1 --stat` must show only `docs/` plus the
   campaign's log and board. **A single production hunk is a stop**, however small and however
   obviously correct — if the agent found a defect it must have recorded it as a §3 issue and
   stopped, and the log must say so.
3. **`verify_production_path.py` was run unedited.** It must not appear in the diff. Neither may
   anything at or above §8 of `docs/gktk-remedial-verification.md`:
   ```bash
   git diff HEAD~1 -- docs/gktk-remedial-verification.md
   ```
   must show an **appended §9 and nothing else**.
4. **Every LambdaCDM row is bit-identical** to `/tmp/qcdbg-verify-base.txt`. Check this yourself,
   by diff. LambdaCDM has no `T(z)` spline (README §2 (g)), so a moved LambdaCDM number means a
   prompt in this campaign leaked into a shared path. **Stop, and name the suspect prompt from the
   logs.**
5. **Every moved QCD number has a stated cause.** Read the diff of the two runs and the document
   side by side. A moved number with no cause is a stop.
6. **The audit script's §1 and §2 are unchanged** — the equation of state was not touched, which is
   README §0.5's boundary made checkable.
7. **The framing is right, and this is the substantive check.** `docs/qcd-background-verification.md`
   must say, in its own words, that §3.5 and §3.6 score a consumer against a producer built from
   the same background, that the error this campaign removed **cancels in them**, and that the
   consumer numbers are therefore *not required to improve*. A document that presents an unchanged
   consumer table as a disappointment, or an improved one as the campaign's headline, has
   misunderstood §0.2 and will mislead every later reader. **The headline is prompt 01's guard**:
   $\int\mathrm{d}z/H$, 3.461e-08 → the floor.
8. **§6, the caveats, is present and honest.** At minimum: the production-$x$ ceiling
   (`docs/OPEN_ISSUES.md` §5); `[00-consumer-anchoring-floor]` and
   `[02-consumer-phi-below-the-storage-granularity]` untouched and limiting `theta_deriv` at
   $k\ge10^7$; `[00-eos-branch-joins-do-not-match]` unrepaired;
   `[02-verify-script-builds-its-own-Gk-consumer]` naming which of §3.5's rows are blind; and that a
   pre-campaign datastore is refused rather than migrated.
9. **Three suites pass** when you run them, with no count fallen from your baseline.

## Continue or stop

The campaign's ungated part is complete when the log is `COMPLETE`-class, every check above passes,
and the board reads **CLOSED at 9 / 12 (workstream D gated)**.

**Stop** on a production hunk, on a moved LambdaCDM number, on a rewritten §1–§8 of
`docs/gktk-remedial-verification.md`, or on a §0.2 misframing in the new document.

## Completion criterion and report

Report to the user:

> The QCD background campaign's ungated part is complete; the tree is at `<SHA>`. The background's
> systematic error in conformal time is `<before>` → `<after>`, which is `<...>` rad at
> $k=3\times10^8$/Mpc against a 9.15e-04 rad floor.

Then, in this order:

1. **T1** — the guard, before and after, and the three phases against their floors.
2. **The representation** — the audit §4 table's three columns and which prompt moved each.
3. **The break-point set** — 407 → 3, and what the `BackgroundModel` build cost did.
4. **The consumers** — what moved, what did not, and why the "did not" is the expected answer.
5. **Cost** — per call and per build.
6. **What the user must now do**: a datastore written before this campaign is **refused**, not
   migrated, and the QCD half must be regenerated. Give the scope from prompt 03's log.
7. **What is left**: workstream D (prompts 10–12), gated on README §7 D7, and the §7 D6 question
   for the equation of state's authors, which is the user's to send.
