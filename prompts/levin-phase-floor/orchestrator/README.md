# Orchestrator prompts — the levin-phase-floor campaign

One orchestrator note per campaign prompt, written just before dispatch. Each dispatches the
prompt's executing agent, reviews its work against fixed criteria, and either lands it or stops and
reports to the user.

| # | Prompt | File | Written? | Character |
|---|---|---|---|---|
| 01 | The phase-group census of A3 v2 | [`prompt-01.md`](prompt-01.md) | **yes** | Written by another laptop session (`757ef01`). **Orchestrated on the Mac Studio**, where the store is, in parallel with `handover` 11 in the same checkout; it commits nothing there. Its records come back to the laptop and are committed there (campaign README §4.1) |
| 02 | Phase-noise injection | [`prompt-02.md`](prompt-02.md) | **yes** | **Stopped at step 3 and superseded by 03 (user, 2026-10-25); records `87813ed`, reviewed.** On the laptop, a fresh-context subagent in the main checkout. Its first noised probe cell ran ~69 minutes unfinished: width-independent noise above `rtol` makes every Levin call bisect the full depth-20 tree. The review of the records commit is §6 of the note |
| 03 | Instrument seven depth-20 rows | [`prompt-03.md`](prompt-03.md) | **yes** | Written 2026-10-25, after 01 landed. **Orchestrated on the Mac Studio**, as 01, alone in its checkout, which the laptop pins to the dispatch SHA and which must not move until it ends: M0 needs the pre-B1 tree, so `handover` 12's R8 follows it there. Commits nothing there. The review turns on M0's bit-identity (checked independently against the store), the wrappers being transparent, the predictions being saved before the launch, and M1–M5 scored against prompt §3's rules as written. **Ran 2026-10-25 on the Mac Studio and passed 9/9** (three readings for the user: `elapsed` not gated, M5 inapplicable on R7, the pid in `status.json`); route (a), Ray-free; M0 bit-identical 7/7, and R1 re-checked from the laptop. **The mechanism as written does not hold**: M1, M2, M4 and M5 hold, M3 fails on its slope clause, and G1 decides how to read it. Records in the bookkeeping commit, `d5f5a2b` |
| 04 | remove | — | not yet | Held on 03 |

## Running them

01 and 02 run at the same time (campaign README §2): 01 on the Mac Studio, 02 on the laptop. They
share no file except the campaign's board, `docs/OPEN_ISSUES.md` and
`docs/levin-phase-floor/VERIFICATION.md`, and each touches those only at its end. **Whichever lands
second merges the other's board row and index lines by reading them, never by overwriting them.**
02 creates `VERIFICATION.md` with a §1 placeholder if 01 has not landed when it commits; 01's
commit then replaces the placeholder and keeps §2.

## The rules that bind the orchestrator

The ones `prompts/handover/orchestrator/README.md` sets out, unchanged:

1. **You do not write code.** Not the script, not a fix, not the wrapper check. You read the
   agent's records, and re-run what the review names.
2. **One prompt, one agent, one commit.** The orchestrator's own commits are the dispatch note
   before and the residue after (the commit hash on the board and in the log).
3. **Do not re-derive the work.** Check that the prompt's own checks pass when *you* run them, that
   the log classifies every deviation, that the board and `docs/OPEN_ISSUES.md` were updated in the
   same commit, and that the diff stayed inside its allowed files. Re-scoring a prediction from the
   checkpoint is a check of the scoring, and is allowed.
4. **Give the agent only its own prompt** and the files that prompt tells it to read. It does not
   read `orchestrator/`.
5. **Relay every agent question verbatim.** Do not answer it yourself.
6. **Stop rather than repair.** Do not fix a failed check, revert it, or dispatch a follow-up agent
   to patch it. Report the specific check and what the log says about it.
7. **An agent must never assume `HEAD` is its own.** Other sessions commit to this branch, and
   01's records land on it from the Mac Studio's orchestration. The dispatch says so.
8. **Correctness is the only objective** (campaign README §5 rule 4). An argument from cost or
   time for a shortcut is a deviation to record, not a reason to accept.
