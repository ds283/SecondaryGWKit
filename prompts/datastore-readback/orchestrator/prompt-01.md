# Orchestrator — prompt 01, the `QuadSourceIntegral` read-back

Read [`../README.md`](../README.md) first, especially §0. **You do not write code** — not the fix,
not the test, and not the deliberate breakage that shows the test works.

**The prompt:** [`01-quadsourceintegral-readback.md`](../01-quadsourceintegral-readback.md)
**Board item:** R1

## 0. What makes this prompt unusual

The fix is one line and will pass. The review is about the other two things:

- **does the guard actually fail on the unfixed code?** A test written against a bug you have
  already fixed, never run against the bug, is not known to catch anything. Prompt §6 item 3
  requires both directions demonstrated. Check the record, and check it names the test.
- **did the audit open rather than fix?** A sweep that repairs eleven factories in the same commit
  destroys the revert-per-prompt property and cannot be reviewed. Findings are §3 issues.

An agent that returns "one line changed, one test added, audit found nothing else" has done the job
**provided** both of the above hold.

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`**, and record the SHA. The agent must be told it is not its own.
2. **Baseline all four suites** — `ComputeTargets`, `CosmologyModels`, `LiouvilleGreen` **and
   `AdaptiveLevin`**, which the hand-over campaign never baselined. Record them. `Datastore/tests`
   does not exist; it will after this prompt.
3. **Record the datastore's row counts** before dispatch, per shard:
   ```bash
   for f in var/datastores/handover-A3-baseline-lambdacdm-shard*.sqlite; do
     echo "$f $(./venv/bin/python -c "import sqlite3;print(sqlite3.connect('$f').execute('select count(*) from QuadSourceIntegral').fetchone()[0])")"
   done
   ```
   Expect 1890 / 1888 / 1889 / 1885 and `GkSourcePolicyData` 290 each. **Check them again
   afterwards**: the prompt forbids writing to the store, and this is how you know.
4. **Confirm the backup exists** at `var/datastores/backup-pre-resume-20260921T091011` and record
   its size. It must still exist afterwards.
5. **Reproduce the failure yourself**, so you are not taking the diagnosis on trust: read one
   stored `QuadSourceIntegral` row through `object_get` and confirm it raises
   `NoSuchColumnError` on `numeric_quad`. Keep the traceback. You will compare the agent's "after"
   against your own "before".
6. `git status` clean.

## 2. Dispatch

One fresh-context subagent. The prompt file, the campaign README, the SHA, the baselines, the
datastore path and its row counts, and the files the prompt's "Read first" names. Nothing else.

Tell it plainly: **one commit**; log at `logs/01-quadsourceintegral-readback.md`; the board created
per §8; `docs/OPEN_ISSUES.md` in the same commit; `black` before committing; **read the datastore,
never write to it**; **do not delete the backup**; **do not start the pipeline**; and that §7's stop
conditions mean *stop and ask*.

Tell it also: if anything long-running becomes necessary, launch it detached and end the turn
rather than polling. Nothing here should need it.

## 3. The review — eight checks

1. **The fix is one line.** `git diff HEAD~1 HEAD -- Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`
   should add `table.c.numeric_quad` and nothing else. A reorganised query, or the two `SELECT`s
   factored together, is a stop — prompt §5 forbids it explicitly.
2. **The read-back works, checked by you.** Re-run your §1.5 script. It must now return the row,
   and the `numeric_quad` value must match what is stored in the table for that serial.
3. **The guard fails on the unfixed code.** The log must show it, naming the test and the error.
   Verify independently: stash the one-line fix, run that test, confirm it fails, restore. This is
   the one check worth doing by hand, because a guard that cannot fail is decoration.
4. **The guard needs no Ray and no datastore.** Read it. If it opens a file under `var/`, that is a
   stop.
5. **What the guard cannot see is written down**, in its docstring and the log. A static check that
   claims complete coverage of the class is overselling; the honest list is the deliverable.
6. **The audit ran across all 22 factories and fixed none of them.**
   `git diff --name-only HEAD~1 HEAD` must show exactly one file under `ObjectFactories/`.
   Findings must appear as §3 issues, each naming file, attribute, omitting `SELECT`, and
   reachability.
7. **Datastore untouched.** Row counts as in §1.3, backup still present at its recorded size.
8. **Suites.** The four baselines unchanged; `Datastore/tests` up by exactly the tests added.
   `black --check` clean. Board in the §8 shape; `docs/OPEN_ISSUES.md` gains a subsection for this
   campaign with the count and date corrected.

## 4. What a good outcome looks like

- One line of production code, and a test that is *demonstrated* to fail without it.
- An honest statement of the guard's blind spots — `getattr` reads, conditional `add_columns`
  paths, `row_data` sourced from another module.
- An audit that names all 12 candidate factories, clean or not, so the next person knows the sweep
  was complete rather than selective.
- If more defects are found: opened, not fixed, with reachability stated for each.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate.

- The fix is not one line, or the diagnosis in README §0 turns out to be wrong.
- The read-back still fails, or `numeric_quad` disagrees with the stored value.
- **The audit finds a defect reachable in a running pipeline**, not only on resume. That is more
  urgent than this prompt.
- The static guard cannot be made honest and the round-trip fallback is also impractical.
- The agent proposes writing to the datastore, deleting the backup, or starting the pipeline.
- Any check in §3 fails. **Report it; do not repair it.**

## 6. After it lands

Report: the commit; the one-line diff; your own before-and-after read-back, with the
`numeric_quad` value; the guard, whether you saw it fail on the unfixed code, and its stated blind
spots; the audit across all 22 factories with any issues opened; the datastore row counts before
and after; and the suite counts.

Then stop. Finishing the interrupted pipeline run is the user's, and so is deciding whether the
audit's findings deserve a prompt 02.
