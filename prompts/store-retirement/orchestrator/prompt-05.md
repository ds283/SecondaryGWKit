# Orchestrator — prompt 05, retire the two stores

Read [`../README.md`](../README.md) first: §3, §5 rule 10 and §6. Then read the prompt in full.
**You do not write code, and you run nothing that writes under `var/`.** The user runs every
`store retire` and the `store amend`.

**The prompt:** [`05-retire-the-two-stores.md`](../05-retire-the-two-stores.md)
**Board items:** R10–R12 · **Closes:** nothing · **Gate:** prompts 01–04 have landed and been
reviewed (04 in `00e0ec6`)

## 0. What makes this prompt unusual

It is the only prompt in the campaign that touches real data, and its central step cannot be
undone. So it runs in three phases, with the user acting between them:

| Step | Who | What |
|---|---|---|
| A | agent | the before-picture, the two dry runs, the reasons and the exact commands |
| 1 | **user** | `store retire` on the sweep store |
| check 1 | you | the sweep's five files gone, its sidecar changed, nothing else changed |
| 2 | **user** | `store retire` on the backup |
| check 2 | you | the backup's five files gone, its sidecar changed, nothing else changed, the live A3 store byte-identical |
| B | agent | the tombstones checked; the amendment drafted |
| 3 | **user** | `store amend` on the live A3 sidecar's `backup` field |
| check 3 | you | only the live A3 sidecar changed |
| C | agent | the amendment checked, the records written, **one commit** |

The review turns on one property: **the live A3 store's five files are byte-identical from start
to finish, and its sidecar changes only by the amendment.** The backup's primary names the live
store's shards by absolute path (audit §2.7). Prompt 01's resolver and its central test are what
stand between that and the live store. This prompt checks it on the real files: in the dry run
before, and by hash after.

**Carried from the reviews of prompts 03 and 04.**
- **The sweep's references report could be broad.** The sweep store sits directly in
  `var/datastores/`. The prompt predicts four runs and no sidecar, from the shipped matcher run
  over today's sidecars, so a broad report would now be a surprise, and the prompt makes it one.
- **`--without-fingerprint` takes any fingerprint exception as D4's error.** It is never passed
  here. A dry run that asks for it is a stop.
- **`store amend`'s identical-value check compares with Python `==`**
  (`[04-amend-calls-true-1-and-1-0-identical]`). The amendment changes `retained` from `true` to
  `false` and changes `reason`, so the check does not reach it.
- `[03-the-package-docstring-still-says-the-registry-deletes-nothing]` stays open. It is not this
  prompt's.

**How the phases are run.** One agent carries all three phases, so it keeps what it measured. You
dispatch it for Phase A and continue it with `SendMessage` for B and C. The log draft lives in the
tree, uncommitted, between phases. If the agent is lost, a fresh one resumes from the draft. The
user's commands take seconds, not hours, so the run registry's launch rules do not apply. But
nothing is left waiting unattended either: if the user is away, the sequence waits at a hand-back,
where nothing is half done.

## 1. Before you dispatch

1. Record the branch and `HEAD`. `HEAD` must be the commit that added this file and the prompt.
2. `python -m RunRegistry list`: nothing `running`. The two `unknown` entries, `a3-pilot` and
   `a3-pilot-resume`, predate the registry and are not runs.
3. **Your own snapshot, read-only.** Record everything under `var/datastores/` (size, mtime and
   SHA-256 of every file, and the listings), everything under `var/runs/` (size and mtime), and the
   repository root's `physics-test-n20-lambdacdm-zend0p1*.sqlite`. Reuse `orch_04_snapshot.py` if
   this session has it; otherwise write it again in the session scratchpad, under an `orch_`
   prefix, never in the tree. This snapshot is independent of the agent's before-picture, and
   every later check is against it.
4. **No suites.** The prompt changes no code, and check 1 of §4 proves that. The board's §5 says
   the suites were not re-run, and why.
5. `git status` is clean, apart from untracked paths that are not this campaign's (today
   `docs/datastore-integrity-audit*` and `prompts/datastore-integrity/`). Tell the agent to leave
   them alone.

## 2. Dispatch Phase A

One fresh-context subagent (**Opus**), in the foreground, since your next step depends on it. Give
it the prompt, the campaign README, the audit, logs 03 and 04, `HEAD` and your snapshot's path
(read-only, for reference). Tell it plainly:
- **it runs nothing that writes under `var/`.** The prompt's §0 lists what it may run. Anything
  else, including `store retire` without `--dry-run`, `store amend`, `store fingerprint --write`
  and `begin`, is the user's or nobody's;
- **never `--without-fingerprint`;**
- it works in phases, and **ends Phase A by handing back**. It does not wait, poll or go on;
- the log draft is written as it goes, in the tree, and is not committed until Phase C;
- **one commit, at the end of Phase C, and none before;**
- it does not touch `orch_*` files, or untracked paths that are not its own;
- the prompt's §5 stop conditions mean *stop and hand back*.

## 3. The user steps, and your checks between them

**Before step 1**, check the agent's Phase A yourself:
- both dry runs exit 0;
- both `fingerprint check:` lines are `matched`, with the digests in the prompt's §1 table;
- the backup's `files to be deleted:` are all inside `backup-pre-resume-20260921T091011/`;
- the references reports match the prompt's prediction, or every difference is explained;
- your snapshot, re-taken, is identical to §1.3's.

Then show the user the following, and ask the user to run the sweep's command and report its exit
code and last line:
- each dry run's output, trimmed to the four sections;
- the two reasons;
- the two commands.

The user may reword a reason. If they do, continue the agent to repeat that dry run with the final
text, before the user runs the command.

**Check 1, after the sweep.** Re-take your snapshot and diff it against §1.3's. The differences
allowed are exactly:
- the sweep's four shards and primary, which are gone;
- the sweep's sidecar, which has changed;
- the mtime of `var/datastores/` itself.

The live A3 store's five files and sidecar, and the backup's six files, must be identical. Run
`store show` on the sweep's primary: it exits 0, and the reading is a completed tombstone. Only
then ask the user to run the backup's command.

**Check 2, after the backup.** The same, against check 1's snapshot. The differences allowed are
exactly:
- the backup's four shards and primary, which are gone;
- its sidecar, which has changed;
- the mtime of its directory.

**The live A3 store's five files and its sidecar must be byte-identical, and their mtimes
unchanged.** Run `store show` on the backup's primary: it exits 0 and reads as a completed
tombstone.

**If either command fails.**
- **Exit 1 with "Nothing was written or deleted"** is a refusal. Stop the sequence, and bring the
  message to the user. It is not retried with a flag.
- **Exit 1 with "failed at step"** is an interrupted retirement. Continue the agent into Phase B,
  which reads the state and hands back the remedy: the same command, with the same reason. The
  user runs it. Nothing else is tried.

**Continue the agent into Phase B** (`SendMessage`, with your two checks' results and the user's
exit codes). When it hands back the amendment, check two things:
- the `--json` value parses;
- it keeps `path`, and sets `retained` to `false`.

Then show the command to the user. The value is the user's judgement (D5), and the user may
change it. If they do, have the agent check again that the new value parses and differs from the
current one.

**Check 3, after the amendment.** Against check 2's snapshot, the one difference allowed is the
live A3 sidecar. Diff that sidecar's JSON:
- `backup` is the new value;
- `history` gains exactly one `amend` entry, whose `before` is the old `backup`, verbatim;
- nothing else differs.

The live store's five files are still byte-identical to §1.3's.

**Continue the agent into Phase C.**

## 4. The review — eight checks, on the Phase C commit

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only:
   - `logs/05-retire-the-two-stores.md` and `logs/05-tombstones.json`;
   - this campaign's board;
   - `prompts/run-registry/IMPLEMENTATION_STATE.md` and `prompts/handover/IMPLEMENTATION_STATE.md`;
   - `docs/OPEN_ISSUES.md`, only if an issue was opened.

   No `.py` file changed. `docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md` is unchanged. The
   `run-registry` and `handover` hunks are **additions only**:
   `git diff HEAD~1 HEAD -- prompts/run-registry prompts/handover | grep '^-[^-]'` is empty.
2. **The live A3 store.** Its five files' SHA-256, size and mtime in your final snapshot equal
   §1.3's. This is the check the prompt exists to pass. Confirm that the log's final hashes agree
   with yours.
3. **The tombstones.** `store show` on each retired primary exits 0 and reads as a completed
   tombstone:
   - state `retired`, with `completed` set;
   - the reason as run;
   - the condition `matched`, with the recorded digest;
   - `files` equal to its dry run's list;
   - the history ends in one `retire`;
   - the `fingerprint` field is value-identical to §1.3's copy of the sidecar.

   `logs/05-tombstones.json` equals the three sidecars on disk, by `json.load`.
4. **The amendment.** The live sidecar differs from §1.3's copy only in `backup` and in one
   appended `amend` entry, whose `before.value` equals the old `backup`, verbatim. `store show`
   exits 0, with `kind: registry` and `problems: none`.
5. **Nothing else under `var/`.** Your final snapshot differs from §1.3's by exactly the ten
   deleted files, the three changed sidecars and the two directory mtimes. `var/runs/` is
   identical, `BACKUP_PATH` included, and so is the `physics-test-n20-*` store.
   `python -m RunRegistry list` still shows nothing `running`.
6. **The log.** It holds:
   - Phase A's before-picture, `list` output and dry-run outputs, verbatim, with the backup's
     resolver notice quoted;
   - the check against the prediction;
   - the user's commands and their exit codes;
   - Phase B's `store show` outputs and field-by-field check;
   - Phase C's amendment check;
   - the live store's hashes before and after.

   Deviations are classified. The deliberate-breakage section says there is no code, so there are
   no mutations.
7. **The records.**
   - **This board:** the §1 row for 05, R10–R12 and §5 are updated, and the header says the
     campaign is complete.
   - **The `run-registry` note** is dated. It explains `var/runs/a3-pilot/BACKUP_PATH` without
     editing it.
   - **The `handover` note** is dated. It says the sweep store is retired, and that
     `QUADSOURCE-TOLERANCE-SWEEP.md:15-17` is now true, unedited.
   - **`docs/OPEN_ISSUES.md`** changed only if an issue was opened, with the count and the date
     right.
8. **The commit.** One commit, in `CLAUDE.md`'s form. The tree is clean afterwards, apart from the
   untracked paths that are not this campaign's.

**After the review.** Record it on the board, in its own commit, as for 01–04. With 05 landed,
the campaign is complete. Its open issues stay open on its board §3, and in the index at §1.14.
Closing the campaign is a separate task, as `store-fingerprint`'s was (`42d4910`).
