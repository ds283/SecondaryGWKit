# Orchestrator — prompt 01, the phase-group census of A3 v2

**You do not write code.** You dispatch one fresh-context subagent, review its records against the
fixed checks in §3, and report. The arrangement is `handover` 10's
([`prompts/handover/orchestrator/prompt-10.md`](../../handover/orchestrator/prompt-10.md)); read it
for the pattern, but take every fact from this note.

**The prompt:** [`prompts/levin-phase-floor/01-phase-group-census.md`](../01-phase-group-census.md)
**Board item:** 01 · **Closes:** nothing · **Model:** Opus, as the prompt recommends.

## 0. What makes this dispatch unusual

**You run on the Mac Studio.** You are a Claude Code session over the saved `MacStudio` SSH
connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`, started by the laptop session that
wrote this note. You dispatch the executing agent **in that checkout**, and review its records
there, where the store is. **You commit nothing.** The bookkeeping commit is made on the laptop
afterwards (§4).

**A second session runs at the same time, in the same checkout.** It is `handover` prompt 11, A3
v2's seam record, dispatched together with this one at the user's request. Neither prompt uses Ray,
takes the lock, or writes outside its own `var/` directory. Both open A3 v2 `mode=ro` only. So they
cannot interfere, **except through git**:
- **The laptop orchestrator has already fast-forwarded the checkout** to the dispatch SHA, just
  before starting both sessions.
- **The agent must not run `git fetch`, `git merge`, or any other git command that writes.** Two
  sessions taking `.git`'s locks at once can fail.
- **Prompt step 1 becomes a check:** `HEAD` is the dispatch SHA and `git status --short` is empty,
  with the `pip freeze` recorded once. The agent records this as a `STRUCTURALLY REQUIRED`
  deviation: parallel dispatch.
- The other session's directory is `var/handover-11/`. **Neither session reads or writes the
  other's.**

**Facts checked by the orchestrator on 2026-10-25,** on the Mac Studio over `ssh macstudio-tunnel`,
read-only. Pass them on.
- **The checkout:** `handover-remedial` at `c9026af` before the fast-forward, `git status --short`
  empty. `venv` is ignored by `.gitignore:7`. Python 3.13.16.
- **The registry:** nothing `running` or stale. `~/.raypoolkit/` is empty, and
  `~/ScienceData/.rehearsal-lock` does not exist.
- **`var/`** holds `datastores`, `handover-06`, `handover-07`, `handover-08`, `handover-10`,
  `py313-venv-02` and `runs`. There is **no `var/levin-phase-floor-01/`**. 117 GiB are free.
- **The store:** `var/datastores/handover-A3-v2-lambdacdm.sqlite`, four shards
  `-shard000[0-3].sqlite`, and the sidecar `handover-A3-v2-lambdacdm.manifest.json`. Fingerprint
  `0e32f24a…`.
- **The group records are there.** The orchestrator read the depth-20 rows over 100 s with
  [`measure/recon_groups.py`](../measure/recon_groups.py) (output
  [`measure/out/recon_groups.txt`](../measure/out/recon_groups.txt)):
  - 629 rows, 282.9 CPU-hr;
  - `G+r` 52.7 %, `G-r` 33.5 %, the lone `G` 6.3 %, the four `Gqr` groups 7.6 %.

  The agent's census must reproduce these numbers on the same population (prompt step 5).
- **The sign convention.** The prompt has the agent confirm it from the code before the census,
  and it is a stop if the code disagrees: `G+r` is the slow group on `r == k`, by
  `prompts/source-remediation/logs/07-phase-group-algebra.md` note 4. Check that the agent's
  reading cites file and line, not the log note alone.
- **Shell.** Every command that takes arguments from a variable runs through `bash`, not zsh.

## 1. Before you dispatch

1. **The Gate**, re-read by you:
   - `git rev-parse HEAD` is the dispatch SHA, and so is `origin/handover-remedial` as last fetched;
   - `git status --short` is empty;
   - `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale;
   - `store show var/datastores/handover-A3-v2-lambdacdm.sqlite` reports live, fingerprint
     `0e32f24a…`;
   - there is no `var/levin-phase-floor-01/`.

   Anything else is a stop. Do not fetch.
2. **The `var/` marker:** immediately before dispatch, `touch` a file in your session scratchpad,
   for check 1.
3. **No suite baselines.** The prompt changes no tracked file.

## 2. Dispatch

One fresh-context subagent, on **Opus**, in the Mac Studio's checkout. Give it:
- the prompt;
- the dispatch SHA;
- §0 of this note.

It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.** Orchestration commits land on this branch.
- **Commit nothing, and run no git command that writes**, `fetch` included (§0).
- **Write only under `var/levin-phase-floor-01/`.** A write anywhere else is a stop, and that
  includes `var/handover-11/` and every file of the store.
- **Save every command and its output**, as the prompt says. The review reads the outputs, not the
  log's summary of them.
- **Write the three predictions into the log before running the census**, copied from prompt §1
  unchanged. Score them afterwards against those words.
- **Stop and ask** at any of prompt §4's conditions.
- **The census should take minutes.** If it is in the background, wait for the notification and do
  not poll.

## 3. The review — on the Mac Studio, after the agent reports

1. **Scope.**
   - `git status --short` is empty, and `HEAD` is the dispatch SHA.
   - `find var -newer <marker>` lists only `var/levin-phase-floor-01/` and `var/handover-11/`, the
     other session's. Nothing else: not the store's files, not its sidecar.
2. **The store.** `diff` `state/store_files_before.txt` against `store_files_after.txt`: identical.
   Re-run `store fingerprint` without `--write` yourself; it must match the sidecar's `0e32f24a…`.
   There must be no journal beside any store file.
3. **The population.** `census_01.txt` reports 9280 rows at `(1e-32, 1e-7)`. Count `groups_01.csv`'s
   distinct `(shard, serial)` yourself.
4. **The reconnaissance reproduced.** Re-derive the four shares of §0 from `groups_01.csv` yourself,
   on depth-20 rows over 100 s; a few lines of Python with `-I` will do. They must equal
   `recon_groups.txt`, or the log must explain the difference.
5. **The sign convention.** The log cites code (file and line) for which label is slow, and it
   agrees with prompt §1.
6. **The predictions.**
   - The log's "Predictions and verdicts" quotes P1–P3 word for word from prompt §1.
   - Each is scored holds, fails or undecided, with the number it was scored on.
   - The numbers match T4 in `census_01.txt`.
   - **Recompute P1's $S$ yourself** from `groups_01.csv`, using the cancelling flag as the prompt
     defines it. Check the flag's definition in `group_census.py` against prompt §1 by reading the
     code.
7. **Findings §2.5.** The non-finite rows are identified to group level. The 1086 / 257 / 132 /
   5.0e-2 counts are recomputed, and agree, or the difference is explained.
8. **The candidates.** `candidates_01.txt` holds at least two rows for each of the four classes, or
   says which class has none.
9. **The records.**
   - `log.md` follows campaign README §5.1 and classifies every deviation. Step 1's no-op is one.
   - `verification.md` is written as `docs/levin-phase-floor/VERIFICATION.md` §1, with its links
     relative to `docs/levin-phase-floor/`.
   - `issues.md` exists, empty or not.

Write your verdict on each check to `var/levin-phase-floor-01/review.md`, with the commands you ran
and what they printed. End the turn with a report:
- pass or stop;
- P1–P3 with their verdicts;
- $S$;
- the fingerprint.

## 4. After the review — on the laptop

The laptop session is woken when this one ends, and does the bookkeeping. It copies
`var/levin-phase-floor-01/` back with `rsync -a`, gzipping `groups_01.csv` if it is over 10 MB.
It re-checks one thing over `ssh macstudio-tunnel` itself: check 4's shares. Then, in **one
commit**, pushed, and separate from 11's:
- everything to `docs/levin-phase-floor/01/`;
- `verification.md` as the new `docs/levin-phase-floor/VERIFICATION.md`, §1;
- `log.md` to `logs/01-phase-group-census.md`;
- the board: 01's row, the hypotheses table's state, and anything `issues.md` opens;
- the narrowings on the boards that hold them (`handover`, `GkTk-remedial`, `levin-refactor`);
- `docs/OPEN_ISSUES.md` to match, with the count and date corrected by counting.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. Stop on the prompt's §4 conditions, or on any failed check in §3
here: report it, do not repair it. **Never write to the store**, and never run a git command that
writes.
