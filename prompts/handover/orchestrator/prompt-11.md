# Orchestrator — prompt 11, A3 v2's seam record

Read [`README.md`](README.md) in this directory first, and [`prompt-10.md`](prompt-10.md), whose
arrangement this prompt reuses. **You do not write code.**

**The prompt:** [`prompts/handover/11-a3-v2-seam-record.md`](../11-a3-v2-seam-record.md)
**Board item:** 11 · **Closes:** nothing · **Model:** Sonnet, as the prompt recommends. You, the
orchestrator, review on whatever model you were started with.

## 0. What makes this dispatch unusual

**You run on the Mac Studio, as 10's orchestrator did.** You are a Claude Code session over the
saved `MacStudio` SSH connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`. You dispatch the
executing agent **in that checkout**, review there, and **commit nothing**. The bookkeeping commit
is made on the laptop (§4).

**A second session runs at the same time, in the same checkout.** It is `levin-phase-floor` prompt
01, a census of A3 v2's phase-group records, dispatched together with this one at the user's
request. Neither prompt uses Ray, takes the lock, or writes outside its own `var/` directory. Both
open A3 v2 `mode=ro` only. So they cannot interfere, **except through git**:
- **The laptop orchestrator has already fast-forwarded the checkout** to the dispatch SHA, just
  before starting both sessions.
- **The agent must not run `git fetch`, `git merge`, or any other git command that writes.**
- **Prompt §1 step 1 becomes a check:** `HEAD` is the dispatch SHA and `git status --short` is
  empty, with the `pip freeze` recorded once. The agent records this as a `STRUCTURALLY REQUIRED`
  deviation: parallel dispatch.
- The other session's directory is `var/levin-phase-floor-01/`. **Neither session reads or writes
  the other's.**

**Facts checked by the orchestrator on 2026-10-25,** on the Mac Studio over `ssh macstudio-tunnel`,
read-only. Pass them on.
- **The checkout:** `handover-remedial` at `c9026af` before the fast-forward, clean. `venv` is
  ignored by `.gitignore:7`. Python 3.13.16.
- **The registry:** nothing `running` or stale. The lock is free: `~/.raypoolkit/` is empty.
- **`var/`** holds `datastores`, `handover-06`, `-07`, `-08`, `-10`, `py313-venv-02` and `runs`.
  There is **no `var/handover-11/`**. 117 GiB are free.
- **The store:** `var/datastores/handover-A3-v2-lambdacdm.sqlite`, four shards and the sidecar.
  Fingerprint `0e32f24a…`.
- **The build's log** is `var/runs/handover-07-a3-v2-20261022T211155/stdout.log`. Its summary block
  is at lines 89–102:
  - 1160 instances;
  - `numeric` 363, all `complete`;
  - `mixed` 72: 71 `complete`, 1 `minimal`;
  - `WKB` 725, all `complete`.

  The table count must equal it in every cell. `type` and `quality` are stored as integers;
  `ObjectFactories/GkSourcePolicyData.py:13-41` decodes them.
- **The field the gap census reads** is `partition["subintervals"][i]["clamp_gaps_log1pz"]`, **not**
  `max_clamp_gap_log1pz`, which is the allowance (prompt "Read first" item 3).
  `HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5` (`ComputeTargets/QuadSourceIntegral.py:111`).
- **Shell.** Every command that takes arguments from a variable runs through `bash`, not zsh.

## 1. Before you dispatch

1. **The Gate**, re-read by you:
   - `git rev-parse HEAD` is the dispatch SHA;
   - `git status --short` is empty;
   - `runregistrykit list` shows nothing `running` or stale;
   - `store show` reports live, fingerprint `0e32f24a…`;
   - there is no `var/handover-11/`.

   Anything else is a stop. Do not fetch.
2. **The `var/` marker:** `touch` a file in your session scratchpad immediately before dispatch.
3. **No suite baselines.** The prompt changes no tracked file.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**, in the Mac Studio's checkout. Give it:
- the prompt;
- the dispatch SHA;
- §0 of this note.

It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.**
- **Commit nothing, and run no git command that writes**, `fetch` included.
- **Write only under `var/handover-11/`.** A write anywhere else is a stop, and that includes
  `var/levin-phase-floor-01/`, the store's files and `stdout.log`.
- **Save every command and its output.**
- **Stop and ask** at any of prompt §3's conditions.
- **Do not interpret the oracle comparison** beyond what prompt §1 step 4 (d) states it can show.

## 3. The review — on the Mac Studio, after the agent reports

1. **Scope.**
   - `git status --short` is empty, and `HEAD` is the dispatch SHA.
   - `find var -newer <marker>` lists only `var/handover-11/` and `var/levin-phase-floor-01/`, the
     other session's. It does not list the store's files, its sidecar, or the A3 v2 run directory.
2. **The store and the log.**
   - `diff` `store_files_before.txt` against `store_files_after.txt`: identical, the `stdout.log`
     hash included.
   - Re-run `store fingerprint` without `--write` yourself; it matches `0e32f24a…`.
   - There is no journal beside any store file.
3. **The policy summary.**
   - `policy_11.txt` holds the block verbatim. Check it yourself against `sed -n 85,105p` of the
     log.
   - Its table count agrees in every cell.
   - The one `mixed`/`minimal` row is identified by $k$, $z_{\rm response}$ and `crossover_z`.
4. **The population.** 9280 rows. Count `subintervals_11.csv`'s distinct `(shard, serial)` yourself.
5. **The gap census.**
   - The script reads `clamp_gaps_log1pz`. Check by reading the code that it does not read
     `max_clamp_gap_log1pz` as the gap.
   - **Recompute the `Tq` and `Tr` non-zero counts and medians yourself** from the CSV, and check
     them against `seam_census_11.txt`.
   - No gap exceeds its row's allowance.
   - The 2026-09 figures stand beside the new ones.
6. **The oracle comparison.**
   - Seam band only.
   - Strata as prompt step 4 (d) says.
   - Rows excluded for a zero or non-finite `analytic_rad` are counted.
   - The limits are stated in the output and in `verification.md`. Decades below about $10^8$ must
     not be read as the clamp's error.
7. **The records.**
   - `log.md` has every heading of `logs/10-reread-a3-v2.md` and classifies each deviation. The
     step 1 no-op is one.
   - `verification.md` is written as `docs/handover/A3-V2-VERIFICATION.md` **§5 "The seam
     record"**, with links relative to `docs/handover/` (the `a3-v2/seam/…` form; 10's check 9).
   - `issues.md` exists.

Write your verdict on each check to `var/handover-11/review.md`, with the commands you ran and what
they printed. End the turn with a report:
- pass or stop;
- the policy table;
- the `Tq` and `Tr` gap counts and medians beside 2026-09's;
- the oracle table's headline;
- the fingerprint.

## 4. After the review — on the laptop

The laptop session is woken when this one ends. It copies `var/handover-11/` back with `rsync -a`,
gzipping `subintervals_11.csv` if it is over 10 MB. It re-checks check 3's table count over
`ssh macstudio-tunnel` itself. Then, in **one commit**, pushed, and separate from
`levin-phase-floor` 01's:
- everything to `docs/handover/a3-v2/seam/`;
- `verification.md` appended as `A3-V2-VERIFICATION.md` §5;
- `log.md` to `logs/11-a3-v2-seam-record.md`;
- the board: 11's row and item;
- `[12-handover-clamp-error-in-production]` narrowed on the `source-remediation` board;
- the policy result recorded against README §2 (p);
- anything `issues.md` opens;
- `docs/OPEN_ISSUES.md` to match, with the count and date corrected by counting;
- this directory's README row for 11.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. **Never write to the store or to the A3 v2 run directory**, and
never run a git command that writes.
