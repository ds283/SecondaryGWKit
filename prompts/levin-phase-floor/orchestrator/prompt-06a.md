# Orchestrator — prompt 06a, the depth census of every A3 v2 row at the current code

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code.** The harness is the agent's.

**The prompt:** [`prompts/levin-phase-floor/06a-depth-census.md`](../06a-depth-census.md)
**Board item:** 06a · **Closes:** nothing by itself. The user's L2 decision, made on its records,
closes or narrows `[04-x-at-1e-8-still-reaches-depth-20]`, `[03-depth20-regions-are-all-clenshaw-curtis]`
and `[05-b1-uncaps-most-cheap-depth-20-rows]` · **Model:** Opus for the agent, as the prompt
recommends. You, the orchestrator, review on whatever model you were started with.

## 0. The shape, and why

- **One part, on the Mac Studio.** The prompt changes no code, so there is no laptop part and no
  code commit. There is one executing agent, which you dispatch and then review. The laptop makes
  the one records commit afterwards (campaign README §4.1).
- **One tree, a worktree** at the dispatch SHA. Its code is 06's $s$ tree's (`f5cee4c`) byte for
  byte, and the prompt's Gate checks that. The main checkout supplies the venv, the stores, `var/`
  and the registry. **It does not move**, and no tree runs from it.
- **No Ray and no lock.** The census reads 06's certified scratch copy of A3 v2 through 06's
  stand-in pool, in-process. Other projects may take the shared lock meanwhile, and that does not
  concern this prompt. A Ray session under `/tmp/ray-sgk` appearing during the run is a stop.
- **At most six worker processes** (the Mac Studio's `--cpus 6` rule: it also serves a website's
  object store).
- **07 may be written on the laptop meanwhile.** It touches nothing here. Its rehearsals, which
  take the lock and six cores, wait until this run is over.

### 0.1 What 05, 05a and 06 taught, which this note carries

- **Every deviation is numbered and classified before the agent reports,** failed attempts
  included.
- **A question relayed to the user carries the state of the numbers.** Before relaying, check what
  the run has already written, and say so. In 05a the laptop told the user no number existed when
  one did.
- **No fact in this note is to be trusted over the store or the code.** 06's note said `GkSource`
  stores `z_init` null for a pure-WKB value. It does not, and the run found that out late. The
  agent checks what it relies on.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest`
  and work counters (`[05-store-value-diff-compares-volatile-columns]`). The census compares
  `QuadSourceIntegral` results unit by unit, not stores, so this matters only if the agent reaches
  for `tools/store_value_diff.py`, which it should not need.

## 1. Facts, checked 2026-10-27 at 02:07 UTC, read-only

Re-checked by the laptop at dispatch, and corrected in an addendum if they have moved.
- **The main checkout** is `handover-remedial` at `28ebaf0`, `git status --short` empty, one
  worktree, itself. 06's two worktrees have been removed.
- **The host:** 10 cores and 32 GiB, 108 GiB free on the data volume. The load is about 6. That is
  macOS's `suggestd`, ChronoSync, `fileproviderd` and OrbStack, not Python. `~/.ray/auth_token` is
  mode 600.
- **The registry** holds no run `running` or stale. 06's three runs are `done`, and so are both of
  A4's runs (the first ended `failed`).
- **The lock is free:** `status` says `~/.raypoolkit/rehearsal-lock` does not exist, and
  `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none.
- **`var/`** holds `datastores`, `handover-03`, `-04`, `-06`, `-07`, `-08`, `-10`, `-11` and `-12`,
  `levin-phase-floor-01`, `-03`, `-04`, `-05`, `-05a` and `-06`, `levin-phase-floor-measure`,
  `py313-venv-02` and `runs`. There is no `levin-phase-floor-06a`.
- **The scratch copy** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, five files. Its
  certification, `var/levin-phase-floor-06/wider/state/copy_files_certified.txt` (written
  2026-10-26T22:22:28Z), gives the primary as `1542e05c…` and the shards as `bb007a4c…`,
  `aae63ccd…`, `5e2bda9b…` and `ecc3330d…`.
- **06's run** `var/runs/levin-phase-floor-06-wider-20261026T223634/` holds 124 files. It contains
  `checkpoint.jsonl`, whose units C0 and C4 compare against, and the unit pickles holding their
  region arrays.
- **06's per-unit timings**, read from that checkpoint, are what the prompt's §3.1 estimate is
  built on:
  - each `s` unit's re-run took about 41 s, of which 36.8 s was `G_compute_store` (about 1,500
    objects) and 2.9 s factory lookups;
  - `GkSource` took 0.08 s, `GkSourcePolicyData` 0.06 s and $T$ 0.15 s;
  - Levin wall time had a median of 0.30 s at `1e-7` and 0.41 s at `1e-8`.
- **A3 v2** is `var/datastores/handover-A3-v2-lambdacdm{,-shard000[0-3]}.sqlite`, digest
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.

## 2. The dispatch — done by the laptop

When §1 still holds, the laptop does the following. It is the only party that writes git on the
Mac Studio.
1. It pushes the dispatch commit: the prompt, this note and the board's 06a row marked written.
2. It runs `git fetch origin` in the main checkout. That moves no working tree.
3. It makes `git worktree add --detach ../SecondaryGWKit-lpf06a-<short SHA> <dispatch SHA>`, with
   no venv in it.
4. It starts you: a Claude Code session over the saved `MacStudio` SSH connection, in the main
   checkout, told the dispatch SHA and the worktree's path. You read this note and the prompt
   **from the worktree**.

## 3. What you do

**Your Gate.** Re-read §1's state yourself, read-only, and stop if anything differs:
- the main checkout at `28ebaf0`, clean;
- the worktree at the dispatch SHA, `git status --short --ignored` empty;
- the registry, the lock and `var/`;
- the copy's five SHA-256 against the certification.

Do not fetch. Then `touch` a marker file in your scratchpad, immediately before the dispatch.

**One agent, fresh-context, on Opus.** Give it:
- the prompt, all of it in scope;
- the dispatch SHA and the worktree's path;
- §1 of this note;
- what follows.

It does not read `orchestrator/`.
- **Where it writes:** only `var/levin-phase-floor-06a/` in the main checkout, and its own run
  directory under `var/runs/`. Nothing in the worktree, and nothing in 06's `var/` or run
  directory, which it reads only.
- **How it runs Python:** every command calls `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python`
  by that absolute path. In-tree code runs with `PYTHONPATH` set to the worktree, `-B` and
  `PYTHONDONTWRITEBYTECODE=1`. Copied scripts run with `-I` where they import nothing from the
  tree.
- **No git command that writes,** in any tree, `fetch` included. Read git with
  `--no-optional-locks`.
- **`HEAD` is not its own.** Other sessions commit to this branch on the laptop. The worktree is
  pinned, and the main checkout is not moved by anyone while the run lives.
- **The order:**
  1. the Gate;
  2. C0, all five clauses;
  3. the pilot and its projection;
  4. `state/predictions.md`, the prompt's §4 copied byte for byte;
  5. the launch.

  Each step's output is saved before the next begins. The mtimes are the review's evidence.
- **Launch, one check, end the turn.** It reports:
  - the run id and the run directory;
  - the driver's pid and the `sh` pid;
  - the first checkpoint line;
  - the pilot's projection.

  **You** then wait on the driver's pid once, in the background: no polling loop and no
  `pgrep`. When it exits, you resume the agent with the final `status.json`. If the projection
  was more than about four hours, tell the laptop session before you wait.
- **The over-budget pass** follows the prompt's §3.2. More than twelve over-budget units is the
  agent's stop. Relay it.
- **Its records:**
  - `log.md`, in campaign README §5.1's form, which becomes `logs/06a-depth-census.md`;
  - `verification.md`, which becomes `VERIFICATION.md` §6a "The depth census", with its links
    relative to `docs/levin-phase-floor/` in the `06a/…` form;
  - `issues.md`;
  - `out/`, `scripts/` and `state/`, with every command and its output in `out/cmd/NN_*.txt`.
- **It stops and asks** at any of the prompt's §9 conditions. Relay the question verbatim, with
  the state of the numbers.

## 4. The review — on the Mac Studio, after the agent reports

Use scratch readers of your own, in your scratchpad, never the agent's scripts, unless a check says
to re-run one. Compare floats by bit pattern (`float.hex`), never `isclose`.

1. **Scope.**
   - The main checkout is clean and still at `28ebaf0`.
   - The worktree is clean and at the dispatch SHA, and `find <worktree> -newer <marker>` lists
     nothing.
   - `find var -newer <marker>` lists only `var/levin-phase-floor-06a/` and the run directory.
   - 06's `var/levin-phase-floor-06/` and its run directory are unchanged: same file count, and no
     file newer than the marker.
2. **The run.**
   - `runregistrykit list` shows it `done`.
   - Its manifest names `levin-phase-floor`, `06a` and `census`, with `git_dirty: false`.
   - Its purpose string names the worktree's SHA, the copy's fingerprint, 9,280 rows and both
     rtols.
   - `checkpoint.jsonl` holds 18,560 units, one per (rtol, row), and each row of `state/rows.json`
     appears once per rtol. Over-budget re-runs are recorded beside these, not in place of them.
3. **The tree, read.**
   - Every unit's record carries the worktree's `HEAD` and an empty status.
   - The module evidence of each worker shows every package imported from the worktree.
   - `stdout.log` shows the per-$k$ workers, at most six alive at once (from their start and exit
     lines), with the `1e-7` set finished before the `1e-8` set began.
4. **C0 and the order of work.**
   - C0's five clauses are present in the saved outputs, and every comparison in them is by bit
     pattern.
   - C0, the pilot and `state/predictions.md` are each earlier than the run's `manifest.json`.
   - The predictions file equals the prompt's §4 byte for byte.
   - **Independently:** compare `ctrl-low-1`'s census unit at `1e-7` with 06's `s-1e-7-ctrl-low-1`,
     from the two `checkpoint.jsonl`s. Compare `total`, `total_abserr`, and every call's value and
     abserr in `float.hex`.
5. **The stores.**
   - Every file of A3 v2 and of the copy has the same SHA-256, size and mtime before and after.
   - No journal, `-wal` or `-shm` file sits beside either.
   - **Run the fingerprints yourself**, without `--write`: `runregistrykit store fingerprint` on
     A3 v2 from the main checkout, and the agent's copied `fingerprint_copy.py` on the copy from
     the worktree. Both must give `0e32f24a…`.
   - `raypoolkit.rehearsal_lock status` and `ray-processes --ray-temp-dir /tmp/ray-sgk` show
     nothing of this run. Run each as a command of its own.
   - `/tmp/ray-sgk` gained no session directory after the marker.
6. **C4, re-scored** from the two checkpoints:
   - all 39 of 05's rows at `1e-7`, and the 31 depth-20 rows at `1e-8`, against 06's `s` units;
   - `total`, `total_abserr`, every call's value and abserr, and the per-call region counts and
     maximum depth;
   - one row's region arrays, from both runs' pickles.
7. **C1–C3, re-scored** from the checkpoint by your own reader:
   - each row's depth at each rtol;
   - the count at 18 or more, and the largest;
   - over-budget and errored units, treated as the rules say.

   The verdicts must be the agent's.
8. **The descriptive items** of the prompt's §5, 1–9, are all present in `out/census_summary.txt`.
   - `out/census_rows.csv.gz` has 18,560 data lines.
   - Spot-check the deepest row at `1e-7` against its checkpoint record: depth, regions and
     `total` in `float.hex`.
   - The stationary-point scan's counts are stated with its grid spacing beside them.
9. **The harness.**
   - The diff against 06's copy, saved at launch and at the end, shows only the prompt's §2 items.
     Any other change is a numbered deviation.
   - `convergence_kinds.py` is either byte-identical to `prompts/levin-phase-floor/measure/`'s,
     or its diff is recorded.
   - Every script's SHA-256 in the records matches the file in `scripts/`.
10. **The records.**
    - `log.md` follows campaign README §5.1, gives C1–C4 against the rules as written, and
      numbers and classifies every deviation.
    - `verification.md` is written as `VERIFICATION.md` §6a.
    - `issues.md` exists.
    - The run directory's size is recorded.

Write your verdict on each check to `var/levin-phase-floor-06a/review.md`, with the commands you ran
and what they printed. End the turn with a report whose first line is the verdict, pass or stop.
Then give:
- C0;
- C1–C4, each with its numbers;
- the distribution of row depth at each rtol;
- the deepest rows;
- the stationary-point scan's two counts: calls with an interior sign change, and regions accepted
  across one;
- the over-budget and errored units;
- the run's wall time.

## 5. After the review — on the laptop

The laptop copies `var/levin-phase-floor-06a/` back with `rsync -a`, without pickles or any file over
2 MB other than `census_rows.csv.gz` if it is larger. It then re-checks over `ssh macstudio-tunnel`
itself:
- check 4's comparison on one other row of 05's 39;
- one line of `census_rows.csv.gz` against its checkpoint record.

Then **one records commit**, pushed, containing:
- the records in `docs/levin-phase-floor/06a/`, scripts byte-identical and unformatted. Their
  directory is added to `CLAUDE.md`'s `black` paragraph if any output records their SHA-256. A
  `KEPT_ON_MAC_STUDIO.md` lists what stayed behind, with SHA-256;
- `log.md` as `logs/06a-depth-census.md`;
- `verification.md` as `docs/levin-phase-floor/VERIFICATION.md` §6a;
- the board: 06a's row; §2.3, with the census's numbers added as a new subsection beside 06's
  evidence and not in place of it; and anything `issues.md` opens, on the user's word;
- `docs/OPEN_ISSUES.md` to match, with the count and date corrected by counting;
- this directory's README row for 06a, and the campaign README's.

**The three issues the census bears on are not closed in that commit.** The laptop puts C1–C4 and
the depth distribution to the user for L2. The user's decision closes or narrows them, in the commit
that records it. The worktree is removed with `git worktree remove`, and the scratch copy of A3 v2
is deleted, each only on the user's word.

## 6. Stop and ask the user

Relay verbatim; do not adjudicate. Stop at the prompt's §9 conditions, or at any failed check in
§4: report it, do not repair it.
- **Never write to A3 v2 or its copy,** or to 05's, 05a's or 06's records on the Mac Studio.
- **Never move the main checkout,** never take or release the lock, and never end another
  session's run.
