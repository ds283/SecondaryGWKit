# Orchestrator — prompt 10, re-read A3 v2

Read [`README.md`](README.md) in this directory first, and [`prompt-08.md`](prompt-08.md), whose
arrangement this prompt reuses. **You do not write code.**

**The prompt:** [`prompts/handover/10-reread-a3-v2.md`](../10-reread-a3-v2.md)
**Board item:** 10 · **Closes:**
`[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]`, when the
prompt's acceptance 2 holds · **Model:** Opus, as the prompt recommends.

## 0. What makes this prompt unusual

**The orchestrator runs on the Mac Studio.** 06–08 were orchestrated from the laptop and executed by
a Mac Studio session. Here the user asked for the orchestration itself to be remote: a Claude Code
session over the saved `MacStudio` SSH connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`,
started by the laptop session that wrote this note. It dispatches one fresh-context subagent **in
that checkout** to execute the prompt, and reviews its records there, where the store and the
readers' output are. **It commits nothing.** The bookkeeping commit is made on the laptop
afterwards (§4), as for 06–08.

The prompt itself is short: the checkout moves, `raypoolkit` moves to `v0.3.0`, the lock is taken,
the readers run, the lock is released. Its review turns on two things a reader's exit code cannot
show: that the GkWKB read-back is **complete** (prompt §1 step 6's counts agree, the failing pair
included), and that the store's files are **byte-identical** from before the first reader to
after the last. It is the first time the Mac Studio's checkout moves since A3 v2's launch, and the
first time the `v0.3.0` lock is used there.

**Facts checked by the orchestrator on 2026-10-25** (on the laptop, and on the Mac Studio over
`ssh macstudio-tunnel`, read-only). Pass them on.

- **The laptop:** `handover-remedial` at `6d32683`, clean, equal to `origin/handover-remedial`.
  Nothing `running` or stale in the registry (53 runs: 51 finished, 2 `unknown` pre-registry). The
  lock is free.
- **The Gate's `git diff`.** `git diff --stat b90752d 6d32683 -- ObjectFactories config main.py`
  lists `config/ray_local.py` (new, 7 lines: a comment and `RAY_TEMP_DIR = "/tmp/ray-sgk"`) and
  `main.py` (+8 −2: the import, and `ray.init` gaining
  `_temp_dir=RAY_TEMP_DIR if args.ray_address == "local" else None`). That is exactly the
  amendment's allowance and touches no schema. The commit carrying this note adds nothing to it;
  the agent re-takes it at the dispatch SHA.
- **The Mac Studio:** `handover-remedial` at `b90752d`, `git status --short` empty;
  `git check-ignore -v venv` answers `.git/info/exclude:28:venv`. Five runs in its registry, all
  finished (`handover-07-a3-v2-20261022T211155` `done`). Neither `~/.raypoolkit/rehearsal-lock` nor
  `~/ScienceData/.rehearsal-lock` exists. `var/` holds `datastores`, `handover-06`, `handover-07`,
  `handover-08` and `runs`; **no `var/handover-10/`**. The A3 v2 store is the primary (36,864 B)
  and four shards (~106 MB each), last written 2026-10-24 21:56, with the sidecar
  `handover-A3-v2-lambdacdm.manifest.json`; `store show` reports kind `registry`, problems none.
  119 GiB free. Up 18 days. Automatic macOS update installs read `0`.
- **The venv's three packages**, before the prompt's step 2: `datastorekit` `f47b9a25…`,
  `raypoolkit` `eaef0ce6…` (`v0.2.0`), `runregistrykit` `0bae3a3c…`. `requirements.txt` at the
  dispatch SHA pins `raypoolkit` at `v0.3.0`; `pip_06.txt` line 82 carries the `eaef0ce6…` line
  step 2 replaces.
- **The lock, under `v0.3.0`.** The forms are `CLAUDE.md`'s, with the values
  `--project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk --ray-address in-process`. `status` takes
  **no** `--project`; `ray-processes` takes `--ray-temp-dir` alone. The prompt does not call
  `set-pid`, and the readers launch no registered run, so the owner file's `pid` stays `-`: `status`
  then exits 3 with `owner pid: - (no pid recorded yet)` while the lock is held. That is expected
  and not a stop (RayPoolKit `docs/rehearsal-lock.md` §4 and its `status` table, at `v0.3.0`).
  **Corrected 2026-10-25, after the run:** `status` exits **0, always** (`docs/rehearsal-lock.md`
  §7); the "3" above was the row number of §8's layout table, misread as an exit status. The
  agent saw exit 0 and recorded the difference; it was the note's error, not the package's.
  `release` refuses while a process under `/tmp/ray-sgk` is alive; it lists them.
- **Lock commands: one per shell command, no pipe.** `>> file 2>&1`, then `echo $?` in the next
  command. Never `| tee`, never in the shell command that runs the readers. `--purpose` names no
  temp dir.
- **The readers start their Ray with `_temp_dir`** at the dispatch SHA (`rpk-v030-adoption` 01), so
  `ray-processes --ray-temp-dir /tmp/ray-sgk` lists each one while it runs and nothing after.
- **08's reader record**, for comparison: `docs/handover/a3-v2/after/out/readers_08.txt` and
  `A3-V2-VERIFICATION.md` §3.4. `extract_GkWKB_data` then exited 1 after 24 s with 344 files
  (18.2 MB), on `k = 1e5`, `z_source = 1.3633e16` (serial 19), leaving a 1,039-byte PDF fragment and
  no PNG or CSV; the other readers wrote 1593, 873, 72, 172, 681 and 1593 files, 360 MB in all.
  The other readers' counts and sizes may differ now, since 09 filtered their `analytic_G_w`
  series; a difference is recorded, not a stop.
- **Shell.** Every command that takes arguments from a variable runs through `bash`, not zsh (06's
  deviations 8–10). `run_readers.sh` is already `bash`.
- **Prompt §1 step 1 moves the checkout.** Nothing runs from it: the registry's one live-era run,
  A3 v2, is `done`.

## 1. Before you dispatch

1. **Read this note at the dispatch SHA** without moving the checkout:
   `git fetch origin`, then `git show <dispatch SHA>:prompts/handover/orchestrator/prompt-10.md` and
   `git show <dispatch SHA>:prompts/handover/orchestrator/README.md`. The checkout moves at the
   prompt's step 1, which is the agent's.
2. **The Gate, re-read on the Mac Studio**: `origin/handover-remedial` is the dispatch SHA;
   `HEAD` `b90752d`, `git status --short` empty; `./venv/bin/python -m runregistrykit list` shows
   nothing `running` or stale; neither lock path exists; no `var/handover-10/`. Anything else is a
   stop.
3. **The `var/` marker:** `touch` a file in the session scratchpad immediately before dispatch, for
   check 7.
4. There are **no suite baselines** to take: prompt 10 changes no tracked file. Step 2's
   `pip install` is the only change to the environment.

## 2. Dispatch

One fresh-context subagent, on **Opus**, in the Mac Studio's checkout. Give it the prompt (read
with `git show <dispatch SHA>:prompts/handover/10-reread-a3-v2.md` until step 1 has moved the
checkout), the dispatch SHA and §0 of this note. It does not read `orchestrator/`. Tell it plainly:

- **`HEAD` is not its own**; orchestration commits land on this branch.
- **Commit nothing.** Write only under `var/handover-10/`, plus step 1's fast-forward, step 2's
  `pip install` into `./venv`, and the lock. A write anywhere else is a stop.
- **Save every command and its output** under `var/handover-10/`, as the prompt says; the review
  reads the outputs, not the log's summary of them.
- **Stop and ask** at any of prompt §3's conditions. Stopping with the lock held leaves it held and
  prints the owner file.
- **The readers run in the foreground**: about four minutes in 08, longer now that GkWKB completes.
  No background wait and no poll is needed.

## 3. The review — on the Mac Studio, after the agent reports

1. **Scope.** `git status --short` is empty; `HEAD` is the dispatch SHA, so the agent committed
   nothing. `find var -newer <marker>` lists only `var/handover-10/` and nothing else, the store's
   files and sidecar included.
2. **The step 1 diff.** The agent's saved `git diff --stat b90752d <dispatch SHA> -- ObjectFactories
   config main.py` is §0's two files and no more.
3. **The venv.** The saved `pip install --dry-run` would install `raypoolkit-0.3.0` alone. The
   after-freeze, diffed against `pip_06.txt` by you, differs in the one `raypoolkit` line, now at
   `67a75b97…`; none of the three packages is editable.
4. **The store before.** `store show`: live, fingerprint `0e32f24a…`. `state/store_files_before.txt`
   has the primary and four shards, with SHA-256, size and mtime, and no journal.
5. **The readers.** `out/readers_10.txt` and each reader's own log: seven exits, all 0; no
   `Traceback`, `ReadOnlyMiss` or `ReadOnlyWrite`; "store files: identical" after every one; no Ray
   process left after any. Record each one's files and size beside 08's.
6. **The GkWKB read-back, counted by you.** Under `var/handover-10/extract/extract_GkWKB_data/`,
   count the full-range PDFs, PNGs and CSVs yourself (`find … | wc -l` for each). They equal each
   other and the reader's work list as the agent found it; read how the agent found that number, in
   the reader's code or log, not only the number. The pair
   `k=1e+05-k-serial=1/zsource=1.3633e+16-z-serial=19` has a PDF, a PNG and a CSV, the PDF far above
   1,039 bytes. The dropped-point count is reported, with its scope (store-wide or that pair).
7. **The store after.** `state/store_files_after.txt` equals `store_files_before.txt` line for
   line (`diff`, by you). `store fingerprint` without `--write`, **re-run by you**, matches the
   sidecar's. No journal beside any store file.
8. **The lock.** `out/lock_10.txt`: `acquire` 0; `ray-processes` listing the instance during a
   reader, if the agent took one then; `release` 0; `status` free; `ray-processes` empty. Re-run
   `status` and `ray-processes --ray-temp-dir /tmp/ray-sgk` yourself.
9. **The records.** `log.md` has every heading of `logs/08-after-the-a3-v2-build.md` and classifies
   each deviation (step 2's `raypoolkit` move is the prompt's own amendment, not a deviation).
   `verification.md` is written as `A3-V2-VERIFICATION.md` §4, "Re-read with the fixed readers",
   with links relative to `docs/handover/`. `issues.md` exists, empty or not.

Write your verdict on each check to `var/handover-10/review.md`, after check 1 has been taken, with
the commands you ran and what they printed. Then end the turn with a report: pass or stop, the
GkWKB counts, the fingerprint, the lock.

## 4. After the review — on the laptop

The laptop session that dispatched this one is woken when it ends, and does the bookkeeping. It
copies `var/handover-10/` back without `extract/` (`rsync -a --exclude extract/`), and checks one
thing over `ssh macstudio-tunnel` itself: the GkWKB counts of check 6. Then, in one commit, pushed:

- `out/`, `state/`, any `scripts/` and `review.md` to `docs/handover/a3-v2/reread/`;
- `log.md` to `logs/10-reread-a3-v2.md`, with 08's header note on where its paths now are;
- `verification.md` as `docs/handover/A3-V2-VERIFICATION.md` §4;
- the board: 10's row and item; `[08-the-gkwkb-reader-fails-…]` moved to §4, closed; anything
  `issues.md` opens added to §3; `docs/OPEN_ISSUES.md` to match, with the count and date corrected by
  counting; this README's 10 row.

`extract/` stays on the Mac Studio. Whether to keep it is the user's decision.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §3 conditions, or any failed check in §3 here:
report it, do not repair it. **Never write to the store**, never re-run a build, and never remove a
lock: a lock that will not release is the user's.
