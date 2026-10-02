# Orchestrator — prompt 00, the readiness audit

Read [`../README.md`](../README.md) first: §0, §2, §3, §4 and §5. Read also `CLAUDE.md`'s
"Long-running jobs — the run registry". Nothing here is long-running, but its liveness rule binds
the probes' process checks.

**You do not write code.** You may:
- re-run any committed probe, to reproduce its output;
- read the A3 baseline read-only, to check the agent's S4 figure;
- list processes, and kill only those you can show this session started.

**The prompt:** [`00-audit-the-readiness-defects.md`](../00-audit-the-readiness-defects.md)
**Board items:** R1–R3, S1–S4, V1–V2, M1–M2 · **Closes:** nothing · **Widens or narrows:**
`[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` and
`[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]`, on the
`datastore-integrity` board · **Model:** Opus
**Gate:** none. `HEAD` is `b3eed5f` (*"Plan the A3 v2 readiness campaign"*) or the commit that adds
this note. 01, 05 and 06 may land before 00, but **not while it runs**: one prompt at a time in
this checkout.

## 0. What makes this prompt unusual

**It changes no code, but it runs real Ray.** The S probes start local Ray instances, send
themselves `SIGTERM`, and in one case are *meant* to be killed by Ray's abort. Prompt 10's version
of this left a `gcs_server` (pid 6426) running with parent 1 until it was killed with `SIGKILL`. So
the review checks process hygiene as hard as the findings: no Ray process the agent started
survives, and **no process it did not start is touched**.

**It touches `var/` once, read-only.** S4 times a fingerprint of the A3 baseline,
`var/datastores/handover-A3-baseline-lambdacdm*`, the only comparator until A3 v2 exists. Two traps:
- `Run.finish(fingerprint=True)` calls `fingerprint_store(results, write=reading.ok, …)`, which
  **writes the sidecar**. S4 must call `RunRegistry.stores.fingerprint_store(primary, write=False)`
  directly, and must not `begin` a run, which would create a run directory under `var/runs/`.
- The store predates `datastore-integrity` 07–09. Its sidecar's fingerprint is `fingerprint_format`
  1, and the tree is at 6. The current fingerprint may refuse it, or report absent columns.
  **That is a finding, not a stop.** The agent records what happened and how long it took.
  - **The prior:** `prompts/store-fingerprint/logs/05-fingerprint-the-real-stores.md`, Phase A,
    measured **6.66 s wall** (7.0 s with
    `--listing`) for the whole fingerprint command on this store, imports included, on the
    2026-09-25 tree. That is already twice Ray's roughly 3 s abort. It is probably why prompt 10's
    `resumed-b`, `SIGTERM`ed with 379 computes in flight, recorded nothing.
  - If HEAD's fingerprint refuses, the log quotes log 05's figure as the bound and says so. It does
    not read the store at its own SHA to get a number.

**What the review exists to establish.** All else is mechanics.
- **(A1) Measured is measured.** Every finding is marked measured, read or inferred (prompt §2).
  A measured one has a committed probe that reproduces it. An inferred one says what it was
  inferred from. Nothing becomes measured because the argument is good. Prompt 10's two inferences
  are the ones to watch: that Ray's Cython signal check turns `SystemExit(15)` into `SystemExit(1)`,
  and that the abort is absl's alarm.
- **(A2) An instrument that counts zero has first been shown to count one.** R2 and M2 report write
  counts on a full store, which should be small and may be zero. Each probe must first count a
  known write (a deliberate `object_store`, or one `object_get` miss), or its zero means nothing.
- **(A3) The scope of `[08-…]` is settled.** R1's table covers all six scripts,
  `build_model_list`, and every helper they call. R2 measures at least one script's whole setup
  sequence, not just the `GkSourcePolicy` pair.
- **(A4) What 02, 03 and 04 rest on is stated, with any user decision and a recommendation.** The
  audit's closing section is what the orchestrator writes 02–04 from. If it is vague, they cannot
  be written.

**Carried from `datastore-integrity`'s reviews.** All bind the agent, and the brief says so.
- **Staging and reporting.**
  - Stage by explicit path only, never `git add -A`, `.` or `commit -a`. Show
    `git diff --cached --name-only` before committing.
  - Another session may write in this checkout; stop and ask if one modifies a file the agent needs.
  - **Count, then paste.** Paste probe output unfiltered.
  - A log states only what the diff and the probes show. Line numbers come from `git show`, not a
    working-tree view, and the two production files are cited by function name (README §5).
- **Load-sensitive tests.** On 2026-10-02 at 12:03 the load average was 6.4 / 10.8 / 15.2 on 10
  CPUs. Under heavy load the known flake `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
  `test_background_tau.test_build_cost` and `test_numeric_phase_cut.test_the_print_is_gated_both_ways`
  fail. Re-run a failure alone before calling it real. Load also stretches S2's timings: the abort
  window is measured as a window, not a point.
- **Residue and scratch.**
  - Small residue found in review is the orchestrator's, fixed in a follow-up commit (the user,
    2026-09-29).
  - The agent works in its own scratch subdirectory, never the session scratchpad's top level.
    Probes are committed under `docs/a3-v2-readiness/`, never left in a scratchpad.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. If
  it commits but sends no report, review from the commit, its log and the probe outputs.
- **What exists to build on.**
  - `Datastore/tests/standin_pool.py`: a real `ShardedPool` on stand-in shards. Its hooks see
    every actor call, and `in_flight_records(pool)` reads the primary's record. This is R2's and
    M2's instrument.
  - `Datastore/tests/real_store_fixtures.py`: a store the real factories wrote, in a temporary
    directory.
  - `Datastore/store_reader.py`: `read_only_url`, and the journal refusal.

  The probes import these; they do not copy them.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` are both empty.
   - `git worktree list` shows this checkout alone.
2. **The registry.** `python -m RunRegistry list`: nothing `running`. At `b3eed5f` there were 12
   runs, 10 finished and 2 `unknown` (`a3-pilot`, `a3-pilot-resume`, which have no manifest).
   Report any change, without acting on it.
3. **Ray and the machine.**
   - `RAY_ADDRESS` is unset, and `./venv/bin/ray status` finds no cluster. `ray` is not on `PATH`;
     use the venv's.
   - `pgrep -lf 'gcs_server|raylet|ray::'` is empty. It was at 12:03 on 2026-10-02. **If it is not
     empty, do not dispatch.** The agent could not tell its own processes from another session's.
     Ask the user.
   - Record the Ray and Python versions (2.43.0 and 3.12.14 on 2026-10-02), the load average and
     `sysctl -n hw.ncpu` (10).
4. **The snapshot.** Snapshot `var/datastores/` and `var/runs/` read-only, with `orch_00_snapshot.py`
   in the session scratchpad. Record the SHA-256, size and mtime of every file, and the directory
   list. Open files only as bytes, never as databases. **Nothing may change**, the A3 baseline's
   five files above all. Record also that no `-journal`, `-wal` or `-shm` file exists beside any of
   them.
5. **Baselines** (README §7, at `f4f1896`; nothing since touches code): AdaptiveLevin 32,
   ComputeTargets 665, CosmologyModels 39, Datastore 551, LiouvilleGreen 148 (1 skipped),
   RunRegistry 199. Re-measure all six on `HEAD`, serially, with README §7's command.
6. **The index.** Count its rows (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`) and tell the agent the
   number. It is **104** at `b3eed5f`.

## 2. Dispatch

One fresh-context subagent, on Opus. Give it:
- the prompt;
- the campaign README;
- this board;
- `docs/datastore-integrity-verification.md` §2.1.1, §2.3 and §2.6, and the two probes and
  `sigterm_probes.txt` beside it;
- `HEAD`, the baselines, the index count, and the versions;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is at `logs/00-audit-the-readiness-defects.md`, in README
  §5.1's form, with the prompt's §7 additions. "Deliberate breakage" becomes the record of each
  instrument counting a known write (A2), since an audit has no mutations.
- **The files it may add or change:** `docs/a3-v2-readiness-audit.md`; probes and their `.txt`
  outputs under `docs/a3-v2-readiness/`; the log; this board; the `datastore-integrity` board's
  entries for `[08-…]` and `[10-a-sigterm-…]` (a dated **Widened** or **Narrowed** line, nothing
  rewritten); `docs/OPEN_ISSUES.md` (§1.17's hooks, and rows for anything it opens). **Nothing
  else.** No production file, no driver, no test, nothing under `orchestrator/`.
- **Processes.**
  - Each S probe records the Ray processes before it starts and after it ends. Everything new is
    its own. Kill only those, `kill` and then `kill -9`, with each pid in the probe's output.
  - Never touch a process that was present before.
  - `pgrep` is used here only to list processes, never to decide whether a run is alive
    (`CLAUDE.md` rule 2).
- **`var/`.** S4 calls `fingerprint_store(<A3 primary>, write=False)` and nothing else. It never
  calls `Run.finish`, `RunRegistry.begin`, `store` subcommands that write, or `ShardedPool`, on
  that store. It shows the five files' SHA-256 before and after.
- **Stop and ask** on any of the prompt's §5 conditions. The one most likely to fire is the third:
  Ray's abort cannot be stopped from Python. If so, the agent stops after S, with S1–S4 written up,
  and the user decides what 04 becomes.

## 3. The review — nine checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's list.
   - `git diff HEAD~1 HEAD -- '*.py' ':!docs/a3-v2-readiness/'` is empty.
   - The `datastore-integrity` board diff adds lines only (`--numstat`): no deletion.
2. **Processes and `var/`.**
   - `pgrep -lf 'gcs_server|raylet|ray::'` is empty, and `ray status` finds no cluster.
   - Every probe output ends with its before/after process lists and the pids it killed.
   - Re-take §1.4's snapshot. Every file is byte-identical, and no journal file has appeared.
3. **(A1) The marks.** Read every finding in the audit. Each carries measured, read or inferred.
   For each measured one, find the probe and the output line. For each inferred one, the source is
   named. In particular, S2 says which of these it established: what the abort is, and whether
   `alarm(0)` stops it. Not "probably".
4. **(A2) The instruments.** Re-run the R2 and M2 probes yourself. Each shows its instrument
   counting a known write before reporting a count on the full store, and your output matches the
   committed `.txt` row for row. A probe that needs Ray is a §5 violation for R and M.
5. **(A3) R1 is complete.** Independently, `grep -n 'pool\.' extract_*.py config/model_list.py` and
   every helper they call. Every call appears in R1's table with its class and its insert-on-miss
   answer. Compare the replicated column with `config/sharding.py` yourself.
6. **S, re-run once.** Re-run S1 and S2 yourself, in the order the log ran them, with the process
   check before and after. Your exception type and code match S1's. S2's survival times fall inside
   its recorded window.
7. **S4.** The figure, the store's size, and what HEAD's fingerprint did with a format-1 store
   (succeeded, refused, or reported problems). If it refused, log 05's 6.66 s is quoted as the
   bound, with the reason. Either way: is the time inside or outside Ray's abort window? That is
   what 04 needs to know.
8. **(A4) The closing section.** For each of 02, 03 and 04 it gives:
   - the facts its method rests on, each pointing at a finding;
   - any user decision, with a recommendation.

   For 03, M1's table says, for every write at open, skip, refuse or keep under `read_only=True`,
   with the reason. A row reading "to be decided" is a decision for the user, and is listed as one.
9. **Records and suites.**
   - The board: the §1 row for 00, R1–M2, §3 for anything opened.
   - The two `datastore-integrity` entries: dated lines that say what changed, and the index hooks
     in §1.17 changed with them.
   - The index count is 104 plus any opened, by counting.
   - All six suites run serially in the checkout, every count unchanged. `black --check` is clean
     on every new Python file.

## 4. After the review

- **Record it on this board**, after 00's row, in `datastore-integrity`'s review form. Fix any
  small residue in a follow-up commit of its own.
- **Then report to the user.** Say:
  - what R found: how many writes a reader makes, and of which classes;
  - what S found: whether a terminal state can be written before Ray's abort, and how;
  - what V and M found;
  - each decision the audit puts to them, with its recommendation.
- **On the user's decisions, write 02, 03 and 04** against the audit, in one commit, recording the
  decisions in README §6.2 and on the board, and releasing them there.
- Each of 02–04 gets its orchestrator note just before its dispatch, carrying this review's
  findings.
