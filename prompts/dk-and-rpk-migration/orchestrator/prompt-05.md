# Orchestrator — prompt 05, rehearse through `datastorekit` (R1)

Read [`../README.md`](../README.md) first: §0, §0.1, §2.2 (05's charter), §3, §4 and §4.1, §5 and
§6.2 (C12, C14). Read also `CLAUDE.md`'s "The datastore layer is `datastorekit`" and "Long-running
jobs — the run registry", whose rules bind you as much as the agent. Then read
[`prompt-02.md`](prompt-02.md), the note for R0, whose protocol, wait and review this note carries
over, and this board's *Orchestrator review of prompt 02* and *of prompt 04*.

**You do not write code.** You may:
- run the interrupter's dummy tests in the scratchpad;
- run the one wait per launch that §2.1 describes;
- read the two new stores read-only, fingerprint them, and run the census, `PRAGMA
  integrity_check`, `foreign_key_check` and `read_inventory` on them;
- read R0's stores' files only to hash them, and its run directories;
- run the suites;
- fix small residue in a follow-up commit of your own (§4).

**While a build is running you write nothing in the repository**, outside `var/` and the session
scratchpad. A file you create makes the next manifest stamp `git_dirty: true`, which is the
prompt's stop.

**The prompt:** [`05-rehearse-through-datastorekit.md`](../05-rehearse-through-datastorekit.md)
**Board items:** L1–L4 (R1), G-DK · **Gate it decides:** DatastoreKit's G2 · **Closes:** nothing ·
**Narrows:** nothing · **Opens:** anything found · **Model:** Opus, as the prompt recommends.

**Gate:**
- 04 landed (`0c9e2cf`) and was reviewed (`0b6565e`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- **Never dispatched concurrently with anything else in this checkout**, and never while any Ray
  instance of this repository is alive.

## 0. What makes this prompt unusual

**It is the first build through code this repository no longer owns.** R0 ran the layer at
`b510bc9`; 04 swapped it for `datastorekit` 0.2.2 and changed nothing else that runs. So R1 is R0
again, and **if it differs from R0, the package is the only cause**. A miss is not a bug to fix
here: it is a DatastoreKit defect (README §5 rule 11, C14). The agent records it and stops, the
user reports it, and the campaign waits for a release and a new pin prompt.

**It runs for an hour or so, and no conversation may own it.** As at 02: the agent launches each
build detached, through the registry, checks once, and ends its turn. You run **one** wait per
launch (§2.1) and resume it when the run is no longer alive. Four launches, four waits.

**It decides DatastoreKit's G2.** Four parts (prompt §2): 04's deletions, 03's pin, the six roots,
and R1's fingerprint. The board records it as holding only when all four do, each naming its
commit or run.

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `944afb8`.
Two statements are wrong or incomplete. Each correction below governs, and the agent logs it as an
`IMPLEMENTATION CHOICE` citing this note.

1. **`prune_serials.py` is not run** (prompt §1.3, last paragraph; README §4.1 L2). It hardcodes
   07's serials (`UNVALIDATED_AFTER_RESUMED_A`), so R0 did not run it either (02's log, line 40;
   verification §1.3.2). R0 ran `prune_check.py` in its place, reading the serials from its own
   `closed_store_state.py` output, and R1 does the same. `closed_store_state.py`,
   `replicated_serial_census.py`, `compare_stores.py` and `interrupt_when.py` run from where R0 ran
   them, as the prompt says.
2. **The manifests' `script_sha256` is not R0's** (prompt §2 lists what every manifest holds; it
   is silent on this). 04 rewrote `docs/handover/quadsource_atol_sweep.py:693`
   (`from Datastore.shard_paths` → `from datastorekit.shard_paths`), its only change since R0. So
   R0's `d745783e65bfff99…` becomes **`b7935877cdbc4e54f6e85a9769f16c2182f7840eaa4bfcb0af2da5bfb9bd3cac`**,
   the same in all four R1 manifests. The log states both and the reason.

**What moves on purpose:**
- Nothing tracked is modified but the board, and `docs/OPEN_ISSUES.md` if an issue is opened.
- Added: `docs/dk-and-rpk-migration-verification.md` gains §2 (§1 is not edited); saved outputs
  under `docs/dk-and-rpk-migration/rehearsal/` named with `05`; the log.
- No script under `docs/dk-and-rpk-migration/rehearsal/` changes. A new one of the agent's, if any,
  goes beside them and is named in the log.
- **README does not move.** Unlike 02, 05 ships no interface, so README §4 gains nothing.
- Under `var/`: the two stores and their sidecars, four run directories and their launch logs, and
  `var/dk-and-rpk-migration-05/`.
- The index is 108, and moves only if an issue is opened.

**The facts, checked by the orchestrator on 2026-10-11 at `944afb8`** (the code is `0c9e2cf`'s:
`git diff --stat 0c9e2cf 944afb8 -- . ':!prompts'` is empty). Pass them on.

- **The tree and the machine.** Branch `handover-remedial`, `git status --short` empty, this
  checkout alone in `git worktree list`. `RAY_ADDRESS` unset. `pgrep -lf
  'gcs_server|raylet|ray::'` and the whole-table search find **nothing**. 10 CPUs. **The load was
  64–92 at 19:14–19:15**, from macOS services (`assistant_service`, `coreaudiod`, `duetexpertd`,
  `contactsd`), not from anything of this repository. 76 GB free on `var/`'s volume (92 % used).
- **The registry.** 21 runs: 19 finished, 2 `unknown` (the pre-registry `a3-pilot` pair). None
  `running`. R0's four are `done`, `killed`, `killed`, `done`.
- **The venv, and what the manifests should hold.** `pip show datastorekit` gives 0.2.2 in this
  venv's `site-packages`; `pip show raypoolkit` finds nothing; `pip check` is clean.
  `RunRegistry.package_provenance()` gives, with keys sorted:
  `{"datastorekit": {"commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "editable": false,
  "requested_revision": "v0.2.2", "url": "https://github.com/ds283/DatastoreKit", "version":
  "0.2.2"}, "raypoolkit": null}`, which is the prompt's §2 record. `datastorekit.__file__` is
  `venv/lib/python3.12/site-packages/datastorekit/__init__.py`.
- **The layer is gone.** In a fresh `python -I` with the repository on its path,
  `Datastore.SQL.ShardedPool`, `Datastore.store_inventory` and `Datastore.SQL.Datastore` each raise
  `ModuleNotFoundError`. The agent checks all 15 (04's log has the list).
- **The build still applies to `main.py`.** `substitute_main_py(filter_cases=False)`, loaded by
  path from `quadsource_atol_sweep.py`, replaces each of its four literals exactly once (the two
  `np.logspace` grids, `tol=DEFAULT_QUADRATURE_ATOL`, `tol=DEFAULT_QUADRATURE_RTOL`), and the
  result compiles, with Ray down. So the four `replaced 1 occurrence` lines of verification §1.2.5
  should recur.
- **The names are free.** `store show` on `var/datastores/dk-and-rpk-migration-05-reference.sqlite`
  and `…-resumed.sqlite`: kind `absent`, problems none, no runs naming either. No file under
  `var/datastores/` and no directory under `var/runs/` contains `dk-and-rpk-migration-05`, and
  `var/dk-and-rpk-migration-05/` does not exist.
- **R0's stores are as 02 left them**, read before this dispatch only by 04, its review and this
  check, all read-only. The twelve files (`var/datastores/dk-and-rpk-migration-02-*`), SHA-256
  prefix and mtime:

  | File | SHA-256 (16) | mtime |
  |---|---|---|
  | `-reference.sqlite` | `9b7a0acc8bcca221` | 14:30:17 |
  | `-reference-shard0000.sqlite` | `3e469052e8810e1e` | 14:38:00 |
  | `-reference-shard0001.sqlite` | `210f494aedfd154f` | 14:38:00 |
  | `-reference-shard0002.sqlite` | `37b12c1592c2f03e` | 14:30:17 |
  | `-reference-shard0003.sqlite` | `db35d513a3d6278f` | 14:38:00 |
  | `-reference.manifest.json` | `755da82a40a26c4b` | 14:38:55 |
  | `-resumed.sqlite` | `bf22863f9ea8b6cf` | 14:45:54 |
  | `-resumed-shard0000.sqlite` | `e1340524ecc5bb94` | 14:48:46 |
  | `-resumed-shard0001.sqlite` | `35908f3a09e66838` | 14:48:46 |
  | `-resumed-shard0002.sqlite` | `fafbe696ba83ac3b` | 14:39:24 |
  | `-resumed-shard0003.sqlite` | `09b8aef1061ef5ed` | 14:48:41 |
  | `-resumed.manifest.json` | `c93c629616fa92a1` | 14:48:49 |

  No `-journal`, `-wal` or `-shm` file under `var/datastores/`. The agent records the full digests
  before its first launch and after its last step (prompt §2), and they must equal these.
- **The scripts run through the package.** On R0's reference store, read-only, with every file's
  hash and mtime unchanged afterwards:
  - `docs/dk-and-rpk-migration/rehearsal/store_soundness.py` exits 0, `=> clean`, 0 problems,
    `QuadSourceIntegral` 864. Its `:30` imports `datastorekit.store_inventory`;
  - `compare_with_recorded.py <primary> --against var/runs/dk-and-rpk-migration-02-reference-20261011T143003`
    exits 0: format 6, `39809dca…`, 21 classes, 0 differences against the sidecar and the run,
    `verdict: equal`.

  The other S1 scripts import no layer module: `stage_counts.py` imports `RunRegistry` and
  `docs/datastore-integrity-verification/stage_counts.read`; `prune_check.py`,
  `policy_statistics.py`, `closed_store_state.py` and `replicated_serial_census.py` use `sqlite3`
  alone; `interrupt_when.py` and `compare_stores.py` use `RunRegistry`. Since R0 (`7650102`) only
  `store_soundness.py:30` has changed among them.
- **The interrupter passes at `944afb8`.** Both dummy tests, run into the scratchpad at 19:15–19:16,
  exit 0 with every verdict as R0's record gives it (five cases; three `--build` cases), and no
  child left alive. The agent runs them again itself, as the prompt says.
- **What the package changes at run time, by reading it.** The set of `print`, `warn` and logger
  messages in `datastorekit/SQL/ShardedPool.py` and `SQL/Datastore.py` equals `b510bc9`'s layer's.
  The one behaviour that is new at close is `ShardedPool._kill_actors()`: on close, if Ray is
  running, it calls `ray.kill(handle, no_restart=True)` on each shard actor and the broker,
  swallowing any exception (`ShardedPool.py:430-456`). R0's actors died with the driver's Ray
  session instead. So:
  - a difference in the launch logs at close (Ray reporting a killed actor, say) is plausible and
    is a finding to list and explain, not by itself a stop. A traceback is a stop.
  - After a killed launch, fewer orphans are possible, not more; R0 saw none.
- **What R0's logs held**, against which §2's log comparison is made. Each R0 `stderr.log` held, at
  most: the handler's two lines (killed runs) and **one SciPy `IntegrationWarning`** ("The
  occurrence of roundoff error is detected…") from a `compute_QuadSource_integral` worker, in
  `reference` and `resumed-c`. That warning belongs to one integral, so in R1's chain it falls in
  whichever launch computes that integral, which depends on where `resumed-b` is killed. Its moving
  between `resumed-b` and `resumed-c` is explained by the point, not a finding. Ray's
  `[repeated Nx across cluster]` folding of the integrity warnings varies run to run (verification
  §1.3.2).
- **The stage times, and the points.** R0's reference (verification §1.5.5), at load 7–35:
  `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` **32.2 s**, no progress line;
  `CALCULATE QUADRATIC SOURCE INTEGRALS` **3 m 32.3 s**, no progress line; the whole build 8 m 0 s.
  - **The points come from R1's own reference `stdout.log`**, as the prompt says. At a higher load
    the stages will be slower; size each `--hold` from R1's reference, not R0's.
  - The QSI queue cannot print a progress line before 300 s (`main.py:119`, `:4155-4156`: a 300 s
    minimum and a batch of 2000). So unless R1's reference QSI stage exceeds 300 s and printed a
    line, interrupt `resumed-b` from the banner with `--hold`, about a third of the stage, with the
    pattern **not anchored at its end** (`'^CALCULATE QUADRATIC SOURCE INTEGRALS'`), as R0 did.
  - `resumed-a`: from the `GkSource` banner with `--interval 5 --hold <S>`, `<S>` putting the signal
    some 10–15 s into the stage and well before its end.
  - **If an interruption misses**, so that a launch meant to be killed ends `done`, **stop and
    ask**. Do not relaunch on the store, and do not build a third store.
- **The commands are R0's**, verbatim in verification §1.1 and §1.1.2, with the names of the
  prompt's §1.1.
- **No Ray process left: list it properly** (`[07-a-killed-builds-ray-agents-outlive-the-driver]`).
  `pgrep -lf 'gcs_server|raylet|ray::'` and the whole-table search, `ps -axo pid,ppid,lstart,command`
  for lines holding both `SecondaryGWKit/venv` and `ray`. Wait for any listed agent **by pid**,
  never by pattern, and record how long it took. Then launch.
- **Shell traps** (04's note). `stat` here is GNU coreutils: use `/usr/bin/stat -f '%m %N'` or
  Python's `os.stat` for mtimes. `grep` in the agent's shell may be a function wrapping `ugrep`; use
  `git grep` or Python for any regex the log relies on.
- **Expected counts**, at `944afb8` (§1 item 5): Datastore **320** (54 + 103 + 101 + 62),
  RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148
  (1 skipped). No count moves at 05.
- **The index:** 108.

**What the review exists to establish.** All else is mechanics.
- **(R1) Every launch ended as recorded.** `done`/0 and `killed`/143, each with a fingerprint, each
  driver's exit status in its launch log, and no Ray process of this repository left.
- **(R2) R1 is R0.** Both stores equal each other, `39809dca…`, and the fingerprints recorded by
  R0's reference, 07's reference and 09's rehearsal, class by class. Nothing stored was
  recomputed; the prune removed exactly the unvalidated rows.
- **(R3) The stores are sound**, by this campaign's `store_soundness.py` and the census.
- **(R4) Every reader works through the package, and none touches the store.**
- **(R5) The manifests say the package ran.** Every `packages` is the prompt's §2 record, every
  `git_dirty` `false`, every `git_head` the dispatch `HEAD`, and every `script_sha256`
  `b7935877…`.
- **(R6) L4 equals R0's**, by the identities of the prompt's §2, and every log difference from R0
  is listed and explained.
- **(R7) G-DK holds**, part by part, each part resting on a named commit or run.
- **(R8) R0 was not touched.** Its twelve files keep their hashes and mtimes.

**Conventions.** 02's, carried. All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §7's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each **in the foreground**, serially, within the turn, and **only after the last
  launch has ended**. Datastore in the four groups. Under load, the four tests of README §7 are
  known to fail: re-run a failure alone before calling it real.
- **Scratch.** The agent's own subdirectory of the session scratchpad. Anything that must outlive
  the session, the working notes and between-launch captures included, goes under
  `var/dk-and-rpk-migration-05/` (`CLAUDE.md` rule 5). No scratch file goes anywhere else in the
  repository, even for a moment, while a build may yet be launched.
- **The venv.** The agent installs and uninstalls nothing.
- **DatastoreKit.** Read only, if at all. Nothing there is changed (README §5 rule 12).
- **Never touch a process you did not start.**
- **Stop conditions are real.** A stop leaves the stores, the run directories and the tree as they
  are: no relaunch, no deletion, no fix. It goes to the user through the orchestrator, and the
  agent resumes with `SendMessage`.
- **Usage limits.** A usage stop does not stop a detached build. Resume the agent with
  `SendMessage`, saying exactly where it stopped. If it commits but sends no report, review from the
  commit, its log and its outputs.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 0c9e2cf..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`. Report
   any stale or unknown run, without acting on it.
3. **Ray and the machine.** `pgrep -lf 'gcs_server|raylet|ray::'`, and the whole-table search of
   §0. No process from this repository's `venv`. `RAY_ADDRESS` unset. Record the load, and say so
   if `var/` has less than 10 GB free.
4. **The venv.** `package_provenance()` gives §0's record; `pip check` is clean.
5. **Baselines.** Re-measure all six roots at `HEAD`, serially, with the load average (README §7),
   and give the agent the numbers.
6. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 108. Tell the agent.
7. **The names are free**, and **R0's twelve files** hash as §0's table.
8. **The interrupter.** Run both dummy tests at `HEAD`, into the scratchpad. Both must pass.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt;
- the campaign README and this board;
- `CLAUDE.md`'s two sections named above;
- 02's and 04's logs, and verification §1;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every launch, every check and the suites. Its log is
  `logs/05-rehearse-through-datastorekit.md`, in README §5.1's form, with the prompt's §6 additions.
- **The hand-off.**
  - After each launch: one check, then a report giving the run id, the Python pid, the `sh` pid,
    and the interrupter's pid and arguments if there is one, then the end of its turn.
  - You resume it with the run's final `status.json`, the last lines of its launch log, and the
    wait's verdict.
  - It never waits, polls or sleeps on a build itself. Waiting by pid for a killed build's orphaned
    Ray agents, before the next launch, is the one bounded wait it runs.
- **The files it may change** are the prompt's §6 list, nothing else. **It touches none of them, and
  creates nothing in the repository outside `var/`, until the last build launch has ended**, so
  that every manifest stamps `git_dirty: false`.
- **R0's stores are not opened.** Hash them before the first launch and after the last step;
  compare R1 with R0 through `compare_with_recorded.py --against` R0's run directory.
- **The interruption points** come from R1's reference `stdout.log`, as §0 says. A missed
  interruption is a stop.
- **Stop and ask** on any of the prompt's §4 conditions. A fingerprint miss is the one R1 exists
  for: record both fingerprints, `compare_fingerprints` by class, and stop. It is a package defect
  (README §5 rule 11), and nothing is fixed or worked around here.

### 2.1 The one wait per launch

02's, unchanged. When the agent reports a launch, run **one** background command that ends when
that run is no longer alive, or after the cap:

```bash
PYTHONPATH=. ./venv/bin/python - <<'EOF'
import json, time
import RunRegistry
status_path = "var/runs/<run id>/status.json"
start, cap, seen_stale = time.time(), 2 * 3600, False
while True:
    status = RunRegistry.read_json(status_path) or {}
    state = RunRegistry.liveness(status, 900)
    if state == "finished" or not RunRegistry.pid_alive(status.get("pid")):
        print("ENDED", state, json.dumps(status)); break
    seen_stale = seen_stale or state == "stale"
    if time.time() - start > cap:
        print("CAP", state, json.dumps(status)); break
    time.sleep(30)
print("heartbeat went stale at least once" if seen_stale else "heartbeat never stale")
EOF
```

- It identifies the run by its pid and heartbeat, never by a command-line match (`CLAUDE.md`
  rule 2). It exits when the build exits. On `CAP` it leaves the build running.
- When it ends, send the agent the run's final `status.json` (less the fingerprint's per-class
  detail), the launch log's last lines, and the verdict line, and nothing else.
  - On `ENDED`, it carries on.
  - On `CAP`, the two-hour stop is met. Tell the agent, and ask the user before anything else
    happens.
- Do not run a second wait while one is alive, and do not check on the build between its start and
  its end.

## 3. The review — thirteen checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 files.
   - `git diff HEAD~1 HEAD --diff-filter=M --name-only` is the board, the verification document
     and, if an issue was opened, `docs/OPEN_ISSUES.md`. README is not modified. No tracked code,
     test, tool or rehearsal script is modified.
   - In the verification document, the diff adds §2 and touches no line of the header or §1 (its
     header already names §2).
2. **(R1) The runs.** Read every manifest, `status.json`, `stderr.log`, interrupter log and launch
   log under `var/runs/dk-and-rpk-migration-05-*` yourself.
   - `reference` and `resumed-c` are `done`/0; `resumed-a` and `resumed-b` are `killed`/143, each
     with `fingerprint` and no `fingerprint_error`.
   - Each launch log ends `** driver exit status` 0 or 143, matching.
   - Each killed `stderr.log` has the handler's lines and no traceback.
   - Each interrupter log shows one `SIGTERM`, with the pid and stage it acted on, and the
     handler's line came within the second or two.
   - The next launch's counts show the killed stage had stored some rows and not all.
   - No process the agent started is alive, and no Ray process of this repository.
3. **(R5) The manifests.** For each: `git_head` the dispatch `HEAD`, `git_dirty` `false`;
   `packages` the prompt's §2 record exactly; `script_sha256` `b7935877…` in all four; `argv` holds
   `--build --rehearsal`, with `--resume` on `resumed-b` and `-c` only; `scope` begins
   `REHEARSAL build, not A3 v2:`; `results` names the right store; each `--purpose` names R1, the
   package and its commit, and `39809dca`.
4. **(R2) Nothing stored was recomputed.** Run `stage_counts.py` on the four run directories
   yourself, and compare it with the log's table and with R0's `stage_counts_02.txt`.
   - The `reference` column equals R0's, stage by stage.
   - Every stage completed before an interruption computes 0 on the next launch.
   - `resumed-b`'s `GkSource` = 291 − (stored by `resumed-a`) + (pruned); `resumed-c`'s QSI = 864 −
     (stored by `resumed-b`), and nothing else. The totals per class over the chain equal the
     reference's. Any excess equals rows in flight at a `SIGTERM`, as the log says.
5. **(R2) The prune.** Run `prune_check.py` on the agent's saved `closed_store_state.py` output
   after `resumed-a`: every listed serial absent now (or reused by a later row), read `mode=ro`.
   `read_inventory` reports no `duplicate`. `resumed-b`'s integrity warnings count the same rows.
6. **(R2) The fingerprints.** Fingerprint both closed stores yourself, read-only:
   - `compare_stores.py <reference> <resumed>`: equal, each equal to its sidecar;
   - `compare_with_recorded.py` on both primaries `--against` R0's reference run directory, 07's
     reference and 09's rehearsal: 0 differences each;
   - format 6, `39809dca…`, 21 classes.
7. **(R3) Integrity.** Run `store_soundness.py` and `replicated_serial_census.py` on both primaries
   yourself: `integrity_check` `ok`, `foreign_key_check` empty, one `version` row per shard (label
   `2025.1.1`), no census divergence, no inventory problem, `replication_in_flight` empty, no
   journal file.
8. **(R4) The readers.** Each reader's log under `var/dk-and-rpk-migration-05/extract/`: exit 0, no
   `ReadOnlyMiss`, `ReadOnlyWrite` or traceback; hashes and mtimes equal around each;
   `extract_tensor_source_data.py` exit 0; the `nomodels extract_Gk_data` run printed the skip line.
   Run one reader yourself through `run_readers.sh` into the scratchpad, and hash the store before
   and after.
9. **(R6) L4.**
   - Run `policy_statistics.py` on R1's reference yourself: 14 lines; the counts R0's (291; 109 /
     26 (25 complete, 1 minimal) / 156); `diff` against `policy_statistics_02_reference.txt`
     empty, or differing only in order, sorted equal, as the log says.
   - The `** ` banner lines of R1's reference `stdout.log`, in order, equal R0's (`grep '^\*\* '`
     on both, times stripped).
   - The resumed-chain identities of check 4 are in the log as a table beside verification §1.5.4,
     and logged as the `STRUCTURALLY REQUIRED` reading of README §2.2.
10. **(R6) The log differences.** For each slug, compare R1's `stderr.log` and launch log with R0's
    yourself, times, pids, ports and paths stripped. Every line kind in one and not the other is in
    the log, with an explanation that holds (§0: the `IntegrationWarning`'s launch, Ray's folding,
    anything at close from `_kill_actors`). No traceback, no `ReadOnlyMiss`/`ReadOnlyWrite`, no
    refused open.
11. **(R7, R8) G-DK and R0.**
    - In a fresh `python -I`, each of the 15 layer modules raises `ModuleNotFoundError`, and
      `datastorekit.__file__` is in the venv's `site-packages`.
    - G-DK's four parts are each stated with their commit or run: `0c9e2cf` and its review
      (`0b6565e`); `9ec603a` and its review (`41d9bc0`) with this prompt's manifests; the six roots
      at the dispatch `HEAD`; R1's `reference` and `resumed-c`.
    - R0's twelve files hash as §0's table, before the agent's first launch (its record) and now.
12. **The records.** The board: the §1 row for 05, L1–L4 (R1) and G-DK ✅, the §2 "DatastoreKit G2"
    row holding with its evidence, `### 5.5 After prompt 05`; §3 entries and the index for anything
    opened, the count right by counting. The verification document's §2 has §1's subsections in
    order, every command verbatim, and its numbers match the run directories. The log covers the
    prompt's §6 list and classifies §0's two corrections.
13. **Suites and `black`.** All six, serially, at `HEAD`, every count as §0. `black --check` is clean
    on any Python file added.

## 4. After the review

- **Record it on this board** as a §5.5 *Orchestrator review of prompt 05* paragraph, in 02's and
  04's form. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board and in the log's header;
  - the board's §2 "DatastoreKit G2" row, if the agent left it ⬜ with every part holding;
  - this note added to the board's list of notes, marked "used for 05".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - R1–R8, each in a line;
  - the fingerprints against `39809dca…`, R0's, 07's and 09's;
  - where each interruption fell, how it was chosen, and what the resume pruned and recomputed;
  - every log difference from R0 and its explanation, `_kill_actors` at close in particular;
  - whether DatastoreKit's G2 holds, and that recording it in DatastoreKit is the user's (C12);
  - anything opened; the stores, which stay (README §3).
- **Then 06**, the `raypoolkit` pin. Its note is written after this review.
