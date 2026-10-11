# Orchestrator — prompt 02, rehearse the start tree (R0)

Read [`../README.md`](../README.md) first: §0.1, §2.2 (02's charter), §3, §4 and §4.1, §5 and §7.
Read also `CLAUDE.md`'s "Long-running jobs — the run registry", whose rules bind you as much as
the agent. Then read [`prompt-01.md`](prompt-01.md) §0's conventions, this board's *Orchestrator
review of prompt 01*, and
[`../../a3-v2-readiness/orchestrator/prompt-07.md`](../../a3-v2-readiness/orchestrator/prompt-07.md),
the note for the rehearsal whose protocol this one runs.

**You do not write code.** You may:
- run the interrupter's dummy tests in the scratchpad;
- run the one wait per launch that §2.1 describes;
- read the two new stores read-only, fingerprint them, and run the census, `PRAGMA
  integrity_check`, `foreign_key_check` and `read_inventory` on them;
- run the suites;
- fix small residue in a follow-up commit of your own (§4).

**While a build is running you write nothing in the repository**, outside `var/` and the session
scratchpad. A file you create makes the next manifest stamp `git_dirty: true`, which is the
prompt's stop.

**The prompt:** [`02-rehearse-the-start-tree.md`](../02-rehearse-the-start-tree.md)
**Board items:** S1, L1–L4 (R0) · **Gate:** R0 · **Closes:** nothing · **Narrows:** nothing ·
**Opens:** anything found · **Model:** Opus, as the prompt recommends.

**Gate:**
- 01 landed (`f54f1c5`) and was reviewed (`c87dc2d`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- **Never dispatched concurrently with anything else in this checkout**, and never while any Ray
  instance of this repository is alive.

## 0. What makes this prompt unusual

**It is the campaign's first real run, and the only one at the start tree.** The layer is
`b510bc9`'s, byte for byte, and the pool is the frozen `RayTools/`. The tree differs from
`b510bc9` only by 01's manifest field, which runs no layer code, and by `prompts/`. R0 runs the
three layer commits no rehearsal has run (`fee1fbf`, `4c34f13`, `086c81a`; README §0.1). R1 and
R2 are then compared with it.

**It runs for an hour or so, and no conversation may own it.** The agent launches each build
detached, through the registry, checks once that it started, and ends its turn. You run **one**
wait per launch (§2.1), and resume the agent when the run is no longer alive. Neither of you
polls. There are four launches, so four waits: `reference`, then `resumed-a`, `-b` and `-c`.

**The fingerprint is the acceptance, and a miss is the stop R0 exists for.** If either store
misses `39809dca…`, or differs from 07's reference, the regression is in SGK's own tree between
`53d4e18` and `HEAD`. It is not this campaign's to fix. The agent records and stops, and nothing
after 02 is dispatched until the user decides.

**What moves on purpose:**
- Nothing tracked is modified. Everything added is a record:
  - `docs/dk-and-rpk-migration-verification.md`, new, with its §1;
  - `docs/dk-and-rpk-migration/rehearsal/`, new: S1's three scripts, any other script of the
    agent's, and the small saved outputs;
  - the log, this board, and `docs/OPEN_ISSUES.md` if an issue is opened.
- **README §4's "02 ships" line.** The prompt's §6 does not list it, but 01 recorded its interface
  there, and 05 and 08 read it. The agent adds one line under "02 ships", saying the scripts
  shipped as named or naming what differs, and logs it as `STRUCTURALLY REQUIRED`. Nothing else in
  README moves.
- Under `var/`: the two stores and their sidecars, four run directories and their launch logs, and
  `var/dk-and-rpk-migration-02/`.
- The index is 104, and moves only if an issue is opened.

**The facts, checked by the orchestrator on 2026-10-11 at `c87dc2d`** (the code is `f54f1c5`'s:
`git diff --stat f54f1c5 c87dc2d -- . ':!prompts'` is empty). Pass them on.

- **The names are free.** `store show` on `var/datastores/dk-and-rpk-migration-02-reference.sqlite`
  and `…-resumed.sqlite`: kind `absent`, problems none, no runs naming either. No file under
  `var/datastores/` and no directory under `var/runs/` begins `dk-and`.
  `var/dk-and-rpk-migration-02/`, `docs/dk-and-rpk-migration/` and the verification document do
  not exist.
- **The registry.** 17 runs: 15 finished, and the 2 `unknown` pre-registry `a3-pilot` runs. None
  `running`.
- **Ray and the machine.** `pgrep -lf 'gcs_server|raylet|ray::'` finds **nothing**: the user's
  ChamPBH cluster is not up today. `RAY_ADDRESS` is unset. 10 CPUs, load 9–15, 86 GB free on
  `var/`'s volume (91 % used). `git worktree list` shows this checkout alone.
- **The venv.** `pip show datastorekit raypoolkit` finds neither. So every manifest's `packages`
  should be `{"datastorekit": null, "raypoolkit": null}` (01's log, *State handed*).
- **The interrupter passes at `c87dc2d`.** The orchestrator ran both dummy tests into the
  scratchpad: `interrupter_dummy_test.sh` (five cases) and `interrupter_build_dummy_test.sh`
  (three cases) each exit 0, with every verdict as 07's record gives it, and no child left alive.
  The agent runs them again itself, as the prompt says.
- **The reference fingerprint.** `var/runs/a3-v2-readiness-07-reference-20261004T012909/status.json`
  holds `fingerprint` with `fingerprint_format` 6 (the key is `fingerprint_format`, not `format`),
  digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, 21 classes, problems
  `{}`. `datastore-generic`'s 09 rehearsal (`var/runs/datastore-generic-09-rehearsal-20261006T152452/`)
  holds the same digest, and its `classes` equal 07's exactly. Both runs were `git_dirty: false`.
  `RunRegistry.stores.compare_fingerprints(recorded, current)` takes two such dicts and never
  compares `taken`, so a `status.json` fingerprint can be compared with a store's directly.
- **The statistics block** (L4). In both earlier logs `** GkSourcePolicyData SUMMARY STATISTICS`
  is at line 83, and the block is lines 83–96. Line 97 is
  `   @@ QuadSourceIntegral triangle filter: kept 864 of 1728 (k,q,r) triples (50.00%)`, then the
  `** CALCULATE QUADRATIC SOURCE INTEGRALS` banner. 07's and 09's lines 83–97 are identical:
  291 instances, 109 numeric, 26 mixed (25 complete, 1 minimal), 156 WKB. `policy_statistics.py`
  must stop before the `@@` line; how it finds the end is the agent's choice, and the log says
  what it chose.
- **The stages run much faster at low load, and 07's second interruption point may not exist.**
  The two reference builds, from their `stdout.log`:

  | Stage | 07 reference (`dbd8a51`, load 13–44) | 09 rehearsal (`53d4e18`) |
  |---|---|---|
  | `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` | 1 m 2.16 s, no progress line | 24.9 s, no progress line |
  | `APPLY GKSOURCE POLICIES` | 24.1 s | 9.38 s |
  | `CALCULATE QUADRATIC SOURCE INTEGRALS` | 7 m 1.74 s; one progress line at 5 m 10.2 s, reading 100.00 % | **1 m 30.4 s, no progress line** |
  | whole build, manifest to last heartbeat | 12 m 58 s | 4 m 10 s |

  - `RayWorkPool` prints a progress line only after 60 s since the last one *and* either 300 s or
    500 batched items (`RayTools/RayWorkPool.py:15-17`, `:524-545`). At this scope the QSI stage
    is 2 work items, so a line, when it comes, reads 100.00 %.
  - **07 interrupted `resumed-b` at that line** (`'^CALCULATE QUADRATIC SOURCE INTEGRALS, ' 100`).
    If R0's reference prints no QSI progress line, that trigger never fires: the stage completes,
    the interrupter exits without signalling (its case E), and `resumed-b` runs to `done`.
  - So, as the prompt says, **choose both points from R0's own reference `stdout.log`**. If its QSI
    stage printed no progress line, interrupt `resumed-b` from the banner with `--hold`, as 07 did
    for `GkSource`: `'^CALCULATE QUADRATIC SOURCE INTEGRALS$' 0 --interval 5 --hold <S>`, with
    `<S>` well inside the reference's stage time (a third of it is a fair aim). Log it as an
    `IMPLEMENTATION CHOICE`, with the reason.
  - `GkSource` likewise. Size `--hold` from R0's reference, not from 07's `15`: at 09's 24.9 s, a
    15 s hold plus a 5 s interval leaves little margin.
  - **If an interruption misses**, so that a launch meant to be killed ends `done`, **stop and
    ask**. Do not relaunch on the store, and do not build a third store. The protocol then has a
    hole, and only the user can say whether R0 stands without it.
- **The scripts 07 left, and what each hardcodes.** All are under `docs/a3-v2-readiness/rehearsal/`
  unless named otherwise.
  - `run_readers.sh` hardcodes 07's store, its four shard names and its output directory. It has
    three modes: `six`, `nomodels <script>`, and `sorted`. `sorted` ran a wrapper for the
    tensor-source reader's random draw, which 07a fixed. S1's copy needs the first two only. It
    takes `<primary> <output dir>` (README §4), so the `nomodels` run needs an optional trailing
    argument, such as `run_readers.sh <primary> <output dir> [nomodels <script>]`. The shards come
    from the primary's `shards` table or a glob, never a hardcoded four.
  - `stage_counts_07.py` labels columns by splitting the run id on `-07-`. It reads through
    `docs/datastore-integrity-verification/stage_counts.read(run_dir)`.
  - **`prune_serials.py` hardcodes 07's serials** (`UNVALIDATED_AFTER_RESUMED_A`), so it cannot
    check R0's prune. The agent writes its own check, reading the serials from its own
    `closed_store_state.py` output after each killed launch. Make it reusable like S1, since 05 and
    08 need it too, and name it in README §4's line. This is outside S1's three, and the prompt's
    §6 allows it ("any other script of yours").
  - `closed_store_state.py <primary>` and `store_soundness.py <primary>...` take their stores as
    arguments, and read the shards from the primary's `shards` table. They run unchanged. So do
    `docs/datastore-integrity-verification/compare_stores.py <reference> <resumed>` and
    `docs/datastore-integrity-audit/replicated_serial_census.py <primary>...`.
  - 07's originals are records, and are not edited.
- **The commands are 07's**, verbatim in `docs/a3-v2-readiness-verification.md` §1.1 and §3, with
  the names of the prompt's §1.1.
- **No Ray process left: list it properly** (`[07-a-killed-builds-ray-agents-outlive-the-driver]`,
  verification §2.1.3).
  - After a kill, Ray's dashboard agent and runtime-env agent are orphaned to pid 1. 07 saw them
    exit on their own 1–2 minutes later.
  - `pgrep -f 'gcs_server|raylet|ray::'` lists the dashboard agent only because its command line
    holds `--raylet-name=…`, and misses the runtime-env agent. 07's agent added a whole-table
    search, `ps -axo pid,ppid,lstart,command`, for lines holding both `SecondaryGWKit/venv` and
    `ray`. The driver itself shows as `/opt/local/…/Python`, not the venv path.
  - Wait for each listed agent **by pid**, never by pattern, and record how long each took. Then
    launch.
- **Disk.** Each store is a few MB; 07's shard 2 was 1 077 248 bytes. 86 GB free is ample.
- **Expected counts**, at `c87dc2d` (§1 item 5): AdaptiveLevin 32, CosmologyModels 39,
  LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693, Datastore 739
  (163 + 185 + 200 + 191). No count moves at 02.
- **The index:** 104.

**What the review exists to establish.** All else is mechanics.
- **(R1) Every launch ended as recorded.** `done`/0 and `killed`/143, each with a fingerprint, each
  driver's exit status in its launch log, and no Ray process of this repository left once its
  agents have exited.
- **(R2) The resume is the reference.** Nothing stored was recomputed, the unvalidated rows were
  pruned and nothing is duplicated, and both stores equal each other, `39809dca…` and 07's
  reference, class by class.
- **(R3) The stores are sound.** The version row on every shard, no census divergence,
  `integrity_check` `ok`, `foreign_key_check` empty, and no inventory problem.
- **(R4) Every reader works, and none touches the store.** All six exit 0, refuse nothing, and
  leave every file byte-identical.
- **(R5) The manifests say what ran.** Every `packages` is both `null`, every `git_dirty` is
  `false`, and every `git_head` is the dispatch `HEAD`.
- **(R6) L4 is a baseline 05 and 08 can be compared with**, and S1's scripts produced it. R0's
  statistics block equals 09's line for line.

**Conventions.** 01's, carried. All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §7's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each **in the foreground**, serially, within the turn, and **only after the last
  launch has ended**: a suite run beside a build changes both timings. Datastore in the four
  groups. Under load, the four tests of README §7 are known to fail: re-run a failure alone before
  calling it real.
- **Scratch.** The agent's own subdirectory of the session scratchpad. Anything that must outlive
  the session, the working notes and between-launch captures included, goes under
  `var/dk-and-rpk-migration-02/` (`CLAUDE.md` rule 5). No scratch file goes anywhere else in the
  repository, even for a moment, while a build may yet be launched.
- **The venv.** The agent installs nothing and runs no `pip install`. 03 is the first prompt that
  changes the venv.
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
   - `git log --oneline c87dc2d..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`. Report
   any stale or unknown run, without acting on it.
3. **Ray and the machine.** `pgrep -lf 'gcs_server|raylet|ray::'`, and the whole-table search of
   §0. No process from this repository's `venv`. Record anything else by pid and start time, so
   that the review can show it was not touched. `RAY_ADDRESS` unset. Record the load, and say so if
   `var/` has less than 10 GB free.
4. **The venv.** `./venv/bin/pip show datastorekit raypoolkit` finds neither.
5. **Baselines.** Re-measure all six roots at `HEAD`, serially, in the foreground, with the load
   average (README §7), and give the agent the numbers.
6. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 104. Tell the agent.
7. **The names are free.** `store show` on both store paths: kind `absent`, no runs.
8. **The interrupter.** Run both dummy tests at `HEAD`, into the scratchpad. Both must pass.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt;
- the campaign README and this board;
- `CLAUDE.md`'s run-registry section;
- 01's log;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every launch, every check and the suites. Its log is
  `logs/02-rehearse-the-start-tree.md`, in README §5.1's form, with the prompt's §6 additions.
- **The hand-off.**
  - After each launch: one check, then a report giving the run id, the Python pid, the `sh` pid,
    and the interrupter's pid if there is one, then the end of its turn.
  - You resume it with the run's final `status.json`, the last lines of its launch log, and the
    wait's verdict.
  - It never waits, polls or sleeps on a build itself. Waiting by pid for a killed build's orphaned
    Ray agents, before the next launch, is the one bounded wait it runs.
- **The files it may change** are the prompt's §6 list and README §4's one line, nothing else.
  **It touches none of them, and creates nothing in the repository outside `var/`, until the last
  build launch has ended**, so that every manifest stamps `git_dirty: false`.
- **The interruption points** come from R0's reference `stdout.log`, as §0 says. A missed
  interruption is a stop.
- **Stop and ask** on any of the prompt's §4 conditions. A fingerprint miss is the one R0 exists
  for: record, and stop.

### 2.1 The one wait per launch

When the agent reports a launch, run **one** background command that ends when that run is no
longer alive, or after the cap:

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
  rule 2). It exits when the build exits, so it cannot outlive what it waits on. On `CAP` it leaves
  the build running.
- When it ends, send the agent the run's final `status.json` (less the fingerprint's per-class
  detail), the launch log's last lines, and the verdict line, and nothing else.
  - On `ENDED`, it carries on.
  - On `CAP`, 07 §5's two-hour stop is met. Tell the agent, and ask the user before anything else
    happens.
- Do not run a second wait while one is alive, and do not check on the build between its start and
  its end.

## 3. The review — twelve checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 files and README §4's one line.
   - `git diff HEAD~1 HEAD --diff-filter=M --name-only` is the board, `README.md` and, if an issue
     was opened, `docs/OPEN_ISSUES.md`. No tracked code, test or tool is modified.
   - In `README.md`, only one line under §4's "02 ships" moves.
   - The diff over everything outside `docs/dk-and-rpk-migration-verification.md`,
     `docs/dk-and-rpk-migration/` and `prompts/dk-and-rpk-migration/` (but `OPEN_ISSUES.md`) is
     empty.
2. **(R1) The runs.** Read every manifest, `status.json`, `stderr.log`, interrupter log and launch
   log under `var/runs/dk-and-rpk-migration-02-*` yourself.
   - `reference` and `resumed-c` are `done`/0; `resumed-a` and `resumed-b` are `killed`/143, each
     with `fingerprint` and no `fingerprint_error`.
   - Each launch log ends `** driver exit status` 0 or 143, matching.
   - Each killed `stderr.log` has the handler's line and no traceback. No `*** SIGTERM received`
     banner.
   - Each interrupter log shows one `SIGTERM`, with the pid and stage it acted on, and the handler's
     line came within the second.
   - The killed launch's last progress line, and the next launch's counts, show the stage had
     stored some rows and not all.
   - No process the agent started is alive: neither interrupter, no Ray process of this repository.
     Anything §1 item 3 recorded has the same pids and start times.
3. **(R5) The manifests.** For each: `git_head` is the dispatch `HEAD` and `git_dirty` is
   `false`; `packages` is `{"datastorekit": null, "raypoolkit": null}`; `script_sha256` is the same
   in all four; `argv` holds `--build --rehearsal`, with `--resume` on `resumed-b` and `-c` only;
   `scope` begins `REHEARSAL build, not A3 v2:`; `results` names the right store.
4. **(R2) Nothing stored was recomputed.** Run S1's `stage_counts.py` on the four run directories
   yourself, and compare it with the log's table.
   - Every stage completed before an interruption computes 0 on the next launch.
   - Summed over the resumed launches, less the pruned rows, each stage stores what `reference`
     stored. Any excess equals the rows in flight at a `SIGTERM`, as the log says.
5. **(R2) The prune.** The log's serials of `validated = 0` rows after each killed launch are absent
   from the store now (or reused by a later row), read `mode=ro`. `read_inventory` reports no
   `duplicate`.
6. **(R2) The fingerprints.** Fingerprint both closed stores yourself, read-only
   (`compare_stores.py`).
   - `compare_fingerprints` gives nothing; both are at format 6, `39809dca…`, all 21 classes;
     each equals its sidecar's.
   - `compare_fingerprints` of each against 07's reference `status.json` fingerprint gives nothing.
7. **(R3) Integrity.** Run `store_soundness.py` and `replicated_serial_census.py` on both primaries
   yourself.
   - `integrity_check` `ok` and `foreign_key_check` empty on every shard.
   - One `version` row per shard, label `2025.1.1`, the same serial on every shard of a store.
   - No census divergence; no inventory problem; the primary's `replication_in_flight` empty.
   - No `-journal`, `-wal` or `-shm` file beside any of them.
8. **(R4) The readers.**
   - Read each reader's log under `var/dk-and-rpk-migration-02/extract/`: exit 0, its run
     selection, what it read, and no `ReadOnlyMiss` or `ReadOnlyWrite`.
   - The hashes before and after each reader are equal, and so are the mtimes.
   - The run without `--models` printed the skip line and exited 0.
   - `extract_tensor_source_data.py` exited 0.
   - Run one reader yourself through S1's `run_readers.sh` into the scratchpad, and hash the store
     before and after.
9. **(R6) S1 and L4.**
   - Read the three scripts in full: usage in each docstring, no prompt number inside, every path
     an argument or derived from one. `run_readers.sh` derives its shards; `stage_counts.py` takes
     its labels from the manifest.
   - Run `policy_statistics.py` on R0's reference and on 09's rehearsal yourself: equal line for
     line, 14 lines, and the script exits non-zero on a log without the block (try `resumed-a`'s).
   - Run `stage_counts.py` on 07's four run directories: it labels them, where 07's original
     script was needed.
   - The verification document's §1 holds L4's tables in a form 05 can be set beside.
10. **A3 v2's guard.** The log's foreground `--build` refused, naming the existing files. The store's
    hashes are unchanged, and no manifest was written for it.
11. **The records.**
    - The board: the §1 row for 02, S1 and L1–L4, the §2 R0 row, and `### 5.2 After prompt 02`.
    - README §4: the one line, naming the scripts as shipped, the prune check among them.
    - The index: anything opened added, its section created, and the count right by counting.
    - The verification document quotes every command verbatim, and its numbers match the run
      directories.
12. **Suites and `black`.** All six, serially, at `HEAD`, every count as §1 item 5. `black --check`
    is clean on every Python file added.

## 4. After the review

- **Record it on this board** as a §5.2 *Orchestrator review of prompt 02* paragraph, in 01's form.
  Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA;
  - the board's §2 R0 gate, if the agent left it ⬜ with the fingerprint matched;
  - this note marked "used for 02" in the board's list.
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - R1–R6, each in a line;
  - the fingerprints against `39809dca…` and 07's reference;
  - where each interruption fell, how it was chosen, and what the resume pruned and recomputed;
  - how long the orphaned Ray agents took to exit;
  - anything opened;
  - the stores, which stay (README §3).
- **Then 03**, the `datastorekit` pin. It is the first prompt that changes the venv, and 04 reads
  R0's reference store read-only (its acceptance 3). Its note is written after this review.
