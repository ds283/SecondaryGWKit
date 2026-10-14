# Orchestrator — prompt 04, rehearse through the new DatastoreKit (R1)

Read [`../README.md`](../README.md) first: §0.1, §2.2 (04's charter), §3, §4.1 (the protocol and
the lock), §5 (rules 8–11, 14, 15) and §6.2 (C3). Read also `CLAUDE.md`'s "Long-running jobs — the
run registry", the freeze section and "The shared rehearsal lock", whose rules bind you as much as
the agent. Then [`prompt-02.md`](prompt-02.md), the note for R0, whose protocol, wait and review this
note carries over except where §0 below says otherwise, and the board's *Orchestrator review* of
02 (§5.2), 02a (§5.4) and 03 (§5.6).

**You do not write code.** You may:
- run the interrupter's dummy tests in the scratchpad;
- run the one wait per launch that §2.1 describes;
- read the four live stores read-only, fingerprint them, and run the census, `PRAGMA
  integrity_check`, `foreign_key_check` and `read_inventory` on them;
- read R0's run directories, and hash the sidecars under `var/datastores/`;
- run `tools.rehearsal_lock status` and `ray-processes`, never `acquire`, `set-pid` or `release`
  while the agent holds the lock;
- run the suites, and a comparison script from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**While a build is running you write nothing in the repository**, outside `var/` and the session
scratchpad. A file you create makes the next manifest stamp `git_dirty: true`, which is the
prompt's stop. This note is committed before dispatch for that reason.

**The prompt:** [`04-rehearse-through-the-new-datastorekit.md`](../04-rehearse-through-the-new-datastorekit.md)
**Board items:** L1–L4 (R1), W1 · **Gate it decides:** R1 · **Closes:** nothing · **Narrows:**
nothing · **Opens:** anything found · **Model:** Opus, as the prompt recommends.

**Gate:**
- 03 landed (`5b81f2e`) and was reviewed (`e150335`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code (anything outside
  `prompts/` and `docs/`). If a later commit touches code, re-check every fact below that it could
  move, and hand the agent the corrections as an addendum.
- **Never dispatched concurrently with anything else in this checkout**, and never while any Ray
  instance of this repository is alive or the rehearsal lock is held.

## 0. What makes this prompt unusual

**It is R0 again, through `datastorekit` 0.4.0.** Between R0 and the dispatch `HEAD` the only change
to what a build runs is the pin: `git diff --stat 74eff08 e150335 -- . ':!prompts' ':!docs'` lists
`requirements.txt` (one line), and 02a's `tools/rehearsal_lock.py`, its tests and `CLAUDE.md`, none
of which a build or a reader imports. Under `datastorekit/`, `v0.2.2` → `v0.4.0` changes `SQL/factory_base.py`
(`v0.2.3`), `object.py` (`v0.3.0`) and adds `store_fingerprint.py`; `store_inventory.py`, which
SGK's fingerprint reads, is unchanged (board §5.5). **So a miss is DatastoreKit's** (README §4.1,
§5 rule 11), and the first suspects are `object.py`'s `_timestamp` and `factory_base.py`.

**It is the first rehearsal under the shared lock.** The agent takes the lock with
`tools/rehearsal_lock.py` before `reference`, sets its `pid` at each launch, holds it through the
readers, and releases it. Between launches the lock is held and its pid is dead; that is the
protocol, not a stale lock. You never touch it while the agent holds it. If the agent stops early
and cannot release, the user removes it by hand.

**It runs for half an hour or more, and no conversation may own it.** As at 02: the agent launches
each build detached, through the registry, checks once, and ends its turn. You run **one** wait per
launch (§2.1) and resume it when the run is no longer alive. Four launches, four waits.

**It adds one script, after the builds.** `docs/runregistry-extraction/fingerprint_both_ways.py`
(W1) is written only after `resumed-c` has ended, and run on R1's two stores and R0's two.

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `e150335`.
Each correction below governs, and the agent logs it as an `IMPLEMENTATION CHOICE` citing this note,
or `STRUCTURALLY REQUIRED` where the prompt cannot be followed as written.

1. **02's corrections carry over**, as 02's log records them (deviations 2, 3, 7): `prune_serials.py`
   is not run, and `prune_check.py` runs on R1's own `closed_store_state.py` output; the six scripts
   of `docs/dk-and-rpk-migration/rehearsal/` are the ones run, `store_soundness.py` among them, not
   `docs/a3-v2-readiness/rehearsal/store_soundness.py`; the other tools run from where R0 ran them
   (verification §1.1.2, §1.3, §1.4), none copied. Every capture goes under
   `var/runregistry-extraction-04/` until `resumed-c` has ended. `docs/runregistry-extraction/` now
   exists, from 02, but nothing is written into it, nor anywhere else in the repository, before then.
2. **The `resumed-a` point** (prompt §1, "R0's 40 % hold on `GkSource` … is the precedent"). Size the
   hold at **about 40 % of R1's own `reference` `GkSource` stage time**, rounded to whole seconds,
   and not below 10 s. R0's was 78 s at a load up to 339, so 31 s; R2 of `dk-and-rpk-migration` took
   41.8 s at load 13–33. Pattern `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$'`,
   `--interval 5`.
   - If `resumed-a` stores no `GkSource` row, that is recorded, not a stop: the run was killed in the
     right stage, and the prune clause holds vacuously, as at R2.
   - If the stage completes during the hold and the build runs on, that is a missed interruption
     and a stop.
3. **The `resumed-b` point.** Pattern `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`, **not anchored at
   its end**, `--interval 5`, `--hold` about a third of R1's `reference` QSI stage **and at most
   250 s**, inside the queue's 300 s minimum notify interval (`main.py:119`).
4. **A progress line may or may not appear.** R0's reference QSI stage took 1230.1 s and printed
   three; R2's took 299.2 s and printed none. Whether R1's does is the 300 s threshold and the load,
   not a finding. Compare any that appear with R0's (verification §1.5.6).
5. **`pkg == sgk` does not compare order.** Python's `dict` equality ignores key order, and the
   prompt's §2 item 4 asks for "every class, in the same order". The script also checks
   `list(pkg_without_encoding) == list(sgk)` and `list(pkg["classes"]) == list(sgk["classes"])`, and
   prints the first differing position if not. `tag_sets` are lists, so `==` already compares their
   order. Its breakage record gains no sixth mutation for this; the log says how order is checked.
6. **Reading DatastoreKit.** The prompt's Read-first item 6 reads DatastoreKit's repository with
   `git -C … show v0.4.0:…`. That is a read, which README §5 rule 12 allows. Nothing in
   DatastoreKit is run or changed, and nothing there is checked out.
7. **The verification document's §2** has §1's subsections §2.1–§2.9 in the same order (R0's
   §1.1–§1.9). §1.10, the factory move, was R0's own and has no counterpart; **§2.10 is W1**, the
   fingerprint both ways. The header's list already names §2 as R1's; it is left as it is.
8. **The pre-launch Ray check is the helper's `ray-processes`**, exit 0 (README §4.1). A machine-wide
   `pgrep 'gcs_server|raylet|ray::'` may also be recorded, and will list SI's or ChamPBH's clusters
   if they are up; it is not the test, and a cluster of another project is never touched.

**What moves on purpose:**
- Added: `docs/runregistry-extraction/fingerprint_both_ways.py`;
  `docs/runregistry-extraction/rehearsal/*_04*` outputs, `lock_04.txt` and
  `fingerprint_both_ways_04.txt` among them; the log.
- Modified: `docs/runregistry-extraction-verification.md` (§2 appended; §1 not edited); this board;
  `docs/OPEN_ISSUES.md` only if an issue is opened.
- **README does not move**, and neither does `prompts/INDEX.md` (the orchestrator's, §4). No other
  script anywhere changes, and nothing under `RunRegistry/`.
- Under `var/`: R1's two stores and their sidecars, four run directories and their launch logs, and
  `var/runregistry-extraction-04/`. **Nothing else under `var/` changes**: the 14 tombstones and R0's
  twelve store files included.
- Under `~/ScienceData/`: `.rehearsal-lock` created and removed again.
- The index is 107, and moves only if an issue is opened.

**The facts, checked by the orchestrator on 2026-10-14 at `e150335`.** Pass them on.

- **The tree and the machine.** Branch `handover-remedial`, `git status --short` empty, this
  checkout alone in `git worktree list`. `git log 5b81f2e..e150335` is `e150335` alone, which
  touches `prompts/` only. `RAY_ADDRESS` unset. 10 CPUs. **The one-minute load was 157 at 15:36,
  rising from 43 (fifteen-minute)**; the top processes by CPU were the Claude app, VS Code and macOS
  services, nothing of this repository. 74 GB free on `var/`'s volume (93 % used).
- **The lock and Ray.** `tools.rehearsal_lock status`: `lock: free`, exit 0. `ray-processes`:
  `none`, exit 0. `~/ScienceData/` holds `ChamPBH-rehearsals/` alone.
- **The registry.** 33 runs, 31 finished, 2 `unknown` (the `a3-pilot` pair); none `running` or stale.
- **The venv, and what every manifest should hold.** `pip check` clean; `pip show runregistrykit`
  finds nothing. `RunRegistry.package_provenance()` gives, keys sorted:
  `{"datastorekit": {"commit": "c8570e849334d09085bbd84876095fc9e22bd005", "editable": false,
  "requested_revision": "v0.4.0", "url": "https://github.com/ds283/DatastoreKit", "version":
  "0.4.0"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable":
  false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version":
  "0.2.0"}}`. `raypoolkit`'s entry is R0's exactly. Compare as JSON, not as text.
- **The launcher is R0's.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` is
  `b7935877cdbc4e54f6e85a9769f16c2182f7840eaa4bfcb0af2da5bfb9bd3cac`; its last change is `0c9e2cf`.
  Every R1 manifest's `script_sha256` should be this.
- **The names are free.** `store show` on `var/datastores/runregistry-extraction-04-reference.sqlite`
  and `…-resumed.sqlite`: kind `absent`, problems none, no runs naming either, exit 0. Nothing under
  `var/runs/` contains `runregistry-extraction-04`; `var/runregistry-extraction-04/` does not exist.
- **What is under `var/datastores/`**: 26 files, the 14 tombstone sidecars (one under
  `backup-pre-resume-20260921T091011/`) and R0's twelve (two primaries, eight shards, two sidecars).
  `find var/datastores -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256` is
  `3e0e6badf16a57c7…`; over the 14 tombstones alone, `023d139fb714e229…`, as at R0. The agent records
  the full list with SHA-256, size and mtime before its first launch and after its last step: the 26
  unchanged, and the only new files R1's ten store files and two sidecars.
- **The recorded fingerprints to compare against**, as run directories for
  `compare_with_recorded.py --against`: R0's `var/runs/runregistry-extraction-02-reference-20261012T091921`
  and `var/runs/runregistry-extraction-02-resumed-c-20261012T101145`, each `status.json` format 6,
  `39809dca…`, 21 classes, problems `{}`.
- **The scripts run at `e150335`, on R0's stores and runs, read-only.**
  - `stage_counts.py` on R0's four run directories reproduces `stage_counts_02.txt` byte for byte;
    `policy_statistics.py` on R0's `reference` reproduces `policy_statistics_02_reference.txt`.
  - `store_soundness.py` on R0's two primaries exits 0, `=> clean`, its output R0's but for the
    command line; `replicated_serial_census.py` exits 0, `=> no divergence` twice; `compare_stores.py`
    exits 0, `verdict: equal`; `compare_with_recorded.py` against R0's two run directories exits 0,
    0 differences in each comparison; `closed_store_state.py` exits 0.
  - R0's twelve store files hash identically (SHA-256, size, `mtime_ns`) before and after all of it.
  - So a script that fails in the agent's run fails on the build's store, not on the pin.
- **The fingerprint both ways, prototyped.** A scratch script doing the prompt's §2 on R0's two
  stores: every check holds on both. The package's fingerprint has SGK's keys plus
  `encoding_format`, second; with it removed, the dictionaries are equal and in the same order;
  `ENCODING_FORMAT` is 1; the listings are 7542 lines each, equal but for the first, which gains
  ` encoding_format 1`; `compare_fingerprints(sgk, pkg)` through the package is one entry, kind
  `encoding`, `recorded` `None`, `current` 1; each sidecar compares empty with `sgk`. 7 s for both.
  The agent writes its own script; this one is not handed over.
- **The interrupter passes at `e150335`.** Both dummy tests, into the scratchpad, exit 0, 40 and 20
  lines, each equal to R0's once times, pids and paths are stripped. The agent runs them again.
- **What R0's logs held**, against which the log comparison is made (verification §1.2.5): each
  killed `stderr.log` the handler's two lines (`resumed-a` `SystemExit(143)`, `resumed-b`
  `SystemExit(1)`); the SciPy `IntegrationWarning` in `reference` and `resumed-b`; the prune's
  integrity warnings in `resumed-b`; three progress lines in `reference`'s QSI stage and one in
  `resumed-a`'s GkWKB stage. No line names `datastorekit` in any R0 build log but through paths.
- **The stage times.** R0's reference (verification §1.5.5), at load up to 339: `GkSource` **78 s**,
  QSI **1230.1 s**, the build 29 m 19 s. `dk-and-rpk-migration` R2's, at load 13–33: 41.8 s,
  299.2 s, 11 m 25 s.
- **The commands are R0's**, verbatim in verification §1.1, §1.1.2 and §1.3–§1.4, with
  `runregistry-extraction-04`, `--campaign runregistry-extraction --prompt 04` and the store names of
  the prompt's §1.
- **No Ray process left.** `ray-processes` is the test. A killed build's agents are waited for **by
  pid**, never by pattern, and how long they took is recorded beside R0's (none outlived either
  killed driver).
- **Shell traps.**
  - `stat` here is GNU coreutils: use `/usr/bin/stat -f '%m %N'` or Python's `os.stat`.
  - `grep` may wrap `ugrep`: use `git grep` or Python for any regex the log relies on.
  - The shell is `zsh`: an unquoted `$var` holding spaces is **not** split into words (use
    `${=var}` or an array), and a bare word beginning `=` is command expansion. Quote it.
- **Expected counts**, at `e150335`, serially, load 46–57 (one-minute):

  | Root | Tests | Result | Wall |
  |---|---|---|---|
  | AdaptiveLevin | 32 | OK | 1 s |
  | CosmologyModels | 39 | OK | 2 s |
  | ObjectFactories | 271 | OK | 15 s |
  | tools | 91 | OK | 12 s |
  | LiouvilleGreen | 148 | OK (1 skipped) | 25 s |
  | RunRegistry | 286 | OK | 85 s |
  | ComputeTargets | 693 | OK | 146 s |

  Every count is 03's. `test_wall_time_per_object` passed this time; it is README §8's first known
  flake, and has failed in every other recent run. A failure of it is re-run alone, not a stop.

- **The index:** 107.

**What the review exists to establish.** All else is mechanics.
- **(R1) Every launch ended as recorded.** `done`/0 and `killed`/143, each with a fingerprint, each
  driver's exit status in its launch log, and no Ray process of this repository left.
- **(R2) R1 is R0.** Both stores equal each other, `39809dca…`, and the fingerprints recorded by R0's
  `reference` and `resumed-c`, class by class. Nothing stored was recomputed; the prune removed
  exactly the unvalidated rows, if there were any.
- **(R3) The stores are sound**, by `store_soundness.py` and the census.
- **(R4) Every reader works, and none touches the store.**
- **(R5) The manifests say what ran.** Every `packages` is §0's record, every `git_dirty` `false`,
  every `git_head` the dispatch `HEAD`, and every `script_sha256` `b7935877…`.
- **(R6) L4 equals R0's**: the statistics counts, the reference stage counts, the chain's
  identities, the stage sequence, the printed lines and registry fields. Every log difference from
  R0 is listed and explained, and any line naming `datastorekit`, a `DeprecationWarning` or
  `_timestamp` is accounted for.
- **(R7) W1: the fingerprint both ways.** The script does what the prompt's §2 says, exits 0 on all
  four stores, and fails on each of the five breakages; the stores are untouched by it.
- **(R8) The lock.** Taken before `reference`, `pid` set at each launch, held through the readers,
  released; free now.
- **(R9) Nothing else under `var/` moved.** The 14 tombstones and R0's twelve files keep their hashes.

**Conventions.** 02's, carried. All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §8's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each **in the foreground**, serially, within the turn, and **only after the last
  launch has ended and the lock is released**. Under load, the four tests of README §8 are known to
  fail: re-run a failure alone before calling it real.
- **Scratch.** The agent's own subdirectory of the session scratchpad. Anything that must outlive
  the session, the working notes and between-launch captures included, goes under
  `var/runregistry-extraction-04/` (`CLAUDE.md` rule 5). No scratch file goes anywhere else in the
  repository, even for a moment, while a build may yet be launched. The breakage copies of the
  script live in the scratchpad and are never committed.
- **The venv.** The agent installs and uninstalls nothing.
- **`RunRegistry/` is frozen.** The agent changes nothing under it (README §5 rule 14). A defect
  found in it is recorded and the agent stops.
- **DatastoreKit and RayPoolKit.** Read only as correction 6 says; nothing run or changed there
  (README §5 rule 12).
- **The lock.** Every helper command and its exit status in `lock_04.txt`. Never wait for, retry or
  remove a lock; exit 3 or 4 at `acquire`, or a refusal naming another owner, is a stop.
- **Ray addresses.** The driver forces `address="local"`; `run_readers.sh` passes `local`. Nothing
  runs `main.py` or an `extract_*.py` at its default. **`ray stop` is never run.**
- **Never touch a process you did not start.**
- **Stop conditions are real.** A stop leaves the stores, the run directories and the tree as they
  are: no relaunch, no deletion, no fix. It goes to the user through the orchestrator, and the
  agent resumes with `SendMessage`. If it stops while holding the lock, it releases once no Ray
  process of this repository is alive, or says it cannot.
- **Usage limits.** A usage stop does not stop a detached build. Resume the agent with
  `SendMessage`, saying exactly where it stopped. If it commits but sends no report, review from the
  commit, its log and its outputs.

## 1. Before you dispatch

Done on 2026-10-14 at `e150335`, before this note was committed. Repeat steps 1–3 at the dispatch
`HEAD` if anything lands in between.

1. **The tree.** Branch and `HEAD`; `git status --short` and `git diff --cached` empty; `git worktree
   list` this checkout alone; `git log --oneline 5b81f2e..HEAD -- . ':!prompts' ':!docs'` empty.
2. **The registry.** Nothing `running` or stale.
3. **The lock and Ray.** `status`: free; `ray-processes`: none. `RAY_ADDRESS` unset. Record the load,
   and say so if `var/` has less than 10 GB free.
4. **The venv.** `package_provenance()` gives §0's record; `pip check` is clean.
5. **Baselines**, as §0's table.
6. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 107.
7. **The names are free**, and `var/datastores/` holds the 26 files, hashing as §0.
8. **The interrupter.** Both dummy tests at `HEAD`, into the scratchpad. Both pass.
9. **The scripts and the prototype**, on R0's stores and runs, as §0.
10. **The `var/` marker**: immediately before dispatch, `touch` a marker in the scratchpad.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt, and through it 02 and `dk-and-rpk-migration`'s 08;
- the campaign README and this board;
- `CLAUDE.md`'s three sections named above;
- 02's log, 03's log and 02a's log's hand-off, and verification §1;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every launch, the readers, the lock's release, W1 and the
  suites. Its log is `logs/04-rehearse-through-the-new-datastorekit.md`, in README §5.1's form, with
  what 02's log records, against R0, and W1's five breakages as diffs.
- **The hand-off.**
  - After each launch: one check, including `set-pid`, then a report giving the run id, the Python
    pid, the `sh` pid, the interrupter's pid and arguments if there is one, and the lock's `status`,
    then the end of its turn.
  - You resume it with the run's final `status.json`, the last lines of its launch log, and the
    wait's verdict.
  - It never waits, polls or sleeps on a build itself. Waiting by pid for a killed build's orphaned
    Ray agents, before the next launch or before `release`, is the one bounded wait it runs.
- **The files it may change** are the prompt's §7 list, nothing else. **It touches none of them, and
  creates nothing in the repository outside `var/`, until the last build launch has ended**, so
  that every manifest stamps `git_dirty: false`.
- **The board's header** says 5 of 10 landed once it commits; `prompts/INDEX.md` is the
  orchestrator's.
- **It retires no store**, R0's included. The orchestrator retires R0's and R1's four after the
  review (D6).
- **Stop and ask** on any of the prompt's §5 conditions. A fingerprint miss, or a W1 difference
  other than `encoding_format`, is the one R1 exists for: record, change nothing, stop. **A
  statistics count that differs, with the fingerprint holding, is not a stop**: it is a finding,
  recorded, explained if it can be, and opened as an issue.

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

## 3. The review — fifteen checks

1. **Scope.**
   - `git diff --name-status HEAD~1 HEAD`: `A` for the log, the script and the `*_04*` outputs; `M`
     for the verification document, this board and, if an issue was opened, `docs/OPEN_ISSUES.md`.
     README, `prompts/INDEX.md` and `orchestrator/` are not modified. No other tracked code, test,
     tool or script is modified, and nothing under `RunRegistry/`.
   - `git diff HEAD~1 HEAD -- docs/runregistry-extraction-verification.md` adds §2 and nothing else:
     §1 and the header are byte-identical.
2. **(R1) The runs.** Read every manifest, `status.json`, `stderr.log`, interrupter log and launch
   log under `var/runs/runregistry-extraction-04-*` yourself.
   - `reference` and `resumed-c` are `done`/0; `resumed-a` and `resumed-b` are `killed`/143, each
     with `fingerprint` and no `fingerprint_error`.
   - Each launch log ends `** driver exit status` 0 or 143, matching.
   - Each killed `stderr.log` has the handler's lines and no traceback.
   - Each interrupter log shows one `SIGTERM`, with the pid and stage it acted on, and the handler's
     line came within the second or two. The holds are those corrections 2 and 3 give, computed
     from R1's own reference.
   - The next launch's counts show the killed stage had stored some rows and not all, or, for
     `resumed-a`, none, as the log says.
   - No process the agent started is alive, and `ray-processes` lists nothing.
3. **(R5) The manifests.** For each: `git_head` the dispatch `HEAD`, `git_dirty` `false`; `packages`
   §0's record exactly, compared as JSON; `script_sha256` `b7935877…` in all four; `argv` holds
   `--build --rehearsal`, with `--resume` on `resumed-b` and `-c` only; `scope` begins `REHEARSAL
   build, not A3 v2:`; `results` names the right store; each `--purpose` names R1 of this campaign,
   the tree's commit, `datastorekit` 0.4.0 at `c8570e8`, and `39809dca`.
4. **(R2) Nothing stored was recomputed.** Run `stage_counts.py` on the four run directories
   yourself, and compare it with the log's table and with `stage_counts_02.txt`.
   - The `reference` column equals R0's, stage by stage.
   - Every stage completed before an interruption computes 0 on the next launch.
   - `resumed-b`'s `GkSource` = 291 − (stored by `resumed-a`) + (pruned); `resumed-c`'s QSI = 864 −
     (stored by `resumed-b`), and nothing else. The totals per class over the chain equal the
     reference's.
5. **(R2) The prune.** Run `prune_check.py` on the agent's saved `closed_store_state.py` output after
   `resumed-a`: every listed serial absent now (or reused by a later row), read `mode=ro`.
   `read_inventory` reports no `duplicate`. `resumed-b`'s integrity warnings count the same rows,
   Ray's folding included. If `resumed-a` stored none, the clause holds vacuously, and the log says
   so.
6. **(R2) The fingerprints.** Fingerprint both closed stores yourself, read-only:
   - `compare_stores.py <reference> <resumed>`: equal, each equal to its sidecar;
   - `compare_with_recorded.py` on both primaries `--against` R0's two run directories: 0
     differences each;
   - format 6, `39809dca…`, 21 classes.
7. **(R3) Integrity.** Run `store_soundness.py` and `replicated_serial_census.py` on both primaries
   yourself: `integrity_check` `ok`, `foreign_key_check` empty, one `version` row per shard (label
   `2025.1.1`), no census divergence, no inventory problem, `replication_in_flight` empty, no
   journal file.
8. **(R4) The readers.** Each reader's log under `var/runregistry-extraction-04/extract/`: exit 0, no
   `ReadOnlyMiss`, `ReadOnlyWrite`, `ModuleNotFoundError`, unpickling error or traceback; hashes and
   mtimes equal around each; the `nomodels extract_Gk_data` run printed the skip line. Each log
   compared with R0's same reader as line kinds. Run one reader yourself only if the lock is free
   and you take it for the run, and release it after.
9. **(R6) L4.**
   - Run `policy_statistics.py` on R1's `reference`, `resumed-b` and `resumed-c` yourself: 14 lines
     each; the counts R0's (291; 109 / 26 (25 complete, 1 minimal) / 156); against R0's same slug,
     equal line for line or equal sorted, as the log says.
   - The `** ` banner lines of each run, in order, equal R0's same slug (numbers and the campaign
     prefix stripped).
   - The resumed-chain identities of check 4 are in the log as a table beside R0's verification
     §1.5.3.
10. **(R6) The printed lines and the log differences.** For each slug, against R0's: the counts of
    `ALL WORK ITEMS COMPLETE in time`, `Queue summary:` and `(no compute items were evaluated`
    lines, each form equal with numbers stripped; any progress line in correction 4's terms; each
    final `status.json` `stage`. Then R1's `stderr.log`, launch log and `stdout.log` against R0's
    as line kinds, times, pids, ports, paths, numbers and the campaign prefix stripped: every kind
    in one and not the other is in the log with an explanation that holds. Search all twelve build
    logs and the reader logs yourself for `datastorekit` outside paths, `DeprecationWarning`,
    `_timestamp`, `Traceback`, `ModuleNotFoundError`.
11. **(R7) W1, by reading.** `fingerprint_both_ways.py` does the prompt's §2 items 1–5 in order:
    `read_inventory` and `factories` imported as `fingerprint_store` imports them; SGK's
    `FINGERPRINT_FORMAT` passed, not typed; the dictionary comparison with `encoding_format`
    removed, and order checked as correction 5 says; every differing path printed with both values;
    the listing check on the first line and the rest; the three comparisons; the sidecar. It writes
    nothing, imports nothing that starts Ray, exits 0 iff every check holds, and is `black`-clean.
12. **(R7) W1, run here.** Run it on the four stores: exit 0, output equal to
    `fingerprint_both_ways_04.txt` but for times; every store file byte-identical around it. Apply
    the log's five breakage diffs to a scratchpad copy: each exits non-zero and names what the
    prompt says it must.
13. **(R8) The lock.** `lock_04.txt`: `acquire` exit 0 before `reference`'s manifest was created;
    `set-pid` exit 0 at each launch, with that run's driver pid; `release` exit 0 after the last
    reader's log ends, and any refusal before it explained. Now `status`: free; `ray-processes`:
    none. `~/ScienceData/` holds what it held before dispatch.
14. **(R9) `var/`.** The file list under `var/datastores/`, with hashes, before the agent's first
    launch (its record) and now: the 26 unchanged, the only additions R1's twelve. A3 v2's guard
    refused and changed nothing. `find var -newer <marker>` lists only R1's runs, stores, sidecars
    and working directory.
15. **The records and the suites.**
    - The board: the header (5 landed), the §1 row for 04, L1–L4 (R1) and W1 ✅, the §2 R1 row
      holding with the two runs that finished `done`, `### 5.7 After prompt 04`; §3 entries and
      the index for anything opened, the count right by counting. Verification §2 has §2.1–§2.10 as
      correction 7 says, every command verbatim, and its numbers match the run directories. The log
      classifies §0's eight corrections and the five breakages are diffs.
    - All seven roots, serially, at the commit, every count as §0.

## 4. After the review

- **Record it on this board** as a §5.7 *Orchestrator review of prompt 04* paragraph, in 02's form.
  Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board and in the log's header;
  - the board's §2 R1 row, if the agent left it ⬜ with every part holding;
  - this note added to the board's list of notes, marked "used for 04";
  - `prompts/INDEX.md`'s row: 5 of 10 landed, R1 holds.
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:** R1–R9, each in a line; the fingerprints against `39809dca…` and R0's two;
  W1 on the four stores; the statistics counts against R0's; where each interruption fell and what
  the resume pruned and recomputed; every log difference from R0 and its explanation; the lock;
  anything opened.
- **Then D6.** The four stores, R0's and R1's, are retired with `store retire`, each after a dry run,
  in a commit of its own, as `718e650` did, while SGK's own code still compares them
  (`[00-recorded-fingerprints-lack-the-encoding-format]`). It deletes store files: **ask the user
  before running it.** 05–09 stay held on E2.
