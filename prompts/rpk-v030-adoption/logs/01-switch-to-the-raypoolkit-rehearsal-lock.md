# Log — rpk-v030-adoption prompt 01, switch to the `raypoolkit` rehearsal lock (`v0.3.0`)

- **Subject:** [`01-switch-to-the-raypoolkit-rehearsal-lock.md`](../01-switch-to-the-raypoolkit-rehearsal-lock.md),
  board items P1, D1, T1, C1, R1 and B1, and gate FP.
- **Commit:** the commit that adds this log (parent `5add52d`).
- **Result:** landed. Acceptance 1–6 hold, and no stop condition of the prompt's §4 fired. In
  ComputeTargets, README §7's known wall-clock test, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
  failed under load both before and after. Re-run alone, it passed once each time; it is recorded
  below. Every other test of the seven roots passes.
- **Model:** Opus 5.5.
- **Date:** 2026-10-25. The clocks went back from BST to GMT at 02:00, during the after-roots.
  Times before that are BST, and later ones GMT, each as recorded with its offset.

## What shipped

- **The pin.** `requirements.txt:76` moved from `…/RayPoolKit@v0.2.0` to `…@v0.3.0`; no other line
  moved.
  - The venv was installed from it: `raypoolkit` 0.3.0 at `67a75b97a656accb98195a4bda93f64e6a5f3fb2`,
    `requested_revision` `v0.3.0`, not editable. `datastorekit` 0.5.0 (`f47b9a25…`) and
    `runregistrykit` 0.2.0 (`0bae3a3c…`) are as before.
  - The install is not in the commit, and `git revert` does not undo it (README §4.1).
- **`config/ray_local.py`** (new): a comment saying what the value is for, then
  `RAY_TEMP_DIR = "/tmp/ray-sgk"`.
- **The three wrappers** pass `_temp_dir=RAY_TEMP_DIR` to `_real_ray_init`. Each wrapper imports it
  as `from config.ray_local import RAY_TEMP_DIR` in the function that defines it, beside its
  `import ray`. That is the drivers' style: each imports repository modules (`config.model_list`)
  inside the function.
  - `run_build` (`RunDrivers/quadsource_atol_sweep.py`) and `scoped_pipeline_run.main`: the
    keyword alone.
  - `run_child`: `address="local"` and the keyword. Its printed line now says `(address=local,
    num_cpus=…, include_dashboard=False)`.
  - Nothing added to `run_child` contains a lower-case `rehearsal`, and
    `test_run_child_does_not_read_the_scope` passes.
- **The seven scripts**: `main.py` and the six `extract_*_data.py` readers.
  - Each gains `from config.ray_local import RAY_TEMP_DIR` after `from config.model_list import
    build_model_list`, beside its other `config` imports.
  - Each keeps one top-level `ray.init(address=args.ray_address, _temp_dir=RAY_TEMP_DIR if
    args.ray_address == "local" else None)`, as `black` lays it out.
  - `main.py`'s comment above the call gains one sentence: the temp dir lets the lock see the
    instance, and is given only with `local`, because with `auto` it would change which cluster
    Ray finds, and with `host:port` it would split the driver's files from the cluster's.
- **`RunDrivers/tests/test_ray_temp_dir.py`** (new, 5 tests, `ast` only, no Ray). The module walks
  every live `.py` file: 119 files, the 118 the orchestrator counted plus `config/ray_local.py`.
  Its tests are:
  - `test_the_temp_dir_is_the_constant`;
  - `test_the_ray_starts_are_the_ten` (the found starts as a sorted list, its length included);
  - `test_each_wrapper_starts_a_local_instance_in_the_temp_dir`;
  - `test_each_script_gives_the_temp_dir_with_local_only`;
  - `test_each_start_takes_the_temp_dir_from_config`. Each file with a start imports
    `RAY_TEMP_DIR` from `config.ray_local` and binds none of its own.

  Each docstring names the failure the test guards: an SGK Ray instance the lock cannot see.
- **The helper is gone.** `git rm tools/rehearsal_lock.py tools/tests/test_rehearsal_lock.py`
  removed 42 tests.
- **`CLAUDE.md`:**
  - **The pin sentence** now says "pinned at `v0.3.0` (moved to it on 2026-10-25 in the
    `rpk-v030-adoption` campaign, which made its `raypoolkit.rehearsal_lock` this repository's
    rehearsal lock)". The paragraph is rewrapped.
  - **Rule 5's Ray check** is now `./venv/bin/python -m raypoolkit.rehearsal_lock ray-processes
    --ray-temp-dir /tmp/ray-sgk`, and the rule is rewrapped. The rest of the rule stands.
  - **"The shared rehearsal lock"** is rewritten. It now says:
    - the location, the module, the contract at `v0.3.0` and SGK's three values;
    - the protocol, with the full `acquire`, `set-pid` and `release` lines given once;
    - exit 3 means stop and ask, including while the old location exists;
    - the lock finds an instance by its temp dir, which the new test guards;
    - each lock command is a shell command of its own;
    - the address and `ray stop` rules, kept as they were.

    The bullet pointing at StochasticInstantons' brief and `runregistry-extraction` §4.1 is gone.
- **`measure/out/`:**
  - `01-pip.txt`: §2.1 and §2.2;
  - `01-roots.txt`: before and after, with the re-runs;
  - `01-guard.txt`: the module, M1–M5 and the clean run after;
  - `01-lock.txt`: every lock command;
  - `01-rehearsal.txt`: the pre-launch checks, the launch, the check, the final state, the
    sidecar and fingerprint, and the readers.
- **This log and the board:**
  - the header;
  - the §1 row;
  - items P1, D1, T1, C1, R1 and B1;
  - gate FP;
  - §5.1.

## Deviations from the prompt

1. **The checklist was read at RayPoolKit `feae4e5`, not at `v0.3.0`** (Read first 3). The
   checklist, `docs/adoption/rehearsal-lock/secondarygwkit.md`, is not in the tag. `git cat-file -e
   v0.3.0:<path>` says it "exists on disk, but not in 'v0.3.0'". It was written after the release,
   in `127b2b4`, so it was read with `git show feae4e5:<path>` at RayPoolKit's `HEAD`, which is
   clean. The contract was read at `v0.3.0`. `git diff v0.3.0 feae4e5 -- raypoolkit pyproject.toml
   docs/rehearsal-lock.md` is empty, so the contract and code are the tag's either way. RayPoolKit's
   repository was not changed. **STRUCTURALLY REQUIRED.**
2. **The guard test's key is the qualified name of the enclosing function**, compared as a sorted
   list with its length, not as a set.
   - The prompt (§2.4) asks for the set of `(file, enclosing function or <module>)`.
   - The three wrapper calls are each inside a nested function named `_local_ray_init`. Two of them
     are in one file, so a key on the innermost function gives a set of nine.
   - The key used here is `run_build._local_ray_init`, `run_child._local_ray_init` or
     `main._local_ray_init`, or `<module>`. So ten stays ten, and a duplicate start fails.

   **IMPLEMENTATION CHOICE** (the orchestrator's note).
3. **A fifth test, `test_each_start_takes_the_temp_dir_from_config`**, beyond the checks §2.4
   lists. It catches a file that sets a local `RAY_TEMP_DIR`, whose spelling the lock may not
   name. That is the same failure the module guards, and the test is within C4's charter.
   **IMPLEMENTATION CHOICE.**
4. **The manifest of the rehearsal says `git_dirty: true`**, at `git_head` `5add52d`, where R3's
   said false.
   - The rehearsal is §2.8, before the one commit, so the tree carries 01's edits. The fingerprint
     records the same, and so does the sidecar's `create` history.
   - The `--purpose` says so: "tree 5add52d with 01's edits uncommitted".

   **STRUCTURALLY REQUIRED.**
5. **`CLAUDE.md`'s lock section grew from 12 lines of body to 25**, more than the prompt's "a few".
   - The prompt asks for the full `acquire`, `set-pid` and `release` lines once. They are a 5-line
     `bash` block, each line over 100 columns, as code blocks elsewhere in `CLAUDE.md` are.
   - The prompt also asks the section to state five new points itself, not by pointer.
   - The prose is otherwise as tight as the content allows.

   **IMPLEMENTATION CHOICE.**
6. **The rehearsal compared the store with two recorded fingerprints**: R3's `reference`, as the
   prompt names, and this run's own `status.json`. The verdict is `equal` against both.
   **IMPLEMENTATION CHOICE.**
7. **The lock commands' timestamps** were taken with zsh's builtin `strftime`, so that each lock
   command was the only process its call started. The calls used no `date` and no pipe.
   **IMPLEMENTATION CHOICE.**
8. **The machine-wide `pgrep -lf 'gcs_server|raylet|ray::'` of R3's pre-launch checks was not
   run.** The brief forbids `pgrep -f`. `ray-processes --ray-temp-dir /tmp/ray-sgk`, the old
   helper's `ray-processes`, and the Gate's checks stood in for it. **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT`.

## Verification performed

**Gate** (2026-10-25 01:41 BST):
- the tree was clean at `5add52d`, and `git diff --stat 2b27509 HEAD -- . ':!prompts' ':!docs'`
  was empty;
- one worktree;
- `runregistrykit list` showed 51 runs (49 finished, 2 unknown), none `running` or stale;
- the venv's provenance was as the Gate states;
- neither lock location existed;
- the old helper's `status` said free and its `ray-processes` none (exit 0);
- `/tmp/ray-sgk` did not exist, and `RAY_ADDRESS` was unset.

`~/ScienceData/` holds `ChamPBH-rehearsals/` and `StochasticInstantons-rehearsals/`. R3 found the
first alone; neither is a lock.

**Acceptance 1, the pin** (`01-pip.txt`):
- The dry run exited 0 and printed `Would install raypoolkit-0.3.0`, with no other `Would install`
  line.
- The install printed `Uninstalling raypoolkit-0.2.0` and then `Successfully installed
  raypoolkit-0.3.0`.
- `diff` of `pip freeze` before and after is the one line, 76:
  `…@eaef0ce6…` to `…@67a75b97a656accb98195a4bda93f64e6a5f3fb2`.
- `pip check` printed "No broken requirements found."
- `package_provenance` gives `raypoolkit` 0.3.0 at `67a75b97…`, `requested_revision` `v0.3.0`,
  `editable` false. The other two are identical to before.
- The first line of `--help` is `usage: python -m raypoolkit.rehearsal_lock [-h]`.

**Acceptance 2, the helper gone:** §2.5's grep found 18 lines before the edits: 4 in `CLAUDE.md`,
5 in the helper and 9 in its tests. It found 4 after `git rm`, all in `CLAUDE.md`, and **1** after
§2.6, `CLAUDE.md:169`, the sentence naming the old location.

**Acceptance 3, the ten starts** (`01-guard.txt`): the module ran 5 tests, all OK. The deliberate
breakage is below.

**Acceptance 4, the roots** (`01-roots.txt`, serial, foreground, README §7's command):

| Root | Before (`5add52d`, rpk 0.2.0) | After (01's edits, rpk 0.3.0) | Change |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 700, 1 failure (wall clock) | 700, 1 failure (wall clock) | 0 |
| CosmologyModels | 39 OK | 39 OK | 0 |
| LiouvilleGreen | 148 OK (skipped=1) | 148 OK (skipped=1) | 0 |
| ObjectFactories | 280 OK | 280 OK | 0 |
| RunDrivers | 115 OK | 120 OK | **+5**, the new module |
| tools | 91 OK | 49 OK | **−42**, the helper's tests |

- **The wall-clock test.** The one ComputeTargets failure each time is
  `test_tk_wkb_phase.TestCost.test_wall_time_per_object`. It took 0.061 s before and 0.103 s
  after, against 0.06 s.
  - Re-run alone, before (load 32–34): fail, pass, fail, fail.
  - Re-run alone, after (load 14–28): pass, fail, fail.
  - It is README §7's known wall-clock test, not a stop. The 699 other tests pass.
- **The four tests that pin the forms pass:** `test_ray_address_default`,
  `test_inventory_consumers`, `test_killed_handler` and `test_build_rehearsal_scope`, each in its
  root.
- **`black --check`** is clean on all 11 touched `.py` files.
- The counts equal the orchestrator's baselines at `2b27509`.

**Acceptance 5, under real Ray** (`01-lock.txt`, `01-rehearsal.txt`): the lock transcript, the
run and the readers are below. **Acceptance 6** is the staged list, shown before the commit.

## The lock transcript, in brief

Every lock command was a Bash call of its own, run from the repository root as `./venv/bin/python
-m raypoolkit.rehearsal_lock …`. Each was redirected with `>> 01-lock.txt 2>&1`, with its exit
recorded by the builtin `echo`, and no pipe. No `--purpose` named the temp dir.

| When (GMT) | Command | Exit | What it said |
|---|---|---|---|
| 01:03:47 | `acquire --project SecondaryGWKit --purpose "rpk-v030-adoption 01: rehearsal under the new lock" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | 0 | acquired; owner file `pid: -` |
| 01:04:36 | `set-pid … 73154` (the driver's Python pid) | 0 | recorded |
| 01:04:44 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | **1** | 18 processes: `gcs_server` 73180 and `raylet` 73217 (both children of 73154), 3 more Python children of 73154, 2 of the raylet, and 11 workers (`ray::SerialPoolBroker`, `ray::Datastore` ×4, `ray::numeric_with_phase_cut` ×6). Not the driver, as the contract says (§6) |
| 01:04:51 | `release …` | **1** | "the recorded pid 73154 is alive; not releasing" |
| 01:05:00 | `acquire …` again, the same values | **3** | held; the owner file printed; "stop and ask the user" |
| 01:05:07 | `status --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | 0 | `lock: HELD`; `owner pid 73154: alive`; `Ray processes of /tmp/ray-sgk: 18`; no `old location` line |
| 01:19:06 | `ray-processes …`, after the driver died | 0 | none |
| 01:27:50 | `ray-processes …`, after the readers | 0 | none |
| 01:27:58 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | 0 | released |
| 01:28:03 | `status --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | 0 | `lock: free (… does not exist)`; `Ray processes of /tmp/ray-sgk: none`; no `old location` line |

The owner file, between `set-pid` and `release`, read `project: SecondaryGWKit`; `purpose:
rpk-v030-adoption 01: rehearsal under the new lock`; `acquired: 2026-10-25T01:03:48+00:00`; `host:
mps026088`; `ray_temp_dir: /tmp/ray-sgk`; `ray_address: in-process`; `pid: 73154`.

## The rehearsal

- **The launch**, at 01:03:57 GMT, after the `acquire`, in a call of its own, as R3 §1.1 launched
  it:
  - `nohup sh -c "PYTHONPATH=. ./venv/bin/python -u RunDrivers/quadsource_atol_sweep.py --build
    --rehearsal --database var/datastores/rpk-v030-adoption-01-reference.sqlite --job-name
    rpk-v030-adoption-01 --atol 1e-32 --rtol 1e-7 --cpus 6 --register reference --purpose \"\$0\"
    --campaign rpk-v030-adoption --prompt 01; echo …" "<purpose>" > var/runs/rpk-v030-adoption-01-reference.launch.log 2>&1 &`
    and then `disown`;
  - the `sh` pid was **73146** and the Python pid **73154**;
  - the purpose names this prompt, `5add52d` with 01's edits uncommitted, `raypoolkit` 0.3.0
    `67a75b9`, `datastorekit` 0.5.0 `f47b9a2`, `runregistrykit` 0.2.0 `0bae3a3`, `39809dca` and
    R3's `reference`;
  - the load at launch was 21.99 26.29 26.55.
- **The one check**, made once launch-log line 12 said "Started a local Ray instance." (01:04:07),
  after a bounded wait on that line:
  - **The manifest:**
    - `packages` equals the venv's `package_provenance`, with `raypoolkit` 0.3.0 at `67a75b97…`;
    - `git_head` `5add52d…` and `git_dirty` true (deviation 4);
    - `host` is this machine;
    - `script` is `RunDrivers/quadsource_atol_sweep.py`, with `script_sha256` `5b984c75…`;
    - `scope` is the rehearsal's.
  - **`status.json`:**
    - `running`, pid 73154, which answered `kill -0`;
    - `host` `id` and `boot` equal to `this_host()`;
    - stage "CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS".
  - **The registry:** `runregistrykit list` shows the run `running`.
  - **The lock:** the five lock commands of the table, each in its own call.
- **The wait.** The agent ended its turn. The orchestrator made one wait on pid 73154, which ended
  at 01:18:38 GMT. Neither of them polled.
- **The final state:**
  - `status.json` says `done`, `exit_code` 0, with the last heartbeat at 01:18:34 and stage
    "CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete";
  - the launch log ends "** driver exit status 0";
  - neither 73154 nor 73146 answers `kill -0`;
  - no line of the launch log, `stdout.log` or `stderr.log` holds a `Traceback`, a
    `DeprecationWarning` or a `ConfigError`;
  - `list` shows the run `done`.
- **Wall time** was 14 m 36 s (01:03:58–01:18:34), against R3's 3 m 52 s. The 15-minute load was
  38.6 at 01:19, against R3's 11–14. Nothing a build computes moved (the fingerprint, below).
- **The fingerprint** (§2.8.5, as R3 §1.1 and §1.3.4 did):
  - `store create` ran on the closed primary and wrote sidecar `store_id`
    `0213c3c625fc49598db4a5cf032711e5`.
  - `store fingerprint --write` gave digest
    **`39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`**, format 6, 21 classes
    (QuadSourceIntegral 864, GkSource 291, GkWKBIntegration 3477, GkNumericIntegration 1368, …),
    problems none, and "it replaced none".
  - The run's own fingerprint (`encoding_format` 1) is the same digest. Its `fingerprint_sidecar`
    is "not written … (there is none)", as R3's `reference` was before its `store create`.
  - `compare_with_recorded.py <store> --against var/runs/rrk-v020-adoption-03-reference-20261018T232838
    var/runs/rpk-v030-adoption-01-reference-20261025T010358` exited 0, verdict `equal`. There were
    0 differences against the sidecar, against R3's `reference` (taken at `aab4075`, `git_dirty`
    false) and against this run's.
  - The store's five files had the same SHA-256, mtime and size before `store create` and after
    the comparison, and no journal appeared.
- **The readers** (§2.8.6), 01:19:38–01:27:29 GMT, with the lock held: `bash
  docs/dk-and-rpk-migration/rehearsal/run_readers.sh var/datastores/rpk-v030-adoption-01-reference.sqlite
  var/rpk-v030-adoption-01/extract`. The script exited 0.

  | Reader | Exit | Wall | Files written |
  |---|---|---|---|
  | `extract_Gk_data` | 0 | 53 s | 369 |
  | `extract_GkWKB_data` | 0 | 236 s | 2436 |
  | `extract_GkSource_data` | 0 | 67 s | 166 |
  | `extract_TkWKB_data` | 0 | 30 s | 27 |
  | `extract_tensor_source_data` | 0 | 26 s | 27 |
  | `extract_QuadSourceIntegral_data` | 0 | 47 s | 76 |

  - For every reader the store's files were identical before and after (SHA-256, mtime, size), and
    no journal appeared.
  - No reader log holds `Traceback` or `Error`.
  - **Each reader's Ray used the temp dir.** `/tmp/ray-sgk/` holds seven sessions: the build's
    (`…_73154`) and one per reader, timed with each (01:19:42, 01:20:38, 01:24:38, 01:25:46,
    01:26:17, 01:26:43).
  - The 1-minute load reached 187 at their end.
  - Then `ray-processes` exited 0, `release` 0, and `status` showed the lock free (the table above).

## The deliberate-breakage record

Each mutation was applied to the working tree by a script, as below. The module was run, and the
file was then restored from a byte-copy (`cmp` equal), or removed (M4). The module then ran clean
again: 5 tests, OK. The outputs are in `01-guard.txt`.

**M1** — drop `_temp_dir` from `extract_Gk_data.py`'s `ray.init`.
```diff
@@ -109,7 +109,6 @@
 # attaches
 ray.init(
     address=args.ray_address,
-    _temp_dir=RAY_TEMP_DIR if args.ray_address == "local" else None,
 )
```
Failed: `test_each_script_gives_the_temp_dir_with_local_only`, subtest
`('extract_Gk_data.py', '<module>')`, "ray.init passes no temp dir". 1 failure.

**M2** — drop `address="local"` from `run_child`'s wrapper.
```diff
@@ -487,7 +487,6 @@
             f"(address=local, num_cpus={args.cpus}, include_dashboard=False)"
         )
         return _real_ray_init(
-            address="local",
             num_cpus=args.cpus,
             include_dashboard=False,
             ignore_reinit_error=True,
```
Failed: `test_each_wrapper_starts_a_local_instance_in_the_temp_dir`, subtest
`('RunDrivers/quadsource_atol_sweep.py', 'run_child._local_ray_init')`, "the real ray.init passes no
address". 1 failure.

**M3** — make `main.py`'s `_temp_dir=RAY_TEMP_DIR` unconditional.
```diff
@@ -316,7 +316,7 @@
 # with host:port it would split this driver's files from the cluster's
 ray.init(
     address=args.ray_address,
-    _temp_dir=RAY_TEMP_DIR if args.ray_address == "local" else None,
+    _temp_dir=RAY_TEMP_DIR,
 )
```
Failed: `test_each_script_gives_the_temp_dir_with_local_only`, subtest `('main.py', '<module>')`,
on the `ast.unparse` comparison. 1 failure.

**M4** — a new file `tools/_probe.py`.
```diff
--- /dev/null
+++ b/tools/_probe.py
@@ -0,0 +1,3 @@
+import ray
+
+ray.init(address="local")
```
Failed: `test_the_ray_starts_are_the_ten`: 11 found against 10, with `('tools/_probe.py', '<module>')`
in the list. 1 failure. The script and wrapper tests check only the ten keys, so they did not see
it, by design.

**M5** — `RAY_TEMP_DIR = "/tmp/ray-sgk/"`.
```diff
@@ -4,4 +4,4 @@
 # instance by its temp dir, so an instance started without it is invisible to the lock. The lock
 # compares the string as given, so it is spelled exactly so everywhere, and on the lock's command
 # line as `--ray-temp-dir /tmp/ray-sgk`. It is short because Ray's Unix sockets live under it.
-RAY_TEMP_DIR = "/tmp/ray-sgk"
+RAY_TEMP_DIR = "/tmp/ray-sgk/"
```
Failed: `test_the_temp_dir_is_the_constant`, `'/tmp/ray-sgk/' != '/tmp/ray-sgk'`. 1 failure.

## Observations not acted on

1. **README lines 9–13** say the checklist is read at the tag `v0.3.0`. It is not in the tag
   (deviation 1). The orchestrator's note assigns that correction to the review.
2. **Four `docs/handover/a3-v2/**/scripts/*.sh`** still call `tools.rehearsal_lock`, which is now
   deleted. They are records of dated runs, which §5 of the prompt and README §1 leave as written.
   Re-run, they would fail.
3. **The rehearsal's wall time was 3.8 × R3's under a 15-minute load near 39.** It is noted only to
   explain the time; nothing a build computes moved.
4. **`RunDrivers/scoped_pipeline_run.py:24`**, the module docstring, still describes "main.py's own
   `ray.init(address=...)`". It is true as written. The wrapper ignores `main.py`'s arguments, the
   temp dir included.
5. **`/tmp/ray-sgk/` keeps the seven sessions' files**, as Ray left them, just as it leaves them
   under `/tmp/ray`. Nothing here deletes them. The lock matches processes, not files, so they do
   not affect it: `ray-processes` exits 0 with them present.
6. **The registry after:** `list` shows 52 runs (50 finished, 2 unknown), none `running` or stale.
7. **`run_child` is checked by `ast` only.** The rehearsal ran `--build` (`run_build`); no real-Ray
   run of the tolerance sweep's child is in scope. The orchestrator's note says the same.

Nothing was opened on the board's §3, so `docs/OPEN_ISSUES.md` is unchanged.

## State handed on

- **To `handover` 09:**
  - the venv on `raypoolkit` 0.3.0, at `67a75b97…`;
  - the lock commands of `CLAUDE.md` with SGK's values;
  - every SGK Ray start giving `/tmp/ray-sgk`;
  - the lock released, and `~/.raypoolkit/rehearsal-lock` absent.

  `~/.raypoolkit/` itself remains; `acquire` made it.
- **To `handover` 10** (the Mac Studio): its checkout still pins `v0.2.0` until it runs
  `pip install -r requirements.txt` (README §4.2). Its Gate allows 01's two edits to `main.py` and
  `config/`.
- **The rehearsal store** `var/datastores/rpk-v030-adoption-01-reference.sqlite` (sidecar
  `0213c3c6…`) and the readers' output under `var/rpk-v030-adoption-01/` are left for the user to
  retire (README §3).
- **ChamPBH and StochasticInstantons** still pin `v0.2.0` and use the old location (U2). While
  either holds `~/ScienceData/.rehearsal-lock`, SGK's `acquire` exits 3.
