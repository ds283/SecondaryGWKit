# Log — runregistry-extraction prompt 02a, the rehearsal-lock helper

- **Subject:** [`02a-the-rehearsal-lock-helper.md`](../02a-the-rehearsal-lock-helper.md), board items H1–H4.
- **Commit:** `c863055` (parent `96a7f6e`).
- **Result:** landed. H1–H4 pass; no stop condition of the prompt's §4 fired. `ComputeTargets` shows
  its known wall-clock failure, and see "Verification performed" for how that re-ran alone.
- **Model:** Sonnet 5.5.

The orchestrator's note for this prompt (its §0, "Corrections to the prompt", cited below as "the
note, correction N") governs where it differs from the prompt.

## What shipped

- **`tools/rehearsal_lock.py`** (new), standard library only. Commands `acquire --purpose`,
  `set-pid`, `release`, `status`, `ray-processes`, with the exit statuses of §2.1's table, usage
  errors 2, and no option to break, wait for or retry. The owner file is the seven keys of the
  brief, in order, written to `owner.tmp` and renamed with `os.replace`. `ray_processes()` takes
  every process whose command line holds this checkout's `venv/` path followed by
  `/site-packages/ray/`, as written and as `realpath`, plus every descendant, and never itself or
  its ancestors. It reads `ps` through one function, `_ps_output`.
- **`tools/tests/test_rehearsal_lock.py`** (new): 42 tests, the eight points of §2.2.
- **`CLAUDE.md`**: one inserted hunk, "The shared rehearsal lock", directly before "Repository
  mechanics" (that is, directly after "`RunRegistry/` is frozen"). Nothing else changed.
- **`measure/out/02a-interop.txt`** and **`02a-real.txt`**: the two proofs.
- This log, and the board: header, the §1 row, H1–H4, `### 5.4`.

## Deviations from the prompt

1. **SI's and ChamPBH's `release_lock` are not run; direction 1 shows their refusal by reading,
   and direction 2 leaves the other project's lock in place** in its temporary `HOME`. **STRUCTURALLY
   REQUIRED** (the note, correction 1: Claude Code's removal check denies it; the user's ruling).
   `02a-interop.txt` quotes both functions (SI lines 130–147, ChamPBH 131–148). H3's "no temporary
   lock left" is read as the note says: none of SGK's, and each other project's where its
   `acquire_lock` put it. One temporary `HOME` per project, direction 1 then 2.
2. **One reason to refuse per refusal test.** **IMPLEMENTATION CHOICE** (the note, correction 2). Two
   extra tests per command (`set-pid`, `release`) follow from it: an owner of another project with
   `ray_temp_dir: in-process`, and an owner with SGK's project name and a cluster's temp dir. Without
   them a mutation of the `project` check survives behind the `ray_temp_dir` check, because SI's and
   ChamPBH's owners fail both.
3. **`set-pid`'s test starts from a fixed past `acquired`** (`2026-01-01T00:00:00+00:00`), not from
   `acquire`, and compares the file byte for byte but for `pid`. No `sleep`. **IMPLEMENTATION CHOICE**
   (the note, correction 3).
4. **Functions take `lock_dir=None`, resolved at call time, and `setUp` patches `HOME` for every
   test** as well as passing a lock directory. **IMPLEMENTATION CHOICE** (the note, correction 4).
5. **§2.2 point 6's `pgrep`/`grep` lines** are R0's `pgrep -lf gcs_server|raylet|ray::` and
   `grep ray`, neither naming the venv; no rule excludes processes by `argv[0]`. **IMPLEMENTATION
   CHOICE** (the note, correction 5).
6. **Proof mechanics** as the note's correction 6 gives them. **IMPLEMENTATION CHOICE.** The
   `sleep 300 &` was accepted, so no fallback was used. The probe was started with `nohup … > probe.out
   2>&1 &`, and its pid from `$!` and from its printed line agree (2778). Wait by pid, a bounded
   loop of `kill -0` every 2 s, 300 s ceiling, no `pgrep`. Each proof was run from a shell script in
   the scratch directory (`interop.sh`, `real.sh`) that prints each command, its output and its exit
   status. Nothing was denied; no route was used to avoid a denial.
7. **`hostname -s` itself** is run (`subprocess`), with `socket.gethostname()` cut at its first dot as
   the fallback if it cannot be run. **IMPLEMENTATION CHOICE** (the note, correction 7). On this machine
   both are `mps026088`.
8. **`ray_processes()` also never lists this process's own descendants** (its own `ps`), and uses
   this process and its ancestors as seeds for the descendant walk, though it does not list them
   (siblings under an ancestor Ray process are Ray's). **IMPLEMENTATION CHOICE.** The prompt says
   "never its own process or its ancestors"; without this a helper run from inside a Ray process would list
   its own `ps`. A test (`test_skips_its_own_pid_and_its_ancestors`) fixes it.
9. **A process table that cannot be read is a refusal.** `acquire` releases and returns 4,
   `release` returns 1, `ray-processes` returns 1 and `status` says so. **IMPLEMENTATION CHOICE**, the
   safe direction; the prompt is silent.
10. **`acquire`'s `--purpose` must be one non-empty line** (a newline would break the seven-line
    format); otherwise a usage error, exit 2. **`set-pid` refuses a pid of zero or below** (exit 1,
    since `os.kill(0, 0)` signals a process group). **IMPLEMENTATION CHOICE.**
11. **Mutation 2 is applied to `_not_sgk_reason`**, which `release` and `set-pid` share, so that it
    also fails `set-pid`'s tests. **IMPLEMENTATION CHOICE**: the project check is one function.
12. **`black` reformatted both new files once** before the tests were re-run. **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT`.

## Verification performed

**The roots**, README §8's command, serially, in the foreground, the real lock free (it was released
before any suite). Load at the `tools` run 21.9 (`uptime`). Outputs are in the scratch directory.

| Root | Baseline `c89fc9e` | Now |
|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK |
| CosmologyModels | 39 OK | 39 OK |
| ObjectFactories | 271 OK | 271 OK |
| tools | 49 OK | **91 OK** (49 + 42 new) |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) |
| RunRegistry | 286 OK | 286 OK |
| ComputeTargets | 693, `test_wall_time_per_object` only | 693, `test_wall_time_per_object` only (0.0791 s against 0.06 s) |

- **The flake alone.** Re-run alone at load 26–27: failures at 0.0717, 0.0714 and 0.0910 s, one pass.
  The baseline passed alone twice, at load 31. `git status` shows this prompt changes nothing
  under `ComputeTargets/` or any code the test runs; the test is README §8's first known flake.
  A fifth alone run's verdict line was not captured, only its cost line.
- The AdaptiveLevin and CosmologyModels runs were issued in the same turn as the `tools` run, so they may
  have overlapped it; all three are short and passed. The other four roots ran one per call.
- `black --check` on the two new files: unchanged.
- `git diff --stat HEAD -- RunRegistry/ main.py 'extract_*.py' docs/` is empty. `CLAUDE.md`'s diff
  is one hunk (`git diff CLAUDE.md | grep -c '^@@'` gives 1).
- **H3**, `02a-interop.txt`: SGK holds, for SI then ChamPBH: `set-pid` 0; each script's `status`
  prints SGK's owner with "owner pid N: alive"; `acquire_lock` returns 3 (the owner on stderr);
  SGK's own `release` exits 1 while the pid is alive (the owner's sha256 is the same before and
  after); after the `sleep` is killed `release` exits 0 and the lock is gone. The other holds: SGK
  `acquire` exits 3 and prints the owner, `set-pid` 1, `release` 1, `status` 0 shows it, and the
  owner's sha256 (SI `913da846dcbe5181`) is unchanged. Both other-project locks were left in place.
- **H4**, `02a-real.txt`: status free; `acquire` 0 at 14:42:29; probe pid 2778; `set-pid` 0;
  `ray-processes` 1, listing nine processes (a `gcs_server`, a `raylet`, six `Python` processes with a bare interpreter as `argv[0]`, and a `ray::IDLE` worker);
  `release` 1 with the pid alive; after `kill -TERM` every one of
  the ten pids was gone 3–5 s later; `ray-processes` 0; `release` 0 at 14:42:56; `status` free.
  The real lock was held 27 s. `~/ScienceData` holds `ChamPBH-rehearsals` only.
- `python -m RunRegistry list` before and after: 33 runs, 31 finished, 2 unknown, none running.

## The deliberate-breakage record

Each mutation applied alone to `tools/rehearsal_lock.py`, the new tests run
(`PYTHONPATH=. ./venv/bin/python -m unittest tools.tests.test_rehearsal_lock`), the failing tests
recorded, and the file restored from a copy (the diff after the sixth was empty; `git status` showed only
the intended files). Nothing was committed. The baseline is 42 OK. All six were caught.

**Mutation 1 acquire creates the lock with os.makedirs(..., exist_ok=True)**

```diff
--- a/tools/rehearsal_lock.py
+++ b/tools/rehearsal_lock.py
@@ -267,5 +267,5 @@
     d.parent.mkdir(parents=True, exist_ok=True)
     try:
-        os.mkdir(d)  # never exist_ok: this is the atomic step
+        os.makedirs(d, exist_ok=True)
     except FileExistsError:
         print(f"the rehearsal lock is held ({d}); its owner file:", file=sys.stderr)
```

Result: FAILED (failures=4). Failing tests:

- `test_held_by_sgk_returns_3_and_prints_the_owner` (TestAcquire)
- `test_held_by_si_returns_3_and_prints_the_owner` (TestAcquire)
- `test_held_with_no_owner_file_returns_3_and_says_so` (TestAcquire)
- `test_commands_and_exit_statuses` (TestCommandLine)

**Mutation 2 release drops the project check**

```diff
--- a/tools/rehearsal_lock.py
+++ b/tools/rehearsal_lock.py
@@ -158,6 +158,4 @@
 def _not_sgk_reason(owner):
     """``None`` when the owner is SGK's (``project`` and ``ray_temp_dir``), otherwise why not."""
-    if owner.get("project") != PROJECT:
-        return f"project is {owner.get('project')!r}, not {PROJECT!r}"
     if owner.get("ray_temp_dir") != IN_PROCESS:
         return f"ray_temp_dir is {owner.get('ray_temp_dir')!r}, not {IN_PROCESS!r}"
```

Result: FAILED (failures=2). Failing tests:

- `test_other_project_with_in_process_temp_dir_is_refused` (TestRelease)
- `test_other_project_with_in_process_temp_dir_is_refused` (TestSetPid)

**Mutation 3 release drops the live-pid check**

```diff
--- a/tools/rehearsal_lock.py
+++ b/tools/rehearsal_lock.py
@@ -361,5 +361,5 @@
     _, owner = held
     pid = _recorded_pid(owner)
-    if pid is not None and pid_alive(pid):
+    if False:
         print(
             f"release: the recorded pid {pid} is alive; not releasing", file=sys.stderr
```

Result: FAILED (failures=2). Failing tests:

- `test_commands_and_exit_statuses` (TestCommandLine)
- `test_live_pid_is_refused` (TestRelease)

**Mutation 4 write_owner swaps acquired and host**

```diff
--- a/tools/rehearsal_lock.py
+++ b/tools/rehearsal_lock.py
@@ -99,6 +99,6 @@
         f"project: {PROJECT}",
         f"purpose: {purpose}",
+        f"host: {host}",
         f"acquired: {acquired}",
-        f"host: {host}",
         f"ray_temp_dir: {IN_PROCESS}",
         f"ray_address: {IN_PROCESS}",
```

Result: FAILED (failures=1). Failing tests:

- `test_free_lock_is_created_with_the_seven_keys` (TestAcquire)

**Mutation 5 set-pid rewrites acquired as well as pid**

```diff
--- a/tools/rehearsal_lock.py
+++ b/tools/rehearsal_lock.py
@@ -346,4 +346,5 @@
         return 1
     lines[index[0]] = f"pid: {pid}"
+    lines[2] = "acquired: " + datetime.now().astimezone().isoformat(timespec="seconds")
     _replace_owner(d, "\n".join(lines) + "\n")
     print(f"recorded pid {pid} in {d / 'owner'}")
```

Result: FAILED (failures=1). Failing tests:

- `test_live_pid_changes_pid_and_only_pid` (TestSetPid)

**Mutation 6 ray_processes() leaves out descendants**

```diff
--- a/tools/rehearsal_lock.py
+++ b/tools/rehearsal_lock.py
@@ -239,5 +239,5 @@
         changed = False
         for p, (ppid, _) in table.items():
-            if p not in marked and ppid in marked:
+            if p not in marked and ppid in set():
                 marked.add(p)
                 changed = True
```

Result: FAILED (failures=5). Failing tests:

- `test_commands_and_exit_statuses` (TestCommandLine)
- `test_a_worker_child_of_a_raylet_is_listed_though_it_names_no_path` (TestRayProcesses)
- `test_finds_gcs_raylet_agent_and_the_workers_below_a_raylet` (TestRayProcesses)
- `test_reads_the_process_table_through_one_function` (TestRayProcesses)
- `test_skips_its_own_pid_and_its_ancestors` (TestRayProcesses)

## Observations not acted on

- **Agent command lines are cut at 150 columns** by `ray-processes`, as the prompt asks, so the
  `agent.py` names of the six `Python` lines are not visible in `02a-real.txt`; the listing
  is correct, the display is short. The probe ran with the dashboard off.
- **A shell whose command line names the venv's Ray path is listed** by `ray_processes()` (the note,
  correction 5), a false positive in the safe direction. None was seen.
- **No process outlived the signal.** The ten pids of the probe died 3–5 s after the `SIGTERM`, not
  the 1–2 min of `[07-a-killed-builds-ray-agents-outlive-the-driver]`. That was a graceful driver
  exit under `SIGTERM`; it says nothing about a `SIGKILL`ed driver.
- **The `ComputeTargets` flake is now failing alone as well**, at load 26–27 (above).
- `[00-main-and-the-readers-default-to-ray-address-auto]` is untouched: `main.py`, the readers and
  the rehearsal tools were not changed (prompt §5).

## State handed to the next prompt

- `tools/rehearsal_lock.py` is on the tree; README §4.1's lock steps (04 and 08, R1 and R2) are its
  commands. It was not edited (prompt §5).
- The real lock `~/ScienceData/.rehearsal-lock` is free and no Ray process of this repository is
  running.
- The tree has the new tests; `tools` is at 91. The other six roots are at the baseline counts.
- The registry shows 33 runs, none running. The open-issue index is unchanged at 106: nothing was
  opened or closed.
- 03–09 stay held on E1 and E2 as the board says.
