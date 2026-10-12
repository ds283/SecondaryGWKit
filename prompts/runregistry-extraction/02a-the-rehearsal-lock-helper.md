# Prompt 02a — a helper for the shared rehearsal lock

**Campaign:** [`README.md`](README.md) · **Board items:** **H1**–**H4** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-12, after R0, at the user's request, at the commit that adds it to the plan.
**Gate:**
- the commit that adds this prompt is `HEAD` or an ancestor of it, and `git status` is clean;
- `python -m RunRegistry list` shows nothing `running`;
- `ls ~/ScienceData/.rehearsal-lock` fails: the shared lock is free. If it is held, do §2.1–§2.4 and
  stop before §2.5 (§4).

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. One small standard-library module, its tests, a short section of
`CLAUDE.md`, and two proofs whose outputs are saved.

**Read first:**

1. `CLAUDE.md` in full, and especially "Long-running jobs — the run registry" and "Repository
   mechanics".
2. [`README.md`](README.md): §2.2 (02a's charter), §3, §4 (02a's line), §4.1 ("The rehearsal lock"
   and "Ray addresses"), and §5 (rules 4, 7, 8, 10 and 14).
3. The board's §3 entry `[00-main-and-the-readers-default-to-ray-address-auto]` and §5.3.
4. StochasticInstantons' (SI's) brief, which is the contract:
   `git -C /Users/ds283/Documents/Code/StochasticInstantons show d6d550b:.prompts/dk-and-rpk-migration/rehearsal-lock-brief.md`,
   §2 (the lock) and §5 (SGK).
5. SI's script, for its lock functions only (`write_owner`, `acquire_lock`, `release_lock`,
   `cmd_status`):
   `git -C /Users/ds283/Documents/Code/StochasticInstantons show 8e04a7c:.documents/dk-and-rpk-migration/rehearsal/ray_cluster.sh`.
   ChamPBH's is the same contract, at its `288b385`, same path.
6. `docs/runregistry-extraction/rehearsal/prelaunch_checks_02.txt`, for how R0 checked for Ray
   processes.

---

## 1. What is wanted

Three projects on this machine, SGK, ChamPBH and SI, run Ray builds. They share one lock,
`~/ScienceData/.rehearsal-lock`, so that only one rehearsal or long build runs at a time (README
§4.1). SI and ChamPBH take it in code, inside their `ray_cluster.sh`. SGK takes it by hand, by
following §4.1's steps with `mkdir` and a hand-written owner file. This prompt gives SGK the same
footing: **a helper that performs each of §4.1's steps, and refuses each step that would break the
contract.**

It also fixes the test for Ray processes before a launch. R0's pre-launch check ran a machine-wide
`pgrep -lf 'gcs_server|raylet|ray::'`. With SI's and ChamPBH's clusters isolated and allowed to
run, that check would find their processes and stop an SGK rehearsal for nothing. The test §3 and
§4.1 actually ask for is narrower: **no Ray process of this repository**. The helper provides it.

The helper never starts or stops Ray, never waits for the lock, never retries, and never removes a
lock that is not SGK's. It is not part of the run registry: `RunRegistry/` is frozen (rule 14), and
the registry records, it does not lock (`CLAUDE.md`).

## 2. What to do

**2.1 The module, `tools/rehearsal_lock.py`.** Standard library only. Run from the repository root
as `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock <command>`.

- **The lock directory** is `Path.home() / "ScienceData" / ".rehearsal-lock"`, and its owner file
  is `owner` inside it. Every function takes the lock directory as an argument, defaulting to that
  path, so that tests never touch the real one.
- **The owner file** is UTF-8, exactly seven `key: value` lines, in this order and no other:
  - `project: SecondaryGWKit`;
  - `purpose: <the --purpose text>`;
  - `acquired:` local time with its offset, to the second
    (`datetime.now().astimezone().isoformat(timespec="seconds")`, e.g. `2026-10-12T14:03:11+01:00`);
  - `host:` the short hostname, as `hostname -s` gives it;
  - `ray_temp_dir: in-process`;
  - `ray_address: in-process`;
  - `pid:` `-`, or a pid.

  It is always written to `owner.tmp` in the lock directory and renamed over `owner` with
  `os.replace`. The parser reads any project's file: one `key: value` per line, the value after the
  first `": "`.
- **SGK's Ray processes.** `ray_processes()` lists every process whose command line contains this
  checkout's `venv/` path, followed by Ray's package directory (`…/site-packages/ray/`), together
  with every descendant of those. It matches both the path as written and its `realpath`. Ray's
  `gcs_server`, `raylet` and agents name that path. Its workers show only `ray::<name>`, so they
  are reached as descendants. It reads `ps -axww -o pid= -o ppid= -o command=` through one
  function, which the tests replace with fixed text. It never lists its own process or its
  ancestors.

**The commands.** Each prints what it did and exits with the status given here:

| Command | Does | Exit |
|---|---|---|
| `acquire --purpose TEXT` | `mkdir -p` the parent; `os.mkdir` the lock (never `exist_ok`), the atomic step; write `owner` with `pid: -`. Then, if `ray_processes()` is not empty, list them, release, and refuse. | 0 held by SGK now; **3** already held, printing its `owner` (or "no owner file") and touching nothing; **4** SGK Ray processes alive, lock released |
| `set-pid PID` | The lock exists and is SGK's (`project: SecondaryGWKit`, `ray_temp_dir: in-process`), and `PID` is a live process (`os.kill(pid, 0)`). Rewrite `owner` with only `pid` changed. | 0; **1** refused (no lock, another's, no `owner`, pid not alive), touching nothing |
| `release` | The lock is SGK's, its `pid` is `-` or not alive, and `ray_processes()` is empty. Then remove `owner` and `owner.tmp` if present, and `rmdir`. | 0 released, or no lock to release (say so); **1** refused, printing why: another project's owner or no `owner` (print both sides), the pid alive, or the processes listed |
| `status` | The lock's state, its `owner` verbatim, whether a numeric `pid` is alive, and `ray_processes()`. | 0 |
| `ray-processes` | `ray_processes()`, one line each: pid, ppid and command, cut to 150 columns. | 0 none; 1 some |

Usage errors exit 2. There is no option to break a lock, wait, or retry. A stale lock is the
user's to remove by hand (brief §2). The module docstring says this, names the brief at SI's
`d6d550b` as the contract, and gives the five commands.

**2.2 The tests, `tools/tests/test_rehearsal_lock.py`.** `unittest`, no Ray, nothing under `var/`,
and never the real `~/ScienceData`. Each test uses a lock directory in a `tempfile` directory, and
`ray_processes()` reads fixed `ps` text. At least:
1. `acquire` on a free lock creates it, with the seven keys in order, `project: SecondaryGWKit`,
   both Ray fields `in-process`, `pid: -`, an `acquired` that parses with
   `datetime.fromisoformat` and carries an offset, and no `owner.tmp` left behind.
2. `acquire` on a held lock, owned by SGK, by SI's owner text, or with no owner file, returns 3,
   prints the owner (or says it is missing), and leaves the directory and `owner` byte-identical.
3. `acquire` with SGK Ray processes in the `ps` text returns 4 and leaves no lock.
4. `set-pid` with a live pid (the test's own) changes `pid` and only `pid`. It refuses a dead pid,
   SI's lock, a missing owner and a missing lock, and changes nothing when it refuses.
5. `release` removes SGK's lock when its pid is `-` or dead and there are no processes. It refuses,
   leaving everything byte-identical, for SI's or ChamPBH's owner text, a missing owner, a live
   pid, and SGK Ray processes. With no lock it returns 0.
6. `ray_processes()` on fixed text: it finds a `gcs_server`, a `raylet` and an agent under this
   checkout's venv, and a `ray::IDLE` worker that is a raylet's child. It skips SI's and ChamPBH's
   processes (another venv, `/tmp/ray-si`, `/tmp/ray-cpbh`), a `pgrep` or `grep` whose arguments
   name Ray, and its own pid.
7. The default lock directory is `$HOME/ScienceData/.rehearsal-lock`, with `HOME` patched to a
   temporary directory. Nothing is created by computing it.
8. The owner parser reads SI's and ChamPBH's owner files as their scripts write them. Copy one
   example of each into the test as text, from the scripts' `write_owner`.

**2.3 `CLAUDE.md`.** Add a section directly after "`RunRegistry/` is frozen — it is moving to
RunRegistryKit", titled **"The shared rehearsal lock"**. In a few lines, say:
- three projects on this machine run Ray builds, and one lock lets only one rehearsal or long build
  run at a time;
- a rehearsal or long build takes it with `python -m tools.rehearsal_lock acquire`, sets the pid at
  each launch, and releases it when its last Ray process is gone. A held lock means stop and ask
  the user. Never wait, retry, or remove it;
- every Ray driver names its address (`local` for SGK's), never `auto`, and `ray stop` is never run;
- the contract is SI's brief, and this campaign's README §4.1 is SGK's protocol.

Change nothing else in `CLAUDE.md`.

**2.4 The interoperability proof**, saved as `prompts/runregistry-extraction/measure/out/02a-interop.txt`.
Use a temporary `HOME` (`mktemp -d` under the session scratchpad), so that every script's lock is
`$HOME/ScienceData/.rehearsal-lock` in it, never the real one. Extract SI's script at `8e04a7c`
and ChamPBH's at `288b385` with `git -C <repo> show <sha>:<path>` into that directory, and run
those copies, not the working trees. Use only their lock functions and `status`: run each with
`bash <copy> status`, or with `bash -c 'source <copy>; acquire_lock "02a proof"; echo rc=$?'` and
the same for `release_lock`. **Never call their `start` or `stop`.**
For each of SI and ChamPBH, record every command and its exit status:
1. SGK holds the lock (`acquire`, then `set-pid` to a `sleep 300` started for the purpose):
   - the other script's `status` prints SGK's owner and says the pid is alive;
   - its `acquire_lock` returns 3, and its `release_lock` refuses;
   - the lock is byte-identical afterwards;
   - kill the `sleep`, and SGK's `release` removes the lock.
2. The other project holds it (its `acquire_lock`):
   - SGK's `acquire` exits 3 and prints the other project's owner;
   - SGK's `set-pid` and `release` exit 1;
   - SGK's `status` shows it;
   - the lock is byte-identical afterwards;
   - the other project's `release_lock` then removes it.

**2.5 The real proof**, saved as `prompts/runregistry-extraction/measure/out/02a-real.txt`. This is
the one step that touches the real lock. It holds the lock for a few minutes at most. Record each
command, its output and exit status:
1. `status`: free.
2. `acquire --purpose "runregistry-extraction 02a (lock helper proof)"`: 0.
3. Start a probe in the background:
   `./venv/bin/python -c 'import os, time, ray; ray.init(address="local", num_cpus=1, include_dashboard=False); print(os.getpid(), flush=True); time.sleep(600)'`.
   It starts a private in-process instance and nothing else. Read its pid from its output.
4. `set-pid <probe pid>`: 0.
5. `ray-processes`: 1. Record which of `gcs_server`, `raylet` and the agents it lists.
6. `release`: 1, refused while the probe is alive. The lock is still there.
7. `kill -TERM` the probe, then wait **by pid** until `ray-processes` exits 0. A killed driver's
   agents can outlive it by a minute or two (`[07-a-killed-builds-ray-agents-outlive-the-driver]`).
   Wait by pid; never `pgrep -f` in a loop.
8. `release`: 0. `status`: free.

If `status` shows the lock held at step 1, or `acquire` exits 3, stop (§4).

**2.6 The roots.** Run the seven, once each, by README §8's command. `tools` gains this prompt's
tests; the other six are at 02's counts. Run `./venv/bin/python -m black --check` on the two new
files.

## 3. Acceptance

1. **H1.** `tools/rehearsal_lock.py` has the five commands, their exit statuses, and the owner
   format of §2.1, imports only the standard library, and has no option to break, wait for, or
   retry a lock.
2. **H2.** `tools/tests/test_rehearsal_lock.py` covers §2.2's eight points. The `tools` root passes
   at 49 + the new count; the other six roots are at 02's counts, apart from README §8's known
   wall-clock tests, each re-run alone.
3. **H3.** `02a-interop.txt` shows both directions for both SI and ChamPBH, every exit status as
   §2.4 says, the lock byte-identical wherever a refusal left it, and no temporary lock left at the
   end.
4. **H4.** `02a-real.txt` shows §2.5's eight steps with their exit statuses. At the end the real
   lock is free and `ray-processes` exits 0.
5. **`CLAUDE.md`** has §2.3's section, and its diff is one inserted hunk.
6. **Deliberate breakage** (rule 8). Apply each of these alone, run the new tests, record the
   failing test names, and revert. Record each mutation as a diff, exactly as applied:
   - `acquire` creates the lock with `os.makedirs(..., exist_ok=True)`;
   - `release` drops the `project` check;
   - `release` drops the live-pid check;
   - `write_owner` swaps `acquired` and `host`;
   - `set-pid` rewrites `acquired` as well as `pid`;
   - `ray_processes()` leaves out descendants.

   Each must be caught. A mutation that survives means a missing test: add the test.
7. `git diff --stat HEAD -- RunRegistry/ main.py extract_*.py docs/` is empty: no frozen file, no
   driver and no rehearsal tool changes.

## 4. Stop conditions — stop and ask

- The real lock is held at the gate or at §2.5's first step, or `acquire` exits 3 there. Report the
  owner file. Do §2.1–§2.4, then stop without committing: H4 needs the lock.
- `list` shows a run `running` or stale.
- The probe's `ray-processes` lists nothing. The command-line match is then wrong for this
  machine's Ray: record the probe's `ps` lines and stop.
- After the probe is killed, a process stays in `ray-processes` for more than five minutes. Record
  it, do not kill it, and leave the lock held.
- SI's or ChamPBH's script behaves otherwise than §2.4 says. That is a finding about the shared
  contract: record it and stop. Do not change their scripts, or this helper, to make the two agree.
- A root fails other than by README §8's known tests.

## 5. What this prompt does not do

- It changes no rehearsal tool, `quadsource_atol_sweep.py`, `run_readers.sh`, `main.py` or any
  `extract_*.py`. The `auto` default is `[00-main-and-the-readers-default-to-ray-address-auto]`,
  assigned to `handover`.
- It changes nothing under `RunRegistry/` or `var/` (the probe registers no run), or in any other
  repository. SI's and ChamPBH's scripts are run from extracted copies, and only their
  lock functions and `status`.
- It does not edit README §4.1. That section already names this helper's commands, from 02a on.
- It runs no build.

## 6. The log and the board

**The files you may change, and nothing else:** `tools/rehearsal_lock.py` (new);
`tools/tests/test_rehearsal_lock.py` (new); `CLAUDE.md`; the two files under `measure/out/`; the
log; this campaign's board; and `docs/OPEN_ISSUES.md` only if §3 gains an entry.

**The log**, at `logs/02a-the-rehearsal-lock-helper.md`, in README §5.1's form.

**The board:** the §1 row for 02a, items H1–H4, and `### 5.4 After prompt 02a (<date>)`.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
