# Orchestrator — prompt 02a, the rehearsal-lock helper

Read [`../README.md`](../README.md) first: §2.2 (02a's charter), §3, §4 (02a's line), §4.1 ("The
rehearsal lock" and "Ray addresses"), §5 (rules 4, 7, 8, 10 and 14) and §8. Then this board's §3
entry `[00-main-and-the-readers-default-to-ray-address-auto]`, §5.3, and the *Orchestrator review of
prompt 02* in §5.2. Then SI's brief at `d6d550b`, §2 and §5, which is the contract the helper
implements:
`git -C /Users/ds283/Documents/Code/StochasticInstantons show d6d550b:.prompts/dk-and-rpk-migration/rehearsal-lock-brief.md`.

**You do not write code.** You may:
- run the suites, `python -m RunRegistry list`, and SI's and ChamPBH's lock functions and `status`
  from extracted copies under a temporary `HOME` in the scratchpad (never their `start` or `stop`,
  and never their `release_lock`: correction 1);
- run the helper the agent ships, against a temporary `HOME` only, never the real lock;
- apply the six mutations to a copy of `tools/` in the scratchpad, never in the repository;
- fix small residue in a follow-up commit of your own (§4).

**While the agent holds the real lock (§2.5) you run nothing**: no suite, no Ray, no `ps` loop.

**The prompt:** [`02a-the-rehearsal-lock-helper.md`](../02a-the-rehearsal-lock-helper.md)
**Board items:** H1–H4 · **Closes:** nothing · **Narrows:** nothing · **Opens:** anything found ·
**Model:** Sonnet, as the prompt recommends.

**Gate:**
- 02 landed (`74eff08`) and was reviewed (`f11e3e7`). The lock was adopted (`e560d55`), its issue
  assigned (`0fbbc4e`), and 02a planned (`c89fc9e`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`, `docs/` and `CLAUDE.md`.
  - If a later commit touches `tools/` or `CLAUDE.md`, re-check §0's facts and hand the agent the
    corrections as an addendum.
- `list` shows nothing `running`, and `~/ScienceData/.rehearsal-lock` does not exist.
- **Never dispatched concurrently with anything else in this checkout.** 02a runs a Ray probe and
  holds the shared lock for a few minutes; no SGK suite or Ray process may be running then.

## 0. What makes this prompt unusual

**It is the first prompt of this campaign that adds code, and the code guards something outside
this repository.** The lock at `~/ScienceData/.rehearsal-lock` is shared with SI and ChamPBH. A
helper that wrote a malformed owner file, or removed a lock that was not SGK's, would break their
rehearsals, not SGK's. So the review weighs three things above the rest:
- **it never removes or rewrites a lock that is not SGK's**, and never one whose pid or Ray
  processes are alive;
- **the tests never touch the real lock**, even when a test forgets to pass a lock directory;
- **each refusal is caught by a test whose only reason to refuse is that check**, so that a
  mutation cannot hide behind another one (correction 2).

**It touches the real lock once, briefly** (§2.5). While it holds it, a session of SI or ChamPBH that
tries to start a rehearsal gets exit 3 and stops to ask the user. That is the contract working, and
the window should be minutes.

**The check for this repository's Ray processes matches a path anywhere in the command line, not
`argv[0]`.** Ray's dashboard agent shows `python -u …/site-packages/ray/dashboard/agent.py`, with a
bare `python` as `argv[0]` (ChamPBH's lines of 2026-10-02, recorded in
`docs/a3-v2-readiness/sigterm_handler_probe.txt`). After a killed driver its agents are
re-parented to pid 1 (`[07-a-killed-builds-ray-agents-outlive-the-driver]`; `a3-v2-readiness`
board, pids 49729 and 49730, `ray/dashboard/agent.py` and `ray/_private/runtime_env/agent/main.py`,
both from this checkout's venv). So a rule keyed on `argv[0]`, or on descent from the driver alone, would miss
exactly the orphans the release check exists for. The prompt's rule (the venv path followed by
`site-packages/ray/`, anywhere in the line, plus descendants) is right; the agent must not narrow
it.

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `c89fc9e`.
Each correction below governs, and the agent logs it as an `IMPLEMENTATION CHOICE` citing this note,
or `STRUCTURALLY REQUIRED` where the prompt cannot be followed as written.

1. **SI's and ChamPBH's `release_lock` are not called** (`STRUCTURALLY REQUIRED`; the user's ruling,
   2026-10-12). Claude Code's built-in removal check denies a `bash -c` that sources a script
   containing `rm`, and only a person can approve it. The orchestrator's dry run was denied on
   exactly that form. Do not work around it with a wrapper file, a hand `rm`, or any other route.
   So §2.4 changes in two places, and nowhere else:
   - **Direction 1** (SGK holds): "its `release_lock` refuses" is shown by reading, not running.
     Quote the copy's `release_lock` (SI's lines 130–147, ChamPBH's 131–148) in `02a-interop.txt`,
     and say
     that the project and `ray_temp_dir` check returns 1 before any `rm`.
   - **Direction 2** (the other holds): the other project's lock is **left in place** in the
     temporary `HOME`. `02a-interop.txt` records its path and its `owner`, byte-identical to when
     it was written. Acceptance H3's "no temporary lock left at the end" becomes "no temporary lock
     of SGK's left, and each other project's lock where its `acquire_lock` put it".
   - Use one temporary `HOME` per project, direction 1 first, then direction 2. Direction 1 ends with
     SGK's `release` removing SGK's lock, so direction 2 starts free.
   - Their `acquire_lock` and `status` are not affected: they ran under the check here.
2. **One reason to refuse per refusal test.** §2.2 points 4 and 5 list several refusals. Each test
   must leave exactly one reason standing, or mutations 2 and 3 survive behind another check:
   - "another project's owner" uses an owner whose `pid` is `-` or dead, with an empty `ps` text;
   - "a live pid" uses SGK's own owner with the test's pid and an empty `ps` text;
   - "SGK Ray processes" uses SGK's owner with `pid: -` and a `ps` text that lists them.
3. **The `set-pid` mutation needs an `acquired` that a rewrite would change.** A test that runs
   `acquire` and `set-pid` within the same second cannot see `acquired` rewritten. Write the
   starting owner with a fixed past `acquired` (for example `2026-01-01T00:00:00+00:00`), not with
   `acquire`, and compare every other line byte for byte. No `sleep`.
4. **The default lock directory is computed at call time, and every test isolates `HOME`.** A default
   computed at import (`def acquire(lock_dir=DEFAULT_LOCK_DIR)`) fails §2.2 point 7, and a test
   that forgot its argument would reach the real lock. So:
   - functions take `lock_dir=None` and resolve it when called;
   - the test module's `setUp` patches `HOME` to a temporary directory for **every** test, as well as
     passing an explicit lock directory, so that a forgotten argument cannot reach
     `~/ScienceData`.
5. **§2.2 point 6's `pgrep`/`grep` case** is R0's own line, `pgrep -lf gcs_server|raylet|ray::`, and
   a `grep ray` line, neither naming this checkout's venv. Do not add a rule that excludes
   processes by `argv[0]` (§0). A shell whose text names the venv's Ray path would be listed. That
   is a false positive in the safe direction, a refusal, and goes in "Observations" if it is seen.
6. **The two proofs' mechanics.**
   - **The live pid for direction 1** is a background `sleep 300 &` and its `$!`. If the tool
     refuses the `sleep`, use `./venv/bin/python -c 'import time; time.sleep(300)' &` instead, and
     log it.
   - **The probe (§2.5 step 3)** is started with `nohup … > <scratch>/probe.out 2>&1 &`. Its pid
     is taken from `$!` and from its printed line; the two must agree.
   - **The wait (§2.5 step 7).** Right after `kill -TERM`, record the pids `ray-processes` lists.
     Then run one bounded loop, `until` none of those pids answers `kill -0` or 300 s pass, polling
     every 2–5 s. Then run `ray-processes` once, which must exit 0. Record how long each pid lived
     after the signal, beside `[07-a-…]`'s 1–2 min.
   - **The window.** Prepare every command first, then run §2.5's eight steps in one stretch. Record
     the times of `acquire` and `release`. Run no suite while the real lock is held (brief §2).
     §2.4 comes before §2.5, and §2.6 after it.
7. **`hostname -s`** is `mps026088` here. Python's `socket.gethostname()` gives the full name,
   `mps026088.phys.susx.ac.uk`. Use `hostname -s` itself, or the gethostname value cut at the
   first `.`, and say which. The two agree on this machine.

**What moves on purpose:**
- Added: `tools/rehearsal_lock.py`, `tools/tests/test_rehearsal_lock.py`,
  `measure/out/02a-interop.txt`, `measure/out/02a-real.txt`, and
  `logs/02a-the-rehearsal-lock-helper.md`.
- Modified: `CLAUDE.md` (one inserted hunk), this board, and `docs/OPEN_ISSUES.md` only if an issue
  is opened.
- **README does not move** (prompt §5), and neither do `prompts/INDEX.md` or `orchestrator/` (the
  orchestrator's, §4).
- No file under `docs/`, `RunRegistry/`, `main.py`, any `extract_*.py` or `var/` changes. The probe
  registers no run.
- The index is 106, and moves only if an issue is opened.

**The facts, checked by the orchestrator on 2026-10-12 at `c89fc9e`.** Pass them on.

- **The tree and the machine.** Branch `handover-remedial`, `git status --short` empty, this
  checkout alone in `git worktree list`. Since 02's commit, `74eff08`, the four commits touch only
  `prompts/` and `docs/OPEN_ISSUES.md`. `RAY_ADDRESS` unset. **No Ray process of any project is
  running**: nothing in `ps -axww` names `gcs_server`, `raylet`, `ray::` or `site-packages/ray/`.
  10 CPUs. Load 19–32.
- **The lock is free.** `~/ScienceData` exists and holds one entry, `ChamPBH-rehearsals`, and no
  `.rehearsal-lock`.
- **The registry.** 33 runs: 31 finished, 2 `unknown` (the `a3-pilot` pair). None `running` or
  stale.
- **Python and Ray.** The venv is Python 3.12.15 and Ray 2.43.0 at
  `/Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.12/site-packages/ray/`. `venv` is a
  real directory, and the checkout's `realpath` equals its path, so the "as written" and `realpath`
  spellings coincide here. The test for both must use a fixed text where they differ.
- **`tools` is a package** (`tools/__init__.py`). With `PYTHONPATH=.`, `import tools` resolves to
  the repository's, so `python -m tools.rehearsal_lock` will too.
- **SI's and ChamPBH's scripts.** `git show` extracts them cleanly:
  - SI `8e04a7c:.documents/dk-and-rpk-migration/rehearsal/ray_cluster.sh`, 398 lines, SHA-256
    `b934a75c7edb612b…`;
  - ChamPBH `288b385:` the same path, 400 lines, `ad4b90210a884e71…`.

  Their lock functions are the same code; `diff` shows only the project name, temp dir, ports,
  default CPUs and comments. `bash` on `PATH` is MacPorts' 5.3.20 (`/opt/local/bin/bash`).
- **Their behaviour, dry-run under a temporary `HOME`** in the orchestrator's scratchpad (each
  through `bash -c 'source <copy>; …'`, and `bash <copy> status`):
  - `acquire_lock` on a free lock: rc 0, and an `owner` of seven lines with `pid: -`, for example
    SI's:
    ```text
    project: StochasticInstantons
    purpose: orch 02a dry run
    acquired: 2026-10-12T14:19:31+01:00
    host: mps026088
    ray_temp_dir: /tmp/ray-si
    ray_address: 127.0.0.1:6390
    pid: -
    ```
    ChamPBH's has `project: ChamPBH`, `ray_temp_dir: /tmp/ray-cpbh` and
    `ray_address: 127.0.0.1:6380`. These are §2.2 point 8's two examples.
  - `acquire_lock` on a held lock: rc 3, the owner printed to **stderr** with a four-space indent,
    and "stop and ask the user".
  - `status`: rc 0. It prints `lock: HELD (<dir>)`, the owner indented, then `owner pid: -` with "(no
    cluster pid recorded yet)" or `owner pid N: alive` / `NOT alive`, the project's cluster
    processes and its four ports. It reads `ps` and `lsof` and writes nothing.
  - `release_lock`: denied by the removal check (correction 1). By its code it returns 0 with "no
    lock to release" when there is no lock, and 1 with both sides printed when `project` or
    `ray_temp_dir` is not its own, before any `rm`.
- **What a Ray process of this checkout looks like** (§0): `gcs_server` and `raylet` are binaries
  under `…/site-packages/ray/core/src/ray/`. The agents are `python -u …/site-packages/ray/dashboard/agent.py`
  and `…/ray/_private/runtime_env/agent/main.py`. Workers carry only their title, `ray::…`. Taken
  from earlier records, not from a probe run here: the orchestrator ran no Ray, so as not to hold
  the lock twice. §2.5 is the first observation, and the prompt's stop covers a miss.
- **Shell traps.**
  - The shell is `zsh`. An unquoted `$var` holding spaces is not split (use an array), and a bare
    word beginning `=` is command expansion. Quote it.
  - `HOME=<tmp> cmd` sets `HOME` for that one command, in `zsh` and in `bash`. `Path.home()` reads
    `HOME`.
  - Foreground `sleep N` is refused by the tool. A bounded `until …; do sleep 2; done` loop is not.
  - `grep` may wrap `ugrep`. Use Python or `git grep` for any regex the log relies on.
  - `stat` is GNU coreutils. Use `/usr/bin/stat -f '%m %N'` or Python's `os.stat`.
- **Expected counts**, at `c89fc9e`, serially, load 19–38:

  | Root | Tests | Result |
  |---|---|---|
  | AdaptiveLevin | 32 | OK |
  | CosmologyModels | 39 | OK |
  | ObjectFactories | 271 | OK |
  | tools | 49 | OK |
  | LiouvilleGreen | 148 | OK (1 skipped) |
  | RunRegistry | 286 | OK |
  | ComputeTargets | 693 | 1 failure: `test_wall_time_per_object` (0.095 s against 0.06 s); alone, passed twice, load 31 |

  Every count is 02's. The wall-clock test is README §8's first known flake. After 02a, `tools` is 49 plus the new module's count, and the rest are
  unchanged.
- **The index:** 106.

**What the review exists to establish.**
- **(K1) The helper keeps the contract.** The owner format is the brief's, byte for byte in key
  order. `acquire` is atomic and never overwrites. `set-pid` changes `pid` alone, and only on SGK's
  lock. `release` removes only SGK's lock, and only with its pid dead and no SGK Ray process. There
  is no break, wait or retry.
- **(K2) The Ray check finds this checkout's processes and nobody else's**, orphans included,
  and never itself or its ancestors.
- **(K3) The tests cannot reach the real lock**, and each of the six mutations is caught by a test
  with one reason to fail.
- **(K4) The other projects and SGK respect each other's lock**, in both directions, with every
  refused lock byte-identical (H3, as correction 1 amends it).
- **(K5) The real lock was held briefly and left free**, with the probe's processes gone, waited
  for by pid (H4).
- **(K6) Nothing else moved.** `CLAUDE.md` gains one hunk, and nothing under `docs/`, `RunRegistry/`
  or `var/` changes. The roots are at 02's counts plus the new tests.

**Conventions.** 01's, carried. All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §8's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each in the foreground, serially, within the turn, after §2.5. Under load, the
  four tests of README §8 are known to fail: re-run a failure alone before calling it real.
- **Scratch.** The agent's own subdirectory of the session scratchpad, which holds the temporary
  `HOME`s, the extracted scripts and the probe's output. Never `/tmp` directly, never `var/`.
- **The venv.** The agent installs and uninstalls nothing.
- **`RunRegistry/` is frozen** (README §5 rule 14). The helper does not import it.
- **Other repositories.** SI and ChamPBH are read with `git -C … show` only. Nothing in their trees
  is run, changed or checked out.
- **Never touch a process you did not start.** The helper signals nothing. The agent kills only its
  `sleep` and its probe.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`. If it stops while it holds the real lock, it releases it with the
  helper once its probe's processes are gone. If it cannot, its report says so first, so the user
  can remove the lock by hand.

## 1. Before you dispatch

Done on 2026-10-12 at `c89fc9e`, before this note was committed. Repeat steps 1–4 at the dispatch
`HEAD` if anything lands in between.

1. **The tree.** Record the branch and `HEAD`. `git status --short` empty, this checkout alone in
   `git worktree list`, `git diff --stat 74eff08 HEAD -- . ':!prompts' ':!docs'` empty.
2. **The registry.** Nothing `running` or stale.
3. **Ray, the lock and the machine.** No Ray process of any project. `ls ~/ScienceData/.rehearsal-lock`
   fails. `RAY_ADDRESS` unset. Record the load.
4. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 106.
5. **Baselines**, as §0's table.
6. **The markers**, for check 9: immediately before dispatch, `touch` a marker in the
   orchestrator's scratch directory and save `ls -la ~/ScienceData`.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it:
- the prompt, and through it the campaign README, this board, `CLAUDE.md`, SI's brief and the two
  scripts' commits;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after both proofs and the suites. Its log is
  `logs/02a-the-rehearsal-lock-helper.md`, in README §5.1's form.
- **The files it may change** are the prompt's §6 list, and nothing else.
- **The order**: §2.1–§2.3, then the mutations, then §2.4, then §2.5 in one stretch, then §2.6.
- **The board's header** says 3 of 10 written and 3 landed (01, 02, 02a); `prompts/INDEX.md` is the
  orchestrator's.
- **Stop and ask** on any of the prompt's §4 conditions. A stop with the real lock held is reported
  first.

## 3. The review — eleven checks

1. **Scope.**
   - `git diff --name-status HEAD~1 HEAD`: `A` for the two `tools/` files, the two `measure/out/`
     files and the log; `M` for `CLAUDE.md` and this board, and `docs/OPEN_ISSUES.md` only with an
     issue opened.
   - README, `prompts/INDEX.md`, `orchestrator/`, `docs/`, `RunRegistry/`, `main.py` and every
     `extract_*.py` are not modified (acceptance 7).
2. **(K1) The module, by reading.** Read `tools/rehearsal_lock.py` in full.
   - Its imports, listed with `ast`, are all in `sys.stdlib_module_names`.
   - The five commands and their exit statuses are §2.1's table. Usage errors exit 2. `argparse`
     offers nothing to break, wait for, or retry a lock.
   - `os.mkdir` without `exist_ok`, after `mkdir -p` of the parent. `owner.tmp` then `os.replace`.
     Seven keys in the brief's order. The parser splits on the first `": "`.
   - The default lock directory is resolved at call time (correction 4).
   - `set-pid` and `release` check both `project` and `ray_temp_dir`. `release` removes `owner`,
     `owner.tmp` and the directory, and nothing else: no `shutil.rmtree`.
   - `ray_processes()` reads `ps` through one function. It matches the venv's Ray path anywhere in
     the line, in both spellings, adds descendants, and excludes itself and its ancestors. It does
     not key on `argv[0]` (§0).
   - The docstring names the brief at `d6d550b` as the contract, gives the five commands, and says a
     stale lock is the user's to remove.
3. **(K1) The commands, by running them** in a temporary `HOME` in the scratchpad. Never the real
   lock. On a free lock: `status` 0, `acquire` 0, `acquire` 3 (owner printed, byte-identical),
   `set-pid <a live pid of mine>` 0 (only `pid` changed), `release` 1 (pid alive), then kill that
   process: `release` 0, `release` 0 again ("no lock"). `set-pid 1x` exits 2. `ray-processes` 0.
4. **(K3) The tests, by reading.**
   - §2.2's eight points are each present.
   - Every refusal test has one reason standing (correction 2), and the `set-pid` test starts from a
     fixed past `acquired` (correction 3).
   - `setUp` patches `HOME` for every test (correction 4).
   - No `ray` import, nothing under `var/`, no `subprocess` call to `ps` that is not replaced.
5. **(K3) The mutations.** On a copy of `tools/` in the scratchpad, apply each of the log's six
   diffs alone and run the new module. Each fails, with the test names the log records.
   `git status` in the repository stays clean throughout.
6. **(K4) The interop proof.** Read `02a-interop.txt` in full. For SI and for ChamPBH:
   - direction 1: SGK's `acquire` 0 and `set-pid` 0; the other's `status` shows SGK's owner and
     "alive", and its `acquire_lock` 3; `release_lock` quoted, not run (correction 1); `owner`
     byte-identical (hashes before and after); SGK's `release` 0 after the kill;
   - direction 2: the other's `acquire_lock` 0; SGK's `acquire` 3 printing the other's owner,
     `set-pid` 1, `release` 1 printing both sides, `status` 0 showing it; `owner` byte-identical;
     the lock left, and its path recorded;
   - every temporary `HOME` under the agent's scratchpad, the extracted copies' SHA-256 as §0's, and
     no call of `start`, `stop` or `release_lock`.

   Then re-run direction 2 for SI yourself, in a fresh temporary `HOME`: the same exit statuses.
7. **(K2, K5) The real proof.** Read `02a-real.txt` in full.
   - The eight steps with their exit statuses: `status` free; `acquire` 0; the probe's pid, the same
     from `$!` and from its output; `set-pid` 0; `ray-processes` 1, listing a `gcs_server`, a
     `raylet` and at least one agent, and not the agent's own shell; `release` 1; the wait, each
     pid's lifetime after the signal; `release` 0; `status` free.
   - The acquire-to-release window and its times.
   - Now: `ls ~/ScienceData/.rehearsal-lock` fails, and `ray-processes` exits 0.
8. **(K6) `CLAUDE.md`.** `git diff HEAD~1 HEAD -- CLAUDE.md` is one inserted hunk, directly after the
   freeze section, titled "The shared rehearsal lock", making §2.3's four points in a few lines. It
   says nothing the helper does not do. It does not restate README §4.1.
9. **(K3, K6) Nothing outside moved.** `find var -newer <marker>` is empty. `ls -la ~/ScienceData`
   equals the saved listing.
10. **The records.**
    - The board: the header; the §1 row for 02a (✅, "this commit", the log); H1–H4 ✅;
      `### 5.4 After prompt 02a (<date>)`. §3 entries and the index for anything opened, the count
      right by counting.
    - The log has every §5.1 heading. It classifies this note's seven corrections and records the six
      mutations as diffs. Its counts agree with the outputs.
    - The commit message is in `CLAUDE.md`'s form, ending `Co-Authored-By: Claude Sonnet …`.
11. **The suites and `black`.** All seven roots, serially, at the commit: `tools` at 49 plus the new
    count, the rest as §0. Run the `tools` root while `ls -la ~/ScienceData` is saved before and
    after: equal. `black --check` is clean on the two new files.

## 4. After the review

- **Record it on this board** as a §5.4 *Orchestrator review of prompt 02a* paragraph, in 02's form.
  Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board and in the log's header;
  - this note added to the board's list of notes, marked "used for 02a";
  - `prompts/INDEX.md`'s row: 3 of 10 written, 3 landed, with 02a's helper in place.
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - K1–K6, each in a line;
  - the six mutations and the tests that caught them;
  - both proofs' outcomes, the real lock's window, and how long the probe's agents outlived it;
  - the other projects' temporary locks left in the scratchpad (correction 1);
  - anything opened.
- **Then 03 waits on E1.** 04 is the first rehearsal to take the lock with the helper.
