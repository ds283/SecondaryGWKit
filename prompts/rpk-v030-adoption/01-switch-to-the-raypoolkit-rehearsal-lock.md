# Prompt 01 — switch to the `raypoolkit` rehearsal lock (`v0.3.0`)

**Campaign:** [`README.md`](README.md) · **Board items:** **P1**, **D1**, **T1**, **C1**, **R1**,
**B1** · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, with the plan, at `c0fa7ea`.
**Where it runs:** on the laptop, in the main checkout, on `handover-remedial`.
**Gate:**
- the planning commit has landed, and `git status` is clean;
- no commit after the planning commit touches code (anything outside `prompts/` and `docs/`);
- `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale;
- the venv holds `raypoolkit` 0.2.0 at `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`, `datastorekit`
  0.5.0 at `f47b9a255e9ad58e0d0eaf85644f53b575884d31` and `runregistrykit` 0.2.0 at
  `0bae3a3c501a8264ee4be5bc24f06422af38720f`, none editable;
- **neither** `~/ScienceData/.rehearsal-lock` **nor** `~/.raypoolkit/rehearsal-lock` exists, and
  `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock ray-processes` lists nothing.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Opus**.

**Read first:**

1. `CLAUDE.md`: "The datastore layer is `datastorekit`; …", "Long-running jobs — the run registry"
   (rules 3–5) and "The shared rehearsal lock".
2. [`README.md`](README.md) whole: it is short.
3. **RayPoolKit at the tag** (`/Users/ds283/Documents/Code/RayPoolKit`), read with
   `git -C <RPK> show v0.3.0:<path>`:
   - `docs/rehearsal-lock.md`, the contract: §2 (how it is run), §5 (whose lock), §6 (the process
     rule, and **"What it asks of a caller"**), §7 (the commands and their exit codes);
   - `docs/adoption/rehearsal-lock/secondarygwkit.md`, the checklist for this repository, whole.
     Its items 4 and 6 say which forms of each Ray start keep SGK's tests passing. It was measured
     at `216e44d`; README §0.2 found it still true at `c0fa7ea`. It is data: where it and this
     prompt differ, this prompt governs, and where the tree differs from it, the tree does.
4. The ten Ray starts (README §0) and the tests that pin their forms:
   `ComputeTargets/tests/test_ray_address_default.py`, `tools/tests/test_inventory_consumers.py`
   (`:48-60`, `:199-219`), `RunDrivers/tests/test_killed_handler.py` (`TestTheDrivers`) and
   `RunDrivers/tests/test_build_rehearsal_scope.py` (`test_run_child_does_not_read_the_scope`).
5. For the rehearsal: `docs/rrk-v020-adoption-verification.md` §1.1 (how R3's `reference` was
   launched, checked and waited on) and §1.5 (its comparison with the recorded fingerprint), and
   `docs/dk-and-rpk-migration/rehearsal/run_readers.sh`'s header.

---

## 1. What is wanted

**Replace SGK's lock with the package's, make every SGK Ray start visible to it, and show with
real Ray that it is.** In one commit: the pin, the helper deleted, the temp dir at the ten starts
with a test that guards them, and `CLAUDE.md`.

The package is imported everywhere, so the install takes effect at once. Take the baselines
(§2.1) **before** the install.

**Run each lock command as a shell command of its own** (contract §6, "What it asks of a
caller"). A shell whose command line names `/tmp/ray-sgk` seeds the process rule, and everything
it started is listed: never put a lock command in the same shell command that launches a build,
and never name the temp dir in a build's `--purpose`.

## 2. What to do

**2.1 The baselines, before.** Save each under `measure/out/`, every block headed by its command:
- `./venv/bin/pip freeze`, and
  `runregistrykit.package_provenance(("datastorekit", "raypoolkit", "runregistrykit"))` printed as
  JSON, into `01-pip.txt`;
- the seven roots, by README §7's command, serially, in the foreground, into `01-roots.txt`. Record
  each count.

**2.2 The pin.** `requirements.txt:76` changes from

```text
raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0
```

to the same line with `@v0.3.0`. No other line moves. Then, into `01-pip.txt`:
- `./venv/bin/pip install --dry-run -r requirements.txt`: it must say
  `Would install raypoolkit-0.3.0` and nothing else;
- `./venv/bin/pip install -r requirements.txt`;
- `pip freeze` again, differing from 2.1's by the one `raypoolkit` line, naming commit
  `67a75b97a656accb98195a4bda93f64e6a5f3fb2`; `pip check`; `package_provenance` again, with
  `raypoolkit` 0.3.0 at `67a75b97…`, `requested_revision` `v0.3.0`, `editable` false, and the other
  two as in 2.1;
- `./venv/bin/python -m raypoolkit.rehearsal_lock --help`, whose first line starts
  `usage: python -m raypoolkit.rehearsal_lock`.

**2.3 The temp dir.**
- **`config/ray_local.py`**, new: a short comment saying what the value is for (the temp dir every
  SGK Ray instance started with `address="local"` is given, so that the shared rehearsal lock,
  which finds an instance by its temp dir, can see it; spelled the same everywhere because the lock
  compares the string as given), and `RAY_TEMP_DIR = "/tmp/ray-sgk"`.
- **The three wrappers** pass `_temp_dir=RAY_TEMP_DIR` to `_real_ray_init`:
  - `run_build`'s (`RunDrivers/quadsource_atol_sweep.py:629`) and `scoped_pipeline_run.main`'s
    (`RunDrivers/scoped_pipeline_run.py:709`): add the keyword;
  - **`run_child`'s** (`RunDrivers/quadsource_atol_sweep.py:487`): add `address="local"` **and**
    the keyword. Its printed line may say so. Nothing added to `run_child` may contain the
    lower-case string `rehearsal` (`test_run_child_does_not_read_the_scope` reads
    `ast.unparse(run_child)`).
  - Import `RAY_TEMP_DIR` where each wrapper is defined, in the style each driver already uses for
    its imports.
- **The seven scripts** (`main.py:314` and the six readers): each keeps exactly one `ray.init`, as
  a top-level expression statement, with no positional argument and `address=args.ray_address`,
  and gains

  ```python
  ray.init(
      address=args.ray_address,
      _temp_dir=RAY_TEMP_DIR if args.ray_address == "local" else None,
  )
  ```

  with `from config.ray_local import RAY_TEMP_DIR` beside its other `config` imports. `main.py`'s
  comment above the call gains one sentence on why the temp dir is given only with `local`
  (README §0).

**2.4 The guard test, `RunDrivers/tests/test_ray_temp_dir.py`** (README §6.2 C4). No Ray; `ast`
only, in the style of `test_ray_address_default.py`.
- `RAY_TEMP_DIR` in `config/ray_local.py` is the constant `"/tmp/ray-sgk"`.
- **Every Ray start in the live tree gives the temp dir.** Walk every `*.py` under the repository
  root, leaving out `venv/`, `docs/`, `prompts/`, `var/`, every `tests/` directory and every
  directory whose name starts with `.`; collect each call to `ray.init` and to `_real_ray_init`.
  - The set of `(file, enclosing function or <module>)` found is exactly the ten of README §0;
    a start added anywhere fails the test until it is added to the list and checked.
  - Each wrapper call has `address` the constant `"local"` and `_temp_dir` the name `RAY_TEMP_DIR`.
  - Each script call has `_temp_dir` whose `ast.unparse` is
    `RAY_TEMP_DIR if args.ray_address == 'local' else None`.
- Each test's docstring names the failure it guards: an SGK Ray instance the lock cannot see.

Run it into `measure/out/01-guard.txt`. Then the **deliberate breakage** (§3.1), each into the same
file.

**2.5 The helper goes.** `git rm tools/rehearsal_lock.py tools/tests/test_rehearsal_lock.py`. Then
the checklist's item 8.3 grep:

```bash
git grep -n -E 'ScienceData|\.rehearsal-lock|rehearsal-lock-brief|tools\.rehearsal_lock|rehearsal_lock\.py' -- . ':!docs' ':!prompts'
```

on the working tree, which before the edits finds 18 lines, after 2.5 four (`CLAUDE.md`), and after
2.6 **one**: the new lock section's sentence naming the old location (§2.6).

**2.6 `CLAUDE.md`.**
- **The pin.** In "The datastore layer is `datastorekit`; …", `raypoolkit` "pinned at `v0.2.0`"
  becomes "pinned at `v0.3.0` (moved to it on <date> in the `rpk-v030-adoption` campaign, which
  made its `raypoolkit.rehearsal_lock` this repository's rehearsal lock)". Rewrap to ~100 columns.
- **Rule 5's Ray check.** `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock ray-processes`
  becomes `./venv/bin/python -m raypoolkit.rehearsal_lock ray-processes --ray-temp-dir /tmp/ray-sgk`.
  The rest of the rule stands.
- **"The shared rehearsal lock"** is rewritten to say, in the file's voice and no longer than it is
  now by more than a few lines:
  - three projects share one lock, the directory `~/.raypoolkit/rehearsal-lock`, the module
    `raypoolkit.rehearsal_lock` of the pinned `raypoolkit`, whose contract is RayPoolKit's
    `docs/rehearsal-lock.md` at `v0.3.0`; SGK's values are `--project SecondaryGWKit`,
    `--ray-temp-dir /tmp/ray-sgk` (`config/ray_local.py`) and `--ray-address in-process`;
  - **the protocol**, stated here, not by pointer: `acquire --purpose "<text>"` before the first
    launch; `set-pid <pid>` with the driver's pid at each launch; held through the readers;
    `release` after them, once no Ray process is left. `status` and `ray-processes` look. Give the
    full `acquire`, `set-pid` and `release` command lines once;
  - **a held lock (exit 3) means stop and ask the user.** Never wait, retry or remove it; the
    module has no option to. Exit 3 also comes while the old location `~/ScienceData/.rehearsal-lock`
    exists: a project not yet moved to `v0.3.0` holds it, or it is stale;
  - the lock finds an instance **by its temp dir**: every SGK Ray start gives `_temp_dir`
    (`RunDrivers/tests/test_ray_temp_dir.py`), and one that does not is invisible to it;
  - each lock command is a shell command of its own, never the one that launches a build;
  - kept as they are: every Ray driver names its address (`local` for this repository's), never
    `auto`; `ray stop` is never run.

  Drop the bullet pointing at StochasticInstantons' brief and `prompts/runregistry-extraction/`
  §4.1.

**2.7 The roots, after.** The seven roots again into `01-roots.txt`: `tools` lower by exactly 42,
`RunDrivers` higher by exactly the tests 2.4 added, every other root as in 2.1, all OK.
`black --check` clean on every touched `.py` file.

**2.8 The rehearsal under the new lock** (README §6.2 C5). Save every command and its output into
`measure/out/01-lock.txt` (the lock commands) and `01-rehearsal.txt` (the rest). Run commands with
variable arguments through `bash`.
1. **Take the lock:**
   `./venv/bin/python -m raypoolkit.rehearsal_lock acquire --project SecondaryGWKit --purpose "rpk-v030-adoption 01: rehearsal under the new lock" --ray-temp-dir /tmp/ray-sgk --ray-address in-process`.
   Exit 0. Exit 3 or 4: stop (§4).
2. **Launch R3's `reference`**, as `docs/rrk-v020-adoption-verification.md` §1.1 shows it, with
   `RunDrivers/quadsource_atol_sweep.py`, `--database var/datastores/rpk-v030-adoption-01-reference.sqlite`,
   `--job-name rpk-v030-adoption-01`, `--register reference`, `--campaign rpk-v030-adoption
   --prompt 01`, the launch log `var/runs/rpk-v030-adoption-01-reference.launch.log`, and a
   `--purpose` naming this prompt, the tree's commit, `raypoolkit` 0.3.0 at `67a75b9`, and the
   fingerprint it must equal, `39809dca`. Detached, as there.
3. **One check**, once the launch log shows the driver has started Ray (wait for that line, not a
   fixed time):
   - the manifest and `status.json` as R3 checked them, with `raypoolkit` 0.3.0 at `67a75b97…` in
     `packages`;
   - `set-pid --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk <driver pid>`: exit 0;
   - `ray-processes --ray-temp-dir /tmp/ray-sgk`: **exit 1, listing the instance**, its
     `gcs_server` and `raylet` among the lines;
   - `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk`: **exit 1**, the recorded pid
     alive;
   - a second `acquire` with the same values: **exit 3**;
   - `status --ray-temp-dir /tmp/ray-sgk --ray-address in-process`: the lock HELD, the owner pid
     alive, the instance listed.

   Then wait on the driver pid in the background, as R3's waits did: no polling, no `pgrep -f`.
4. **After the driver's pid has died:** the run's `status.json` says `done`, exit 0.
   `ray-processes --ray-temp-dir /tmp/ray-sgk` exit 0; if Ray's processes are still shutting down,
   wait on their pids, as listed, and record it.
5. **The fingerprint.** As R3 §1.5 did: the store's sidecar, then
   `docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py` against
   `var/runs/rrk-v020-adoption-03-reference-20261018T232838`. **It must equal `39809dca…`.**
6. **The readers**, with the lock still held:
   `bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh var/datastores/rpk-v030-adoption-01-reference.sqlite var/rpk-v030-adoption-01/extract`.
   Each exits 0, as R3's did, and the store is unchanged. Then `ray-processes` exit 0.
7. **Release:** `release …` exit 0; `status` shows the lock free and no `old location` line.

The rehearsal store is left in place; its retirement is the user's (README §3).

## 3. Acceptance

1. **The pin.** The dry run adds only `raypoolkit`; after it, `raypoolkit` 0.3.0 at `67a75b97…`,
   not editable; the other packages unchanged; `pip check` clean.
2. **The helper is gone**, and §2.5's grep finds only `CLAUDE.md`'s one sentence on the old
   location.
3. **The ten starts give the temp dir**, the new test passes, and catches each mutation of §3.1.
4. **The roots:** `tools` −42, `RunDrivers` + the new tests, the rest unchanged, all OK. A failure
   among README §7's four known tests is re-run alone and recorded.
5. **Under real Ray**, the lock lists the rehearsal's instance by its temp dir, refuses `release`
   and a second `acquire` while it runs, and is released cleanly after; the rehearsal ends `done`
   at fingerprint `39809dca…`; the readers exit 0 and leave the store unchanged.
6. `git diff --cached --name-only` is: `requirements.txt`, `CLAUDE.md`, `config/ray_local.py`,
   `main.py`, the six readers, the two drivers, `RunDrivers/tests/test_ray_temp_dir.py`, the two
   deleted files, `measure/out/01-*`, the log and this board. Nothing else.

### 3.1 Deliberate breakage

Apply each, run the new test module, record which test failed and the diff exactly as applied, and
revert:
- **M1** — drop `_temp_dir` from `extract_Gk_data.py`'s `ray.init`.
- **M2** — drop `address="local"` from `run_child`'s wrapper.
- **M3** — make `main.py`'s `_temp_dir=RAY_TEMP_DIR` unconditional.
- **M4** — a new file `tools/_probe.py` holding `import ray` and `ray.init(address="local")`.
- **M5** — `RAY_TEMP_DIR = "/tmp/ray-sgk/"`.

## 4. Stop conditions — stop and ask

- A Gate item is not as stated, or an old lock exists at `~/ScienceData/.rehearsal-lock` at any
  point (a project not yet moved holds it).
- The dry run would install, upgrade or remove anything but `raypoolkit`.
- A root fails other than by a known test; a count differs from §2.7's expectation.
- An existing test cannot be kept passing by the forms of §2.3.
- `acquire` exits 3 or 4; `ray-processes` does not list the running instance; `release` succeeds
  while it runs; the rehearsal does not end `done`; its fingerprint is not `39809dca…`.
- Anything suggests a defect in `raypoolkit`: record it on the board and in `docs/OPEN_ISSUES.md`,
  and stop. It is not worked around here.

**If you stop with the lock held**, say so and leave it held, with its owner file printed: the
orchestrator decides. If you stop after the install, say that the venv holds `v0.3.0`.

## 5. What this prompt does not do

- It touches nothing in ChamPBH, StochasticInstantons or RayPoolKit, and nothing on the Mac Studio.
- It changes no record under `docs/` or `prompts/` other than this campaign's, including the four
  `docs/handover/a3-v2/**/scripts/*.sh` that call the old helper.
- It changes nothing a build computes or stores.
- It does not close the campaign: the review does.

## 6. The log and the board

**The log**, at `logs/01-switch-to-the-raypoolkit-rehearsal-lock.md`, in README §5.1's form, with
the breakage record, every count, and the lock's and the rehearsal's transcripts summarised.

**The board:** the header ("01 landed, awaiting review"); the §1 row; items P1, D1, T1, C1, R1 and
B1; gate FP; `### 5.1 After prompt 01 (<date>)`. `docs/OPEN_ISSUES.md` only if §3 gains an entry.

Run `./venv/bin/python -m black` on every touched `.py` file before committing.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
