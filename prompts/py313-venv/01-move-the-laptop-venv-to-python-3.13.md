# Prompt 01 — move the laptop's venv to Python 3.13, `ray` 2.59.0 and `SQLAlchemy` 2.0.55

**Campaign:** [`README.md`](README.md) · **Board items:** **B1**, **P1**, **V1**, **G1**, **C1**,
**S1**, **R6**, **K1** · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, with the plan, at `ebfae11`.
**Where it runs:** on the laptop, in the main checkout, on `handover-remedial`.
**Gate:**
- the planning commit has landed, and `git status` is clean;
- no commit after the planning commit touches code (anything outside `prompts/` and `docs/`);
- `git worktree list` shows the main checkout alone. A worktree's venv is a symlink to this one, so
  it would move with it;
- `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale;
- `./venv/bin/python --version` is 3.12.15, and `pip freeze` names `ray==2.43.0` and
  `SQLAlchemy==2.0.39`;
- `/opt/local/bin/python3.13 --version` is 3.13.16;
- the rehearsal lock is free (`status`), and `ray-processes --ray-temp-dir /tmp/ray-sgk` lists
  nothing;
- `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-<today>` does not exist.

**Closes:** `[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]` (`handover` board §3),
when acceptance 1–5 hold. **Opens:** only what the work finds.
**Recommended model:** **Opus**. A new interpreter, a Ray sixteen minor releases on, six registered
builds and a hard kill, under the lock.

**Read first:**

1. `CLAUDE.md`, all of it. The run-registry rules and the lock section are used throughout.
2. [`README.md`](README.md), all of it. It is short.
3. **R3**: `rrk-v020-adoption`'s
   [`03-rehearse-through-v0.2.0.md`](../rrk-v020-adoption/03-rehearse-through-v0.2.0.md) §1–§3,
   which R6 runs again with the names and differences below. Where this prompt is silent, that 03
   governs, and through it the chain it names. Its lock commands are `tools.rehearsal_lock`'s.
   Use `CLAUDE.md`'s `raypoolkit.rehearsal_lock` commands instead, as
   [`rpk-v030-adoption` 01](../rpk-v030-adoption/01-switch-to-the-raypoolkit-rehearsal-lock.md)
   §1 and §2.8 use them.
4. [`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md) §1
   (R3: how each launch went, with the commands verbatim) and §2 (the hard-kill pair).
   `docs/rrk-v020-adoption/rehearsal/*_03.txt` are R3's saved outputs. Yours should look like
   them.
5. [`docs/handover/A3-V2-VERIFICATION.md`](../../docs/handover/A3-V2-VERIFICATION.md) §1 (R4, R3
   on the Mac Studio). It is the most recent run of the four launches, and its interruption points
   and stage times are the ones to compare against.
6. `ComputeTargets/tests/test_ray_address_default.py`'s docstring. It is the test that should
   fail if Ray changed what `address="local"` means.
7. The `handover` board's §3 entries `[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]`
   and `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`.

---

## 1. What is wanted

**Move the laptop's venv to Python 3.13.16 with `ray` 2.59.0 and `SQLAlchemy` 2.0.55. Show that
the suite passes as it stands, and that a rehearsal build stores exactly what it stored under
3.12.** Do it in one commit: the two pins, `distributions`, `CLAUDE.md`, the records and the board.

The venv is not in `git`, so the move takes effect at once and the commit does not record it. Take
the baselines (§2.1) **before** the old venv is moved.

**Run each lock command as a shell command of its own**, the only process that command starts, with
output by `>>` redirection and never a pipe (`CLAUDE.md`, "The shared rehearsal lock").

## 2. What to do

Save every command and its output under `measure/out/`, each block headed by its command. Run
commands with variable arguments through `bash`.

**2.1 The baselines, under 3.12.**
- Into `01-pip.txt`: `./venv/bin/python --version`; `cat venv/pyvenv.cfg`; `./venv/bin/pip
  freeze`; and `runregistrykit.package_provenance(("datastorekit", "raypoolkit", "runregistrykit",
  "ray", "SQLAlchemy", "numpy", "scipy"))` printed as JSON.
- Into `01-roots.txt`: the seven roots, by README §7's command, serially, in the foreground. Record
  each count, and the wall time of each root.
- Into `01-black.txt`: `./venv/bin/python -m black --check --extend-exclude '^/prompts/' .`. The
  tree outside `prompts/` is clean (`CLAUDE.md`). Record the file count.

**2.2 The old venv, moved aside.** `mkdir -p ~/Documents/Code/retired-venvs && mv venv
~/Documents/Code/retired-venvs/SecondaryGWKit-py312-<today>`. Record the path in `01-pip.txt`. It
is the rollback (README §4.1), and retiring it is the user's.

**2.3 The pins.** In `requirements.txt`, line 75 changes from `ray==2.43.0` to `ray==2.59.0`, and
line 88 changes from `SQLAlchemy==2.0.39` to `SQLAlchemy==2.0.55`. No other line moves.

**2.4 The new venv.** Into `01-pip.txt`:
- `/opt/local/bin/python3.13 -m venv venv`. `git status --short` stays empty (`.gitignore` names
  `venv/`).
- `./venv/bin/pip install -r requirements.txt`. Do not upgrade `pip` unless the install fails
  without it, and say so if it does.
- `./venv/bin/python --version`, which is 3.13.16, and `cat venv/pyvenv.cfg`.
- `pip freeze`. Diff it against 2.1's freeze. **They differ in exactly two lines, `ray` and
  `SQLAlchemy`.** That includes the three git packages, at the same commits and not editable.
- `pip check`, which is clean.
- `package_provenance` as in 2.1: `ray` 2.59.0, `SQLAlchemy` 2.0.55, and the rest as in 2.1.
- `./venv/bin/python -c "import ray, sqlalchemy, sqlite3; print(ray.__version__, sqlalchemy.__version__, sqlite3.sqlite_version)"`.
  Record SQLite's version against 3.12's. It comes from the interpreter's build, so a change here
  is a finding (README §5 rule 5).

**2.5 `distributions`** (README §6.2 C3). In `runregistrykit.toml`, `distributions =
["datastorekit", "raypoolkit"]` becomes `["datastorekit", "raypoolkit", "ray", "SQLAlchemy",
"numpy", "scipy"]`. Add a one-line comment above it saying that the last four are there so that a
manifest says which venv wrote a store.

**2.6 `CLAUDE.md`** (README §6.2 C6). In "Repository mechanics", add one bullet saying that the
venv is Python 3.13 on the laptop and on the Mac Studio, built with
`/opt/local/bin/python3.13 -m venv venv` and then `pip install -r requirements.txt`. Say that it
moved from 3.12 on <date> in the `py313-venv` campaign. Write it in the file's voice and wrap it to
~100 columns. The packages section's sentence "This repository keeps its own `ray` and `SQLAlchemy`
pins" stays as it is.

**2.7 The roots and `black`, under 3.13.** Into the same files as 2.1:
- **The seven roots**, run the same way. **Every count equals 2.1's, with the same skips and every
  root OK.** A failure among README §7's four known tests is re-run alone and recorded. Run
  `test_ray_address_default` on its own as well and record its result: it passing is the evidence
  that `local` still means a fresh instance under 2.59.0. Record each root's wall time beside
  2.1's.
- **`black --check`**, as in 2.1, which is clean with the same file count.
- **Warnings.** Run one root, `ComputeTargets`, once more with `-W default`. Record any
  `DeprecationWarning` or `FutureWarning` that names a file in this repository or one of the three
  packages. Record warnings from third-party internals as a count. This is a finding, not a stop
  (README §5 rule 5).

**2.8 R6: R3's four launches.** Run R3's §1 as written, with these substitutions:
- **Stores:** `var/datastores/py313-venv-01-reference.sqlite` and
  `var/datastores/py313-venv-01-resumed.sqlite`, each `absent` to `store show` before its first
  launch.
- **Runs:** `--campaign py313-venv --prompt 01`, with slugs `reference`, `resumed-a`, `resumed-b`
  and `resumed-c`, and `--job-name py313-venv-01` on every launch. The flags are R3's (`--atol
  1e-32 --rtol 1e-7 --cpus 6 --rehearsal`). Each `--purpose` names R6 of `py313-venv` prompt 01 on
  the laptop, the tree's commit, Python 3.13.16 with `ray` 2.59.0 and `SQLAlchemy` 2.0.55, the
  fingerprint it must equal (`39809dca`), and R3's run that recorded it
  (`rrk-v020-adoption-03-reference-20261018T232838`). No `--purpose` names `/tmp/ray-sgk`.
- **Launch logs** go to `var/runs/py313-venv-01-<slug>.launch.log`, and **reader outputs** to
  `var/py313-venv-01/extract/`.
- **Saved outputs** go to `measure/out/`, named after R3's with `01`
  (`01-prelaunch_checks.txt`, `01-lock.txt`, `01-stage_counts.txt`, …).
- **The lock.** `acquire --purpose "SecondaryGWKit py313-venv 01 (R6)"`. It is held from before
  `reference` until after the hard-kill pair (§2.9), with `set-pid` at each launch that begins a
  run, and then `release`.
- **The waits.** Make one check after each launch. Then wait for the driver's pid to die with **one
  background command** that exits when `kill -0` fails, and resume when it reports. Never use a
  foreground poll or `pgrep -f`.
- **Interruption points** are chosen as R4's were, from R6's `reference` `stdout.log`, and
  recorded with it.

**What is new against R3:**
- **The interpreter.** Every manifest's `host.python` is `3.13.16`, where R3's was `3.12.15`.
- **`packages`** names seven distributions, not three, because of 2.5: `ray` 2.59.0, `SQLAlchemy`
  2.0.55, `numpy` 2.2.4 and `scipy` 1.16.3, with `url` and `commit` null, beside the three git
  packages as in R3.
- **Ray 2.59.0.** During `reference`'s one check, `ray-processes --ray-temp-dir /tmp/ray-sgk`
  exits 1 and lists the instance, with its `gcs_server` and `raylet` among the lines. This is the
  evidence that the lock still sees a 2.59.0 instance by its temp dir. **If it lists nothing while
  the build runs, stop**: it is a `raypoolkit` finding (§4).
- **The `SIGTERM`ed launches.** `resumed-a` and `resumed-b` end `killed`, through the handler, as
  R3's and R4's did. Record how long each one's Ray processes took to go, beside R3's.
- **The stage times** are recorded beside R3's and R4's. They are not compared.

**2.9 R6: R3's hard-kill pair** (README §6.2 C4). Run R3's §2 as written, after `resumed-c` and the
readers, under the same lock, on `var/datastores/py313-venv-01-hardkill.sqlite`, with the slugs
`hardkill-a` and `hardkill-b`, and with `01` in place of `03` in every name. Save it as
`measure/out/01-hardkill.txt`. **Its §2.3 decides the stop.** If any Ray process of this
repository is alive ten minutes after the driver's death, report each pid, its command and its
parent. Do not signal it, do not run `ray stop`, do not release the lock, and stop: the user
decides. Whatever happens, record how long the GCS server took to go, beside R3's 20 m 50 s.

**2.10 What to verify.** R3's §3, L1–L4, on R6's three stores and six runs, with the substitutions
above. In particular:
- every manifest's `git_head` is the dispatch `HEAD`, and its `packages` and `host.python` are as
  above. Its `git_dirty` is `true`: the pin edits, `runregistrykit.toml` and the records are
  uncommitted until the prompt's one commit, as in `rpk-v030-adoption` 01 and `handover` 09. No
  `.py` file is among the dirty paths. Record `git status --short` at each launch to show it;
  *corrected 2026-10-25 before dispatch, where the plan had `false`*;
- **every store is format 6, `39809dca…`, 21 classes, `encoding_format` 1**, equal to its sidecar;
- `compare_with_recorded.py` on each primary against R3's `reference` and `resumed-c` run
  directories: every comparison empty;
- the readers, as R3 ran them, each exit 0 and leave every store file's SHA-256 and mtime as it
  was;
- **L4 against R4**, the latest full run: the statistics block's counts, as counts and as sorted
  lines; the reference stage counts; the resumed chain's identities; the stage sequence; and the
  printed lines. A new line is recorded and explained (README §5 rule 5).

**2.11 The verification record.** Write `docs/py313-venv-verification.md`. It opens with a short
header saying what the campaign moved and which prompt wrote which section. Then add **§1 "R6 —
the laptop on Python 3.13"**, in the shape of R3's §1 and §2 in
`docs/rrk-v020-adoption-verification.md`. It opens with the venv: the interpreter, the two moved
pins, the SQLite version, and the path the old venv was moved to. Leave §2 for prompt 02, with a
single line saying so.

## 3. Acceptance

1. **The venv.** It is Python 3.13.16. Its `pip freeze` differs from 3.12's in exactly the `ray`
   and `SQLAlchemy` lines, the three git packages are at their commits and not editable, and
   `pip check` is clean. The old venv was moved, not deleted, and its path is recorded.
2. **The suite.** All seven roots have the same counts as under 3.12, all OK, with
   `test_ray_address_default` passing. `black --check` is clean.
3. **R6's four launches.** `reference` and `resumed-c` end `done`, and `resumed-a` and `resumed-b`
   end `killed` at the points chosen. Both stores are `39809dca…`, equal to R3's recorded
   fingerprints. The lock lists the running instance by its temp dir.
4. **The hard-kill pair** runs R3's route: `stale`, the refused resume, `end` (`lost`, verified),
   `hardkill-b` to `done`, and its store `39809dca…`. Ray's last process is gone within ten
   minutes, or the stop of §2.9 was taken.
5. **The readers** exit 0 on R6's stores and leave them unchanged. The lock was released at the
   end, and afterwards `ray-processes` lists nothing.
6. `git diff --cached --name-only` lists exactly:
   - `requirements.txt`, `runregistrykit.toml` and `CLAUDE.md`;
   - `docs/py313-venv-verification.md` and `docs/OPEN_ISSUES.md`;
   - `prompts/py313-venv/measure/out/01-*`, the log and this campaign's board;
   - `prompts/handover/IMPLEMENTATION_STATE.md`, and `prompts/rrk-v020-adoption/IMPLEMENTATION_STATE.md`
     if §6 narrows the GCS issue.

   Nothing else.

## 4. Stop conditions — stop and ask

- A Gate item is not as stated.
- **`pip install` fails**, or the freeze differs from 3.12's in any line but `ray` and
  `SQLAlchemy`. Record the diff, and do not pin, upgrade or install anything by hand.
- **A root's count differs**, a test fails other than one of the four known tests, or
  `test_ray_address_default` fails.
- **An R6 store's fingerprint differs from `39809dca…`.** Record both fingerprints and
  `compare_fingerprints`, change nothing and relaunch nothing. The bisection of README §4.3 is the
  user's call.
- A manifest's `packages` or `host.python` is not as §2.8 says, or a `.py` file is dirty at a
  launch.
- **The lock does not list the running instance**, `acquire` exits 3 or 4, or `release` refuses.
  Anything that suggests a defect in `raypoolkit`, `datastorekit` or `runregistrykit` is recorded
  on the board and in `docs/OPEN_ISSUES.md`, and not worked around here.
- §2.9's ten-minute condition.

**If you stop with the lock held**, say so and leave it held, with its owner file printed: the
orchestrator decides. **If you stop after §2.2**, say that the venv is 3.13 and where the 3.12 venv
is. Do not move it back yourself.

## 5. What this prompt does not do

- It changes no `.py` file and no test.
- It touches nothing on the Mac Studio, in `main`, in ChamPBH, StochasticInstantons or the three
  packages.
- It moves no pin but the two.
- It deletes no venv and retires no store.
- It does not fix what it finds.

## 6. The log and the board

**The log** goes to `logs/01-move-the-laptop-venv-to-python-3.13.md`, in README §5.1's form, with
every count, the two freezes' diff, and the rehearsal's and the hard-kill pair's transcripts
summarised.

**The board:**
- the header ("01 landed, awaiting review"), the §1 row, and items B1, P1, V1, G1, C1, S1, R6 and
  K1;
- gate FP6, and `### 5.1 After prompt 01 (<date>)`;
- **the issue closed.** On the `handover` board, move
  `[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]` from §3 to §4 with a
  `**Closed <date> by py313-venv 01:**` line. Say that this branch is now on 3.13.16 with `ray`
  2.59.0, later than `main`'s 2.45.0, and that a merge resolves `requirements.txt:75` to this
  branch's line. Record it in this board's §4 as closed on another board. Delete its row from
  `docs/OPEN_ISSUES.md` §1.23, and correct that file's count and date.
- **The GCS issue.** If the hard-kill pair measured it, narrow
  `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` on the `rrk-v020-adoption` board, and
  its row in `docs/OPEN_ISSUES.md` §1.1, with what Ray 2.59.0 did.

Run §2.1's `black --check` once more before committing. It must still be clean.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`. Then `git push`: 02 fetches it on the Mac Studio.
