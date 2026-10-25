# Prompt 02 — move the Mac Studio's venv to Python 3.13, and read A3 v2 under it

**Campaign:** [`README.md`](README.md) · **Board items:** **M1**, **M2**, **R7**, **A1** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, with the plan, at `ebfae11`. **Held** until 01 has landed and been
reviewed.
**Where it runs:** **on the Mac Studio** (`mps026298`), in a Claude Code session opened over the
saved `MacStudio` SSH connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`, as `handover`
06–10 were. **It commits nothing there.** Everything it writes is under `var/py313-venv-02/` (§6).
**Gate:**
- 01 has landed and been reviewed, gate FP6 holds, and the dispatch commit is 01's review commit or
  later, pushed to `origin/handover-remedial`;
- **no commit since `b90752d` touches `ObjectFactories/`, `config/` or `main.py`** except
  `rpk-v030-adoption` 01's `config/ray_local.py` and `main.py`'s `ray.init`, which is `handover` 10's
  Gate restated. A3 v2 was built at `b90752d`, and this is what makes reading it at the dispatch
  SHA sound (`CLAUDE.md`). `git diff --stat b90752d <dispatch SHA> -- ObjectFactories config
  main.py` shows that commit's two files and no other;
- on the laptop, `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale;
- on the Mac Studio, `runregistrykit list` shows nothing `running` or stale, the rehearsal lock is
  free, `git status --short` is empty, the venv is 3.12.15, and `/opt/local/bin/python3.13
  --version` is 3.13.16;
- `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-<today>` does not exist there.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Opus**.

**Read first:**

1. `CLAUDE.md`, all of it, at the dispatch SHA. It now names the 3.13 venv (01).
2. [`README.md`](README.md), and [`01-move-the-laptop-venv-to-python-3.13.md`](01-move-the-laptop-venv-to-python-3.13.md),
   which this prompt runs again here, without the pin edits and without the hard-kill pair.
3. 01's log and `docs/py313-venv-verification.md` §1 (R6). R7 should look like R6.
4. [`prompts/handover/06-a3-v2-host-and-rehearsal.md`](../handover/06-a3-v2-host-and-rehearsal.md)
   §1–§3, the last time this host's venv was rebuilt and rehearsed on (R4). Its lock commands are
   the old helper's. Use `CLAUDE.md`'s instead.
5. [`prompts/handover/10-reread-a3-v2.md`](../handover/10-reread-a3-v2.md) §1, the last read of A3
   v2. Its steps 3, 5 and 7 are repeated here.

---

## 0. What this prompt is

The Mac Studio builds and analyses, and A3 v2 lives there. This prompt puts that host on the venv
01 showed on the laptop, and shows two things. Its own rehearsal stores what R6 stored, which is
**R7**. And the A3 v2 store, written under 3.12, reads the same under 3.13. After it, the queued
analysis jobs run here on the new venv.

---

## 1. Steps

Make `var/py313-venv-02/{out,state}` first. Save every command and its output, in order, under
`out/`, named with `02`. Run commands with variable arguments through `bash`.

1. **The host and the checkout** (`out/host_02.txt`). `hostname`, `uptime`, `df -h ~`, and the ten
   processes using the most CPU. Then `git fetch origin && git merge --ff-only
   origin/handover-remedial`. `HEAD` is the dispatch SHA, `git status --short` is empty, and the
   Gate's `git diff --stat` is as stated.
2. **The baselines, under 3.12** (`out/pip_02.txt`, `out/roots_02.txt`). The same as 01 §2.1:
   `python --version`, `pyvenv.cfg`, `pip freeze`, and `package_provenance` of the seven
   distributions. Then the seven roots, serially, in the foreground, with counts and wall times.
3. **The old venv, moved aside.** `mv venv
   ~/Documents/Code/retired-venvs/SecondaryGWKit-py312-<today>`, and record the path. Leave
   `SecondaryGWKit-py313-20261022` beside it as it is.
4. **The new venv** (`out/pip_02.txt`). `/opt/local/bin/python3.13 -m venv venv`, then `./venv/bin/pip
   install -r requirements.txt`, without upgrading `pip`. Then `python --version` (3.13.16), `pip
   check`, `package_provenance`, and the `ray`/`sqlalchemy`/`sqlite3` versions line of 01 §2.4.
   **`pip freeze` is identical to the laptop's**, the last freeze in
   `prompts/py313-venv/measure/out/01-pip.txt`. `git status --short` stays empty.
5. **The roots, under 3.13** (`out/roots_02.txt`). Every count equals 01's under 3.13, all OK. A
   failure among README §7's four known tests is re-run alone and recorded. Record the wall times
   beside step 2's.
6. **The A3 v2 store, before** (`state/store_files_before.txt`, `out/store_02.txt`). `store show
   var/datastores/handover-A3-v2-lambdacdm.sqlite`: it is live, with fingerprint `0e32f24a…`.
   Record the SHA-256, size and mtime of the primary and its four shards.
7. **The lock.** `acquire --purpose "SecondaryGWKit py313-venv 02 (R7 and the A3 v2 read)"`, held
   through steps 8–10, and then `release`. Exit 3 means stop and ask.
8. **R7.** R6's four launches (01 §2.8), with these substitutions:
   - `02` for `01` in every name, store, run, launch log and saved output;
   - `var/py313-venv-02/` for `measure/out/`;
   - each `--purpose` naming R7 on the Mac Studio;
   - `--cpus 6`, as R4;
   - `compare_with_recorded.py`'s `--against` naming R3's two run directories under
     `var/handover-06/recorded/`.

   Make one check after each launch, then use one background wait on the driver's pid. During
   `reference`'s check, `ray-processes` lists the instance. Interruption points are chosen as R6's
   were. **The hard-kill pair is not run** (README §6.2 C4). Then run 01 §2.10, L1–L4, against R6
   and R4. Both stores are `39809dca…`.
9. **The A3 v2 readers** (`out/readers_02.txt`). `bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh
   var/datastores/handover-A3-v2-lambdacdm.sqlite var/py313-venv-02/extract`, all six, then
   `extract_Gk_data` with `nomodels`, as 10 ran them. Record each one's exit, wall time, files
   written and size, beside 10's (`docs/handover/a3-v2/reread/out/readers_10.txt`). The counts can
   differ, because the readers' samples are unseeded (`[10-the-readers-draw-their-work-lists-unseeded]`):
   a difference is recorded, not a stop. A new warning in a reader's log is a finding (README §5
   rule 5).
10. **The A3 v2 store, after** (`state/store_files_after.txt`). Every store file is unchanged
    against step 6. Run `store fingerprint` on it without `--write`. **It equals the sidecar's
    `0e32f24a…`, computed now under 3.13.** No Ray process is left.
11. `release`. Then `status` reports the lock free and `ray-processes` lists nothing
    (`out/lock_02.txt`). `runregistrykit list` shows R7's four runs and nothing `running` or stale
    (`out/run_records_02.txt`).

## 2. Acceptance

1. **The checkout** is at the dispatch SHA, and `git status --short` is empty before the first
   launch and after the last.
2. **The venv** is Python 3.13.16, its `pip freeze` identical to the laptop's, and `pip check` is
   clean. The old venv was moved, not deleted.
3. **The roots** have 01's counts under 3.13.
4. **R7.** Both stores are `39809dca…`, equal to each other, to their sidecars and to R3's recorded
   fingerprints. Every manifest records `host.python` 3.13.16 and the seven distributions, with
   `git_dirty` false.
5. **A3 v2** reads under 3.13. Every reader exits 0, every store file is byte-identical, and its
   fingerprint, computed under 3.13, is `0e32f24a…`.
6. **The lock** was taken and released.
7. **The records of §6** are complete.

## 3. Stop conditions — stop and ask

- A Gate item is not as stated.
- `pip install` fails, or the freeze differs from the laptop's in any line. Record the diff, and do
  not pin, upgrade or install anything by hand.
- A root's count differs, or a test fails other than one of the four known tests.
- **An R7 store is not `39809dca…`, or A3 v2's fingerprint under 3.13 is not `0e32f24a…`.** Record
  both fingerprints and `compare_fingerprints`, and change nothing.
- A reader exits non-zero, or any A3 v2 store file changes.
- The lock is held, does not list the running instance, or will not release.

**If you stop after step 3**, say that the venv is 3.13 and where the 3.12 venv is. Do not move it
back yourself.

## 4. What this prompt does not do

- It changes no tracked file on either machine. The laptop's records are the orchestrator's (§6).
- It moves no pin, deletes no venv and retires no store. A3 v2 is read and nothing more.
- It runs no hard kill and no `end`.
- It does not fix what it finds.

## 5. Handed on

**To the queued analysis jobs:** the Mac Studio's venv on 3.13.16, `ray` 2.59.0 and `SQLAlchemy`
2.0.55, and A3 v2 shown to read the same under it.

## 6. The records

**On the Mac Studio, under `var/py313-venv-02/` only:**
- `out/` and `state/`, every saved output of §1;
- `log.md`, the log, in README §5.1's form;
- `verification.md`, which is **§2 "R7 — the Mac Studio on Python 3.13, and A3 v2 read under it"**
  of `docs/py313-venv-verification.md`, in §1's shape;
- `issues.md`, holding a board §3 entry, in this board's form, for anything opened.

**On the laptop, the orchestrator** copies `out/` and `state/` to `prompts/py313-venv/measure/out/02/`,
the log to `logs/02-move-the-mac-studio-venv-to-python-3.13.md`, and `verification.md` into
`docs/py313-venv-verification.md` §2. It reviews them, updates this board and
`docs/OPEN_ISSUES.md`, and makes **one commit** in `CLAUDE.md`'s form. The reader output,
`extract/`, stays on the Mac Studio.
