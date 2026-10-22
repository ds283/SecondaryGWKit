# Prompt 06 — the A3 v2 host: the Mac Studio's checkout and venv, and rehearsal R4 there

**Campaign:** [`README.md`](README.md) · **Board item:** **06** (outside workstreams A–E, like 05) ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-22, with prompt 07, at `c081eed`, on the user's decisions of the same day (§0.2).
**Where it runs:** **on the Mac Studio** (`mps026298`), in a Claude Code session opened over the
saved `MacStudio` SSH connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`. **It commits
nothing there** (§7).
**Gate:**
- the dispatch commit is on `origin/handover-remedial`, so the Mac Studio can fetch it;
- on the laptop, `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale;
- the Mac Studio's checkout has an empty `git status --short` (it read so on 2026-10-22, on `main`
  at `9678f28`).

**Opens:** anything you find, **without fixing it**.
**What it decides:** **R4**, which 07 needs (board §2, item 06).
**Recommended model:** **Opus**. A venv built on another machine, four registered builds, the
readers, under the lock.

**Read first:**

1. `CLAUDE.md`: "Long-running jobs — the run registry", "The shared rehearsal lock", the packages
   section, and "Repository mechanics".
2. **`rrk-v020-adoption`'s
   [`03-rehearse-through-v0.2.0.md`](../rrk-v020-adoption/03-rehearse-through-v0.2.0.md)** §0, §1
   and §3, which this prompt runs again with the names and differences below. Where this prompt is
   silent, that 03 governs, and through it the chain it names (`runregistry-extraction` 08, 04 and
   02, `dk-and-rpk-migration` 08). **Its §2, the hard-kill pair, is not run here** (§0.3).
3. [`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md) §1 (R3:
   how each launch went, the commands verbatim) and §3.5 (what A3 v2 inherits, with the commands in
   their current spelling). `docs/rrk-v020-adoption/rehearsal/*_03.txt` are R3's saved outputs,
   which yours should look like.
4. [`docs/a3-v2-readiness-verification.md`](../../docs/a3-v2-readiness-verification.md) §2.4, A3
   v2's launch checklist. Its command names `docs/handover/quadsource_atol_sweep.py`; the launcher
   has been `RunDrivers/quadsource_atol_sweep.py` since `031bcb4`.
5. [`prompts/dk-v050-adoption/measure/out/01-pip.txt`](../dk-v050-adoption/measure/out/01-pip.txt),
   the laptop venv's `pip freeze` after the `v0.5.0` install (its second section), and
   `01-roots.txt` beside it, the seven roots' counts.
6. This board's §3, `[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]`.

---

## 0. What this prompt is

### 0.1 Why

A3 v2 runs on the Mac Studio (`a3-v2-readiness` README §3; the user, 2026-10-22). **No rehearsal
has run there**, and three things have changed since the last rehearsal, R3 (2026-10-18/19), on the
laptop:
- **the host.** The Mac Studio's checkout is on `main` at `9678f28`, 336 commits behind this
  branch, with a Python 3.13 venv holding `ray` 2.45.0 and none of the three packages;
- **the launcher's bytes.** R3 ran `docs/handover/quadsource_atol_sweep.py` at SHA-256
  `2a985303…`. Prompt 05 of `rrk-v020-adoption` made its refusals clean (`9b40e457…`), and
  `031bcb4` moved it to `RunDrivers/` (`b2fa4ef1…` at `c081eed`). Neither change touches what runs
  after `begin`, and neither has been through a rehearsal build;
- **`datastorekit`**, moved from `v0.4.0` to `v0.5.0` by `dk-v050-adoption` (`542ddd1`). Its gate
  FP held on the golden-fingerprint test, not on a build.

**R4 is R3 again, on the Mac Studio, at this tree.** If its stores reproduce `39809dca…`, the host,
the venv, the launcher as it now stands and the `v0.5.0` pin change nothing a build writes or a
fingerprint reads, and 07 may launch.

### 0.2 The user's decisions (2026-10-22)

- **The Mac Studio's venv is Python 3.12**, from this branch's `requirements.txt` (`ray` 2.43.0),
  matching the laptop and every rehearsal. Moving to 3.13 is later work
  (`[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]`).
- **The Mac Studio runs the builds and the analysis; the campaign records are committed on the
  laptop.** No development lands on this repository while A3 v2 runs.
- **`--cpus 6`**, as rehearsed: the Mac Studio also serves a low-traffic website's object store.
- **Route B**: a Claude Code session on the Mac Studio does the work there.

### 0.3 What this prompt is not

- It changes no code and no test, and **writes no file in the repository** on the Mac Studio
  outside `var/`, which is ignored: every manifest must record `git_dirty: false`, and 07 needs the
  same checkout clean. Everything you produce goes under `var/handover-06/` (§7).
- **It does not run R3's hard-kill pair.** R3 showed the route after a hard kill on this launcher,
  through `runregistrykit` `v0.2.0`, which is unchanged; what the route checks (`host`, `boot`,
  `kill -0`) is the package's, and works the same on any host. A Ray process that outlives a
  `SIGKILL` here is handled by `CLAUDE.md` rule 5's manual step. An orchestrator's choice; the user
  may ask for it.
- It keeps no store. R4's two stores are the user's to retire after the review, or not.

---

## 1. The host, before anything is launched

Make `var/handover-06/out/` first (`mkdir -p`; `var/` is in `.gitignore`). Save each step's
commands and output, in order, in `var/handover-06/out/host_06.txt`.

1. **The machine.** `hostname`; `sw_vers`; `sysctl -n hw.ncpu hw.memsize machdep.cpu.brand_string`;
   `uptime`; `df -h ~`; `pmset -g`; the ten processes using the most CPU (`ps -Ao
   pid,pcpu,etime,comm -r | head -11`).
2. **The checkout.** `git status --short` (empty), `git branch --show-current`, `git log -1
   --format='%H %s'`. Then `git fetch origin`, `git switch handover-remedial` (it tracks
   `origin/handover-remedial`), and record `git log -1 --format=%H` — it must be the dispatch SHA
   the orchestrator gives you — and `git status --short`, empty.
3. **The old venv.** Record `cat venv/pyvenv.cfg` and `./venv/bin/pip freeze` into
   `var/handover-06/out/old_venv_06.txt`. Then **move it out of the checkout**, not delete it:
   `mkdir -p ~/Documents/Code/retired-venvs && mv venv
   ~/Documents/Code/retired-venvs/SecondaryGWKit-py313-20261022`. The user retires it.
4. **The new venv.** `/opt/local/bin/python3.12 -m venv venv`, then `./venv/bin/pip install -r
   requirements.txt`. Save the install's output in `var/handover-06/out/pip_install_06.txt`. Do not
   upgrade `pip` unless the install fails without it, and say so if it does. Check that `venv` is
   ignored here (`git check-ignore -v venv`, and `git status --short` still empty).
5. **The venv against the laptop's.** Save `./venv/bin/python --version` and `./venv/bin/pip
   freeze` in `var/handover-06/out/pip_06.txt`, and diff the freeze against the **second** freeze
   in `prompts/dk-v050-adoption/measure/out/01-pip.txt` (the laptop's after the `v0.5.0` install).
   **The freeze must be identical.** In particular `datastorekit @ …@f47b9a255e9ad58e0d0eaf85644f53b575884d31`,
   `raypoolkit @ …@eaef0ce60e2fc354bb32f0ad576010e6a83c1976`,
   `runregistrykit @ …@0bae3a3c501a8264ee4be5bc24f06422af38720f`, `ray==2.43.0`. Then
   `runregistrykit.package_provenance(("datastorekit", "raypoolkit", "runregistrykit"))`, as
   `01-pip.txt` runs it: all three `editable: false`. A Python patch version other than the
   laptop's 3.12.15 is recorded, not a stop.
6. **The seven roots**, each by `CLAUDE.md`'s command, at the dispatch SHA. The counts must equal
   `01-roots.txt`'s: `AdaptiveLevin` 32, `ComputeTargets` 700, `CosmologyModels` 39,
   `LiouvilleGreen` 148 (`skipped=1`), `ObjectFactories` 279, `RunDrivers` 115, `tools` 91. Save
   them in `var/handover-06/out/roots_06.txt`. `ComputeTargets`'
   `test_tk_wkb_phase.…test_wall_time_per_object` is a known wall-clock flake: if it alone fails,
   run that root once more and record both.
7. **The registry and the lock.** `./venv/bin/python -m runregistrykit list` (no runs: there is no
   `var/runs/` here yet), and `this_host()` as R3's pre-launch check reads it. `PYTHONPATH=.
   ./venv/bin/python -m tools.rehearsal_lock status` (no lock; `~/ScienceData` did not exist on
   2026-10-22, and `acquire` makes it) and `ray-processes` (nothing). `echo "${RAY_ADDRESS-unset}"`
   is `unset`.

---

## 2. R4 — R3's four launches, on the Mac Studio

**Run R3's §1 as written** (`rrk-v020-adoption` 03 §1), with these substitutions:
- **The launcher:** `RunDrivers/quadsource_atol_sweep.py`. Record its SHA-256 before the first
  launch: `b2fa4ef1bbdbb09fea1780952710eeb53dee401ef6c0febcd385dd98aa0526ef` at `c081eed`, and
  unchanged at the dispatch SHA unless a commit since touched it, which is a stop.
- **Stores:** `var/datastores/handover-06-reference.sqlite` and
  `var/datastores/handover-06-resumed.sqlite`, each `absent` to `store show` before its first
  launch.
- **Runs:** `--campaign handover --prompt 06`, slugs `reference`, `resumed-a`, `resumed-b` and
  `resumed-c`, and `--job-name handover-06` on every launch; `--atol 1e-32 --rtol 1e-7 --cpus 6
  --rehearsal`, as R3. Each `--purpose` names R4 of `handover` prompt 06 on the Mac Studio, the
  tree's commit, `datastorekit` 0.5.0 at `f47b9a2`, the fingerprint it must equal and R3's run that
  recorded it (`rrk-v020-adoption-03-reference-20261018T232838`).
- **Launch logs:** `var/runs/handover-06-<slug>.launch.log`.
- **Reader outputs:** `var/handover-06/extract/`.
- **Saved outputs:** `var/handover-06/out/`, named as R3's with `06` (`prelaunch_checks_06.txt`,
  `lock_06.txt`, `stage_counts_06.txt`, …).
- **The lock:** `acquire --purpose "SecondaryGWKit handover 06 (R4 on the Mac Studio)"`, held from
  before `reference` until after the readers and §3's checks, `set-pid` at each launch that begins
  a run, then `release`.
- **The recorded fingerprints.** R3's run directories are on the laptop. The orchestrator copies
  `rrk-v020-adoption-03-reference-20261018T232838/` and
  `rrk-v020-adoption-03-resumed-c-20261018T233941/` to `var/handover-06/recorded/` here before
  dispatch. `compare_with_recorded.py`'s `--against` names those two.
- **The waits.** One check after each launch; then wait for the driver's pid to die with **one
  background command** that exits when `kill -0` fails (`while kill -0 <pid> 2>/dev/null; do sleep
  15; done`), and resume when it reports. Never a foreground poll, never `pgrep -f`. The waits for a
  killed launch's Ray agents are by pid, as R3's.
- **Interruption points** are your own, chosen from R4's `reference` `stdout.log` by R3's reasoning,
  and recorded with it.

**What is new against R3:**
- **The host.** Every manifest's and status's `host.id` is the Mac Studio's, equal to §1.7's
  `this_host()`; `boot` matches its `uptime`.
- **`datastorekit` 0.5.0** at `f47b9a255e9ad58e0d0eaf85644f53b575884d31` (`requested_revision`
  `v0.5.0`) in every manifest's `packages`, where R3's named 0.4.0 at `c8570e8…`.
- **The launcher's path and hash** in every manifest's `script` and `script_sha256`.
- **The stage times** are the Mac Studio's. Record them beside R3's; they are not compared, but 07
  scales A3 v2's wall-clock estimate from them.

---

## 3. What to verify

**L1–L3 are R3's §3**, on this prompt's two stores and four runs, with the substitutions above:
- every manifest's `packages` names the three packages at their pins, none editable; every
  `git_dirty` is `false`; every `git_head` is the dispatch SHA; every `host` the Mac Studio's;
- **both stores are format 6, `39809dca…`, 21 classes, `encoding_format` 1**, each equal to its
  sidecar's, and `compare_fingerprints` between them is empty;
- `compare_with_recorded.py` on both primaries against R3's two recorded run directories: every
  comparison empty. Save it as `fingerprint_against_recorded_06.txt`.

**L4 — equal to R3**, as R3 stated it against R2 (`rrk-v020-adoption` verification §1): the
statistics block's counts, as counts and as sorted lines
(`[02-the-policy-statistics-block-orders-its-lines-by-completion]` makes raw order vary); the
reference stage counts; the resumed chain's identities; the stage sequence; the printed lines and
the registry's fields. A new line in a log, against R3's same slug, that names `datastorekit`
outside a path, a `DeprecationWarning`, or a refusal is a finding to explain.

**The readers** (L3) run as R3 ran them, every one with `--ray-address local`, leaving every store
file's SHA-256 and mtime as it was. They are the A3 v2 store's readers, and the analysis runs here.

**The registry after R4.** `list` shows R4's four runs and nothing else, nothing `running` or
stale, none counted "with no host recorded". Save it as `run_records_06.txt`.

---

## 4. Acceptance

1. **The checkout** is on `handover-remedial` at the dispatch SHA, `git status --short` empty
   before the first launch and after the last.
2. **The venv** is Python 3.12, its `pip freeze` identical to the laptop's, the three packages at
   their pins and not editable; the old venv moved, not deleted.
3. **The seven roots** pass at `01-roots.txt`'s counts.
4. **L1–L3 hold:** both R4 stores are `39809dca…`, equal to each other, their sidecars and R3's
   recorded fingerprints; every manifest is as §3 says.
5. **L4** against R3, or each difference recorded, explained and opened.
6. **The readers** pass and leave the stores untouched.
7. **The lock** was taken before `reference`, set at each launch, released at the end; afterwards
   `status` reports no lock and `ray-processes` lists nothing.
8. **The records of §7** are complete under `var/handover-06/`.

## 5. Stop conditions — stop and ask

R3's §5 conditions that bear on the four launches apply, with R3 in place of R2, and:
- **`pip install` fails**, or the freeze differs from the laptop's in any line. Record the diff;
  do not pin, upgrade or install anything by hand.
- **A root's count differs**, or a test fails other than the one flake of §1.6.
- **An R4 store's fingerprint differs from `39809dca…`.** Record both fingerprints and the
  package's `compare_fingerprints`, change nothing, relaunch nothing, and stop. With the venv
  identical, the cause is the host, the launcher's two changes, or `v0.5.0`; finding which is not
  this prompt's.
- **The launcher's SHA-256 is not `b2fa4ef1…`** at the dispatch SHA.
- A manifest's `packages` or `host` is not as §3 says, or a `git_dirty` is `true`.
- The lock is held (exit 3), a Ray process of this checkout is alive at `acquire` (exit 4), or
  `release` refuses.
- Anything on the Mac Studio outside this checkout, `~/Documents/Code/retired-venvs` and
  `~/ScienceData/.rehearsal-lock` would need changing.

## 6. What this prompt does not do

- It changes no tracked file, on either machine; the laptop's records are the orchestrator's (§7).
- It does not touch `main` or the Mac Studio's other checkouts, and does not delete the old venv.
- It runs no hard kill, no `end`, and calls `Run.finish` on no run.
- It retires no store and launches nothing at production scope.
- It does not fix what it finds.

## 7. The records

**On the Mac Studio, under `var/handover-06/` only:**
- `out/`: every saved output of §1–§3, named with `06`;
- `log.md`: the log, in `qcd-background-audit` README §5.1's form, classifying every deviation as
  `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`. It records what R3's log
  records, against R3, and §1's host and venv;
- `verification.md`: **§1 "R4 — the Mac Studio"** of a new `docs/handover/A3-V2-VERIFICATION.md`,
  in R3's §1 shape, opening with §1's host and venv;
- `issues.md`: a board §3 entry, in this board's form, for anything you opened.

**On the laptop, the orchestrator** copies `out/` to `docs/handover/a3-v2/rehearsal/`, the log to
`logs/06-a3-v2-host-and-rehearsal.md` and `verification.md` into
`docs/handover/A3-V2-VERIFICATION.md`, reviews them, updates this board (the 06 row, the 06
item, §3 and §4) and `docs/OPEN_ISSUES.md`, and makes **one commit** in `CLAUDE.md`'s form.
That commit is the one 07's checkout fast-forwards to.
