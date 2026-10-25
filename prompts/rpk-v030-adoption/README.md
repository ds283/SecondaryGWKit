# Campaign — rpk-v030-adoption

**Written:** 2026-10-25 on branch `handover-remedial` at `c0fa7ea`, by Claude Opus 5.5. It moves
SGK from `raypoolkit` `v0.2.0` to `v0.3.0` and replaces SGK's own rehearsal lock,
`tools/rehearsal_lock.py`, with the package's `raypoolkit.rehearsal_lock`, so that the lock is one
piece of code shared by every project that uses it. The user asked for the move on 2026-10-25,
before `handover` prompt 09 runs, landing directly on `handover-remedial` (§6.1).

It has **no audit document of its own**. The audit is RayPoolKit's:
- the contract, `docs/rehearsal-lock.md` ("contract §N" below), at the tag `v0.3.0`
  (`67a75b97a656accb98195a4bda93f64e6a5f3fb2`);
- the checklist for this repository, `docs/adoption/rehearsal-lock/secondarygwkit.md` ("the
  checklist"), measured read-only at SGK `216e44d`. Its own words: it is data, not instructions.
  **It is not in the tag**: it was written after the release, in RayPoolKit `127b2b4`, and is read
  at `feae4e5`, where the contract and `raypoolkit/` are the tag's. (Corrected 2026-10-25 in the
  review of 01, whose log records the reading as deviation 1; as first written, this paragraph put
  both documents at the tag, and prompt 01's "Read first" 3 still does.)

§0.2 reconciles it with `c0fa7ea`. 01 measures again before acting, and its numbers govern.

## 0. What changes

**What `v0.3.0` adds, and nothing else under `raypoolkit/`:** `raypoolkit/rehearsal_lock.py`, its
tests, and an import-guard test (`git diff --stat v0.2.0 v0.3.0 -- raypoolkit pyproject.toml`).
`RayWorkPool`, its arguments and its printed lines are `v0.2.0`'s, so nothing a build computes or
stores can move.

**What it means for SGK** (the checklist, items 1–7):
- **The lock moves** from `~/ScienceData/.rehearsal-lock` to `~/.raypoolkit/rehearsal-lock`, and
  the commands become `./venv/bin/python -m raypoolkit.rehearsal_lock …` with SGK's values:
  `--project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk --ray-address in-process`.
- **The process rule changes.** SGK's helper finds Ray processes by SGK's venv; the module finds an
  instance by its **temp dir** (contract §6). So every Ray start SGK runs under the lock must give
  Ray that temp dir, or the lock cannot see it. There are ten:
  - three `_real_ray_init` wrappers in `RunDrivers/`: `run_build`'s and `run_child`'s in
    `quadsource_atol_sweep.py`, and `main`'s in `scoped_pipeline_run.py`. **`run_child`'s passes no
    address today**; it gains `address="local"` with the temp dir, as the other two have;
  - seven module-level `ray.init(address=args.ray_address)`: `main.py` and the six
    `extract_*_data.py` readers. They pass the temp dir only when the address is `"local"`: with
    `auto` or a `host:port`, `_temp_dir` would change which cluster Ray finds or split the
    driver's files from the cluster's (the checklist, item 4).
- **SGK's helper and its 42 tests are deleted.** The `tools/tests` root falls by 42.
- **`CLAUDE.md`** states the new lock and its commands, and the hard-kill route's Ray check
  becomes `ray-processes --ray-temp-dir /tmp/ray-sgk`.

### 0.1 The other two projects

ChamPBH and StochasticInstantons share the lock, and still pin `v0.2.0` and take the old location.
Until they move, they cannot see SGK's lock, and SGK's `acquire` refuses (exit 3) while their old
lock exists. RayPoolKit's plan (its decision D3) has the three move on one day. **The user has
ruled that SGK moves now and that the user moves the other two when they can** (§6.1 U2): the
hazard is two projects' Ray running side by side on the laptop, and the user is the one person who
starts them. This campaign does nothing in either repository.

### 0.2 Reconciliation against `c0fa7ea`

Measured by the orchestrator on 2026-10-25 while writing this plan.
- **The tree since the checklist's `216e44d`:** outside `docs/` and `prompts/`, only `.gitignore`
  changed. The checklist's grep for Ray starts finds the same ten calls at the same lines
  (`quadsource_atol_sweep.py:487`, `:629`; `scoped_pipeline_run.py:709`; `main.py:314`; the readers
  at `extract_GkSource_data.py:118`, `extract_GkWKB_data.py:110`, `extract_Gk_data.py:109`,
  `extract_QuadSourceIntegral_data.py:130`, `extract_TkWKB_data.py:106`,
  `extract_tensor_source_data.py:110`), and its grep for the old lock finds the same **18** live
  lines.
- **The registry:** 51 runs, none `running` or stale. **Neither lock location exists**, on the laptop
  or on the Mac Studio.
- **The venv:** `raypoolkit` 0.2.0 at `eaef0ce6…`, `datastorekit` 0.5.0 at `f47b9a25…`,
  `runregistrykit` 0.2.0 at `0bae3a3c…`, none editable. A dry run of `requirements.txt` with the
  pin moved says `Would install raypoolkit-0.3.0` and nothing else.
- **The tests the edits must keep passing** are the checklist's item 6, and the forms that keep them
  passing are its item 4's two tables: one `ray.init` per script with no positional argument and
  `address=args.ray_address`; `main.py`'s call a top-level expression statement; each wrapper's
  `address` the constant `"local"`; and no lower-case `rehearsal` anywhere in `run_child`.

## 1. Scope

**In scope (01):**
- the pin, `requirements.txt:76`, and the venv installed from it;
- `tools/rehearsal_lock.py` and `tools/tests/test_rehearsal_lock.py` deleted;
- the temp dir in one constant, given at the ten Ray starts; a test that every Ray start in the
  repository gives it;
- `CLAUDE.md`'s pin, its lock section and its hard-kill route;
- a rehearsal build and the readers under the new lock, with real Ray, as the check that the lock
  sees SGK's instances and that the fingerprint does not move.

**Out of scope:**
- **ChamPBH, StochasticInstantons and RayPoolKit** (§0.1). A defect in the package is recorded and
  reported, never patched or worked around here (`CLAUDE.md`).
- **What the code stores.** No table, column, key or value; the rehearsal fingerprint stays
  `39809dca…`.
- **Records.** `prompts/` and dated `docs/` stay as written, including the four
  `docs/handover/a3-v2/**/scripts/*.sh` that call the deleted helper: they are records of dated
  runs. `prompts/runregistry-extraction/README.md` §4.1, the old protocol, stays as written;
  `CLAUDE.md` states the protocol itself from now on.
- **The Mac Studio's venv.** It moves when `handover` 10 moves that checkout (§4.2).

## 2. Prompts

| # | Prompt | Covers | Model | Status |
|---|---|---|---|---|
| 01 | [`01-switch-to-the-raypoolkit-rehearsal-lock.md`](01-switch-to-the-raypoolkit-rehearsal-lock.md) | the pin; the helper deleted; the temp dir at the ten starts and its test; `CLAUDE.md`; a rehearsal and the readers under the new lock | Opus | landed `40d10b1`, reviewed 2026-10-25 |

There is no close prompt. The orchestrator's review of 01, in its own commit, closes the campaign:
the board's header and §1, and the campaign's line in `prompts/INDEX.md`. Residue the review finds
is fixed in a follow-up commit before the close.

## 3. Datastores

01 builds one throwaway rehearsal store, `var/datastores/rpk-v030-adoption-01-reference.sqlite`,
and reads it. Retiring it is the user's, after the review.

## 4. The interfaces

### 4.1 Rollback

01 is one commit, reverted by `git revert`. **The venv must be put back too**, since the install is
not in `git`:

```bash
./venv/bin/pip install "raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0"
```

A revert while the new lock is held leaves `~/.raypoolkit/rehearsal-lock` behind, which the old
helper cannot see: release it first.

### 4.2 What 01 hands on

- **To `handover` 09:** the venv on `v0.3.0`, and the lock commands of `CLAUDE.md`. 09's
  rehearsal and readers run under them. 09 is held until 01 has landed and been reviewed; its
  prompt file says so.
- **To `handover` 10, on the Mac Studio:** 10 is the first time that checkout moves, so it installs
  the new pin there (`pip install -r requirements.txt`) before it reads A3 v2, and its Gate allows
  01's two edits to `main.py` and `config/`. Its prompt file says so. RayPoolKit's switch-over
  procedure considers one machine only; the Mac Studio is a second host that takes the lock.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:
1. **Existing tests are not modified.** The helper's own tests are deleted with it; nothing else
   under `*/tests/` changes but the new test module.
2. **Deliberate breakage.** The new test catches the mutations 01 names; the log records each as a
   diff, exactly as applied. Mutations are never committed.
3. **No test needs Ray.** The real-Ray check is the rehearsal, not a test.
4. **The pin is a tag.** The commit carries the pin and what the pin makes necessary, which here is
   the whole switch: with the helper still present, two locks would exist side by side.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE`
  or `UNINTENDED DRIFT`;
- **Verification performed**, with every count against the baselines taken before;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed on** (to `handover` 09 and 10).

## 6. Decisions

### 6.1 Made by the user (2026-10-25)

| | Decided |
|---|---|
| **U1** — when | Move to `v0.3.0` now, before `handover` 09 |
| **U2** — the other projects | SGK moves alone; the user moves ChamPBH and StochasticInstantons when they can. No branch held back for a common day |

### 6.2 Choices of method made in writing this plan

The orchestrator's; the user can reverse any of them.
- **C1 — the temp dir is `/tmp/ray-sgk`**, the checklist's advice: 12 bytes, so Ray's socket paths
  stay inside macOS's limit; named by no other process or project. Always spelled exactly so: the
  lock compares strings as given (contract §5).
- **C2 — one constant, `RAY_TEMP_DIR`, in a new module `config/ray_local.py`.** `main.py`, the
  readers and the drivers import it. `config/defaults.py` is numerical tolerances.
- **C3 — the readers' and `main.py`'s form** is
  `ray.init(address=args.ray_address, _temp_dir=RAY_TEMP_DIR if args.ray_address == "local" else None)`,
  which the checklist found keeps every existing test passing. `_temp_dir=None` is Ray's default.
- **C4 — the guard test** lives in `RunDrivers/tests/test_ray_temp_dir.py`. It finds every
  `ray.init(…)` and `_real_ray_init(…)` call in the repository's live `.py` files by `ast`, so a
  start added later without the temp dir fails it.
- **C5 — the real-Ray check is R3's reference rehearsal** (about 4 minutes) plus the six readers,
  under the new lock, rather than a toy script: it exercises SGK's own starts, and its fingerprint
  is known.
- **C6 — `CLAUDE.md` states the protocol itself** (take before the first launch, `set-pid` at each
  launch, hold through the readers, release after) and stops pointing at a closed campaign's §4.1,
  the checklist's advice.
- **C7 — the name** `rpk-v030-adoption`, after `dk-v050-adoption`.

## 7. Baselines

Re-measure every root at `HEAD`, clean, before dispatch:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . > <file> 2>&1
grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' <file>
```

**Under load, four tests are known to fail on wall clock or chance:**
`test_tk_wkb_phase.TestCost.test_wall_time_per_object`; `test_background_tau.test_build_cost`;
`test_numeric_phase_cut.test_the_print_is_gated_both_ways`; LiouvilleGreen's
`test_3bessel_analytic.test_JJJ`. Re-run a failure alone before calling it real.
