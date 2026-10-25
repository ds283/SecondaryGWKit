# rpk-v030-adoption campaign — implementation state

**Last updated:** 2026-10-25 · **Status: 01 landed, awaiting review.** U1 and U2 ruled by the
user.

The plan is [`README.md`](README.md), written on 2026-10-25 at `c0fa7ea` from RayPoolKit's
`v0.3.0` contract and its switch-over checklist for SGK.

The campaign moves SGK's `raypoolkit` pin from `v0.2.0` to `v0.3.0` and replaces SGK's own
rehearsal lock with the package's, `raypoolkit.rehearsal_lock`. The lock moves to
`~/.raypoolkit/rehearsal-lock` and finds Ray processes by their temp dir, so every SGK Ray start
gives Ray `/tmp/ray-sgk`.

**Campaign:** [`README.md`](README.md) ·
**Code:** `requirements.txt`, `CLAUDE.md`, `config/ray_local.py`, `main.py`, the six
`extract_*_data.py` readers, `RunDrivers/quadsource_atol_sweep.py`,
`RunDrivers/scoped_pipeline_run.py`, `RunDrivers/tests/test_ray_temp_dir.py`; `tools/rehearsal_lock.py`
and its tests deleted ·
**Verification:** 01's rehearsal and readers under the new lock, in `measure/out/` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) (no section yet: no issue is open)

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row is added, moved or deleted, and the count and date in its header are corrected. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-25 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** — when | move to `v0.3.0` now, before `handover` 09 |
| **U2** — the other projects | SGK moves alone, on `handover-remedial`; the user moves ChamPBH and StochasticInstantons when they can |

The orchestrator's choices of method, C1–C7, are in README §6.2.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Switch to the `raypoolkit` rehearsal lock](01-switch-to-the-raypoolkit-rehearsal-lock.md) | the pin; the helper deleted; the temp dir at the ten Ray starts and its test; `CLAUDE.md`; a rehearsal and the readers under the new lock | Opus | ✍️ 2026-10-25 | ✅ 2026-10-25 | `40d10b1` | [log](logs/01-switch-to-the-raypoolkit-rehearsal-lock.md) |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

The review of 01 closes the campaign (README §2).

**Orchestrator notes** are written just before dispatch, under `orchestrator/`:
[`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-25, for 01). The planning commit
is `2b27509`.

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **P1** | 01 | `requirements.txt:76` at `v0.3.0`; the venv installed from it; `CLAUDE.md`'s pin | ✅ 01: the dry run installs `raypoolkit-0.3.0` alone; venv 0.3.0 at `67a75b97…`, not editable; freeze differs by one line; `pip check` clean |
| **D1** | 01 | `tools/rehearsal_lock.py` and `tools/tests/test_rehearsal_lock.py` deleted; no live line names the old lock | ✅ 01: both deleted (42 tests); the grep finds 18 → 1, `CLAUDE.md`'s sentence on the old location |
| **T1** | 01 | `config/ray_local.py`'s `RAY_TEMP_DIR`; `_temp_dir` at the three wrappers (and `address="local"` at `run_child`'s) and the seven scripts; `RunDrivers/tests/test_ray_temp_dir.py` | ✅ 01: the ten starts give it; the module's 5 tests pass, and each of M1–M5 fails one |
| **C1** | 01 | `CLAUDE.md`'s lock section, protocol and hard-kill route | ✅ 01 |
| **R1** | 01 | the rehearsal under the new lock: the lock sees the instance, refuses `release` while it runs, and the fingerprint is `39809dca…`; the six readers exit 0 on its store | ✅ 01: `ray-processes` 1 (18 processes), `release` 1, second `acquire` 3; `done`/0 at `39809dca…`; readers exit 0, store unchanged; released 0 |
| **B1** | 01 | the baselines: `pip freeze`, provenance, the seven roots before and after | ✅ 01: tools 91 → 49, RunDrivers 115 → 120, the rest unchanged; the known wall-clock test fired both times |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| **E** | RayPoolKit has tagged `v0.3.0` with a contract and a checklist for SGK | — | ✅ holds (2026-10-25): `v0.3.0` on `67a75b9`; `docs/rehearsal-lock.md`; `docs/adoption/rehearsal-lock/secondarygwkit.md` |
| **FP** | the rehearsal fingerprint is `39809dca…` through `v0.3.0` and the temp dir | 01 | ✅ holds (2026-10-25): `39809dca…`, `compare_with_recorded.py` equal against R3's `reference` |

The other two projects' move is not a gate here (U2).

## 3. Active and unresolved issues

None.

## 4. Resolved issues

None.

## 5. Dispatch record

### 5.0 Planning (2026-10-25)

Proposed in conversation on 2026-10-25. The user ruled U1 and U2: move now, alone, with no branch
held back for a common day with the other two projects. Measured at `c0fa7ea` (README §0.2): the
checklist still describes the tree; no run `running` or stale; neither lock location exists on the
laptop or the Mac Studio; a dry run of the moved pin installs `raypoolkit-0.3.0` alone.

The planning commit also holds `handover` 09 on 01, and amends `handover` 10's Gate and step 2 for
the new pin on the Mac Studio (README §4.2).

### 5.1 After prompt 01 (2026-10-25)

Landed in one commit on `5add52d`. P1, D1, T1, C1, R1 and B1 hold, and so does gate FP; no stop
condition fired. Log:
[`logs/01-switch-to-the-raypoolkit-rehearsal-lock.md`](logs/01-switch-to-the-raypoolkit-rehearsal-lock.md);
measurements: `measure/out/01-{pip,roots,guard,lock,rehearsal}.txt`.
- **The pin.**
  - `requirements.txt:76` is `@v0.3.0`.
  - The dry run said `Would install raypoolkit-0.3.0` and nothing else.
  - The venv holds `raypoolkit` 0.3.0 at `67a75b97…`, `requested_revision` `v0.3.0`, not editable.
    `datastorekit` and `runregistrykit` are unchanged, `pip check` is clean, and `pip freeze`
    differs by the one line.
- **The temp dir.**
  - `config/ray_local.py` holds `RAY_TEMP_DIR = "/tmp/ray-sgk"`.
  - The three wrappers pass `_temp_dir=RAY_TEMP_DIR`, and `run_child`'s now passes
    `address="local"` too.
  - The seven scripts pass it only with `local`.
  - `RunDrivers/tests/test_ray_temp_dir.py` has 5 tests, keyed on the qualified enclosing function.
    M1–M5 each fail one of its tests and were reverted.
- **The helper is gone**, with its 42 tests. §2.5's grep finds 18 lines before and 1 after, the
  sentence on the old location in `CLAUDE.md`'s lock section, which now states the protocol itself.
- **The roots**, before and after, serially:

  | Root | Before | After |
  |---|---|---|
  | AdaptiveLevin | 32 | 32 |
  | ComputeTargets | 700 | 700 |
  | CosmologyModels | 39 | 39 |
  | LiouvilleGreen | 148 (1 skipped) | 148 (1 skipped) |
  | ObjectFactories | 280 | 280 |
  | RunDrivers | 115 | 120 |
  | tools | 91 | 49 |

  README §7's `test_wall_time_per_object` failed under load both times and passed once alone each
  time. Every other test passes.
- **The rehearsal under the new lock** (`rpk-v030-adoption-01-reference-20261025T010358`, Python
  pid 73154):
  - `acquire` exited 0 before the launch and `set-pid 73154` 0.
  - `ray-processes` exited 1 and listed 18 processes, `gcs_server` and `raylet` among them.
  - `release` exited 1 while the driver lived, and a second `acquire` exited 3.
  - The run ended `done`/0, with fingerprint **`39809dca…`**, equal to R3's `reference` and to the
    new sidecar (`store_id` `0213c3c6…`).
  - The six readers exited 0 under the held lock, each leaving the store byte-identical, and each
    Ray instance's session is under `/tmp/ray-sgk`.
  - Then `ray-processes` exited 0, `release` 0, and `status` showed the lock free with no old
    location.
- **No issue opened or closed;** `docs/OPEN_ISSUES.md` is unchanged.
- **The venv is ahead of `git revert`:** reverting 01 leaves `v0.3.0` installed (README §4.1).
- **For the review:** README lines 9–13 say the checklist is read at the tag, and it is not there.
  It was read at RayPoolKit `feae4e5` (log, deviation 1).
