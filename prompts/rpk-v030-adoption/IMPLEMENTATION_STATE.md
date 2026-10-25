# rpk-v030-adoption campaign — implementation state

**Last updated:** 2026-10-25 · **Status: PLANNED — 01 written, not dispatched.** U1 and U2 ruled by
the user.

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
| 01 | [Switch to the `raypoolkit` rehearsal lock](01-switch-to-the-raypoolkit-rehearsal-lock.md) | the pin; the helper deleted; the temp dir at the ten Ray starts and its test; `CLAUDE.md`; a rehearsal and the readers under the new lock | Opus | ✍️ 2026-10-25 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

The review of 01 closes the campaign (README §2).

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **P1** | 01 | `requirements.txt:76` at `v0.3.0`; the venv installed from it; `CLAUDE.md`'s pin | ⬜ |
| **D1** | 01 | `tools/rehearsal_lock.py` and `tools/tests/test_rehearsal_lock.py` deleted; no live line names the old lock | ⬜ |
| **T1** | 01 | `config/ray_local.py`'s `RAY_TEMP_DIR`; `_temp_dir` at the three wrappers (and `address="local"` at `run_child`'s) and the seven scripts; `RunDrivers/tests/test_ray_temp_dir.py` | ⬜ |
| **C1** | 01 | `CLAUDE.md`'s lock section, protocol and hard-kill route | ⬜ |
| **R1** | 01 | the rehearsal under the new lock: the lock sees the instance, refuses `release` while it runs, and the fingerprint is `39809dca…`; the six readers exit 0 on its store | ⬜ |
| **B1** | 01 | the baselines: `pip freeze`, provenance, the seven roots before and after | ⬜ |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| **E** | RayPoolKit has tagged `v0.3.0` with a contract and a checklist for SGK | — | ✅ holds (2026-10-25): `v0.3.0` on `67a75b9`; `docs/rehearsal-lock.md`; `docs/adoption/rehearsal-lock/secondarygwkit.md` |
| **FP** | the rehearsal fingerprint is `39809dca…` through `v0.3.0` and the temp dir | 01 | ⬜ |

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
