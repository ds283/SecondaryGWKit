# rrk-v020-adoption campaign — implementation state

**Last updated:** 2026-10-18 · **Status: PLANNED.** 01–03 written, none dispatched; 04 held until
03's review (README §2.1). U1–U3 ruled by the user (README §6.1). One issue assigned here from
`handover` (§3).

The plan is [`README.md`](README.md), written on 2026-10-18 at `79434bf` from RunRegistryKit's
`v0.2.0` release notes and its adoption note for SGK.

The campaign moves SGK's `runregistrykit` pin from `v0.1.0` to `v0.2.0`, puts the route after a
hard kill (`end`, then the resume) into `CLAUDE.md`, and shows it on the A3 v2 launcher in a
rehearsal, R3, before the A3 v2 compute run depends on it.

**Campaign:** [`README.md`](README.md) ·
**Code:** `requirements.txt`, `CLAUDE.md`, `docs/datastore-integrity-verification/interrupt_when.py`,
`docs/a3-v2-readiness/interrupter_build_dummy_test.sh`, a new test module in `RunDrivers/tests/` ·
**Verification:** `docs/rrk-v020-adoption-verification.md` (created by 03) ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.22

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row is added, moved or deleted, and the count and date in its header are corrected. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-18 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** — the shape | four prompts: pin, adopt, rehearse, close |
| **U2** — the hard kill | a pair of launches on a third store, after the unchanged four-launch protocol |
| **U3** — Ray after a `SIGKILL` | stop and ask if a Ray process of this repository outlives the killed driver |

The orchestrator's choices of method, C1–C8, are in README §6.2. **C3 departs from the plan as
accepted** (the drivers keep raising `begin`'s refusal with its traceback); it is the user's to
reverse (README §6.3).

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Pin `runregistrykit` at `v0.2.0`](01-pin-runregistrykit-at-v0.2.0.md) | the pin; the baselines; `CLAUDE.md`'s pin mentions | Sonnet | ✍️ 2026-10-18 | ⬜ | — | — |
| 02 | [Adopt the hard-kill route](02-adopt-the-hard-kill-route.md) | `CLAUDE.md`'s rules; `interrupt_when.py --signal`; the driver test | Sonnet | ✍️ 2026-10-18 | ⬜ | — | — |
| 03 | [Rehearse through `v0.2.0` (R3)](03-rehearse-through-v0.2.0.md) | **R3**; the hard-kill pair; closes `[a3-v2-a-hard-killed-run-blocks-its-store]` | Opus | ✍️ 2026-10-18 | ⬜ | — | — |
| 04 | Close the campaign | the final measurements; G1 drafted; the index's notes; what A3 v2 inherits | Sonnet | ⏸ after 03's review | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Written in batches (README §2.1).** 01–03 now; 04 after 03's review. The hold is for **method**,
not scope: every charter is fixed in README §2.2.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, under
`orchestrator/`.

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **K1** | 01 | `requirements.txt:81` moved to `v0.2.0`; the venv installed from it | ⬜ |
| **B1** | 01 | the baselines: `pip freeze`, provenance, `list`, `store show` on the 21 sidecars, the seven roots | ⬜ |
| **A1** | 02 | `CLAUDE.md`: liveness by host and boot; the route after a hard kill; no `Run.finish` by hand | ⬜ |
| **A2** | 02 | `interrupt_when.py --signal {TERM,KILL}`; the build dummy test's `KILL` case | ⬜ |
| **A3** | 02 | the driver test: a `SIGKILL`ed run refuses `register`; `end`; `register` accepted | ⬜ |
| **L1**–**L4** (R3) | 03 | README §4.1's four launches through `v0.2.0`; L4 against R2 | ⬜ |
| **H1** | 03 | the hard-kill pair: `hardkill-a` killed, the refused resume, `end`, `hardkill-b` to `done`, `39809dca…` | ⬜ |
| **H2** | 03 | no Ray process outlives the `SIGKILL`ed driver (U3) | ⬜ |
| **Z1** | 04 | the final measurements | ⬜ |
| **Z2** | 04 | G1 drafted for RunRegistryKit's `interrupted-runs` board | ⬜ |
| **Z3** | 04 | `docs/OPEN_ISSUES.md`'s notes on the transferred issues `v0.2.0` closed | ⬜ |
| **Z4** | 04 | what A3 v2 inherits | ⬜ |
| **Z5** | 04 | the records: verification §3, this board, README §2, `prompts/INDEX.md`, `docs/OPEN_ISSUES.md` | ⬜ |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| **E** | RunRegistryKit has tagged `v0.2.0` with release notes and an SGK adoption note | — | ✅ holds (2026-10-18): tag object `895e3c70…` on `0bae3a3c…`; `docs/releases/v0.2.0.md` and `docs/adoption/v0.2.0/secondarygwkit.md` in the tag; nothing under `runregistrykit/` or in `pyproject.toml` changed after it (RunRegistryKit `f6d4b80`) |
| R3 | the four launches through `v0.2.0` reproduce `39809dca…`, equal to R2's recorded fingerprints | 03 | ⬜ |
| HK | a `SIGKILL`ed build's resume is refused at `begin`, `end` records the run `lost`, and the resume then reproduces `39809dca…` with its fingerprint recorded | 03 | ⬜ |
| RunRegistryKit G1 | SGK has adopted `v0.2.0` in a campaign of its own, its pin moved in a commit of its own | 01 | ⬜ — recorded on RunRegistryKit's board by the user (README §6.3); 04 drafts the record |

## 3. Active and unresolved issues

- **[a3-v2-a-hard-killed-run-blocks-its-store]** — *opened 2026-10-16 on the
  [`handover`](../handover/IMPLEMENTATION_STATE.md) board, §3, which holds it. Assigned to this
  campaign on 2026-10-18 (README §6.2 C6).* RunRegistryKit closed the defect in `v0.2.0`: `end`
  (its `interrupted-runs` 04) and `begin`'s second-writer refusal (05). It reaches SGK by 01's pin.
  **Next step:** 03 shows the route on the real launcher (README §2.2, 03, acceptance 3), then
  closes the entry on the `handover` board's §4, with `docs/OPEN_ISSUES.md` §1.22, in its commit.

## 4. Resolved issues

None yet.

---

## 5. Dispatch record

### 5.0 Planning (2026-10-18)

The plan was proposed in conversation on 2026-10-18 and accepted as proposed (U1–U3). It was
measured at `79434bf` (README §0.2):
- 45 runs, none `running` or stale;
- 21 sidecars, all tombstones;
- the seven roots at AdaptiveLevin 32, ComputeTargets 700, CosmologyModels 39, LiouvilleGreen 148
  (1 skipped), ObjectFactories 279, RunDrivers 99 and tools 91, all OK;
- a dry run of the moved pin installs `runregistrykit-0.2.0` and nothing else.

Writing 02 against the drivers led to **C3**: the plan's clean-error change to the two drivers is
not taken, and the reason is recorded in README §6.2. The planning commit assigns
`[a3-v2-a-hard-killed-run-blocks-its-store]` here: the `handover` board gains the **Assigned**
line, and the index row moves from `docs/OPEN_ISSUES.md` §1.1 to §1.22.
