# dk-v050-adoption campaign — implementation state

**Last updated:** 2026-10-22 · **Status: 01 landed, awaiting review — 1 of 1.** U1 and U2 ruled by the
user.

The plan is [`README.md`](README.md), written on 2026-10-22 at `031bcb4` from DatastoreKit's
`v0.5.0` addendum for SGK and RunRegistryKit's amended guard recipe.

The campaign moves SGK's `datastorekit` pin from `v0.4.0` to `v0.5.0`, to keep current. The release
adds `InventorySpec.extra_values`, which SGK does not use; the one consequence for SGK is a
condition in its spec guard, so that an empty `extra_values` does not move every spec's digest.

**Campaign:** [`README.md`](README.md) ·
**Code:** `requirements.txt`, `CLAUDE.md`, `ObjectFactories/tests/test_fingerprint_format_guard.py` ·
**Verification:** none of its own (README §0.1); `measure/out/` holds 01's baselines ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) (no section yet: no issue is open)

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row is added, moved or deleted, and the count and date in its header are corrected. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-22 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** — whether to move | move to `v0.5.0`, to keep current |
| **U2** — the shape | one prompt; the review closes the campaign |

The orchestrator's choices of method, C1–C7, are in README §6.2. The one left for the user, a guard
that does not break on the next field, is README §6.3.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Pin `datastorekit` at `v0.5.0`](01-pin-datastorekit-at-v0.5.0.md) | the pin; the guard's condition and its test; `CLAUDE.md`'s pin; the baselines | Sonnet | ✍️ 2026-10-22 | ✅ 2026-10-22 | this commit | [log](logs/01-pin-datastorekit-at-v0.5.0.md) |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Written with the plan (README §2.1).** There is no later prompt; the review of 01 closes the
campaign (README §2.2).

**Orchestrator notes** are written just before dispatch, under `orchestrator/`:
[`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-22, for 01). The planning commit
is `99199ca`.

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **K1** | 01 | `requirements.txt:17` moved to `v0.5.0`; the venv installed from it; `CLAUDE.md`'s pin | ✅ |
| **G1** | 01 | the guard's `plain` leaves out an empty `extra_values`; its docstrings; `test_an_empty_extra_values_is_left_out`; `SPEC_DIGESTS` and `FINGERPRINT_FORMAT` unchanged | ✅ |
| **B1** | 01 | the baselines: `pip freeze`, provenance, the guard unedited and edited, the seven roots | ✅ |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| **E** | DatastoreKit has tagged `v0.5.0` with an SGK addendum, and RunRegistryKit's recipe gives the condition | — | ✅ holds (2026-10-22): tag object `6a6809d5…` on `f47b9a25…`, on GitHub; the addendum in `docs/adoption/secondarygwkit.md` at the tag; RunRegistryKit `70ee1e8`, pushed |
| **FP** | the golden fingerprint is reproduced unchanged through `v0.5.0` | 01 | ✅ holds (2026-10-22): `test_store_fingerprint_golden`, 6 tests, OK under the installed `v0.5.0` |

There is no gate on a DatastoreKit board for this move: the `extra-values` campaign's U5 lets SGK
stay on `v0.4.0` or move.

## 3. Active and unresolved issues

None.

## 4. Resolved issues

None.

## 5. Dispatch record

### 5.0 Planning (2026-10-22)

The plan was proposed in conversation on 2026-10-22 and accepted as proposed (U1, U2). It was
measured at `031bcb4` (README §0.2):
- 51 runs, none `running` or stale; the rehearsal lock free; no Ray process of this repository;
- 24 sidecars, all tombstones: no primary on disk;
- a dry run of the moved pin installs `datastorekit-0.5.0` and nothing else;
- under `v0.5.0`, in a scratch overlay, the guard fails on every class as it stands and passes
  with the condition; the condition changes no digest under `v0.4.0`;
- the seven roots under `v0.5.0` pass but for the guard, the golden-fingerprint tests included.

Writing the plan corrected one thing proposed in conversation: a read-only fingerprint of a real
store through the new pin is not possible, since no store's primary is on disk. The golden test,
which builds a real store, is the check (README §0.1, C7).

### 5.1 After prompt 01 (2026-10-22)

Landed at this commit, on `62de2c4`. K1, G1 and B1 pass; no stop condition fired. Log:
[`logs/01-pin-datastorekit-at-v0.5.0.md`](logs/01-pin-datastorekit-at-v0.5.0.md); measurements:
`measure/out/01-{pip,guard,roots}.txt`.
- **The pin.** `requirements.txt:17` is `@v0.5.0`. The dry run said `Would install datastorekit-0.5.0`
  and nothing else. The venv holds `datastorekit` 0.5.0 at `f47b9a25…`, `requested_revision`
  `v0.5.0`, not editable; `raypoolkit` and `runregistrykit` unchanged; `pip check` clean; `pip
  freeze` differs by the one line. `InventorySpec` has seven fields, `ENCODING_FORMAT` is `1`.
- **The guard.** Unedited under `v0.5.0` it failed with "spec changed" for 21 classes and no other
  line; with the condition it passes against `SPEC_DIGESTS[6]` as it stands. `FINGERPRINT_FORMAT`,
  `SPEC_DIGESTS` and `config/fingerprint.py` are untouched. M1 and M2 each fail the tests the prompt
  names, and were reverted.
- **The roots**, before and after, serially: AdaptiveLevin 32, ComputeTargets 700, CosmologyModels
  39, LiouvilleGreen 148 (1 skipped), ObjectFactories 279 then 280, RunDrivers 115, tools 91, all
  OK; no known flake fired. The six golden-fingerprint tests pass.
- **The addendum's claims held:** 21 `InventorySpec(...)` calls, none passing `extra_values`; no
  other file reads `extra_values`, `extra_value_counts`, `as_json` or walks `dataclasses.fields`.
- **No issue opened or closed;** `docs/OPEN_ISSUES.md` stays at 100.
- **The venv is ahead of `git revert`:** reverting 01 leaves `v0.5.0` installed (README §4.1).
