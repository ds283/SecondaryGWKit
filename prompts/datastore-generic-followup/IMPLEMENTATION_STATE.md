# datastore-generic-followup campaign — implementation state

**Last updated:** 2026-10-07 · **Status: IN PROGRESS — 1 of 3 prompts landed (01).** Prompts
01–03 are written; 01 landed on 2026-10-07 and closed the drop issue on `datastore-generic`'s
board.

The campaign was opened on 2026-10-06 to carry out the user's decisions on the three issues
`datastore-generic` left open when it closed (README §0, §6.1). The issues stay on
`datastore-generic`'s board, which holds their measurements, and close there; this board records
the work.

**Campaign:** [`README.md`](README.md) ·
**Code:** `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/schema.py`, `Datastore/store_inventory.py`,
`Datastore/store_reader.py` ·
**Issues:** [`../datastore-generic/IMPLEMENTATION_STATE.md`](../datastore-generic/IMPLEMENTATION_STATE.md)
§3.1 · **Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.19

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, or out of `datastore-generic`'s board for an issue this campaign owns,
> [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-06 (README §6.1):**

| Decision | Decided |
|---|---|
| **1** — a drop that would leave references | refused, naming the tables; a reference is a foreign key or a declared `inventory_spec` parent, followed transitively |
| **2** — an absent nullable parent in `resolve()` | rendered as `None`, as a parent-set member is |
| **3** — the doubled closed-store prefix | removed in `_read_closed_store`; each caller states its own |

The orchestrator's choices of method are in README §6.2.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Refuse a drop that leaves references](01-refuse-a-drop-that-leaves-references.md) | **D1**–**D3** | Opus | ✍️ yes, 2026-10-06 | ✅ 2026-10-07 | this commit | [log](logs/01-refuse-a-drop-that-leaves-references.md) |
| 02 | [Resolve an absent parent as `None`](02-resolve-an-absent-parent-as-none.md) | **V1** | Sonnet | ✍️ yes, 2026-10-06 | ⬜ | — | — |
| 03 | [State each closed-store prefix once](03-state-each-closed-store-prefix-once.md) | **P1**–**P2** | Sonnet | ✍️ yes, 2026-10-06 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not landed · ✅ landed.

The measurement every prompt was written against is [`measure/`](measure/): the rough change of
all three, the scripts, and the verdicts at `6de692a` (README §0.2).

---

## 2. Items

| Item | Prompt | What | Issue | Status |
|---|---|---|---|---|
| **D1** | 01 | `schema.dependent_tables(dropped, factories)`: the tables naming a dropped one, by foreign key or declared parent, transitively, in registry order | `[06-dropping-one-group-…]` | ✅ this commit |
| **D2** | 01 | `ShardedPool._refuse_drop_that_leaves_references()`, after the undeclared-table refusal, before the primary is opened | `[06-dropping-one-group-…]` | ✅ this commit |
| **D3** | 01 | `test_drop_refuses_dangling_references` (11 tests); `TestPruningAfterRepair.test_an_interrupted_store` re-fixtured | — | ✅ this commit |
| **V1** | 02 | `StoreInventory.resolve` gives `None` for a `None` key parent; two tests in `TestResolve` | `[08-resolve-shows-…]` | ⬜ |
| **P1** | 03 | `_read_closed_store` raises the bare reason and takes no `verb` | `[04-closed-store-refusals-…]` | ⬜ |
| **P2** | 03 | `open_read_only` wraps it with its own prefix; `test_closed_store_refusals` (5 tests); one `test_store_reader` test changed | `[04-closed-store-refusals-…]` | ⬜ |

---

## 3. Active and unresolved issues

None opened by this campaign. The two it still owns are on `datastore-generic`'s board, §3.1.

## 4. Resolved issues

None opened by this campaign. Closed on `datastore-generic`'s board, §4, which holds the record:
- `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`, by prompt 01 (2026-10-07, this
  commit).

## 5. Baselines

### 5.1 At the plan (2026-10-06, `6de692a`)

Taken at `53d4e18` (datastore-generic 09's review), with no code changed since. The rough change
of all three prompts was measured against an export of `6de692a` (README §0.2).

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 693 OK |
| CosmologyModels | 39 OK |
| Datastore | 721 OK (147 + 183 + 200 + 191) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 280 OK |

Expected after each prompt: Datastore 732 (01), 734 (02), 739 (03); every other suite unchanged.

### 5.2 After prompt 01 (2026-10-07, this commit)

Each suite run in the foreground, serially, with README §7's command, at load 5–11. Every suite
OK on the first run; no flake.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 693 OK |
| CosmologyModels | 39 OK |
| Datastore | 732 OK (158 + 183 + 200 + 191) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 280 OK |

Datastore's `[a-f]` group is 147 + 11: the new
`test_drop_refuses_dangling_references` module. Every other count is §5.1's.
