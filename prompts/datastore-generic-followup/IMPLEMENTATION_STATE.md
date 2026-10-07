# datastore-generic-followup campaign — implementation state

**Last updated:** 2026-10-07 · **Status: IN PROGRESS — 2 of 3 prompts landed (01, 02).** Prompts
01–03 are written; 01 and 02 landed on 2026-10-07 and closed the drop issue and the resolve issue
on `datastore-generic`'s board.

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
| 01 | [Refuse a drop that leaves references](01-refuse-a-drop-that-leaves-references.md) | **D1**–**D3** | Opus | ✍️ yes, 2026-10-06 | ✅ 2026-10-07 | `fee1fbf` | [log](logs/01-refuse-a-drop-that-leaves-references.md) |
| 02 | [Resolve an absent parent as `None`](02-resolve-an-absent-parent-as-none.md) | **V1** | Sonnet | ✍️ yes, 2026-10-06 | ✅ 2026-10-07 | this commit | [log](logs/02-resolve-an-absent-parent-as-none.md) |
| 03 | [State each closed-store prefix once](03-state-each-closed-store-prefix-once.md) | **P1**–**P2** | Sonnet | ✍️ yes, 2026-10-06 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not landed · ✅ landed.

The measurement every prompt was written against is [`measure/`](measure/): the rough change of
all three, the scripts, and the verdicts at `6de692a` (README §0.2).

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
(2026-10-06, used for 01).

**Orchestrator review of prompt 01 (2026-10-07).** Dispatched from `3e1e7f0` to one Opus
subagent. Reviewed against its commit `fee1fbf`, with nothing landed after it. **Every check of the
note's §3 passed**, and acceptance 1–3 are met. No stop condition fired, and the agent ran its
suites in the foreground.

*At dispatch.* The note's gate held at `3e1e7f0`:
- the tree was clean, and the only worktree was this checkout;
- `6de692a..3e1e7f0` touches only `docs/OPEN_ISSUES.md` and `prompts/`;
- the registry held 17 runs (15 finished, 2 `unknown`, none `running`), and no Ray process was up;
- the venv imported `scipy.sparse.linalg`;
- the baselines at `33308fe` were README §7's, all OK.

*The checks.*
1. **Scope.** `fee1fbf` touches 9 files, all on the prompt's §6 list. Outside the layer's two
   files, the four tests and the records, the diff is empty.
2. **E1, by running.** `dependent_tables` gives `f2_closure.txt`'s list for each of the ten
   groups, in order. Each group plus its list gives `[]`, and so do all ten together.
   `["GkSource", "NotATable"]` raises `ValueError` naming `NotATable` alone.
3. **E1 and E2, by reading.** The derivation reads only `build_schema` and `inventory_specs`. Its
   docstring states the rule, that it is structural, and why each kind of reference sees what the
   other does not, naming no project table. The refusal is the line after
   `_refuse_undeclared_drop_tables()`. For `gk-source-policy-records` alone it raises:
   `Cannot open sharded datastore "<primary>": cannot drop ['GkSourcePolicy'] alone: the rows of
   ['GkSourcePolicyData', 'QuadSourceIntegral', 'QuadSourceIntegral_tags'] would name rows that
   are gone, so they must be dropped too. Nothing was opened`.
4. **E3, by `ast`** against the pre-images, docstrings stripped. `ShardedPool.py` adds
   `_refuse_drop_that_leaves_references`, and `__init__` differs by the one call; nothing else
   differs. `schema.py` adds `dependent_tables` and the import, and every existing function dumps
   identically. In `test_reconcile_at_open.py` only `test_an_interrupted_store` differs, by its drop
   and its docstring.
5. **The tests, by reading.** Test 1's lists are literals. Tests 4–6 assert the structural fact
   before the dependent one. Tests 8–10 also assert that no actor or engine was made, test 8 that
   the checksums are unchanged, test 9 that the directory stays empty, and test 11 that rows
   existed before it asserts they are gone.
6. **E5, the pins.** Each of (i)–(vi), replayed in `bash` from the log, fails the tests the prompt
   names: (i) 5 groups of test 1 and test 4; (ii) and (iii) 8 groups of test 1, and tests 5 and 6
   respectively; (iv) tests 8 and 9; (v) test 10; (vi) the re-fixtured test and its inherited copy,
   both subtests. Each reverts with `git apply -R`, and `git status` is clean.
7. **E4, the probe** (the scripts of `datastore-generic` measure-09, with a drop argument). A
   `build_full_store` was built at the parent's export. Fresh copies were opened read-write through
   the stand-in pool at the parent and at `fee1fbf`, with (a) no drop, twice, (b) every group, and
   (c) test 11's accepted drop. At the two trees, the printed lines agree once the store path and
   the INTEGRITY WARNING's wall-clock stamp are normalised. So do the reconciliation and prune
   records and the version, and the post-close dumps are byte-identical (207, 112 and 193 rows).
   (d) `gk-source-policy-records` alone opened at the parent, leaving 203 rows with
   `GkSourcePolicy` emptied, and is refused at `fee1fbf` with every file unchanged.
8. **E6, the records.** The drop issue is in `datastore-generic`'s §4 with a dated "Closed" line
   naming `dependent_tables`, the refusal, this campaign and §0.2's undercount. It is gone from its
   §3.1 and from the index, which counts 106 rows, matching the header. README §4 says 01 shipped
   as named.
9. **Suites**, serially at `fee1fbf`, load 12–148 (another session's load peaked during the
   Datastore groups):
   - Datastore 732 (158 + 183 + 200 + 191), OK; RunRegistry 280, OK; ComputeTargets 693, OK;
   - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).

   No flake appeared. `black --check` is clean on the four Python files.

*The agent's deviations.* Seven, all IMPLEMENTATION CHOICE but `black`'s docstring reflow
(STRUCTURALLY REQUIRED). The orchestrator accepts each. The two of substance:
- `schema.py`'s module docstring states its new import, since otherwise it would name a false
  import list;
- the message lists `sorted(set(drop_tables))`, so a repeated name appears once.

*Residue, fixed in the follow-up commit that records this review.*
- "this commit" → `fee1fbf` in §1, §2, §4, §5.2, the log and `datastore-generic`'s §4.
- README §2's status column for 01 said "written". It now says landed, with the SHA.

*Handed on.* Nothing to 02, which touches no file 01 touched. The agent's observation that
`test_layer_registry`'s `DROP = ["GkSource", "NotATable"]` now passes only because the
undeclared-table refusal comes first is pinned by test 10 and breakage (v), and is not an issue.

---

## 2. Items

| Item | Prompt | What | Issue | Status |
|---|---|---|---|---|
| **D1** | 01 | `schema.dependent_tables(dropped, factories)`: the tables naming a dropped one, by foreign key or declared parent, transitively, in registry order | `[06-dropping-one-group-…]` | ✅ `fee1fbf` |
| **D2** | 01 | `ShardedPool._refuse_drop_that_leaves_references()`, after the undeclared-table refusal, before the primary is opened | `[06-dropping-one-group-…]` | ✅ `fee1fbf` |
| **D3** | 01 | `test_drop_refuses_dangling_references` (11 tests); `TestPruningAfterRepair.test_an_interrupted_store` re-fixtured | — | ✅ `fee1fbf` |
| **V1** | 02 | `StoreInventory.resolve` gives `None` for a `None` key parent; two tests in `TestResolve` | `[08-resolve-shows-…]` | ✅ this commit |
| **P1** | 03 | `_read_closed_store` raises the bare reason and takes no `verb` | `[04-closed-store-refusals-…]` | ⬜ |
| **P2** | 03 | `open_read_only` wraps it with its own prefix; `test_closed_store_refusals` (5 tests); one `test_store_reader` test changed | `[04-closed-store-refusals-…]` | ⬜ |

---

## 3. Active and unresolved issues

None opened by this campaign. The one it still owns, `[04-closed-store-refusals-repeat-their-prefix]`,
is on `datastore-generic`'s board, §3.1.

## 4. Resolved issues

None opened by this campaign. Closed on `datastore-generic`'s board, §4, which holds the record:
- `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`, by prompt 01 (2026-10-07, `fee1fbf`  commit);
- `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`, by prompt 02 (2026-10-07, this commit).

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

### 5.2 After prompt 01 (2026-10-07, `fee1fbf`)

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

### 5.3 After prompt 02 (2026-10-07, this commit)

Each suite run in the foreground, serially, with README §7's command, at load 8–33. Every suite
OK on the first run; no flake.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 693 OK |
| CosmologyModels | 39 OK |
| Datastore | 734 OK (158 + 185 + 200 + 191) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 280 OK |

Datastore's `[g-o]` group is 183 + 2: the two new methods of
`test_inventory_declarations.TestResolve`. Every other count is §5.2's.
