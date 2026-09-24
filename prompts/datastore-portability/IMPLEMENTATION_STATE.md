# Datastore portability campaign — implementation state

**Last updated:** 2026-09-24 · **Status: COMPLETE — 1 of 1 prompts landed (01).** Prompt 01
measured what a moved store does on the unfixed tree: it **opens silently and recreates its old
directory with empty shards**. It does not raise. Prompt 01 then made `ShardedPool` fail closed on
a missing shard before any actor exists, record shard paths relative to the primary, and read
legacy absolute records as siblings by name, **never as the absolute path**, through one resolver
shared with the audit tool. A copy of the atol-sweep store, in a new directory and opened through
`main.py`, read its own shards; the originals and the backup are byte-identical afterwards.
This closes `run-registry`'s `[04-sharded-store-paths-are-absolute-and-so-stores-are-not-portable]`
on that board's §4. Two issues are open in §3. One of them, the whole-store rename, is the user's
design decision.

**Campaign:** [`README.md`](README.md) ·
**Code:** `Datastore/shard_paths.py`, `Datastore/SQL/ShardedPool.py`, `tools/shard_key_audit.py` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.12

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**:
> the row is added, moved or deleted, and the count and date in its header are corrected. An
> issue owned by another board is moved to **that** board's §4, and its row is deleted from the
> index. The index is an index: one line per issue, pointing here. Where the two disagree, this
> board is right. See `CLAUDE.md`.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Relative shard paths](01-relative-shard-paths.md) | **P0**–**P4** | Opus 5.5 | ✍️ yes | ✅ 2026-09-24 | *"Record shard paths relative to the primary and refuse missing shards"* | [`logs/01-…`](logs/01-relative-shard-paths.md) |

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| P0 | **MEASUREMENT** | Settle what a moved or renamed store does on the unfixed tree, before changing anything: does it raise, or open? | 01 | ✅ **Done, 2026-09-24.** A throwaway 2-shard store was built through the real constructor in session scratch space, its directory was moved, and the store was reopened. **It opened**, with no exception and no warning. The pool reported the old paths, and **the old directory was recreated** holding two new, empty 37-table shard databases, each with one `version` row. The moved store's own shards sat unused. README §0's reading of the code was right. The `run-registry` issue's "a nonexistent shard raises" was wrong. The throwaway was deleted. |
| P1 | **REMEDY** | Fail closed on a missing shard when opening an existing pool, before any actor is created. | 01 | ✅ **Done, 2026-09-24.** `ShardedPool._check_shard_files()`, called right after `_read_shard_data()`. `SerialPoolBroker` is now created after the branch, so no actor exists when the check runs. The check refuses a shard that is missing, is not a regular file, or is a symlink, and refuses two serials that resolve to one file. The message gives the primary and, for each problem, the serial, the stored record, the resolved path and the reason. On a real throwaway after the change, a deleted shard raised, and nothing was recreated. |
| P2 | **REMEDY** | Write shard paths relative to the primary's directory. | 01 | ✅ **Done, 2026-09-24.** `_write_shard_data` records `relative_to(primary.parent)`, which is the bare name, and checks that the record resolves back to the same file. A new store made through the real constructor recorded `[(0, 'p0store-shard0000.sqlite'), (1, 'p0store-shard0001.sqlite')]`. |
| P3 | **REMEDY** | Read both record forms through one resolver. A legacy absolute record is a sibling by name, never the absolute path. Never rewrite legacy rows. | 01 | ✅ **Done, 2026-09-24.** `Datastore/shard_paths.py`'s `resolve_shard_path`. It returns an absolute `primary.parent / name` and has **no fallback**. It refuses empty, `.`, `..`, separators, NUL, non-strings, and a relative primary. One `!!` line per store when a legacy record is relocated. **Deliberate breakage:** on the unfixed tree the legacy test and both copy tests FAIL, with *A*'s paths where *B*'s were expected. The fail-closed tests ERROR (no guard exists), or FAIL where the unfixed pool picks the original's file. On the fixed tree all pass. §4: the sweep store was copied into a new directory and opened through `main.py --inventory`. All **28** per-table counts match the original's, and a discriminating edit to the copy (7,706 → 7,705) was seen in the inventory. The originals and the backup are identical in mtime, size, SHA-256 and row count. |
| P4 | **REMEDY** | The audit tool resolves the shard it attaches the same way, stays read-only and standalone, and the function is not copied. | 01 | ✅ **Done, 2026-09-24.** `tools/shard_key_audit.py` imports `resolve_shard_path` and `shard_file_problem` from `Datastore.shard_paths`, which is standard library only and outside `Datastore.SQL`, so it pulls in no `ray` or `sqlalchemy`. The tool adds its own repository root to `sys.path`, so it runs from any directory with no `PYTHONPATH`. It now names the path it attaches. Run on the §4 copy, it attached **the copy's** shard 0. The unfixed tool, in the test fixture, reported the original's row count. |

---

## 3. Active and unresolved issues

- **[01-whole-store-rename-is-unsupported]** *(opened 2026-09-24 by prompt 01, per its §5)*:
  **the user's decision.** Shards are recorded by file name, so moving a store's directory and
  renaming its primary on its own both work. Renaming **the shards too** does not. The records
  still carry the old names, P3 reads a record by its name, and P1 then refuses to open, naming
  every shard. This is fail-closed, so nothing is ever silently wrong. It was observed on a real
  store in prompt 01's §4. A copy of the atol-sweep store with all five files renamed to a new
  stem, opened through `main.py`, exited 1 with `Cannot open sharded datastore … shard #0: stored
  record "…/var/datastores/handover-atol-sweep-shard0000.sqlite" resolves to
  "…/var/portability-check/handover-atol-sweep-shard0000.sqlite", which does not exist; …`. The
  copy and the originals were untouched.

  **Two options, neither chosen.**
  **(a) A rename tool** that renames the primary and shards together and rewrites the `shards`
  rows to the new bare names, in one step that can be checked. The format stays self-describing,
  and the only writer of `shards.filename` besides the creator is explicit. The cost is a second
  writer of the table, which this campaign set out to reduce.
  **(b) Derive shard names from the primary's stem** at read time
  (`<stem>-shard{serial:04d}<suffix>`), keeping the table for serials and as a cross-check. A
  whole-store rename then needs no tool. But the table stops being the authority, and a store
  whose shards were ever named otherwise would need a rule for which wins.

  **Impact:** a whole-store rename is refused loudly, not mis-opened. Until the decision, rename
  the primary alone, or move the directory. Measurement:
  [`logs/01-relative-shard-paths.md`](logs/01-relative-shard-paths.md), "The real-store
  demonstration" and "Deviations" item 1. Indexed at `docs/OPEN_ISSUES.md` §1.12.

- **[01-atol-sweep-check-expects-absolute-shard-records]** *(opened 2026-09-24 by prompt 01)*:
  `docs/handover/quadsource_atol_sweep.py` `assert_store_is_self_consistent` (`:589`) compares
  `shards.filename` with `str(p.resolve())` for each expected shard. So it **rejects any store
  created after prompt 01**, whose records are bare names, unless the store has first been
  through `prepare()`'s `UPDATE`, which rewrites the rows to absolute. **Impact:** none in the
  script's own workflow, because `SWEEP_STORE` is only ever made by `prepare()`. The check would
  refuse wrongly if the function were reused on a freshly created store, and its docstring's
  account of `ShardedPool` ("reads them, as absolute paths") is now out of date. With prompt 01
  in, the re-pointing `UPDATE` itself is redundant but not wrong: P3 reads its absolute sibling
  paths as those same siblings. **Next step:** if the script is ever edited again for another
  reason, compare through `Datastore.shard_paths.resolve_shard_path` instead of against literal
  absolute paths, and drop the `UPDATE`. Not done here, because the script is the record of a
  measurement (prompt 01 §2). Indexed at `docs/OPEN_ISSUES.md` §1.12.

---

## 4. Resolved issues

*None opened by this campaign has closed.* The issue this campaign was created to close,
`[04-sharded-store-paths-are-absolute-and-so-stores-are-not-portable]`, is owned by the
[`run-registry`](../run-registry/IMPLEMENTATION_STATE.md) board and is closed on **that** board's
§4, with the P0 finding and the backup of README §0.1 recorded there as a second instance.

---

## 5. Baselines

Measured at `71c4c66` before dispatch, and re-measured at this prompt's commit.

| Suite | At `71c4c66` | After prompt 01 |
|---|---|---|
| `AdaptiveLevin` | 32 OK | 32 OK |
| `ComputeTargets` | 552, 1 failure: `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, the known wall-clock flake | 552, the same 1 failure (0.0607 vs 0.06); it passed when re-run on its own 3 times |
| `CosmologyModels` | 39 OK | 39 OK |
| `Datastore` | 10 OK | **37 OK** (+27) |
| `LiouvilleGreen` | 148 OK (skipped=1) | 148 OK (skipped=1) |
| `RunRegistry` | 38 OK | 38 OK |

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s Datastore/tests -t .
```
