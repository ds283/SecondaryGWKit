# Implementation state — backport of shared Ray/Datastore modules

**Campaign:** [`README.md`](README.md) · **Source audit:** [`docs/backport-modules-audit.md`](../../docs/backport-modules-audit.md)
**Baseline commit:** `79f0360` (`main`, clean)
**Last updated:** 2026-09-04 — after prompt 05, plus an out-of-sequence fix for
`[02-shard-config-reader]` (see §4)

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status, fill in the commit SHA and the log link, and add or clear entries in
> §3 (Active issues). Do not edit rows other than your own except to close an issue you resolved.

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Backport proper (audit items)

| # | Prompt | Items | Status | Commit | Log |
|---|---|---|---|---|---|
| 01 | [Shard-key persistence](01-shard-key-persistence.md) | B1, B5 | ✅ | `2610abe` | [log](logs/01-shard-key-persistence.md) |
| 02 | [Shard-config reader](02-shard-config-reader.md) | B2, D2 | ✅ | `9206704` | [log](logs/02-shard-config-reader.md) |
| 03 | [Robustness fixes](03-robustness-fixes.md) | F6, B4, D1, D3, D4, F3 | ✅ | `e81b145` | [log](logs/03-robustness-fixes.md) |
| 04 | [`read_table` service](04-read-table-service.md) | B3 | ✅ | *(SHA intentionally not embedded — see §5 note 11)* | [log](logs/04-read-table-service.md) |
| 05 | [`persist_handler` split](05-persist-handler-split.md) | E1 | ✅ | *(SHA intentionally not embedded — see §5 note 11)* | [log](logs/05-persist-handler-split.md) |

### `inventory()` sub-campaign (F2)

| # | Prompt | Items | Status | Commit | Log |
|---|---|---|---|---|---|
| 06 | [Inventory plumbing](06-inventory-plumbing.md) | F2a | ⬜ | — | — |
| 07 | [Replicated factories](07-inventory-replicated-factories.md) | F2b | ⬜ | — | — |
| 08 | [Sharded factories + merge config](08-inventory-sharded-factories.md) | F2c | ⬜ | — | — |
| 09 | [Inventory reporting](09-inventory-reporting.md) | F2d | ⬜ | — | — |

### Close-out

| # | Prompt | Items | Status | Commit | Log |
|---|---|---|---|---|---|
| 10 | [Verification pass](10-verification.md) | audit §8 + F2 | ⬜ | — | — |

**Progress:** 5 / 10 complete.

---

## 2. Item-level tracking

Traceability from the audit's finding IDs to the prompt that discharges them.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| B1 | **Critical** | `_assign_shard_keys` inserts `key_id` into a `key_serial` PK column → silent shard misrouting after restart | 01 | ✅ |
| B5 | Low–Medium | `_assign_shard_keys` does not dedup within a batch (hard prerequisite for B1) | 01 | ✅ |
| B2 | High | `_read_shard_data` reads `row.key_attr` from a `key_type`-only select → `AttributeError` on every reopen | 02 | ✅ |
| D2 | Latent | `raise print(f"…")` raises `TypeError`, and the branch is unreachable | 02 | ✅ (kept & repaired — see log; branch found reachable, not dead) |
| F6 | Low | Unguarded empty-list insert into `sharded_tables` | 03 | ✅ (`replicated_tables` insert also guarded — implementation choice) |
| B4 | Low | `RayWorkPool` rejects a task builder returning `None` | 03 | ✅ |
| D1 | Latent | `ShardedPool.__init__` error path references non-existent `self._db_file` | 03 | ✅ |
| D3 | Latent | `_default_serial_batch_size[table]` raises `KeyError` for unlisted classes | 03 | ✅ (fallback = 500, `ClientPool`'s own default) |
| D4 | Latent | `_last_num_available_complete` assigned from `_num_store_complete` | 03 | ✅ |
| F3 | Optional | `object_get("version", …)` by name, dropping the `MetadataConcepts` coupling | 03 | ✅ (taken, not skipped — both `Datastore.py` and `ShardedPool.py`) |
| B3 | High | `read_table_config` method generation is broken in four ways; replace with `read_table()` | 04 | ✅ |
| E1 | Feature | `store_handler` / `persist_handler` split in `RayWorkPool` — **confirmed wanted** | 05 | ✅ |
| F2a | Feature | `inventory()` plumbing: `Datastore`, `ShardedPool`, `_merge_queue`, numeric merge policies | 06 | ⬜ |
| F2b | Feature | `inventory()` on the 13 replicated-table factories | 07 | ⬜ |
| F2c | Feature | `inventory()` on the 15 sharded-table factories + `inventory_config` | 08 | ⬜ |
| F2d | Feature | Inventory reporting entry point | 09 | ⬜ |

**Explicitly not backported** (audit §5 — do not schedule these): X1, X2, X3, X4.
**Out of scope:** F4 (licence headers — repo-wide or not at all; see `README.md` §6).

---

## 3. Active and unresolved issues

- **[01-shard-key-persistence]** *(opened by prompt 01, 2026-09-03)* — Audit §8 checklist items
  1–2 (a fresh `SGWK` datastore has no `_assign_shard_keys MISMATCH` lines and every
  `shard_keys.key_serial` equals its `wavenumber.serial`; a stopped-and-resumed run finds all
  previously-written records) need a real Ray pipeline run and were not exercised. The B1/B5 code
  fix itself was verified against synthetic sqlite fixtures built for this prompt (see
  [log](logs/01-shard-key-persistence.md) §Verification), not against a genuine `SGWK` pipeline
  run — no current-schema `SGWK` datastore exists in the tree to test against. **Impact:**
  behavioural confirmation of the fix on a real pipeline is outstanding. **Next step:** prompt 10
  should run a real (or minimal) `SGWK` pipeline against a fresh datastore, inspect for `MISMATCH`
  output, and do a stop/resume cycle.

- **[04-read-table-service]** *(opened by prompt 04, 2026-09-04)* — Audit §8 checklist items 5–6
  (each `extract_*.py` script constructs its `ShardedPool` and returns the same wavenumber/redshift
  arrays as before, under a real multi-shard Ray run) need a live Ray cluster with multiple shard
  actors and were not exercised. The `read_table` implementation itself was verified two other ways
  instead (see [log](logs/04-read-table-service.md) §Verification items 5–6): a real `Datastore`
  instance (not Ray-wrapped) against a fresh SQLite file, read via `read_table` and cross-checked
  against a direct SQL query of the same file; and `ShardedPool.read_table`'s two required negative
  cases (sharded-class rejection, unconfigured-class rejection) against a minimally-constructed
  `ShardedPool` object. Neither exercises the actual multi-shard random-selection dispatch
  (`shard.read_table.remote(...)`) against live shard actors. **Impact:** behavioural confirmation of
  `read_table` end-to-end, across real shards, under real Ray, is outstanding — same class of gap as
  the `[01-shard-key-persistence]` issue above. **Next step:** prompt 10 should run at least one
  `extract_*.py` script (or a minimal equivalent) against a fresh multi-shard `ShardedPool`, and
  confirm `read_table("wavenumber", ...)`/`read_table("redshift", ...)` return correct data by
  cross-checking against direct SQL on the shard chosen.

- **[05-persist-handler-split]** *(opened by prompt 05, 2026-09-04)* — Audit §8 checklist item 7
  (a real driver run exercising the `store_handler`/`persist_handler` split end-to-end, confirming
  results are still stored exactly as before) needs a live Ray cluster and was not exercised. The
  split itself was verified statically and with a synthetic in-process harness instead (see
  [log](logs/05-persist-handler-split.md) §Verification items 1-5): grep-verified 35/35
  `store_handler=None`/`persist_handler=None` pairings across all 7 call-site files, 45 unchanged
  `RayWorkPool(` constructions, `py_compile` and `black --check` clean on all 8 touched files, and a
  throwaway harness exercising `_default_store_handler`/`_default_persist_handler` call order and
  both constructor-validation branches directly against the real `RayTools.RayWorkPool` module.
  Neither exercises an actual Ray task graph. **Impact:** behavioural confirmation that a real
  `RayWorkPool` run still stores results correctly under the new two-hook split is outstanding — same
  class of gap as `[01-shard-key-persistence]` and `[04-read-table-service]` above. **Next step:**
  prompt 10 should run at least one real (or minimal) Ray-backed `RayWorkPool` with default handlers
  and confirm results are stored identically to a pre-split baseline (or, if no baseline is
  practical, confirm the stored objects are correct against a direct datastore query).

> Add an entry here whenever a prompt finishes with something unresolved: a verification step that
> could not be run, an assumption that could not be confirmed, a deviation that a later prompt has
> to work around. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them.

---

## 4. Resolved issues

- **[02-shard-config-reader]** *(opened by prompt 02, 2026-09-03; resolved 2026-09-04, outside the
  prompt sequence, at the user's direct request)* — `LiouvilleGreen/WKBtools.py:6` did
  `from defaults import DEFAULT_ABS_TOLERANCE`, but the top-level `defaults.py` module was deleted in
  `a2bd966` (2025-12-15 — the same commit that introduced B2) and its contents moved to
  `config/defaults.py`. This broke any real (non-stubbed) import of `Datastore.SQL.ShardedPool` and
  everything downstream, and had been worked around with a `sys.modules` stub in throwaway
  verification harnesses across prompts 02, 03 and 04. **Fix:** one-line import correction to
  `from config.defaults import DEFAULT_ABS_TOLERANCE`; confirmed `import Datastore.SQL.ShardedPool`
  now succeeds with no stub required. Not one of this campaign's tracked audit items
  (B1–B5/D1–D4/F2/F3/F6/E1) — fixed as a standalone commit rather than folded into any prompt's
  commit, since it did not originate from this campaign's audit and touches a file (`WKBtools.py`)
  outside every prompt's stated file list.

- **[commit-sha-links-stale]** *(opened by prompt 03, 2026-09-03; resolved by prompt 04, 2026-09-04)*
  — The commit SHAs recorded for prompts 01, 02 and 03 in §1 (`fbc3a90`, `34380ba`, `3e8a984`) and in
  their own logs were **not reachable from the branch tip**: `git merge-base --is-ancestor <sha> HEAD`
  failed for all three. The commits actually on the branch, with identical messages and diffs, are
  `2610abe` (prompt 01), `9206704` (prompt 02) and `e81b145` (prompt 03). This was a structural
  consequence of the self-referential requirement in the old wording of README.md §5.1 — the
  log/status board had to record the commit's own SHA, but editing the file to embed a guessed SHA
  and then amending to correct it necessarily produces a *new* SHA, which cannot itself be embedded
  without repeating the problem. Each of the first three prompts amended once to inject a best-guess
  SHA and stopped, leaving the recorded value one amend behind the true final commit — prompt 03
  diagnosed this for 01/02 but then reproduced it a third time for its own commit rather than
  avoiding it. **Fix:** (a) corrected all three `Commit` columns in §1 and both affected log headers
  (01, 02) plus 03's own log header to the branch-reachable SHAs above; (b) prompt 04 stops writing a
  guessed self-referential SHA at all — see §5 note 11 for the convention adopted from prompt 04
  onward, which removes the underlying cause rather than re-chasing convergence each time.

---

## 5. Standing notes for all implementers

Established by the planning pass against the working tree at `79f0360`. These do not need
re-deriving.

1. **No `SGWK` datastore in the tree uses the current schema.** `test-qcd-db.sqlite` has
   `shard_keys.wavenumber_serial` and lacks `shard_key_config`, `replicated_tables` and
   `sharded_tables` — it predates commit `a2bd966` (2025-12-15). Consequences: (a) it corroborates
   that the B2/B3 code paths have never been exercised, so those findings need no re-derivation;
   (b) it cannot be used as a fixture for "reopen an existing pool" tests — a fresh datastore must
   be created first; (c) any pre-existing datastore must be rebuilt regardless of B1, purely on
   schema grounds.
2. **There is no test infrastructure for `Datastore`, `ShardedPool` or `RayWorkPool`.** The only
   pytest suites are `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/`. Verification in this
   campaign is by static check plus purpose-written scripts; be explicit in your log about what you
   actually executed versus what you only reasoned about.
   **`ray` (and this project's other runtime dependencies) are only installed in the repo's own
   `./venv`**, not in the ambient `python3` — use `./venv/bin/python3` for any throwaway harness
   that imports `Datastore.SQL.*` or `RayTools.*` (confirmed while verifying prompt 03's B4 fix).
3. **Only two factories expose `read_table`:** `ObjectFactories/redshift.py:76`
   (`conn, table, tables, is_source, is_response, model_proxy` → `tables_arg: True`) and
   `ObjectFactories/wavenumber.py:90` (`conn, table, units, is_source, is_response` →
   `tables_arg: False`). Both are `@staticmethod`s on the factory class.
4. **Verified call-site counts** at `79f0360`: 35 `store_handler=None` across 7 files; 45
   `RayWorkPool(` constructions; 17 `read_wavenumber_table`/`read_redshift_table` calls across 6
   `extract_*.py` scripts. If your grep disagrees, something landed between then and now — stop and
   reconcile before editing.

### Notes specific to the F2 sub-campaign (prompts 06–09)

5. **Factories are classes with `@staticmethod`s, not instances.** `SGWK` registers
   `sqla_x_factory` (the class); `CPBH`/`SI` register `sqla_x_factory()` (an instance). So every
   upstream `inventory` method takes `self` and **cannot be copied across unaltered** — `SGWK`'s
   signature is `@staticmethod def inventory(conn, table, tables, *args, **kwargs)`. This is the
   most likely silent failure in prompts 07 and 08: a stray `self` binds `conn` and the error
   surfaces far from its cause.
6. **`inventory` must stay optional.** `ObjectFactories/base.py` is an ABC whose members are all
   `@staticmethod` **and** `@abstractmethod`. Making `inventory` abstract would oblige all ~37
   factory classes to implement one, including the tag-association factories where it is
   meaningless. Discover it with `hasattr`, as upstream does. **Do not modify `base.py`.**
7. **Class counts:** 13 replicated (`version`, `store_tag`, `redshift`, `wavenumber`,
   `wavenumber_exit_time`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`,
   `BackgroundModel`, `BackgroundModelValue`, `GkSourcePolicy`, `QuadSourcePolicy`) and 15 sharded
   (`TkNumericIntegration`, `TkNumericValue`, `TkWKBIntegration`, `TkWKBValue`, `QuadSource`,
   `QuadSourceValue`, `GkNumericIntegration`, `GkNumericValue`, `GkWKBIntegration`, `GkWKBValue`,
   `GkSource`, `GkSourceValue`, `GkSourcePolicyData`, `QuadSourceIntegral`, `OneLoopIntegral`).
   Note the audit says 14 replicated; the config lists 13.
8. **Tables with no `timestamp` column** (registered `"timestamp": False`): `version`,
   `BackgroundModelValue`, and all six sharded value tables (`TkNumericValue`, `TkWKBValue`,
   `QuadSourceValue`, `GkNumericValue`, `GkWKBValue`, `GkSourceValue`). These can only be counted,
   which is why prompt 06 must add a numeric `"sum"` merge policy that upstream lacks.
9. **Tables with a `validated` column:** `BackgroundModel`, `GkNumericIntegration`, `GkSource`,
   `GkWKBIntegration`, `QuadSource`, `TkNumericIntegration`, `TkWKBIntegration`. These take the
   labelled validated/unvalidated inventory shape. `OneLoopIntegral` mentions `validated` once —
   prompt 08 must check what that occurrence actually is rather than assuming.
10. **Zero factories expose `inventory` today.** Prompts 07 and 08 write all ~28 from scratch;
    there is nothing to reconcile against in this tree.
11. **Do not embed a commit's own SHA in that same commit.** Prompts 01–03 each tried to record
    their own commit hash in this file and the prompt's log (per README.md §5.1), then amended once
    to fix a guessed value — but an amend produces a new SHA, so the recorded value was always one
    amend behind the real branch tip (see the resolved `[commit-sha-links-stale]` issue in §4). From
    prompt 04 onward: do not write a guessed SHA into `Commit` cells or log headers. Either leave a
    placeholder that says the SHA is intentionally omitted and why (as prompt 04's own row does), or
    — if a later prompt's edit touches this file anyway — fill in *previous* prompts' real SHAs
    (verified reachable with `git merge-base --is-ancestor <sha> HEAD`) while leaving the current
    prompt's own row/log without one. The gap is harmless: `git log --oneline -- prompts/backport-modules/`
    always recovers the true mapping.
