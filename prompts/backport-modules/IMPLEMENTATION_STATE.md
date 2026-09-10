# Implementation state — backport of shared Ray/Datastore modules

**Campaign:** [`README.md`](README.md) · **Source audit:** [`docs/backport-modules-audit.md`](../../docs/backport-modules-audit.md)
**Baseline commit:** `79f0360` (`main`, clean)
**Last updated:** 2026-09-04 — after prompt 10. **Campaign complete: 10/10 prompts done.**

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status, fill in the commit SHA and the log link, and add or clear entries in
> §3 (Active issues). Do not edit rows other than your own except to close an issue you resolved. **Any change to §3 or §4 must also update the project-wide index
> [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit** (see `CLAUDE.md`).

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
| 06 | [Inventory plumbing](06-inventory-plumbing.md) | F2a | ✅ | *(SHA intentionally not embedded — see §5 note 11)* | [log](logs/06-inventory-plumbing.md) |
| 07 | [Replicated factories](07-inventory-replicated-factories.md) | F2b | ✅ | *(SHA intentionally not embedded — see §5 note 11)* | [log](logs/07-inventory-replicated-factories.md) |
| 08 | [Sharded factories + merge config](08-inventory-sharded-factories.md) | F2c | ✅ | *(SHA intentionally not embedded — see §5 note 11)* | [log](logs/08-inventory-sharded-factories.md) |
| 09 | [Inventory reporting](09-inventory-reporting.md) | F2d | ✅ | *(SHA intentionally not embedded — see §5 note 11)* | [log](logs/09-inventory-reporting.md) |

### Close-out

| # | Prompt | Items | Status | Commit | Log |
|---|---|---|---|---|---|
| 10 | [Verification pass](10-verification.md) | audit §8 + F2 | ✅ | *(SHA intentionally not embedded — see §5 note 11)* | [log](logs/10-verification.md) |

**Progress:** 10 / 10 complete. **Campaign complete.** See
[`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md) for the full
verification record: 8 of 12 audit-checklist items now have live confirmation against a real,
multi-shard, Ray-actor-backed pool; the remaining 3 need actual physics compute and are handed to the
user with a cost estimate.

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
| F2a | Feature | `inventory()` plumbing: `Datastore`, `ShardedPool`, `_merge_queue`, numeric merge policies | 06 | ✅ |
| F2b | Feature | `inventory()` on the 13 replicated-table factories | 07 | ✅ |
| F2c | Feature | `inventory()` on the 15 sharded-table factories + `inventory_config` | 08 | ✅ |
| F2d | Feature | Inventory reporting entry point | 09 | ✅ |

**Explicitly not backported** (audit §5 — do not schedule these): X1, X2, X3, X4.
**Out of scope:** F4 (licence headers — repo-wide or not at all; see `README.md` §6).

---

## 3. Active and unresolved issues

Three of the four issues open before prompt 10 are now resolved (moved to §4) or narrowed. See
[`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md) for the full
account of what prompt 10 ran and how.

- **[04-read-table-service]** *(opened by prompt 04, 2026-09-04; narrowed by prompt 10, 2026-09-04)*
  — Audit §8 checklist items 5–6. **Item 5 is now closed**: prompt 10 ran
  `pool.read_table("GkSource")`/`pool.read_table("LambdaCDM")` against a real, live, multi-shard
  `ShardedPool` (not the `__new__`-constructed stand-in prompt 04's own verification used), both
  raising the intended `RuntimeError`s — see
  [`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md) §4 check 5.
  **Item 6 remains open**: each `extract_*.py` script constructing its pool and returning the correct
  wavenumber/redshift arrays under a real multi-shard Ray run, against data produced by an actual
  compute pipeline, was not exercised — this needs a populated datastore, which needs real physics
  compute (`main.py`'s wavenumber sample size is hardcoded at 50+50 with no CLI override to shrink
  it). **Impact:** end-to-end confirmation of the six extract scripts under a live multi-shard run is
  outstanding. **Next step:** see
  [`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md) §5 for a
  concrete, scoped-down approach (a small driver script outside production code, 5–10 wavenumbers,
  one queue enabled) and a rough time estimate.

- **[05-persist-handler-split]** *(opened by prompt 05, 2026-09-04)* — Audit §8 checklist item 7
  (a real driver run exercising the `store_handler`/`persist_handler` split end-to-end, confirming
  results are still stored exactly as before) needs a live Ray cluster and was not exercised. The
  split itself was verified statically and with a synthetic in-process harness instead (see
  [log](logs/05-persist-handler-split.md) §Verification items 1-5). Prompt 10 separately re-confirmed
  the *related* B4 checklist item (a task builder returning `None`) live, against a real pool (see
  [`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md) §4 check 6)
  — but that is not this issue's ask. **Impact:** behavioural confirmation that a real `RayWorkPool`
  run still stores a genuine compute result correctly under the new two-hook split is outstanding —
  this needs one real compute-target object (the `.available`/`.compute()`/`.store()` contract, e.g.
  `TkNumericIntegration`) carried through a live `RayWorkPool`, i.e. actual physics compute. **Next
  step:** see
  [`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md) §5 — the
  same scoped-down driver script that would close `[04-read-table-service]`'s item 6 would also close
  this, since both need one real computed-and-stored object to check against.

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

- **[01-shard-key-persistence]** *(opened by prompt 01, 2026-09-03; resolved by prompt 10,
  2026-09-04)* — Audit §8 checklist items 1–2 needed a real Ray pipeline run, which did not exist
  anywhere in this tree when prompt 01 landed. Prompt 10 built one: a real 2-shard `ShardedPool`
  under a locally-bootstrapped Ray runtime, inserted 5 wavenumbers via the real
  `pool.object_get(...)` → `_assign_shard_keys` path, and confirmed (a) zero `MISMATCH` lines during
  insertion, (b) every `shard_keys.key_serial` matches its shard's `wavenumber.serial` — cross-checked
  both by direct SQL and by running the actual `tools/shard_key_audit.py` tool, which returned
  `VERDICT: OK`, exit 0; (c) stopping (`ray.shutdown()`) and reopening a second pool against the same
  primary file found the same 5 records (identical `store_id`s, no duplication) with no
  `AttributeError`. See
  [`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md) §4 checks
  1–3 for the full account. **Scope note carried forward:** this exercises the shard-key/datastore-
  reopen dedup path B1/B5/B2 concern, not a full compute-target stop/resume (no object was computed
  and interrupted mid-flight) — that gap is now folded into the `[04-read-table-service]`/
  `[05-persist-handler-split]` entries above, since closing it needs the same real compute run.

- **[08-inventory-sharded-factories]** *(opened by prompt 08, 2026-09-04; partially closed by prompt
  09, 2026-09-04; fully closed by prompt 10, 2026-09-04)* — Verification step 5 needed a live Ray
  cluster to exercise `ShardedPool.inventory()`'s real fan-out (`ray.get([shard.inventory.remote(...)
  ...])`) for one class per group. Prompt 09 closed the Group C (count-only) case live
  (`TkNumericValue`, `{"count": 11}` = 7+4 across two real shard files). Prompt 10 closed the
  remaining two groups against the same live pool: **Group A** (`TkNumericIntegration`, labelled) —
  one `validated` row inserted on each of two real shards with different labels and timestamps;
  `pool.inventory(...)` returned both labels merged (not one shard's), with the timestamp range
  correctly spanning both days. **Group B** (`GkSourcePolicyData`, flat-with-timestamps) — 3 rows on
  one shard, 2 on the other; `pool.inventory(...)` returned `{"count": 5, ...}` (summed, not `3` or
  `2`), timestamp range spanning both. See
  [`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md) §4.1 for the
  full record. All three groups are now confirmed live; this issue is closed.

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
12. **`inventory()` plumbing (prompt 06) is complete and inert, exactly as planned — see
    [log](logs/06-inventory-plumbing.md) for the full contract.** The load-bearing facts prompts 07/08
    must follow: every factory `inventory` method is `@staticmethod def inventory(conn, table, tables,
    *args, **kwargs):` (a stray `self` binds `conn` incorrectly — note 5 above); `tables` is passed
    positionally and *unconditionally* (no `tables_arg` switch, unlike `read_table`); every field a
    sharded-class factory returns must use one of the merge policies `_merge_queue` supports —
    `"extend"` (list/set), `"earliest"`/`"latest"` (datetime), `"sum"`/`"min"`/`"max"` (int/float, never
    bool) — or `inventory_config` merging will raise at call time; a labelled-shape factory (`{label:
    {field: value}}`) must return **the same set of labels on every shard regardless of row count**,
    because the merge path reads `data_queue[0]` to determine both the shape and the label set — a
    shard-dependent label set will surface as a bare `KeyError` inside `_merge_queue`'s field-lookup on
    whichever shard is missing a label, not as the diagnosed "label not in config" error (which only
    catches a label absent from *config*, not one absent from *some other shard's dict*). Also:
    `ShardedPool.inventory`'s replicated-class branch `ray.get`s and returns a **value**, not an
    `ObjectRef` (asymmetric with `read_table`) — prompt 09 can call `pool.inventory(...)` directly for
    either kind of class with no `ray.get` of its own.
13. **`inventory()` on all 15 sharded factories (prompt 08) is complete, and `config/sharding.py`
    now exports `inventory_config` for all 15 — see
    [log](logs/08-inventory-sharded-factories.md) for the full contract.** Load-bearing facts for
    prompt 09: every Group A/B class's `labels`/count fields carry **unresolved foreign-key
    serials**, not human-readable values (e.g. `"wavenumber_exit=1, model=1, atol=1, rtol=1"`) —
    resolving them to physical values is left to prompt 09 if it wants that, at the cost of one
    query per distinct serial; Group A's `labels` lists are **not deduplicated** (one entry per row,
    across all shards after merging), so `len(labels)` is a row count, not a distinct-configuration
    count; Group B (`GkSourcePolicyData`, `QuadSourceIntegral`, `OneLoopIntegral`) carries no
    `labels` field at all, only `count`/`earliest_timestamp`/`latest_timestamp`. Also:
    `OneLoopIntegral.py`'s `store()` has a dead `"validated": False` insert key with no matching
    column (silently dropped by SQLAlchemy Core, confirmed harmless, not fixed — see the log's
    Observations section) — `OneLoopIntegral` correctly has no validated/unvalidated split despite
    that string appearing in the file.
14. **F2 is complete end to end as of prompt 09 — see [log](logs/09-inventory-reporting.md) for the
    full report contract.** `main.py --inventory` (with `--inventory-verbose` for untruncated
    label/value lists) reports and exits without running any compute; the formatting logic lives in
    `tools/inventory_report.py` (a new package — `tools/__init__.py` was added, since `tools/` had
    none before). Two facts worth carrying into prompt 10:
    - **`main.py` never passed `inventory_config` to `ShardedPool(...)` before this prompt** — prompt
      06 added the constructor parameter and prompt 08 populated the config dict, but no prompt wired
      the one production call site to actually use it. Fixed in prompt 09's own commit (it is the
      first thing that calls `pool.inventory()` from `main.py` at all, so there was nothing to leave
      alone). If any other config value is ever added to `ShardedPool.__init__`'s signature in a
      future prompt, check `main.py`'s call site is updated in the *same* commit — this is the second
      time a constructor parameter and its call-site wiring have landed in different commits (see
      also `read_table_config`, which was wired correctly, for contrast).
    - **A local `ray.init()` (no `address=`, no pre-existing cluster) successfully starts a real,
      multi-actor Ray runtime in this environment** — prompts 01/04/05/06/08 all recorded "no live Ray
      cluster available" and fell back to hand-simulated or non-Ray-wrapped verification. That was
      true for *connecting to an existing cluster* but nobody had tried starting a fresh local one
      until prompt 09, which used exactly that to build a real 2-shard `ShardedPool` and exercise the
      genuine `ray.get([shard.X.remote(...) ...])` fan-out end to end. **This likely unblocks the
      live-Ray verification steps still outstanding in the `[01-shard-key-persistence]`,
      `[04-read-table-service]`, `[05-persist-handler-split]`, and (partially)
      `[08-inventory-sharded-factories]` issues in §3** — prompt 10 should try a local `ray.init()`
      before assuming any of those checks are out of reach. One caveat: `ShardedPool` registers its
      broker actor under the fixed name `"SerialPoolBroker"`, so only one `ShardedPool` can exist per
      Ray runtime at a time — building a second one in the same process needs `ray.shutdown()` /
      fresh `ray.init()` first.
15. **Campaign complete as of prompt 10 — see
    [`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md) for the
    full verification record.** Note 14's prediction was confirmed: a local `ray.init()` did unblock
    live verification of `[01-shard-key-persistence]` (fully closed) and
    `[08-inventory-sharded-factories]` (fully closed), plus the `read_table`-negative-case half of
    `[04-read-table-service]`. What it did *not* unblock is anything requiring an actual computed
    result — `[04-read-table-service]`'s `extract_*.py` half and all of
    `[05-persist-handler-split]` still need a real `TkNumericIntegration`-style object carried through
    a live `RayWorkPool`'s compute→store→persist cycle, which is genuine physics compute, not
    infrastructure the local-Ray trick can shortcut. `main.py`'s wavenumber sample size is hardcoded
    (50 source + 50 response, `main.py` lines 2669/2682) with no CLI flag to shrink it, so even "the
    smallest configuration `main.py` accepts" is a real compute run — closing the two remaining issues
    needs a small custom driver script outside production code instead (see the verification
    document §5 for a scoped-down approach and time estimate), not a `main.py` invocation.
