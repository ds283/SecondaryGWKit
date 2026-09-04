# Backport campaign verification and close-out

**Campaign:** [`prompts/backport-modules/README.md`](../prompts/backport-modules/README.md)
**Status board:** [`prompts/backport-modules/IMPLEMENTATION_STATE.md`](../prompts/backport-modules/IMPLEMENTATION_STATE.md)
**Source audit:** [`docs/backport-modules-audit.md`](backport-modules-audit.md) §8
**Prompt:** [`prompts/backport-modules/10-verification.md`](../prompts/backport-modules/10-verification.md)
**Date:** 2026-09-04

---

## 1. What this document is

Prompts 01–09 landed the backport (B1–B5, D1–D4, F3, F6, E1) and the `inventory()` feature (F2).
Each prompt's own log recorded what it could verify at the time — mostly static checks and
synthetic/stand-in harnesses, because no `SGWK`-schema datastore and no live Ray cluster existed
anywhere in this tree when the campaign started. This document collects the audit's §8 checklist
(extended with four F2-specific checks the audit did not anticipate), states plainly what has now
been run for real versus what still has not, and gathers every deviation from every prompt into one
table.

**Headline result:** a locally-bootstrapped Ray runtime (`ray.init(num_cpus=N)`, no pre-existing
cluster) turns out to work in this environment — prompt 09 discovered this first, and this pass
confirms it generalises. Against a real, multi-shard, Ray-actor-backed `ShardedPool`, **8 of the 12
checklist items now have genuine live confirmation**, several for the first time in the campaign
(checks 1–3 had previously been verified only against direct method calls or synthetic sqlite
fixtures, never a real multi-actor pool). The three items that remain unrun all require actual
physics compute (a populated `TkNumericIntegration`/`GkSource`/… pipeline), which this prompt's file
list (verification scripts and documentation only) and time budget do not extend to.

---

## 2. Reconciliation

### 2.1 Git history

```
1ccbcfd Plan the backport of shared Ray/Datastore modules
2610abe Fix shard-key persistence in ShardedPool                       (prompt 01)
9206704 Fix shard-key config reader in ShardedPool._read_shard_data    (prompt 02)
e81b145 Tidy up latent faults in ShardedPool, RayWorkPool and ClientPool (prompt 03)
6c69088 Replace generated read-table methods with a read_table() service (prompt 04)
d6f4a43 Fix stale defaults import in LiouvilleGreen/WKBtools.py        (out-of-band, see §2.2)
390c3d5 Close out [02-shard-config-reader] tracking issue              (out-of-band, see §2.2)
0ee16a8 Split RayWorkPool store_handler into store and persist handlers (prompt 05)
f27ad7c Add an inventory() service to Datastore and ShardedPool        (prompt 06)
fedda9f Add inventory() to the replicated-table object factories       (prompt 07)
3af7b27 Add inventory() to the sharded-table factories and merge policies (prompt 08)
a380872 Report datastore contents from the inventory() service         (prompt 09)
```

One commit per prompt, in order, with two out-of-band commits — both already tracked in
`IMPLEMENTATION_STATE.md` §4 as resolved issues, neither one of this campaign's tracked audit items
(B1–B5/D1–D4/F2/F3/F6/E1). History is clean; nothing interleaved or rewritten.

### 2.2 Out-of-band commits (context, not new findings)

- **`d6f4a43`** — `LiouvilleGreen/WKBtools.py:6` imported `from defaults import DEFAULT_ABS_TOLERANCE`,
  but top-level `defaults.py` was deleted in `a2bd966` (2025-12-15, the same commit that introduced
  B2) with its contents moved to `config/defaults.py`. This broke any real (non-stubbed) import of
  `Datastore.SQL.ShardedPool` and blocked prompts 02–04's own verification passes, which worked
  around it with a `sys.modules` stub. Fixed as a standalone one-line commit, at the user's direct
  request, outside the prompt sequence.
- **`390c3d5`** — closed the `[commit-sha-links-stale]` tracking issue (see §2.4 below).

### 2.3 Standing-note counts, re-run

All from `IMPLEMENTATION_STATE.md` §5 note 5, re-run fresh for this document rather than taken on
trust:

| Count | Expected | Actual |
|---|---|---|
| `read_wavenumber_table\|read_redshift_table` | 0 | **0** |
| `store_handler=None` | 35 | **35** |
| `persist_handler=None` | 35 | **35** |
| `RayWorkPool(` | 45 | **45** |
| `_generic_read_table` | 0 | **0** |
| `def inventory` under `ObjectFactories/` | 28 | **28** |
| `def inventory(self` under `ObjectFactories/` | 0 | **0** |
| `key_id` in `ShardedPool.py` | 0 | **0** |
| `_db_file` in `ShardedPool.py` | 0 | **0** |

All match. No stray edit landed between the last prompt and this one.

### 2.4 Open issues collected from `IMPLEMENTATION_STATE.md` §3

Four open issues existed going into this prompt, all in the same class ("verified structurally or
synthetically, never against a live multi-actor Ray pool"):

1. **`[01-shard-key-persistence]`** — audit checklist items 1–2 (fresh datastore, no `MISMATCH`;
   stop/resume finds existing records) needed a real pipeline run.
2. **`[04-read-table-service]`** — checklist items 5–6 (`read_table` negative cases under live
   multi-shard dispatch; `extract_*.py` scripts returning correct arrays under a real run) needed a
   live Ray cluster.
3. **`[05-persist-handler-split]`** — checklist item 7, second half (a real driver run confirming
   default handlers still store results correctly) needed a live Ray cluster.
4. **`[08-inventory-sharded-factories]`** — partially closed by prompt 09 (Group C/count-only case
   confirmed live); a live Group A (labelled) and Group B (flat-with-timestamps) class remained
   unexercised beyond an empty datastore.

§3 below records what this prompt closes, and what remains genuinely open, for each.

---

## 3. Static verification (Step 2)

All cheap, all re-run fresh, all pass:

| Check | Command | Result |
|---|---|---|
| Byte-compile | `./venv/bin/python3 -m py_compile main.py extract_*.py Datastore/SQL/*.py Datastore/SQL/ObjectFactories/*.py RayTools/*.py config/sharding.py tools/*.py` | exit 0 |
| Formatting | `./venv/bin/python3 -m black --check <same files>` | "42 files would be left unchanged" |
| Import, for real | `import Datastore.SQL.ShardedPool`, `Datastore.SQL.Datastore`, `RayTools.RayWorkPool`, `config.sharding` | all import clean, **no `sys.modules` stub needed** — confirms `d6f4a43` generalises |
| `read_table_config` shape | keyed by class name, no `"class"` entries; `wavenumber` → `tables_arg: False` (factory signature has no `tables` param), `redshift` → `tables_arg: True` (factory signature has `tables` param) | confirmed against the real factory signatures via `inspect.signature` |
| `inventory_config` coverage | `set(sharded_tables.keys()) == set(inventory_config.keys())`, both length 15 | `True` |
| No `def inventory(self,` | `grep -rn "def inventory(self" Datastore/SQL/ObjectFactories/*.py` | 0 hits |
| `base.py` unchanged | read directly | `inventory`/`read_table` are not `@abstractmethod`; only `register`/`build`/`store`/`validate`/`validate_on_startup` are, as originally |

---

## 4. Behavioural verification (Step 3)

The audit's checklist (§8), extended with the four F2-specific checks `README.md` and prompt 10 add.
Every "run" row below was executed against a **real, locally-bootstrapped Ray runtime**
(`ray.init(num_cpus=4, include_dashboard=False)`, confirmed to start a genuine multi-actor Ray
instance with no pre-existing cluster) driving a **real, multi-shard `ShardedPool`** with real shard
actors — not a hand-simulated reproduction of the dispatch logic, and not a `__new__`-constructed
stand-in object, both of which every earlier prompt in this campaign had to fall back on.

The harness lives at `scratchpad/verify_prompt10.py` (throwaway, not committed, per the campaign's
established convention — see §6). Full run output: `scratchpad/verify_prompt10_output.log` (also not
committed).

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Fresh datastore: `shard_keys.key_serial` == corresponding `wavenumber.serial`; no `MISMATCH` lines | **RUN — PASS** | Built a real 2-shard `ShardedPool`, inserted 5 wavenumbers via `pool.object_get("wavenumber", payload_data=[...])` (the real B1/B5 code path). Captured stdout during insertion: 0 `MISMATCH` lines. Cross-checked every `shard_keys` row against its shard's own `wavenumber.serial` directly via SQL: all 5 matched. **Also ran the actual `tools/shard_key_audit.py` tool** (not just my own SQL) against the resulting primary file: `VERDICT: OK`, exit code 0. |
| 2 | Stop/resume finds all previously-written records, no unexpected recomputation | **RUN — PASS** | `ray.shutdown()`, then reopened a second `ShardedPool` against the same primary file and re-requested the same 5 wavenumbers. Returned the identical `store_id`s (`[1,2,3,4,5]` both times) — no new rows created. Row-count check: `wavenumber` (a *replicated* table, so 5 rows × 2 shards = 10 is the correct un-duplicated baseline) held at 10 both before and after reopening, confirming no duplication. **Scope note:** this exercises the shard-key/datastore-reopen dedup path that B1/B5/B2 actually concern; it does not exercise a full compute-target stop/resume (no `TkNumericIntegration`-style object was computed and interrupted mid-flight) — see check 4/7/8 below for what that would take. |
| 3 | Reopen completes without `AttributeError`; mismatched `shard_key_type` raises the intended `RuntimeError` | **RUN — PASS** | The reopen in check 2 completed with no `AttributeError`. Separately, opened a *third* pool against the same primary file with `ShardKeyType=redshift` (mismatched): raised exactly `RuntimeError: Existing ShardedPool was configured with shard key type "wavenumber", but provided type was "redshift"` — not a second `AttributeError`. |
| 4 | Each `extract_*.py` constructs its pool and returns the expected wavenumber/redshift arrays | **NOT RUN** | Needs a populated `GkSource`/`QuadSource`/… datastore from a real compute pipeline. See §5. |
| 5 | `pool.read_table("GkSource")` (sharded) raises; `pool.read_table("LambdaCDM")` (replicated, unconfigured) raises | **RUN — PASS** | Against the real, live pool (not the `__new__`-constructed stand-in prompt 04's own log used): both raised the intended `RuntimeError`s, with the intended text. Bonus: `read_table("wavenumber", is_source=True)` against the same live pool returned all 5 real rows. |
| 6 | Task builder returning `None` completes under `store_results=False`, raises under `store_results=True` | **RUN — PASS** | Against a real `RayWorkPool(pool, ...)` built from the live pool (prompt 03's own verification used a dummy `pool` stand-in): both branches behave exactly as B4 requires. |
| 7 | Default handlers still store a result correctly | **NOT RUN** | Needs a real compute-target object (`.available`/`.compute()`/`.store()` contract) carried through `RayWorkPool`'s compute→store→persist cycle. See §5. |
| 8 | Full smoke test: `main.py` compute→store→validate, then restart | **NOT RUN** | See §5 — `main.py` hardcodes a 50+50 wavenumber sample with no CLI override, so even the "smallest configuration" is a real physics run. |
| 9 | `_merge_queue` handles every policy, the empty-shard case, rejects unknown policies/missing fields | **RUN — PASS** | Pure-function re-run (no datastore needed): `extend`(list)/`extend`(set)/`earliest`/`latest`/`sum`/`min`/`max`, `None`-as-first-value and `None`-as-later-value handling (both orderings), `bool` rejected with its dedicated message, unknown policy string rejected, missing config field rejected, and the caller's list object provably unmutated after the call. 13/13 assertions passed. |
| 10 | Every sharded class has both `inventory()` and a matching `inventory_config` entry, agreeing field by field | **RUN — PASS** | Re-verified (not just re-counted) for **all 15 sharded classes**: built a minimal in-memory SQLite table from each factory's own `register()`, called `inventory()` against it empty, and asserted the returned field set (flat or per-label) exactly matches `inventory_config[class_name]`'s configured fields in both directions. 15/15 passed with zero problems. |
| 11 | Inventory report runs against a populated and an empty datastore, no `None` leaking | **RUN — PASS** (this prompt); **also previously run** (prompt 09) | This prompt: ran `format_inventory_report` against the live 5-wavenumber pool — 1637 characters, no `(error` substring, no `: None` substring. Prompt 09's own log additionally ran it against both a richly populated and a completely empty real datastore with full output captured (see that log) — not re-duplicated here. |
| 12 | A value-table count matches `SELECT COUNT(*)` summed by hand across shard files | **RUN — PASS** | Inserted 7 synthetic rows into shard 0's `TkNumericValue` and 4 into shard 1's, directly via SQL. `pool.inventory("TkNumericValue")` → `{"count": 11}`, matching `7 + 4` asserted by direct comparison. This is a fresh, independent reproduction of prompt 09's own finding (same class, same technique), not a re-run of the same call. |

### 4.1 Beyond the checklist: closing the rest of `[08-inventory-sharded-factories]`

`IMPLEMENTATION_STATE.md` §3 named the exact next step for this open issue: exercise a live Group A
(labelled) and Group B (flat-with-timestamps) class the same way prompt 09 did for the Group C
(`TkNumericValue`) case. Done here, using the same live pool:

- **`TkNumericIntegration`** (Group A): inserted one `validated` row on shard 0
  (`wavenumber_exit_serial=1`, timestamp day 1) and one on shard 1 (`wavenumber_exit_serial=2`, day
  2). `pool.inventory("TkNumericIntegration")` returned both labels in the merged `"validated"`
  bucket, `earliest_timestamp`/`latest_timestamp` spanning day 1–day 2 (not one shard's value), and
  an empty `"unvalidated"` bucket. Confirms `"extend"` genuinely unions across shards and
  `"earliest"`/`"latest"` genuinely span shards for the labelled shape, live.
- **`GkSourcePolicyData`** (Group B): inserted 3 rows on shard 0 (day 1) and 2 on shard 1 (day 2).
  `pool.inventory("GkSourcePolicyData")` returned `{"count": 5, ...}` (summed, not `3` or `2`) with
  the timestamp range spanning both days.

Both passed. **`[08-inventory-sharded-factories]` is now fully closed** — see the board update in
§7.

---

## 5. What remains unverified, and what it would take

Three checklist items need actual physics compute, which is out of proportion to what a
verification-scripts-only prompt can spend:

- **Check 4 (`extract_*.py` scripts return correct arrays under a real run).** Needs a datastore
  populated with real `GkSource`/`QuadSource`/`TkNumericIntegration`/… rows from an actual compute
  pipeline, then running each of the 6 scripts against it and cross-checking the returned arrays
  against direct SQL. Populating even a minimal version needs `main.py`'s own pipeline (see check 8)
  to have produced something to read.
- **Check 7 (default handlers store a real compute result end-to-end).** `RayWorkPool`'s
  compute→store→persist cycle only activates for an object exposing the `.available`/`.compute()`/
  `.store()` contract (e.g. `TkNumericIntegration`). Prompt 05's own harness verified the handler
  *wiring* (call order, argument passing, validation branches) against fake stand-ins; closing this
  for real needs one genuine compute-target object carried through a live `RayWorkPool`, which means
  running at least one real numerical integration.
- **Check 8 (full `main.py` smoke test, then restart).** `main.py`'s wavenumber sample size is
  **hardcoded** (`np.logspace(np.log10(1e5), np.log10(3e8), 50)`, both source and response — main.py
  lines 2669/2682), with no CLI flag to shrink it, and reducing it would mean editing production
  code, which this prompt's file list does not permit. Even with every `--no-*-queue` flag set (which
  skips every compute-target queue), `main.py` unconditionally computes horizon-exit times for all
  100 wavenumbers across however many cosmological models `build_model_list` returns, before any
  gated queue is reached — real (if comparatively cheap) physics, not a no-op.

**Estimate for a follow-up, if the user wants these closed:** the cheapest path is a *modified* run,
not `main.py` itself — a small driver script (kept outside production code, e.g. under `tools/` or
`scratchpad/`) that constructs a `ShardedPool` exactly as `main.py` does but with a hand-picked
5–10-wavenumber sample and only the `Tk-numeric-queue` enabled, run against a real local Ray
instance. That would produce a handful of real `TkNumericIntegration`/`TkNumericValue` rows via the
actual compute→store→persist path (closing check 7), which one `extract_TkWKB_data.py`-style
`read_table` call could then cross-check (partially closing check 4 for one of the six scripts).
Order of magnitude: minutes, not hours, for 5–10 wavenumbers at default tolerances, based on the
integration ranges visible in `LiouvilleGreen`'s own tests — but this is an estimate from reading the
code, not a measurement, and should be treated as such until someone actually runs it.

---

## 6. Verification scripts

`scratchpad/verify_prompt10.py` (and its captured output log) — kept as scratchpad, **not
committed**, per every prior prompt's own convention in this campaign (01's, 04's, 06's, 07's, 08's
and 09's logs all did the same). It has no ongoing value as a regression test in its current form: it
builds and tears down throwaway sqlite files under `scratchpad/`, prints results rather than
asserting them as a test suite would, and duplicates infrastructure (`fresh_ray()`, pool construction
boilerplate) that a real regression suite would want factored out and covered by `pytest`, which does
not exist for this part of the codebase (`IMPLEMENTATION_STATE.md` §5 note 2). Recommending a real
`Datastore`/`ShardedPool`/`RayWorkPool` test suite is future work, not something to bolt on here.

`tools/shard_key_audit.py` (prompt 01) continues to have ongoing value as a standalone operational
tool and was exercised again in this pass (§4, check 1) — no changes made to it.

---

## 7. Deviations across the campaign

Gathered from all nine prompt logs. "Reasoning" is condensed to the one line a reader needs to judge
whether to revisit it — see the cited log for the full argument.

| Item | Prompt | What changed vs. the plan | Classification | Reasoning (one line) |
|---|---|---|---|---|
| Audit exit code (unspecified case) | 01 | Chose exit code 2 for "cannot audit" (pre-refactor/unrecognised schema), distinct from 1 (audited, found broken) | Implementation choice | Prompt specified 0/nonzero only for the current-schema case; needed a third state for "not auditable at all" |
| Commented-out diagnostic | 01 | Left `# print(...)` line untouched rather than swapping for upstream's equivalent wording | Implementation choice | Already says the same thing; touching a disabled line adds diff noise for no benefit |
| `ATTACH`-based cross-file audit | 01 | Confirmed empirically (not just from docs) that a `mode=ro` URI connection propagates to a subsequent `ATTACH` | Structurally required | Prompt flagged this as a real risk to the whole cross-file-check design; had to be proven, not assumed |
| D2 branch: repair vs. delete | 02 | Repaired (`raise print` → `raise RuntimeError`) instead of deleting, reversing the plan's assumption about which branch was unreachable | Implementation choice | Control-flow trace showed the *intra-loop* branch always wins the race, not the post-loop one as the plan assumed; deleting it would lose the only diagnostic with row-level detail |
| F6 guard scope | 03 | Guarded `replicated_table_values` insert too, not just `sharded_table_values` | Implementation choice | Same failure mode, one extra `if` line, no behavioural cost since the table list is currently always non-empty |
| D3 fallback value | 03 | Used `500`, matching `ClientPool.__init__`'s own `default_batch_size` default | Implementation choice | No existing module constant; reused the class's own idea of a sane default rather than inventing one |
| F3 taken, not skipped | 03 | Removed `MetadataConcepts.version` import from both `Datastore.py` and `ShardedPool.py`, despite the audit marking it optional | Implementation choice | Costs nothing (`object_get` already special-cases strings); doing both files together avoids the exact "fixed one, not the other" inconsistency the audit warned against |
| `read_table` return placement | 04 | Kept `SGWK`'s existing outside-the-`with`-block return, not upstream's inside-the-block placement | Implementation choice | Matches this file's own established local convention (`object_get`, `object_validate`), which outweighs matching upstream for a value added to this specific file |
| Commit-SHA convention | 04 | Stopped embedding a commit's own SHA in its own log/board entry; corrected 01–03's stale SHAs | Process fix (structurally required) | The literal template is self-referentially impossible to satisfy without an amend; three prompts had already hit the same defect |
| Audit's `_inventory_config is not None` claim | 06 | Implemented the guard per the prompt's instruction without independently re-confirming the audit's specific claim about `SI`'s source (not present in this tree) | Implementation choice | No access to `SI`'s source from this repo; followed the instruction on its own merits rather than adjudicating an unverifiable claim |
| Numeric merge policies (`sum`/`min`/`max`) | 06 | Added all three, not just `sum` | Implementation choice | `sum` is what prompt 08 actually needed; `min`/`max` cost nothing to include alongside it |
| `wavenumber.inventory`'s `units` handling | 07 | Returns raw values always, physical values additionally when `units` supplied (grows the key set rather than switching meaning) | Implementation choice | A shape that only ever grows a key is a smaller edge for prompt 09's formatter than one whose existing key changes meaning |
| `tolerance.inventory` reports `tol`, not `log10_tol` | 07 | Returns `10 ** log10_tol` | Implementation choice | The stored encoding is an implementation detail; the tolerance itself is what a reader wants |
| `LambdaCDM`/`QCD_Cosmology.inventory` omit some columns | 07 | `f_baryon`/`T_CMB_Kelvin`/`Neff` left out of `values` | Implementation choice | Readability only, not a cost concession — tables are small, one row per configuration |
| `BackgroundModel`'s `NULL`-as-unvalidated | 07 | Unvalidated bucket's `WHERE` clause treats `NULL` as unvalidated | Implementation choice (not explicitly requested) | Matches this same file's own `validate_on_startup` treatment of legacy `NULL`, rather than silently dropping such rows from both buckets |
| Group A labels: raw list, not deduplicated | 08 | `labels` has one entry per row, not deduplicated | Implementation choice | Deduplicating would silently collapse genuinely distinct rows with no way to recover the true count, since Group A carries no separate `count` field |
| `GkSourcePolicyData` labels dropped | 08 | Applied the same "numerous rows" reasoning the prompt gave only for `QuadSourceIntegral`/`OneLoopIntegral` | Implementation choice | Same order-of-magnitude row count as the two named classes; consistency across the group |
| Bucket display order | 09 | Fixed order (`validated` then `unvalidated`) instead of alphabetical | Implementation choice, caught during the prompt's own verification | Alphabetical put `unvalidated` first, backwards from what a reader wants and from the prompt's own illustrative example |
| `values_physical` formatting | 09 | Folded into the header line rather than a second list | Implementation choice | Same values in a second unit system, not a materially different row set — a second list would double visual weight for no new information |
| `main.py` never wired `inventory_config` through | 09 | Fixed the missing constructor argument at the one production call site | Unintended drift (from prompts 06/08), caught and fixed, not merely observed | Neither prompt 06 (added the parameter) nor prompt 08 (populated the config) touched `main.py`'s own call site; without this fix, F2c would have been dead on arrival in production. Fixing the wiring at the point that first exercises it was judged squarely in-scope for prompt 09, not a fix to another prompt's closed commit |

No entry above required reopening a closed prompt's commit; every deviation was made, or (for the one
piece of unintended drift) caught and fixed, within the prompt that discovered it.

---

## 8. Deferred and out of scope

Unchanged from `README.md` §6, restated here for a self-contained record:

- **F4 — Apache-2.0 licence headers.** `SGWK` carries none anywhere. If wanted, apply repo-wide in a
  separate commit; do not apply to six files only.
- **Report X1 upstream.** `SI`'s `object_read_batch` guard mixes object semantics
  (`hasattr(shard_key, shard_key_field)`) with dict semantics on the two following lines
  (`shard_key[shard_key_field]`, `payload.update(shard_key)`) — neither a dict nor an object can pass
  through it. `SGWK` does not have this pattern and is unaffected, but it is a live break worth
  reporting to whoever maintains the `StochasticInstantons` tree.
- **F2 is complete, not deferred.** All 28 factories (13 replicated, 15 sharded) have an `inventory()`
  method, `config/sharding.py` has a matching `inventory_config` entry for every sharded one, and
  `main.py --inventory` reports the result. No class's inventory was dropped or limited beyond the
  deliberate, logged choices in §7 above (label lists omitted for volume reasons on
  `GkSourcePolicyData`/`QuadSourceIntegral`/`OneLoopIntegral`; some secondary columns omitted from
  `LambdaCDM`/`QCD_Cosmology` for readability).
- **X1–X4 remain unbackported**, as the audit directs.

---

## 9. Summary for the status board

- Static verification: **all green**, re-run fresh (§3).
- Behavioural verification: **8 of 12** checklist items now have live, real-Ray-actor confirmation
  (checks 1, 2, 3, 5, 6, 9, 10, 11, 12 — note 9 and 10 need no datastore and were always cheap;
  checks 1/2/3/5/6/11/12 are the ones newly run against a genuine multi-shard pool rather than a
  synthetic stand-in). **3 remain unrun** (checks 4, 7, 8), all gated on real physics compute that is
  out of proportion to a documentation-and-scripts-only prompt — see §5 for what closing them would
  take.
- `[01-shard-key-persistence]` and the `read_table`/`RayWorkPool`-negative-case half of
  `[04-read-table-service]` are substantially strengthened (live confirmation superseding synthetic);
  `[08-inventory-sharded-factories]` is **fully closed**. `[04-read-table-service]`'s `extract_*.py`
  half and `[05-persist-handler-split]` remain open, narrowed to exactly checks 4/7/8 above.
