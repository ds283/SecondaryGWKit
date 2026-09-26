# Datastore integrity audit — what must be right before an archival store is built

**Date:** 2026-09-25 · **Tree:** `handover-remedial` at `226889f`, clean · **Author:** Claude Opus
5.5 · **Status:** read-only. No tracked file was changed to write it. Under `var/`, the three real
stores' shard files were read `mode=ro` by one census script (§4.4), and their SHA-256 were
checked unchanged afterwards. Everything else ran in memory or in temporary directories under
`docs/datastore-integrity-audit/`, which also holds every probe cited here, with its saved output
where it has one.

**Why it exists.** The user is about to build the **A3 v2** LambdaCDM store
(`docs/handover/quadsource_atol_sweep.py --build`, which `exec`s `main.py`'s own pipeline). It is
the first store meant to have archival value. The user asked that every known datastore-integrity
defect — broken schema, wrong foreign keys, wrong lookup keys — be fixed before it is built. Seven
such defects were already open, recorded by the `store-fingerprint` audit
([`docs/store-fingerprint-audit.md`](store-fingerprint-audit.md)) and left unowned when that
campaign closed at 5 / 5 on 2026-09-25; four more sat on other boards. This audit re-verifies all
eleven against the current tree, sweeps the same defect classes across every factory, and finds
more. The campaign that fixes them is
[`prompts/datastore-integrity/`](../prompts/datastore-integrity/README.md).

**The user's decisions, 2026-09-25**, taken after the first triage and not reopened here:
1. make the replicated write atomic;
2. keep `QuadSourcePolicy` and add it to `QuadSourceIntegral`'s key — it is a shell today but may
   gain settings, and if kept it must be treated correctly;
3. change the exit-time tolerance lookup to an exact match;
4. leave the adjacent issues out of scope — `[02-oneloopintegral-is-a-ninth-keyed-object-type]`,
   `handover` C2's `[03-gksource-policy-accepts-a-value-that-raises]` and
   `[03-quadsource-policy-vocabulary-differs]`, and
   `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` — as downstream work.

**How to read the citations.** **[probed]** — run while writing this document, output as quoted.
**[checked]** — re-read in the source while writing it. **[reported]** — established by one of
three fresh-context audit agents, by reading or by a probe under
`docs/datastore-integrity-audit/`, and not independently re-run here. Every **[reported]** claim
names its probe where one exists, so it can be re-run with
`PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/<probe>.py`.

---

## 1. The storage model, in the terms the defects need

- **Shards.** A `ShardedPool` is a plain object in the driver. Each shard is a separate SQLite file
  owned by a `Datastore` Ray actor, which runs one call at a time. Serials come from a single
  `SerialPoolBroker` actor, so a serial is unique across shards. **[reported]**
- **Replicated tables** (`config/sharding.py:3-17`) are copied onto every shard: `version`,
  `store_tag`, `redshift`, `wavenumber`, `wavenumber_exit_time`, `tolerance`, `LambdaCDM`,
  `QCD_Cosmology`, `IntegrationSolver`, `BackgroundModel` (+ its tags and values), `GkSourcePolicy`,
  `QuadSourcePolicy`. **[checked]**
- **Sharded tables** live on one shard, chosen by a wavenumber: by `k` for the Tk/Gk numeric and
  WKB integrations, `GkSource`, `GkSourcePolicyData`, `QuadSourceIntegral`, `OneLoopIntegral`; by
  `q` for `QuadSource`. An exit time maps to its wavenumber, so every row for one `k` is on one
  shard. **[reported]**
- **Foreign keys are never enforced.** Nothing sets `PRAGMA foreign_keys`;
  `fk_sweep_probe.py` reads `0`. Every foreign key in the schema is documentation, and a wrong one
  is a false statement nothing checks. **[reported]** A foreign key to a table on *another* shard
  cannot be declared at all, because each shard is its own file.
- **Journal mode** is SQLite's default rollback journal (`delete`), `synchronous=FULL`; no PRAGMA
  in the tree changes it. **[reported]**

---

## 2. Schema and foreign keys

`fk_sweep_probe.py` builds every factory's table in memory and lists each integer reference column
with its foreign key, nullability and shard placement; each value was traced through its `store()`.
Every `version`, `z_*_serial`, `model_serial`, `wavenumber_exit_serial`, tolerance, solver, tag and
value reference is correct except the rows below. **[reported]**

**2.1 `QuadSource.Tq_serial` points at the wrong table** — `[00-quadsource-tq-serial-has-the-wrong-foreign-key]`.
It declares `ForeignKey("QuadSource.serial")` (`Datastore/SQL/ObjectFactories/QuadSource.py:113-118`)
**[checked]** and holds a `TkNumericIntegration` serial (`ComputeTargets/QuadSource.py:99`, `:473`)
**[checked]**. `Tq` is the Tk numeric row at `k = q`, and `QuadSource` shards on `q`, so the parent
is always on the same shard and a correct foreign key is possible. `tq_fk_probe.py`, with
enforcement on: the current constraint rejects a real row, `ForeignKey("TkNumericIntegration.serial")`
accepts it. **[reported]** `Tr_serial` rightly has none (`Tr` is on `r`'s shard), but the comment
at `:124` names `QuadSource.serial`, the same error. `Datastore/tests/real_store_fixtures.py:1185`
repeats it (`{"Tr_serial": "QuadSource"}`). **[reported]**

**2.2 `GkSourcePolicyData.wavenumber_exit_serial` has no foreign key** (`GkSourcePolicyData.py:68-70`)
**[checked]**. Its referent is replicated, so one is valid. New.

**2.3 WKB rows record nothing of the numeric row they start from** — `[20-wkb-rows-consume-numeric-initial-data]`,
and wider than its board entry.
- Tk: `main.py` looks up the numeric row (`:1463-1511`), then passes `z_init`, `T_init`,
  `Tprime_init` from its stop point (`:1533-1552`); with no numeric row it raises (`:1519-1531`), so
  a Tk WKB row always has a parent. Gk: it seeds from `stop_G`/`stop_Gprime` when the numeric row
  exists (`:2184-2226`), and otherwise builds a **pure-WKB** row with `G_init=0`, `Gprime_init=1`
  (`:2250-2268`) — so a Gk parent column must be nullable. **[reported]**
- The lookup keys (`TkWKBIntegration.py:280-296`, `GkWKBIntegration.py:261-281`) filter `z_init`
  only optionally, with an absolute `1e-7` against `z ~ 1e9`–`1e12`. The existence queries in
  `main.py` (`:1416-1434`, `:2025-2046`) and the stage lookup at `:3344-3348` pass **no** `z_init`
  at all, so once any WKB row exists for `k` it is served whatever numeric row now exists.
  **[reported]**
- Both tables shard on `k`, so a real foreign key to the numeric table is possible.
- `Datastore.py:141`'s `_drop_order` claims to respect foreign keys but drops `tk-wkb` after
  `tk-numeric` (`:149-150`); once a WKB→numeric key exists that order is wrong. **[checked]**

**2.4 Derived rows are not keyed on the rows they were computed from.** New, and the class that
2.3 is one instance of.
- `QuadSource` records `Tq_serial`/`Tr_serial` and looks up only on model, `q`, `r` and tags
  (`QuadSource.py:147-164`). **[reported]**
- `QuadSourceIntegral` records `source_serial` and `data_serial` without looking up on either
  (`QuadSourceIntegral.py:230-240`), and records nothing of the four Tk rows it consumes
  (`main.py:376-381`). **[reported]**
- `GkSource` records no Gk numeric or WKB parent (`GkSource.py:219-229`). **[reported]**
- `compute()` checks that the Tk rows are for the right `k`, but not that its `Tq` is the source's
  `Tq_serial`. **[reported]**

The consequence: change an upstream tolerance, grid or policy in a store that already holds
downstream rows, and the downstream rows are served stale. Only a run-label tag prevents it, and tag
joins are superset matches. **Inert for a store built once from scratch** — which is how A3 v2 is
planned — but a resumed or extended archival store is exactly where it bites. Two constraints on a
fix (**[reported]**): `store_inventory.py:236-245` resolves key parents only within one shard, so
`Tr_serial`, `QuadSourceIntegral.source_serial` and the QSI→Tk references cannot become fingerprint
identity until the inventory resolves parents across shards; and those same references can be
indexed columns but never foreign keys.

**2.5 `QuadSourcePolicy` is referenced by nothing** — `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`.
`main.py` creates two (`QuadSource_policy_1pt5`, `_5pt0`, `:3672-3688`) and passes them into
`run_pipeline` (`:981-982`), which never uses either; `QuadSourceIntegral` keys on `GkSourcePolicy`
(`QuadSourceIntegral.py:135-141`). Every `QuadSourceIntegral` and `GkSourcePolicyData` site uses
`GkSource_policy_1pt5` (`main.py:2722`, `:2818`, `:2945`, `:3065`, `:3211`, `:3432`), so
`GkSource_policy_5pt0` is unused too. **[reported]** `QuadSourcePolicy` is replicated, so a real
foreign key from the `k`-sharded QSI table is valid. The touch list is long but mechanical:
the compute target's constructor (`ComputeTargets/QuadSourceIntegral.py:1740-1756`), the factory's
`register`/`build`/`store`/`read_batch`/`inventory_records`, `main.py`'s lookup (`:3061-3080`) and
compute call (`:3428-3440`), `extract_QuadSourceIntegral_data.py:1062-1099`, and the tests that
construct the target by keyword. **[reported]** Because no computation reads the policy's fields,
the 1.5 and 5.0 policies would key two identical computations differently — correct as a record,
worth knowing.

**2.6 Two witnesses pin the schema and the fingerprint.** `Datastore/tests/test_schema_builder.py`
compares the built schema with `Datastore/tests/data/schema_at_base.json`, which its docstring says
"is never regenerated to make these tests pass" **[checked]** — it witnessed a refactor, not a
design. `RunRegistry/tests/test_store_fingerprint.py:60` pins `full_store_fingerprint.json`. Every
schema change in this audit breaks the first; every key change breaks the second. The campaign
needs a stated rule for deliberate changes to both.

---

## 3. Lookup keys and read paths

`docs/datastore-integrity-audit/lookup_keys/` holds seven probes, each building the schema in
memory with `build_schema(MetaData(), Datastore.SQL.Datastore._factories)` and calling the real
factory code, no Ray. Two (`01b`, `04b`) apply a proposed fix to the factory source in memory and
never write the file.

**3.1 Every tagged `read_batch` of `QuadSourceIntegral` fails** — `[00-tagged-read-batch-joins-an-unselected-alias]`,
**confirmed**. `QuadSourceIntegral.py:679` and `OneLoopIntegral.py:332` join
`tab.c.parent_serial == query.c.serial`. `probe_01_tagged_read_batch.py`, three rows tagged
`{A,B}`, `{A}`, none **[probed]**:

| tags | result |
|---|---|
| `[]` | `[1, 2, 3]`, correct |
| `[A]`, `[A,B]` | `OperationalError: no such column: anon_1.serial` — the join names `anon_1`, `anon_2`, outside FROM |
| control, `table.c.serial` | `[1,2,3]`, `[1,2]`, `[1]`, all correct |

`extract_QuadSourceIntegral_data.py:1087-1099` passes the run label as a tag whenever the store
records a run, so **extraction of `QuadSourceIntegral` from any run-labelled store fails** — A3 v2
included. No test calls any `read_batch`. The fix is one line in each file (`probe_01b`).
**[reported]**

**3.2 The same method returns the wrong `store_id` for its exit times.** New.
`QuadSourceIntegral.read_batch` selects `q_wavenumber_tab.c.serial.label("q_serial")` (`:572`,
`:576`) and uses it as the exit time's `store_id` (`:703`, `:709`) **[checked]**: a wavenumber
serial where an exit-time serial belongs. Latent — the extraction script only plots.

**3.3 The numeric value reads omit `break_point_kind`** — `[00-numeric-value-parent-lookup-omits-break-point-kind]`,
and wider than its entry: the live `GkNumericValue.read_batch` (`GkNumericIntegration.py:961-995`)
has the same omission as the two `_build_impl_model` paths (`TkNumericIntegration.py:862-878`,
`GkNumericIntegration.py:863-878`). `probe_05`, two parents differing only in `break_point_kind`:
both `build` paths raise `MultipleResultsFound`; **`read_batch` silently returns both parents'
values**, and `main.py:2466-2475` then keeps whichever came last. At the same site
`TkNumericValue.build` dispatches on `payload["z_init"]` and `_build_impl_model` reads
`payload["z_source"]`. **[reported]** Latent while every store uses one break-point kind.

**3.4 `GkWKBValue` reads omit `z_init`.** New. `GkWKBIntegration.py:1131-1162` (`read_batch`, live
at `main.py:2434`) and `:1026-1040`: with two parents at different `z_init`, `read_batch` returns
both and `build` raises. **[reported]**

**3.5 The exit-time lookup is an inequality, and raises once a tighter row exists** —
`[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`, worse than recorded. `wavenumber.py:216-228`
filters `stored − requested <= 1e-7` on log10 atol and rtol and `stepping >= requested`, then orders
descending **[checked]**. `probe_04_exit_time_and_solver.py` **[probed]**, stored (1e-10, 1e-9):

| request | served |
|---|---|
| (1e-10, 1e-9) | serial 1 |
| (1e-10, 1e-8), (1e-8, 1e-8) | **serial 1, reporting (1e-10, 1e-9)** |
| (1e-12, 1e-9) | miss |
| second row (1e-12, 1e-9) also stored; request (1e-10, 1e-9) | **`MultipleResultsFound`** |

The `ORDER BY` is dead code, because `one_or_none()` raises on two rows whatever their order. So the
first time a tighter exit time is stored at some `k`, the production lookup at the original pair
raises from then on. The other keyed types match a tolerance by **serial equality**
(`tolerance.build` has already mapped the value to one serial); that is the exact form, verified in
memory by `probe_04b`. **[reported]** Callers: `main.py:993-1001` and six `extract_*.py` scripts
request the stored pair and are unchanged; `analyse_greens_and_source.py` and
`run_quadsource_integrals.py` rely on the inequality (`[05a-two-verification-scripts-…]`) and would
then miss.

**3.6 The query runs inside the loop** — `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]`,
confirmed at `wavenumber.py:235` **[checked]** (the board's `:270` is stale): six statements per
lookup, the last one complete.

**3.7 `IntegrationSolver` is served the wrong row** — `[03-integrationsolver-stepping-minimum-lookup]`,
worse than recorded. `integration_metadata.py:30-34` matches `stepping >= requested` and takes
`.scalar()` **[checked]**. `probe_04` **[probed]**: with rows `{1: 8, 2: 4}`, requests for 4, 6 and 8
are **all** served serial 1 (stepping 8), and the object reports the requested stepping. An exact
row for 4 exists and is not used.

**3.8 `.scalar()` identity lookups never detect a duplicate.** New. `tolerance.py:35`,
`store_tag.py:26`, `version.py:26`, `integration_metadata.py:34`, `GkSourcePolicy.py:39`,
`QuadSourcePolicy.py:39`, `LambdaCDM.py:55`, `QCD_Cosmology.py:99`. Verified for `tolerance`: two
rows at log10 −10 and serial 3 is served silently. **[reported]** These are the replicated classes,
so this is where a duplicate from §4 would be hidden.

**3.9 `redshift.build` at `z = 0` never matches.** New. `redshift.py:40` divides by `z`
**[checked]**; SQLite returns NULL for `0/0`, so every call inserts a new row (three calls, three
rows). **[reported]** Latent: production's `--zend` is 0.1.

**3.10 The read-back guard sees only `build()`** — `[01-read-batch-is-outside-the-guard]`.
`Datastore/tests/test_factory_select_columns.py` checks, by `ast`, that every column a `build()`
reads off a result was selected. It never visits `read_batch`, whose rows reach a nested helper's
parameter, nor the `_build_impl_*` helpers `build()` delegates to. `probe_02` runs the guard's own
analyser over both **[reported]**, and finds a real defect it would have caught:
`row_data.WKB_criterio` at `GkWKBIntegration.py:947` and `TkWKBIntegration.py:1001` **[checked]**
(every hit raises `AttributeError`). It cannot see column existence; `probe_03` resolves 1176
`<table>.c.<name>` references against the schema and finds three wrong names: `GkSource.py:766`
(`z_serial`), `QuadSource.py:672` (`model_id`), and `OneLoopIntegral.py:235` (an insert key
`validated` with no column). **[reported]**

**3.11 Smaller read-path defects**, all latent (no live caller), all **[reported]** with the probe
named:
- `GkNumericIntegration.py:226` and `:1029` label `redshift.source` as `…_is_response`
  **[checked]**;
- `TkWKBValue.read_batch` (`TkWKBIntegration.py:1257-1279`) never passes the required `friction`;
- `tags = payload.get("tags", None)` then `for tag in tags` at eleven sites raises on omitted tags;
- `raise ("…")` with a string at `GkWKBIntegration.py:921`, `TkWKBIntegration.py:974`,
  `GkSource.py:777` raises `TypeError` instead of its message.

**3.12 `OneLoopIntegral`** — `[00-oneloop-lookup-joins-the-wrong-tag-table]`, confirmed, and not the
factory's only defect. `build` joins `QuadSourceIntegral_tags` (`OneLoopIntegral.py:141`)
**[checked]**, so a correctly tagged row is missed and an untagged one can match a QSI row's tags by
coincidence of serials (`probe_06`). Beyond that: every hit raises `KeyError: 'source_serial'`
(the payload has no such key and the table no such column); `store()` writes a `validated` key to no
column; `read_batch` is a module-level function with no `return`. The target is never built
(`main.py` has no site), and its keying is the user's out-of-scope decision (decision 4 above).

**3.13 Floats matched with an absolute epsilon.** `wavenumber.py:47` (absolute `1e-7` on `k` up to
`3e8`, ≈ 1.7 ulp) is effectively bit equality and fine for regenerated grids. The WKB `z_init` match
(§2.3) is the fragile one, and removing `z_init` from the key removes it. **[reported]**

---

## 4. The replicated write

**4.1 The path.** A replicated object is written first on a **controlling shard chosen at random
per call** (`ShardedPool.py:970-975`, and again at `:1157-1162` and `:1227-1232`) **[checked]**, in
one transaction, then re-sent to every other shard, each in its own transaction:
- **get-or-insert** classes (`tolerance`, `redshift`, `wavenumber`, the policies, …) re-run
  `object_get(serial=store_id)` on each replica, and **the replicas' answers are discarded**
  (`:1008-1036`, comment at `:1043`) **[checked]** — a replica that holds the key under a different
  serial returns it and nothing notices;
- **store** classes (`wavenumber_exit_time`, `BackgroundModel`) send the controller's object to
  every replica (`:1173`) **[checked]**, whose `store()` inserts with `serial=_my_id` and no lookup;
- **validate** (`BackgroundModel`) validates on a controller, then each shard counts its own value
  rows and sets its own flag (`:1224-1251`). **[reported]**

Each shard stamps its own `timestamp` (`Datastore.py:582-583`) **[checked]**. `BackgroundModelValue`
rows draw their serials per shard, harmlessly — nothing references them. **[reported]**

**4.2 How it diverges.** Any failure after the controller's commit — an exception on one replica, an
actor death, a SIGTERM of the driver — leaves the controller's row on some shards and not others.
A SIGTERM mid-transaction also leaves a **hot journal** (`hot_journal_probe.py`: one of 9728 bytes),
which the read-only fingerprint reader refuses and a read-write open rolls back. **[reported]**

**4.3 How divergence turns into a wrong answer.** `replicated_write_fault_probe.py` runs the real
`ShardedPool`, `Datastore`, factories and broker without Ray (actor classes unwrapped, `.remote()` a
direct call, faults injected per actor and method). **[probed]**

```
2. replica fault: shard0001-store.object_get(tolerance) killed
   per shard: {0: [(1, -8.0)], 1: [(1, -8.0)], 2: [(1, -8.0), (2, -6.0)]}
4. resume, controller lacks the row: driver now holds serial 3
   per shard: {0: [(1, -8.0), (3, -6.0)], 1: [(1, -8.0), (3, -6.0)], 2: [(1, -8.0), (2, -6.0)]}
5. replica fault: shard0001-store.object_get(wavenumber) killed
   resume with lacking controller: driver holds k serial 101
   per shard: {0: [(101, 0.5)], 1: [(101, 0.5)], 2: [(1, 0.5)]}
   primary shard_keys: {101: 2}
6. inventory problems: (none)
```

So an interrupted replication followed by a resume whose random controller lacks the row produces a
**serial split**: one key, two serials, on different shards. Every shard's own counts still agree.
Sharded rows then join to a serial their shard does not hold and are silently dropped from lookups
and reads; for a wavenumber the shard key is assigned under the new serial, so its sharded work
lands on a shard where `k` has another serial. **The store inventory reports nothing** (step 6).
`replicated_divergence_probe.py` shows the inventory catches a missing row, a duplicate, a
`validated` difference and a value-count difference, and is blind to a serial-only split, to the
`source`/`response` flags (which `read_table` serves from a random shard to the `extract_*.py`
scripts), to timestamps and to value contents. **[reported]** So `store-fingerprint` prompt 05's
"no divergence" on the three real stores did not rule a split out. §4.4 does.

**4.4 The three real stores, serial by serial.** `replicated_serial_census.py` opens every shard
`mode=ro` and compares each replicated table's full rows across shards, excluding only `timestamp`
(stamped per shard by design) and `BackgroundModelValue.serial` (drawn per shard by design).
**[probed]** Nothing was `running` (`python -m RunRegistry list`), no journal was present, and the
primaries' and shards' SHA-256 were identical before and after.

| store | replicated tables | result |
|---|---|---|
| live A3, `handover-A3-baseline-lambdacdm` | 14, all four shards | **no divergence** |
| sweep, `handover-atol-sweep` | 14, all four shards | **no divergence** |
| backup, `backup-pre-resume-20260921T091011/…` | 14, all four shards | **no divergence** |

So no store today holds a split, and a check at open that refuses divergence would refuse none of
them.

**4.5 Options.** **[reported]**, with the reasoning re-read here.
- *Two-phase commit across actors:* SQLite has no durable prepared state, so a crash between the
  commits still diverges; it narrows the window and adds no recovery. Rejected.
- *One connection with `ATTACH`:* atomic across attached files in rollback mode, but
  `SQLITE_LIMIT_ATTACHED` is 10 here and `main.py`'s default is 20 shards; it would run factory
  code outside the actors. Rejected.
- *Content-derived serials:* lookups match within tolerances, so content has no single hash; it
  re-keys every foreign key. Rejected.
- **Recommended: checked, idempotent replication with repair at open.** Nothing in the write path
  deletes a replicated row, the controller always commits first, and replicas only copy what it
  committed; so the correct state after any crash is always "every shard holds what the controller
  holds". Four parts: an in-flight record in the primary around each replicated write; every
  replica's answer checked against the controller's serial, raising on a mismatch; replica stores
  made insert-or-verify, with one driver-chosen timestamp and value-row serials passed through; and
  a driver-side step in `ShardedPool.__init__`, before any actor exists, that rolls back hot
  journals, compares every replicated table across shards, **repairs by copying forward** from the
  in-flight record's controller, and **refuses to open** on any other difference.
- The inventory cannot be the verifier: it is blind to splits and flags (§4.3) and refuses a store
  with a journal, which is what an interrupted build leaves. **[reported]**
- The design can be tested without Ray, with the stand-in shards `replicated_write_fault_probe.py`
  already uses. **[reported]**

---

## 5. What follows for the campaign

**5.1 In scope, by the user's decisions:** everything in §2–§4 except the exclusions of decision 4 above.

**5.2 Open questions for the user**, stated with recommendations in the campaign README §6:
- **§2.4** — key every derived row on the rows it was computed from? The user asked for every
  known integrity defect, and this is the class §2.3 belongs to.
- **§3.10–3.11** — fix the latent defects on read paths no one calls today? The extended guard
  (§3.10) would otherwise have to carry them as known-unfixed.
- **§3.12** — how much of `OneLoopIntegral`'s factory to repair, given that its keying is out of
  scope.
- **§2.6** — the rule for deliberately changing the schema witness and the golden fingerprint.
- **Existing stores.** After the key changes (§2.3–2.5), the three stores in `var/` lack columns the
  new lookups filter on, so the post-campaign tree cannot read them; each stays readable at the git
  SHA it was written under. The fingerprint's identity changes too, so a fingerprint recomputed at
  the new tree will not match one recorded before — which `store-retirement`'s `retire` checks.
  The **sequencing consequence** is in README §2.

**5.3 Out of scope, recorded as issues** on the campaign board §3: the inventory's blind spots
(§4.3); `OneLoopIntegral`'s round-trip defects that are part of its keying (§3.12); the unused 5.0
policy pair (§2.5).
