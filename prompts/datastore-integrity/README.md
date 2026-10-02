# Campaign — datastore integrity

**Written:** 2026-09-25 on `handover-remedial`, by Claude Opus 5.5, from
[`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) and the user's
decisions recorded in §6.1. **Prompts 01–04 are written.** Their tasks are fixed by §6.1 alone.
05–10 are held: each waits on a decision in §6.2, or on what an earlier prompt ships (§2).
**Decisions recorded 2026-09-26:** the user decided D2–D6 (§6.2). That releases 06, 07 and 08 to be
written. 05 still waits on 04's list, 09 on D1, which the user has asked to have explained, and 10
on 01–09. **The user decided D1 later on 2026-09-26 (yes), which releases 09 too.** **06–09 were
written on 2026-09-26.** `store-retirement` closed on 2026-09-26 at 6 / 6, which discharges both of §2's
cross-campaign constraints. **04 and 04b landed and were reviewed by 2026-09-28, which releases 05
to be written.** **05 was written on 2026-09-28 as three prompts, 05a–05c, for one orchestrator session, and landed
the same day, all three reviewed.** **06, 07 and 08 landed and were reviewed by 2026-09-30.** **09 was split by the
user on 2026-09-30 into three prompts, 09a–09c, for one orchestrator session**
([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md)). **09a–09c landed and were reviewed by 2026-10-01; 09d, a small follow-up on the inventory report, was written, landed and reviewed the same day. 10 was written on 2026-10-01, and the user took its five decisions (its §1.1) the same day, all as recommended.** **10 landed on 2026-10-02 and was reviewed the same day: the resumed store equals the reference once pruning runs, and the run opened four issues, one assigned to `handover`.**

## 0. Why this campaign exists

The user is about to build the **A3 v2** LambdaCDM store, the first store meant to have archival
value, and asked that every known datastore-integrity defect be fixed first. Eleven were open on
four boards, seven of them left unowned when `store-fingerprint` closed. The audit re-verified all
eleven and found the classes wider than recorded:
- **The replicated write can diverge** (audit §4). A crash between shards' commits leaves copies
  different. A resume can then turn a missing row into a **serial split**, which the store
  fingerprint cannot see. All three real stores were checked serial by serial, and none is split
  (§4.4).
- **Lookups serve the wrong row** (§3.5–§3.9). The exit-time lookup is an inequality, and raises
  once a tighter row exists. `IntegrationSolver` ignores an exact row. Identity lookups on
  replicated classes take `.scalar()`, which hides a duplicate.
- **Read paths are broken** (§3.1–§3.4, §3.10–§3.12). Every tagged `QuadSourceIntegral`
  `read_batch` fails, so extraction from a run-labelled store — A3 v2 included — fails. Value reads
  omit parts of their parent's key. The read-back guard sees only `build()`.
- **The schema says false things** (§2.1–§2.2). One foreign key names the wrong table, another is
  missing.
- **Derived rows are not keyed on what they were computed from** (§2.3–§2.5). WKB rows are not
  keyed on their numeric row, and `QuadSource`, `QuadSourceIntegral` and `GkSource` not on theirs.
  `QuadSourcePolicy` is keyed on by nothing.

**0.1 Correctness is the only objective.** Work is sequenced by epistemic dependency.
1. The replicated write comes first, because every other object's rows join to replicated rows; a
   store whose replicated tables can diverge cannot be trusted to verify anything else.
2. Then the identity lookups on those same classes, which is where a duplicate would hide.
3. Then the instrument before the remedy: the read-back guard is extended, and records what it
   finds, before the read paths are repaired against it.
4. Then the schema and the keys, from the foreign keys that are simply wrong up to the design
   question of keying derived rows on their parents.
5. Last, an end-to-end build on a throwaway store, so that the first archival store is not the
   first thing to exercise the result.

## 1. Scope

**In scope:**
- `Datastore/SQL/ShardedPool.py` and `Datastore/SQL/Datastore.py`: the replicated write path, and a
  check at pool open (prompts 01, 02);
- the factories under `Datastore/SQL/ObjectFactories/`: lookups, read paths, schema and keys
  (prompts 03, 05–09);
- `Datastore/tests/test_factory_select_columns.py`, the read-back guard, and a column-existence
  test beside it (prompt 04);
- the compute targets whose constructors must carry a new key: `ComputeTargets/TkWKBIntegration.py`,
  `GkWKBIntegration.py`, `QuadSource.py`, `QuadSourceIntegral.py`, `GkSource.py` (07–09);
- `main.py`, only where a key change needs it, and within §5 rule 12;
- the `extract_*.py` scripts, only where a key change breaks their lookup (07–09);
- `Datastore/store_inventory.py`, the factories' `inventory_records`, and the golden fingerprint,
  where a key change moves the fingerprint's identity (07–09), under D5;
- `Datastore/tests/data/schema_at_base.json`'s test, under D5;
- the eleven issues assigned here (board §3), and the boards that own them.

**Out of scope** (the user's decision 6.1.4, and the audit's §5.3):
- `[02-oneloopintegral-is-a-ninth-keyed-object-type]` — whether `OneLoopIntegral` is decoupled or
  keyed as the others are. Prompt 05's repairs to its factory are limited by D3;
- `handover` C2's `[03-gksource-policy-accepts-a-value-that-raises]` and
  `[03-quadsource-policy-vocabulary-differs]`;
- `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]`;
- the store inventory's blind spots (audit §4.3), recorded as an issue. Prompt 02's check at open is
  the verifier this campaign ships;
- `docs/handover/quadsource_atol_sweep.py`, which is a measurement record, except as §5 rule 12
  protects it;
- every store under `var/`, which no prompt writes (§3).

## 2. Prompts

| # | Prompt | Covers | Status |
|---|---|---|---|
| 01 | [`01-checked-replicated-write.md`](01-checked-replicated-write.md) | An in-flight record in the primary around each replicated write. Every replica's answer checked against the controller's serial. Replica stores made insert-or-verify, with value-row serials passed through. One driver-chosen timestamp. | **written** |
| 02 | [`02-repair-or-refuse-at-open.md`](02-repair-or-refuse-at-open.md) | A driver-side step in `ShardedPool.__init__`, before any actor exists: roll back hot journals, compare every replicated table across shards, copy forward from the in-flight record's controller, **refuse to open** on anything else. Closes `[00-replicated-writes-can-diverge-across-shards]`. | **written**; after 01 |
| 03 | [`03-exact-identity-lookups.md`](03-exact-identity-lookups.md) | Exit time by serial equality, the query out of the loop. `IntegrationSolver` by `==`. `.scalar()` → `one_or_none()` on every identity lookup. `redshift` at `z = 0`. Closes three issues. | **written**; after 02 |
| 04 | [`04-extend-the-read-back-guard.md`](04-extend-the-read-back-guard.md) | The guard over `read_batch` and every `_build_impl_*`, and a column-existence test. Records what it finds as known-unfixed. Closes `[01-read-batch-is-outside-the-guard]`. | **written**; independent of 01–03 |
| 04b | [`04b-tighten-the-read-back-guard.md`](04b-tighten-the-read-back-guard.md) | Findings 1–3 of the review of 04: every function visited, not a list of names; the function in the coverage identity; G2 declares every unresolved table and unchecked insert, and walks module-level functions. Fixes no factory. *(Added 2026-09-28, by the user's decision on the review.)* | **written** 2026-09-28; after 04 |
| 05 | **Split 2026-09-28 into 05a–05c**, one session, one commit each ([`orchestrator/prompt-05.md`](orchestrator/prompt-05.md)). Charter unchanged: The tagged join; the exit time's `store_id`; `break_point_kind` and `z_init` in the value reads, each refusing more than one parent per `z_source`; the wrong flag columns; the three wrong column names, `WKB_criterio` and `friction`; tags defaulting to `[]`; `OneLoopIntegral`'s tag table, as D3 bounds. Empties prompt 04's known-unfixed list. Closes three issues. | **landed** 2026-09-28, as 05a–05c, all reviewed |
| 05a | [`05a-repair-the-mechanical-read-defects.md`](05a-repair-the-mechanical-read-defects.md) | P6, P7: `WKB_criterio`, `z_serial`, `model_id`, `friction`; tags default to `[]`; the three `raise ("…")`. Empties `KNOWN_UNFIXED`. | **landed** 2026-09-28 |
| 05b | [`05b-repair-the-integral-read-batches.md`](05b-repair-the-integral-read-batches.md) | P1, P2, P8: the tagged join; the exit times' serials; `OneLoopIntegral` as D3 bounds, `validated` included. Empties `KNOWN_UNFIXED_COLUMNS`. Closes two issues. | **landed** 2026-09-28 |
| 05c | [`05c-key-the-value-reads-on-their-parent.md`](05c-key-the-value-reads-on-their-parent.md) | P3, P4, P5, P9: `break_point_kind` and required tolerances in the numeric value reads; `z_init` in the Gk **and Tk** WKB value reads; a second parent refused; the two labels; a pin on every `read_batch`'s round trip. Closes one issue. | **landed** 2026-09-28 |
| 06 | `06-correct-the-foreign-keys.md` | `Tq_serial` → `TkNumericIntegration`; the `Tr_serial` comment; `GkSourcePolicyData.wavenumber_exit_serial`; the fixtures' map; a test that turns `PRAGMA foreign_keys` on over a store the real factories wrote; the exit-time and `IntegrationSolver` registrations' `"stepping": "minimum"` made to say what prompt 03 made true; the first deliberate schema change, under D5. Closes one issue. | **written** 2026-09-26; after 03 |
| 07 | `07-key-wkb-rows-on-their-numeric-row.md` | A `numeric_serial` foreign key, NOT NULL on Tk and nullable on Gk (NULL = pure WKB), in the lookup; `z_init` and the initial values leave the key; `main.py`'s existence queries pass the parent; `_drop_order`. Closes `[20-wkb-rows-consume-numeric-initial-data]`. | **written** 2026-09-26; after 05 and 06 |
| 08 | `08-key-quadsourceintegral-on-its-policy.md` | `QuadSourcePolicy` in `QuadSourceIntegral`'s key, with a foreign key, through factory, target, `main.py` and the extraction script. Closes `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`. | **written** 2026-09-26; after 07 |
| 09 | **Split 2026-09-30 into 09a–09c** by the user, one session, one commit each ([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md)). Charter unchanged, in [`09-key-derived-rows-on-their-parents.md`](09-key-derived-rows-on-their-parents.md): `QuadSource` on `Tq`/`Tr`, `QuadSourceIntegral` on its source, policy data and Tk rows, `GkSource` on its Gk rows; same-shard parents as foreign keys, cross-shard as indexed columns; the inventory resolves parents across shards. | **written** 2026-09-26; split 2026-09-30 |
| 09a | [`09a-key-quadsource-on-its-transfer-functions.md`](09a-key-quadsource-on-its-transfer-functions.md) | K4, and K7's cross-shard `Parent`: `Tq_serial` and `Tr_serial` in `QuadSource`'s key, `Tr_serial` indexed; the `QuadSource` stage and the `QuadSourceIntegral` stage's `QuadSource` lookup keyed on them. | **written** 2026-09-30; after 08 |
| 09b | [`09b-key-gksource-on-its-parent-set.md`](09b-key-gksource-on-its-parent-set.md) | K6, and K7's set-valued parent: `GkSource_parents` and `parents_digest`; the Gk value reads hand back their parent; all six `GkSource` lookups in `main.py` keyed on the set. Closes `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]`. | **written** 2026-09-30; after 09a |
| 09c | [`09c-key-quadsourceintegral-on-its-inputs.md`](09c-key-quadsourceintegral-on-its-inputs.md) | K5: `QuadSourceIntegral` keyed on its source, its policy data and its four Tk rows, with the relations between them asserted; its stage resolves every work item's parents before the existence query. | **written** 2026-09-30; after 09b |
| 09d | [`09d-refer-to-derived-parents-by-label-in-the-inventory-report.md`](09d-refer-to-derived-parents-by-label-in-the-inventory-report.md) | The inventory report refers to every derived (sharded) parent, and every `GkSource` parent set, by a label, printing each row and set once, so record lines stop growing with sets and chains of parents while distinct records stay distinct lines. Closes `[09b-inventory-report-renders-every-member-of-a-parent-set]`. *(Added 2026-10-01, by the user's decision on the review of 09c.)* | **landed** 2026-10-01, reviewed |
| 10 | [`10-build-a-throwaway-store-end-to-end.md`](10-build-a-throwaway-store-end-to-end.md) | Remedial. A scoped build through the registry into a store under `var/datastores/` that is not A3 v2; then a resume; then extraction of every target from it, the fingerprint, and the census of audit §4.4. Also amends `test_inventory_report.test_verbose_renders_every_distinct_record_as_a_distinct_line` to compare record lines with their own label `  #<n>` removed, since 09d that label makes every line of a labelled class distinct whatever its fields; closes `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]`. *(Added 2026-10-01, by the user's decision on the review of 09d.)* Closes the campaign. | **landed** 2026-10-02, reviewed |

**Why 05–10 are held.** Their charters above are fixed and cannot drift to fit what 01–04 find.
What waits is their method. 05's size depends on D2 and D3, and it is written against the list 04's
guard produces. 06–09 depend on D5, the rule for deliberately changing the schema witness and the
golden fingerprint, and 07–09 on D4. 09 exists only if D1 is taken. 10 is written last, against what
01–09 ship.

**Order.**
- 01 → 02 → 03: 02 repairs what 01 records, and 03 changes lookups on the classes 01–02 replicate.
- 04 is independent of 01–03 and may run beside them; 04b follows 04, and 05 follows 04b, as
  05a → 05b → 05c. 05's prompts touch no factory file 02 or 03 touches, but neither pair is
  dispatched concurrently in one checkout.
- 06 → 07 → 08 → 09: each changes the schema, and one change at a time keeps each revertible. 08
  and 09 both change `QuadSourceIntegral`'s key; 08 is the decided one and goes first.
- 06 follows 03, whose two `stepping_mode` flags it corrects. 07 follows 05 as well as 06, because
  05 and 07 both edit the WKB factories' read paths, and 09 relies on 05's value reads refusing more
  than one parent per `z_source`. *(Added 2026-09-26, when 06–09 were written.)*
- 09a → 09b → 09c: `QuadSource`, then `GkSource`, then `QuadSourceIntegral`, so that a row's key
  reaches its parents before anything downstream is keyed on it. *(Added 2026-09-30, when 09 was
  split.)*
- 10 last.

**Across campaigns (D4).** `prompts/store-retirement` is live in the same tree. Two constraints:
- **`ShardedPool.py`.** Its prompt 01 is adding `delete_store` to it now. Nothing of this campaign
  is dispatched while another campaign's uncommitted change to `ShardedPool.py` is in the tree, and
  every prompt here cites that file by function name, not line number.
- **Fingerprints.** Its prompt 03's `retire` refuses unless a store's recorded fingerprint matches
  one recomputed at the current tree. Prompts 07–09 change the fingerprint's identity and add
  columns old stores lack, so after them the tree cannot recompute a matching fingerprint for any
  existing store. **So 07–09 are not dispatched until `store-retirement` prompt 05 has retired the
  sweep store and the backup**, unless the user decides otherwise under D4. 01–06 do not move the
  fingerprint.
- **Discharged 2026-09-26.** `store-retirement` closed at 6 / 6 (`1e37a8a`): its `ShardedPool.py`
  work is committed, and its prompt 05 retired the sweep store and the backup (`81c0a9a`). Neither
  constraint now holds anything back. §5 rule 13 stays, for any later campaign.

## 3. Datastores

Three stores live under `var/datastores/`: the live A3 baseline, the sweep store and the backup.
All three were read `mode=ro` by the audit's census and are clean serial by serial (audit §4.4).

**Prompts 01–09 never open anything under `var/`.** Their tests build stores in temporary
directories or in memory. **Prompt 10 is the one exception**, and it writes only a new store of its
own, under a name no other store or run uses, and never A3 v2's.

**After 07–09, no existing store is readable by the tree** for the targets whose keys changed. Each
stays readable at the git SHA it was written under, and its sidecar records that SHA. This is
accepted under D4, and it is why §2 orders `store-retirement` first.

**Before touching any store,** run `python -m RunRegistry list` and confirm that nothing is
`running`.

## 4. The interfaces between prompts

The **names** below are fixed here, so that each prompt can be written against the one before it.
The internals are the implementing prompt's to design.

- **Prompt 01** ships:
  - a table **`replication_in_flight`** in the **primary** file, holding at most one row: the
    operation (`get`, `store` or `validate`), the class name, the controlling shard's id, the
    `store_id` once known, and the start time. It is committed before the controller's call and
    deleted after the last replica's call returns;
  - **`ShardedPool._replicated_write(...)`**, or an equivalently named single helper, through which
    all three replicated paths go, so that the record is written in one place;
  - a per-call **timestamp override** that `Datastore._insert` uses in place of `datetime.now()`;
  - the exception **`ReplicationMismatch`**, raised when a replica's answer differs from the
    controller's, naming the shard, the class and both serials.
- **Prompt 02** ships:
  - **`ShardedPool._reconcile_replicated_tables()`**, run by the constructor after
    `_check_shard_files` and before any actor is created, needing no Ray;
  - the exception **`ReplicatedDivergence`**, raised to refuse an open, naming the class, the shard
    and the serial of each difference;
  - the check's result recorded in the pool, as **`ShardedPool.reconciliation`**, a small dict
    saying what was compared, what was repaired and from which controller.
- **Prompt 04** ships **`KNOWN_UNFIXED`** in `test_factory_select_columns.py`: every defect its
  extended guard finds, each as (module, function, attribute), which prompt 05 must empty.
- **Prompt 07** ships the column **`numeric_serial`** on `TkWKBIntegration` and `GkWKBIntegration`.
- **Prompt 08** ships the column **`quad_policy_serial`** on `QuadSourceIntegral`.
- **Prompt 09** ships, as recommended (its §2 K6 allows a justified alternative), the table
  **`GkSource_parents`** and the key column **`parents_digest`** on `GkSource`; four columns on
  `QuadSourceIntegral` naming its `Tq`/`Tr` numeric and WKB rows, named by 09 and added here when it
  lands; and a cross-shard form of the inventory's `Parent`. *(Added 2026-09-26.)* Since the split
  (2026-09-30): 09a ships the cross-shard `Parent`; 09b the table, the column and a set-valued
  inventory parent; 09c the four columns, **`Tq_numeric_serial`**, **`Tq_WKB_serial`**,
  **`Tr_numeric_serial`** and **`Tr_WKB_serial`** (NOT NULL, indexed, no foreign key; landed
  2026-09-30).

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit**, plus `docs/OPEN_ISSUES.md`,
   plus the owning board of any issue it closes.
4. **Do not fix things the prompt did not ask for.** Record them, and open a §3 issue.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified**, except where a prompt names the test and the reason; D5
   governs the two witnesses. A prompt may add tests beside them.
8. **Deliberate breakage.** Each prompt names mutations its tests must catch, and the log records
   each as a diff, exactly as applied, so the orchestrator can replay it with `git apply`.
   Mutations are never committed.
9. **No test needs Ray, and no test opens anything under `var/`.** Stand-in shards follow
   `docs/datastore-integrity-audit/replicated_write_fault_probe.py`: actor classes unwrapped through
   `__ray_metadata__.modified_class`, `.remote()` a direct call, faults injected per actor and
   method, patches scoped with `unittest.mock.patch`.
10. **No agent writes a store under `var/`**, except prompt 10's own new store.
11. **A schema change is never costed** in rows, migration or regeneration. There is nothing to
    rebuild. What is costed is reader breakage: every script that stops working is named in the log.
12. **`main.py` edits keep `quadsource_atol_sweep.py` working.** That script `exec`s `main.py` with
    five textual substitutions, each required to match exactly once
    (`substitute_main_py`, `docs/handover/quadsource_atol_sweep.py:337`). A prompt that edits
    `main.py` runs `substitute_main_py()` afterwards and shows every substitution still matches once.
    It never edits the sweep script to make that true.
13. **`ShardedPool.py` is shared with `store-retirement`.** Cite it by function name. Do not start
    while another campaign's change to it is uncommitted.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt**.

## 6. Decisions

### 6.1 Made by the user (2026-09-25), and not reopened here

1. **The replicated write is made atomic.** The audit's recommended form (§4.5) is taken: checked,
   idempotent replication, with repair or refusal at open.
2. **`QuadSourcePolicy` is kept and added to `QuadSourceIntegral`'s key.** It is a shell today and
   may gain settings; if kept, it is treated correctly.
3. **The exit-time tolerance lookup becomes an exact match.**
4. **The adjacent issues are out of scope** (§1), as downstream work.

### 6.2 Decisions D1–D6

Each was stated on 2026-09-25 with a recommendation. **The user decided D2–D6 on 2026-09-26**, as
recorded under each, and D1 later the same day, once it had been explained. The recommendations as first
written are kept under each, since they are what the user decided against.

- **D1 — key derived rows on the rows they were computed from (audit §2.4)? Decided: yes.**
  The user asked, 2026-09-26, whether the gist is that rows on another shard cannot carry explicit
  foreign keys. It is not; see *What D1 is about* below. Once that was explained, the user agreed
  with the recommendation the same day, noting that the likely original reason for the narrow key
  was to let a `QuadSourceIntegral` be looked up from `k`, `q`, `r` alone, without first looking up
  the intermediate Tk objects — and that the result is inconsistent. **So prompt 09 must say what
  becomes of that convenience.** A reader that holds only physical labels (`k`, `q`, `r`, the
  tolerances and the tags; `extract_QuadSourceIntegral_data.py` is the case) may still select by
  them, provided it **refuses when more than one stored row matches**, naming the parents that
  distinguish them, rather than choosing one. `build()`, which decides whether to compute, keys on
  the parents.
  - *As recommended on 2026-09-25:* the user asked for every known integrity defect, and this is
    the class the WKB issue belongs to: a row computed from specific upstream rows is served for any
    upstream rows. Same-shard parents (`QuadSource.Tq_serial`, `QuadSourceIntegral.data_serial`,
    the Tk rows on `k`'s shard) join the key with foreign keys; cross-shard parents (`Tr_serial`,
    `QuadSourceIntegral.source_serial`) join it as indexed columns without one, and the inventory
    learns to resolve them across shards so they can be fingerprint identity. *Alternative:* rely on
    building every archival store once, from scratch, and record the class as an issue. That holds
    for A3 v2 as planned, and fails the first time an archival store is resumed or extended.
  - **What D1 is about.** It is about the **lookup key**, not the foreign key. When `main.py` asks
    "is there already a `QuadSource` for this model, `q` and `r`?", the query
    (`QuadSource.py:147-164`) names only those three and the tags. It does not say which Tk rows
    the stored source was computed from, although the row records them in `Tq_serial` and
    `Tr_serial`. So if the Tk rows it would be computed from today are different rows — a changed
    Tk tolerance, grid or break-point policy, under the same run tags — the old `QuadSource` is
    served as if it matched. `QuadSourceIntegral` (on its source, its `GkSourcePolicyData` and its
    four Tk rows) and `GkSource` (on its Gk rows) are the same. The fix is to add the parents'
    serials to each lookup. The shard question is secondary: a parent on the **same** shard can
    also be declared as a foreign key; one on **another** shard cannot, but serials are unique
    across shards, so it can still be an indexed key column with no declared foreign key. The only
    real cost of the cross-shard ones is in the fingerprint: its inventory resolves a parent only
    within one shard, so it would have to learn to look across shards (item K7).
- **D2 — fix the latent read-path defects no caller reaches today (audit §3.10–§3.11)? Decided:
  yes.** The user, 2026-09-26: they must be fixed, as part of what this campaign is for.
  *As recommended:* yes. *Alternative, not taken:* fix only what a live path reaches.
- **D3 — how much of `OneLoopIntegral`'s factory to repair (audit §3.12)? Decided: the mechanical
  defects only.** The user, 2026-09-26: the rest "come about because `OneLoopIntegral` hasn't been
  implemented yet, not that it has been implemented wrongly", and are logged and resolved when it is
  built. So prompt 05 fixes the wrong tag table, the tagged join, `read_batch` made a method with a
  return value, and tags defaulting to `[]`. `[00-oneloop-factory-cannot-round-trip-a-row]` records
  the rest, for when `OneLoopIntegral` is built. *As recommended:* the same. *Alternative, not
  taken:* leave the factory alone.
- **D4 — existing stores, and order against `store-retirement`. Decided: readers are not made
  version-aware.** The user, 2026-09-26: that "makes the datastore layer much larger than we want".
  So after 07–09 no existing store is readable by the tree for the changed targets, and each stays
  readable at the SHA it was written under. The ordering half was discharged the same day, when
  `store-retirement` closed (§2). *As recommended:* the same. *Alternative, not taken:* a
  version-aware fingerprint and readers.
- **D5 — deliberately changing the schema witness and the golden fingerprint. Decided: as
  recommended.** The user approved it, 2026-09-26. The first schema-changing prompt (06) amends
  `test_schema_builder.py` to compare against the witness for the **current** schema design; each
  schema-changing prompt then adds a new witness beside the old ones —
  `schema_at_<campaign>-<NN>.json`, never overwriting one — repoints the test, and puts the diff
  between the two witnesses in its log as the reviewable statement of what changed. The golden
  fingerprint is re-captured the same way, and never to make a test pass that the prompt did not
  expect to fail. *Alternative, not taken:* regenerate both in place and rely on git history.
- **D6 — which `QuadSourcePolicy` each `QuadSourceIntegral` gets. Decided: `QuadSource_policy_1pt5`
  only, for now.** The user, 2026-09-26. It pairs with the `GkSource_policy_1pt5` every site uses
  today (audit §2.5); nothing reads the policy's fields, so it changes the record and no number. The
  unused 5.0 pair stays as `[00-the-5pt0-policy-pair-is-created-and-never-used]`, not removed.
  *As recommended:* the same. *Alternative, not taken:* key on both.

## 7. Baselines

Measured 2026-09-25 while writing this campaign, with `store-retirement` prompt 01 uncommitted in
the tree:

| Suite | Result |
|---|---|
| RunRegistry | 128 OK |
| Datastore | **not measurable**: 206 run, 60 failing, all in `store-retirement`'s uncommitted `test_delete_store.py` |

### 7.1 Re-measured 2026-09-26, at `3964bf8`, clean

`store-retirement` had closed, so the tree was clean.

| Suite | Result |
|---|---|
| Datastore | 206 OK |
| RunRegistry | 199 OK |

The other suites were not re-measured; the orchestrator measures them before the first dispatch.

Re-measure every suite at `HEAD`, clean, before every dispatch, with:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . 2>&1 | tail -40
```

The suites print banners, so `| tail -5` will not show the verdict. Do not set
`THREE_BESSEL_DIAGNOSTIC_PLOTS`. `ComputeTargets`' `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
is a known wall-clock flake.
