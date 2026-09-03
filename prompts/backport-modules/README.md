# Backport campaign: shared Ray/Datastore infrastructure modules

**Source document:** [`docs/backport-modules-audit.md`](../../docs/backport-modules-audit.md)
**Planned:** 2026-09-03
**Target branch:** `main` (clean at `79f0360` when this plan was written)
**Status board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)

---

## 1. What this campaign does

`SecondaryGWKit` (`SGWK`) is the oldest lineage of six reusable Ray/Datastore modules that were
later carried into `ChamPBH` (`CPBH`) and `StochasticInstantons` (`SI`). The audit established that
`SI` is a strict superset of `CPBH` for all six modules, so `SI` is the single upstream reference.

`SGWK` carries four confirmed defects (two of which make currently-committed code paths
unrunnable), plus a handful of smaller latent faults. This campaign backports the fixes, replaces
one broken API with the upstream redesign, takes one upstream API extension (**E1**), and builds out
the datastore-contents reporting service (**F2**).

Ten prompts, each landing exactly one commit, each independently revertible.

> **Scope note.** The audit recommended deferring F2 and treating E1 as discretionary. **The user
> has since asked for both.** E1 is prompt 05; F2 is prompts 06–09. The reasoning that led the audit
> to defer F2 was sound — the plumbing is inert without ~28 per-factory methods, which is a feature
> project rather than a reconciliation — so it is planned as four prompts rather than folded into
> one.

---

## 2. Reconciliation of the audit against the codebase

Before this plan was written, every finding in the audit was checked against the working tree at
`79f0360`. **All of them reproduce at the stated locations**, and all the stated call-site counts
are exact:

| Audit claim | Verified |
|---|---|
| B1 — `{"key_id": …}` insert vs. `key_serial` PK column | ✅ `ShardedPool.py:251`, `ShardedPool.py:800-803` |
| B2 — `row.key_attr` on a `key_type`-only select | ✅ `ShardedPool.py:334-337`, three uses at `342`, `344`, `349` |
| B3 — `for method_name, method_config in read_table_config:` over a dict | ✅ `Datastore.py:228`, `ShardedPool.py:190` |
| B4 — no `None` branch in the task-builder dispatch | ✅ `RayWorkPool.py:285-297` |
| B5 — no intra-batch dedup in `_assign_shard_keys` | ✅ `ShardedPool.py:772-781` |
| F6 — unguarded `sharded_tables` insert | ✅ `ShardedPool.py:303-308` |
| E1 — 35 `store_handler=None` sites across 7 files; 45 `RayWorkPool(` constructions | ✅ exactly 35 and 45 |
| B3 — 17 `read_*_table` call sites across 6 scripts | ✅ exactly 17 (+2 config lines = 19 grep hits) |
| D1 `self._db_file` / D2 `raise print(...)` / D3 `_default_serial_batch_size[table]` / D4 `_num_store_complete` | ✅ all four |
| F2 — zero factories expose `inventory` | ✅ `grep "def inventory" ObjectFactories/*.py` → no matches |
| Only `redshift` and `wavenumber` factories expose `read_table` | ✅ |

### 2.1 Two things the audit did not report

**(a) The audit's §2.2 open question is answered by the on-disk evidence.** The audit asks a
planning agent to confirm with the user whether `main.py` / `extract_*.py` have been run against an
existing sharded datastore since 2025-12-15, because if they had, the B2/B3 analyses would have to
be wrong. The datastore checked into the working tree settles this without needing to ask:

```
test-qcd-db.sqlite:  tables = ['shards', 'shard_keys']
                     shard_keys PK column = "wavenumber_serial"   (50 rows)
```

That is the **pre-`a2bd966` schema**. There is no `shard_key_config` table, no `replicated_tables`
table and no `sharded_tables` table, and the shard-key PK is the old hardcoded `wavenumber_serial`
rather than the generic `key_serial`. So the most recent datastore on disk predates the refactor
that introduced B2 and B3, which is consistent with — and strong evidence for — the conclusion that
those two paths have never been exercised. **The B2 and B3 findings stand as written; no
re-derivation is needed.**

Two consequences flow into the plan:

- The B1 data audit is a **forward-looking detector**, not a rescue operation. There is no
  new-schema `SGWK` datastore in the tree that could already be corrupt. Prompt 01 still ships the
  auditor (it is the only cheap ongoing detector, and the user may hold datastores outside the
  repo), but it must not be scoped as a repair of known damage.
- Any pre-existing `SGWK` datastore has to be rebuilt regardless of B1, because its schema is
  simply not readable by the current code. The "remedy is rebuild" advice in the audit is therefore
  unconditional, not a B1-specific concession.

**(b) B3 has a fourth fault the audit does not list.** The audit enumerates three faults in the
`read_table_config` block (dict iteration, unbound `setattr`, late-binding closure). There is a
fourth, in the consumer rather than the generator: `Datastore._generic_read_table` already passes
`tables=self._tables` unconditionally (`Datastore.py:761`), while the generated wrapper *also*
injects `kwargs["tables"] = self._tables` when `tables_arg` is set (`Datastore.py:230-232`). Even if
the first three faults were fixed, a `tables_arg: True` class would raise
`TypeError: got multiple values for keyword argument 'tables'`. The upstream redesign removes this
by construction (`factory.read_table(conn, tab, *args, **kwargs)` with `tables` injected only via
`kwargs`), so no extra work is needed — but the implementer should know the current code is broken
in four places, not three, and should not be surprised when a partial fix still fails.

### 2.2 Deviations from the audit's recommendations

The plan follows the audit's recommendation column with three deliberate departures, all noted in
the prompts that carry them:

1. **F3 is folded into prompt 03, not left dangling.** The audit marks it "optional, fold into
   whichever commit touches `Datastore.py` if convenient" and warns against creating an
   inconsistency by fixing `Datastore.py` while leaving `ShardedPool.py:155`. Prompt 03 either does
   both or neither, and defaults to doing both.
2. **D2's suggested remedy is narrowed.** The audit offers "simplest fix is to delete it". Prompt 02
   deletes the unreachable `elif num_config > 1:` branch rather than repairing it, because the
   `num_config > 1` check immediately below already raises correctly — but the prompt requires the
   agent to confirm that reachability claim before deleting, not to take it on trust.
3. **A verification prompt is added.** The audit's §8 checklist spans every code commit and cannot
   be discharged inside any one of them. Prompt 10 collects it, runs everything that can be run
   without a long compute campaign, and hands the user an explicit list of what still needs a real
   pipeline run.
4. **F2 is implemented rather than deferred, at the user's request** — see the scope note in §1. Two
   corrections to the audit's F2 analysis are carried into prompt 06: one thing it flags is not
   actually a defect, and one defect it does not mention is forced by `SGWK`'s schema. Both are in
   §2.3 below.
5. **E1 is confirmed wanted**, so prompt 05 no longer asks the user before proceeding.

Items the audit says to skip are skipped: **X1**, **X2**, **X3**, **X4** are not backported, and
each prompt that touches nearby code carries an explicit "do not change this" note so a
well-meaning agent does not drift into them. **F4** (licence headers) remains out of scope.

### 2.3 Two corrections to the audit's F2 analysis

Both were established by inspecting `SI`'s implementation directly, and both are carried into
prompt 06.

**(a) The defect the audit reports is not there.** Audit §4 F2 says `SI`'s `ShardedPool.inventory`
"calls `self._inventory_config[class_name]` without first checking `self._inventory_config is not
None` in the label-merge branch". It does check — `SI/ShardedPool.py:958` raises when the config is
`None`, before either merge branch runs. There is a real fault a few lines later, but it is a
different one: `self._inventory_config[class_name][label]` raises a bare `KeyError` when a factory
returns a label the config does not mention, which is a failure mode prompt 08 makes easy to hit and
worth a proper message.

**(b) The defect the audit calls hypothetical is mandatory.** The audit notes in passing that
`_merge_queue`'s policy dispatch "falls through to a `RuntimeError` for any type not in
`{list, set, datetime, None}` (e.g. an `int` count field)" and suggests tightening it "on the way
in". For `SGWK` this is not a tidy-up — it is a blocking requirement. Six of the 15 sharded tables
are value tables (`TkNumericValue`, `TkWKBValue`, `QuadSourceValue`, `GkNumericValue`,
`GkWKBValue`, `GkSourceValue`), all registered `"timestamp": False`, so they have no timestamp
column and their only meaningful inventory is a row count — which must be **summed** across shards.
Upstream's policy vocabulary cannot express that. Prompt 06 therefore adds numeric merge policies
that do not exist upstream, and prompt 10 checks a real count against hand-summed SQL, because a
`"latest"`-style bug applied to a count would report one shard's value and look entirely plausible.

A third, smaller structural difference: `SGWK` registers factory **classes** with `@staticmethod`s
while `SI` registers **instances** with instance methods (audit §5 **X4**, deliberately not
backported). Every upstream `inventory` method therefore takes `self` and cannot be copied across
unaltered. Prompts 07 and 08 both call this out, because it is a silent failure — a stray `self`
binds `conn` and the error surfaces far from its cause.

---

## 3. The prompts

| # | Prompt | Items | Files touched | Risk |
|---|---|---|---|---|
| 01 | [`01-shard-key-persistence.md`](01-shard-key-persistence.md) | B1, B5 | `ShardedPool.py`, new `tools/shard_key_audit.py` | Low code risk, critical correctness value |
| 02 | [`02-shard-config-reader.md`](02-shard-config-reader.md) | B2, D2 | `ShardedPool.py` | Low |
| 03 | [`03-robustness-fixes.md`](03-robustness-fixes.md) | F6, B4, D1, D3, D4, F3 | `ShardedPool.py`, `RayWorkPool.py`, `ClientPool.py`, `Datastore.py` | Low |
| 04 | [`04-read-table-service.md`](04-read-table-service.md) | B3 | `Datastore.py`, `ShardedPool.py`, `config/sharding.py`, 6 `extract_*.py` | **High blast radius** |
| 05 | [`05-persist-handler-split.md`](05-persist-handler-split.md) | E1 | `RayWorkPool.py`, `main.py`, 6 `extract_*.py` | **Source-incompatible** |
| 06 | [`06-inventory-plumbing.md`](06-inventory-plumbing.md) | F2a | `Datastore.py`, `ShardedPool.py` | Low — additive, inert until 07/08 |
| 07 | [`07-inventory-replicated-factories.md`](07-inventory-replicated-factories.md) | F2b | 13 replicated factories | Low — additive |
| 08 | [`08-inventory-sharded-factories.md`](08-inventory-sharded-factories.md) | F2c | 15 sharded factories, `config/sharding.py` | Medium — config/factory drift |
| 09 | [`09-inventory-reporting.md`](09-inventory-reporting.md) | F2d | `main.py` or `tools/inventory_report.py` | Low — first end-to-end exercise |
| 10 | [`10-verification.md`](10-verification.md) | audit §8 + F2 | verification scripts + docs only | None |

Prompts 06–09 are the F2 (`inventory()`) sub-campaign. They are split four ways because the work
has three genuinely different characters: pool/datastore plumbing (06), ~28 repetitive per-factory
query methods (07, 08), and a formatting/reporting layer that is the first thing to exercise any of
it (09). Splitting also keeps each commit reviewable — a single commit adding 28 factory methods
plus plumbing plus a report would not be.

## 4. Dependencies and ordering

```
01 (B1+B5) ─► 02 (B2) ─► 03 (small fixes) ─► 04 (B3) ─┬─► 05 (E1) ────────────────────┐
     │                                                 │                               │
     └── B5 is a hard prerequisite for B1              └─► 06 (F2 plumbing)            ├─► 10 (verify)
                                                             │                         │
                                                             ├─► 07 (replicated) ──┐   │
                                                             │                     ├───┘
                                                             └─► 08 (sharded) ─────┴─► 09 (report)
```

**Hard dependencies**

- **B5 before (or with) B1.** Once B1 makes the INSERT actually bind the primary key, a duplicated
  shard-key object in one batch turns a previously silent double-INSERT into a primary-key
  violation. They must land together; prompt 01 does both in one commit.
- **02 before 04.** Prompt 04 requires reopening an existing sharded datastore to verify anything,
  and `_read_shard_data` cannot complete until B2 is fixed. Doing 04 first leaves it unverifiable.
- **04 before 06.** `read_table` and `inventory` are near-identical dispatch methods sitting side by
  side in both classes. Writing `inventory` against a settled `read_table` keeps them consistent;
  writing it first means matching a shape that is about to change.
- **06 before 07 and 08.** Both implement against the factory signature and merge-policy vocabulary
  that 06 fixes.
- **07 before 08.** 08 reuses the labelled compute-target return shape that 07 establishes for
  `BackgroundModel`, and must match its field names exactly.
- **06, 07 and 08 before 09.** The report formats what they return.

**Soft dependencies**

- **04 before 05.** Both touch the same six `extract_*.py` scripts. Either order works, but doing
  the semantic change first means 05's mechanical 35-site edit lands on settled call sites, and a
  revert of 05 does not disturb 04.
- **05 anywhere after 04.** E1 is independent of the whole F2 sub-campaign. It is placed at 05 so
  the backport proper (01–05) finishes before the feature work (06–09) starts, which keeps the
  revert story clean: reverting all of F2 is `git revert` of four contiguous commits.

**Independence**

- Prompt 03's six items are mutually independent and independent of everything else. It is placed
  third only because it is cheap and clears the file of distractions before the two large changes.
- 07 and 08 could in principle run in parallel, but 08 depends on 07's field names, so in a
  fresh-context campaign it is simpler to keep them sequential.

**Recommended ordering: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10.**

**Natural stopping points.** The campaign has three, if the user wants to pause and assess:

| After | State |
|---|---|
| **03** | All pure bug fixes landed. No API surface change, no call-site churn. Minimal risk. |
| **05** | The backport proper is complete — everything the audit recommended, and nothing else. |
| **09** | F2 delivered; the datastore can report its own contents. |

04 is not really optional in practice: the `extract_*.py` scripts cannot construct a pool without
it.

## 5. Rules that apply to every prompt

Each prompt restates these, but they are collected here so the campaign's invariants are visible in
one place.

1. **One commit per prompt.** Do not amend or squash across prompts. The commit boundary is the
   rollback boundary.
2. **Commit message format** matches this repository's existing convention: an imperative,
   capitalised subject line under ~72 characters with no prefix tag; a blank line; a prose body
   explaining *why*, wrapped at ~80 columns; and the trailer
   `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
3. **Every prompt writes a log** to `prompts/backport-modules/logs/NN-<name>.md` using the template
   in §5.1 below, and the log is included in that prompt's commit.
4. **Every prompt updates** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) in the same commit.
5. **Do not fix things the prompt did not ask for.** If you spot something, record it in the log's
   "Observations not acted on" section and leave the code alone. Scope creep destroys the
   revert-per-prompt property.
6. **Do not backport X1, X2, X3 or X4** under any circumstances. See audit §5.

### 5.1 Log format (mandatory)

The log has to be good enough that a later reader can tell what shipped, and *why it differs from
the prompt*, without re-deriving anything from the code. Every deviation must be classified:

- **Structurally required** — the prompt could not be implemented as written (the code was not
  shaped as the prompt assumed, a name differed, an ordering constraint forced a change). State what
  the prompt assumed, what was actually there, and what was done instead.
- **Implementation choice** — the prompt left it open and the agent picked. Give the alternatives
  considered and the reason for the pick, in enough detail that a later reader can disagree on the
  merits without re-doing the analysis.
- **Unintended drift** — noticed after the fact, not deliberate. Say so plainly and say whether it
  was reverted or kept.

Template:

```markdown
# Log NN — <prompt title>

**Prompt:** prompts/backport-modules/NN-<name>.md
**Commit:** <sha> — <subject>
**Date:** <YYYY-MM-DD>
**Result:** COMPLETE | COMPLETE WITH DEVIATIONS | PARTIAL | BLOCKED

## What shipped
<Per item: file:line before → after. Enough that a reader knows the change without opening the diff.>

## Deviations from the prompt
<One subsection per deviation, each tagged STRUCTURALLY REQUIRED / IMPLEMENTATION CHOICE /
UNINTENDED DRIFT. "None" is an acceptable and expected answer.>

## Verification performed
<Exactly what was run and what it printed. Distinguish "I ran this and it passed" from
"I reasoned that this is correct" from "this needs a pipeline run the user must do".>

## Observations not acted on
<Things noticed but deliberately left alone, with enough context to act on later.>

## State handed to the next prompt
<Anything the next prompt needs to know that is not already in its own text.>
```

---

## 6. Deferred and out of scope

Both F2 and E1 have moved into scope at the user's request; what remains is:

- **F4 — Apache-2.0 licence headers.** `SGWK` carries none anywhere. If wanted, apply repo-wide in a
  separate commit; do not apply to six files only.
- **Report X1 upstream.** `SI`'s `object_read_batch` guard was converted to object semantics
  (`hasattr(shard_key, shard_key_field)`) while the two following lines stayed on dict semantics
  (`shard_key[shard_key_field]`, `payload.update(shard_key)`). Neither a dict nor an object can pass
  through it. `SGWK` is internally coherent and is left alone, but this is a live break in the
  `StochasticInstantons` tree worth telling its maintainer about.
- **Two upstream `_merge_queue` weaknesses, if F2 is ever pushed back upstream.** Prompt 06 fixes
  both on the way in: `merge_queue.pop()` mutates the caller's list, and `"extend"` mutates the
  shard's returned dict in place. Harmless in the current call pattern, but they make the function
  unsafe to reuse.

### F2 scope, as planned

The audit estimated F2 at "~15 sharded and ~14 replicated classes" of per-factory work. That is the
full theoretical surface and it is what prompts 07 and 08 cover — but it is worth knowing that
**upstream did not do all of it**: `SI` has 17 factories with an `inventory` method out of a much
larger set, and its `inventory_config` configures exactly four sharded classes. Incremental scoping
is legitimate. If any class turns out to be more trouble than it is worth, the right move is to
leave it without an `inventory` — the service already raises a clear "does not provide an inventory
service" error, and prompt 09's report is required to survive it — and record the omission as a
deliberate choice in the log.

What is *not* optional: any class that gets an `inventory` and is sharded must also get an
`inventory_config` entry that matches it field for field. A half-configured class fails at report
time, not at import time.
