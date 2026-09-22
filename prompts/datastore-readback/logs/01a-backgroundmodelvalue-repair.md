# Log 01a — Repair the `BackgroundModelValue` factory's two dead-code defects

**Prompt:** none. Direct user instruction, in session, immediately after prompt 01 landed:
*"Can you also fix both of those failures in the `BackgroundModelValue` factory. They look like
one-line fixes: `Hubble` -> `Hubble_GeV`, and the correct insert keys are presumably inferable
from the `BackgroundModel` write path."*
**Commit:** *(this commit)* — "Make the BackgroundModelValue factory's build path work"
**Model:** Claude Opus 5
**Date:** 2026-09-22
**Result:** COMPLETE — **neither fix was one line**, for reasons given below

## What shipped

**`Datastore/SQL/ObjectFactories/BackgroundModel.py`**, in
`sqla_BackgroundModelValue_factory.build()`:

- `:964` `"wkb_serial"` → `"model_serial"`. One line, and the user's inference from the
  `BackgroundModel.store()` write path (`:664-686`) was exactly right: that payload and this one
  are the same eighteen keys, and `model_serial` is the only one that differed.
- the two consistency checks moved from below the block that replaces the payload values with the
  stored ones to above it, and the `Hubble` one rewritten to compare in the stored GeV
  representation with a **relative** bound.

**`Datastore/tests/test_backgroundmodelvalue_roundtrip.py`** (new, 6 tests) — the in-memory round
trip, which is prompt 01 §3's shape **(b)** applied to the one factory that needed it. The table is
built from the factory's own `register()` output, so the schema under test is the production
schema; the two foreign-key targets are stubs, because SQLite does not enforce foreign keys unless
`PRAGMA foreign_keys` is on and the production `Datastore` does not turn it on. No Ray, no
datastore file.

**`Datastore/tests/test_factory_select_columns.py`** — `KNOWN_UNFIXED` emptied, with a note saying
what it held and when it was struck.

Boards: the issue closes on `prompts/GkTk-remedial`'s §4 (see below), the duplicate is retired on
this campaign's §4, and `docs/OPEN_ISSUES.md` loses both rows.

## Why neither was a one-line fix

### The insert key — one line, but not for the stated reason

The correction is a single token. What it does is not what the key's spelling suggests.
**SQLAlchemy Core's `insert()` drops a dict key matching no column silently**, rather than raising
— a note already in the tree at `OneLoopIntegral.py:253-257` records exactly this behaviour for a
different key, and it was re-verified here. So `"wkb_serial": model_serial` did not fail as an
unknown column; the key vanished, the row was attempted with `model_serial` unset, and what
refused it was that column's own `nullable=False`:

```
sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError)
NOT NULL constraint failed: BackgroundModelValue.model_serial
```

This matters beyond bookkeeping: **had `model_serial` been nullable, this branch would have written
silently orphaned rows instead of failing.** The insert-key class of defect is not self-announcing,
and the schema is what happened to catch this one.

`wkb_serial` is a real column of the `GkWKBValue` and `TkWKBValue` tables, which is where it was
copied from.

### The `Hubble` read — not a rename

`row_data.Hubble` → `row_data.Hubble_GeV` would have made the code *run* and would have made the
check *meaningless*. The comparison sat below this block:

```python
Hubble = row_data.Hubble_GeV * GeV      # the payload Hubble is gone from here on
...
if fabs(row_data.Hubble - Hubble) > DEFAULT_FLOAT_PRECISION:   # <- was here
```

so by the time it ran, `Hubble` was the stored value. Renaming the attribute in place would have
compared the stored value against itself: a check that passes for every input, including a stored
row that disagrees with the payload by any amount whatever. The sibling `wBackground` check two
lines below is *not* affected, because `wBackground` is never reassigned — which is why the two
looked alike and are not.

Both checks now run above that block, where the payload value still exists.

The `Hubble` comparison is also now **relative, and in the stored GeV representation**, because no
single absolute bound serves this column. Over a production redshift grid the internal-unit Hubble
runs from **2.37e-04 to 9.19e+26**:

| end of grid | internal-unit $H$ | what `DEFAULT_FLOAT_PRECISION = 1e-7` means there |
|---|---|---|
| bottom | 2.37e-04 | a **relative** tolerance of 4.2e-04 — accepts a stored $H$ wrong in its fourth significant figure |
| top | 9.19e+26 | one ulp is 1.37e+11, i.e. **1.4e+18 ×** the bound — the check degenerates to bit equality |

`wBackground` is dimensionless and $O(1)$, so its absolute bound is the right instrument and is
unchanged; it moved only to stay beside its sibling.

**A claim I made and withdrew.** The first version of the comment justified the relative bound by
saying an absolute one "would fire on the round trip through GeV alone". That is false in this
tree: over all 1,740 stored `Hubble_GeV` values on shard 0, the internal → GeV → internal round
trip is **exact in every case**. The argument that holds is the scale argument in the table above,
which does not depend on round-trip luck, and that is what the comment and the test now say.

## Verification performed

### Deliberate breakage, both halves

- **Insert key.** Restoring `"wkb_serial"`: all **6** tests in the new module fail with
  `IntegrityError: NOT NULL constraint failed: BackgroundModelValue.model_serial`. Restored: 6 OK.
- **`Hubble` read.** Restoring `row_data.Hubble` is caught by prompt 01's static guard, which is
  the point of that guard.

### The guard held the fix, unprompted

Fixing the factory made `test_known_unfixed_entries_are_still_present` fail —
*"`BackgroundModel.py:sqla_BackgroundModelValue_factory` no longer falls short of its SELECT: if it
has been fixed, remove its `KNOWN_UNFIXED` entry"* — and it stayed failing until the stale entry
was struck. That mechanism was written in prompt 01 on the argument that a carried defect must not
be allowed to rot; it did its job on the first opportunity, one commit later.

### The three branches

`test_backgroundmodelvalue_roundtrip.py` exercises insert, agreeing read-back and disagreeing
read-back, and asserts that **both** consistency checks fire — neither ever has. It also asserts
the scale property above directly, rather than inferring it from a sampled round trip.

### The write-side sweep

The static guard cannot see insert-key defects: it compares reads against a `SELECT`, and this is
a dict of keys against a column list. Every insert-payload dict literal in `ObjectFactories/` was
therefore swept by hand against its table's columns. **Nothing was found beyond this factory**, with
one known exception: `sqla_OneLoopIntegral_factory.store()` passes a `"validated"` key its table
does not declare, which is already documented in place at `OneLoopIntegral.py:253-257` as
understood, is not `NOT NULL`, and is silently dropped with no consequence. Recorded so the sweep's
result is on the record; **not** opened as an issue, and **not** changed.

### Suites

| Suite | Baseline | This tree | Verdict |
|---|---|---|---|
| `ComputeTargets` | 552 | **552**, OK | unchanged; the `test_wall_time_per_object` flake did not fire this run |
| `CosmologyModels` | 39 | **39**, OK | unchanged |
| `LiouvilleGreen` | 148 (skipped=1) | **148**, OK (skipped=1) | unchanged |
| `AdaptiveLevin` | 32 | **32**, OK | unchanged |
| `Datastore` | 4 | **10**, OK | +6, exactly the tests added |

All five green, no failures anywhere. `black --check` clean over 273 files.

## Deviations from the prompt

There is no prompt; the deviations are from what the user asked for.

1. **The `Hubble` fix is not `Hubble` → `Hubble_GeV`.** — **STRUCTURALLY REQUIRED.** The rename
   alone produces a check that cannot fail. Stated at the top of this log and in the source comment.

2. **The two checks moved, and the `Hubble` tolerance changed from absolute to relative.** —
   **STRUCTURALLY REQUIRED** for the move; **IMPLEMENTATION CHOICE** for the tolerance. Turning on
   a check that has never run, in a form that cannot be satisfied at one end of its own column and
   is worthless at the other, would be worse than leaving it broken.

3. **A new test module was added, which the user did not ask for.** — **IMPLEMENTATION CHOICE.**
   This change takes a `build()` from never-executed to executable; shipping that with no test is
   not defensible, and the issue's own recorded "next step" — written in 2026-09-11 — asked for
   exactly this test. Six tests, 0.06 s, no Ray, no datastore.

4. **The issue closes on the `prompts/GkTk-remedial` board, not this one.** — **STRUCTURALLY
   REQUIRED** by `CLAUDE.md`: an issue owned by another board is moved to that board's §4.

5. **Prompt 01's `[01-backgroundmodelvalue-hubble]` is retired as a duplicate.** — **UNINTENDED
   DRIFT, in prompt 01.** See below.

## Observations not acted on

1. **Prompt 01 opened a duplicate issue.** The defect was already indexed, as
   `[03-backgroundmodelvalue-build-path]`, opened by `GkTk-remedial` prompt 03 on **2026-09-11** —
   with both halves named, both failure modes named correctly, and "a two-line fix in its own
   commit, with a test that exercises `build()` against an in-memory SQLite store" as its next
   step. Prompt 01's audit re-found the read half and opened a new row **without reading the index
   that exists to prevent exactly this**. Nothing is wrong in the tree as a result — both rows
   described the same real defect, and it is now fixed and both are closed — but prompt 01's log
   and board entry claim the audit *found* something that was on the record for eleven days, and
   that claim is wrong. Per `CLAUDE.md` invariant 6 the prompt-01 log is not rewritten; this log
   and both board entries carry the correction. **The rule it breaks is the first one in
   `CLAUDE.md`: check `docs/OPEN_ISSUES.md` before opening.**

2. **The write-side defect class has no guard.** `[01-read-batch-is-outside-the-guard]` covers the
   read side's remaining gap; nothing covers insert payloads. The sweep above was done by hand and
   is clean, but a hand sweep is a measurement, not a guard. Extending
   `test_factory_select_columns.py` with an insert-key check is a real candidate — the sweep's
   prototype resolved parent-inserts-into-child's-table correctly and found only the documented
   `validated` key — but it is a second analyser, it was not asked for, and shipping it in this
   commit would bury a two-token production change. **Not opened as an issue**: it is a
   suggestion, and the user should decide whether it is worth a prompt.

3. **`sqla_BackgroundModelValue_factory.build()` is still not called by anything.** This change
   makes it correct, and the new tests make it exercised, but no production path reaches it:
   `BackgroundModel.build()` reads its sample rows with its own `SELECT`. Whether the factory
   should be reachable, or should be deleted, is a design question this change deliberately does
   not answer.

## State handed to the next prompt

- `Datastore/tests` is at **10** tests.
- `docs/OPEN_ISSUES.md` is at **85**; `prompts/GkTk-remedial`'s §3 has one entry fewer.
- One issue open on this campaign's board: `[01-read-batch-is-outside-the-guard]`.
- The interrupted `handover-A3-baseline-lambdacdm` run is still the user's to resume; nothing in
  this commit touches the datastore.
