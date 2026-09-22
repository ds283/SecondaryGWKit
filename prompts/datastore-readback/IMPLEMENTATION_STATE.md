# Datastore read-back campaign — implementation state

**Last updated:** 2026-09-22 · **Status: STARTED — 1 of 1 written prompts landed (01), plus one
direct follow-up.** Prompt 01 landed the one-line `QuadSourceIntegral` fix, the static guard in the
new `Datastore/tests/` root, and the audit of every object factory. **Stored `QuadSourceIntegral`
rows read back**; the interrupted `handover-A3-baseline-lambdacdm` run is resumable, which is the
user's to do. The audit's own finding, `[01-backgroundmodelvalue-hubble]`, was then **closed at the
user's direction** (§4), which took `sqla_BackgroundModelValue_factory.build()` from never-executed
to covered. One issue remains open in §3.

**Campaign:** [`README.md`](README.md) ·
**Datastore:** `var/datastores/handover-A3-baseline-lambdacdm.sqlite` and its manifest ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.10

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit** —
> the row added, moved or deleted, and the count and date in its header corrected. An issue owned
> by another board is moved to **that** board's §4 and its row deleted from the index. The index is
> an index: one line per issue, pointing here. Where the two disagree, this board is right.
> See `CLAUDE.md`.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [The `QuadSourceIntegral` read-back](01-quadsourceintegral-readback.md) | **R1**, **R2**, **R3** | Opus 5 | ✍️ yes | ✅ 2026-09-22 | *"Add the missing numeric_quad column to the QuadSourceIntegral read"* | [`logs/01-…`](logs/01-quadsourceintegral-readback.md) |
| — | *(no prompt file — direct user instruction, in session, immediately after 01)* | **R4** | Opus 5 | — | ✅ 2026-09-22 | *"Repair the BackgroundModelValue factory's two dead-code defects"* | [`logs/01a-…`](logs/01a-backgroundmodelvalue-repair.md) |

The second row is **not a numbered prompt**: the user read prompt 01's audit and asked for
`[01-backgroundmodelvalue-hubble]` to be closed directly. It keeps its own commit and its own log,
per campaign README §5 items 1 and 2; it has no prompt file because none was written.

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| R1 | **REMEDY** | Add `table.c.numeric_quad` to the `SELECT` that `sqla_QuadSourceIntegral_factory.build()` executes. One line; the schema and the stored data are correct. | 01 | ✅ **Done, 2026-09-22.** `QuadSourceIntegral.py:246`. Row `serial=20806` of shard 0 reads back through the real `Datastore.object_get` with `numeric_quad = -1.2033078530407802e-11`, equal as a float to the stored value; the same read against the unfixed tree raises `Could not locate column in row for column 'numeric_quad'`. Store read-only throughout; shard mtimes unchanged at 2026-09-21 04:32; backup untouched; pipeline not started. |
| R2 | **FACILITY** | The guard: a test that catches the whole class — `build()` reading a column its own `SELECT` does not request — with no Ray and no datastore. | 01 | ✅ **Done, 2026-09-22.** `Datastore/tests/test_factory_select_columns.py`, 516 lines, **4 tests**, 0.3 s. Shape **(a)**, static `ast` consistency, as the prompt recommended, and it did **not** have to be weakened to pass. Two soundness corrections were forced by running it: a dynamically named column may clear a binding but may never convict one, and a variable rebound part-way through a `build()` is two bindings — merging them produced a false positive on `sqla_BackgroundModelFactory`. Beyond the subset assertion the guard pins its own reach (`EXPECTED_COVERAGE`, 20 bindings), its one declared blind spot (`KNOWN_BLIND`) and the currency of `KNOWN_UNFIXED`, so that a refactor which blinds it fails rather than passing vacuously. **Deliberate-breakage record:** unfixed tree → `FAILED (failures=1)` naming `numeric_quad`; fixed tree → `OK`. Six holes stated in the module docstring, three of them live in this tree. |
| R4 | **REMEDY** | The two defects the audit found in `sqla_BackgroundModelValue_factory.build()`: the read of `row_data.Hubble`, and the insert key `"wkb_serial"`. | — | ✅ **Done, 2026-09-22.** Closes `[01-backgroundmodelvalue-hubble]` (§4). **Neither was a one-line fix.** The insert key was, once the mechanism was established: SQLAlchemy Core drops an unmatched key silently — the note at `OneLoopIntegral.py:253` records the same for a different key — so the row was attempted with `model_serial` unset and `nullable=False` refused it. Had the column been nullable it would have written a silently orphaned row. The `Hubble` read could **not** be repaired by renaming the attribute: the comparison sat *below* the block that replaces the payload values with the stored ones, so `Hubble` had already become the stored value and a renamed check would have compared it with itself and never fired. It is moved above that block and compared in the stored GeV representation with a **relative** bound, because the internal-unit value spans 2.37e-04 to 9.19e+26 over a production grid — at the bottom an absolute `DEFAULT_FLOAT_PRECISION` is a 4.2e-04 relative tolerance, at the top one ulp is 1.37e+11, i.e. 1.4e+18 times the bound. `Datastore/tests/test_backgroundmodelvalue_roundtrip.py` (**6 tests**, in-memory SQLite built from the factory's own `register()`) now exercises all three branches; it is prompt 01 §3's shape **(b)**, which the static guard by construction cannot replace on the write side. `KNOWN_UNFIXED` is back to empty, as intended. |
| R3 | **AUDIT** | Run the guard across every object factory; **open** what it finds, fix nothing. | 01 | ✅ **Done, 2026-09-22.** 21 modules, **38 classes defining `build()`**, of which **24 read nothing off a query result** (verified independently by scanning each for an attribute read, an `_mapping` subscript or a `getattr` on a row-named variable — none has one) and **14 read rows** across **21 bindings**, 20 decidable and 1 declared blind. **One further defect found, not fixed:** `sqla_BackgroundModelValue_factory.build` reads `row_data.Hubble` where the column is `Hubble_GeV` — `[01-backgroundmodelvalue-hubble]` in §3. It is **not** on a path reachable from a running pipeline or a resume, so prompt §7's third stop condition was not met. Everything else is clean. The campaign README's "22 factories, 12 reading `row_data`" counts registry entries; the figures here are class by class and supersede it. |

---

## 3. Active and unresolved issues

- **[01-read-batch-is-outside-the-guard]** *(opened 2026-09-22 by prompt 01)* — the guard covers
  `build()` only, because that is what prompt §3 scopes it to. **`read_batch()` reads rows too and
  is checked by nothing.** `sqla_QuadSourceIntegral_factory.read_batch` (`:530`) alone reads some
  sixty attributes off a `row` that is the parameter of a nested `make_object`, from a query
  extended by `add_columns` in two loops — neither shape is one the static analyser can follow, so
  extending the existing check to it is not a matter of widening a glob. The defect class is equally
  possible there, and `read_batch` is on the **live** path (`Datastore.object_read_batch`), not only
  on resume: it has been exercised, which is weak evidence that it is currently clean but is no
  guarantee for the columns a given call does not touch. **Impact:** the guard's coverage is
  narrower than "the factories are guarded" would suggest, and a later reader could take the
  green suite for more than it says. The module docstring and the log both state the limit; this
  issue is so that it is not only stated. **Next step:** either teach the analyser to link a nested
  helper's row parameter to the query it is called with, or write prompt §3 shape **(b)**, the
  in-memory round trip, for `read_batch`. Not attempted in prompt 01: widening the guard is a change
  with its own risk and would have obscured the one-line diff that commit exists to prove.
  Measurement: [`logs/01-quadsourceintegral-readback.md`](logs/01-quadsourceintegral-readback.md),
  "What it cannot see" item 3. Indexed at `docs/OPEN_ISSUES.md` §1.10.

---

## 4. Resolved issues

- **[01-backgroundmodelvalue-hubble]** *(opened 2026-09-22 by prompt 01; **retired 2026-09-22 as a
  duplicate**, and the defect fixed under item R4)* — `sqla_BackgroundModelValue_factory.build()`
  carried two defects, both fatal on the first call and both invisible because nothing called it:
  `row_data.Hubble` where the column is `Hubble_GeV`, and the insert key `"wkb_serial"` where the
  column is `model_serial`.

  **This issue should never have been opened.** The same defect — both halves of it — had been on
  `docs/OPEN_ISSUES.md` since 2026-09-11 as
  [`[03-backgroundmodelvalue-build-path]`](../GkTk-remedial/IMPLEMENTATION_STATE.md) §4, opened by
  `GkTk-remedial` prompt 03, with the two failure modes correctly named and with "a two-line fix in
  its own commit, with a test that exercises `build()` against an in-memory SQLite store" as its
  next step. Prompt 01's audit found the read half independently and opened a new row **without
  checking the index that exists precisely to prevent this**. The fix therefore closes on the
  `GkTk-remedial` board, per `CLAUDE.md`'s rule that an issue owned by another board is moved to
  that board's §4; this entry is the record of the duplicate, and it is deleted from the index.

  **What the repair established that neither entry did.** Neither defect was the one-line fix it
  looked like.

  - *The insert key.* SQLAlchemy Core's `insert()` drops a dict key that matches no column
    **silently**, rather than raising — the note at `OneLoopIntegral.py:253-257` records exactly
    this for a different key, and it was re-verified here. So the row was attempted with
    `model_serial` unset, and what refused it was the column's own `nullable=False`:
    `IntegrityError: NOT NULL constraint failed: BackgroundModelValue.model_serial`. Both entries
    were right that the branch could never have run, and `[03-…]` was right that the symptom is an
    `IntegrityError`; what neither said is that the constraint, not the key, is what catches it —
    **had that column been nullable, the branch would have written silently orphaned rows instead
    of failing.** `wkb_serial` is a real column of the `GkWKBValue` and `TkWKBValue` tables, which
    is where it was copied from.
  - *The `Hubble` read.* It could **not** be closed by correcting the attribute name. The
    comparison sat *below* the block that replaces the payload values with the stored ones, so by
    the time it ran, `Hubble` had already become the stored value: a renamed check would have
    compared the stored value against itself, passed always, and looked repaired. It is now above
    that block, where the payload value still exists. It is also compared in the stored GeV
    representation with a **relative** bound, because no single absolute bound serves this column:
    over a production redshift grid the internal-unit value runs 2.37e-04 to 9.19e+26, so
    `DEFAULT_FLOAT_PRECISION` is a 4.2e-04 relative tolerance at the bottom — loose enough to
    accept a Hubble wrong in its fourth significant figure — while at the top one ulp is 1.37e+11,
    1.4e+18 times the bound, and the check degenerates to bit equality. The `wBackground` check
    beside it is dimensionless and O(1), so its absolute bound is right and is unchanged; it moved
    only to stay with its sibling.

  **Verification.** `Datastore/tests/test_backgroundmodelvalue_roundtrip.py`, 6 tests, in-memory
  SQLite built from the factory's own `register()` output so the schema under test is the
  production schema. All three branches are exercised — insert, agreeing read-back, disagreeing
  read-back — and both consistency checks are shown to *fire*, which neither has ever done.
  Deliberate breakage: restoring `"wkb_serial"` fails all six with the `IntegrityError` above;
  restoring the `Hubble` read fails the static guard in `test_factory_select_columns.py`. That
  guard's `KNOWN_UNFIXED` is back to empty, and `test_known_unfixed_entries_are_still_present`
  refused to pass until the stale entry was struck — the guard held the fix, as designed.

  **Not addressed, and not a defect:** `sqla_OneLoopIntegral_factory.store()` passes a `"validated"`
  key its table does not declare. A sweep of every insert-payload dict in `ObjectFactories/`
  against its table's columns found this and nothing else; it is already documented in place at
  `OneLoopIntegral.py:253-257` as understood and deliberate, and the column is not `NOT NULL`, so
  it is silently dropped with no consequence. Recorded here so the sweep's result is on the record.

*`[R1]` was never indexed as an issue: the campaign was created to fix it and prompt 01 closed it
in the same commit that opened the board. Its record is item R1 in §2 above and the `run_history`
entry of `var/datastores/handover-A3-baseline-lambdacdm.manifest.json`.*

---

## 5. Baselines

Campaign README §4's, plus the two this campaign measured.

| Suite | Count | Measured |
|---|---|---|
| `ComputeTargets` | 552 | `ab7079c`; `test_tk_wkb_phase.TestCost.test_wall_time_per_object` is a known wall-clock flake — re-run that module alone before attributing a failure |
| `CosmologyModels` | 39 | `ab7079c` |
| `LiouvilleGreen` | 148 (skipped=1) | `ab7079c` |
| `AdaptiveLevin` | **32** | **first baselined 2026-09-22 by prompt 01**, on the pre-change tree |
| `Datastore` | **10** | **created 2026-09-22 by prompt 01** at 4; item R4 added 6 the same day |

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s Datastore/tests -t .
```
