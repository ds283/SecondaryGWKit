# Log 07 — Report the `T(z)` representation in the QCD cosmology inventory

**Prompt:** prompts/background-solver-robustness/07-inventory-representation.md
**Commit:** (this commit) — *"Report the T(z) representation in the QCD cosmology inventory"*
**Model:** Sonnet
**Date:** 2026-09-16
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

`Datastore/SQL/ObjectFactories/QCD_Cosmology.py:150` (`inventory()`) — before: selected and
labelled `name`, `omega_m`, `omega_cc`, `h`, `log10_max_z`. After: also selects and labels
`T_z_representation`, placed immediately **before** `log10_max_z` rather than appended at the end
— `name`, `omega_m`, `omega_cc`, `h`, `T_z_representation`, `log10_max_z`. Justification (also in
the code comment): `T_z_representation` and `log10_max_z` both describe how this row's background
was computed and over what range, while `name`/`omega_m`/`omega_cc`/`h` are the cosmology's
physical parameters that precede them; reading the two together as "computed under representation
X, valid out to log10_max_z" states the row's identity in the natural order, rather than trailing
it as an afterthought (prompt §2 item 3's instruction).

`tools/inventory_report.py` — **not in the diff**. Read in full: `_format_value_list` /
`_format_value_lines` render every entry of an `inventory()` "values" list generically via
`str(dict)` (`tools/inventory_report.py:105-118, 172-202`); no function names the QCD columns.
Confirmed by the new test's own inspection of the returned dict, not assumed.

New public symbol: none. No new class, function or constant was added to production code — the
only production change is the two-line extension of the `values`/`select` lists quoted above.

New test module: `ComputeTargets/tests/test_qcd_cosmology_inventory.py` (3 tests, class
`TestInventoryReportsTheRepresentation`). Builds an in-memory SQLite table from
`sqla_QCDCosmology_factory.register()`'s own output — the pattern
`ComputeTargets/tests/test_cosmology_representation_key.py` already uses for this same factory —
and calls the production `inventory()` against it directly; no Ray, no project `Datastore`
machinery, no new package.

`T_Z_REPRESENTATION_VERSION` is **6** before and after (`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:428`);
this prompt does not touch that file.

## The two equality redshifts

Not applicable to this prompt — it is workstream D (housekeeping), not workstream A or B, and
touches no file that computes them. Not measured.

## Deviations from the prompt

### D1 — STRUCTURALLY REQUIRED: the test's location changes the `ComputeTargets` count

The prompt's acceptance table (§3) states `ComputeTargets` **449 → 449**, i.e. unchanged, and its
files-may-touch line says "a test under `Datastore/tests/`, or the nearest existing home for
datastore-factory tests (find it; do not create a new package)". There is no `Datastore/tests/`
package anywhere in the tree (`find Datastore -type d` shows none, and no `__init__.py` under a
path of that name), and no `sqlite`-based test of any object factory exists outside
`ComputeTargets/tests/` — specifically `test_cosmology_representation_key.py` and
`test_numeric_break_point_key.py`, both of which build an in-memory SQLite table from a factory's
own `register()` and call its static methods directly, exactly the pattern this prompt asks for.
That **is** "the nearest existing home for datastore-factory tests", so the new test module was
added there rather than as a new package. The acceptance table's assumption that the count would
stay at 449 was written before this was confirmed and does not survive it: a genuine new test
module necessarily raises the count. Measured: `ComputeTargets` **449 → 452** (+3, exactly the
three tests in the new module); `CosmologyModels` **39 → 39**, unchanged. Per the board's standing
note 3 ("a count that falls is a stop... a rise must be a test a prompt says it added"), a rise
attributed to a named new test module is not a stop, and this one is.

### D2 — STRUCTURALLY REQUIRED: the issue this prompt closes had already been widened, to a file out of scope

`docs/OPEN_ISSUES.md` §1.7's row for `[03-qcd-inventory-does-not-report-the-representation]`
(and the `qcd-background-audit` board's own §3 entry for it) read as a single-file issue in the
prompt's own §1, but the board entry itself carries a "Widened by prompt 14 (2026-09-15)" paragraph
this prompt's text does not mention: the same gap exists on
`sqla_BackgroundModelFactory.inventory()` (`source_grid_digest`, `source_grid_construction`), and
the qcd-background-audit board's own recorded next step was "add `T_z_representation` to
`sqla_QCDCosmology_factory.inventory()` *and* the two grid columns to
`sqla_BackgroundModelFactory.inventory()`'s per-bucket report, in one commit."

This prompt's own files-may-touch list does not include `BackgroundModel.py`, and its stop
condition §4 item 3 is explicit: "You find a second reporting site that has the same gap. Record
it; do not fix it — that is a new issue, not this prompt's." That is exactly this situation, so
`BackgroundModel.py` was left untouched (confirmed not in the diff) and the widened half is
recorded rather than silently dropped: the `qcd-background-audit` board's `[03-…]` entry is
**narrowed**, not deleted — the `QCD_Cosmology` half moves to that board's §4 as resolved (citing
this commit), and the `BackgroundModel` half stays in that board's §3 under the same ID, unassigned,
with its own next step. `docs/OPEN_ISSUES.md` §1.8's row for the old (assigned, gated) state is
deleted and replaced by a new row in §1.7 for the narrowed remainder — net effect on the index's
open-issue count is zero (one row removed, one added), so the header's "69 open" is unchanged. This
is a deviation from the prompt's literal deliverable #4 ("`[03-…]` moved to the qcd-background-audit
board's §4 ... row deleted from `docs/OPEN_ISSUES.md`"), which assumed full closure; full closure
would have silently stopped tracking a known, already-documented defect, which the prompt's own stop
condition forbids.

### D3 — none of the prompt's other assumptions needed correcting

The column exists exactly as `[03-…]`'s index entry implied (`T_z_representation`, an `Integer`,
part of the lookup key), and `tools/inventory_report.py` did not need editing, both confirmed by
reading rather than assumed.

## Verification performed

- `PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_qcd_cosmology_inventory -v`
  → **3 tests, OK** (0.028 s). Covers: two rows differing only in `T_z_representation` render as
  two distinct label dicts (`{4, 6}`, not one indistinguishable pair — the prompt's §3 acceptance
  criterion, demonstrated rather than argued); the column appears before `log10_max_z`, not after;
  two rows at the *same* representation still report two labels with the expected fields.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` →
  **Ran 452 tests in 158.103s, OK**. Before (prompt 06's close-out): 449.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .` →
  **Ran 39 tests in 0.676s, OK**. Unchanged from 39.
- No suite covers `Datastore/` directly (`find . -type d -iname tests` finds no such package); the
  new test is the first exercise of any `Datastore/SQL/ObjectFactories` `inventory()` method against
  SQLite, added to `ComputeTargets/tests/` per D1 above.
- `./venv/bin/python -m black --check Datastore/SQL/ObjectFactories/QCD_Cosmology.py
  ComputeTargets/tests/test_qcd_cosmology_inventory.py` → clean, both files unchanged.
- `git diff Datastore/SQL/ObjectFactories/QCD_Cosmology.py` reviewed by eye: `build()`, `register()`
  and the lookup key are not in the diff; only `inventory()`'s two lists changed.
- `tools/inventory_report.py` read in full; not in the diff (confirmed by `git status`).

## Observations not acted on

- The `BackgroundModel` half of `[03-qcd-inventory-does-not-report-the-representation]` (D2 above).
  Recorded on the `qcd-background-audit` board's §3 (narrowed entry) and in
  `docs/OPEN_ISSUES.md` §1.7. Next step: add `source_grid_digest` and `source_grid_construction` to
  `sqla_BackgroundModelFactory.inventory()`'s per-bucket report, in whichever prompt next has
  `Datastore/SQL/ObjectFactories/BackgroundModel.py` in scope. Not assigned to any campaign.
- No second reporting site beyond `BackgroundModel` was found. A grep for other `inventory()`
  implementations under `Datastore/SQL/ObjectFactories/` was not exhaustively re-audited against
  every table's lookup key beyond what the qcd-background-audit board had already flagged; that
  wider audit is out of scope for a two-line, gated housekeeping prompt.

## State handed to the next prompt

Workstream D's other prompt (08) is independent of this one and needs nothing from it beyond what
its own text already states. For whoever eventually takes `Datastore/SQL/ObjectFactories/BackgroundModel.py`
into scope: the columns to add are `source_grid_digest` and `source_grid_construction`
(`sqla_BackgroundModelFactory`'s per-bucket `inventory()` report), and `tools/inventory_report.py`
will again need no change — `_format_bucketed` (its bucketed-report formatter) renders whatever a
bucket's `labels` list contains generically, the same way `_format_value_list` does for the flat
case this prompt exercised.
