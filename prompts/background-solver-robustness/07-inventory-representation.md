# Prompt 07 — Report the `T(z)` representation in the QCD cosmology inventory

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Workstream D — GATED.** Do not run this without the user's go-ahead (README §7 **D3**).
**Closes:** `[03-qcd-inventory-does-not-report-the-representation]` on the
[`qcd-background-audit` board](../qcd-background-audit/IMPLEMENTATION_STATE.md) §3
**Depends on:** nothing in this campaign. It may run at any point after the gate opens, including
after 06.
**Recommended model:** **Sonnet** — two lines and a test, in files no other prompt in this campaign
touches. The work is confirming the column name and that nothing downstream parses the inventory's
shape.

**Files you may touch:** `Datastore/SQL/ObjectFactories/QCD_Cosmology.py` (`inventory()`, `:150`),
`tools/inventory_report.py`, a test under `Datastore/tests/` or the nearest existing home for
datastore-factory tests (**find it; do not create a new package**), plus this campaign's log, both
boards and `docs/OPEN_ISSUES.md`.

**Do not touch:** the table schema, the lookup key, `build()`, or anything that writes. **This
prompt changes what is *reported*, never what is *stored*.**

**Read first:** `Datastore/SQL/ObjectFactories/QCD_Cosmology.py` in full — it is short — paying
attention to `:50` and `:91`, the comments prompt 03 of `qcd-background-audit` left about the
representation column and how it is compared; `tools/inventory_report.py`; and the index row for
`[03-qcd-inventory-does-not-report-the-representation]` in `docs/OPEN_ISSUES.md` §1.7.

---

## 1. What is wrong

`sqla_QCDCosmology_factory.inventory()` (`:150`) selects `name`, `omega_m`, `omega_cc`, `h` and
`log10_max_z` and builds one label per row. Since prompt 04 of `qcd-background-audit` the table also
carries the `T_z_representation` column (added by prompt 03), which is **part of the lookup key**:
two rows differing only in their representation are two different cosmologies. The inventory does
not select it, so to `tools/inventory_report.py` — the only tool in the tree that inspects a
datastore — they render as **indistinguishable duplicates**.

`T_Z_REPRESENTATION_VERSION` has been **6** since `qcd-background-audit` prompt 13, and stores
written under versions 4, 5 and 6 all exist in principle. A user looking at an inventory to work out
which rows to keep currently cannot.

The index calls it *"One line in `inventory()`"*. Confirm that before assuming it.

## 2. The change

1. **Add the representation column to the select and to the per-row label** in `inventory()`. Use
   the column's actual name — read it from the factory's table definition, do not guess it from the
   constant's name.
2. **Check `tools/inventory_report.py`.** If it renders the label dict generically, nothing is
   needed. If it names the five keys, add the sixth. Say which in the log.
3. **Order.** Put the representation next to the field it qualifies, not at the end, so the label
   reads as a statement about the cosmology rather than as an afterthought. Justify the position in
   one sentence.
4. **A test.** Find where datastore object-factory behaviour is tested today. If `inventory()` has
   no test anywhere, write the smallest one that exercises it against an in-memory SQLite table —
   **and if that requires standing up machinery this prompt has no business standing up, stop and
   say so** rather than building it. In that case the deliverable is the two-line change plus a log
   entry recording that the path is untested and why, and the board keeps a §3 issue for it.

## 3. Acceptance

| Check | Threshold |
|---|---|
| `inventory()` reports the representation | demonstrated against a table with two rows differing only in that column, which must render distinguishably |
| Anything that **writes** | unchanged; `build()` not in the diff |
| The lookup key | unchanged |
| `ComputeTargets` suite | **449 → 449**, OK (447 before prompt 09, which added two `test_source_grid.py` tests) |
| `CosmologyModels` suite | unchanged, OK |
| Whatever suite covers `Datastore/` | count and result quoted before and after |
| `black --check` | clean |

## 4. Stop conditions

- **The column is not in the table**, or is named differently from what the index implies. Report
  what is actually there and stop.
- **Making `inventory()` testable requires new test infrastructure.** Ship the change, record the
  gap, open the issue. Do not build a datastore test harness inside a gated housekeeping prompt.
- **You find a second reporting site** that has the same gap. Record it; do not fix it — that is a
  new issue, not this prompt's.

## 5. Deliverables

1. The change in `inventory()` and, if needed, in `tools/inventory_report.py`.
2. A test, or a recorded reason there is none.
3. `logs/07-inventory-representation.md` per README §5.1.
4. Board row 07, and `[03-qcd-inventory-does-not-report-the-representation]` moved to the
   **`qcd-background-audit` board's §4** with its row deleted from `docs/OPEN_ISSUES.md` — count
   and date corrected, in the same commit.
5. One commit, README §5 rule 2.
