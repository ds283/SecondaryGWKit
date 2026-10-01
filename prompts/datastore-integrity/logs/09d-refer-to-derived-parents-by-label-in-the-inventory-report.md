# Log 09d — Refer to derived parents by label in the inventory report

**Prompt:** [`prompts/datastore-integrity/09d-refer-to-derived-parents-by-label-in-the-inventory-report.md`](../09d-refer-to-derived-parents-by-label-in-the-inventory-report.md)
**Commit:** *(this commit)* — "Refer to derived parents by label in the inventory report"
**Model:** Claude Sonnet 5.5
**Date:** 2026-10-01
**Result:** COMPLETE. The report prints a reference to a derived (sharded) row as `<Class>#<n>` and a
parent set as `S<n>`, defines each once, and keeps the property that two distinct records never
print as one line. It closes `[09b-inventory-report-renders-every-member-of-a-parent-set]` and
opens one issue, `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]`. No stop
condition triggered:
- no existing test was edited (`git diff --cached --name-status` shows one modified file,
  `tools/inventory_report.py`, and one added, `Datastore/tests/test_inventory_report_labels.py`,
  besides the log, the board and the index);
- the property of §2 holds with record lines that do not grow with a set or a chain (the measured
  sizes below);
- the fix needed nothing outside `tools/inventory_report.py` and a new test module;
- no class whose parents are all replicated and that nothing names changed: `OneLoopIntegral` and
  the twelve replicated classes print the same lines as at `HEAD` (below).

Suites after this commit, serially, on the staged tree:
- Datastore **532 → 551** (the 19 of the new `test_inventory_report_labels.py`);
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), RunRegistry 199 and
  ComputeTargets 665, unchanged in count. ComputeTargets shows the known wall-clock flake
  `test_tk_wkb_phase.TestCost.test_wall_time_per_object` under a load average of 50–120 from other
  work: it fails alone in the checkout (0.0804 s against the 0.06 s limit) and alone in an export of
  `HEAD` (0.0805 s), so it is not this commit's.

```
AdaptiveLevin: Ran 32 tests OK
CosmologyModels: Ran 39 tests OK
LiouvilleGreen: Ran 148 tests in 54.546s OK (skipped=1)
RunRegistry: Ran 199 tests in 174.060s OK
Datastore: Ran 551 tests in 503.710s OK
ComputeTargets: Ran 665 tests in 254.588s FAILED (failures=1), the flake alone
```

The Datastore suite was run twice on the staged tree. The first run failed one test,
`test_inventory_retired.TestNoReferenceRemains.test_the_grep_prints_nothing`: the new module's
helper was a method named `inventory`, which the retired-name `git grep` matches (it sees only a
staged file, as log 08 deviation 8 and log 09b deviation 15 found). The method was renamed
`as_inventory`, the module re-staged, and the full suite re-run: the 551 above.

The index `docs/OPEN_ISSUES.md` stays at **99** rows, counted
(`grep -c '^| \`\[' docs/OPEN_ISSUES.md`): one deleted (the 09b report issue), one opened.

## What shipped

**`tools/inventory_report.py`** (the only code file changed).
- **Which references are labels.** `_Renderer.labelled()`: the classes of the inventory that are
  sharded and that some class's `parents` or `parent_sets` members name. On the fixture these are
  `TkNumericIntegration`, `TkWKBIntegration`, `GkNumericIntegration`, `GkWKBIntegration`,
  `GkSource`, `GkSourcePolicyData` and `QuadSource`. `ref_text` prints a reference to one of them
  as `<Class>#<n>`; a reference to any other class is rendered from the parent's key as before, and
  an unresolvable one keeps `<unresolved … reference>`.
- **The label.** `_Renderer.ordinals(name)`: a record's ordinal in its class's verbose listing, tag
  set by tag set, each in the report's physical order, counting from 1. Two records with the same
  reference digest (a `duplicate` problem) each have an ordinal; a reference to that digest takes
  the lower. It uses no serial. The order is the one the report already used, now held by
  `groups(name)`, which `_class_lines` also reads.
- **The record's own label.** `record_line` appends `  #<n>` for a class in `labelled()`, after
  the fields and any `[unvalidated]` marker and before `| values:`, so beyond the first double space
  that `fields_of` stops at.
- **Parent sets.** `set_numbers()` gives each distinct, resolvable set of a class a number, in the
  physical order of its members (the sort key `set_sort` already computed, then the digest for a
  tie), counting on through every class that holds sets. `set_text` is now `S<n>`, and the set is
  printed once, after its class's records, as `parent set S<n>: [{z_source=…, numeric=…, wkb=…},
  …]` (`set_line`, `members_text`), the members in the order and form `_set_members` already
  gave, each field a label or `none`.
- **Every label is defined.** `needed()` computes, from the records the report lists, the rows they
  name that the report does not list, and so on for those rows (a worklist closed under reference),
  and the sets that all of these name. `_class_lines` prints those rows after its records, one line
  each under the prefix `> `, headed `N row(s) referred to by the records above, and not shown:`,
  each followed by `  tags: …` because a listed row is not under its tag set's line. With
  `verbose` every record is listed, so only the sets remain, and all of them are printed.
- **The module docstring** describes this in place of the old account of how a parent and a set were
  rendered, and says why a summary of a set would not keep the property.

**`Datastore/tests/test_inventory_report_labels.py`** (new, 19 tests, no Ray, nothing under
`var/`). The stores of tests 1, 2 (second store) and 4 are inventories built in memory by `_Stand`
from the fixture's replicated rows; tests 2, 3 and 5 also read the fixture store with
`build_full_store`.

| Class | Tests | What it pins |
|---|---|---|
| `TestDistinctStaysDistinct` | 5 | §3 test 1: `GkSource` pairs differing only in a Gk numeric row or only in a Gk WKB row at one `z_source` (same member count and range); `GkSourcePolicyData` and `QuadSourceIntegral` pairs differing only through the set; `QuadSourceIntegral` pairs differing in each one of the four Tk parents, and only in the `QuadSource`; two Tk parents differing only in their tags. The record's own label is taken off first. |
| `TestEveryLabelIsDefinedOnce` | 6 | §3 test 2, in the fixture and a stand-in, verbose and not and with `MAX_SHOWN` 1 and 2: one defining line per label; the defining line equals the line of the same label in a report that lists everything; labels run 1, 2, 3, … down a verbose section and the sets S1, S2, …; a reference names the row it points to (each Tk row has a leaf of its own); rows referred to and not shown are listed; every reference to a sharded class is a label and every set a `S<n>`. |
| `TestLabelsAreStable` | 3 | §3 test 3: the same store twice; the fixture under `relabel_serials`; the fixture inventory with every class's records reversed. |
| `TestTheSizeIsBounded` | 3 | §3 test 4, at 103, 403 and 1,703 members: the record lines of the five classes that carry a derived parent are equal at every size; a set's line grows by at most 100 characters a member; the rest of the report (less the sets and the replicated redshift rows the members name) is the same size. |
| `TestNothingElseChanged` | 2 | §3 test 5: the replicated classes and `OneLoopIntegral` print no label, listing or set; `QuadSourceIntegral`, which nothing names, prints no own label. |

## The shape chosen, and why

The recommended shape, unchanged: a label for a reference to a sharded row and for a set, each
printed once in its own section, and the rows referred to but not shown listed. The prompt's
alternatives were not needed. It is better than a summary because a label is distinct for every
distinct row and set, so no key field is dropped by naming a parent; and better than keeping the
parents inline because a line is then a few hundred characters whatever the depth of its chain.

## The fixture, before and after

Taken from `format_inventory_report(read_inventory(build_full_store(tmp).primary), "build_full_store",
verbose=True)` at `HEAD` (`0dda6c8`) and with this change, the first record line of each class
whose lines changed, without its leading `            - `.


**TkNumericIntegration**, before (401 characters):

```text
model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0  | values: 3
```

after (405 characters):

```text
model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0  #1  | values: 3
```

**TkWKBIntegration**, before (565 characters):

```text
k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, numeric={model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0, tags=[Run_fixture, grid-A]}  | values: 1
```

after (173 characters):

```text
k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, numeric=TkNumericIntegration#1  #1  | values: 1
```

**QuadSource**, before (1200 characters):

```text
q={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, r={cosmology_type=0, k=0.5, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, Tq={model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0, tags=[Run_fixture, grid-A]}, Tr={model={cosmology_type=1, tau_gauss_order=6, cs_tau_gauss_order=6, friction_F_gauss_order=6, source_grid_digest=fixture-digest-qcd, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046, log10_max_z=10.0, T_z_representation=3}, tags=[Run_fixture]}, k={cosmology_type=1, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046, log10_max_z=10.0, T_z_representation=3}}, rtol=-5.0, tags=[Run_fixture, grid-B]}  [unvalidated]  | values: 2
```

after (336 characters):

```text
q={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, r={cosmology_type=0, k=0.5, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, Tq=TkNumericIntegration#2, Tr=TkNumericIntegration#3  [unvalidated]  #1  | values: 2
```

**GkNumericIntegration**, before (137 characters):

```text
k={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}  | values: 1
```

after (141 characters):

```text
k={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}  #1  | values: 1
```

**GkWKBIntegration**, before (137 characters):

```text
k={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}  | values: 1
```

after (141 characters):

```text
k={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}  #1  | values: 1
```

**GkSource**, before (550 characters):

```text
k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, z_response=10.0, parents=[{z_source=10.0, numeric=none, wkb={k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, tags=[Run_fixture, grid-A]}}, {z_source=100000.0, numeric={k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, tags=[Run_fixture, grid-A]}, wkb=none}]  | values: 2
```

after (170 characters):

```text
k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, z_response=10.0, parents=S1  #1  | values: 2
```

**GkSourcePolicyData**, before (762 characters):

```text
source={k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, z_response=0.0, parents=[{z_source=10.0, numeric=none, wkb={k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, tags=[Run_fixture, grid-A]}}, {z_source=100000.0, numeric={k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, tags=[Run_fixture, grid-A]}, wkb=none}], tags=[Run_fixture, grid-A]}, policy={Levin_threshold=0.1, numeric_policy=maximize-numeric}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}
```

after (210 characters):

```text
source=GkSource#2, policy={Levin_threshold=0.1, numeric_policy=maximize-numeric}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}  #1
```

**QuadSourceIntegral**, before (4316 characters):

```text
k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, q={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, r={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, z_response=10.0, source={q={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, r={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, Tq={model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0, tags=[Run_fixture, grid-A]}, Tr={model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0, tags=[Run_fixture, grid-A]}, tags=[Run_fixture, grid-A]}, data={source={k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, z_response=10.0, parents=[{z_source=10.0, numeric=none, wkb={k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, tags=[Run_fixture, grid-A]}}, {z_source=100000.0, numeric={k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, tags=[Run_fixture, grid-A]}, wkb=none}], tags=[Run_fixture]}, policy={Levin_threshold=0.5, numeric_policy=maximize-WKB}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}}, Tq_numeric={model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0, tags=[Run_fixture, grid-A]}, Tq_WKB={k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, numeric={model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0, tags=[Run_fixture, grid-A]}, tags=[Run_fixture]}, Tr_numeric={model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0, tags=[Run_fixture, grid-A]}, Tr_WKB={k={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, numeric={model={cosmology_type=0, tau_gauss_order=8, cs_tau_gauss_order=8, friction_F_gauss_order=8, source_grid_digest=fixture-digest, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}, tags=[Run_fixture, grid-A]}, k={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, rtol=-7.0, tags=[Run_fixture, grid-A]}, tags=[Run_fixture]}
```

after (565 characters):

```text
k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, q={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, r={cosmology_type=0, k=1.0, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, z_response=10.0, source=QuadSource#2, data=GkSourcePolicyData#2, Tq_numeric=TkNumericIntegration#1, Tq_WKB=TkWKBIntegration#1, Tr_numeric=TkNumericIntegration#2, Tr_WKB=TkWKBIntegration#2
```

**A `parent set` line** (the same report, verbose; the fixture's two `GkSource` sets each have two
members):

```text
         parent set S1: [{z_source=10.0, numeric=none, wkb=GkWKBIntegration#2}, {z_source=100000.0, numeric=GkNumericIntegration#2, wkb=none}]
```

The `GkSource` record lines above name it `parents=S1`; before this change each carried the whole
bracketed list.

**A referenced-but-not-shown line.** At the fixture's size no row is left unshown (a class shows up
to five records per tag set, and none has more than three), so these are from the same store
reported without `verbose` and with `MAX_SHOWN` set to 1 by `unittest.mock.patch`
(`TestEveryLabelIsDefinedOnce` does the same). In the `TkWKBIntegration` section, where the one
shown row names `TkNumericIntegration#2` (listed in that class's own section in the same way):

```text
         1 row referred to by the records above, and not shown:
            > k={cosmology_type=0, k=1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}, numeric=TkNumericIntegration#2  #2  | values: 2  tags: Run_fixture
```

and in `GkSourcePolicyData`, where `QuadSourceIntegral` names a row the section did not show:

```text
         1 row referred to by the records above, and not shown:
            > source=GkSource#1, policy={Levin_threshold=0.5, numeric_policy=maximize-WKB}, k={cosmology_type=0, k=0.1, cosmology={omega_m=0.31, omega_cc=0.69, h=0.67, f_baryon=0.16, T_CMB_Kelvin=2.7255, Neff=3.046}}  #2
```

### The classes whose lines changed

Compared section by section, the fixture's report at `HEAD` and with this change, verbose and not
(a script in the scratch directory, `render.py` and an `re` split on the `@@` headers):

| Class | How its lines changed | Verbose record line, characters |
|---|---|---|
| `TkNumericIntegration` | its own label `  #n` (named by `TkWKBIntegration`, `QuadSource`, `QuadSourceIntegral`) | 401–492 → 405–496 |
| `TkWKBIntegration` | `numeric=` is a label, and its own label | 565 → 173 |
| `QuadSource` | `Tq`, `Tr` are labels, and its own label | 1109–1200 → 321–336 |
| `GkNumericIntegration` | its own label (named by a set's members) | 137 → 141 |
| `GkWKBIntegration` | its own label (named by a set's members) | 137 → 141 |
| `GkSource` | `parents=` is `S<n>`, its own label, and its sets' lines after its records | 533–550 → 169–170 |
| `GkSourcePolicyData` | `source=` is a label, and its own label | 746–762 → 206–210 |
| `QuadSourceIntegral` | `source`, `data`, `Tq_numeric`, `Tq_WKB`, `Tr_numeric`, `Tr_WKB` are labels | 4316–4462 → 564–565 |

The sections of `version`, `store_tag`, `LambdaCDM`, `QCD_Cosmology`, `BackgroundModel`,
`wavenumber`, `redshift`, `wavenumber_exit_time`, `tolerance`, `IntegrationSolver`,
`GkSourcePolicy`, `QuadSourcePolicy` and `OneLoopIntegral` are identical, line for line, in both
reports, verbose and not. `GkNumericIntegration`, `GkWKBIntegration` and `TkNumericIntegration`
have no sharded parent and change only because another class now names them, which §5's last
stop condition allows. The whole fixture report goes from 31,466 to 14,776 characters verbose, and
from 31,198 to 14,664 not.

### The measured sizes (§3 test 4)

On `production_stand` (the test module's stand-in): a `GkSource` over a parent set of 103, 403 or
1,703 members (the fixture's three redshifts and 100, 400 or 1,700 more, each member naming a Gk
numeric or a Gk WKB row), a second `GkSource` over another set, their two `GkSourcePolicyData`
rows, two `QuadSourceIntegral` rows and the `QuadSource` and Tk rows they name. Longest record line
of the class, in characters, without the bullet; before is `HEAD` (`0dda6c8`) with the same
stand-in, after is this change:

| Members | verbose | GkSource | GkSourcePolicyData | QuadSourceIntegral | QuadSource | TkWKBIntegration | a set's line, after | whole report |
|---|---|---|---|---|---|---|---|---|
| 103 | no | 53,949 → 27 | 53,945 → 21 | 53,943 → 25 | 589 → 587 | 66 → 67 | 5,793 and 5,797 | 333,999 → 21,515 |
| 103 | yes | 54,577 → 27 | 54,573 → 21 | 54,571 → 25 | 591 → 589 | 66 → 67 | 6,003 and 6,007 | 339,915 → 24,069 |
| 403 | no | 208,299 → 27 | 208,295 → 21 | 208,293 → 25 | 589 → 587 | 66 → 67 | 22,593 and 22,597 | 1,260,099 → 55,115 |
| 403 | yes | 210,727 → 27 | 210,723 → 21 | 210,721 → 25 | 591 → 589 | 66 → 67 | 23,403 and 23,407 | 1,283,415 → 65,469 |
| 1,703 | no | 877,850 → 27 | 877,846 → 21 | 877,844 → 25 | 589 → 587 | 66 → 67 | 96,094 and 96,098 | 5,277,411 → 202,123 |
| 1,703 | yes | 888,078 → 27 | 888,074 → 21 | 888,072 → 25 | 591 → 589 | 66 → 67 | 99,504 and 99,508 | 5,376,826 → 246,976 |

- **Before**, a record line grew by about 515 characters a member (`(877,850 − 53,949) / 1,600`), and
  the same set was repeated on every `GkSource`, `GkSourcePolicyData` and `QuadSourceIntegral` line.
- **After**, those three lines are 27, 21 and 25 characters at every size, and a set's line grows by
  about 56 characters a member (`(96,094 − 5,793) / 1,600`), printed once. The test bounds it at 100.
- `QuadSource` and `TkWKBIntegration` change little in the stand-in, whose Tk rows differ in one
  leaf each, so their lines were short already; on the fixture, where those rows differ in several
  fields, they go from 1,109–1,200 and 565 to 321–336 and 173.
- The whole report at 1,703 members is 5.4 MB before and 0.25 MB after, verbose. It is still built as
  one string (§6). Nothing is measured at the production grid itself.

## Deviations from the prompt

1. **The own label sits before `| values:`, and a listed row has a `tags:` suffix.** The prompt
   gives `  #3` "beside `| values:`" and leaves the listing's prefix open. The own label comes after
   `[unvalidated]` and before `| values:`; a listed row's line ends with `  tags: …`, because it is
   not under its tag set's line. The prefix is `> `, headed by a line counting the rows.
   IMPLEMENTATION CHOICE.
2. **The set line keeps the existing member form.** Each member is `{z_source=…, numeric=…,
   wkb=…}`, as `_set_members` rendered it, in a bracketed list, not the prompt's example `z=10:
   numeric=none wkb=…`, which says it is an example. A member is about 56 characters.
   IMPLEMENTATION CHOICE.
3. **`S<n>` counts through every class that holds sets.** Only `GkSource` does, so the label is
   `S1`, `S2`, …; counting on keeps a label unique if another class gains a set. The prompt says "the
   set's ordinal in physical order". IMPLEMENTATION CHOICE.
4. **A class is labelled if the inventory declares it as a parent, whether or not this store has a
   record that names it.** `labelled()` reads `parents` and `parent_sets`, not the records, so the
   report's shape does not depend on a store's contents. IMPLEMENTATION CHOICE.
5. **`_class_lines` lost its `verbose` argument.** Which records a class lists is now
   `_Renderer.shown`, which `needed()` also uses; the function is private. IMPLEMENTATION CHOICE.
6. **The tests build their stand-in stores in memory.** §3 test 4 says "stand-in store"; a parent
   set of 1,703 members and the rows that name it are `ClassInventory` records made over the
   fixture's replicated rows (`_Stand`), not rows in SQLite. The renderer reads only the inventory.
   Tests 2, 3 and 5 also run on the fixture store itself. IMPLEMENTATION CHOICE.
7. **Mutation (ii) cannot label by a serial**, because a `Record` carries none: the inventory
   resolves every reference to a digest. The mutation numbers a class's records in the order the
   reader gave them, which is the serials' order; `TestLabelsAreStable` reverses that order.
   STRUCTURALLY REQUIRED.
8. **Mutation (iv) inlines `TkNumericIntegration`, not only `QuadSource`'s `Tq`.** The prompt says
   "for example `QuadSource`'s `Tq`"; the one-line mutation that does that for `Tq` and `Tr` does it
   for a WKB row's `numeric` and an integral's `Tq_numeric` and `Tr_numeric` too. IMPLEMENTATION
   CHOICE.
9. **The mutations run the three modules that import the report, not the whole suites.** Only
   `test_inventory_report_labels`, `test_inventory_report` and `test_inventory_consumers`
   (48 tests) and `ComputeTargets.tests.test_qcd_cosmology_inventory_record` (3) import
   `tools/inventory_report.py` (`git grep`); logs 09b and 09c ran whole suites. IMPLEMENTATION
   CHOICE.

## Verification performed

- **Existing tests unmodified.** `git diff --cached --name-status` lists `M tools/inventory_report.py`
  and `A Datastore/tests/test_inventory_report_labels.py` among the code; nothing under
  `Datastore/tests/` is modified. `main.py`, `Datastore/store_inventory.py`, the factories, the
  witnesses and the goldens are untouched.
- **Class by class.** Above: eight sections change, thirteen do not.
- **The consumers.** `main.py --inventory` prints this report; `test_inventory_consumers` passes
  unchanged.
- **The six suites**, serially, on the staged tree: above.
- **Black.** `./venv/bin/python -m black --check tools/inventory_report.py
  Datastore/tests/test_inventory_report_labels.py` is clean.
- **The size test measures, and a mutation fails it.** Mutation (i) fails `TestTheSizeIsBounded`
  as well as the distinct-lines test.
- **Replays.** Before the commit, the five blocks below were extracted from this file and applied
  with plain `git apply` in a `git archive` export of the staged tree made into its own git
  repository (`venv` linked, so that the tests that call `git` pass); the result is in the final
  report, since this file cannot record a run that follows itself. After the commit they are
  replayed from `git show HEAD:prompts/datastore-integrity/logs/09d-refer-to-derived-parents-by-label-in-the-inventory-report.md`;
  that result goes to the orchestrator.

## The deliberate-breakage record

Each mutation was applied with `git apply` in a `git archive` export of the staged tree made into its
own git repository, so that `git diff` there showed the mutation alone; the modules that import the
report were run (`python -m unittest Datastore.tests.test_inventory_report_labels
Datastore.tests.test_inventory_report Datastore.tests.test_inventory_consumers`, and
`ComputeTargets.tests.test_qcd_cosmology_inventory_record`); and it was reverted with `git apply
-R`, after which `git diff --name-only -- '*.py'` was empty. Each failing entry is one `FAIL:` or
`ERROR:` line; a sub-test is its own entry, with its parameters. The diffs are pasted from `git
diff`, exactly as applied. The unmutated export passes all 48 and all 3. The ComputeTargets module
passed under every mutation.

### Mutation (i) — label the parent sets by their member count instead of their identity

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 8cd970c..69bff14 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -445,10 +445,10 @@ class _Renderer:
                         digests, key=lambda d: (self.set_sort(name, field, d), d)
                     )
                     out[(name, field)] = {
-                        d: count + i for i, d in enumerate(ordered, start=1)
+                        d: len(self._set_members(name, field, d)) for d in ordered
                     }
                     self._set_digests[(name, field)] = {
-                        count + i: d for i, d in enumerate(ordered, start=1)
+                        len(self._set_members(name, field, d)): d for d in ordered
                     }
                     count += len(ordered)
             self._set_numbers = out
```

`Datastore` (the three modules that import the report): `Ran 48 tests in 22.231s`, `FAILED (failures=9)`; 9 failing entries, 7 of them sub-tests:

- `FAIL` `test_inventory_report_labels.TestDistinctStaysDistinct.test_gksource_differing_in_the_gk_row_at_one_z_source` — sub-test `cls='GkSource', verbose=False, differing='numeric'`
- `FAIL` `test_inventory_report_labels.TestDistinctStaysDistinct.test_gksource_differing_in_the_gk_row_at_one_z_source` — sub-test `cls='GkSource', verbose=True, differing='numeric'`
- `FAIL` `test_inventory_report_labels.TestDistinctStaysDistinct.test_gksource_differing_in_the_gk_row_at_one_z_source` — sub-test `cls='GkSource', verbose=False, differing='wkb'`
- `FAIL` `test_inventory_report_labels.TestDistinctStaysDistinct.test_gksource_differing_in_the_gk_row_at_one_z_source` — sub-test `cls='GkSource', verbose=True, differing='wkb'`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_labels_are_the_ordinals_of_the_listing` — sub-test `store='fixture'`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_labels_are_the_ordinals_of_the_listing` — sub-test `store='stand-in'`
- `FAIL` `test_inventory_report_labels.TestTheSizeIsBounded.test_a_record_line_does_not_grow_with_the_members_of_a_set` — sub-test `cls='GkSource'`
- `FAIL` `test_inventory_report_labels.TestTheSizeIsBounded.test_a_sets_line_grows_by_a_bounded_number_of_characters_a_member` — the test itself (no sub-tests)
- `FAIL` `test_inventory_report_labels.TestTheSizeIsBounded.test_the_whole_report_is_bounded_by_its_sets` — the test itself (no sub-tests)

`ComputeTargets` (`test_qcd_cosmology_inventory_record`): `Ran 3 tests`, `OK`.

### Mutation (ii) — number a class's records in the order they were read instead of the physical order (the stand-in for the serial order, which the reader's order follows)

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 8cd970c..d9dcf2e 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -417,7 +417,7 @@ class _Renderer:
             of_record: Dict[int, int] = {}
             of_digest: Dict[str, int] = {}
             listing: List[Record] = []
-            for _, members in self.groups(name):
+            for _, members in [(None, self.cls(name).records)]:
                 for record in members:
                     listing.append(record)
                     of_record[id(record)] = len(listing)
```

`Datastore` (the three modules that import the report): `Ran 48 tests in 17.920s`, `FAILED (failures=7)`; 7 failing entries, 6 of them sub-tests:

- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_labels_are_the_ordinals_of_the_listing` — sub-test `store='fixture'`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_labels_are_the_ordinals_of_the_listing` — sub-test `store='stand-in'`
- `FAIL` `test_inventory_report_labels.TestLabelsAreStable.test_the_same_inventory_with_every_class_read_in_another_order` — sub-test `verbose=False, shown=None`
- `FAIL` `test_inventory_report_labels.TestLabelsAreStable.test_the_same_inventory_with_every_class_read_in_another_order` — sub-test `verbose=False, shown=1`
- `FAIL` `test_inventory_report_labels.TestLabelsAreStable.test_the_same_inventory_with_every_class_read_in_another_order` — sub-test `verbose=True, shown=None`
- `FAIL` `test_inventory_report_labels.TestLabelsAreStable.test_the_same_inventory_with_every_class_read_in_another_order` — sub-test `verbose=True, shown=1`
- `FAIL` `test_inventory_report.TestFloats.test_the_last_bit_of_k_shows_in_verbose_mode_only` — the test itself (no sub-tests)

`ComputeTargets` (`test_qcd_cosmology_inventory_record`): `Ran 3 tests`, `OK`.

### Mutation (iii) — without `verbose`, define only the rows shown, not the ones referred to and not shown

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 8cd970c..0ad9c54 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -509,7 +509,7 @@ class _Renderer:
                             continue
                         visited.add((of, found))
                         if (of, found) not in listed:
-                            extra.setdefault(of, set()).add(found)
+                            pass
                         work.append((of, found))
             self._needed = (
                 {n: sorted(v) for n, v in extra.items()},
```

`Datastore` (the three modules that import the report): `Ran 48 tests in 16.241s`, `FAILED (failures=5, errors=1)`; 6 failing entries, 6 of them sub-tests:

- `ERROR` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_a_reference_names_the_row_it_points_to` — sub-test `verbose=False, shown=1`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_every_label_has_exactly_one_defining_line` — sub-test `store='fixture', verbose=False, shown=1`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_every_label_has_exactly_one_defining_line` — sub-test `store='fixture', verbose=False, shown=2`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_every_label_has_exactly_one_defining_line` — sub-test `store='stand-in', verbose=False, shown=1`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_without_verbose_the_rows_referred_to_but_not_shown_are_listed` — sub-test `store='fixture'`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_without_verbose_the_rows_referred_to_but_not_shown_are_listed` — sub-test `store='stand-in'`

`ComputeTargets` (`test_qcd_cosmology_inventory_record`): `Ran 3 tests`, `OK`.

### Mutation (iv) — render one sharded parent inline again (`TkNumericIntegration`, which is `QuadSource`'s `Tq` and `Tr`, and the `numeric` of a WKB row)

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 8cd970c..10c09fe 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -287,7 +287,7 @@ class _Renderer:
             found = self.index(of).get(reference, [])
             if len(found) == 0:
                 self._ref_text[cache] = f"<unresolved {of} reference>"
-            elif of in self.labelled():
+            elif of in self.labelled() - {"TkNumericIntegration"}:
                 # a derived row is named by its label, and printed once, in its own section
                 self._ref_text[cache] = f"{of}#{self.ordinals(of)[1][reference]}"
             else:
```

`Datastore` (the three modules that import the report): `Ran 48 tests in 30.603s`, `FAILED (failures=4, errors=4)`; 8 failing entries, 8 of them sub-tests:

- `ERROR` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_a_reference_names_the_row_it_points_to` — sub-test `verbose=True`
- `ERROR` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_a_reference_names_the_row_it_points_to` — sub-test `verbose=False`
- `ERROR` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_a_reference_names_the_row_it_points_to` — sub-test `verbose=False, shown=1`
- `ERROR` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_a_reference_names_the_row_it_points_to` — sub-test `verbose=False, shown=2`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_every_reference_to_a_sharded_class_is_a_label` — sub-test `store='fixture', verbose=False`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_every_reference_to_a_sharded_class_is_a_label` — sub-test `store='fixture', verbose=True`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_every_reference_to_a_sharded_class_is_a_label` — sub-test `store='stand-in', verbose=False`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_every_reference_to_a_sharded_class_is_a_label` — sub-test `store='stand-in', verbose=True`

`ComputeTargets` (`test_qcd_cosmology_inventory_record`): `Ran 3 tests`, `OK`.

### Mutation (v) — number a class's records in the order their labels are met, not the physical order

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 8cd970c..8883beb 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -185,6 +185,7 @@ class _Renderer:
             str, Tuple[Dict[int, int], Dict[str, int], List[Record]]
         ] = {}
         self._labelled: Optional[frozenset] = None
+        self._met: Dict[Tuple[str, str], int] = {}
         self._set_numbers: Optional[Dict[Tuple[str, str], Dict[str, int]]] = None
         self._set_digests: Dict[Tuple[str, str], Dict[int, str]] = {}
         self._needed: Optional[
@@ -289,7 +290,10 @@ class _Renderer:
                 self._ref_text[cache] = f"<unresolved {of} reference>"
             elif of in self.labelled():
                 # a derived row is named by its label, and printed once, in its own section
-                self._ref_text[cache] = f"{of}#{self.ordinals(of)[1][reference]}"
+                met = self._met.setdefault(
+                    (of, reference), 1 + sum(1 for k in self._met if k[0] == of)
+                )
+                self._ref_text[cache] = f"{of}#{met}"
             else:
                 record = found[0]
                 fields, tags = self._ref_fields(of, record)
```

`Datastore` (the three modules that import the report): `Ran 48 tests in 16.514s`, `FAILED (failures=8)`; 8 failing entries, 8 of them sub-tests:

- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_a_reference_names_the_row_it_points_to` — sub-test `verbose=True`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_a_reference_names_the_row_it_points_to` — sub-test `verbose=False`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_a_reference_names_the_row_it_points_to` — sub-test `verbose=False, shown=1`
- `FAIL` `test_inventory_report_labels.TestEveryLabelIsDefinedOnce.test_a_reference_names_the_row_it_points_to` — sub-test `verbose=False, shown=2`
- `FAIL` `test_inventory_report_labels.TestLabelsAreStable.test_the_same_inventory_with_every_class_read_in_another_order` — sub-test `verbose=False, shown=None`
- `FAIL` `test_inventory_report_labels.TestLabelsAreStable.test_the_same_inventory_with_every_class_read_in_another_order` — sub-test `verbose=False, shown=1`
- `FAIL` `test_inventory_report_labels.TestLabelsAreStable.test_the_same_inventory_with_every_class_read_in_another_order` — sub-test `verbose=True, shown=None`
- `FAIL` `test_inventory_report_labels.TestLabelsAreStable.test_the_same_inventory_with_every_class_read_in_another_order` — sub-test `verbose=True, shown=1`

`ComputeTargets` (`test_qcd_cosmology_inventory_record`): `Ran 3 tests`, `OK`.

## Observations not acted on

1. **The existing `test_verbose_renders_every_distinct_record_as_a_distinct_line` can no longer
   fail for a labelled class.** A labelled class's record line ends in its own label, different for
   every record, so comparing whole lines is satisfied whatever the fields. The other existing test,
   `test_verbose_shows_every_key_field_of_every_class`, still catches a dropped field. The new tests
   take the label off before comparing. Not changed (README §5 rule 7); opened as
   `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]`.
2. **The renderer's cost at production size is not measured.** `set_sort` is recomputed for each
   `GkSource` record and again in `set_numbers`, and the sort key of an integral carries its
   parents' sort keys, which for a `GkSource` is a tuple of its members' keys. Nothing here
   measures a store with ~7,000 sets of ~870 members.
3. **The report is still one string** (§6). At 1,703 members and verbose it is 0.25 MB on the
   stand-in.
4. **A duplicate row (two records with one key and tag set) has two labels**, and a reference to
   its digest takes the lower; the duplicate is still its own named problem.
5. **Without `verbose`, every set of a class is numbered but only those named are printed**, so
   `S1` and `S3` can appear without `S2`.

## State handed to the next prompt

- **Names.** `tools/inventory_report.py`: `_Renderer.labelled`, `.groups`, `.ordinals`, `.shown`,
  `.set_numbers`, `.needed`, `.set_line`, `.members_text`; the label forms `<Class>#<n>` and
  `S<n>`, the line forms `  #<n>` (a record's own), `            > ` (a listed row) and
  `         parent set S<n>: ` (a set).
- **For a reader of the report.** A record line's fields end at the first double space, as before;
  anything that parses a report for a parent's key must follow the label to the row that carries
  it.
- **Prompt 10** is the next prompt, and `main.py --inventory` on its throwaway store is the first
  print of this report on rows written by the keyed stages.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 665 (the known flake aside),
  CosmologyModels 39, Datastore **551**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
