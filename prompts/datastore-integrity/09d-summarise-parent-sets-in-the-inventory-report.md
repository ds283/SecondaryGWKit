# Prompt 09d — the inventory report prints each parent set once

**Campaign:** [`README.md`](README.md) · **Board items:** none (a follow-up to 09b and 09c) ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-01, by the user's decision on the review of 09c, as a small prompt of its own
(the pattern of 04b). **Gate:** 09c landed and reviewed (board §1); `git status` is clean.
**Closes:** `[09b-inventory-report-renders-every-member-of-a-parent-set]` (board §3.3), whose index
row is deleted. **Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Sonnet**. One display function changes, in one file, under a property the
existing tests already pin.

**Read first:**

1. [`README.md`](README.md): §5 (rules 1–9) and §5.1.
2. The issue, on the board's §3.3, with its **Widened** line, and its index row in
   [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md).
3. [`logs/09b-key-gksource-on-its-parent-set.md`](logs/09b-key-gksource-on-its-parent-set.md): how
   the parent set came to be rendered (the user's decision on the 09b sketch, note §0.7 decision 2),
   and the inventory's `ParentSet`, `ClassInventory.parent_sets` and `.members`.
   [`logs/09c-key-quadsourceintegral-on-its-inputs.md`](logs/09c-key-quadsourceintegral-on-its-inputs.md),
   observation 1.
4. `tools/inventory_report.py`: the module docstring, `_Renderer` (`ref_text`, `ref_sort`,
   `_set_members`, `set_text`, `set_sort`, `value_text`, `value_sort`), `_class_lines` and
   `format_inventory_report`.
5. `Datastore/tests/test_inventory_report.py` and `Datastore/tests/inventory_report_parsing.py`,
   which parses a report back into its parts. Note `test_verbose_renders_every_distinct_record_as_a_distinct_line`,
   `test_verbose_shows_every_key_field_of_every_class` and `test_no_digest_appears`.
6. `Datastore/tests/test_inventory_consumers.py`: `main.py --inventory` prints this report.

---

## 1. What is wrong

The report renders each record by its physical labels, and a parent reference by that parent's own
resolved key, recursively. Since 09b a `GkSource`'s key includes its **parent set**, one member per
`z_source`, each naming a Gk numeric and/or Gk WKB row. The report prints every member in full, with
each Gk row's `k`, cosmology and tags, at about **196 characters a member**. The set is repeated
wherever a `GkSource` is referenced: on every `GkSourcePolicyData` line and, since 09c, on every
`QuadSourceIntegral` line.

**Measured on `build_full_store()`** (two members a set), record-line lengths in characters:

| Class | Before 09 (`2252b8c`) | At `1e8364d` |
|---|---|---|
| `GkSource` | ~165 | ~557 |
| `GkSourcePolicyData` | ~370 | ~760 |
| `QuadSourceIntegral` | ~400 | ~4,300 |

**At production size** (estimated from the grids, not measured on a store):
- the source grid has about 1,740 `z` samples (the A3 run's log), so a set has up to ~1,740
  members, about 870 on average, and a `GkSource` line is about 170 kB on average and up to ~340 kB;
- there are about 7,000 `GkSource` rows (50 response `k` × ~145 `z_response`), at least as many
  `GkSourcePolicyData` rows, and of order 500,000–750,000 `QuadSourceIntegral` rows.

So `main.py --inventory`, which shows at most five records per tag set, prints a few MB of lines
nobody can read. `--inventory-verbose` would be about 1.2 GB for `GkSource` alone, the same again
for `GkSourcePolicyData`, and of order 100 GB for `QuadSourceIntegral`. The report is built as one
string before it is printed, so in practice it would exhaust memory first. Nothing stored, and
nothing in the fingerprint, is wrong; this is the display alone.

## 2. What is wanted, and what must stay true

**The property the report exists to keep** (store-fingerprint prompt 03, and the test
`test_verbose_renders_every_distinct_record_as_a_distinct_line`): the display drops no key field, so
**two distinct records never render as the same line**. A count-and-range summary of a set (`870
members, z_source 1e14…12`) would break it: two `GkSource`s differing in the Gk row at one
`z_source` — exactly the case 09b keys on — would print identically. So a set cannot simply be
summarised in place.

**Recommended shape: print each distinct parent set once, under a short label, and refer to it by
the label.**
- In the `GkSource` section, after its records, a listing of the parent sets, one line each:
  `         parent set S<n>: <members>`. The labels `S1`, `S2`, … are ordinals in the sets'
  physical order (the order `set_sort` already gives), so a report of the same store always numbers
  them the same way. A label is not a digest, and no 64-hex digest is printed anywhere.
- Every record that carries a set, at whatever depth of a parent reference (a `GkSource` line, a
  `GkSourcePolicyData` line's `source=`, a `QuadSourceIntegral` line's `data=`), renders it as
  `parents=S<n>`. Distinct sets have distinct labels, so the property above holds.
- A set's line is compact. Its members' Gk rows share most of their identity (normally the source's
  own `k`, model and tags), so the listing factors that out once per set, as the report already
  factors out "common to every record". Each member then shows only its `z_source`, which of
  numeric and WKB it has, and any field of its rows that is not common. Aim for tens of characters
  a member, not hundreds. A member whose rows differ from the common part in any field still shows
  that field, so two sets that differ only there are still distinct.
- Without `verbose`, list only the sets that the records shown reference, so the report stays
  bounded. With `verbose`, list every set.
- A set the inventory cannot resolve keeps today's `<unresolved parent set>`.

*Your choice:* another shape is allowed if it keeps every requirement of §4, and the log says why
it is better.

## 3. Tests — new modules beside the existing ones, no Ray, nothing under `var/`

1. **Distinct sets stay distinct.** Two `GkSource` rows identical but for the Gk row named at one
   `z_source`, with the same member counts and `z_source` range, render as two distinct lines with
   two different labels. Do the same for two `GkSourcePolicyData`s and two `QuadSourceIntegral`s
   that differ only through their `GkSource`'s set.
2. **Every label is defined, once.** Every `S<n>` printed anywhere in a report, verbose or not, has
   exactly one `parent set S<n>:` line, and the line's members are the set's members.
3. **The labels are stable.** The same store reported twice, and the same store with its serials
   relabelled (the fixtures' `relabel_serials`), give the same labels for the same sets.
4. **The size is bounded.** On a stand-in store with a `GkSource` of production size (a few
   hundred to ~1,700 members), measure:
   - the record lines of `GkSource`, `GkSourcePolicyData` and `QuadSourceIntegral` (each should be
     independent of the member count);
   - the set's own line, in characters per member;
   - the whole report, verbose and not.
   Record the numbers in the log. The test asserts that a record line does not grow with the member
   count, and that a set's line grows by no more than a stated number of characters per member.
5. **Nothing else changed.** Every existing test passes **unmodified**, `test_inventory_report.py`
   and `inventory_report_parsing.py` included. Every class other than the set-carrying ones renders
   exactly as at `HEAD` (compare the full fixture report before and after, outside the
   `GkSource`, `GkSourcePolicyData` and `QuadSourceIntegral` sections).

**Deliberate breakage.** Each must make a test fail; then restore it:
- (i) label sets by their member count instead of by their identity;
- (ii) list, without `verbose`, only the sets of the `GkSource` records shown, and not those
  referenced from other classes' records;
- (iii) number the sets in the order they are met rather than in physical order.

Each goes in the log as a `git diff`, exactly as applied, with every failing test, counted, and is
replayed twice as README §5 rule 8 and the 09 logs did.

## 4. Acceptance

1. The report prints each parent set once, refers to it by a label everywhere else, and keeps the
   property: two distinct records never render as the same line (§3 test 1).
2. No 64-hex digest appears in any report.
3. Record lines no longer grow with the size of a parent set; the measured sizes are in the log.
4. Every existing test passes unmodified. `black --check` is clean.
5. The issue is closed: moved to the board's §4 with a closing note naming this commit by subject,
   its index row deleted, and the index's count and date corrected by counting.
6. On the board: the §1 row for 09d, the §4 entry, and a new §5 subsection
   `### 5.15 After prompt 09d (<date>)`. All in the same commit.

## 5. Stop conditions — stop and ask

- An existing test would have to change.
- The property of §2 cannot be kept with lines that do not grow with the member count.
- The fix needs a change outside `tools/inventory_report.py` and new test modules (for example to
  `Datastore/store_inventory.py`, whose `ClassInventory.members` the renderer reads).
- A class other than `GkSource`, `GkSourcePolicyData` and `QuadSourceIntegral` renders differently.

## 6. What this prompt does not do

- It does not change the inventory, the fingerprint, any key, any witness or golden, or `main.py`.
- It does not shorten the **other** parent references a `QuadSourceIntegral` line carries since
  09c: its `QuadSource` (with that source's `Tq` and `Tr`) and its four Tk rows, each rendered in
  full. On the fixture they account for most of its growth, about 3,500 characters a line beyond
  the set. Measure that part separately and record it in the log as an observation; it is not this
  prompt's to fix.
- It does not make the report stream rather than build one string.

## 7. The log and the board

`logs/09d-summarise-parent-sets-in-the-inventory-report.md`, in README §5.1's form. In addition:
the shape chosen and why; a sample of the new `GkSource`, `GkSourcePolicyData` and
`QuadSourceIntegral` lines and of a `parent set` line, from the fixture; the measured sizes of §3
test 4, before and after; and the other-parents observation of §6.

`IMPLEMENTATION_STATE.md`: the §1 row for 09d, the issue on §4, and §5.15. `docs/OPEN_ISSUES.md`:
the row deleted, count and date corrected.
