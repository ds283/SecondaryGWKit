# Prompt 09d — the inventory report refers to a derived parent by a label

**Campaign:** [`README.md`](README.md) · **Board items:** none (a follow-up to 09b and 09c) ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-01, by the user's decision on the review of 09c, as a small prompt of its own
(the pattern of 04b). The user widened it the same day from the parent sets alone to every
reference to a derived row. **Gate:** 09c landed and reviewed (board §1); `git status` is clean.
**Closes:** `[09b-inventory-report-renders-every-member-of-a-parent-set]` (board §3.3), whose index
row is deleted. **Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Sonnet**. One display module changes, under a property the existing tests
already pin; the care is in keeping that property and in leaving every other class as it prints
today.

**Read first:**

1. [`README.md`](README.md): §5 (rules 1–9) and §5.1.
2. The issue, on the board's §3.3, with its **Widened** and **Assigned** lines, and its index row in
   [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md).
3. Logs [`09b`](logs/09b-key-gksource-on-its-parent-set.md) and
   [`09c`](logs/09c-key-quadsourceintegral-on-its-inputs.md):
   - how the parent set came to be rendered (the user's decision on the 09b sketch, note §0.7
     decision 2);
   - the inventory's `ParentSet`, `ClassInventory.parent_sets` and `.members`;
   - log 09c's observation 1.
4. `tools/inventory_report.py`:
   - the module docstring;
   - `_Renderer`: `_ref_fields`, `ref_text`, `ref_sort`, `_set_members`, `set_text`, `set_sort`,
     `value_text`, `value_sort`;
   - `_class_lines` and `format_inventory_report`.
5. `Datastore/tests/test_inventory_report.py`, and `Datastore/tests/inventory_report_parsing.py`,
   which parses a report back into its parts. Note especially:
   - `test_verbose_renders_every_distinct_record_as_a_distinct_line`;
   - `test_verbose_shows_every_key_field_of_every_class`;
   - `test_without_verbose_at_most_five_records_per_tag_set`;
   - `test_no_digest_appears`;
   - that `fields_of` reads a record line only up to its first double space, where the
     `| values:` marker begins.
6. `Datastore/tests/test_inventory_consumers.py`: `main.py --inventory` prints this report.
7. `config/sharding.py`: which classes are replicated and which sharded.

---

## 1. What is wrong

The report renders each record by its physical labels, and a parent reference by the parent's own
resolved key (`ref_text`: the fields of that key that vary across the parent's class). This is
recursive: a parent's fields include its own parents. Prompts 07–09 put **derived rows**, which are
sharded, into the keys of other derived rows. So a record's line now carries, in full, the whole
chain of rows it was computed from.
- `TkWKBIntegration` and `GkWKBIntegration` carry their numeric row (07).
- `QuadSource` carries its `Tq` and `Tr` (09a).
- `GkSource` carries its parent set: one member per `z_source`, each naming a Gk numeric and/or Gk
  WKB row (09b), at about **196 characters a member**.
- `GkSourcePolicyData` carries its `GkSource`, and with it the whole set.
- `QuadSourceIntegral` carries its `QuadSource` (with that source's `Tq` and `Tr`), its
  `GkSourcePolicyData` (with its `GkSource` and the set) and its four Tk rows, the WKB rows each with
  their numeric row (09c).

**Measured on `build_full_store()`** (two members a set), record-line lengths in characters:

| Class | Before 09 (`2252b8c`) | At `1e8364d` |
|---|---|---|
| `GkSource` | ~165 | ~557 |
| `GkSourcePolicyData` | ~370 | ~760 |
| `QuadSourceIntegral` | ~400 | ~4,300 |

One `QuadSourceIntegral` line at `1e8364d`, by field:

| Field | Characters | What it is |
|---|---|---|
| `k`, `q`, `r` | ~120 each | replicated exit times, as before 09 |
| `source` | 1,118 | the `QuadSource` |
| `data` | 747 | the `GkSourcePolicyData` |
| `Tq_numeric`, `Tr_numeric` | ~415 each | the numeric Tk rows |
| `Tq_WKB`, `Tr_WKB` | ~570 each | the WKB Tk rows |

**At production size** (estimated from the grids, not measured on a store):
- **Sets.** The source grid has about 1,740 `z` samples (the A3 run's log), so a set has up to
  ~1,740 members, about 870 on average. A `GkSource` line is then about 170 kB on average, and up to
  ~340 kB.
- **Row counts.** About 7,000 `GkSource` rows (50 response `k` × ~145 `z_response`), at least as
  many `GkSourcePolicyData` rows, and of order 500,000–750,000 `QuadSourceIntegral` rows.
- **`main.py --inventory`.** It shows at most five records per tag set, but prints a few MB of lines
  nobody can read.
- **`--inventory-verbose`.**
  - The parent sets alone: about 1.2 GB for `GkSource`, the same again for `GkSourcePolicyData`,
    and of order 100 GB for `QuadSourceIntegral`.
  - Even without the sets, the other derived parents make each `QuadSourceIntegral` line about
    4 kB, which is of order 3 GB for that class.
  - The report is built as one string before it is printed, so in practice it would exhaust
    memory first.

Nothing stored is wrong, and nothing in the fingerprint; this is the display alone.

## 2. What is wanted, and what must stay true

**The property the report exists to keep** (store-fingerprint prompt 03, and the test
`test_verbose_renders_every_distinct_record_as_a_distinct_line`): the display drops no key field, so
**two distinct records never render as the same line**. So a reference cannot be replaced by a
summary that two distinct parents could share, such as a set's member count and `z_source` range.
Two `GkSource`s differing in the Gk row at one `z_source`, which is exactly the case 09b keys on,
would then print alike.

**Recommended shape: a reference to a derived row is a label, and the row is printed once, in its
own section.**
- **Which references.** A parent reference whose class is **sharded** (`config/sharding.py`) is
  rendered as a label. These are the derived rows: the Tk and Gk numeric and WKB integrations,
  `QuadSource`, `GkSource`, `GkSourcePolicyData`. A reference to a **replicated** class (a model,
  an exit time, a redshift, a tolerance, a policy) renders exactly as it does today.
- **The label** is `<Class>#<n>`, for example `QuadSource#3` and `TkWKBIntegration#2`. `n` is the
  record's ordinal in its own class's verbose listing (tag set by tag set, in physical order, as the
  report already sorts them). So it is the same in every report of the same store, and the same
  after the store's serials are relabelled. It is not a serial, and not a digest.
- **The record carries its label.** Every record line of a class that is referenced shows its own
  label after its fields, beyond the first double space (for example `  #3`, beside `| values:`),
  where `fields_of` does not read it.
- **A parent set** is labelled the same way. In the `GkSource` section, after its records, each
  distinct set gets one line, `         parent set S<n>: <members>`.
  - `S<n>` is the set's ordinal in physical order.
  - Each member is its `z_source` with its numeric and WKB rows **by label**, or `none`: for
    example `z=10: numeric=none wkb=GkWKBIntegration#1`. That is tens of characters a member,
    not hundreds.
  - A record, at whatever depth, renders the set as `parents=S<n>`.
- **Every label printed is defined in the report.** Without `verbose`, a class shows at most five
  records per tag set, so a referenced row may not be among them. Each class section then also
  lists the rows of that class that the report refers to but has not shown, one line each, under a
  prefix that is not a record line's. The same goes for the parent sets referenced. The listing is
  closed under reference, which keeps the report bounded: it holds only what the shown records
  name, and what those name. With `verbose`, every record and every set is shown anyway.
- An unresolvable reference or set keeps today's `<unresolved … reference>` /
  `<unresolved parent set>`.

Distinct parents have distinct labels, and distinct sets distinct set labels, so the property
holds. Every key field is still on the line, as a label where it is a derived parent. A
`QuadSourceIntegral` line should come back close to its pre-09 length.

*Your choice:* another shape is allowed if it meets every requirement of §4, and the log says why it
is better.

## 3. Tests — new modules beside the existing ones, no Ray, nothing under `var/`

1. **Distinct stays distinct.** Each of these pairs renders as two distinct lines:
   - two `GkSource` rows identical but for the Gk row at one `z_source`, with the same member counts
     and `z_source` range;
   - two `GkSourcePolicyData`s, and two `QuadSourceIntegral`s, that differ only through their
     `GkSource`'s set;
   - two `QuadSourceIntegral`s that differ only in one Tk parent, and two only in their `QuadSource`.
2. **Every label is defined, once.** In every report, verbose or not, every `<Class>#<n>` and
   `S<n>` printed has exactly one defining line, and that line is the row or set it names.
3. **Labels are stable.** The same store reported twice, and the same store with its serials
   relabelled (the fixtures' `relabel_serials`), give the same labels to the same rows and sets.
4. **The size is bounded.** On a stand-in store with a `GkSource` of production size (a few hundred
   to ~1,700 members), and integrals that reference it, measure:
   - the record lines of every class that carries a derived parent, which should not grow with the
     member count;
   - a set's line, in characters per member;
   - the whole report, verbose and not.

   Record the numbers in the log. The test asserts that a record line does not grow with the member
   count, and that a set's line grows by no more than a stated number of characters a member.
5. **Nothing else changed.**
   - Every existing test passes **unmodified**, including `test_inventory_report.py` and
     `inventory_report_parsing.py`.
   - A class none of whose parents is sharded, and that nothing references, renders exactly as at
     `HEAD`. Compare the full fixture report before and after, class by class.
   - Name, in the log, every class whose lines changed and how.

**Deliberate breakage.** Each must make a test fail; then restore it:
- (i) label sets by their member count instead of by their identity;
- (ii) label a derived row by its serial instead of by its ordinal;
- (iii) without `verbose`, define only the rows shown, and not the ones referenced but not shown;
- (iv) render one sharded parent's reference inline again (for example `QuadSource`'s `Tq`);
- (v) number a class's records in the order they are met rather than in physical order.

Each goes in the log as a `git diff`, exactly as applied, with every failing test, counted, and is
replayed twice, as README §5 rule 8 and the 09 logs did.

## 4. Acceptance

1. Every reference to a sharded row, and every parent set, is a label in the report, defined once in
   it. References to replicated rows are unchanged.
2. Two distinct records never render as the same line (§3 test 1).
3. No 64-hex digest appears in any report.
4. Record lines no longer grow with the size of a parent set or the depth of a chain of derived
   parents. The measured sizes are in the log, before and after.
5. Every existing test passes unmodified. `black --check` is clean.
6. The issue is closed: moved to the board's §4 with a closing note naming this commit by subject,
   its index row deleted, and the index's count and date corrected by counting.
7. On the board: the §1 row for 09d, the §4 entry, and a new §5 subsection
   `### 5.15 After prompt 09d (<date>)`. All in the same commit.

## 5. Stop conditions — stop and ask

- An existing test would have to change.
- The property of §2 cannot be kept with record lines that do not grow with a set or a chain.
- The fix needs a change outside `tools/inventory_report.py` and new test modules. For example,
  `Datastore/store_inventory.py`, whose classes and members the renderer reads.
- A class none of whose parents is sharded, and that nothing references, renders differently.

## 6. What this prompt does not do

- It does not change the inventory, the fingerprint, any key, any witness or golden, or `main.py`.
- It does not change how a replicated parent is rendered. `k`, `q` and `r` keep their ~120
  characters each, as before 09.
- It does not make the report stream rather than build one string. Measure the verbose report's
  size on the stand-in, and record it.

## 7. The log and the board

`logs/09d-refer-to-derived-parents-by-label-in-the-inventory-report.md`, in README §5.1's form. In
addition:
- the shape chosen and why;
- from the fixture, before and after:
  - a record line of each class whose lines changed;
  - a `parent set` line;
  - a referenced-but-not-shown line;
- the measured sizes of §3 test 4, before and after;
- the list of classes whose lines changed.

`IMPLEMENTATION_STATE.md`: the §1 row for 09d, the issue on §4, and §5.15. `docs/OPEN_ISSUES.md`:
the row deleted, the count and date corrected.
