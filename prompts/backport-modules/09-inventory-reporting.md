# Prompt 09 — The inventory reporting entry point (F2, part 4 of 4)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit section:** §4 F2
**Depends on:** prompts 06, 07, 08 (all of them — this is the first thing that actually calls the
service end to end)
**Files you may touch:** `main.py` **or** a new `tools/inventory_report.py` (see below), plus the log
and status board.

**Read the logs from prompts 06, 07 and 08 first.** They record the return shape of every class,
which is what you are formatting. Do not re-derive it from the factory source if the logs say it.

---

## What this delivers

The point of F2. After this prompt the user can ask a datastore what it contains and get a readable
answer — which is the capability they asked for, and the only reason the previous three commits are
worth anything.

This is also the first end-to-end exercise of the service, so **expect to find defects in 06–08**.
That is a normal and useful outcome. Where a defect is small and clearly a slip, fix it here and say
so in the log. Where it is a design problem, open a §3 issue and report it rather than reworking
someone else's commit inside this one.

---

## Task 1 — choose where it lives

Two options. Pick one deliberately and record the reasoning.

**(a) A flag on `main.py`.** What upstream does — `SI/main.py:1240-1300` builds the report inline.
`SGWK`'s `main.py` is 2660 lines and already constructs the `ShardedPool` with all its
configuration, units and cosmology model, so a `--inventory` flag that reports and exits reuses all
of that for free. The cost is more code in an already-large file.

**(b) A standalone `tools/inventory_report.py`.** Cleaner separation, and consistent with
`tools/shard_key_audit.py` from prompt 01. The cost is duplicating `main.py`'s pool-construction
preamble — which is not trivial, and which then has two copies to keep in step.

**Recommendation: (a)**, unless reading `main.py` shows the pool construction is already factored
into something a separate tool can import cheaply. The duplication risk in (b) is real and the
whole point of the report is to be run against the same datastore `main.py` uses.

If you take (a), the flag must **report and exit** without running any compute. Check how `main.py`
is structured — it may need the report placed after pool construction but before the work begins.

## Task 2 — the report

Structure it by category rather than dumping 28 classes in registration order. A useful shape:

```
== Datastore inventory: <db file> ==

   -- Cosmology models
      ...
   -- Grid definitions (wavenumbers, redshifts, tolerances)
      ...
   -- Compute targets
      @@ GkSource: 412 validated, 3 unvalidated | 2026-08-01 09:12 – 2026-08-14 17:40
      ...
   -- Value tables
      GkSourceValue: 8,213,904 rows
      ...
```

Requirements:

- **Handle a class that raises.** Upstream wraps each call in `try/except` and prints
  `(error: {e})` (`SI/main.py:1258-1259` and similar). Do the same. A datastore predating some of
  these tables, or a factory whose `inventory` has a bug, must not abort the whole report — and the
  error text is exactly what makes the failure diagnosable. Do not swallow the exception silently.
- **Format timestamps and counts readably.** Thousands separators for counts; a short date-time
  format for ranges; `–` or `to` between the endpoints. Print something explicit like `(empty)` for
  a class with no rows rather than a bare `0` next to a `None` range.
- **Do not print unbounded label lists.** Several classes return label lists that can run to
  thousands of entries. Print the count always, and the labels only under a verbosity flag, or
  truncated with an `… and N more`. Upstream has a `_print_value_list` helper for this shape; look
  at what it does before writing your own.
- **Sort for stability.** Sorted labels and values, so two runs against the same datastore produce
  comparable output.

`ShardedPool.inventory` returns a **value, not an `ObjectRef`** — it `ray.get`s internally in both
branches (prompt 06's log records this). Do not wrap the call in `ray.get`.

## Task 3 — the `store_handler` note

Prompt 05 split `store_handler` from `persist_handler` specifically so a caller can mint associated
datastore objects between compute and persist. If while writing the report you find a natural use
for that hook, it is worth noting — but **do not manufacture one**. E1 stands on its own and does
not need a consumer to justify it. This is a note, not a task.

---

## Do not

- Do not add `inventory()` to any factory here. If one is missing, that is a gap in 07 or 08 — open
  a §3 issue.
- Do not change the return shape of any factory to make formatting easier. Format what is there, or
  report the shape as a problem.
- Do not run compute from the inventory path.
- Do not make the report the default behaviour of `main.py`.

---

## Verification

This prompt is where the feature either works or doesn't, so the verification is the deliverable as
much as the code is.

1. The touched file parses; `black --check` clean if available.
2. **Run the report against a real datastore.** This is the point. You need one in the current
   schema — `test-qcd-db.sqlite` is pre-`a2bd966` and will not serve (`IMPLEMENTATION_STATE.md` §5
   note 1). If prompt 10's smallest-viable-datastore work has not been done yet, doing it here is
   well spent: create the smallest datastore the pipeline will produce and run the report against
   it. **Paste the actual output into the log.**
3. Run it against an **empty** datastore too — freshly created, nothing computed. Every class should
   report empty cleanly, with no exceptions and no `None` leaking into the formatted output. This
   catches the empty-table cases that 07 and 08 could only partially test.
4. Confirm counts against direct SQL. For at least one value table, compare the reported count with
   `SELECT COUNT(*)` summed by hand across the shard files. This is the check that proves prompt
   06's `"sum"` merge policy actually works — a `"latest"`-style policy applied to a count would
   report one shard's value and look plausible.
5. Confirm a class whose factory has no `inventory` (a tag-association factory, say) produces the
   intended "does not provide an inventory service" error rather than a crash, if the report can
   reach one.

If you genuinely cannot create a datastore, this prompt cannot be completed as intended — mark it
⛔ blocked on the board with the specific obstacle rather than committing an unexercised report.

---

## Finish

1. Write `prompts/backport-modules/logs/09-inventory-reporting.md` per `README.md` §5.1. Include:
   - where the entry point went and why (task 1);
   - **the actual report output**, on both a populated and an empty datastore;
   - the count cross-check from verification step 4;
   - every defect found in 06–08, split into what you fixed here and what you raised as a §3 issue;
   - anything about the return shapes that made formatting awkward, as feedback for whether the
     shapes chosen in 07/08 were right.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 09 row, the F2d item row, progress, "last updated",
   any §3 issues. If F2 is now complete end to end, say so on the board.
3. Commit. Suggested message:

```
Report datastore contents from the inventory() service

Last of four commits adding datastore-contents reporting, and the first thing
that exercises the service end to end.

Groups the report by category -- cosmology models, grid definitions, compute
targets, value tables -- rather than by registration order, since the useful
question is usually "how far has this run got" rather than "what is in table
N". Compute targets show validated and unvalidated counts with the timestamp
range; value tables show row counts summed across shards.

Each class is queried independently and a failure is reported inline rather
than aborting the report, so a datastore predating some tables still yields a
useful answer and a faulty factory is diagnosable from the output.

Label lists are counted rather than printed in full: several classes return
lists running to thousands of entries, which is not useful in an interactive
report.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Adjust to match where you put the entry point and what you found.
