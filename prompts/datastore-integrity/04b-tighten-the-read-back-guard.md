# Prompt 04b — tighten the read-back guard: every function, per-function coverage, nothing skipped unsaid

**Campaign:** [`README.md`](README.md) · **Board items:** **G4**–**G6** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** after 04 (`6a97d8a`) and its review (`85cbe28`); `git status` and `git diff --cached`
clean. Prompt 05 is held until this prompt has landed and been reviewed.
**Closes:** nothing on any board. It acts on findings 1–3 of the orchestrator's review of prompt 04
(board §1), which the user decided on 2026-09-28 to fix before prompt 05 rather than carry.
**Opens:** nothing expected.
**Recommended model:** **Sonnet**. This is a small change to a test module that already exists.
The care goes into the new pins, which must be able to fail.

**Read first:**

1. [`README.md`](README.md): §0.1, §2, §4 and §5, including §5.1's log template.
2. The board's **Orchestrator review of prompt 04**, its five findings above all.
3. [`logs/04-extend-the-read-back-guard.md`](logs/04-extend-the-read-back-guard.md).
4. `Datastore/tests/test_factory_select_columns.py` in full, as 04 left it.
5. [`orchestrator/review_04_measurements.py`](orchestrator/review_04_measurements.py) and its
   `.out`. It is the measurement behind the findings and the target figures below. Run it; do not
   edit it.

---

## 1. What is wanted

Prompt 04's guard is the instrument prompt 05 is scored against. The review found three places
where it can go blind with no test failing:

- **It visits a list of names, not every function** (finding 1). The list is `build`,
  `read_batch` and `_ROW_HELPERS`. `validate_on_startup` holds a decidable row binding in seven
  factories, and it runs on every pool open. None of the seven is visited. All seven are clean
  today.
- **Its coverage pin has no function in its identity** (finding 2). A class's
  `_build_impl_serial` and `_build_impl_model` collapse into one `module:class.variable` key, so an
  analyser that stops visiting `_build_impl_model` fails no test. One that stops visiting
  `_build_impl_serial` fails only while the two `WKB_criterio` entries remain, and prompt 05
  deletes them.
- **G2 drops what it cannot resolve without declaring it** (finding 3). There are three cases:
  - 44 `<base>.c.<name>` references whose base is not resolved to a table, at 12
    (function, base) pairs;
  - the module-level `OneLoopIntegral.read_batch`, which G2 never walks;
  - 13 insert calls whose payload is not a dict literal.

  A `tables["X"]` or `inserters["X"]` naming no schema table would also be skipped rather than
  reported.

This prompt closes those three holes. It changes no factory and records no new defect unless the
tightened guard finds one.

---

## 2. What to change

The only file changed is **`Datastore/tests/test_factory_select_columns.py`**. README §5 rule 7
allows it because this prompt names it. No other test changes. No new module is needed; add one
beside it only if you have a stated reason.

**G4 — every function.**
- Visit every `FunctionDef` in the body of every factory class, and every module-level function,
  in every module under `ObjectFactories/`. Decide what to visit without a list of names.
  `_ROW_HELPERS` goes, or stops gating anything.
- Apply `inline_nested_row_helper` wherever it links something, not only in functions named
  `read_batch`. Keep the manufactured undecidable finding for a `read_batch` that links nothing,
  which is the module-level `OneLoopIntegral.read_batch` today. Do not manufacture findings for
  functions that read no rows: most functions (`register`, `add_tag`, …) read none, and that is
  not a hole.
- **Expected:** the 7 `validate_on_startup` bindings join coverage, decidable and clean
  (`review_04_measurements.out` §A). A different result is explained in the log.

**G5 — the function in the coverage identity.**
- `EXPECTED_COVERAGE` and `KNOWN_BLIND` are keyed `module:class.function.variable`. Within one
  function, a variable rebound part-way through is still one key, as before.
- **`KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` keep their keying and their contents.** They are
  the lists prompt 05 is written against (README §4). If the key they use (`Finding.key`) has to
  stay as it is, give the coverage identity its own property.
- **Nothing weakened.** Rewriting the key form is the one permitted change to an existing entry.
  Every entry of 04's `EXPECTED_COVERAGE` (30) must be the projection, with the function dropped,
  of at least one new entry. Show this in the log by a computation, not by inspection.
- **Expected:** 43 decidable entries and 2 blind (`review_04_measurements.out` §D).
  - The projection covers all 30 of 04's entries, plus the 7 `validate_on_startup` ones.
  - The 16 `_build_impl_*` and `read_batch` bindings now match `probe_02.out`'s 16. That closes
    04's deviation 1.
  - The blind two are `BackgroundModelFactory.build.row_data` and the module-level
    `OneLoopIntegral.read_batch`. The last component of its key is yours to name; the
    manufactured finding's `row_var` is currently `read_batch`.

**G6 — G2 declares everything it does not check.**
- Walk module-level functions as well as class methods.
- A `tables["X"]` or `inserters["X"]` naming no table of the built schema is a **problem**, reported
  like a missing column, never skipped. There are none today.
- Every `<base>.c.<name>` whose base does not resolve to a table is recorded as
  (function site, base) and pinned against a declared dict with a reason per entry. Call it
  `KNOWN_UNRESOLVED_TABLES`, alongside `KNOWN_BLIND_COLUMNS`, with a test that fails on any pair
  that is added or removed.
  - **Expected:** 12 pairs (`review_04_measurements.out` §C): nine `subquery`, two `query`, and the
    module-level `OneLoopIntegral.read_batch`'s `table`, a parameter.
  - *Your choice:* you may **resolve** a case instead of declaring it, if the rule is sound and
    stated. Examples: `table` in a module-level function of a module that registers exactly one
    factory, or a `subquery` whose columns are all plain `table.c.x` of a resolved table.
    Whatever is resolved is then checked, and a check that can find nothing is not a resolution.
- Every insert call G2 does not check is pinned the same way, in its own declared dict with a
  reason per entry: `inserter(conn, x)` or `value_inserter(conn, x)` where `x` is not a dict
  literal.
  - **Expected:** 13 sites. They are listed in the log with how each was measured.
  - *Your choice:* resolve some. Typically `x` is a local bound to a dict literal and then
    extended by `x["k"] = …`, and its literal keys can be checked. A payload built by another
    function stays declared.
- Update docstring point 6 to say what G2 now checks and what it declares.

---

## 3. Verification

1. The suite passes on the tree, with `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` unchanged.
2. **Deliberate breakage.** Each of the following must make a test fail. Then restore it:
   - (i) make the analyser skip functions named `_build_impl_model` and nothing else. The coverage
     test must fail. This is finding 2's case.
   - (ii) make it skip `validate_on_startup`. The coverage test must fail.
   - (iii) rename a column read in one `validate_on_startup` to one its query does not select. The
     guard proper must fail.
   - (iv) change one factory's `tables["X"]` to a table name the schema does not have. G2 must
     fail.
   - (v) make G2 skip an unresolved base silently again, which is 04's behaviour. The new
     unresolved-table pin must fail.
   - (vi) make G2 stop walking module-level functions. A test must fail.

   **Every diff in the log is the output of `git diff` taken while the mutation was applied, and
   is pasted unaltered. None is written by hand.** Then, from the committed log, extract each diff
   to a file outside the tree and run, for each: `git apply --check`, `git apply`, the module,
   and `git apply -R`. Record the failing tests. Prompt 04's mutation (v) had a wrong hunk header
   and did not apply (review finding 4), and this rule exists because of it.
3. All six suites, with README §7's command, run serially. Datastore rises by exactly the tests
   added. The others match board §5.2. `black --check` is clean.

---

## 4. Acceptance

1. G4–G6 in the tree. No Ray, no engine, no store, nothing under `var/`. `Datastore/SQL/` untouched.
2. `EXPECTED_COVERAGE` function-qualified, its projection covering all 30 of 04's entries, and the
   log's figures against the expected 43 / 2 / 12 / 13.
3. The deliberate-breakage record, (i)–(vi), replayed from the committed log.
4. On the board: the §1 row for 04b, items G4–G6, and a new §5.3 for the baselines. Do not
   rewrite §5.2. Leave 05 held; its release is the orchestrator's. `docs/OPEN_ISSUES.md` is
   unchanged unless an issue is opened.
5. One commit, with the log at `logs/04b-tighten-the-read-back-guard.md`.

---

## 5. Stop conditions — stop and ask the user

- The tightened guard finds a **new** defect on a path `main.py` reaches during a build.
  `validate_on_startup` runs on every pool open, so a defect there counts. Prompt 05 is held, and
  the user may want the fix moved ahead of it. A new defect off the build path is recorded in
  `KNOWN_UNFIXED` / `KNOWN_UNFIXED_COLUMNS` with a P-item, marked **new**, and does not stop you.
- Any entry of 04's `EXPECTED_COVERAGE` cannot be kept as the projection of a new one.
- The change needs an analyser that runs factory code, or a change to any other test.

---

## 6. What this prompt does not do

- It fixes no factory, and it neither empties nor re-keys `KNOWN_UNFIXED` or
  `KNOWN_UNFIXED_COLUMNS`. That is prompt 05's.
- It does not merge the two unfixed lists (review finding 5). Prompt 05 is written to empty both.
- It does not edit log 04, which the review corrects, or the audit's probes, or
  `orchestrator/review_04_measurements.py`.
