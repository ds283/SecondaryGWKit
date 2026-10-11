# Prompt 09 — close the migration

**Campaign:** [`README.md`](README.md) · **Board items:** **Z1**–**Z6** (and **Z7** if the user
chooses it) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-11, with prompts 05–08, after the review of 04 (`0b6565e`), at the user's
request. README §2.1 had held it until 08's review so that it could carry 08's findings; its
orchestrator note, written after that review, hands it every count and finding that 05–08
produce, and governs where it differs from this file.
**Gate:**
- 08 landed and reviewed, with **both G2s holding** (board §2);
- `git status` is clean;
- nothing is `running` in the registry;
- the user's decisions of README §6.3 are taken and recorded in the orchestrator's note (§1, Z7).

**Closes:** `[04-comments-name-the-layers-old-module-paths]`; the three inherited layer issues on
their own boards (C11); `[04-six-docs-probes-import-factories-from-the-layer]` only if the user
chooses Z7. **Narrows or closes:** `[01-cross-filesystem-move-advice-says-delete-by-hand]` (C11).
**Recommended model:** **Sonnet**. Records, prose and measurements; one small, exact code-free
sweep of comments.

**Read first:**

1. [`README.md`](README.md): §2.2 (09's charter), §4.3, §5, §6.2 (C9–C12) and **§6.3**.
2. This board in full: §2 (the gates), §3 (every open issue), and §5.4–§5.8 with their reviews.
3. [`docs/dk-and-rpk-migration-verification.md`](../../docs/dk-and-rpk-migration-verification.md)
   §1–§3.
4. The four inherited issues, each on its own board:
   - `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]` and
     `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]`:
     `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md` §3;
   - `[00-the-inventory-cannot-see-a-serial-split]`: `prompts/datastore-integrity/IMPLEMENTATION_STATE.md` §3;
   - `[01-cross-filesystem-move-advice-says-delete-by-hand]`: `prompts/store-retirement/IMPLEMENTATION_STATE.md` §3.

   And DatastoreKit's `docs/OPEN_ISSUES.md` §1.2, which already lists all four as inherited from
   SGK. **Read DatastoreKit and RayPoolKit; edit nothing there.**
5. Each package's extraction board, §2 (the gates), so the G2 drafts match its form:
   `/Users/ds283/Documents/Code/DatastoreKit/prompts/extraction/IMPLEMENTATION_STATE.md` and
   `/Users/ds283/Documents/Code/RayPoolKit/prompts/extraction/IMPLEMENTATION_STATE.md`.
6. `CLAUDE.md`, "Repository mechanics", whose test command this prompt rewrites, and
   [`prompts/INDEX.md`](../INDEX.md).

---

## 1. What is wanted

The packages are adopted and both gates hold. 09 leaves the repository saying so, with nothing
outside the dated records pointing at code that is gone, and hands each issue that belongs to a
package to that package.

- **Z1 — the four inherited issues (C11).**
  - **The three that are wholly layer code** close on their own boards' §4, as *transferred to
    DatastoreKit*: each entry moves from §3 to §4 with a line saying when, by this campaign, and
    where it now lives (DatastoreKit's `docs/OPEN_ISSUES.md` §1.2, and its board when it has one).
    Their rows leave SGK's `docs/OPEN_ISSUES.md`.
  - **`[01-cross-filesystem-move-advice-says-delete-by-hand]`.** Read the path: the advice text is
    the package's (`ShardedPool.move_store`'s failure message); `RunRegistry.stores.move_store`
    calls it (`RunRegistry/stores.py`, find the line at your `HEAD`). If SGK only relays the
    package's message, **transfer it whole**, as the other three. If SGK adds or shapes the
    advice itself, **narrow it** to that SGK half: its board entry says what remains SGK's and
    what went to DatastoreKit, and its index row's hook changes. Show the reading either way.
  - Each board you edit gets only that move and the line saying so. Its own §5 and its other
    entries do not change.
- **Z2 — the prose 04 left (`[04-comments-name-the-layers-old-module-paths]`).** Its 23 lines,
  and any 07's log recorded, name a layer module or file, or `RayTools/`, by a path that no longer
  exists here. Rewrite each to name what is current, changing the path and nothing else:
  - `Datastore.store_inventory…`, `Datastore.shard_paths…`, `Datastore.SQL.schema…` →
    `datastorekit.<same>`;
  - `Datastore.SQL.Datastore._build_schema` → `datastorekit.SQL.Datastore._build_schema`; and
    `Datastore.SQL.Datastore._factories`, which has been wrong since `19f07ee`, →
    `config.datastore.factories`, the registry those tests pass;
  - file paths: `Datastore/store_inventory.py` → `datastorekit/store_inventory.py`, and the like;
    ``Datastore/SQL/`` in `config/datastore.py:4` → `datastorekit.SQL`.

  Each line keeps its sentence. Re-wrap only where the new name pushes it past 100 columns.
  Close the issue on the board's §4.
- **Z3 — the test command.** `CLAUDE.md`'s "Repository mechanics" gives one `unittest discover`
  line, for `ComputeTargets/tests`. Make it name all six roots (`AdaptiveLevin`, `ComputeTargets`,
  `CosmologyModels`, `Datastore`, `LiouvilleGreen`, `RunRegistry`), each a command of README §7's
  form, and the Datastore root's four groups (`-p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`),
  with one sentence saying why the groups exist (a Bash call is capped at ten minutes). Change
  nothing else in that section.
- **Z4 — the final measurements**, at your `HEAD`, saved under `measure/out/09-*`:
  - DatastoreKit's `measure_client_imports.py sgk --commit HEAD`: 15 layer files absent, 0 imports,
    strings and `import_module` calls outside `prompts/`;
  - RayPoolKit's `measure_client_pools.py sgk --commit HEAD`: `RayTools/` absent, 0 imports, 54
    calls all `constructs`;
  - the six roots' counts, against 08's;
  - `pip freeze` against 06's after-freeze, and `package_provenance()`.
- **Z5 — the close of the verification record.** `docs/dk-and-rpk-migration-verification.md` gains
  **§4, "The close"**: a table of R0, R1 and R2 (tree, packages, fingerprint, each run's end state),
  Z4's measurements, and **the two G2 records, drafted** for DatastoreKit's and RayPoolKit's
  extraction boards, each in that board's §2 form, naming SGK's commits and runs that each part
  rests on. They are **not** committed in those repositories (C12): the user makes those commits,
  or has a session there make them.
- **Z6 — the records.** This board closes: its status line, every §1 row, §2, and a final
  `### 5.9 After prompt 09 (<date>)` listing every issue this campaign leaves open, with where
  each now stands. README §2's table marks 05–09 landed. `prompts/INDEX.md`'s row says closed, with
  the date, and its open-issue column taken by counting §1.20's rows. `docs/OPEN_ISSUES.md`'s count
  and date are corrected by counting.
- **Z7 — only if the user chooses it at dispatch** (README §6.3): repair the six `docs/` probes of
  `[04-six-docs-probes-import-factories-from-the-layer]`. In each, the one statement importing
  `_factories` from `datastorekit.SQL.Datastore` becomes `from config.datastore import factories as
  _factories`, and nothing else changes. Show each now gets past its imports, by the method 04
  used (five with no arguments, `order_audit.py` through `d3_tables()`), stopping it there. Close
  the issue. **If the user does not choose it, the issue stays open**, unassigned, and §5.9 says so.

**Left for the user, not done here** (README §6.3): retiring the six rehearsal stores, the two G2
commits in the package repositories, and whether to rename the `Datastore` package.

---

## 2. Verification

1. **Nothing outside `prompts/` and dated `docs/` names a deleted file as a current path**
   (README §2.2, 09's acceptance 2). Run, and give in the log, a regex over every tracked file
   outside `prompts/` and `docs/`, excluding import statements and `import_module` calls, for:
   the 15 layer paths in their file (`Datastore/…/x.py`) and dotted (`Datastore.x`) forms,
   `ObjectFactories/base`, `tools/sharded_store`, `tools/shard_key_audit`, and `RayTools`. It
   prints nothing. Show that it is not vacuous: before Z2, it prints 04's 23 lines.
2. **Z2 changes prose only.** For every Python file Z2 touches, `ast.dump` of the module with
   docstrings removed is identical before and after, and comments are not in the AST, so name each
   file's changed lines from `git diff -U0`.
3. **The index.** `docs/OPEN_ISSUES.md`'s header count equals its rows by counting; this campaign's
   §1.20 and the three boards' rows agree with their boards. `prompts/INDEX.md`'s row agrees with
   this board.
4. **The suites.** All six roots, at 08's counts (and, under Z7, unchanged: no test imports the six
   probes). `black --check` clean on every Python file touched.
5. **Z1's reading** of the move advice is shown with the lines it rests on.

---

## 3. Acceptance

1. **Z1–Z6** as §1 says; **Z7** if chosen.
2. §2.1's regex prints nothing, and printed 04's lines before Z2.
3. The index's count and date agree with its rows; every board edited agrees with the index.
4. The six roots pass at 08's counts.
5. **No deliberate breakage beyond §2.1's non-vacuity run**: 09 adds no code and no test.
6. **One commit**, with the log `logs/09-close-the-migration.md` in README §5.1's form.

---

## 4. Stop conditions — stop and ask

- A G2 does not hold on this board when you start.
- An inherited issue is not on DatastoreKit's `docs/OPEN_ISSUES.md` §1.2, or its SGK board says
  something that makes the transfer wrong (an SGK half you cannot place).
- A Z2 line cannot be rewritten without changing its meaning, or names something with no current
  equivalent.
- §2.1's regex finds a current path that is code, not prose.
- Any suite's count moves.
- Anything would need an edit in DatastoreKit or RayPoolKit.

---

## 5. What this prompt does not do

- **Files it changes:** the Z2 lines' files; `CLAUDE.md`'s test command; the three inherited
  issues' boards (`a3-v2-readiness`, `datastore-integrity`, `store-retirement`), their entries
  only; `docs/OPEN_ISSUES.md`; `docs/dk-and-rpk-migration-verification.md` (§4 added); the
  `measure/out/09-*` files; README §2; this board; `prompts/INDEX.md`; the log. Under Z7, the six
  probes. Nothing else.
- It runs no build, opens no store, retires nothing, and starts no Ray.
- It edits nothing in DatastoreKit or RayPoolKit.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
