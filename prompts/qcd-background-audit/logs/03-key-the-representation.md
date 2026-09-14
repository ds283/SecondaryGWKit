# Log 03 — Put the `T(z)` representation in the datastore lookup key

**Prompt:** prompts/qcd-background-audit/03-key-the-representation.md
**Commit:** *(this commit)* — Key the QCD cosmology on its temperature representation
**Model:** Claude Opus 5
**Date:** 2026-09-14
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

**`T_Z_REPRESENTATION_VERSION` before this commit: did not exist. After this commit: `1`.**
Prompts **04, 05, 06 and 07 each bump it** — they change the node solve's tolerance, the quantity
splined, the node count and spline order, the segmentation, and the break-point set respectively,
and every one of those is a different background under an unchanged parameter block.

README §7 **D1 option (i)** was taken, unchanged: the column, filtered on in `build()`, fed by a
class constant. Option (ii) was not considered preferable at any point — the argument in the
prompt is decisive, in that a documented "regenerate; there is no migration" cannot *detect* the
stale row it warns about, and the failure this campaign is repairing is invisible precisely
because nothing detects it.

### `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`

- **line 73 (new), inside the `LambdaCDM_GenericEOS` class body before `__init__`:**
  `T_Z_REPRESENTATION_VERSION: int = 1`, preceded by a 38-line comment block on the shape of
  `TkNumericIntegration.BREAK_POINT_KIND`'s. The comment states (a) what the constant identifies —
  the node solve's tolerance, the quantity tabulated and splined, the node count, the spline
  order, the segmentation and its edge placement, and the set
  `integration_break_points` declares; (b) that every prompt in `prompts/qcd-background-audit/`
  that changes any of those bumps it, as a table
  (`version | prompt | what changed`, one row so far: `1 | 03 | nothing numerically; the key
  exists`) that prompts 04–07 append to; and (c) that a stale row is otherwise undetectable, with
  the 3.461e-08 / 1.4e5 rad figures, which is the whole reason it exists.
- Declared on `LambdaCDM_GenericEOS`, **not** on `QCD_Cosmology`, and inherited — `QCD_Cosmology`
  has no copy of its own in `__dict__`, which a test asserts. No other line of the file changed.

### `Datastore/SQL/ObjectFactories/QCD_Cosmology.py`

- **lines 1–19 (new): a module docstring**, a SCHEMA NOTE on the precedent of
  `Datastore/SQL/ObjectFactories/BackgroundModel.py`'s. It records what the column holds, why the
  parameter key cannot see it, that there is no migration, and — the fact a later reader most
  needs — that `Datastore._ensure_tables()` creates absent tables but never alters an existing
  one, so an old datastore keeps its old table and the mismatch surfaces at `build()`'s select.
- **line 24:** `from sqlalchemy.exc import SQLAlchemyError` added.
- **`register()`, lines 52–54 (new):**
  `sqla.Column("T_z_representation", sqla.Integer, index=True, nullable=False)`, last in the
  column list after `log10_max_z`.
- **`build()`, line 78 (new):** `T_z_representation = QCD_Cosmology.T_Z_REPRESENTATION_VERSION`
  — read from the class, never written out as a literal.
- **`build()`, line 94 (new):** `table.c.T_z_representation == T_z_representation` added inside the
  existing `sqla.and_(...)`, as an equality beside — not instead of — the seven
  `abs(column - value) < DEFAULT_FLOAT_PRECISION` comparisons and `log10_max_z`. Not a float
  comparison: this is an identifier and `DEFAULT_FLOAT_PRECISION` has no business near it.
- **`build()`, lines 80–117:** the select is now bound to a local `query` and executed inside
  `try: store_id = conn.execute(query).scalar()` / `except SQLAlchemyError as e:`. The handler
  matches `"T_z_representation" in str(e)` and re-raises `RuntimeError(...) from e`; anything else
  re-raises untouched. This is the pattern at `BackgroundModel.py:300-309` and
  `TkNumericIntegration.py:283-299`. Before this commit the select was a single unguarded
  `conn.execute(...).scalar()` expression.
- **`build()`, line 130 (new):** `"T_z_representation": T_z_representation` in `insert_data`,
  reading the same local the filter reads.
- `inventory()` unchanged; no other file in `Datastore/SQL/ObjectFactories/` touched.

**What a pre-campaign datastore now does, exactly.** `Datastore._build_schema()` builds the
`sqla.Table` handed to `build()` from the factory's `register()` — that is, from the *code* — so
`table.c.T_z_representation` always exists as an attribute and the missing column never presents
as a silently absent one. `Datastore._ensure_tables()` (`Datastore/SQL/Datastore.py:404-407`)
creates a table only `if not self._inspector.has_table(name)`, so an existing `QCD_Cosmology` table
is left exactly as it was written. The mismatch therefore appears only when SQLite executes the
select, as

```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError)
    no such column: QCD_Cosmology.T_z_representation
```

from `conn.execute(query)` on the `store_id = ...` line of `sqla_QCDCosmology_factory.build()`.
That is caught and re-raised as

> `RuntimeError`: QCD_Cosmology.build(): the QCD_Cosmology table has no "T_z_representation"
> column. This datastore predates the T(z) representation becoming part of the cosmology lookup
> key (prompts/qcd-background-audit, prompt 03) and its QCD half must be regenerated; there is no
> migration. Its rows record no representation, so the BackgroundModel, Gk and Tk rows hanging
> from them cannot be told apart from rows computed against the background now in the code.

with the `OperationalError` preserved as `__cause__`. This was verified against a real SQLite
database holding the old table, not reasoned about — see "Verification performed" below.

### `ComputeTargets/tests/test_cosmology_representation_key.py` (new, 555 lines, 15 tests)

No Ray, no project `Datastore` machinery; 0.19 s. Public symbols are the test classes:

- `TestTheDeclaration` (3) — the constant is on `LambdaCDM_GenericEOS`, is **not** shadowed on
  `QCD_Cosmology`, is a non-bool `int` ≥ 1, and is readable from the class without an instance
  (which is load-bearing: `build()` filters on it before any model exists).
- `TestTheKeyField` (2) — `register()` carries `T_z_representation` as a non-nullable
  `sqla.Integer` and not a `sqla.Float`; `build()`'s compiled `WHERE` clause carries it as an
  equality *beside* all seven parameter columns and `log10_max_z`.
- `TestTheFilterReadsTheConstant` (3) — prompt §3 item 2. By `ast`: the single `Compare` whose
  left is `table.c.T_z_representation` has an `ast.Name`, not an `ast.Constant`, as its
  comparator; that name has exactly one binding in `build()`, to
  `QCD_Cosmology.T_Z_REPRESENTATION_VERSION`; and the `insert_data` dict writes **the same name**.
  Then dynamically: re-pointing the single declaration moves the compiled query's criterion with
  it, and restoring it moves it back.
- `TestTwoRepresentationsTwoSerials` (2) — prompt §3 item 1, against a real in-memory SQLite
  database. Version 1 twice returns one serial (`_new_insert` then `_deserialized`, one row);
  version 2 then returns a **different** serial and a second row; the two rows carry 1 and 2. A
  second case checks the row written carries the running version.
- `TestOldSchemaFailsLoudly` (3) — prompt §3 item 3. A genuine old-schema table (the production
  registration with the one column dropped) created in SQLite, queried through the production
  `table` object, and the `RuntimeError`'s message checked for `T_z_representation`, `prompt 03`,
  `regenerated`, `no migration` and `QCD_Cosmology`, with `__cause__` an `OperationalError`. Plus:
  an unrelated `OperationalError` ("database is locked") propagates undisguised.
- `TestLambdaCDMIsUntouched` (2) — prompt §3 item 4. `sqla_LambdaCDM_factory.register()`'s column
  list is exactly the seven parameter names it always was, and `LambdaCDM.py`'s source mentions
  neither `T_z_representation` nor `T_Z_REPRESENTATION_VERSION`.

## Deviations from the prompt

### 1. The named test precedent does not stand up an in-memory datastore — STRUCTURALLY REQUIRED

**The prompt assumed:** §3 — "It needs a datastore, which `test_numeric_break_point_key.py` shows
how to stand up in-memory; if it cannot be done without one, use `ast` to assert the *structure*
instead."

**What was actually there:** `ComputeTargets/tests/test_numeric_break_point_key.py` stands up **no
database at all**. Its own docstring says so: it builds `sqla.Table` objects from the factories'
`register()` output and hands `build()` a stand-in connection that raises `_QueryCaptured` instead
of executing, so that the query can be read back and compiled with `literal_binds`. There is no
engine and no SQLite anywhere in it. So the prompt's first branch had no worked example to follow
and its second branch (`ast` only) was the fallback it offered.

**What was done instead:** both, and neither exactly as offered. An in-memory SQLite engine
(`sqla.create_engine("sqlite://")`) is created directly and the table created on it from the
factory's own `register()` output, with an inserter reduced from `Datastore._insert` to what this
table needs (the engine's serial, plus a timestamp). This is three lines of setup and needs
nothing from `Datastore/SQL/Datastore.py`, so the prompt's worry that it might not be doable
without the real datastore does not arise. That gives prompt §3 items 1 and 3 *real* assertions —
what SQLite does with a candidate row, and what SQLite raises for a genuinely missing column —
rather than assertions about a compiled query. The stand-in-connection technique from
`test_numeric_break_point_key.py` is used as well, for the two tests that are about the *shape* of
the query rather than its result. `ast` is used for prompt §3 item 2, as the prompt directs.

This touches no README §2 design fact.

### 2. A module docstring was added to the factory — IMPLEMENTATION CHOICE

The prompt's §2 item 3 says to follow `sqla_BackgroundModel_factory`'s precedent, and that
precedent *is* a module docstring (a "SCHEMA NOTE" block) — the `RuntimeError` message in that
file's `build()` is a one-sentence pointer at the docstring, not a self-contained explanation. The
alternative was to carry the whole explanation in the exception message, which would have made a
runtime string the only record of a schema decision and would not be readable by someone opening
the file. The docstring is where a reader who has just been handed the `RuntimeError` will look.

### 3. The column is `index=True, nullable=False` — IMPLEMENTATION CHOICE

The prompt specifies "a `T_z_representation` integer column" and no more. `nullable=False` follows
`break_point_kind`'s precedent and is the correct statement: there is no such thing as a QCD
cosmology row with no representation, and a NULL would be exactly the undetectable state the
column exists to abolish. `index=True` follows the same precedent; the table is tiny (a handful of
rows), so the index buys nothing measurable and costs nothing measurable — it is there for
consistency with every other keyed column in the tree, and the alternative (omitting it) would
have been equally defensible.

### 4. `build()`'s select was restructured into `query = ...` / `try: conn.execute(query)` — IMPLEMENTATION CHOICE

Before this commit the select was one expression,
`store_id = conn.execute(sqla.select(...).filter(...)).scalar()`. The missing-column handler needs
the execute inside a `try`, and wrapping the whole expression would have put 15 lines of query
construction inside the `try` where they cannot raise a `SQLAlchemyError` anyway. Binding the
query to a local first is what `TkNumericIntegration.py`'s `build()` does, and it also made the
stand-in-connection test possible (the query is formed before anything executes). No criterion,
ordering or comparison in the query changed.

### 5. The test asserts the version is an `int` ≥ 1, not `== 1` — IMPLEMENTATION CHOICE

Pinning the literal `1` would turn every one of prompts 04–07's required bumps into a test edit,
and a test that must be edited to stay green teaches the next agent to edit it. The properties
that matter — one declaration, on the base class, unshadowed, an integer, and *actually the thing
`build()` filters on* — are asserted directly instead, and
`test_repointing_the_constant_moves_the_query_with_it` is what catches a later prompt that bumps
the constant but breaks the link to the key. The value at each prompt boundary is recorded on the
board's §1 version table, which is where README §8 item 3 says a later reader should find it.

**No `UNINTENDED DRIFT`.**

## Verification performed

All of the following were run on this tree; none is a reasoned claim.

### No number moves — bit-identical, measured

A scratch script dumped `T_photon`, `Hubble` and `rho` as exact `float.hex()` strings on a
4,001-point grid — `z = 0` plus 4,000 points log-spaced in `1+z` over
`z ∈ [10^-3, 10^19]` — for three models: `QCD_Cosmology` (`Planck2018`, `Mpc_units`, `max_z=1e20`),
`LambdaCDM` (`Planck2018`), and `LambdaCDM_GenericEOS` on the constant-`gs` `PureRadiationEOS` of
`CosmologyModels/tests/test_wPerturbations.py` (`max_z=1e20`). 12,003 lines, 36,009 floats. It was
run in a `git worktree` at `3478ae3` (`HEAD~1`) and on this tree:

```
$ cmp before.txt after.txt && echo BIT-IDENTICAL
BIT-IDENTICAL
$ md5 -q before.txt after.txt
022cbbc1faf075233f54f84d5d0959f8
022cbbc1faf075233f54f84d5d0959f8
```

Byte-identical files, identical MD5. Every LambdaCDM, `PureRadiationEOS` and QCD value is
unchanged to the last bit, which is README §0.5's acceptance test and stop condition. The worktree
was removed afterwards (`git worktree list` shows only the main checkout).

### The QCD reference fixture

```
$ PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py --dry-run
  ... 12 science keys, each "unchanged" ...
No change: the QCD block's science content (12 keys) is bit-identical to the shipped file.
Nothing written (not even the provenance fields).
Total wall time 117.118 s (build 0.658 s, reference quadrature 116.449 s).
```

`ComputeTargets/tests/wkb_reference_data.json` is untouched; `git status --porcelain` does not
list it. Prompt 02's `_t_z_representation_version()` now returns `(1, None)` instead of
`(None, "not yet introduced (prompts/qcd-background-audit prompt 03)")`, checked directly, so from
prompt 04 onward a regeneration will stamp `T_Z_REPRESENTATION_VERSION=1` (or later) into the
block's `method` provenance. It prints nothing here because that path runs only when the science
has moved, which it has not.

### Suites

| Suite | Before (`3478ae3`) | After | Time |
|---|---|---|---|
| `CosmologyModels/tests` | 18 OK | **18 OK** | 0.140 s |
| `ComputeTargets/tests` | 339 OK | **354 OK** (+15, this prompt's module) | 155.9 s |
| `LiouvilleGreen/tests`, fast set | 143 OK | **143 OK** | 14.2 s |

The LiouvilleGreen figure is the fast set: every module **except `test_3bessel_analytic`**, run by
name, as prompt 02's log did. `test_3bessel_analytic` was excluded because the full suite takes
~1,400 s and nothing in this commit is reachable from `LiouvilleGreen/` — the diff touches one
class constant, one datastore factory and one new test module, and `LiouvilleGreen/` imports none
of them. No count fell.

The new module alone: `Ran 15 tests in 0.193s / OK`.

### Formatting

```
$ ./venv/bin/python -m black --check ComputeTargets CosmologyModels Datastore Quadrature \
      LiouvilleGreen config Units CosmologyConcepts MetadataConcepts
All done! 145 files would be left unchanged.
```

(`black --check .` from the repository root reports 54 files it would reformat; all of them are
under `venv/` and `thirdparty/`, which black's default excludes do not cover — that is the state
of `3478ae3` too and is not this commit's.)

## Observations not acted on

- **`inventory()` does not report the representation, and neither does `tools/inventory_report.py`.**
  `sqla_QCDCosmology_factory.inventory()` returns `name`, `omega_m`, `omega_cc`, `h` and
  `log10_max_z` per row, and `tools/inventory_report.py:46` lists `QCD_Cosmology` among the tables
  it summarises. From prompt 04 onward a datastore can legitimately hold several QCD cosmology
  rows that differ **only** in `T_z_representation` — same name, same parameters, same
  `log10_max_z` — and the only tool that inspects the datastore will show them as indistinguishable
  duplicates. Adding `"T_z_representation": row.T_z_representation` to the `values` list is one
  line, but the prompt's §2 item 4 is explicit that this commit changes the key and nothing else,
  and `inventory()` is not part of the key. Opened as
  `[03-qcd-inventory-does-not-report-the-representation]` on the board's §3.
- **Nothing else keys on the representation, and nothing else needs to.** `BackgroundModel` is
  keyed on `cosmology_type` + `cosmology_serial`
  (`Datastore/SQL/ObjectFactories/BackgroundModel.py`), and every `Gk`/`Tk` row is keyed on a
  `model_serial` that hangs from it, so a new cosmology serial propagates the whole way down
  without any further column. This was checked, not assumed; it is why prompt §2 item 4's "keying
  the cosmology serial is sufficient" holds.
- **The replica path is unaffected.** `build()`'s `if "serial" in payload` branch, used when a
  shard is handed a serial by the broker, inserts `T_z_representation` alongside the supplied
  serial like any other column.
- **The constant is inherited by every `LambdaCDM_GenericEOS`, but only `QCD_Cosmology` has a
  factory that keys on it.** `PureRadiationEOS`-backed models are built directly in tests and are
  never stored, so nothing is missing today. If a second generic-EOS cosmology acquires a factory,
  it must filter on the constant too; no issue opened, because there is no such factory to fix.

## State handed to the next prompt

- **`T_Z_REPRESENTATION_VERSION = 1`**, declared at
  `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:73` on `LambdaCDM_GenericEOS` and inherited
  by `QCD_Cosmology`. **Prompt 04 must set it to `2`** and add a row to the
  `version | prompt | what changed` table in the comment block immediately above it. Change it on
  `LambdaCDM_GenericEOS`, not on `QCD_Cosmology` —
  `test_the_constant_lives_on_the_base_class_and_qcd_inherits_it` fails if a copy is added to the
  subclass.
- **Nothing else needs to be touched for the datastore to notice.** The column is
  `T_z_representation` on the `QCD_Cosmology` table;
  `Datastore/SQL/ObjectFactories/QCD_Cosmology.py:78` reads the constant into a local and
  `:94` / `:130` use that local in the filter and the insert. Bumping the constant is sufficient
  and is the *only* thing that must be done; it is also *necessary*, and a prompt that moves a
  number without bumping it leaves a datastore silently stale.
- **A pre-prompt-03 datastore raises `RuntimeError` from `sqla_QCDCosmology_factory.build()`**,
  with the message quoted in "What shipped" above, caused by
  `OperationalError: no such column: QCD_Cosmology.T_z_representation`. There is no migration. A
  QCD datastore written before this commit must have its QCD half regenerated.
- **The test module to extend is
  `ComputeTargets/tests/test_cosmology_representation_key.py`** (15 tests, 0.19 s, no Ray). It
  does not pin the version's value, so prompts 04–07 need not edit it. It *does* assert that
  `build()`'s filter reads an `ast.Name` bound to `QCD_Cosmology.T_Z_REPRESENTATION_VERSION` and
  that `insert_data` writes the same name, so a prompt that inlines a literal there will fail
  `test_the_filter_is_not_a_literal`.
- **The bit-identity harness is reproducible.** Dump `T_photon` / `Hubble` / `rho` as
  `float.hex()` on 4,001 points over `z ∈ [0, 10^19]` for `QCD_Cosmology`, `LambdaCDM` and
  `LambdaCDM_GenericEOS(PureRadiationEOS)`, in a `git worktree` at the parent commit and on the
  tree under test, and `cmp` them. On this commit the two files are byte-identical, MD5
  `022cbbc1faf075233f54f84d5d0959f8`. Prompts 04–06 must expect the **QCD** lines to move and the
  **LambdaCDM and `PureRadiationEOS`** lines to stay byte-identical (README §2 (g), §0.5); the
  `PureRadiationEOS` lines are the ones prompt 05 must keep exact rather than merely accurate.
  (A worktree has no `./venv`; symlink the main checkout's.)
- **The QCD reference fixture regenerates with**
  `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py`
  (add `--dry-run` to report without writing). ~117 s, of which 116 s is the reference quadrature.
  On this commit it reports **no change** in all 12 science keys. From prompt 04 it will report a
  change and stamp `T_Z_REPRESENTATION_VERSION=<n>` into the block's `method` provenance, so the
  bump must land in the same commit as the regeneration or the provenance records the wrong
  version. Note `[01-convergence-block-has-a-separate-generator]`: the JSON's top-level
  `convergence` block is **not** written by this generator.
- **Suite counts at this commit:** `CosmologyModels` 18, `ComputeTargets` 354, `LiouvilleGreen`
  143 on the fast set (`test_3bessel_analytic` excluded) / 148 full.
- **No numerical state was created or changed by this prompt.** There are no node counts, spline
  orders, segment edges or accuracies to hand on; prompt 04 inherits exactly the background
  `3478ae3` had.
