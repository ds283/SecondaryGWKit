# Log 14 — name a run, verify what it names, and key the background on the grid that built it

**Prompt:** prompts/qcd-background-audit/14-key-the-source-grid-construction.md
**Commit:** *(this commit)* — Name a run at write time and verify it at read time
**Model:** Claude Opus 5
**Date:** 2026-09-15
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

`T_Z_REPRESENTATION_VERSION` is **6** before and after. **No background number moves**: this
commit changes which objects a query matches and what a row records about itself, and nothing
else. The production source grid is **byte-identical** to `HEAD~1` — measured, §1 below.

### 1. `CosmologyConcepts/wavenumber.py` — the construction version

New module constant, at the head of the cosmology-aware-source-grid block (`:25-51`):

```python
SOURCE_GRID_CONSTRUCTION_VERSION = 1
```

Its comment block says what it identifies (the *algorithm*, not the grid), why it is not
substitutable for the content digest prompt 11 introduced (a digest cannot be inverted or
range-queried, and two cosmologies in one run legitimately have different grids and the same
construction), that **a bump is the only signal a datastore gets**, and what version 1 is:
prompt 11's protected set at `SOURCE_GRID_BREAK_STANDOFF = 0.25`,
`SOURCE_GRID_BREAK_HALF_WIDTH = 5`, `SOURCE_GRID_BREAK_REFINEMENT = 2`, the two equality
redshifts, and a uniform base density. Nothing else in the file changed; `build_z_sample` and
`populate_source_grid` are untouched.

`CosmologyConcepts/redshift.py` was **not** touched. Prompt 11's `redshift_grid_digest` and
`redshift_array.digest()` already expose everything this prompt needs.

### 2. `extract_common.py` — the run-naming convention, and the read path's selection

New module docstring, and a new section carrying the whole mechanism (`:44-211`):

| symbol | signature | what it is |
|---|---|---|
| `RUN_LABEL_TAG_PREFIX` | `"Run_"` | the prefix that lets a reader recover the set of runs from the `store_tag` table alone |
| `SOURCE_GRID_CONSTRUCTION_TAG_PREFIX` | `"SourceGridConstruction_"` | likewise for the grid's generation |
| `run_label_tag(run_label: str) -> str` | | the store_tag label a run records itself under |
| `source_grid_construction_tag(version: int) -> str` | | the store_tag label recording the construction |
| `RunSelection` | `NamedTuple(label: Optional[str], tags: list, description: str)` | which run a reader is reading |
| `add_run_selection_argument(parser) -> None` | | adds `--run-label`, defaulting to `None` |
| `available_run_labels(pool) -> List[str]` | | the runs a datastore holds, from `pool.inventory("store_tag")` |
| `choose_run_label(available, requested) -> RunSelection` | | **pure**: the four-case decision, testable with no datastore |
| `resolve_run_selection(pool, requested) -> RunSelection` | | the above, with the chosen run's `store_tag` materialised |
| `describe_background_generation(model) -> str` | | renders what `build()` found, including "unknown" |

`main.py` imports `run_label_tag` and `source_grid_construction_tag` from here, so the writer and
the six readers agree on the tag spelling by construction rather than by a test. Deviation 1
argues that direction of dependency.

### 3. `Datastore/SQL/ObjectFactories/BackgroundModel.py` — the key

New `SCHEMA NOTE (prompt 14)` in the module docstring, stating the defect and — explicitly — that
the two paths give **different answers** to a pre-prompt-14 store.

- `register()` gains `source_grid_digest` (`String(DEFAULT_STRING_LENGTH)`, indexed, non-nullable)
  and `source_grid_construction` (`Integer`, indexed, non-nullable). Neither is a `Float`: both
  are compared for equality and `DEFAULT_FLOAT_PRECISION` has no business near them.
- `build()` is restructured around a local `_build_query(with_grid_identity: bool)`. It reads
  `source_grid_construction = SOURCE_GRID_CONSTRUCTION_VERSION` and
  `source_grid_digest = z_sample.digest() if z_sample is not None else None` into locals and
  filters on the locals — never a literal, asserted by `ast` (prompt §3 item 3).
  - **compute path** (`z_sample` supplied): both are in the `WHERE` clause. A model tabulated on
    a different grid, or built by a different construction, misses.
  - **read path** (`z_sample is None`): the two columns are *selected* but not filtered on,
    because the caller is asking which grid was used rather than asserting one.
  - `one_or_none()` became `list(conn.execute(...))`, so that a match spanning more than one
    generation can be **named**: the `RuntimeError` says how many rows and how many generations it
    found, lists every `(construction version, digest)`, and says to narrow with `--run-label`. Two
    rows of the *same* generation are no more distinguishable, so they refuse too, with the same
    message. The old `print("!! ...multiple results found...")` line is gone, replaced by it.
  - a missing column raises the campaign's `RuntimeError` on the compute path (naming prompt 14,
    "regenerated", "no migration", and pointing at `extract_*.py` for the archival case), and on
    the read path silently falls back to `_build_query(False)`.
  - the returned object carries `obj._source_grid_identity` — `(construction, digest)`, or `None`
    for a store that records neither.
- `store()` writes both columns, from `obj.z_sample.digest()` and the same single declaration
  `build()` reads.

### 4. `main.py` — the argument, the two tags, and 27 tag lists

- `DEFAULT_RUN_LABEL = "default"` (`:78-91`), with the comment block arguing why it is a fixed
  string rather than a generated unique one (deviation 2).
- `--run-label` (`:120-131`).
- `run_label = args.run_label` (`:3145`), beside `zend` / `source_samples_per_log10z`.
- Two new tags in the tag block (`:683-706`): `RunLabelTag`, from `run_label_tag(run_label)`, and
  `SourceGridConstructionTag`, from
  `source_grid_construction_tag(SOURCE_GRID_CONSTRUCTION_VERSION)`. Both built from the helpers,
  never as f-string literals — asserted by `ast`.
- A line announcing the run and its grid (`:726-729`).
- Both tags added to the `BackgroundModel` lookup and to the **26** further tag lists — every
  list in the file that names `TkProductionTag` or `GkProductionTag`, counted by `ast`. `LargestSourceZTag`,
  `SmallestSourceZTag` and `SourceSamplesPerLog10ZTag` are untouched, per prompt §2's note on the
  descriptors; `SourceSamplesPerLog10ZTag` is prompt 15's to retire.

### 5. `extract_*.py` — six scripts, the same three hunks each

`extract_Gk_data.py`, `extract_GkSource_data.py`, `extract_GkWKB_data.py`,
`extract_QuadSourceIntegral_data.py`, `extract_TkWKB_data.py`, `extract_tensor_source_data.py`:

1. `add_run_selection_argument(parser)` before `args = parser.parse_args()`;
2. `run_selection = resolve_run_selection(pool, args.run_label)` as the first statement inside the
   `with ShardedPool(...) as pool:` block, with its `description` printed;
3. `tags=run_selection.tags` on the `BackgroundModel` lookup — plus a printed
   `describe_background_generation(model)` line — and on every compute-target lookup the script
   makes: `GkNumericIntegration`, `GkWKBIntegration`, `TkNumericIntegration`,
   `TkWKBIntegration`, `GkSource`, `QuadSource`, `QuadSourceIntegral`. `GkSourcePolicyData` is
   *not* tagged, because `main.py` does not tag it either — it is keyed on its `GkSource` proxy.

**No script reproduces `main.py`'s tag lists** (prompt §2 item 4, last bullet), and a test asserts
that none of them so much as mentions `SourceZGridSizeTag`, `SourceSamplesPerLog10ZTag`,
`OutsideHorizonEfoldsTag`, `TkOneLoopDensity` or `GkOneLoopDensity`. The scripts are readers;
their requirement is selection and disambiguation.

### 6. Tests

- `ComputeTargets/tests/test_run_identity.py` (**new**, 30 tests, 0.3 s): eight classes covering
  prompt §3 items 2–9. The three that matter most run the production `store()`, `validate()` and
  `build()` against a real in-memory SQLite database built from the factories' own `register()`
  output — the same pattern as prompt 03's `test_cosmology_representation_key.py`, widened to the
  six tables `BackgroundModel` reaches into (`store_tag`, `tolerance`, `IntegrationSolver`,
  `redshift`, `BackgroundModel`, `BackgroundModel_tags`, `BackgroundModelValue`).
- `ComputeTargets/tests/test_source_grid.py`: `+2` tests
  (`TestTheConstructionVersionNamesThisAlgorithm`), pinning version 1 to the three constants it
  names and pinning both production grids by length **and digest**.
- **Board:** closes `[11-background-model-not-keyed-on-the-source-grid]` (§3 → §4), opens
  `[14-archival-read-stops-at-the-pre-gktk-value-columns]` and
  `[14-no-archive-of-grid-construction-algorithms]` (§3), widens
  `[03-qcd-inventory-does-not-report-the-representation]`; `docs/OPEN_ISSUES.md` moves from 64 to
  **65** in the same commit.
- `ComputeTargets/tests/test_main_plumbing.py` was **not** changed: its
  `load_main_py_functions` is used as it stands, and the new `main.py` assertions are made against
  the `ast` tree directly, because they are about *every* tag list rather than about one function.

## Deviations from the prompt

### 1. `IMPLEMENTATION CHOICE` — the run-naming convention lives in `extract_common.py`, and `main.py` imports it

The prompt's file list allows `main.py`, the two `CosmologyConcepts` modules "only to expose the
construction version", the `BackgroundModel` factory, `extract_common.py` and the `extract_*.py`
scripts. The tag spelling has to be known to **both** the writer and the readers, and none of
those is a natural shared home:

- `main.py` cannot be imported (`CLAUDE.md`), so a constant declared there reaches the scripts only
  as a second copy, kept in step by a test — which is exactly the "never write a literal" failure
  mode prompt §2 item 3 is about, one level out.
- `CosmologyConcepts/wavenumber.py` and `redshift.py` are permitted "only to expose the
  construction version", and a *run label* is not a redshift or wavenumber concept in any case.
- a new top-level module is not on the file list.

So `extract_common.py` owns it and `main.py` does `from extract_common import run_label_tag,
source_grid_construction_tag`. The direction is admittedly backwards from the module's name —
it reads as the readers' module — but the import is cheap (`extract_common` pulls in `ray`,
`ComputeTargets` and `CosmologyConcepts`, all of which `main.py` already imports, and no
matplotlib), there is no cycle, and the alternative is two copies of a string. The module
docstring says so in terms. **If a later prompt is allowed to add a top-level module, move both
prefixes and the two helpers there**; nothing else needs to change.

### 2. `IMPLEMENTATION CHOICE` — the default run label is fixed (`"default"`), not generated

Prompt §2 item 1 says "a generated default that is still unique and still human-legible is the
obvious answer, but argue it". It is the wrong answer here, and the reason is the decision made
in the same item: the label follows `TkProductionTag`'s mechanism, so it is a **selection
criterion** and not merely a descriptive tag.

A label that changed from run to run would then make every unlabelled run miss every object in
the store. The bill for that is measured: log 11 §5 records **15,020 objects and 6 m 35 s plus an
unfinished `QuadSourceIntegral` stage** for a *single-model, 5×5-wavenumber* run, with production
×10 in each wavenumber sample, ×85 on `QuadSource` and of order ×850 on `QuadSourceIntegral`, over
two models. A user who ran `main.py` twice without arguments would silently pay it.

A fixed default instead means unlabelled runs all extend one run — which is precisely the
behaviour every run before this commit had — and naming a run becomes something the user opts
into. It also makes the read path's "exactly one run in the store needs no `--run-label`" case the
normal one.

**A label may be reused deliberately, and that is the supported way to extend a run.** Re-running
under an existing name finds everything already tagged with it and computes only what is missing;
that is the same gesture as not naming a run at all. A *new* name starts a new run and computes it
from scratch. Both are stated in the `--run-label` help text and in `DEFAULT_RUN_LABEL`'s comment.

### 3. `IMPLEMENTATION CHOICE` — verification is one construction and one digest *per lookup*, not per label

Prompt §2 item 4 says "verify that everything pulled under that label is internally consistent:
one construction version, one digest". Taken literally across a whole label that is **false by
construction**: a production run holds two models, and their grids legitimately differ — QCD 1,773
samples and digest `303f9ce7`, LambdaCDM 1,732 and `0960e169`. One digest per label would refuse
every real run.

The invariant that is both true and sufficient is *per lookup*: the tags a reader supplies,
together with the cosmology and tolerances it is asking about, must match **at most one**
generation. That is what `build()` enforces, and it catches exactly the case prompt §3 item 5
names — a label reused across a changed configuration — because the two rows then differ in their
digest under the same `(cosmology, atol, rtol, label)`. Verified in the tree by
`TestALabelSpanningTwoConfigurationsIsRefused`, both for two digests at one construction and for
two constructions.

### 4. `IMPLEMENTATION CHOICE` — verification is in the factory, not in `extract_common.py`

The prompt puts "verify by digest" under §2 item 4, the `extract_common.py` item. It is
implemented in `sqla_BackgroundModelFactory.build()` instead, and `extract_common` only renders
the answer. Three reasons: the grid identity is a *column* on `BackgroundModel` and a tag on
nothing, so only the factory can see it; `BackgroundModel` is the first lookup every one of the
six scripts makes, so a refusal there stops the script before it reads anything else; and putting
it in the factory means the same check also protects `main.py` and any future reader, rather than
only the six scripts that remember to call it.

### 5. `IMPLEMENTATION CHOICE` — `test_main_plumbing.py` and `test_source_grid.py` were touched only lightly

Prompt §3 item 8 asks for the `main.py` assertions "by `ast` through
`test_main_plumbing.load_main_py_functions`". `load_main_py_functions` *executes* named top-level
functions; the assertions this prompt needs are about the argument parser and about every `tags=`
list in the file, neither of which is inside a function. They are therefore made against
`ast.parse(MAIN_PY.read_text())` directly, reusing `test_main_plumbing.MAIN_PY` so there is still
exactly one record of where `main.py` is. `test_source_grid.py` gained two tests and no
modifications to existing ones.

## Verification performed

### 1. The grid is unchanged — element for element, on both models

Prompt §4 row 1. Reproducing `main.py`'s own grid construction
(`cosmology_feature_redshifts` read out of `main.py` with `ast`, then `build_z_sample`, then
`winnow(12, protect=...)`) on both production models, at `HEAD` and in a detached worktree at
`9d57ff3` (= `HEAD~1`):

| | samples | digest | response | response digest | protected |
|---|---|---|---|---|---|
| LambdaCDM | **1,732** | `0960e169` | **145** | `69050b4c` | 0 |
| QCD | **1,773** | `303f9ce7` | **156** | `197b46de` | 8 |

Identical at both commits, and identical to log 11 §5's four production tag labels
(`SourceRedshiftGrid_1732_0960e169`, `ResponseRedshiftGrid_145_69050b4c`,
`SourceRedshiftGrid_1773_303f9ce7`, `ResponseRedshiftGrid_156_197b46de`) to the character.

Stronger, because a digest is only a 32-bit summary: every source, response and protected value of
both models dumped as exact `float.hex()` — **3,814 lines, MD5 `d5ecc0aa85f38578d8c57c051d3f11e0`
— is byte-identical between `HEAD` and `9d57ff3`** (`diff` reports no difference). The grid did
not move.

### 2. The acceptance table

| Quantity | Requirement | Measured |
|---|---|---|
| Source grid, both models | element-for-element identical to `HEAD~1` | **3,814 `float.hex()` lines byte-identical**, §1 above |
| `BackgroundModel` across two generations | misses; same generation hits | `test_a_different_construction_version_misses`, `test_a_different_grid_misses`, `test_the_same_grid_and_generation_reuse_one_row` — two rows in the table, the pre-existing row not served |
| Run round-trip by label | same objects back | `test_a_named_run_reads_back_its_own_objects` — written under `Run_march` and `Run_september`, read back by tag, same serial, same label, same sample count, right digest |
| Label spanning two configurations | refused, both named | `test_it_refuses_and_names_both_generations` (two digests) and `test_two_constructions_under_one_label_are_also_refused` (two construction versions); the message carries both and the words `--run-label` |
| Store with no label | reads, labelled unknown | `test_the_read_path_still_reads_it_and_labels_it_unknown` — the row deserialises, `_source_grid_identity` is `None`, `describe_background_generation` says "grid generation unknown (this datastore predates prompt 14)" |
| Store with one run, no label given | reads, says which | `test_a_store_with_exactly_one_run_needs_no_label_and_says_which` |
| Tag literals in any filter | none | `test_the_construction_filter_is_not_a_literal`, `test_the_digest_filter_is_not_a_literal`, `test_the_insert_writes_what_the_filter_reads`, `test_the_tag_labels_are_built_from_the_shared_helpers` |
| `T_Z_REPRESENTATION_VERSION` | unchanged | **6** before and after (read from `LambdaCDM_GenericEOS`) |

### 3. What a pre-prompt-14 store does, exactly — and it is two different answers

Both measured, not reasoned about; the old schema is reproduced by dropping the two columns from
the factory's own `register()` output, because `Datastore._ensure_tables()` creates absent tables
but never alters an existing one, so an old database keeps its old table while
`Datastore._build_schema()` builds the `Table` object from the code.

**Compute path** (`main.py`; `z_sample` supplied) — SQLite raises
`OperationalError: no such column: BackgroundModel.source_grid_digest` from `build()`'s select,
and `build()` converts it to:

> `BackgroundModel.build(): the BackgroundModel table has no "source_grid_digest" /
> "source_grid_construction" columns. This datastore predates the source grid becoming part of the
> background model's lookup key (prompts/qcd-background-audit, prompt 14) and must be regenerated;
> there is no migration. Its rows record no grid identity, so a background tabulated on a grid that
> no longer exists cannot be told apart from one tabulated on the grid this run builds. Read such a
> store with the extract_*.py scripts, which report it as an unknown generation rather than
> failing.`

with the `OperationalError` preserved as `__cause__`. An unrelated `OperationalError` ("database
is locked") is **not** disguised — asserted.

**Read path** (`extract_*.py`; `z_sample=None`) — the same `OperationalError` is caught and the
query is re-issued without the two columns. The row deserialises exactly as it did before this
commit, `_source_grid_identity` is `None`, and the script prints
`<model> background model: grid generation unknown (this datastore predates prompt 14)`.
`resolve_run_selection` finds no `Run_*` tags and reports
*"this datastore records no run labels: reading it as a single unnamed run of unknown generation"*.
**A superseded datastore stays readable and plottable**, which is prompt §2 item 4's stated
requirement and its stated failure condition.

### 4. What `source_samples_per_log10z` turned out to feed (prompt §2 item 5)

Traced, not assumed. It reaches exactly three places in production code:

1. **`build_z_sample`'s base density** (`main.py:639` → `populate_source_grid`) — the grid itself,
   which is prompt 15's.
2. **`SourceSamplesPerLog10ZTag`** (`main.py:717`) — a label, and a filter in 27 tag lists.
3. **`delta_logz=1.0 / float(source_samples_per_log10z)`** at `main.py:916`
   (`TkNumericIntegration`) and `main.py:1533` (`GkNumericIntegration`). **This is the one that
   looks computational and is not.** `Quadrature/integrators/numeric_with_phase_cut.py:640-645`
   accepts `delta_logz` and hands it to `NumericIntegrationSupervisor` purely "so that callers
   (main.py) need not change"; the oscillation-resolution diagnostic has been computed after the
   solve, from the actual spacing of the returned samples, since `GkTk-remedial` prompt 11
   (`scan_sample_grid_for_unresolved_osc`). Inside the supervisor the value is read only by
   `report_wavelength`, which that module's own docstring records as having **no callers**, and by
   the three `has_unresolved_osc` / `unresolved_z` / `unresolved_efolds_subh` properties, whose
   values are not the ones returned — the returned dict is `osc_diagnostic`, from the scan.

**So nothing downstream computes from `source_samples_per_log10z`, and prompt 15 can retire the
tag without a numerical consequence** — but it should delete or re-source the two `delta_logz`
arguments in the same commit rather than leave them reading a parameter that no longer describes
the grid.

### 5. Which extract scripts were executed, and which were not

**None of the six was executed.** Each opens a Ray connection and a `ShardedPool` at module scope
and then requires a populated datastore, and the only store in the tree is the LambdaCDM
`physics-test-n20-lambdacdm-zend0p1` shard set at the repository root — which is **exactly the
archival case** and was used as one (§6 below), not as a pipeline target.

What **was** executed:

- every one of the six compiles (`python -m py_compile`), and every one parses;
- `extract_common`'s mechanism was exercised directly, both in the test tree
  (`TestChoosingARun`, six tests) and at the interpreter, including `available_run_labels` against
  a stand-in pool returning `['Run_march', 'Run_sept', 'TkOneLoopDensity']` → `['march', 'sept']`;
- the *query construction* of all six was asserted by `ast`
  (`TestEveryExtractScriptSelectsARun`): each adds the argument, resolves a run, passes
  `run_selection.tags` — and, specifically, passes exactly `run_selection.tags` as the
  `BackgroundModel` lookup's `tags=` — and none of them mentions any of the writer's tags or the
  run prefix;
- the read path itself — a tagged `BackgroundModel` lookup with `z_sample=None`, including the
  archival fallback — was executed against a real SQLite database in
  `TestARunRoundTripsByName` and `TestAStoreWithNoGridIdentity`.

**Nothing here claims that a script produces the same plots as before.** What is claimed is that
its queries are constructed as described and that the lookup they all begin with behaves as
measured.

### 6. The read path against a **real** archival store — and the limit it revealed

`physics-test-n20-lambdacdm-zend0p1-shard0000.sqlite` (repository root) is a genuine pre-campaign
datastore: its `QCD_Cosmology` table has no `T_z_representation` column (pre-prompt 03), its
`BackgroundModel` table has neither of this prompt's columns, and its nine `store_tag` labels
include `SourceRedshiftGrid_1732` — prompt 11's digest is absent — and no `Run_*` at all. A copy
was opened read-only and the production `build()` run against it directly (no Ray, no
`ShardedPool`):

```
store_tag labels present (9): ['GkOneLoopDensity', 'LargestSourceRedshift_2.0636e+16',
  'OutsideHorizonEfolds_e3', 'ResponseRedshiftGrid_145', 'ResponseSparsenessZ_12',
  'SmallestSourceRedshift_0.1', 'SourceRedshiftGrid_1732', 'SourceSamplesPerLog10Z_100',
  'TkOneLoopDensity']
runs recovered: []
choose_run_label -> this datastore records no run labels: reading it as a single unnamed run of
  unknown generation (it predates prompts/qcd-background-audit prompt 14)
```

**The prompt-14 fallback fires and works**: `build()`'s first select raises
`no such column: BackgroundModel.source_grid_digest`, the read path re-issues
`_build_query(False)`, and the row is found. **It then fails one guard further in**, on the
*pre-existing* `GkTk-remedial` prompts 03/04 refusal:

> `BackgroundModel.build(): the BackgroundModelValue table has no "tau_lo_Mpc" column. This
> datastore predates the double-double conformal-time, sound-horizon and friction node tables
> (prompts/GkTk-remedial, prompts 03 and 04) and must be regenerated; there is no migration.`

**That is a real limit on the archival guarantee and it is recorded, not argued away.** Prompt 14
makes a store that predates *prompt 14* readable; it does not, and could not, make one readable
that predates the columns whose **values** it needs — `tau_lo_Mpc`, `cs_tau_Mpc`, `cs_tau_lo_Mpc`
and `friction_F` are quantities the old store never computed, where a grid identity is only
metadata the old store never recorded. Softening that guard would mean fabricating background
values. It is observation 5 below. The one store in the tree is therefore old enough to exercise
prompt 14's fallback but too old to be read all the way; a store written between
`GkTk-remedial` prompt 04 and this commit reads through.

### 7. Suites

| suite | before (`9d57ff3`) | after | |
|---|---|---|---|
| `CosmologyModels/tests` | 30 | **30** | OK |
| `ComputeTargets/tests` | **392** (measured in a detached worktree at `9d57ff3`, 182.2 s) | **424** | OK |
| `LiouvilleGreen/tests` | 148 (log 13, **full** set) | **148** | OK, 1,422.6 s — the **full** set, not the fast set prompts 02–08 used |

+32 in `ComputeTargets` = 30 (`test_run_identity.py`) + 2 (`test_source_grid.py`). None falls.

**One anomalous run, recorded rather than passed over.** The *first* full `ComputeTargets` run of
this prompt reported `Ran 424 tests ... FAILED (failures=1)`. It was taken while the
`CosmologyModels` suite and a `black` pass over the whole tree were running in the same shell, and
its output filter (`grep -E "^(Ran|OK|FAILED|ERROR)"`) discarded the failure's identity. **Three
subsequent runs on an unloaded machine are `424 OK`**, one of them with the complete unfiltered
output retained, so it did not reproduce and cannot be attributed. The only load-sensitive
assertion in the suite is `test_background_tau.py`'s `assertLess(t_on, 50.0)` µs throughput guard,
which reads **0.34 µs** on both models when run alone — a margin of 147×, so even that is an
uncomfortable fit for the explanation. Stated here because an unexplained failure is a fact about
the run, not about the commit, and the next reader should know it happened.

`black` (no configuration) reports all twelve changed and added files unchanged.

## Observations not acted on

1. **`sqla_BackgroundModelFactory.inventory()` does not report the grid identity.** It reports
   labels and timestamps, bucketed by `validated`. From this commit a datastore can legitimately
   hold several background rows for the same cosmology and tolerances differing **only** in their
   grid digest or construction version, and `tools/inventory_report.py` will render them as
   indistinguishable duplicates. This is the same defect as
   `[03-qcd-inventory-does-not-report-the-representation]` one level out, and that entry has been
   widened to say so rather than a near-duplicate being opened. Not done here because prompt §2
   item 3 is about the key and `inventory()` is not part of it.

2. **`docs/source-remediation-verification/run_quadsource_integrals.py:203-207` builds its tag
   labels as literals** — `f"SourceRedshiftGrid_{len(z_source_sample)}"` and
   `f"SourceSamplesPerLog10Z_{args.samples_per_log10z}"`. The first is already stale (prompt 11
   added the digest) and the second is now incomplete (a run written by `main.py` also carries
   `Run_*` and `SourceGridConstruction_*`). `docs/` is not on this prompt's file list and that
   script is a scoped verification harness, not production, so it was left alone. Whoever next
   uses it should have it call `build_grid_tag_labels` and the two helpers in `extract_common`
   rather than re-spelling the labels. **No §3 issue was opened**: it has been stale since prompt
   11, it is a scoped harness rather than production, and its failure mode is loud — it finds no
   objects rather than the wrong ones.

3. **`CosmologyConcepts.redshift.check_zsample` still has no callers**, which log 11 recorded.
   This prompt closes the hazard the other way — by keying the lookup — so the function is now
   redundant rather than merely unused; nothing was deleted, because deleting a public helper is
   not what the prompt asked for.

4. **The run label being a selection criterion invalidates every compute target in an existing
   store**, on top of what prompt 11's digest already invalidated. That is not a new cost in
   practice — log 11 §5 establishes that all eight stored object types were already unfindable —
   but it is worth stating plainly that `HEAD~1` and `HEAD` share no compute-target rows.

5. **The archival guarantee stops at `GkTk-remedial` prompts 03/04, and the only store in the tree
   is older than that.** Measured (§6): the prompt-14 fallback fires correctly on
   `physics-test-n20-lambdacdm-zend0p1-shard0000.sqlite` and finds its background row, and then
   `build()` raises the *pre-existing* refusal on `BackgroundModelValue.tau_lo_Mpc`. The two are
   not the same kind of thing — a grid identity is metadata the old store never *recorded*, while
   `tau_lo_Mpc` / `cs_tau_Mpc` / `cs_tau_lo_Mpc` / `friction_F` are values it never *computed* —
   so softening the second guard would mean fabricating background values and was not considered.
   Nothing was changed. But it means the user's archival requirement is met for stores written
   after `GkTk-remedial` prompt 04 and not for anything older, and no such store exists here to
   demonstrate the happy path on. If that matters, the lever is a reader that stops at
   `BackgroundModel` and never asks for the value rows — which is a design, not a patch.

## State handed to the next prompt

**Prompt 15 owns the grid's density. Everything below is what it needs and did not ask for.**

- **`T_Z_REPRESENTATION_VERSION` is 6**, unchanged by this commit.
- **`SOURCE_GRID_CONSTRUCTION_VERSION` is 1**, declared at
  `CosmologyConcepts/wavenumber.py:51`. **Prompt 15 must bump it to 2 in the same commit that
  changes how the grid is built**, and add its line to the version table in that comment block.
  `ComputeTargets/tests/test_source_grid.py::TestTheConstructionVersionNamesThisAlgorithm::test_version_1_is_prompt_11s_construction`
  pins version 1 to `SOURCE_GRID_BREAK_STANDOFF = 0.25`, `SOURCE_GRID_BREAK_HALF_WIDTH = 5` and
  `SOURCE_GRID_BREAK_REFINEMENT = 2` and **will fail** if any of them moves without the bump —
  that is deliberate, and updating it is part of the bump.
  `test_the_production_grids_are_the_ones_the_campaign_recorded` pins both production grids by
  length and digest and will fail for the same reason; prompt 15 replaces both numbers with its
  own and records the old ones beside them.
- **The production grid, for the record:** QCD 1,773 samples, digest `303f9ce7`, response 156 /
  `197b46de`, 8 protected points; LambdaCDM 1,732 / `0960e169`, response 145 / `69050b4c`, 0
  protected. Byte-identical to prompt 11's.
- **`SourceSamplesPerLog10ZTag` is still true and was deliberately left alone** (prompt §2). It is
  prompt 15's to retire, in the commit that makes it false. When it does: `source_samples_per_log10z`
  also feeds `delta_logz` at `main.py:916` and `main.py:1533`, which is **vestigial** (§4 above) —
  nothing computes from it — but those two arguments should go or be re-sourced in the same commit.
- **Tag spellings**, both from `extract_common`: `Run_<label>` and
  `SourceGridConstruction_<version>`. The default run label is `"default"`
  (`main.py:DEFAULT_RUN_LABEL`).
- **`BackgroundModel`'s lookup key** is now `(cosmology_type, cosmology_serial, atol_serial,
  rtol_serial, source_grid_digest, source_grid_construction)` plus the supplied tags, on the
  compute path; on the read path (`z_sample=None`) the two grid columns are selected and checked
  for uniqueness rather than filtered on. `obj._source_grid_identity` is `(construction, digest)`
  or `None`.
- **A datastore written before this commit cannot be computed into and can still be read.** The
  exact errors and messages are §3 above. There is no migration. **The archival read has a floor:**
  a store older than `GkTk-remedial` prompts 03/04 stops on `BackgroundModelValue.tau_lo_Mpc`
  instead — §6, and `[14-archival-read-stops-at-the-pre-gktk-value-columns]`.
- **`HEAD~1` and `HEAD` share no compute-target rows**, because the run label and the construction
  tag are in every lookup. That is not a new cost: log 11 §5 established that the grid-tag change
  had already made all eight stored object types unfindable.
- **The read path's entry point** for any new script is
  `extract_common.resolve_run_selection(pool, args.run_label)` after
  `extract_common.add_run_selection_argument(parser)`; pass `selection.tags` to every
  `pool.object_get` for a stored compute target, and *not* to `GkSourcePolicyData`, which carries
  no tags on the write side either.
- **The one command that regenerates the QCD reference fixture** is unchanged and was not run by
  this prompt (no number moved):
  `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py`.
