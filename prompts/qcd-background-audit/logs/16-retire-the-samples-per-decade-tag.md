# Log 16 — Retire the samples-per-decade tag and the arguments it fed

**Prompt:** prompts/qcd-background-audit/16-retire-the-samples-per-decade-tag.md
**Commit:** *(this commit)* — "Retire the samples-per-decade tag the grid no longer obeys"
**Model:** Claude Sonnet 5
**Date:** 2026-09-15
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

`T_Z_REPRESENTATION_VERSION` is **6** before and after (unchanged: no `CosmologyModels/` file is in
the diff). `SOURCE_GRID_CONSTRUCTION_VERSION` is **2** before and after (unchanged: no
`CosmologyConcepts/` file is in the diff). Three files touched, plus one new test module:

1. **`main.py`** (the argument-parsing and tagging hunks only):
   - `SourceSamplesPerLog10ZTag`'s declaration (in the tuple-unpacking assignment) and its
     construction (`pool.object_get("store_tag", label=f"SourceSamplesPerLog10Z_{...}")`) are both
     removed; the `ray.get([...])` list it was built in now has one fewer element, in the same
     position order as its matching tuple target.
   - All **27** `tags=[...]` uses of `SourceSamplesPerLog10ZTag` across the `BackgroundModel`
     lookup and the six tagged compute-target queues (`TkNumericIntegration`,
     `GkNumericIntegration`, `QuadSource`, `QuadSourceIntegral`, and their two `_do_not_populate`
     probe variants) are removed. Total occurrences of the literal `SourceSamplesPerLog10Z` in
     `main.py` before this commit: 29 (1 declaration + 27 tag-list uses + 1 label f-string); after:
     **0**.
   - The two `delta_logz=1.0 / float(source_samples_per_log10z)` arguments (at the
     `TkNumericIntegration` and `GkNumericIntegration` `object_get` calls) are removed.
   - `--source-samples-log10z`'s help text is rewritten to describe what the number now does (sets
     the base lattice the curvature criterion's cap is measured against) rather than what it used
     to do (set the grid's density directly). The argument itself, its default
     (`DEFAULT_SOURCE_SAMPLES_PER_LOG10_Z`), and the four surviving uses of
     `source_samples_per_log10z` (the base-grid `build_z_sample` call, the production
     `populate_source_grid` call, the status print, and `args.source_samples_log10z` at parse
     time) are untouched.
2. **`Datastore/SQL/ObjectFactories/BackgroundModel.py`** (module docstring only, prose, no logic):
   the SCHEMA NOTE for prompt 14 named `SourceSamplesPerLog10ZTag` as one of the tags the
   pre-prompt-14 factory keyed on via `main.py`; it now says that tag existed only "until
   prompts/qcd-background-audit prompt 16 retired it".
3. **`ComputeTargets/tests/test_run_identity.py`** (docstring only, prose, no logic): the same
   correction to the module docstring's description of what `sqla_BackgroundModel_factory.build()`
   used to filter on.
4. **New module `ComputeTargets/tests/test_retire_samples_per_decade_tag.py`** (3 classes, 8 tests,
   0.05 s): `TestTheTagIsGoneFromEveryWriter` (source-text checks that the tag name and label are
   gone, `delta_logz` is gone, and the switch itself with its corrected help text and exactly its
   four surviving uses of `source_samples_per_log10z` survive), `TestNeitherVersionConstantMoved`
   (direct pins of `SOURCE_GRID_CONSTRUCTION_VERSION == 2` and
   `LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION == 6`), and
   `TestRemovingATagFromTheLookupBroadensIt` (the real-SQL test that answers section 2 item 2, see
   below).

`docs/source-remediation-verification/run_quadsource_integrals.py` is **not** in the diff — see
deviation 2.

## Deviations from the prompt

### 1. IMPLEMENTATION CHOICE — the new tests live in a new module, not in `test_run_identity.py` or `test_source_grid.py`

The prompt names both existing files as touchable and offers a new module as a third option. I put
all four new test classes (tag absence, version pins, and the broadening demonstration) in a new
`ComputeTargets/tests/test_retire_samples_per_decade_tag.py` rather than adding classes to either
existing file. Reasons: (i) `test_run_identity.py`'s own docstring and tests are prompt 14's
record of *that* prompt's work, and prompt 16 section 3 item 5 is explicit that prompt 14's tests
"still pass unmodified except for the docstring" — adding new test classes there would blur whose
work a reader is looking at; (ii) `test_source_grid.py`'s existing
`TestTheConstructionVersionNamesThisAlgorithm` already carries the grid-identity pin this prompt's
item 2 asks for, so nothing needed to be added there, and the module's docstring already frames it
as being about the grid's construction and density, not its datastore tags; (iii) the broadening
question (section 2 item 2) is a property of the tag-join idiom shared by every tagged factory, not
a property of `BackgroundModel` or of the source grid specifically, so a dedicated module keeps
the claim's scope visible in one place. The alternative (folding these into `test_run_identity.py`,
since it already has the SQL harness in scope) was considered and rejected on (i).

### 2. IMPLEMENTATION CHOICE — `run_quadsource_integrals.py` is left alone

Section 2 item 5 asks for a decision, stated and reasoned, on whether
`docs/source-remediation-verification/run_quadsource_integrals.py:207` (which builds
`f"SourceSamplesPerLog10Z_{args.samples_per_log10z}"` into its own `tag_labels` list) should follow
`main.py`. I left it untouched. The script's own comment names its source as "the six production
tags main.py builds from the grid (main.py:430-465)" and its list is exactly those six: it has
never been updated for `main.py`'s tagging since prompts 11 and 14 of this campaign landed — it
still builds `SourceRedshiftGrid_{len}` (the pre-prompt-11 size-only tag, superseded by the
content-digest tag `build_grid_tag_labels` now constructs) and it tags neither `RunLabelTag` nor
`SourceGridConstructionTag`. `git log` confirms the file's last two commits are both from the
`source-remediation` campaign; no `qcd-background-audit` prompt has touched it. Removing only the
one tag this prompt happens to retire, while leaving the script five tagging-generations behind
`main.py` in every other respect, would misrepresent it as "kept in step" when it is not, and a
full re-sync (adding `RunLabelTag`/`SourceGridConstructionTag`, switching to the digest-based grid
tags) is well outside this prompt's file list and its "do not touch: the grid, its construction, or
`SOURCE_GRID_CONSTRUCTION_VERSION`" boundary. `CLAUDE.md`'s "verification documents are additive"
rule was considered and found not to bear directly here (it governs the `.md` documents, not this
script), which is why the reason above is about the script's own already-stale state and not about
that rule.

### 3. IMPLEMENTATION CHOICE — the help text's exact wording

Section 2 item 4 asks only that the help text say what the number now does. I wrote: "specify the
base number of z-sample points per log10(z) for the source term; the curvature criterion of
docs/qcd-background-verification.md §10 may refine individual intervals above this density, capped
so that no interval is ever coarser than this uniform lattice would have made it
(SOURCE_GRID_MAX_SPACING_FACTOR)". The alternative considered was a one-line version without the
document pointer or the constant name; the longer form was chosen because `--help` output is the
one place a future reader unfamiliar with this campaign is likely to meet this switch cold, and the
pointer costs one line.

None of the three deviations touches a README section 2 design fact, and none was kept against the
prompt's instructions — all three are choices the prompt left open.

## Verification performed

**Section 2 item 2 (measured, not assumed): removing a tag from a lookup broadens it.** Read
`Datastore/SQL/ObjectFactories/TkNumericIntegration.py:262-274` and
`Datastore/SQL/ObjectFactories/BackgroundModel.py:283-293` (the two idioms; every other tagged
factory in the tree — `GkNumericIntegration`, `QuadSource`, `QuadSourceIntegral`, `GkSource`,
`TkWKBIntegration`, `GkWKBIntegration` — follows the same `for tag in tags: q = q.join(tab, ...)`
shape): for each tag in the caller's list, the query is joined once against the tag-association
table, requiring that the candidate row have *an* association with that tag's serial. There is no
companion `NOT EXISTS` or count check that the row's *total* tag set equals the caller's list, so a
row with additional tags the caller did not ask for still matches. Consequence: dropping a tag from
the list can only **admit** rows that used to be excluded by it (a superset match), never exclude
rows that used to pass. Demonstrated against a real SQLite database
(`ComputeTargets/tests/test_retire_samples_per_decade_tag.py::TestRemovingATagFromTheLookupBroadensIt`,
reusing the harness `test_run_identity.py`'s `_Schema` built for prompt 14): a `BackgroundModel` row
written with `tags=[run_tag, retired_tag]` (simulating a pre-prompt-16 write, which still attached
`SourceSamplesPerLog10ZTag`) resolves under `read_model(grid, tags=[run_tag])` (the shorter,
post-prompt-16 query) — **`read.available` is `True`** and `read.store_id == written.store_id`. The
converse is checked in the same class: a row missing a tag the caller *does* still request
continues to miss (`read.available` is `False`). **Consequence for this commit: no datastore
regeneration.** Every row any pre-prompt-16 run wrote keeps `SourceSamplesPerLog10ZTag` as harmless
archival history and continues to satisfy the shorter post-prompt-16 tag list built from the same
run label and construction version.

**Section 2 item 3 (delta_logz): confirmed, not assumed, that nothing downstream reads the
value.** Read `Quadrature/integrators/numeric_with_phase_cut.py:640-646` (the module's own comment:
"`delta_logz` is still accepted and still handed to the supervisor ... but the oscillation-
resolution diagnostic no longer uses it") and `Quadrature/supervisors/numeric.py` in full:
`NumericIntegrationSupervisor.report_wavelength` is the only method that reads `self._delta_logz`
for anything beyond the `is None` gate on its three properties (`has_unresolved_osc`,
`unresolved_z`, `unresolved_efolds_subh`), and `report_wavelength`'s own docstring states plainly
that neither `GkNumericIntegration.RHS` nor `TkNumericIntegration.RHS` calls it any more (confirmed
by `grep -rn report_wavelength` returning only its definition and that docstring's mention of it —
no call site anywhere in the tree). The flag those three properties actually carry on a stored
object comes from `scan_sample_grid_for_unresolved_osc`, which takes `omega_sq` and the returned
sample grid, never `delta_logz`. `TkNumericIntegration.__init__` and `GkNumericIntegration.__init__`
both declare `delta_logz: Optional[float] = None`, so the two call sites this prompt edits pass no
argument and get the harmless default, identical in effect to what `report_wavelength` already did
with it (nothing, since it is never called).

**Section 3 item 1 — the tag is gone from every writer.**
`TestTheTagIsGoneFromEveryWriter` (4 tests): `SourceSamplesPerLog10Z` (covering the tag name, its
declaration comment and its label f-string) appears 0 times in `main.py`; `delta_logz` appears 0
times; `"--source-samples-log10z"` is still declared and its old help sentence
("specify number of z-sample points per log10(z) for the source term") no longer appears verbatim;
`source_samples_per_log10z` appears exactly 4 times (the base-grid call, the production-grid call,
the status print, and the argparse-to-local assignment) — matching `grep -c` on the file directly.

**Section 3 item 2 — the grid did not move.** This prompt's diff touches no file under
`CosmologyConcepts/` or `CosmologyModels/`, and no construction logic in `main.py`. The existing,
unmodified `ComputeTargets/tests/test_source_grid.py::TestTheConstructionVersionNamesThisAlgorithm`
still asserts QCD 1,996 samples / digest `a2c32f67` and LambdaCDM 1,778 / digest `60a3205a`, element
for element, plus `SOURCE_GRID_CONSTRUCTION_VERSION == 2` — and it passed, unedited, in the full
`ComputeTargets` run below.

**Section 3 item 3 — see the item-2 verification above** (the same test class carries both the
broadening measurement and the "still resolves" assertion).

**Section 3 item 4.** `SOURCE_GRID_CONSTRUCTION_VERSION == 2` and
`LambdaCDM_GenericEOS.T_Z_REPRESENTATION_VERSION == 6`, pinned directly in
`TestNeitherVersionConstantMoved` and confirmed by reading
`CosmologyConcepts/wavenumber.py:61` and `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:414`
directly (neither file is in this commit's diff).

**Section 3 item 5.** `ComputeTargets/tests/test_run_identity.py` was run unmodified except for the
docstring edit; every one of its existing test classes passed (see suite counts below), including
`TestMainPyNamesItsRun::test_every_tag_list_carries_the_run_and_the_construction`, which does not
name `SourceSamplesPerLog10ZTag` and is unaffected by its removal, and
`TestEveryExtractScriptSelectsARun::test_no_script_reproduces_the_writers_tag_configuration`, which
asserts the *absence* of `SourceSamplesPerLog10ZTag` (among others) from the six `extract_*.py`
scripts — unaffected, since those scripts never carried it.

**`black --check`** on every touched and new file: clean (the new module was formatted with
`black` before the final run).

**Suite counts**, from the repository root:

```
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_compatibility \
  LiouvilleGreen.tests.test_bessel_near_region LiouvilleGreen.tests.test_bessel_phase \
  LiouvilleGreen.tests.test_bessel_reference LiouvilleGreen.tests.test_bessel_tail \
  LiouvilleGreen.tests.test_bessel_two_region LiouvilleGreen.tests.test_phase_spline \
  LiouvilleGreen.tests.test_range_reduce LiouvilleGreen.tests.test_scipy_bessel_domain \
  LiouvilleGreen.tests.test_three_bessel LiouvilleGreen.tests.test_wkbtools
```

| Suite | Before | After |
|---|---|---|
| `CosmologyModels` | 30 | 30 |
| `ComputeTargets` | 439 | **447** (+8, all new, `test_retire_samples_per_decade_tag.py`) |
| `LiouvilleGreen` (fast set, `test_3bessel_analytic` excluded per prompts 02–15's convention) | 143 | 143 |

No count falls. `ComputeTargets` was run as a full `discover` (167.4 s); `CosmologyModels` and the
`LiouvilleGreen` fast set as shown above (0.60 s and 14.8 s).

## Observations not acted on

None beyond the two already recorded in `docs/OPEN_ISSUES.md` and referenced above
(`run_quadsource_integrals.py`'s own tagging staleness, which predates this prompt and is out of
its scope to fix wholesale; and the wavenumber-set half of
`[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]`, which is the user's
recorded no-action decision, not an open question this prompt could act on).

## State handed to the next prompt

- `T_Z_REPRESENTATION_VERSION` is **6**. `SOURCE_GRID_CONSTRUCTION_VERSION` is **2**. Neither moved
  and neither is expected to move on account of this prompt.
- `SourceSamplesPerLog10ZTag` no longer exists anywhere in `main.py`; a store written before this
  commit still carries it on its rows as harmless history, and every lookup that used to require it
  now succeeds without it (measured, not assumed — see above). No regeneration is needed or was
  performed.
- `--source-samples-log10z` (and its parsed local `source_samples_per_log10z`) is unchanged in
  behaviour: it still sets `populate_source_grid`'s and `build_z_sample`'s base lattice, now
  correctly described in `--help` as the base the curvature criterion's cap is measured against
  rather than as the density itself.
- The two `delta_logz=` arguments are gone from `main.py`; `numeric_with_phase_cut` and
  `NumericIntegrationSupervisor` still accept the keyword (untouched, since neither file is in this
  campaign's file list), so no other caller is affected.
- `docs/source-remediation-verification/run_quadsource_integrals.py` still builds
  `SourceSamplesPerLog10Z_{args.samples_per_log10z}` into its own tag list, deliberately left alone
  (deviation 2) — a later prompt that wants to bring it into step with `main.py`'s current tagging
  should expect to fix all six tags at once, not one.
- `[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]` stays open on the board,
  narrowed: its tag half is closed, its wavenumber-set half is the user's recorded no-action
  decision and needs no further work.
- The campaign is **16 / 16 complete**; workstream E now runs 13–16.
