# Prompt 14 — Name a run, verify what it names, and key the background on the grid that built it

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[11-background-model-not-keyed-on-the-source-grid]` on this campaign's board §3
**Implements:** the campaign's own house style — **a human-readable handle plus a derived check** —
applied to the source grid and to the scripts that read it. Prompt 03 declared a readable constant
and required the filter to *read* it rather than inline a literal; prompt 11 kept the readable
`SourceRedshiftGrid_1773` label and added a content digest beside it. This prompt does the same one
level out, and makes the read path use it.
**Depends on:** 13. **Blocks 15**, for the reason prompt 03 blocked prompt 04: the key must exist
before the thing it keys moves.
**Recommended model:** **Opus** — a schema, tagging and CLI decision whose failure mode is silent.

**Files you may touch:** `main.py` (**the argument parsing, grid-construction and tagging hunks
only**), `CosmologyConcepts/wavenumber.py` and `redshift.py` (**only to expose the construction
version**), `Datastore/SQL/ObjectFactories/BackgroundModel.py`, **`extract_common.py` and every
`extract_*.py`**, `ComputeTargets/tests/test_main_plumbing.py`,
`ComputeTargets/tests/test_source_grid.py`, a new test module if you want one, plus this campaign's
log and board and `docs/OPEN_ISSUES.md`.

> **Two scope notes, both authorised by the user, both to be recorded in your log.**
>
> 1. **`extract_*.py` has been a stop condition for every prompt in this campaign** (README §0.5 and
>    §4 name it). **That exclusion is lifted for this prompt only.** It remains in force elsewhere —
>    prompt 15 must not touch them.
> 2. **`main.py` gains a new command-line argument.** Every prompt so far has treated `main.py` as
>    nearly untouchable. The user has explicitly blessed this change, because a run cannot be named
>    at read time if it was never named at write time.

**Do not touch:** `CosmologyModels/`; `ComputeTargets/BackgroundModel.py`'s *numerics* — prompt 13
owns its splines, you touch only its **factory**; the grid's **density**, its construction, or the
number of samples per decade — that is prompt 15; `QCD_EOS.py`; `Quadrature/`; any compute target;
the Bessel, WKB or numeric stages of `main.py`.

**Read first:** `prompts/qcd-background-audit/logs/11-cosmology-aware-source-grid.md` §5 — the
blast-radius table and the digest it introduced; `main.py` `build_grid_tag_labels` (`:510-530`), the
tag block at `:650-700`, and the eight `tags=` call sites; `Datastore/SQL/ObjectFactories/BackgroundModel.py`'s
`build()` and module docstring; `Datastore/SQL/ObjectFactories/QCD_Cosmology.py` **as prompt 03 left
it — the shape to follow**; `extract_common.py` and `extract_Gk_data.py:389`, whose `query_payload`
passes `"z_sample": None` and no tags at all; board entry
`[11-background-model-not-keyed-on-the-source-grid]`.

---

## 1. What is wrong — three things, one root

**(a) `BackgroundModel` is not keyed on the grid it was computed on.**
`sqla_BackgroundModel_factory.build()` filters on `(cosmology_type, cosmology_serial, atol_serial,
rtol_serial)` plus `LargestSourceZTag`, `SmallestSourceZTag` and `SourceSamplesPerLog10ZTag`. When
prompt 11 changed the grid's *shape*, **all three of those stayed the same** — the endpoints and the
samples-per-decade did not change — and the factory never filters on `z_sample` at all. So a
pre-prompt-11 datastore returns its **1,732-node** background for the new **1,773-node** grid. It is
the *one surviving row* in a store whose every compute target the grid-tag change invalidated, which
makes it worse than a miss: the next run finds it, uses it, and is wrong.

**(b) The read path has no notion of which run it is reading.** All seven of `extract_common.py`,
`extract_Gk_data.py`, `extract_GkSource_data.py`, `extract_GkWKB_data.py`,
`extract_QuadSourceIntegral_data.py`, `extract_TkWKB_data.py` and `extract_tensor_source_data.py`
pass **zero** tags, against `main.py`'s eight-plus tagged call sites. This was a reasonable
development choice — while a store held exactly one complete run, a tagless query needed no
adjustment whenever the compute configuration changed, and that ergonomic property is worth
preserving. It stops being safe the moment a store holds two generations, which is what a partial
regeneration or an archived store read beside a current one produces: the scripts cannot tell them
apart and will mix silently or pick arbitrarily.

**(c) Nothing names a run.** The store records what a run *was* only as a scatter of descriptive
tags. There is no handle to ask for, which is why the archival case — *"read me the run I did in
March, which I can no longer reproduce"* — cannot be expressed at all.

## 2. The model to build

**A human-readable handle, plus a derived check.** They answer different questions and neither
substitutes for the other.

| | what it is | answers | fails how |
|---|---|---|---|
| **run label** | a string the user chooses at write time | *which run?* | silently, if reused across configurations — hence the check |
| **construction version** | an integer naming the grid *algorithm* | *which generation?* | only if someone forgets to bump it |
| **grid digest** | content hash of the grid's values (prompt 11) | *which exact grid?* | it cannot — it is derived |
| **largest / smallest z** | the grid's extent | *which runs reached $z\ge X$?* | — |

**On the descriptors, because it is easy to get backwards.** `LargestSourceZTag` and
`SmallestSourceZTag` are **not** redundant with the digest. A digest is one-way: you cannot recover
an extent from a hash, and you cannot range-query one. They are **the only queryable projection of
the grid's extent**, and cross-generation questions depend on them. Keep them, keep them recorded,
and keep them queryable — just never load-bearing for correctness.
`SourceSamplesPerLog10ZTag` is the different case: it is still true today, so **leave it alone
here**; prompt 15 retires it, in the same commit that makes it false.

1. **A run label, applied at write time.** `main.py` takes a new argument naming the run; the label
   becomes a `store_tag` carried by everything the run writes, alongside the tags it already
   carries. Decide and state: what happens when the user supplies none (a generated default that is
   still unique and still human-legible is the obvious answer, but argue it), and whether a label
   may be reused deliberately to extend an existing run — which is a real workflow and should not be
   forbidden by accident. `TkProductionTag` and `GkProductionTag` (`"TkOneLoopDensity"`,
   `"GkOneLoopDensity"`) are the existing precedent for a hand-chosen label string: follow their
   mechanism rather than inventing a second one.

2. **A source-grid construction version**, in the style of prompt 03's
   `T_Z_REPRESENTATION_VERSION`: an integer naming the **algorithm** that built the grid, declared
   in one place, carried as a `store_tag`, and **bumped by any prompt that changes how the grid is
   constructed**. Its comment block must say what it identifies, that a bump is the only signal a
   datastore gets, and which change landed at which version. **Version 1** is what ships today:
   prompt 11's protected set, `SOURCE_GRID_BREAK_HALF_WIDTH = 5`, `SOURCE_GRID_BREAK_REFINEMENT = 2`,
   `SOURCE_GRID_BREAK_STANDOFF = 0.25`, uniform base density.

3. **`BackgroundModel`'s factory filters on the digest and the construction version.** Follow prompt
   03's shape exactly: read the constants into locals, filter on them, insert them, and **never
   write a literal** — prompt 03's `test_the_filter_is_not_a_literal` exists because a literal stops
   tracking the constant the moment someone bumps it. A store predating the column must fail with a
   message naming this campaign and saying there is no migration, as `sqla_QCDCosmology_factory` now
   does.

4. **The extract scripts select by label and verify by digest.** This is the substance.
   - **Select** on the run label — the primary query surface, and the only thing a user should have
     to type.
   - **Verify** that everything pulled under that label is internally consistent: one construction
     version, one digest. **If it is not, refuse and name what was found** — do not return a
     mixture, and do not pick.
   - **Preserve the single-run ergonomics.** If no label is given and the store holds exactly one
     run, use it and *say which*. Only an ambiguous store should require the user to choose. That
     is the property that made the tagless design pleasant and it should survive.
   - **A store written before this prompt still reads.** Rows carrying no label and no version are
     not an error; they are *unknown*, and must remain queryable and plottable, labelled as such.
     **A design that makes old stores unreadable fails this prompt**, however clean — the user's
     requirement is that a superseded datastore keeps archival value even once it can no longer
     serve as a numerical base, and `extract_*.py` is how that value is realised.
   - Put the mechanism in `extract_common.py` once. **Do not copy `main.py`'s `tags=` lists into
     seven scripts**: the scripts are *readers*, and their requirement is selection and
     disambiguation, not reproduction of the writer's configuration.

5. **Check, do not assume, what `source_samples_per_log10z` feeds.** It is read here as a CLI
   parameter that labels the grid, but that has not been traced through every consumer. If anything
   downstream *computes* from it rather than merely labelling with it, say so in the log — prompt 15
   is about to retire the tag and must not be surprised.

6. **Do not build the archival library.** The user has noted that an archive of grid-construction
   algorithms may be wanted later. **Record it as a §3 issue** in the user's framing so it survives,
   and build nothing towards it beyond the version integer that would be its key.

7. **Do not change the grid.** No density change, no construction change, no new protected points.
   The grid this commit produces must be **element-for-element identical** to prompt 11's, on both
   models, and that is an acceptance test.

## 3. Tests

1. **The grid is unchanged** — QCD 1,773 and LambdaCDM 1,732, element for element against `HEAD~1`,
   same digests. Quote the comparison.
2. **Two generations do not collide.** The same parameters under construction version 1 and 2 give
   two distinct `BackgroundModel` rows; the same version twice reuses one. Prompt 03's
   `test_the_same_representation_reuses_a_row_and_a_different_one_does_not` is the precedent.
3. **The filter reads the constants, not literals** — by `ast`, following prompt 03's test of the
   same name.
4. **A run round-trips by name**: write under a label, read it back through the extract path by that
   label, and get the same objects.
5. **A label spanning two configurations is refused, naming both.** This is the check that makes
   naming safe and it is the single most important test in the prompt.
6. **A store with no label reads, labelled unknown** — the archival requirement, and it needs a real
   test rather than a docstring.
7. **A store with exactly one run needs no label** and says which it used.
8. **`main.py` tags what it builds, and parses the new argument** — by `ast` through
   `test_main_plumbing.load_main_py_functions`, since `main.py` cannot be imported (`CLAUDE.md`).
9. **Every extract script still runs.** They need a datastore; if a full run is not possible in the
   test tree, assert their *query construction* by `ast` and **say in the log what you could not
   execute**. Do not claim a script works if you did not run it.

## 4. Acceptance

| Quantity | Requirement |
|---|---|
| Source grid, both models | **element-for-element identical** to `HEAD~1` |
| `BackgroundModel` across two generations | **misses**; same generation **hits** |
| Run round-trip by label | **same objects back** |
| Label spanning two configurations | **refused, both named** |
| Store with no label | **reads, labelled unknown** |
| Store with one run, no label given | **reads, says which** |
| Tag literals in any filter | **none** |
| `T_Z_REPRESENTATION_VERSION` | **unchanged** — this prompt moves no background number |

All three suites pass; counts quoted and not falling. `black` clean.

## 5. Log and commit

Log to `logs/14-key-the-source-grid-construction.md` per README §5.1, carrying: the label mechanism
and the alternatives rejected; what happens when no label is supplied and whether a label may be
reused deliberately; what a pre-prompt-14 store does, exactly — which error, from which call, with
which message — **for both the compute path and the read path**, which are different answers; what
`source_samples_per_log10z` turned out to feed; and which extract scripts you executed against a
real store versus asserted by `ast`.

Close `[11-background-model-not-keyed-on-the-source-grid]` on the board §3 → §4, open the
archival-library issue per §2 item 6, and update `docs/OPEN_ISSUES.md` in the same commit with its
count and date corrected.

**Record on the board, do not decide:** prompt 12's density recommendation
(`[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]`) stays open and is
prompt 15's.

**If this will not land coherently in one commit, stop and say so** rather than splitting it
yourself. The write half and the read half are deliberately one prompt: a label with no verification
is the weak design the campaign rejected in README §7 D1, verification with no label is unusable,
and the acceptance test is end-to-end.

Commit subject, or something equally specific:
`Name a run at write time and verify it at read time`
