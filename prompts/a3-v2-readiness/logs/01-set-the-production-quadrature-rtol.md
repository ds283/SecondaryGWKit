# Log — prompt 01: set the production quadrature `rtol` to `1e-7`

**Subject:** *Set the production quadrature rtol to 1e-7* · **Model:** Claude Sonnet 5.5 ·
**Date:** 2026-10-02 · **Base:** `67e7e50`, clean · **Result:** landed. `DEFAULT_QUADRATURE_RTOL` is
`1e-7`, with its comment above it; two stale comments in `ComputeTargets/QuadSourceIntegral.py` are
corrected; the provenance, the `handover` and `levin-refactor` boards and the index are updated;
`[00-quadsource-extraction-reads-at-the-default-rtol]` is closed. Neither stop condition of the
prompt's §5 fired. The commit is the one that adds this file ("this commit" below, since its SHA
does not exist yet).

## What shipped

- **`config/defaults.py`:** `DEFAULT_QUADRATURE_RTOL = 1e-08` → `1e-07`, with a 14-line comment
  **above** it. The `atol` comment below the constant and the `atol` line are byte-identical.
- **`ComputeTargets/QuadSourceIntegral.py`:** two comments, no code (hit table below).
- **`docs/TOLERANCE-PROVENANCE.md`:** a new subsection, *Update 2026-10-02*, beneath the
  `DEFAULT_QUADRATURE_RTOL` entry (46 lines added, 0 deleted). The entry's heading and text are not
  rewritten.
- **`prompts/handover/IMPLEMENTATION_STATE.md`** (6 added, 0 deleted) and
  **`prompts/levin-refactor/IMPLEMENTATION_STATE.md`** (10 added, 0 deleted): one dated paragraph
  each, on `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` and
  `[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]`. **Both issues stay open.**
- **`ComputeTargets/tests/test_quadrature_tolerance_pin.py`:** four tests, in two classes (below).
- **`docs/OPEN_ISSUES.md`:** the closed row deleted, `[06-quadrature-…]`'s hook extended with a dated
  sentence, the header at **106 open** (107 − 1, by `grep -c '^| \`\['`). The "Last updated" date
  was already 2026-10-02.
- **This campaign's board:** §1's row for 01, items T1–T2, §3.2 (the issue's entry removed), §4 (it,
  closed), §5.2.

### The new comment's numbers, and where each is in the sweep document

Every figure was read from `docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md` (the working tree equals
`HEAD` for that file) at the line quoted:

| In the comment | Quoted from the sweep |
|---|---|
| mid band **1.9e-06**, "sweep section 2" | §2 table, line 106: `\| **1e-32** \| **1e-7** \| **1.9e-06** \| **1060.5 s** \| ×6.8 \|` |
| low band **2.84e-06**, "section 5" | §5, line 180: `Worst relative difference against \`1e-8\`: **1e-5 → 4.86e-06, 1e-6 → 1.08e-05, 1e-7 → 2.84e-06**.` |
| seam band **8.26e-07**, "section 4" | §4, line 156: `Worst over the nine: **1e-6 vs 1e-7 → 9.97e-07**, **1e-7 vs 1e-8 → 8.26e-07**.` (8.26e-07 is the second figure; the first is `1e-6` against `1e-7`) |
| **1.1e-02** for `1e-6` in the mid band, "sections 0 and 2.1" | §0 answer 3, line 39: `` `rtol = 1e-6` is ruled out by the mid band alone, where it is **1.1e-02** wrong.``; §2 table line 105: `\| 1e-32 \| 1e-6 \| 1.1e-02 \| 31.1 s \| ×233 \|` |
| low $z$ is representation-limited, "section 0, answer 4" | §0, line 41: `**4. Low $z_{\rm response}$ is representation-limited, not quadrature-limited.**` |
| the loosest rung converged in all three bands | §0 answer 3, line 39: `` `rtol = 1e-7` is the loosest rung converged in all three response-redshift bands`` |
| the check is one rung tighter, `1e-8` | not a sweep figure: README §6.1, D1 |

The provenance subsection repeats the same figures at the same places, and adds 1060.5 s against
7237.2 s (§2 table, lines 106 and the zero point at line 95, ×6.8), and "stored unconverged on 5 of
the 9" (the `handover` board's qualification (i), and the sweep's line 95 "5 unconverged").

### The T1 search, every hit

Command, as the prompt gives it: `grep -rnE '1e-0?8\b' --include='*.py' main.py extract_*.py config
ComputeTargets AdaptiveLevin LiouvilleGreen CosmologyModels Datastore RayTools RunRegistry tools |
grep -v /tests/ | grep -iE 'quad|rtol|QuadSource|levin|tol'`. It returned 12 lines. Without the final
filter it returns 24; the 12 extra are in the same table. Line numbers are `HEAD`'s (`git show HEAD:<file>`).

| Where | What it says | Action | Why |
|---|---|---|---|
| `config/defaults.py:24` | `DEFAULT_REL_TOLERANCE = 1e-8` | left | not the quadrature `rtol`: the shared default |
| `config/defaults.py:40`, `:44`, `:51` | the `DEFAULT_HEXIT_REL_TOLERANCE` comment: a displacement of 3.81e-8, "previous shared 1e-8", `(1e-10, 1e-8)` | left | not the quadrature `rtol`; the exit-time target's history |
| `config/defaults.py:78` | the GK-numeric comment: "at 1e-8 that error is 2.6e-07" | left | not the quadrature `rtol` |
| `config/defaults.py:86` | `DEFAULT_GK_NUMERIC_REL_TOLERANCE = 1e-8` | left | not the quadrature `rtol`; a different constant that happens to hold the same value |
| `config/defaults.py:100`, `:111`, `:116` | the TK-numeric comment: "from the shared 1e-8", "the previous 1e-8", "the old 1e-8" | left | not the quadrature `rtol`; history of `DEFAULT_TK_NUMERIC_REL_TOLERANCE` |
| `config/defaults.py:138`, `:153` | the TK-numeric comments: "rtol = 1e-8" | left | not the quadrature `rtol`; the TK sector's own measurements |
| `config/defaults.py:159` | `DEFAULT_QUADRATURE_RTOL = 1e-8` | **changed** to `1e-07`, new comment above | T1 |
| `config/defaults.py:163` | the `atol` comment: "`rtol` was confirmed not to bind (1e-8 -> 1e-11 …)" | left, byte-identical | the prompt requires it; standing note 26 and `[06-quadrature-atol-is-inert-…]` record that it is stale. A sentence in the new comment points at it |
| `ComputeTargets/QuadSourceIntegral.py:73` | "The module constants LEVIN_RELERR = 1e-8 / LEVIN_ABSERR = 1e-23 were deleted by …" | left | history of two deleted constants, not a statement of the current value |
| `ComputeTargets/QuadSourceIntegral.py:124` | `MIN_SUBINTERVAL_LOG_WIDTH`: "1e-7/11.4 ~ 1e-8 of the integral -- at the quadrature rtol" | **corrected** | no longer true: 1e-8 is a tenth of `1e-7`. Now names `DEFAULT_QUADRATURE_RTOL` and says "a tenth of". See below |
| `ComputeTargets/QuadSourceIntegral.py:1548` | `analytic_integral`: "with 1e-21 / 1e-8 hardwired on both calls" | left | history of the literals the call replaced |
| `ComputeTargets/QuadSourceIntegral.py:1550` | "the pipeline supplies DEFAULT_QUADRATURE_ATOL = 1e-25 and DEFAULT_QUADRATURE_RTOL = 1e-8, so the absolute tolerance is now 1e4 tighter than the retired literal" | **corrected** | wrong in both halves of the pair today, and the `1e4` is true only at `atol = 1e-25`. Now names both constants, says that when the audit-09 log was written they were 1e-25 and 1e-8, so the absolute tolerance was then 1e4 tighter than the retired 1e-21, and that neither constant has that value now. No number in it is stated as current |
| `ComputeTargets/TkNumericIntegration.py:447` | "6.1e-08, 2.0e-07 and 3.5e-08 then" | left | a substring match on other numbers (`1e-08` inside `6.1e-08`); not a tolerance |
| `LiouvilleGreen/bessel_near_region.py:478` | "a step of 1e-11 contributes 1e-8" | left | not a tolerance setting; Bessel near region |
| `LiouvilleGreen/bessel_tail.py:136` | "exceeds 1e-8 (DRAFT-PLAN.md §5.2, §5.3)" | left | the Bessel tail's own threshold |
| `LiouvilleGreen/bessel_phase.py:57` | "3.51e-8 relative derivative jump" | left | a measurement, substring match |
| `CosmologyModels/GenericEOS/QCD_EOS.py:59` | `11: -9.33311e-8,` | left | a fit coefficient, substring match |
| `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:423`, `:853` | "3.461e-08 relative in conformal time", "geometries differ by 1e-8 of a bracket width" | left | not the quadrature `rtol` |

Counting, for the reviewer: the first group of rows (`defaults.py` 24, 40, 86, 100, 138, 153, 159,
163, and `QuadSourceIntegral.py` 73, 124, 1548, 1550) is the 12 lines the filtered command returns;
the other 12 (`defaults.py` 44, 51, 78, 111, 116; `TkNumericIntegration.py` 447; the three
`LiouvilleGreen` lines; `QCD_EOS.py` 59; `LambdaCDM_GenericEOS.py` 423, 853) come from the same
command without its last `grep -iE`. **Every line either returns is in the table.**

**Other forms and other places.** A second search over `*.py`, `*.json`, `*.toml`, `*.yaml`,
`*.yml`, `*.cfg` for `1.0e-8`, `10**-8`, `1E-8`, `0.00000001` and `1e-8`, outside `venv/`, `var/`,
`docs/`, `prompts/` and `tests/`, found nothing further in the listed packages. It found one
line outside the T1 list of directories: `CosmologyConcepts/wavenumber.py:269` ("perturbation of
`z_init` from 1e-16 to 1e-8"), which is about the source grid, not a tolerance; **left**. Under
`docs/` the same search finds many per-campaign measurement scripts holding their own literal
`1e-8` (`docs/handover/quadsource_atol_sweep.py`'s pair table, `tolerance-convergence`'s sweeps,
`docs/one-loop-planning/fpt_levin_comparison.py:14`'s `ATOL, RTOL, ORDER = 1e-32, 1e-8, 24`, and so
on). None reads the default, none is production, and **none was touched**; they are records.

**The tests were not edited.** `ComputeTargets/tests/test_quadsource_integral.py`'s `run()` passes
`rtol=1e-8` explicitly (so no test runs quadrature at the default, and the prompt's first stop
condition could not fire); `test_main_plumbing.py` pins the constant's *name*.

### Every use of `DEFAULT_QUADRATURE_RTOL` in production (the second stop condition)

`grep -rn DEFAULT_QUADRATURE_RTOL --include='*.py' .` outside `venv/` and `var/`, production only:

| Where | Use | A quadrature tolerance? |
|---|---|---|
| `main.py:71` (import), `:4256` | `pool.object_get("tolerance", tol=DEFAULT_QUADRATURE_RTOL)`, bound to `quad_rtol` | yes: `quad_rtol` is read only at `main.py:3668` (the `QuadSourceIntegral` payload's `"rtol"`) and `:4090` (`rtol=quad_rtol`) |
| `extract_QuadSourceIntegral_data.py:49` (import), `:1056` | the same lookup; `quad_rtol` used at `:1205` as `rtol=quad_rtol` | yes |
| `ComputeTargets/QuadSourceIntegral.py:23` (import), `:750`, `:982` | the default of `rtol` in `evaluate_QuadSource_integral` and `compute_QuadSource_integral` | yes |
| `ComputeTargets/QuadSourceIntegral.py:116` | a comment recording the removed region guard | none: a comment |
| `config/defaults.py` | the definition | n/a |

None is a region guard or any other non-tolerance use. The condition did not fire.

### `MIN_SUBINTERVAL_LOG_WIDTH`: does the argument still hold at `rtol = 1e-7`?

Yes. The comment's argument is that a discarded sub-interval, of width at most `1e-7` in
$\log(1+z')$ out of 11.4, carries at most `1e-7/11.4` of an integral, which is about 1e-8, and that
this is negligible. What changed is only the comparison: 1e-8 was equal to the quadrature `rtol` and
is now a **tenth** of it, so the discarded part is smaller than the tolerance by a decade, not equal
to it. It is also still below every representation floor the comment cites, which the sweep puts at
about 1e-05 to 1e-06 (§0 answer 4; §4's "floor ~1e-06", the `handover` board's table), by 2 to 3
decades. The bound is the comment's own, a worst case for a uniform integrand, and was not
re-measured. It is stated, not verified, in the same way the original was.

## Deviations from the prompt

1. **Deliberate breakage (iii) is not caught by `test_main_plumbing.py`; a test was added to this
   prompt's own module that catches it.** The prompt (§3 (iii)) and the brief say
   `test_main_plumbing` already catches it. It does not. With `extract_QuadSourceIntegral_data.py`'s
   `rtol` lookup moved to `DEFAULT_GK_NUMERIC_REL_TOLERANCE`, `test_main_plumbing` runs 27 tests and
   passes. The reason: `ReaderToleranceAgreementTestCase` compares the constants each reader passes
   to `object_get` and `object_get_vectorized`, and the extractor reads `QuadSourceIntegral` with
   `object_read_batch` (`extract_QuadSourceIntegral_data.py`, the `build_plot_QuadSourceIntegral_work`
   helper). For that file `lookup_constants` returns `['wavenumber_exit_time']` only, measured. The
   new module's second class, `TestExtractorQueriesUnderTheQuadraturePair`, reads the extractor with
   `ast`, reuses `test_main_plumbing._defaults_constant_bindings`, asserts `quad_atol` and `quad_rtol`
   are bound to `DEFAULT_QUADRATURE_ATOL` and `DEFAULT_QUADRATURE_RTOL`, and that the
   `QuadSourceIntegral` `object_read_batch` passes those two names. Mutation (iii) then fails it.
   `test_main_plumbing.py` is unchanged. — **STRUCTURALLY REQUIRED**: the prompt's acceptance (the
   mutation "must make a test fail") cannot pass otherwise, and the new test module is in the files
   this prompt may add. The gap in `test_main_plumbing` itself is under *Observations*.
2. **A sentence in the new `rtol` comment pointing at the `atol` comment below it.** It says the
   comment directly below, which is `DEFAULT_QUADRATURE_ATOL`'s, records log 12's "`rtol` was
   confirmed not to bind" at `atol = 1e-25`, is left as written, and does not transfer. The prompt
   lists what the comment says and this is not on the list. — **IMPLEMENTATION CHOICE**: without it
   the two adjacent comments disagree and nothing says which is current. It adds no number and does
   not edit the `atol` comment, which is byte-identical.
3. **The provenance subsection is a `####` heading**, one level under the entry, titled "Update
   2026-10-02". — **IMPLEMENTATION CHOICE**: the file's `###` headings are one per parameter and the
   prompt asks for "a new subsection beneath" an entry.
4. **The `handover` and `levin-refactor` lines are paragraphs, not single sentences.** The
   `levin-refactor` one has the three points the prompt names, as (i)–(iii). — **IMPLEMENTATION
   CHOICE**.
5. **The `MIN_SUBINTERVAL_LOG_WIDTH` comment adds "it was 1e-8, the same size, when this was
   written".** — **IMPLEMENTATION CHOICE**: the sentence "at the quadrature rtol" was true when it
   was written and the reader should be able to see that.

No `UNINTENDED DRIFT` is known.

## Verification performed

### The AST of every production file touched, against its parent

`ast.dump(ast.parse(...))` of the working-tree file against `git show HEAD:<file>`, by the script
`astcmp.py` in this agent's scratch directory:

| File | Result |
|---|---|
| `ComputeTargets/QuadSourceIntegral.py` | **AST identical** to `HEAD`'s |
| `config/defaults.py` | **AST differs**, and only in one top-level statement: `DEFAULT_QUADRATURE_RTOL = 1e-08` → `DEFAULT_QUADRATURE_RTOL = 1e-07`. Both modules have 17 top-level statements; the one pair that differs is that one |

No docstring changed in either file; the QuadSourceIntegral edits are `#` comments.

### `substitute_main_py(filter_cases=False)`

`main.py` is not edited, and the four substitutions still match once each:

```
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_ATOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_RTOL' in main.py
```

### The six suites

Serially, each with README §7's command, output to a file in this agent's scratch directory,
verdict read from it. The first pass was a background script; its `ComputeTargets` run stopped about
27 s in with output to a progress dot and **no verdict line**, for a reason I did not find (no
failure text, no traceback). It was re-run alone in the foreground, and that is the result below.

| Suite | Result | Wall time | Load (1 / 5 / 15 min), at start |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 0.06 s | 7.4 / 10.6 / 12.6 |
| ComputeTargets | **669 OK** (665 + 4 new) | 145.9 s | ~9 / 10.5 / 11.7 (the re-run, 15:05–15:08) |
| CosmologyModels | 39 OK | 0.7 s | 6.3 / 10.1 / 12.4 |
| Datastore | 551 OK | 141.4 s | 6.3 / 10.1 / 12.4 |
| LiouvilleGreen | 148 OK (skipped=1) | 22.9 s | 7.1 / 9.1 / 11.6 |
| RunRegistry | 199 OK | 69.7 s | 10.5 / 9.8 / 11.8 |

Every count other than `ComputeTargets` is unchanged from README §7; `ComputeTargets` rose by the
four tests this prompt adds. No flake fired. The machine was shared with unrelated work (another
repository's suites ran during this one's).

### Black

`./venv/bin/python -m black config/defaults.py ComputeTargets/QuadSourceIntegral.py
ComputeTargets/tests/test_quadrature_tolerance_pin.py` left the two production files unchanged and
reformatted the new test once (before it was committed); `black --check` on the three is clean.

## The deliberate-breakage record

Each mutation is a real diff, made with `diff -u` between the clean file and the mutated copy,
applied with `git apply`, the tests run, and reverted with `git apply -R`; the working tree was
byte-identical to its pre-mutation copy afterwards (`cmp`). **No mutation was committed.** The
diffs below replay with `git apply` from a clean tree at this commit; `git apply --check` was run
against them at the content the commit holds.

**(i) `DEFAULT_QUADRATURE_RTOL = 1e-8`**

```diff
--- a/config/defaults.py
+++ b/config/defaults.py
@@ -170,7 +170,7 @@
 # which is DEFAULT_QUADRATURE_ATOL's, records log 12's "rtol was confirmed not to bind", measured
 # at atol = 1e-25; it is left as written and does not transfer to atol = 1e-32
 # (docs/TOLERANCE-PROVENANCE.md, standing note 26).
-DEFAULT_QUADRATURE_RTOL = 1e-7
+DEFAULT_QUADRATURE_RTOL = 1e-8
 # was 1e-25; tightened per prompts/source-remediation/IMPLEMENTATION_STATE.md
 # [12-atol-too-loose-for-the-source-integral] -- at 1e-25, ~58% of production QuadSourceIntegral
 # work items had |total|/(1+z_response) below this floor, so their quadrature "converged" before
```

Fails: `ComputeTargets.tests.test_quadrature_tolerance_pin.TestProductionQuadratureTolerance.test_rtol_is_the_a3_v2_rung`
(4 run, 1 failure). `test_main_plumbing`: 27 OK, as expected (it pins names, not values).

**(ii) `DEFAULT_QUADRATURE_ATOL = 1e-25`**

```diff
--- a/config/defaults.py
+++ b/config/defaults.py
@@ -177,7 +177,7 @@
 # doing any work. rtol was confirmed not to bind (1e-8 -> 1e-11 was bit-identical on 159 items);
 # atol does. 1e-32 is the exact value the source-remediation campaign measured (log 12) as
 # halving the residual against the analytic oracle at an unconverged test point.
-DEFAULT_QUADRATURE_ATOL = 1e-32
+DEFAULT_QUADRATURE_ATOL = 1e-25
 
 # value of dtheta/dz where we will switch to a Levin integration strategy
 DEFAULT_LEVIN_THRESHOLD = 1.0
```

Fails: `…test_quadrature_tolerance_pin.TestProductionQuadratureTolerance.test_atol_is_the_a3_v2_floor`
(4 run, 1 failure).

**(iii) the extractor looks its `rtol` up under `DEFAULT_GK_NUMERIC_REL_TOLERANCE`**

```diff
--- a/extract_QuadSourceIntegral_data.py
+++ b/extract_QuadSourceIntegral_data.py
@@ -1053,7 +1053,7 @@
             pool.object_get("tolerance", tol=DEFAULT_HEXIT_ABS_TOLERANCE),
             pool.object_get("tolerance", tol=DEFAULT_HEXIT_REL_TOLERANCE),
             pool.object_get("tolerance", tol=DEFAULT_QUADRATURE_ATOL),
-            pool.object_get("tolerance", tol=DEFAULT_QUADRATURE_RTOL),
+            pool.object_get("tolerance", tol=DEFAULT_GK_NUMERIC_REL_TOLERANCE),
         ]
     )
 
```

Fails: `…test_quadrature_tolerance_pin.TestExtractorQueriesUnderTheQuadraturePair.test_the_tolerance_lookups_are_under_the_quadrature_constants`
(4 run, 1 failure). **`test_main_plumbing`: 27 tests, OK.** It does not catch this, contrary to the
prompt and the brief (deviation 1). `DEFAULT_GK_NUMERIC_REL_TOLERANCE` is not imported by the
extractor, so the mutated file would also raise `NameError` if it were run; the tests read it with
`ast` and never run it.

The `git status` after each revert showed only this prompt's uncommitted changes, with the mutated
file absent from the list. (The tree was not clean in the commit sense during this work, because
the mutations were made before the commit; it was clean of the mutation.)

## Observations not acted on

1. **After this commit a store built at any pair other than `(1e-32, 1e-7)` is unreadable by
   `extract_QuadSourceIntegral_data.py`.** It looks its rows up at the default pair, which is now
   `(1e-32, 1e-7)`. `quadsource_atol_sweep.py --build` takes explicit `--atol`/`--rtol` and
   substitutes them into `main.py`, so a build at, say, `(1e-32, 1e-8)` (the A3 baseline's pair) is
   unreadable by the extractor at the default. The A3 baseline holds no `QuadSourceIntegral` rows
   (audit S4: `QuadSourceIntegral: 0`), so it loses nothing; A3 v2 does not exist yet. Prompt §6:
   not fixed here.
2. **Five records read the default at their own tree, and a re-run of each would now measure at
   `1e-7`.** None was touched and none re-run (prompt §6):
   - `docs/tolerance-convergence/quadsource_readonly.py` and
     `docs/tolerance-convergence/QUADSOURCE-READONLY.md`: the first takes
     `PRODUCTION = TolerancePair(atol=DEFAULT_QUADRATURE_ATOL, rtol=DEFAULT_QUADRATURE_RTOL)` and
     builds its axes around it; the document is its output at `1e-8`;
   - `docs/tolerance-convergence/inventory.py` and `TOLERANCE-INVENTORY.md`: `inventory.py:723-724`
     resolves `@config.defaults.DEFAULT_QUADRATURE_RTOL` and writes the value, and its row names
     `config/defaults.py:159`; the document is its output at `1e-8`;
   - `docs/source-remediation-verification/run_quadsource_integrals.py`: its `--rtol` defaults to
     the constant (`:110`), so a run with no argument is now at `1e-7`.

   And a sixth, listed for completeness: `docs/a3-v2-readiness/reader_writes_probe.py` imports the
   constant, builds its stand-in store from it and deletes by it, so it is self-consistent at either
   value and needs no change.
3. **`test_main_plumbing.ReaderToleranceAgreementTestCase` does not check any reader's
   `QuadSourceIntegral` lookup.** `lookup_constants` scans `object_get` and
   `object_get_vectorized` only. Measured on the tree: for the six extractors it returns
   `wavenumber_exit_time` alone for `extract_GkSource_data.py`, `extract_tensor_source_data.py` and
   `extract_QuadSourceIntegral_data.py`, `GkNumericIntegration` and `wavenumber_exit_time` for the
   two Gk extractors, and `TkNumericIntegration` and `wavenumber_exit_time` for `extract_TkWKB_data.py`.
   The `QuadSourceIntegral` read in the extractor is an `object_read_batch`. This prompt closes the
   gap for that one extractor, in its own test (deviation 1). The comparison's blind spot (any
   reader that reads a target by `object_read_batch` or `object_read_*`) is not closed, and no §3
   issue was opened for it, because the brief fixes the index count at 106; whether to open one is
   the orchestrator's call.
4. **`[06a-readme-and-config-comments-state-the-superseded-quadrature-rtol-regime]`**
   (`tolerance-convergence`, `docs/OPEN_ISSUES.md` §1.5) names three comment sites: README
   §6.2, `config/defaults.py:163`, and `QuadSourceIntegral.py:1550`. This commit corrects the third
   site's values and leaves the second as the prompt requires. The row and its board were not
   touched; a reviewer may want that row narrowed.
5. **The `handover` board's phase-2/3 table, and the index row of
   `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`, give the seam band's
   `rtol = 1e-7` worst deviation as 9.97e-07.** The sweep's §4 gives 9.97e-07 for `1e-6` against
   `1e-7`, and **8.26e-07** for `1e-7` against `1e-8`; the other two bands' figures in that table are
   against the tightest rung, which is `1e-8`. The new comment and the provenance subsection use
   8.26e-07. The existing text on the board and in the index was not edited (this prompt's rule).
6. **Line numbers that now name the old layout.** `docs/TOLERANCE-PROVENANCE.md`'s entry headings
   say `config/defaults.py:159` (`rtol`) and `:166` (`atol`), and `inventory.py:723` carries `:159`
   as a literal. The constants are now at `:173` and `:180`. The headings are the record of the tree
   on 2026-09-18 and are not rewritten; the new subsection says `:173`.
7. **`config/defaults.py`'s `atol` comment is still the superseded finding**, as the prompt requires.
   `[06-quadrature-atol-is-inert-and-rtol-is-the-binding-half]` stays open on that sentence.
8. **The T1 search's directory list does not include `CosmologyConcepts`.** One non-tolerance hit
   there (`wavenumber.py:269`) was found by the wider second search. Nothing follows from it.
9. **The first background run of `ComputeTargets` stopped without a verdict** (see Verification).
   Cause not found; the re-run passed.

## State handed to the next prompt

- `config.defaults.DEFAULT_QUADRATURE_RTOL == 1e-7` and `DEFAULT_QUADRATURE_ATOL == 1e-32`, pinned
  by `ComputeTargets/tests/test_quadrature_tolerance_pin.py`. `main.py`, the build and
  `extract_QuadSourceIntegral_data.py` agree on the pair without the substitution of `RTOL_SITE`
  (which still matches once and is now an identity for a build at `1e-7`).
- Prompt 03 (read-only readers) may rely on the extractor's tolerance lookup resolving to the
  `1e-7` row. It may also want `test_quadrature_tolerance_pin`'s second class when it edits the
  extractor: the class reads the extractor's `quad_atol`/`quad_rtol` bindings and its
  `object_read_batch` call, so a restructuring of that helper has to keep them or change the test
  deliberately.
- Baselines after this commit: AdaptiveLevin 32, **ComputeTargets 669**, CosmologyModels 39,
  Datastore 551, LiouvilleGreen 148 (1 skipped), RunRegistry 199. Index: **106 open**.
- Registry: 12 runs, 10 finished, 2 unknown, nothing running; this prompt launched nothing.
