# Log 02 — The accuracy-parameter inventory

**Prompt:** `prompts/tolerance-convergence/02-accuracy-parameter-inventory.md`
**Commit:** *(this prompt's own commit)* — "Inventory every accuracy parameter in the pipeline"
**Model:** Opus 5
**Date:** 2026-09-16
**Result:** COMPLETE WITH DEVIATIONS — the inventory is delivered in full, and it **triggers prompt
02 §8's second stop condition**: there is a ninth keyed object type, `OneLoopIntegral`. It is
recorded unassigned and the user decides where it goes.

---

## What shipped

Two new documents. **Zero production files and zero test files in the diff**, as prompt 02 §7
requires.

### `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` (new, 583 lines, of which 366 are written prose before the generated block)

- **§0** says which parts are generated and which are written, and warns against hand-editing
  inside the markers.
- **§1** the headline: §2 (a)'s table is right row by row; there is a ninth keyed type; two policy
  tables are keyed on `Levin_threshold`; README §0.1's *summary* of §2 (a) undercounts the live
  cases; and `wavenumber_exit_time` is keyed by inequality.
- **§2** the keyed object types and the four corrections, each with its evidence:
  §2.1 `OneLoopIntegral`; §2.2 `Levin_threshold` and the two policy tables; §2.3 the inequality
  key; §2.4 how the Gauss orders reach an `IntegrationSolver` label that no lookup filters on.
- **§3** prompt 02 §4's four questions, answered explicitly and in its own terms.
- **§4** the exclusion boundary, eight classes, one line each, plus the one exclusion that had to
  be argued rather than applied.
- **§5** the generated block.
- **§6** how to regenerate, and — importantly — **what a re-run will not catch**.

### `docs/tolerance-convergence/inventory.py` (new, 1,308 lines)

Regenerates §5 between `<!-- BEGIN GENERATED … -->` / `<!-- END GENERATED -->`.

Public surface:

- `lookup_predicates(factory) -> List[Predicate]` — the comparison predicates a factory's
  `build()` filters its own table on, by `ast`. The heuristic is in the docstring: an
  `ast.Compare` mentioning exactly one `<alias>.c.<column>` is a filter; one mentioning two is a
  join and is dropped; only the object's own table and `tables["…"].alias(…)` aliases bound in the
  same function are kept.
- `keyed_object_types() -> List[Tuple[str, List[str], List[Predicate]]]` — every registered table
  whose predicate set mentions `atol_serial`, `rtol_serial`, `log10_tol` or `Levin_threshold`.
- `imported_constants() -> Dict[str, object]` — 39 constants, by `importlib`, never by parsing.
- `hard_coded_literals() -> List[Tuple[str, int, int, str, str, str]]` —
  `(path, call line, argument line, called function, keyword, literal)`.
- `render() -> str`, `splice(document, generated) -> str`, `main()`.
- Data: `PARAMETERS` (the nine-column inventory, 39 rows in seven groups), `OBJECT_COUNTS`,
  `ACCURACY_COLUMNS`, `BOOKKEEPING_COLUMNS`, `TOLERANCE_KEYWORDS`, `PRODUCTION_TREES`,
  `PRODUCTION_FILES`.

CLI: no argument prints to stdout; `--write` rewrites the block; `--check` exits 1 if it is stale.
No Ray, no datastore; `main.py` is read with `ast` and never imported.

### Board, index, log

`prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` — row 02, item **T3**, four new §3 issues;
`docs/OPEN_ISSUES.md` §1.5 — four new rows, count 70 → 74, date corrected. Same commit
(`CLAUDE.md`).

---

## Deviations from the prompt

### 1. The prompt's acceptance quotes `ComputeTargets` 452; it is 484 — **STRUCTURALLY REQUIRED**

Prompt 02 §7 says "`ComputeTargets` suite 452 → 452". That figure was written before prompt 01
landed a test module. The suite is **484** at `4a94148` and was 484 before this prompt's commit as
well; the number this prompt must hold unchanged is 484, and it did. `CosmologyModels` is 39 → 39
as the prompt says. Nothing was done about the prompt text.

### 2. `ComputeTargets` is not green, and was not green before this prompt — **STRUCTURALLY REQUIRED**

`test_tk_wkb_phase.TestCost.test_wall_time_per_object` fails: `cold = 0.0605 s` in the full-suite
run and `0.0601 s` when re-run alone, against `COST_COLD_WALL_TIME_LIMIT = 0.06`. The diff for this
prompt contains **no `.py` file that the suite imports** — two new files under
`docs/tolerance-convergence/`, neither on any test's import path — so the failure is a property of
the machine at this commit and not of this change.

It is already on the record. `prompts/phase-representation/IMPLEMENTATION_STATE.md:114` records
the same test failing on this machine at `0c61799` "with no `.py` file changed — 0.0638–0.0979 s
against its 0.06 s limit, four runs of four", and attributes it to
`[07-tk-per-object-cost-is-all-setup]` on the `GkTk-remedial` board, which is open. **No new issue
is opened for it** (README §5 rule 4: it is neither this prompt's to fix nor a new observation).
Recorded here so that the acceptance row reads honestly: **484 tests, 483 pass, 1 pre-existing
wall-clock failure.**

### 3. A ninth keyed object type was found — **STRUCTURALLY REQUIRED, and a stop**

Prompt 02 §8: "You find a ninth keyed object type … Stop and report; the user decides where it
goes." `OneLoopIntegral` is the ninth. The prompt also asks, in §4.3, whether any parameter keys an
object type §2 (a) does not list, and in §4 says that a corrected table "is a successful outcome
for this prompt, not a problem" — so the inventory was **completed** rather than abandoned, because
abandoning it would have left the campaign with neither the table nor the finding. What was *not*
done is the thing §8 reserves for the user: `OneLoopIntegral` is **not** allocated to prompt 03, 04
or 05. Its row says "nobody — the user decides", and the issue is unassigned.

### 4. Two extra kinds of keyed table are reported beside the nine — **IMPLEMENTATION CHOICE**

`GkSourcePolicy` / `QuadSourcePolicy` (keyed on `Levin_threshold`) and the `tolerance` table itself
(keyed on `log10_tol` to `DEFAULT_FLOAT_PRECISION`) satisfy the script's filter and are reported,
making the generated count **12 tables**. The alternative was to filter them out and report nine.

Reported, because: `DEFAULT_LEVIN_THRESHOLD` is a constant README §1.2 names by hand as one the
provenance note must cover, so prompt 06 needs its keying to be on the record; `GkSourcePolicyData`
is keyed through `GkSourcePolicy`, so the parameter does reach stored objects; and the `tolerance`
row is what makes it visible that `DEFAULT_FLOAT_PRECISION` is the resolution at which two distinct
tolerance requests become one object. §3.1 of the document states all three counts (9 / 11 / 12)
and says where each boundary is drawn, so a reader who wants "nine" can have it.

### 5. The inventory reaches further than prompt 02 §2's list 1–4 — **IMPLEMENTATION CHOICE**

§2's list is "at minimum". The delivered table has 39 rows in seven groups, adding: the source-grid
construction constants (in the `BackgroundModel` key through the grid digest); the background
derivative-fit and spline orders; the $T(z)$ representation's node count and spline order (keyed
through `T_Z_REPRESENTATION_VERSION`); `CHEBYSHEV_ORDER`, `DEFAULT_LEVIN_MAX_DEPTH` and
`scipy.quad`'s `limit=100`; and `DEFAULT_HEXIT_TOLERANCE`.

The alternative was to stop at the list and note the rest in prose. Rejected because prompt 02 §4.4
asks for a *count* of the accuracy parameters that are in no key, `[20-…]` asserts five, and the
question cannot be answered honestly without enumerating the others — the answer is seventeen
(document §3.4). Each added row names the campaign that owns it, and none is proposed for this
campaign's prompts.

### 6. The literal sweep reports two line numbers per site — **IMPLEMENTATION CHOICE**

`prompts/background-solver-robustness/PROVENANCE.md` cites `LambdaCDM_GenericEOS.py:636`, the
`root_scalar(` line; this campaign's README §3.2 cites `main.py:1119`, the `phase_atol=` line. Both
conventions are in the record and neither is wrong, so the table has a **Call site** column and an
**Argument line** column and a reader can match either citation. `integration_tools.py` is
therefore reported as `:92` / `:95-96` where README §0.5 cites `:95`.

### 7. `black` reformatted `inventory.py` after it was written — **UNINTENDED DRIFT, kept**

The first draft was not `black`-clean (one `frozenset` literal and one list argument). `black` was
run per `CLAUDE.md` and the reformatting kept; the generated block is byte-identical before and
after, confirmed by re-running `--check`.

---

## Verification performed

Every command below was run from the repository root at the commit this log describes.

| Prompt §7 check | Result |
|---|---|
| Every parameter of §2's list 1–4 appears in the table | **Yes**, and more (deviation 5). The five `config/defaults.py` accuracy constants and `DEFAULT_LEVIN_THRESHOLD`: rows A1–A6. The four `*_GAUSS_ORDER` and the margin: B1–B5. Every production `root_scalar` carrying a tolerance: C1, C3, C4, C5 (plus C2 the acceptance guard and C6 the test-only probe). Every `atol`/`rtol` **column** on an `ObjectFactories/` table: generated §5.1, all twelve. The sweep is made exhaustive by *two independent passes* — the factory sweep (§5.1, by `register()` + `ast`) and the literal sweep (§5.3, by `ast` over the production trees) — and the document says so in §0 and §2 |
| Every row has all nine columns filled | **Yes**, 39 rows × 9. `"unknown"` appears nowhere; **"never chosen" appears in 12 rows**: `DEFAULT_ABS_TOLERANCE`, `DEFAULT_REL_TOLERANCE`, `DEFAULT_LEVIN_THRESHOLD`, `RESIDUAL_WKB_REGION_MARGIN` ("never chosen against a criterion"), `_solve_horizon_exit`'s pair, `DEFAULT_HEXIT_TOLERANCE`, `find_phase_extremum`'s pair, `temperature_crossing_log1pz`, `DEFAULT_LEVIN_MAX_DEPTH`, `limit=100`, `DEFAULT_LEVIN_ABSTOL`, `DEFAULT_LEVIN_RELTOL` |
| §4's four questions answered explicitly, in the document and in the log | **Yes** — document §3.1–§3.4, and "The four answers" below |
| The exclusion boundary stated, one line per excluded class | **Yes** — document §4, eight classes, plus the argued non-exclusion of `DEFAULT_ABS_TOLERANCE` |
| `inventory.py` re-run reproduces the table section byte-for-byte | **Yes.** `PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --check` → `inventory.py: TOLERANCE-INVENTORY.md is up to date`, exit 0. Re-checked after `black` reformatted the script: still exit 0 |
| The three `LambdaCDM_GenericEOS.py` entries cite `background-solver-robustness/PROVENANCE.md` at the re-anchored line numbers, not re-derived | **Yes** — rows C4 (`:636`), C5 (`:1137`) and C6 (`CosmologyModels/tests/T_z_reference.py:285`), each carrying "Lifted from … PROVENANCE.md §N, not re-derived". Independently confirmed by the literal sweep, which finds `:636`/`:637` and `:1137`/`:1138` and finds **no** third solve in production |
| `ComputeTargets` suite | **484 → 484**, 483 pass, **1 pre-existing failure** (deviation 2): `Ran 484 tests in 201.448s`, `FAILED (failures=1)`, `test_wall_time_per_object` `0.0605 > 0.06` |
| `CosmologyModels` suite | **39 → 39, OK.** `Ran 39 tests in 0.786s`, `OK` |
| Production and test files in the diff | **Zero.** `git status --short` before the commit shows `?? docs/tolerance-convergence/` and nothing else; the commit adds two files there plus the log, the board and `docs/OPEN_ISSUES.md` |
| `black --check` on `inventory.py` | **Clean.** `1 file would be left unchanged` |

### The four answers

**(1) Are there eight keyed object types, or more?** **Nine**, counting as README §2 (a) counts —
object types whose lookup key carries an `atol`/`rtol` pair. The ninth is `OneLoopIntegral`:
`ObjectFactories/OneLoopIntegral.py:101-116` declares both columns as indexed non-nullable foreign
keys into `tolerance` and `build()` filters on both at `:153-154`; the table is registered
(`Datastore.py:120`), sharded on `k` (`config/sharding.py:34`) and given a client-pool budget
(`ClientPool.py:43`). `main.py` never builds one — the only `OneLoop` strings in it are two
`store_tag` labels at `:1021-1022` — and `ComputeTargets/OneLoopIntegral.py:94-112`'s `compute()`
does nothing but replace a label. **Eleven** if `GkSourcePolicy` and `QuadSourcePolicy` are counted
(`Levin_threshold`); **twelve** if the `tolerance` table's own identity query is. The enumeration
is from `Datastore/SQL/Datastore.py`'s `_factories` map, not from the README.

**(2) Live / vestigial / vestigial-in-computation-but-load-bearing-in-key?** **Four live**:
`wavenumber_exit_time` (`root_scalar`, `wavenumber.py:982-983`), `GkNumericIntegration` (DOP853,
`:379-380`), `TkNumericIntegration` (DOP853, `:412-413`), `QuadSourceIntegral` (Levin and
`scipy.quad`). **Four vestigial in the computation and load-bearing in the key**:
`BackgroundModel`, `GkWKBIntegration`, `TkWKBIntegration`, `GkSource`. **One vestigial in both**:
`OneLoopIntegral`, which has no solver *and* no row for the key to identify.

`GkSource` is the fourth vestigial case and differs in kind, as the prompt says: it assembles and
integrates nothing, so there is no order to put in its key. What it does carry —
`DEFAULT_G_WKB_DIFF_ABS_TOLERANCE = 1e-3`, `DEFAULT_G_WKB_DIFF_REL_TOLERANCE = 1e-2`
(`GkSource.py:42-43`) — are warning thresholds read at `:278` and `:295`, not accuracy requests,
and are excluded under boundary class 4 with the exclusion named rather than silent.

**Correction to README §0.1, not to §2 (a).** §0.1 says six object types share one `(atol, rtol)`
pair "and of those six only one actually uses it". **Two** do: `GkNumericIntegration` and
`wavenumber_exit_time`. §2 (a)'s own table marks both "yes", and `RECONCILIATION.md` §2.1 says
"three are live" counting `TkNumericIntegration`, which is not one of the six. The table is right;
the sentence is not. Per prompt 02 §6 the README was not edited — prompt 06 reconciles it.

**(3) Does any parameter key an object type §2 (a) does not list?** **Yes, three times.**
(i) the shared pair keys `OneLoopIntegral`; (ii) `Levin_threshold` keys `GkSourcePolicy` and
`QuadSourcePolicy`, and through `policy_serial` it keys `GkSourcePolicyData`; (iii)
`DEFAULT_FLOAT_PRECISION` is the matching tolerance of the `tolerance` table's own identity query
(`ObjectFactories/tolerance.py:31-33`) and of both policy tables'. (iii) is excluded from the
inventory as an identity precision, and **reported rather than silently excluded** as prompt 02 §2
requires, because there is one site where it does accuracy work: it is the slack in
`wavenumber_exit_time`'s tolerance inequality. As an identity precision it cannot bind on anything
prompt 05 might do — 1e-7 in $\log_{10}$ of the tolerance is a factor of 1.00000023 in the
tolerance, and every candidate here is separated by decades. Indirectly, two more reach keys
without appearing in one: `_find_rho_equality`'s pair (through the equality redshifts in the grid
digest) and `find_phase_extremum`'s `xtol=1e-6, rtol=1e-4` (through the `z_init` it fixes).

**(4) Does any object type carry an accuracy parameter that is in no key at all?
`[20-…]` says five.** **The five it names are right; five is an undercount of the phenomenon.**
Counting every parameter that sets the accuracy of a stored quantity and is in no lookup key
anywhere: **seventeen** (document §3.4, itemised). The twelve beyond the issue's five are
`CHEBYSHEV_ORDER`, `DEFAULT_LEVIN_MAX_DEPTH`, `limit=100`, `phase_atol=1e-12`,
`amplitude_rtol=1e-12`, `DERIVATIVE_SPLINE_ORDER`, `STORED_SAMPLE_SPLINE_ORDER`, the four
`DERIVATIVE_FIT_*` and `DEFAULT_HEXIT_TOLERANCE`; all belong to other campaigns and none is
proposed for this one. **This is not a proposal to widen D3.** It is what prompt 06 has to cover,
because README §1.2 requires that no accuracy parameter be unexplained when the campaign closes.

**And the mechanism behind `[20-…]`, measured rather than assumed** (document §2.4):
`TAU_GAUSS_ORDER` *does* reach a stored field — `TAU_SOLVER_LABEL` is
`f"{base}-stepping{TAU_GAUSS_ORDER}"` (`BackgroundModel.py:49`) and `main.py:3510-3514` fetches the
`IntegrationSolver` with `stepping=TAU_GAUSS_ORDER`; `RHO_GAUSS_ORDER` likewise through
`PHASE_SOLVER_STEPPING` (`WKB_phase_function.py:86`, `main.py:3519-3523`). But **no lookup filters
on `solver_serial`**: the derived predicate set for `BackgroundModel` is `cosmology_type`,
`cosmology_serial`, `atol_serial`, `rtol_serial`, `source_grid_digest`, `source_grid_construction`,
`z_init_serial`, and the solver table is joined at `BackgroundModel.py:259-262` only so the label
can be read back. **So raising `TAU_GAUSS_ORDER` today and re-running serves the order-4 row that
already exists, with a stored `solver_label` saying `stepping4`.** That is D3's "an order change
that silently serves a stale row", in the concrete. `CS_TAU_GAUSS_ORDER` and
`FRICTION_F_GAUSS_ORDER` do not reach even a label; `RESIDUAL_WKB_REGION_MARGIN` reaches no key,
label or tag at all.

### Stop conditions, checked one by one (prompt 02 §8)

| Condition | Status |
|---|---|
| A parameter reaches a solver that §2 (a) says is not keyed on one, or vice versa, **in a way that changes which prompt owns a target** | **No.** Every §2 (a) row is confirmed. The one correction that touches a live/vestigial verdict is to README §0.1's summary sentence, not to the table, and it moves no target between prompts: `wavenumber_exit_time` was already prompt 03's (board T6) |
| **A ninth keyed object type**, or an accuracy parameter on a target neither 03 nor 04 covers | **TRIGGERED.** `OneLoopIntegral`. Reported, unassigned, issue opened. Also reported: `Levin_threshold` on the two policy tables, which is on no compute target and is covered by prompt 06's provenance note rather than by 03 or 04 |
| `inventory.py` cannot derive the key columns programmatically without importing `main.py` or standing up a datastore | **No.** It derives them by importing the factory classes and walking `build()` with `ast`; `main.py` is never imported; no datastore, no Ray connection. Runtime ~9 s |
| You need to change a production file, a test file or `config/defaults.py` | **No.** Zero in the diff |

---

## Observations not acted on

1. **`extract_TkWKB_data.py:439` can never find a production `TkNumericIntegration` row.** It
   passes `"atol": atol` with `atol = DEFAULT_ABS_TOLERANCE` (`:364`) into an `object_get` for
   **both** `TkNumericIntegration` and `TkWKBIntegration` (`:444-445`), and `main.py` writes every
   `TkNumericIntegration` under `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` (`main.py:3199-3203`, and the
   comment at `:3475-3479` saying "Every `TkNumericIntegration` `object_get` — the work items and
   **every lookup** — must use it"). `TkNumericIntegration`'s lookup filters `atol_serial ==`
   (generated §5.1), so the query misses. None of the six `extract_*.py` readers imports
   `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`. Not verified by running the script — that needs a datastore,
   which this prompt may not stand up. Issue
   `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]`.
2. **`DEFAULT_ABS_TOLERANCE` doubles as a float-comparison epsilon at seven sites** —
   `ComputeTargets/GkSource.py:96`, `:104`, `:275`;
   `Quadrature/integrators/numeric_with_phase_cut.py:618`, `:737`, `:790`;
   `LiouvilleGreen/WKBtools.py:83`. All seven are `fabs(a - b) < DEFAULT_ABS_TOLERANCE`-shaped
   comparisons with no connection to the Green's-function ODE. Prompt 05 is expected to retune or
   split this constant; if it does, all seven move with it. Issue
   `[02-shared-atol-doubles-as-a-float-comparison-epsilon]`.
3. **`DEFAULT_LEVIN_THRESHOLD = 1.0` is not merely unchosen; nothing takes it.** Both policy
   constructors default to it and `main.py:3542`, `:3548`, `:3560`, `:3566` pass `1.5` and `5.0`
   explicitly, so it never reaches the datastore. The two values that do are bare literals with no
   comment, repeated in `extract_GkSource_data.py:855`, `:861` and
   `extract_QuadSourceIntegral_data.py:1063`, `:1069`. Recorded in the inventory row and in
   document §2.2; **no separate issue** — README §1.2 already schedules this constant for
   `docs/TOLERANCE-PROVENANCE.md`, and "never chosen, and never used" is the entry prompt 06 will
   write.
4. **`ComputeTargets/QuadSourceIntegral.py:1550` still says the pipeline supplies
   `DEFAULT_QUADRATURE_ATOL = 1e-25`.** It has been `1e-32` since `source-remediation` prompt 12.
   Already recorded on this board under "Recorded by the rebase, not owned here"; re-confirmed at
   `4a94148` and not re-opened. README §0.4 puts that file out of bounds.
5. **`wkb_reference.production_source_grid` is still imported by 28 call sites in 20 files.**
   Unchanged from prompt 01's narrowing of `[00-three-production-grid-reproductions]`; nothing in
   this prompt touches it, and nothing in this prompt's figures depends on a grid.
6. **`OneLoopIntegral.compute()` has two bugs of its own**, seen while establishing that it is a
   stub: `:105-107` raises "value haa already been computed" when `self._value is **None**` —
   inverted condition and a typo — and `:116` raises inside `store()` before an unreachable
   comment. Not acted on: it is a production file, and it is the same object the §8 stop condition
   is about, so fixing it would pre-empt the user's decision. Folded into
   `[02-oneloopintegral-is-a-ninth-keyed-object-type]` rather than opened separately.
7. **`ComputeTargets/tests/test_tk_wkb_phase.TestCost.test_wall_time_per_object` fails on this
   machine.** Pre-existing, no `.py` in this diff, already attributed to
   `[07-tk-per-object-cost-is-all-setup]` by `prompts/phase-representation` log 02 observation 5.
   No new issue.
8. **`three_bessel_integrals.quad_JJJ` / `quad_YJJ` and `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` have
   no production caller.** Production imports only `BesselPhaseGroup` from that module
   (`QuadSourceIntegral.py:18`). Excluded under boundary class 6 and named there, because a keyword
   grep for `atol=` in `LiouvilleGreen/` finds 35 hits in that file and every one of them is dead
   as far as the pipeline is concerned.

---

## State handed to the next prompt

### Prompt 03's target list — the adaptive solvers

Three targets, by name, with the object count of each sector and the current value of the parameter
that keys it. **Confirmed unchanged from README §3.3; prompt 03 owns exactly these three.**

| Target | Object count | Parameter today | What it reaches | Notes prompt 03 needs |
|---|---|---|---|---|
| `GkNumericIntegration` | **~65,000 per model** (one per $(k, z_{\rm source})$, `main.py:1770-1791`; the count is `GkTk-remedial` README §6's and is a **version-0** figure, not re-taken here) | `atol = DEFAULT_ABS_TOLERANCE = 1e-10`, `rtol = DEFAULT_REL_TOLERANCE = 1e-8` | DOP853 via `GkNumericIntegration.py:379-380` → `numeric_with_phase_cut` | Both are **never chosen**: no log, document or comment records a measurement behind either. `scipy` clamps `rtol` at 100 eps = 2.22e-14 (`numeric_with_phase_cut.py:62`), which is the hard floor of the `rtol` axis |
| `TkNumericIntegration` | **50 per model** (`main.py:1215`) | `atol = DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` (**settled**, D1, user 2026-09-12), `rtol = DEFAULT_REL_TOLERANCE = 1e-8` | DOP853 via `:412-413` | Only `rtol` is open. Its key also carries `break_point_kind` (generated §5.1) which prompt 05 must not disturb |
| `wavenumber_exit_time` | **50 per model** (`NUMBER_SOURCE_K_VALUES = 50`, `main.py:3584`; the response sample is the same 50 values, `:3596`), each solved at **1 + 3 superhorizon + 5 subhorizon** offsets (`wavenumber.py:1021-1033`) | `xtol = DEFAULT_ABS_TOLERANCE = 1e-10`, `rtol = DEFAULT_REL_TOLERANCE = 1e-8`, both in $u = \log(1+z)$ | `root_scalar`, Brent, **bracketed** by a geometric widening search (`wavenumber.py:930-977`), `:982-983` | **Two things prompt 03 must know.** (i) Its lookup is an **inequality**, not an equality: a loose request is served a tighter stored row (document §2.3), so a sweep must call `_solve_horizon_exit` directly and never go through the datastore. (ii) `DEFAULT_HEXIT_TOLERANCE = 1e-2` at `:993` is an acceptance guard **eight orders looser** than the `xtol` it guards and is not the solve's error bound |

Also for prompt 03: `find_phase_extremum`'s `xtol=1e-6, rtol=1e-4`
(`LiouvilleGreen/integration_tools.py:92`, arguments at `:95-96`) fixes the numeric stop point
$z_{\rm init}$ that is a **key column** on `TkNumericIntegration` (`z_init_serial`) and on both WKB
targets (`z_init`). It is the hand-over campaign's (`[11-stop-point-root-tolerance]`) and is **not
retuned here**, but a `TkNumericIntegration` sweep moves through it and should say so.

### Prompt 04's target list — the order-governed targets

| Target | Object count | Knob today | In a key? | Notes prompt 04 needs |
|---|---|---|---|---|
| `BackgroundModel` | **1 per (cosmology, source grid)** (`main.py:1062`) | `TAU_GAUSS_ORDER = 4`, `CS_TAU_GAUSS_ORDER = 4`, `FRICTION_F_GAUSS_ORDER = 4` (`BackgroundModel.py:35`, `:43`, `:44`) | **No.** `TAU_GAUSS_ORDER` reaches `TAU_SOLVER_LABEL`/`stepping` and thence `solver_serial`, **which the lookup does not filter on**; the other two reach nothing at all | `atol`/`rtol` are accepted "for signature compatibility … the table has no tolerances" (`:409`). Raising an order today serves the stale row — document §2.4 |
| `GkWKBIntegration` | **~65,000 per model** (version-0 count) | `RHO_GAUSS_ORDER = 4` (`phase_residual.py:90`), `RESIDUAL_WKB_REGION_MARGIN = 0.5` (`:238`) | **No.** `RHO_GAUSS_ORDER` reaches `PHASE_SOLVER_STEPPING` and the solver label, not filtered on; the margin reaches **no key, label or tag** | `atol`/`rtol` vestigial (`GkWKBIntegration.py:334-336`) |
| `TkWKBIntegration` | **50 per model** | as above | as above | `atol`/`rtol` vestigial (`TkWKBIntegration.py:38-39`, `:349`) |

For D3, the schema-churn cost prompt 04 must price: the three targets carry `atol_serial` and
`rtol_serial` as indexed non-nullable foreign keys into `tolerance`
(`BackgroundModel.py:149-159`, `GkWKBIntegration.py:106-116`, `TkWKBIntegration.py:106-116`), and
each lookup filters on both (generated §5.1). Replacing them with an order column means: the column
definitions, the `build()` filter, the `store()` payload, the `read_table`/`inventory` selects that
join `tolerance` twice to report `log10_atol`/`log10_rtol`, and — because the orders are not
per-object but per-module — a decision about whether one column carries all four of
`BackgroundModel`'s or whether it gets three. `GkSource` is the fourth vestigial case and has
**nothing to put there**, so its options are drop or keep-as-inert; this document does not
recommend (prompt 02 §6).

### Prompt 05's hand-off

1. **The `ast` guard's predicate must widen past `"…Integration"`** (README §2 (g)) and the list it
   must reach is **nine**, not eight: `wavenumber_exit_time`, `BackgroundModel`,
   `TkNumericIntegration`, `TkWKBIntegration`, `GkNumericIntegration`, `GkWKBIntegration`,
   `GkSource`, `QuadSourceIntegral`, **`OneLoopIntegral`** — unless the user rules the ninth out.
2. **README §2 (g)'s "a new parameter makes every existing row unreachable" is false for
   `wavenumber_exit_time`.** Its key is an inequality (document §2.3): tightening misses as
   expected, loosening *hits* a tighter stored row and returns its `z_exit`, and the object's
   `atol`/`rtol` properties report the **stored** pair rather than the requested one
   (`wavenumber.py:729-734`). `MultipleResultsFound` is caught at `:260`, so two qualifying rows
   raise rather than choosing.
3. **Retuning `DEFAULT_ABS_TOLERANCE` moves seven float comparisons with it** (observation 2). If
   prompt 05 splits the constant per target, those seven sites need a name of their own or they
   silently follow whichever target keeps the old one.
4. **The six `extract_*.py` readers are `extract_Gk_data.py`, `extract_GkSource_data.py`,
   `extract_GkWKB_data.py`, `extract_QuadSourceIntegral_data.py`, `extract_TkWKB_data.py`,
   `extract_tensor_source_data.py`.** All six import `DEFAULT_ABS_TOLERANCE` and
   `DEFAULT_REL_TOLERANCE`; only `extract_QuadSourceIntegral_data.py` imports the quadrature pair;
   **none** imports `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`, which is already a live defect
   (observation 1) and will become a wider one the moment more constants are split.
5. **Both numeric factories' keys already carry `break_point_kind`** (generated §5.1) — prompt 05
   must not disturb it (README §2 (g)).

### Prompt 06's hand-off — the provenance fields

`docs/TOLERANCE-PROVENANCE.md` assembles from the inventory's nine columns plus prompts 03–05's
measurements. The rows that already carry a complete provenance and need no measurement:
`DEFAULT_TK_NUMERIC_ABS_TOLERANCE`, `DEFAULT_QUADRATURE_ATOL`, `DEFAULT_QUADRATURE_RTOL`,
`_solve_T_z`, `_find_rho_equality`, the two Bessel budgets, `CHEBYSHEV_ORDER`, and the
source-grid group. The rows whose provenance entry is **"never chosen"**, in those words, are the
twelve listed in the acceptance table above. **Seventeen accuracy parameters set the accuracy of a
stored quantity and are in no lookup key** (document §3.4) — all seventeen need an entry, and
twelve of them belong to campaigns other than this one.

---

## Issues opened

| Issue | Assigned |
|---|---|
| `[02-oneloopintegral-is-a-ninth-keyed-object-type]` | **unassigned — the user decides** (prompt 02 §8) |
| `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` | candidate for prompts 03 and 05 |
| `[02-shared-atol-doubles-as-a-float-comparison-epsilon]` | candidate for prompt 05 |
| `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]` | unassigned; prompt 05 touches the same six files |

`docs/OPEN_ISSUES.md`: **70 → 74 open**, §1.5, same commit.
