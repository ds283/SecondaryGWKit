# Log 12 — A separate absolute tolerance for the transfer-function numeric run

**Prompt:** prompts/GkTk-remedial/12-tk-numeric-atol.md
**Commit:** *(this commit)* — Give the transfer-function numeric run its own absolute tolerance
**Model:** Claude Opus 5
**Date:** 2026-09-12
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### `config/defaults.py`

New constant after `DEFAULT_REL_TOLERANCE`:

```python
DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13
```

with a 25-line comment carrying review §12.5's argument and this prompt's measurements: why
`DEFAULT_ABS_TOLERANCE = 1e-10` is right for $G_k$ (|G| ~ 1e10 in `Mpc_units`, so the absolute
floor never binds — tightening it moves $G$ by 8.5e-10 of the envelope for 0.4 % more
right-hand-side evaluations) and wrong for $T_k$ (|T| ~ 1.2e-2 at the stop point and ~1e-5 deeper
in, so 1e-10 is a 1e-5 *relative* tolerance), the three-row `atol` table (9.9e-6 / 2.5e-6 / 2.5e-6
of the envelope at 1e-10 / 1e-13 / 1e-16), and the statement that below 1e-13 the floor is the
super-horizon initial condition, not the solver (`[00-tk-superhorizon-ic-series]`, out of scope).

### `main.py`

- `:46-54` — the `config.defaults` import gains `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`.
- `:2980-2998` — a fifth `tolerance` object is built beside `atol`/`rtol`/`quad_atol`/`quad_rtol`:
  `atol, rtol, quad_atol, quad_rtol, Tk_numeric_atol = ray.get([... , pool.object_get("tolerance",
  tol=DEFAULT_TK_NUMERIC_ABS_TOLERANCE)])`, with a comment saying that every `TkNumericIntegration`
  `object_get` must use it because the tolerance is part of the datastore key.
- **The five `TkNumericIntegration` `object_get` sites** (`grep -n '"TkNumericIntegration"'`, then
  the batch dict behind each `**x` dispatch; line numbers after the edit):

  | # | Site | Line | What it is | Before → after |
  |---|---|---|---|---|
  | 1 | `build_Tk_numeric_work`, `query_batch` → `RayWorkPool` at `:702` | `:684` | the "does this object already exist?" query | `"atol": atol` → `"atol": Tk_numeric_atol` |
  | 2 | `build_Tk_numeric_work`, direct `pool.object_get` | `:755` | the work item itself | `atol=atol` → `atol=Tk_numeric_atol` |
  | 3 | `build_Tk_WKB_work`, `payload_batch` → `RayWorkPool` at `:919` | `:895` | the numeric object read for the WKB initial condition (`Tk.stop_T`, `Tk.stop_Tprime`) | `"atol": atol` → `"atol": Tk_numeric_atol`, **and the queue is now dispatched over `payload_batch` instead of `query_batch`** — see deviation 1 |
  | 4 | `build_tensor_source_work`, `Tk_lookup_batch` → `RayWorkPool` at `:1119` | `:1102` | the $T_q$, $T_r$ lookup for `QuadSource` | `"atol": atol` → `"atol": Tk_numeric_atol` |
  | 5 | `build_QuadSourceIntegral_batch`, `Tk_numeric_lookup_batch` → `RayWorkPool` at `:2730` | `:2713` | the $T_q$, $T_r$ lookup for `QuadSourceIntegral` | `"atol": atol` → `"atol": Tk_numeric_atol` |

  Each carries a three-line comment naming prompt 12 and review §12.5 and saying that the work
  item and every lookup have to agree. `main.py:309, :311` also contain the string
  `"TkNumericIntegration"`, but as a label in the missing-data warning list, not an `object_get`.
  `TkWKBIntegration`, `GkNumericIntegration` and `GkWKBIntegration` keep `atol` at all nine of
  their sites.

### `ComputeTargets/tests/test_main_plumbing.py` (extended, nothing existing disturbed)

New section after prompt 16's, with module-level helpers `_own_nodes`, `_object_get_calls`,
`_atol_values_under`, `_name_of` and `tk_numeric_tolerance_sites() -> (sites, unclassified)`, plus
`TkNumericToleranceWiringTestCase` (5 tests). The finder classifies **every** integration-class
`object_get`/`object_get_vectorized` in `main.py` — 14 of them — and reports the tolerance each
carries: from the call's own `atol=` keyword where it has one, and otherwise from the batch dict
the enclosing `RayWorkPool` dispatches over. The tests assert that nothing is left unclassified,
that exactly five `TkNumericIntegration` sites exist, that all five carry `Tk_numeric_atol`, that
the other nine carry `atol`, and that `Tk_numeric_atol` is the `tolerance` built from
`DEFAULT_TK_NUMERIC_ABS_TOLERANCE` (by position inside the `ray.get` list) rather than an alias
of `atol`. `load_main_py_functions` is not used and not changed: these calls are inside
`run_pipeline`, and it extracts module-level `FunctionDef` nodes only.

### `ComputeTargets/tests/test_tk_numeric_atol.py` (new, 8 tests)

Review §12.5's geometry through the undecorated `numeric_with_phase_cut._function`: exact
radiation (`wkb_reference.RadiationModel`), $T=1, T'=0$ five e-folds outside the horizon, the
production source grid (100 per decade of $z$) truncated at $0.85z_{e6}$, the $(z_{e3},z_{e6})$
stop window, `rtol = 1e-8`, `warn_unresolved_osc=False`. Public helpers `x_of_z`, `T_exact`,
`dT_dx`, `T_envelope`, `exact_initial_data`, `geometry`, `run`, `cached_run`, `max_T_error`,
`max_G_difference`; each `(sector, atol, initial data)` is solved once and shared.

## Deviations from the prompt

### 1. `build_Tk_WKB_work`'s numeric lookup was dispatched over the wrong batch — STRUCTURALLY REQUIRED

The prompt says to pass `atol=Tk_numeric_atol` in **every** `pool.object_get("TkNumericIntegration",
…)`, work-item creation and every lookup. Four of the five are `task_builder=lambda x:
pool.object_get("TkNumericIntegration", **x)`, so the tolerance is whatever the batch dict holds.
In `build_Tk_WKB_work` the batch dict fed to that queue was **`query_batch`** — the
`TkWKBIntegration` existence query, built over the whole batch — while `payload_batch`, built
immediately above over `missing` alone and identical in every key, was never used
(`main.py:899-913` before the edit; the slip dates from `8e96750`, 2025-05-23). The
`GkWKBIntegration` twin of the same function passes `payload_batch` (`main.py:1526-1528`), which
is what the code intends.

This cannot be worked around inside the prompt's scope: `query_batch` feeds the `TkWKBIntegration`
query, which must keep `atol`, and the same dict feeds the `TkNumericIntegration` lookup, which
must carry `Tk_numeric_atol`. Leaving it would have produced exactly the failure this prompt
exists to prevent — with one difference, in this case it is loud rather than silent: the lookup
would find no row, `Tk.available` would be false, and the stage would print "attempt to compute WKB
solution … without a pre-computed initial condition" and raise `RuntimeError`. So the queue is now
dispatched over `payload_batch`, with a six-line comment saying why.

The repair has a second, pre-existing effect that is worth recording: the results of this queue are
consumed as `zip(missing, lookup_queue.results)` with an `assert Tk._k_exit.store_id ==
k_exit.store_id`. Over `query_batch` the results are one per element of `batch`, so whenever
`missing` was a strict subset the zip was misaligned and that assertion would have fired; over
`payload_batch` they are one per element of `missing` and the pairing is right by construction.
Nothing else changes: the two dicts agree key for key, `_do_not_populate: True` included.

### 2. `Tk_numeric_atol` is a module-level name, not a `run_pipeline` parameter — IMPLEMENTATION CHOICE

`run_pipeline` takes `atol` and `rtol` as parameters but reads `quad_atol`/`quad_rtol` as globals
of the `with ShardedPool(...)` block. Adding a sixth parameter would have touched the signature,
the one call site and nothing else; following `quad_atol` keeps the diff to the five sites plus the
construction and matches the existing convention for a tolerance used by one stage. The structural
test does not depend on which was chosen.

### 3. The wiring test cannot be the keyword check the prompt describes — STRUCTURALLY REQUIRED

Prompt §1 item 3 asks for a check that "every `Call` whose first positional argument is the string
`"TkNumericIntegration"` … has a keyword `atol` whose value is the name `Tk_numeric_atol`". Four of
the five sites reach `object_get` as `**x` from inside a `task_builder` lambda and have no `atol`
keyword at all, so that check would have passed on one site and been silent on the four that
matter. The test therefore resolves each `**x` dispatch back to the `RayWorkPool`'s batch argument
and reads the `"atol"` entry of the dict literal assigned to that name in the same scope (scope
being "this `def`, not counting nested `def`s"; lambdas are deliberately not a boundary).

Two additions beyond the prompt, both IMPLEMENTATION CHOICE within that: `object_get_vectorized`
is included, because three `*Integration` sites use it (`GkNumericIntegration` ×2,
`GkWKBIntegration`) and a future vectorized `TkNumericIntegration` lookup would otherwise be
invisible to the guard; and `tk_numeric_tolerance_sites` returns an `unclassified` list which a
test asserts is empty, so a site written in a spelling the finder does not understand fails the
test instead of silently not being checked.

### 4. The test wavenumber is $k=10^6/{\rm Mpc}$ — IMPLEMENTATION CHOICE

The prompt fixes the model, the initial data, the grid, the window and `rtol`, but not $k$. At
$k=10^6$ all four right-hand-side evaluation counts of review §12.5's table are reproduced
**exactly** — 6476 / 6401 / 7379 / 7829 — so that is the review's own geometry and every number
asserted is comparable with the review's line by line. The alternative, sweeping the production
$k$-range in the test module, was rejected on runtime and because the comparison with §12.5 is
what the thresholds are written against; the sweep was run once here instead, and its result is
recorded as an observation below.

## Verification performed

All figures below are measured, not reasoned. `PYTHONPATH=. ./venv/bin/python …` from the worktree
root.

### Review §12.5 reproduced (RadiationModel, $k=10^6$, production source grid, `rtol = 1e-8`)

486 samples requested from $z_{\rm source}=1.4841\times10^8$ ($x_i = 0.003890$), 478 returned (the
ODE terminates on the $z_{e6}$ event), $x_{\rm end} = 275.8$.

| initial data | `atol` | RHS evals (review) | max $\delta T/{\rm env}$ (review) |
|---|---|---|---|
| $T=1,T'=0$ | 1e-10 | **6476** (6476) | **9.928e-6** (1.1e-5), at $x=219.06$ |
| exact | 1e-10 | **6401** (6401) | **1.160e-5** (1.1e-5), at $x=219.06$ |
| exact | 1e-13 | **7379** (7379) | **3.275e-7** (3.6e-7), at $x=209.20$ |
| exact | 1e-16 | **7829** (7829) | **1.344e-7** (1.5e-7), at $x=209.20$ |
| $T=1,T'=0$ | **1e-13 (shipped)** | **7403** | **2.534e-6**, at $x=28.22$ |

### The prompt's four acceptance items

1. `atol=1e-10`, production initial data: **9.928e-6**, inside the required $[9\times10^{-6},
   1.3\times10^{-5}]$.
2. `atol=DEFAULT_TK_NUMERIC_ABS_TOLERANCE`: **2.534e-6** ≤ 3e-6. With exact initial data,
   **3.275e-7** ≤ 5e-7 — so the residue at the shipped tolerance is the super-horizon initial
   condition (7.7× larger), not the solver, exactly as review §12.5 says.
3. RHS evaluations: **6476 → 7403, +14.31 %** ≤ 25 % (review: 15 %). Accuracy gain over the same
   step: factor **3.92** (9.928e-6 → 2.534e-6); the review's "factor 30" is the exact-initial-data
   row, where this measurement gives 1.160e-5 → 3.275e-7, factor **35.4**.
4. `GkNumericIntegration` at `atol=1e-10` vs `1e-13` on the same background: max
   $|\Delta G|/{\rm env}$ = **8.538e-10** ≤ 1e-9, at $z=2.577\times10^6$; 12767 → 12812 RHS
   evaluations, **+0.35 %**. `atol` does not bind for $G_k$, so `GkNumericIntegration` keeps the
   shared tolerance and no $G_k$ datastore key moves.

### README §6

Row "$T_k$ numeric $\delta T/{\rm env}$, production initial data": now 1.1e-5, target ≤3e-6, floor
2.5e-6. **Measured 2.534e-6** on the review's geometry — at the target, and at the initial-condition
floor to three figures. See the observation below for the largest wavenumbers.

### Plumbing

`grep -n '"TkNumericIntegration"' main.py` → `:309`, `:311` (warning labels, not `object_get`),
`:702`, `:746`, `:919`, `:1119`, `:2730`. The five `object_get` sites and the tolerance each
carries, printed by the new `tk_numeric_tolerance_sites()`:

```
('TkNumericIntegration', 'Tk_numeric_atol', 'build_Tk_numeric_work: RayWorkPool(query_batch) line 702')
('TkNumericIntegration', 'Tk_numeric_atol', 'build_Tk_numeric_work: direct call, line 745')
('TkNumericIntegration', 'Tk_numeric_atol', 'build_Tk_WKB_work: RayWorkPool(payload_batch) line 919')
('TkNumericIntegration', 'Tk_numeric_atol', 'build_tensor_source_work: RayWorkPool(Tk_lookup_batch) line 1119')
('TkNumericIntegration', 'Tk_numeric_atol', 'build_QuadSourceIntegral_batch: RayWorkPool(Tk_numeric_lookup_batch) line 2730')
unclassified: []
```

The other nine integration sites (`TkWKBIntegration` ×3, `GkNumericIntegration` ×3,
`GkWKBIntegration` ×3 — four direct, five through a queue) all report `'atol'`.

### Tests

- `python -m unittest ComputeTargets.tests.test_tk_numeric_atol` — **8 tests, OK**, 0.28 s.
- `python -m unittest ComputeTargets.tests.test_main_plumbing` — **21 tests, OK** (16 before, 5
  added).
- `python -m unittest discover -s ComputeTargets/tests -t .` — **299 tests, OK**, 149 s. The
  baseline was 286; 8 + 5 = 13 added, none lost.
- `python -m black --check main.py config/defaults.py ComputeTargets/tests/` — clean.

**Not run:** any pipeline execution. `main.py` cannot be imported (README §5 / `CLAUDE.md`), so the
`main.py` half of this change is verified structurally, by `ast`, and not by execution — which is
the reason the guard in `test_main_plumbing.py` exists.

## Observations not acted on

1. **At the top of the production $k$-range, `atol = 1e-13` does not always reach the
   initial-condition floor.** Sweeping the same radiation control over
   $k\in\{10^5,3\times10^5,10^6,3\times10^6,10^7,3\times10^7,10^8,3\times10^8\}$, the shipped
   tolerance gives 2.5e-6 of the envelope everywhere **except $k=3\times10^8$, where it gives
   2.56e-4** — an isolated excursion over samples 340–350, $x\approx9.8$–12.4, whose absolute size
   in $T$ is ~8e-6. It is not a monotone tolerance floor: at $k=3\times10^8$ the *looser* 1e-10
   gives 1.17e-5, and at $k=10^8$ the pattern is inverted (3.15e-4 at 1e-10, 2.49e-6 at 1e-13).
   `atol = 1e-16` gives 2.53e-6 at every $k$ swept, at **fewer** evaluations than 1e-13 at
   $k=3\times10^8$ (7265 against 8483). The mechanism is consistent with the state vector's second
   component: $dT/dz \sim (x^2/A)\,dT/dx$ with $A = k/\sqrt3$, so the same absolute tolerance is a
   $k$-fold looser *relative* tolerance on the derivative, and step selection near $x\sim10$ becomes
   erratic. Not acted on: the prompt fixes the constant at 1e-13 and 1e-16 is a separate decision
   with its own cost; opened as `[12-tk-numeric-atol-largest-k-excursion]` for prompt 13, which
   should re-measure on the real backgrounds rather than the radiation control.
2. **The QCD and LambdaCDM backgrounds were not measured**, only the exact radiation control the
   review and the prompt specify. The production models' Hubble rates are splines, and the
   excursion in item 1 may or may not survive there.
3. **`build_Tk_WKB_work`'s numeric lookup queries with `_do_not_populate: True`** and then reads
   `Tk.stop_T`/`Tk.stop_Tprime`. That is unchanged by this prompt (both candidate dicts set it) and
   production evidently works, so the two scalars are not part of the withheld payload; noted only
   because deviation 1 moved the queue and a reader may wonder whether the flag moved with it. It
   did not.

## State handed to the next prompt

- **The constant.** `config.defaults.DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13`, for the
  `TkNumericIntegration` run alone. `DEFAULT_ABS_TOLERANCE` is unchanged at 1e-10 and still serves
  `TkWKBIntegration`, `GkNumericIntegration` and `GkWKBIntegration`.
- **The `main.py` name.** `Tk_numeric_atol`, a `tolerance` object built in the same `ray.get` as
  `atol`, `rtol`, `quad_atol`, `quad_rtol` (`main.py:2987`, fifth in the tuple); a module-level
  name of the `with ShardedPool(...)` block, read inside `run_pipeline` the way `quad_atol` is.
- **The five sites it reaches**, by line number after this commit: `:684` (`build_Tk_numeric_work`
  existence query), `:755` (the work item), `:895` (`build_Tk_WKB_work`'s initial-condition
  lookup — whose `RayWorkPool` at `:919` now dispatches over `payload_batch`, not `query_batch`),
  `:1102` (`build_tensor_source_work`), `:2713` (`build_QuadSourceIntegral_batch`).
- **A fresh datastore is required for prompt 13.** The tolerance is part of every
  `TkNumericIntegration` row's lookup key, so every such row written before this commit is
  unreachable by the new lookups: a scoped pipeline run on an old datastore will recompute all of
  them, and a run that mixes the two will hold both. This is on top of the regeneration prompts 03
  and 04 already attached (schema and values).
- **Measured, for prompt 13's verification document** (RadiationModel, $k=10^6$, production source
  grid, $T=1,T'=0$, `rtol=1e-8`): $\delta T/{\rm env}$ **9.928e-6 → 2.534e-6**, RHS evaluations
  **6476 → 7403 (+14.3 %)**; with exact initial data **1.160e-5 → 3.275e-7**. $G_k$ is unmoved by
  the same change: **8.538e-10** of the envelope, **+0.35 %** evaluations. Review §12.5's four
  evaluation counts are reproduced exactly at this $k$.
- **The remaining $T_k$ numeric floor is the initial condition**, 2.5e-6 of the envelope
  (`[00-tk-superhorizon-ic-series]`, README §0.4). Nothing in this commit touches $T=1,T'=0$.
- **The structural guard.** `ComputeTargets/tests/test_main_plumbing.tk_numeric_tolerance_sites()`
  returns `(sites, unclassified)` for the whole of `main.py`; `EXPECTED_TK_NUMERIC_SITES = 5` must
  be updated by any prompt that adds or removes a `TkNumericIntegration` lookup.
