# Log 16 — The `has_unresolved_osc` print policy: a per-$k$ summary

**Prompt:** prompts/GkTk-remedial/16-unresolved-osc-print-policy.md
**Commit:** *(this commit)* — Summarise unresolved-oscillation warnings per wavenumber
**Model:** Claude Opus 5
**Date:** 2026-09-12
**Result:** COMPLETE WITH DEVIATIONS

**The warning was relocated, not deleted** (README §2 (h), §7 D2 option (ii), taken by the user
2026-09-11). `scan_sample_grid_for_unresolved_osc` still runs prompt 11's test, unchanged to the
bit; the two `print` calls are still in the file and still reachable, now behind a keyword-only
`warn: bool = True`; the three payload keys, the two object factories and the persisted columns
are untouched. **A datastore written before this prompt is readable after it, and vice versa: the
only thing that moved is where the information is printed.**

## What shipped

### `Quadrature/integrators/numeric_with_phase_cut.py`

- `scan_sample_grid_for_unresolved_osc(model, k, k_float, omega_sq, sampled_z, object_label, *,
  warn: bool = True) -> dict` — the new parameter is **keyword-only** and gates **only** the two
  `print` calls (now inside `if warn:` at `:95`). The loop, the failing-pair
  rule, the early `return` on the first violation and all three returned values are byte-for-byte
  prompt 11's. The docstring gains a `:param warn:` block saying why the production integrators
  opt out and why the default stays `True`.
- `numeric_with_phase_cut(..., object_label="(object)", warn_unresolved_osc: bool = True)` — the
  new parameter is **appended at the end** of the signature (see deviation 2) and is passed
  through as `warn=warn_unresolved_osc` at the single call site (`:350`). A three-line comment
  above that call says the gate is on the print and never on the test or its result.

### `ComputeTargets/GkNumericIntegration.py`, `ComputeTargets/TkNumericIntegration.py`

- Each `numeric_with_phase_cut.remote(...)` call gains `warn_unresolved_osc=False` with a
  six-line comment recording that the *warning* is suppressed, not the test, and that `main.py`
  now summarises. Nothing else in either file changed: the `omega_sq=` argument, the `atol`/`rtol`
  lines (prompt 12's landing site) and the payload read-back are untouched.

### `main.py` — two new module-level functions and the wiring

Both functions are **module level** (`record_unresolved_osc` at `:337`, `format_unresolved_osc_summary` at `:408`, between
`build_QuadSourceIntegral_payload` and `run_pipeline` at `:458`) and the formatter **returns** its lines rather than printing them, because
`ComputeTargets/tests/test_main_plumbing.load_main_py_functions` extracts module-level
`ast.FunctionDef` nodes only:

```python
def record_unresolved_osc(summary: dict, obj) -> None
def format_unresolved_osc_summary(summary: dict, sector_label: str) -> List[str]
```

`record_unresolved_osc` keys `summary` by `float(obj.k.k_inv_Mpc)` and holds, per wavenumber,
`{"recorded", "flagged", "unpopulated", "efolds_min", "efolds_max"}`. Every accessor is guarded:
`obj is None`, a missing/unfloatable `k`, the `RuntimeError` that `has_unresolved_osc` and
`unresolved_efolds_subh` raise before population (`TkNumericIntegration.py:235-238` and the
`GkNumericIntegration` twin), and a `None` flag (what the payload carries when `omega_sq` was not
supplied) are all counted or ignored rather than allowed to kill the pipeline in a reporting path.

`format_unresolved_osc_summary` returns a header line, one line per wavenumber that flagged at
least once — the wavenumber, how many objects flagged out of how many were recorded, and the range
of `unresolved_efolds_subh` over them — and a closing totals line; plus a trailing line if any
object arrived unpopulated. Wavenumbers are listed in ascending $k$; a wavenumber that recorded
objects but never flagged is counted in the totals and not listed.

Wiring, in both `if args.{Tk,Gk}_numeric_queue:` blocks (`main.py:791-815` and `:1378-1402`): an
accumulator dict created before the queue, `post_handler=lambda obj: record_unresolved_osc(<acc>,
obj)` on the `RayWorkPool` construction, and `for line in format_unresolved_osc_summary(<acc>,
<sector label>): print(line)` after `.run()`. `post_handler` returning `None` is safe under
`store_results=False` (`RayTools/RayWorkPool.py:361-363, :398-400, :417-419, :489-491, :513-515`
each guard on `replacement_obj is not None and self._store_results`). Sector labels are
`"tensor Green's functions, numerical part"` and `"matter transfer functions, numerical part"`.

**`main.py`'s `delta_logz=` arguments, its `0.85` truncation constants and its tolerance plumbing
are untouched** — `git diff main.py | grep -E "^[-+].*(delta_logz|0\.85|atol|rtol)"` returns
nothing.

### Tests

`ComputeTargets/tests/test_numeric_phase_cut.py` — new `TestUnresolvedOscillationWarningIsGated`
(4 tests), extending prompt 11's module without touching any of its existing tests or constants.
`ComputeTargets/tests/test_main_plumbing.py` — new `UnresolvedOscSummaryTestCase` (6 tests) and
`UnresolvedOscWiringTestCase` (3 tests, the `ast` guard), plus `StubNumericObject`, `"List": List`
in `load_main_py_functions`'s namespace (the `-> List[str]` annotation is evaluated at function
creation), and the module-level `NUMERIC_QUEUE_TITLES`.

## Deviations from the prompt

### 1. `format_unresolved_osc_summary` returns `[]` only for an *empty* accumulator — IMPLEMENTATION CHOICE

The prompt's suggested snippet carries the docstring *"Returns [] when nothing flagged"*, but its
own §2.2 prose says *"When nothing flagged, say so in one line rather than printing an empty
block; a run that prints nothing at all is indistinguishable from a run where the wiring was
dropped."* The two cannot both hold. I followed the prose: `[]` is returned only when the
accumulator is empty — the queue did not run, so there is nothing to report and no block to print
— and an accumulator holding objects, none of which flagged, produces the two-line all-clear form.
The signature is exactly as suggested. Alternative considered and rejected: return `[]` for
"nothing flagged" as the docstring says, which would reinstate precisely the silent failure §2.2
names as this prompt's main exposure.

### 2. `warn_unresolved_osc` is appended at the end of `numeric_with_phase_cut`'s signature — IMPLEMENTATION CHOICE

The prompt does not say where. Placing it next to `omega_sq`, where it belongs by subject, would
shift the positional index of `atol`, `rtol`, `delta_logz` and `mode`. Every caller in the tree
passes those by keyword, but the four reproduction scripts under
`docs/gk-wkb-review-fable-2026-09-09/` pass the first seven arguments **positionally**
(`run(proxy, kmock, redshift(...), zsamp, 0.0, 1.0, GN.RHS, atol=..., ...)`), so appending after
`object_label` is the placement with no way to break them. The parameter is a plain
positional-or-keyword with a `True` default; the scan's `warn`, which the prompt does specify, is
keyword-only.

### 3. `"List": List` added to `load_main_py_functions`'s namespace — IMPLEMENTATION CHOICE

The prompt's suggested formatter signature annotates `-> List[str]`, and annotations are evaluated
when the function object is created, so `List` must exist in the namespace the extracted functions
are `exec`'d in. The alternative was to annotate `-> list` and leave the loader alone; I kept the
prompt's signature verbatim and added the one name, which is the same pattern the loader already
uses for `redshift`, `wavenumber_exit_time` and the six other annotation names.

### 4. A `None` flag is counted as unpopulated — IMPLEMENTATION CHOICE

The prompt says to guard the `RuntimeError`. `has_unresolved_osc` can also legitimately be `None`
without raising: `numeric_with_phase_cut` returns `None` for all three fields when no `omega_sq`
was supplied, and the integrators' `_has_unresolved_osc` is then `None`, which makes the property
*raise*. The remaining `None`-without-raise route is a stand-in or a future object that sets the
attribute directly. Treating it as unpopulated rather than as "did not flag" keeps the totals
honest; treating it as `False` would quietly inflate the all-clear count.

### 5. Four tests in `test_numeric_phase_cut.py` where §3 names three — IMPLEMENTATION CHOICE

§3 item 3 is *"the integrators opt out"*, and the test it describes — a production-geometry `Gk`
run that prints nothing and still flags — demonstrates the *parameter*, not the *call site*, since
the test must supply the argument itself. I added
`test_both_integrators_pass_warn_unresolved_osc_False`, which reads both integration modules with
`ast` and asserts the literal `False` reaches `warn_unresolved_osc` in each
`numeric_with_phase_cut.remote(...)` call. Neither module can be imported without Ray, so `ast` is
the available instrument; this is the same device §3 item 5 asks for on `main.py`.

## Verification performed

All run from the worktree root.

**1. The full `ComputeTargets` suite passes, with no test lost.**

```
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
Ran 286 tests in 171.765s
OK
```

273 before (prompt 11's figure) + 13 new = 286. `LiouvilleGreen/tests` also passes untouched
(`Ran 141 tests in 1803.673s / OK`), as expected — nothing in this commit is under
`LiouvilleGreen/`. No pre-existing test was modified: prompt 11's
`TestBitIdentity`, `TestUnresolvedOscillationFlag`, `TestD2FireRate`, `TestCost`, `TestModeGuard`
and `TestPhaseSteppedExtremumSearch` all pass unchanged, including the bit-identity constants and
the RHS-evaluation counts.

**2. The flag does not depend on the warning.** `test_flag_does_not_depend_on_the_warning` asserts
equality of all three returned keys between `warn_unresolved_osc=True` and `False` on prompt 11's
coarse grid, *and* equality of the whole dict returned by `scan_sample_grid_for_unresolved_osc`
called directly both ways. Both pass.

**3. The print is gated both ways.** `test_the_print_is_gated_both_ways`: with the warning on,
stdout is exactly **2 lines** (the header and the detail line — the warning has always been two
prints, prompt 11's observation 1) and contains `may have developed unresolved oscillations` once;
with it off, stdout is the empty string. `has_unresolved_osc is True` in both cases.

**4. The integrators opt out.** `test_production_geometry_Gk_run_is_silent_when_opted_out`: a
LambdaCDM $k=10^7$ run on the production source/response geometry prints **nothing** and returns
`has_unresolved_osc is True` with both companion fields populated.
`test_both_integrators_pass_warn_unresolved_osc_False` confirms each integration module's single
`numeric_with_phase_cut.remote(...)` call carries the literal `False`.

**5. The summary functions.** Six tests on stand-ins: two wavenumbers with a mix of flagged and
unflagged objects give the right counts (`2 of 3`, `1 of 1`), the right e-folds range (`3.5 to
4.25`) and the right totals (`3 of 4 objects flagged, over 2 of 2 wavenumbers`); a wavenumber that
never flagged is counted but not listed; an all-clear accumulator gives the one-line form; an empty
accumulator gives `[]`; an object whose flag is unpopulated **does not raise**, is counted
separately and is reported on its own trailing line; a flagged object with no e-folds value still
reports.

**6. The `ast` wiring guard bites.** With the guard passing on the shipped tree, I deleted the
`post_handler=` line from the `Gk` queue and re-ran `UnresolvedOscWiringTestCase`:

```
AssertionError: 'post_handler' not found in {...} : CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS
FAILED (failures=1)
```

and restored the file (`git diff` identical afterwards; the class passes again). This is the
silent failure the guard exists for.

**7. The prompt's four file-level checks.**

- `grep -n "print" Quadrature/integrators/numeric_with_phase_cut.py` → the two warning prints at
  `:96` and `:99`, inside the `if warn:` at `:95`, still present and reachable (the default is `True`; the
  gating test exercises them).
- `git diff --stat` touches exactly six files — `main.py`,
  `Quadrature/integrators/numeric_with_phase_cut.py`, the two integrators and the two test
  modules. **No `Datastore/`.**
- `git diff main.py` is the two module-level functions, the two accumulators, the two
  `post_handler=` keywords and the two summary loops. `grep -E "^[-+].*(delta_logz|0\.85|atol|rtol)"`
  over that diff returns nothing.
- `black --check` clean on all six files.

**8. The summary block a production run would now print.** `format_unresolved_osc_summary`, fed
from the test's `StubNumericObject`s, shaped as production: ~65,000 `GkNumericIntegration` objects
over 50 wavenumbers is ~1,300 per $k$, every one flagging (prompt 11's 2,149 of 2,149), with
first-trip depths $\ln x$ for $x = 26.5$–$66.6$ (prompt 11's measured range):

```
-- UNRESOLVED-OSCILLATION SUMMARY | tensor Green's functions, numerical part
|  k = 1e+05/Mpc: 1300 of 1300 objects flagged | e-folds inside horizon at first unresolved sample: 3.28 to 4.2
|  TOTAL: 1300 of 1300 objects flagged, over 1 of 1 wavenumbers
```

That is **3 lines in place of 2,600** for this wavenumber ($1{,}300 \times 2$ prints), and one
block per sector rather than per object: ~52 lines per model instead of ~$1.3\times10^5$. The
transfer-function queue, whose objects do not flag (0 of 6 measured), prints:

```
-- UNRESOLVED-OSCILLATION SUMMARY | matter transfer functions, numerical part
|  no object reported unresolved oscillations (50 objects over 1 wavenumbers)
```

The real block carries one line per flagging $k$, so a full model prints a header, up to 50
wavenumber lines and a total.

## Observations not acted on

1. **`NumericIntegrationSupervisor.report_wavelength` still has no caller.** Prompt 11 kept it,
   corrected, noting that under option (iii) it is where the per-step form goes back and that
   under option (ii) "it should probably go", and did not open an issue because
   `[00-unresolved-osc-print-policy]` owned the decision. Option (ii) has now landed and that
   issue is closed, so the retention rationale is spent — but
   `Quadrature/supervisors/numeric.py` is **not** in this prompt's file list, and the prompt fixes
   the `docs/OPEN_ISSUES.md` count at 51 → 50, which opening a new §3 row would contradict. It is
   therefore recorded in the §4 entry on the board, where a reader of the closed issue meets it,
   rather than lost to this log: a one-line deletion for prompt 13's clean-up or a later tidy.
2. **The all-clear line reads "1 wavenumbers" when exactly one $k$ recorded objects.** Cosmetic;
   production has 50 $k$ per model, so it is a stand-in artefact. Not worth a plural helper, but a
   later editor who dislikes it should know it was seen.
3. **The summary is per queue, not per model.** Both accumulators live inside `run_pipeline`'s
   `if args.*_numeric_queue:` blocks and cover every wavenumber that queue processed, which is one
   model per invocation. If the pipeline ever runs several models in one process, the sector label
   would need the model name.
4. **Objects already in the datastore are recorded too.** `post_handler` fires on the "available"
   exit path as well as the compute path (`RayWorkPool.py:417-419, :489-491, :513-515`), so a
   re-run that recomputes nothing still prints a full summary from the stored flags. That is the
   wanted behaviour — the flag describes the grid, not the run — but it means the counts are
   "objects seen", not "objects computed".
5. **The scan's early return still reports only the first failing pair.** Unchanged from prompt 11
   by design; the summary's e-folds range is therefore a range over *first* failures, which is
   what §2.2 asks for.

## State handed to the next prompt

- **New parameter names and defaults.**
  `scan_sample_grid_for_unresolved_osc(model, k, k_float, omega_sq, sampled_z, object_label, *,
  warn: bool = True)` — keyword-only, gates the two prints only.
  `numeric_with_phase_cut(..., object_label="(object)", warn_unresolved_osc: bool = True)` —
  appended last in the signature, so no positional index moved; passed through as
  `warn=warn_unresolved_osc`. Both production integrators pass `warn_unresolved_osc=False`; every
  other caller (tests, `docs/` scripts) keeps the warning by default.

- **The two `main.py` module-level functions**, extractable by
  `test_main_plumbing.load_main_py_functions`:

  ```python
  def record_unresolved_osc(summary: dict, obj) -> None
  def format_unresolved_osc_summary(summary: dict, sector_label: str) -> List[str]
  ```

  `summary` is keyed by `float(obj.k.k_inv_Mpc)`; each value is
  `{"recorded", "flagged", "unpopulated", "efolds_min", "efolds_max"}`. The formatter returns
  lines; the caller prints them. `[]` means the accumulator was empty. `load_main_py_functions`'s
  namespace now also supplies `List`.

- **The summary format** (header, one line per flagging $k$ in ascending $k$, a totals line, and a
  trailing unpopulated line when needed):

  ```
  -- UNRESOLVED-OSCILLATION SUMMARY | <sector label>
  |  k = 1e+05/Mpc: 1300 of 1300 objects flagged | e-folds inside horizon at first unresolved sample: 3.28 to 4.2
  |  TOTAL: 1300 of 1300 objects flagged, over 1 of 1 wavenumbers
  ```

  all-clear form:

  ```
  -- UNRESOLVED-OSCILLATION SUMMARY | <sector label>
  |  no object reported unresolved oscillations (50 objects over 1 wavenumbers)
  ```

  Sector labels in production: `"tensor Green's functions, numerical part"`,
  `"matter transfer functions, numerical part"`.

- **Nothing stored changed.** Payload keys, the two Datastore factories and the persisted columns
  are untouched; a datastore written before this commit is readable after it and vice versa.
  Prompt 12's landing site — the `atol=self._atol.tol` line in each integrator's
  `numeric_with_phase_cut.remote(...)` call, and `config/defaults.py` — is untouched; the only
  line added to those calls is `warn_unresolved_osc=False` just above `**payload`.

- **Test counts.** `ComputeTargets/tests` is at **286** tests (273 at prompt 11 + 13). A prompt
  that does not touch the numeric integrators or `main.py`'s numeric queues must leave all of them
  passing; a prompt that edits either numeric `RayWorkPool` construction must keep its
  `post_handler` and the formatter call, or `UnresolvedOscWiringTestCase` fails.
