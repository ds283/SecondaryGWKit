# Prompt 06 — Restrict `QuadSource` to the both-numeric region (A3, A2 part 2)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §0.2 A2, A3; `QS-report.md` QS-5, QS-6 (with the reproduction at
`docs/spec-code-audit/scripts/QS_04_coverage_and_jacobian.py`), QS-9
**Depends on:** 05 (uses the same hand-over definition). Touches the QuadSource stage of `main.py`;
prompt 04 touches a different stage, prompt 10 the QuadSourceIntegral stage.
**Recommended model:** Opus
**Files you may touch:** `ComputeTargets/QuadSource.py`, `main.py` (QuadSource stage,
`build_tensor_source_work` and neighbours only), new `ComputeTargets/tests/test_quadsource.py`,
plus the log and the status board. **No change** to
`Datastore/SQL/ObjectFactories/QuadSource.py` — the schema stays; rows just get fewer.

Read first: `QuadSource.py` in full (477 lines); `main.py:480-560` (how `TkNumericIntegration` is
scheduled with a truncated grid and `mode="stop"`) and `main.py:860-1000` (how `QuadSource` is
scheduled); QS-report §2 QS-5 and QS-6.

---

## 1. Character of this commit

`QuadSource` currently tries to be *the* source term $f(z'\mid q,r)$ over the whole source grid:
it samples $f$ at every $z'$ and splines the samples. Two things are wrong with that (audit A2,
A3): sub-horizon the samples are of an oscillation the spline cannot represent, and the
`TkNumericIntegration` values it reads stop 3–6 e-folds inside the horizon, so the loop runs off
the end of the transfer-function grid and raises.

After this commit `QuadSource` is *the smooth part of the source term*: it is defined on exactly
the region where **both** $T_q$ and $T_r$ are in their numeric representation (plus the
super-horizon default above it), and it is correct there. Everything below that — where at least
one factor is oscillatory — is handled by the phase-group machinery of prompts 07–08, which
consumes `TkSourceFunctions` (prompt 05) directly and never reads a sampled $f$.

The stored columns do not change. The `values` list simply ends at the hand-over redshift of the
*larger* of $q, r$ (the one that enters the horizon first).

## 2. What to do

### 2.1 Truncate the grid in `compute_quad_source`

`QuadSource.py:55-190`. Currently `z_sample` is the full `z_source_sample` passed from `main.py:910`.

- Compute the hand-over redshift for each factor exactly as prompt 05 does:
  $z^{\rm X}_q = q.z_{\rm exit} - $ `Tq.stop_deltaz_subh` (and the same for $r$). Both `Tq` and
  `Tr` are `TkNumericIntegration` objects and carry `stop_deltaz_subh`; confirm the attribute is
  populated on a stored object (`TkNumericIntegration.py:263-268`, and the factory's read path).
- Define the both-numeric region as $z' \ge \max(z^{\rm X}_q, z^{\rm X}_r)$. Walk only the part of
  `z_sample` in that region.
- Keep the existing super-horizon default ($T=1$, $T'=0$) for a factor whose numeric grid has not
  started yet (`:116-121`), which is correct and is exactly the "leading mismatch" the current
  code tolerates. Replace the index-walking alignment (`:82-102`) with a lookup by `store_id`
  (a dict from `store_id` to value) so that a missing sample is detected explicitly and raised with
  a message naming the factor, rather than by `IndexError`. **Keep the `RuntimeError` for an
  interior gap**: the grids are supposed to be nested, and a hole is a pipeline error, not
  something to interpolate over.
- Record on the object the two hand-over redshifts and the region actually used. They need not
  be persisted (they are recomputable from the `Tk` objects), but expose them as properties
  (`crossover_z_q`, `crossover_z_r`, `numeric_region`) so prompt 08 can assert consistency with
  its `TkSourceFunctions`. If you find the object cannot recover them after a datastore round-trip
  without the `Tk` objects, expose them as `None` in that case and document it — prompt 08 passes
  the `Tk` objects anyway.

### 2.2 `_create_functions`

`QuadSource.py:294-309`. The spline now covers only the both-numeric region, which is the only
place a spline of $f$ is valid (audit QS-5 table: the region ends 3–6 e-folds sub-horizon, i.e.
at most ~40 cycles of the *faster* factor; measure the residual error there in §3).
`QuadSourceFunctions` gains the region bounds so a consumer can refuse to evaluate outside them.
Keep `ZSplineWrapper` (its out-of-range behaviour is what `QuadSourceIntegral` relies on) with the
label from prompt 02.

### 2.3 `main.py` — QuadSource stage

`main.py:900-1000`. `z_sample=z_source_sample` is still the right thing to pass (the object
truncates internally) — but make sure the `SourceZGridSizeTag` etc. remain meaningful, and add a
one-line comment at the `object_get("QuadSource", ...)` site explaining that the stored values end
at the both-numeric hand-over. Nothing else in `main.py` changes in this prompt.

### 2.4 Datastore round-trip

`Datastore/SQL/ObjectFactories/QuadSource.py` reconstructs `z_sample` from the stored values (check
how — the `TkNumericIntegration` factory does `len(values)`; QS-6 cites its lines 336-346, 420).
Confirm a `QuadSource` with a truncated value list reads back with a consistent `z_sample` and that
`available` is `True`. You may not edit the factory; if it cannot cope, stop and open a §3 issue.

## 3. Tests — `ComputeTargets/tests/test_quadsource.py`

Use mock `Tk` objects shaped like the real ones (the audit's `QS_04_coverage_and_jacobian.py`
already builds these — reuse the shape, not the file) and the analytic constant-$w$ transfer
function for values:

1. **A3 regression.** Full source grid, `Tq`/`Tr` grids truncated at both ends with
   `stop_deltaz_subh` set: `compute_quad_source` completes, the value list ends at
   $\max(z^{\rm X}_q, z^{\rm X}_r)$, and the first samples use the $T=1$ default. The same inputs
   raise `IndexError` on the pre-commit code (state this in the log by running the old code once
   from `git stash`/`git show`).
2. **Interior gap** in one `Tk` grid raises `RuntimeError` naming the factor.
3. **Kernel unchanged.** `source_function` is untouched; assert on a handful of points that the
   stored `source` equals the audit's independent transcription of spec 03 R22 (copy the few
   lines from `QS_01_sympy_f.py`/`QS_02_deriv_and_f.py` into the test) to $10^{-14}$.
4. **Spline residual inside the region.** Build the exact analytic radiation source for
   $q = r$ on the production grid density over the both-numeric region ending at
   `z_exit_subh_e6·0.85` (the latest possible hand-over), spline it as `_create_functions` does, and
   report (not assert) the max error relative to the local envelope. Put the number in the log and
   in `IMPLEMENTATION_STATE.md` §3 as an open observation: it bounds the accuracy of the
   all-smooth region of the source integral, and if it is worse than ~$10^{-3}$ the hand-over
   window in `main.py:505-507` (or the grid density) is the knob to turn — **not** in this prompt.

## 4. Verification

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` passes.
- `QS_04_coverage_and_jacobian.py` part (a) no longer reproduces the `IndexError` when pointed at
  the new function (adapt a copy if its mocks need the new attributes).

## 5. Log and commit

Log to `logs/06-quadsource-regions.md`. **State handed to the next prompt:** the property names
for the hand-over redshifts and region, the `QuadSourceFunctions` fields, and the measured spline
residual from §3.4. Board: row 06, items A3 and A2 (2/3); §5 note that existing `QuadSource` rows
are stale. One commit; body says what region the object now covers, why, and that the schema is
unchanged.
