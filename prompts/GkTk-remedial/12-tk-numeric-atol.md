# Prompt 12 — A separate absolute tolerance for the transfer-function numeric run

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §12.5 (the `atol` mis-scaling: $T\sim3/x^2$, $|T|\sim10^{-5}$ deep in the
window, so `atol = 1e-10` is a $10^{-5}$ *relative* tolerance; dropping to `1e-13` buys 30× for
15 % more evaluations; the super-horizon initial condition is then the $2.5\times10^{-6}$ floor),
§12.7 last line ("a one-constant change").
**Design facts:** README §2 (d).
**Depends on:** nothing (independent). Recommended after 11.
**Recommended model:** Opus (the risk is silent plumbing, not numerics)
**Files you may touch:** `config/defaults.py`, `main.py` (**every** `pool.object_get("TkNumericIntegration", …)`
and the tolerance construction at `:2789-2795`), new `ComputeTargets/tests/test_tk_numeric_atol.py`,
`ComputeTargets/tests/test_main_plumbing.py` (add a check), plus the log and the status board.
**Do not touch:** `TkNumericIntegration.py` (the class takes `atol` as given), the initial
condition `T=1, T'=0` (`[00-tk-superhorizon-ic-series]`, out of scope), the Datastore factories.

Read first: README §2 (d), §0.4 (the initial-condition series is *not* done here);
`RECONCILIATION.md` §1 item 14; review §12.5; `ComputeTargets/tests/test_main_plumbing.py`
(`load_main_py_functions`, the `ast` extraction pattern — `main.py` cannot be imported).

---

## 1. What to build

1. `config/defaults.py`: `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` with a comment carrying review
   §12.5's numbers (production $1.1\times10^{-5}$ of the envelope at `atol=1e-10`;
   $3.6\times10^{-7}$ at `1e-13`; $1.5\times10^{-7}$ at `1e-16`; the initial condition then floors
   at $2.5\times10^{-6}$), and why the shared `DEFAULT_ABS_TOLERANCE = 1e-10` is right for $G_k$
   (whose $|G|$ is enormous in these units) and wrong for $T_k$.
2. `main.py`: build one more `tolerance` object, `Tk_numeric_atol`, next to `atol`/`rtol`
   (`:2791-2794`), and pass it as `atol=Tk_numeric_atol` in **every**
   `pool.object_get("TkNumericIntegration", …)` — work-item creation *and* every lookup query
   (`RECONCILIATION.md` §1 item 14: the tolerance is part of the lookup key, so a missed site makes
   lookups miss silently and the pipeline recompute). On `9ff59d5` the sites are near `:601-621`,
   `:745-755`, `:940-960`, `:2536-2545`; **grep, do not trust this list**. `TkWKBIntegration`,
   `GkNumericIntegration`, `GkWKBIntegration` keep `atol`.
3. Add a test in `test_main_plumbing.py` (via `ast`, not import): every `Call` whose first
   positional argument is the string `"TkNumericIntegration"` and whose callee ends in `object_get`
   has a keyword `atol` whose value is the name `Tk_numeric_atol`; every other
   `*Integration` `object_get` has `atol=atol`. This is the guard against the silent-plumbing
   failure.

## 2. Tests (`test_tk_numeric_atol.py`)

Undecorated `numeric_with_phase_cut` with the `TkNumericIntegration` RHS on `RadiationModel`
(prompt 01, or a local copy if Workstream A has not run), production initial data $T=1,T'=0$ at 5
e-folds outside the horizon, production grid and stop window, `rtol=1e-8`:

1. `atol=1e-10`: max envelope-relative error of $T$ against $3(\sin x-x\cos x)/x^3$ between
   $9\times10^{-6}$ and $1.3\times10^{-5}$ (review: $1.1\times10^{-5}$) — pins the baseline.
2. `atol=DEFAULT_TK_NUMERIC_ABS_TOLERANCE`: $\le3\times10^{-6}$ (review: $2.5\times10^{-6}$ with
   the production initial condition; $3.6\times10^{-7}$ with exact initial data — also assert the
   latter with exact initial data, $\le5\times10^{-7}$, to show the remaining floor is the initial
   condition, README §2 (d)).
3. RHS evaluations rise by $\le25\,\%$ (review: 15 %).
4. `GkNumericIntegration`'s RHS on the same background with `atol=1e-10` vs `1e-13`: identical to
   $10^{-9}$ of the envelope (the review's claim that `atol` never binds for $G_k$ — so nothing is
   gained by changing it there).

## 3. Verification and acceptance

- New tests pass; `test_main_plumbing.py` passes; `discover -s ComputeTargets/tests -t .` passes.
- README §6 row "$T_k$ numeric $\delta T/{\rm env}$" met.
- `grep -n '"TkNumericIntegration"' main.py` — every hit is within an `object_get` carrying
  `atol=Tk_numeric_atol` (list them in the log).
- `black --check` clean.

## 4. Log and commit

"State handed to the next prompt": the constant's value; the list of `main.py` sites changed
with line numbers; the measured errors and evaluation counts. Note for prompt 13: a fresh
datastore is required (the tolerance is part of every `TkNumericIntegration` row's key).

Commit subject, or something equally specific: `Give the transfer-function numeric run its own absolute tolerance`.
