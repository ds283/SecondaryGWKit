# Prompt 10 — Supply the transfer-function objects to `QuadSourceIntegral` and settle `QuadSourcePolicy` (A4, wiring)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §0.2 A4; `QI-report.md` QI-6; reconciliation document §3.4 (`QuadSourcePolicy` "nothing
consumes it yet")
**Depends on:** 04 (same `main.py` stage), 08 and 09 (hard). Read `logs/08-…` §6 (the cost ratio)
before starting.
**Recommended model:** Opus
**Files you may touch:** `main.py` (QuadSourceIntegral stage, `build_QuadSourceIntegral_batch` and
its lookup queues; the policy-creation block at `:2620-2650`), `ComputeTargets/QuadSourceIntegral.py`
(only if the payload contract needs a final adjustment), `MetadataConcepts/QuadSourcePolicy.py`
(docstring/comment only), plus the log and the status board.

Read first: `main.py:2200-2520` in full (the QuadSourceIntegral stage: how `GkSourcePolicyData`
and `QuadSource` are looked up in vectorised batches, cached by `store_id`, and packed into
`compute_payload`); `main.py:600-700` (how `TkNumericIntegration`/`TkWKBIntegration` are looked up
per $k$ in the Tk-WKB stage — the pattern to copy); `main.py:2620-2650` (policy objects).

---

## 1. Character of this commit

Pipeline wiring only. After prompt 08, `compute_QuadSource_integral` needs `Tq_numeric`, `Tq_WKB`,
`Tr_numeric`, `Tr_WKB` in its payload; `main.py` does not supply them, so the pipeline is
non-runnable. This prompt supplies them and closes the `QuadSourcePolicy` question.

## 2. Payload plumbing

In the QuadSourceIntegral stage:

1. Collect the set of distinct $q$ and $r$ wavenumbers among the `missing_labels` (post-04, only
   triangle-closing triples). It is at most the 50 `source_k_exit_times`.
2. Look up `TkNumericIntegration` and `TkWKBIntegration` for each, once, with a vectorised
   `object_get` batch exactly as the Tk-WKB stage does for `TkNumericIntegration` at
   `main.py:606-680` (query-only objects: `z_sample=None`, `z_init=None`, the same tags, and
   `_do_not_populate=True` **only where the existing code uses it** — check whether a
   `_do_not_populate` object exposes `values`; `TkWKBIntegration.values` raises if it is set
   (`TkWKBIntegration.py:317-325`), so the objects handed to the payload must be fully populated).
3. Cache by `store_id` and add the four objects to `compute_payload`. Raise with the existing
   `!! MISSING DATA WARNING` style if any is unavailable.
4. Memory: each `TkWKBIntegration` carries ~900 values; 50 of each kind is small compared with the
   `GkSourcePolicyData` objects already shipped per work item. Note the existing TODO at
   `main.py:2418-2421` about putting shared objects in the Ray object store; do not act on it,
   but say in the log whether the added payload makes it more pressing (measure the serialised
   size of one payload if you can do so cheaply).

## 3. `QuadSourcePolicy`

`MetadataConcepts/QuadSourcePolicy.py` is a persisted config object with a `Levin_threshold` and a
`numeric_policy`, created in `main.py:2643-2655` and never read. Decide according to prompt 08's
§6 measurement:

- **Ratio ≤ 3×** (the expected outcome): the total-variation gate inside `adaptive_levin_sincos`
  is doing the job. Leave `QuadSourcePolicy` persisted and threaded (removing it is a schema and
  signature change for no gain), add a docstring to the class saying it is currently unused by the
  integrator and why, and delete the dead `LEVIN_MIN_2PI_CYCLES`/`LEVIN_MIN_PHASE_DIFF` constants in
  `QuadSourceIntegral.py` that prompt 08 left in place.
- **Ratio > 3×**: the orchestrator should already have stopped on prompt 08's §3 issue. If the
  user has decided to reinstate a threshold, wire `QuadSourcePolicy.Levin_threshold` in as the
  minimum total phase variation (in cycles, across a sub-interval, of the fastest group) below
  which the sub-interval is integrated by `scipy.quad` of the *phase-group sum* (via
  `evaluate_sum`) instead of by Levin — never by the old `WKB_quad_integral` of $G$ alone, which is
  incorrect when a $T$ is oscillatory. Pass the policy through the payload. This branch needs a
  test in `test_quadsource_integral.py` showing both routes agree.

Record which branch you took and the ratio that decided it.

## 4. `GkSourcePolicyData.Levin_z`

Prompt 08 stopped using `Levin_z` for control flow. It is still computed and persisted by
`GkSourcePolicyData._classify_Levin` and is harmless. Do **not** remove it (schema); add a comment
at the persistence site saying it is diagnostic only since this commit. `GkSourcePolicy.Levin_threshold`
(the Green's-function one) likewise stays.

## 5. Verification

- `PYTHONPATH=. ./venv/bin/python -c "import ast; ast.parse(open('main.py').read())"`.
- A **dry structural test**: factor the payload-assembly into a small function that takes the
  caches and a `(z_response, k, q, r)` and returns the payload dict, and unit-test it with stub
  objects (`ComputeTargets/tests/` or a new `tests/test_main_plumbing.py` at the repository root —
  say where and why). Assert the four keys are present and refer to the right `store_id`s.
- A live run is prompt 12's job. State plainly that this commit has not been exercised end to end.

## 6. Log and commit

Log to `logs/10-qsi-main-plumbing.md`. Board: row 10, item A4 (3/3); close the "pipeline
non-runnable" §3 issue from prompt 08 (move to §4) — the tree is runnable again, pending 12's
confirmation. One commit; body names the payload keys added and the `QuadSourcePolicy` decision.
