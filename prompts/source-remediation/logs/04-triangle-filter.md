# Log 04 — Schedule only triangle-closing (k,q,r) triples (A5)

**Prompt:** prompts/source-remediation/04-triangle-filter.md
**Commit:** (this commit) — Filter QuadSourceIntegral work items to triangle-closing triples
**Model:** Claude Sonnet 5
**Date:** 2026-09-09
**Result:** COMPLETE

## What shipped

- **A5** `main.py:246-256` (new) — added a module-level helper

  ```python
  def closes_triangle(
      k: wavenumber_exit_time, q: wavenumber_exit_time, r: wavenumber_exit_time
  ) -> bool:
      k_val = k.k.k
      q_val = q.k.k
      r_val = r.k.k

      tol = DEFAULT_FLOAT_PRECISION * max(k_val, q_val, r_val)
      return (abs(q_val - r_val) - tol) <= k_val <= (q_val + r_val + tol)
  ```

  placed immediately before `def run_pipeline(`, since `run_pipeline` is the only top-level
  function in `main.py` and every other helper is nested inside it — there is no pre-existing
  top-level "helpers" section to join. `.k.k` reaches the physical (dimensionful) wavenumber
  through `wavenumber_exit_time.k` (a `wavenumber`) `.k` (`k_inv_Mpc / units.Mpc`), the same
  quantity `|q-r| <= k <= q+r` must hold for. The tolerance is relative, scaled by
  `max(k, q, r)`, so a triple sitting exactly on `k = q+r` or `k = |q-r|` is kept rather than
  dropped by floating-point rounding, per the prompt.

- **A5** `main.py:2519-2528` (was `main.py:2488-2492` in the prompt's line numbers, and
  `main.py:2496-2501` before this commit's own `closes_triangle` insertion shifted it a further
  14 lines) — after building
  `qsi_work_items` as `(z, k, q, r)` tuples, added:

  ```python
  num_before_triangle_filter = len(qsi_work_items)
  qsi_work_items = [
      (z, k, q, r) for z, k, q, r in qsi_work_items if closes_triangle(k, q, r)
  ]
  num_after_triangle_filter = len(qsi_work_items)
  print(
      f"   @@ QuadSourceIntegral triangle filter: kept {num_after_triangle_filter} of "
      f"{num_before_triangle_filter} (k,q,r) triples "
      f"({100.0 * num_after_triangle_filter / num_before_triangle_filter:.2f}%)"
  )
  ```

  This is the only change to the work-item construction: the `(q,r)` pair grid, the response
  grid, and the `QuadSource`-stage `combinations_with_replacement` (`main.py:961`, untouched) are
  unchanged.

## Deviations from the prompt

None.

## Verification performed

- Ran the file-parses check the prompt specifies:
  `PYTHONPATH=. ./venv/bin/python -c "import ast,sys; ast.parse(open('main.py').read())"` — passed
  (`PARSE_OK`).
- Reproduced `QI_04_triangle.py`'s arithmetic with `closes_triangle`'s exact predicate (same
  50-point log-spaced source/response grids, `1e5` to `3e8`), in a standalone script (not
  committed; scratch only) that inlines the same `k_val/q_val/r_val` and
  `tol = DEFAULT_FLOAT_PRECISION * max(...)` logic as the committed function:

  ```
  n source k = 50  n response k = 50
  combinations_with_replacement pairs = 1275
  (k,q,r) triples = 63750
  triples satisfying closes_triangle : 5133 (8.05%)
  ```

  This matches the audit's and the original `QI_04_triangle.py`'s figures exactly: 63,750 triples
  before the filter, 5,133 (8.05%) after. Also spot-checked the boundary case `q = r = 5.0`,
  `k = q + r = 10.0` (an exact degenerate triangle) is kept by `closes_triangle`, confirming the
  tolerance does not drop boundary points.
- Did not run `main.py` itself (needs Ray and a datastore; not required by the prompt).

## Observations not acted on

- The prompt's cited line numbers (`main.py:2488-2492`) are `main.py:2496-2501` on this branch's
  `main.py` (prior commits in this campaign, e.g. prompt 02's `main.py` edits and earlier history,
  shifted line numbers by a few lines). This is a citation drift, not a structural mismatch — the
  code shape (`itertools.product(...)` then a list comprehension re-ordering to `(z, k, q, r)`) is
  exactly as the prompt describes, so no deviation tag is warranted; noted here only for a later
  reader cross-referencing line numbers.

## State handed to the next prompt

- `closes_triangle(k, q, r)` is a public, module-level function in `main.py` (not nested inside
  `run_pipeline`), taking three `wavenumber_exit_time` objects. Prompt 10, which also edits the
  `QuadSourceIntegral` stage of `main.py`, should reuse it rather than re-deriving the predicate,
  and should keep the filter applied before any further payload plumbing it adds to
  `qsi_work_items`.
- The diagnostic print uses the `"   @@ "` prefix style already used elsewhere in `main.py`
  (e.g. `main.py:314`, `422`) for the same kind of one-shot summary line.
