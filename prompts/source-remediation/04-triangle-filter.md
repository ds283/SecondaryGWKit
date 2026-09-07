# Prompt 04 — Schedule only triangle-closing $(k,q,r)$ triples (A5)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §0.2 A5; `QI-report.md` QI-12; `docs/resonance-scaffolding/sigw-resonance-reconciliation.md` §0.2 F3
**Depends on:** nothing. Must land **before** prompt 10 (same `main.py` stage).
**Recommended model:** Sonnet
**Files you may touch:** `main.py` (the QuadSourceIntegral work-item construction only), plus the
log and the status board.

---

## Character of this commit

A filter on the work-item list and a diagnostic printout. No compute target changes, no schema
change. It recovers ~92 % of the scheduled `QuadSourceIntegral` work.

## The defect

`main.py:2488-2492`:

```python
    qsi_work_items = itertools.product(
        z_source_integral_response_sample,
        itertools.combinations_with_replacement(source_k_exit_times, 2),
        response_k_exit_times,
    )
    qsi_work_items = [(z, k, q, r) for z, (q, r), k in qsi_work_items]
```

Nothing enforces $|q-r|\le k\le q+r$. On the shipped 50-point grid that is 63,750 $(k,q,r)$
triples of which 5,133 (8.05 %) close a triangle (`docs/spec-code-audit/scripts/QI_04_triangle.py`
re-derives this). A non-triangle has no $\theta\in[0,\pi]$ realising it and is not a point of the
spec 03 §0.3 integrand at all; computing it is pure waste.

## What to do

1. Add the filter when building `qsi_work_items`: keep $(k,q,r)$ only if
   $|q-r| \le k \le q+r$, comparing `.k.k` (physical wavenumbers, all in the same units). Use a
   relative tolerance of `DEFAULT_FLOAT_PRECISION` at the boundaries so grid points that sit
   exactly on $k=q+r$ or $k=|q-r|$ are **kept**, not dropped (they are legitimate degenerate
   triangles and future $(s,d)$ grids will place nodes there deliberately).
2. Put the predicate in a small named function (e.g. `closes_triangle(k, q, r)`) near the other
   helpers at the top of `main.py`, with a one-line docstring citing spec 03 §0.3.
3. Print, once, after the filter: the number of triples before and after and the percentage kept,
   in the same style as the surrounding `print` diagnostics.
4. Do **not** change the $(q,r)$ pair grid, the response grid, or anything about `QuadSource`
   scheduling (the `combinations_with_replacement` in the QuadSource stage is correct: every
   unordered pair is needed for *some* $k$).

## Verification

- Run `QI_04_triangle.py` and confirm your filter's count (5,133 on the shipped grid) matches its
  arithmetic, including the boundary handling — modify a copy of the script to call your
  `closes_triangle` if that is the cleanest way.
- `PYTHONPATH=. ./venv/bin/python -c "import ast,sys; ast.parse(open('main.py').read())"` — the
  file still parses; a full `main.py` run needs Ray and is not required here.

## Log and commit

Log to `logs/04-triangle-filter.md`. Board: row 04, item A5. One commit; body states the before/after
counts.
