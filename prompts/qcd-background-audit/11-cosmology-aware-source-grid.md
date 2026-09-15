# Prompt 11 — A source grid that knows the cosmology it samples

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Implements:** audit §8 recommendation **5**, its mechanical half · **Measurements:** audit §7
**Depends on:** 07 (the 3-point set), 09 (a verified base), 10 (the consumer that must resolve the
step). **Gated: README §7 D7.**
**Recommended model:** **Opus** — the change is small and its datastore consequences are not.

**Files you may touch:** `CosmologyConcepts/wavenumber.py` (`populate_z_sample`),
`CosmologyConcepts/redshift.py` (`winnow`), `main.py` (**the grid-construction and tag hunks at
`:520-590` only**), `ComputeTargets/tests/test_main_plumbing.py`, a new
`CosmologyConcepts/tests/` module or `ComputeTargets/tests/test_source_grid.py`, plus this
campaign's log and board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `CosmologyModels/`; any compute target; any `Datastore/` factory other than
through the tag labels `main.py` builds; the Bessel or Levin stages of `main.py`; the number of
samples per decade, which is prompt 12's question and not this one's.

**Read first:** audit §7 in full; `CosmologyConcepts/wavenumber.py:250-293` (`populate_z_sample` —
note that **the cosmology is never consulted**); `CosmologyConcepts/redshift.py:226` (`winnow`, a
blind stride `[::-n]`); `main.py:520-590` (the universal grid, the winnow to the response grid, and
the nine tags); `ComputeTargets/tests/test_main_plumbing.py` (the `ast` pattern for asserting
`main.py`'s structure without importing it — `main.py` **cannot be imported**, `CLAUDE.md`);
`ComputeTargets/tests/wkb_reference.py`'s `production_source_grid`.

---

## 1. What is wrong

`populate_z_sample` returns `logspace(log10(z_init), log10(z_end), num=...)`, where `z_init` comes
from the wavenumber's horizon-exit time and the density from a command-line number. **The cosmology
is never consulted.** `main.py:532` builds one universal grid from the earliest-exiting $k$ and the
response grid is a blind decimation of it.

So the grid lands on the features that matter only by luck:

- the **three genuine break points** — and after prompt 06 the background has a **step** at each,
  which a consumer's cubic will interpolate straight across unless there is a sample tight on
  either side;
- **matter–radiation equality** ($z=3407$) and **matter–$\Lambda$ equality** ($z=0.303$), which the
  model computes in its own constructor and then discards;
- each $k$'s **horizon exit**, and the numeric→WKB **hand-over**.

Two mechanical consequences the audit names:

- **`winnow` is a blind stride `[::-n]`**, so any protected point would have to survive decimation
  for the response grid to resolve the same feature;
- **the grid tags are `SourceRedshiftGrid_{len}`** (`main.py:566`), which labels *size only*, so two
  different grids of equal length **collide in the datastore** — a real hazard the moment the grid
  stops being a pure function of `(z_init, z_end, samples_per_log10z)`.

## 2. The change

1. **A protected set, declared by the cosmology and injected into the grid.** Extend
   `populate_z_sample` with a keyword-only parameter carrying the points that must appear — and
   have `main.py` fill it from `_cosmology_break_points(cosmology, z_end, z_init)` plus the two
   equality redshifts the model already computes. **Do not import an equation-of-state module into
   `CosmologyConcepts/`**: pass values, or pass the duck-typed cosmology, following
   `_cosmology_break_points`'s own precedent (`BackgroundModel.py:203`), so that a cosmology
   declaring nothing produces **exactly the grid it produces today**. That bit-identity is an
   acceptance test.

2. **Two samples per break point, not one.** The background steps there. A single sample *on* the
   break is ambiguous — which side is it? — and the campaign's redshift rule (README §2 (i)) says
   an equality comparison on a recovered $z$ is not safe. Place a pair straddling each break at a
   relative standoff in $(1+z)$, choose the standoff by measurement against what prompt 10's
   consumer spline needs, and **say what you chose and why**. `BREAK_POINT_STANDOFF = 1e-12` in
   `numeric_with_phase_cut.py` is the existing precedent for a standoff constant and its docstring
   argues its size; do not simply reuse the number without arguing it for this purpose.

3. **Protected points survive `winnow`.** Either give `winnow` a set it must retain, or build the
   response grid by decimating the unprotected points and re-inserting the protected ones. The
   invariant that matters is stated in `main.py`'s own comment: **the response sample must remain a
   subset of the source sample**. Assert it in a test.

4. **Make the grid tag identify the grid.** `SourceRedshiftGrid_{len}` labels size alone. Give it
   something that distinguishes two grids of equal length — a short digest of the construction
   parameters and the protected set is the obvious shape. **This changes a datastore tag**, so
   every object carrying the old tag becomes unfindable under the new one: say so explicitly in the
   log, quantify what it invalidates, and note that prompt 03's representation key has already
   invalidated the QCD half for a different reason. `ResponseRedshiftGrid_{len}` has the same
   defect and the same fix.

5. **`[03-derivative-pad-clamp-on-coarse-grids]` is adjacent, and is not yours.** The background
   derivative-fit padding is clamped near $z=0$ and binds at 50 samples per decade. If your
   protected points change the local spacing anywhere near $z=0$, **measure whether that clamp now
   binds** and record it; do not change it.

## 3. Tests

1. **A cosmology declaring nothing gives the grid it gives today** — bit-identical, element for
   element, for LambdaCDM at the production parameters. This is the test that keeps the change
   inert everywhere it should be.
2. **Every protected point is in the source grid**, and the pair straddling each break is at the
   chosen standoff, on `QCD_Cosmology` at the production parameters.
3. **The response grid is a subset of the source grid**, and every protected point survives the
   winnow.
4. **The grid is still monotone, still descending, has no duplicates**, and its length is what the
   tag says.
5. **Two grids that differ only in their protected set get different tags** — the collision this
   prompt closes. Use `ast` against `main.py` for the tag construction, following
   `test_main_plumbing.py`.
6. **The consumer resolves the step.** With prompt 10's `PrimitivePhase`, the interpolation error
   at and around each break on the new grid, against the old: quote both.

## 4. Acceptance

| Quantity | Now | Target |
|---|---|---|
| Protected points present in the production source grid | 0 of 5 by construction | **all**, by construction |
| Samples straddling each break | none (no break coincides with a sample) | **2 per break**, at a justified standoff |
| Protected points surviving `winnow` | not applicable | **all** |
| Grid tag collision between two equal-length grids | possible | **impossible** |
| LambdaCDM production grid | — | **bit-identical** |
| Consumer error at the breaks, QCD, both sectors, all three $k$ | prompt 10's figures | **quoted**, improvement expected |

## 5. Log and commit

Log to `logs/11-cosmology-aware-source-grid.md` per README §5.1. It must state the standoff chosen
and its justification, the tag scheme, and — prominently — **exactly what the tag change
invalidates in an existing datastore**, because that is the consequence a reader will need and the
one least visible from the diff. Append a dated section to `docs/qcd-background-verification.md`.

Commit subject, or something equally specific:
`Build the source grid around the features the cosmology declares`
