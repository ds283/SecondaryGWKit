# Log 02 — Solve the complexified `N×N` system

**Prompt:** prompts/levin-refactor/02-complexified-solve.md
**Commit:** Solve the Levin collocation system in its complex N x N form; SHA intentionally omitted — see README §5 rule 5
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

All items in `AdaptiveLevin/levin_quadrature.py` and `AdaptiveLevin/tests/test_levin_quadrature.py`.

1. **`_Basis_SinCos.supports_complexified_solve`** (new property, `levin_quadrature.py`) — returns
   `True`. The driver asks this rather than sniffing `isinstance(BasisData, _Basis_SinCos)`, so a
   future basis with a different structure or component count simply answers `False` (falls
   through `getattr(BasisData, "supports_complexified_solve", False)`).

2. **`_Basis_SinCos.build_Levin_data`** — signature gained `need_AmatT: bool = True`. The phase is
   still sampled exactly once regardless of path; the method now also returns `theta_prime_Cheb`
   (needed by the complex path's diagonal, previously computed but not returned), and skips the
   `np.block` construction of the `A^T` block entirely when `need_AmatT=False`. Return tuple
   changed from `(AmatT, w0, wk, phase_span, theta_scale)` to
   `(AmatT, theta_prime_Cheb, w0, wk, phase_span, theta_scale)`; `AmatT` is `None` when
   `need_AmatT` is `False`.

3. **`_adaptive_levin_subregion_impl`** — gated at `m == 2 and BasisData.supports_complexified_solve`
   (`use_complex_solve`). On that path:
   - `LevinL = Dmat.astype(complex)` with `LevinL[np.diag_indices(N)] += 1j * theta_prime_Cheb`,
     replacing the `np.block` assembly of the realified `2N×2N` operator.
   - `rhs = f_Cheb[:N] + 1j * f_Cheb[N:]` (previously `f_Cheb` directly), matching
     `q = p1 + i*p2` solving `(D + i diag(theta')) q = f1 + i*f2` (Chen et al. (168)).
   - The three-tier solver ladder (LU → `lstsq(rcond=None)` → `pinv`) is otherwise **untouched
     code**, now operating on `LevinL`/`rhs` instead of `LevinL`/`f_Cheb` — the same lines serve
     both the real and complex paths.
   - `P = np.vstack([sol.real, sol.imag])` on the complex path, replacing
     `P = sol.reshape(m, chebyshev_order)` on the real path — both produce the same `P[j, i]`
     layout (component `j` at collocation point `i`).
   - `lower_limit`/`upper_limit`/`p_endpoint_l1` rewritten to index `P[i, -1]`/`P[i, 0]` directly
     instead of a flattened `p` vector (`p[(i+1)*N-1]`/`p[i*N]`). This was necessary because the
     complex path never produces a flattened `p` of length `m*N` — only `P` — and it is a genuine
     simplification: the flattened indices were always equal to `P[i, -1]`/`P[i, 0]` by
     construction of `reshape`, so nothing changes for the real path either (confirmed
     numerically, see below).
   - **The residual sanity check on the LU branch was removed** (see judgement call 3 below); the
     LU branch is now accepted on finiteness of `sol_direct` alone, same as the other two branches
     already were.
   - The now-dead `_LEVIN_DIRECT_SOLVE_RESIDUAL` constant and its defining comment were deleted
     (module scope, previously just above `INTERVAL_TYPE_LEVIN`).

4. **Tests** (`AdaptiveLevin/tests/test_levin_quadrature.py`):
   - `test_sincos_basis_reports_complex_support` — locks in the capability contract.
   - `test_lstsq_and_direct_solve_paths_converge` — exercises both branches of the complexified
     solver ladder explicitly (phase span below/above `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`).
   - Suite grew from 9 to 11 tests, runtime unchanged at ~0.01s. Every pre-existing test now
     exercises the complex path incidentally too, since all of them use `_Basis_SinCos` with
     `m == 2`.

## Numerical evidence

Required by README §5 rule 9. `PYTHONPATH=. ./venv/bin/python`, comparisons via `git stash`/`git
stash pop` against baseline `0002fb4`.

### Accuracy A/B — sin/cos test-suite problems plus `e^-x sin(omega x)` at three frequencies

Best-of-5 wall time; `atol=1e-15, rtol=1e-10, chebyshev_order=12` throughout.

| Problem | `\|Δvalue\|` | reported `abserr` | regions before/after | evaluations before/after | speedup |
|---|---|---|---|---|---|
| `SinIntegral` (∫₁^50000 sin x) | 1.11e-16 | 1.00e-11 | 1/1 | 3/3 | 1.44× |
| `CosIntegral` (∫₁^500000 cos x) | 2.22e-16 | 1.00e-10 | 1/1 | 3/3 | 1.51× |
| `SincIntegral` (∫₁^100 sin(100x)/x) | 1.04e-17 | 1.57e-13 | 10/10 | 39/39 | 1.47× |
| `GRZIntegral` (λ=100) | 2.86e-17 | 9.74e-13 | 4/4 | 15/15 | 1.52× |
| `e^-x sin(ωx)`, ω=1e4, [1/3,7/3] | 0.00 | 1.90e-16 | 1/1 | 3/3 | 1.32× |
| `e^-x sin(ωx)`, ω=1e8 | 4.14e-25 | 1.90e-16 | 1/1 | 3/3 | 1.42× |
| `e^-x sin(ωx)`, ω=1e12 | 0.00 | 1.90e-16 | 1/1 | 3/3 | 1.39× |

Every `|Δvalue|` is at or far below the reported `abserr` for that run (the largest, 2.22e-16, is
still 4.5×10⁻¹⁰ times the reported `abserr` of 1.00e-10). `num_regions` and `evaluations` are
identical before/after in every case — the driver's control flow is unaffected, only the linear
algebra changed. Speedups of 1.32–1.52× are within, but at the low end of, the audit's reported
1.4–1.8× end-to-end range; see the discussion of the J000 result below for why very small
single-region problems under-realise the theoretical per-solve saving.

### Accuracy A/B — three-Bessel oracles (production integrand, exercises `theta_deriv`)

`quad_JJJ`, `k=1.3, q=1.7, s=2.1, max_x=1e5`, phase built with `bessel_phase(atol=1e-25,
rtol=5e-14)`, quad `atol=1e-14, rtol=1e-10`, `chebyshev_order=12` (the production default from
`three_bessel_integrals.py`).

| Oracle | before | after | `\|Δvalue\|` | analytic | speedup |
|---|---|---|---|---|---|
| `J000` | `0.16923010755911763` | `0.16923010755911763` | 0.00 (bit-identical) | `0.16923037349654133` | 1.04× |
| `J110` | `0.006509508070508839` | `0.0065095080705088355` | ≈3.5e-18 | `0.0065088605190977405` | — |

Both agree at round-off, as the audit's own correction (§2.3.h — "bit-identical" means the
reported value, not bit-identical arithmetic) predicts.

**On the 1.04× J000 speedup** (materially below the 1.4–1.8× audit figure, investigated per the
prompt's instruction rather than shipped uninspected): `three_bessel_integrals.py:13-25`'s own
comment states the cost of each Levin solve at `chebyshev_order=12` "is dominated by evaluating the
phase and modulus splines at the collocation points" rather than by the linear algebra. A component
microbenchmark (below) confirms the assembly/solve speedups this commit delivers are intact and, if
anything, *better* than the audit's figures in isolation — they are simply a smaller fraction of the
total per-region cost once four phase groups × three Bessel-phase-spline evaluations per collocation
point are counted. This is consistent with, not contrary to, the audit's own scoping (§4 measures
the Levin linear algebra in isolation) and is not evidence against the change. The synthetic
sin/cos problems above, whose amplitude functions are cheap Python lambdas, show the expected
1.3–1.5× end-to-end range because the linear algebra there *is* the dominant cost.

### Component microbenchmark (assembly, LU solve, `lstsq`), best-of-2000 iterations

Standalone benchmark reproducing the audit's own §4.1/§4.2 table shape, using this module's actual
`chebyshev_matrices()` and the exact assembly code now in `_adaptive_levin_subregion_impl`:

| N | assembly real→complex (speedup) | LU solve real→complex (speedup) | `lstsq` real→complex (speedup) |
|---|---|---|---|
| 12 | 21.90→3.29 µs (6.66×) | 9.74→8.13 µs (1.20×) | 91.63→43.51 µs (2.11×) |
| 16 | 26.03→4.81 µs (5.41×) | 12.80→6.42 µs (1.99×) | 92.25→66.63 µs (1.38×, re-measured — see below) |
| 24 | 21.32→3.54 µs (6.02×) | 19.89→8.13 µs (2.45×) | 173.99→92.95 µs (1.87×) |
| 32 | 24.03→3.86 µs (6.23×) | 23.84→13.92 µs (1.71×) | 298.75→146.02 µs (2.05×) |
| 48 | 31.81→4.28 µs (7.43×) | 42.45→27.22 µs (1.56×) | 612.99→307.54 µs (1.99×) |

The first `lstsq` measurement at N=16 gave a spurious 0.67× (123.45→184.88 µs); re-run in isolation
with a larger iteration count and warm-up gave 92.25→66.63 µs (1.38×), consistent with the
neighbouring orders. Reported as measurement noise (background system load), not a regression —
recorded here rather than silently discarded per README §5 rule 9's instruction to distinguish
measurement from reasoning. All figures are the same order as the audit's own table (§4.2):
assembly 5–7× (audit: 5.1×), LU solve 1.2–2.4× (audit: 1.5–2.0×), `lstsq` 1.4–2.1× (audit:
1.7–2.3×).

### Ill-conditioned branch (`lstsq` fallback), including the `rcond` threshold check (step 3)

Forced by choosing `phase_span = 10π < _LEVIN_DIRECT_SOLVE_PHASE_SPAN = 20π`:
`x_span=(0,1)`, `f=[exp(-x), 0.3*cos(3x)]`, `theta(x) = (10π)x`, `chebyshev_order=12`, called
directly against `_adaptive_levin_subregion_impl`.

| | value | phase_span | metadata |
|---|---|---|---|
| before (`0002fb4`) | `0.019970785799531935` | 31.42 (< 62.83 gate) | `{}` (lstsq path, no SVD error) |
| after | `0.019970785799531966` | 31.42 | `{}` (lstsq path, no SVD error) |

`|Δvalue| ≈ 3.1e-17`, below the region's own `phase_err = 1.95e-16`. `metadata == {}` in both cases
confirms the `lstsq` branch (not LU, not `pinv`) was exercised on both sides, and that the
`rcond=None` truncation-threshold halving (`max(M,N)` is now `N` rather than `2N` on the complex
path) is immaterial here, as predicted.

### `lstsq`-share re-measurement (step 6, deliverable for a later RRQR decision)

Instrumented temporarily (removed before commit; not part of the shipped diff) to count true LU
vs. `lstsq` vs. `pinv` solves per region across the full solver ladder — the returned
`num_direct_solves` counter counts only the parent-region solve, which is why this needs
instrumenting externally (C10; prompt 07 fixes the counters).

Measured on two three-Bessel oracles (`k=1.3, q=1.7, s=2.1, max_x=1e5`, `chebyshev_order=12`,
production defaults), which between them invoke `adaptive_levin_sincos` once per phase group (4
groups) per oracle:

| Oracle | LU solves | `lstsq` solves | `pinv` solves | `lstsq` share |
|---|---|---|---|---|
| `J000` | 95 | 213 | 0 | **69.2%** |
| `J110` (incremental) | 118 | 386 | 0 | **76.6%** |

This is **substantially higher** than the audit's own synthetic-problem reference points (0% on
three problems, 25% worst-case at k=12, 10% at k=32, §4.4). The likely explanation, consistent with
README standing note 7: `three_bessel_integrals.py`'s four sum-and-difference phase groups include
difference-type combinations (`e_nu, e_sigma` with opposite signs) whose *net* frequency can be
small even when the individual Bessel arguments are large, so a much larger fraction of the m=2
subregions in the production integrand fall below the `_LEVIN_DIRECT_SOLVE_PHASE_SPAN` gate than in
the audit's synthetic sweep. This measurement is recorded as the evidence the audit's own §4.4
instructs be gathered before any RRQR work is considered — no such work was done here (see *Do
not*, unchanged).

## Deviations from the prompt

### Implementation choice: how `build_Levin_data` returns what the complex path needs (step 1)

Considered two designs: (a) a single capability call that both answers "does this basis support
complexification" and hands back everything the complex path needs in one shot, or (b) keep
`build_Levin_data` as the single sampling point for *all* derived quantities (real and complex
alike), add a boolean property purely for the capability question, and add a `need_AmatT` flag to
`build_Levin_data` so the (basis-owned) `A^T` construction is skipped when it won't be used.

Took (b). Reasons: `A^T`'s block structure is specific to a basis's own algebraic relationship
between its components (`w' = A w`), not something the generic `m`-component driver should
construct — standing note 10 requires the generic path to remain intact for future bases, and a
future basis with a different `A` structure should build its own `A^T` the same way `_Basis_SinCos`
does today. Moving `A^T` construction into the driver (as a hard-coded 2-block `np.block`, since
that's the only construction that exists) would have made the "generic" path silently `m=2`-only
even though the loop over `row_list` is written for general `m`. Keeping `build_Levin_data` as the
one sampling point, with a flag to skip the part of its output the caller won't use, preserves that
separation while still sampling `theta_prime_Cheb` exactly once regardless of path.

A consequence worth recording: this means no separate endpoint-phase / `e^{iθ}` construction was
needed at all. The prompt raises the possibility of returning "the endpoint phases" so the complex
path can form `e^{iθ}` directly and compute `Im[q(b)e^{iθ(b)} - q(a)e^{iθ(a)}]`. Verifying the
prompt's own worked-out equivalence (`Im[(p1+ip2)(cosθ+isinθ)] = p1 sinθ + p2 cosθ`) shows this is
*exactly* the existing `lower_limit`/`upper_limit` expression already computed from `P` and
`w0`/`wk` — which are needed by the real path regardless and cost nothing extra to keep computing
on the complex path. So `w0`/`wk` (sin/cos pairs, not raw angles) are reused unchanged for both
paths, and no `e^{iθ}` value is ever constructed in the code. This was checked, not assumed: see
*Verification performed*.

### Implementation choice: no persistent buffer (step 2)

Did not use a persistent module-level buffer. `Dmat.astype(complex)` already allocates a fresh
array on every call (`astype` copies by default unless `copy=False` and the dtype already
matches — irrelevant here since `Dmat` is real and the target is complex), so there is nothing to
share, and the component microbenchmark above shows the resulting assembly cost (2.8–4.3 µs across
N=12–48) is already the audit's claimed win without needing to avoid a per-call allocation. A
persistent buffer would also have to be one-per-Chebyshev-order and explicitly excluded from
cross-thread sharing (the driver runs under Ray workers); avoiding that hazard entirely, for no
measured benefit, was the easy call.

### Implementation choice: removed the residual sanity check (step 5)

Removed. The A/B the prompt requires was run by temporarily logging, for every LU-branch attempt,
whether the finiteness check alone would have differed from the finiteness-and-residual check
together. Across the full `AdaptiveLevin/tests/` suite plus the `J000` oracle (**221 direct-solve
attempts** spanning every existing test problem), the relative residual `‖Lq − rhs‖ / ‖rhs‖` never
exceeded **4.65e-15** against the `1e-10` threshold — nine orders of magnitude of margin — and the
two gates (finite-only vs. finite-and-residual) agreed on every single call (221/221 accepted by
both). This is exactly the behaviour the code's own pre-existing comment predicted ("in the
near-singular regime the residual is small... a residual test alone would silently accept
garbage" — but the phase-span gate, not the residual, is what actually protects against that
regime; the residual check was only ever a secondary guard against "outright failure", which the
existing finiteness check on `sol_direct` already catches). Combined with the audit's own
measurement that the check cost 58–72% of the solve it guarded, removing it is a clear win with no
observed behavioural change. The instrumentation itself was not committed (removed before the final
diff); the measurement is recorded here per the prompt's instruction.

## Verification performed

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` — actually run:
  `Ran 11 tests in 0.009s`, `OK` (grew from 9 to 11 with the two new tests in this commit).
- `./venv/bin/black --check AdaptiveLevin/levin_quadrature.py AdaptiveLevin/tests/test_levin_quadrature.py`
  — clean after running `black` once on the test file (whitespace only).
- Accuracy A/B (item 2 of Verification): actually run, both `git stash`/`git stash pop` against
  `0002fb4`, `repr()`-precision values compared; see *Numerical evidence* above. All seven
  synthetic problems and both three-Bessel oracles checked.
- Endpoint-extraction equivalence (prompt's "check this yourself" instruction, §*The mathematics*):
  worked the algebra by hand (`Im[(p1+ip2)(cosθ+isinθ)] = p1 sinθ + p2 cosθ`, matching the existing
  `w0`/`wk`-based extraction exactly) and confirmed no separate complex-endpoint code path exists in
  the diff — the same `lower_limit`/`upper_limit` lines run on both paths, fed by `P` built two
  different ways.
- Speedup measurement (item 4): best-of-5 wall time on the seven synthetic problems and best-of-1/3
  on the two three-Bessel oracles (the latter too slow for repeats=5); actually run and tabulated
  above, not estimated.
- Ill-conditioned branch (item 5): actually run standalone against `_adaptive_levin_subregion_impl`
  directly, before and after, with `phase_span` confirmed below the gate and `metadata` confirmed
  empty (lstsq path) on both sides — see table above.
- `lstsq`-share re-measurement (item 6, step 6): actually run with temporary counters against two
  three-Bessel oracles (`J000`, `J110`); counters removed before the final commit diff.
- Component microbenchmark: actually run, best-of-2000 (best-of-5000 for the N=16 `lstsq`
  re-measurement), warm-up iterations excluded from timing.

## Observations not acted on

- The `lstsq`-share measurement (69–77% on production three-Bessel integrands, against the audit's
  synthetic worst case of 25%) is new information relevant to the deferred RRQR decision (audit
  §4.4, README §6). Not acted on here — prompt 02's *Do not* list explicitly defers that decision —
  but it changes the cost/benefit case for prompt-adjacent future work: RRQR would address a larger
  share of solves on the real production integrand than the audit's own reference points suggested.
  Recorded for whoever picks that up.
- `metadata["direct_solve"]` still counts only the parent-region solve, not the comparison
  region (`dataL`/`dataR`) solves, exactly as before this commit — untouched, in scope for prompt 07
  (C10).
- The spurious N=16 `lstsq` timing measurement (0.67× on the first pass) is a reminder that these
  microbenchmarks are noisy at the tens-of-microseconds scale on a shared machine; the component
  table above already carries the re-measured, more reliable number, but any future re-measurement
  should use enough iterations (thousands) and a warm-up phase, as this one now does.

## State handed to the next prompt

- `_adaptive_levin_subregion_impl` now uses `sol`/`rhs`/`LevinL` as the shared names for the solve
  ladder's inputs/output on both the real and complex paths, `P` (shape `(m, chebyshev_order)`) as
  the sole downstream representation of the solved antiderivatives, and `use_complex_solve` as the
  gate variable. Prompt 03 restructures this function around a second cell type (Clenshaw–Curtis);
  it should preserve `use_complex_solve`'s gating (`m == 2 and BasisData.supports_complexified_solve`)
  and must not reintroduce a flattened `p` vector when adding new logic that reads antiderivative
  values — index `P` directly.
- `_Basis_SinCos.build_Levin_data` now takes `need_AmatT` and returns `theta_prime_Cheb` as its
  second element (return signature: `(AmatT, theta_prime_Cheb, w0, wk, phase_span, theta_scale)`).
  Any new caller of `build_Levin_data` must match this signature.
- `_LEVIN_DIRECT_SOLVE_RESIDUAL` no longer exists (removed as dead code alongside the residual
  check). If a future prompt wants a residual-based guard back, it needs a fresh justification —
  see the measurement above for why the current gate does not need one.
- The complex fast path is active for every existing test and every production caller today (all
  go through `_Basis_SinCos` with `m == 2`), so from prompt 03 onward, "the Levin path" and "the
  complexified path" are the same code in practice unless a future basis opts out.
