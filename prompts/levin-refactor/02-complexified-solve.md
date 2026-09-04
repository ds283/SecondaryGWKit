# Prompt 02 — Solve the complexified `N×N` system

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §4.1, §4.2, §4.4; §5 recommendation 6
**Depends on:** prompt 01 (its finiteness guards live in the code you are rewriting — carry them across)
**Files you may touch:** `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_quadrature.py`,
plus the log and the status board.

---

## Character of this commit

A pure efficiency change with no intended effect on the returned value beyond round-off. It is the
single largest measured win in the audit, it is the paper's own formulation, and it makes the code
simpler rather than more complicated.

The measured picture, which inverts what the earlier performance campaign assumed: **at the default
order the linear solve is 15% of a subregion evaluation and `np.block` assembly alone is 35%.**

---

## The mathematics

For the `(sin θ, cos θ)` basis the real `2N×2N` system is the realification of a complex `N×N` one.
The current code builds

```
LevinL = blockdiag(D, D) + AmatT,     AmatT = [[0, -diag(θ')], [diag(θ'), 0]]
```

acting on `p = [p₁; p₂]` (each block of length `N`) with right-hand side `f_Cheb = [f₁(grid); f₂(grid)]`.
Written out, the two row blocks are

```
D p₁ − diag(θ') p₂ = f₁
diag(θ') p₁ + D p₂ = f₂
```

Setting `q = p₁ + i p₂` and adding `i ×` the second to the first:

```
(D + i diag(θ')) q = f₁ + i f₂
```

which is Chen et al. (168). The endpoint extraction (171) becomes

```
I_region = Im[ q(b) e^{iθ(b)} − q(a) e^{iθ(a)} ]
```

**Check this against the current code before you write anything.** The current extraction at
`levin_quadrature.py:772-777` is `Σᵢ pᵢ(b) wᵢ(b) − Σᵢ pᵢ(a) wᵢ(a)` with `w = (sin θ, cos θ)`, i.e.
`[p₁(b) sin θ_b + p₂(b) cos θ_b] − [p₁(a) sin θ_a + p₂(a) cos θ_a]`. And
`Im[(p₁ + i p₂)(cos θ + i sin θ)] = p₁ sin θ + p₂ cos θ`. They agree. Confirm this yourself rather
than taking it on trust — it is the one place where a sign or a conjugate error would produce a
plausible wrong answer rather than an obvious one.

**The grid is descending** (`chebyshev_matrices` returns `x[0] = b`, `x[-1] = a`; the comment at
`:414` documents this and is load-bearing). So `q[0]` is at the upper endpoint and `q[-1]` at the
lower. The current code's index arithmetic — `p[i*N]` for the upper endpoint, `p[(i+1)*N - 1]` for
the lower — encodes the same thing in the flattened layout.

## Verified before this plan was written

- **Exact equivalence.** Solving both systems on the same data gives `max|Δp| ≈ 2e-21` at
  N = 12, 24, 32 on solutions of size `O(10⁻⁵)`, i.e. agreement at round-off.
- **Identical conditioning.** `cond(LevinL)` and `cond(D + i diag(θ'))` agree to 4 significant
  figures at every N tested. The complexification does not trade accuracy for speed.
- **Timings** (best-of-7, this machine, µs):

| N | `np.block` assembly | complex assembly | `solve` 2N | `solve` N | speedup | `lstsq` 2N | `lstsq` N | speedup |
|---|---|---|---|---|---|---|---|---|
| 12 | 14.11 | **2.76** | 6.92 | 4.59 | 1.51× | 67.0 | 37.3 | 1.80× |
| 16 | 14.79 | **2.81** | 9.48 | 5.33 | 1.78× | 101.2 | 57.2 | 1.77× |
| 24 | 16.81 | **2.97** | 15.30 | 7.76 | 1.97× | 178.4 | 104.5 | 1.71× |
| 32 | 18.99 | **3.30** | 21.72 | 12.67 | 1.71× | 311.1 | 161.0 | 1.93× |
| 48 | 33.98 | **8.07** | 40.76 | 25.14 | 1.62× | 766.4 | 340.8 | 2.25× |

Assembly is 5.1× cheaper at N = 12 and it is the *largest single component* of a subregion
evaluation. The audit's end-to-end measurement is **1.4–1.8×** across six problems.

**One correction to the audit.** §4.2 says the results are "bit-identical". They are not, and cannot
be: the complex path routes through LAPACK's complex drivers rather than the real ones, so the
arithmetic genuinely differs. What is true is that the two agree at round-off, and that the audit's
end-to-end `|Δvalue|` was `≤ 3.3e-24` on values of size `O(10⁻⁷)`. **Do not write an acceptance test
that demands exact equality** — see *Verification* for what to demand instead.

---

## What to do

### 1. Add a complex fast path in `_adaptive_levin_subregion_impl`

Gate it on `m == 2` **and** on the basis being `_Basis_SinCos`. Ask the basis object rather than
sniffing types: give `_Basis_SinCos` a property or method that says it supports the complexified
form and returns what the complex path needs. The generic `m`-component path **must be retained** —
standing note 10, and audit §4.2's closing sentence: it exists for the future bases sketched in the
prior review's §8.3.

`build_Levin_data` currently returns `(AmatT, w0, wk, phase_span, theta_scale)` and builds `AmatT`
by `np.block` at `:490-493`. For the complex path you need `theta_prime_Cheb` itself, and the
endpoint phases rather than (or as well as) the endpoint basis vectors `w0`/`wk`. Restructure the
return so that both paths get what they need from **one** sampling of the phase — do not sample
twice, and do not build `AmatT` on the complex path at all.

The endpoint phases are already computed inside `build_Levin_data` (`theta0_mod_2pi`/`thetak_mod_2pi`
at `:509-510`, or `theta0`/`thetak` at `:526-527`); currently only `sin`/`cos` of them survive. Return
the angles so the complex path can form `e^{iθ}` — or return `w0`/`wk` and construct
`e^{iθ} = w[1] + i·w[0]` from them, which is exactly equivalent and avoids a second `sin`/`cos`.
Either is fine; pick one and say why in the log.

### 2. Assemble into a preallocated buffer

The operator is `D` with `iθ'` added on the diagonal, so `np.block` disappears entirely. **Watch
two things:**

- `_chebyshev_base` returns arrays marked read-only and shared between callers (`:391-392`, and the
  docstring at `:362-363` says so explicitly). `chebyshev_matrices` forms a *new* `D` by
  `2.0 * D_base / (b - a)` (`:426`), so the `D` you receive is writable and private to this call —
  but confirm that before mutating it, and do not mutate `D_base`.
- A preallocated buffer that persists between calls must be per-order and must not be shared across
  threads. The driver is single-threaded but is called from Ray workers; a module-level mutable
  buffer is a hazard worth avoiding. `M = D.astype(complex)` followed by
  `M[np.diag_indices(N)] += 1j * theta_prime_Cheb` allocates once per call and measured **2.76 µs at
  N = 12** against `np.block`'s 14.11. That is already the win; a persistent buffer is not needed to
  get it. If you use one anyway, justify it and make it safe.

### 3. Keep the solver ladder intact

The three-tier ladder — LU fast path gated on `phase_span > _LEVIN_DIRECT_SOLVE_PHASE_SPAN`, then
`lstsq(..., rcond=None)`, then `pinv` — stays exactly as it is, with `np.linalg` operating on the
complex system. All three support complex input.

**Do not** change the gate. The comment at `:679-685` is right and load-bearing: in the
near-singular regime the residual is small even when `p` is completely wrong, so a residual test
alone would silently accept garbage. The phase span is the primary guard.

`lstsq(..., rcond=None)` truncates at `eps·max(M,N)·σ₁`, which is the paper's step-5 truncation at
`‖A‖ε₀` (audit §1.1, verified). Note that `max(M,N)` is now `N` rather than `2N`, so the truncation
threshold changes by a factor of two. This is a real behavioural difference on the ill-conditioned
branch. It is almost certainly immaterial, but **check it** on a deliberately ill-conditioned region
(phase span below `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`) and record what you found.

### 4. Preserve every derived quantity exactly

`P = p.reshape(m, N)` at `:756` becomes `P = np.vstack([q.real, q.imag])` — check the ordering
against the flattened layout, do not assume. Everything downstream must be unchanged in meaning:
`p_means`, `p_mean_max`, `p_ratios`, `p_use`, `p_sample`, `lower_limit`, `upper_limit`,
`p_endpoint_l1`. `p_sample` is consumed by `_write_progress_data` (`:1444-1462`) and by the returned
`p_points`; keep its `(x, [p_1, p_2])` shape.

**Carry prompt 01's finiteness guards across.** The check on the solved `p` becomes a check on `q`;
the `f_Cheb` check becomes a check on whatever you build the complex right-hand side from. Do not
lose one in the rewrite — that would silently undo the campaign's most important commit.

### 5. Consider the residual sanity check

Audit §4.3: the residual check at `:691-696` costs **4.6 µs against the 8.2 µs solve it guards**
(reproduced here: 4.0–5.0 µs against 6.9 µs, i.e. 58–72%). The code's own comment (`:682-685`) says
the phase span is the primary guard and the residual is "only a secondary guard against non-finite
values and outright failure" — a purpose a bare `np.isfinite(q).all()` discharges at ~0.3 µs.

The audit says "worth an A/B before removing, but the cost/benefit is poor as it stands". **Do the
A/B, decide, and record it.** If you keep the residual check, say what it caught that the finiteness
check would not. If you remove it, confirm that prompt 01's finiteness guard on the solved vector is
in place and covers the "outright failure" case, and that the LU path still cannot accept a wrong-
but-finite `p` on a region with `phase_span` above the gate. This is a genuine judgement call and
either answer is defensible; an unrecorded one is not.

### 6. Re-measure the `lstsq` share

Audit §4.4 defers the rank-revealing-QR question with a specific instruction: *"Do complexification
first, then re-measure the `lstsq` share on the real three-Bessel integrands before writing any
pivoted-QR-plus-rank-determination code."*

You are the "first". **Instrument the module temporarily** (not committed) to count true LU solves
versus true `lstsq` solves across all three solves per region — the returned counters count only the
parent solve, which is why the audit had to measure this externally (C10; prompt 07 fixes the
counters). Run it on at least two three-Bessel oracles from
`LiouvilleGreen/tests/test_3bessel_analytic.py` and record the split in your log.

This is the evidence a later decision rests on. The audit's reference points, measured on synthetic
problems before complexification: `lstsq` share 0% on three problems, 25% on the worst (`lorentz
peak` at k=12), 10% at k=32.

---

## Do not

- Do not remove the generic `m`-component path.
- Do not change `_LEVIN_DIRECT_SOLVE_PHASE_SPAN` or the reasoning behind the gate.
- Do not write pivoted-QR code. Audit §4.4 and README §6: the decision is deferred pending the
  measurement you are taking in step 6.
- Do not change the gate at `:924`, `theta_scale`, `_phase_error`, the acceptance test, or `p_use`'s
  selection rule. Those are prompts 03, 04, 05 and 06.
- Do not vectorise the sampling loops. That is prompt 08, and doing it here would confound the
  timing A/B this commit rests on.

---

## Verification

1. `AdaptiveLevin/tests/` passes (should be ~9 tests after prompt 01).
2. **Accuracy A/B (required).** For each of at least five problems — the four in
   `AdaptiveLevin/tests/`, plus `∫_{1/3}^{7/3} e^{−x} sin(ωx) dx` at ω = 10⁴, 10⁸, 10¹² against its
   closed form — record `value`, `abserr`, `num_regions` and `evaluations` before and after.
   **Acceptance criterion:** `|Δvalue|` at or below the reported `abserr` for that run, and the
   error against the closed form no worse after than before. Do **not** demand bit-equality (see the
   correction above). Report the actual `|Δvalue|` numbers; the audit's were `≤ 3.3e-24`.
3. **At least one three-Bessel oracle** from `LiouvilleGreen/tests/test_3bessel_analytic.py` agrees
   before and after to the same number of digits. This is the production integrand and it exercises
   `theta_deriv`, which the `AdaptiveLevin` tests do not (C12).
4. **Speedup measurement.** End-to-end wall time on the same five problems, before and after,
   best-of-N. Report the ratios. The audit's expectation is 1.4–1.8×; if you measure materially
   less, say so and investigate rather than shipping the claim.
5. **The ill-conditioned branch.** Confirm the `lstsq` path is actually exercised (force it by
   choosing a region with `phase_span` below the gate) and that its result is unchanged in the same
   sense as item 2, including after the `rcond` truncation-threshold change noted in step 3.
6. `black --check` clean if available.

---

## Finish

1. Write `prompts/levin-refactor/logs/02-complexified-solve.md`. Under *Numerical evidence*, include
   the accuracy A/B table (item 2), the three-Bessel check (item 3) and the speedup table (item 4).
   Under a clearly-headed section, record **the `lstsq`-share measurement from step 6** — that is the
   deliverable a later RRQR decision depends on, and it must be findable without re-reading the diff.
   This prompt contains **three explicit judgement calls**: how the endpoint phases are returned from
   `build_Levin_data` (step 1), whether a persistent buffer is used (step 2), and whether the
   residual sanity check is kept (step 5). Each needs an implementation-choice entry.
2. Update `IMPLEMENTATION_STATE.md`.
3. Commit in one commit. Suggested message:

```
Solve the Levin collocation system in its complex N x N form

For the (sin, cos) basis the real 2N x 2N Levin super-operator is the
realification of a complex N x N one: with q = p1 + i p2 the two row blocks
collapse to (D + i diag(theta')) q = f1 + i f2, and the endpoint extraction
becomes Im[q(b) exp(i theta(b)) - q(a) exp(i theta(a))]. This is Chen et
al.'s own formulation, (168) and (171), rather than a reformulation of it.

The motivation is measurement rather than flop count. Profiling a subregion
evaluation component by component showed the linear solve to be only 15% of
its cost at the default order, while assembling the block operator with
np.block was 35% -- the reverse of what the earlier performance campaign
assumed. In the complex form the operator is D with i*theta' added on the
diagonal, so the assembly is a copy and an in-place diagonal update, and
np.block disappears from the hot path entirely.

Measured per operation: assembly 14.1 -> 2.8 us at N = 12, solve 1.5-2.0x
faster and lstsq 1.7-2.3x faster across N = 12 to 48. End to end this is
<FILL IN>x on the problems measured, with values agreeing to <FILL IN>.

The two systems are exactly equivalent, not approximately: they have
identical condition numbers and their solutions agree at round-off. Note
that agreement is at round-off and not bit-for-bit, since the complex path
uses LAPACK's complex drivers rather than the real ones.

The generic m-component path is retained for bases other than (sin, cos).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Fill in the measured numbers; do not ship the placeholders.
