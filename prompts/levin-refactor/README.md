# Levin refactor campaign: `AdaptiveLevin/levin_quadrature.py`

**Source document:** [`docs/adaptive-levin-audit-2026-09.md`](../../docs/adaptive-levin-audit-2026-09.md)
**Planned:** 2026-09-04
**Target branch:** `main` (clean at `c8a1918` when this plan was written)
**Status board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)

---

## 1. What this campaign does

`AdaptiveLevin/levin_quadrature.py` implements the adaptive Levin scheme of Bremer, Chen & Yang
(arXiv:2211.13400v3, §5) for rapidly oscillatory integrals. The audit's verdict is that the Levin
core is **correct** — the differentiation matrix, the `Aᵀ` construction, the descending-grid
convention and the endpoint extraction all verify exactly, and the module delivers the
frequency-independent absolute accuracy the paper predicts.

What it does not do is **either return a meaningful result or refuse**. Three defects break that
property, two of them capable of returning a confidently-wrong number:

- a non-finite amplitude sample makes a region contribute exactly 0 with a reported error of
  exactly 0 (**C1**);
- the weakly-oscillatory gate tests *net* phase change rather than total variation, so an integral
  with an interior stationary point is handed wholesale to `scipy.quad` and comes back **1590%**
  wrong, with the Levin rule never invoked (**C2**);
- `atol` is a per-region tolerance that is never compared against what the caller asked for
  (**C3**).

Beyond correctness there is a measured 1.4–1.8× end-to-end speedup available from solving the
paper's own complexified `N×N` system instead of the realified `2N×2N` one, and a reported
round-off floor that is optimistic by up to ten orders of magnitude on exactly the code path
production uses.

**Ten prompts, each landing exactly one commit, each independently revertible.**

---

## 2. Reconciliation of the audit against the codebase

The audit states it was performed at commit `68cff5d`. **`AdaptiveLevin/levin_quadrature.py` is
byte-identical between `68cff5d` and the current `HEAD` (`c8a1918`)**, confirmed with
`git diff 68cff5d HEAD -- AdaptiveLevin/levin_quadrature.py` (empty), and the working tree is clean
for `AdaptiveLevin/` and `Quadrature/`. Every line number in the audit therefore resolves directly.

### 2.1 Line references — all resolve exactly

Spot-checked all 31 line numbers cited in the audit body. Representative sample:

| Audit citation | Resolves to |
|---|---|
| `:662` `np.isfinite(LevinL).all()` | ✅ `levin_quadrature.py:662` |
| `:769` `p_use = [r > rtol for r in p_ratios]` | ✅ `levin_quadrature.py:769` |
| `:924` `phase_diff = np.fabs(BasisData.raw_theta(b) - ...)` | ✅ `levin_quadrature.py:924` |
| `:521` `theta_scale = TWO_PI` | ✅ `levin_quadrature.py:521` |
| `:498` `phase_span = mean|θ'|·width` | ✅ `levin_quadrature.py:498` |
| `:1059` `relerr_denom = max(min(...), atol)` | ✅ `levin_quadrature.py:1059` |
| `:1083` `resolved = abserr < atol or relerr < rtol` | ✅ `levin_quadrature.py:1083` |
| `:1112` `if current_region.depth > max_depth:` | ✅ `levin_quadrature.py:1112` |
| `:972` fallback-branch `continue` | ✅ `levin_quadrature.py:972` |
| `:1393` unguarded `min(|est|,|ref|)` in diagnostics | ✅ `levin_quadrature.py:1393` |
| `:66-67` `seaborn` / `matplotlib.pyplot` module-scope imports | ✅ `levin_quadrature.py:66-67` |

### 2.2 Findings reproduced numerically

Every critical and high finding was re-run against `HEAD` in the repository venv
(`./venv`, Python 3.12, NumPy 2.2.4, SciPy 1.15.2) with `PYTHONPATH=.`:

| Finding | Audit's measurement | Reproduced |
|---|---|---|
| **C1** partial integral | `value = -5.191968760836506e-07`, `abserr = 1.03e-15`, 4 regions | ✅ `-5.191968760785887e-07`, `1.025e-15`, 4 regions |
| **C1** all-NaN | `value = 0.0, abserr = 0.0` | ✅ exactly |
| **C2** stationary phase | rel. err `1.59e+01`, 1 region, 0 Levin solves, reported `abserr` 0.19 | ✅ rel. err `1.590e+01`, 1 region, 0 solves, `abserr = 0.18889` |
| **C2** split at ½ | 12 correct digits | ✅ rel. err `8.0e-10` (16 regions; the audit's 20-region figure used a different driver) |
| **C3** `atol = 1e-10` | reported `abserr = 1.26e-10`, exceeds `atol` | ✅ `1.26e-10`, exceeds; at `atol = 1e-12` it does not |
| **C4** reduced-phase floor | optimistic by `5.5e10` at ω = 10¹² | ✅ optimistic by `8.1e9` at ω = 10¹² (`atol=1e-18`); raw path flat at `1.90e-16` and never exceeded |
| **C6** `max_depth` | never updated on the fallback branch | ✅ reports `max_depth = 0` on a run that is entirely fallback |
| **C8** `atol = 0` | 256 regions / 1023 solves at `depth_max = 8` | ✅ exactly 256 / 1023, against 1 region / 3 solves at `atol = 1e-15` |
| **C9** validation matrix | 8 rows | ✅ all 8 reproduce verbatim, including the silently-truncated 3-tuple `x_span` |
| **C11** import cost | 1.13–1.36 s of 1.57 s | ✅ **worse** — `seaborn` (cumulative) is 1.20–2.00 s of a 1.26–2.10 s total import, i.e. >95% |
| **§2.2** CC accuracy | `CC₁₃`/`CC₂₅` errors at 0.5π…10π | ✅ reproduces to within a factor 1.1 at every span |
| **§4.2** complexification | solve 1.42–1.89×, `lstsq` 1.91–2.09× | ✅ solve **1.51–1.97×**, `lstsq` **1.71–2.25×** (best-of-7, N = 12…48) |
| **§4.2** equivalence | identical condition numbers, agreement ≤7.5e-16 | ✅ condition numbers identical to 4 s.f.; `max\|Δp\| ≈ 2e-21` |
| **§4.1** `np.block` dominance | `np.block` 19.9 µs vs solve 8.2 µs at N=12 | ✅ `np.block` 14.1 µs vs solve 6.9 µs; complex assembly **2.8 µs** (5.1× cheaper than `np.block`) |
| **§4.3** residual check cost | 4.6 µs against an 8.2 µs solve (56%) | ✅ 4.0–5.0 µs against 6.9 µs (58–72%) |

**Baseline test state:** `AdaptiveLevin/tests/` — 4 tests, all pass, 0.008 s. There is no `pytest`
in the venv; use `python -m unittest`.

### 2.3 Eight corrections to the audit

These are established by measurement or by direct inspection, and each is carried into the prompt
that needs it. **They are not optional reading**: three of them change what the prompts should do.

**(a) Clenshaw–Curtis nesting works for *every* N, not only odd N.** §4.5 advises that "odd orders
are preferable if §2.2's nested Clenshaw–Curtis fallback is adopted (nesting needs `N−1` even)".
That constraint does not exist. The extremal grid is `cos(kπ/(N−1))`, `k = 0…N−1`; the `2N−1`-point
grid is `cos(jπ/(2N−2))`, and `j = 2k` gives `cos(kπ/(N−1))` for any `N ≥ 2` because `2N−2` is even
by construction. Verified exactly (`max|diff| = 0.0`) at **N = 12, 13, 16, 17, 25, 33**.
*Consequence:* there is no conflict between prompt 03 (nested CC) and prompt 08 (raise the default
order to 16, an even number). Do not introduce an odd-order constraint.

**(b) `three_bessel_integrals.py`'s "hard-coded 64" is stale.** §4.5 says
"`three_bessel_integrals.py`'s hard-coded 64 is worth re-tuning on its own integrands". It is
already 12 — `DEFAULT_3BESSEL_CHEBYSHEV_ORDER = 12` at `three_bessel_integrals.py:26`, retuned in
commit `cc64ae4`, with the supporting measurement recorded in the source comment above it (identical
relative error at orders 12–64 against the analytic oracles; 3–6.6× runtime saved). The file that
still hard-codes 64 is **`ComputeTargets/QuadSourceIntegral.py:29`** (`CHEBYSHEV_ORDER = 64`, used
at 9 call sites). *Consequence:* prompt 08 retargets that advice.

**(c) The `three_bessel` retuning comment independently corroborates recommendation 5's second
half.** That comment records that the delivered relative error "is set by the accuracy of the phase
and modulus splines, not by the spectral order". That is precisely the error source the module
cannot currently express and reports as ~10⁻¹⁵ — the case for the optional `theta_abserr` in the
phase contract. Prompt 04 cites it.

**(d) C12 overstates the test gap.** It says `_GRZIntegral(10.0)` "takes [the fallback] by accident
… so **that test never exercises Levin at all**". `test_GRZIntegral` calls `_GRZIntegral` at
λ = 10, 100 **and** 1000; only the λ = 10 case falls through to direct quadrature. The test does
exercise Levin, twice. The narrower true statement — that one of the three sub-cases silently never
reaches the Levin rule — still deserves fixing.

**(e) C10 is a naming problem at least as much as a counting bug.** The audit reads
`num_direct_solves` under-counting by ~3× as a defect. The source comment at
`levin_quadrature.py:977-980` states the per-region accounting is *deliberate*: "the metadata of the
comparison regions `dataL`/`dataR` has never been accumulated, so reusing them preserves the
existing accounting exactly". Both readings are defensible; what is indefensible is a counter named
`num_direct_solves` that counts regions. Prompt 07 is scoped as *make the names match the
semantics, and add true solve counts alongside* — not as "fix an under-count".

**(f) Recommendation 15's stated target has no production caller.** `quad_JJJ`/`quad_YJJ` are called
only from `LiouvilleGreen/tests/test_3bessel_analytic.py` and the `docs/adaptive-levin-benchmark/`
harness. The production consumer of Levin quadrature is **`ComputeTargets/QuadSourceIntegral.py`**,
which calls `adaptive_levin_sincos` **directly at 9 sites** and discards `abserr` at every one of
them. Prompt 09 covers both, and treats `QuadSourceIntegral.py` as the load-bearing one.

**(g) Equation (151) aggregates gracefully under subdivision; the endpoint model does not.** Measured
while spot-checking recommendation 5: summing the eq. (151) bound over a cell decomposition of
`∫_{1/3}^{7/3} e^{−x} sin(ωx) dx` gives a total that is **invariant under the number of cells**
(it is `≈ 3·eps·‖f‖_∞·(h_total/2)` once `G₁ > k²`), whereas the raw endpoint bound summed over the
same 400 cells is 400× larger (`2.5e-14` against `9.3e-17`). This is a *further* argument for
recommendation 5 that the audit does not make: the current floor grows linearly with region count,
so a heavily-subdivided integral is reported as less accurate the harder the driver works on it.
Prompt 04 records it. **Caveat on my own check:** 10 cells of one problem family, eq. (151) valid
(worst true/bound ratio 0.001) in all of them — this is a sanity check, *not* a reproduction of the
audit's 26-cell validation, which stands on its own evidence.

**(h) "bit-identical" in §4.2 means the reported value, not the arithmetic.** Complexification
routes through LAPACK's complex drivers (`zgesv`/`zgelsd`) rather than the real ones, so the
operations are genuinely different and bit-identity is not available. What is true, and what
prompt 02 must verify, is that the two solutions agree at round-off: measured `max|Δp| ≈ 2e-21` on a
solution of size `O(10⁻⁵)`. Do not write an acceptance test that demands exact equality.

### 2.4 Deviations from the audit's recommended ordering

The audit's §5 priority table orders by (severity or payoff) ÷ effort: 1–4 and 10 trivial, then
5, 6, 7, 8. This plan follows it with **three deliberate departures**, each noted in the prompt that
carries it:

1. **Recommendation 7 (total-variation gate + CC fallback) is scheduled before recommendation 5
   (the round-off floor)** — prompt 03 before prompt 04, inverting the audit's 5-then-7. Two
   reasons. First, C2 is a *critical* wrong-answer finding and C4 is a *high* wrong-error-bar
   finding; the wrong answer goes first. Second, recommendation 7 **deletes** the
   `simple_quadrature` fallback branch, which carries its own `direct_phase_err` computed through
   `_phase_error`/`phase_scale`. Landing 5 first would mean reworking a code path that 7 then
   removes, and then writing the new floor for the CC branch anyway — the same work twice.
2. **Recommendation 10 (raise `_LEVIN_PHASE_ERROR_SAFETY` to 4–8) is folded into prompt 04, not
   shipped as a standalone trivial change.** The audit's own measurement is that the *endpoint*
   bound was exceeded by 1.24× in one of 30 cells at a factor of 1.0, while **eq. (151) has 4.5×
   margin at `C = 1`**. Raising a single shared constant to 8 before replacing the model it applies
   to would inflate every reported error bar eightfold for one prompt and then be wrong for the new
   model. Prompt 04 introduces two separately-named constants and sets each from the audit's own
   measurement for that model.
3. **Recommendation 16 (tests) is distributed, not deferred.** Every prompt adds the tests for its
   own change; prompt 10 completes the matrix and verifies the campaign. The audit's list of
   uncovered cases is mostly a list of *the defects this campaign fixes*, so writing them up front
   would mean committing a suite that is red for eight prompts.

Nothing the audit recommends is dropped. The two optimisations it measured and **rejected**
(p-refinement, §2.3; global error balancing, §2.3) are not scheduled, and prompt 03 carries an
explicit note not to reintroduce them.

### 2.5 Facts about the callers that the prompts depend on

| Caller | `theta` keys supplied | Order | Consumes `abserr`? |
|---|---|---|---|
| `LiouvilleGreen/three_bessel_integrals.py:199` (×4 phase groups) | `theta`, `theta_mod_2pi`, `theta_deriv` | 12 | no — `data["value"]` only (`:208`) |
| `ComputeTargets/QuadSourceIntegral.py` (×8, `:473`–`:605`) | `theta`, `theta_mod_2pi` | **64** | no |
| `ComputeTargets/QuadSourceIntegral.py:1013` | `theta`, `theta_mod_2pi` (`theta_deriv` commented out) | **64** | no |
| `AdaptiveLevin/tests/` (×4) | `theta` only | 12 | no |
| `docs/adaptive-levin-benchmark/levin_bench/` | varies | varies | reads `num_regions`, `num_simple_regions`, `evaluations`, `max_depth`, `num_order_changes`, `chebyshev_min_order`, `num_direct_solves`, `regions` |

Two consequences the prompts act on:

- **`QuadSourceIntegral.py` supplies `theta_mod_2pi` but not `theta_deriv`.** So it takes the
  `need_theta_Cheb` branch (θ′ by spectral differentiation of the raw phase) **and** the
  `theta_scale = TWO_PI` branch. It is the most C4-exposed caller in the tree, and the reason C4 is
  scored high rather than medium.
- **The benchmark harness reads the returned dictionary by key.** Any prompt that renames a key
  breaks `docs/adaptive-levin-benchmark/levin_bench/{runners,sweeps}.py`. Prefer adding keys; if you
  must rename, update the harness in the same commit.

---

## 3. The prompts

| # | Prompt | Audit items | Files touched | Risk |
|---|---|---|---|---|
| 01 | [`01-refuse-or-report.md`](01-refuse-or-report.md) | recs 1–4 · C1, C3a, C6, C8, C9 | `levin_quadrature.py` | Low code risk, **critical** correctness value |
| 02 | [`02-complexified-solve.md`](02-complexified-solve.md) | rec 6 · §4.1, §4.2 | `levin_quadrature.py` | Medium — numerical core |
| 03 | [`03-total-variation-gate.md`](03-total-variation-gate.md) | rec 7 · C2, C7, §2.2, §2.4 | `levin_quadrature.py` | **High** — largest structural change |
| 04 | [`04-roundoff-floor.md`](04-roundoff-floor.md) | recs 5, 10 · C4, §3.2, §3.3, §3.4 | `levin_quadrature.py` | Medium — changes every reported error bar |
| 05 | [`05-global-tolerance.md`](05-global-tolerance.md) | rec 8 · C3b | `levin_quadrature.py` | Low–Medium — changes what `atol` means |
| 06 | [`06-mode-filter.md`](06-mode-filter.md) | rec 9 · C5 | `levin_quadrature.py` | Low |
| 07 | [`07-diagnostics-hygiene.md`](07-diagnostics-hygiene.md) | recs 11, 12 · C10, C11 | `levin_quadrature.py`, benchmark harness | Low — no numerics |
| 08 | [`08-order-and-sampling.md`](08-order-and-sampling.md) | recs 13, 14 · §4.3, §4.5 | `levin_quadrature.py`, `QuadSourceIntegral.py` | Low — needs measurement |
| 09 | [`09-caller-propagation.md`](09-caller-propagation.md) | rec 15 · §3.4 | `three_bessel_integrals.py`, `QuadSourceIntegral.py`, tests, harness | Medium — **source-incompatible** |
| 10 | [`10-test-matrix.md`](10-test-matrix.md) | rec 16 · C12 + campaign verification | `AdaptiveLevin/tests/`, `docs/` | None |

## 4. Dependencies and ordering

```
01 (refuse or report)
 │
 ├─► 02 (complexify) ─► 03 (TV gate + CC) ─► 04 (eq. 151 floor) ─┬─► 05 (global atol) ─┐
 │                                                                │                     │
 │                                                                └─► 06 (p_use) ───────┤
 │                                                                                      │
 └─► 07 (hygiene, independent of all numerics) ────────────────────────────────────────►├─► 10
                                                                                        │
                                            04 ─► 08 (order + sampling) ─► 09 (callers)─┘
```

**Hard dependencies**

- **01 before everything.** The finiteness checks are the foundation: prompt 02 rewrites the solve
  and must carry them into the complex path, and prompt 03 introduces a second cell type that also
  needs them. Landing 01 first means each later prompt extends one guard rather than inventing one.
- **02 before 03.** Both rewrite `_adaptive_levin_subregion_impl`. 02 changes the *solve*; 03
  restructures the routine *around* the solve. Doing 02 first means 03's restructure lands on the
  final solve shape and its diff stays readable. Doing 03 first means writing the sample-once
  structure twice.
- **03 before 04.** 03 fixes the region taxonomy — Levin cells and Clenshaw–Curtis cells — and
  deletes the `simple_quadrature` branch with its `direct_phase_err`. 04 must define the round-off
  floor once for each surviving cell type; doing it before 03 means defining it for a branch that is
  about to be deleted, then again for its replacement. See §2.4 note 1.
- **04 before 05.** `phase_limited` arms on `phase_err > atol`. Prompt 05 changes what `atol` means
  per region; prompt 04 changes what `phase_err` is. Landing 04 first means 05 tunes a settled
  quantity.
- **04 before 06.** 06 adds the discarded-mode contribution to the region's `abserr`; 04 defines the
  `abserr` component structure it must slot into.
- **02 and 04 before 06.** 02 changes the storage layout of `p`, which is exactly what `p_use`
  indexes into.
- **04 before 08.** Equation (151) contains a `max(G₁, k²)` term where `k` is the Chebyshev order.
  Changing the default order changes every reported floor, so the order must be retuned *after* the
  floor it feeds is settled, not before.
- **04 before 09.** 09 propagates the error components that 04 defines.

**Soft dependencies**

- **07 is independent of all numerics** and could run at any point. It is placed after 06 so that
  moving `seaborn`/`matplotlib` and rewriting `_write_progress_data` lands on a settled subregion
  API — 02 and 03 both change the shape of what `_write_progress_data` re-solves.
- **08 before 09** only because 09's verification is cheaper once the order is settled. Either order
  works.
- **10 last** by construction: it asserts the final behaviour of everything above it.

**Recommended ordering: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10.**

**Natural stopping points**, if the user wants to pause and assess:

| After | State |
|---|---|
| **01** | The module refuses bad input and reports honestly. No numerical behaviour change on valid input. The single best value-for-effort point in the campaign. |
| **03** | Both **critical** findings closed (C1, C2), plus the measured speedup. The module no longer has a path to a confidently-wrong answer. |
| **04** | All critical and high findings closed (C1–C4). The reported error bar means what it says on the production code path. |
| **06** | Every audit finding C1–C9 closed. What remains is hygiene, tuning, caller plumbing and tests. |

## 5. Rules that apply to every prompt

Each prompt restates the ones it needs; they are collected here so the invariants are visible in one
place.

1. **One commit per prompt.** Do not amend or squash across prompts. The commit boundary is the
   rollback boundary.
2. **Commit message format** matches this repository's convention: an imperative, capitalised
   subject line under ~72 characters with no prefix tag; a blank line; a prose body explaining *why*,
   wrapped at ~80 columns; and the trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
3. **Every prompt writes a log** to `prompts/levin-refactor/logs/NN-<name>.md` using the template in
   §5.1, and the log is included in that prompt's commit.
4. **Every prompt updates** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) in the same commit.
5. **Do not embed a commit's own SHA in that commit.** Amending to inject a guessed SHA produces a
   new SHA, so the recorded value is always one amend stale. Leave a placeholder in your own row and
   log header; fill in *earlier* prompts' real SHAs (verified with
   `git merge-base --is-ancestor <sha> HEAD`) if your edit touches the board anyway.
   `git log --oneline -- prompts/levin-refactor/` always recovers the true mapping.
   *(This convention was learned the hard way in the `backport-modules` campaign — see that
   campaign's `IMPLEMENTATION_STATE.md` §4, `[commit-sha-links-stale]`.)*
6. **Do not fix things the prompt did not ask for.** Record them under *Observations not acted on*
   and leave the code alone. Scope creep destroys the revert-per-prompt property.
7. **Do not reintroduce the two rejected optimisations.** p-refinement before bisection (1.5–2.2×
   more integrand evaluations) and global worst-first error balancing (unsound with the step-(4)
   residual, which is a difference of two rules and not a decreasing per-region bound). Audit §2.3
   has the measurements and the reasoning; do not re-litigate them from first principles.
8. **The four existing tests in `AdaptiveLevin/tests/` must pass at the end of every prompt** unless
   the prompt explicitly says otherwise and explains why. Run them with
   `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` — there is no
   `pytest` in the venv. Baseline: 4 tests, OK, 0.008 s.
9. **Every prompt that changes numerics must record a before/after accuracy comparison** against a
   closed form (see §5.2) in its log. "It still passes the existing tests" is not sufficient
   evidence: those four tests only assert `|value − truth| < 1e-10`, which is eight orders of
   magnitude looser than the accuracy this module actually delivers.
10. **Use `./venv/bin/python` with `PYTHONPATH=.`.** The ambient `python3` does not have this
    project's dependencies.

### 5.1 Log format (mandatory)

The log has to be good enough that a later reader can tell what shipped, and *why it differs from
the prompt*, without re-deriving anything from the code. Every deviation must be classified:

- **Structurally required** — the prompt could not be implemented as written (the code was not
  shaped as the prompt assumed, a name differed, an ordering constraint forced a change). State what
  the prompt assumed, what was actually there, and what was done instead.
- **Implementation choice** — the prompt left it open and you picked. Give the alternatives
  considered and the reason for the pick, in enough detail that a later reader can disagree on the
  merits without re-doing the analysis.
- **Unintended drift** — noticed after the fact, not deliberate. Say so plainly, and say whether it
  was reverted or kept.

Template:

```markdown
# Log NN — <prompt title>

**Prompt:** prompts/levin-refactor/NN-<name>.md
**Commit:** <subject line; SHA intentionally omitted — see README §5 rule 5>
**Date:** <YYYY-MM-DD>
**Result:** COMPLETE | COMPLETE WITH DEVIATIONS | PARTIAL | BLOCKED

## What shipped
<Per item: file:line before → after. Enough that a reader knows the change without opening the diff.>

## Numerical evidence
<Required for any prompt that changes numerics (rule 9). The before/after table, the problem used,
the closed form it was checked against, and the actual numbers. Distinguish measurement from
reasoning.>

## Deviations from the prompt
<One subsection per deviation, tagged STRUCTURALLY REQUIRED / IMPLEMENTATION CHOICE / UNINTENDED
DRIFT. "None" is an acceptable and expected answer.>

## Verification performed
<Exactly what was run and what it printed. Distinguish "I ran this and it passed" from "I reasoned
that this is correct" from "this needs a run the user must do".>

## Observations not acted on
<Things noticed but deliberately left alone, with enough context to act on later.>

## State handed to the next prompt
<Anything the next prompt needs that is not already in its own text.>
```

### 5.2 Closed forms available for verification

Use these rather than inventing references. A "high-effort run of the same core" is legitimate for
comparing *policies* and **not** for establishing absolute accuracy — the audit's §7 caveats apply.

| Integral | Closed form | Where |
|---|---|---|
| `∫₁^X sin x dx`, `∫₁^X cos x dx` | trivial | already in `AdaptiveLevin/tests/` |
| `∫₋₁¹ cos(λ·atan x)/(1+x²) dx` | `(2/λ)·sin(πλ/4)` | already in `AdaptiveLevin/tests/` (Gradshteyn & Ryzhik) |
| `∫_a^b e^{−x} sin(ωx) dx` | `[−e^{−x}(sin ωx + ω cos ωx)/(1+ω²)]_a^b` | used throughout this reconciliation; the audit's §3.2/§3.3 tables use `a = 1/3, b = 7/3` |
| Three-Bessel `JJJ`/`YJJ` | seven analytic oracles | `LiouvilleGreen/tests/test_3bessel_analytic.py` |
| Stationary-phase oracle | `∫₀¹ e^{−x} sin(10⁶(x−x²)) dx = -6.879079716900e-04` | audit §1.3, by a 79 578-panel Gauss–Legendre sum; **reproduced here to 8.0e-10 by splitting the interval at x = ½**, which is independent confirmation that the oracle is right and the module is wrong |

---

## 6. Deferred and out of scope

- **`LiouvilleGreen/bessel_phase.py`, `phase_spline.py`, `range_reduce_mod_2pi.py`.** The audit
  explicitly did not examine these (§6). Two live items sit there: the prior review's §7.1
  initial-condition defect (`/ m` should be `/ sqrt(m)`) and the campaign's recommendation 2 (delete
  the bespoke range reduction). **Neither is scheduled here.** They warrant their own audit and
  campaign.
- **`phase_spline` reporting its own fit accuracy.** Prompt 04 adds the *quadrature-side* half of
  recommendation 5.2: `adaptive_levin_sincos` will accept and honour an optional `theta_abserr`.
  Nothing in `LiouvilleGreen/` computes such a number today (`phase_spline.py` has no accuracy API),
  so the parameter ships unused by production callers. Making `bessel_phase`/`phase_spline` report a
  real fit accuracy — which the audit measured at a uniform 2×10⁻⁸ relative floor across all seven
  three-Bessel oracles — is the natural follow-up, and is what would make prompt 09's propagated
  error bar tell the truth on the production path.
- **Rank-revealing QR in place of `lstsq`.** Audit §4.4 is explicit: complexification already halves
  `lstsq`, and the LU fast path already covers 75–100% of solves, so RRQR can address at most 25% of
  solves on the worst problem measured. **Do complexification (prompt 02) first, then re-measure the
  `lstsq` share on the real three-Bessel integrands before writing any pivoted-QR code.** The
  module's existing comment at `levin_quadrature.py:703-713` reaches the same conclusion and should
  stand. Prompt 02 is required to record the post-complexification `lstsq` share so that this
  decision can be made on evidence later.
- **Whether the four-phase-group decomposition in `three_bessel_integrals.py` is optimal** (audit
  §6). Not examined, not scheduled.
- **Thread/process safety under Ray** beyond the cwd-relative file writes that prompt 07 fixes.
- **`_write_progress_data` as a whole.** Prompt 07 fixes its paths, its imports and its unguarded
  denominator. It does not rewrite the 500×(m+1)-evaluations-per-region diagnostic itself; that is
  a diagnostics-only cost paid only under `emit_diagnostics=True`.
