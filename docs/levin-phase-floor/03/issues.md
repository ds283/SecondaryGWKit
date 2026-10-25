# levin-phase-floor 03 — found and not acted on

These are for the orchestrator to place on the boards. Nothing here was changed.

## I1. M3's slope clause cannot be met: the jitter-free discrepancy is at round-off from depth 10

**Proposed tag:** `[03-m3-slope-clause-is-badly-posed-at-round-off]`.

**What was measured** (`out/instrument_03.txt`, the ancestry table and M3):
- All 6380 ancestry spans are Clenshaw–Curtis, under both the real and the control phase. They are 580 sampled depth-20 regions in 29 scored groups, each with its spans at depths 10–20.
- With `theta_mod_2pi` replaced on the depth-10 span by a degree-12 polynomial fit, and nothing else changed, the median normalised discrepancy D is about 2e-16 to 1e-13 at **every** depth from 10 to 20.
  - That is the round-off of a 24/47-point nested Clenshaw–Curtis pair.
  - The real phase gives 4e-8 to 5e-6 at the same depths.
- So the control's slope of ln D on ln h over depths 12–20 is −0.02 to 0.15 per row (median over scored groups). It is < 2 on every row.
- The median depth-20 ratio real/control is 3.1e6 to 2.4e8 per row, ≥ 100 on every row.

**Why the clause is badly posed.** A discrepancy that is already at its floor at depth 10 cannot fall with h at any slope. "Slope ≥ 2" assumes algebraic convergence above a floor, and the nested pair on a smooth integrand over these spans is spectrally converged before depth 12. The clause therefore cannot separate "converges once the jitter is removed" from "does not converge". The ratio clause is the one that carries the intended comparison, and it holds by six to eight orders of magnitude.

**Impact.**
- As written, M3 fails. "The mechanism holds if M1–M4 hold" is therefore not met, and §3's clause "If M2 holds and M3 fails … the floor mechanism is refuted" fires by its letter.
- The measurement points the other way: removing the jitter alone removes the discrepancy. Moreover the control's fit residual (0.5e-6 to 3e-5 rad) equals σ_θ, and the routing is unchanged on every span.
- Per campaign README §5 rule 3, this is recorded as a badly posed rule and not re-posed.

**Next step.** The orchestrator and the user decide how the verdict is read at G1.

## I2. Every depth-20 region is a Clenshaw–Curtis fallback region

**Proposed tag:** `[03-depth20-regions-are-all-clenshaw-curtis]`.

**What was measured.**
- In the 29 scored groups of the seven rows, **189,196 of 189,196** depth-20 regions have type `direct`.
- No depth-20 region of any group, scored or not, is a Levin region (the per-group histograms in `instrument_03.txt`).
- The bisection that reaches `DEFAULT_LEVIN_MAX_DEPTH` is therefore entirely in `_adaptive_levin`'s `is_direct` branch (`levin_quadrature.py:1975-2064`).
  - There, the region's `abserr` is the nested-pair estimate. It carries the jitter at D/σ_θ ≈ 0.1.
  - Its `phase_err` (eq. 151) reaches `abserr` in only 0.85 % of the regions, so `phase_limited` is out of reach in more than 99 % of them.

**Impact.**
- Findings §2.7 layer 2, "tell the driver the input's floor", must act on the Clenshaw–Curtis branch's acceptance test. Acting only on the Levin branch would not reach these regions.
- The Clenshaw–Curtis fallback is the subject of `[10-levin-wholesale-cc-fallback]` (OPEN_ISSUES §1.3), which campaign README §0.1 puts out of scope. That is a scope question for gate G1.
- It also narrows 01's finding that "most" depth-20 regions in the expensive groups are `simple_regions`: on these rows it is all of them.

## I3. `StoreFixtures` cannot carry A3 v2's objects as they stand

**Proposed tag:** `[03-storefixtures-standin-pickle-and-exit-rows]`.

1. **`standin_pool` serialises with standard `pickle`.** `StandinRef` and `_copy` both do. The standard `pickle` cannot serialise a `GkSource`'s `NumericData`, a namedtuple bound as `_NumericData` (`ComputeTargets/GkSource.py:22`). A stand-in pool on a real store therefore fails at the first `GkSource` lookup. Ray uses `ray.cloudpickle`, which can. This prompt patched both names to cloudpickle inside its own harness (`scripts/harness.py`).
2. **`real_store_fixtures`' default `wavenumber_exit_time` rows carry `z_exit` only.** `TkNumericIntegration`'s constructor reads `z_exit_suph_e3` (`ComputeTargets/TkNumericIntegration.py:152`), so the default store cannot return a `TkNumericIntegration`. This prompt supplied the efold columns through `build_real_store(replicated=…)`.

**Impact:** none on 03's results. The next harness on a real store (04) needs the same two workarounds, or a change to `StoreFixtures/`, which is test-support code and outside 01–04's remit.

## I4. A `PhaseGroup` cannot be pickled

`f_sin` and `f_cos` close over `build_phase_groups`' local `terms_at`, which is wrapped in `functools.lru_cache` (`ComputeTargets/phase_groups.py:474-475`). `ray.cloudpickle` cannot pickle an `lru_cache` wrapper of a local function.

This prompt's `scripts/harness.py` has a reducer that pickles the wrapper as the function it wraps plus its cache size. The function is pure, so the rebuilt empty cache computes the same values. That was checked: 580 of 580 recorded region discrepancies were reproduced exactly from the pickles.

**Impact:** relevant to 04 only. No proposed tag; it is a harness note.

## I5. The prompt's "Read first" item 4 is inaccurate about `used_interval`

It says each accepted region records "the `abserr_history` and `relerr_history` of its ancestors, keyed by depth". `used_interval` (`levin_quadrature.py:450-640`) has no such fields. Only the internal `_levin_interval` carries them, and they are not kept when a region is accepted.

Step 7's ancestry recomputes the ancestors' discrepancies with the driver's own routine, so nothing was lost. This is a correction for the prompt's record only.
