# Review of the Liouville–Green construction for the tensor Green's function

**Date:** 2026-09-09. **Tree inspected:** `f06f587` (branch `claude/gk-wkb-integration-review-029104`).
**Files:** `ComputeTargets/GkWKBIntegration.py`, `ComputeTargets/WKB_Gk.py`,
`Quadrature/integrators/WKB_phase_function.py`, `LiouvilleGreen/WKBtools.py`,
`LiouvilleGreen/phase_spline.py`; consumers `ComputeTargets/GkSource.py` (phase rectifier) and
`ComputeTargets/GkSourcePolicyData.py:587-711` (line numbers as on `main` at `d249baa`; 14 lower at `f06f587`). §12 extends the review to the transfer-function
analogue: `ComputeTargets/TkWKBIntegration.py`, `ComputeTargets/WKB_Tk.py`, `TkNumericIntegration.py`
and the consumer `ComputeTargets/TkSourceFunctions.py`.
**Out of scope:** the numeric→WKB hand-over (where it sits, how wide the overlap is, continuity across
it). Everything below starts inside the WKB regime with specified initial data.

**Related prior work.** `docs/gk-wkb-numerical-review-2026-09.md` (commit `39ed7fc`, branch
`bessel-remedial-plan`, 2026-09-08) reviewed the same code on an exact-radiation control. Its central
measurements are reproduced here (§2, §4, §5) and extended to the real Planck2018 background, to the
production phase spans, to the QCD production model and to a concrete replacement measured on the real
background (§7). Where this document differs from it, that is stated.

Reproduction, from the repository root:

```sh
export PYTHONPATH=.; PY=./venv/bin/python; D=docs/gk-wkb-scheme-review-2026-09
$PY $D/t1_span.py; $PY $D/t2_solver.py; $PY $D/t3_dense.py; $PY $D/t4_primitive.py
$PY $D/t4b_production_real.py 1e5; $PY $D/t4b_production_real.py 3e8
$PY $D/t5_spline.py; $PY $D/t6_sweep.py; $PY $D/t7_jitter.py; $PY $D/t8_qcd.py
$PY $D/tn1_numeric_rad.py; $PY $D/tn2_numeric_real.py
$PY $D/tk1_span_residual.py; $PY $D/tk2_production_real.py 1e5; $PY $D/tk2_production_real.py 3e8
$PY $D/tk3_radiation_LG.py; $PY $D/tk5_numeric_rad.py; $PY $D/tk5b_atol.py; $PY $D/tk6_dense_real.py
```

Python 3.12, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0. No Ray, no datastore: the production functions
`integrate_phase_function`, `stage_2_evolution`, `shift_theta_sample` and `phase_spline` are called
directly with a duck-typed model. References are mpmath (40 digits) on the same floating-point inputs.

---

## 0. Summary

1. **The formulas are right; the numerics do not deliver what the comments promise.** `WKB_Gk.py`
   implements the correct effective frequency for the friction-removed equation, and `store()`
   implements the correct LG amplitude and initial-data algebra (§1). But the phase solver's error is
   a fixed fraction of the *accumulated phase*, ~1e-9 at production tolerances, and production phase
   spans are 1.4e9–4.1e12 rad (§2). Measured on the real Planck2018 background with the production
   code path: **13.9 rad** error at z=0.1 for k=1e5/Mpc and **7.4e3 rad** for k=3e8/Mpc (§4).

2. **The `Q` change of variables is algebraically exact but numerically counter-productive.** Q is
   not "close to unity" (it reaches −224 for k=1e5 and −11069 for k=3e8); the phase error is
   ω_i(1+u)δQ, so the tolerance on Q buys nothing; stage 1's reset scheme is 40–300× more accurate at
   the same tolerance, and Q additionally exposes DOP853's dense-output error, amplified by the same
   factor: for constant ω the reported samples are off by **0.33 rad** while the step endpoints are
   accurate to 1.7e-7 (§3). What Q does buy is cost: stage 1 costs ∝ span, 2.5e6 RHS evaluations
   (64 s) per (k, z_source) object at k=3e8.

3. **`phase_spline` chunking does nothing useful and does measurable harm** (§5): interpolation error
   is bit-for-bit identical with and without it (it is h⁴x/384 either way), while chunk rebasing
   inflates the spline ordinates 64× above the data span, worsens knot residuals 30–50×, and
   introduces a 1.4e-4 rad discontinuity at the chunk switch. The comment in `GkSourcePolicyData`
   that chunking "keeps the rebasing well-conditioned over many cycles" is false.

4. **The phase is, to 1e-8 rad, k times conformal time.** On the LambdaCDM background the entire
   departure of the LG phase from k[τ(z_i) − τ(z)] over the whole WKB range is ≤ 2.5e-7 rad; on the
   QCD production model it is ≤ 1.5e-3 rad, the same size as the LG truncation error itself (§6).
   The current scheme therefore re-solves, per (k, z_source), an ODE whose answer is a
   k-independent function of z times k. A shared τ primitive built by Gauss–Legendre on the existing
   grid reaches the double-precision floor (6e-15 relative) in 0.02 s and 3.7k Hubble evaluations
   for the whole range (§7). This removes stages, resets, Q, per-object solves, the cross-object
   cycle-count rectifier and the growing-phase spline in one step, and makes phase groups
   θ_G ± θ_q ± θ_r cancel exactly in their leading term.

5. **Physical floor.** The LG truncation error (the term the WKB criterion |ω'|/ω² does not measure)
   is ~1e-8 rad in LambdaCDM and ~5e-4–1e-3 rad in the QCD model for k ≥ 1e8/Mpc (§6). Numerical
   accuracy targets tighter than this are wasted unless the LG order is raised.

6. **The numeric region is sound.** `GkNumericIntegration` reproduces G and G′ to 2e-7 of the
   envelope at production tolerances for 0.1 s per object, independently of where the source sits;
   only cleanup is recommended, chiefly removing a per-step diagnostic that costs 45 % of the run
   and has no consumer (§10).

7. **Relation to the prior review's plan.** Its per-k local-quadrature primitive is the right
   design and §7 is that design; its "optional" leading-term split turns out to be the whole
   answer once τ accuracy and residual size are measured, and the comparative ODE-versus-quadrature
   study it calls for is settled by the numbers in §2 and §7 (§11).

8. **The transfer-function WKB shares the defects and the fix, with three adjustments** (§12). It uses
   the same phase solver: 2.0 rad (k=1e5) and 5.1e3 rad (k=3e8) error at z=0.1 on the real background,
   with a late-time jump that is stepping error of the Q equation itself. Its phase is the
   sound-horizon integral k∫c_s dz/H to within −0.09 rad on both models, so a second k-independent
   table replaces its solver, and a third replaces its friction ODE. Not shared: it is one object per
   k (no cross-object stitching), its LG truncation floor is ~1e-4 of the envelope at the production
   hand-over (Gk's is zero in radiation), and its numeric run is limited to 1e-5 of the envelope by
   a mis-scaled absolute tolerance, not by the solver.

Smaller findings: the `sin_coeff` "sign fix" is a provable no-op; the zero-length check compares a
redshift to the ODE tolerance; stage 1's event restart passes a 1-element array into `math.fmod` and
will break on a future NumPy; several comments are stale (§8). The per-object `div_2pi` rebase in
`shift_theta_sample` produces ±1-cycle inconsistencies between neighbouring objects, but the
`GkSource` rectifier repaired every one in 4,000 simulated objects (§8.3).

---

## 1. What is computed, and what is correct

With s = 1+z and ε = s H'/H, `GkNumericIntegration` integrates
G'' + (ε/s)G' + [k²/H² + (ε−2)/s²]G = 0. Removing the friction term with G = √(H_i/H) Y gives
Y'' + ω²Y = 0 with

ω² = k²/H² − ε'/(2s) + (3ε/2 − ε²/4 − 2)/s²,

which is what `Gk_omegaEff_sq` evaluates (checked by hand: −p'/2 − p²/4 with p = ε/s gives exactly
the extra ε/2 − ε²/4 in the s⁻² coefficient). The LG solution is
G = √(H_i/(Hω)) [A sin θ + B cos θ], θ(z) = ∫_{z_i}^z ω dz' (negative for z < z_i). `store()`
fixes (A, B) from (G_init, G'_init) correctly, and the rotation into a single sine with
δ = atan2(raw_cos, raw_sin), amplitude B = hypot(raw_cos, raw_sin) is exact. In exact radiation
(H ∝ s², ε = 2) the s⁻² coefficient vanishes, ω = k/s², θ = k(1/s_i − 1/s), and the LG solution is
the exact Green's function G = (s_i²/k) sin θ. Every control below uses this.

Two facts about the production geometry (`main.py`, `t1_span.py`) frame everything else:

| k [1/Mpc] | z_exit | z at 3 e-folds | x = k/(aH) at z=0.1 | phase, 3-e-fold point → z=0.1 | ε·θ (FP floor) | 1e-8·θ |
|---|---|---|---|---|---|---|
| 1e5 | 4.6e10 | 2.3e9 | 4.6e8 | 1.373e9 rad | 3e-7 rad | 14 rad |
| 1e6 | 4.6e11 | 2.3e10 | 4.6e9 | 1.373e10 | 3e-6 | 140 |
| 1e7 | 4.6e12 | 2.3e11 | 4.6e10 | 1.373e11 | 3e-5 | 1.4e3 |
| 1e8 | 4.6e13 | 2.3e12 | 4.6e11 | 1.373e12 | 3e-4 | 1.4e4 |
| 3e8 | 1.4e14 | 6.9e12 | 1.4e12 | 4.118e12 | 9e-4 | 4.1e4 |

So (i) even a perfect double-precision representation of the absolute phase at z=0.1 is uncertain
by up to 1e-3 rad, and (ii) any solver whose error is a fixed fraction rtol of the state cannot
resolve a cycle. Note also the structure of the work: one `GkWKBIntegration` object per
(k, z_source), each integrating its own phase from z_source (or from the numeric hand-over z_init)
down to z_end = 0.1; ~1300 source redshifts per k at 100/decade.

---

## 2. Solver accuracy on the exact radiation control (`t2_solver.py`)

`integrate_phase_function` on H = s², x_init = 30, z_end = 0.1, response grid 100/12 per decade
(the production response sparseness) and 100 per decade; error against mpmath at every sample.

| span [rad] | grid/dec | (rtol, atol) | path | max δθ [rad] | δθ/span | RHS evals (stage 1 + 2) | resets |
|---|---|---|---|---|---|---|---|
| 1e5 | 8.3 | (1e-8, 1e-10) | 1→2 | 9.0e-5 | 9.0e-10 | 853 + 353 | 1 |
| 1e5 | 8.3 | (5e-14, 1e-16) | 1→2 | 8.7e-10 | 8.7e-15 | 1669 + 941 | 1 |
| 1e7 | 8.3 | (1e-8, 1e-10) | 1→2 | **9.7e-3** | 9.7e-10 | 1582 + 773 | 10 |
| 1e7 | 100 | (1e-8, 1e-10) | 1→2 | 9.7e-3 | 9.7e-10 | 1639 + 812 | 10 |
| 1e7 | 8.3 | (1e-11, 1e-13) | 1→2 | 3.0e-5 | 3.0e-12 | 1840 + 1061 | 10 |
| 1e7 | 8.3 | (5e-14, 1e-16) | 1→2 | 2.1e-7 | 2.1e-14 | 3106 + 1973 | 10 |
| 1e7 | 8.3 | (1e-8, 1e-10) | stage 1 forced | 2.6e-4 | 2.6e-11 | 65947 | 1000 |
| 1e7 | 8.3 | (5e-14, 1e-16) | stage 1 forced | 8.2e-9 | 8.2e-16 | 108655 | 1000 |
| 1e9 | 8.3 | (1e-8, 1e-10) | 1→2 | **0.98** | 9.8e-10 | 7947 + 1169 | 104 |
| 1e9 | 8.3 | (5e-14, 1e-16) | 1→2 | 1.6e-4 | 1.6e-13 | 13026 + 2873 | 104 |

Observations. The error is a fixed fraction of the span at fixed tolerance (9e-10 at rtol=1e-8),
independent of the output grid density, so denser sampling cannot help. Stage 1 alone is 37× more
accurate than the two-stage path at the same tolerance and reaches the floating-point floor at tight
tolerance, at 28–35× the cost. Tightening the two-stage path to (5e-14, 1e-16) costs 2.5× and
reaches 2e-14 relative at 1e7 rad, but degrades to 1.6e-13 relative at 1e9 rad (the 104 stage-1
segments' errors accumulate); extrapolated to the production span of 4e12 rad this is still ~1 rad.
No tolerance setting rescues the absolute phase at production spans.

The prior review's two-stage figures (9.35e-3 rad at a 1e7 span, 0.955 rad at 1e9) are
reproduced to within 5%.

---

## 3. The `Q` variable (`t3_dense.py`, `t2_solver.py`)

Stage 2 writes θ = θ_i + ω_i(1+u)Q, u = z_i − z, and integrates
Q' = −[ω/ω_i + Q]/(1+u), Q(0) = 0. The comment says "Q will typically be fairly close to unity". Exactly,
Q = −(1/(ω_i(1+u)))∫_0^u ω dt = −⟨ω⟩/ω_i, and since H falls steeply, ⟨ω⟩ ≫ ω_i: production runs give
Q_min = −224 (k=1e5) and −11069 (k=3e8) (§4). Three consequences:

- **The tolerance protects the wrong quantity.** δθ = ω_i(1+u)δQ; the solver controls
  `atol + rtol·|Q|` per step, so the phase error scales with the span exactly as for a direct θ
  integration, and worse than stage 1's rebased θ (whose state is bounded by 1e4 rad, so its local
  error scale is ≤ 1e-4 rad per step). This is the 37× of §2 and the 300× of §4.
- **Dense output is amplified.** `t_eval` values come from DOP853's interpolant, not from accepted
  steps. For constant ω = 1e7 (θ linear, Q = −u/(1+u) rational), from u=0 to 98.9 at
  (1e-8, 1e-10): 26 steps; max error **1.7e-7 rad at step endpoints, 0.105 rad at step midpoints,
  0.33 rad on a fine grid**; at rtol=1e-12 still 1.7e-4 rad on the fine grid. A direct θ
  integration of the same problem: 7 steps, 2.5e-6 rad everywhere. In the radiation control the
  accumulated step error (9.3e-3 rad) happens to dominate the interpolation error (7.1e-3), so the
  trap is hidden there; it is not hidden in general, and it means "check the final point" misses it.
  This reproduces the prior review's 0.337 rad and identifies the mechanism.
- **Coordinate round-trip.** z is recovered as z_init − u; the error is half an ulp of z_init. In
  production stage 2 starts where ω² = 1e6, i.e. z_init ≈ 6.1e3 (k=1e5) to 3.7e5 (k=3e8), so the
  worst case is 3e-11, inside the hard-coded 1e-10 assertion at
  `WKB_phase_function.py:359` by a factor of three. Safe today; a `DEFAULT_OMEGA_WKB_SQ_MAX` of
  1e5 or a larger k would trip it.

What Q achieves: cost. Stage 1's reset scheme spends ~66 RHS evaluations per 1e4 rad; stage 2 spends
~1e3 evaluations regardless of span. Q is a cost optimisation that was described as an accuracy
mechanism.

---

## 4. The production path on the real background (`t4b_production_real.py`)

Planck2018 LambdaCDM with analytic ε, ε', ε'' (the code path `BackgroundModel` takes when the
cosmology supplies them), `integrate_phase_function` from the 3-e-fold point to z=0.1 on the
production response grid, production tolerances. Reference: mpmath quadrature of k/H plus the
rationalised residual ∫ C/(ω + k/H) dz (§6).

**k = 1e5/Mpc** (z_e3 = 2.31e9; stage 1: 46702 evaluations, 676 resets; stage 2: 1271; 1.3 s):

| z | θ_ref [rad] | error [rad] | relative |
|---|---|---|---|
| 1.75e9 | −6.39 | 6.8e-9 | 1.1e-9 |
| 1.24e7 | −3.73e3 | 7.9e-7 | 2.1e-10 |
| 1.68e4 | −2.64e6 | 1.6e-4 | 6.1e-11 |
| 3213 (stage 2) | −1.18e7 | −2.5e-3 | 2.1e-10 |
| 21.7 | −3.07e8 | −0.17 | 5.5e-10 |
| 0.1 | −1.3728e9 | **13.9** | 1.0e-8 |

**k = 3e8/Mpc** (z_e3 = 6.92e12; stage 1: **2,538,351 evaluations, 37,301 resets**; stage 2: 1883;
**63.7 s for one object**):

| z | θ_ref [rad] | error [rad] | relative |
|---|---|---|---|
| 7.8e8 | −1.78e5 | 1.5e-4 | 8.3e-10 |
| 1.05e6 (end of stage 1) | −1.32e8 | 1.8e-4 | **1.3e-12** |
| 1.16e5 (stage 2) | −1.19e9 | −0.35 | 2.9e-10 |
| 1416 | −6.90e10 | −70 | 1.0e-9 |
| 16.3 | −1.07e12 | 292 | 2.7e-10 |
| 0.1 | −4.1184e12 | **7366** | 1.8e-9 |

Stage 1 delivers 1.3e-12 relative on the real background; the moment stage 2 takes over the relative
error jumps to 3e-10 and stays there. Below z ≈ 1e3 the stored phase for the largest k is wrong by
tens to thousands of cycles. Since `G_WKB` is built from `theta_mod_2pi`, these stored values are
noise in that region.

Cost. For k=3e8 the ~730 source redshifts above the stage-1/2 switch each cost ~2.5e6 evaluations:
~1.8e9 evaluations, ~13 CPU-hours, for one k. The cost is O(N_source × phase span) for what §6 shows
is a single k-independent quadrature.

---

## 5. `phase_spline` chunking (`t5_spline.py`)

Consumer geometry, as in `GkSourcePolicyData._create_functions`: θ(z_source) at fixed z_response,
exact radiation, sources s = 10…1e4 (x_source = k/10…k/1e4), `x_is_redshift=True`,
`increasing=False`, `chunk_logstep=125` versus `None`. Error evaluated at 10 points per interval
against mpmath, excluding three intervals at each end.

| k | grid/dec | chunks | interior max δθ | incl. ends | predicted h⁴x_max/384 | max spline ordinate | knot residual | switch jump |
|---|---|---|---|---|---|---|---|---|
| 1e6 | 100 | 1 | 8.26e-5 | 7.7e-4 | 7.3e-5 | 9.99e4 rad | 5.8e-11 | — |
| 1e6 | 100 | 2 | 8.26e-5 | 7.7e-4 | 7.3e-5 | **6.87e6 rad** | 3.0e-9 | −3.8e-7 rad |
| 1e8 | 100 | 1 | 8.26e-3 | 7.7e-2 | 7.3e-3 | 9.99e6 | 9.3e-9 | — |
| 1e8 | 100 | 2 | 8.26e-3 | 7.7e-2 | 7.3e-3 | **6.44e8 rad** | 2.7e-7 | **−1.4e-4 rad**, δθ'/θ' 3.3e-8 |
| 1e8 | 300 | 1 | 1.07e-4 | 9.7e-4 | 9.0e-5 | 9.99e6 | 7.5e-9 | — |
| 1e8 | 300 | 2 | 1.07e-4 | 9.7e-4 | 9.0e-5 | 6.44e8 | 2.5e-7 | −3.5e-8 rad |

Conclusions:

- **Interpolation error is unchanged by chunking** and follows h⁴x_source/384 (the follow-up
  document's law, here with the local source x, which is the right variable for this consumer).
  At the production x_source of 1e9–1e12 the 100/decade cubic spline is meaningless.
- **Chunking makes the ordinates larger, not smaller.** `_build_log_chunks_negative` places
  boundaries at multiples of 125 cycles (0, 126, 11751, 1101626, …) and `_chunk_spline` rebases to
  the chunk's *far boundary*, not to a sample. The data span 1.6e6 cycles; the chunk holding them is
  (−1101626, −8813) and its ordinates reach 6.4e8 rad, 64× the global span. A single global rebase
  (what `chunk_logstep=None` does) already bounds ordinates by the span.
- **Chunk selection is a hard switch** between two independent splines (`_match_chunk`'s
  distance-to-centre penalty); the two disagree by 1.4e-4 rad at 100/decade. A Levin consumer of
  `theta_deriv` sees the derivative jump.
- `_build_log_chunks_positive` has no progress guard: start=1 with `chunk_logstep` < 2 gives
  end=round(logstep+1)=2, next start=round(1.0)=1, forever (by inspection; production uses 125,
  which progresses).
- The claim in log 05 of "≤125 cycles per chunk" and the comment at `GkSourcePolicyData.py:676-680`
  are both false. The `MINIMUM_SPLINE_DATA_POINTS` merge removes any bound on chunk span.

Chunking should be removed (`chunk_step=None, chunk_logstep=None`), which makes `phase_spline` a
plain cubic spline of a globally rebased phase. That does not fix the interpolation error, which
needs a different representation (§7).

---

## 6. The phase is k·Δτ plus a tiny residual (`t1_span.py`, `t8_qcd.py`)

Write ω² = (k/H)² + C(z) with C = −ε'/(2s) + (3ε/2 − ε²/4 − 2)/s². Then exactly

θ(z; z_i) = k[τ(z_i) − τ(z)] + ρ_k(z; z_i),  ρ_k = ∫ C / (ω + k/H) dz,  τ = ∫ dz/H,

the rationalised form avoiding the subtraction ω − k/H. C vanishes identically in radiation (ε=2,
ε'=0) and is O(1)/s² in matter and Λ eras, where H/k ≪ 1, so ρ ~ ∫ C H/(2k) dz = O(1/x).

| model | k [1/Mpc] | ρ over the whole WKB range (3 e-folds → z=0.1) |
|---|---|---|
| LambdaCDM | 1e5 | −2.5e-7 rad |
| LambdaCDM | 1e6 | −2.4e-8 rad |
| LambdaCDM | 3e8 | −5.1e-11 rad |
| QCD_Cosmology | 1e7 | 3.0e-5 rad |
| QCD_Cosmology | 1e8 | −1.45e-3 rad |
| QCD_Cosmology | 3e8 | −1.26e-3 rad |

In the QCD model ε departs from 2 by up to 0.150 at z ≈ 1.2e12 (the QCD transition), C(z)s² ≈
−0.0087 there, and modes k ≥ 1e8 are at x = 90–270 during the transition, hence the 1e-3 rad
residual; for k=1e7 the transition occurs at x ≈ 9, inside the numeric region. (ε here is a quintic
spline derivative of ln H at 2000 points per decade, since `QCD_Cosmology` supplies no analytic
`d_lnH_dz`.)

**LG truncation.** Y = ω^{-1/2} sin∫ω satisfies Y'' + ω²Y = RY with
R = ¾(ω'/ω)² − ½ω''/ω ≈ (ε²/4 − ε/2)/s² + ε'/(2s) deep inside the horizon. The next-order phase
correction ∫ R/(2ω) dz is: LambdaCDM, same order as ρ (~1e-8 rad); QCD, **−5.3e-4 rad** (k=1e7),
**−1.1e-3** (1e8), **−4.4e-4** (3e8). This is the physical floor of the current representation for
the QCD model. The WKB diagnostic |ω'|/ω² is 0.067 at the 3-e-fold point in radiation while R = 0
there, so it neither measures nor bounds this error. The prior review's constant-w figures
(2.3e-3 of envelope at w=0.2, 1.2e-2 at w=0 from x_s=30) are the same effect in a regime the
production LambdaCDM run never enters, because production k re-enter at z ≳ 5e10 in pure radiation.

**Consequences.** (a) For LambdaCDM, θ = kΔτ to 3e-7 rad over the whole range; ρ is negligible.
(b) For QCD, ρ is 1e-3 rad and smooth, and can be quadratured to 1e-6 absolute at negligible cost.
(c) A numerical phase target below ~1e-3 rad (QCD) buys nothing physical without a higher-order LG
treatment; but the current *numerical* errors (§4) are 1e4–1e7 times larger than that floor.
(d) The leading term is k times a k-independent function, so it should be computed once.

---

## 7. A shared τ primitive, measured (`t4_primitive.py`)

Real LambdaCDM background, the production grid (100 per decade of 1+z from z=2.3e9 to 0.1, 1040
intervals), Gauss–Legendre per interval in u = log(1+z), cumulative sum with `math.fsum`, checked
against mpmath at nine nodes.

| Gauss order | Hubble evaluations (total) | time | max relative error of τ at nodes |
|---|---|---|---|
| 4 | 3,732 | 0.02 s | 5.7e-15 |
| 8 | 7,464 | 0.02 s | 5.9e-15 |
| 12 | 11,196 | 0.03 s | 5.9e-15 |

Order 4 is already at the double-precision floor. Off-grid evaluation matters as much as the nodes:

| off-grid method (25 random points) | relative error | as phase at k=1e5 | at k=3e8 |
|---|---|---|---|
| nearest node + local 8-point Gauss | **2.1e-16** | 3e-7 rad | 8.6e-4 rad |
| quintic spline of the nodes | 1.8e-14 | 2.5e-5 rad | 0.07 rad |
| cubic spline of the nodes | 1.4e-9 | **1.9 rad** | 5.4e3 rad |

So a spline of τ is not an acceptable route (a cubic reproduces the h⁴ problem in a new place; this
is the caveat the prior review raised about `model.functions.tau`, now quantified); local
quadrature from an anchor node is. The residual absolute error is the ε·kτ floor of §1, which is
irreducible for the *absolute* phase in double precision and irrelevant for phase *differences*
formed from interval sums over short baselines.

**What this replaces.** θ(z_r; z_s) = k[τ(z_s) − τ(z_r)] + ρ_k(z_r; z_s) for every k, z_source and
z_response, from one τ table per background model (and, for T_k, one sound-horizon table
∫ c_s dz/H). Compared with the current scheme:

- No ODE, no stage 1/2 switch, no phase resets, no Q, no `t_eval` interpolation, no per-object
  solve: 3.7k Hubble evaluations per model instead of ~1e9 per k (§4).
- Phases for different z_source are consistent by construction, so `shift_theta_sample`'s rebase
  and `GkSource`'s 2π rectifier (§8.3) are unnecessary: the initial-data offset δ from a numeric
  hand-over is simply added.
- The consumer need not spline a growing phase: it evaluates θ from τ (and a spline of the small ρ,
  whose derivatives are O(1e-3)). The h⁴x/384 problem and all of `phase_spline`'s chunking disappear
  for G. `theta_deriv` is −ω(z_s) in closed form.
- Phase groups become θ_G − θ_q − θ_r = (k − q − r)Δτ + (ρ_k − ρ_q − ρ_r): the leading cancellation is
  done on wavenumbers in exact arithmetic, and the error in τ enters multiplied by |k − q − r|, not
  k. Independent per-mode solves cannot cancel their errors this way. This is the cosmological
  analogue of the Bessel campaign's θ = x + c_ν + r_ν design, with kτ in the role of x.
- Retained unchanged: `Gk_omegaEff_sq`, the LG amplitude √(H_i/(Hω)), the (B, δ) algebra, the
  `GkWKBValue` schema (div/mod pairs can still be emitted from the primitive if consumers need them).

Not measured here and needed before adoption: per-interval Gauss accuracy for ρ on the QCD
background, whose ε' comes from a spline with knots (the mpmath/`quad` references here hit
subdivision limits only from dynamic range, not roughness, but this should be checked on the grid);
the T_k sound-horizon analogue; and the hand-over offset δ, which is out of scope.

---

## 8. Smaller defects and traps

### 8.1 Dead and misleading logic in `GkWKBIntegration.store()`

- **The sign fix is a no-op.** With G_init = norm·B·sin δ exactly (norm > 0), sgn(sin δ)·sgn(G_init) =
  +1 whenever G_init ≠ 0, and the code's `>= 0` conventions give +1 when G_init = 0. `sin_coeff` is
  always +B. Confirmed on 4,000 simulated objects (`t6_sweep.py`, "sign-fix ≠ +1 for 0 objects").
  It can be deleted.
- `shift_theta_sample` subtracts the first sample's wrap shift from every `div_2pi` "to cut down
  unnecessary shifts". Within one object this is harmless; across objects it is the source of the
  ±1-cycle inconsistencies in §8.3. It should not be done (store θ + δ exactly).

### 8.2 `WKB_phase_function.py`

- **Zero-length check** (`:671`): `fabs(z_init − z_sample.min.z) < atol` compares a redshift interval
  with the ODE tolerance. Unreachable in production (grid spacing ≫ 1e-10 and the constructor
  forbids samples above z_init), so the prior review's −8.26 rad example cannot occur there; but it
  is the wrong kind of test and costs nothing to make exact.
- **NumPy deprecation.** On a phase reset, `sol.y_events[0][0]` is a 1-element array and is passed
  to `WKB_mod_2pi` → `math.fmod`. With `DeprecationWarning` promoted to an error this raises
  "Conversion of an array with ndim > 0 to a scalar is deprecated … will error in future"
  (`t9_warn.py`, NumPy 2.2.4). Every production stage-1 run with a reset hits this path. Use `values[0][0]`
  or `float(...)`.
- **Two terminal events in one step**: the recycle branch wins and stage 1 continues past
  ω² = 1e6 with no terminate event ever firing again (harmless, but undocumented).
- Stale comments: `:403` "cutting the integral after we've stepped about 1E3 in redshift" (it is
  1e4 rad of phase); `:262-264` "Q … fairly close to unity"; `main.py:424-427` describes the Bessel
  Q with the same premise.

### 8.3 Cycle-count consistency across objects (`t6_sweep.py`)

`GkSourcePolicyData` splines θ(z_source) through samples drawn from independent
`GkWKBIntegration` objects, and `GkSource.py:166-233` "rectifies" their `div_2pi` with a heuristic
that fires only when θ fails to be monotone and then picks the nearest of {0, ±1} cycles. Simulated
in exact radiation for the numeric-initialised band z_source ∈ [√(z_e3 z_e4), z_e3] with the
production grids, `store()`'s (B, δ) algebra, `shift_theta_sample`, and a faithful copy of the
rectifier, for 15 k values × 3 response points × two stop conventions (stop at an extremum of G,
δ = ±π/2; stop at a zero, δ = 0 or π):

- 60 of 330 objects (extremum) and 165 of 330 (zero) carried a −1 rebase offset; none carried +1.
- Every offset, and every 2π wrap of δ between neighbours, was in the monotonicity-violating
  direction and was repaired: **0 kinks** in the rectified θ(z_source), max deviation 2e-13 rad.

So the bookkeeping works in the tested geometry. It works because the phase increment per source
sample in that band is 0.5–0.8 rad < π; the rectifier has no way to detect a jump of the opposite
sign (a −2π step is monotone), which the simulation shows does not arise from these two mechanisms
but would from any third. The shared primitive (§7) removes the need for it.

### 8.4 Per-object solve jitter (`t7_jitter.py`)

148 independent production-tolerance solves of θ(z_r = 0.1; z_s), k = 1.1e7, x_source 1e3–3e4:
max error 9.7e-3 rad but neighbour-to-neighbour jitter only 2.9e-5 rad (the solves share most of
their path, so errors are correlated). Relative roughness 6e-7; the consumer spline's `theta_deriv`
degrades from 1.2e-9 (exact samples) to 7.7e-7 relative. Minor: the phase error is a smooth bias,
not noise.

---

## 9. Answers to the questions posed

**Does the change of variables to Q achieve its purpose?** No. Its stated purpose is accuracy of θ
mod 2π via a bounded state; it delivers cost reduction at a 40–300× accuracy penalty relative to
stage 1 (§2, §4), because the phase error is ω_i(1+u)δQ and Q itself grows to −1e4. Traps: DOP853
dense-output error amplified by the same factor (0.33 rad for a linear phase, §3); the
z = z_init − u round-trip, safe by 3× today; the false premise Q ≈ −1 in every comment.

**Does `phase_spline` chunking achieve anything?** No (§5). Interpolation error is unchanged;
ordinates grow 64×, knot residuals 30–50×, and a 1.4e-4 rad switch discontinuity appears. Remove it.

**Major sources of error**, in production order of magnitude at z=0.1: solver error ∝ rtol·span,
14–7400 rad (§4); consumer cubic spline of the growing phase, h⁴x/384, 7e-3 rad at x=1e7 rising to
O(1) at x ≥ 1e9 (§5); floating-point floor ε·kτ, 3e-7–9e-4 rad (§1); LG truncation, 1e-8 rad
(LambdaCDM) or 1e-3 rad (QCD, k ≥ 1e8) (§6). Amplitude, jitter and cycle bookkeeping are minor
(§8).

**Can the logic be simplified?** Yes, structurally: the LG phase is k·Δτ + ρ with ρ ≤ 1.5e-3 rad
(§6), so one Gauss–Legendre τ table per model (0.02 s, floating-point-floor accuracy, §7) replaces
the two-stage ODE, the resets, Q, the per-object solves, the cross-object rectifier and the
growing-phase spline. The frequency, amplitude and initial-data algebra stay as they are.

---

## 10. The numeric region (`GkNumericIntegration`, `numeric_with_phase_cut`)

Added 2026-09-09 after the WKB review. Scripts `tn1_numeric_rad.py` (exact radiation) and
`tn2_numeric_real.py` (real Planck2018 background). The integrator is called directly with
`mode="stop"`, the production response grid (100/12 per decade, truncated to 0.85·z_e6) and the
production stop window (z_e3, z_e6), from source redshifts between 5 e-folds outside the horizon
and 3.9 e-folds inside.

### 10.1 Accuracy and cost

Exact radiation, G = (s_s²/k) sin θ, errors relative to the local envelope over all returned samples:

| x_source | (atol, rtol) | RHS evals | time | max δG/env | max δG'/env' | stop at x | G_stop/env | δG_stop/env |
|---|---|---|---|---|---|---|---|---|
| e⁻⁵ | (1e-10, 1e-8) | 12767 | 0.09 s | 2.3e-7 | 1.7e-7 | 23.6 | +1.000000 | 1.3e-8 |
| e⁻⁵ | (1e-13, 1e-11) | 28550 | 0.19 s | 2.1e-10 | 1.5e-10 | 23.6 | +1.000000 | 9.8e-12 |
| 1 | (1e-10, 1e-8) | 12026 | 0.08 s | 2.1e-7 | 1.0e-7 | 24.6 | +1.000000 | 7.2e-9 |
| e^3.9 | (1e-10, 1e-8) | 10013 | 0.08 s | 2.3e-7 | 5.6e-8 | 54.1 | +1.000000 | 1.5e-10 |

The error is set by `rtol` and is independent of where the source sits; tightening `rtol` by 10³
costs 2.3× and buys 10³. On the real background (k = 1e5 and 3e8/Mpc, source 5 e-folds outside)
the run costs 12.8k evaluations and agrees with the radiation oracle `compute_analytic_G` to
1e-8–4e-7 relative, the residual being the matter fraction Ω_m/(Ω_r(1+z)) ≈ 7e-8 at z ≈ 1e10 rather
than solver error. Nothing here needs changing for accuracy: 2e-7 of the envelope at 0.1 s per
object is a sound conventional integration, and the consumer's cubic spline of the numeric G in
log(1+z) (follow-up document §1.1: 1e-5 to 1e-4 near the hand-over) is the larger error by two
orders.

### 10.2 Observations and small recommendations

- **The per-RHS diagnostic costs 45 % of the run and is never consumed.** `RHS` evaluates
  `Gk_omegaEff_sq` (three extra background calls) at every step to feed
  `report_wavelength`; measured 0.13 s with it against 0.09 s without on LambdaCDM (it will be
  worse on the QCD model, whose ε comes from splines). `has_unresolved_osc` is stored but no
  consumer reads it, and `store()` already computes `omega_WKB_sq` and `WKB_criterion` on the
  sample grid. Remove the call from the RHS.
- **`delta_logz` is used as Δln(1+z) but supplied as Δlog₁₀(1+z)** (`main.py` passes
  `1/source_samples_per_log10z`; `report_wavelength` forms `(1+z)·delta_logz`). The grid spacing is
  under-estimated by ln 10 = 2.3, so the "unresolved oscillation" flag fires at x ≳ 630 rather than
  the intended x ≳ 273. Since the integration stops at z_e6 (x = 403) the flag never fires in
  production, which is why every object reports `False`.
- **The stop point is a maximum, not a minimum.** `find_phase_minimum` looks for G' changing from
  negative to positive as z decreases; at that point G_stop/env = +1.000000 in every run. The
  label and the comment "cutting at a point of fixed phase where G' = 0 at a minimum" are wrong,
  but nothing depends on which extremum is chosen: `store()` in the WKB object rotates arbitrary
  (G, G') into a pure sine, so the "fixed phase to avoid jitter" motivation is also obsolete.
- **The search step is safe only inside the window.** `find_phase_minimum` steps 1e-3 in relative
  z, i.e. 2π/(1e-3·x) samples per cycle: 15 per cycle at x = 403, fewer than one at x > 6283. The
  window (z_e3, z_e6) keeps it safe; widening the window past ~8.7 e-folds would let it skip
  cycles silently. Stepping in phase (using ω) rather than in z would make it robust.
- **The `mode=None` path is broken**: `mode.lower()` at `numeric_with_phase_cut.py:50` runs before
  the `None` check. Production always passes `"stop"`, so it is dead code; either fix it or delete
  the non-stop branch.
- **Samples below z_e6 are requested but never produced.** `main.py` truncates the response grid to
  0.85·z_e6, but in stop mode the ODE terminates at z_e6 + 1e-7 and the `expected_values` check is
  skipped, so the trailing samples are silently absent. Harmless (consumers cope), but the
  truncation constant is misleading.
- **Structure.** Like the WKB region, this is one solve per (k, z_source): ~500 objects × 12k
  evaluations ≈ 6e6 per k, negligible next to stage 1's 1.8e9 (§4), so cost is not a reason to
  change it. If the WKB region moves to the shared primitive of §7, the natural companion is to
  build G(z; z_s) for all z_s from two homogeneous solutions per k via the Wronskian,
  G = [u₁(z_s)u₂(z) − u₂(z_s)u₁(z)]/W(z_s), which makes G continuous in z_s by construction and
  reduces the hand-over to a single matching per k. In radiation both basis solutions are bounded
  oscillations, so there is no growing/decaying-mode cancellation to worry about in the production
  numeric range; this was not tested and is an option, not a recommendation.

---

## 11. Relation to the prior review's plan

`docs/gk-wkb-numerical-review-2026-09.md` (commit `39ed7fc`) was read before any test here was run,
so this review is not independent of it; several scripts exist to reproduce or falsify its claims.
Its §6 proposes a per-(background, k) phase primitive F_k(z) built by local error-controlled
quadrature, compensated accumulation, anchor-plus-local-integral evaluation, and reuse across
source redshifts via F_k(z) − F_k(z_i). It derives the split θ = k∫dz/H + ∫C/(ω + k/H) dz but labels
it optional, saying the leading integral "needs its own adequate representation and accuracy
budget" and warning that splining τ moves the error rather than removing it. Its §7–§8 decline to
recommend a replacement and call for a bounded comparative study at matched accuracy, keeping the
stage-1 rebased ODE as a serious comparator.

**Agreement.** The primitive, anchor-plus-local-Gauss evaluation, compensated accumulation, reuse
across z_source and the warning about splining τ are all correct, and §7 here is that design. The
warning is quantified in §7: a cubic spline of τ nodes is 1.4e-9 relative (1.9 rad at k=1e5); local
Gauss from the nearest node is 2e-16. Its §8 list of difficult geometries (nearby source/response
pairs, off-grid starts, many short segments) and its insistence on independent references are
stronger than anything here and were not run.

**Differences, with the measurements that decide them.**

- *The leading term is the whole design, not an option.* The prior review treated ∫dz/H as needing
  its own accuracy study. §7 is that study: Gauss–Legendre order 4 on the existing grid gives τ to
  6e-15 relative in 3.7k evaluations, and §6 gives the residual ρ ≤ 2.5e-7 rad (LambdaCDM) and
  ≤ 1.5e-3 rad (QCD) over the whole WKB range. With those two numbers F_k collapses to k·τ plus a
  quantity small enough to spline, one table per model rather than per k, and the phase groups
  θ_G ± θ_q ± θ_r cancel exactly in their leading term, which the prior review did not note and
  which bears directly on prompts 07–08 of the source-remediation campaign.
- *The matched-accuracy comparison it asks for is decided.* Stage 1 at tight tolerance does reach
  the floating-point floor (§2: 8e-16 relative), at 108k evaluations for a 1e7 rad span with cost
  proportional to span. Gauss order 4 reaches the same floor for the full 4e12 rad span in 3.7k
  evaluations. A factor of 10⁴–10⁷ at equal accuracy does not need a further study to rank. The
  prior review's remark that "an ODE solver integrating a state-independent RHS is itself
  performing numerical quadrature" is correct and is the argument against retaining the ODE.
- *Its error-target framing lacks the two floors.* It rightly says not to infer a target from
  `rtol`, but names neither the LG truncation floor (1e-3 rad for QCD at k ≥ 1e8, 1e-8 in LambdaCDM,
  §6) nor the double-precision floor on the absolute phase (up to 9e-4 rad, §1). Those bound any
  useful phase target and shrink its proposed study considerably.

**Where its caution still applies here.** The one substantive advantage of an adaptive method over
a fixed-order rule on a fixed grid is robustness to features between nodes. The QCD model's H(z)
comes from a spline of T(z) whose knots need not align with the production grid, so a fixed Gauss
rule for ρ could converge slowly across them. That is unmeasured (§7) and should be the first test
of any implementation. If it fails, the fallback is an adaptive rule for ρ alone, not a return to
the ODE, because the leading term is smooth regardless. The T_k analogue (a sound-horizon primitive
∫c_s dz/H, with a possibly larger residual through the QCD c_s transition) is likewise untested.

In short: right design, wrong emphasis, and a study in place of a decision, because it stopped
short of measuring the two quantities that settle the question.

---

## 12. The transfer-function analogue: `TkWKBIntegration`

Added 2026-09-09. Scripts `tk1_span_residual.py`, `tk2_production_real.py`, `tk3_radiation_LG.py`,
`tk5_numeric_rad.py`, `tk5b_atol.py`, `tk6_dense_real.py`.

### 12.1 What is shared and what is not

`TkWKBIntegration.compute()` calls the same `WKB_phase_function` as the Green's function, with
`Tk_omegaEff_sq`/`Tk_d_ln_omegaEff_dz` in place of the Gk pair and one addition: a separate friction
ODE, `integrate_friction_function`, integrating F' = (3/2)(1+c_s²)/(1+z). `store()` runs the same
(B, δ) algebra with the friction term (ε − 3(1+c_s²))/(1+z) and the same `shift_theta_sample`, and
forms T = √(H_i/(Hω)) e^F B sin(θ+δ). So stages 1 and 2, the resets, Q, the `t_eval` interpolation,
the NumPy deprecation (§8.2), the no-op sign fix and the stale comments are all inherited verbatim.

Structural differences: **one object per k**, not per (k, z_source), integrated from the numeric stop
point (found at k/(aH) ≈ 27, i.e. x_T = k c_s/(aH) ≈ 15.5) down to z_end on the *source* grid
(100/decade, 933–1280 samples). There is therefore no cross-object cycle stitching and no
rectifier (§8.3 does not arise), and the total cost is 50 objects per model rather than ~65,000.

Formulas. Removing the friction p = (ε − 3(1+w))/s from T'' + pT' + qT = 0 gives
ω² = q − p'/2 − p²/4 = w k²/H² + (3w'/2 − ε'/2)/s + [a/2 + εa/2 − 3ε/2 − ε²/4 − a²/4]/s² with
a = 3(1+w), which is term-for-term what `Tk_omegaEff_sq` evaluates; `Tk_d_ln_omegaEff_dz` agrees with a
central finite difference of ½ln ω² to 3e-9–1e-8 relative at z = 1e9…0.5 (the difference is the
finite-difference noise). Both confirm audit TK-report rows R23–R30.

### 12.2 Production spans and the leading term (`tk1_span_residual.py`)

With x_T = k c_s/(aH) and the split ω_T² = w k²/H² + C_T, θ_T = k[τ_s(z_i) − τ_s(z)] + ρ_T with the
**sound-horizon primitive** τ_s = ∫c_s dz/H and ρ_T = ∫C_T/(ω_T + k c_s/H) dz. From the 3-e-fold
point to z = 0.1:

| model | k | θ_T span [rad] | ρ_T [rad] | F(0.1) | x_T at z=0.1 | ε·θ | 1e-8·θ |
|---|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | 6.175e7 | −0.0863 | 38.9 | 4.8e6 | 1.4e-8 | 0.6 |
| LambdaCDM | 1e7 | 6.175e9 | −0.0863 | 48.1 | 4.8e8 | 1.4e-6 | 62 |
| LambdaCDM | 3e8 | 1.852e11 | −0.0863 | 54.9 | 1.44e10 | 4.1e-5 | 1.9e3 |
| QCD | 1e5 | 6.174e7 | −0.0887 | 39.1 | 4.8e6 | 1.4e-8 | 0.6 |
| QCD | 1e8 | 6.174e10 | −0.0979 | 53.2 | 4.8e9 | 1.4e-5 | 620 |
| QCD | 3e8 | 1.852e11 | −0.0933 | 55.4 | 1.44e10 | 4.1e-5 | 1.9e3 |

The spans are 22× smaller than the Green's function's (c_s = 1/√3 in radiation, and c_s → 0.01 after
equality, so x_T actually *falls* below z ≈ 1). The residual is the same −0.09 rad for every k: it is
dominated by the radiation-era term, which in the exact radiation control (§12.4) is
ρ_T = 1/x_i − 1/x, i.e. 0.05 rad from the 3-e-fold start, plus matter-era contributions. Unlike
the Green's function's 1e-8 rad, ρ_T is not negligible, but it is smooth, k-independent to a few
per cent, and cheap. The friction integral F is likewise a smooth **k-independent** function.

### 12.3 The production path on the real background (`tk2_production_real.py`, `tk6_dense_real.py`)

LambdaCDM, `WKB_phase_function` with the Tk frequency and `friction_RHS`, from z_e3 to z = 0.1 on
the 100/decade source grid, production tolerances. Reference: mpmath quadrature of k c_s/H plus the
rationalised residual.

**k = 1e5/Mpc** (stage 1: 35,626 evaluations, 508 resets; stage 2: 1,061; friction: 1,811; 1.1 s):

| z | x_T | θ_ref [rad] | phase error [rad] | relative | δF |
|---|---|---|---|---|---|
| 3.8e8 | 70 | −58.1 | 2.3e-8 | 4.0e-10 | 2.4e-9 |
| 5.5e4 | 4.6e5 | −4.75e5 | 1.9e-4 | 4.0e-10 | −6.8e-8 |
| 268 | 7.3e6 | −2.06e7 | 1.1e-2 | 5.2e-10 | −1.1e-7 |
| 6.8 | 7.8e6 | −4.78e7 | −4.4e-2 | 9.3e-10 | −1.8e-7 |
| 0.32 | 5.6e6 | −6.08e7 | **2.17** | 3.6e-8 | −1.8e-7 |
| 0.1 | 4.8e6 | −6.17e7 | **2.01** | 3.3e-8 | −2.3e-7 |

**k = 3e8/Mpc** (stage 1: 1,932,588 evaluations, 28,343 resets; stage 2: 1,697; 58 s):

| z | x_T | θ_ref [rad] | phase error [rad] | relative |
|---|---|---|---|---|
| 3.0e6 | 2.7e7 | −2.71e7 | 1.7e-4 | 6.4e-12 |
| 2.6e5 (end of stage 1) | 3.1e8 | −3.10e8 | 1.5e-5 | 4.9e-14 |
| 2.2e4 (stage 2) | 3.1e9 | −3.33e9 | 0.42 | 1.2e-10 |
| 169 | 2.3e10 | −7.18e10 | −105 | 1.5e-9 |
| 13.8 | 2.4e10 | −1.28e11 | −193 | 1.5e-9 |
| 0.1 | 1.4e10 | −1.85e11 | **5.1e3** | 2.8e-8 |

Same picture as §4: stage 1 at 1e-9 falling to 5e-14 relative, stage 2 at 1e-9, and the phase
unresolved below z ≈ 1e3 for the largest k. The friction integral carries a relative amplitude
error of 2.3e-7 (k=1e5) to 4.1e-7 (k=3e8), set by `rtol`.

**The late-time jump is the Q equation, not dense output.** Between z ≈ 2 and z ≈ 1.5 the error
jumps from −0.8 to +2.0 rad (k=1e5) and by a factor 30 (k=3e8). Re-solving the stage-2 Q ODE with
`dense_output=True` shows the 2.0 rad already present at the *accepted step endpoints*
(z = 1.54, 0.80, 0.13), so it is stepping error, made in one or two long steps where c_s k/H turns
over in the matter/Λ era. A direct θ' = −ω integration at the same tolerance gives −0.083 rad at the
same points with fewer evaluations (788 vs 1139). This is §3's accuracy penalty of Q, here 25× in
one step.

### 12.4 The LG truncation floor for T (`tk3_radiation_LG.py`)

Exact radiation (w = 1/3, H ∝ s², τ = 1/s), where the exact transfer function is
T = 3(sin x − x cos x)/x³ and the LG frequency is ω² = k²/(3s⁴) − 2/s². The full `store()`
reconstruction (exact initial data at x_i, the (B, δ) algebra, θ = ∫ω, F = 2 ln(s/s_i),
√(H_i/(Hω))) against the exact T, relative to the envelope 3√(1+x²)/x³:

| x_init | max |δT|/env over x_i…1e4 | frozen amplitude offset at x=1e4 | ρ_T = θ − (x_i − x) |
|---|---|---|---|
| 24 | **3.8e-5** | 1.5e-6 | 0.0416 (= 1/x_i − 1/x) |
| 50 | 4.1e-6 | 1.0e-7 | 0.0199 |
| 100 | 5.1e-7 | 3.7e-9 | 0.0099 |
| 400 | 7.8e-9 | 1.3e-11 | 0.0024 |

The error scales as x_i⁻³ and is largest at the start; matching at x_i also freezes in a relative
amplitude offset of order x_i⁻⁴. The production hand-over is at x_T ≈ 15.5, where the x_i⁻³ law
gives ~1.4e-4 (extrapolated). This is the transfer function's analogue of §6: where Gk's LG is exact
in radiation, Tk's is not, and this floor is above everything the τ-primitive would leave. It is the
regime the bessel-remedial campaign's exact θ = x + c_ν + r_ν(x) representation handles exactly for
constant w; audit TK-8(e) measured the same effect (3.7e-4–5.6e-3 in T at 3 e-folds).

### 12.5 The numeric region (`tk5_numeric_rad.py`, `tk5b_atol.py`)

Exact radiation, T = 1, T' = 0 at 5 e-folds outside the horizon, production grid and stop window:

| initial data | (atol, rtol) | RHS evals | max δT/env |
|---|---|---|---|
| T=1, T'=0 (production) | (1e-10, 1e-8) | 6476 | 1.1e-5 |
| exact | (1e-10, 1e-8) | 6401 | **1.1e-5** |
| exact | (1e-13, 1e-8) | 7379 | 3.6e-7 |
| exact | (1e-16, 1e-8) | 7829 | 1.5e-7 |
| T=1, T'=0 (production) | (1e-13, 1e-11) | 11777 | 2.5e-6 |
| exact | (1e-13, 1e-11) | 11681 | 2.3e-8 |

At production tolerances the Tk numeric run is 50× less accurate than Gk's (§10) and the cause is
the **absolute tolerance**: T decays as 3/x², so |T| is 1.2e-2 at the stop point and ~1e-5 deeper in
the window, and `atol = 1e-10` there is a 1e-5 *relative* tolerance. Dropping `atol` to 1e-13 buys a
factor 30 for 15 % more evaluations. Once that is done the super-horizon initial condition becomes
the floor at 2.5e-6 (audit TK-7), removable with the series T ≈ 1 − x²/10. `GkNumericIntegration`
is unaffected because |G| is enormous in these units and `atol` never binds. The §10.2 cleanup
items apply to the Tk integrator identically (same `numeric_with_phase_cut`, same per-RHS
diagnostic, same log₁₀/ln slip).

### 12.6 The consumer

`TkSourceFunctions._build_WKB` splines the stored phase with `phase_spline(chunk_logstep=125)`, so
§5 applies verbatim: no benefit, inflated ordinates, a switch discontinuity. Its interpolation error
is h⁴x_T/384 in the local x_T: at 100/decade, 3.5e-3 rad at k=1e5 (x_T = 4.8e6 at z=0.1) and 10 rad
at k=3e8 (1.4e10), extrapolated from the law measured to 1e7 in §5. Prompt 05's decision to supply
ω in closed form rather than the spline derivative was the right one. It also splines F from the
stored samples; with a tabulated F that spline disappears.

### 12.7 Does one fix fit both?

Yes, with three adjustments and one caveat.

- **Two more k-independent tables.** The Green's function needs τ = ∫dz/H. The transfer function
  needs τ_s = ∫c_s dz/H and F = (3/2)∫(1+c_s²)dz/(1+z). All three are smooth, non-oscillatory,
  built once per model by the same Gauss–Legendre-on-the-grid machinery (§7), and evaluated off-grid
  the same way. The friction ODE and its 2–4e-7 amplitude error go away.
- **The residual is not negligible for T.** ρ_T ≈ −0.09 rad must be included; it is smooth,
  O(0.1) with O(0.1) derivatives, so it can be splined or quadratured per k at negligible cost.
  Whether a fixed-order Gauss rule for ρ_T converges across the QCD model's spline knots is the same
  open test as for ρ_G (§7, §11).
- **No stitching layer.** One object per k means `shift_theta_sample`'s rebase and the `GkSource`
  rectifier have no Tk counterpart; the Tk path is simpler and the primitive drops straight in.
- **Caveat: the physical floor differs.** Gk's LG is exact in radiation; Tk's is not, and at the
  production hand-over x_T ≈ 15.5 its truncation error is ~1e-4 of the envelope (§12.4), with a
  frozen amplitude offset of order 1e-5. Below that no numerical improvement is visible. The
  remedies are the bessel-remedial exact constant-w representation for the radiation era, a
  higher-order LG frequency, or a later hand-over (x_T = 50 gives 4e-6). This is a T_k-only decision.
- **Phase groups.** θ_G ± θ_q ± θ_r = kτ ± qτ_s ± rτ_s. The leading terms of the two transfer
  functions cancel exactly on wavenumbers; the Green's-function term uses a different primitive, so
  G-versus-T cancellation is a difference of two floating-point-floor-accurate quantities rather than
  exact. Since x_T ≤ 1.4e10 the floor on that difference is ≤ 4e-5 rad, well below ρ_T.

Fix the `atol` scaling in the Tk numeric run independently; it is a one-constant change.
