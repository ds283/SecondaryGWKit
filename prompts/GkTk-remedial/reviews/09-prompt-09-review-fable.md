# Review of prompt 09 (`c1c3717`, gktk-remedial) — `PrimitivePhase` and the Gk consumer

Reviewer: Claude Fable 5.1, 2026-09-11. Implementation by Claude Opus 5 (Fable unavailable).
Independently re-run on the `gktk-remedial` worktree at `aae6ac5`.

## Verdict

**Accept.** The implementation does not require improvement. The one missed number — prompt §4
test 1's ≤1e-8 rad — is a defect in the prompt text, not in the code, and the agent's claim about it
is correct in substance. Recommended action: amend prompt 09 §4 test 1 to the campaign's own
numbers (README §6: ≤1e-6 rad, plus the >1e5 ratio), record it as a prompt-text correction, mark
prompt 09 ✅, and check prompt 10 §4 for the same carried-over threshold before dispatch.

## 1. The threshold claim, verified

Geometry fixed by the prompt: exact radiation, k=1e8, z_r=0.1, z_s∈[10,1e4].

| quantity | value | source |
|---|---|---|
| \|θ\| over the geometry | 8.18e7 – 9.09e7 rad | θ = k(1/s_r − 1/s_s) |
| 1 ulp of θ | 1.490e-8 rad | `np.spacing` |
| prompt threshold / ulp | **0.671** | |
| εkτ floor (README §2 (d)) | 2.019e-8 rad | ε·k·(1/s_r) |
| achieved (mpmath dps=50 ref at `mpf(float(z))`) | 4.189e-8 rad = 2.81 ulp | reproduced |
| `phase_spline` of the same samples | 7.286e-3 rad | reproduced |
| ratio | 1.739e5 (> 1e5) | reproduced |

Error decomposition (my measurement, 2890 interior points):
- `TablePrimitive.delta` returns a *double*; its relative error vs exact reaches 4.0e-16 (≈3.6
  half-ulps), contributing up to **2.39 ulp** of θ once multiplied by k. This is the table, the
  off-grid Gauss partial and the hi+lo→double rounding together.
- The product `k*delta` adds **0.50 ulp**.
- Total 2.86 ulp worst case; the shipped 2.81 ulp is this budget.

Precision of the agent's wording: "unreachable in double precision" is a slight overstatement — a
correctly-rounded double result would be 0.5 ulp = 7.45e-9 rad and would pass. But reaching that
would require `delta` to return a double-double and `PrimitivePhase` to form the product with a
compensated (Dekker/FMA) multiply. That is a change to `BackgroundModel.py` (prompt 03's file, not
in prompt 09's allowed list) and would be chasing a number **below the campaign's own declared
floor** εkτ = 2.0e-8 rad, which IMPLEMENTATION_STATE §5 standing note 2 says is not a target. The
conclusion therefore stands: not achievable within prompt 09's scope, and not worth achieving.

Origin of the defect: prompt 08's test 1 was at **k=1e6** on the same s∈[10,1e4] band (span
~9e5 rad, ulp 1.2e-10; 1e-8 rad ≈ 86 ulp). Prompt 09 says "prompt 08's test 1 geometry, scaled to
k=1e8" — the k was scaled by 100 to reproduce the review's 8.26e-3 rad `phase_spline` figure, the
threshold was not. The same 1e-8 number is also README §6's prompt-06 exact-radiation control row,
which is at span 1e7 rad (5.4 ulp) and is reachable there. Both sources make 1e-8 look right; on
this geometry it is not.

Is the substituted assertion sound? Yes: ≤1e-6 (README §6 consumer row, the acceptance table of
record) plus ≤6 ulp of the span. The 6-ulp bound is the useful one: it pins the result to the
representation floor and would catch a regression that a 1e-6 bound would not.

## 2. Implementation against the prompt, item by item

§2 `PrimitivePhase`: interface, `x_is_log` via `expm1`, `theta_mod_2pi` via `WKB_mod_2pi` with the
D6 docstring and the `[00-consumer-anchoring-floor]` pointer, closed-form `theta_deriv` (both
forms), `make_interp_spline` cubic/quintic, `SPLINE_TOP_BOTTOM_CUSHION` imported not copied,
`num_chunks == 1`, `sign` ±1 as one formula — all as specified. No spline of the full phase.

§3 `GkSourcePolicyData`: `phase_spline` import and both constructions gone (grep empty, verified);
the false "chunking keeps the rebasing well-conditioned" comment deleted; `_classify_Levin` uses
`theta_deriv(log_derivative=True)` on the same object; `k` via the public `source.k.k` (identical
to `_k_exit.k.k`); `model` via `source.model_proxy.get()` (the prompt's primary path; no threading
needed). `GkWKBSplineWrapper` verified to call only `theta_mod_2pi`.

§4 tests: 26 tests across the three modules, OK (0.46 s, re-run). Items 1–5 and 1–3 all present.
Rectifier copy read against `GkSource.py:166-233` by the orchestrator; sweep counts (990 objects,
90 transitions) match log 06 exactly; pure-WKB band shows 0 corrections and `rectified == raw`.

§5: full `discover -s ComputeTargets/tests -t .` = 249 tests OK (re-run, 147 s);
`test_quadsource_integral.py` and `test_phase_groups.py` untouched; `black --check` clean on all
six touched modules; diff touches only allowed files (+ OPEN_ISSUES/board/log per README rule 4).

§6: log follows §5.1, every deviation classified, "State handed to the next prompt" verbatim
interface, sign convention, model access, measured errors and φ statistics. Commit message per §5
rule 2, Co-Authored-By names Opus. The board row is ⚠️ pending this decision.

## 3. Deviations — assessment

1. Threshold (STRUCTURALLY REQUIRED) — correct tag; see §1.
2. Prompt 06 `_sweep` not reusable (STRUCTURALLY REQUIRED) — correct: it returns the combined
   phase, and re-reducing a ~1e8 rad double is the anti-pattern §2 (e) forbids; `store_algebra` is
   imported verbatim so the producer algebra under test is the shipped one.
3. Sweep samples on background-grid nodes (IMPLEMENTATION CHOICE) — sound and production-faithful;
   counts reproduce log 06 exactly, redshifts within one grid spacing as stated.
4. Test 3 geometry starting at the response point (IMPLEMENTATION CHOICE) — the only geometry on
   which the prompt's literal `rectified == raw` can hold, and it is production's; the
   geometry-independent statement (0 corrections) is asserted alongside. Good.
5. `build_phi_samples`, `phi()`, `_build_phase` additions — small, justified, reused by prompt 10.
6. Only φ clamped at the cushion (IMPLEMENTATION CHOICE) — an improvement over `phase_spline`'s
   behaviour: θ stays continuous instead of acquiring a k·Δτ(cushion) kink that a Levin consumer
   would differentiate. Pinned by a test.
7. `source.k.k` — equivalent to the prompt's private path.

## 4. Minor observations (none blocking)

- `build_phi_samples` forms φ as the difference of two ~kτ-sized doubles, so `phi_samples` carry a
  few ulp of θ (4e-8 rad at k=1e8, ~1e-3 at 3e8). This is the D6 floor the prompt §1 anticipates,
  and the spline of φ does not amplify it, but the docstring could say so explicitly next to the
  "constant multiple of 2π" note.
- `_get_x(x_is_log=True)` recovers z with `expm1`; a consumer passing the log of a grid node gets a
  z that may not bit-match the node, so the accessor treats it as off-grid (4 integrand evaluations
  instead of 0). Accuracy unaffected; cost is the 5.85 µs vs 3.11 µs figure the log already reports.
- `spline_wrappers.GkWKBSplineWrapper.__init__` type hint still names `phase_spline` — noted by
  the agent; file is "verify, do not edit" here.

## 5. Recommendation for prompt 10

Prompt 10 §4 was written by the same hand; check any absolute-radian threshold against
ulp(k·Δτ_s) on its own geometry before dispatch, and give it README §6's numbers where the prompt's
are below the floor.
