# Numerical review of the tensor Green's-function WKB construction

Review date: 2026-09-08. Production code inspected at `c4c49055cdb40dcb21499467225d8bc1aefc9a2b`.

**Conclusion:** the frequency and amplitude formulas implement the intended leading-order LG approximation, but the numerical machinery does not establish small **absolute phase error** over long evolutions. Stage-1 ODE phase resets do improve the solver's local error scale, and materially improved accuracy in the measured control. Stage 2 replaces these resets with `Q` rescaling, which improves cost but restores amplification of state error by the accumulated stage-2 phase. The separate, subsequent `phase_spline` chunking does not cure interpolation error and can make roundoff and continuity worse. A shared phase primitive built by local, error-controlled quadrature is a promising candidate, not yet a validated replacement. It should be compared with an improved rebased ODE method at matched accuracy before choosing remedial work.

**Clarification after discussion:** ODE phase resets and post-processing spline chunks must be distinguished. The measurements below are unchanged. The explanation of rebasing and the recommendation for next work have been revised to make that distinction explicit and to require comparative validation before selecting a replacement.

This review excludes the numeric/WKB handover: its location, overlap, matching-data accuracy, and continuity across it. All propagation comparisons start directly inside the WKB regime with specified initial data. No production implementation was changed.

## Evidence and scope

Read the source-remediation campaign overview, workstream B orchestration prompt, logs 05 and 06, the follow-up document, the GK audit and its relevant scripts, and the Bessel remediation overview and reconciliation. The orchestration prompt was treated as a historical description, not a request to restart that campaign.

Reproduction:

```sh
PYTHONPATH=. ./venv/bin/python docs/gk-wkb-review/measure.py
PYTHONPATH=. ./venv/bin/python docs/gk-wkb-review/alternatives.py
PYTHONPATH=. ./venv/bin/python docs/spec-code-audit/scripts/GK_02_sympy_omega.py
```

The [main measurements](gk-wkb-review/measurements.json) and [alternative-method measurements](gk-wkb-review/alternatives.json) accompany the scripts. Python 3.12, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0 on Apple Silicon. The radiation phase reference uses 60-digit mpmath arithmetic on the actual floating-point inputs; NumPy `longdouble` is only float64 on this host. Constant-w Green's-function references use SciPy J/Y or scaled Hankel functions at arguments no larger than 1000, not the repository's `bessel_phase` oracle.

The production functions `integrate_phase_function`, `stage_2_evolution` and `phase_spline` are exercised directly, without a Ray cluster or datastore. For the short-interval test, the Ray function's underlying implementation is called with only unit-compatibility checking mocked. The main propagation numbers isolate the phase solver rather than invoke `GkWKBIntegration.store()`. Its reconstruction is checked algebraically and used in the constant-w tests. These are controlled numerical experiments, not a survey of the cached production population or a full cosmological pipeline run.

## 1. What the implementation computes correctly

Write \(s=1+z\), \(\epsilon=sH'/H\). The homogeneous equation actually integrated numerically is

\[
G''+\frac{\epsilon}{s}G'+\left[\frac{k^2}{H^2}+\frac{\epsilon-2}{s^2}\right]G=0.
\]

Removing the first derivative with \(G=\sqrt{H_i/H}\,Y\) gives

\[
Y''+\omega^2Y=0,\qquad
\omega^2=\frac{k^2}{H^2}-\frac{\epsilon'}{2s}
 +\frac{3\epsilon/2-\epsilon^2/4-2}{s^2}.
\]

`WKB_Gk.py` implements this expression correctly. Its `Gk_d_ln_omegaEff_dz` is also the correct derivative, provided the supplied background derivatives are consistent with H. Re-running GK_02 gives zero for all four symbolic differences and confirms the logarithmic derivative identity.

The LG ansatz is

\[
G_{\rm LG}=\sqrt{\frac{H_i}{H\omega}}
 [A\sin\theta+B\cos\theta],\qquad
\theta(z)=\int_{z_i}^{z}\omega(t)\,dt.
\]

The initial coefficients in `GkWKBIntegration.store()` reproduce the specified G and G' for this ansatz; the `atan2` rotation into a single sine is algebraically legitimate. For the directly initialized retarded solution, \(G_i=0,G_i'=1\), the result simplifies to

\[
G_{\rm LG}(z,z_i)=\sqrt{\frac{H_i}{H(z)\omega_i\omega(z)}}
\sin\left(\int_{z_i}^{z}\omega(t)\,dt\right).
\]

The phase is negative for forward evolution to smaller redshift. This is the repository's normalization, not a new convention.

An essential distinction from the Bessel work: here \(\int\omega\,dz\) is a **leading-order approximation**, not an exact Bessel amplitude-phase decomposition. Correct formulas and highly accurate integration do not make this the exact solution in a general background. The general distinction between LG approximants and their error terms is discussed in [DLMF §2.7(iii)](https://dlmf.nist.gov/2.7#iii); the oscillatory residual below follows directly by substitution into this repository's equation.

## 2. Q: exact algebra, unsuccessful accuracy protection

### ODE rebasing and spline chunking are different operations

Stage 1 integrates a local theta, stops at roughly −10⁴ radians, transfers completed cycles to an external integer offset, and restarts with a small remainder. The accumulated offset is not part of the state used in the solver's error test. At `rtol=1e-8`, the relative contribution to its local error scale is consequently of order at most 10⁻⁴ radians on accepted segments, rather than 0.1 radians for a state of magnitude 10⁷. This is a sound and effective reason to rebase. Event detection can involve an accepted step extending past the threshold, so the reset length is not a strict bound on every trial state or error estimate.

This local error scale is not a bound on global error: errors already made in earlier segments remain in the accumulated phase, and event localization and output interpolation also contribute. Nevertheless, the measured stage-1-only control below is about 178 times more accurate than stage 2 at the tested default tolerances. The report does not reject ODE rebasing as an accuracy mechanism.

`phase_spline` performs a different operation after integration: it subtracts cycle offsets from already-computed samples before fitting splines. Those offsets never enter the ODE solver and cannot change its integration error. Section 3 concerns this later operation only.

### Stage 2 scales the accumulated increment rather than rebasing it

In stage 2, with \(u=z_i-z\), the code uses

\[
\theta=\theta_i+\omega_i(1+u)Q,\qquad
Q'=-\frac{\omega(z_i-u)/\omega_i+Q}{1+u},\quad Q(0)=0.
\]

This change of variables is algebraically correct. Its numerical consequence is

\[
\delta\theta=\omega_i(1+u)\,\delta Q
\]

before accounting for frequency, coordinate, or multiplication errors. Reducing the result modulo \(2\pi\) cannot remove that error.

SciPy controls an estimate of local error on the **integrated state**, using `atol + rtol*abs(y)`; it does not promise a bound on the final transformed quantity. DOP853 also uses a separate interpolation polynomial to supply requested output times. See [SciPy's solve_ivp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html). In this scheme both the absolute and relative Q-error scales are multiplied back into phase units. A small numerical state is not an absolute-phase accuracy guarantee.

### Exact radiation control

Take \(H=s^2\), \(\epsilon=2\), in normalized units. Then

\[
\omega=k/s^2,\quad
\theta=k(1/s_i-1/s),\quad
G=(s_i^2/k)\sin\theta.
\]

The LG approximation is exact. There are no background interpolation or truncation errors in this control.

| Configuration | Tolerances `(rtol, atol)` | Maximum phase error at requested nodes |
|---|---|---:|
| Full two-stage path, k=10⁴, s_i=k/30, s_f=1 | (10⁻⁸, 10⁻¹⁰) | 6.92e-6 rad |
| Full two-stage path, k=10⁷, s_i=k/30, s_f=1 | (10⁻⁸, 10⁻¹⁰) | **9.35e-3 rad** |
| Stage 2 only, k=10⁷, s_i=100, s_f=1 | (10⁻⁸, 10⁻¹⁰) | **9.56e-3 rad** |
| Same stage-2 geometry, k=10⁹ | (10⁻⁸, 10⁻¹⁰) | **9.55e-1 rad** |
| Stage 2, k=10⁷ | (10⁻¹², 10⁻¹⁰) | 1.11e-5 rad |
| Stage 2, k=10⁷ | (10⁻¹², 10⁻¹⁴) | 2.41e-6 rad |
| Stage 2, k=10⁷ | (5e-14, 10⁻¹⁶) | 1.54e-7 rad |

The k=10⁷ full path has ten phase resets before switching to Q at z=99. Its phase span is 9,999,970 radians. These results directly contradict the follow-up document's §2.3 statement that the production phase solve is protected and accuracy is lost only afterwards.

The same failure is present outside radiation: a smooth, internally consistent background with a localized change in epsilon gives **9.82e-4 rad** error over 7.03e5 radians, compared with independent quadrature whose reported error estimate is at most 1.13e-8 rad. This is a controlled synthetic transition, not a measurement of the production EOS table.

### The premise Q≈−1 does not generally hold

Exactly,

\[
Q=-\frac{1}{\omega_i(1+u)}\int_0^u\omega(z_i-t)\,dt.
\]

For constant omega it tends to −1. For radiation,

\[
Q=-\frac{s_i u}{s(1+u)}.
\]

In the s_i=100, s_f=1 test, Q ends at −99. H falls strongly during cosmological evolution, so omega is generally not approximately constant. The older GK_09 audit already observed this, but tested relative agreement of two numerical solves on a modest domain; that does not establish absolute oscillatory accuracy over millions of radians.

Even **constant omega** exposes an avoidable problem: theta is linear, while Q is the rational function \(-u/(1+u)\). For omega=10⁷ and z=99→0, the production Q path gives **0.337 rad** maximum error at output samples, while its final phase is accurate to roundoff. A direct phase solve in z gives 2.40e-6 rad maximum error on the same samples. The transformation has made an elementary integral harder, especially for the solver's output interpolation. Checking only the final point would miss this failure.

### What Q does accomplish

It avoids a number of restarts proportional to accumulated phase. For the k=10⁷, s_i=100 radiation case:

| Method | RHS evaluations | Maximum phase error |
|---|---:|---:|
| Current stage 2 | 824 | 9.56e-3 rad |
| Force stage 1 throughout, existing 10⁴-radian resets | 64,619; 990 resets | 5.38e-5 rad |
| Direct theta in log(1+z), same default tolerances | 161 | 1.23e-2 rad |

Thus Q buys cost relative to the restart strategy, at a cost in accuracy here. Plain log-coordinate integration is cheaper again, but **also fails to guarantee absolute phase accuracy**. Neither removing Q nor tightening a single tolerance is a complete remedy. These are same-tolerance comparisons, not matched-accuracy efficiency comparisons: they do not establish which algorithm is best after each has been configured to achieve the same absolute phase target.

### Additional traps

- **Cancellation in z=z_i−u.** Both RHS evaluation and output reconstruction lose low-redshift resolution when z_i is large. At z_i=10⁸, the round-trip error on a logarithmic sample grid reaches 7.45e-9 in z, exceeding the hard-coded 1e-10 sample assertion. An actual stage-2 radiation run with k=1.1e19 and s_i=100000001 raises `AssertionError` at line 359. These are normalized stress-test parameters; incidence in production was not measured. Changing the assertion alone would leave the RHS coordinate error in place.
- **False zero-length evolution.** `WKB_phase_function` line 671 compares a redshift interval to the state tolerance `atol`. With z_i=10, z_f=10−10⁻¹¹ and k=10¹⁴ in radiation, it returns zero phase, although the exact phase is **−8.264 rad**. This is a distinct, reproduced correctness bug, not a handover problem. Exact equality or a frequency-aware phase bound is needed.
- **Stored sample density does not constrain adaptive steps.** `t_eval` specifies output locations, so requesting more samples does not by itself repair phase integration or its output interpolation.
- **The stage transition threshold is not an accuracy criterion.** omega²=10⁶ and phase reset length 10⁴ do not encode a desired absolute phase accuracy. Stage-1 reset errors accumulate too.

## 3. phase_spline: interpolation error survives, and chunking can hurt

The consumer of interest splines G as a function of **source** redshift at fixed response. This is different from the response coordinate used by `GkWKBIntegration`.

For directly initialized radiation data with fixed response s_r,

\[
\theta_G(s_s)=k/s_s-k/s_r,\qquad
\frac{d\theta_G}{d\log s_s}=-k/s_s.
\]

After removing any constant cycle offset, its fourth log-coordinate derivative still has magnitude \(x_s=k/s_s\). Cubic interpolation therefore has a typical interior phase error of order

\[
|\delta\theta|\sim h^4 x_s/384.
\]

The controlling quantity is the **local source phase curvature**, not necessarily the entire source-to-response accumulated phase. A huge constant response phase does not contribute to interpolation truncation error. The follow-up document's statement of identical scaling needs this coordinate qualification for G.

Tests use exact radiation source phases at s_s=10…10⁴, fixed s_r=1, with the common integer offset removed as in `GkSource`. All these sources are well inside the horizon. Entries below exclude three intervals at each global end for “interior”; no numeric/WKB seam is involved.

| k | Samples/decade | Chunks (`None` / `125`) | Interior max phase error, either setting | Maximum including spline ends |
|---|---:|---:|---:|---:|
| 10⁶ | 100 | 1 / 2 | 8.26e-5 rad | 7.09e-4 rad |
| 10⁶ | 300 | 1 / 2 | 1.07e-6 rad | 8.90e-6 rad |
| 10⁸ | 100 | 1 / 2 | **8.26e-3 rad** | **7.09e-2 rad** |
| 10⁸ | 300 | 1 / 2 | 1.07e-4 rad | 8.90e-4 rad |

The expected linear scaling in k and approximately fourth-power improvement with grid density are present. Changing chunk size does not solve this. A phase error δθ can produce an error of order δθ relative to the oscillation envelope; relative error divided by G itself is unsuitable near its zeros.

There are additional implementation problems:

1. **125 is a multiplier, not a cycle limit.** `_build_log_chunks_negative` multiplies successive cycle boundaries by 125. The last boundary can overshoot the observed phase range greatly. Merging chunks with fewer than 30 samples further removes any bound on their spans. The log 05 claim of at most 125 cycles per chunk is false.
2. **The rebase may enlarge the ordinates.** `_chunk_spline` uses the nominal lower cycle boundary, not an actual central sample. In the k=10⁸ example, the globally rebased phase spans 9.99e6 rad, but a chunk stores spline ordinates as large as **6.44e8 rad**. Its knot-level circular discrepancy from its own supplied remainders rises from **2.88e-9** without chunking to **2.28e-7** with `chunk_logstep=125`. Even interpolation at the supplied knots loses accuracy.
3. **Chunk selection is a hard switch between different splines.** The two fits need not agree at the point selected by the distance-to-centre rule. In that same example, the switch has a **1.08e-4 rad** phase jump and a 3.51e-8 relative derivative jump. These were measured by solving for the actual switching coordinate and evaluating both fits there. Derivatives within each chunk do not account for a discontinuity in the assembled function. This matters to Levin consumers as well as pointwise G evaluation.
4. **Decreasing-phase merges can create inverted interval keys.** The merge code uses `(start, next_end)` / `(prev_start, end)` even when chunks were sorted in decreasing phase order. The test constructs a key `(-126, -8813)`. Evaluation happened to complete because merged data are retained and actual x ranges are used, but the claimed interval bookkeeping is wrong. A repair would take min/max of the union irrespective of ordering.

For this G-source geometry, the current chunking has no demonstrated numerical advantage at the tested scales and has demonstrated disadvantages. Rebasing in principle remains useful when storing a small local phase; that does not validate this particular chunk construction. Disabling chunking would remove its extra discontinuities and roundoff here, but leave the much larger cubic interpolation error.

## 4. The independent error from the LG approximation

For \(Y=\omega^{-1/2}\sin\int\omega\), substitution gives

\[
Y''+\omega^2Y=R Y,\qquad
R=\frac34\left(\frac{\omega'}{\omega}\right)^2-\frac12\frac{\omega''}{\omega}.
\]

Thus the amplitude curvature, including second derivatives of frequency, matters. The diagnostic \(|\omega'|/\omega^2<1\) is not an error budget, and certainly does not certify 10⁻⁸ accuracy. In radiation R=0 even though that diagnostic is nonzero.

Using exact constant-w backgrounds \(H=s^p\), \(p=3(1+w)/2\), and \(x=k\tau\), the tests fix G=0,G'=1 at source x_s and propagate to x=1000. The small phase correction is integrated at tight tolerances; ordinary full-phase integration and production phase splines are excluded to isolate truncation.

| w | Source x_s | Initial WKB diagnostic | Max error / exact envelope | Max phase error |
|---|---:|---:|---:|---:|
| 1/3 | 30 | 0.0667 | 1.12e-15 | 2.90e-16 rad |
| 0.2 | 30 | 0.0750 | **2.27e-3** | 2.27e-3 rad |
| 0.2 | 100 | 0.0225 | 6.32e-4 | 6.33e-4 rad |
| 0 | 30 | 0.1001 | **1.21e-2** | 1.21e-2 rad |
| 0 | 100 | 0.0300 | 3.37e-3 | 3.37e-3 rad |

These discrepancies survive arbitrarily accurate evaluation of the current LG ansatz. They are not evidence that `WKB_Gk.py` has the wrong formula. They show that its intended approximation may be insufficient for a stringent scientific error target even for sources already well subhorizon. Evolving deeper inside the horizon does not erase an already accumulated phase offset.

## 5. Error budget for the present computation

The leading errors depend on the region and background; there is no universal ranking by a single number:

- **LG truncation:** zero in radiation, but 10⁻³–10⁻² of envelope in the non-radiation propagation examples above.
- **Numerical phase integration and output interpolation:** roughly 10⁻² rad at a 10⁷-radian span in the default radiation example; potentially order unity for larger spans.
- **Phase resampling:** roughly 10⁻² rad at local x_s~10⁷ on a 100/decade grid, worse at global spline ends; independent of the first two errors.
- **Background errors:** \(\delta\omega\simeq-(k/H)\delta H/H\) deep inside the horizon, so small systematic relative H errors can accumulate into a large absolute phase error. Epsilon and its derivatives enter the subleading frequency and its diagnostic. Real EOS/background interpolation must be included in a final production budget; exact-background tests intentionally remove it. `BackgroundModel.functions.Hubble` delegates to the cosmology, while several derivatives and tau can be spline-derived.
- **Amplitude interpolation:** a separate source-consumer error. For the radiation amplitude H_s/k=s_s²/k, the same 100/decade cubic grid has maximum relative error **1.19e-7**, falling to 1.45e-9 at 300/decade. This is smaller than the large-phase errors here and does not acquire their cycle-count factor.
- **Floating-point coordinates, reconstruction, and range reduction:** secondary at the moderate spans tested, but ultimately unavoidable constraints. Stage 2 forms the large product before range reduction; the split cycle count cannot recover its lost bits or the Q solve's error. Chunk rebasing can aggravate this floor, as measured above.

Downstream quadrature tolerances do not repair an inaccurate supplied amplitude or phase. Supplying an analytic frequency to a derivative slot also does not repair a wrong phase value. In particular, for G one must distinguish differentiation in response redshift from differentiation in source redshift: −omega(source) is valid for the directly initialized primitive-difference representation, but should not be substituted indiscriminately for every assembled source-phase derivative.

## 6. A candidate construction requiring comparative validation

The key observation is that the phase equation has **no dependence on the phase state**:

\[
\theta'=\omega(z).
\]

It is a quadrature problem. The particular state rescaling, phase-cut events, stage transition, repeated sample-list mutation and independently fitted overlapping phase chunks are not essential to solve it. Local quadrature retains the useful principle behind ODE rebasing: integrate a local increment while keeping completed increments outside the state whose error is being controlled. It does not eliminate the need for accurate offset accumulation or a global error budget.

A candidate replacement, preserving the current LG approximation, would:

1. Construct a frequency quadrature representation once per (background,k), using a coordinate appropriate to its variation, normally log(1+z). Store local interval integrals and local antiderivative information.
2. Accumulate phase offsets with compensated summation or a deliberately validated split representation. Allocate **absolute phase error** across the whole consumed interval, not just a per-interval relative tolerance.
3. Evaluate from a nearby anchor plus a local integral, or evaluate a sufficiently accurate local polynomial antiderivative. Choose interval refinement/order from an independent error estimate. Do not fit the entire growing phase with a fixed cubic grid.
4. Reuse the primitive for different initial/source redshifts: \(\theta(z;z_i)=F_k(z)-F_k(z_i)\), retaining whatever initial coefficient/phase-offset data were supplied. This changes neither the boundary conditions nor the handover policy. Avoid a naive subtraction of huge, nearby floating-point primitive values; use interval differences locally.
5. Report achieved phase-error estimates and validate actual sin/cos outputs, sample values, off-grid values, and derivatives. An embedded quadrature estimate needs independent validation, especially across EOS features.

A small prototype on the same k=10⁸ source-phase geometry uses Gauss-Legendre quadrature in each of the existing log-grid intervals and evaluates each midpoint from its interval anchor. With compensated accumulation:

| Gauss order | Maximum midpoint phase error over the full range |
|---|---:|
| 2 | 6.50e-4 rad |
| 4 | **4.23e-9 rad** |
| 8 | **2.24e-9 rad** |

This compares with **7.09e-2 rad** for the current cubic phase spline. The order-4 local midpoint integral alone has error at most 5.58e-11 rad. These are measurements on smooth exact radiation, not a universal four-point rule recommendation. Production needs refinement and a global budget; the measured floor also motivates careful offset arithmetic at still larger phases.

This comparison demonstrates room for improvement, not the superiority of quadrature over a properly configured rebased ODE. The prototype has not yet demonstrated reliable error estimates, matched-accuracy cost, arbitrary off-grid queries, or robustness on the actual cosmological background. An ODE solver integrating a state-independent RHS is itself performing numerical quadrature; the substantive choices are interval construction, error estimation, local evaluation, and accumulation, rather than the labels “ODE” and “quadrature”.

Local intervals need not span less than a cycle. Their increments must instead be representable and computable to the allocated absolute accuracy. Conversely, assigning the same fixed absolute tolerance to every interval is insufficient when the number of intervals grows: the global budget must cover accumulation of their errors. More intervals or more frequent rebasing is not, by itself, a convergence argument.

### Optional leading-term separation

There is an exact analogue of the useful Bessel design principle, although the general cosmological leading term is not a known elementary function. Write

\[
\omega^2=(k/H)^2+C(z),\qquad
\theta(z;z_i)=k\int_{z_i}^z\frac{dt}{H(t)}+
\int_{z_i}^z\frac{C(t)}{\sqrt{(k/H(t))^2+C(t)}+k/H(t)}\,dt.
\]

The rationalized second integrand avoids subtracting nearly equal frequencies. The leading integral is \(k[\tau(z_i)-\tau(z)]\). This decomposition is exact for the **current LG phase** and vanishes in its residual part for radiation. It can make the interpolated residual inexpensive and well-conditioned in the deep subhorizon regime.

However, using the existing cubic `model.functions.tau` without an accuracy study merely moves the large-phase interpolation error into kτ. The leading integral needs its own adequate representation and accuracy budget. Likewise, storing the current Q spline would still give \(\delta\theta=\omega_i(1+u)\delta Q\); it does not automatically remove the phase amplification. Preserve the leading/residual split during trigonometric evaluation if their scales warrant it.

The Bessel closed-form tail itself cannot simply be transplanted into a varying cosmological background. Removing numerical error and improving the physical LG approximation are two separate developments. Higher-order LG or an exact nonoscillatory phase construction would need a separate design if the truncation numbers above are unacceptable.

## 7. Consequences for the earlier documents and next work

Three statements should no longer guide remediation:

- Log 05's “125 cycles per chunk” explanation is wrong; the later follow-up correctly retracted the interpolation claim but did not establish useful numerical conditioning for the actual negative-phase chunk implementation.
- Follow-up §2.3's exemption of the production phase solve is contradicted by the radiation tests.
- Follow-up §2.5's suggestion that storing Q removes the growing phase-error factor is not sufficient: reconstruction multiplies Q interpolation error too, and this Q is not generally bounded near unity.

The Bessel campaign correctly leaves cosmological phase construction to separate work. Its strongest transferable lesson is to represent and budget the quantities that control the oscillation's absolute error, and to validate against an independent reference.

Recommended next step: a bounded comparative numerical study before drafting a replacement campaign. The demonstrated short-interval and coordinate-loss defects can be handled separately; they do not determine which phase construction should replace the current one. Post-processing spline chunking should not be treated as protection against ODE error, while stage-1 ODE rebasing should remain a serious comparator. No conclusion here relies on changing or testing the numeric/WKB handover.

## 8. Proposed validation study before selecting remediation

**Objective:** establish whether a local-quadrature construction can meet a specified absolute phase target, with trustworthy error estimates and acceptable construction/evaluation cost, more simply than an improved rebased ODE. Preserve the current LG frequency and supplied initial data throughout. Do not change the numeric/WKB handover or use agreement with the exact Green's function to judge numerical integration of an approximate LG phase.

1. **Define the workload and error targets.** Map the required k, source/response redshifts, phase spans and query patterns from the production configuration. Sweep several absolute phase targets to expose accuracy/cost curves and the floating-point floor; do not infer a scientific acceptance target from the current solver `rtol`. Allocate separate budgets for phase construction, accumulation, off-grid evaluation and background uncertainty.
2. **Use independent references.** Retain analytic radiation and constant-frequency integrals. Add high-precision quadrature of smooth, nontrivial backgrounds and constant-w cases, with independently checked convergence. For actual background functions, compare two independently refined numerical constructions of the same supplied frequency; separately refine the background representation to avoid mistaking agreement on a common inaccurate background for physical accuracy.
3. **Compare credible candidates at matched achieved accuracy.** Include the current two-stage method as a baseline; stage-1-style rebased theta integration with varied reset lengths, improved coordinates and explicit error budgeting; and adaptive local quadrature with varied orders, interval sizes and accumulation representations. Test leading-plus-residual quadrature as an additional candidate rather than assuming it is necessary. Couple the viable integration methods to local evaluation so that the existing cubic phase spline does not obscure construction accuracy.
4. **Exercise the difficult geometries.** Test long accumulated phases, nearby source/response pairs, large starting redshifts, many short segments, starts and queries off the construction grid, and both source- and response-coordinate evaluation. Include smooth transitions of varying widths, the actual EOS/background features and their interpolation knots, and frequencies approaching the least adiabatic part of the allowed WKB domain. All tests remain inside the WKB propagation problem.
5. **Test error estimates and accumulation explicitly.** Compare observed absolute phase and circular sin/cos errors with the claimed global bounds, including dense and randomized off-grid samples and interval boundaries. Check derivative consistency, anchor continuity and additivity of phase increments across intermediate points. Vary the number of segments independently of total phase span where possible, and record error versus reset length; do not assume segment errors cancel randomly. Deliberately place narrow features between quadrature nodes to test whether refinement detects them. Keep rounding of large primitive values separate from quadrature truncation error.
6. **Measure the complete cost.** Record frequency/background evaluations, construction time, query time, memory, number of intervals/resets and opportunities for reuse across sources. Compare these at the same achieved error. A shared primitive may save construction while making queries more expensive; both count.

The deliverable should be a reproducible comparison with a supported operating range, measured accuracy/cost curves, identified failures and a justified recommendation. Recommend a production replacement only if a candidate meets the chosen targets across that workload, with validated error estimates and clear implementation advantages. If a revised rebased ODE is competitive and simpler to integrate into the existing pipeline, prefer it. If neither meets the target, investigate the precision/background limitation before prescribing a rewrite. LG truncation remains a separate error budget whatever construction wins.
