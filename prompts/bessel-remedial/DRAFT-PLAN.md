# Draft plan: accurate Bessel amplitude and phase construction

**Date:** 2026-09-08  
**Status:** proposal; production implementation has not been changed  
**Scope:** the Bessel amplitude–phase representation, its numerical evaluation, and the consumers needed to preserve its accuracy  
**User clarification:** there is no requirement to retain the phase ODE. The phase function is the required output; the method of obtaining it may be replaced.

## 1. Recommendation

Replace the current normalized-phase ODE and subsequent full-phase spline with a representation that separates the known leading oscillation from a smooth correction:

\[
J_\nu(x)=A_\nu(x)\sin\theta_\nu(x),\qquad
Y_\nu(x)=-A_\nu(x)\cos\theta_\nu(x),
\]

\[
\boxed{
\theta_\nu(x)=x+c_\nu+r_\nu(x),\qquad
A_\nu(x)=\sqrt{\frac{2}{\pi x}}\,a_\nu(x),\qquad
c_\nu=\frac\pi4-\frac{\pi\nu}{2}.
}
\]

Construct the normalized amplitude \(a_\nu\) and residual phase \(r_\nu\) using the exponentially scaled Hankel function. Interpolate the residual and the logarithm of the normalized amplitude, with adaptive refinement and explicit error checks. Preserve the leading term and correction separately when evaluating trigonometric functions and combining phases.

This replaces the ODE rather than merely retuning it. It also removes the separate phase-offset root solve. There is no proposed change to the underlying Bessel equation or to the sine/cosine convention used by the repository.

The initial implementation should use the scaled special function across the existing supported domain. An explicit asymptotic tail is an optional later optimization, not a prerequisite: using a scaled Hankel function does not itself truncate an asymptotic expansion.

## 2. Context and evidence reviewed

The workstream B campaign introduced a transfer-function representation and restricted the sampled quadratic source to its appropriate region. Its constant-equation-of-state fixtures use the Bessel amplitude–phase construction as an analytic reference. An error in that reference consequently limits what those tests can establish.

The review covered:

- [Campaign README](../source-remediation/README.md), especially the exact constant-\(w\) fixture and closed-form LG amplitude.
- [Workstream B orchestration prompt](../source-remediation/orchestrator/workstream-B.md).
- [Prompt 05 log](../source-remediation/logs/05-tk-source-functions.md), especially deviation 7, verification results, and its superseding note.
- [Prompt 06 log](../source-remediation/logs/06-quadsource-regions.md), for the distinct source-spline accuracy floor and downstream context.
- [Phase and hand-over follow-up](../../docs/lg-phase-and-handover-followup-2026-09.md), especially §§2.4–2.5.
- [Bessel construction](../../LiouvilleGreen/bessel_phase.py), [phase spline](../../LiouvilleGreen/phase_spline.py), [range reduction](../../LiouvilleGreen/range_reduce_mod_2pi.py), and [Bessel tests](../../LiouvilleGreen/tests/test_bessel_phase.py).
- [Production phase integration](../../Quadrature/integrators/WKB_phase_function.py), [production Bessel setup](../../main.py), and [three-Bessel consumers](../../LiouvilleGreen/three_bessel_integrals.py).

The experiments reported below were temporary review calculations against the current implementation. They did not modify production code. The reproduction appendix supplies the central comparison without depending on temporary files.

## 3. Findings about the existing construction

### 3.1 Normalizing the phase does not bound absolute phase error

The Bessel constructor already evolves \(Q=\theta/x\) in \(u=\log x\):

\[
\frac{dQ}{du}=\frac{2}{\pi x(J_\nu^2+Y_\nu^2)}-Q.
\]

The state remains of order unity at large \(x\), which is useful for numerical scaling. But reconstruction gives

\[
\delta\theta=x\,\delta Q.
\]

The relevant error for \(\sin\theta\) and \(\cos\theta\) is absolute phase error, measured in radians. Controlling the relative error of a state approaching unity does not provide a uniform absolute-phase bound as \(x\) increases. ODE local error tolerances are also not certificates of global solution error.

This supports the follow-up's attribution of the default-tolerance Bessel error to phase evolution. It also qualifies the discussion of the Green's-function normalization. A representation of the form \(\theta=\theta_0+LQ\) has the same amplification \(L\delta Q\). The normalization can help, but its presence alone does not establish that integration error is negligible. This review has not measured the Green's-function production solve and does not assign it a new numerical error estimate.

### 3.2 Full-phase interpolation introduces a separate growing error

After integrating \(Q\), the Bessel constructor forms \(xQ\), reduces it to cycles and remainder, and constructs a `phase_spline` in \(\log x\). Within each chunk, that class reconstructs an unwrapped phase relative to a constant cycle offset and fits a cubic spline.

For a leading phase \(\theta\simeq x=e^u\), its fourth derivative with respect to \(u\) is also approximately \(x\). Cubic interpolation therefore has an error of order \(h^4x\). Subtracting an integer number of cycles changes the constant term, not the fourth derivative. Chunking can improve arithmetic conditioning but cannot remove this interpolation mechanism.

At 250 samples per e-fold, the interior scale \(h^4x/384\) is approximately \(6.7\times10^{-10}\) at \(x=10^3\) and \(6.7\times10^{-6}\) at \(x=10^7\). This is a useful scale estimate, not an endpoint bound or an exact formula for every spline configuration.

Evaluating the existing `Q` spline directly can reduce this particular interpolation error. It still multiplies its state and interpolation errors by \(x\), retains the ODE error, and requires correct handling of the phase offset. It is not the recommended final construction.

### 3.3 The phase-offset root solve adds an avoidable floor

The constructor initially determines a phase using `asin(J/sqrt(J*J+Y*Y))`. It later solves a scalar matching equation against \(J\), using `xtol=1e-6` and `rtol=1e-4`, and subtracts the resulting offset `phi`.

In the reviewed low-order cases this introduces the following offsets:

| Order | Computed `phi`, radians |
|---|---:|
| \(1/2\) | approximately \(-1.45\times10^{-8}\) |
| \(3/2\) | \(-1.14935\times10^{-8}\) |
| \(7/4\) | \(-2.04439\times10^{-8}\) |
| \(5/2\) | \(-4.83654\times10^{-8}\) |

For the \(3/2\), \(7/4\), and \(5/2\) cases the matching location is the initial node, whose phase was already fixed from the Bessel value. The extra solve creates a spurious small correction. At tight ODE tolerances and \(x\le10^3\), undoing `phi` in a diagnostic reduced the error to approximately \(6\times10^{-10}\), consistent with the remaining full-phase interpolation scale.

More generally, matching only a sine value is an unnecessarily ambiguous way to establish a phase quadrant. If a phase anchor is needed, use

\[
\theta_0=\operatorname{atan2}(J_\nu(x_0),-Y_\nu(x_0))
\]

and an explicit branch convention. This uses both Bessel functions and avoids a root solve. Removing `phi` indiscriminately from the old implementation is not proposed as a general branch-handling fix.

### 3.4 Fixture tolerances and production tolerances differ

The follow-up correctly identifies the defaults used by the fixtures: `rtol=1e-8`, `atol=1e-10`. However, `main.py` already builds its Bessel objects with `rtol=5e-14`, `atol=1e-25`, currently for orders \(1/2\) and \(5/2\).

Thus the default-tolerance fixture error should not be quoted as the measured production Bessel error. Tightening defaults would improve the fixtures but leave the phase-offset and interpolation mechanisms in place. Diagnostic runs using the production tolerances still gave approximately \(6\times10^{-6}\) phase-pair error through \(x=10^7\).

### 3.5 Amplitude error is small in the measured low-order interval

For \(19.2\le x\le10^3\) or \(10^7\), the existing amplitude spline agreed with \(\sqrt{J^2+Y^2}\) at roughly \(10^{-13}\) relative error or better in the tested low-order cases. The principal defect there is phase accuracy.

Normalizing the amplitude is nevertheless worthwhile. It factors out known algebraic variation, preserves positivity when interpolating its logarithm, and makes the far tail approach a constant. These are useful structural improvements, not evidence that amplitude error dominated the workstream B findings.

## 4. Numerical experiments

### 4.1 Error definition

Let \(A=\operatorname{hypot}(J,Y)\), evaluated from the reference functions. Define

\[
E_\theta=\max_x\max\left(
\left|\sin\theta_{\rm ours}-J/A\right|,
\left|-\cos\theta_{\rm ours}-Y/A\right|\right).
\]

This is a phase-pair error, not a directly unwrapped phase difference. For small errors it measures the same local effect on Bessel values, normalized by their envelope, without dividing by a function near a zero. Amplitude error is measured separately as \(E_A=\max|A_{\rm ours}/A-1|\).

### 4.2 Low-order comparisons

The comparison used 3,000 logarithmically spaced evaluation points from \(19.2\) to \(x_{\max}\), a construction endpoint of \(1.02x_{\max}\), and the existing density of 250 samples per e-fold. These points do not exhaust the domain or isolate every spline endpoint interval.

| Order | Construction | \(E_\theta\), through \(10^3\) | \(E_\theta\), through \(10^7\) |
|---|---|---:|---:|
| \(3/2\) | Existing defaults | \(2.019\times10^{-6}\) | \(1.588\times10^{-3}\) |
| \(3/2\) | Existing, `rtol=1e-12`, `atol=1e-14` | \(1.203\times10^{-8}\) | \(5.910\times10^{-6}\) |
| \(3/2\) | Scaled-Hankel residual, cubic | \(3.042\times10^{-14}\) | \(2.792\times10^{-14}\) |
| \(7/4\) | Existing defaults | \(1.565\times10^{-6}\) | \(2.119\times10^{-3}\) |
| \(7/4\) | Existing, `rtol=1e-12`, `atol=1e-14` | \(2.101\times10^{-8}\) | \(6.469\times10^{-6}\) |
| \(7/4\) | Scaled-Hankel residual, cubic | \(4.535\times10^{-14}\) | \(4.527\times10^{-14}\) |

The two orders correspond to the campaign's radiation and \(w=0.2\) transfer-function fixtures. The prototype's normalized-amplitude reconstruction had errors of a few \(10^{-14}\) in these comparisons.

This demonstrates a substantial improvement in the tested interval. It does not certify machine-precision accuracy for all orders, endpoints, or input conventions. `jv`, `yv`, and `hankel1e` are all SciPy special functions and may share numerical machinery; agreement between them alone is insufficient as independent validation.

### 4.3 Turning-point and higher-order checks

A separate calculation tested all log-interval midpoints from the existing lower construction bound to \(\max(1000,10\nu)\). Both cubic and quintic interpolation were tried at 250 samples per e-fold.

| Order | Cubic \(E_\theta\) | Quintic \(E_\theta\) | Quintic \(E_A\) |
|---|---:|---:|---:|
| \(3/2\) | \(4.07\times10^{-12}\) | \(1.02\times10^{-14}\) | \(2.38\times10^{-14}\) |
| \(7/4\) | \(7.21\times10^{-12}\) | \(1.88\times10^{-14}\) | \(3.29\times10^{-14}\) |
| \(5/2\) | \(2.37\times10^{-11}\) | \(1.01\times10^{-14}\) | \(2.29\times10^{-14}\) |
| \(20.5\) | \(8.62\times10^{-9}\) | \(2.24\times10^{-12}\) | \(2.04\times10^{-11}\) |
| \(100.5\) | \(5.29\times10^{-7}\) | \(2.94\times10^{-9}\) | \(1.04\times10^{-8}\) |

At \(\nu=1000.5\), the naive fixed-grid prototype failed with order-unity phase-pair errors. The residual can advance by more than a branch-tracking-safe angle per grid interval, and ordinary `unwrap` cannot infer missed turns. Increasing interpolation degree does not fix incorrectly unwrapped samples.

For that order, increasing the uniform density and using quintic interpolation gave:

| Samples per e-fold | \(E_\theta\) | \(E_A\) |
|---:|---:|---:|
| 1000 | \(9.66\times10^{-9}\) | \(2.34\times10^{-8}\) |
| 2000 | \(9.08\times10^{-11}\) | \(4.64\times10^{-10}\) |
| 4000 | \(9.70\times10^{-12}\) | \(7.97\times10^{-12}\) |

These are diagnostic refinement results, not a recommendation to impose a 4000-point density everywhere. They motivate adaptive sampling concentrated where the order and proximity to the turning point require it. The asymptotic claim that the residual is small is for fixed order as \(x\) grows; it is not uniform in order near the turning point.

### 4.4 Independent large-argument spot checks

The direct scaled-Hankel construction and angle-addition reconstruction were compared with 70-digit `mpmath` Bessel values at \(x=10^3,10^7,10^{12},10^{15}\), for orders \(3/2\) and \(7/4\). Phase-pair discrepancies were below \(1.4\times10^{-16}\) at these selected points.

These checks used the supplied floating-point argument directly and did not test a spline over the entire range. They show that the leading-plus-residual evaluation can preserve the correction even at very large arguments. They do not establish the accuracy of a physical argument computed from uncertain model quantities, or of a round trip through `log` and `exp`.

## 5. Proposed mathematical construction

### 5.1 Obtain amplitude and phase from a scaled Hankel function

SciPy defines

\[
\operatorname{hankel1e}(\nu,x)=e^{-ix}H_\nu^{(1)}(x),
\qquad H_\nu^{(1)}=J_\nu+iY_\nu.
\]

Construct

\[
S_\nu(x)=\sqrt{\frac{\pi x}{2}}\,
e^{i(\pi\nu/2+\pi/4)}\operatorname{hankel1e}(\nu,x).
\]

Under the repository's convention,

\[
H_\nu^{(1)}=A_\nu e^{i(\theta_\nu-\pi/2)},
\]

so substitution gives

\[
S_\nu=a_\nu e^{ir_\nu}.
\]

Therefore \(a_\nu=|S_\nu|\) and \(r_\nu\) is its continuously tracked argument. The construction never obtains the residual by subtracting \(x\) from a large computed phase. The scaled routine supplies the oscillation-removed quantity directly.

The fixed-order large-argument expansion gives

\[
r_\nu(x)=\frac{4\nu^2-1}{8x}+O(x^{-3}),\qquad
a_\nu(x)=1+O(x^{-2}).
\]

Consequently the logarithmic derivatives of the residual decay in the tail, whereas those of the old full phase grow like \(x\). These asymptotics explain the improved interpolation conditioning; they are not used as a finite approximation in the prototype.

### 5.2 Branch tracking is part of correctness

A principal complex argument alone is insufficient to construct a differentiable unwrapped phase. The implementation must:

1. Establish an anchor consistent with both \(J\) and \(Y\), and document the allowed constant integer-cycle offset.
2. Refine sampling before accepting an interval whose residual variation could conceal a wrap.
3. Track the continuous residual branch through accepted intervals.
4. Validate phase continuity and derivative behavior across refinement and interpolation boundaries.

Do not rely only on endpoint principal-angle differences: an interval may contain an undetected full turn. Use a derivative-informed sampling criterion and additional interior checks. The exact relation \(dr/d\log x=x(a^{-2}-1)\) can help estimate variation, with appropriate care about tail cancellation and the distinction between an estimate and a rigorous interval bound.

An integer \(2\pi\) offset does not change Bessel values, but inconsistent offsets between intervals invalidate interpolation and can confuse consumers that inspect raw phase differences. Fixing this explicitly is preferable to the old ad hoc rebasing and scalar sine matching.

### 5.3 Interpolation and derivatives

Interpolate \(r(u)\) and \(\ell(u)=\log a(e^u)\), with \(u=\log x\). A quintic spline is a reasonable starting candidate given the measurements, but spline degree alone is not the acceptance criterion. Check interpolation error at additional points, refine, and include endpoint intervals in the checks. Piecewise Chebyshev interpolation is another valid implementation choice if it provides simpler error estimation.

Reconstruct

\[
A(x)=\sqrt{\frac{2}{\pi x}}e^{\ell(\log x)},\qquad
\theta'(x)=1+\frac{r_u(\log x)}x,
\]

\[
\frac{d\log A}{dx}=-\frac1{2x}+\frac{\ell_u(\log x)}x.
\]

Use derivatives of the chosen interpolants so the derivative accessor describes the represented function. Independently verify the exact Wronskian relation

\[
A^2\theta'=\frac{2}{\pi x},\qquad \theta'=a^{-2}.
\]

An accurate amplitude alone does not ensure an accurate interpolated phase derivative. Conversely, substituting \(a^{-2}\) for a poor spline derivative can hide inconsistency. The value and derivative errors both need to meet the budget, especially for Levin quadrature.

### 5.4 Preserve the split during evaluation

Let \(d=c_\nu+r_\nu(x)\). Evaluate

\[
\sin\theta=\sin x\cos d+\cos x\sin d,
\]

\[
\cos\theta=\cos x\cos d-\sin x\sin d.
\]

At large \(x\), forming `x + d` first can round away part or all of the correction. Angle addition avoids that loss and lets the platform trigonometric routines reduce the original argument internally.

Similarly, reducing a huge phase against a double-precision `TWO_PI` creates an error proportional to the cycle count. A bounded-angle accessor can instead use `atan2(sin_theta, cos_theta)`, with any documented interval convention applied only to this bounded result. If a consumer actually requires an accurate integer cycle count, design and validate that operation separately; a good bounded angle is not by itself a cycle-count algorithm.

Keep `raw_theta` for compatibility and diagnostics, documenting its floating-point precision limit. Accurate oscillatory evaluation must use the split or bounded-angle path, not a large raw phase reconstructed for convenience.

### 5.5 Argument accuracy has its own limit

The new representation can accurately evaluate Bessel functions at the supplied floating-point \(x\). It cannot recover uncertainty already present in \(x=k\eta\), nor undo a lossy `exp(log(x))` round trip.

When raw \(x\) is supplied, preserve it for the leading oscillation and use \(\log x\) only to query the correction interpolants. If only \(u=\log x\) is supplied, define the evaluation as being at the computed \(e^u\), and test against that same argument. At very large \(x\), input-coordinate error can exceed the residual interpolation error by many orders of magnitude.

## 6. API and consumer integration

### 6.1 Preserve the useful Bessel interface

Keep the constructor entry point and the useful dictionary members `phase`, `mod`, `bessel_j`, `bessel_y`, `min_x`, and `max_x`. The `phase` object need not remain an instance of `phase_spline`; it should implement the behavior its consumers require, including raw/log inputs and ordinary/log derivatives.

Audit every caller before finalizing the adapter. In particular, diagnostic consumers in `plot_besssel_phase.py` and `ComputeTargets/QuadSourceIntegral_debug.py` read `Q`. The current `Q` is the pre-offset ODE state, whereas the returned phase also incorporates `phi` and cycle rebasing. Do not silently replace `Q` with a different diagnostic quantity under the same undocumented meaning. Either update those consumers to use the residual, or provide a clearly documented compatibility quantity and deprecation path. Treat `phi` similarly.

Existing `atol` and `rtol` arguments describe ODE tolerances. Introduce explicit absolute phase and relative amplitude accuracy settings for the new construction. Update callers rather than silently claiming that the old arguments have identical semantics. The migration can accept deprecated arguments temporarily, but their translation and precedence must be documented.

`BesselPhaseProxy` transfers the object through Ray. Verify serialization of the new representation and its interpolants. No Bessel-specific datastore schema change is indicated by this design; the separate persisted cosmological phase problem should not be folded into this patch by assumption.

### 6.2 Preserve structure in Bessel phase groups

The current three-Bessel helper sums raw phases or bounded phases and sums their derivatives. Bounded phases can preserve trigonometric values, but raw phase differences and derivative cancellation still deserve explicit treatment.

For a shared variable \(t\), assemble

\[
\theta_\mu(kt)+\epsilon_\nu\theta_\nu(qt)+\epsilon_\sigma\theta_\sigma(st)
=Kt+C+R(t),
\]

\[
K=k+\epsilon_\nu q+\epsilon_\sigma s.
\]

Combine the leading coefficients before multiplying by \(t\), and combine the residuals separately. Form group derivatives from that same expression. This avoids subtracting independently reconstructed large phases near resonance.

Use appropriately accurate summation for the coefficients and residuals, and test exact and near cancellation. Input coefficient uncertainty still limits what can be established near resonance; compensated arithmetic does not make uncertain inputs exact.

### 6.3 Separate oracle improvement from transfer-function re-splining

An accurate Bessel object makes the constant-\(w\) oracle more useful. It does not repair a downstream consumer that samples its full phase and then fits another coarse spline in \(\log(1+z)\).

Re-run the workstream B comparisons with two distinct questions:

1. Does the Bessel amplitude–phase object reproduce independently evaluated Bessel functions?
2. How much error does `TkSourceFunctions` introduce when consuming the sampled fixture?

Keep the exact Bessel fixture separate from the approximate physical LG fixture. Improving the Bessel representation does not remove physical LG truncation error. Preserve and report the consumer interpolation floor until a separate residual-phase or local-phase-evaluation design addresses it.

## 7. Proposed implementation sequence

### Stage 1 — Establish independent regression measurements

- Add a reproducible diagnostic covering the metrics and parameter ranges in §4.
- Add exact half-integer references for \(\nu=1/2,3/2,5/2\), and selected high-precision non-half-integer references.
- Record construction cost, sample count, evaluation cost, phase-pair error, amplitude error, derivative error, and location of each maximum.
- Distinguish sample nodes, interior test points, and endpoint intervals.

**Reason:** the present basic Bessel test allows 50% relative error, and higher-order tests mainly detect gross failures. They cannot establish the intended numerical improvement. References built from the same Bessel phase object would also conceal common error.

### Stage 2 — Implement scaled-Hankel residual construction

- Replace the ODE and offset matching with normalized scaled-Hankel samples.
- Implement explicit branch tracking and adaptive interpolation.
- Retain the current domain restrictions initially; validate inputs and insufficient sample requests rather than extending behavior implicitly.
- Add finite-value and construction-failure checks, including a refinement cap that reports unmet accuracy instead of silently accepting it.
- Implement amplitude, residual, and derivative accessors.

**Reason:** this addresses state-error amplification, removes spurious phase matching, and avoids interpolating the growing leading phase in one coherent change.

### Stage 3 — Implement accurate evaluation and compatibility

- Implement split sine/cosine evaluation and a bounded-phase accessor.
- Preserve raw arguments during logarithmic interpolation lookup.
- Provide the phase adapter required by existing consumers.
- Migrate diagnostic `Q`/`phi` usage and production tolerance arguments.
- Verify serialization and scalar/log-input behavior.

**Reason:** a better construction is ineffective if evaluation immediately rounds its correction away or sends it through the old full-phase spline.

### Stage 4 — Integrate Bessel phase-group consumers

- Preserve the analytic leading term when forming three-Bessel phases and derivatives.
- Test resonant and nearly resonant combinations, Bessel integrals, and amplitude/phase sign conventions.
- Keep quadrature error estimates distinct from input representation error; do not interpret agreement under quadrature refinement as a certificate of Bessel accuracy.

**Reason:** consumers can otherwise reintroduce cancellation or large-phase errors after the individual Bessel functions have been fixed.

### Stage 5 — Revalidate campaign fixtures and document remaining floors

- Re-run the \(w=1/3\) and \(w=0.2\) fixture comparisons using independent Bessel references.
- Quantify the remaining transfer-function re-spline error separately.
- Update the follow-up document with the measured replacement accuracy, the additional offset finding, and the fixture/production tolerance distinction.
- Remove stale blanket statements that the Bessel oracle necessarily has an \(x\times10^{-8}\) floor, while retaining historical measurements as historical evidence.

**Reason:** the objective is both a better representation and reliable downstream interpretation of its accuracy.

## 8. Acceptance criteria and error budget

The following are proposed engineering targets, not already-certified bounds:

| Coverage | Initial acceptance target |
|---|---|
| Low orders \(1/2,3/2,7/4,5/2\), existing domain through \(10^7\) | \(E_\theta,E_A\le10^{-11}\), including endpoint checks |
| Orders \(20.5,100.5,1000.5\), lower bound through \(\max(1000,10\nu)\) | \(E_\theta,E_A\le10^{-10}\), with adaptive refinement |
| Ordinary phase derivative | Relative error at most \(10^{-9}\) against an independent reference over the tested domain |
| Phase groups near derivative cancellation | Absolute derivative checks scaled to constituent frequencies, plus an independent group reference; no division by a vanishing group derivative |
| Selected large arguments through \(10^{15}\) | Independent split-evaluation checks at identical supplied arguments; no claim of full-domain coverage from spot checks |

These targets leave margin above the observed low-order prototype errors and require substantial improvement over the old representation. The implementation should expose requested accuracy and report failure when it cannot meet it. If validation shows that a reference or input-coordinate floor dominates, document that floor before changing an acceptance target.

Allocate phase error among scaled-function sampling, branch tracking, interpolation, derivative representation, and evaluation arithmetic. Amplitude relative error and phase error both contribute to envelope-normalized Bessel error; controlling either alone is insufficient. Adaptive midpoint comparisons are practical estimators, not mathematical supremum bounds. Use multiple check points, refinement comparisons, adversarial tests, and independent references before describing the result as validated.

Required coverage includes:

- Bessel zeros and extrema, where pointwise relative errors are misleading.
- Construction endpoints and the turning-point neighborhood.
- Changes of interpolation interval and phase branch.
- Raw and logarithmic input modes, with explicitly matched reference arguments.
- Consistency of phase values, derivatives, and amplitude through the Wronskian identity.
- Supported high orders already exercised by the existing tests.
- Three-Bessel values and integrals, including phase-group cancellation.
- Serialization and existing fixture consumers.

Benchmark performance rather than assuming it improves. Removing adaptive ODE integration is promising, but tighter interpolation and high-order refinement have their own costs. The design should retain cheap evaluation after construction and avoid work proportional to the total number of oscillations in the fixed-order far tail.

## 9. Explicit boundaries and deferred work

- Numeric/WKB hand-over overlap and numeric spline endpoint padding are separate work. This plan does not change them.
- General cosmological transfer-function and Green's-function stored phases need their own leading-term or local-integration design. The Bessel leading term \(x\) is special and cannot simply be assumed for a general background.
- Physical LG truncation error remains distinct from numerical Bessel representation error.
- Extending the Bessel domain below the current lower bound, or extending supported orders, requires explicit validation and is not implied by replacing the constructor.
- An explicit asymptotic tail can be considered after the scaled-Hankel implementation is validated. Its switch must depend on order and a remainder/error test, not only on a fixed large value of \(x\).
- Retaining a residual ODE is unnecessary under the user's clarified requirement and is not part of the recommended implementation.

## 10. Reproduction of the central comparison

Run from the repository root using the repository environment, for example `PYTHONPATH=. ./venv/bin/python <script.py>`. This is a diagnostic prototype, not production code: its simple fixed-grid `unwrap` is intentionally limited to the low-order comparison and must not be copied as the high-order branch algorithm.

```python
import contextlib
import io
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.special import hankel1e, jv, yv
from LiouvilleGreen.bessel_phase import bessel_phase


def pair_error(sine, minus_cosine, J, Y, amplitude):
    return max(
        np.max(np.abs(sine - J / amplitude)),
        np.max(np.abs(minus_cosine - Y / amplitude)),
    )


for nu in (1.5, 1.75):
    for xmax in (1e3, 1e7):
        xs = np.geomspace(19.2, xmax, 3000)
        J, Y = jv(nu, xs), yv(nu, xs)
        amplitude = np.hypot(J, Y)

        for rtol, atol in ((1e-8, 1e-10), (1e-12, 1e-14)):
            with contextlib.redirect_stdout(io.StringIO()):
                old = bessel_phase(nu, 1.02 * xmax, rtol=rtol, atol=atol)
                theta = np.array([
                    old["phase"].theta_mod_2pi(x) for x in xs
                ])
            print("existing", nu, xmax, rtol,
                  pair_error(np.sin(theta), -np.cos(theta), J, Y, amplitude),
                  "phi", old["phi"])

        lo, hi = np.sqrt(nu * nu - 0.25), 1.02 * xmax
        count = round(250 * np.log(hi / lo) + 0.5)
        grid = np.linspace(np.log(lo), np.log(hi), count)
        gx = np.exp(grid)
        S = (np.sqrt(np.pi * gx / 2) * hankel1e(nu, gx)
             * np.exp(1j * (np.pi * nu / 2 + np.pi / 4)))
        residual = np.unwrap(np.angle(S))
        rs = make_interp_spline(grid, residual, k=3)
        ls = make_interp_spline(grid, np.log(np.abs(S)), k=3)

        small = -np.pi * nu / 2 + np.pi / 4 + rs(np.log(xs))
        sine = np.sin(xs) * np.cos(small) + np.cos(xs) * np.sin(small)
        cosine = np.cos(xs) * np.cos(small) - np.sin(xs) * np.sin(small)
        ours_amplitude = np.sqrt(2 / (np.pi * xs)) * np.exp(ls(np.log(xs)))
        print("residual", nu, xmax,
              pair_error(sine, -cosine, J, Y, amplitude),
              "amplitude", np.max(np.abs(ours_amplitude / amplitude - 1)))
```

For independent direct-construction spot checks, use `mpmath` at 70 decimal digits and pass `mp.mpf(float_x)` so that the reference uses the same supplied argument. Compare normalized \(J,Y\) with angle addition from a direct `hankel1e` evaluation. To reproduce the turning-point experiment, evaluate at every log-grid midpoint starting from the construction lower bound, and compare cubic and quintic interpolants. That experiment deliberately exposes the inadequacy of fixed-density unwrapping at high order.

## 11. Mathematical references

- [NIST DLMF §10.18 — Modulus and phase functions](https://dlmf.nist.gov/10.18): exact amplitude–phase identities, including the Wronskian relation. DLMF's phase convention differs from the repository's by \(\pi/2\).
- [NIST DLMF §10.17 — Large-argument asymptotic expansions](https://dlmf.nist.gov/10.17): the leading Hankel oscillation and fixed-order residual behavior.
- [SciPy `hankel1e` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel1e.html): definition and implementation of the scaled Hankel function.
- [Bremer, phase function methods for second-order ODEs with turning points](https://arxiv.org/abs/2209.14561): the reference cited by the existing constructor; retaining its current ODE-based realization is not required for the replacement.
