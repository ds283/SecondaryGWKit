# Draft plan: accurate Bessel amplitude and phase construction

**Date:** 2026-09-08
**Revision:** 2. Revision 1 is commit `34d5cc0`; this document supersedes it and the diff against
that commit is the review record.
**Status:** proposal; production implementation has not been changed
**Scope:** the Bessel amplitude–phase representation, its numerical evaluation, and the consumers
needed to preserve its accuracy
**User clarifications:**
1. There is no requirement to retain the phase ODE. The phase function is the required output; the
   method of obtaining it may be replaced.
2. Whether an ODE-retaining variant is viable was asked explicitly. §5 answers it: the phase is a
   quadrature rather than an ODE, the obvious residual reformulation is not computable in double
   precision, and the viable variant converges on the same two-region structure recommended here.
3. Whether `phase_spline`'s disabled chunking and its `chunk_logstep=125` setting are the right
   choices was asked explicitly. §4.6 answers it.

## 0. What changed in revision 2

Every claim in revision 1 that was checkable was reproduced, and all of them held. The changes
below come from measurements revision 1 did not make.

| # | Change | Where |
|---|---|---|
| 1 | The asymptotic tail is **promoted from an optional later optimization to a required part of the construction**. This single change removes the `hankel1e` domain exposure, makes construction cost independent of `x_max`, and deletes the cycle-count and chunking machinery from this module. | §1, §6, §8 |
| 2 | New finding: `hankel1e` returns **exactly `-0j`, silently**, above x≈7.1e8 for ν≳100 and above x≈2.25e15 for all ν. A finite-value check does not catch it. | §4.4 |
| 3 | New finding: the residual is **worse** conditioned than the full phase near the turning point, not better. The split pays only for x ≳ 1.5ν. | §4.5 |
| 4 | New finding: the derivative, not the value, is the binding constraint, and revision 1's 1e-9 derivative target fails at fixed density. Derivative errors are now measured. | §4.3, §9 |
| 5 | New finding: reading θ′ off the amplitude interpolant as a *value* (`θ′ = e^{-2ℓ}`) beats differentiating the residual interpolant in 7 of 8 configurations. | §4.3, §7.3 |
| 6 | New finding: `phase_spline`'s chunking has **no measurable effect on accuracy**, its shipped setting cannot reduce the splined dynamic range by design, and the code contradicts its own comment. | §4.6 |
| 7 | The ODE-retaining alternative is analysed and its failure mode measured. | §5 |
| 8 | `theta_abserr`, an existing `AdaptiveLevin` hook for exactly this purpose, is added to the integration plan. | §8.1 |
| 9 | Revision 1's characterization of `raw_theta` as compatibility-only is **confirmed correct**, against an apparent contradiction in the Levin docstring. | §8.1 |
| 10 | The performance argument is promoted from a caveat to a motivation: the existing construction is a documented scalability blocker. | §1, §4.7 |
| 11 | New finding: the tail needs only **one** series. \(a_\nu\) follows from the phase series through the exact Wronskian, so DLMF 10.18.17 is not required. | §7.2 |
| 12 | User decision: the high-order acceptance target is reduced from \(10^{-10}\) to \(10^{-6}\), making correct branch tracking rather than sample density the binding requirement above \(\nu=20.5\). | §10, §11 |

## 1. Recommendation

Replace the normalized-phase ODE and the subsequent full-phase spline with a representation that
separates the known leading oscillation from a smooth correction:

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

Construct the normalized amplitude \(a_\nu\) and residual phase \(r_\nu\) in **two regions**:

- **Near region**, \(x\lesssim x_\star(\nu)\approx100\nu\): sample the exponentially scaled Hankel
  function, track the residual branch explicitly, and interpolate \(r_\nu\) and \(\log a_\nu\) in
  \(u=\log x\) with adaptive refinement and explicit error checks.
- **Tail**, \(x>x_\star(\nu)\): evaluate \(r_\nu\) and \(a_\nu\) from their fixed-order
  large-argument expansions in closed form. No samples, no interpolation, no branch tracking.

Preserve the leading term and correction separately when evaluating trigonometric functions and
combining phases.

This replaces the ODE rather than retuning it, and removes the separate phase-offset root solve.
There is no proposed change to the underlying Bessel equation or to the sine/cosine convention used
by the repository.

**The two-region structure is not an optimization; it is what makes the construction sound.**
Revision 1 treated the tail as optional and would have sampled `hankel1e` across the whole domain.
That is the one part of revision 1 that does not survive review: `hankel1e` fails silently at large
argument (§4.4), and sampling to \(x_{\max}\) makes both the node count and the peak splined phase
grow with \(x_{\max}\) for no benefit, since the tail is where the correction is analytically known
and smallest. With the crossover at \(\sim100\nu\), `hankel1e` is never evaluated within three
decades of its failure boundary, the interpolated domain spans a fixed \(\approx4.6\) e-folds
regardless of \(x_{\max}\), and \(r_\nu\) never exceeds \(O(1)\) — so the cycle-count split and the
chunking machinery are not needed here at all.

A secondary motivation is scalability, which revision 1 listed only as a caveat. The existing
construction is a **documented blocker**: `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py:275`
records that a `bessel_phase` build for \(x_{\max}\sim8\times10^{15}\) does not complete in ~25
minutes and caps the production scan at \(\kappa=100\) — *"the phase layer, not the Levin core, is
the scalability limit."* Measured here, the ODE build does not complete in 600 s at
\(x_{\max}=10^{13}\). The two-region construction is O(1) in \(x_{\max}\).

## 2. Context and evidence reviewed

The workstream B campaign introduced a transfer-function representation and restricted the sampled
quadratic source to its appropriate region. Its constant-equation-of-state fixtures use the Bessel
amplitude–phase construction as an analytic reference. An error in that reference consequently
limits what those tests can establish.

The review covered:

- [Campaign README](../source-remediation/README.md), especially the exact constant-\(w\) fixture
  and closed-form LG amplitude.
- [Workstream B orchestration prompt](../source-remediation/orchestrator/workstream-B.md).
- [Prompt 05 log](../source-remediation/logs/05-tk-source-functions.md), especially deviation 7,
  verification results, and its superseding note.
- [Prompt 06 log](../source-remediation/logs/06-quadsource-regions.md), for the distinct
  source-spline accuracy floor and downstream context.
- [Phase and hand-over follow-up](../../docs/lg-phase-and-handover-followup-2026-09.md), especially
  §§2.4–2.5.
- [Bessel construction](../../LiouvilleGreen/bessel_phase.py),
  [phase spline](../../LiouvilleGreen/phase_spline.py),
  [range reduction](../../LiouvilleGreen/range_reduce_mod_2pi.py), and
  [Bessel tests](../../LiouvilleGreen/tests/test_bessel_phase.py).
- [Production phase integration](../../Quadrature/integrators/WKB_phase_function.py),
  [production Bessel setup](../../main.py), and
  [three-Bessel consumers](../../LiouvilleGreen/three_bessel_integrals.py).
- Added in revision 2: [Levin quadrature](../../AdaptiveLevin/levin_quadrature.py) (the phase
  interface, `phase_span`, and `theta_abserr`), and
  [the Bessel benchmark tier](../../docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py) (the
  recorded scalability limit).

The experiments reported below were temporary review calculations against the current
implementation. They did not modify production code. §12 supplies reproductions that do not depend
on temporary files.

## 3. Reproduction of revision 1's central comparison

Every figure in revision 1's headline table reproduced to the quoted digits, independently:

| Order | Construction | \(E_\theta\) to \(10^3\) | \(E_\theta\) to \(10^7\) |
|---|---|---:|---:|
| \(3/2\) | Existing defaults | 2.0194e-06 | 1.5883e-03 |
| \(3/2\) | Existing, `rtol=1e-12` | 1.2028e-08 | 5.9100e-06 |
| \(3/2\) | Scaled-Hankel residual, cubic | 3.0420e-14 | 2.7922e-14 |
| \(7/4\) | Existing defaults | 1.5653e-06 | 2.1186e-03 |
| \(7/4\) | Existing, `rtol=1e-12` | 2.1012e-08 | 6.4689e-06 |
| \(7/4\) | Scaled-Hankel residual, cubic | 4.5353e-14 | 4.5269e-14 |

Normalized-amplitude errors were 1.40e-14 to 4.65e-14 across the same cases.

The algebra of §6.1 is **exact, not asymptotic**. Under the repository's convention
\(H^{(1)}_\nu=A_\nu e^{i(\theta_\nu-\pi/2)}\), and substituting \(\theta=x+c_\nu+r_\nu\) with
\(c_\nu=\pi/4-\pi\nu/2\) makes the two \(\pi/4\) terms and the \(-\pi/2\) cancel identically,
collapsing \(S_\nu\) to \(a_\nu e^{ir_\nu}\). The DLMF phase convention differs from the
repository's by exactly \(+\pi/2\) (\(\theta_{\rm repo}=\theta_{\rm DLMF}+\pi/2\)), which is what
turns DLMF 10.18.18's \(-(\tfrac12\nu+\tfrac14)\pi\) into \(c_\nu\). The Wronskian
\(A^2\theta'=2/(\pi x)\) then gives \(\theta'=a^{-2}\), and hence the exact identity
\(dr/d\log x=x(a^{-2}-1)\) used throughout below.

## 4. Findings about the existing construction

### 4.1 Normalizing the phase does not bound absolute phase error

The Bessel constructor evolves \(Q=\theta/x\) in \(u=\log x\):

\[
\frac{dQ}{du}=\frac{2}{\pi x(J_\nu^2+Y_\nu^2)}-Q .
\]

The state remains of order unity at large \(x\), which is useful for numerical scaling. But
reconstruction gives \(\delta\theta=x\,\delta Q\). The relevant error for \(\sin\theta\) and
\(\cos\theta\) is absolute phase error in radians; controlling the relative error of a state
approaching unity provides no uniform absolute-phase bound as \(x\) increases. ODE local error
tolerances are also not certificates of global solution error.

This supports the follow-up's attribution of the *default-tolerance* Bessel error to phase
evolution. It also qualifies the discussion of the Green's-function normalization: a representation
\(\theta=\theta_0+LQ\) has the same amplification \(L\,\delta Q\). The normalization can help, but
its presence alone does not establish that integration error is negligible. This review has not
measured the Green's-function production solve and does not assign it a new numerical error
estimate.

### 4.2 Full-phase interpolation introduces a separate growing error

After integrating \(Q\), the constructor forms \(xQ\), reduces it to cycles and remainder, and
constructs a `phase_spline` in \(\log x\). For a leading phase \(\theta\simeq x=e^u\), the fourth
derivative with respect to \(u\) is also approximately \(x\), so cubic interpolation has an error of
order \(h^4x\). Subtracting an integer number of cycles changes the constant term, not the fourth
derivative.

At 250 samples per e-fold the interior scale \(h^4x/384\) is \(\approx6.7\times10^{-10}\) at
\(x=10^3\) and \(\approx6.7\times10^{-6}\) at \(x=10^7\). Both are confirmed: the tight-tolerance
measurements in §3 are 1.2e-8 (offset-dominated, see §4.3) and 5.91e-6. §4.6 confirms
independently, on exact phase samples, that this mechanism accounts for the entire error and that
chunking does not affect it.

Evaluating the existing `Q` spline directly can reduce this particular interpolation error. It still
multiplies its state and interpolation errors by \(x\), retains the ODE error, and requires correct
handling of the phase offset. It is not the recommended construction.

### 4.3 The phase-offset root solve adds an avoidable floor, and it dominates at production tolerances

The constructor first determines a phase from `asin(J/sqrt(J*J+Y*Y))`, then solves a scalar
matching equation against \(J\) with `xtol=1e-6, rtol=1e-4`, and subtracts the resulting offset
`phi`.

For every \(\nu>1/2\) the domain lower bound is \(\min x=\sqrt{\nu^2-\tfrac14}>1\), so
\(\log\min x>0\) and the match point **is the initial node** — where the phase was already fixed
exactly from the Bessel value. The matching function is then
\(\sqrt m\,\sin(\operatorname{asin}(J/\sqrt m)-\phi)-J\), which vanishes identically at \(\phi=0\).
The offset is entirely an artefact of the root solve's own loose tolerances. Confirmed, and the
offset is the whole tight-tolerance error:

| Order | Computed `phi` | \(E_\theta\) as shipped | \(E_\theta\) with `phi` undone |
|---|---:|---:|---:|
| \(3/2\) | \(-1.149353\times10^{-8}\) | \(1.2028\times10^{-8}\) | \(7.42\times10^{-10}\) |
| \(7/4\) | \(-2.044386\times10^{-8}\) | \(2.1012\times10^{-8}\) | \(8.12\times10^{-10}\) |
| \(5/2\) | \(-4.836537\times10^{-8}\) | \(4.8774\times10^{-8}\) | \(8.92\times10^{-10}\) |

\(E_\theta\approx|\phi|\) to two digits, and removing it lands on the \(6.7\times10^{-10}\)
interpolation floor of §4.2. For \(\nu=1/2\) the match point is instead \(x=1\), an interior point,
so that case is genuinely different and is not covered by this argument.

This matters for the interpretation of the follow-up. `main.py:429-437` already builds its Bessel
objects with `rtol=5e-14, atol=1e-25` (orders \(1/2\) and \(5/2\); `b_value = 0.0`), whereas the
fixtures use the `config/defaults.py` values `rtol=1e-8, atol=1e-10`. So the error budget
decomposes into three mechanisms, each dominant in its own regime:

| regime | measured | mechanism |
|---|---:|---|
| defaults, \(x\le10^3\) | 2.0e-6 | ODE error \(x\,\delta Q\) |
| tight tolerances, \(x\le10^3\) | 1.2e-8 | **spurious `phi`** |
| tight tolerances, \(x\le10^7\) | 5.9e-6 | full-phase interpolation \(h^4x/384\) |

The follow-up's blanket \(x\times10^{-8}\) figure is therefore correct only at the fixture
tolerances. At the tolerances production actually uses, the dominant error is a constant offset that
should not exist, costing a factor of ~50.

If a phase anchor is needed, use \(\theta_0=\operatorname{atan2}(J_\nu(x_0),-Y_\nu(x_0))\) with an
explicit branch convention: it uses both Bessel functions and avoids a root solve. Removing `phi`
indiscriminately from the old implementation is not proposed as a general branch-handling fix.

### 4.4 `hankel1e` fails silently at large argument, and a finite-value check does not catch it

*New in revision 2. This is the finding that makes the tail mandatory.*

Validating the **samples themselves** (no interpolation) against `mpmath` at 60 digits, the scaled
Hankel route is excellent where it works and catastrophic where it does not:

| ν | \(\max\lvert\delta r\rvert\) (rad) | \(\max\lvert\delta a/a\rvert\) | usable to |
|---|---:|---:|---|
| \(1/2\)–\(5/2\) | ~1e-16 | ~1e-16 | 2.25e15 |
| 20.5 | 1.09e-14 | 1.42e-15 | 2.25e15 |
| 100.5 | 1.26e-14 | 1.14e-15 | **7.1e8** |
| 1000.5 | 2.96e-13 | 1.00e-14 | **7.1e8** |

Beyond those bounds `hankel1e` returns **exactly `-0j`**, with no warning and no exception:
`hankel1e(100.5, 1e9) == -0j`. Consequently `np.abs → 0`, `np.log(np.abs) → -inf`, and
`np.angle(-0j) → -0.0`. Revision 1's Stage 2 called for "finite-value and construction-failure
checks", but **`-0j` is finite**, so that guard passes and the failure propagates into the
log-amplitude interpolant as `-inf`. The guard must be a plausibility band on \(a_\nu\), which is
\(\gtrsim1\) above the turning point.

Two qualifications, so the finding is not overstated. First, this is a **pre-existing SciPy/Amos
limitation, not a regression**: `yv(1000.5, 1e10)` is also exactly `-0.0`, which corrupts
\(m=J^2+Y^2\) in the current ODE's right-hand side too, and the failure is **not monotonic in
\(x\)** (`jv`/`yv` recover at \(10^{16}\) for \(\nu=1000.5\) while failing at \(10^{10}\)), so no
simple ceiling can be certified for the `jv`/`yv` route either. Second, the plan's own acceptance
table never combined high order with large argument, so revision 1's stated targets were reachable.
The problem is that a constructor promising 1e-11 must not accept arbitrary \((\nu,x_{\max})\) and
silently return `-inf` splines.

With the mandatory tail of §1 this exposure disappears rather than merely being guarded: the
crossover at \(\sim100\nu\) is at most \(\sim10^5\) even for \(\nu=1000.5\), three decades below
7.1e8. The guard and the documented supported domain are still required, as defence in depth.

### 4.5 The residual is worse conditioned than the full phase near the turning point

*New in revision 2. This determines the refinement design.*

Revision 1's §5.1 said the residual's logarithmic derivatives "decay in the tail, whereas those of
the old full phase grow like \(x\)". True, but it describes only the tail. From the exact identity
\(dr/d\log x=x(a^{-2}-1)\):

| ν | \(x/\nu\) | \(\lvert d\theta/d\log x\rvert\) | \(\lvert dr/d\log x\rvert\) | gain |
|---|---:|---:|---:|---:|
| 1.5 | 1.0 | 0.943 | 0.471 | 2.0 |
| 100.5 | 1.0 | 17.2 | 83.3 | **0.21** |
| 1000.5 | 1.0 | 79.6 | 920.9 | **0.086** |
| 1000.5 | 1.5 | 1119 | 382 | 2.9 |
| 1000.5 | 100 | 1.0e5 | 5.0 | 2.0e4 |

At the turning point the residual is **11× worse** conditioned than the full phase at
\(\nu=1000.5\). The split pays only for \(x\gtrsim1.5\nu\), and pays enormously only in the tail.

This is also the mechanism behind revision 1's reported \(\nu=1000.5\) failure. At 250 samples per
e-fold the residual advance per interval is 0.0019 rad at \(\nu=1.5\), 0.054 at 20.5, 0.289 at
100.5, and 2.92 at 1000.5 just above the turning point — rising to 3.68 rad, **exceeding \(\pi\)**,
as \(x\to\sqrt{\nu^2-\tfrac14}\). Ordinary `unwrap` cannot recover, and raising the interpolation
degree cannot repair incorrectly unwrapped samples.

Two consequences for the design. Refinement near the turning point is **mandatory and
order-dependent**, not a nicety; and the adaptivity must be **two-sided**. In the tail
\(r(u)\approx Ce^{-u}\), so every \(u\)-derivative is as small as \(r\) itself and the required
density collapses far below 250 per e-fold. Revision 1 discussed only refining. Under §1's
recommendation the tail is closed-form and the question does not arise, but the near region still
needs both directions.

### 4.6 `phase_spline`'s chunking: the code contradicts its comment, and the setting achieves nothing

*New in revision 2, in answer to user clarification 3.*

`bessel_phase.py:260-270` carries the comment "currently force spline to use a single chunk /
multi-chunk implementation seems currently not mature enough for production use", but passes
`chunk_logstep=125`. Since that is not `None`, `phase_spline.__init__` takes the
`elif chunk_logstep is not None` branch (`phase_spline.py:259`) and builds log-spaced chunks. The
single-chunk path is the commented-out `chunk_logstep=None` line. Measured: **2 chunks at
\(x_{\max}=10^3\), 4 at \(10^7\), 6 at \(10^{12}\)**, with boundaries at
\(x\approx591,\,5.5\times10^4,\,5.2\times10^6,\,4.9\times10^8,\,4.6\times10^{10}\). The comment is
false, and a reader currently believes the multi-chunk path is disabled in production.

Whether it matters was measured directly. Machine-accurate phase samples were built from the scaled
Hankel function — so the comparison contains no ODE error — and identical
\((\operatorname{div}2\pi,\operatorname{mod}2\pi)\) arrays were fed to `phase_spline` under
different configurations. Any difference is the chunking:

| \(x_{\max}\) | config | chunks | \(E_\theta\) | max rebased \(\lvert y\rvert/\theta_{\max}\) |
|---|---|---:|---:|---:|
| 1e3 | single chunk | 1 | 6.420e-10 | 1.000 |
| 1e3 | **logstep=125 (shipped)** | 2 | **6.422e-10** | 0.783 |
| 1e3 | logstep=4 | 5 | 6.421e-10 | 0.510 |
| 1e7 | single chunk | 1 | 6.225e-06 | 1.000 |
| 1e7 | **logstep=125 (shipped)** | 4 | **6.223e-06** | 0.673 |
| 1e7 | logstep=4 | 13 | 6.224e-06 | 0.673 |
| 1e7 | logstep=20 | 6 | 6.223e-06 | 0.590 |
| 1e10 | single chunk | 1 | 5.465e-03 | 1.000 |
| 1e10 | **logstep=125 (shipped)** | 5 | **5.464e-03** | 0.952 |
| 1e10 | logstep=4 | 20 | 5.465e-03 | 0.715 |
| 1e10 | logstep=20 | 8 | 5.464e-03 | 0.895 |

Identical to four significant figures in every configuration at every \(x_{\max}\), and equal to
\(h^4x/384\) throughout (6.7e-6 predicted at \(10^7\), 6.7e-3 at \(10^{10}\)). **Chunking has no
measurable effect on accuracy.** This is §4.2's argument measured rather than inferred, and it
disposes of the 05 log's superseded claim that the floor is set by the phase range per chunk.

The setting is nevertheless the wrong knob, for a structural reason. A chunk running from `start` to
`125·start` in cycle count spans a factor 125 in \(\theta\), so rebasing to the chunk base removes
only \(\sim1/\)`logstep` of the chunk's range — 0.8% in principle, and measured **5% at
\(x_{\max}=10^{10}\)**. Geometric chunking in cycle count cannot bound the dynamic range of the
splined phase. Bounding it requires a bounded number of cycles per chunk, i.e. the linear
`chunk_step`; but `_build_linear_chunks` advances by \(0.75\times\)`chunk_step` per iteration, so
for a Bessel phase of \(10^{12}\) rad it runs \(1.59\times10^{11}/150\approx1.06\times10^9\)
iterations and hangs. That is presumably why `chunk_logstep` exists. So `chunk_logstep=125` is not a
tuned value; it is a value that avoids both failure modes while accomplishing nothing.

The class docstring's premise is also the wrong diagnosis. "Avoid loss of precision when the result
of the spline is evaluated mod 2pi" describes a floating-point problem; the error is interpolation
error, which no rebasing addresses.

**Under §1's recommendation this becomes moot for `bessel_phase`.** With a closed-form tail, the
interpolated residual is \(O(1)\) and never exceeds a cycle, so `simple_mod_2pi` is unnecessary,
\(\operatorname{div}2\pi\) is identically zero, and there is nothing to chunk. The
`(div_2pi, mod_2pi)` representation and the `phase_spline` dependency both leave this module.

`phase_spline` must remain for the cosmological phases, where \(\theta\) genuinely reaches
\(10^{15}\) and there is no closed-form leading term to subtract. Two items follow for it, both
**out of scope here** and listed in §11:

- Its chunking should be re-examined on its own terms. The measurement above says it is unlikely to
  be helping there either, but the cosmological case has a different \(\theta(u)\) and must be
  measured, not assumed.
- `_build_log_chunks_positive` has no progress guard: its next start is
  `round(0.75*end - 0.5)`, which does not advance for small `logstep`. `logstep=1.2` and
  `logstep=1.5` both stick at `(1, 2)` and loop forever. Nothing currently passes such a value, so
  this is latent, but the loop should assert progress.

### 4.7 Cost, and the derivative as the binding constraint

*Extended in revision 2.*

Construction cost, ν=5/2, matched settings (`atol=1e-25, rtol=5e-14`, 250 per e-fold):

| \(x_{\max}\) | ODE build | scaled-Hankel build | speedup |
|---|---:|---:|---:|
| 1e3 | 0.02 s | 0.001 s | 14× |
| 1e7 | 0.03 s | 0.003 s | 13× |
| 1e11 | 0.05 s | 0.004 s | 11× |
| 1e13 | **>600 s (did not complete)** | 0.005 s | — |

The \(10^{13}\) row independently confirms the benchmark's recorded limit (§1). Revision 1's "do not
assume it improves" was too cautious; the improvement is the removal of a standing blocker.

Derivative accuracy was **not measured in revision 1**, although §8 set a 1e-9 target for it. It is
the harder quantity by two to three orders of magnitude, against the exact oracle
\(\theta'=(2/\pi)/(x(J^2+Y^2))\), measured at sample-grid midpoints:

| ν | cubic, 250/e-fold | quintic, 250/e-fold | quintic, 1000/e-fold |
|---|---:|---:|---:|
| 3/2 | 1.52e-09 | 1.94e-13 | 5.24e-13 |
| 5/2 | 6.07e-09 | 8.05e-13 | 6.72e-13 |
| 20.5 | 5.30e-07 | 1.96e-10 | 1.35e-12 |
| 100.5 | 1.13e-05 | **8.06e-08** | 3.16e-12 |

Cubic fails 1e-9 at *every* order. Quintic at 250 per e-fold fails at \(\nu=100.5\). **Every
maximum lies in the interval adjacent to the turning point**, consistent with §4.5. The target is
reachable but only with order-dependent refinement there, which §9 now states.

Two further results. First, reading \(\theta'\) off the amplitude interpolant as a *value* rather
than differentiating the residual interpolant is consistently better — \(\theta'=a^{-2}=e^{-2\ell}\)
won 7 of 8 configurations, by 4× to 15× (4.66e-14 vs 1.94e-13 at \(\nu=3/2\); 2.09e-8 vs 8.06e-8 at
\(\nu=100.5\), both quintic at 250 per e-fold). Second, the Wronskian residual
\(\lvert a^2\theta'-1\rvert\) tracked the derivative error to two digits in every row (1.515e-9 vs
1.523e-9, and so on). That makes it a sharp, self-contained detector of the dominant error — a
stronger endorsement than revision 1 gave it — but it also means that a Wronskian check on a
representation whose \(\theta'\) is *derived from* \(a\) is close to a tautology. §7.3 addresses
both.

### 4.8 Amplitude error is small in the measured low-order interval

For \(19.2\le x\le10^3\) or \(10^7\), the existing amplitude spline agreed with \(\sqrt{J^2+Y^2}\)
at roughly \(10^{-13}\) relative error or better in the tested low-order cases. The principal defect
there is phase accuracy.

Normalizing the amplitude is nevertheless worthwhile: it factors out known algebraic variation,
preserves positivity when interpolating its logarithm, makes the far tail approach a constant, and —
per §4.7 — supplies the better route to \(\theta'\). These are structural improvements, not evidence
that amplitude error dominated the workstream B findings.

## 5. The ODE-retaining alternative

*New in revision 2, in answer to user clarification 2.*

### 5.1 The phase is a quadrature, not an ODE

The exact phase relation is

\[
\frac{d\theta}{dx}=\frac{2/\pi}{x\,m(x)},\qquad m=J_\nu^2+Y_\nu^2 ,
\]

whose right-hand side **does not contain \(\theta\)**. The phase is a quadrature. The substitution
\(Q=\theta/x\) manufactures a linear ODE, \(dQ/du=(2/\pi)/(x m)-Q\), out of a pure integral, and then
pays twice: an adaptive stepper that was not needed, and the \(\delta\theta=x\,\delta Q\)
amplification of §4.1. Recognizing this is what makes the offset finding of §4.3 and the
reformulation below available at all.

### 5.2 The obvious residual reformulation is not computable

Writing \(\theta=x+c_\nu+r\) gives the residual quadrature

\[
\frac{dr}{d\log x}=\frac{2/\pi}{m(x)}-x=x\left(a^{-2}-1\right),
\]

which has the right conditioning in principle — \(\delta r\) *is* \(\delta\theta\), with no
amplification. **It fails in practice, and not because of the integrator: the right-hand side cannot
be evaluated in double precision.** Since \((2/\pi)/m\to x\) in the tail, this subtracts two nearly
equal quantities of size \(x\) to obtain one of size \(\nu^2/x\):

| \(x\) | true RHS | naive \((2/\pi)/m-x\) | rel err |
|---|---:|---:|---:|
| 1e2 | \(-3.00000\times10^{-2}\) | \(-3.00000\times10^{-2}\) | 8.0e-13 |
| 1e4 | \(-3.00000\times10^{-4}\) | \(-3.00000\times10^{-4}\) | 1.1e-08 |
| 1e6 | \(-3.00000\times10^{-6}\) | \(-3.00037\times10^{-6}\) | 1.2e-04 |
| 1e8 | \(-3.00000\times10^{-8}\) | \(-1.49012\times10^{-8}\) | **5.0e-01** |
| 1e10 | \(-3.00000\times10^{-10}\) | \(-1.90735\times10^{-6}\) | 6.4e+03 |
| 1e12 | \(-3.00000\times10^{-12}\) | \(-2.44141\times10^{-4}\) | 8.1e+07 |

Total loss of significance by \(x\approx10^8\); by \(10^{12}\) the computed right-hand side is eight
orders of magnitude wrong. The accumulated error of such a quadrature is
\(\int\varepsilon x\,d\log x\sim\varepsilon x_{\max}\) — the same \(\varepsilon x\) floor as the raw
phase, i.e. 0.22 rad at \(x=10^{15}\). **No integrator or tolerance can repair this.** A residual ODE
or quadrature driven by the closed-form right-hand side is therefore not viable on its own.

### 5.3 What remains viable, and why it converges on the same design

Escaping §5.2 requires \(a^{-2}-1\) from its asymptotic series — and once that series is available,
\(r\) is available directly and no integration is needed in the tail at all. The viable
ODE-retaining design is therefore a hybrid: quadrature near the turning point (where
\(a^{-2}\approx0.08\)–\(0.17\) and there is no cancellation), asymptotic series in the tail. That
is the same two-region structure as §1.

Measured, the DLMF 10.18.18 series against `mpmath` at 60 digits:

| ν | 2 terms | 3 terms |
|---|---|---|
| 5/2 | 1.77e-10 at \(x=125\) | 3.68e-17 at \(x=2500\) |
| 20.5 | 7.64e-10 at \(x=1025\) | 1.60e-16 at \(x=2.05\times10^4\) |
| 100.5 | 4.01e-09 at \(x=5025\) | 8.34e-16 at \(x=1.005\times10^5\) |

Machine precision for \(x\gtrsim50\)–\(100\nu\). The cancellation floor of §5.2 is relative
\(\varepsilon\cdot8x^2/(4\nu^2-1)\), which stays below \(10^{-8}\) out to \(x\lesssim4800\nu\). So
there is a wide safe matching window \([100\nu,\,4800\nu]\) and no delicate crossover.

The hybrid has three genuine advantages over the recommendation of §1:

1. It needs only the **modulus** \(m=J^2+Y^2\), never \(\arg H^{(1)}_\nu\), so it does not make
   SciPy's Hankel *phase* load-bearing (§10) and does not touch the `hankel1e` boundaries of §4.4 at
   all.
2. \(\theta'=a^{-2}\) is a **primitive** rather than a differentiated interpolant, which makes the
   Wronskian a genuine independent check instead of the near-tautology measured in §4.7.
3. Construction is O(1) in \(x_{\max}\) — though §1's recommendation shares this once its tail is
   mandatory.

It is not recommended, for two reasons. It does not dodge the hard part: the near region still needs
an interpolant with exactly the conditioning of §4.5, and the derivative there is still binding
(§4.7). And it is more implementation — two regimes plus a quadrature scheme plus a remainder test —
where the scaled-Hankel route is already validated to ~1e-16 below the crossover (§4.4).

**The two designs share their hard part and differ only in what supplies the near-region data.** If
the scaled-Hankel near region should later prove unsatisfactory — for instance if a SciPy change
degrades `hankel1e`'s phase — §5.3 is the drop-in replacement for that region alone, with the tail,
the interpolation scheme, the evaluation path and the consumers unchanged. §7.1 keeps the near-region
sampler behind a boundary so that substitution stays cheap.

## 6. Numerical experiments not already tabulated

### 6.1 Error definition

Let \(A=\operatorname{hypot}(J,Y)\) from the reference functions. Define

\[
E_\theta=\max_x\max\left(
\left|\sin\theta_{\rm ours}-J/A\right|,
\left|-\cos\theta_{\rm ours}-Y/A\right|\right),
\qquad E_A=\max_x\left|A_{\rm ours}/A-1\right| .
\]

This is a phase-pair error, not an unwrapped phase difference. For small errors it measures the same
local effect on Bessel values, normalized by their envelope, without dividing by a function near a
zero.

### 6.2 Turning-point and higher-order checks

All log-interval midpoints from the construction lower bound to \(\max(1000,10\nu)\), cubic and
quintic at 250 samples per e-fold:

| Order | Cubic \(E_\theta\) | Quintic \(E_\theta\) | Quintic \(E_A\) |
|---|---:|---:|---:|
| \(3/2\) | \(4.07\times10^{-12}\) | \(1.02\times10^{-14}\) | \(2.38\times10^{-14}\) |
| \(7/4\) | \(7.21\times10^{-12}\) | \(1.88\times10^{-14}\) | \(3.29\times10^{-14}\) |
| \(5/2\) | \(2.37\times10^{-11}\) | \(1.01\times10^{-14}\) | \(2.29\times10^{-14}\) |
| \(20.5\) | \(8.62\times10^{-9}\) | \(2.24\times10^{-12}\) | \(2.04\times10^{-11}\) |
| \(100.5\) | \(5.29\times10^{-7}\) | \(2.94\times10^{-9}\) | \(1.04\times10^{-8}\) |

At \(\nu=1000.5\) the naive fixed-grid prototype failed with order-unity phase-pair errors; §4.5
supplies the mechanism and the numbers. Increasing uniform density and using quintic interpolation
gave \(E_\theta=9.66\times10^{-9}\), \(9.08\times10^{-11}\), \(9.70\times10^{-12}\) at 1000, 2000 and
4000 samples per e-fold. These are diagnostic refinement results, not a recommendation to impose a
4000-point density anywhere; they motivate adaptive sampling concentrated where the order and
proximity to the turning point require it.

### 6.3 Large-argument evaluation and the split

`np.sin` and `math.sin` were confirmed correctly rounded against `mpmath` out to \(x=10^{16}\) on
this platform, so the angle-addition path of §7.4 rests on a verified premise. The cost of *not*
splitting is large — forming `x + d` first, versus angle addition, for \(\nu=3/2\):

| \(x\) | naive \(\sin(x+d)\) | angle addition |
|---|---:|---:|
| 1e3 | 3.20e-14 | 1.11e-16 |
| 1e7 | 1.21e-10 | 0 |
| 1e12 | 2.72e-06 | 1.11e-16 |
| 1e15 | **4.73e-02** | 1.11e-16 |

Fourteen orders of magnitude at \(x=10^{15}\). Revision 1 motivated the split correctly but did not
quantify it.

Direct scaled-Hankel construction and angle-addition reconstruction were also compared with 70-digit
`mpmath` Bessel values at \(x=10^3,10^7,10^{12},10^{15}\) for orders \(3/2\) and \(7/4\), with
phase-pair discrepancies below \(1.4\times10^{-16}\). Those checks used the supplied floating-point
argument directly and did not test a spline over the range; note also that \(10^{15}\) is within a
factor 2.25 of the universal `hankel1e` boundary of §4.4, which is one more reason the tail must be
closed-form rather than sampled.

## 7. Proposed mathematical construction

### 7.1 Near region: amplitude and phase from a scaled Hankel function

SciPy defines \(\operatorname{hankel1e}(\nu,x)=e^{-ix}H_\nu^{(1)}(x)\) with
\(H_\nu^{(1)}=J_\nu+iY_\nu\). Construct

\[
S_\nu(x)=\sqrt{\frac{\pi x}{2}}\,
e^{i(\pi\nu/2+\pi/4)}\operatorname{hankel1e}(\nu,x)
\;=\;a_\nu e^{ir_\nu}
\]

exactly, by the cancellation shown in §3. Therefore \(a_\nu=|S_\nu|\) and \(r_\nu\) is its
continuously tracked argument. The construction never obtains the residual by subtracting \(x\) from
a large computed phase; the scaled routine supplies the oscillation-removed quantity directly.

Sample only for \(x\le x_\star(\nu)\). Validate every sample against a plausibility band on
\(a_\nu\) — **not** `isfinite`, per §4.4 — and fail construction loudly if any sample is rejected.

Keep this sampler behind a narrow internal boundary that returns \((a_\nu,r_\nu)\) on a grid, so
that the §5.3 quadrature can replace it without touching the tail, the interpolation, the evaluation
path or the consumers.

### 7.2 Tail: closed form

For \(x>x_\star(\nu)\) evaluate the fixed-order large-argument expansions directly. With
\(\mu=4\nu^2\), DLMF 10.18.18 in the repository's convention gives

\[
r_\nu(x)=\frac{\mu-1}{8x}+\frac{(\mu-1)(\mu-25)}{384x^3}
+\frac{(\mu-1)(\mu^2-114\mu+1073)}{15360x^5}+\cdots
\]

**The amplitude needs no separate series.** The Wronskian \(\theta'=a^{-2}\) is exact, and
\(\theta=x+c_\nu+r\), so

\[
a_\nu(x)=\bigl(1+r_\nu'(x)\bigr)^{-1/2}
\]

with \(r_\nu'\) obtained by differentiating the expansion above term by term. Measured against
`mpmath`, the relative error of \(a_\nu\) from this route matches the phase series' own convergence:
with 2 terms, \(1.9\times10^{-12}\) at \(x=50\nu\) and \(3.1\times10^{-14}\) at \(x=100\nu\), for
both \(\nu=20.5\) and \(\nu=100.5\); exact for \(\nu=1/2\), where \(\mu-1=0\) gives \(a\equiv1\) and
\(r\equiv0\) as it must. So DLMF 10.18.17 is **not** required, one series governs both quantities,
and the crossover test of §5.3 covers the amplitude automatically. §5.3 measures the phase accuracy
achieved and the safe crossover window.

Choose \(x_\star(\nu)\) by a **remainder test at the requested accuracy**, not by a fixed constant
and not by a fixed multiple of \(\nu\): evaluate the first omitted term, and require in addition that
the near-region interpolant and the series agree to the accuracy budget at the crossover. Record
\(x_\star\) and the agreement achieved. The series is asymptotic, so its remainder must be tested,
never assumed.

This region needs no samples, no interpolation, no branch tracking, and no cycle-count
representation, and it is where the overwhelming majority of the domain lies.

### 7.3 Branch tracking, interpolation and derivatives

A principal complex argument alone is insufficient to construct a differentiable unwrapped phase.
The near-region implementation must:

1. Establish an anchor consistent with both \(J\) and \(Y\), and document the allowed constant
   integer-cycle offset.
2. Refine before accepting an interval whose residual variation could conceal a wrap. Use the exact
   \(dr/d\log x=x(a^{-2}-1)\) as the variation estimator, with the caveat that it cancels in the tail
   (§5.2) — which is harmless, because the tail is closed-form and the estimator is needed only where
   \(a^{-2}-1\) is \(O(1)\).
3. Track the continuous residual branch through accepted intervals.
4. Validate continuity and derivative behaviour across refinement and interpolation boundaries.

Do not rely on endpoint principal-angle differences alone: an interval may contain an undetected full
turn. An integer \(2\pi\) offset does not change Bessel values, but inconsistent offsets between
intervals invalidate interpolation and can confuse consumers that inspect raw phase differences.

Interpolate \(r(u)\) and \(\ell(u)=\log a(e^u)\) with \(u=\log x\). Quintic is the reasonable
starting candidate on the §4.7 and §6.2 evidence, but degree alone is not the acceptance criterion:
check at additional points, refine, and include endpoint intervals. Adaptivity must be **two-sided**
(§4.5). Piecewise Chebyshev interpolation is a valid alternative if it makes error estimation
simpler.

Reconstruct

\[
A(x)=\sqrt{\frac{2}{\pi x}}e^{\ell(\log x)},\qquad
\frac{d\log A}{dx}=-\frac1{2x}+\frac{\ell_u(\log x)}x,
\]

and take the phase derivative from the **amplitude interpolant as a value**,

\[
\theta'(x)=a^{-2}=e^{-2\ell(\log x)},
\]

rather than from \(1+r_u(\log x)/x\). §4.7 measures this as better in 7 of 8 configurations and it
avoids differentiating an interpolant.

That choice changes what the Wronskian check means, and the change must be handled explicitly. With
\(\theta'\) derived from \(\ell\), the identity \(a^2\theta'=1\) is satisfied by construction and
checks nothing. So verify instead that

- \(e^{-2\ell}\) and \(1+r_u/x\) agree to the budget — an independent consistency check between the
  two interpolants, which is what §4.7 measured; and
- both agree with \((2/\pi)/(x(J^2+Y^2))\) from the reference functions.

Revision 1 warned that substituting \(a^{-2}\) for a poor spline derivative can hide inconsistency.
That warning is correct and is why the first check above is required rather than optional.

### 7.4 Preserve the split during evaluation

Let \(d=c_\nu+r_\nu(x)\). Evaluate

\[
\sin\theta=\sin x\cos d+\cos x\sin d,\qquad
\cos\theta=\cos x\cos d-\sin x\sin d .
\]

At large \(x\), forming `x + d` first rounds away part or all of the correction (§6.3). Angle
addition avoids that loss and lets the platform trigonometric routines reduce the original argument
internally, which §6.3 confirms they do correctly to \(10^{16}\).

Similarly, reducing a huge phase against a double-precision `TWO_PI` creates an error proportional to
the cycle count. A bounded-angle accessor can instead use `atan2(sin_theta, cos_theta)`, with any
documented interval convention applied only to this bounded result. If a consumer genuinely requires
an accurate integer cycle count, design and validate that separately; a good bounded angle is not a
cycle-count algorithm.

Keep `raw_theta` for compatibility and diagnostics, documenting its \(\varepsilon x\) precision
limit. Accurate oscillatory evaluation must use the split or bounded-angle path.

### 7.5 Argument accuracy has its own limit

The representation can accurately evaluate Bessel functions at the supplied floating-point \(x\). It
cannot recover uncertainty already present in \(x=k\eta\), nor undo a lossy `exp(log(x))` round trip.

When raw \(x\) is supplied, preserve it for the leading oscillation and use \(\log x\) only to query
the correction interpolants. If only \(u=\log x\) is supplied, define the evaluation as being at the
computed \(e^u\) and test against that same argument. At very large \(x\), input-coordinate error can
exceed the residual interpolation error by many orders of magnitude.

## 8. API and consumer integration

### 8.1 Preserve the useful Bessel interface

Keep the constructor entry point and the useful dictionary members `phase`, `mod`, `bessel_j`,
`bessel_y`, `min_x`, `max_x`. The `phase` object need not remain a `phase_spline` instance — under
§7.2 it will not be one — but must implement the behaviour its consumers require, including raw/log
inputs and ordinary/log derivatives.

**`raw_theta` is genuinely compatibility-only, and this was checked.** The `AdaptiveLevin` docstring
at `levin_quadrature.py:2748` says `theta` is "always used to decide subdivision", which appears to
contradict revision 1. It does not: `phase_span` is computed from `theta_prime_Cheb`
(`levin_quadrature.py:1090`), and `need_theta_Cheb` is `False` whenever both `theta_mod_2pi` and
`theta_deriv` are supplied (`:1038`). So `raw_theta` is never evaluated by Levin **provided the
object supplies both accessors**. The docstring is stale, not the plan. The corollary is that the
derivative carries double duty — basis conditioning *and* subdivision — which is a further reason
§7.3's budget for it must be met rather than finessed.

**Declare `theta_abserr`.** `AdaptiveLevin` already accepts a caller-declared phase error
(`levin_quadrature.py:962`), and the comment at `:2360` says it exists for exactly this case: "where
the phase's own construction is the real limit (e.g. a fitted spline), a declared theta_abserr so the
caller sees an honest number instead of an artificially small one." Neither `test_bessel_phase.py`
nor `three_bessel_integrals.py` passes one today. Since this work changes achieved accuracy by eight
orders of magnitude, the new object should report its own achieved phase accuracy and its consumers
should pass it through.

Audit every caller before finalizing the adapter. Diagnostic consumers in
`plot_besssel_phase.py:22` and `ComputeTargets/QuadSourceIntegral_debug.py:55` read `Q`. The current
`Q` is the pre-offset ODE state, whereas the returned phase also incorporates `phi` and cycle
rebasing. Do not silently replace `Q` with a different diagnostic quantity under the same
undocumented meaning: either update those consumers to use the residual, or provide a clearly
documented compatibility quantity and a deprecation path. Treat `phi` similarly — and note that §4.3
means `phi` should be reported as identically zero for \(\nu>1/2\), not quietly dropped.

Existing `atol`/`rtol` arguments describe ODE tolerances and will no longer have a referent.
Introduce explicit absolute phase and relative amplitude accuracy settings. Update callers rather
than silently claiming the old arguments have identical semantics; deprecated arguments may be
accepted temporarily, but their translation and precedence must be documented. `main.py:429-437` is
the production caller to migrate.

`BesselPhaseProxy` (`ComputeTargets/QuadSourceIntegral.py:52`) transfers the object through Ray via
`ray.put`. Verify serialization of the new representation and its interpolants. No Bessel-specific
datastore schema change is indicated; the separate persisted cosmological phase problem should not be
folded into this patch by assumption.

### 8.2 Preserve structure in Bessel phase groups

`three_bessel_integrals.py:179-201` sums raw phases or bounded phases and sums their derivatives.
Bounded phases can preserve trigonometric values, but raw phase differences and derivative
cancellation still deserve explicit treatment.

For a shared variable \(t\), assemble

\[
\theta_\mu(kt)+\epsilon_\nu\theta_\nu(qt)+\epsilon_\sigma\theta_\sigma(st)=Kt+C+R(t),
\qquad K=k+\epsilon_\nu q+\epsilon_\sigma s .
\]

Combine the leading coefficients before multiplying by \(t\), and combine the residuals separately.
Form group derivatives from that same expression. This avoids subtracting independently reconstructed
large phases near resonance. Use appropriately accurate summation, and test exact and near
cancellation. Input coefficient uncertainty still limits what can be established near resonance;
compensated arithmetic does not make uncertain inputs exact.

### 8.3 Separate oracle improvement from transfer-function re-splining

An accurate Bessel object makes the constant-\(w\) oracle more useful. It does not repair a
downstream consumer that samples its full phase and then fits another coarse spline in
\(\log(1+z)\). Re-run the workstream B comparisons with two distinct questions:

1. Does the Bessel amplitude–phase object reproduce independently evaluated Bessel functions?
2. How much error does `TkSourceFunctions` introduce when consuming the sampled fixture?

Keep the exact Bessel fixture separate from the approximate physical LG fixture. Improving the Bessel
representation does not remove physical LG truncation error. Preserve and report the consumer
interpolation floor until a separate residual-phase or local-phase-evaluation design addresses it.

Note that the prompt 05 "exact" fixture defines its phase as \(\theta=\pi-\vartheta(x)\) with
\((m,\vartheta)\) from `bessel_phase`, so the fixture inherits this object's accuracy directly and
its sign convention must be re-checked against §1's zero-point.

## 9. Proposed implementation sequence

### Stage 1 — Establish independent regression measurements

- Add a reproducible diagnostic covering the metrics and parameter ranges in §4 and §6.
- Add exact half-integer references for \(\nu=1/2,3/2,5/2\), and **permanent `mpmath` references at
  the \((\nu,x)\) corners of §4.4** — including \(\nu\in\{20.5,100.5,1000.5\}\) and the largest
  arguments the supported domain admits. Record the SciPy version, because §4.4 is a property of the
  bundled Amos library and not a guarantee.
- Record construction cost, sample count, evaluation cost, phase-pair error, amplitude error,
  **derivative error**, and the location of each maximum.
- Distinguish sample nodes, interior test points, and endpoint intervals.

**Reason:** the present basic Bessel test allows 50% relative error
(`test_bessel_phase.py:34`) and the higher-order tests are set at \(10^{-3}\) to "catch garbage
rather than to measure precision" (`:113`). They cannot establish the intended improvement.
References built from the same Bessel phase object would conceal common error, and references built
only from `jv`/`yv` share the Amos library with `hankel1e`.

### Stage 2 — Implement the two-region construction

- Replace the ODE and the offset matching with normalized scaled-Hankel samples **in the near region
  only**, behind the boundary of §7.1.
- Implement the closed-form tail of §7.2 and the remainder-tested crossover. This is **required**,
  not deferred. One series (DLMF 10.18.18) supplies both \(r_\nu\) and \(a_\nu\).
- Implement explicit branch tracking and two-sided adaptive interpolation.
- Retain the current domain restrictions initially; state the supported \((\nu,x_{\max})\) domain
  explicitly and reject requests outside it, rather than extending behaviour implicitly.
- Add the \(a_\nu\) plausibility guard of §4.4 — not `isfinite` — plus a refinement cap that reports
  unmet accuracy instead of silently accepting it.
- Implement amplitude, residual and derivative accessors, with \(\theta'\) from \(e^{-2\ell}\) per
  §7.3.

**Reason:** this addresses state-error amplification, removes spurious phase matching, avoids
interpolating the growing leading phase, bounds the interpolated domain independently of
\(x_{\max}\), and keeps `hankel1e` three decades away from its failure boundary — in one coherent
change.

### Stage 3 — Implement accurate evaluation and compatibility

- Implement split sine/cosine evaluation and a bounded-phase accessor.
- Preserve raw arguments during logarithmic interpolation lookup.
- Provide the phase adapter existing consumers require, and declare `theta_abserr` (§8.1).
- Migrate diagnostic `Q`/`phi` usage and the production tolerance arguments.
- Verify serialization and scalar/log-input behaviour.

**Reason:** a better construction is ineffective if evaluation immediately rounds its correction
away, or reports an artificially small error to the quadrature.

### Stage 4 — Integrate Bessel phase-group consumers

- Preserve the analytic leading term when forming three-Bessel phases and derivatives.
- Test resonant and nearly resonant combinations, Bessel integrals, and amplitude/phase sign
  conventions.
- Keep quadrature error estimates distinct from input representation error; do not read agreement
  under quadrature refinement as a certificate of Bessel accuracy.
- Re-check `test_3bessel_analytic.py` and `test_three_bessel.py`: their tolerances were set against
  the old accuracy, and an eight-order improvement can expose a different limiting error rather than
  simply passing more easily.

**Reason:** consumers can otherwise reintroduce cancellation or large-phase errors after the
individual Bessel functions have been fixed.

### Stage 5 — Revalidate campaign fixtures and document remaining floors

- Re-run the \(w=1/3\) and \(w=0.2\) fixture comparisons using independent Bessel references.
- Quantify the remaining transfer-function re-spline error separately.
- Re-run the capped benchmark tier of `bessel_tier.py` at \(\kappa=1000\) and update the note at
  `:275` if the phase layer is no longer the scalability limit.
- Update the follow-up document with the measured replacement accuracy, the offset finding, the
  fixture/production tolerance distinction, and the chunking measurement of §4.6.
- Remove stale blanket statements that the Bessel oracle necessarily has an \(x\times10^{-8}\) floor,
  while retaining the historical measurements as historical evidence.

**Reason:** the objective is both a better representation and reliable downstream interpretation of
its accuracy.

## 10. Acceptance criteria and error budget

Engineering targets, not already-certified bounds:

| Coverage | Initial acceptance target |
|---|---|
| Low orders \(1/2,3/2,7/4,5/2\) — the orders production builds — existing domain through \(10^7\) | \(E_\theta,E_A\le10^{-11}\), including endpoint checks |
| Ordinary phase derivative, low orders | Relative error \(\le10^{-9}\) against an independent reference. §4.7: quintic reaches 1.9e-13 to 8.1e-13 at 250 per e-fold, so this has margin |
| Orders \(\ge20.5\), lower bound through \(\max(1000,10\nu)\) | \(E_\theta,E_A\le10^{-6}\) and phase derivative relative error \(\le10^{-6}\). **Accuracy is not the objective at these orders; correctness is** — see the row below |
| Orders \(\ge20.5\), structural requirements | Construction succeeds or fails loudly; every sample passes the \(a_\nu\) plausibility band; branch tracking is *verified*, not assumed, with an explicit test that a fixed-density `unwrap` would fail at \(\nu=1000.5\) (§4.5) and that the shipped tracker does not |
| Crossover to the closed-form tail | Near-region interpolant and series agree to the phase **and amplitude** budget at \(x_\star\); first omitted series term below budget; \(x_\star\) recorded |
| Supported domain boundary | Construction fails loudly outside the declared \((\nu,x_{\max})\) domain; every sample passes the \(a_\nu\) plausibility band; no `-inf` or `-0j` reaches an interpolant |
| Phase groups near derivative cancellation | Absolute derivative checks scaled to constituent frequencies, plus an independent group reference; no division by a vanishing group derivative |
| Selected large arguments through \(10^{15}\) | Independent split-evaluation checks at identical supplied arguments; no claim of full-domain coverage from spot checks |

**On the high-order target (user decision, 2026-09-08).** Revision 2 initially proposed
\(10^{-10}\) for orders 20.5, 100.5 and 1000.5. That was reduced to \(10^{-6}\) deliberately.
Production builds only \(\nu=1/2\) and \(5/2\) (`main.py:429-437`), the existing tests contract for
only \(10^{-3}\) at high order and say so explicitly — "to catch garbage rather than to measure
precision" (`test_bessel_phase.py:113`) — and §6.2 shows that reaching \(10^{-10}\) at
\(\nu=1000.5\) needs of order 4000 samples per e-fold near the turning point. Paying that for orders
nothing currently consumes would make the hardest part of the work also the least useful part.

The consequence is worth stating, because it simplifies the implementation substantially: at
\(10^{-6}\), §6.2's quintic at 250 samples per e-fold already delivers \(2.94\times10^{-9}\)
(\(E_\theta\)) and \(1.04\times10^{-8}\) (\(E_A\)) at \(\nu=100.5\), and §4.7's derivative
route delivers \(8.1\times10^{-8}\). So **at high order the binding requirement is correct branch
tracking, not sample density** — the \(\nu=1000.5\) failure in §4.5 is a wrap that `unwrap` cannot
detect, and no density fixes an incorrectly unwrapped sample. Order-dependent refinement is still
required near the turning point, but it is sized by the branch-safety criterion of §7.3 rather than
by an accuracy target.

Accuracy at high order remains available later: it needs only a denser near region, with no change
to the tail, the evaluation path or the consumers. §11 records it as deferred.

These targets leave margin above the observed low-order prototype errors and require substantial
improvement over the old representation. The implementation should expose requested accuracy and
report failure when it cannot meet it. If validation shows a reference or input-coordinate floor
dominates, document that floor before changing a target.

Allocate phase error among near-region sampling, branch tracking, interpolation, the series
remainder, derivative representation, and evaluation arithmetic. Amplitude relative error and phase
error both contribute to envelope-normalized Bessel error; controlling either alone is insufficient.
Adaptive midpoint comparisons are practical estimators, not mathematical supremum bounds. Use
multiple check points, refinement comparisons, adversarial tests and independent references before
describing the result as validated.

Required coverage includes:

- Bessel zeros and extrema, where pointwise relative errors are misleading.
- Construction endpoints, the turning-point neighbourhood, **and the tail crossover \(x_\star\)**.
- Changes of interpolation interval and phase branch.
- Raw and logarithmic input modes, with explicitly matched reference arguments.
- Consistency of \(e^{-2\ell}\), \(1+r_u/x\) and the reference \(\theta'\) (§7.3).
- Supported high orders already exercised by the existing tests.
- Three-Bessel values and integrals, including phase-group cancellation.
- Serialization and existing fixture consumers.

Benchmark performance rather than assuming it improves — but note §4.7: the existing construction
does not complete at \(x_{\max}=10^{13}\), so the relevant question is whether the replacement is
O(1) in \(x_{\max}\) as designed, not whether it is faster.

## 11. Explicit boundaries and deferred work

- Numeric/WKB hand-over overlap and numeric spline endpoint padding are separate work. This plan does
  not change them.
- General cosmological transfer-function and Green's-function stored phases need their own
  leading-term or local-integration design. The Bessel leading term \(x\) is special and cannot be
  assumed for a general background.
- **`phase_spline`'s chunking for the cosmological consumers** is out of scope. §4.6 measured it to
  be ineffective for the Bessel phase and structurally unable to bound the splined dynamic range, but
  the cosmological \(\theta(u)\) differs and must be measured on its own rows before any change.
- **The `_build_log_chunks_positive` progress guard** (§4.6) is a latent robustness bug in
  `phase_spline`, unrelated to this work, affecting any caller that passes a small `chunk_logstep`.
- Physical LG truncation error remains distinct from numerical Bessel representation error.
- **Tightening the high-order accuracy target** beyond the \(10^{-6}\) of §10 — for instance for
  non-Limber angular power spectra, the use case named in `test_bessel_phase.py:87-92` — is deferred.
  It requires only a denser near region and a cost budget for the node counts in §6.2; the tail, the
  evaluation path and the consumers are unaffected.
- Extending the Bessel domain below the current lower bound, or extending supported orders, requires
  explicit validation and is not implied by replacing the constructor. §4.4 in particular means the
  supported domain is now a statement about SciPy's behaviour and must be revalidated when SciPy
  changes.
- Retaining a residual ODE is unnecessary under the user's clarified requirement, and §5.2 shows the
  natural residual formulation is not computable in double precision in any case. §5.3 records the
  viable hybrid as the designated fallback for the near region only.

## 12. Reproduction

Run from the repository root using the repository environment, for example
`PYTHONPATH=. ./venv/bin/python <script.py>`. These are diagnostic prototypes, not production code:
the fixed-grid `unwrap` below is intentionally limited to the low-order comparison and must not be
copied as the high-order branch algorithm.

### 12.1 The central comparison (§3)

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
                theta = np.array([old["phase"].theta_mod_2pi(x) for x in xs])
            print("existing", nu, xmax, rtol,
                  pair_error(np.sin(theta), -np.cos(theta), J, Y, amplitude),
                  "phi", old["phi"])

        lo, hi = np.sqrt(nu * nu - 0.25), 1.02 * xmax
        count = round(250 * np.log(hi / lo) + 0.5)
        grid = np.linspace(np.log(lo), np.log(hi), count)
        gx = np.exp(grid)
        S = (np.sqrt(np.pi * gx / 2) * hankel1e(nu, gx)
             * np.exp(1j * (np.pi * nu / 2 + np.pi / 4)))
        rs = make_interp_spline(grid, np.unwrap(np.angle(S)), k=3)
        ls = make_interp_spline(grid, np.log(np.abs(S)), k=3)

        small = -np.pi * nu / 2 + np.pi / 4 + rs(np.log(xs))
        sine = np.sin(xs) * np.cos(small) + np.cos(xs) * np.sin(small)
        cosine = np.cos(xs) * np.cos(small) - np.sin(xs) * np.sin(small)
        ours_amplitude = np.sqrt(2 / (np.pi * xs)) * np.exp(ls(np.log(xs)))
        print("residual", nu, xmax,
              pair_error(sine, -cosine, J, Y, amplitude),
              "amplitude", np.max(np.abs(ours_amplitude / amplitude - 1)))
```

### 12.2 The `hankel1e` failure boundary (§4.4)

```python
import numpy as np
from scipy.special import hankel1e, jv, yv
np.seterr(all="ignore")

for nu in (0.5, 2.5, 20.5, 50.5, 100.5, 1000.5):
    lo = np.sqrt(max(nu * nu - 0.25, 1e-10))
    xs = np.geomspace(max(1.5 * lo, 1.0), 1e16, 6000)
    aH = np.abs(np.sqrt(np.pi * xs / 2) * hankel1e(nu, xs))
    aJ = np.hypot(jv(nu, xs), yv(nu, xs)) * np.sqrt(np.pi * xs / 2)

    def contiguous(ok):
        i = np.argmax(~ok) if (~ok).any() else len(ok)
        return xs[i - 1] if i > 0 else float("nan")

    print(f"nu={nu}: hankel1e to {contiguous(np.isfinite(aH) & (aH > 1e-3)):.4g}, "
          f"jv/yv to {contiguous(np.isfinite(aJ) & (aJ > 1e-3)):.4g}")

print(repr(hankel1e(100.5, 1e9)))          # -0j: finite, so isfinite() passes
print(np.angle(hankel1e(100.5, 1e9)),      # -0.0
      np.log(np.abs(hankel1e(100.5, 1e9))))  # -inf
```

### 12.3 The residual RHS cancellation (§5.2)

```python
import numpy as np
from mpmath import mp, mpf, besselj, bessely, pi as mpi
from scipy.special import jv, yv
mp.dps = 60

nu = 2.5
for x in (1e2, 1e4, 1e6, 1e8, 1e10, 1e12):
    naive = (2.0 / np.pi) / (jv(nu, x) ** 2 + yv(nu, x) ** 2) - x
    X, NU = mpf(repr(x)), mpf(repr(nu))
    a2 = (mpi * X / 2) * (besselj(NU, X) ** 2 + bessely(NU, X) ** 2)
    true = float(X * (1 / a2 - 1))
    print(f"x={x:.0e} true={true:.5e} naive={naive:.5e} rel={abs(naive/true-1):.2e}")
```

### 12.4 Chunking has no effect on accuracy (§4.6)

Build phase samples from the scaled Hankel function so the comparison contains no ODE error, then
feed identical `(div_2pi, mod_2pi)` arrays to `phase_spline` under different configurations. Compare
`num_chunks`, \(E_\theta\), and `max(abs(y))` over each chunk's `_y_points` against the peak raw
phase. Note that `chunk_step=200` (the `phase_spline` default) must not be used for a Bessel phase
spanning \(10^{12}\) rad: `_build_linear_chunks` would run \(\sim10^9\) iterations. `chunk_logstep`
values at or below about 1.9 do not terminate at all (§4.6).

### 12.5 Derivative accuracy and the two routes (§4.7)

Compare `1 + rs.derivative()(u)/x` and `np.exp(-2*ls(u))` against
`(2/np.pi)/(x*(jv(nu,x)**2 + yv(nu,x)**2))` at the log-midpoints of the sample grid, for
\(k=3,5\) and 250 and 1000 samples per e-fold. Report the location of the maximum; it should fall in
the interval adjacent to \(\sqrt{\nu^2-\tfrac14}\).

For independent direct-construction spot checks, use `mpmath` at 70 decimal digits and pass
`mp.mpf(float_x)` so the reference uses the same supplied argument.

## 13. Mathematical references

- [NIST DLMF §10.18 — Modulus and phase functions](https://dlmf.nist.gov/10.18): exact
  amplitude–phase identities, the Wronskian relation, and 10.18.18 (phase series). DLMF's phase
  convention differs from the repository's by exactly \(+\pi/2\). The modulus series 10.18.17 is
  not needed: §7.2 derives \(a_\nu\) from the phase series through the Wronskian.
- [NIST DLMF §10.17 — Large-argument asymptotic expansions](https://dlmf.nist.gov/10.17): the
  leading Hankel oscillation and fixed-order residual behaviour.
- [SciPy `hankel1e` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hankel1e.html):
  definition and implementation of the scaled Hankel function. See §4.4 for its measured domain of
  validity, which the documentation does not state.
- [Bremer, phase function methods for second-order ODEs with turning points](https://arxiv.org/abs/2209.14561):
  the reference cited by the existing constructor; retaining its ODE-based realization is not
  required, and §5.1 notes that the Bessel phase is a quadrature rather than an ODE.
