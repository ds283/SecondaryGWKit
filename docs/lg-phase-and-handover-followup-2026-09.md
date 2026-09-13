# Follow-up: the numeric→WKB hand-over and the accuracy of stored phases

**Status:** deferred. Not part of the `prompts/source-remediation/` campaign; to be picked up after
it lands.
**Origin:** review of prompt 05 (`TkSourceFunctions`, commit `e3348e4`) on 2026-09-08. The
prompt's relaxed test tolerances were accepted, but the measurements that justified them turned up
two problems that are wider than the prompt.
**Related:** `docs/spec-code-audit-2026-09.md` (A2, A7, TK-8); `docs/spec-code-audit/TK-report.md`
TK-8(e); `docs/spec-code-audit/QS-report.md` QS-5; `prompts/source-remediation/logs/05-tk-source-functions.md`
(deviation 7, §3 issue `[05-numeric-region-is-now-the-accuracy-floor]`).

All numbers below were measured on the prompt 05 test fixtures
(`ComputeTargets/tests/test_tk_source_functions.py`, class `Fixture`) with $w=1/3$ unless stated,
at the production grid density of 100 samples per decade of $1+z$. Reproduction snippets are in §5.
Statements marked *inference* have not been measured.

---

## 0. Summary

1. **The numeric→WKB hand-over sits in the worst interval of the numeric spline.** A cubic
   interpolating spline (`make_interp_spline`, not-a-knot ends) loses one to two orders of accuracy
   in its last two intervals. For $T_k$ the numeric grid ends at the hand-over, so the seam between
   the two representations is exactly where the numeric-region spline of $T$ and $dT/dz$ is least
   accurate (7.4e-6 and 2.9e-6 of the super-horizon value in the last two intervals, against
   ~5e-7 in the interior at the same node density). The same end effect appears at the low-$z$ end
   of the WKB-region phase spline. This is the same phenomenon prompt 03 fixed in
   `BackgroundModel._build_derivative` (audit A7) by padding the fit grid. The Green's function
   hand-over has an overlap and may be protected; that has not been checked (§1.3).

2. **A phase stored on a $\log(1+z)$ grid has an interpolation error that grows linearly with
   the number of cycles.** With $u=\log(1+z)$ and $x=kc_s a_0\eta$, $d\theta/du\simeq x$ in
   radiation, so every derivative of $\theta(u)$ is $O(x)$ and the cubic-spline error is
   $\simeq h^4x/384$. Measured: 6.9e-7 rad at $x=10^3$ and 6.2e-6 at $x=10^4$ (predicted 7.3e-7,
   7.3e-6); $\times81$ smaller at 300 per decade. Extrapolated to the shipped configuration
   ($x\sim1.4\times10^7$ at $z_{\rm end}$ for the largest $k$, QS-5) this is $\sim10^{-2}$ rad,
   i.e. $\sim1\%$ of the envelope in $\sin\theta$ (*inference*: verified only to $x=10^4$). The 05
   log's statement that the floor is set by the chunk range rather than the total cycle count is
   wrong: chunking protects floating-point precision, not interpolation error. This affects every
   consumer of a stored `theta_div_2pi`/`theta_mod_2pi` pair splined in $\log(1+z)$ — the
   `TkSourceFunctions` WKB region, `GkSourcePolicyData._create_functions`, and the phase handed to
   the Levin quadrature.

3. **`LiouvilleGreen/bessel_phase` is accurate to ~2e-6 of the envelope at $x=10^3$**, with the
   error entirely in the phase (modulus exact to 1e-14) and growing with $x$. Its own 250-per-e-fold
   spline predicts ~1e-9, so the spline is not the cause; the size is consistent with the ODE
   tolerance `rtol=1e-8` on $Q$ amplified by $\theta=xQ$ (*inference*). This matters because
   `bessel_phase` is the "exact" oracle for the constant-$w$ fixtures used by prompts 05, 07, 08
   and the audit scripts: those oracles have a floor of order $x\times10^{-8}$ in phase, and no
   test built on them can meaningfully assert better than that sub-horizon.

None of this is the Liouville–Green truncation error (the LG amplitude differs from the exact
Bessel envelope by ~1e-5 in $d\ln M/dz$ and by 3.7e-4–5.6e-3 in $T$ at 3 e-folds sub-horizon,
audit TK-8(e)). That is a property of the representation, is grid-independent, and is not
addressed here.

---

## 1. The hand-over

### 1.1 What was measured

Numeric region of the $w=1/3$ fixture: 370 nodes, from 5 e-folds super-horizon to the hand-over
at 3.5 e-folds sub-horizon ($x\approx19$, ~3 cycles, 14 nodes per cycle at the end). The spline
is `make_interp_spline` (cubic, not-a-knot) through exact samples of $T$ in $\log(1+z)$, evaluated
at the log-midpoints of every interval. Error relative to the super-horizon value $T=1$:

| interval, counted from the hand-over | $x$ at midpoint | $\lvert\delta T\rvert$ |
|---|---|---|
| 1 (last) | 18.90 | 7.4e-06 |
| 2 | 18.47 | 2.9e-06 |
| 3 | 18.05 | 7.7e-09 |
| 4 | 17.64 | 5.0e-07 |
| 5–8 | 16.1–17.2 | 3.6e-08 – 4.8e-07 |

The interior error at this $x$ is what a cubic spline at 14 nodes per cycle should give
(~5e-7 absolute, ~4e-5 of the *local* envelope $\approx3/x^2$); the last two intervals are 15× and
6× worse. The derivative spline shows the same shape: interior median 6e-12, last interval 1.6e-7.
Banded by $x$, error relative to the local envelope rises smoothly from 6e-10 ($x<1$) to 2e-5
($10<x<15$) and then jumps to 6e-4 in the last band, the jump being the end effect.

The WKB-region phase spline (`phase_spline`, `chunk_logstep=125`) does the same at its low-$z$
end: last two intervals 6.1e-6 and 2.8e-6 of the local envelope at $x=10^3$, third-from-last
1.3e-7, interior 1–2e-7.

### 1.2 Why it matters for $T_k$

README §2(a) of the campaign: the transfer function has *no* numeric/WKB overlap. `main.py`
integrates `TkNumericIntegration` with `mode="stop"` to a phase minimum and starts
`TkWKBIntegration` at exactly that point. `TkSourceFunctions` therefore splines the numeric
samples up to their last node and the WKB samples from their first node, and the region prompt 08
integrates across is partitioned at `crossover_z`. Both splines' worst intervals abut the seam.
The consequence is a systematic error in the source integrand concentrated in a two-grid-step
window around every $q$ and $r$ hand-over, and a discontinuity in the reconstructed $T$ across the
seam of the same order (not measured: the fixtures are exact on both sides, so the seam is
continuous by construction there).

### 1.3 The Green's function (*not yet checked*)

`GkSourcePolicyData._create_functions` (`GkSourcePolicyData.py:573-697`) splines the numeric $G$
over `(z_sample.max, numeric_smallest_z)` and the WKB $G$ over `(primary_WKB_largest_z,
z_sample.min)`, with `crossover_z` chosen *inside* the overlap. If the overlap is at least two
grid intervals on each side of `crossover_z`, the end intervals of both splines fall outside the
consumed range and the seam is protected. This should be measured on real rows (the overlap width
is a policy output, `GkSourcePolicyData`'s classification bands, and can be as small as the
`"WKB_minimal"` band allows). Two further differences from the $T_k$ case:

- the Green's-function WKB amplitude `sin_coeff*sqrt(H_ratio/sqrt(omega_WKB_sq))` **is** splined
  (`:644-652`), not assembled from closed forms as `TkSourceFunctions.M` is, so it has an end
  interval of its own;
- the numeric $G$ spline's low end is where the numeric ODE was stopped, i.e. deep sub-horizon
  where $G$ oscillates fastest, so its end-interval error is largest in absolute terms exactly
  where the WKB representation is supposed to take over.

### 1.4 Remediation options

- **Overlap by construction.** Let `TkNumericIntegration` continue two or more grid nodes past the
  `mode="stop"` point and store them, so `TkSourceFunctions` can spline past `crossover_z` and
  the consumed range ends two intervals short of the spline's end. This is the analogue of prompt
  03's padding. Cost: two extra ODE steps per $k$; a `main.py` and `TkNumericIntegration` change;
  existing rows unaffected in value but shorter than new ones.
- **Start the WKB grid above the hand-over** by the same margin, using the WKB solution's
  validity a few nodes above the stop point (it is already 3–6 e-folds sub-horizon there).
- **Use the overlap for a continuity check** (audit §4.1 asks for exactly this for $G$): with
  both representations available across two intervals, the mismatch of $T$ and $dT/dz$ at
  `crossover_z` becomes a stored diagnostic rather than an assumption.
- A different end condition on the spline (`bc_type`) helps only if the true end derivatives are
  known; `dT/dz` is stored, so `bc_type=((1, Tprime_end), ...)` at the hand-over end is available
  for the $T$ spline but not for the $dT/dz$ spline. Cheap, partial.

---

## 2. Stored-phase accuracy

### 2.1 The mechanism

A stored phase is consumed by splining $\theta(u)$, $u=\log(1+z)$, through the samples
$(\theta_{\rm div\,2\pi},\theta_{\rm mod\,2\pi})$ and evaluating $\sin\theta$. The spline never
needs to resolve a cycle — $\theta(u)$ is smooth and monotone — but in radiation
$\theta\simeq x\propto e^{-u}$, so $\theta^{(n)}(u)=O(x)$ for every $n$ and a cubic
interpolant's error is

$$\delta\theta\;\simeq\;\frac{h^4}{384}\,\theta''''\;\simeq\;\frac{h^4x}{384},\qquad
h=\frac{\ln 10}{\text{samples per decade}}.$$

At 100 per decade $h=0.023$: the phase advances $xh$ per node — 0.46 rad at $x=20$, 23 rad
($3.7$ cycles) at $x=10^3$, $3\times10^5$ rad at $x=1.4\times10^7$ — and the absolute error grows
with $x$ although the *relative* error of the spline is excellent ($10^{-7}$ rad on $10^3$ rad).
$\sin\theta$ responds to the absolute error.

### 2.2 Measurements

`TkSourceFunctions.T_WKB` against the fixture's own $M\sin\theta$ (so only the re-spline is
tested), interior of the top decade in $x$, excluding the last three intervals:

| grid | $x_{\max}$ | measured max | predicted $h^4x/384$ | last interval |
|---|---|---|---|---|
| 100/decade | $10^3$ | 6.9e-07 | 7.3e-07 | 6.1e-06 |
| 100/decade | $10^4$ | 6.2e-06 | 7.3e-06 | 6.0e-05 |
| 300/decade | $10^3$ | 3.0e-09 (max incl. ends) | 9.0e-09 | 3.0e-08 |

The amplitude assembly contributes 1e-14 throughout. Linear growth in $x$ and $h^4$ scaling are
both confirmed. The fixture's $k$ puts $x=10^5$ below $z=0$, so nothing beyond $10^4$ was measured.

### 2.3 Where it bites

- **`TkSourceFunctions` WKB region** (prompt 05): as above. Prompt 08's error budget should use
  $h^4x/384$ at the relevant $x$, not the 6e-6 the 05 log quotes for $x=10^3$.
- **`GkSourcePolicyData._create_functions` phase spline** (`:657-671`): identical construction,
  identical scaling, with the Green's-function phase $\theta_G$ in place of $\theta_T$. Every
  `QuadSourceIntegral` region that evaluates $G$ through this spline inherits it.
- **The Levin phase input.** The phase groups $\theta_G\pm\theta_q\pm\theta_r$ of prompts 07/08
  are sums of three such splines. Levin's accuracy depends on the phase *derivative* being right;
  prompt 05 deliberately supplies `omega` in closed form for this reason, and 07/08 should use the
  closed-form frequencies rather than `phase.theta_deriv` wherever a derivative is needed.
- **Not** the production phase *solve*. `WKB_phase_function.stage_2_evolution`
  (`Quadrature/integrators/WKB_phase_function.py:238-380`) integrates $Q$ with
  $\theta=\theta_{\rm init}+\omega_{\rm init}(1+u)Q$, exactly to avoid accumulating error in a
  large $\theta$; the accuracy is lost afterwards, when $\theta$ is stored as
  `(div_2pi, mod_2pi)` on the grid and re-splined by the consumer.

### 2.4 `bessel_phase` as an oracle

> **Superseded in part (2026-09-10, `prompts/transfer-remedial`).** Everything in this section
> describes the ODE-and-root-solve construction that existed when this document was written —
> integrate $Q$, solve for a phase offset `phi`, spline the full phase in chunks. That
> construction was replaced outright (commit `f6cbb29`, "Rebuild the Bessel phase from a
> two-region amplitude and residual", and the five commits after it) by a two-region
> amplitude/residual representation: a sampled near region below a remainder-tested crossover
> $x_\star$, and a closed-form asymptotic tail above it. The measurements below are **retained as
> the historical record of the construction they describe** — they were correct measurements of
> that construction — and are superseded by §2.4.1, which is the current state.

`LiouvilleGreen/bessel_phase.py` integrates $Q$ in $\ln x$ (DOP853, `atol=1e-10`, `rtol=1e-8`
from `config/defaults.py`), 250 samples per e-fold, then forms $\theta=xQ$, range-reduces, and
splines it. Measured against `scipy.special.jv`/`yv` at 20,000 points, $\nu=3/2$:

| $x$ | $\lvert m\sin\vartheta-J_\nu\rvert/m$ | $\lvert\delta m\rvert/m$ |
|---|---|---|
| 19–50 | 1.7e-08 | 8e-14 |
| 50–100 | 4.9e-07 | 5e-14 |
| 100–300 | 7.8e-07 | 4e-14 |
| 300–1000 | 2.0e-06 | 4e-14 |

The 250-per-e-fold spline predicts $\delta\theta\sim h^4x/384\approx7\times10^{-10}$ at
$x=10^3$, two thousand times smaller than measured, so the spline is not the limit. An `rtol` of
$10^{-8}$ on $Q\sim1$ gives $\delta\theta=x\,\delta Q\sim10^{-5}$ at $x=10^3$, the right order
(*inference*; tightening `rtol` and re-measuring would settle it in minutes). Consequences
(**historical — see §2.4.1; the blanket claim below is no longer true of the Bessel oracle
itself**):

- ~~every "exact" constant-$w$ fixture in this codebase (prompt 05 tests, audit scripts `TK_03`,
  `TK_04`, `GK_03`, `QI_02`, `QI_05`, and whatever prompts 07/08 build) has a phase floor of order
  $x\times10^{-8}$; a test asserting agreement with such an oracle to better than that sub-horizon
  is asserting agreement between two errors~~;
- ~~the 05 log's "vs scipy $J_\nu$" column (1.985e-06) is this floor, correctly identified there as
  `bessel_phase`'s and not `TkSourceFunctions`'s~~;
- ~~`bessel_phase` is also the analytic-branch oracle for `QuadSourceIntegral.analytic_integral`
  and `_three_bessel_Levin` (audit QI report), so the same floor sits under the analytic
  comparison prompt 08 will lean on~~.

#### 2.4.1 The replacement, measured (2026-09-10)

The `transfer-remedial` campaign's acceptance-table and attribution-table numbers are in
`docs/transfer-remedial-verification.md`; the four points this document's readers need are:

1. **Measured replacement accuracy.** The envelope-normalized phase-pair error $E_\theta$ at the
   orders and arguments production builds falls from 2.0e-06 (fixture tolerances, $x\le10^3$) and
   1.2e-08 (production tolerances, offset-dominated) to $\sim3\times10^{-14}$ — six to eight
   orders. The most striking figure is a correctness one, not an accuracy one: at $x=10^{12}$ and
   $10^{15}$, the old construction's $E_\theta$ was **1.509 and 1.512 — order unity** (`eps*theta`
   rounding of `sin(x+d)` at large $x$ destroys the reconstruction entirely), against 1.110e-16
   and 5.274e-16 for the split-evaluation replacement. `test_tk_source_functions`'s `err_scipy` —
   the one number in that fixture that isolates the Bessel oracle from the fixture's own re-spline
   — falls from 1.985e-06 to 3.021e-08 at $w=1/3$ (and 1.550e-06 to 2.234e-08 at $w=0.2$), landing
   exactly on the fixture's own re-spline error `err_T`.
2. **The offset finding.** `phi` was a pure artefact of a loose root solve (`xtol=1e-6,
   rtol=1e-4`) at a match point where the phase was already exact — for every $\nu>1/2$ the match
   point is the domain's initial node, where the phase is fixed exactly from the Bessel value
   itself — and it **was** the whole tight-tolerance error: $\phi=-4.836537\times10^{-8}$ and
   $E_\theta=4.873\times10^{-8}$ at $\nu=5/2$, agreeing to two significant figures. `phi` is now
   identically `0.0` at every order (there is no root solve at all).
3. **The fixture/production tolerance distinction.** The `bessel_phase` floor above was measured
   at the fixture tolerances (`config/defaults.py`'s `rtol=1e-8, atol=1e-10`). Production
   (`main.py`) used `rtol=5e-14, atol=1e-25` and the dominant error there was the constant `phi`
   offset, not the $x\times10^{-8}$ interpolation floor — the two regimes had different
   mechanisms. `main.py` now builds with `phase_atol=1e-12, amplitude_rtol=1e-12` (prompt 06),
   arguments with a direct meaning under the new construction rather than translated ODE
   tolerances.
4. **The chunking measurement.** `phase_spline`'s chunking (`chunk_logstep=125`) had **no**
   measurable effect on the Bessel phase's accuracy — identical to four significant figures at
   every $x_{\max}$ tested — and `chunk_logstep=125` cannot, by construction, bound the splined
   dynamic range; the shipped code also contradicted its own comment about what the constant did.
   This is a statement about the *Bessel* phase specifically. The **cosmological** $\theta(u)$ this
   document's §1–§3 concern is a different quantity and has since been measured on its own rows,
   more harshly, by `docs/gk-wkb-review-astra-pathfinder-2026-09-08.md` §3 (commit `39ed7fc`): "no
   demonstrated numerical advantage at the tested scales and has demonstrated disadvantages" — a
   hard chunk-switch discontinuity of 1.08e-4 rad phase and 3.51e-8 relative derivative (which a
   Levin consumer sees), a rebase that can *enlarge* the stored ordinates (6.44e8 rad from a
   9.99e6 rad span), knot-level accuracy degrading from 2.88e-9 to 2.28e-7, and inverted interval
   keys from decreasing-phase merges. `phase_spline` itself was not touched by either finding;
   both are arguments against its chunking, not fixes to it.

The `bessel_phase` oracle's blanket $x\times10^{-8}$ floor is therefore gone; what stood under it
in the two `ComputeTargets` fixture modules (`test_tk_source_functions.py`,
`test_phase_groups.py`) is now, wherever it is not the oracle, the consumer's own $h^4$ re-spline
of the sampled fixture, or physical Liouville–Green truncation — separately measured and named in
`docs/transfer-remedial-verification.md`, and **not** removed by this replacement, since neither
is a Bessel-oracle defect.

### 2.5 Remediation options

- **Store something with small derivatives.** The production solve already has $Q$; storing $Q$
  (or $\theta-\theta_{\rm LG,0}$ for an analytic leading term $\theta_{\rm LG,0}(u)$) and
  reconstructing $\theta$ at evaluation time turns the splined quantity into one whose $u$
  derivatives are $O(1)$, removing the factor $x$. This is a schema change on
  `TkWKBIntegration`/`GkWKBIntegration` values (or an additional column) and a consumer change in
  `phase_spline`/`TkSourceFunctions`/`GkSourcePolicyData`.
  **(2026-09-10: the supporting example no longer holds.)** `bessel_phase` no longer keeps a `Q`
  member at all — prompt 06 of `prompts/transfer-remedial` removed it, because there is no ODE, no
  offset and no state left for it to name (`data["Q"]` now raises `KeyError`). More importantly,
  `DRAFT-PLAN.md` §4.2 (the design document behind that campaign) gives the reason this was never
  the right general answer in the first place: evaluating a $Q=\theta/x$ spline directly still
  multiplies its own state and interpolation error by $x$ at reconstruction, so it does not remove
  the mechanism this section is about — it only relocates it. This bullet's general
  recommendation (store a quantity whose derivatives stay $O(1)$, e.g. the residual against an
  analytic leading term, as the Bessel replacement now does for the Bessel leading term $x$) is
  unaffected; only the `bessel_phase`-`Q` illustration is retired.
- **Evaluate $\theta$ by local integration of the closed-form $\omega_{\rm eff}$** from the
  nearest stored node: $\theta(u)=\theta_i+\int_{u_i}^{u}\omega\,(1+z)\,du'$ with a fixed-order
  Gauss rule. Removes the spline entirely at the cost of a few $\omega_{\rm eff}$ evaluations per
  call; needs $\omega_{\rm eff}$ to be cheap (it is closed-form for $T_k$; for $G_k$ it is
  `Gk_omegaEff_sq`, also closed-form).
- **Denser grid**: $h^4$ — 1000 per decade buys $10^4$, which is not enough at $x=10^7$ and
  costs every stage of the pipeline.
- ~~For `bessel_phase`: tighten `rtol`/`atol` (cheap; check first that this is the limit), and
  state the oracle floor in every test that uses it.~~ **Done, by replacement rather than by
  tightening** — see §2.4.1. Tightening `rtol`/`atol` on the old ODE would not have removed the
  `phi` offset or the growing full-phase interpolation error, which were structural to that
  construction, not artefacts of its tolerances.

These last three bullets are about the **cosmological** transfer-function and Green's-function
stored phases (this document's §1–§3), which remain out of scope for `prompts/transfer-remedial`
(its README §1.1) and are unaffected by the Bessel oracle's replacement.

---

## 3. What should be measured before remediating

1. **Green's-function overlap width** on real `GkSourcePolicyData` rows: is `crossover_z` at least
   two grid intervals inside both `numeric_region` and `WKB_region`? If not, §1 applies to $G$
   as well and with larger absolute errors.
2. **Continuity across the $T_k$ seam on real rows**: $T_{\rm numeric}$ vs $M\sin\theta$ and
   $dT/dz$ vs $M(d\ln M/dz\,\sin\theta+\omega\cos\theta)$ at `crossover_z` (prompt 12 partly covers
   this; audit §4.1 is the $G$ analogue).
3. **Phase error at production $x$**: repeat §2.2 with a fixture whose $k$ keeps $x=10^5$–$10^7$
   above $z_{\rm end}$, to confirm the linear extrapolation rather than rely on it.
4. **`bessel_phase` with `rtol=1e-12`**: does the 2e-6 fall? If so, §2.4's attribution is right
   and the fix is a constant.
5. **Cost of storing $Q$**: whether `phase_spline` consumers can be switched without touching the
   Levin gate (`GkSourcePolicyData._classify_Levin` builds a single-chunk θ spline for a
   derivative test, audit B9).

---

## 4. Relation to the current campaign

Nothing in prompts 06–12 should be blocked by this. Prompt 08 should (a) partition on
`crossover_z` but not assume the integrand is smooth to better than ~1e-5 within two grid steps
of it, (b) budget the LG-branch phase error as $h^4x/384$ at the local $x$, and (c) treat the
`bessel_phase`-based analytic oracle as accurate to $\sim x\times10^{-8}$ in phase. Prompt 12's
tolerances should be set with the same three facts in mind. The 05 log's §3 issue
`[05-numeric-region-is-now-the-accuracy-floor]` remains correct in its conclusion (the numeric
region is the floor) but attributes the floor to grid density; the two-interval end effect is the
larger part of the quoted number.

---

## 5. Reproduction

All from the repository root with `PYTHONPATH=.` and the venv Python; each runs in seconds.

Numeric-region end effect (§1.1):

```python
import numpy as np
from ComputeTargets.tests.test_tk_source_functions import Fixture, log_midpoints
f = Fixture(1/3); fn = f.exact_functions()
mids = log_midpoints(f.z_numeric)
err = [abs(fn.T(z) - f.T_exact(z)) for z in mids]
print([f"{e:.1e}" for e in err[-8:]])      # last 8 intervals, hand-over end last
```

Phase-spline scaling (§2.2):

```python
from math import sin, log
h = log(10)/100
for xmax in (1e3, 1e4):
    f = Fixture(1/3, x_max=xmax); fn = f.exact_functions()
    mids = np.array(log_midpoints(f.z_WKB)); xm = np.array([f.x(z) for z in mids])
    err = np.array([abs(fn.T_WKB(z) - f.M_exact(z)*sin(f.theta_exact(z)))/abs(f.M_exact(z)) for z in mids])
    top = err[xm > 0.3*xmax][:-3]
    print(xmax, top.max(), h**4*xmax/384, err[-1])
```

`bessel_phase` floor (§2.4):

```python
from scipy.special import jv, yv
from LiouvilleGreen.bessel_phase import bessel_phase
bp = bessel_phase(1.5, 1.02e3)
xs = np.logspace(np.log10(19.2), 3, 20000)
J, Y = jv(1.5, xs), yv(1.5, xs); m = np.sqrt(J*J + Y*Y)
T = np.array([bp["mod"](x) for x in xs]) * np.sin([bp["phase"].raw_theta(x) for x in xs])
for lo, hi in [(19,50),(50,100),(100,300),(300,1000)]:
    s = (xs>=lo)&(xs<hi); print(lo, hi, np.max(np.abs(T[s]-J[s])/m[s]))
```
