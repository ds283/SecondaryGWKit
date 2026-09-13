# The QCD background: `T(z)` representation and the source sample grid

**Subject.** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` at commit `b3e3769`
(`gktk-remedial`) — `_solve_T_z` (`:144`), `_build_T_z_spline` (`:189`),
`integration_break_points` (`:256`) and the `rho`/`Hubble` path that consumes them (`:340-400`) —
together with the production source grid built by
`CosmologyConcepts/wavenumber.py:populate_z_sample` and `main.py:532`.

**Occasion.** `prompts/phase-representation` prompt 02 stopped (`BLOCKED`, log
`prompts/phase-representation/logs/02-primitive-phase-break-point-knots.md`) on the finding that a
repeated-knot vector for `PrimitivePhase` is singular on every production grid, because
`BREAK_POINT_ALL` returns 407 points across ~1,400 samples. This audit asks where those 407 points
come from. They are not features of the cosmology.

**Prior work this builds on.** `docs/gktk-remedial-verification.md` §3.5–§3.7 (the consumer
measurements this audit's defects sit underneath); `GkTk-remedial` log 02 (the $4.4\times10^{-4}$
jump in $H(z)$ at $z=4.24\times10^7$, measured but not attributed); the board entries
`[02-qcd-T-z-spline-node-tolerance]`, `[13-consumer-spline-crosses-eos-break-points]` and
`[00-consumer-anchoring-floor]`.

**Method.** Everything asserted below is a measurement, scored against a *tight* reference: the
defining equation $T\,g_s(T)^{1/3} = T_{\rm CMB}\,g_s(T_{\rm CMB})^{1/3}(1+z)$ root-solved to
`rtol=1e-14`. The shipped `_solve_T_z` is **not** a reference — it is one of the things measured.
Reproduction is §10, one script, 1.0 s, no Ray and no datastore. No repository files were modified.

---

## 0. Summary

### 0.1 Verdict

The QCD background's temperature is represented by a cubic interpolating spline over 500 points
uniform in $u=\log(1+z)$, whose node values are root-solved to `rtol=1e-4`. Both numbers appear to
have been chosen to get something running and never revisited. The representation is wrong in three
independent ways, and it sits underneath every QCD quantity in the project.

The consequence that matters is not an interpolation error. It is a **systematic error in the
background itself**, of order $3.5\times10^{-8}$ relative in conformal time, which is **common mode
between the producer and the consumer** and therefore invisible to every measurement
`docs/gktk-remedial-verification.md` takes. That campaign polished the consumer's *self-consistency*
to 1.00 ulp of the span while the background underneath both sides carried, at
$k=3\times10^8$/Mpc, of order $1.4\times10^5$ radians.

| | Finding | Severity |
|---|---|---|
| **T1** | The shipped `T(z)` gives a relative error of **3.461e-08** in $\int\mathrm{d}z/H$, i.e. of order **48 / 4.8e3 / 1.4e5 rad** of phase at $k=10^5/10^7/3\times10^8$, against 1-ulp floors of 3.05e-7 / 3.05e-5 / 9.15e-4 rad. Common mode, so **no existing test can see it**. | **critical** |
| **T2** | `_solve_T_z` root-solves to `rtol=1e-4`, so each spline node is independently wrong by up to **2.496e-05**, uncorrelated with its neighbours. This *is* `[02-qcd-T-z-spline-node-tolerance]`'s "scatter between neighbouring nodes", at its root. | **high** |
| **T3** | The spline interpolates $T$ against $u$, spending its resolution on the $(1+z)$ ramp that is known analytically, rather than on the entropy factor, which is the only part that needs approximating. Costs ~3 orders of typical accuracy at fixed node count. | medium |
| **T4** | The representation is a single global spline across points where $T(z)$ is genuinely **discontinuous**, so its maximum error is pinned at the jump height (**7.2e-04**) at *any* node count. | **high** |
| **G1** | 404 of the 407 `BREAK_POINT_ALL` points are knots of that auxiliary 500-point spline — an implementation artefact, not cosmology. They force a quadrature panel split every 4.04 grid intervals throughout `BackgroundModel`, and they are the sole cause of prompt 02's Schoenberg–Whitney failure. | **high** |

All five are ours and all five are fixable. §4 exhibits a representation, measured, that takes
`T(z)` from **7.18e-04 max / 1.89e-07 median** to **6.81e-11 max / 1.77e-16 median**, reproduces the
conformal-time integral bit-identically to the exact background at 17 digits, and is **cheaper per
call** than the one it replaces (2.21 µs against 2.44 µs).

### 0.2 What is not ours to fix

The equation of state is an upstream data fixture. It is **discontinuous at three of its four
declared branch joins** (§1). That is recorded here, and a sharp question is proposed for its
authors, but this audit does not repair it — and, importantly, **it does not need to be repaired
for §4 to land**: a segmented representation reproduces a discontinuous function exactly. The
discontinuity remains a genuine physical feature that the *sample grid* must resolve (§7), not an
obstacle to the representation.

---

## 1. The equation of state as handed to us

Entropy conservation gives $T\,g_s(T)^{1/3}\propto(1+z)$, so a step in $g_s$ forces a step in
$T(z)$ at fixed $z$, of size $\Delta T/T = -\tfrac13\,\Delta g_s/g_s$. Measured at
$T_{\rm break}(1\pm10^{-9})$:

| $T_{\rm break}$ [GeV] | $g$ below | $g$ above | $\Delta g/g$ | $g_s$ below | $g_s$ above | $\Delta g_s/g_s$ | forced $\Delta T/T$ |
|---|---|---|---|---|---|---|---|
| $10^{16}$ | 105.219908 | 106.750000 | +1.454e-02 | 105.281615 | 106.750000 | +1.395e-02 | **−4.649e-03** |
| 0.12 | 19.767510 | 19.763408 | −2.075e-04 | 19.100021 | 19.092871 | −3.744e-04 | **+1.248e-04** |
| 0.002 | 10.684024 | 10.684024 | +1.751e-11 | 10.685943 | 10.685943 | +1.755e-11 | −5.849e-12 |
| $10^{-5}$ | 3.380000 | 3.383000 | +8.876e-04 | 3.940000 | 3.931000 | −2.284e-03 | **+7.614e-04** |

**The 0.002 GeV join matches to round-off and the other three do not.** That asymmetry is the
evidence that the mismatches are a defect rather than the parametrisation's intent: if
discontinuous joins were designed in, none of the four would match to 1.8e-11.

**Proposed question for the equation of state's authors.** *Your branch join at $T=0.002$ GeV is
continuous in both $g$ and $g_s$ to 1.8e-11. The joins at $10^{16}$, 0.12 and $10^{-5}$ GeV jump by
1.4e-2, 3.7e-4 and 2.3e-3 in $g_s$. Are the latter intended, or are the branch coefficients or the
domain boundaries slightly off?* Until that is answered, §4 reproduces whatever the fixture says,
exactly.

---

## 2. `T(z)` is genuinely discontinuous, not merely kinked

$F(u) \equiv \log\!\big(T/[T_{\rm CMB}(1+z)]\big) = -\tfrac13\log\!\big(g_s(T)/g_s^{\rm CMB}\big)$,
across the lowest crossing $z_c = 4.25337\times10^7$:

| $z$ | $F(u)$ |
|---|---|
| 4.040700e+07 | 0.000000000000 |
| 4.210835e+07 | 0.000000000000 |
| 4.249115e+07 | 0.000000000000 |
| 4.257622e+07 | 0.000762292290 |
| 4.295902e+07 | 0.000762292290 |
| 4.466037e+07 | 0.000762292290 |

A step function, flat to twelve decimals on both sides. $g_s$ is piecewise constant in this
neighbourhood (3.940 → 3.931), so on each branch $T\propto(1+z)$ *exactly*, and $T(z)$ jumps by a
relative 7.614e-04 at $z_c$.

This closes a loop left open by `GkTk-remedial` log 02, which measured $H(z)$ jumping by
$4.4\times10^{-4}$ at $z=4.24\times10^7$ and recorded it without attribution. The chain is: EOS
branch join → step in $g_s$ → step in $T(z)$ → step in $H(z)$ → kink in $\varphi$ → the consumer's
worst QCD error at $z=4.24\times10^7$ in `docs/gktk-remedial-verification.md` §3.5, in both sectors.
One root cause, upstream of everything `prompts/phase-representation` has been fighting.

**Consequence for any representation.** No single smooth approximant can represent a step, at any
node count. §3's candidate table shows the maximum error pinned near the jump height whether 500,
2,000 or 5,000 nodes are used. A representation *segmented at the jump* reproduces it exactly.

The segment edges must be located by bisecting the monotone $T(z)$ against $T_{\rm break}$, **not**
by root-finding on $T(z)-T_{\rm break}$: that difference need not have a root at a discontinuity,
and a bracketing solver lands beside the jump. An edge misplaced by one node leaves the full error
in place — measured at 5.7e-04 on a first attempt that made exactly this mistake. This is the one
detail an implementation must get right.

---

## 3. The representation: three independent defects

Measured on 640 probe points placed on *and between* production grid nodes (a spline is worst
between its knots), $z\in[1,9.5\times10^{15}]$:

| | max | p90 | median |
|---|---|---|---|
| shipped `_solve_T_z` (its own `rtol=1e-4`) | 2.496e-05 | 1.246e-05 | 1.222e-08 |
| shipped `T(z)` spline (500 pts over those nodes) | **7.177e-04** | 1.323e-05 | 1.890e-07 |

**T2 — the nodes are sloppy.** `_solve_T_z` (`:179`) passes `xtol=1e-6, rtol=1e-4` to
`root_scalar`. The spline is built from 500 such values, each converged independently, so adjacent
nodes carry uncorrelated errors of up to 2.5e-05. A cubic through scattered nodes scatters, and its
derivative scatters worse. The cost of tightening is zero at runtime: the solve is build-time only,
at 8.3 µs per node.

**T3 — the wrong quantity is splined.** `T` against `u` is dominated by the $(1+z)$ ramp, which is
known in closed form; the spline spends its degrees of freedom re-deriving it. $F(u)$ is bounded,
$O(1)$, and *exactly constant* wherever $g_s$ is. At 500 nodes with accurate values, splining $F$
instead of $T$ improves the median by **400×** (1.071e-07 → 2.599e-10).

**T4 — no segmentation.** §2.

---

## 4. A representation that works

$T(z) = T_{\rm CMB}(1+z)\exp F(u)$, with $F$ splined per branch, edges on the jumps, nodes solved
to `rtol=1e-14`. All candidates built over the same interval the shipped spline covers, scored on
the same probe set:

| representation | max | p90 | median |
|---|---|---|---|
| shipped (500 pts, sloppy nodes, $T$ against $u$) | 7.177e-04 | 1.323e-05 | 1.890e-07 |
| plain $T$ spline, 500 pts, accurate nodes | 7.261e-04 | 1.936e-07 | 1.071e-07 |
| entropy factor, 500 pts | 7.236e-04 | 8.912e-08 | 2.599e-10 |
| entropy factor, 2000 pts | 5.066e-05 | 2.508e-10 | 1.476e-13 |
| entropy factor, 2000 pts, $k=5$ | 5.580e-05 | 9.002e-14 | 2.320e-16 |
| **segmented entropy factor, 3000 pts, $k=5$** | **6.807e-11** | **3.123e-15** | **1.773e-16** |

The three defects are independent and the table separates them: accurate nodes fix the p90, the
entropy factor fixes the median, segmentation fixes the max. All three are needed.

**Cost** (best of 3 over 200 calls): shipped spline 2.435 µs, entropy factor 2.193 µs, **segmented
entropy factor 2.210 µs**, accurate root solve 8.344 µs (build time only, ~25 ms for 3,000 nodes).
The recommended representation is slightly *cheaper* at runtime than the one it replaces.

---

## 5. Downstream: what it costs today

$\rho_r = \text{RadiationConstant}\cdot g(T)\cdot T^4$ and $H\propto\sqrt\rho$, so a relative error
$\varepsilon$ in $T$ reaches $H$ at roughly $2\varepsilon$ in the radiation era. On the production
grid:

| | max | p90 | median |
|---|---|---|---|
| $H(z)$, shipped | 1.278e-03 | 2.895e-05 | 3.424e-07 |
| $H(z)$, improved | 1.690e-10 | 6.314e-15 | 2.803e-16 |

Since $\theta = -[k\,\Delta\tau + \Delta\rho]$, a relative error in $\tau$ is a phase error
proportional to $k\tau$. Measured on $\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$, with the
integrator given the jump locations as break points:

| representation | rel. error in $\tau$ | $k=10^5$ | $k=10^7$ | $k=3\times10^8$ |
|---|---|---|---|---|
| shipped | **3.461e-08** | **4.75e+01 rad** | **4.75e+03 rad** | **1.43e+05 rad** |
| improved | 0.000e+00 † | 0.00 | 0.00 | 0.00 |
| *1 ulp of the span, for scale* | | 3.05e-07 | 3.05e-05 | 9.15e-04 |

† bit-identical to the exact-background integral at all 17 digits.

The figure is converged against the integrator: 3.4744e-08, 3.4602e-08, 3.4605e-08 and 3.4605e-08
at `epsrel` = 1e-9, 1e-10, 1e-11, 1e-12.

**Scope of this claim.** The 3.461e-08 is measured on $\int\mathrm{d}z/H$ over a representative
range, not on the exact $\Delta\tau$ limits each production object uses, and $\theta$ also carries
$\Delta\rho$. The phase columns are therefore "of order", not exact predictions for a given object.
The order is what matters: eight to ten decades above the floor the consumer path has been polished
to.

**What this does and does not mean.** For the *amplitude* of an observable, $3.5\times10^{-8}$ in
conformal time is negligible. It matters wherever the answer is phase-coherent — the one-loop
integral, where the Green's function's oscillation beats against the source. At
$k=3\times10^8$/Mpc the QCD oscillation phase is, as things stand, not meaningful.

---

## 6. Why no existing test catches it

Every measurement in `docs/gktk-remedial-verification.md` §3.5–§3.6 scores a **consumer against a
producer**. Both are built from the same `BackgroundModel`, hence the same $H$, hence the same
$\tau$. An error in the background cancels exactly in that comparison, so §3.5 can read 1.00 ulp
while §5 above is true.

This is the failure mode the `phase-representation` orchestrator names one level lower down — *"a
self-consistency test that passes both before and after proves nothing"* — applied to the
background rather than to the reduction. Closing it needs a test that scores the background against
an **independent** background, which is what §10's script does for the first time.

---

## 7. The source sample grid

`populate_z_sample` (`CosmologyConcepts/wavenumber.py:250`) returns
`logspace(log10(z_init), log10(z_end), num=...)`, with `z_init` from the wavenumber's horizon-exit
time and the density from a command-line number. **The cosmology is never consulted.** `main.py:532`
builds one universal grid from the earliest-exiting $k$; the response grid is a decimation of it.

Measured against that grid (1,732 samples, median spacing 2.3032e-02 in $u$):

| kind | count in range | median spacing | relative to grid |
|---|---|---|---|
| `BREAK_POINT_ALL` | 407 | 9.2936e-02 | 4.04 × |
| `BREAK_POINT_DISCONTINUITY` | 2 | 9.9194e+00 | 430.68 × |

**404 of the 407 are knots of the `T(z)` spline** — the uniform lattice of `linspace(..., 500)` in
`_build_T_z_spline`, intersected with the production range. That is finding **G1**, and it has two
costs beyond `T(z)` itself: `BackgroundModel` splits a Gauss–Legendre panel every 4.04 grid
intervals throughout, for an artefact; and `prompts/phase-representation` prompt 02 was blocked
entirely by it, since a multiplicity-3 knot at each of 407 points cannot satisfy
Schoenberg–Whitney against ~1,400 samples.

**A representation built as §4 has no knot lattice to declare**, because $F$ is splined per branch
and the branches are the physics. `BREAK_POINT_ALL` then collapses to the 3 genuine crossings, and
grid design becomes a three-point problem: a sample either side of each jump, tight enough that the
consumer's cubic resolves the step rather than interpolating across it.

Beyond that, the grid should carry the features it currently lands on only by luck — matter–radiation
equality ($z=3407$), matter–$\Lambda$ ($z=0.303$), each $k$'s horizon exit, and the numeric→WKB
hand-over — and its density should be set by a measured criterion on $\varphi$'s curvature rather
than by a uniform `samples_per_log10z`. Two mechanical consequences of making the grid
cosmology-dependent: `winnow` (`CosmologyConcepts/redshift.py:226`) is a blind stride `[::-n]`, so
a protected set would have to survive decimation if the response grid is ever required to resolve a
feature; and the grid tags at `main.py:566` are `SourceRedshiftGrid_{len}`, which labels size only,
so two different grids of equal length would collide in the datastore.

---

## 8. Prioritised recommendations

1. **Tighten `_solve_T_z`** to `rtol≈1e-14` (T2). One line, build-time cost only. Largest
   accuracy-per-line in the audit.
2. **Re-express the representation as a segmented entropy-factor spline** (T3, T4) — §4. The three
   sub-changes are separable and should be measured separately, because the table in §4 is the
   acceptance test: nodes fix the p90, the entropy factor the median, segmentation the max.
3. **Re-derive `integration_break_points`** once the knot lattice is gone (G1), and confirm
   `BREAK_POINT_ALL` falls to the genuine crossings. Expect a quadrature speed-up in
   `BackgroundModel` as a side effect; measure it.
4. **Add a background-against-background test** (§6). Without one, T1 can silently return. This is
   the only recommendation here that adds a permanent guard rather than a fix.
5. **Redesign the source grid** (§7) on the now-3-point break set.
6. **Re-run `prompts/phase-representation` prompt 02**, whose blocker is removed by 3 and whose
   premise is corrected by §2 — the kink is real, but it is a *step*, and it is 2–3 points.
7. **Ask the equation of state's authors** the §1 question. Independent of 1–6.

Items 1–3 are the campaign; 4 belongs with them; 5–6 are a second campaign that depends on 3.

## 9. What this audit did **not** examine

- The `LambdaCDM` path, which has no equation of state and is untouched by all of this. Every
  LambdaCDM figure in `docs/gktk-remedial-verification.md` stands.
- Whether the Saikawa & Shirai fits are correctly transcribed. §1 measures the joins only.
- `[00-consumer-anchoring-floor]`, which `prompts/phase-representation` prompt 02 established is
  what limits `theta_deriv` at $k\ge10^7$ (the recovered $\varphi$ spans 2–6 ulp of the stored
  phase there). **Nothing in this audit touches it**, and no amount of background accuracy will.
- The response grid and the $k\tau$ oscillation an $\Omega_{\rm GW}$ post-processing step would
  need to resolve in order to fit an RMS amplitude. Not yet designed; noted so it is not forgotten.
- Any re-measurement of the consumer tables of `docs/gktk-remedial-verification.md` §3.5 under a
  corrected background. That is work for the campaign, and it is the real acceptance test for T1.

## 10. Reproduction

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py
```

1.0 s, no Ray, no datastore. Sections 0–6 of its output correspond to §§3, 1, 2, 4, 5, 7 and 4
(cost) of this document. Taken on `b3e3769`.
