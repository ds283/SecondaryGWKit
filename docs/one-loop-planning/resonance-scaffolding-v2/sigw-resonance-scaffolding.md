# Adaptive scaffolding of the SIGW one-loop integral for a general time-dependent equation of state

**Status:** design proposal, not yet implemented.
**Purpose:** hand this to a coding session together with a target codebase, to work out how (and whether) it fits.

---

## 1. Context and notation

Scalar-induced gravitational waves are computed from a one-loop momentum
integral in which a kernel weights the product of two input scalar power
spectra. Writing the loop momentum as `q` and the external momentum as `k`,
and setting `s = k − q` (so `|s| = |k − q|`):

```
P_h(k) ∝ ∫ d³q/(2π)³  W · K(|q|, |s|; k) · P(|q|) P(|s|)
```

`K` is the (time-averaged, squared) kernel built from the tensor Green's
function and the scalar transfer functions; `W` is the polarisation/projection
factor.

### 1.1 Notation warning

The SIGW literature (Kohri–Terada, Domènech, SIGWfast) uses `s` and `d` for the
*sum and difference* variables, which collides with `s = k − q` above. This
document uses capitals for the sum/difference pair:

```
Σ = (|q| + |s|)/k  ∈ [1, ∞)        (literature "s"; = u + v)
Δ = (|q| − |s|)/k  ∈ [−1, 1]       (literature "d"; = u − v)
u = |s|/k          v = |q|/k
```

`K` is symmetric under `Δ → −Δ`, so integrate `Δ ∈ [0,1]` and double.

### 1.2 Measure

With `|s|² = k² + |q|² − 2k|q|μ`, the angular integral can be traded for `|s|`:

```
∫ d³q/(2π)³  →  (1 / 4π²k) ∫∫ |q| |s| d|q| d|s|
             =  (k³ / 32π²) ∫∫ (Σ² − Δ²) dΣ dΔ
```

The two forms are the two natural coordinate systems; §5 uses both, for
different parts of the domain.

### 1.3 The feature to be resolved

For a **constant** equation of state `w` with adiabatic sound speed
`c_s² = w`, `K` has a singular feature on the line

```
Σ = 1 / c_s          (= √3 for radiation domination)
```

This is the Kohri–Terada resonance. Codes such as SIGWfast handle it by
splitting the `Σ` range at `1/√w` and bunching quadrature nodes towards the
split point from both sides with a fixed power law (see `libraries/sdintegral.py`,
functions `arrays_w`, `arrays_r`, `arrays_1`).

**The problem this document addresses:** when the background has an arbitrary
time-dependent equation of state `w(z)` and sound speed `c_s²(z)`, and the
Green's function and transfer functions are obtained numerically rather than in
closed Bessel form, the location of the singular feature is no longer known a
priori — and it is no longer a single line.

Throughout, `z` denotes the time variable used by the numerical code
(conformal time, e-folds, or whatever the codebase uses).

---

## 2. Physics of the feature (what determines where to put nodes)

### 2.1 Origin

The tensor mode is a driven oscillator with natural frequency `≈ k`
(gravitational waves propagate at `c = 1`). The source is bilinear in the
scalar potential, whose sub-horizon behaviour is a sound wave oscillating at
`c_s |q|`. Writing the product of two such waves as sum- and
difference-frequency terms, the inner time integral has the form

```
∫ dz A(z) exp[ i Θ(z) ]        Θ = θ_q + θ_s − θ_k^T
```

where `θ_p(z) = ∫^z ω_p` are the WKB / Liouville–Green phases of the
numerically computed mode functions, `ω_p ≈ c_s(z) p` sub-horizon, and
`ω_k^T ≈ k` for the tensor mode.

The stationary-phase condition `Θ'(z_*) = 0` reads

```
ω_q(z_*) + ω_s(z_*) = ω_k^T(z_*)
   ⇔   c_s(z_*) (|q| + |s|) = k
   ⇔   Σ = 1 / c_s(z_*)
```

Physically: two phonons of momenta **q** and **k−q** can produce an on-shell
graviton of momentum **k**. Momentum conservation is automatic; energy
conservation requires `Σ = 1/c_s > 1`, which is a genuine threshold reachable
by non-collinear triangles.

The difference-frequency branch would need `c_s |Δ| = 1`, unreachable for
`c_s ≤ 1` since `|Δ| ≤ 1`. It can be ignored except in the degenerate `c_s → 1`
case, where both branch points migrate onto the domain boundary.

### 2.2 What changes when c_s varies with time

For constant `c_s`, *every* time is stationary on the line `Σ = 1/c_s` — that
degeneracy is exactly why the kernel diverges there.

For `c_s = c_s(z)`:

- **The line becomes a band.** Every `Σ` in the range of `1/c_s(z)` resonates,
  but only for a finite stretch of time around its own `z_*(Σ)`. Define

  ```
  R(z) ≡ 1 / c_s(z),        band = [ min R , max R ]
  ```

- **Inside the band the kernel is finite**, of order
  `A(z_*) √(2π / |Θ''|)` with

  ```
  Θ'' = k · Σ · c_s'(z_*)
  ```

  So the kernel is largest where `c_s` changes most *slowly*.

- **The true peaks are caustics**, at the band edges: values of `Σ = 1/c_s(z_e)`
  where `z_e` is either a stationary point of `c_s(z)` or an endpoint of the
  era / a sharp transition.

### 2.3 Widths — the key input to grid design

| Feature | Local normal form | Width in `Σ` | Scaling with `k` |
|---|---|---|---|
| Interior fold (`c_s' = 0`, `c_s'' ≠ 0`) | Airy | `ΔΣ/Σ ~ (c_s''/k²)^(1/3)` | `k^(−2/3)` |
| Endpoint / sharp transition | Fresnel endpoint | `ΔΣ ~ 1/(k Δz_feature)` | `k^(−1)` |
| `c_s` constant over `Δz` | degenerate | `ΔΣ ~ 1/(k Δz) → 0` | logarithmic divergence |

The last row recovers the constant-`w` result: peak height `~ ln(kz)`, width
`~ 1/(kz)`.

**Consequence for implementation:** the transverse grid must be rebuilt for each
`k`. A fixed `(Σ,Δ)` mesh cannot remain correct across decades in `k` once
feature widths scale as `k^(−2/3)` or `k^(−1)`.

### 2.4 Two independent sources of structure, in two different coordinates

This is the fact that drives the whole sampling design in §5:

| Enhancement | Locus | Behaviour under changing `k` |
|---|---|---|
| `P` peaked at `|q| = p_*` | `Σ + Δ = 2p_*/k` and mirror | fixed in **absolute** momentum |
| `K` resonance | `Σ = 1/c_s(z_*)` | fixed in **Σ**, moves in absolute momentum |

The two families of lines are not parallel and do not transform together. No
single tensor-product grid resolves both cheaply.

---

## 3. Assumed capabilities of the numerical code

This proposal is written for a code that already:

- integrates the tensor Green's function `G_k(z,z′)` **directly** outside the
  horizon (where a Liouville–Green representation is inefficient) and with a
  **WKB/LG** scheme inside;
- does the same for the scalar transfer functions `T_p(z)`;
- performs the inner time integral with a **Levin** method;
- can evaluate `K(|q|,|s|)` for arbitrary `|k|, |q|, |s|`.

Consequences used below: an LG phase–amplitude representation is available
**inside the horizon only**, which is exactly where the stationary-phase
analysis lives; and the Levin solver has a known failure mode at stationary
points (§5.5).

### 3.1 Which equations are which

Two distinct 1D problems are involved, and Step 0 needs both.

**Scalar (Bardeen potential) evolution equation** — the equation whose solution
is the transfer function `T_p`. In Newtonian gauge with no anisotropic stress:

```
Φ'' + 3H(1 + c_s²) Φ' + [ 2H' + (1 + 3c_s²) H² ] Φ + c_s² p² Φ = S_nad
```

For constant `w` with `c_s² = w` the mass-like bracket vanishes identically,
which is why the closed-form solution collapses to `J_{3/2+b}(c_s p z)`. Once
`w` varies that cancellation fails and the bracket must be carried numerically.
This is the underlying reason the resonance stops being a single line.

**Tensor equation** — the homogeneous solutions of

```
χ'' + (k² − a''/a) χ = 0,        χ = a·h
```

from which `G_k(z,z′)` is built. No first-derivative term, so the Wronskian is
exactly constant: use it as a free numerical check on the tensor integrator.

The resonance condition couples the phases of both: `θ_q + θ_s − θ_k^T`
stationary. The scalar phases supply `c_s|q|` and `c_s|s|`; the tensor phase
supplies `≈ k`.

**Caveat for time-dependent `w`.** The source grouping

```
S ⊃ 2 Φ_q Φ_s + (4 / 3(1+w)) (H⁻¹Φ'_q + Φ_q)(H⁻¹Φ'_s + Φ_s)
```

still follows from the first-order momentum constraint, but the scalar
evolution no longer closes with `c_s² = w`. Carry `w(z)` and `c_s²(z)` as
**independent** inputs, and be explicit about whether non-adiabatic pressure is
included.

---

## 4. Core architectural decision: decouple ODE solves from quadrature nodes

The expensive objects are all one-dimensional:

- `T_p(z)` — one solve per momentum `p`, reusable for **every** pair `(q,s)`
  and **every** `k`;
- `G_k(z,z′)` — one solve per external `k`, already on the output grid;
- `K(|q|,|s|)` — a Levin integral given the above; cheap by comparison.

So maintain a **static catalogue** of momenta for the ODE solves, and let the
quadrature nodes **float freely per `k`**, interpolating the catalogued mode
data. Cost becomes `O(N_p)` ODE solves plus `O(N_k · N_nodes)` Levin integrals,
instead of an ODE solve per node.

The catalogue is built once, before the `k` loop, spanning the support of `P`
plus whatever range the kinematics demand (see §5.6 on the tail).

### 4.1 Making the interpolation safe

Interpolating `T_p(z)` naively in `ln p` fails: inside the horizon the LG phase
is large and nearly linear in `p`, so a small *relative* interpolation error is
a large error in radians. Factor out the leading behaviour first. With the
sound horizon `X(z) = ∫^z c_s dz′`:

```
θ_p(z) = p · X(z) + δθ_p(z)
A_p(z) = p^(−α) · Â_p(z)
```

`δθ_p` and `ln Â_p` are smooth and `O(1)` in `ln p` and interpolate to spectral
accuracy on a modest catalogue. Reconstruct `θ_p` exactly from the analytic
part.

Outside the horizon there is no LG form, but there is also no oscillation:
`T_p(z)` itself is smooth and interpolates directly in `ln p`. Switch at the
horizon-crossing time already used by the solver.

**Calibration, not guesswork:** compare interpolated `T_p` against a fresh ODE
solve at a few interstitial momenta, and set the catalogue spacing from the
measured error.

---

## 5. Sampling strategy over `|q|` and `|s|`

### 5.1 Locating the resonance by sweeping time, not by root-finding

Do **not** root-find in `(|q|,|s|)`. Sweep `z` instead.

The LG representation supplies local frequencies `ω_p(z) = θ_p'(z)` inside the
horizon; the tensor solver supplies `ω_k^T(z)`. For each `z`, the set of
`(|q|,|s|)` with `Θ'(z) = 0` is a single curve — a straight line
`|q| + |s| = k/c_s(z)` in the deep sub-horizon limit, mildly curved once the
horizon corrections in `ω_p` are retained. Sweeping `z` sweeps a one-parameter
family of lines; the **band** is the region they cover and the **caustics** are
the **envelope** of the family. Both come out of an `O(N_z)` pass with no
root-finding.

Per `k`:

1. Restrict to `z` where **both** `q` and `s` are sub-horizon — outside that
   window there is no oscillation to be stationary.
2. Tabulate `Σ_res(z) = ω_k^T(z) / [ω_q(z) + ω_s(z)]` along a few `Δ = const`
   slices; the locus is only weakly `Δ`-dependent.
3. Band edges from the extrema of `Σ_res(z)` and from the endpoints of the
   sub-horizon window.
4. Width at each edge from `Θ''` and `Θ'''` (table in §2.3).

This also answers a question needed downstream: whether the band is **narrow**
(sharp feature, dedicated patch required) or **broad** (`c_s` varying fast,
feature smeared, ordinary adaptive quadrature suffices).

### 5.2 Split the domain with a partition of unity

Use a smooth partition rather than a hard cut, so each piece can use coordinates
aligned to its own features and no seam artefacts appear:

```
I = ∫ χ·(integrand)  +  ∫ (1−χ)·(integrand)
χ = χ( (Σ − Σ_res(Δ)) / w_res )          smooth bump, width a few × w_res
```

### 5.3 Band piece — coordinates `(Σ, Δ)`

- **Transverse (`Σ`):** Gauss–Legendre nodes over a few widths either side of
  `Σ_res(Δ)`. Alternative that self-tunes: change variable `Σ → z_*`
  (resonance time) and use a uniform `z_*` grid — the Jacobian
  `dΣ/dz_* ∝ c_s'` vanishes at the caustics, so nodes bunch there by
  construction.
- **Along the line (`Δ`):** `K` varies slowly; a coarse grid suffices.
- Typical size: `n_Σ ~ 20–40` transverse × `n_Δ ~ 16–32`.

### 5.4 Bulk piece — coordinates `(ln|q|, ln|s|)`

Here the structure comes from `P`, which is fixed in absolute momentum, so this
grid can be **shared across all `k`** up to the triangle cut
`| |q| − |s| | ≤ k ≤ |q| + |s|`.

Seed it directly from the known features of `P` — its support, peak, spectral
breaks — since `P` is an input and its features are known analytically. No
surrogate kernel is needed here. Refine `h`-adaptively on top of that seeding.

### 5.5 Levin fails exactly where it matters — guard it

The Levin collocation system is **singular when the phase derivative vanishes**,
i.e. precisely on the resonance. Without a guard, the inner integrator loses
accuracy in the one region the whole scheme exists to resolve, and does so
silently by returning plausible numbers.

Required guard:

1. Evaluate `Θ'` on each Levin panel.
2. If `Θ'` changes sign, or `|Θ'|` falls below a threshold set by the panel
   length, split the panel.
3. Handle the stationary panel separately: Filon with a locally quadratic phase
   model, numerical steepest descent, or a **uniform (Airy)** treatment if two
   stationary points are coalescing.
4. Regression-test against a constant-`w` case where the exact kernel is known.

Also watch for conditioning of the Levin system near the horizon-crossing
switch, where the LG phase is least accurate.

### 5.6 Boundaries and tail

- **Triangle edges** `Σ = 1` (flattened) and `|Δ| = 1` (squeezed): `K` has
  endpoint structure there, and the triangle cut slices through bulk cells.
  Give them dedicated graded nodes (`tanh`-type or polynomial grading into the
  edge) rather than expecting adaptive refinement to discover them.
- **`c_s → 1`:** the resonance migrates *onto* `Σ = 1`, so the band patch and
  the edge treatment must merge. Needs an explicit branch. Compare SIGWfast's
  separate `arrays_1` path.
- **`c_s → 0`:** the band runs off to large `Σ` (UV); the grid extent must track
  it rather than using a fixed `Σ_max`.
- **Tail:** for near-scale-invariant `P` the `Σ` integration is unbounded. Cut
  at `Σ_max` where `P(uk)P(vk)` times the kernel's asymptotic falloff drops
  below tolerance, and add an **analytic tail estimate** rather than extending
  the grid.

### 5.7 Order of operations

```
build catalogue of T_p(z) for p over the required range      # once
for each k in the output grid:
    solve G_k(z,z′)                                          # 1 ODE system
    sweep z → resonance locus Σ_res(Δ), band, caustics, widths
    build band nodes (Σ,Δ) and bulk nodes (ln|q|, ln|s|)
    for each node:
        interpolate (A_q, θ_q), (A_s, θ_s) from catalogue
        Levin integral with stationary-point guard → K
    accumulate with the measure and P(|q|) P(|s|)
    refine: h/2 on the band and on any bulk cell failing its error estimate
```

---

## 6. Optional refinement: surrogate kernel for monitor-driven grids

If the seeded grid of §5 proves insufficient, add a cheap surrogate `K̃(Σ,Δ)`
evaluated by **uniform** stationary phase (Chester–Friedman–Ursell), so that it
stays finite *at* the caustics rather than diverging exactly where the grid must
be densest. Cost is table lookups only, so `K̃` can be evaluated on a very fine
mesh.

Use it to place nodes by equidistributing a weighted arc-length monitor:

```
M(Σ) = sqrt( 1 + [ d/dΣ ( 𝒲(Σ) · K̃(Σ,Δ) ) ]² )
𝒲(Σ) = P(uk) · P(vk) · W(Σ,Δ)
```

Including the weight matters: there is no point resolving a caustic where
`P·P ≈ 0`, and a modest feature inside the support of a peaked `P` deserves
nodes. `K̃` is for grid design only and never enters the answer.

---

## 7. Error control and validation

### 7.1 Error control

Refine on the residual: evaluate the true kernel on grids `h` and `h/2` near
each flagged feature and refine until the local contribution to `Ω_GW(k)`
stabilises. Physics-informed seeding plus a posteriori refinement is safer than
either alone — pure adaptivity can step straight over a `k^(−2/3)`-wide caustic,
and pure seeding can miss features coming from structure in `P` rather than in
the background.

### 7.2 Validation ladder

1. **Constant `w = 1/3`:** reproduce the Kohri–Terada `Ī²` including prefactor.
2. **Constant `w ≠ 1/3`:** reproduce the general-`w` `I_J²`, `I_Y²`
   (SIGWfast `sdintegral.py` is a convenient reference implementation).
3. **Frozen-`c_s` limit of the full machinery:** with `c_s` constant, confirm
   the numerically detected band collapses to a line, the peak height grows like
   `ln(kz)` and the width closes like `1/(kz)`.
4. **Slowly varying `c_s(z)`:** check the constant-`w` logarithm is recovered as
   the variation is switched off.
5. **Monochromatic `P` (delta function):** the kernel is probed at a single
   `(Σ,Δ)`, so grid error is unambiguous and cleanly separable from kernel error.
6. **Interpolation check:** interpolated `T_p` vs. fresh ODE solve at
   interstitial momenta.
7. **Convergence:** `Ω_GW(k)` stable under independently doubling `n_Σ`, `n_Δ`,
   the bulk grid, and the catalogue resolution.

---

## 8. Integration questions for the target codebase

Work these out against the actual code before writing anything.

**Data flow**
- Where does the background come from — analytic `w(z)`, a tabulated thermal
  history, or a solved scalar-field model? Is `c_s²(z)` available independently
  of `w(z)`, or currently assumed equal?
- Are `T_p(z)` stored as fields, or as LG amplitude and phase? Is the phase
  available separately from the amplitude, and is `δθ_p` (phase minus `p·X`)
  recoverable?
- Is there a shared time grid between background, scalar and tensor solvers, or
  does each own its own?
- Does the code already cache `T_p` across calls, or re-solve per kernel
  evaluation? This is the single largest cost lever (§4).

**Integrator structure**
- Is the `(|q|,|s|)` grid currently built once and reused across all `k`, or
  per-`k`? Per-`k` rebuilds of the band are required here.
- Is the loop integral vectorised over grid points — in which case a ragged,
  per-`k` node set breaks the array shapes — or looped? **If vectorised, this
  refactor may dominate the work.**
- Where does the Levin driver live, and can it expose `Θ'` on each panel for the
  stationary-point guard?
- Where would a `resonance_scan` module sit so that both the grid builder and
  the kernel evaluator can see it?

**Interfaces to propose**

```python
BackgroundTables   # a, H, a''/a, w, c_s^2 on a common z grid
ModeCatalogue      # A_p(z), δθ_p(z) on (ln p, z); sub- and super-horizon branches
TensorSolution     # G_k(z,z'), omega_k^T(z)
ResonanceMap       # band, caustics, widths, z_*(Sigma), Theta'' — per k
GridSpec           # band nodes + weights, bulk nodes + weights — per k
```

**Scope control**
- §5.1 (the `z`-sweep resonance scan) is self-contained and testable in
  isolation against the known constant-`w` answer, before the integrator is
  touched. Natural first increment.
- §5.5 (the Levin guard) is independent of everything else and can be done in
  parallel; it is also the change most likely to alter existing results.
- §5.2–5.4 (the sampling scheme) replace the grid builder.
- §6 (surrogate kernel) is optional and should only be attempted if the seeded
  grid proves inadequate.

---

## 9. Known pitfalls

- **Levin at stationary points.** Singular collocation system exactly on the
  resonance; fails silently. See §5.5. Highest-priority guard.
- **Phase interpolation.** Interpolating `T_p` without factoring out `p·X(z)`
  gives radian-level phase errors from sub-percent amplitude errors.
- **Multi-valued `z_*(Σ)`.** If `c_s(z)` is non-monotonic, a given `Σ` has
  several stationary points whose contributions interfere. Sum the **complex
  amplitudes**, not the squared moduli.
- **Coalescing stationary points.** When two `z_*` merge, ordinary stationary
  phase fails — this is why §6 specifies *uniform* asymptotics.
- **`c_s → 1`** and **`c_s → 0`**: see §5.6.
- **`w → 1/3` in analytic cross-checks.** Closed-form general-`w` kernels carry
  `Γ(1±b)` and `1/sin(πb)` factors whose `b → 0` poles cancel only in the limit;
  they are numerically singular exactly at radiation. SIGWfast nudges `w` off
  `1/3` by `1e-4` (`vv()` in `sdintegral.py`). Any validation harness needs the
  same guard.
- **Polarisations.** Check whether the codebase's `P_h` is per polarisation or
  summed; a factor 2 is easy to lose.
- **Overall source normalisation.** Convention-sensitive (projector
  normalisation `e_ij e^ij = 1` vs `2`, and the sign/placement of the TT
  projection). Pin it down against a known constant-`w` result rather than by
  inspection.
- **Horizon-crossing switch.** The LG representation is least accurate near the
  switch, which is also where `ω_p` enters the resonance scan. Check the scan's
  sensitivity to the switch location.

---

## 10. Suggested first session goal

Implement §5.1 only — the `z`-sweep resonance scan — plus a plotting harness
that overlays, for three values of `k`:

- `Σ_res(z)` and the derived band,
- the flagged caustics with their predicted widths,
- the true kernel `K` on a brute-force dense `Σ` slice at fixed `Δ`.

If the flagged positions and widths line up with the brute-force kernel, the
rest of the scheme is mechanical. If they do not, the phase extraction (§4.1)
and the horizon-crossing switch are the first places to look.

A useful second, independent increment is the Levin stationary-point guard
(§5.5), since it can be validated on its own against a constant-`w` case.
