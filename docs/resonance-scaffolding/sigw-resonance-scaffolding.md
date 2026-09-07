# Adaptive scaffolding of the SIGW one-loop integral for a general time-dependent equation of state

**Status:** design proposal, not yet implemented.
**Purpose:** hand this to a coding session together with a target codebase, to work out how (and whether) it fits.

---

## 1. Context

Scalar-induced gravitational waves are computed from a one-loop momentum
integral in which the kernel weights the product of two input scalar power
spectra:

```
P_h(k) ∝ ∫ ds ∫ dd  W(s,d;k) · K(s,d;k) · P_ζ(uk) P_ζ(vk)
```

with the standard variables

```
u = |k - q| / k      v = q / k
s = u + v ∈ [1, ∞)   d = |u - v| ∈ [0, 1]
```

`K` is the (time-averaged, squared) kernel built from the tensor Green's
function and the scalar transfer function; `W` is the polarisation/projection
factor.

For a **constant** equation of state `w` with adiabatic sound speed
`c_s² = w`, `K` has a well-known singular feature on the line

```
s = 1 / c_s          (= √3 for radiation domination)
```

This is the Kohri–Terada resonance. Codes such as SIGWfast handle it by
splitting the `s` integration range at `1/√w` and bunching quadrature nodes
towards the split point from both sides with a fixed power law (see
`libraries/sdintegral.py`, functions `arrays_w`, `arrays_r`, `arrays_1`).

**The problem this document addresses:** when the background has an arbitrary
time-dependent equation of state `w(η)` and sound speed `c_s²(η)`, and the
Green's function and transfer function are obtained numerically rather than in
closed Bessel form, the location of the singular feature is no longer known a
priori — and it is no longer a single line.

---

## 2. Physics of the feature (what determines where to put nodes)

### 2.1 Origin

The tensor mode is a driven oscillator with natural frequency `≈ k`
(gravitational waves propagate at `c = 1`). The source is bilinear in the
scalar potential, whose sub-horizon behaviour is a sound wave oscillating at
`c_s q`. Writing the product of two such waves as sum- and difference-frequency
terms, the inner time integral has the form

```
∫ dη' A(η') exp[ i Ψ(η') ]        Ψ = θ_q + θ_r − θ_k^T
```

where `θ_q(η) = ∫^η ω_q` are the WKB phases of the numerically computed mode
functions, `ω_q ≈ c_s(η) q` sub-horizon and `ω_k^T ≈ k` for the tensor mode.

The stationary-phase condition `Ψ'(η_*) = 0` reads

```
ω_q(η_*) + ω_r(η_*) = ω_k^T(η_*)
   ⇔   c_s(η_*) (q + r) = k
   ⇔   s = 1 / c_s(η_*)
```

Physically: two phonons of momenta **q** and **k−q** can produce an on-shell
graviton of momentum **k**. Momentum conservation is automatic; energy
conservation requires `s = 1/c_s > 1`, which is a genuine threshold reachable
by non-collinear triangles.

The difference-frequency branch would need `c_s |u−v| = 1`, unreachable for
`c_s ≤ 1` since `d ≤ 1`. It can be ignored except in the degenerate `c_s → 1`
case, where both branch points migrate onto the domain boundary.

### 2.2 What changes when c_s varies with time

For constant `c_s`, *every* time is stationary on the line `s = 1/c_s` — that
degeneracy is exactly why the kernel diverges there.

For `c_s = c_s(η)`:

- **The line becomes a band.** Every `s` in the range of `1/c_s(η)` resonates,
  but only for a finite stretch of time around its own `η_*(s)`. Define

  ```
  R(η) ≡ 1 / c_s(η),        band = [ min R , max R ]
  ```

- **Inside the band the kernel is finite**, of order
  `A(η_*) √(2π / |Ψ''|)` with

  ```
  Ψ'' = k · s · c_s'(η_*)
  ```

  So the kernel is largest where `c_s` changes most *slowly*.

- **The true peaks are caustics**, at the band edges: values of `s = 1/c_s(η_e)`
  where `η_e` is either a stationary point of `c_s(η)` or an endpoint of the
  era / a sharp transition.

### 2.3 Widths — the key input to grid design

| Feature | Local normal form | Width in `s` | Scaling with `k` |
|---|---|---|---|
| Interior fold (`c_s' = 0`, `c_s'' ≠ 0`) | Airy | `Δs/s ~ (c_s''/k²)^(1/3)` | `k^(−2/3)` |
| Endpoint / sharp transition | Fresnel endpoint | `Δs ~ 1/(k Δη_feature)` | `k^(−1)` |
| `c_s` constant over `Δη` | degenerate | `Δs ~ 1/(k Δη) → 0` | logarithmic divergence |

The last row recovers the constant-`w` result: peak height `~ ln(kη)`, width
`~ 1/(kη)`.

**Consequence for implementation:** the grid must be rebuilt for each `k`.
A fixed `(s,d)` mesh cannot remain correct across decades in `k` once feature
widths scale as `k^(−2/3)` or `k^(−1)`.

---

## 3. Proposed algorithm

### Step 0 — Background and mode tables in phase–amplitude form

Solve the background once and tabulate on a common time grid:

```
a(η), H(η) ≡ a'/a, a''/a, w(η), c_s²(η)
```

Then, for each `q` on a logarithmic grid, solve the scalar mode equation and
store **amplitude and phase**, not the oscillating field:

```
φ(q, η) = A_q(η) cos θ_q(η)
```

Two reasons:

1. `A` and `θ` are smooth in `ln q`, so they can be interpolated to arbitrary
   `q`. Interpolating the oscillating `φ` directly is hopeless once the mode is
   deep inside the horizon.
2. The phases are exactly the objects the resonance scan in Step 1 needs.

Obtain `(A, θ)` either from a Prüfer/Riccati formulation of the mode equation,
or by evolving a complex solution matched to the WKB branch once sub-horizon
and taking modulus and argument.

Do the same for the two tensor homogeneous solutions per `k`, Wronskian-
normalised. Since the `χ = a·h` equation has no first-derivative term, the
Wronskian is exactly constant — use this as a free numerical check on the
tensor integrator.

**Caveat on the source term.** For time-dependent `w` the grouping

```
S ⊃ 2 Φ_q Φ_r + (4 / 3(1+w)) (H^{-1}Φ'_q + Φ_q)(H^{-1}Φ'_r + Φ_r)
```

still follows from the first-order momentum constraint, but the scalar
evolution equation no longer closes with `c_s² = w`. Carry `w(η)` and `c_s²(η)`
as **independent** inputs, and be explicit about whether non-adiabatic pressure
is included.

### Step 1 — Resonance scan (cost: negligible)

From the tables:

1. Tabulate `R(η) = 1/c_s(η)`. If horizon corrections matter, tabulate
   `R(η; q, r)` from the stored `ω`'s instead.
2. For each candidate `s`, solve `R(η) = s` by bracketing on the table
   → `η_*(s)`, possibly multi-valued.
3. Locate `dR/dη = 0` and the era endpoints → caustic positions `s_c`.
4. Evaluate `Ψ''` and `Ψ'''` at each stationary point → feature widths from the
   table in §2.3.

Output: the band `[s_min, s_max]`, the caustic list, and a width for each.

### Step 2 — Cheap surrogate kernel

Using the same tables, evaluate an approximate kernel `K̃(s,d)` by **uniform**
stationary phase — Chester–Friedman–Ursell (Airy-type uniform asymptotics), so
that it stays finite *at* the caustics rather than blowing up exactly where the
grid needs to be densest.

Cost is table lookups only, so `K̃` can be evaluated on a very fine mesh. It is
used solely for grid design, never for the final answer.

### Step 3 — Equidistribute the grid against a weighted monitor

Place `s` nodes by equidistributing an arc-length monitor function

```
M(s) = sqrt( 1 + [ d/ds ( 𝒲(s) · K̃(s,d) ) ]² )
𝒲(s) = P_ζ(uk) · P_ζ(vk) · W(s,d)
```

Including the actual weight matters. There is no point resolving a caustic that
sits where `P_ζ P_ζ ≈ 0`; conversely a modest feature inside the support of a
peaked `P_ζ` deserves nodes. Equidistribution reproduces SIGWfast-style bunching
where it is needed and coarsens elsewhere, with no hand-tuned bunching exponent.

**Cheaper variant** that captures most of the benefit: change variables from `s`
to the resonance time,

```
s = 1 / c_s(η_*)        ds = −(c_s' / c_s²) dη_*
```

and use a uniform `η_*` grid inside the band. The Jacobian vanishes exactly
where `c_s' → 0`, so nodes pile up at the caustics by construction. Union this
with a coarse logarithmic grid covering `s` outside the band.

### Step 4 — Inner time integral

With `Ψ` known:

- Use Filon- or Levin-type oscillatory quadrature away from stationary points.
- Integrate through `η_*` with a locally refined panel, or numerical steepest
  descent.

This avoids resolving every oscillation from `η_0` to `η`.

**Fallback:** solve the sourced ODE for `h_{qrk}` directly with an adaptive
integrator. More robust, considerably slower, and it does **not** remove the
need for Steps 1–3 — the outer `(s,d)` grid still has to resolve the feature.

### Step 5 — Error control

Refine adaptively on the residual: evaluate the true kernel on grids `h` and
`h/2` near each flagged feature and refine until the local contribution to
`Ω_GW(k)` stabilises.

Physics-informed seeding plus a posteriori refinement is safer than either
alone: pure adaptivity can step straight over a `k^(−2/3)`-wide caustic, and
pure seeding can miss features that come from structure in `P_ζ` rather than in
the background.

### Step 6 — Validation ladder

1. Constant `w = 1/3`: reproduce the Kohri–Terada `Ī²` including prefactor.
2. Constant `w ≠ 1/3`: reproduce the general-`w` `I_J²`, `I_Y²` expressions
   (SIGWfast `sdintegral.py` is a convenient reference implementation).
3. Slowly varying `c_s(η)`: check the constant-`w` logarithm is recovered as the
   variation is switched off, with peak width closing like `1/(kΔη)`.
4. Monochromatic `P_ζ` (delta function): kernel is probed at a single `(s,d)`,
   so grid errors show up unambiguously.
5. Convergence: `Ω_GW(k)` stable under doubling `n_s`, `n_d`, and the mode-table
   resolution independently.

### The `d` direction

`K` is smooth in `d` except at the `d → 1` edge (and at `d = 1/c_s`, unreachable
for `c_s ≤ 1`). Gauss–Legendre in `d` at each `s` node is sufficient. All the
adaptive machinery lives in `s`.

---

## 4. Integration questions for the target codebase

Work these out against the actual code before writing anything:

**Data flow**
- Where does the background come from — analytic `w(η)`, a tabulated thermal
  history, or a solved scalar-field model? Is `c_s²(η)` available independently
  of `w(η)`, or currently assumed equal?
- Are the scalar transfer functions already solved numerically, or still using
  the Bessel form? If numerical, are they stored as fields or as amplitude and
  phase?
- Is there an existing time grid shared between background and modes, or does
  each solver own its own?

**Integrator structure**
- Is the `(s,d)` grid currently built once and reused across all `k` (SIGWfast
  style), or per-`k`? Per-`k` rebuilds are required here — what does that cost
  in the existing loop structure?
- Is the loop integral vectorised over grid points (so a ragged, per-`k` grid
  breaks the array shapes), or looped?
- Where would a `resonance_scan` module have to sit so that both the grid
  builder and the kernel evaluator can see it?

**Interfaces to propose**

```python
BackgroundTables      # a, H, a''/a, w, c_s^2 on a common η grid
ModeTables            # A_q(η), θ_q(η) on (ln q, η); same for tensor
ResonanceMap          # band, caustics, widths, η_*(s), Ψ'' — per k
GridSpec              # s nodes + weights, d nodes + weights — per k
```

**Scope control**
- Steps 0–2 are self-contained and testable in isolation: they consume the
  background and produce a `ResonanceMap` plus a surrogate kernel. This is the
  natural first increment, and it can be validated against the known constant-`w`
  answer before any of the integrator is touched.
- Step 3 is a drop-in replacement for the existing grid builder.
- Step 4 is optional and independent; the existing inner integration can be
  kept initially.

---

## 5. Known pitfalls

- **Multi-valued `η_*(s)`.** If `c_s(η)` is non-monotonic, a given `s` has
  several stationary points, whose contributions interfere. Sum the complex
  amplitudes; do not sum `|·|²` separately.
- **Coalescing stationary points.** When two `η_*` merge, ordinary stationary
  phase fails — this is exactly why Step 2 specifies *uniform* asymptotics.
- **`c_s → 1`.** The resonance migrates onto the domain boundary `s = 1`, and
  the `d = 1` edge becomes singular too. Needs endpoint treatment, not interior
  bunching. Compare SIGWfast's separate `arrays_1` path.
- **`c_s → 0`.** The band runs off to large `s` (UV), and the `s` grid extent
  must track it rather than using a fixed `s_max`.
- **`w → 1/3` in any analytic cross-check.** Closed-form general-`w` kernels
  carry `Γ(1±b)` and `1/sin(πb)` factors whose `b → 0` poles cancel only in the
  limit; they are numerically singular exactly at radiation. SIGWfast handles
  this by nudging `w` off `1/3` by `1e-4` (`vv()` in `sdintegral.py`). Any
  validation harness needs the same guard.
- **Polarisations.** Check whether the codebase's `P_h` is per polarisation or
  summed; a factor 2 here is easy to lose.
- **Overall source normalisation.** The factor in front of the quadratic source
  is convention-sensitive (projector normalisation `e_ij e^ij = 1` vs `2`, and
  the sign/placement of the transverse-traceless projection). Pin it down
  against a known constant-`w` result rather than by inspection.

---

## 6. Suggested first session goal

Implement Steps 0–1 only, plus a plotting harness that overlays:

- `R(η) = 1/c_s(η)` and the derived band,
- the flagged caustics with their predicted widths at three values of `k`,
- the true kernel `K(s, d=const; k)` evaluated on a brute-force dense grid.

If the flagged positions and widths line up with the brute-force kernel, the
rest of the scheme is mechanical. If they do not, the phase extraction in Step 0
is the first place to look.
