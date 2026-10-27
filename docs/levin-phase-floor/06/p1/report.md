# levin-phase-floor 06, finding P1 — which of R13's and R14's totals is right, and why

> **Written by a laptop subagent** on 2026-10-27, which ran every command on the Mac Studio over
> SSH. Its tool environment refused to write this file ("Subagents should return findings as
> text"). It did not get round that over SSH, and returned the report as text. **The laptop
> orchestrator saved that text here unchanged** in 06's records commit, with three additions taken
> from the laptop's brief to the subagent: this note, the table identifying the two rows under the
> verdict, and the parenthesis defining H1 and H2 at the end of §1. Its `scripts/` and `out/` are
> beside it. The `.npz` files stay on the Mac Studio
> ([`../KEPT_ON_MAC_STUDIO.md`](../KEPT_ON_MAC_STUDIO.md)), as the key files at the end say.

VERDICT: R13 nearer. Reference G+q+r values on z ∈ [250, 5.42099e9], scaled by (1 + z_response) as
stored: row 1 −1.3579758964e-20 (R13's integrand) and −1.3579759064e-20 (R14's); row 2
+7.8664300997e-21 and +7.8664300993e-21. The totals with that group corrected: row 1
5.918585470e-20 (R13) and 5.918585406e-20 (R14); row 2 −3.3961904781e-19 (R13) and
−3.3961904975e-19 (R14). R13's stored totals are within 0.01× their declared total_abserr of
these. R14's are 1.28e4× (row 1) and 555× (row 2) their declared total_abserr away.

The rows are:

| row (k, q, r) /Mpc | R13 serial (shard) | R14 serial (shard) | total R13 → R14 |
|---|---|---|---|
| 1: (3.162e5, 3.162e5, 3.162e5), z_response 250 | 10096 (0) | 10193 (0) | 5.918585e-20 → 5.927866e-20 |
| 2: (1e5, 3.162e5, 3.162e5), z_response 250 | 70 (3) | 270 (3) | −3.396190e-19 → −3.396284e-19 |

## 1. In one paragraph

Both rows' G+q+r phase, θ = θ_G + θ_q + θ_r, has exactly one stationary point inside the
sub-interval: at z = 10211.3 for row 1, where c_s = 1/2, and at z = 275.0 for row 2, where
2qc_s = k. The leading stationary-phase contribution of that point is −3.697e-25 for row 1 and
+3.739e-26 for row 2 (unscaled). Under R14, `adaptive_levin_sincos` accepts the whole sub-interval
as one Levin region. That region's estimate lacks exactly this contribution: the estimate minus the
reference is +3.6975e-25 and −3.7392e-26, which matches the stationary-phase estimate to 1e-5 and
5e-5. The step-(4) residual cannot see the miss, because the parent region and the half that holds
the stationary point both lack the same contribution. R13's root estimate lacks it too, and is the
same to 3.5e-8. R13 was bisected only because its noisier phase pushed the root's residual over
rtol, by factors of 2.05 and 1.008. The cause is noise in the high-z half, which has nothing to do
with the stationary point. Once bisected, the halves holding the stationary point fail their own
tests, and the driver resolves it. **H1 holds, strongly; H2 is refuted.** R13 got the right answer
by luck, and both trees share the defect, which lies in AdaptiveLevin.

(H1: `AdaptiveLevin` falsely accepts a single region under R14's noise-free phase, where R13's
noisy φ forced bisection. H2: R14's integrand, its phase or amplitude, really differs from R13's
by more than round-off.)

## 2. Harness, and every change to it

The scripts were copied from 06's `wider/scripts/` (originals' SHA-256 in
`out/originals_sha256.txt`, re-checked unchanged at the end). Only `harness.py`, `rebuild.py` and
`prov.py` were copied. `rerun.py` and `worker.py` were not needed; `worker.py`'s `tree_evidence`
was cut down into `p1lib.py`.

- **`harness.py`** is unchanged apart from one docstring paragraph.
- **`rebuild.py`**, each change marked "p1" in the file:
  1. `A3V2_*` became `SCOPE_*` with the rehearsal's values from `build_scope` and
     `_main_py_argv`. The k sample is `np.logspace(np.log10(1e5), np.log10(1e6), 3)` and `--zend`
     is `float("250")`. Unchanged: atol 1e-32, rtol 1e-7, source samples 100, response sparseness
     12, run label `default`, LambdaCDM only.
  2. `PRIMARY`/`shard_file` became `use_store(stem)`, which points at `levin-phase-floor-06-r13*`
     for 9f66622 and `-r14*` for f5cee4c. There is no default store.
  3. The row's QuadSource is the stored `QuadSource_obj`, not 05's recomputation. Each store was
     built by the tree reading it, so main.py's stage used the stored read-back.
     `recompute_QuadSource` still runs as a check, and found 0 differing columns at all 428 nodes
     in all four rebuilds.
  4. The re-run path raises an error if called; it is never used.
- **`prov.py`** names the trees r13 (9f66622) and r14 (f5cee4c) and each row's (shard, serial).
- **New scripts:**
  - `p1lib.py`: tree evidence, the rebuild, transparent capture wrappers on
    `QSI_MOD.build_phase_groups` and `adaptive_levin_sincos`, and the validation.
  - `validate.py`, `analyse.py` (steps 3a–c and 5), `reference.py` (3d), `rootdiag.py` (the root
    decision), `samples.py` and `compare.py` (step 4), `summary.py`.
  - `run.sh`, the recorder.
- **How every run was made:** `PYTHONPATH=<worktree>`, the main venv's python by absolute path,
  `-B`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, cwd `p1/out`. Each run checked its tree
  evidence first: every production module comes from that worktree, HEAD is the expected SHA,
  status is empty, `-B` is in force and Ray is not initialised.
- **No Ray.** Ray was never started, and no lock was needed.

Three runs failed on bugs in my own scripts and were re-run after the fix. Their logs were
overwritten, and none touched a store:
- `validate`, first try: a `TypeError` in my comparator on a list field.
- `analyse`, first try: brentq refused `rtol=4e-16`.
- `samples` ran twice in its first form, before knot recording was added.

**Read-only.** All ten R13/R14 files have the same SHA-256 before and after
(`out/store_sha256_before.txt`, `out/store_sha256_after.txt`). Both worktrees were clean
afterwards, ignored files included. Nothing else under `var/` was opened.

## 3. Validation (step 2): bit for bit, in both trees

Each tree rebuilt both rows from its own store. For each row, all 54 compared items were identical
by `float.hex`:
- `total`, `total_abserr`, `total_converged`, `numeric_quad`, `WKB_Levin`;
- for both sub-intervals: z_max, z_min, value, abserr, converged and regime;
- for every group: value, abserr, converged, regions, simple_regions and evaluations.

All 22 rebuild checks held in each case. Re-calling `adaptive_levin_sincos` with the captured
arguments reproduces each captured value and abserr bit for bit. The tables are in
`out/validate_r1{3,4}.json`.

| row | store (shard/serial) | total | total_abserr |
|---|---|---|---|
| 1 | R13 0/10096 | 5.918585465520117e-20 | 5.105e-27 |
| 1 | R14 0/10193 | 5.9278660352359123e-20 | 7.270e-27 |
| 2 | R13 3/70 | −3.3961904798375587e-19 | 1.938e-26 |
| 2 | R14 3/270 | −3.3962843526143793e-19 | 1.692e-26 |

The low sub-interval is x = log(1+z) ∈ [5.5254529391317835, 22.413544878488572]. Each group is
called with atol 1.617e-33, rtol 1e-7, chebyshev_order 24 and `theta_deriv` supplied.

## 4. Stationary points (step 3a)

θ′ is `group.theta_deriv`, the closed form handed to the driver. It was sampled at 40,001 points
and every sign change refined by brentq. Both trees find identical points.

| row | group | total variation of θ (rad) | sign changes of θ′ | z_s (x_s) | θ″ (per unit x²) | rates there: θ_G′, θ_q′ = θ_r′ | envelope | √(2π/θ″) | stationary-phase estimate (unscaled) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | G+q+r | 1.11295e8 | **1** | 10211.3179 (9.2313499) | 1.5537e6 | −1.24293e7, +6.21465e6 (so c_s = 1/2) | 1.842e-22 | 2.01e-3 | −3.6974e-25 |
| 2 | G+q+r | 5.66757e7 | **1** | 275.00859 (5.6204320) | 2.1270e7 | −4.59884e7, +2.29942e7 (so 2qc_s = k) | 7.104e-23 | 5.44e-4 | +3.7394e-26 |

The control groups have no sign change of θ′. Their θ′ ranges are:
- row 1: G±(q−r) [−1.53e8, −27.0], G−q−r [−1.99e8, −58.1];
- row 2: G±(q−r) [−4.84e7, −8.55], G−q−r [−9.47e7, −39.6].

In every control the smallest |θ′| is at the z = 5.4e9 end, with no interior minimum.

## 5. Every method's G+q+r value (steps 3b–d)

Values are unscaled; the stored value is 250.99999999999994 times this. "Δref" is the value minus
that tree's own Φ = 1e4 reference.

**Row 1**

| method | R13 tree | regions | Δref | R14 tree | regions | Δref |
|---|---|---|---|---|---|---|
| (b) rtol 1e-7, as stored | −5.410262554830e-23 (abserr 6.8e-31) | 18 | −1.9e-31 | **−5.373287957651e-23** (abserr 1.0e-29) | **1** | **+3.697e-25 (6.8e-3)** |
| (b) rtol 1e-9 | −5.410262535846e-23 | 62 | −2.5e-33 | −5.373287957651e-23 | **1** | +3.697e-25 |
| (b) rtol 1e-11 | −5.410262535554e-23 | 131 | +4.4e-34 | −5.373287957651e-23 | **1** | +3.697e-25 |
| (c) 2^3 pieces | −5.410262554430e-23 | 20 | −1.9e-31 | −5.410262605113e-23 | 19 | −3.0e-31 |
| (c) 2^6 pieces | −5.410262553162e-23 | 71 | −1.8e-31 | −5.410262592823e-23 | 71 | −1.8e-31 |
| (c) 2^9 pieces | −5.410262535600e-23 | 515 | −2.5e-35 | −5.410262575281e-23 | 515 | −3.7e-35 |
| (c) 2^12 pieces | −5.410262535598e-23 | 4098 | −1.6e-36 | −5.410262575278e-23 | 4098 | −1.4e-35 |
| (d) Φ = 1e3 / **1e4** / 3e4 | −5.410262535593 / **−5.410262535598** / −5.410262535593 e-23 | | ≤4.5e-35 | −5.410262575274 / **−5.410262575277** / −5.410262575273 e-23 | | ≤4.5e-35 |

**Row 2**

| method | R13 tree | regions | Δref | R14 tree | regions | Δref |
|---|---|---|---|---|---|---|
| (b) rtol 1e-7, as stored | +3.134035827265e-23 (abserr 1.3e-32) | 16 | −6.9e-31 (2.2e-8) | **+3.130296649474e-23** (abserr 2.2e-30) | **1** | **−3.739e-26 (1.2e-3)** |
| (b) rtol 1e-9 | +3.134035827267e-23 | 18 | −6.9e-31 | +3.134035953026e-23 | 24 | +5.7e-31 |
| (b) rtol 1e-11 | +3.134035896207e-23 | 188 | −6.8e-34 | +3.134035896071e-23 | 187 | −7.8e-34 |
| (c) 2^3 pieces | +3.134035953089e-23 | 20 | +5.7e-31 | +3.134035952134e-23 | 20 | +5.6e-31 |
| (c) 2^6 pieces | +3.134035929884e-23 | 76 | +3.4e-31 | +3.134035930090e-23 | 76 | +3.4e-31 |
| (c) 2^9 pieces | +3.134035896264e-23 | 518 | −1.1e-34 | +3.134035896139e-23 | 518 | −9.9e-35 |
| (c) 2^12 pieces | +3.134035896266e-23 | 4124 | −9.0e-35 | +3.134035896141e-23 | 4124 | −8.3e-35 |
| (d) Φ = 1e3 / **1e4** / 3e4 | +3.134035896273 / **+3.134035896275** / +3.134035896273 e-23 | | ≤2.2e-35 | +3.134035896148 / **+3.134035896149** / +3.134035896147 e-23 | | ≤2.1e-35 |

Forced subdivision (c) converges in both trees to the reference. The 2^0 case reproduces (b) at
rtol 1e-7.

### The reference (d) and its error

- **Inside the window, no Levin.** The window is W(Φ) = {x : |θ(x) − θ(x_s)| ≤ Φ}, with edges
  found by brentq on θ.
  - It is cut into pieces of at most 0.75 rad of phase each. That is the fine resolution,
    (Δφ, n) = (0.5, 20); the coarse one is (1.0, 12), at most 1.5 rad.
  - Pieces are marched out from x_s, and every piece's phase span is measured.
  - Each piece is integrated by n-point Gauss–Legendre on f_sin sin ψ + f_cos cos ψ, with
    ψ = `group.theta_mod_2pi`. Pieces are summed with fsum.
- **Outside the window, Levin.** There |θ′| ≥ 5.4e4 (row 1) and 2.1e5 (row 2) for Φ = 1e3. The
  reference uses `adaptive_levin_sincos` as QSI calls it, forced onto 2^9 equal pieces.

| Φ | row 1 window, z | row 2 window, z | GL pieces (fine) |
|---|---|---|---|
| 1e3 | 9856 – 10590 | 272.35 – 277.70 | 3,998 |
| 1e4 | 9160 – 11501 | 266.70 – 283.63 | 39,996 |
| 3e4 | 8505 – 12646 | 260.82 – 290.16 | 119,996 |

Error of the reference at Φ = 1e4:
- GL fine against coarse: ≤1.4e-35.
- Spread over the three Φ: 4.5e-35 (row 1) and 2.2e-35 (row 2). A wider window replaces Levin by
  GL on the annulus between, so this spread measures the hand-off directly.
- Levin outer, rtol 1e-7 against 1e-10: ≤1e-35.
- **Tight total: 4.5e-35 (row 1) and 7.7e-35 (row 2), i.e. 8e-13 and 2.5e-12 relative.**
- Conservative, also adding Levin outer m6 against m9: 1.35e-32 and 5.9e-32.

Applied over the window, the same Levin forced subdivision (m = 9) agrees with GL to 6e-11
relative.

### Totals (`out/summary.json`)

| row | store | stored G+q+r | stored − ref | ÷ declared group abserr | stored total − corrected | ÷ total_abserr |
|---|---|---|---|---|---|---|
| 1 | R13 | −1.3579759013e-20 | −4.83e-29 | 0.28 | −4.83e-29 | 0.0095 |
| 1 | R14 | −1.3486952774e-20 | **+9.2806e-23** | **3.6e4** | **+9.2806e-23** | **1.28e4** |
| 2 | R13 | +7.8664299264e-21 | −1.73e-28 | 54 | −1.73e-28 | 0.0089 |
| 2 | R14 | +7.8570445902e-21 | **−9.3855e-24** | **1.7e4** | **−9.3855e-24** | **555** |

- The two trees' corrected totals differ by 6.4e-28 (row 1) and 1.9e-27 (row 2). That gap comes
  from the other groups' ~1e-8 tree differences, well inside total_abserr.
- Also replacing the other three low-sub-interval groups by their 2^12 sums gives:
  - row 1: R13 5.9185854809e-20, R14 5.9185854251e-20;
  - row 2: R13 −3.3961904654e-19, R14 −3.3961904983e-19.

## 6. Why R14 accepted one region and R13 did not (`out/rootdiag_*.json`)

Each probe calls `adaptive_levin_sincos` with `depth_max=0` on the root, then on the halves
leading down to x_s, and records the acceptance quantities the driver uses.

| row | tree | root estimate | step-(4) residual | relerr (rtol 1e-7) | eq.(151) floor | decision |
|---|---|---|---|---|---|---|
| 1 | R13 | −5.373288144e-23 | 1.104e-29 | 2.05e-7 | 1.024e-29 | bisect (residual is 8% above the floor) |
| 1 | R14 | −5.373287958e-23 | 3.23e-30 | 6.0e-8 | 1.024e-29 | **accepted, "resolved"** |
| 2 | R13 | +3.130296710e-23 | 3.15e-30 | **1.008e-7** | 9.5e-31 | bisect (by 0.8%) |
| 2 | R14 | +3.130296649e-23 | 2.19e-30 | 7.0e-8 | 9.5e-31 | **accepted, "resolved"** |

- **Both root estimates miss the stationary term.** They agree with each other to 3.5e-8.
- **The residuals differ because of the right half, [13.97, 22.41] (z ≈ 1e6–5e9).** For row 1
  that half's own residual is 1.28e-29 under R13 and 9.1e-31 under R14. This is where R13's φ
  noise acts, far from x_s.
- **The halves holding x_s carry tiny absolute residuals.** The left half's is 1.2e-32 (relerr
  1e-6), and its estimate also lacks the −3.7e-25. Deeper x_s-holding halves have relerr 2e-6 to
  3e-2, so once the root is bisected the driver resolves the stationary point.
- **Under R14, row 1 stays wrong at rtol 1e-9 and 1e-11.** The root is then accepted as
  phase-limited: the floor, inflated by a small but finite G0 = min|θ′| on the nodes, exceeds the
  residual. Tightening rtol cannot fix it.
- **Row 2 under R14 recovers at rtol 1e-9**, because its floor is below its residual.

## 7. The two trees' integrands (step 4; `out/compare.json`)

The common points are 28,002 per row: 20,001 uniform, plus 8,001 within |x − x_s| ≤ 0.05.

**Amplitudes are bit-identical at every point, in both rows.** That covers f_sin and f_cos of all
four groups, A_G, M, dlnM/dz and ω of T_q and T_r, H and w.

**Phases differ, R14 against R13, by:**
- G+q+r raw θ: 2.98e-8 rad, which is 1 ulp (row 1) and 2 ulp (row 2) of max|θ|.
- G+q+r θ mod 2π: up to 4.55e-8 rad cyclically (row 1) and 3.53e-8 (row 2), which is 1.5 and 2.4
  ulp of max|θ|. Within the window it is 1.0–1.1e-8.
- G+q+r θ′, relative to its envelope: 3.5e-14 and 1.0e-13 (pointwise 3.5e-12 and 3.4e-11).
- φ_G: 4.55e-8 and 3.53e-8.
- φ_T, the same for q and r: 9.8e-9, which is 1.3 ulp of max|θ_T|.

**What the phase differences are.** They are not round-off of the remainder: they are about 5e7
times ulp(2π) = 8.9e-16. Nor are they larger than the granularity of the unreduced phase: they are
1–2.4 ulp of k τ ≈ 2.4e8 rad. That is R13's known quantisation of φ,
`[02-consumer-phi-below-the-storage-granularity]`.

**The knots of the φ spline show R13 is the noisy side.** The test takes each knot's residual from
the cubic through its four neighbours, over about 730 knots inside the sub-interval:
- φ_G under R13 is a staircase: median residual 1e-23 rad (exactly flat), maximum 7.0e-8 rad (2.3
  ulp). Under R14 it is smooth: median 7e-19, maximum 1.2e-9 rad (0.04 ulp).
- φ_T: median 7.8e-11 and maximum 9.9e-9 under R13, against 6.5e-13 and 3.0e-9 under R14.

**Effect on the integral.** The two trees' G+q+r references differ by 4.0e-31 (7.3e-9 relative)
for row 1 and 1.3e-33 (4e-11) for row 2.

## 8. Controls (step 5): the other three groups

None of the other three groups has a stationary point.
- **All six one-region calls are phase-limited.** In both trees, all three groups in row 1 and
  G−q−r in row 2 are accepted as one region, 3 evaluations, "converged" False, at every rtol.
- **Forced subdivision changes them only inside their declared abserr.** The m ≥ 9 sums differ
  from the single-region values by 1e-8 to 2e-8 relative.
- **Row 2's G±(q−r) take 2 regions at rtol 1e-7.** Their gap to the forced value is 3.1e-8 under
  R13 and below 1e-9 under R14, inside abserr either way.
- **The trees agree to 5e-9 to 2e-8 relative** in their forced m = 12 sums.

## 9. Which hypothesis

**H1 (false acceptance under the noise-free phase): supported, strongly.**
- The composed phase has one interior stationary point.
- The single region misses that point's contribution to within 1e-5 of the leading-order
  estimate.
- The miss is invisible to the step-(4) residual in both trees.
- Under R14 the root passes the relative test (6.0e-8 and 7.0e-8 against 1e-7), because the
  denominator is dominated by the high-z half. For row 1, the floor would also accept the root at
  any tighter rtol.

One refinement to H1 as stated. R13's noise did not "force bisection of the stationary point". It
lifted the root's residual over rtol, by factors of 2.05 and 1.008, through the unrelated high-z
half, and bisection then exposed the point. That is luck.

**H2 (R14's integrand differs by more than round-off): refuted.**
- The amplitudes are bit-identical.
- The phases differ by about 1 ulp of the unreduced phase, and the noisy side is R13.
- Integrated correctly, the two trees' values agree to 7e-9 and 4e-11 relative.

## 10. What this means

**For 06 (dc9d28c): no defect in 06's code.** R14's integrand is the more accurate of the two. 06
removed the phase noise that, by chance, made AdaptiveLevin bisect these two roots.
- **R14's stored values for these rows are wrong.** The G+q+r group is off by 6.8e-3 and 1.2e-3;
  the total by 1.6e-3 and 2.8e-5.
- **R13's are right**, to 3.6e-9 and 2.2e-8 of the group.
- **Other rows are exposed, in stores built at either tree.** Any QSI row whose phase group has an
  interior stationary point inside a Levin sub-interval is at risk. These are mainly G+q+r groups
  where 2c_s q ≈ k. Such a row survives only if noise or a tight rtol happens to force bisection.
- **The exposed rows can be listed from stored metadata.** The census would select an
  all-oscillatory sub-interval, a group whose θ′ changes sign, and few regions.

**For AdaptiveLevin: its error control is not valid for a Levin region that holds an interior
stationary point.**
1. The step-(4) residual compares estimates that all miss the stationary-phase term. Here the
   residual was about 1e5 times smaller than the true error.
2. The relative test can be passed through another part of the region.
3. The eq.(151) floor, with a small but finite G0, accepts the region as phase-limited at any
   rtol. The docstring already warns about this for G0 at the noise floor; here G0 is well above
   it.

**Possible remedies, not implemented here (out of scope):**
- θ′ is already sampled at the nodes, for G0 and G1. A sign change among them could be made to
  exclude both Levin acceptance and the phase-limited exit, until the region is small enough to
  route to the Clenshaw–Curtis fallback.
- Or QSI could split each group's call at the zeros of its closed-form `theta_deriv`.

This should be opened as a §3 issue on the board that owns AdaptiveLevin and QSI, and indexed in
`docs/OPEN_ISSUES.md`.

**Observations not acted on:**
- R13's row 2 G+q+r at rtol 1e-7 and 1e-9 is off by 2.2e-8, which is 54× its declared abserr of
  1.3e-32. It is the same blindness at a smaller scale.
- The control groups are routinely accepted as one phase-limited region with "converged" False.
- I waited for my two unregistered `reference.py` processes with one blocking `pgrep -f` loop. They
  are not registered runs, and nothing was polled across turns.

**Key files:**
- `out/validate_r13.json` and `out/validate_r14.json`: the bit-for-bit validation.
- `out/analyse_r13.json` and `out/analyse_r14.json`, with their `_pieces.npz` (kept on the Mac
  Studio): steps (a), (b), (c) and the controls.
- `out/reference_r13.json` and `out/reference_r14.json`: step (d).
- `out/rootdiag_r13.json` and `out/rootdiag_r14.json`: the root decision.
- `out/samples_r13.npz`, `out/samples_r14.npz` (kept on the Mac Studio) and `out/compare.json`:
  step 4.
- `out/summary.json`: all the tables above.
- `out/store_sha256_before.txt` and `out/store_sha256_after.txt`: the stores' hashes.
