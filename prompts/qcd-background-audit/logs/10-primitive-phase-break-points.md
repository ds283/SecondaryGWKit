# Log 10 — `PrimitivePhase` on the 3-point break set: the knots are refuted, the grid is the cause

**Prompt:** [`prompts/qcd-background-audit/10-primitive-phase-break-points.md`](../10-primitive-phase-break-points.md)
**Commit:** *(this commit)* — Measure the consumer phase spline against the cosmology's break points
**Model:** Claude Opus 5
**Date:** 2026-09-15
**Parent:** `bc8c3e4` (prompt 09) · **Campaign base:** `2a5e0fa` (numerically `e8f746d`)
**Result:** COMPLETE WITH DEVIATIONS

**The one-sentence version.** Prompt §1 said the first job is not to build anything but to
re-measure, and the measurement says **no break-point-aware knot vector helps**: on the corrected
background the two constructions the prompt names are **2.09× and 5.00× worse** than the shipped
default at QCD $k=10^5$, the best of the four controls is 1.21× better against a 4.25× rise to
recover, and every other row in the twelve is untouched by every scheme. What *does* recover them is
**samples**: a plain cubic of the same $\varphi$, default knots and no break-point treatment
whatever, goes **34.11 → 1.48 ulp** ($G_k$) and **1876.61 → 72.76 ulp** ($T_k$) when the production
source grid is doubled, and **34.11 → 1.64 ulp / 1876.61 → 74.60 ulp** when only the ±5 grid
intervals around the crossing are doubled. So the residue at `QCD_EOS`'s `T_LO` crossing is the
**production sample grid's**, not `PrimitivePhase`'s — which is
`prompts/phase-representation/IMPLEMENTATION_STATE.md` §5 note 6 put to the test on the corrected
background rather than inherited, and which hands the defect to this campaign's **G2** (prompts
11–12), not to the spline. **No production file was changed.**

---

## What shipped

**No production code.** `ComputeTargets/primitive_phase.py`, `GkSourcePolicyData.py`,
`TkSourceFunctions.py` and `BackgroundModel.py` are byte-identical to `bc8c3e4`; no
`CosmologyModels/`, `Quadrature/`, `LiouvilleGreen/`, `Datastore/` or `main.py` file is in the diff.
**`T_Z_REPRESENTATION_VERSION` is 5 before and after** — nothing numerical moved, and nothing could.

Two files of substance:

1. **`docs/qcd-background-audit/consumer_knot_scheme_scan.py`** (new, 808 lines, 119.6 s at load average 6.1, no Ray, no
   datastore). Reproduces `docs/gktk-remedial/verify_production_path.py`'s `consumers` geometry —
   both models, both sectors, all three production wavenumbers, ten points per production interval,
   the same fixtures and the same reference — and scores **nine knot schemes plus two controls**
   against one set of producer runs. Public names: `repeated_knot_vector(sites, breaks, order,
   multiplicity)`, `SegmentedSpline(sites, values, breaks, order)`, `build_cases(models, k_values,
   verbose=True)`, `resolution_ladder(case, u_break)`, `SCHEMES`, `DEFAULT_SCHEMES`. Its `base`
   column reproduces `docs/qcd-background-verification.md` §3.1 and §3.2 to every printed digit,
   which is its own check that it measures the same thing.

2. **`ComputeTargets/tests/test_primitive_phase.py`** gains
   `TestBreakPointKnotsBuyWhatTheSamplesResolve` (2 tests, 0.007 s), which puts the rule in the tree
   rather than only in this log: a slope discontinuity the samples **do** resolve is recovered by a
   $C^0$ knot essentially exactly (5.22e+12×), and a feature whose width is the sample spacing is
   **not** recovered by any knot placement (the $C^0$ knot is 4.4× *worse*) and **is** recovered by
   sampling (2× density, 25.8× better). Those two synthetic factors bracket the production ones
   (2.09× worse; 23–26× better) without being tuned to them.

### The design decision, as prompt §6 requires it to be stated

**Neither construction was taken, and no third one was substituted.** Prompt §2 item 2 makes a
repeated-knot vector the default and per-segment splines the fallback. Both were implemented in the
scan, scored at production geometry on all six QCD and all six LambdaCDM (sector, $k$) cases, and
both are **worse**:

| at QCD $k=10^5$ | $G_k$ | $T_k$ |
|---|---|---|
| shipped default knots (`base`) | 8.1062e-06 rad (34.00 ulp) | 1.3982e-05 rad (1876.61 ulp) |
| **repeated multiplicity-3 ($C^0$) knot** at each declared break | **1.6928e-05 (71.00)** — 2.09× worse | **2.9315e-05 (3934.6)** — 2.10× worse |
| **per-segment splines**, one per interval between breaks | **4.0531e-05 (170.00)** — 5.00× worse | **7.0770e-05 (9498.6)** — 5.06× worse |

`num_chunks` is untouched and still returns 1 — nothing was split and nothing was chunked, so
`phase-representation` README §2 (d)'s distinction does not arise, and `WKB_phase_spline_chunks`
cannot have moved.

**Why the repeated knot loses, mechanically.** An interpolating knot vector has fixed length
$n+k+1$, so a multiplicity-3 knot must be paid for by removing three knots, and prompt 02 measured
that Schoenberg–Whitney permits that payment only **locally** — the three knots nearest the break.
The $C^0$ freedom it buys is worth less than the coarsening it costs, because what is at the
crossing is not a resolved corner. **Why the segments lose:** every declared break lies strictly
inside a grid interval (fractional position 0.6424 at `T_LO` on both $k=10^5$ grids), so each side's
spline must **extrapolate** up to 0.64 of an interval past its last data site to reach the break;
both sectors' maxima move onto exactly that point ($z=4.25278\times10^7$, against $4.24388\times10^7$
for `base`).

---

## Deviations from the prompt

### 1. `STRUCTURALLY REQUIRED` — prompt §2's change was not made, because §1's measurement says not to

Prompt §2 is written as "the change, if a change is warranted", and prompt §1 says the first job is
to re-take prompt 02's measurement and that closing the issue with a measurement and no code is
`COMPLETE`. §3 below is that measurement and it warrants no change: the two constructions §2 names
are 2.09× and 5.00× worse, the four controls move nothing outside $k=10^5$ and buy at most 1.21×
there, and §3.4 identifies what does work. Nothing in §2 items 1–5 was implemented: there is no
`break_points` parameter, no call site changed, `BackgroundModel.py` was not touched, `spline_order`
is still 3, and $\varphi$ alone is still what is splined.

### 2. `IMPLEMENTATION CHOICE` — `[13-consumer-spline-crosses-eos-break-points]` is **closed**, and a correctly-named successor is opened

Prompt §6 says to move the entry to the `GkTk-remedial` board's §4. `prompts/phase-representation`
prompt 02 was told the same thing and declined, on the ground that moving an unfixed defect records
a fix that does not exist. The accuracy defect is still unfixed, so that argument has not gone away —
and I have taken the other decision. The reasons, so a later reader can disagree on the merits:

- What the entry *names* is refuted, not merely un-attempted. Its title, its original "next step"
  ("give `PrimitivePhase` a knot vector that repeats a knot at each `integration_break_points`
  value") and prompt §2's two constructions have now been measured on **both** backgrounds, at all
  three wavenumbers and in both sectors, across the whole constructible family. There is nothing
  left for anyone to try under that framing.
- The entry has been narrowed four times and its supersession prediction falsified once. A fifth
  narrowing under a title that names the wrong remedy is the drift `docs/OPEN_ISSUES.md` exists to
  prevent.
- The defect is not lost. `[10-consumer-phi-unresolved-at-the-eos-crossing]` opens on **this**
  campaign's board §3 with the same numbers, the correct mechanism and a *measured* remedy, assigned
  to **prompt 11**. The index count is unchanged: one row out, one row in.

The alternative — leave it open, narrowed a fifth time, and open nothing — keeps the history in one
place at the cost of a title that no longer describes the defect. If the orchestrator prefers it,
reverting this commit's two board edits and the index swap restores it exactly.

### 3. `IMPLEMENTATION CHOICE` — a new script under `docs/qcd-background-audit/`, which is not on the prompt's file list

Prompt §3 asks for before/after at all three wavenumbers in both sectors.
`docs/gktk-remedial/verify_production_path.py` cannot supply it: it may not be edited, it builds its
$G_k$ consumer by calling `PrimitivePhase(...)` directly (`[02-verify-script-builds-its-own-Gk-consumer]`,
restated in `docs/qcd-background-verification.md` §3.4), and it exercises one scheme per run where
nine were needed. `consumer_knot_scheme_scan.py` reproduces its geometry and its `base` column
exactly and pays for the producer runs once. The precedent is prompts 08 and 09, which added
`per_sector_policy_remeasure.py` and `consumer_break_point_profile.py` to the same directory. It
touches no production file and no other campaign's driver.

### 4. `IMPLEMENTATION CHOICE` — the scan substitutes the spline object rather than passing a parameter

The controls the attribution needs — multiplicity 1 and 2, and the `phi_zero` column that removes
$\varphi'$ altogether — are not constructions anything should be able to ask for in production, and
there is no production parameter to pass in any case. So the scan builds the production object
(through `TkSourceFunctions` in the $T_k$ sector, exactly as the production call site does) and then
replaces `_spline` / `_spline_deriv`. `base` reproducing §3.1 and §3.2 to the printed digit is the
check that the substitution is faithful.

### 5. `IMPLEMENTATION CHOICE` — the two new tests are synthetic, and assert a rule rather than a production number

Prompt §4's four tests all presuppose a production change. With none made, the useful thing to lock
in the tree is the *rule the measurement establishes*, at the scale of one grid interval, in 7 ms:
a $C^0$ knot recovers what the samples resolve and nothing else. The production measurement itself
is a 117 s script and belongs in `docs/`, where it is. Prompt §4 item 1's "show it fails on the old
code" is answered in the opposite direction and is reported as such: on the geometry that matters
the proposed new code fails and the old code does not.

Nothing is tagged `UNINTENDED DRIFT`. No README §2 design fact was touched.

---

## Verification performed

Everything below was run on this tree. **Machine load averages are quoted with every wall time**;
this machine had been under erratic external load (above 140 the previous day), and no conclusion
here rests on a clock.

### 1. The base is reproduced before anything is measured

`PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py --section consumers`,
unedited, at `bc8c3e4`: all twelve §3.1 rows and all twelve §3.2 column groups reproduce
`docs/qcd-background-verification.md` **to every printed digit**, including
`8.1062e-06 rad at z = 4.24388e+07 (34.00 ulp)` and
`1.3982e-05 rad at z = 4.24388e+07 (1876.61 ulp)`, and `1.0232e-06` / `3.0630e-06` for
`theta_deriv`. The scan's `base` column then reproduces those same numbers, which is what licenses
every other column in it.

### 2. The consumer phase, nine schemes × twelve rows (prompt §3 bullet 1)

`max |theta - ref|` in rad, ulp of the span in brackets, at ten points per production interval.
`ALLx3` is the prompt's repeated-knot vector at `BREAK_POINT_ALL`; `DISCx3` the same at
`BREAK_POINT_DISCONTINUITY`; `segALL` / `segDISC` the per-segment splines; `x1` and `x2` are
**controls, not proposals** (a multiplicity-1 knot drops no continuity at all — it merely moves one
knot onto the break).

| model | $k$ | sector | base | segDISC | segALL | DISCx3 | ALLx3 | DISCx1 / ALLx1 | DISCx2 / ALLx2 |
|---|---|---|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | $G_k$ | 2.3842e-07 (1.00) | *identical in all eight* | | | | | |
| LambdaCDM | 1e7 | $G_k$ | 0.0000e+00 (0.00) | *identical* | | | | | |
| LambdaCDM | 3e8 | $G_k$ | 0.0000e+00 (0.00) | *identical* | | | | | |
| LambdaCDM | 1e5 | $T_k$ | 7.4506e-09 (1.00) | *identical* | | | | | |
| LambdaCDM | 1e7 | $T_k$ | 9.5367e-07 (1.00) | *identical* | | | | | |
| LambdaCDM | 3e8 | $T_k$ | 3.0518e-05 (1.00) | *identical* | | | | | |
| **QCD** | **1e5** | **$G_k$** | **8.1062e-06 (34.00)** | 4.0531e-05 (170.0) | 4.0531e-05 (170.0) | 1.6928e-05 (71.0) | 1.6928e-05 (71.0) | 6.6757e-06 (28.0) | 6.6757e-06 (28.0) |
| **QCD** | **1e5** | **$T_k$** | **1.3982e-05 (1876.6)** | 7.0770e-05 (9498.6) | 7.0770e-05 (9498.6) | 2.9315e-05 (3934.6) | 2.9315e-05 (3934.6) | 1.1631e-05 (1561.1) | 1.1528e-05 (1547.2) |
| QCD | 1e7 | $G_k$ | 1.5259e-05 (1.00) | *identical in all eight* | | | | | |
| QCD | 1e7 | $T_k$ | 9.5367e-07 (1.00) | *identical* | | | | | |
| QCD | 3e8 | $G_k$ | 4.8828e-04 (1.00) | *identical* | | | | | |
| QCD | 3e8 | $T_k$ | 3.0518e-05 (1.00) | *identical* | | | | | |

**Three things this table says, and only the first was expected.**

- **All six LambdaCDM rows are identical under every scheme**, because LambdaCDM declares no break
  points and `_cosmology_break_points` returns an empty array; prompt §2 item 5's acceptance test
  is met by construction and is confirmed here rather than assumed.
- **The ten rows at 1.00 ulp stay at 1.00 ulp under every scheme, including `ALLx1`.** This is new,
  and it is prompt 07's doing: when prompt 02 measured, `ALLx1` broke QCD $G_k$ at $10^7$ and
  QCD $T_k$ at $3\times10^8$ from 1.00 to 3.00 ulp, because `BREAK_POINT_ALL` was then 226–325
  points on those grids. It is now 1–3 points and no scheme in the family touches a row other than
  the two at $k=10^5$. **Prompt 02's ranked option (a) — that no scheme is a strict improvement —
  is therefore no longer true as stated**, and the trap it warned of (a scheme that looks good at
  $k=10^5$ and regresses $10^7$ / $3\times10^8$) has been removed by prompt 07 rather than by
  anything here. Scoring all three wavenumbers is still what establishes that, which is why it was
  done.
- **Nothing reaches the target.** Prompt §5 wants ≤ 1e-06 rad and ≤ 2 ulp, and the campaign base is
  1.907e-06 / 3.186e-06 rad. The best of the nine is 6.6757e-06 rad (28.00 ulp) and
  1.1528e-05 rad (1547.2 ulp) — 1.21× better than shipped, against the 4.25× and 4.39× that would
  have to be recovered, and 3.5× and 3.6× above the campaign base. **A break-point-aware knot
  vector does not bring those rows back.**

### 3. `theta_deriv` against $\omega$, split and attributed (prompt §3 bullet 2)

Relative error of $|{\rm theta\_deriv}(z)|$ against $\omega$ over the consumer's own samples, deep
interior. `phi_zero` is the decisive control: the same object with the $\varphi$ spline's derivative
**removed**, so that `theta_deriv` is the closed-form leading term alone. `phi range` is the dynamic
range of the recovered $\varphi$ in ulp of the stored phase, which is what
`[02-consumer-phi-below-the-storage-granularity]` is about.

| model | $k$ | sector | base | best break-point scheme | `phi_zero` | $\varphi$ range [ulp] |
|---|---|---|---|---|---|---|
| LambdaCDM | 1e5 | $G_k$ | 4.3707e-10 | 4.3707e-10 (none declared) | 4.3707e-10 | 1.0 |
| LambdaCDM | 1e5 | $T_k$ | 8.8279e-12 | 8.8279e-12 | 4.7163e-03 | 1.16e+07 |
| LambdaCDM | 1e7 | $G_k$ | 4.3760e-12 | 4.3760e-12 | 4.3760e-12 | 0.0 |
| LambdaCDM | 1e7 | $T_k$ | 8.3889e-12 | 8.3889e-12 | 4.5137e-03 | 8.85e+04 |
| LambdaCDM | 3e8 | $G_k$ | 1.4363e-13 | 1.4363e-13 | 1.4363e-13 | 0.0 |
| LambdaCDM | 3e8 | $T_k$ | 8.5379e-12 | 8.5379e-12 | 4.5828e-03 | 2.79e+03 |
| **QCD** | **1e5** | **$G_k$** | 1.0232e-06 | **6.5730e-07** (`DISCx3`, 1.56×) | 9.9272e-06 | 1.12e+03 |
| **QCD** | **1e5** | **$T_k$** | 3.0630e-06 | **1.9771e-06** (`DISCx3`, 1.55×) | 4.6918e-03 | 1.18e+07 |
| **QCD** | **1e7** | **$G_k$** | 6.0229e-06 | 6.0229e-06 (**no scheme moves it**) | **3.9489e-06** | **6.0** |
| QCD | 1e7 | $T_k$ | 3.0630e-10 | 1.9770e-10 (`DISCx3`, 1.55×) | 5.0634e-03 | 9.31e+04 |
| **QCD** | **3e8** | **$G_k$** | 2.5418e-04 | 2.5418e-04 (**no scheme moves it**) | **1.5976e-05** | **2.0** |
| QCD | 3e8 | $T_k$ | 7.3236e-06 | **1.2456e-05 — 1.70× worse** under both $C^0$ schemes | 4.8542e-03 | 3.02e+03 |

**The split the review asked for, in one sentence each.**

- **The break points' share is at most 36 %, and only at $k=10^5$.** Every $C^0$ scheme recovers
  1.0232e-06 → 6.5730e-07 ($G_k$) and 3.0630e-06 → 1.9771e-06 ($T_k$) — 35.8 % and 35.5 % of the
  base figure — and it costs 2.09× and 2.10× of consumer phase to buy it. Prompt 02 measured the
  same share (38 %) at the same price on the defective background; that half of its finding is
  **unchanged by the corrected background**.
- **`[02-consumer-phi-below-the-storage-granularity]`'s share is 100 % of the two rows that miss the
  target.** QCD $G_k$ at $10^7$ and $3\times10^8$ are the only rows above $10^{-6}$ that are not at
  $k=10^5$, and there the recovered $\varphi$ spans **6.0** and **2.0 ulp** of the stored phase.
  **No scheme moves either figure by a printed digit**, and removing $\varphi'$ altogether
  *improves* them by **1.53×** and **15.9×** — differentiating a 3-valued staircase is worse than
  contributing nothing. That is prompt 02's item 4 reproduced exactly, on a background whose $T(z)$
  error has fallen by seven orders, which is the strongest available evidence that it is neither
  the knots' nor the representation's. README §0.5 puts it out of this campaign's scope and it
  stays there.
- **A third row is a regression the $k=10^5$-only reader would have shipped.** QCD $T_k$ at
  $3\times10^8$ goes 7.3236e-06 → **1.2456e-05** under both $C^0$ schemes. Prompt 02's warning —
  "`DISC × 3` improves `theta_deriv` while doubling the phase error, and a one-number acceptance
  test would have shipped it" — fires again, in a different row.

### 4. What *does* recover the crossing: the resolution ladder (section E of the scan)

$\varphi$ reconstructed from the dense reference, then splined with **default knots and no
break-point treatment at all**, given different *samples*. The stored node values are kept exactly
where they exist, so the first row is the shipped consumer. Scored against the reference's own
$\varphi$ at the points not used in the fit; $h = 2.3030\times10^{-2}$ in $u$.

| samples given to a plain cubic | QCD $G_k$ $k=10^5$ | QCD $T_k$ $k=10^5$ |
|---|---|---|
| **the production grid (the shipped consumer)** | 8.1329e-06 rad, **34.11 ulp** | 1.3982e-05 rad, **1876.61 ulp** |
| +4 samples inside the break's own interval alone | 17.42 ulp | 960.11 ulp |
| refine ±1 interval by 5× | 15.80 | 848.39 |
| refine ±2 intervals by 5× | 7.93 | 447.57 |
| refine ±3 intervals by 5× | 2.29 | 170.86 |
| **refine ±5 intervals by 2×** | **1.64** | **74.60** |
| refine ±5 intervals by 5× | 1.64 | 26.97 |
| refine ±10 intervals by 5× | 1.04 | 26.97 |
| uniform 2× over the whole range | 1.48 | 72.76 |
| uniform 5× over the whole range | 0.95 | 2.87 |

**Read it as three statements.**

1. **The feature is real and it is resolvable.** Doubling the grid takes $G_k$ to 1.48 ulp — the
   floor the other ten rows sit at — where the best knot vector leaves it at 28.00.
2. **It is not confined to the break's own interval.** Four extra samples *inside* that interval buy
   1.96× and stall; the error only collapses once ±3 to ±5 intervals are refined. That is prompt
   02's "±3 grid interval" arch structure, measured again now that the $T(z)$ knot lattice that was
   blamed for it is gone (`T(z)` max 7.18e-04 → 6.807e-11, prompt 06) — so it is $\varphi$'s own,
   exactly as prompt 02 said before the supersession paragraph doubted it.
3. **±5 intervals at 2× is the cheap fix, and it is a grid fix.** 10 extra samples in 1,016 —
   **1.0 %** — take $G_k$ to 1.64 ulp (3.9121e-07 rad, inside prompt §5's ≤ 1e-06 rad and ≤ 2 ulp)
   and $T_k$ to 5.5584e-07 rad, also inside the rad target. At 5× over the same ±5 intervals (40
   extra samples, 3.9 %) $T_k$ reaches 2.0096e-07 rad. **Nothing in `PrimitivePhase` can do this**,
   because the information is not in the node values it is given.

**At $k=10^7$ and $3\times10^8$ the ladder reads 0.00 ulp near the break in both sectors, at every
density**: the crossing contributes nothing at all there, and those rows' maxima are elsewhere and
at the representation floor. That is the independent confirmation that the crossing is a $k=10^5$
phenomenon and that the $k\ge10^7$ `theta_deriv` misses belong to the other issue.

### 5. Cost (prompt §3 bullet 3)

**No production code changed, so no production cost moved**; the row is measured rather than
asserted. Best of 9, load average 12.28 at the time of measurement:

| model | sector | samples | `PrimitivePhase(...)` build | per sample | integrand evaluations |
|---|---|---|---|---|---|
| LambdaCDM | $G_k$ | 1,361 | 3.18e-04 s | 0.234 µs | **0** |
| QCD | $G_k$ | 1,377 | 3.38e-04 s | 0.246 µs | **0** |

The constructor performs an `argsort`, a `log1p` and one `make_interp_spline`; $\varphi$ is supplied
by the caller, so it evaluates no integrand at all. Prompt §3's figure to beat — **0.0010 s / 468
evaluations** — is `gktk-remedial-verification` §3.9's cost of a whole `GkWKBIntegration` *producer*
object, not of a `PrimitivePhase` build (prompt 02's log called that row vacuous for the same
reason). Against it the build is 0.32–0.34× and the stop condition (2×) is nowhere near. The
`TkSourceFunctions(...)` end-to-end construction, which is a different quantity, measured 0.0103 s
(LambdaCDM) and 0.1795 s (QCD) at $k=3\times10^8$ — unchanged, and dominated by the per-$k$ residual
table (`[07-tk-per-object-cost-is-all-setup]`).

### 6. The tree

| suite | before (`bc8c3e4`) | after | command |
|---|---|---|---|
| `CosmologyModels/tests` | 30 OK | **30 OK** (0.63 s) | `discover -s CosmologyModels/tests -t .` |
| `ComputeTargets/tests` | 359 OK | **361 OK** (153.3 s) | `discover -s ComputeTargets/tests -t .` |
| `LiouvilleGreen/tests` | 143 OK (fast set) | **143 OK** (14.2 s) | every module except `test_3bessel_analytic` |

**The `LiouvilleGreen` figure is the fast set**, as prompts 02–08 used: `test_3bessel_analytic` is
excluded and the full set is 148 (prompt 09 ran it). This commit touches no `LiouvilleGreen` file
and no file any of its tests import. No count fell; the +2 are the new test class.

`black` is clean on both files this commit adds or edits.

---

## Observations not acted on

1. **`docs/gktk-remedial/verify_production_path.py` still builds its own $G_k$ consumer.**
   `[02-verify-script-builds-its-own-Gk-consumer]`, open on the `phase-representation` board. It did
   not obstruct this prompt — the scan reproduces its geometry and can pass anything — but it means
   that if prompt 11 changes the source grid, six of §3.5's twelve rows will still not exercise the
   production $G_k$ construction. Not fixed: not this prompt's file.

2. **`DISCx3` and `ALLx3` are indistinguishable to every digit on every row, as are `segDISC` and
   `segALL`, and `DISCx1`/`ALLx1`.** After prompt 07 the two kinds differ only by `EOS_T_LO`, where
   $w$ kinks and $g_s$ does not, and the measurement says $\varphi$ cannot tell: prompt §2 item 1's
   "measure which `kind` serves $\varphi$ better; do not assume" is answered **neither**. If a later
   attempt does route break points into a consumer, this is evidence that the choice of kind is not
   where the accuracy is. No issue opened — it is a measurement, not a defect.

3. **The one-sided $[\varphi']$ fit at the crossing is still window-dependent on the corrected
   background**, which is the signature prompt 02 identified. `kink_fit` in the scan (section E)
   fits cubics to the dense reference over windows of 1, 2 and 3 grid intervals either side of
   `T_LO` and gives $[\varphi'] = +7.52$e-05,
   $-3.00$e-03, $-6.15$e-03 ($G_k$) and $-2.17$e-04, $+5.19$e-03, $+1.06$e-02 ($T_k$). A genuine
   slope discontinuity gives a window-independent jump; this does not, at any window the production
   grid supports. **That is why a $C^0$ knot cannot help**, and it is the quantitative form of §4's
   conclusion. Prompt §1's third row — "probably still true, and it is the thing to establish
   first" — is therefore confirmed, with the corrected background underneath it.

4. **`[02-consumer-phi-below-the-storage-granularity]` now has a second, stronger measurement
   behind it** (§3 above: 6.0 and 2.0 ulp of range; `phi_zero` better by 1.53× and 15.9×) taken on a
   background whose $T(z)$ error is seven orders smaller. Its entry on the `phase-representation`
   board was not edited — it is not this prompt's to narrow, and the numbers are here and in
   `docs/qcd-background-verification.md` §8 for whoever takes it.

---

## State handed to the next prompt

1. **The tree is `bc8c3e4` plus a script, a test class and documentation.** No `.py` file outside
   `ComputeTargets/tests/` and `docs/` differs from prompt 09's commit, and
   `T_Z_REPRESENTATION_VERSION` is **5**, unchanged. Reverting this commit costs nothing but the
   record and the two tests.

2. **`PrimitivePhase` keeps `make_interp_spline`'s default knots, on measurement.** There is no
   `break_points` parameter, no call site passes one, `spline_order` is 3, `num_chunks` is 1, and
   `ComputeTargets/BackgroundModel.py` was not touched. Anyone re-opening this must first beat
   `docs/qcd-background-audit/consumer_knot_scheme_scan.py`'s `base` column — nine schemes already
   do not.

3. **The measured remedy, for prompt 11, in one line:** refine the production source grid over the
   **±5 grid intervals around each declared discontinuity** by a factor of **2** — 10 extra samples
   in 1,016, 1.0 % — and QCD $k=10^5$ goes 34.11 → **1.64 ulp** ($G_k$, 3.9121e-07 rad) and
   1876.61 → **74.60 ulp** ($T_k$, 5.5584e-07 rad), both inside prompt §5's 1e-06 rad target, with
   $G_k$ also inside its 2-ulp target. A factor of 5 over the same ±5 intervals takes $T_k$ to
   2.0096e-07 rad. Refining the break's own interval alone buys 1.96× and stalls — **the feature is
   ±3 to ±5 intervals wide, not a point**, and a grid design that protects only the crossing itself
   will not work.

4. **The three declared crossings, unchanged from prompts 06 and 07**, in $u=\log(1+z)$:
   `17.565806941870026` (`T_LO`, the only one in range at $k=10^5$ and the one all four risen rows
   sit at), `23.197460552819653`, `27.485391822044257`. `BREAK_POINT_ALL` is 3 on the production
   source grid and `BREAK_POINT_DISCONTINUITY` is 2; $\varphi$ cannot tell them apart (observation 2).

5. **The reproduction, one command, ~120 s, no Ray and no datastore:**
   ```bash
   PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/consumer_knot_scheme_scan.py
   ```
   `--schemes` selects a subset, `--models` / `--k` narrow the geometry, `--no-ladder` skips
   section E, `--json` writes the raw results. The QCD reference fixture is untouched by this commit
   and its regeneration command is unchanged:
   `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py --dry-run`.

6. **The issue moved.** `[13-consumer-spline-crosses-eos-break-points]` is **closed** on the
   `GkTk-remedial` board §4 (deviation 2), and `[10-consumer-phi-unresolved-at-the-eos-crossing]` is
   open on this campaign's board §3 and in `docs/OPEN_ISSUES.md`, **assigned to prompt 11**. The
   index count is unchanged.

7. **What this does not touch.** `[02-consumer-phi-below-the-storage-granularity]` and
   `[00-consumer-anchoring-floor]` are now measured to be 100 % of the two QCD $G_k$ `theta_deriv`
   rows that miss $10^{-6}$, and README §0.5 keeps them out of this campaign. They are per-region
   anchoring and no grid change touches them either.
