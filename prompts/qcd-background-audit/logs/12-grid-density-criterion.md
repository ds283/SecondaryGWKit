# Log 12 — a measured criterion for the grid's density

**Prompt:** [`prompts/qcd-background-audit/12-grid-density-criterion.md`](../12-grid-density-criterion.md)
**Commit:** *(this commit)* — Measure what the source grid's density buys and what it wastes
**Model:** Claude Opus 5
**Date:** 2026-09-15
**Parent:** `4f725c0` (prompt 11) · **Campaign base:** `2a5e0fa` (numerically `e8f746d`)
**Result:** COMPLETE WITH DEVIATIONS

**The one-sentence version.** The uniform `source_samples_per_log10z = 100` is wrong in **both**
directions at once — the consumer's $\varphi$ spline is under-resolved by **7.86×** and **7.84×**
the storage floor in the top decade of the $T_k$ band at $k=10^5$ on LambdaCDM and QCD, and
over-resolved by up to **$10^{19}$** at the bottom of the range — and the criterion that fixes it,
$h^4|\varphi''''|/384 \le \varepsilon$, is **computable before the grid exists** from $H$, $c_s^2$
and $k$ alone and predicts the realised error **to ±2 %** over 500-odd intervals in the
transfer-function sector on both models at all three wavenumbers. At the **same** sample count it
puts every production row inside its floor; at the **same** accuracy it needs **1.75×** (QCD) and
**2.06×** (LambdaCDM) fewer samples. **Nothing was changed**: `T_Z_REPRESENTATION_VERSION` is 5
before and after and no production file is in the diff. The recommendation, its cost and its
ceiling are §10.0 of `docs/qcd-background-verification.md` and
`[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]` on the board.

**And one thing the prompt did not ask for but could not be measured around.** Building the QCD
background on the grid prompt 11 ships — which is what `main.py` does — moves the consumer's
$\varphi$ error at the `T_LO` crossing from prompt 11's measured **3.97e-07** to **2.49e-05**, a
factor of **1.77 worse than the base grid** rather than 35× better, because `epsilon` on QCD is a
spline of $\log H$ over a refinement of the source grid itself and refining it at a genuine step in
$H$ makes the ringing narrower faster than it makes it smaller. Prompt 11 held the background fixed
on the base grid, correctly for what it measured. §10.4 and
`[12-background-derivative-fit-grid-rings-at-a-step]`.

---

## What shipped

**`T_Z_REPRESENTATION_VERSION` is 5 before and after.** `git diff --stat` touches
`docs/` and `prompts/` only; no `CosmologyModels/`, `ComputeTargets/`, `CosmologyConcepts/`,
`Datastore/` or `main.py` file is in the diff, and no test or fixture is either.

### 1. `docs/qcd-background-audit/grid_density_criterion.py` — new, the measurement

One command, **88.4 s**, no Ray and no datastore. Run twice on this tree; **every printed figure is
bit-identical between the two runs** except the wall clocks.

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py \
    --models QCDModel --k 1e5 --sections A B --json /tmp/density.json
```

Public names: `production_grids(cosmology)`; `class Case` (the oracle and the pre-grid closed form
for one (model, sector, $k$), with `.phi(u)`, `.dphi_du(u)`, `.d4phi_du4(u)`, `.score(u_fit, exact=,
away_only=)`); `criterion_shapes(case, u, breaks_u)`; `spacing_profile(kind, d1, d4, scale)`;
`march(u_lo, u_hi, u_prof, h_prof, h_min=, h_max=)`; `tune_to_target(...)`, `tune_to_count(...)`,
`uniform_to_target(...)`; `derivative_pad(u_grid)`; `decade_rows(...)`; `masked(u, breaks_u)`.
Constants: `SCORE_POINTS_PER_INTERVAL = 4`, `CUBIC_ERROR_CONST = 1/384`, `DERIV_STEP_U = 1e-3`,
`CROSSING_HALO_U = 0.15`, `CONSUMER_TARGET_RAD = 1e-06`, `ORACLE_NOISE_RAD = 1e-15`,
`H_MAX_FACTOR = 10.0`, `CAP_LADDER = (1, 2, 4, 8, 32)`.

Five sections: **A** the criterion's inputs are pre-grid (three checks); **B** what the shipped grid
delivers, per decade, twelve (model, sector, $k$), plus **B2** the four-way background/sample-grid
separation; **C** whether $h^4|\varphi''''|/384$ predicts; **D** two criteria costed per case, plus
**D2** one envelope grid for the whole production set over a cap ladder, each with its
`[03-derivative-pad-clamp-on-coarse-grids]` verdict; **E** the response grid and the $k\tau$
oscillation.

### 2. `docs/qcd-background-verification.md` — a dated §10 appended

**377 insertions, 0 deletions; 1,070 → 1,447 lines, and §§1–9 are untouched** (`CLAUDE.md`: additive). §10.0 is the recommendation and
the production-$x$ ceiling; 10.1 the oracle; 10.2 sub-question 1; 10.3 sub-question 2; 10.4 the
crossing exclusion and the background/grid coupling; 10.5 sub-question 3; 10.6 the two constraints
of prompt §2; 10.7 the $k\tau$ oscillation of prompt §3; 10.8 suites.

### 3. The board and the index

`IMPLEMENTATION_STATE.md` row 12, the progress line, §2's **G2** row, and §3 gain two issues;
`docs/OPEN_ISSUES.md` §1.7 gains the prompt-12 paragraph and the two rows, with the count 61 → 63.

---

## Deviations from the prompt

### 1. `STRUCTURALLY REQUIRED` — the reference is the phase-residual table, not $\varphi$ recovered from a stored $\theta$

Prompt §1 item 1 asks for "interpolation error of the consumer's $\varphi$ spline as a function of
local sample spacing, per decade". Taken literally — with $\varphi$ recovered from a stored
$\theta$, as prompts 10 and 11 did — **that measurement cannot be made**, because prompt §2 item 2
is true: the recovered $\varphi$ carries the $\epsilon k\tau$ rounding of
`[02-consumer-phi-below-the-storage-granularity]`, which is 3.05e-07 rad at $k=10^5$ and 9.15e-04 at
$3\times10^8$, while the interpolation error this prompt has to resolve runs from 5.9e-08 down to
7e-20. Ten of the twelve rows would have read "noise".

So the reference is $\rho$ itself, from
`ComputeTargets.phase_residual.build_phase_residual`'s `CumulativeTable`, whose `delta` does genuine
Gauss quadrature between arbitrary endpoints rather than interpolating
(`cumulative_table.py:360`). `PrimitivePhase`'s module docstring is the warrant — $\varphi$ is
$\rho$ plus the producer's constant offsets, and a constant is invisible to an interpolation error
— and the floor then enters where it belongs, as the line below which no grid decision is visible.
**It is cross-validated against prompt 11's own number** on a completely different path: the
configuration prompt 11 measured (background on the base grid, samples from the shipped grid) reads
1.4055e-05 → 3.9744e-07 here against its 1.3982e-05 → 4.9012e-07, agreeing to 0.5 % at the start
and 23 % at the end (§10.4).

### 2. `IMPLEMENTATION CHOICE` — a third script, and `build_cases` is *not* imported

The orchestrator's standing instruction was to reuse
`docs/qcd-background-audit/consumer_knot_scheme_scan.py` and `source_grid_consumer_check.py` rather
than build a third scorer. Deviation 1 makes that impossible in the literal sense: those two tools
*are* the stored-$\theta$ scorer, and this prompt's whole question lives below their floor. What is
reused is the **result**: prompt 11's number is the cross-check in §10.4, reproduced rather than
re-run.

The alternative considered was to keep `build_cases` for the producer runs and score against the
dense reference it builds. It was rejected on measurement: that reference is itself a producer run
at ten points per interval, so it carries the same $\epsilon k\tau$ granularity as the nodes, and
the ladder rows in prompt 10 §8.4 bottom out at 1.48 ulp for exactly that reason. The residual
table has no such floor; it is also ~60× cheaper, which is why twelve (model, sector, $k$) at five
sections cost 88 s where prompt 10's nine schemes at three wavenumbers cost 120 s.

### 3. `IMPLEMENTATION CHOICE` — the QCD background is built on the grid that ships, not on the base grid

Prompt 10's and prompt 11's harnesses build `qcd_model_with_tables(production_source_grid(...))` —
the 1,732-point base grid — and then vary only the *consumer's* sample set. This prompt builds the
model on the 1,773-point grid prompt 11 ships, because sub-question 1 asks what the **current** grid
delivers and `main.py` passes `z_sample=z_source_sample` to `BackgroundModel`.

That choice is what produced §10.4, and it is a finding rather than a preference: on QCD the
background's `epsilon`, `d_epsilon_dz` and `d2_epsilon_dz2` are stacked splines over a refinement of
the source grid, so which grid the background is built on changes $\omega_{\rm eff}$ and therefore
$\rho$. Both configurations are reported, in a four-way table, rather than one being chosen and the
other suppressed. **Nothing here says prompt 11 measured wrongly** — it measured the configuration
its own harness defines, and away from a crossing its grid is better on every row — only that the
production configuration is the fourth row of that table and not the second.

### 4. `IMPLEMENTATION CHOICE` — the criterion declines to answer within 0.15 in $u$ of a declared crossing

`C(z)` is built from derivatives of $H$, and $H$ **steps** at each declared crossing (audit §2), so
no derivative of it means anything within a few steps of one; and prompt 11 already puts a
straddling pair and a $\pm5$-interval refinement there on a rule rather than on a curvature. The
halo is `CROSSING_HALO_U = 0.15`, about 6.5 production intervals, comfortably outside prompt 11's
neighbourhood. Every table reports the crossing column **and** the away column, so nothing is
hidden; the criterion is fitted and scored on the away column alone.

The alternative was to let the criterion see the crossings, which was tried: the shape then asks for
unbounded density at three points and the envelope is decided entirely by a spline artefact.

### 5. `IMPLEMENTATION CHOICE` — a spacing cap, and a cap *ladder* rather than one number

No candidate is allowed to be more than `H_MAX_FACTOR = 10` times coarser than today's spacing, and
the envelope grids in §10.5 are reported over `CAP_LADDER = (1, 2, 4, 8, 32)`. The cap is not
cosmetic. `SLOPE` divides by $|\varphi'|$ and takes an unbounded step wherever $\varphi$ is
stationary, so without a cap the comparison would be about that accident. More importantly the
source grid carries the numeric ODE's samples, four `CumulativeTable`s' Gauss panels and
`QuadSourceIntegral`'s abscissae, **none of whose requirements this prompt measures** — so a
$\varphi$-only criterion is a lower bound on the density and the cap is where the unmeasured
consumers would have to be let in. Reporting one number would have concealed that.

### 6. `IMPLEMENTATION CHOICE` — `phi_fast`, and the control that licenses it

A candidate grid puts samples anywhere and an exact table call costs ~0.13 ms, so a ladder of thirty
candidates of $10^4$ samples would be minutes of quadrature per case. `Case._phi_fast` is an
order-5 interpolant of the exact values already computed at the nodes and the four-per-interval
scoring set (spacing $h/5$), used **only** for a candidate grid's own ordinates; the scoring set's
values are always the exact table's, and the shipped grid is scored with `exact=True`. Measured
against the table at points it was not built on: **1.6e-14 … 3.7e-13 rad away from a crossing**
across the twelve cases, against targets of 7.5e-09 and up — and 2.0e-09 … 3.4e-09 *at* a crossing,
where it smooths the kink, which is one more reason §10.5 scores away from crossings only.

Nothing is tagged `UNINTENDED DRIFT`. No README §2 design fact is touched: no production file is in
the diff at all, the only $u\to z$ conversions are sample *locations* and quadrature limits and
never an equality comparison (§2 (i)), no author convention is restated, and the equation of state
is not read except through the break points the cosmology already declares.

---

## Verification performed

Load average is quoted with every wall clock; this machine has been under erratic external load
(above 140 earlier in the campaign). The measurement itself rests on **no** clock: the two runs
below agree bit for bit on every number, and the only wall times quoted are the criterion's own
cost, which is reported beside the `compute_background` cost it must be compared against.

### 1. The whole measurement, twice

`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py`, load average
**4.63** and **7.26**, **88.4 s** and 88.9 s. `diff` of the two outputs from section B onwards: the
only differing lines are five "shape cost 0.01/0.02 s" wall clocks. Every measured figure in
`docs/qcd-background-verification.md` §10 is therefore reproducible to the digit.

### 2. Section A — the criterion is pre-grid (the prompt's third review criterion)

* **A1**, the closed form $\mathrm{d}\varphi/\mathrm{d}u = -(1+z)C/(\omega+\omega_0)$ against the
  residual table's own central difference, mid-band, twelve cases: **2.40e-11** (LambdaCDM $G_k$,
  $k=10^5$) to **1.24e-05** (LambdaCDM $T_k$, $3\times10^8$) relative, the large end being the
  difference's own noise where $\varphi' \sim 2.5\times10^{-8}$ over a step of $2\times10^{-4}$.
* **A2**, $\epsilon$ as the model supplies it against $\epsilon$ from a central difference of the
  **cosmology's** pointwise `Hubble` — which is what a criterion running before
  `populate_z_sample` would have to use — on 400 points over the production range, away from a
  declared crossing: **max 2.525e-09 / median 8.190e-10** (LambdaCDM) and **max 3.907e-09 / median
  8.091e-10** (QCD). Within 0.5 in $u$ of a crossing: **5.361e-04**.
* **A3**, the same at a ladder of offsets from each of the three declared crossings, on both
  backgrounds: the table in §10.4. `EOS_T_LO` — the kink where $g_s$ is continuous to 1.8e-11 — is
  the control and reads **1.6e-09 … 1.9e-09** at every offset on both grids.
* **Cost:** $5N$ closed-form evaluations, **0.01–0.27 s** for the production profile against
  **0.599 s** for a QCD `compute_background` (§4.2).

### 3. Section B — sub-question 1

The twelve-row table of §10.2. **Two rows miss their floor and they are the same row on both
models**: $T_k$ at $k=10^5$, **7.86 ulp** (LambdaCDM, 5.8576e-08 rad against a 7.4506e-09 floor, at
$z=1.3152\times10^{10}$) and **7.84 ulp** (QCD, 5.8437e-08 rad, at $z=1.4554\times10^{10}$) — both
in the **top decade of the band**, within a few grid intervals of horizon entry. Every other row is
inside its floor: the largest of the remaining ten is QCD $G_k$ at $k=10^5$ at **7.04e-02** of its
floor, and the smallest is LambdaCDM $G_k$ at $3\times10^8$ at **1.87e-12**.

Per decade the error falls about one order per decade downwards and spans **eight orders inside a
single band** (QCD $T_k$, $k=10^5$: 5.84e-08 at decade 10 to 3.33e-16 at decade −1) while $h_u$ is
constant to four digits over fourteen decades — and *falls* to 5.49e-03 below $z=1$, because
`populate_z_sample` is log-spaced in $z$ rather than in $1+z$, so the density is highest exactly
where the curvature is lowest. Headroom (floor / max-away) at the bottom of the LambdaCDM $G_k$
$3\times10^8$ band: **2.10e+19**.

**Where the criterion stops meaning anything** (the prompt's fourth review criterion): the floor
grows like $k$ and $\varphi$ falls like $1/k$, so the $T_k$ away/floor column reads **7.84** at
$k=10^5$, **1.62e-02** at $10^7$ and **1.92e-03** at $3\times10^8$. That is the same fact prompt 10
measured from the other side — $\varphi$ spans 6.0 ulp of the stored phase at $10^7$ and 2.0 ulp at
$3\times10^8$. **Every scale in this prompt is set at $k=10^5$; above $k\approx10^7$ no grid
decision is visible.**

### 4. Section B2 — the background/grid coupling

Four combinations, all six QCD cases; the LambdaCDM control is skipped because its two grids are
identical. QCD $T_k$ at $k=10^5$ (1 ulp = 7.4506e-09 rad):

| background on | samples from | $n$ | max | near a crossing | away |
|---|---|---|---|---|---|
| base | base | 1,117 | 1.4055e-05 | 1.4055e-05 | 1.0523e-07 |
| base | shipped | 1,145 | **3.9744e-07** | 3.9744e-07 | 5.8439e-08 |
| shipped | base | 1,117 | 1.1851e-04 | 1.1851e-04 | 5.9047e-08 |
| shipped | shipped | 1,145 | **2.4859e-05** | 2.4859e-05 | 5.8437e-08 |

The same ordering holds in all six: QCD $G_k$ $10^5$ 8.1208e-06 / 2.2855e-07 / 6.8532e-05 /
1.4356e-05; $G_k$ $10^7$ 7.4677e-05 / 2.2338e-06 / 6.2594e-04 / 2.1574e-04; $G_k$ $3\times10^8$
2.4515e-06 / 7.4601e-08 / 2.0963e-05 / 7.1558e-06; $T_k$ $10^7$ 1.7899e-04 / 4.9524e-06 /
1.2540e-03 / 4.4783e-04; $T_k$ $3\times10^8$ 5.0859e-06 / 1.5424e-07 / 3.8747e-05 / 1.3272e-05.
**The away column improves on every one of the six** when the background moves to the shipped grid
(e.g. $G_k$ $10^7$: 3.6721e-07 → 9.1102e-10).

### 5. Section C — sub-question 2, and whether the indicator earns its place

$h^4|\varphi''''|/384$ against the realised error, per production interval, away from a crossing and
above 1e-15 rad. **Transfer-function sector, both models, all three $k$: median 0.923–0.928, p10 to
p90 spread 1.0× over 517–551 intervals** — the textbook constant, no fitting, ±2 %. QCD $G_k$:
median 1.016–1.247, p90 3.2–10.4. LambdaCDM $G_k$: median 0.010, i.e. the indicator over-predicts
100×, which is the five-point stencil's roundoff where $\varphi'\sim10^{-8}$ and the divisor is
$2\delta^3 = 2\times10^{-9}$; it errs towards more samples, and on that row the true answer is "any
density will do" (headroom 8.6e+04 to 2.1e+19).

### 6. Section D / D2 — sub-question 3

Per case, `CURV` against the two uniform controls at the same target (§10.5's second table): the
two binding rows are $T_k$ at $k=10^5$, where **uniform in $\log_{10}z$ needs 1,898 (LambdaCDM) and
1,908 (QCD) band samples** against the 1,113 and 1,145 the grid has — today's
`source_samples_per_log10z = 100` would have to be about **167** to meet the floor uniformly — while
`CURV` needs **204** and **295**.

Accuracy at the shipped sample count: **2.1427e-12** (LambdaCDM $T_k$ $10^5$, against 5.8576e-08),
**1.0849e-11** (QCD $T_k$ $10^5$, against 2.4859e-05), **9.7801e-11** (QCD $T_k$ $10^7$, against
4.4783e-04) — ratios of 2.7e+04 to 4.6e+06.

One envelope grid for the whole production set, every case scored on it:

| | cap 1× | cap 2× | cap 4× | cap 8× | cap 32× |
|---|---|---|---|---|---|
| QCD `CURV`, samples (shipped 1,773) | **1,761** | **1,015** | 703 | 588 | 521 |
| worst row / its target | 0.14 | 0.21 | 1.65 ✗ | 0.92 | 4.17 ✗ |
| LambdaCDM `CURV`, samples (shipped 1,732) | **1,634** | **842** | 471 | 307 | 218 |
| worst row / its target | 0.34 | 1.09 ✗ | 5.09 ✗ | 1.06 ✗ | 1.35 ✗ |
| `[03-derivative-pad-clamp-on-coarse-grids]` | does not bind | **BINDS** | BINDS | BINDS | BINDS |

The worst row is $T_k$ at $k=10^5$ in nine of the ten columns. The ratios are non-monotone in the
cap because the march is a forward Euler and where a sample lands relative to the peak curvature
varies; the counts are the simple implementation's, not an optimal grid's, and the log says so
rather than smoothing it.

**`SLOPE` is refuted.** On the envelope at cap 1× it needs **114,281** samples (QCD) and **4,229**
(LambdaCDM), 64× and 2.4× *more* than the grids they replace, and above cap 4× it misses by up to
1,723× (QCD $G_k$, $3\times10^8$). Per case it fails outright at QCD $G_k$ $k=10^5$ and
$3\times10^8$ ("never meets the target at any scale") and needs 15,463 samples where `CURV` needs
295.

### 7. `[03-derivative-pad-clamp-on-coarse-grids]` — measured on every candidate grid

`derivative_pad()` transcribes `BackgroundModel.py:99-104`. On both shipped grids and on the cap-1×
criterion grids the first refined interval is **7.6773e-03** against the floor cap
$-\log(0.9)/12 = $ **8.7800e-03**, and the clamp **does not bind**. At cap 2× the first interval is
1.5355e-02, at 4× 3.0709e-02, at 8× 6.1419e-02 and at 32× 2.4567e-01, and the clamp **binds** in
every one. **The saving and the clamp are the same lever**: the whole of the criterion's saving is
coarsening at low $z$, which is precisely what raises the lowest interval past the cap. Nothing was
changed; the issue is unchanged and now has a number for the coarse grids it warned about.

### 8. Prompt §2's two constraints, and §3's two records

* **The response grid stays a decimation of the source grid**: a blind stride is a subset of any
  grid by construction, and `winnow(12)` was checked on all twenty candidate grids — `True` in all
  twenty. Its size scales with the source grid: QCD 147 → 85 → 59 → 49 → 44 across the ladder.
  **A 44-sample response grid over twenty decades is almost certainly not usable**, which is a
  second reason the recommendation stops at cap 1–2×; it is not a $\varphi$ question and this
  prompt does not answer it.
* **The large-$k$ floor** — §3 above and §10.6.
* **The $k\tau$ oscillation** (§10.7): $k\tau$ over the whole range is 1.3728e+09 / 1.3728e+11 /
  4.1184e+12 rad; the **median** response interval advances **185–296 rad at $k=10^5$** — 30 to 47
  complete cycles — and 5.6e+05 to 8.9e+05 rad at $3\times10^8$; four samples per cycle would need
  8.740e+08 to 2.622e+12 response samples, a shortfall of **5.9e+06× to 1.8e+10×**. The conclusion
  is not a grid size but that **no grid size exists**: an $\Omega_{\rm GW}$ post-processing step
  must get its RMS amplitude from an oscillatory quadrature, never from sampling.
* **The production-$x$ ceiling** (`docs/OPEN_ISSUES.md` §5) is stated in §10.0, the paragraph that
  carries the recommendation, and not in a footnote.

### 9. The tree

| suite | before (`4f725c0`) | after | command |
|---|---|---|---|
| `CosmologyModels/tests` | 30 OK | **30 OK** (0.576 s) | `discover -s CosmologyModels/tests -t .` |
| `ComputeTargets/tests` | 380 OK | **380 OK** (153.568 s) | `discover -s ComputeTargets/tests -t .` |
| `LiouvilleGreen/tests` | 143 OK (fast set) | **143 OK** (13.923 s) | every module except `test_3bessel_analytic` |

The `LiouvilleGreen` figure is the **fast set**, as prompts 02–08, 10 and 11 used
(`test_3bessel_analytic` excluded; the full set is 148 and prompt 09 ran it). No production file, no
test file and no fixture is in the diff, so "before" and "after" are the same tree outside `docs/`
and `prompts/`; the counts are quoted because README §5 rule 6 requires them. `black --check` is
clean on the one new file. `git status --porcelain` lists only the four files this prompt writes.

### 10. What was *not* run

**No pipeline run, and no grid was changed.** Every candidate grid in §10.5 exists only inside the
measurement script; none was built by `populate_z_sample`, none reached a datastore, and no
`BackgroundModel`, producer or consumer object was ever computed on one. The claim that a re-gridded
production run would behave as §10.5 predicts is an **inference from an interpolation measurement**,
and `docs/OPEN_ISSUES.md` §5's ceiling applies to it in full.

---

## Observations not acted on

1. **The QCD background's derivative fields are splined on the source grid, and they ring at a
   genuine step in $H$.** §10.4. Opened below as
   `[12-background-derivative-fit-grid-rings-at-a-step]`. Not fixed here: the repair is in
   `ComputeTargets/BackgroundModel.py`, this prompt may touch no production file, and the obvious
   repair — splitting `_build_derivative_fit_grid` at `_cosmology_break_points` and fitting one
   spline per branch, exactly as prompt 06 did for $F(u)$ — moves every stored QCD $\omega_{\rm eff}$
   and therefore every stored phase, which is a `T_Z_REPRESENTATION_VERSION` bump and a
   regeneration.

2. **On LambdaCDM the Green's-function sector needs no sample density at all.** Its $\varphi$ is the
   residual of an equation of state that is radiation to a part in $10^4$: the headroom is
   8.6e+04 at the worst decade and 2.1e+19 at the best, and eight uniform samples over the whole
   band meet the floor. Any criterion is dominated by the transfer-function sector and by QCD.
   Recorded because a reader seeing "164 samples" in §10.5 should know it is the *cap* speaking and
   not the curvature.

3. **`ComputeTargets/tests/wkb_reference.py::production_source_z_values` still hands out the base
   1,732-point grid.** Prompt 11's observation 3, unchanged and re-confirmed here: this prompt's
   `production_grids()` goes through `CosmologyConcepts.build_z_sample` and `main.py`'s own
   `cosmology_feature_redshifts` rather than through that helper, precisely so that the grid it
   measures is the one that ships. Whoever regenerates the QCD fixture next still has to decide
   whether that helper gains the protected set.

4. **The five-point stencil is roundoff-limited where $\varphi'$ is very small.** LambdaCDM $G_k$,
   §10.3: the indicator over-predicts by ~100×. It is conservative, it does not affect any
   recommendation, and either a larger `DERIV_STEP_U` or an analytic third derivative of
   $C/(\omega+\omega_0)$ would remove it. Not worth an issue.

5. **A response grid of 44 samples over twenty decades.** The coarse end of the cap ladder produces
   one. Nothing in this campaign has measured what the response grid's density has to be — §10.7
   establishes only that it cannot be set by the oscillation — so the ladder is reported and the
   coarse end is not recommended.

---

## State handed to the next prompt

**There is no next prompt: this is the last of twelve and the campaign closes here.** What follows
is the decision packet for the user, because prompt §4 says the recommendation has to survive the
campaign's close.

1. **The criterion, in one line.** $h(u)^4\,|\varphi''''(u)|/384 \le \varepsilon$ in
   $u=\log(1+z)$, with
   $\varphi'(u) = -(1+z)\,C(z)/\bigl(\omega(z,k)+\omega_0(z,k)\bigr)$ — the integrand of
   `ComputeTargets.phase_residual.phase_residual_integrand`, times $(1+z)$ — and $\varphi''''$ its
   third derivative by a five-point central stencil of step $10^{-3}$ in $u$. It needs $H$, $c_s^2$,
   their $z$-derivatives and $k$; nothing else, and no `BackgroundModel`. On `QCD_Cosmology` the
   $z$-derivatives are not cosmology methods and a pre-grid criterion must take them as finite
   differences of the cosmology's pointwise `Hubble` and `wPerturbations` — which agrees with the
   grid-splined ones to **3.9e-09 relative away from a declared crossing**. Cost $5N$ closed-form
   evaluations, **0.01–0.27 s** against `compute_background`'s 0.599 s.

2. **What it is worth, in numbers the user can weigh.** One universal grid, every (sector, $k$)
   scored on it, against the shipped 1,773 (QCD) and 1,732 (LambdaCDM):

   | | shipped | criterion, cap 1× | criterion, cap 2× |
   |---|---|---|---|
   | QCD samples | 1,773 | **1,761** | **1,015** (1.75× fewer) |
   | worst row against its target | **7.84×** (miss) | **0.14×** | **0.21×** |
   | LambdaCDM samples | 1,732 | **1,634** | **842** (2.06× fewer) |
   | worst row against its target | **7.86×** (miss) | **0.34×** | 1.09× (a 9 % miss) |
   | `[03-derivative-pad-clamp-on-coarse-grids]` | does not bind | does not bind | **binds** |

   **The cap-1× column is the recommendation**: at the same cost as today it takes the one place the
   grid misses from 7.8× its floor to 0.2×, and it triggers nothing. The cap-2× column is the
   saving, and it costs the derivative-pad clamp.

3. **What it would cost to take.** A full regeneration of eight stored object types — the bill is
   log 11 §5 and `docs/qcd-background-verification.md` §9.6: 15,020 objects and 6 m 35 s plus an
   unfinished `QuadSourceIntegral` stage for a measured 5×5-wavenumber single-model run, with
   production ×10 in each wavenumber sample, ×85 on `QuadSource` and of order ×850 on
   `QuadSourceIntegral`, over two models. Prompt 11's grid-tag digest means the invalidation is
   detected rather than silent.

4. **What is still unmeasured, and bounds all of it.** (i) The source grid carries the numeric
   ODE's samples, four `CumulativeTable`s' Gauss panels and `QuadSourceIntegral`'s abscissae; **none
   of those requirements is measured**, so a $\varphi$-only criterion is a lower bound on the
   density and never an upper one — that is what the cap is for. (ii) `docs/OPEN_ISSUES.md` §5: no
   verification run has reached production $x$, and **no pipeline has been run on any grid in this
   section, including the one that ships**. (iii) Above $k\approx10^7$ nothing here is visible:
   $\varphi$ spans 6.0 ulp of the stored phase at $10^7$ and 2.0 ulp at $3\times10^8$, so a
   criterion derived there would be fitting rounding. (iv) The response grid's own requirement:
   §10.7 shows only that it cannot be set by the $k\tau$ oscillation, which is 30 to 47 cycles per
   *median* response interval at the smallest production $k$ and $10^6$–$10^{10}$ times out of
   reach.

5. **The crossing residue is not a density question and no criterion reaches it.**
   `[12-background-derivative-fit-grid-rings-at-a-step]`. Until that is repaired, refining the grid
   at a declared crossing makes the background's own spline artefact **narrower faster than it makes
   it smaller**, and the production configuration — background rebuilt on the refined grid — reads
   2.4859e-05 rad at QCD $T_k$, $k=10^5$ where prompt 11's configuration reads 3.9744e-07. Anyone
   re-measuring prompt 11's rows must say which of the four combinations of §10.4 they are in.

6. **Reproduction, one command, 88 s, no Ray and no datastore:**
   ```bash
   PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py
   ```
   `--models`, `--k`, `--sectors`, `--sections` and `--json` narrow it. Prompts 10's and 11's
   scripts are untouched and their commands are unchanged. The QCD reference fixture is untouched by
   this commit.

7. **Two issues open, none closes.** `[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]`
   (the recommendation itself, so it outlives the campaign) and
   `[12-background-derivative-fit-grid-rings-at-a-step]`. `docs/OPEN_ISSUES.md` goes 61 → 63.
   **G2 is answered, not implemented**, which is what prompt §4 asks for: `COMPLETE` is when the
   measurement is made and the recommendation is stated, not when a grid changes.
