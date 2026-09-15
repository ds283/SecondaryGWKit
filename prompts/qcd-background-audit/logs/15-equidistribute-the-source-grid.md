# Log 15 — Build the source grid by the measured criterion, at the cap that never coarsens

**Prompt:** prompts/qcd-background-audit/15-equidistribute-the-source-grid.md
**Commit:** *(this commit)* — "Equidistribute the source grid on the phase residual's curvature"
**Model:** Claude Opus 5
**Date:** 2026-09-15
**Result:** COMPLETE WITH DEVIATIONS

> **Read this first, because it is where the result is and not a footnote** (`docs/OPEN_ISSUES.md`
> §5). **No pipeline has ever been run on any grid in this line of work, including the one that
> ships**, and no verification run has reached production $x$: `source-remediation`'s run A used
> `zend = 1e7` and reached $x = 4.63\times10^5$ against the $x\sim1.4\times10^7$ production reaches
> at its largest $k$. This prompt changes the production source grid on the strength of a
> **Gauss-quadrature oracle** — the phase residual read off a `CumulativeTable` whose `delta` does
> genuine quadrature between arbitrary endpoints — and not on the strength of a pipeline run. That
> a re-gridded run behaves as the numbers below predict is an inference from an interpolation
> measurement, and it is the only kind of evidence this campaign has ever had for the grid.

---

## What shipped

`T_Z_REPRESENTATION_VERSION` is **6** before and after — no `CosmologyModels/` file is in the
diff and **no background value moves**; what moves is which redshifts are sampled.
`SOURCE_GRID_CONSTRUCTION_VERSION` goes **1 → 2**, which is exactly the change that constant
exists to record.

### `CosmologyConcepts/wavenumber.py`

* `SOURCE_GRID_CONSTRUCTION_VERSION` **1 → 2**, with a version-2 row added to the table in its
  comment: version 1 plus a base density set by the curvature criterion, reaching `build_z_sample`
  as a `spacing` profile so that no cosmology object reaches the module.
* **New: `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0`** — the cap, "never coarser than the uniform
  lattice puts there, anywhere", with the declined 1.75×/2.06× saving, the four things the
  criterion cannot see, and the reason a later prompt cannot move it by changing a number (raising
  it above 1 additionally needs a code path that *drops* base samples, which does not exist).
* New: `SOURCE_GRID_MAX_REFINEMENT = 32` (refusal rather than a grid of unknown size; the largest
  the criterion asks for on the production envelope is 4), `SOURCE_GRID_CUBIC_ERROR_CONST = 1/384`,
  `SOURCE_GRID_CURVATURE_STEP_U = 1e-3`, `SOURCE_GRID_CURVATURE_FD_STEP_U = 1e-4`,
  `SOURCE_GRID_CROSSING_MASK_U = 6e-3`, `SOURCE_GRID_CONSUMER_TARGET_RAD = 1e-6`,
  `SOURCE_GRID_SPLINE_EDGE_INTERVALS = 3`, `SOURCE_GRID_SPLINE_EDGE_FACTOR = 10.0`, and the
  private `_SPACING_UNCONSTRAINED = 1e6`.
* `build_z_sample(..., spacing: Optional[Tuple[Sequence[float], Sequence[float]]] = None)`. The
  early return is now conditioned on `spacing is None` as well as on nothing being declared, and a
  new final block subdivides each base interval by
  `m = max(1, ceil(h_base / min(h_profile at both ends, SOURCE_GRID_MAX_SPACING_FACTOR * h_base)))`,
  placing the interior points in the base grid's **own** coordinate (`t`, i.e. $\log_{10} z$) and
  guarding each with the same `_admits` mesh guards prompt 11 uses. It runs **after** the
  straddling pairs, the feature redshifts and the break neighbourhoods, so every sample prompt 11
  placed is placed identically and the new grid is a strict **superset** of the old one.
* `populate_z_sample` / `populate_source_grid` forward `spacing` unchanged (`**kwargs`).

### `CosmologyConcepts/__init__.py`

Re-exports the eight new public constants.

### `main.py` (grid-construction and tagging hunk only)

* **New: `pre_grid_background_proxy(cosmology)`** — a duck-typed stand-in for `BackgroundModel`
  exposing `.cosmology` and `.functions.{Hubble, epsilon, d_epsilon_dz, wPerturbations,
  d_wPerturbations_dz}`, every one of them from the cosmology pointwise: closed forms where the
  cosmology has them (`LambdaCDM` has all four derivatives), central differences in
  $u=\log(1+z)$ where it does not (`QCD_Cosmology` has none). Every accessor memoised on its
  argument, so the fifty-wavenumber envelope costs the cosmology no more than the first case does.
* **New: `source_grid_spacing_profile(cosmology, base_z_values, k_values, sectors=("Gk","Tk"))`**
  → `(u_profile, h_profile)`. The envelope over every wavenumber the run serves and both sectors
  of $h = \big(\varepsilon/(|\varphi''''|/384)\big)^{1/4}$, with $\varphi'$ from
  `ComputeTargets.phase_residual.phase_residual_integrand` **used unmodified**, $\varphi''''$ by
  the five-point stencil, the band from `residual_node_range` (the producers' own rule), the
  declared crossings masked out and log-interpolated across, and
  $\varepsilon = \min(\mathrm{ulp}(k\tau_{\rm span}), 10^{-6})$ per case — the span by trapezium in
  $u$, which sets an *ulp* and so needs no converged quadrature.
* `run_pipeline` builds the base lattice explicitly, times the criterion, passes `spacing=` to
  `populate_source_grid`, and prints the sample count against what a uniform
  `samples_per_log10z` would have given. `build_z_sample`, `phase_residual_integrand`,
  `residual_node_range` and the eight constants are imported.

### Tests — `ComputeTargets/tests/test_source_grid.py`

21 → **36** test methods. `TestACosmologyThatDeclaresNothingIsUntouched` is renamed
`TestACosmologyThatDeclaresNothingDeclaresNothing` and **updated, not deleted** (prompt §3 item 3):
`build_z_sample` with no `spacing` still reproduces `numpy.logspace` bit for bit and a smooth
cosmology still gets no break points, but its *production* grid is no longer the bare lattice.
Two new classes: `TestTheCurvatureCriterionSetsTheBaseDensity` (11 tests — the cap asserted
directly against the base lattice interval by interval, the prompt-11 grid surviving element for
element, the protected set still straddled, the descending/distinct/datastore-resolvable
invariants, the derivative-pad clamp, the response grid as a decimation and its size, and the two
refusals) and `TestThePreGridCriterionNeedsNoBackgroundModel` (3 tests). The construction-version
test becomes `test_version_2_is_prompt_15s_construction` and the production-grid test records
three grids instead of two.

### New: `docs/qcd-background-audit/equidistributed_grid_check.py`

One command, no Ray, no datastore. Imports `Case`, `derivative_pad` and the constants from
`grid_density_criterion.py` — the tool is reused, not rewritten — and adds the grid comparison:
**A** the grids and the cap, **B** every (model, sector, $k$) row in its own production
configuration before and after, **C** §10.4's four-way table.

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/equidistributed_grid_check.py
```

---

## Deviations from the prompt

### 1. The cap is applied *locally*, and the sample counts are therefore larger, not smaller — STRUCTURALLY REQUIRED

**What the prompt assumed.** §2 item 2 and §3 test 1 say the cap is "never coarser than today,
anywhere", and §3 test 1 states it as "no interval on the new grid exceeds the corresponding
interval on the shipped grid... the one that must not be weakened". §4 then targets **QCD ~1,761**
and **LambdaCDM ~1,634**, from §10.5's cap-1× column.

**What is actually there.** Those two cannot both hold, and the arithmetic says which gives.
§10.5's cap is a **scalar**: `h_today = np.max(np.diff(u_all)) = 2.3032e-02`, the base grid's
spacing above $z\approx10$. But the base grid is uniform in $\log_{10} z$, **not** in
$u=\log(1+z)$, so its spacing in $u$ collapses towards low redshift — 5.4866e-03 in the $z\in[1,10)$
decade and **2.1159e-03** in the bottom interval at $z=0.1$, eleven times finer than the scalar cap.
A march capped at 2.3032e-02 therefore *coarsens* the bottom two decades by up to 10.9×. That is
where LambdaCDM's 1,634 — which is **fewer** than the 1,732 it is supposed never to undercut —
comes from, and QCD's 1,761 against 1,773 likewise. §10.0 statement 3's "the saving is 1 %" at the
1× cap is that same coarsening, counted.

**What was done instead.** `SOURCE_GRID_MAX_SPACING_FACTOR` multiplies the base grid's **own local
interval**, so the cap means what §3 test 1 says it means at every redshift rather than only above
$z\approx10$; and the construction subdivides base intervals by an integer rather than marching, so
the grid is a strict superset of the one it replaces and "never coarser" is a property of the
construction rather than a number to be checked afterwards. The measured consequence is that the
grids are **larger** than §4's targets on both models — QCD **1,996** against ~1,761, LambdaCDM
**1,778** against ~1,634 — which is the direction the user's instruction points ("I would much
rather have the grid slightly more dense than needed, than have it underdense") and the direction
§3 test 1 fixes. §4's two sample-count rows are therefore **missed, deliberately and in the safe
direction**, and are recorded as a miss rather than argued away.

### 2. The grid is built by subdividing the base lattice, not by marching `du/di = h(u)` — STRUCTURALLY REQUIRED

A consequence of deviation 1: a march produces a new lattice with none of the old samples in it,
which cannot be "never coarser than the old one, anywhere" unless the old one happens to be uniform
in $u$. Subdividing has two further properties the prompt asks for and a march does not give:
prompt 11's straddling pairs, feature redshifts and ±5-interval neighbourhoods survive **element
for element** (§2 item 3: "do not fold one into the other"), and the forward-Euler lag
`grid_density_criterion.march` documents — a step taken with $h$ evaluated at the interval's start,
into a region where $h$ is collapsing — does not arise, because each interval takes the smaller of
the two endpoint requirements.

### 3. The criterion needed an end-condition term, and the prompt did not name one — IMPLEMENTATION CHOICE

**The problem, measured.** At $\varepsilon$ = the per-case target the criterion does **not**
recover the two rows §10.2 records as a miss: the permitted spacing at the binding interval comes
out at 2.398e-02 against a base 2.3032e-02, i.e. "the shipped grid is already adequate", while the
realised error there is 7.86× the floor. Taking §10.3's ratio apart per interval says why — it is
not the criterion, it is the **end condition**. Realised/predicted, by distance from the band edge:

| case | 1st interval | 2nd | 3rd | interior |
|---|---|---|---|---|
| LambdaCDM $T_k$ $k=10^5$ | **9.897** | 4.292 | 1.230 | p50 0.923, p90 0.934 |
| QCD $T_k$ $k=10^5$ | **9.792** | 4.189 | ~1 | p50 0.928, p90 0.941 |
| QCD $G_k$ $k=10^5$ | **9.327** | 4.546 | 1.379 | p50 1.223 |

`scipy.interpolate.make_interp_spline` closes a cubic with a not-a-knot condition, whose error
constant in the outermost intervals is much larger than the interior $1/384$ — and §10.2's misses
are in the **topmost interval of the band**, a few grid intervals inside horizon entry, which is
exactly there. Prompt 12 absorbed it silently: its `tune_to_target` bisects a global scale against
the oracle, and the scale it found for LambdaCDM $T_k$ at $k=10^5$ is **8.053e-10** against that
case's target of 7.4506e-09 — a factor of **9.25**, which is this effect and nothing else. A
production criterion cannot do that, because the bisection needs the oracle and the oracle needs a
`BackgroundModel`.

**The two candidates, and the measurement between them.** (i) A **global safety factor** of 10 on
every target: refines **560** of the 1,731 production intervals on QCD and adds **614** samples,
and most of what it refines is QCD $G_k$ at $k=10^5$ over a wide stretch where the realised error
is $10^{-12}$ against a $2.4\times10^{-7}$ floor — density bought for the stencil's roundoff.
(ii) An **edge factor** applied to the outermost `SOURCE_GRID_SPLINE_EDGE_INTERVALS = 3` intervals
of each band: refines **226** intervals and adds **242**. (ii) was taken, because the same
measurement that shows the edges are wrong shows the interior constant is right, and the table
above is transcribed into the constant's comment so a later reader can disagree on the merits
without re-deriving it. A later prompt that wants the belt-and-braces version raises
`SOURCE_GRID_SPLINE_EDGE_INTERVALS`; the cost of doing so is recorded here (EDGE = 2 / 3 / 5 gives
QCD +208 / +242 / +316 samples).

**Residual risk, stated.** The band is `residual_node_range`'s, and a production consumer's spline
ends at its own anchor, which lies *inside* that band. The edge factor therefore protects the
worst place (where $|\varphi''''|$ is largest) and not every place a real spline ends. That is the
one thing in this prompt a global safety factor would have covered and this does not; it is
recorded as an observation below rather than papered over.

### 4. The envelope is over the run's fifty wavenumbers, not §10.5's three — IMPLEMENTATION CHOICE

§10.5 builds its envelope from the three reference wavenumbers. `main.py` builds **one universal
grid** serving fifty (`main.py:3292`, both the source and the response array), each with its own
Liouville–Green band and therefore its own band-top edge, so `source_grid_spacing_profile` takes
the wavenumbers the run actually serves — `[float(k_exit.k) for k_exit in full_k_exit_times]`.
Measured difference on QCD: **+53** samples over prompt 11's grid with three wavenumbers against
**+223** with fifty, and the criterion costs **0.50 s** against **1.82 s**. Using three would leave
forty-seven band tops unrefined; the alternative — refine every band top that any $k$ could produce
— is what fifty already is.

### 5. The criterion lives in `main.py`, not in `CosmologyConcepts/wavenumber.py` — IMPLEMENTATION CHOICE

The criterion needs `Hubble` and `wPerturbations` and reuses `ComputeTargets.phase_residual`, while
`ComputeTargets` imports `CosmologyConcepts`; a module-scope import the other way is circular and a
lazy one inverts the layering for good. Putting it in `main.py` beside `cosmology_feature_redshifts`
keeps prompt 11's design statement literally true — *"no cosmology object reaches this function and
no equation-of-state module is imported here"*, now asserted by a test on the module's own source —
and `build_z_sample` takes two arrays. The **constants** stay in `wavenumber.py` with the
construction version they identify. The alternatives were a lazy import in `wavenumber.py`
(rejected: the layering) and a new module under `ComputeTargets/` (rejected: not in the prompt's
file list, and `load_main_py_functions` already gives the tests an executable handle on `main.py`'s
policy functions, which is prompt 11's precedent).

### 6. A new scoring script rather than an edit to `grid_density_criterion.py` — IMPLEMENTATION CHOICE

That file is not in the prompt's "files you may touch" list and the prompt calls it *"the tool, not
a starting point to rewrite"*. `equidistributed_grid_check.py` **imports** `Case`, `derivative_pad`,
`K_VALUES`, `MODEL_KEYS` and `SECTORS` from it and adds only what prompt 15 needs: three grids
instead of two, and each scored in its own production configuration. Adding a measurement script
under `docs/qcd-background-audit/` is what prompts 09, 10, 11 and 12 each did.

### 7. The prompt-11 bit-identity test was updated rather than deleted — STRUCTURALLY REQUIRED (the prompt asks for this tag)

§3 item 3. LambdaCDM's *production* grid changes here, because the density is not a question about
the equation of state — §10.2 measures its $T_k$ row at $k=10^5$ missing by **7.86×**, the same
miss in the same place as QCD's 7.84×. What is asserted instead is the structural half:
`build_z_sample` with no `spacing` still reproduces `numpy.logspace` element for element, a smooth
cosmology gets no protected points and no break neighbourhoods, `build_z_sample` has no `cosmology`
parameter, and `CosmologyConcepts/wavenumber.py` contains no reference to `CosmologyModels`.

---

## Verification performed

Everything below was **run**, not reasoned about, except where it says otherwise. Reproduction:

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/equidistributed_grid_check.py
```

### 1. The grids, the cap, and the two constraints (§3 items 1, 4, 5)

| | LambdaCDM | QCD |
|---|---|---|
| base lattice | 1,732 (`0960e169`) | 1,732 |
| prompt 11/14 | 1,732 (`0960e169`) | 1,773 (`303f9ce7`) |
| **prompt 15** | **1,778** (`60a3205a`) | **1,996** (`a2c32f67`) |
| widest interval **against the base lattice** | **1.0000×** | **1.0000×** |
| the shipped grid is a subset of it | **True** | **True** |
| protected points | 0 → 0 | 8 → 8, all present, still straddled |
| strictly descending, distinct | True | True |
| closest approach, relative in $(1+z)$ | 2.1181e-03 | 1.0280e-03 |
| lowest interval in $u$ | **2.1159e-03** | **2.1159e-03** |
| response grid (stride 12, `protect=`) | 145 → **149** | 156 → **175** |
| response grid a subset of the source grid | **True** | **True** |
| criterion cost, 50 wavenumbers × 2 sectors | 0.89–1.03 s | 1.76–1.90 s |

The cap row is the user's decision measured directly: **no interval of either production grid is
wider than the interval the uniform 100-per-decade lattice puts at the same place**, and the widest
is exactly 1.0000× because the bottom interval is untouched. `[03-derivative-pad-clamp-on-coarse-grids]`
**does not bind**: `h_lo = 7.0529e-04` on both grids (first 7.0529e-04, floor cap 8.7800e-03), and
the assertion in the tree is on the raw interval, 2.1159e-03 against $-\log(0.9)/12 =
8.7800\times10^{-3}$. §10.6's warning that the ladder's coarse end reaches a 44-sample response grid
is on the far side of the ladder from here: **175 and 149**, both above the 156/145 they replace.

### 2. Accuracy, every row in its own production configuration (§3 item 2)

Background rebuilt on the grid the run samples, samples taken from it — §10.4's row 4, the
configuration prompt 11 fell short of. "away" excludes 0.15 in $u$ either side of a declared
crossing; "floor" is 1 ulp of the $k\tau$ span.

| model | sector | $k$ | away, `HEAD~1` | away, `HEAD` | floor | before/floor | **after/floor** |
|---|---|---|---|---|---|---|---|
| LambdaCDM | $G_k$ | 1e5 | 2.7680e-12 | 2.7680e-12 | 2.3842e-07 | 0.00 | 0.00 |
| LambdaCDM | $G_k$ | 1e7 | 2.7458e-14 | 2.7458e-14 | 1.5259e-05 | 0.00 | 0.00 |
| LambdaCDM | $G_k$ | 3e8 | 9.1527e-16 | 9.1527e-16 | 4.8828e-04 | 0.00 | 0.00 |
| LambdaCDM | $T_k$ | 1e5 | 5.8576e-08 | **5.1164e-09** | 7.4506e-09 | **7.86 ✗** | **0.69** |
| LambdaCDM | $T_k$ | 1e7 | 5.9076e-08 | 5.9076e-08 | 9.5367e-07 | 0.06 | 0.06 |
| LambdaCDM | $T_k$ | 3e8 | 5.3192e-08 | 5.3192e-08 | 3.0518e-05 | 0.00 | 0.00 |
| QCD | $G_k$ | 1e5 | 1.6782e-08 | **1.4078e-09** | 2.3842e-07 | 0.07 | 0.01 |
| QCD | $G_k$ | 1e7 | 9.1102e-10 | 1.1519e-09 | 1.5259e-05 | 0.00 | 0.00 |
| QCD | $G_k$ | 3e8 | 5.1456e-10 | 6.2029e-10 | 4.8828e-04 | 0.00 | 0.00 |
| QCD | $T_k$ | 1e5 | 5.8437e-08 | **8.4161e-10** | 7.4506e-09 | **7.84 ✗** | **0.11** |
| QCD | $T_k$ | 1e7 | 1.5453e-08 | 7.9051e-09 | 9.5367e-07 | 0.02 | 0.01 |
| QCD | $T_k$ | 3e8 | 5.8675e-08 | 2.8478e-08 | 3.0518e-05 | 0.00 | 0.00 |

**The two rows that missed now pass**, 7.86 → **0.69** and 7.84 → **0.11**, and the "before" column
is the measurement on `HEAD~1`'s grid, taken by the same script in the same run — which is prompt
§3 item 2's "show it fails on `HEAD~1`". Ten of twelve rows improve or are unchanged. **Two move
the wrong way and are recorded, not argued away**: QCD $G_k$ at $10^7$ 9.1102e-10 → 1.1519e-09
(1.26×) and at $3\times10^8$ 5.1456e-10 → 6.2029e-10 (1.21×), both at **0.00** of their floors
(7.5e-05 and 1.3e-06 ulp) and both the background's own derivative fields responding to a finer
lattice, which is §10.4's mechanism seen where it costs nothing.

### 3. The four-way table (§10.4), QCD and LambdaCDM $T_k$ at $k=10^5$

| model | background on | samples from | $n$ | max | near a crossing | away |
|---|---|---|---|---|---|---|
| LambdaCDM | shipped | shipped | 1,113 | 5.8576e-08 | — | 5.8576e-08 |
| LambdaCDM | shipped | **equi** | 1,116 | 5.1164e-09 | — | 5.1164e-09 |
| LambdaCDM | **equi** | shipped | 1,113 | 5.8576e-08 | — | 5.8576e-08 |
| LambdaCDM | **equi** | **equi** | 1,116 | **5.1164e-09** | — | **5.1164e-09** |
| QCD | shipped | shipped | 1,145 | 5.8437e-08 | 1.0024e-08 | 5.8437e-08 |
| QCD | shipped | **equi** | 1,165 | 1.0024e-08 | 1.0024e-08 | 6.5546e-10 |
| QCD | **equi** | shipped | 1,145 | 5.8446e-08 | 1.0024e-08 | 5.8446e-08 |
| QCD | **equi** | **equi** | 1,165 | **1.0024e-08** | 1.0024e-08 | **8.4161e-10** |

**The trap prompt 11 fell into is gone, and prompt 13 is why.** In §10.4 the four cells spanned
3.97e-07 to 1.19e-04 — a factor of 300 — because the background's derivative fields were splined
across the step; here the two background columns agree to four digits on LambdaCDM and to 1.5e-04
relative on QCD, so the grid has become a sample set and nothing more. What is left at the crossing
is prompt 11's neighbourhood, unchanged by the density: **1.0024e-08 rad, 1.35 ulp**, against the
1e-06 rad consumer target.

### 4. Log 13 §1's three-crossing row, re-taken on the candidate grid

`[13-segmenting-costs-accuracy-on-a-grid-that-does-not-resolve-the-crossing]` asks whoever takes
prompt 12's recommendation to re-take it. `epsilon` from the model against a central difference of
the cosmology's own `Hubble`, worst over $\mathrm{d}u = \pm0.005, \pm0.01, \pm0.02$:

| crossing | base 1,732 | shipped 1,773 | **equidistributed 1,996** |
|---|---|---|---|
| `T_LO`, $u=17.565806942$ | 4.6082e-08 | 1.6896e-09 | **1.6896e-09** |
| `EOS_T_LO`, $u=23.197460553$ | 8.1901e-08 | 1.9824e-09 | **1.9824e-09** |
| `T_120_MEV`, $u=27.485391822$ | 4.3338e-06 | 4.0481e-09 | **4.0453e-09** |

Identical to the shipped grid at every crossing, one of them marginally better. The concern that
entry raises — that a cap coarser than the base grid would put a segment edge on a lattice that
does not resolve the crossing — **cannot arise at this cap**, because the grid is a strict superset
of the one the row was measured on. The entry is narrowed accordingly.

### 5. The QCD reference fixture (§2 item 6)

Regenerated with prompt 02's generator in this commit:
`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py`.

**Largest relative move per key: none — all 12 science keys are bit-identical and nothing was
written** (178.2 s: build 0.608 s, reference quadrature 177.6 s;
`git status --porcelain ComputeTargets/tests/wkb_reference_data.json` empty before and after; the
keys are `tau_minus_top`, `cs_tau_minus_top`, `friction_F_minus_top`, `rho_G`, `rho_T`,
`rho_anchor_z`, `primitives_at_rho_anchor`, `short_baseline`, `reference_floor`, `grid`, `z_top`,
`checkpoints`, every one of them reported "unchanged"). This is the expected answer and the reason
is structural, not luck: the generator builds
its model on `ComputeTargets/tests/wkb_reference.production_source_z_values`, which is the **bare
`logspace`** and has been since the campaign began — prompt 11 did not change it either. Prompt 15
changes the grid `main.py` builds, not the grid the test tree scores against. Under prompt 04 §2
item 5's rule: **no tolerance had to loosen, and none had to tighten**; zero findings on this
account.

### 6. Suites

| suite | before (`96abdfe`) | after | command |
|---|---|---|---|
| `CosmologyModels/tests` | 30 OK | **30 OK** (0.569 s) | `discover -s CosmologyModels/tests -t .` |
| `ComputeTargets/tests` | 424 OK (log 14) | **439 OK** (159.4 s) | `discover -s ComputeTargets/tests -t .` |
| `LiouvilleGreen/tests` | 143 OK (fast set) | **143 OK** (14.2 s) | every module except `test_3bessel_analytic` |

No count falls. The 15 are `test_source_grid.py` 21 → 36 test methods. The `LiouvilleGreen` figure
is the **fast** set (prompts 02–08's convention), not the full 148 prompts 13 and 14 ran; nothing
in this commit reaches that package.

`black` is clean on all five files in the diff (`main.py`, `CosmologyConcepts/wavenumber.py`,
`CosmologyConcepts/__init__.py`, `ComputeTargets/tests/test_source_grid.py`,
`docs/qcd-background-audit/equidistributed_grid_check.py`). It is **not** clean on the tree as a
whole — 54 files would be reformatted by `black 25.1.0` — and that is pre-existing: the working
tree was clean at `96abdfe` and none of the 54 is in this diff.

### 7. Acceptance table (§4)

| Quantity | Shipped | Target | **Measured** |
|---|---|---|---|
| QCD source samples | 1,773 | ~1,761, none coarser anywhere | **1,996**, none coarser anywhere — **missed, in the safe direction** (deviation 1) |
| LambdaCDM source samples | 1,732 | ~1,634, none coarser anywhere | **1,778**, none coarser anywhere — **missed, in the safe direction** (deviation 1) |
| Worst row against its floor | 7.84× (miss) | ≤ 1×, measured 0.14× | **0.69×** (LambdaCDM $T_k$ $10^5$); QCD's own worst is **0.11×** |
| `[03-derivative-pad-clamp-…]` | does not bind | still does not bind | **does not bind** (2.1159e-03 against 8.7800e-03) |
| Response grid | 156 | quoted, still a subset | **175** (QCD), **149** (LambdaCDM), both subsets |
| Source-grid construction version | 1 | 2 | **2** |

---

## Observations not acted on

1. **The edge factor protects the band's edge, not every edge a real consumer spline has.**
   Deviation 3's residual risk. `residual_node_range`'s band top is where the Liouville–Green
   frequency still keeps half its leading term; a production `PrimitivePhase` object's spline ends
   at that object's own anchor, three e-folds inside the horizon and therefore *inside* the band,
   where $|\varphi''''|$ is smaller but the not-a-knot amplification is the same ~10×. The
   criterion guarantees $C h^4|\varphi''''| \le \varepsilon$ there, so such an end carries up to
   $10\varepsilon$. Nothing in the tree measures it, because every measurement in §10 and here
   scores the band. Opened below as
   `[15-the-edge-factor-is-applied-at-the-band-edge-not-at-the-consumers-own-end]`.

2. **The grid now depends on the wavenumber sample.** Two runs with different `--source-k-samples`
   get different grids and therefore different digests, which is correct and is what prompt 14's
   digest exists to record — but it is a new coupling: the grid was previously a pure function of
   $(z_{\rm init}, z_{\rm end}, \texttt{samples\_per\_log10z})$, and `main.py` hard-codes
   `NUMBER_SOURCE_K_VALUES = 50`, so nothing announces the dependence to a reader of the command
   line. No tag records the wavenumber set. Opened below as
   `[15-the-grid-now-depends-on-the-wavenumber-sample-and-no-tag-says-so]`.

3. **`docs/qcd-background-verification.md` was not extended.** Prompts 09–12 each appended a dated
   section to it; prompt 15's file list does not include it and its §5 asks for the measurements in
   this log, so they are here. A later prompt with that file in scope may want to lift §§1–4 above
   into a §11, which would be additive in `CLAUDE.md`'s sense.

4. **`[13-crossing-neighbourhood-refinement-was-sized-at-k-1e5]` is still open and still belongs to
   the grid.** Prompt 15 changes the *base* density and leaves `SOURCE_GRID_BREAK_HALF_WIDTH = 5`
   and `SOURCE_GRID_BREAK_REFINEMENT = 2` exactly where prompt 10 put them, as §2 item 3 requires.
   The QCD $T_k$ $k=10^7$ figure that entry names is a crossing-neighbourhood figure and this
   commit does not move it; the density criterion cannot, because the crossing neighbourhoods are
   masked out of it by construction.

5. **The 1.75×/2.06× saving is still available and still declined.** It is now a deliberate act in
   code — `SOURCE_GRID_MAX_SPACING_FACTOR` above 1.0 needs a coarsening path that does not exist —
   rather than a number nobody wrote down. If the decision is ever revisited, §10.5's ladder is the
   measurement and `[13-segmenting-costs-accuracy-on-a-grid-that-does-not-resolve-the-crossing]` is
   the constraint it does not carry.

---

## State handed to the next prompt

* **`T_Z_REPRESENTATION_VERSION = 6`**, unchanged by this commit and by prompts 13 and 14.
  **`SOURCE_GRID_CONSTRUCTION_VERSION = 2`** — bumped here; version 1 is prompt 11's construction
  and version 2 is version 1 plus the curvature criterion. A store written under version 1 cannot
  be computed into; `sqla_BackgroundModelFactory` filters on it and on the digest.
* **The production grids.** QCD **1,996** samples, digest **`a2c32f67`**, tag
  `SourceRedshiftGrid_1996_a2c32f67`; LambdaCDM **1,778**, digest **`60a3205a`**, tag
  `SourceRedshiftGrid_1778_60a3205a`. Response grids (stride 12, `protect=`): **175** and **149**.
  The base lattice is unchanged at 1,732 / `0960e169` and prompt 11's grid is unchanged at
  1,773 / `303f9ce7`; both are asserted in `test_source_grid.py`, and the new grids are strict
  supersets of them.
* **New public names.** `CosmologyConcepts.wavenumber.SOURCE_GRID_MAX_SPACING_FACTOR` (1.0),
  `SOURCE_GRID_MAX_REFINEMENT` (32), `SOURCE_GRID_CUBIC_ERROR_CONST` (1/384),
  `SOURCE_GRID_CURVATURE_STEP_U` (1e-3), `SOURCE_GRID_CURVATURE_FD_STEP_U` (1e-4),
  `SOURCE_GRID_CROSSING_MASK_U` (6e-3), `SOURCE_GRID_CONSUMER_TARGET_RAD` (1e-6),
  `SOURCE_GRID_SPLINE_EDGE_INTERVALS` (3), `SOURCE_GRID_SPLINE_EDGE_FACTOR` (10.0), all re-exported
  from `CosmologyConcepts`;
  `build_z_sample(..., spacing: Optional[Tuple[Sequence[float], Sequence[float]]] = None)`;
  `main.pre_grid_background_proxy(cosmology)` and
  `main.source_grid_spacing_profile(cosmology, base_z_values, k_values, sectors=("Gk","Tk"))`,
  both loadable with `load_main_py_functions` (the extra globals they need are listed at the top of
  `test_source_grid.py` and of `equidistributed_grid_check.py`).
* **The criterion's cost**, for anyone timing a run: 0.9 s (LambdaCDM) and 1.8 s (QCD) for the
  fifty-wavenumber envelope, against 0.599 s for a QCD `compute_background`. It runs once per
  model, before the background.
* **The reproduction commands.**
  ```bash
  # the grids, the cap, the twelve rows and the four-way table (~4 min, no Ray, no datastore)
  PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/equidistributed_grid_check.py
  # prompt 12's measurement, unchanged and still reproducing (~88 s)
  PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py
  # the QCD reference fixture (178 s; --dry-run reports without writing)
  PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py
  ```
* **What a next prompt must not assume.** The grid is now a function of the wavenumber sample as
  well as of $(z_{\rm init}, z_{\rm end}, \texttt{samples\_per\_log10z})$ — observation 2 — so
  "the production grid" is only well defined once the fifty production wavenumbers are named.
  `equidistributed_grid_check.PRODUCTION_K_INV_MPC` and `test_source_grid.PRODUCTION_K_INV_MPC` both
  transcribe `main.py:3292`'s `np.logspace(log10(1e5), log10(3e8), 50)`; if that line ever changes,
  both must change with it.
