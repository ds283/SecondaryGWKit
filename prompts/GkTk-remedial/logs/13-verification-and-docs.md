# Log 13 — Verification on both models, and the close-out documents

**Prompt:** prompts/GkTk-remedial/13-verification-and-docs.md
**Commit:** *(this commit)* — Verify the Gk/Tk WKB remediation against both background models
**Model:** Claude Opus 5
**Date:** 2026-09-13
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

No production code was touched. `git diff HEAD~1 --stat` covers only `docs/` and
`prompts/GkTk-remedial/`.

### 1. `docs/gktk-remedial/verify_production_path.py` — Layer 1 (§1)

New, 1,274 lines, seven selectable sections, 46 s for all seven. Everything runs through the production
functions — the undecorated `Quadrature.integrators.WKB_phase_function`, `BackgroundModel`'s
`TablePrimitive` accessors, `TkWKBIntegration.store()` via `test_tk_wkb_phase.build_and_store`,
`ComputeTargets.primitive_phase.PrimitivePhase` and `ComputeTargets.TkSourceFunctions` — on the
reference harness's stand-in models with prompt 06's table fixtures. No Ray, no datastore.

| section | what it measures | prompt 13 item |
|---|---|---|
| `primitives` | $\tau$, $\Delta\tau$, $\tau_s$, $F$, $\rho$ at the production nodes against prompt 01's references; the interval accessor's cost per call | §1 item 4 (README §6 rows 1–5) |
| `producers` | review §4 and §12.3 re-measured, before/after; cost per object, cold and cached | §1 item 1 |
| `stored` | $F$ and $\rho_T$ read back out of `TkWKBValue` after the production `store()` | §1 item 3 |
| `consumers` | `PrimitivePhase` at fixed $z_r=0.1$ over the source grid and `TkSourceFunctions.phase` over its WKB region, at 10 points per grid interval; plus `theta_deriv` against `omega` | §1 item 2, `[10-residual-spline-end-condition]` |
| `cycles` | the self-consistency of the stored `(div, mod)` pair at the production $|\theta|$ | (found here) |
| `margins` | the residual table's cut-to-anchor margin over the production $k$ grid | `[14-residual-range-top-margin]` |
| `throughput` | `raw_theta` in bulk at off-grid abscissae on both models | `[01-offgrid-accessor-cost-on-qcd]` |

### 2. `docs/gktk-remedial/scoped_pipeline_run.py` — the Layer 2 driver (deviation 1)

A copy of `docs/source-remediation-verification/scoped_pipeline_run.py` with one change: the two
wavenumber-grid literals it substitutes in `main.py`. See deviation 1.

### 3. `docs/gktk-remedial/analyse_scoped_run.py` — Layer 2 read-back (§2)

New, 488 lines. Opens every shard `mode=ro` and runs four checks: the persisted `tau_Mpc`/`tau_lo_Mpc`,
`cs_tau_Mpc`/`cs_tau_lo_Mpc` and `friction_F` limbs against an independent reference on the run's
own grid (mpmath at 40 digits for LambdaCDM; for QCD, converged adaptive Gauss–Legendre split at
the cosmology's declared break points — prompt 02's `branch+knots` scheme, which the reference
turned out to need as much as the table does, see "Verification performed"; both from
`reference_lib.py`); the stored `(theta_div_2pi, theta_mod_2pi)` of sampled `GkWKBIntegration` and
`TkWKBIntegration` rows against the offline producer re-run on the object's own stored initial data
and put through the production `store()` algebra and `apply_phase_offset`, requiring **exact**
equality; `GkSourcePolicyData` completion and the `WKB_phase_spline_chunks` histogram; and the
solver provenance plus the NULL-ness of every stage-2 column.

### 4. `docs/gktk-remedial-verification.md` — the verification document (§3)

New, in the shape of `docs/source-remediation-verification.md`: §1 what it is and the headline
results, §2 reconciliation (git history, environment, what "before" means), §3 Layer 1 with the
before/after tables, §4 Layer 2, §5 the README §6 acceptance table with measured values, §6 what
remains open with a one-line status for every §3 board issue, §7 reproduction.

### 5. Additive documentation notes (§4)

- `docs/gk-wkb-review-fable-2026-09-09.md` — a new **§14, "Outcome, 2026-09-13"**, appended below
  §13.5. Nothing at or above §13 was edited or deleted (`git diff` confirms the file is
  append-only). It carries the after-column headline table, states that every structural
  recommendation of §7, §13.2, §13.3 and §13.4 was implemented as written, records the one defect
  the verification found that the review did not, and notes that the stale `main.py` Bessel-stage
  comment is `transfer-remedial` prompt 06's.
- `docs/spec/02-greens-function.md` §0.1 item (1) — one dated parenthesis where the text says
  `ComputeTargets/BackgroundModel.py` "integrates $d(a_0\eta)/dz=-1/H$": since `83ef7c5` it does
  not integrate an ODE; $a_0\eta$ is a per-interval Gauss–Legendre table stored as a double-double
  pair with an interval accessor, and *the quantity and the convention $\tau=a_0\eta$ are
  unchanged*. Nothing else in the spec was touched.
- `docs/OPEN_ISSUES.md` — three rows deleted (closed), three added (opened), one hook corrected,
  the §1.4 header brought up to date; the campaign's row count is unchanged at 24 and the index's
  total at 54, so the header figure did not move.

### 6. Board (`IMPLEMENTATION_STATE.md`)

Row 13 set to ⚠️ with its log link and **Progress: 20 / 20 complete**; M24 filled in; the header
`Last updated` paragraph carries prompt 13's summary; §3 gains three entries and loses three to §4;
one clause of `[10-wrap-theta-loop-at-large-phase]` corrected in place.

## Deviations from the prompt

### 1. A second scoped-run driver, rather than the one the prompt names — **STRUCTURALLY REQUIRED**

Prompt 13 §2 says to use `docs/source-remediation-verification/scoped_pipeline_run.py`. That script
substitutes the two wavenumber grids of `main.py` by **exact text match** on

```
np.logspace(np.log10(1e5), np.log10(3e8), 50)
```

and refuses to run unless it finds exactly two occurrences. `main.py` has not spelt them that way
since `f17f2d4`, which named the sample counts:

```
np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)      # main.py:3094
np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)    # main.py:3106
```

so the script finds **zero** occurrences and raises. The file belongs to the `source-remediation`
campaign's verification folder and is not in prompt 13's "files you may touch"; `CLAUDE.md` also
says verification documents are additive. It was therefore **copied** to
`docs/gktk-remedial/scoped_pipeline_run.py` — which prompt 13 *does* allow ("new scripts under
`docs/gktk-remedial/`") — and the copy substitutes the two literals `main.py` carries today, one
occurrence each. Nothing else differs: the same three interventions (local Ray bootstrap, model-list
filter, textual grid substitution), the same refusal to touch an existing datastore, the same
pass-through of every `main.py` argument. The original is untouched and is recorded as
`[13-scoped-run-driver-k-grid-literal]`.

The alternative — editing the original in place — would have been a one-line fix, but it silently
changes another campaign's verification driver, whose own document quotes the runs it produced.

### 2. The consumer geometry is the pure-WKB band, not the whole source grid — **IMPLEMENTATION CHOICE**

Prompt 13 §1 item 2 asks for `PrimitivePhase` "at fixed $z_r=0.1$ over the full source grid, from
stand-in `GkSourceValue`s built by the production `store()` algebra". The full source grid contains
two populations, and `main.py:1558,1588` separates them at $z_{\rm source} = \sqrt{z_{e3}z_{e4}}$:
above it the object is numeric-initialised and $z_{\rm init}$ is a `root_scalar` root of a numeric
run; below it the object is pure-WKB with `G_init = 0`, `Gprime_init = 1`, $z_{\rm init} =
z_{\rm source}$ and therefore $\delta = {\rm atan2}(0, +) = 0$ exactly.

The measurement was taken over the **pure-WKB band**, which is 1,014–1,377 of the ~1,400 sources at
each wavenumber — everything from the half-e-fold band at the top down to $z=0.1$. Reproducing the
numeric-initialised band faithfully on a real background would need a stand-in for
`find_phase_extremum`'s root and for the numeric run's stop values, and prompt 09 has already
measured exactly that band, on a faithful copy of the `GkSource` rectifier, with the result that
$\varphi$ is constant to 3.64e-12 rad after rectification and the rectifier repairs 90 of 90
stop-point transitions. What prompt 09 could not do, and this does, is the **real background at
production $x$**, which is what §1 item 2 is for. The alternative — a synthetic stop rule on the real
background — would have measured my reconstruction of the production geometry rather than the
production geometry.

Layer 2 covers the numeric-initialised band in the only way that is not a reconstruction: the live
run computes those objects through `GkNumericIntegration` and `GkSourcePolicyData` and the read-back
confirms 660 of 660 `GkSourcePolicyData` rows complete with no `fail`.

### 3. The consumer reference is the producer, not mpmath — **IMPLEMENTATION CHOICE**

Prompt 13 §1 item 2 says "against the reference phase at 10 points per interval". At 10 points per
interval over ~1,400 intervals × 6 (model, $k$) cases that is ~84,000 reference evaluations, which
mpmath at 40 digits cannot deliver in a verification run's budget; and a double-precision adaptive
quadrature reference carries ~1e-14 relative of $\tau$, i.e. 0.02 rad of absolute phase at
$k=3\times10^8$ — 20 times the quantity being measured.

The reference used is therefore the **producer evaluated at those same points**: `WKB_phase_function`
anchored at $z_r$, whose leading term is the double-double table's exact interval and whose residual
is the order-4 residual table's. This is not independent of the *tables* — but it is exactly
independent of what the consumer approximates, which is the cubic spline of $\varphi$, and that is
the quantity README §6's consumer row is about. The tables themselves are scored against genuinely
independent references in §3.1 of the document (mpmath / converged `quad`), and the phase they build
against prompt 01's references in §3.2–§3.3, so the chain is closed by two measurements rather than
one. The consequence is that a reader should read §3.5's figures as *representation and
interpolation* error, not as total error against the physical phase; the document says so.

### 4. The LambdaCDM `QuadSourceIntegral` stage was stopped, and the QCD one not run — **IMPLEMENTATION CHOICE**

Prompt 13 §2 asks for the `GkSourcePolicyData` and `QuadSourceIntegral` stages "if the scoped run
reaches them". On LambdaCDM it reached them. `GkSourcePolicyData` completed; `QuadSourceIntegral`
dispatched all 3,300 work items, stored 2,363 of them — 2,263 in the first two minutes, 100 more in
the following two and a half hours — and was stopped after about three hours with 937 compute tasks
still in flight on eight cores. The QCD run was then given `--no-quad-source-integral-queue` so that
every stage the prompt *requires*, plus `GkSource` and `GkSourcePolicyData`, completed on that model
in bounded time (11 minutes).

The alternative was to let it run to completion, of unknown duration: progress had fallen to ~0.7
rows per minute and the remaining items are a small tail of pathologically expensive source
integrals. The judgement is that this is not this campaign's measurement to take:
`ComputeTargets/QuadSourceIntegral.py` is explicitly out of scope (README §1.1), this campaign
edited nothing in it, its phase consumption goes through the duck-typed protocol that
`PrimitivePhase` implements, and the cost of its Levin/Clenshaw–Curtis fallback is already an open
`source-remediation` issue (`[10-levin-wholesale-cc-fallback]`, "2.65–3.56× wall clock for
1.00–1.44× integrand evaluations"). What the stage had to say about *this* campaign is the
`WKB_phase_spline_chunks` column, and 1,388 stored rows say it, spread over all five wavenumbers and
all fifteen $(q, r)$ pairs. A later reader who wants the complete stage should budget many hours on
eight cores and should read the result as a statement about the source integral, not about the
phase.

No §3 issue is opened for it: the cost is not new, not this campaign's, and already indexed.

### 5. `--k-count 5` over $10^5$–$10^7$/Mpc — **IMPLEMENTATION CHOICE** (the prompt's own suggestion)

Prompt 13 §2 suggests exactly this (`--k-min 1e5 --k-max 1e7 --k-count 5`) and it was used
unchanged, with the production `--zend 0.1` and `--source-samples-log10z 100`. Recorded here only
because the resulting per-stage timings are not the production ones and the document says so: a
production run is 50 wavenumbers over $10^5$–$3\times10^8$, where the largest $k$ is 30× the largest
here. The Layer 1 measurements *do* reach $k=3\times10^8$; Layer 2 does not, which is why the
`WKB_mod_2pi` defect of "Observations not acted on" is a Layer 1 finding and does not appear in the
Layer 2 rows.

## Verification performed

Everything below is "I ran this and it printed X" unless stated otherwise. Raw output is in the
verification document; the tables here are the acceptance thresholds only.

### Layer 1

`PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py` — 46 s, exit 0.

**README §6, every row that this prompt can measure offline** (the four it cannot are named in the
document's §5 with the prompt that measured them):

| README §6 row | target | measured | where the maximum was |
|---|---|---|---|
| $\tau$ at nodes, relative | $\le2\times10^{-14}$ | **3.810e-16** / **2.104e-14** | LambdaCDM $z=1002.5$ / QCD $z=1.005\times10^7$ |
| $\Delta\tau$ adjacent nodes, relative | $\le10^{-13}$ | **2.494e-16** / **9.354e-15** | LambdaCDM $z=1.0006$ / QCD $z=100.19$ |
| accessor cost, QCD | measured; stop if $>50\,\mu$s | **0.33 / 33.3 / 64.7 µs** (on-grid / one off / both off); production is the first two | — |
| $\tau_s$, $F$ at nodes | $\le2\times10^{-14}$ / $\le10^{-13}$ | **2.496e-16, 3.314e-16** / **2.108e-14, 3.340e-16** | LambdaCDM / QCD |
| $\rho$ vs reference, absolute | $\le10^{-6}$ rad | **4.163e-17** / **3.608e-16** rad | LambdaCDM `Tk` 3e8 / QCD `Tk` 3e8 |
| $\theta_G$, LambdaCDM, $k=10^5$, $z=0.1$ | $\le10^{-5}$ rad | **0.0 rad** (max over checkpoints 1.192e-7) | $z=10.012$ |
| $\theta_G$, LambdaCDM, $k=3\times10^8$, $z=0.1$ | $\le5\times10^{-3}$ rad | **9.766e-4 rad** | $z=1.0006$ (1 ulp of 4.118e12 rad) |
| cost per `GkWKBIntegration` object, $k=3\times10^8$ | $\le0.05$ s | **0.0010 s / 468 evaluations** (LambdaCDM), **0.0085 s / 472** (QCD) | best of 5 |
| $\theta_T$, LambdaCDM, $k=10^5$ / $3\times10^8$ | $\le10^{-4}$ / $\le5\times10^{-3}$ rad | **1.490e-8** / **9.155e-5 rad** | $z=0.1$ both |
| cost per `TkWKBIntegration` object | measured and recorded, both figures | **0.0511 s / 11,376** building the table, **0.0181 s / 5,540** with it cached (LambdaCDM); **0.6559 / 12,896** and **0.3257 / 5,608** (QCD) | best of 5 |
| consumer phase interpolation at production $x$ | $\le10^{-6}$ rad | **2.384e-7 rad** ($G_k$) and **7.451e-9 rad** ($T_k$) on LambdaCDM at $k=10^5$, both **1.00 ulp** of their span; 1.00 ulp in ten of the twelve (model, $k$, sector) cases | met on LambdaCDM, **missed on QCD at $k=10^5$** — see below |
| chunk-switch discontinuity | none (single spline) | `WKB_phase_spline_chunks` is **1** wherever it is populated, over 609+ stored `QuadSourceIntegral` rows | Layer 2 |

**Not met, and why** — two places in one row, plus one issue. **No target was loosened.**

- **The consumer row is missed on `QCD_Cosmology` at $k=10^5$:** 1.907e-6 rad ($G_k$, 8 ulp of the
  span) and 3.186e-6 rad ($T_k$, 428 ulp) against $\le10^{-6}$ rad. Both maxima are at
  $z=4.24\times10^7$, `QCD_EOS`'s `T_LO` branch boundary, where $H(z)$ jumps by 4.4e-4 and
  $\varphi$ kinks; `PrimitivePhase` splines $\varphi$ with `make_interp_spline`'s default knots and
  so interpolates straight across it. It is 500× below the ~1e-3 rad Liouville–Green truncation
  floor that bounds any QCD phase claim, and it is
  `[13-consumer-spline-crosses-eos-break-points]`, not a loosened target. LambdaCDM meets the row
  at every wavenumber.
- **The $G_k$ consumer at LambdaCDM, $k=3\times10^8$ is 6.175 rad, not one ulp.** It is not an
  interpolation error: **one** of the 1,361 stored samples, at $z=33{,}226$, carries a $\varphi$ a
  whole cycle from its neighbours. Excluding the 15 grid intervals either side of it the maximum
  over the remaining points is **exactly 0.0 rad**. The cause is a defect in `WKB_mod_2pi` — see
  "Observations not acted on".
- **`[10-residual-spline-end-condition]`, re-measured on the real background as the board
  assigned.** On LambdaCDM the wider of prompt 10's two shipped bounds is met — `[3:-3]` gives
  2.9–3.0e-10 against `< 1e-9` — and the tighter one is **missed by 1.7×**: `[5:-3]` gives 1.66e-11
  against `< 1e-11`, met from about the seventh sample in (1.02e-11) with the deep interior at
  8.5e-12. The shipped assertion is on prompt 10's *fixture*, where it passes, and the whole suite
  passes unchanged; the real background's residual is simply a little less smooth near the hand-over
  than a constant-$w$ closed form, by under a factor two. The end effect itself decays by ~3.8 per
  sample inwards (last five, hand-over end: 9.2e-11, 2.9e-10, 1.2e-9, 4.5e-9, 1.7e-8), reproducing
  the issue's ~3×. **On `QCD_Cosmology` the identity is missed by orders — 2.3e-7 to 3.3e-4 — but
  *not at the ends*:** it is uniform across the interior, so it is the cosmology's own
  non-smoothness, not a spline end condition, and no spline order fixes it. The issue is therefore
  **closed at the cubic** and the QCD half carried forward as
  `[13-consumer-spline-crosses-eos-break-points]`.

**`[01-offgrid-accessor-cost-on-qcd]`, its stated closing condition:** `PrimitivePhase.raw_theta` at
4,000 off-grid abscissae in bulk, best of 3 — **29.3 µs per call and 4.27 integrand evaluations on
`QCD_Cosmology`** (6.86 µs / 4.00 on LambdaCDM), against the 50 µs stop threshold. On-grid: 3.56 µs,
0 evaluations. **Closed.**

**`[14-residual-range-top-margin]`, its stated closing condition:** the cut-to-anchor margin over
the 50-point production $k$ grid, both models, both sectors — worst **1.735 e-folds** (a ratio of
5.67), in the $T_k$ sector on both models, at $k=9.70\times10^6$ (LambdaCDM) and $3.70\times10^5$
(QCD); $G_k$ is 3.86 to 8.00 e-folds. Log 14's single-wavenumber 1.74 generalises. **Closed.**

### Layer 2

Two live runs through `main.py`'s own pipeline on a locally bootstrapped Ray cluster (8 of 10 CPUs)
and a **fresh** sharded datastore per model, both outside the repository. The commands, the stage
timings and the read-back are in the verification document §4; the acceptance points:

- **Every stage prompt 13 §2 requires completed on both models**: background, `TkNumericIntegration`,
  `TkWKBIntegration`, `GkNumericIntegration`, `GkWKBIntegration`, plus `QuadSource`, `GkSource` and
  `GkSourcePolicyData`. LambdaCDM: 6,336 `BackgroundModelValue`, 5 + 5 $T_k$, 2,455 + 7,920 $G_k$,
  660 + 660 `GkSource`/`GkSourcePolicyData`. QCD: 6,416, 5 + 5, 2,538 + 8,020, 670 + 670. **The QCD
  run did not fail**, which prompt 13 §2 allows for; it completed in 11 minutes.
- **`GkSourcePolicyData` is complete with no `fail`**: LambdaCDM 195 `numeric` / 44 `mixed` / 421
  `WKB`, all complete; QCD 203 / 45 / 422, with one `mixed` row `minimal` rather than `complete` —
  the band `source-remediation`'s audit §4.2 established is reachable, not a defect of this campaign.
- **The persisted limbs round-trip and are right.** Against an independent reference rebuilt on each
  run's own grid (mpmath at 40 digits on LambdaCDM, break-aware converged Gauss–Legendre on QCD),
  at three nodes spread over the grid: LambdaCDM $\tau$ **2.611e-16**, $\tau_s$ 2.204e-16, $F$
  2.440e-16 relative; QCD **4.626e-16**, 4.005e-16, 2.422e-16. `tau_lo_Mpc` runs 1e-24 to 1e-13 —
  the bits a single double cannot hold — and comes back unchanged.
  *The QCD reference had to be rebuilt to get that number, and the first attempt is worth
  recording.* A break-**unaware** composite Gauss–Legendre rule refined to 4,096 panels of order 40
  over the whole range puts QCD $\tau$ at **3.93e-09 relative** — stalled, not converged. Splitting
  at the cosmology's 375 declared break points in range (`integration_break_points`, prompt 03) plus
  each decade of $1+z$ takes the same reference to 4.63e-16. That is prompt 02's
  `[01-qcd-eos-branch-boundaries]` finding reproduced from the other side: on `QCD_Cosmology` it is
  the *reference* that needs the break points, and the stored table, which has them, was right.
- **The stored phases are bit-identical to the offline producer.** For 12 + 5 objects per model the
  object's own `z_init` and initial data are read out of the datastore, the three tables are
  reconstructed from the persisted limbs (zero quadrature), `WKB_phase_function` is re-run offline
  and the production `store()` algebra and `apply_phase_offset` applied: **0 of 13,430 samples
  differ**, in either `theta_div_2pi` (an integer) or `theta_mod_2pi` (a double), on either model.
- **`WKB_phase_spline_chunks` is 1**, on all 1,388 of the 2,363 stored `QuadSourceIntegral` rows
  that populate it (NULL on the 975 whose Green's function has no WKB phase at that response
  redshift). Never 2 or more.
- **Solver provenance**: `cumulative-GL` on every `BackgroundModel` row, `wkb-primitive` on every
  `GkWKBIntegration.solver_serial` and both `TkWKBIntegration` solver columns; `stage_2` NULL on
  **15,940 of 15,940** `GkWKBIntegration` and 10 of 10 `TkWKBIntegration` rows over the two models.
- **`has_unresolved_osc` in production**: 0 of 5 $T_k$ objects on each model, **2,455 of 2,455**
  and **2,538 of 2,538** $G_k$ objects, first firing 3.27–4.24 e-folds inside the horizon at every
  wavenumber — prompt 11 predicted 2,149 of 2,149 and 0 of 6 on stand-ins. Prompt 16's per-$k$
  summary prints **16 lines across both models** where the per-object form would have printed 9,986.

The one thing that did **not** complete is the `QuadSourceIntegral` stage on LambdaCDM: all 3,300
work items were dispatched and 2,363 computed and stored, spread over all five wavenumbers and all
fifteen $(q,r)$ pairs, before the run was stopped after about three hours with 937 compute tasks
still in flight. 2,263 of those rows landed in the first two minutes and 100 more in the next two
and a half hours, so the residue is a few pathologically expensive source integrals. That stage is
out of scope for this campaign (README §1.1), this campaign changed nothing inside it, and its cost
is already `source-remediation`'s `[10-levin-wholesale-cc-fallback]`; what it had to say about this
campaign — `WKB_phase_spline_chunks` — it said 1,388 times. The QCD run was therefore started with
`--no-quad-source-integral-queue`. Deviation 4 and verification document §4.4.

### Unit tests

Both suites, on this commit's tree, after every edit:

```
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
    Ran 339 tests in 148.977s      OK

PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .
    Ran 141 tests in 1065.977s     OK
```

339 is the count prompt 20 left; no test was added, removed or changed by this prompt, and none
needed to be — prompt 13 touches no production code and no test module. The suites are the reason
most of Layer 1's figures are assertions elsewhere rather than only printed numbers here, and in
particular `test_tk_source_functions.test_omega_matches_phase_derivative_from_the_primitive` still
passes at its shipped 1e-9 / 1e-11 pair, which is the fixture-side half of
`[10-residual-spline-end-condition]`.

## Observations not acted on

### 1. `WKB_mod_2pi`'s cycle count is a rounded division — `[13-wkb-mod-2pi-cycle-count-inconsistent]`

`LiouvilleGreen/WKBtools.py:15-27` computes `theta_mod_2pi = fmod(theta, TWO_PI)` (exact) and
`theta_div_2pi = int(floor(fabs(theta) / TWO_PI))` (a **rounded** division, then `floor`). When the
exact quotient lies within half an ulp below an integer, the division rounds up across it and the
pair no longer reconstructs its own phase: `div * 2π + mod == theta - 2π`. Worked example from the
production consumer set: `theta = -3832989103139.361`, exact quotient 610039162581.99994,
`fabs(theta)/TWO_PI` rounds to 610039162582.0, `floor` returns 610039162582, and the reconstruction
is 6.283203125 rad low.

**The stored `theta_mod_2pi` is correct** — it is the `fmod` — so no stored *value* of $G$ or $T$
moves. What is wrong is the cycle count, and therefore every consumer that reconstructs the
unwrapped phase, which since prompts 09 and 10 is both of them.

Measured on the production geometry, over the whole (source × response) rectangle for $G_k$ and the
source grid for $T_k$: **1 of 77,975** samples at LambdaCDM $k=3\times10^8$, 0 in all eleven other
(model, sector, $k$) cases (43,434–79,809 samples each for $G_k$). Uniform controls over 400,000
draws: 0 at $|\theta|\sim10^9$, 0 at $10^{11}$, **25 ($6.25\times10^{-5}$) at $4\times10^{12}$**,
against a half-ulp width of $6.1\times10^{-5}$ cycles — so the rate is the half-ulp width and scales
linearly with $|\theta|$, i.e. with $k$. Cost when it fires: 6.17 rad in the consumer, against the
9.15e-4 rad floor everything else sits at.

**The `GkSource` rectifier does not repair it.** Its trigger (`GkSource.py:200`) is
`theta > last_theta` — a cycle count jumping *up* as the source redshift rises — and this defect
makes the stored phase one cycle *more negative*, so the condition is false at the offending sample.
Not fixed here: prompt 13 may not touch production code, and the fix (derive the cycle count from
the `fmod` remainder, `div = (theta - mod) / TWO_PI` rounded to an integer, or use `divmod`-style
consistency) moves a persisted column and carries a datastore regeneration.

This also contradicts one clause of `[10-wrap-theta-loop-at-large-phase]`, which says "`WKB_mod_2pi`
uses `fmod`, is exact". Its *remainder* is; its cycle count is not. Both entries now say so.

### 2. The consumers' $\varphi$ spline crosses `QCD_EOS`'s break points — `[13-consumer-spline-crosses-eos-break-points]`

On `QCD_Cosmology` the consumer's worst error is at $z=4.24\times10^7$ in **both** sectors at
$k=10^5$ — `QCD_EOS`'s `T_LO` branch boundary, where $H(z)$ jumps by 4.4e-4 (log 02). $\varphi$ has
a kink there and a cubic spline of it does not: 1.907e-6 rad ($G_k$, 8 ulp) and 3.186e-6 rad ($T_k$,
428 ulp) against one ulp everywhere else. The same non-smoothness is what makes `theta_deriv` miss
$\omega$ by 2.3e-7 to 3.3e-4 relative across the QCD interior, uniformly rather than at the ends.

The remedy is the one prompts 02, 03, 18 and 19 already built for the quadrature and the ODE — split
at the cosmology's declared break points — applied to `PrimitivePhase`'s residual spline, which
would mean a knot vector rather than `make_interp_spline`'s default. Out of scope here, and the
magnitudes (a few 1e-6 rad against a 1e-3 rad Liouville–Green truncation floor on QCD) do not force
it.

### 3. `[07-tk-per-object-cost-is-all-setup]` re-measured, not closed

The document quotes both figures, as the issue asks. The residual-table build is 5,836 of the
11,376 integrand evaluations of a cold LambdaCDM $T_k$ object and the leading table's anchor panel
is most of the remaining 5,540 — the split prompt 14 applied to $\rho$ and not to $\tau_s$. The
issue's next step is unchanged.

### 4. `[14-rhs-evaluations-depend-on-build-order]`

The document's cost table gives the cold and cached figures separately and says which is which,
which is what the issue asks a reader of the persisted `RHS_evaluations` column to do.

### 5. `[00-consumer-anchoring-floor]`

Measured on the real background at production $x$: the global anchor's $\varepsilon k\tau$ floor is
now the *whole* consumer error at every wavenumber of both models — 1.00 ulp of the span in ten of
the twelve cases. Per-region anchoring would scale that with the region's own phase. The floor is
below the QCD Liouville–Green truncation and three to four orders below the numeric region's, so
nothing downstream is limited by it today; the issue stays open with its next step unchanged.

### 6. A pre-existing Ray head node was stopped on this machine

`ray status` reported a head from `session_2026-09-10_12-54-01` (three days old) with no attached
workload; `ray.init(num_cpus=…)` refuses to bootstrap while one exists, so `ray stop` was run before
the Layer 2 runs. Recorded in case it was wanted.

## State of the tree at close

**Every SHA of the campaign**, on branch `gktk-remedial`, in landing order. Rows marked (—) are
planning, orchestrator or prompt-text commits that touch only `prompts/` and `docs/OPEN_ISSUES.md`;
`e01c31d` is the merge of the two sibling Bessel campaigns, whose own commits are theirs.

| Prompt | SHA | Subject |
|---|---|---|
| — | `4980185` | Plan the Gk/Tk WKB phase remediation as thirteen prompts |
| 01 | `dcd99fd` | Add WKB phase references and a measured primitive prototype |
| 02 | `e912593` | Measure Gauss-order convergence of the WKB primitives on both models |
| — | `e01c31d` | Merge the Bessel phase campaigns into `gktk-remedial` |
| — | `c5e835e` | Point the last four Gk WKB review references at its new name |
| 03 | `83ef7c5` | Build conformal time as a double-double Gauss-Legendre table |
| 04 | `680ed84` | Tabulate the sound horizon and the LG friction integral per model |
| 05 | `cf986b5` | Add the WKB phase residual as a per-k table |
| 06 | `243cb84` | Compute the Green function WKB phase from the conformal-time table |
| — | `2e634fd` | Add a prompt to build the phase residual once per wavenumber |
| 14 | `b183cb8` | Build the WKB phase residual once per wavenumber |
| — | `c0a0a9e` | Correct the stated reason that the residual range is safe |
| — | `42773ba` | Widen prompt 07 to relocate the retired friction ODE |
| 07 | `2873e15` | Compute the transfer-function WKB phase and friction from tables |
| — | `b65539e` | Give the acceptance table its own transfer-function cost row |
| — | `79e0757` | Confirm the transfer-remedial merge before Workstream D |
| — | `837e909` | Record the orchestrator's own verification of prompt 07 |
| — | `7f6c393` | Close prompt 07's outstanding suites and flag wall-clock timings |
| — | `a2ea069` | Forbid subagents rewriting commits they did not author |
| 08 | `bb6a4c8` | Drop the chunked phase spline in favour of one rebased spline |
| — | `e6f88f4` | Record the orchestrator's verification of prompt 08 |
| 09 | `c1c3717` | Evaluate the Green function phase from the conformal-time table |
| — | `aae6ac5` | Record the orchestrator's verification of prompt 09, and stop |
| — | `866eeee` | Correct prompt 09's below-floor threshold and lift the stop |
| 10 | `8ba58e7` | Evaluate the transfer-function phase and friction from tables |
| — | `9689949` | Record the orchestrator's verification of prompt 10, and stop |
| — | `0038f28` | Settle prompt 10's two deviations and close Workstream D |
| — | `003ad5c` | Close the LiouvilleGreen suite at Workstream D close-out |
| — | `121de53`, `340abe9` | Prompt 15 and its orchestrator prompt |
| 15 | `2ed3632` | Give `PrimitivePhase` an explicit leading-rate callable |
| 11 | `8f606c4` | Test oscillation resolution on the sample grid, off the RHS |
| — | `a38c200` | Add prompt 16 to settle the unresolved-oscillation print policy |
| 16 | `45da2cd` | Summarise unresolved-oscillation warnings per wavenumber |
| 12 | `25e5b6f` | Give the transfer-function numeric run its own absolute tolerance |
| — | `dcdf51c` | Accept prompt 11's stop-point bound and assign the issue onward |
| — | `243c239` | Add prompt 17 to measure the Tk numeric tolerance across the k-grid |
| 17 | `6037cf3` | Measure the transfer-function numeric tolerance across the k-grid |
| — | `891d84a` | Add prompt 18 to split the numeric ODE at declared discontinuities |
| — | `4e7b3fc` | Settle the Tk numeric tolerance and require tolerance provenance |
| 18 | `2f5664e` | Split the numeric ODE at the cosmology's declared discontinuities |
| — | `0c4617e` | Add prompt 19 to make the break-point policy a per-sector choice |
| 19 | `5b84d06` | Let each numeric sector choose which declared break points it splits at |
| — | `2b4f30c` | Add the missing plan row for prompt 19 to the campaign README |
| — | `fa87cd8` | Add prompt 20 to key the numeric break-point policy |
| 20 | `ff9ee29` | Put the numeric break-point policy in the datastore lookup key |
| **13** | *(this commit)* | **Verify the Gk/Tk WKB remediation against both background models** |

Two commits on the branch belong to other work and are not this campaign's: `622b84b` ("Plan the
tolerance and convergence campaign") and `532fed4` ("Add a v2 SIGW resonance scaffolding design
proposal").

**The datastore regeneration requirement, in full, for whoever runs production next.** A datastore
written before this campaign **cannot be used**, and every one of these fails loudly rather than
returning a stale row:

1. **`BackgroundModelValue` gained four columns** — `tau_lo_Mpc` (`83ef7c5`), `cs_tau_Mpc`,
   `cs_tau_lo_Mpc`, `friction_F` (`680ed84`), all `Float(64)`, `nullable=False`; `tau_Mpc` is now
   the *high limb* of a double-double table and its value moved. `sqla_BackgroundModelFactory.build()`
   raises a `RuntimeError` naming the missing column.
2. **Every `GkWKBIntegration` and `TkWKBIntegration` value moved** (`243cb84`, `2873e15`): the phase
   is no longer an ODE solution. The rows are keyed on `atol`/`rtol`, which did not change, so they
   would be *served* — regenerate them.
3. **`TkNumericIntegration`'s absolute tolerance changed** to `DEFAULT_TK_NUMERIC_ABS_TOLERANCE =
   1e-13` (`25e5b6f`); it is part of that row's lookup key, so old rows are unreachable and would be
   recomputed rather than mis-served.
4. **QCD numeric values moved twice** — `2f5664e` (split at declared discontinuities, both sectors,
   up to 2.82e-04 of the envelope) and `5b84d06` ($T_k$ only, `BREAK_POINT_ALL`, a further
   1.61e-04). `GkNumericIntegration` rows are bit-identical across `5b84d06`.
5. **Both numeric tables gained `break_point_kind`** (`ff9ee29`), `nullable=False`, filtered on in
   `build()`. A datastore without the column raises a `RuntimeError` naming prompt 20. A datastore
   built *after* `ff9ee29` survives any future change of policy, because the policy is now in the
   key.

A fresh datastore built at this commit is good. Neither Layer 2 datastore is committed; the scratch
paths are recorded verbatim in the verification document §7 and are not expected to survive.

**The open issues, and who owns them.** The board's §3 at close, by owner:

| Owner | Issues |
|---|---|
| **A follow-up prompt of this campaign, if anyone wants one** | `[13-wkb-mod-2pi-cycle-count-inconsistent]` (the only *live* accuracy defect left — 6.17 rad at $k=3\times10^8$, one sample in 78,000, and it grows with $k$); `[13-consumer-spline-crosses-eos-break-points]`; `[00-consumer-anchoring-floor]`; `[07-tk-per-object-cost-is-all-setup]`; `[06-metadata-column-headroom]`; `[20-wkb-gauss-orders-not-in-lookup-key]`; `[20-wkb-rows-consume-numeric-initial-data]` |
| **The hand-over campaign** (`docs/OPEN_ISSUES.md` §1.1) | `[00-tk-lg-truncation-floor]`, `[11-stop-point-root-tolerance]` |
| **`prompts/tolerance-convergence`** | `[12-tk-numeric-atol-largest-k-excursion]` — the user settled `atol = 1e-13` on 2026-09-12; the remaining lever is `rtol`, which keys every integration object |
| **The author, as a spec or model decision** | `[00-tk-superhorizon-ic-series]`, `[02-qcd-T-z-spline-node-tolerance]` |
| **Whoever next has that directory in scope** (comment- or text-only) | `[19-cosmologymodels-docstrings-predate-per-sector-policy]`, `[10-transfer-remedial-tolerance-comments-stale]`, `[13-scoped-run-driver-k-grid-literal]`, `[08-docs-scripts-reference-removed-chunking]`, `[06-docs-scripts-reference-removed-ode]`, `[10-wrap-theta-loop-at-large-phase]`, `[03-backgroundmodelvalue-build-path]`, `[03-integrationsolver-stepping-minimum-lookup]`, `[04-background-rhs-evaluations-count]` |
| **Nobody — recorded floors, not actions** | `[01-lambdacdm-hubble-rounding-floor]`, `[02-qcd-reference-floor]`, `[03-qcd-short-baseline-reference-endpoint-rounding]`, `[14-rhs-evaluations-depend-on-build-order]` |

**If you read only one thing before touching this code again**: the phase is now delivered at the
double-precision floor and the floor is $\varepsilon k\tau$, $3\times10^{-7}$ rad at $k=10^5$/Mpc to
$9\times10^{-4}$ rad at $3\times10^8$ (board §5 note 2). Nothing in this campaign's numbers can be
improved without changing the representation, and a test that asserts below that floor is asserting
agreement between two errors. The one exception — the one place where a real improvement is still
available — is `[13-wkb-mod-2pi-cycle-count-inconsistent]`.
