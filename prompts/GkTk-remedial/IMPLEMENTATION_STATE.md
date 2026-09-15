# Implementation state — Gk/Tk WKB phase remedial campaign

**Campaign:** [`README.md`](README.md) · **Source review:** [`docs/gk-wkb-review-fable-2026-09-09.md`](../../docs/gk-wkb-review-fable-2026-09-09.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md)
**Baseline commit:** `9ff59d5` (`main`, clean)
**Last updated:** 2026-09-15 — `[13-consumer-spline-crosses-eos-break-points]` is **CLOSED** (§4) by `prompts/qcd-background-audit/` prompt 10, which measured nine knot schemes over all twelve production rows and found the remedy this entry named **2.09×/2.10× worse** ($C^0$ repeated knot) and per-segment splines 5.00×/5.06× worse, while ±5 grid intervals of extra *samples* around the crossing bring both rows inside the 1e-06 rad target; the unfixed accuracy defect re-opens as `[10-consumer-phi-unresolved-at-the-eos-crossing]` on the `qcd-background-audit` board, assigned to its prompt 11. Previously 2026-09-14 — `[13-consumer-spline-crosses-eos-break-points]` **narrowed again** by `prompts/qcd-background-audit/` prompt 09: re-measured on the corrected background its two §3.5 rows are **4.25× and 4.39× larger**, because the old `T(z)` spline was smearing the equation of state's step rather than causing the error, and this entry's supersession paragraph's prediction is falsified. Previously 2026-09-14 — `[02-qcd-reference-floor]` and `[03-qcd-short-baseline-reference-endpoint-rounding]` re-measured (narrowed, neither closed) by `prompts/qcd-background-audit/` prompt 06. Previously 2026-09-13 — **Prompt 13 landed and the campaign is closed.** The verification
document is [`docs/gktk-remedial-verification.md`](../../docs/gktk-remedial-verification.md), taken
on `ff9ee29` and changing no production code. **Layer 1** re-measures review §4 and §12.3 through
the production functions on both models at $k\in\{10^5,10^7,3\times10^8\}$: at $z=0.1$ on
LambdaCDM, $\theta_G$ goes **13.9 rad → 0.0** ($k=10^5$) and **7366 → 9.77e-04 rad**
($3\times10^8$, exactly one ulp of the 4.118e12 rad phase), $\theta_T$ **2.01 → 1.49e-08** and
**5.1e3 → 9.16e-05 rad**; on `QCD_Cosmology`, which the review never measured, the same four are
4.77e-07 / 4.88e-04 and 7.45e-08 / 2.44e-04 rad. Cost per object at $3\times10^8$: $G_k$ **0.0010 s
and 468 integrand evaluations** against 63.7 s and 2.54e6, $T_k$ **0.0181 s and 5,540** cached
(0.0511 / 11,376 cold) against 58 s and 1.93e6. The consumers at production $x$, scored at ten
points per grid interval, are at **1.00 ulp of their span** in ten of twelve (model, $k$, sector)
cases. **Layer 2** ran the pipeline on a fresh datastore per model through every stage; the stored
`(theta_div_2pi, theta_mod_2pi)` of sampled rows is **bit-identical** to the offline producer put
through the production `store()` algebra, the persisted limbs agree with an independent reference to
2.6e-16 relative, and `WKB_phase_spline_chunks` is **1** on every row that populates it. **Three
issues closed on measurements the board had assigned to this prompt**:
`[01-offgrid-accessor-cost-on-qcd]` (29.3 µs per call in bulk on QCD, under the 50 µs threshold),
`[14-residual-range-top-margin]` (worst cut-to-anchor margin 1.735 e-folds over the whole production
$k$ range) and `[10-residual-spline-end-condition]` (on the real background the cubic meets the
wider of prompt 10's two shipped windows and misses the tighter by 1.7×; `spline_order=5` is not
taken, because it does not touch what actually dominates there). **Three opened**, none of them fixed here because prompt 13
may not touch production code: `[13-wkb-mod-2pi-cycle-count-inconsistent]` — `WKB_mod_2pi` takes its
remainder from an exact `fmod` but its cycle count from a *rounded* division, so the stored pair can
reconstruct $\theta-2\pi$; 1 of 77,975 production $G_k$ samples at $k=3\times10^8$ on LambdaCDM, and
it costs the consumer **6.17 rad** there, the largest single error left in the chain —
`[13-consumer-spline-crosses-eos-break-points]` and `[13-scoped-run-driver-k-grid-literal]`.
`docs/gk-wkb-review-fable-2026-09-09.md` gains an additive **§14** and `docs/spec/02-greens-function.md`
§0.1 item (1) a dated parenthesis. Prompt 20 landed (the numeric **break-point policy is now part of
both numeric datastore lookup keys**, enacting the user's decision of 2026-09-13 to key the
*configuration* and leave the solver as provenance. Each numeric compute target carries one
declaration of its policy — `GkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_DISCONTINUITY`,
`TkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_ALL`, in the shape prompt 06 used for
`PHASE_SOLVER_LABEL_BASE` — and three uses read it: `compute()` passes it to
`numeric_with_phase_cut` as `self.BREAK_POINT_KIND`, the factory's `store()` writes it, and the
factory's `build()` **filters** on it. Both numeric tables gain a plain
`String(DEFAULT_STRING_LENGTH)` column `break_point_kind`, `nullable=False`, placed after
`rtol_serial`; the vocabulary is `CosmologyModels/GenericEOS`'s own, and no cosmology, temperature
or equation of state enters `Datastore/`. **There is no migration and no default**: a datastore
written before this commit raises a `RuntimeError` naming the prompt and demanding regeneration,
on the `BackgroundModel` prompts 03/04 pattern, rather than silently returning a row computed
under an unknown policy. **`main.py` is untouched** — the policy is not per-call configuration, so
neither numeric `object_get` site needed anything. **Nothing computed moved**: a production QCD
object in both sectors at $k=4.972\times10^7$/Mpc is **bit-identical** to `HEAD~1` (sha256
`565e907…` over every sample as a hex float, 494 + 41 samples, 32,351 + 13,258 RHS evaluations),
`Quadrature/`, `config/defaults.py`, `CosmologyModels/`, every tolerance and every `solver_serial`
being untouched. Suite 328 → **339**, none removed; the one existing case whose expectation moved
is *structural*, not numerical — prompt 19's `ast` call-site test now follows the call site to the
class constant, with three assertions where it had two. **The §5 audit refutes the "no equivalent
free parameter" expectation** for the other three targets: `TAU_GAUSS_ORDER`,
`CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER`, `RHO_GAUSS_ORDER` and
`RESIDUAL_WKB_REGION_MARGIN` are in no lookup key, the last in no label or tag either
(`[20-wkb-gauss-orders-not-in-lookup-key]`), and the WKB targets consume the numeric stop point
while being keyed independently of it — covered in practice only because `z_init` is filtered as
an absolute `1e-7` against $z\sim10^{12}$, and measured: the two policies move QCD $z_{\rm init}$
by 4.59e5 at that wavenumber, so the lookup misses
(`[20-wkb-rows-consume-numeric-initial-data]`). Both opened, neither acted on.
`[18-numeric-solver-not-in-lookup-key]` is **resolved** (§4), its recorded next step corrected in
place: adding `solver_serial` to the queries is a no-op.) Prompt 19 landed (**which kind** of declared break point the numeric
ODE splits at is now the *caller's* choice, enacting the user's decision of 2026-09-13:
`numeric_with_phase_cut` takes a `break_point_kind`, appended last in its signature and defaulting
to `BREAK_POINT_DISCONTINUITY` — so every caller that does not name it, the four `docs/` scripts
and three test modules included, keeps its numbers bit for bit. **Both production integrators name
it anyway**, because the asymmetry is a measured decision in each sector rather than one inherited
by omission: `TkNumericIntegration` passes `BREAK_POINT_ALL` and `GkNumericIntegration`
`BREAK_POINT_DISCONTINUITY`, each under a comment citing `TK-NUMERIC-ATOL-SWEEP.md` §9.1 and §9.7.
**The acceptance test passes**: on `QCDModel` the $T_k$ reference-convergence drift is below the
3.4e-08 criterion at **all 50** production wavenumbers — worst **8.72e-09**, median 3.96e-09,
against 1.97e-07 and **3** offenders with the jumps alone — so §9.7's three-wavenumber measurement
generalises and the other 47 are undisturbed. **$G_k$ reproduces §9 exactly**: 1.94e-11 / 2.1e-11 /
8.41e-09 worst drift at the same wavenumbers, **13320** evaluations per QCD object (§9.3's integer),
and at all 50 wavenumbers the run with the argument omitted is bit-identical to the run with it
named. Both smooth models take the single-`solve_ivp` path under **either** policy and are
bit-identical between them at all 50 wavenumbers in both sectors, prompt 17's two controls included
(2.53e-06 / 7403; 2.56e-04 / 8483). Cost on QCD: $T_k$ 9843 → **31521** evaluations per object
(**+220.23 %**, ~49 s for the whole 50-object sector), $G_k$ 13320 → 13320 (**+0.00 %**, the same
integer — several core-hours per model *not* spent on a quantity already at 8.41e-09). One change
was forced on the shared driver — `_separated_boundaries` drops a segment boundary that does not
clear its predecessor or an endpoint by `BREAK_POINT_STANDOFF`, because asking for every declared
break point cuts a production $T_k$ object at ~127 boundaries rather than one; it is **inert on
every production geometry** (the measured minimum separation of declared points on `QCD_Cosmology`
is 3.4e-03 in $\log(1+z)$, nine orders above the 1e-12 standoff, and the closest approach of a break
to a requested sample is 1.88e-06) and the exact $G_k$ regression is what proves it. No tolerance,
no `CosmologyModels/` declaration, no `Datastore/` factory and no lookup key was touched. **The QCD
$T_k$ answer moves a second time**, by up to **1.61e-04** of the envelope on top of prompt 18's
2.82e-04, so `[18-numeric-solver-not-in-lookup-key]` bites again and a pre-commit QCD datastore is
still unusable for that sector — but **`GkNumericIntegration` rows are bit-identical across this
commit on all three models**. `[17-qcd-reference-not-converged]` is **resolved** (§4).) Prompt 18 landed (the numeric ODE is **split at the cosmology's
declared discontinuities**. `GenericEOSBase` gains `discontinuity_temperatures_GeV`, default `()`
— "a smooth equation of state has none" — and `QCD_EOS` declares `(T_LO, T_120_MEV, T_HI)`,
*measured* rather than transcribed: $G$ and $G_s$ step by 2.1e-4–1.5e-2 there and are continuous
at `EOS_T_LO`, and $H(z)$ steps by 4.437e-04 at $z=4.25\times10^7$ and 1.038e-04 at
$z=8.64\times10^{11}$. `integration_break_points` takes a `kind=` of `BREAK_POINT_ALL` (the
default, **so the quadrature path is byte-for-byte unchanged** and
`test_background_tau.test_qcd_break_points` is untouched) or `BREAK_POINT_DISCONTINUITY` (jumps
only, no knots), and `numeric_with_phase_cut` asks for the latter, integrating the segments
between them in sequence and carrying each final state into the next. A cosmology declaring
nothing — every LambdaCDM model, `RadiationModel`, every stand-in — executes the **unmodified**
`solve_ivp` statement; a LambdaCDM $G_k$ run is asserted bit-identical to a directly issued call.
`t_eval`, `mode="stop"` (including a search window straddling a boundary, through a composite
dense output), the supervisor's aggregated accounting and the `has_unresolved_osc` warning all
survive, and a per-segment failure names its segment. **The acceptance test is met at 47 of 50**:
on `QCDModel` the $T_k$ reference-convergence drift goes from 23 wavenumbers above the 3.4e-08
criterion (worst 6.17e-06) to **3** (worst **1.97e-07**), and all four wavenumbers prompt 17 named
improve **347×–5764×**. **$G_k$ never had the failure** — 1.94e-11 / 2.1e-11 / 8.41e-09 worst over
the grid, the same split or unsplit, the first converged-reference measurement of that sector on
any model. Two things had to be measured rather than assumed: the boundary must stand off the
crossing by 1e-12 relative on the **near** side, because a segment ending exactly on it evaluates
its last Runge–Kutta stage on whichever branch rounding picks — with the boundary on the crossing
$k=4.972\times10^7$ stays at 2.99e-06 — and the synthetic fixture **cannot** demonstrate the
improvement at any jump size or tolerance tried, because DOP853 detects a large jump and grinds
its step down, so the accuracy claim is tested on the real cosmology instead. Cost **+1.38 %**
($T_k$) / **+0.39 %** ($G_k$) per QCD object. **A QCD datastore built before this commit may not
be used**: the split moves production values by up to 2.82e-04 of the envelope and `solver_serial`
is in neither numeric lookup key — `[18-numeric-solver-not-in-lookup-key]`, reported not acted on.
`[17-qcd-reference-not-converged]` is **narrowed, not closed**: the residue is the $T(z)$ spline's
$C^2$ knots, which would close it at **+219 % / +155 %** of the production evaluations — measured,
and left to the user.) Prompt 17 landed (measurement only, no production code: the
transfer function's numeric absolute tolerance is measured across the **whole production
$k$-grid** — 50 wavenumbers × 3 models × `atol` ∈ {1e-10, 1e-13, 1e-16} at `rtol = 1e-8`, each
scored against a converged run of the same integrator, in 276 s. Prompt 12's two control figures
reproduce to 0.02 % and 0.01 %. **The excursion is real off the radiation control and is not a
large-$k$ effect**: at the shipped `atol = 1e-13`, **3 of 50** (Radiation), **13 of 50**
(LambdaCDM) and **8 of 50** (QCD) wavenumbers exceed README §6's $3\times10^{-6}$, worst
**8.64e-4** at $k=8.37\times10^5$ on LambdaCDM — and each is a *level*, not one bad sample: the
median is lifted to 1.7e-5–5.8e-5 and the last returned sample is still wrong by up to 2.2e-4.
**`atol = 1e-16` does not fix it** — LambdaCDM 13 → 10, two of the ten wavenumbers `1e-13`
handles — and is not cheaper across a real background (15 of 50 $k$, grid total +0.5 %/+1.1 %);
prompt 12's "cheaper" was one point on the control. Two diagnostics identify the lever: at fixed
`atol = 1e-13`, one decade of **`rtol`** removes every excursion (8.64e-4 → 7.4e-8) for +23 %
evaluations, and on both production backgrounds a **$10^{-6}$ change in $k$** removes it too, so
it is an accident of the step sequence. **Recommendation: keep `1e-13`; prompt 13 may build its
datastore on it.** Two limits are recorded rather than fixed: `QCD_Cosmology`'s discontinuous
$H(z)$ stops any reference of this construction converging below ~6e-6 of the envelope at four
QCD wavenumbers, and the $T=1,T'=0$ initial condition holds a $k$-independent 2.52e-6 floor,
confirmed against the exact $T$ at all 50 $k$. `[12-tk-numeric-atol-largest-k-excursion]` is
**narrowed, not closed** — the constant is the user's call.) Prompt 12 landed (the transfer function's numeric run has its own
absolute tolerance: `config.defaults.DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13`, built into a fifth
`tolerance` object `Tk_numeric_atol` in `main.py` and threaded through **all five**
`TkNumericIntegration` `object_get` sites — the existence query, the work item, the WKB stage's
initial-condition lookup, and the two source-stage lookups — while `TkWKBIntegration`,
`GkNumericIntegration` and `GkWKBIntegration` keep `atol`. On review §12.5's own geometry, which
reproduces all four of its RHS-evaluation counts exactly at $k=10^6$, $\delta T/{\rm env}$ goes
from **9.928e-6 to 2.534e-6** for **+14.3 %** evaluations (6476 → 7403), and the residue is the
super-horizon initial condition, not the solver: with exact initial data the same run gives
**3.275e-7**. $G_k$ is indifferent, as the review says — 8.538e-10 of the envelope and +0.35 %
evaluations — so nothing about $G_k$ moves. **A fresh datastore is required**: the tolerance is
part of every `TkNumericIntegration` row's lookup key. One structural repair came with it
(deviation 1, log 12): `build_Tk_WKB_work`'s numeric lookup was dispatched over `query_batch` —
the `TkWKBIntegration` query over the whole batch — with `payload_batch`, built over `missing`
alone, unused since `8e96750`; the `GkWKBIntegration` twin passes `payload_batch`, and the two
tolerances can no longer share one dict.) Prompt 16 landed (the unresolved-oscillation warning is
**relocated, not deleted**, enacting README §7 **D2 option (ii)**, which the user took on
2026-09-11 on prompt 11's measurement. `scan_sample_grid_for_unresolved_osc` gains a keyword-only
`warn: bool = True` that gates **only** its two `print` calls — the test, which pair trips it and
all three returned values are bit-identical to prompt 11's — and `numeric_with_phase_cut` threads
it as `warn_unresolved_osc: bool = True`, appended last in the signature so no positional index
moved. Both production integrators pass `warn_unresolved_osc=False`; every other caller keeps
today's behaviour by default. `main.py` gains two module-level functions,
`record_unresolved_osc(summary, obj)` and `format_unresolved_osc_summary(summary, sector_label)
-> List[str]`, and both numeric work queues gain
`post_handler=lambda obj: record_unresolved_osc(<acc>, obj)` plus a summary printed after
`.run()` — `post_handler` is the only seam, because both queues run `store_results=False` and
retain no objects to sweep. Per flagging wavenumber the block gives the counts and the range of
`unresolved_efolds_subh`; production goes from ~$1.3\times10^5$ printed lines per model to ~52.
**Nothing stored changed**: the payload keys, the two Datastore factories and the persisted
columns are untouched, and a datastore written before this commit is readable after it.
`[00-unresolved-osc-print-policy]` is **resolved** (§4) — but only its *printing* half: whether
the response grid should resolve the mode through the numeric→WKB seam belongs to the hand-over
campaign.) Prompt 11 landed (the numeric region's oscillation-resolution
diagnostic is off the ODE right-hand side: `numeric_with_phase_cut` takes an `omega_sq` callable,
and `scan_sample_grid_for_unresolved_osc` runs the test **once, after the solve, on the returned
sample grid** — the grid review §13.1 says the flag is about — instead of at every solver step
against a spacing reconstructed from `delta_logz`. `has_unresolved_osc`, `unresolved_z` and
`unresolved_efolds_subh` are still populated and the two-line warning still prints, verbatim; the
returned $G$, $G'$, $T$, $T'$ samples and the RHS-evaluation counts are **bit-identical**, and a
LambdaCDM $G_k$ object costs 0.0773 s against 0.1299 s. `NumericIntegrationSupervisor.report_wavelength`
is retained, documented as superseded, with its $\ln10$ slip fixed in place.
`find_phase_minimum` is now `find_phase_extremum` (old name an alias), steps
$2\pi/(16\omega)$ when a frequency is available, and its docstring and both integrators' comments
say the stop point is a **maximum**. `mode.lower()` no longer precedes the `None` check, and
`main.py`'s two `0.85 z_e6` comments now say the trailing samples are never produced in stop mode.
**The D2 measurement the orchestrator must put to the user** (`[00-unresolved-osc-print-policy]`):
the corrected test fires on **2,149 of 2,149** $G_k$-like objects — every one, first at $x=26.5$–66.6
— and on **0 of 6** $T_k$-like runs, which peak at 0.807–0.822 of the trip threshold. Today the rate
is 0 of 2,155.) Prompt 15 landed (`PrimitivePhase` takes an explicit, optional
keyword-only `rate` callable — `d/dz[leading.delta(z, z_anchor)]`'s magnitude — defaulting to
`lambda z: 1.0 / model_functions.Hubble(z)` when omitted, so every pre-existing call site is
unchanged; `theta_deriv`'s leading term is now `sign * k * rate(z)` rather than the hard-wired
`sign * k / model_functions.Hubble(z)`. `TkSourceFunctions.py`'s `_SoundHorizonRate` adapter —
which made `model_functions.Hubble` silently return $H/c_s$ — is gone; `TkSourceFunctions`
now passes the model's real `ModelFunctions` as `model_functions` and a module-level
`_sound_horizon_rate(functions)` closure, which carries the same positive-$c_s^2$
`RuntimeError` guard, as `rate`. The reordered expression (`k * (sqrt(c_s^2)/H)` vs. the old
`k / (H/sqrt(c_s^2))`) differs by **2.218e-16 relative** ($w=1/3$) and **2.191e-16** ($w=0.2$,
measured directly at every stored sample) — six-plus orders below prompt 10 §3 item 3's
1e-9/1e-11 window, which reproduces its own figures (1.0492e-10 / 5.559e-12 at $w=1/3$,
7.9502e-11 / 3.757e-12 at $w=0.2$) unchanged to the last printed digit, confirming the
reordering is invisible at that test's precision. `GkSourcePolicyData.py` needed no edit: its
`PrimitivePhase(...)` call passes no `rate` and takes the new default unchanged.
`[10-primitive-phase-leading-rate-is-hardcoded]` is **resolved** (§4).) Prompt 10 landed (the transfer-function *consumer* is on the tables
too: `TkSourceFunctions.phase` is a `PrimitivePhase` with `leading = cs_tau`, `z_anchor = z_init`,
`sign = +1`, and `friction(z)` is `friction_F.delta(crossover_z, z)` read exactly from the
background table with the stored samples kept only as a construction-time cross-check.
`PHASE_SPLINE_CHUNK_LOGSTEP` is gone and no production path builds a `phase_spline` any more. On a
$w=1/3$ fixture reaching $x_T=10^6$ on the production 100/decade grid the phase error falls from
**7.0854e-3 rad** (a cubic spline of the same stored samples) to **3.4482e-10 rad**, a ratio of
**2.055e7**, and the 3.4e-10 is 2.96 ulp of the $10^6$ rad phase — the `div * TWO_PI`
representation floor. Friction matches its closed form to **8.9e-16** absolute. Two deviations went to the
user, and both are now settled. **Prompt 10 §3 item 3's 1e-10 relative on `omega` vs `theta_deriv`
is missed at one abscissa per equation of state** — 1.0492e-10 at $w=1/3$, the not-a-knot end
condition of the residual spline at the top of the WKB region, 5.6e-12 from the fifth sample
inwards, against 4.249e-08 before: the shipped 1e-9 / 1e-11 window pair stands and
`[10-residual-spline-end-condition]` is **left open and assigned to prompt 13**, which re-measures
it on the real background before anyone pays for a quintic. **Five lines of stand-in construction
in `ComputeTargets/tests/test_quadsource_integral.py`, outside the prompt's file list** are
**accepted** — the orchestrator confirmed by restoring the file that §1's friction cross-check
fails there on a genuinely inconsistent fixture — and the prompt's file list is extended in place
with the reasoning (`[10-quadsource-fixture-model-substitution]` resolved, §4).
`[00-transfer-remedial-test-file-overlap]` is **resolved**: `test_phase_groups.py` needed no edit
at all — it is byte-identical to the pre-Workstream-D snapshot — and every line prompt 10 removed
from `test_tk_source_functions.py` blames to `e3348e4`, not to `8ba9159`, so those tolerances are
intact. **Workstream D is complete: the campaign's accuracy claims now hold end to end.**)
Prompt 09 landed (the Green's-function *consumer* stops splining the
growing phase: `ComputeTargets/primitive_phase.py` evaluates
$\theta=-k\,\tau.\mathrm{delta}(z_s,z_r)+\varphi$ from prompt 03's double-double table with a cubic
spline of the small residual $\varphi$ alone, and both `GkSourcePolicyData` call sites — the
`Levin_z` threshold test and `GkSourceFunctions.phase` — now build one. On the review §5 consumer
geometry scaled to $k=10^8$ the error falls from **7.286e-3 rad** to **4.189e-8 rad**, a ratio of
**1.739e5**, and the 4.189e-8 is 2.81 ulp of the 9.09e7 rad span — the $\varepsilon k\tau$
representation floor, not the method's error; $\varphi$ itself is recovered to 2.157e-10 rad. The
`GkSource` rectifier is verified on a faithful copy: 90 of 990 swept objects carry a $+1$-cycle
step at a stop-point transition, the rectifier repairs every one, and after it $\varphi$ is
constant to 3.6e-12 rad; on pure-WKB objects it makes **zero** corrections. **Prompt 09 §4 test
1's 1e-8 rad threshold is below the double-precision floor of its own geometry** — 0.67 ulp — and
was replaced by README §6's 1e-6 rad plus a 6-ulp bound. The orchestrator stopped on that, per
README §4.3; an independent Fable review
([`reviews/09-prompt-09-review-fable.md`](reviews/09-prompt-09-review-fable.md)) accepted the
implementation, confirmed the arithmetic and recommended amending the prompt text, which was done —
`[09-consumer-threshold-below-representation-floor]` is **resolved** (§4).)
Prompt 08 landed (`phase_spline` chunking is gone: one cubic spline
over the whole sample, rebased at the sample's median `theta_div_2pi` rather than selected between
several by a hard switch; `chunk_step`/`chunk_logstep`/`increasing` are accepted no-ops, so
`bessel_phase.py` and the three test fixtures that still pass `chunk_logstep=125` needed no edits;
`num_chunks` always reports 1. The interior interpolation error the review measured, 8.2566e-05 rad at
$k=10^6$ on the `GkSourcePolicyData` geometry (review §5: 8.26e-5), is confirmed unchanged from
before de-chunking —
chunking bought nothing and cost ordinates, knot residuals and a switch discontinuity, all now
gone. Workstream D may proceed to prompt 09.) Workstream C closed; the `transfer-remedial` merge confirmed at `e01c31d`, so Workstream D may start (`[00-transfer-remedial-test-file-overlap]`). Prompt 07 landed (the transfer-function phase and friction now come from the tables too: `friction_RHS` and its state index are gone from the producer and live only in prompt 04's test, `store()`'s no-op sign fix and its cross-sample rebase are gone, and the stored $\theta_T$ is 1.5e-8 rad at $k=10^5$ and 9.2e-5 rad at $3\times10^8$ against prompt 01's references, where the ODE was 2.01 rad and 5.1e3 rad. Review §12.4's LG truncation table reproduced to two figures. **Its cost, 0.049–0.052 s per object, straddles prompt 07 §3 item 6's 0.05 s** — `[07-tk-per-object-cost-is-all-setup]`.) Prompt 14 landed (the residual table is built once per $(model, k, sector)$ on the background grid and memoised in the worker: 142.5 (LambdaCDM) / 163.0 (QCD) residual-integrand evaluations per object over 50 objects of one $k$ against 6,924 / 7,908 — 49× — and 0.0010 s per object at $k=3\times10^8$ against 0.0309 s; $\theta$ bit-identical at every sample of fifteen (model, $k$, sector) cases). Prompt 06 landed (the Green's-function WKB phase is now $-[k\,\Delta\tau+\Delta\rho]$ from the tables: the two-stage phase ODE, the `Q` variable, the resets, the sign fix and the cross-sample rebase are gone; stored phases at the floor; `TkWKBIntegration.compute()` switched, its `store()` awaits 07).

**Post-close-out, 2026-09-13.** `[13-wkb-mod-2pi-cycle-count-inconsistent]` moved from §3 to §4:
**resolved** by prompt 01 of
[`prompts/phase-representation`](../phase-representation/IMPLEMENTATION_STATE.md), which derives
`WKB_mod_2pi`'s (and `simple_mod_2pi`'s) cycle count from the exact `fmod` remainder. LambdaCDM
$G_k$ at $k=3\times10^8$ is **0 inconsistent of 77,975** (was 1), the $|\theta|\sim4\times10^{12}$
uniform control **0 of 400,000** (was 25), and that case's consumer row **0.0000e+00 rad** (was
6.1748). The stored remainder is bit-identical, so no stored $G$ or $T$ moved. This board's
narrative above, and `docs/gktk-remedial-verification.md` §§1–7, are the record of the tree at
`ff9ee29`/`9daa2cb` and were **not** rewritten (`CLAUDE.md`: verification documents are additive).

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status, fill in the commit SHA, model and log link, update the mechanism-level
> table in §2, and add or clear entries in §3 (Active issues). Do not edit rows other than your own
> except to close an issue you resolved. **Any change to §3 or §4 must also update the project-wide
> index [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit** (see `CLAUDE.md`).

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Workstream A — measurement and prototypes

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [Reference harness and prototype](01-reference-harness-and-prototype.md) | review §7, §13.3 | Opus | ⚠️ | *"Add WKB phase references and a measured primitive prototype"* (SHA not embedded, per the campaign convention) | [`logs/01-reference-harness-and-prototype.md`](logs/01-reference-harness-and-prototype.md) |
| 02 | [QCD residual convergence](02-qcd-residual-convergence.md) | review §11, §12.7 | Opus | ⚠️ | *"Measure Gauss-order convergence of the WKB primitives on both models"* (SHA not embedded, per the campaign convention) | [`logs/02-qcd-residual-convergence.md`](logs/02-qcd-residual-convergence.md) |

### Workstream B — the primitives in `BackgroundModel`

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 03 | [τ primitive](03-tau-primitive.md) | review §7, §13.2, §13.3 | **Fable** | ⚠️ | *"Build conformal time as a double-double Gauss-Legendre table"* (SHA not embedded, per the campaign convention) | [`logs/03-tau-primitive.md`](logs/03-tau-primitive.md) |
| 04 | [Sound-horizon and friction tables](04-sound-horizon-and-friction-tables.md) | review §12.7 | Opus | ⚠️ | *"Tabulate the sound horizon and the LG friction integral per model"* (SHA not embedded, per the campaign convention) | [`logs/04-sound-horizon-and-friction-tables.md`](logs/04-sound-horizon-and-friction-tables.md) |

### Workstream C — the producers

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 05 | [Phase residual](05-phase-residual.md) | review §6, §12.2, §12.4 | Opus | ⚠️ | *"Add the WKB phase residual as a per-k table"* (SHA not embedded, per the campaign convention) | [`logs/05-phase-residual.md`](logs/05-phase-residual.md) |
| 06 | [Gk WKB phase from the primitive](06-gk-wkb-phase-from-primitive.md) | review §2–§4, §8, §13.4 | **Fable** | ⚠️ | *"Compute the Green function WKB phase from the conformal-time table"* (SHA not embedded, per the campaign convention) | [`logs/06-gk-wkb-phase-from-primitive.md`](logs/06-gk-wkb-phase-from-primitive.md) |
| 14 | [Residual table reuse](14-residual-table-reuse.md) | §3 `[06-residual-table-per-object]` | Opus | ⚠️ | *"Build the WKB phase residual once per wavenumber"* (SHA not embedded, per the campaign convention) | [`logs/14-residual-table-reuse.md`](logs/14-residual-table-reuse.md) |
| 07 | [Tk WKB phase from the primitive](07-tk-wkb-phase-from-primitive.md) | review §12.1–§12.4 | Opus | ⚠️ | *"Compute the transfer-function WKB phase and friction from tables"* (SHA not embedded, per the campaign convention) | [`logs/07-tk-wkb-phase-from-primitive.md`](logs/07-tk-wkb-phase-from-primitive.md) |

> Row 14 is numbered last because the campaign's numbers are append-only, but it **runs
> between 06 and 07**: it removes the per-object residual-table build that 06 introduced,
> and 07 inherits the same producer.

### Workstream D — the consumers

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 08 | [`phase_spline` de-chunk](08-phase-spline-dechunk.md) | review §5 | Sonnet | ✅ | *"Drop the chunked phase spline in favour of one rebased spline"* (SHA not embedded, per the campaign convention) | [`logs/08-phase-spline-dechunk.md`](logs/08-phase-spline-dechunk.md) |
| 09 | [Gk consumer on `PrimitivePhase`](09-gk-consumer-primitive-phase.md) | review §5, §7, §8.3, §13.3–§13.4 | **Fable** → Opus (Fable unavailable) | ⚠️ | *"Evaluate the Green function phase from the conformal-time table"* (SHA not embedded, per the campaign convention) | [`logs/09-gk-consumer-primitive-phase.md`](logs/09-gk-consumer-primitive-phase.md) |
| 10 | [Tk consumer on the tables](10-tk-consumer-primitive-phase.md) | review §12.6, §12.7 | Opus | ⚠️ | *"Evaluate the transfer-function phase and friction from tables"* (SHA not embedded, per the campaign convention) | [`logs/10-tk-consumer-primitive-phase.md`](logs/10-tk-consumer-primitive-phase.md) |
| 15 | [`PrimitivePhase` explicit rate](15-primitive-phase-explicit-rate.md) | §3 `[10-primitive-phase-leading-rate-is-hardcoded]` | Sonnet | ✅ | *"Give PrimitivePhase an explicit leading-rate callable"* (SHA not embedded, per the campaign convention) | [`logs/15-primitive-phase-explicit-rate.md`](logs/15-primitive-phase-explicit-rate.md) |

> Row 15 is numbered last because the campaign's numbers are append-only, but it belongs to
> Workstream D — see README §3.

### Workstream E — the numeric region

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 11 | [Numeric diagnostics and units](11-numeric-diagnostics-and-units.md) | review §10.2, §12.5, §13.1 | Opus | ⚠️ | *"Test oscillation resolution on the sample grid, off the RHS"* (SHA not embedded, per the campaign convention) | [`logs/11-numeric-diagnostics-and-units.md`](logs/11-numeric-diagnostics-and-units.md) |
| 16 | [Unresolved-osc print policy](16-unresolved-osc-print-policy.md) | §7 D2; §3 `[00-unresolved-osc-print-policy]` | Opus | ✅ | *"Summarise unresolved-oscillation warnings per wavenumber"* (SHA not embedded, per the campaign convention) | [`logs/16-unresolved-osc-print-policy.md`](logs/16-unresolved-osc-print-policy.md) |
| 12 | [Tk numeric `atol`](12-tk-numeric-atol.md) | review §12.5 | Opus | ⚠️ | *"Give the transfer-function numeric run its own absolute tolerance"* (SHA not embedded, per the campaign convention) | [`logs/12-tk-numeric-atol.md`](logs/12-tk-numeric-atol.md) |
| 17 | [Tk numeric `atol` k-sweep](17-tk-numeric-atol-k-sweep.md) | §3 `[12-tk-numeric-atol-largest-k-excursion]` | Opus | ✅ | *"Measure the transfer-function numeric tolerance across the k-grid"* (SHA not embedded, per the campaign convention) | [`logs/17-tk-numeric-atol-k-sweep.md`](logs/17-tk-numeric-atol-k-sweep.md) |
| 18 | [Numeric ODE break points](18-numeric-ode-break-points.md) | §3 `[17-qcd-reference-not-converged]` | Opus | ⚠️ | *"Split the numeric ODE at the cosmology's declared discontinuities"* (SHA not embedded, per the campaign convention) | [`logs/18-numeric-ode-break-points.md`](logs/18-numeric-ode-break-points.md) |
| 19 | [Per-sector break-point policy](19-per-sector-break-point-policy.md) | §3 `[17-qcd-reference-not-converged]` | Opus | ⚠️ | *"Let each numeric sector choose which declared break points it splits at"* (SHA not embedded, per the campaign convention) | [`logs/19-per-sector-break-point-policy.md`](logs/19-per-sector-break-point-policy.md) |
| 20 | [Key the break-point policy](20-key-the-break-point-policy.md) | §3 `[18-numeric-solver-not-in-lookup-key]` | Opus | ⚠️ | *"Put the numeric break-point policy in the datastore lookup key"* (SHA not embedded, per the campaign convention) | [`logs/20-key-the-break-point-policy.md`](logs/20-key-the-break-point-policy.md) |

> Rows 16 and 17 are numbered last because the campaign's numbers are append-only. **16 runs
> between 11 and 12** — the follow-up README §7 D2 anticipated, enacting the user's choice of
> option (ii) (2026-09-11) once prompt 11 had measured the fire rate. **17 runs after 12**, a
> measurement-only prompt approved by the user 2026-09-12: prompt 12 set the $T_k$ numeric
> `atol` on evidence from one $k$, and the tolerance is a datastore key, so the grid is
> measured before prompt 13 builds a datastore on top of it. **18 runs after 17** — the
> repair its check 6 forced — and **19 after 18**, enacting the decision prompt 18 §4
> reserved for the user (taken 2026-09-13): the reusable integrator takes the break-point
> policy as an argument, $T_k$ splitting at jumps and kinks and $G_k$ at jumps only, each
> on the evidence of `TK-NUMERIC-ATOL-SWEEP.md` §9.

### Workstream F — verification

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 13 | [Verification and docs](13-verification-and-docs.md) | review §4, §12.3, §13.5 | Opus | ⚠️ | *"Verify the Gk/Tk WKB remediation against both background models"* (SHA not embedded, per the campaign convention) | [`logs/13-verification-and-docs.md`](logs/13-verification-and-docs.md) |

**Progress:** 20 / 20 complete. **The campaign is closed**; its verification document is
[`docs/gktk-remedial-verification.md`](../../docs/gktk-remedial-verification.md).

---

## 2. Mechanism-level tracking

Traceability from each review finding to the prompt that discharges it. IDs are local to this
campaign; the review section is the authority on each.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| M1 | **DEFECT, accuracy** | Two-stage phase solver: error is a fixed fraction of the *accumulated* phase; 13.9 rad ($k=10^5$) and 7366 rad ($3\times10^8$) at $z=0.1$ on the real background (§2, §4) | 06 | ✅ the ODE is gone; $\theta=-[k\,\Delta\tau+\Delta\rho]$ from the tables. LambdaCDM $k=10^5$: 1.19e-7 rad against prompt 01's references (target 1e-5); $k=3\times10^8$: 9.77e-4 rad = one ulp of $4\times10^{12}$ rad, the representation floor (target 5e-3); QCD $3\times10^8$: 9.77e-4 (log 06) |
| M2 | **DEFECT, accuracy** | The $Q$ variable is not "close to unity" ($-224$…$-11069$); the tolerance protects the wrong quantity; DOP853 dense output amplified by $\omega_i(1+u)$ — 0.33 rad on a linear phase (§3) | 06 | ✅ no $Q$, no dense output, no tolerances: `WKB_phase_function` has no `atol`/`rtol`. Radiation control $10^7$ rad span: 3.7e-9 rad (ODE 9.7e-3); $10^9$ rad: 3.6e-7 (ODE 0.98) |
| M3 | **DEFECT, cost** | Stage 1 cost ∝ span: $2.5\times10^6$ RHS evaluations, 63.7 s per object at $k=3\times10^8$; ~13 CPU-hours per $k$ (§4) | 06, 14 | ✅ 06: **0.031 s and 6,000 integrand evaluations** per object at $k=3\times10^8$ on LambdaCDM over the full response grid (target 0.05 s), 92 % of it the per-object residual-table build. 14 removed that build: **0.0010 s and 468 evaluations** per object (4 residual + 464 leading partials), 142.5 residual evaluations per object amortised over 50 objects of one $k$ against 6,924 (LambdaCDM) and 163.0 against 7,908 (QCD) |
| M4 | **DEFECT, accuracy** | `functions.tau` is a cubic spline of RK45 nodes: $1.4\times10^{-9}$ relative, ~2 rad of *oracle* phase error at $k=10^5$ in `compute_analytic_G/T` and `QuadSourceIntegral`'s η-limits (§7, §13.2) | 03 | ✅ 3.8e-16 relative at the LambdaCDM nodes; the retired accessor measured 3.08 rad off at $k=10^5$ (log 03) |
| M5 | **REQUIREMENT** | Double-double node table and an interval accessor `tau.delta`; a pointwise accessor carries the $\varepsilon\tau$ floor ($9\times10^{-4}$ rad at $3\times10^8$) on short baselines (§13.3) | 03 | ✅ `CumulativeTable` + `TablePrimitive`; one-interval Δτ ≤ 2.5e-16 (LambdaCDM), ≤ 9.4e-15 (QCD) relative |
| M6 | **REQUIREMENT** | Persist the low-order limb (`tau_lo_Mpc`, …); regeneration attached (§13.2; README §7 D1) | 03, 04 | ✅ `tau_lo_Mpc` (03) and `cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F` (04); the factory refuses a datastore lacking any of the four by name |
| M7 | **REQUIREMENT** | Sound-horizon table $\tau_s$ and friction table $F$ per model; the friction ODE ($2.3$–$4.1\times10^{-7}$ relative error) goes (§12.2, §12.7) | 04, 07 | ✅ 04 built both tables (LambdaCDM $\tau_s$ 2.5e-16, $F$ 3.3e-16 relative at the checkpoints; QCD 2.1e-14 / 3.3e-16) and measured the ODE it replaces at 2.261e-07 absolute in $F$. 07 switched `TkWKBIntegration` onto them: `friction_RHS` and `FRICTION_INDEX` are gone from the producer (relocated verbatim into prompt 04's own test as `_friction_RHS`, where the 2.261e-07 measurement is unchanged), and `TkWKBValue.friction` is now bit-equal to `friction_F.delta(z_init, z)` — 6.5e-16 / 4.0e-16 / 2.7e-14 relative against prompt 01's references on (LambdaCDM $10^5$, LambdaCDM $3\times10^8$, QCD $3\times10^8$) |
| M8 | **REQUIREMENT** | The residual $\rho$ carried explicitly: $\le1.5\times10^{-3}$ rad for $G_k$ on QCD, $\approx-0.09$ rad for $T_k$; formed without subtraction (§6, §12.2) | 05, 14 | ✅ `ComputeTargets/phase_residual.py`: `build_phase_residual(model, k, z_nodes, sector, order)` → `CumulativeTable`; the correction comes from `*_omegaEff_sq_correction`, never from a subtraction. Worst 3.61e-16 rad against prompt 01's references over all twelve (model, sector, $k$) cases; $\rho_G$ bit-exactly zero in radiation; $\rho_T$ = −0.086 (LambdaCDM) to −0.093 (QCD). 14 added `residual_node_range` (the grid cut at `RESIDUAL_WKB_REGION_MARGIN = 0.5` of the leading term) and `cached_phase_residual`, one table per $(model, k, sector)$: $\rho$ moves by $\le1.4\times10^{-17}$ rad and $\theta$ is bit-identical |
| M9 | **REQUIREMENT** | Gauss orders decided by measurement on `QCD_Cosmology` across its spline knots; adaptive fallback for $\rho$ alone if needed (§11) | 02, 05 | ✅ 02 fixed all four orders at **4** with no adaptive fallback, but only under **break-point subdivision** on QCD; 05 consumes it — `RHO_GAUSS_ORDER = 4`, `RHO_ADAPTIVE_FALLBACK_REQUIRED = False`, and `build_phase_residual` applies the subdivision itself (1.22–1.23× the evaluations of `order × intervals` on QCD, exactly `order × intervals` on LambdaCDM) |
| M10 | **DEFECT, dead logic** | `sin_coeff` sign fix is provably always $+1$ (§8.1) | 06, 07 | ✅ 06 deleted it from `GkWKBIntegration.store()` (`sin_coeff = B`); 206 $(G,G')$ cases incl. $G=0$, $G<0$ confirm the factor was $+1$ and $B>0$ reproduces the initial data to 3e-16. 07 deleted the copy in `TkWKBIntegration.store()`; 206 $(T,T')$ cases incl. $T=0$, $T<0$ give the factor $+1$ every time and $B>0$ reproduces $T_{\rm init}$ to 8.6e-15, and the shipped `store()` returns `sin_coeff == B > 0`, `cos_coeff == 0.0` exactly |
| M11 | **DEFECT, consistency** | `shift_theta_sample` rebases `div_2pi` to the first sample, producing ±1-cycle offsets between objects (§8.1, §8.3) | 06, 07 | ✅ 06 added `WKBtools.apply_phase_offset` (per-sample wrap, no rebase) and switched `GkWKBIntegration.store()` to it; 990-object sweep: 0 rebase offsets, cycle steps only at stop-point transitions (90 = 90); the old helper would have rebased 180 (= the review's 60 of 330). 07 switched `TkWKBIntegration.store()` to it as well — there is one $T_k$ object per $k$ so no cross-object stitching arises (§12.7), but the stored $\theta+\delta$ is now exact. `shift_theta_sample` itself is retained in `WKBtools` for the two `docs/` scripts that still run (D7, log 07 deviation 5) |
| M12 | **DEFECT, hygiene** | Zero-length check compares a redshift to `atol`; 1-element array into `math.fmod` (NumPy deprecation); stale comments at `:262-264`, `:403` (§8.2) | 06 | ✅ exact test `len(z_sample) == 1 and z_sample[0] == z_init`; the `fmod` path and both comments went with the ODE; `Quadrature/supervisors/WKB.py` deleted |
| M13 | **DEFECT, accuracy + trap** | `phase_spline` chunking: no interpolation benefit, ordinates 64× inflated, knot residuals 30–50× worse, $1.4\times10^{-4}$ rad switch discontinuity, no progress guard for `logstep<2` (§5) | 08 | ✅ chunking deleted (`_build_*chunks*`, `_match_chunk`, `MINIMUM_SPLINE_DATA_POINTS` gone); one spline, rebased at the sample's *median* `theta_div_2pi`; `chunk_step`/`chunk_logstep`/`increasing` are accepted no-ops so `bessel_phase.py` and three test fixtures needed no changes. Interior interpolation error at $k=10^6$, 100/decade confirmed unchanged by chunking at 7–10e-5 rad (review 8.26e-5); the old progress-guard defect cannot recur (code path deleted) |
| M14 | **DEFECT, accuracy** | Consumers spline the growing phase: $h^4x/384$, $O(1)$–$O(10)$ rad at production $x$ (§5, §12.6). Same term as `source-remediation`'s `[12-phase-spline-error-grows-with-x]` | 09, 10 | ⚠️ 09 discharged it for $G_k$: on the review §5 consumer geometry at $k=10^8$, 100/decade, the error is **4.189e-8 rad** against the same samples' `phase_spline` at **7.286e-3 rad** — a ratio of **1.739e5**, and the 4.189e-8 is 2.81 ulp of the 9.09e7 rad span, i.e. the representation floor. $\varphi$ alone is recovered to 2.157e-10 rad, matching $h^4\max|\varphi''''|/384$. 10 discharged it for $T_k$: on a $w=1/3$ fixture at $x_T=10^6$, 100/decade, the error is **3.4482e-10 rad** against the same samples' cubic spline at **7.0854e-03 rad** — a ratio of **2.0548e7** — and the 3.4e-10 is 2.96 ulp of the $10^6$ rad phase, i.e. the `div * TWO_PI` floor. `PHASE_SPLINE_CHUNK_LOGSTEP` is deleted and no production path builds a `phase_spline`. ⚠️ because prompt 10 §3 item 3's 1e-10 on $\omega$ vs `theta_deriv` is missed at one abscissa per $w$ (`[10-residual-spline-end-condition]`) |
| M15 | **REQUIREMENT** | `PrimitivePhase`: $\theta=-k\Delta\tau+\varphi$ with the `phase_spline` protocol; closed-form $\theta'$; global anchor with the recorded floor (§7, §13.3, §13.4) | 09, 15 | ✅ `ComputeTargets/primitive_phase.py`: `PrimitivePhase(k, leading, z_anchor, z_samples, phi_samples, *, sign, model_functions, label, spline_order, rate=None)` with `raw_theta`/`theta_mod_2pi`/`theta_deriv`/`num_chunks`, plus `build_phi_samples`. `sign=-1` for $G_k$ at fixed $z_r$, `sign=+1` for $T_k$ at fixed $z_{\rm init}$ — one object for both sectors (prompt 10 reuses it, unsubclassed). $\theta'=\mathrm{sign}\,k\,\mathrm{rate}(z)+\varphi'$ in closed form, exact to 0.0 relative on the radiation control. Global anchor, floor documented in the module docstring. **Prompt 15 generalised the closed form**: `rate` is now an explicit, optional keyword-only parameter (default `1/H`, built from `model_functions.Hubble` exactly as before), and `TkSourceFunctions`' `_SoundHorizonRate` adapter — which made `model_functions.Hubble` silently return $H/c_s$ — is gone; `TkSourceFunctions` now passes the model's genuine `model_functions` plus `rate=c_s/H` explicitly. `[10-primitive-phase-leading-rate-is-hardcoded]` is **resolved** (§4) |
| M16 | **REQUIREMENT** | The `GkSource` rectifier is retained and verified inert on pure-WKB objects, correct on $\delta$-wraps (§8.3; `RECONCILIATION.md` §2 item 6) | 06, 09 | ✅ 06 documented what it must repair; 09 verified it against a faithful copy of `GkSource.py:166-233` (copied, not imported — `assemble_GkSource_values` is a Ray remote over datastore objects), on prompt 06's geometry with the source samples taken from the background grid. **Before rectification $\varphi$ jumps by exactly $+1$ cycle at every stop-point transition and nowhere else** (2 of 22 objects at $k=10^7$, $x_r=10^3$; **90 of 990** over the full sweep, log 06's figure); **the rectifier repairs every one** (90 corrections = 90 transitions) and $\varphi$ is then constant to **3.64e-12 rad**. On pure-WKB objects $\delta=0$ exactly and the rectifier makes **zero** corrections, leaving `theta_div_2pi` untouched. `GkSource.py` was not edited (D5) |
| M17 | **DEFECT, cost** | Per-RHS `*_omegaEff_sq` diagnostic: 45 % of the numeric run; the warning it feeds is live and must be preserved (§10.2, §13.1) | 11, 16 | ✅ the call is gone from both `RHS` functions; `numeric_with_phase_cut` takes an `omega_sq` callable and runs `scan_sample_grid_for_unresolved_osc` once after the solve. **The warning survives verbatim** (two `print` lines, README §2 (h)) and all three payload fields are still populated. RHS evaluations bit-identical (12,770 / 6,611 / 12,854); LambdaCDM $G_k$ object 0.1299 s → **0.0773 s**, 40.5 % (review measured 0.13 → 0.09 s). 16 **relocated** the warning as README §7 D2 option (ii) directs: the two `print` calls stay in `scan_sample_grid_for_unresolved_osc` and stay reachable, gated by a keyword-only `warn: bool = True` that defaults to on; the two production integrators pass `warn_unresolved_osc=False` and `main.py` prints one per-$k$ block per sector instead. The returned dict is asserted equal with the warning on and off |
| M18 | **DEFECT, units** | `delta_logz` supplied as $\Delta\log_{10}$, used as $\Delta\ln$; and the $G_k$ run is sampled on the response grid, not the source grid the value describes (§10.2, §13.1; `RECONCILIATION.md` §1 item 9) | 11, 16 | ✅ both cured at once: the test now uses the **actual spacing of consecutive returned samples**, so neither `delta_logz` nor which grid it describes enters it. `report_wavelength` is retained and its slip fixed in place (`(1+z) * delta_logz * LN_10`), with the parameter documented as $\Delta\log_{10}(1+z)$; it has no caller left. `main.py`'s `delta_logz=` arguments are untouched — by 11 and by 16, which does not go near them. The consequence was the D2 measurement, and **16 discharges it**: the corrected test fires on essentially every $G_k$ object, so the flag is accumulated per wavenumber and reported once per sector rather than warned per object (§4 `[00-unresolved-osc-print-policy]`) |
| M19 | **DEFECT, dead path** | `mode.lower()` before the `None` check (§10.2) | 11 | ✅ the `None` test comes first; `mode=None` integrates the whole grid, `mode="STOP"` is accepted, `mode="x"` raises `ValueError` — three tests. The `mode != "stop"` branch is kept (`RECONCILIATION.md` §3) |
| M20 | **DEFECT, comments + robustness** | The stop point is a maximum, not a minimum; the "fixed phase to avoid jitter" motivation is obsolete; `find_phase_minimum`'s $10^{-3}z$ step is safe only inside the window (§10.2) | 11 | ✅ renamed **`find_phase_extremum`** with `find_phase_minimum` kept as an alias; docstring and both integrators' comments now say maximum, and say the jitter motivation is obsolete because `store()` rotates $(G,G')$ into a pure sine. Steps $2\pi/(16\omega)$ where $\omega^2>0$, falling back to $10^{-3}z$: inside the window both steps find the same extremum, and at $x=6\times10^3$ — where the old step covers **0.955 of a cycle** — the phase step lands within 0.1 cycle of the first maximum while the old step skips more than a full cycle. Window **not** widened. The stop point moves by $\le1.04\times10^{-7}$ relative, inside `root_scalar`'s own tolerance (`[11-stop-point-root-tolerance]`) |
| M21 | **DEFECT, misleading** | `0.85·z_e6` truncation requests samples never produced in stop mode (§10.2) — comment only | 11 | ✅ both `main.py` comments (`:604-611`, `:1178-1189`) rewritten to say the ODE terminates on the $z_{e6}$ event, the `expected_values` check is skipped in stop mode, and the samples between $z_{e6}$ and $0.85z_{e6}$ are never produced. The constant stays (hand-over decision). `git diff main.py` is comment-only; the 40-of-41 return is pinned by the bit-identity test |
| M22 | **DEFECT, accuracy** | $T_k$ numeric run limited to $1.1\times10^{-5}$ of the envelope by `atol=1e-10` acting as a $10^{-5}$ relative tolerance (§12.5) | 12, 17 | ⚠️ `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` for the `TkNumericIntegration` run alone, carried by a separate `tolerance` object `Tk_numeric_atol` through all five of its `object_get` sites. On review §12.5's geometry (RadiationModel, $k=10^6$, production source grid, $T=1,T'=0$, `rtol=1e-8`, all four of the review's RHS-evaluation counts reproduced exactly): $\delta T/{\rm env}$ **9.928e-6 → 2.534e-6** (README §6 target $\le3\times10^{-6}$) for **+14.3 %** evaluations; with exact initial data **1.160e-5 → 3.275e-7**, so what remains is the $2.5\times10^{-6}$ initial-condition floor (`[00-tk-superhorizon-ic-series]`, out of scope). $G_k$ moves by 8.538e-10 of the envelope for +0.35 % evaluations and keeps `atol`. **Confirmed by the user 2026-09-12 after prompt 17's grid sweep: `1e-13` stands.** ⚠️ because at $k=3\times10^8$ the shipped tolerance leaves an isolated 2.56e-4 excursion near $x\approx10.8$ that `atol=1e-16` removes (`[12-tk-numeric-atol-largest-k-excursion]`), and because deviation 1 had to repair the batch `build_Tk_WKB_work`'s numeric lookup was dispatched over — `query_batch` (the `TkWKBIntegration` query, whole batch) rather than the unused `payload_batch` (the missing subset), which no longer works once the two carry different tolerances. **Prompt 17 measured the whole production $k$-grid** (50 wavenumbers × `RadiationModel`, `LambdaCDMModel`, `QCDModel` × `atol` ∈ {1e-10, 1e-13, 1e-16} at `rtol=1e-8`, each against a converged run of the same integrator; `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`): the excursion is real off the control and is **not** a large-$k$ effect — at the shipped tolerance **3 / 13 / 8 of 50** wavenumbers exceed $3\times10^{-6}$, worst **8.64e-4** at $k=8.37\times10^5$ on LambdaCDM, every one of them a *level* (median 1.7e-5–5.8e-5, last sample up to 2.2e-4) rather than one bad sample. `atol=1e-16` does **not** fix it (LambdaCDM 13 → 10, two of them wavenumbers `1e-13` handles) and is not cheaper on a real background. The lever is `rtol`: at fixed `atol=1e-13`, `rtol` 1e-8 → 1e-9 removes every excursion (8.64e-4 → 7.4e-8) for +23 % evaluations, and on both production backgrounds a $10^{-6}$ change in $k$ removes it too. What `1e-13` buys is the *level*, uniformly: median-over-$k$ of the per-$k$ maximum 1.25e-5 → 3.8e-7, 1.19e-5 → 4.5e-7, 1.38e-5 → 1.0e-6 for +25–30 % evaluations. **Prompt 17 recommends keeping 1e-13**; the constant is the user's call and the ⚠️ stands until it is settled. **Prompt 18 repaired the reference those QCD figures were measured against**: `numeric_with_phase_cut` splits its integration at the cosmology's declared *discontinuities*, and on `QCDModel` the $T_k$ reference-convergence drift falls from **23 of 50** wavenumbers above the 3.4e-08 criterion (worst 6.17e-06) to **3** (worst 1.97e-07 at $k=4.287\times10^6$, median 5.12e-09), the four wavenumbers prompt 17 named improving **347×–5764×**. $G_k$ **never had the failure** on any model — 1.94e-11 / 2.1e-11 / 8.41e-09 worst over the grid on Radiation / LambdaCDM / QCD, the same split or unsplit — which is the first converged-reference measurement for that sector. Nothing about `atol` or `rtol` changed. The residue is the $T(z)$ spline's $C^2$ knots, which would close it at +219 % / +155 % of the production evaluations (log 18, `TK-NUMERIC-ATOL-SWEEP.md` §9.7). **Prompt 19 paid that price in the $T_k$ sector alone** (`TK-NUMERIC-ATOL-SWEEP.md` §10): the break-point policy is now the caller's, `TkNumericIntegration` asks for `BREAK_POINT_ALL` and `GkNumericIntegration` for `BREAK_POINT_DISCONTINUITY`, and on `QCDModel` the $T_k$ drift is below the criterion at **all 50** wavenumbers — worst **8.72e-09**, median 3.96e-09, zero offenders — for **+220.23 %** of the production evaluations (9843 → 31521 per object, ~49 s for the whole 50-object sector). $G_k$ is **bit-identical**: 8.41e-09 worst on QCD at the same wavenumber and 13320 evaluations per object, the same integers as §9. The reference is now converged in both sectors on all three models, so `[17-qcd-reference-not-converged]` is **resolved** and what remains under this ID is the `rtol` question `[12-tk-numeric-atol-largest-k-excursion]` carries to `prompts/tolerance-convergence`. **Prompt 20 closed the datastore half that prompts 18 and 19 kept reporting**: `break_point_kind` is now a `nullable=False` string column on both numeric tables, filtered on in `build()` and written from the same class constant `compute()` passes to the integrator, so a row computed under one policy no longer answers a query for the other; an old-schema datastore raises rather than silently hitting; `main.py`, `Quadrature/`, `config/defaults.py` and every tolerance are untouched, and a production QCD object in both sectors is **bit-identical** to `HEAD~1`. `[18-numeric-solver-not-in-lookup-key]` is **resolved** (§4) |
| M23 | **REQUIREMENT** | Independent references and error definitions; throughput of the interval accessor measured early (§13.3, §13.5) | 01 | ⚠️ |
| M24 | **REQUIREMENT** | Verification on both models against the references; scoped pipeline run; additive docs (§13.5) | 13 | ⚠️ [`docs/gktk-remedial-verification.md`](../../docs/gktk-remedial-verification.md). **Layer 1** (`docs/gktk-remedial/verify_production_path.py`, 46 s, six sections, no Ray, no datastore): review §4 and §12.3 re-measured on both models at $k\in\{10^5,10^7,3\times10^8\}$ — at $z=0.1$ on LambdaCDM, $\theta_G$ **13.9 rad → 0.0** ($k=10^5$) and **7366 → 9.77e-04 rad** ($3\times10^8$, one ulp of 4.118e12 rad); $\theta_T$ **2.01 → 1.49e-08** and **5.1e3 → 9.16e-05 rad**; on QCD, where the review measured nothing, 4.77e-07 / 4.88e-04 and 7.45e-08 / 2.44e-04 rad. Cost per object at $3\times10^8$: $G_k$ **0.0010 s / 468 evaluations** (63.7 s / 2.54e6 before), $T_k$ **0.0181 s / 5,540** cached and 0.0511 / 11,376 cold (58 s / 1.93e6). Consumers at production $x$ (10 points per grid interval over the pure-WKB source band and the whole $T_k$ WKB region): **1.00 ulp of the span** in ten of twelve (model, $k$, sector) cases, the two exceptions being a QCD equation-of-state break point (`[13-consumer-spline-crosses-eos-break-points]`) and one whole-cycle sample (`[13-wkb-mod-2pi-cycle-count-inconsistent]`). Tables, $\rho$ and $F$ re-measured at the nodes and reproduce logs 03/04/05 to the printed digits. **Layer 2**: see the row above and the verification document §4. **Docs**: review §14 appended (nothing at or above §13 edited), `docs/spec/02-greens-function.md` §0.1 item (1) carries a dated parenthesis, `docs/OPEN_ISSUES.md` reconciled. ⚠️ because prompt 13 found a live defect it may not fix (`[13-wkb-mod-2pi-cycle-count-inconsistent]`) and because README §6's $T_k$ numeric row is still met only at the typical wavenumber (`[12-tk-numeric-atol-largest-k-excursion]`, owned by `prompts/tolerance-convergence`) |

**Out of scope (do not schedule):** the numeric→WKB hand-over (window, overlap, clamp,
$\sqrt{z_{e3}z_{e4}}$ limit — `docs/OPEN_ISSUES.md` §1.1); the $T_k$ LG truncation floor at the
hand-over (`[00-tk-lg-truncation-floor]`); raising the LG order; per-region anchoring; the
Wronskian two-solution construction; the super-horizon series initial condition; everything in
README §0.2's `transfer-remedial` file list.

---

## 3. Active and unresolved issues

Opened by the planning pass, 2026-09-10, before any prompt runs.

- **[13-scoped-run-driver-k-grid-literal]** *(opened by prompt 13, 2026-09-13; not this campaign's
  file)* — `docs/source-remediation-verification/scoped_pipeline_run.py` substitutes `main.py`'s two
  wavenumber grids by exact text match on `np.logspace(np.log10(1e5), np.log10(3e8), 50)` and
  refuses to run unless it finds exactly two occurrences. Since `f17f2d4` `main.py` spells them
  `…, NUMBER_SOURCE_K_VALUES)` and `…, NUMBER_RESPONSE_K_VALUES)` (`main.py:3094`, `:3106`), so the
  script finds **zero** and raises `RuntimeError`. **Impact:** the `source-remediation` campaign's
  Layer 2 is no longer reproducible by its own documented command. Prompt 13 copied the script to
  `docs/gktk-remedial/scoped_pipeline_run.py` with the two current literals rather than editing
  another campaign's verification driver, whose document quotes the runs it produced (log 13
  deviation 1). **Next step:** a one-line fix in the original, for whoever next has
  `docs/source-remediation-verification/` in scope — or make both copies match on the
  `np.logspace(np.log10(1e5), np.log10(3e8),` prefix rather than on the whole call.

- **[02-qcd-reference-floor]** *(opened by prompt 02, 2026-09-10; inert)* — the QCD $\tau$ and
  $\tau_s$ references in `wkb_reference_data.json` are themselves accurate only to
  **1.88e-14 / 1.89e-14 relative**, measured as the disagreement between prompt 01's
  per-production-interval `quad` and prompt 02's break-aware `quad`
  (`convergence.models.QCDModel.*.json_vs_reference_max_rel`). The order-4 cumulative errors under
  `branch+knots` are 1.89e-14 and 1.90e-14 — i.e. *at* that floor, not above it. **Impact:** a
  floor on what prompts 03 and 13 may assert for QCD $\tau$ at the nodes; it is below README §6's
  $2\times10^{-14}$ target but only just, and asserting tighter would be asserting agreement
  between two references. LambdaCDM is unaffected (mpmath at 40 digits; its floor is
  `[01-lambdacdm-hubble-rounding-floor]`). **Next step:** if 13 needs more headroom, regenerate the
  QCD block of the JSON break-aware; otherwise none.
  **Re-measured by `prompts/qcd-background-audit/` prompt 06 (2026-09-14): narrowed, not closed.**
  The half of this issue that was *circularity* is gone. The JSON's QCD block was regenerated in
  that commit against a representation that now reproduces the defining equation to 6.807e-11 max
  and 1.765e-16 median, so the `rtol=1e-14` root solve is a genuine higher-precision oracle for the
  background the references are built on, and the block's `method` string says so instead of the
  old "H(z) here is itself a spline evaluation of T(z), so no higher-precision reference exists".
  What moved, measured: the model's order-4 cumulative table now agrees with the JSON at
  **2.104e-15** ($\tau$) and **2.212e-15** ($\tau_s$) -- an order *below* this issue's recorded
  1.88e-14 / 1.89e-14 floor, where before prompt 06 it sat above it -- and the JSON's own
  self-agreement (`models.QCDModel.reference_floor`, which that campaign's generator does write) is
  `quad_epsrel_1e-12` **0.0** for both and `gauss40_bisect` 1.9875e-15 / 2.4488e-14. Both
  `QCD_FLOOR_FACTOR`s in `ComputeTargets/tests/` went back to **3.0** as a result. **The issue's own
  number cannot be re-measured without `docs/gktk-remedial/residual_convergence.py`, which writes
  the `convergence` block and which only `qcd-background-audit` prompt 08 is scoped to run
  (`[01-convergence-block-has-a-separate-generator]` on that board). Next step: that run.**

- **[01-lambdacdm-hubble-rounding-floor]** *(opened by prompt 01, 2026-09-10; inert)* — the
  double-precision evaluation of `LambdaCDM.Hubble` carries 2–9e-15 relative near $z=1$–$10^6$,
  which is the floor on $\Delta\tau$ over one production grid interval whatever the Gauss order
  (4.687e-15 at order 4 and 4.525e-15 at order 20 on the same interval) and whatever the storage
  width. In phase that is $3.8$–$6.1\times10^{-5}$ rad at $k=3\times10^8$ per interval.
  **Impact:** a floor on what prompts 03 and 13 may assert for a single-interval $\Delta\tau$ on
  LambdaCDM; it is *below* README §6's $5\times10^{-3}$ rad target and above the
  $\varepsilon k\tau$ floor, and is a different error from either. **Next step:** none; recorded
  so a later reader does not chase it.

- **[11-stop-point-root-tolerance]** *(opened by prompt 11, 2026-09-11)* — `find_phase_extremum`
  refines the sign change with `root_scalar(..., xtol=1e-6, rtol=1e-4)`, so the stop point is
  located only to $\sim10^{-4}z$ and the derivative there is $O(|G|\omega^2\cdot10^{-4}z)$, not
  zero: measured $|G'|/(|G|\omega)=9.76\times10^{-6}$ before prompt 11 and $6.52\times10^{-5}$
  after, against the $10^{-12}$ prompt 11 §3 item 1 asked to assert (deviation 1 of log 11 — the
  pre-change code misses it by six orders, so it is a property of the root finder, not of the
  change). Value/envelope is nevertheless $+1$ to $2.1\times10^{-9}$, and nothing downstream depends
  on where in the cycle the cut falls because `store()` rotates $(G,G')$ into a pure sine.
  **Impact:** the stop point is reproducible across a change of search step only to
  $\sim10^{-7}$ relative, which is why prompt 11's bit-identity test exempts it; and it is one more
  input to the hand-over campaign, since $z_{\rm init}$ is this root. **Next step:** decide with the
  hand-over campaign whether to tighten the tolerances — doing so moves every stored $z_{\rm init}$
  and forces a datastore regeneration, so it is not a free change.
  **Assigned (2026-09-12): the hand-over campaign** (`docs/OPEN_ISSUES.md` §1.1). The user accepted
  the substituted bound: prompt 11's shipped assertion is
  $|G'|<|G|\omega^2(x_{\rm tol}+r_{\rm tol}z)$, what the root tolerance permits, and **prompt 11
  §3 item 1 has been amended in place** to ask for that instead of the unachievable $10^{-12}$,
  with the pre-change measurements recorded there. No code changed. The issue stays open because
  $z_{\rm init}$ *is* this root: whether to tighten it belongs with the other six seam decisions,
  which must be taken together.

- **[00-consumer-anchoring-floor]** *(planning, 2026-09-10)* — `PrimitivePhase` reduces
  $k\Delta\tau$ against a global anchor ($z_r$ for $G_k$, $z_{\rm init}$ for $T_k$), so its
  `theta_mod_2pi` carries the $\varepsilon k\tau$ floor: $9\times10^{-4}$ rad at $k=3\times10^8$
  (review §13.4). Per-region anchoring would scale the floor with the region's own phase.
  **Impact:** Levin regions at the largest $k$; below the QCD LG floor, three to four orders below
  today's consumer error.
  **Measured by prompt 09 (2026-09-11).** The floor is now the *whole* error: on the review §5
  consumer geometry at $k=10^8$, $z_r=0.1$, $z_s\in[10,10^4]$, `raw_theta` is **4.189e-8 rad**
  from a 50-digit reference, which is **2.81 ulp** of the 9.0899e7 rad span against an
  $\varepsilon k\tau$ floor of 2.019e-8 rad and one ulp of 1.490e-8 rad; at the samples
  themselves it is 3.681e-8 rad. The residual $\varphi$ alone is recovered to **2.157e-10 rad**,
  so everything above the floor has been removed and re-anchoring is the only remaining lever.
  **Next step:** unchanged — a follow-up prompt adding an `anchored(z0)` view to `PrimitivePhase`
  and a Levin-side hook, if the verification in 13 shows the floor matters.

- **[10-transfer-remedial-tolerance-comments-stale]** *(opened by prompt 10, 2026-09-11)* — five
  tolerance comments in `ComputeTargets/tests/test_tk_source_functions.py` that
  `transfer-remedial` prompt 08 (`8ba9159`) wrote now describe a mechanism prompt 10 deleted, and
  quote numbers three to four orders above what the tests measure: the module docstring's
  "consumer re-spline … is now the binding term in `err_T`"; the `err_M` comment's "backed out
  from `M_exact` and re-splined" (measured 1.272e-13, now 1.655e-15); the `err_T` comment's
  "3.021e-08 … the h^4 cubic fit `TkSourceFunctions` puts through the sampled phase" (now
  2.079e-12); `test_phase_convention`'s "`phase_spline` rebases each chunk"; the
  `[grid refinement]` comment's "this *is* the consumer re-spline floor" (6.090e-06, now
  1.776e-10); and `test_omega_matches_phase_derivative`'s "a `phase_spline` through the exact
  integral of `omega_eff` … 4.249e-08" (now 1.049e-10). The test name
  `test_spline_error_dominates_on_the_production_grid` is also now a misnomer. **None was
  edited**: the orchestrator made `8ba9159`'s tolerance constants and comments a stop condition
  for prompt 10, and all six assertions still pass unchanged (three of them now with four extra
  orders of margin). **Impact:** anyone reading those comments to calibrate a new threshold will
  calibrate against the retired representation. **Next step:** a comments-only commit refreshing
  the six blocks with prompt 10's measured values, which needs only the user's confirmation that
  `8ba9159`'s text may be rewritten now that both campaigns have landed.

- **[10-wrap-theta-loop-at-large-phase]** *(opened by prompt 10, 2026-09-11; inert in
  production)* — `LiouvilleGreen.WKBtools.wrap_theta` (`:69-94`) range-reduces by adding
  `TWO_PI` in a `while` loop, so at $|\theta|\sim10^6$ rad it performs ~1.6e5 additions of a
  quantity $10^6$ times smaller than the accumulator and returns a pair that reconstructs
  $\theta$ only to **1.3862e-06 rad** (and costs 1.6e5 iterations). Production is unaffected:
  its only caller is `apply_phase_offset`, which passes `mod + delta` with `mod` already in
  $(-2\pi,0]$, so the loop runs at most twice. `WKB_mod_2pi` uses `fmod`, is exact, and is what
  prompt 10's $x_T=10^6$ fixture uses — **corrected by prompt 13 (2026-09-13): its *remainder* is
  exact, because that is the `fmod`; its *cycle count* was a separate rounded division and was not,
  which is `[13-wkb-mod-2pi-cycle-count-inconsistent]`. That defect was fixed by prompt 01 of
  `prompts/phase-representation` (2026-09-13, §4 below), so `WKB_mod_2pi` is now exact in both
  halves and the fixture recommendation stands without qualification. Everything else in this entry
  stands.** **Impact:** any test fixture that reduces a large
  unwrapped phase with `wrap_theta` — prompt 10's test 3.1 would have been 14× over its own
  1e-7 rad bound on the fixture's arithmetic alone. There is no warning in the docstring.
  **Next step:** a one-line note on `wrap_theta`, or an `fmod` fast path for
  $|\theta| > 2\pi$; three test modules still call it at small $|\theta|$, where it is fine.

- **[00-tk-lg-truncation-floor]** *(planning, 2026-09-10; **assigned to the hand-over campaign**)*
  — the transfer function's LG representation is not exact in radiation: $3.8\times10^{-5}$ of the
  envelope from $x_i=24$, $\sim1.4\times10^{-4}$ at the production hand-over $x_T\approx15.5$,
  scaling as $x_i^{-3}$, with a frozen amplitude offset $\sim x_i^{-4}$ (review §12.4). Below it no
  numerical improvement in this campaign is visible. Remedies — later hand-over ($x_T=50$ gives
  $4\times10^{-6}$), higher-order LG frequency, or the Bessel exact representation in the radiation
  era — are hand-over decisions. **Impact:** the $T_k$ value-level acceptance rows are floored here.
  **Measured by prompt 07 (2026-09-11)**, reproducing review §12.4 to two significant figures with
  the shipped `store()` on the exact radiation background: max $|\delta T|/\text{env}$ over
  $x_i\le x\le10^4$ is **3.8118e-05** ($x_i=24$), **4.0663e-06** (50), **5.0628e-07** (100),
  **7.7760e-09** (400), with frozen amplitude offsets 1.30e-05, 1.14e-06, 1.51e-07 and 2.39e-09 at
  $x=10^4$, and a 24/400 ratio of **4902** against $(400/24)^3=4630$ — the $x_i^{-3}$ law, now
  pinned by `test_tk_wkb_phase.TestRadiationValue`. **Next step:** none here; recorded for the
  hand-over campaign.

- **[00-tk-superhorizon-ic-series]** *(planning, 2026-09-10; inert)* — once prompt 12 lands, the
  $T_k$ numeric floor is the super-horizon initial condition $T=1,T'=0$ at $2.5\times10^{-6}$
  (review §12.5; audit TK-7), removable with the series $T\approx1-x^2/10$ for $w=1/3$ (general $w$
  needs the spec). **Impact:** a floor on what `test_tk_numeric_atol.py` may assert. **Next step:**
  a spec-level decision by the author; not scheduled.

- **[03-backgroundmodelvalue-build-path]** *(opened by prompt 03, 2026-09-11; confirmed)* — the
  `sqla_BackgroundModelValue_factory.build()` path has two latent defects on its
  query-existing-row branch (`RECONCILIATION.md` §2 item 11): the fresh-insert dict uses the key
  `"wkb_serial"` where the column is `model_serial`, and the consistency check reads
  `row_data.Hubble` where the select provides `Hubble_GeV`. Production never takes this path —
  values are inserted through `BackgroundModel.store()` — so neither has fired. Prompt 03 edited
  the neighbouring lines (adding `tau_lo_Mpc`) and did not repair them. **Impact:** anyone who
  calls `pool.object_get("BackgroundModelValue", …)` directly gets an `IntegrityError` (insert) or
  an `AttributeError` (existing row). **Next step:** a two-line fix in its own commit, with a
  test that exercises `build()` against an in-memory SQLite store.

- **[03-qcd-short-baseline-reference-endpoint-rounding]** *(opened by prompt 03, 2026-09-11;
  inert)* — the QCD short-baseline references in `wkb_reference_data.json` (`delta_tau_full`,
  `delta_tau_fraction`) were computed by `quad` in $u=\log(1+z)$ between **rounded double
  endpoints** `log1p(z)`, so each carries up to $\tfrac12{\rm ulp}(u_{\rm hi})+\tfrac12{\rm ulp}(u_{\rm lo})$
  of endpoint error relative to the baseline width $W$: bounds 7.7e-14 / 2.1e-13 (full / 37 %
  fraction at $z=10^6$), 3.9e-14 / 1.1e-13 ($z=10^2$), 9.7e-15 / 2.6e-14 ($z=1$). Measured
  disagreement of the JSON against a `quad` in the exact-width parametrisation
  $1+z=(1+z_{\rm lo})e^t$, $t\in[0,W]$: 4.4e-15 / 3.3e-14, 9.4e-15 / 2.0e-14, 1.9e-15 / 9.6e-15,
  every one inside its bound; the shipped table agrees with the exact-width `quad` to ≤ 8.8e-16
  on all six. The LambdaCDM references are mpmath at exact `mpf(float(z))` endpoints and do not
  carry this. Distinct from `[02-qcd-reference-floor]`, which is the break-unaware/break-aware
  disagreement on the *cumulative* values. **Impact:** a floor on what prompts 03, 04 and 13 may
  assert for QCD short baselines — `test_background_tau.py` asserts README §6's $10^{-13}$, not
  the JSON's recorded self-agreement (~1.6e-15). **Next step:** if 13 needs headroom, regenerate the
  QCD short-baseline records in the exact-width parametrisation; otherwise none.
  **Re-measured by `prompts/qcd-background-audit/` prompt 06 (2026-09-14): unchanged in kind, and
  not closable by a better background.** The cause is the *parametrisation of the quadrature
  limits* -- the references integrate between rounded `log1p(z)` endpoints -- so it is untouched by
  how accurately $T(z)$ is represented, and the regenerated QCD block confirms it: the three
  `short_baseline` `agreement_full` / `agreement_fraction` pairs read 1.4537e-15 / 1.3192e-15,
  1.9903e-16 / 5.3991e-16 and 0.0 / 1.6863e-16, the last two bit-identical to the values before the
  background moved. `test_background_tau`'s QCD short baselines read 4.200e-15, 9.155e-15 and
  1.746e-15 against README §6's 1e-13. **Next step: unchanged** -- regenerate the records in the
  exact-width parametrisation if headroom is ever needed.

- **[03-integrationsolver-stepping-minimum-lookup]** *(opened by prompt 03, 2026-09-11; inert)* —
  `sqla_IntegrationSolver_factory` registers `"stepping": "minimum"` and `build()` matches
  `label == label AND stepping >= stepping`, returning the first such row. Every solver registered
  before this campaign has `stepping=0`, so it never mattered; the τ table registers
  `("cumulative-GL", 4)` with the Gauss order as the stepping. If a second order is ever registered
  under the same label, a query for the lower order can be served by the higher order's row (the
  returned `IntegrationSolver` object still reports the requested stepping, but `solver_serial`
  points elsewhere). **Impact:** none while every table uses order 4 (log 02); a trap for whoever
  changes an order. **Next step:** if a second order is registered, either fold the order into the
  label or query with an exact stepping match.

- **[04-background-rhs-evaluations-count]** *(opened by prompt 04, 2026-09-11; inert)* —
  `compute_background` now builds three Gauss–Legendre tables, but `IntegrationData` is a fixed
  namedtuple with a single evaluation counter and prompt 03's `test_background_tau.test_payload_shape`
  asserts `RHS_evaluations == TAU_GAUSS_ORDER * (nodes - 1)` **exactly** on LambdaCDM. Prompt 04's
  §2 item 1 asked for the new integrand evaluations to be added to the returned `IntegrationData`;
  `test_background_tau.py` is not in prompt 04's "files you may touch", so instead
  `RHS_evaluations` still counts the $\tau$ table alone (6,924 on LambdaCDM, 8,552 on QCD) and the
  other two are reported as the payload keys `cs_tau_evaluations` and `friction_F_evaluations`
  (the same numbers again on each model, 20,772 / 25,656 in total — log 02's figures).
  `compute_time` does cover all three tables. **Impact:** the persisted `RHS_evaluations` column
  of `BackgroundModel` understates the build by 3×; anyone reading it as the job's cost is
  misled. **Next step:** if the aggregate is wanted, relax that one assertion in
  `test_background_tau.py` and sum the three counters — a two-line change in its own commit.

- **[06-metadata-column-headroom]** *(opened by prompt 06, 2026-09-11; narrowed by prompt 14)* —
  the `GkWKBIntegration`/`TkWKBIntegration` `metadata` column is
  `sqla.String(DEFAULT_STRING_LENGTH)` = `String(256)` and holds `json.dumps(obj.metadata)`.
  SQLite does not enforce the length; PostgreSQL would truncate or refuse. **Impact:** anyone
  adding a metadata key (prompt 07's friction bookkeeping is the candidate).
  **Narrowed by prompt 14 (2026-09-11):** with prompt 14's `rho_reused` key the longest payload
  is **227 characters** (LambdaCDM, $k=3\times10^8$, `Gk`, the call that builds the table), so
  **29 characters remain**; the four variants measure 226 / 222 (`Gk` built / reused) and
  223 / 219 (`Tk`), and the `initial_data_only` payload is 82. Prompt 06's 206 was the same
  payload without the key, so a key costs ~20 characters. The count is now asserted by
  `test_residual_table_reuse.TestTableIsShared.test_metadata_still_fits_the_column`, which
  fails rather than overflowing. **Next step:** count before adding; or widen the column in a
  schema-touching commit.

- **[14-rhs-evaluations-depend-on-build-order]** *(opened by prompt 14, 2026-09-11; inert)* —
  prompt 14 §2 item 3 requires `stage_1_data.RHS_evaluations` to count only the integrand
  evaluations a call actually spent, so the first object of a given $(model, k, sector)$ in a Ray
  worker stores ~7,000 and every later one a few hundred. Which object is first depends on the
  scheduler, so two runs over the same inputs can persist different `RHS_evaluations` for the
  same object. The column is payload data and part of no lookup key, so nothing misses.
  **Impact:** anyone reading the column as "the cost of this object"; prompt 13's timing of the
  scoped pipeline run should sum it over a $k$ rather than sample it. **Next step:** none, unless
  13 wants a per-object cost, in which case the table build belongs in its own counter.

- **[08-docs-scripts-reference-removed-chunking]** *(opened by prompt 08, 2026-09-11; inert)* —
  two `docs/` reproduction scripts read private internals of `phase_spline` that prompt 08 deleted
  along with chunking: `docs/gk-wkb-review-fable-2026-09-09/t5_spline.py:26` reads
  `spl._chunk_list`, `spl._splines` and calls `spl._match_chunk(...)`; `docs/gk-wkb-review-astra-pathfinder-2026-09-08/measure.py:163-164,174,176`
  reads `spl._splines` and calls `spl._match_chunk(...)`. Both scripts measured the chunked tree
  they ran on and the documents they support are correct for it; they were not edited (README §5
  "verification documents are additive"), the same treatment prompt 06 gave the phase-ODE removal
  (`[06-docs-scripts-reference-removed-ode]`, below). **Impact:** anyone re-running either script
  gets an `AttributeError` rather than the chunked-vs-unchunked comparison it printed when the
  review was written. `docs/gk-wkb-review-fable-2026-09-09/t7_jitter.py` and
  `docs/spec-code-audit/scripts/GK_05_phase_reassembly.py` also construct `phase_spline` objects
  but only through the public constructor and `raw_theta`/`theta_mod_2pi`, so they are unaffected.
  **Next step:** none; a dated note in the two review folders' READMEs if someone trips over it.

- **[06-docs-scripts-reference-removed-ode]** *(opened by prompt 06, 2026-09-11; inert)* — the
  reproduction scripts `docs/gk-wkb-review-fable-2026-09-09/{t2_solver,t4b_production_real,
  t7_jitter,t9_warn}.py`, `docs/gktk-remedial/baseline_k1e5.py` and
  `docs/gk-wkb-review-astra-pathfinder-2026-09-08/{measure,alternatives}.py` import
  `integrate_phase_function`, `stage_1_evolution`, `stage_2_evolution` or
  `DEFAULT_OMEGA_WKB_SQ_MAX`, none of which exists after prompt 06. They measured the tree they
  ran on and the documents they support are correct for it; they were not edited (README §5
  "verification documents are additive"). **Impact:** anyone re-running them gets an
  `ImportError`/`AttributeError` rather than the old ODE. **Next step:** none; a dated note in
  the two review folders' READMEs if someone trips over it. `t6_sweep.py` and
  `GK_05_phase_reassembly.py` still run (D7).
  **Widened by prompt 07 (2026-09-11):** two more scripts join them for a different removed
  symbol. `docs/spec-code-audit/scripts/TK_04_WKB_reconstruction.py:27` and
  `docs/gktk-remedial/baseline_k1e5.py:32` do `from ComputeTargets.TkWKBIntegration import
  friction_RHS`, which prompt 07 deleted; the function itself is alive and verbatim as
  `_friction_RHS` at `ComputeTargets/tests/test_background_cs_tau_friction.py:79`, so the one-line
  repair for either script is to import it from there. Prose-only references at
  `ComputeTargets/BackgroundModel.py:180`, `ComputeTargets/tests/wkb_reference.py:25` and
  `ComputeTargets/TkSourceFunctions.py:46` (prompt 10's file) now name a symbol that has moved;
  they break nothing.

- **[07-tk-per-object-cost-is-all-setup]** *(opened by prompt 07, 2026-09-11)* — a
  `TkWKBIntegration` object at $k=3\times10^8$ on LambdaCDM over the 1,384-sample source grid
  costs **0.0494–0.0516 s** across seven timed runs, straddling prompt 07 §3 item 6's
  $\le0.05$ s rather than clearing it (the ODE it replaces: 58 s). Essentially all of it is
  setup, in two halves that are independently removable:
  1. **5,840 of the 11,376 integrand evaluations (0.0321 s) build the per-$k$ residual table**,
     and nothing amortises it. Prompt 14's cache is keyed on $(model, k, sector)$ and pays off
     49× for $G_k$, where ~1,700 objects share a key; there is exactly **one**
     `TkWKBIntegration` object per $k$ (`main.py:682-712`, review §12.1), so the $T_k$ sector
     always pays the build in full. With the table cached the same call is **0.0178–0.0186 s**.
  2. **5,536 evaluations are the *leading* table's off-grid anchor panel, recomputed once per
     sample.** `WKB_phase_function` calls `leading.delta(z_init, z)` directly, so
     `CumulativeTable._locate(z_init)` goes off-grid and re-integrates the identical order-4
     panel 1,384 times. Prompt 14 split exactly this anchor off at `nearest_table_node` for the
     residual table and did not do the same for the leading one. $G_k$ pays it too (464
     evaluations on the 12×-sparser response grid, log 06).
  **Impact:** prompt 07 §3 item 6's threshold, which is not met reliably; prompt 13's cost
  measurements, which should quote both figures and say which is which; and the wall-clock of a
  production $T_k$ stage (50 wavenumbers × 2 models ≈ 5 s, so this is a tidiness issue, not a
  throughput one). **Next step:** apply prompt 14's `rho_anchor_node` split to the leading table
  in `Quadrature/integrators/WKB_phase_function.py` — one `nearest_table_node` call and one
  addition per sample, removing item 2 entirely and making the $T_k$ total ~0.034 s. Item 1 is
  irreducible without changing where the residual table's nodes come from. Neither is in prompt
  07's scope.
  **Widened to the consumer by prompt 09 (2026-09-11).** Item 2 is a property of
  `CumulativeTable.delta`, not of the producer: it re-integrates an off-grid *endpoint*'s panel on
  every call. `PrimitivePhase` escapes it only because its anchor is on-grid — `z_response` is a
  background-grid node — so a consumer whose anchor is off the grid would pay one extra order-4
  panel per evaluation on top of the abscissa's. **Prompt 10's $T_k$ anchor is $z_{\rm init}$, a
  `root_scalar` root** (`RECONCILIATION.md` §2 item 5), so it is exactly that case, and the
  `nearest_table_node` split prompt 14 applied to $\rho$ is the same one-line remedy there.
  **Next step:** unchanged for item 1; for item 2, apply the split in `WKB_phase_function` and —
  if prompt 10 anchors off-grid — inside `PrimitivePhase`.

- **[12-tk-numeric-atol-largest-k-excursion]** *(opened by prompt 12, 2026-09-12)* — the shipped
  `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` reaches the $2.5\times10^{-6}$ initial-condition floor
  at every wavenumber swept on the exact radiation control **except $k=3\times10^8$, where the
  maximum is 2.56e-4 of the envelope** — an isolated excursion over samples 340–350,
  $x\approx9.8$–12.4, worth ~8e-6 in $T$ itself. It is not a monotone tolerance floor: at
  $k=3\times10^8$ the looser `atol=1e-10` gives 1.17e-5, and at $k=10^8$ the pattern inverts
  (3.15e-4 at 1e-10, 2.49e-6 at 1e-13). `atol=1e-16` gives 2.53e-6 at every $k$ swept
  ($10^5$…$3\times10^8$) and at $k=3\times10^8$ costs **fewer** evaluations than 1e-13 (7265 against
  8483). The mechanism is consistent with the second state component: $dT/dz\sim(x^2/A)\,dT/dx$ with
  $A=k/\sqrt3$, so one absolute tolerance is a $k$-fold looser *relative* tolerance on the
  derivative and step selection near $x\sim10$ becomes erratic. **Impact:** README §6's
  $\le3\times10^{-6}$ row is met on review §12.5's geometry ($k=10^6$, 2.534e-6) and at $k=10^8$,
  but not at the top of the production $k$-grid; prompt 13's verification should not assume one
  number covers the range. Measured only on `RadiationModel` — the production backgrounds, whose
  Hubble rates are splines, were not swept. **Assigned (2026-09-12): prompt 17**, a
  measurement-only prompt approved by the user — not prompt 13, as first recorded — run before
  prompt 13 because the tolerance is a `TkNumericIntegration` row key.

  **Narrowed by prompt 17 (2026-09-12), not closed** —
  [`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`](../../docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md),
  50 wavenumbers × three models × `atol` ∈ {1e-10, 1e-13, 1e-16} at `rtol=1e-8`, each scored
  against a converged run of the same integrator; prompt 12's two control figures reproduce to
  0.02 % and 0.01 %. Four things change in this issue's picture:

  1. **It is not a large-$k$ effect and not confined to the control.** At the shipped tolerance
     **3 of 50** (Radiation), **13 of 50** (LambdaCDM) and **8 of 50** (QCD) wavenumbers exceed
     $3\times10^{-6}$; the worst is **8.64e-4 at $k=8.37\times10^5$, $x=17.4$** on LambdaCDM, not
     at the top of the grid. Almost every excursion above $10^{-5}$ falls at $x\approx8$–18, an
     e-fold above the stop window's $z_{e3}$ edge.
  2. **They are levels, not isolated samples.** The median of an offending run is lifted to
     1.7e-5–5.8e-5 (against 3e-8–7e-8 in a quiet run) and its last returned sample — what
     `TkWKBIntegration` reads — is still wrong by 8.6e-6 to 2.2e-4. Prompt 12's reading of "an
     isolated excursion" was the control's max/median ratio, not the level.
  3. **`atol=1e-16` does not fix it, and is not cheaper.** LambdaCDM 13 → **10** offenders, two of
     them wavenumbers `1e-13` handles perfectly, worst 4.78e-4; QCD 8 → 4, worst 2.49e-5 at a
     wavenumber `1e-13` handles at 2.2e-6. It is cheaper than `1e-13` at 34 of 50 radiation
     wavenumbers but only 15 of 50 on each real background, where the grid total is +0.5 % and
     +1.1 % *higher*. Prompt 12's "fewer evaluations" was one point on the control.
  4. **The lever is `rtol`, not `atol`.** At fixed `atol=1e-13`, `rtol` 1e-8 → 1e-9 removes the
     excursion at the worst $k$ of every model — 8.64e-4 → 7.37e-8 (LambdaCDM), 2.50e-4 → 1.01e-7
     (Radiation), 2.83e-4 → 9.80e-7 (QCD) — for +23 %–25 % evaluations; and on both production
     backgrounds a **$10^{-6}$ relative change in $k$** removes it as well (8.64e-4 → 3.92e-7),
     so it is an accident of the step sequence, not a property of a wavenumber. On the exact
     radiation control it does *not* move under the same perturbation, because $H=H_0(1+z)^2$
     makes the problem in $x$ $k$-independent and `atol` (acting on $T'$, whose size falls like
     $1/k$) is the only thing that breaks the scaling — which is exactly why the control made the
     excursion look like a function of $k$.

  What `1e-13` does buy is the *level*, uniformly: median-over-$k$ of the per-$k$ maximum
  1.25e-5 → 3.8e-7, 1.19e-5 → 4.5e-7, 1.38e-5 → 1.0e-6 for +24.7 %, +30.5 %, +25.0 %
  evaluations, and `1e-16` improves none of the three by a factor of three. **Impact:** README
  §6's $\le3\times10^{-6}$ row is met at the typical wavenumber on all three models but not at
  all fifty, and prompt 13 must quote the grid distribution rather than one $k$. Two limits came
  with the measurement: `QCD_Cosmology`'s discontinuous $H(z)$ (prompt 02, `RESIDUAL-CONVERGENCE.md`
  §2) stops *any* reference of this construction converging below ~6e-6 of the envelope at
  $k\in\{1.58\times10^7, 4.97\times10^7, 5.86\times10^7, 2.55\times10^8\}$, so nothing below
  that is measurable there; and the $T=1,T'=0$ initial condition holds a **$k$-independent
  2.52e-6 floor** — confirmed against the exact $T$ at all 50 $k$, and $k$-independent because the
  grid starts five e-folds outside the horizon at every $k$ ($x_i=0.00389$).
  **Settled by the user, 2026-09-12: `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` stays at `1e-13`**, as
  prompt 17 recommended, and **prompt 13 may build its datastore on it**. The constant is no longer
  in question and no prompt of this campaign revisits it.
  **Assigned (2026-09-12): [`prompts/tolerance-convergence`](../tolerance-convergence/README.md).**
  What remains open is the excursion itself, which `atol` cannot reach: the lever is `rtol`, one
  shared number keying every integration object (`main.py:2980-2998`), so it is a pipeline-wide
  retuning rather than a $T_k$ one. That campaign sweeps it per sector (its prompt 02) and
  decouples the constants (its prompt 04); the user has confirmed that the resulting proliferation
  of datastore objects is the **intended** outcome, each quantity carrying its own justified
  tolerance pair. This entry closes when that campaign settles `rtol`.

- **[20-wkb-gauss-orders-not-in-lookup-key]** *(opened by prompt 20, 2026-09-13)* — prompt 20 §5
  asked whether the other three compute targets have "any configuration axis that can vary between
  runs and is not in that key". They do. **`BackgroundModel`** keys `cosmology_type`,
  `cosmology_serial`, `atol_serial`, `rtol_serial` and its three `store_tag`s — so the node grid
  *is* keyed — but not `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER` or `FRICTION_F_GAUSS_ORDER`
  (`ComputeTargets/BackgroundModel.py:34-43`, all `= 4`), nor the quadrature break-point scheme the
  cumulative tables use. **`GkWKBIntegration`** and **`TkWKBIntegration`** key
  `wavenumber_exit_serial`, `model_serial`, `atol_serial`, `rtol_serial`, `z_source_serial` and
  `|z_init − z_init| < DEFAULT_FLOAT_PRECISION`, but not `RHO_GAUSS_ORDER = 4`
  (`ComputeTargets/phase_residual.py:90`) or `RESIDUAL_WKB_REGION_MARGIN = 0.5` (`:238`). The
  orders at least move `TAU_SOLVER_LABEL` / `PHASE_SOLVER_LABEL`, hence `solver_serial` — but no
  factory anywhere filters on a solver serial, so that provenance is never consulted. **The margin
  moves nothing at all**: no label, no tag, no column, so a change to it is invisible in every
  row. Distinct from `[03-integrationsolver-stepping-minimum-lookup]`, which is about the
  `IntegrationSolver` lookup itself rather than about who filters on its serial. **Impact:**
  latent, not live — every order is 4 today, and prompt 14 measured the margin's effect at
  $\le1.4\times10^{-17}$ rad in $\rho$ with $\theta$ bit-identical, so nothing in the tree is
  currently mis-keyed. It becomes live the moment anyone re-measures an order or the margin, which
  is exactly the situation prompts 18 and 19 created for the numeric sector.
  **Next step:** the fix is prompt 20's, applied three more times — a `nullable=False` column
  written from one class constant and filtered on in `build()` — or, for the orders alone, folding
  the order into the solver *label* and filtering on `solver_serial`. Either way it carries a
  datastore regeneration, so it belongs with whichever change first moves one of these constants.

- **[20-wkb-rows-consume-numeric-initial-data]** *(opened by prompt 20, 2026-09-13)* — the WKB
  stage takes its initial data from the numeric stop point —
  `z_init = k_exit.z_exit − Tk.stop_deltaz_subh`, `T_init = Tk.stop_T`,
  `Tprime_init = Tk.stop_Tprime` (`main.py:951-953`; the $G_k$ twin at `:1589`) — and the WKB rows
  are keyed independently of the numeric row they came from: there is no foreign key, and
  `T_init`/`Tprime_init` (`G_init`/`Gprime_init`) are stored `nullable=False` and **not** filtered
  on. What does protect them is `z_init`, which *is* filtered, but as
  `|z_init − stored| < DEFAULT_FLOAT_PRECISION = 1e-7` **absolute** against a $z_{\rm init}$ of
  order $10^{11}$–$10^{13}$ — nineteen orders below its own ulp, so in production that comparison
  is exact equality and any movement whatever in the stop point makes every downstream WKB lookup
  miss. **Measured** on `QCD_Cosmology` at $k = 4.972\times10^7$/Mpc, `BREAK_POINT_ALL` against
  `BREAK_POINT_DISCONTINUITY`: $z_{\rm init}$ moves by **4.59e+05** (4.4e-07 relative),
  $T_{\rm init}$ by 3.84e-07 and $T'_{\rm init}$ by 1.37e-22 — so the lookup misses, correctly.
  **Impact:** the protection is incidental, not designed. A change that moved the stop *values*
  while leaving $z_{\rm init}$ bit-identical would be served a stale WKB row. For prompt 13 the
  practical statement is unchanged: regenerate the whole QCD chain, which prompt 19's hand-off
  already requires. **Next step:** either record the numeric row's serial on the WKB row and
  filter on it, or filter on `G_init`/`T_init` as well as `z_init`; both carry a schema change and
  a regeneration, and neither is urgent while `z_init` is compared exactly.

> Add an entry here whenever a prompt finishes with something unresolved: a verification step that
> could not be run, an assumption that could not be confirmed, a deviation a later prompt has to
> work around, a measured cost that changes a later prompt's decision. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them, and update `docs/OPEN_ISSUES.md`.

---

## 4. Resolved issues

- **[13-consumer-spline-crosses-eos-break-points]** *(opened by prompt 13, 2026-09-13)* —
  `PrimitivePhase` splines the residual $\varphi$ with `make_interp_spline`'s default knots, which
  interpolate straight across the points where `QCD_EOS` stops being smooth. On `QCD_Cosmology` the
  consumer's worst error is at $z=4.24\times10^7$ — the `T_LO` branch boundary, where $H(z)$ jumps
  by 4.4e-04 (log 02) — in **both** sectors at $k=10^5$: **1.907e-06 rad** ($G_k$, 8 ulp of the
  span) and **3.186e-06 rad** ($T_k$, 428 ulp), against 1.00 ulp everywhere else and at every other
  wavenumber. The same non-smoothness is what makes `theta_deriv` miss $\omega$ by **2.3e-07 to
  3.3e-04 relative across the QCD interior** — uniformly, not at the ends, which is why it is *not*
  `[10-residual-spline-end-condition]` (that one is closed, §4). Contributing to the QCD figures is
  `[02-qcd-T-z-spline-node-tolerance]`, which makes $\omega^2$ itself scatter between neighbouring
  nodes at the top of the grid. **Impact:** a few $10^{-6}$ rad of consumer phase on QCD at the
  smallest wavenumbers, against a $10^{-3}$ rad Liouville–Green truncation floor on that model, so
  nothing downstream is limited by it today; it matters if anyone ever tightens the QCD phase
  claims.

  **Assigned (2026-09-13): `prompts/phase-representation` prompt 02.** **Narrowed and
  re-attributed by that prompt, 2026-09-13, which stopped rather than fixing it** (`BLOCKED`; its
  §2 item 2 and README §7 D2 reserve the decision). No production file changed. What it measured,
  at the same production geometry as §3.5 and §3.6:

  1. **The knot vector does not construct.** At `BREAK_POINT_ALL` — the kind the remedy named — a
     multiplicity-`spline_order` knot vector is **singular on all six production grids**
     (Schoenberg–Whitney fails at 3, 5, 9, 8, 10 and 14 sites), because that kind puts 226–325
     break points inside a 1,016–1,401 sample range, one per 4.5 samples, and 1–3 of the resulting
     segments hold **no sample at all**. At `BREAK_POINT_DISCONTINUITY` it constructs and is
     **2.0× worse** on $G_k$ (1.907e-06 → 3.815e-06 rad, 8 → 16 ulp) and **2.1× worse** on $T_k$
     (3.186e-06 → 6.790e-06 rad, 428 → 911 ulp) at $k=10^5$.
  2. **Why, structurally.** No declared break point coincides with a sample on any of the six grids
     (fractional position within its interval 0.0011–0.9999), and an interpolating knot vector has
     a fixed length, so the three knots a repeated knot consumes must be removed **locally** —
     Schoenberg–Whitney tolerates at most one net removal below any site. A $C^0$ knot at a break
     therefore always coarsens the spline in the intervals adjacent to it, which is where
     $\varphi$ is least smooth. The quadrature (prompts 02, 03) and the ODE (prompts 18, 19)
     escaped this because they **choose their own abscissae**; an interpolating spline is stuck
     with the samples it is given. That is the asymmetry this entry's original "next step" missed.
  3. **The kink is 1–4 % of the error it was charged with.** Fitted from the dense reference on
     each side of the `T_LO` crossing, one grid interval either side: $[\varphi'] = -5.55$e-06
     ($G_k$) and $-4.99$e-05 ($T_k$), i.e. kink terms of **1.60e-08** and **1.44e-07 rad** against
     the measured 1.907e-06 and 3.186e-06. The base spline's error near the break is a string of
     arches decaying by ~0.7 per interval, not the $(\sqrt3-2)\approx0.27$ of a point kink: the
     non-smoothness is spread over $\pm3$ grid intervals and is $\varphi$'s own structure, which
     the production sample grid does not resolve.
  4. **The `theta_deriv` separation this entry asked for, answered — and it is a third thing.** The
     two rows that miss $10^{-6}$ are QCD $G_k$ at $10^7$ and $3\times10^8$, and at those
     wavenumbers the recovered $\varphi$ spans **6.0** and **2.0 ulp** of the stored phase (7 and
     **3** distinct values over 1,218 and 1,377 samples, every one an exact multiple of
     ${\rm ulp}(\theta)$). Differentiating that staircase is **3× and 10× worse than contributing
     nothing**: 5.9403e-06 and 1.7475e-04 with the $\varphi$ spline derivative, 1.9982e-06 and
     1.7392e-05 without it. So the miss is neither the knots' (which recover at most 38 %, at
     $k=10^5$ only, and only by doubling the phase error) nor
     `[02-qcd-T-z-spline-node-tolerance]`'s — it is the $\varepsilon k\tau$ storage granularity of
     $\varphi$, `[00-consumer-anchoring-floor]`, opened separately as
     `[02-consumer-phi-below-the-storage-granularity]` on the
     [`phase-representation` board](../phase-representation/IMPLEMENTATION_STATE.md) §3.

  **Next step (rewritten):** not a knot vector. Either accept this as a property of the production
  sample grid and mark it inert at the ~$10^{-3}$ rad QCD Liouville–Green truncation floor, 300×
  above the worst figure; or attack the **grid** — split the production source grid at
  `integration_break_points` so a break is resolved from both sides, which is a `main.py` /
  `BackgroundModel` change and a different campaign; or take `[00-consumer-anchoring-floor]` first,
  since item 4 shows it, not the knots, is what limits `theta_deriv` at $k\ge10^7$. Measurements
  and the ranked options are in
  [`prompts/phase-representation/logs/02-primitive-phase-break-point-knots.md`](../phase-representation/logs/02-primitive-phase-break-point-knots.md).
  Note that `docs/gktk-remedial/verify_production_path.py` builds its own $G_k$ consumer, so half
  of §3.5 cannot show a fix routed through `GkSourcePolicyData`
  (`[02-verify-script-builds-its-own-Gk-consumer]`).

  **Superseded (2026-09-13, orchestrator close-out of `prompts/phase-representation`).** None of
  the three options above was taken, because
  [`docs/qcd-background-audit-2026-09.md`](../../docs/qcd-background-audit-2026-09.md) identifies a
  fourth that prompt 02 could not have seen: **404 of the 407 `BREAK_POINT_ALL` points are knots of
  the `T(z)` spline itself** — the uniform lattice of an auxiliary 500-point interpolant in
  `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:189`, spaced 4.04 grid intervals apart, not a
  feature of the cosmology. Remove that artefact and `BREAK_POINT_ALL` falls to the **3** genuine
  crossings, at which point item 1's Schoenberg–Whitney failure cannot occur and a $C^0$ knot
  vector constructs trivially. This also revises item 3: the "±3 grid interval" structure prompt 02
  attributed to $\varphi$ itself has the same 4.04-interval scale as that knot lattice, and the
  audit measures the `T(z)` interpolation error that produces it at up to 7.18e-04 relative
  (median 1.89e-07), so most of it is very likely `[02-qcd-T-z-spline-node-tolerance]` after all —
  testable by rebuilding `T(z)` and re-measuring. **This entry now belongs to the
  `qcd-background` campaign**, whose audit §8 schedules the re-run of prompt 02 after the
  representation and the grid are fixed.

  **Narrowed — the blocker is gone (`prompts/qcd-background-audit` prompt 07, 2026-09-14). The
  defect is not.** `integration_break_points` no longer declares the `T(z)` tabulation's knots:
  `BREAK_POINT_ALL` on the production source grid is **3** and `BREAK_POINT_DISCONTINUITY` **2**,
  and none of them is a knot of anything. Item 1's Schoenberg–Whitney failure is therefore
  **measured to be gone**: a multiplicity-`spline_order` knot vector **constructs on all six** of
  the (sector, $k$) geometries prompt 02 measured as singular, and the same construction with the
  knot lattice restored still raises `LinAlgError` on all six — both asserted, in the tree, by
  `ComputeTargets/tests/test_numeric_break_points.py::TestConsumerKnotVectorConstructs`, which also
  carries the six bands and the local-payment knot construction prompt 10 needs.

  What this does **not** settle is whether a $C^0$ knot at a break is the right representation for
  $\varphi$, which is the substance of items 2–4 and the reason prompt 02 stopped: items 3 and 4
  stand as measured, and item 4's finding — that the $k\ge10^7$ `theta_deriv` miss is the storage
  granularity of $\varphi$ and not the knots — is untouched by anything the background campaign
  does. Prompt 07 also removes the second half of item 3's supersession argument: the "±3 grid
  interval" structure can no longer be attributed to a 4.04-interval knot lattice, because the
  interpolation error that produced it is gone (`T(z)` max 7.18e-04 → 6.807e-11, prompt 06), so
  whatever remains at the `T_LO` crossing is $\varphi$'s own. **Owned by
  `prompts/qcd-background-audit` prompt 10**, which is now unblocked.

  **Narrowed again, and the numbers are larger than the ones above
  (`prompts/qcd-background-audit` prompt 09, 2026-09-14).** The close-out re-ran
  `docs/gktk-remedial/verify_production_path.py` unedited at the campaign base and at the corrected
  background. **This entry's own two rows rose**, and they are now the campaign's one
  un-discharged consequence:

  | | this entry / §3.5 | after the corrected background | |
  |---|---|---|---|
  | $G_k$, $k=10^5$, at $z=4.24\times10^7$ | 1.907e-06 rad (8.00 ulp) | **8.1062e-06 rad (34.00 ulp)** | 4.25× |
  | $T_k$, $k=10^5$, at $z=4.24\times10^7$ | 3.186e-06 rad (427.60 ulp) | **1.3982e-05 rad (1876.61 ulp)** | 4.39× |
  | `theta_deriv`, $G_k$, $k=10^5$ | 5.128e-07 | **1.0232e-06** | 2.00× |
  | `theta_deriv`, $T_k$, $k=10^5$ | 1.100e-06 | **3.0630e-06** | 2.79× |

  **The supersession paragraph's testable prediction is falsified, and the right way round.** It
  said the "±3 grid interval" structure was "very likely `[02-qcd-T-z-spline-node-tolerance]` after
  all — testable by rebuilding `T(z)` and re-measuring". `T(z)` has been rebuilt (max 7.18e-04 →
  6.807e-11, $\int\mathrm{d}z/H$ now bit-identical to an independent exact background) and the
  error **went up**, because the old representation was *hiding* the defect rather than causing it.
  Measured with `docs/qcd-background-audit/consumer_break_point_profile.py` on both trees, the
  equation of state's step in $\mathrm{d}\ln H/\mathrm{d}u$ at the `T_LO` crossing:

  | on production-grid spacing, 25 intervals | pre-campaign | corrected |
  |---|---|---|
  | peak deviation from the local median | +3.071237e-02 | **+8.544581e-02** |
  | total $\sum\lvert$deviation$\rvert$ | 1.201851e-01 | **8.557937e-02** |
  | share inside the crossing's own interval | **25.55 %** | **99.84 %** |
  | intervals carrying > 10 % of the peak | **8** | **1** |

  The 500-node order-3 `T`-against-`u` spline's knot spacing was 4.04× the production grid, so it
  smeared the genuine step over about four grid intervals and presented only a quarter of it inside
  any one of them, with ±3e-03 of its own scatter either side. The corrected background delivers a
  step 2.78× taller with 99.84 % of it inside the single interval the consumer's cubic must span.
  So item 3's "±3 grid interval" structure was the knot lattice's after all — but removing it makes
  the *consumer* worse, not better, because what is left is the cosmology's own discontinuity
  undiluted. **Item 3's kink fit (1.60e-08 / 1.44e-07 rad, 1 % and 4 % of the error) should be
  re-taken by prompt 10 before it designs anything**: it was fitted on the smeared step.

  Still 70–120× below the ~1e-03 rad QCD Liouville–Green truncation floor, where it was 300–500×
  below; still above README §6's 1e-06 rad consumer target, as it was before. Prompt 09 is a
  verification prompt and repaired nothing. Record:
  [`docs/qcd-background-verification.md`](../../docs/qcd-background-verification.md) §3.3 and
  [`prompts/qcd-background-audit/logs/09-close-out-verification.md`](../qcd-background-audit/logs/09-close-out-verification.md).

  **CLOSED on measurement (`prompts/qcd-background-audit` prompt 10, 2026-09-15). The remedy this
  entry names is refuted; the defect it measures is re-opened under a correct name.** Prompt 10
  re-took the measurement on the corrected background with
  [`docs/qcd-background-audit/consumer_knot_scheme_scan.py`](../../docs/qcd-background-audit/consumer_knot_scheme_scan.py)
  (one command, ~120 s, no Ray, no datastore), scoring **nine schemes** over all twelve
  (model, sector, $k$) rows — the two constructions `qcd-background-audit` prompt 10 §2 names plus
  four non-$C^0$ controls — against a `base` column that reproduces §3.5 and §3.6 to every printed
  digit. **No production file was changed.**

  | at QCD $k=10^5$ | $G_k$ | $T_k$ |
  |---|---|---|
  | shipped default knots | 8.1062e-06 rad (34.00 ulp) | 1.3982e-05 rad (1876.61 ulp) |
  | **repeated multiplicity-3 ($C^0$) knot** — this entry's own "next step" | 1.6928e-05 (71.00) — **2.09× worse** | 2.9315e-05 (3934.6) — **2.10× worse** |
  | **per-segment splines** | 4.0531e-05 (170.0) — **5.00× worse** | 7.0770e-05 (9498.6) — **5.06× worse** |
  | best of the four controls (multiplicity 1 or 2) | 6.6757e-06 (28.00) — 1.21× better | 1.1528e-05 (1547.2) — 1.21× better |

  All six LambdaCDM rows are identical under every scheme, and — new since prompt 02, and prompt
  07's doing — **the ten rows at 1.00 ulp stay at 1.00 ulp under every scheme**, `ALLx1` included,
  because `BREAK_POINT_ALL` is now 1–3 points on those grids rather than 226–325. So the family was
  scored without the trap prompt 02 warned of, and still nothing reaches the 1e-06 rad / 2 ulp
  target or even returns to this entry's original 1.907e-06 / 3.186e-06 rad.

  **Item 3's kink fit, re-taken as this entry required, says why.** One-sided cubics on $\varphi$
  from the dense reference either side of `T_LO`, over windows of 1, 2 and 3 grid intervals:
  $[\varphi'] = +7.5232$e-05, $-3.0026$e-03, $-6.1529$e-03 ($G_k$) and $-2.1692$e-04,
  $+5.1925$e-03, $+1.0634$e-02 ($T_k$). **A genuine slope discontinuity gives a window-independent
  jump; this moves two orders and changes sign**, on a background whose $T(z)$ error has fallen
  7.18e-04 → 6.807e-11. Item 3's original reading — smooth-but-unresolved data on a ±3-interval
  scale, $\varphi$'s own — is therefore **confirmed**, and the supersession paragraph's attribution
  of it to the `T(z)` knot lattice is wrong in both directions: the lattice was hiding it, not
  making it.

  **Item 4 is confirmed unchanged.** At QCD $G_k$ $k=10^7$ and $3\times10^8$ the recovered
  $\varphi$ still spans **6.0** and **2.0 ulp** of the stored phase, **no scheme moves either
  `theta_deriv` figure by a printed digit**, and removing $\varphi'$ altogether *improves* them by
  1.53× and 15.9×. The break points' share of `theta_deriv` is **35.8 % / 35.5 %**, at $k=10^5$
  only, bought at 2.09×/2.10× of consumer phase — and QCD $T_k$ at $3\times10^8$ is a **1.70×
  regression** under both $C^0$ schemes.

  **What does work is the sample grid**, which is the third of this entry's own rewritten next
  steps. A plain cubic of the same $\varphi$, default knots, given more samples: refining the ±5
  grid intervals around the crossing by 2× — **10 extra samples in 1,016, 1.0 %** — takes the two
  rows to **1.64 ulp** (3.9121e-07 rad) and **74.60 ulp** (5.5584e-07 rad), both inside the 1e-06
  rad consumer target; a uniform 2× gives 1.48 and 72.76 ulp. Refining the crossing's *own* interval
  alone buys 1.96× and stalls, because the feature is 3–5 grid intervals wide. That is
  `prompts/phase-representation/IMPLEMENTATION_STATE.md` §5 note 6 — *a knot vector cannot resolve a
  break the sample grid does not resolve* — established on the corrected background rather than
  inherited.

  **Where it goes.** The accuracy defect is unfixed and is **not lost**: it re-opens as
  `[10-consumer-phi-unresolved-at-the-eos-crossing]` on the
  [`qcd-background-audit` board](../qcd-background-audit/IMPLEMENTATION_STATE.md) §3, assigned to
  **prompt 11**, which owns the source grid (**G2**). This entry is closed because what it *names* —
  `PrimitivePhase` splining across the declared break points, and a repeated-knot vector as the
  remedy — has now been measured on both backgrounds, at all three wavenumbers, in both sectors and
  across the whole constructible family, and there is nothing left to try under that framing.
  `PrimitivePhase` keeps `make_interp_spline`'s default knots, on measurement. Record:
  [`docs/qcd-background-verification.md`](../../docs/qcd-background-verification.md) §8 and
  [`prompts/qcd-background-audit/logs/10-primitive-phase-break-points.md`](../qcd-background-audit/logs/10-primitive-phase-break-points.md).

- **[19-cosmologymodels-docstrings-predate-per-sector-policy]** *(opened by prompt 19,
  2026-09-13; not this campaign's to fix; **resolved by `prompts/qcd-background-audit` prompt 07**,
  2026-09-14)* — two `CosmologyModels/` docstrings stated as fact what prompt 19 had measured to be
  true of one numeric sector and not the other: `GenericEOS.py`'s
  `discontinuity_temperatures_GeV` said "an *adaptive* ODE solver only has to be split at a jump,
  because a C2 point does not invalidate an embedded Runge-Kutta error estimator", and
  `LambdaCDM_GenericEOS.integration_break_points` said the jumps-only set "is what
  `Quadrature/integrators/numeric_with_phase_cut.py` asks for". Prompt 19 had measured that the C2
  knots cost about an order of magnitude of reference convergence in the $T_k$ sector
  (1.97e-07 → 8.72e-09 worst over the grid), so `TkNumericIntegration` asks for `BREAK_POINT_ALL`:
  a C2 point does not invalidate the estimator but it does degrade it. Documentation only
  throughout; prompt 19 §1 put `CosmologyModels/` out of bounds.

  **Rewritten by prompt 07 of the `qcd-background-audit` campaign**, which had that directory in
  scope because it was changing the declaration itself. Both texts now say that the *kind* is the
  consumer's choice, taken on measurement rather than on principle, and point at
  `numeric_with_phase_cut.py`'s module docstring for which sector asks for which. Four further
  texts were corrected in the same pass for the same reason — the `BREAK_POINT_*` comment block and
  `break_temperatures_GeV` in `GenericEOS.py`, `_cosmology_break_points` in
  `ComputeTargets/BackgroundModel.py`, and three paragraphs of `numeric_with_phase_cut.py` — all of
  which described the `T(z)` spline's knot lattice as part of what the cosmology declares. It is
  not, as of that prompt: `BREAK_POINT_ALL` is **3** points on `QCD_Cosmology`'s production range
  and `BREAK_POINT_DISCONTINUITY` **2**, both crossings of an equation-of-state branch temperature.
  Every rewritten text carries that prompt's own measurement. Full record:
  [`prompts/qcd-background-audit/logs/07-rederive-break-points.md`](../qcd-background-audit/logs/07-rederive-break-points.md).

- **[02-qcd-T-z-spline-node-tolerance]** *(opened by prompt 02, 2026-09-10; not this campaign's;
  **resolved by `prompts/qcd-background-audit` prompt 04**, 2026-09-14)* —
  `LambdaCDM_GenericEOS._solve_T_z` solved $T\,g_S(T)^{1/3}=$ const with
  `root_scalar(..., xtol=1e-6, rtol=1e-4)`, and the 500 node values of `QCD_Cosmology`'s $T(z)$
  spline inherited that: adjacent nodes converged independently, so the interpolant scattered
  between them. **Fixed** by tightening to `xtol=1e-300, rtol=1e-14` — one line, build-time cost
  only (`~13.4 ms` for 500 nodes, against a 100 ms stop threshold; `T_photon`'s per-call cost is
  unchanged at 2.408 µs, since the evaluation path is untouched). **Measured, before → after**, on
  the audit's 640-point probe set (`docs/qcd-background-audit-2026-09.md` §3;
  `CosmologyModels/tests/T_z_reference.py`): node solve against the `rtol=1e-14` defining-equation
  oracle, max **2.496e-05 → 0.0** (bit-identical — the shipped solve now uses the same tolerance
  and algorithm as the reference); $H(z)$ built from the node solve alone (not the spline), same
  probe set, max **7.016e-05 → 0.0**. The full `T(z)` spline (500 pts, still one global spline —
  T3/T4 are prompts 05/06) improves from max/p90/median **7.177e-04/1.323e-05/1.890e-07** to
  **7.2615e-04/1.936e-07/1.071e-07**: accurate nodes fix the p90 as the audit's §4 table predicts;
  the max is untouched (even nudges up slightly) because it is pinned at the un-segmented jump
  height, not by node accuracy. **The $\pm0.1$-scale $\omega^2/\omega_0^2$ scatter this issue was
  blamed for in `ComputeTargets/phase_residual.py:226`'s comment (`RESIDUAL_WKB_REGION_MARGIN`'s
  reason for existing) survives essentially unchanged**: 11 production nodes near $z\sim4\times
  10^{15}$ at $k=3\times10^8$ (Green's function) span $\omega^2/\omega_0^2\in[-0.0713,+0.2336]$
  before and $[-0.0676,+0.2376]$ after — the same span to 0.1 %. **So this issue is not (or not
  primarily) the scatter's cause**; `RESIDUAL_WKB_REGION_MARGIN = 0.5` is not narrowed by this
  prompt and `[20-wkb-gauss-orders-not-in-lookup-key]` is untouched. The QCD reference fixture was
  regenerated in the same commit (largest relative move: `rho_G` 1.10e-02, `rho_T` 3.01e-03,
  `tau_minus_top`/`cs_tau_minus_top` ~1.8e-05, `friction_F_minus_top` 2.05e-08); every dependent
  test re-scored green except two whose expected values were pinned to the shipped nodes'
  crossing location and were updated in the same commit (a hardcoded closed-form-style literal in
  `test_numeric_break_points.py`, and an alignment tolerance against the JSON's separate
  `convergence` block in `test_background_tau.py`, loosened from 1e-9 to 1.4e-05 — attributable to
  `[01-convergence-block-has-a-separate-generator]` on the `qcd-background-audit` board, not a new
  defect). See `prompts/qcd-background-audit/logs/04-tighten-node-solve.md`.

- **[13-wkb-mod-2pi-cycle-count-inconsistent]** *(opened by prompt 13, 2026-09-13; **resolved by
  prompt 01 of [`prompts/phase-representation`](../phase-representation/IMPLEMENTATION_STATE.md)**,
  2026-09-13)* — `LiouvilleGreen/WKBtools.py`'s `WKB_mod_2pi` took its remainder from an exact
  `fmod` but its cycle count from `int(floor(fabs(theta) / TWO_PI))`, a correctly-rounded division:
  when the exact quotient sat within half an ulp *below* an integer the division rounded up across
  it, `floor` returned one cycle too many, and the stored pair reconstructed $\theta-2\pi$.
  `simple_mod_2pi` shared the construction and the defect. **Fixed** by deriving the count from the
  exact remainder — `int(round((fabs(theta) - fabs(theta_mod_2pi)) / TWO_PI))`, whose argument is
  within $n\cdot2^{-52}$ of the true integer $n=\lfloor|\theta|/2\pi\rfloor$ and so rounds to it
  exactly while $|\theta| < 2\pi\cdot2^{51}\approx1.4\times10^{16}$, against a production maximum of
  $\approx5\times10^{12}$. Both functions keep their own (deliberately different) remainder
  conventions, and the remainder itself is bit-identical, so **no stored $G$ or $T$ moved**.
  **Measured**, `docs/gktk-remedial/verify_production_path.py` unedited, before → after: LambdaCDM
  $G_k$ at $k=3\times10^8$ **1 → 0 inconsistent of 77,975**; the eleven other (model, sector, $k$)
  rows 0 → 0; the uniform control at $|\theta|\sim4\times10^{12}$ **25 → 0 of 400,000**, emptying
  the whole 6.104e-05-cycle half-ulp band. At the consumer, the LambdaCDM $k=3\times10^8$
  Green's-function row falls from **6.1748 rad (12,646 ulp of the span) at $z=33{,}296$ to exactly
  0.0000e+00 rad (0.00 ulp)**, and `theta_deriv` against $\omega$ on that case from 5.6245e-08 to
  1.8060e-13. Nothing else in the script's output moved but wall-clock. Cost: 0.2254 → 0.2731 µs
  per reduction (+48 ns), once per stored sample. **Datastore:** `theta_div_2pi` is
  `nullable=False` and in no lookup key, so a pre-fix datastore is served silently with the old
  count at the affected samples; no migration was invented and none is recommended — see
  `prompts/phase-representation/logs/01-wkb-mod-2pi-cycle-count.md`, "State handed to the next
  prompt", for the regeneration list. **This also corrected one clause of
  `[10-wrap-theta-loop-at-large-phase]`**, in §3 above.

- **[01-offgrid-accessor-cost-on-qcd]** *(opened by prompt 01, 2026-09-10; narrowed by prompts 03,
  06 and 09; **resolved by prompt 13**, 2026-09-13)* — the entry's recorded closing condition was
  "a Levin-side measurement on `QCD_Cosmology` (prompt 13) shows the per-call cost acceptable in
  bulk". Measured: `PrimitivePhase.raw_theta` at **4,000 abscissae drawn uniformly in
  $u=\log(1+z)$ across the production band** — the Levin evaluation pattern, off-grid by
  construction — costs **29.3 µs per call and 4.27 integrand evaluations on `QCD_Cosmology`**
  (6.86 µs / 4.00 on LambdaCDM), best of 3; on-grid it is **3.56 µs and exactly 0 evaluations**,
  because the anchor `z_response` is a background-grid node and costs nothing. That is **below
  README §4.3's 50 µs stop threshold**, and it is the one-endpoint-off-grid column prompt 09
  predicted, not the both-off-grid 52–65 µs this issue was opened against. The shipped accessor's
  three-way cost on QCD re-measured for the same document: **0.33 / 33.3 / 64.7 µs** (on-grid / one
  off / both off), against log 03's 0.33 / 26.1 / 52.0. **Closed**: no production or consumer path
  evaluates both endpoints off-grid, and the pattern that does evaluate off-grid is affordable in
  bulk. See [`docs/gktk-remedial-verification.md`](../../docs/gktk-remedial-verification.md) §3.8.

- **[14-residual-range-top-margin]** *(opened by prompt 14, 2026-09-11; corrected by the
  orchestrator; **resolved by prompt 13**, 2026-09-13)* — the entry's recorded next step was
  "prompt 13 measures the cut-to-anchor margin across the production $k$ range on both models and
  both sectors as part of its verification, and records it; if it is ever below ~1 e-fold, the
  margin constant is what to revisit". Measured over the **50-point production $k$ grid**,
  $10^5$–$3\times10^8$/Mpc, both models, both sectors:

  | model | sector | worst cut/anchor | e-folds | at $k$ [1/Mpc] | cut $z$ | anchor $z$ | fewest nodes |
  |---|---|---|---|---|---|---|---|
  | LambdaCDM | `Gk` | 2981 | 8.000 | 3.000e8 | 2.064e16 | 6.923e12 | 1732 |
  | LambdaCDM | `Tk` | **5.668** | **1.735** | 9.704e6 | 1.269e12 | 2.239e11 | 1113 |
  | QCD | `Gk` | 47.48 | 3.860 | 1.160e6 | 1.392e12 | 2.931e10 | 1243 |
  | QCD | `Tk` | **5.666** | **1.735** | 3.696e5 | 5.286e10 | 9.329e9 | 1117 |

  The tightest margin anywhere on the production grid is **1.735 e-folds**, in the `Tk` sector on
  both models, so log 14's single-wavenumber 1.74 generalises and nothing approaches one e-fold.
  **Closed**, with the figure recorded so that a change to `RESIDUAL_WKB_REGION_MARGIN`, to the
  production grid or to a cosmology can be checked against it — which is what "no test pins either
  statement" asked for. The measurement is reproducible as
  `docs/gktk-remedial/verify_production_path.py --section margins` (4 s).

- **[10-residual-spline-end-condition]** *(opened by prompt 10, 2026-09-11; left open for prompt 13
  by the user, 2026-09-11; **resolved by prompt 13**, 2026-09-13, at the cubic)* — the entry's
  recorded next step was "prompt 13 re-measures this identity on the real background for both
  sectors and then either closes this issue at the cubic or escalates to `spline_order=5` for both
  consumers with `MIN_SPLINE_DATA_POINTS` raised to 6. It should also report the error at the second
  and third samples, not only the window maxima, since the end effect is what is in question."

  **Measured on the real background**, $\omega(z)$ from `*_omegaEff_sq` against
  `phase.theta_deriv(z)`, over each consumer's own production sample set:

  | model | $k$ | sector | max, all | `[3:-3]` | `[3:-5]` | `[3:-8]` | deep interior |
  |---|---|---|---|---|---|---|---|
  | LambdaCDM | 1e5 | `Gk` | 5.503e-10 | 5.135e-10 | 4.904e-10 | 4.577e-10 | 4.371e-10 |
  | LambdaCDM | 1e5 | `Tk` | 1.764e-08 | 3.031e-10 | **1.663e-11** | 1.020e-11 | 8.828e-12 |
  | LambdaCDM | 1e7 | `Tk` | 1.681e-08 | 2.890e-10 | 1.580e-11 | 9.689e-12 | 8.389e-12 |
  | LambdaCDM | 3e8 | `Tk` | 1.709e-08 | 2.938e-10 | 1.608e-11 | 9.865e-12 | 8.538e-12 |

  The windows are the shipped test's, read in its sense: `z_WKB` is in redshift order, so `[3:-3]`
  and `[5:-3]` trim 3 and 5 samples from the **high-$z$ (hand-over) end**, which is where the end
  condition lives; the columns above trim 3 from the low-$z$ end throughout and $m$ from the high-$z$
  end.

  **The end effect is real and it decays as the issue says**: the last five `Tk` samples at the
  hand-over end, where $\varphi\sim-1/x$ varies fastest, run **9.2e-11, 2.9e-10, 1.2e-09, 4.5e-09,
  1.7e-08** — a factor ~3.8 per sample inwards, against the ~3× prompt 10 measured on its
  closed-form fixture — and the deep interior is 8.5e-12. Against the two bounds prompt 10 shipped:
  **`< 1e-9` over `[3:-3]` is met**, at 2.9–3.0e-10 (the fixture's 1.0492e-10 for the same window is
  the 4.9 % miss of prompt 10 §3 item 3's 1e-10 this issue was opened for; the real background is 3×
  larger there and comfortably inside the shipped bound). **`< 1e-11` over `[5:-3]` is missed by
  1.7×** on the real background, at 1.66e-11; it is met from about the seventh sample in (1.02e-11
  at `[3:-8]`), and the deep interior is 8.5e-12. The shipped assertion is on the *fixture*, where
  it passes — the whole suite passes unchanged, 339 + 141 tests — so what this says is that the real
  background's residual is a little less smooth near the hand-over than the constant-$w$ closed form
  is, by under a factor two. **`spline_order=5` is not taken**: it would need six samples against
  `TkSourceFunctions.MIN_SPLINE_DATA_POINTS = 5`, it would make the $T_k$ consumer's representation
  differ from prompt 09's $G_k$ consumer, and — decisively — it does not touch the error that
  actually dominates on the real background, which is the next paragraph.

  **What the measurement did turn up is a different defect, and it is carried forward, not
  closed**: on `QCD_Cosmology` the same identity is missed by 2.3e-07 to 3.3e-04 relative — but
  **uniformly across the interior, not at the ends** (`[3:-3]`, `[3:-8]` and the deep interior are
  the same number), so it is not an end condition and no spline order fixes it. It is the
  cosmology's own non-smoothness reaching the residual spline, and it is
  `[13-consumer-spline-crosses-eos-break-points]`, now in §4 — closed by
  `prompts/qcd-background-audit` prompt 10 on the measurement that no knot vector touches it and
  re-opened against the *sample grid* as
  `[10-consumer-phi-unresolved-at-the-eos-crossing]`.

- **[18-numeric-solver-not-in-lookup-key]** *(opened by prompt 18, 2026-09-13; second instance
  recorded by prompt 19; diagnosis corrected and the decision taken by the user 2026-09-13;
  **resolved by prompt 20**, 2026-09-13)* — the two numeric lookup keys carried no record of *how*
  the integration was done. `Datastore/SQL/ObjectFactories/GkNumericIntegration.py:221-227` and
  `TkNumericIntegration.py:225-231` filtered on `validated`, `wavenumber_exit_serial`,
  `model_serial`, `atol_serial` and `rtol_serial` (plus the source/init redshift when supplied) and
  on nothing else, so a `QCDModel` row computed before prompt 18 was indistinguishable by key from
  one computed after it while differing by up to 2.82e-04 of the envelope, and prompt 19's
  per-sector policy moved `TkNumericIntegration` a further 1.61e-04, again invisibly.

  **The entry's original next step was wrong, and is recorded here so that it is not
  re-proposed.** Adding `solver_serial` to the two queries is a **no-op**: the solver cannot vary.
  `numeric_with_phase_cut` hard-codes `method="DOP853"` (lines 358 and 658), takes no solver or
  method argument and returns the constant label `"solve_ivp+DOP853-stepping0"` (line 823);
  `main.py:3033`'s `solvers` dict is never indexed in `main.py`, only passed whole to six
  `object_get` sites and consumed *after* the compute as
  `self._solver_labels[data["solver_label"]]`, so `RK45`, `Radau`, `BDF` and `LSODA` are
  registered and unreachable. Every numeric row ever stored therefore points at one and the same
  `IntegrationSolver` serial, and filtering on it would exclude nothing while making the query
  read as though it were sound. Nor can `IntegrationSolver` carry the policy: its own lookup is
  `label == label AND stepping >= stepping` (`integration_metadata.py:32`), an ordered *quality*
  comparison, and a break-point policy is categorical.

  **What was unkeyed was the configuration, and prompt 20 keyed it.** Each numeric compute target
  declares its policy once — `GkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_DISCONTINUITY`,
  `TkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_ALL` — and `compute()`, `store()` and
  `build()` all read that one declaration; both numeric tables gain a `nullable=False`
  `break_point_kind` string column after `rtol_serial`, **filtered on** in `build()`. A row
  computed under the other policy now misses. A datastore written before the commit has no such
  column and raises a `RuntimeError` naming the prompt and demanding regeneration — no default, no
  migration — on the `BackgroundModel.py:300-309` pattern prompts 03 and 04 set. The solver stays
  as provenance, untouched. **No computed value moved**: a production QCD object in both sectors at
  $k = 4.972\times10^7$/Mpc is bit-identical to `HEAD~1` (sha256 `565e907…`, 494 + 41 samples,
  32,351 + 13,258 RHS evaluations), and `Quadrature/`, `main.py`, `config/defaults.py`,
  `CosmologyModels/`, every tolerance and every `solver_serial` are untouched. The wider audit the
  entry called for is `[20-wkb-gauss-orders-not-in-lookup-key]` and
  `[20-wkb-rows-consume-numeric-initial-data]` in §3.

- **[17-qcd-reference-not-converged]** *(opened by prompt 17, 2026-09-12; narrowed by prompt 18, 2026-09-13; **resolved by prompt 19**, 2026-09-13)* — prompt 17's §2.1
  convergence test **fails on `QCDModel`**: at $k\in\{1.58\times10^7, 4.97\times10^7,
  5.86\times10^7, 2.55\times10^8\}$ two runs of the same integrator a decade apart in tolerance
  differ by 1.6e-6–6.2e-6 of the envelope, against a smallest reported candidate difference of
  3.45e-7 — so the criterion (drift $\le$ a tenth of that, i.e. $\le$3.4e-8) is missed by more
  than an order of magnitude and the sweep is partly measuring its own reference there. **Cause:**
  `QCD_Cosmology`'s $H(z)$ jumps at the `QCD_EOS` branch boundaries (4.4e-4 at `T_LO`, 1.0e-4 at
  `T_120_MEV`; prompt 02, `RESIDUAL-CONVERGENCE.md` §2), and `numeric_with_phase_cut` integrates
  straight across them in one `solve_ivp` call. DOP853's embedded error estimator is invalid across
  a discontinuous RHS: the method drops to first order, so a decade of tolerance buys ~25 % and the
  refinement is not even monotone (measured at $k=10^8$: `(1e-19,1e-13)` moves the reference by
  1.03e-6, `(1e-20,1e-14)` by 7.6e-9), and SciPy clamps `rtol` at 2.22e-14. **Impact:** nothing
  below ~1e-5 of the envelope is measurable on QCD at those $k$ by any tolerance; and because
  `GkNumericIntegration` runs through the same driver, the same failure is expected for $G_k$,
  where **no converged-reference measurement has ever been taken on any model** (prompt 12's $G_k$
  figure was candidate-against-candidate on the radiation control; review §10.1 used the radiation
  oracle on radiation and LambdaCDM). Prompt 17 reported with the drift carried as a per-$k$ column
  and conclusions drawn only 57×–175× above it; the orchestrator stopped on it. **Next step:
  assigned (2026-09-12) to prompt 18**, approved by the user — wire the declaration protocol prompt
  02 already built (`GenericEOS.break_temperatures_GeV` → `integration_break_points` →
  `_cosmology_break_points`, default "smooth", no equation-of-state knowledge in any consumer) to
  the ODE, splitting only at **jumps**: the 404 $T(z)$ spline knots in range are $C^2$ points an
  adaptive stepper absorbs, and only the 3 temperature crossings break the estimator, so the
  declaration must distinguish the two kinds. Prompt 18 **runs before prompt 13**: it changes
  computed values on QCD in both sectors, and `solver_serial` is not part of the
  `GkNumericIntegration` lookup key.

  **Narrowed by prompt 18 (2026-09-13), not closed** —
  [`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`](../../docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md)
  §9, 50 wavenumbers × three models × **both sectors**, split and unsplit, in 740 s.
  `numeric_with_phase_cut` now asks the cosmology for its declared *discontinuities* and
  integrates the segments between them in sequence. Four things change in this issue's picture:

  1. **Three of §4's four failures were the jumps; the fourth was where the boundary is put.**
     All four named wavenumbers are fixed — 1.60e-06 → 4.60e-09 (347×), 2.76e-06 → 6.48e-09
     (426×), 6.17e-06 → 7.94e-09 (777×), 5.29e-06 → 9.18e-10 (5764×). The fourth needed
     `BREAK_POINT_STANDOFF`: a segment ending *exactly* on the crossing evaluates its final
     Runge–Kutta stage there, and which branch of the equation of state answers is a rounding coin
     flip, so the boundary is placed 1e-12 relative on the near side. With it on the crossing,
     $k=4.972\times10^7$ stays at 2.99e-06; on the far side its two neighbours break symmetrically
     (§9.6).
  2. **The criterion is still missed at 3 of 50 QCD $T_k$ wavenumbers** — $k=8.366\times10^5$
     (6.08e-08), $4.287\times10^6$ (**1.97e-07**) and $4.223\times10^7$ (3.46e-08), 1.8× to 5.8×
     the 3.4e-08 line, against 23 offenders and up to 180× before. Each is stable across four
     tightening pairs, so it is genuine non-convergence and not noise in the estimator.
  3. **$G_k$ never had the failure.** First converged-reference figures for that sector, on its
     own production geometry: worst drift **1.94e-11** (Radiation), **2.1e-11** (LambdaCDM),
     **8.41e-09** (QCD), none above the criterion, the QCD column the same split or unsplit.
     Consistent with review §12.5: `atol` binds for $T_k$ and never for $G_k$.
  4. **The residue is the $T(z)$ spline's $C^2$ knots, and closing it is a cost decision.**
     Splitting at all 407 declared break points instead of the 3 jumps takes the three offenders
     to 6.89e-10, 4.65e-09 and 2.30e-09 — all inside the criterion — at **+218.8 %** ($T_k$) and
     **+154.8 %** ($G_k$) of the production right-hand-side evaluations (§9.7). Prompt 18 §4
     forbids doing it without asking, so it was measured and not done.

  **Cost of what shipped:** +1.38 % ($T_k$) and +0.39 % ($G_k$) per QCD object — one extra
  restart, `T_120_MEV` being the only declared discontinuity inside any production numeric range.
  **Impact on prompt 13:** the split moves the production answer on QCD by up to **2.82e-04** of
  the envelope ($T_k$, $k=1.584\times10^7$; $G_k$ worst 1.77e-06), so **a QCD datastore built
  before this commit may not be used**; LambdaCDM and Radiation rows are bit-identical.
  **Next step:** the user decides whether to split at the $C^2$ knots as well. Closes if they do,
  or if they accept a 1.97e-07 floor at three QCD wavenumbers.

  **Resolution (prompt 19, 2026-09-13)** —
  [`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`](../../docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md)
  §10, 50 wavenumbers x three models x both sectors in 574 s. The user ruled (2026-09-13) that the
  cosmology declares all of its potential non-smoothness and each consumer decides what to do with
  it, so `numeric_with_phase_cut` gained a `break_point_kind` and the two sectors ask for different
  things: **$T_k$ for `BREAK_POINT_ALL`, $G_k$ for `BREAK_POINT_DISCONTINUITY`**, each on its own
  measurement and each named at the call site so that neither looks inherited by omission.

  1. **The criterion is met at all 50 QCD $T_k$ wavenumbers.** Worst drift **8.72e-09** (at
     $k=10^5$), median 3.96e-09, **zero** above 3.4e-08 — against 1.97e-07 and three offenders
     with the jumps alone. §9.7 had measured only the three offenders; the other 47 are not
     disturbed, and the three reproduce §9.7's figures exactly (6.89e-10, 4.65e-09, 2.30e-09).
  2. **$G_k$ is bit-identical**, which is what says the shared driver did not change behaviour for
     the sector that did not ask: 1.94e-11 / 2.1e-11 / 8.41e-09 worst drift at the same
     wavenumbers as §9.1, **13320** evaluations per QCD object (§9.3's integer), +0.00 % over the
     grid, and at all 50 wavenumbers the run with the argument omitted equals the run with it
     named sample for sample.
  3. **The cost is where the decision put it.** $T_k$ on QCD 9843 -> **31521** evaluations per
     object (**+220.23 %**), which at 50 objects per model is ~49 s for the whole sector; $G_k$
     unchanged, so the ~65,000 objects per model pay nothing.
  4. **The price is a second movement of the QCD $T_k$ answer**, up to **1.61e-04** of the
     envelope on top of prompt 18's 2.82e-04 — `[18-numeric-solver-not-in-lookup-key]`, which
     stays open and stays the user's. `GkNumericIntegration` rows are bit-identical on all three
     models and need no regeneration.

  **Closed**, on the first branch of prompt 18's "closes if they do": the user chose to split at
  the $C^2$ knots, in the sector where measurement said it was necessary. `prompts/tolerance-convergence`
  is unblocked without a QCD caveat — both sectors now converge at all 50 wavenumbers on all three
  models.

- **[00-unresolved-osc-print-policy]** *(planning, 2026-09-10; decided by the user 2026-09-11;
  resolved by prompt 16, 2026-09-12)* — README §7 D2. With the $\ln10$ slip fixed and the test
  evaluated against the caller's actual sample grid (prompt 11), `has_unresolved_osc` fires on
  **2,149 of 2,149** $G_k$-like objects, first at $x=26.5$–66.6, and **0 of 6** $T_k$-like runs,
  which peak at 0.807–0.822 of the trip threshold; today's rate is 0 of 2,155. The warning is
  **two** printed lines, so keeping it per object would turn 0 printed lines into
  $\sim1.3\times10^5$ per model (~65,000 `GkNumericIntegration` objects). **Decision (user,
  2026-09-11): option (ii)** — store the flag, print a per-$k$ summary in `main.py`. Option (iii)
  ("pass the intended grid explicitly") was rejected because it would test $G_k$ against a grid it
  is not sampled on, suppressing the signal and undoing prompt 11's faithful semantics; option (i)
  is the print storm.
  **Resolution (prompt 16).** The warning is **relocated, not deleted** (README §2 (h)):
  `scan_sample_grid_for_unresolved_osc` gains a keyword-only `warn: bool = True` gating **only**
  its two `print` calls, and `numeric_with_phase_cut` a `warn_unresolved_osc: bool = True`
  appended last in its signature; the test, the failing-pair rule and all three returned values
  are bit-identical either way (asserted). Both production integrators pass
  `warn_unresolved_osc=False`; every other caller — tests, the `docs/` reproduction scripts, a
  future integrator — keeps today's behaviour by default. `main.py` gains module-level
  `record_unresolved_osc(summary, obj)` and
  `format_unresolved_osc_summary(summary, sector_label) -> List[str]`, and both numeric work
  queues gain `post_handler=lambda obj: record_unresolved_osc(<acc>, obj)` — the only seam, since
  both run `store_results=False` and retain no objects — plus a summary printed after `.run()`.
  Per flagging wavenumber: the counts and the range of `unresolved_efolds_subh`, the depth inside
  the horizon at which the grid first failed. Production prints ~52 lines per model instead of
  ~$1.3\times10^5$; an `ast` guard in `test_main_plumbing.py` fails if a later edit drops either
  `post_handler` or either formatter call, which is the silent failure this wiring is exposed to.
  **Nothing stored changed**: payload keys, both Datastore factories and the persisted columns are
  untouched, and a datastore written before the commit is readable after it.
  **What this does *not* close.** Only *where the information is printed*. The **semantic**
  question the flag now detects is untouched and is owned by the hand-over campaign
  (`docs/OPEN_ISSUES.md` §1.1): the flag trips at $x=19.74$ against $x=e^3=20.09$ at the
  stop-search window's floor, so it is `False` before the numeric→WKB hand-over window and `True`
  across all of it — the same condition as `[05-numeric-region-is-now-the-accuracy-floor]` and
  `[06-source-spline-residual-vs-handover]`. Whether the response grid should resolve the mode
  through the seam is that campaign's question, and the fire rate is an output of its design, not
  a knob to tune here.
  **Loose end, recorded rather than opened.** `NumericIntegrationSupervisor.report_wavelength`
  still has no caller. Prompt 11 kept it, corrected, because this issue owned the decision, noting
  that under option (iii) it is where the per-step form goes back and that under option (ii) "it
  should probably go". Option (ii) landed, but `Quadrature/supervisors/numeric.py` is outside
  prompt 16's file list — a one-line deletion for prompt 13's clean-up or a later tidy, not a new
  §3 row. See [`logs/16-unresolved-osc-print-policy.md`](logs/16-unresolved-osc-print-policy.md).

- **[10-primitive-phase-leading-rate-is-hardcoded]** *(opened by prompt 10, 2026-09-11; resolved by
  prompt 15, 2026-09-11)* — `PrimitivePhase.theta_deriv` computed the leading derivative as
  `sign * k / model_functions.Hubble(z)`, correct for the Green's function's $\tau$ but wrong by
  $1/c_s$ for the transfer function's $\tau_s$; prompt 10 worked around it with a
  `TkSourceFunctions._SoundHorizonRate` adapter that made `.Hubble` silently return $H/c_s$.
  **Resolution:** `PrimitivePhase.__init__` gained an explicit, optional, keyword-only `rate`
  parameter — `d/dz[leading.delta(z, z_anchor)]`'s magnitude — defaulting to `None`, in which
  case `lambda z: 1.0 / model_functions.Hubble(z)` is built exactly as before, so every call site
  that predates this parameter (`GkSourcePolicyData.py`'s two, confirmed unedited by `git diff`)
  needs no change and sees the same value; `theta_deriv` now computes
  `sign * k * rate(z) + phi'(z)`. `TkSourceFunctions._SoundHorizonRate` is deleted; a module-level
  `_sound_horizon_rate(functions)` closure carries the same positive-$c_s^2$ `RuntimeError` guard
  and is passed as `rate`, while `model_functions=self._model.functions` (the real one) is passed
  unmodified, so `phase._Hubble` on a $T_k$ `PrimitivePhase` is now genuinely $H(z)$.
  **Measured:** the reordered expression (`k*(c_s/H)` vs. the old `k/(H/c_s)`) differs by
  **2.218e-16 relative** ($w=1/3$) and **2.191e-16** ($w=0.2$) at every stored sample of prompt
  10's `test_omega_matches_phase_derivative_from_the_primitive` fixture — machine epsilon, six-plus
  orders below its 1e-9/1e-11 window — and that test's own figures are unchanged to the last
  printed digit (1.0492e-10 / 5.559e-12 at $w=1/3$, 7.9502e-11 / 3.757e-12 at $w=0.2$). See
  [`logs/15-primitive-phase-explicit-rate.md`](logs/15-primitive-phase-explicit-rate.md).

- **[10-quadsource-fixture-model-substitution]** *(opened by prompt 10, 2026-09-11; resolved by the
  user, 2026-09-11)* — prompt 10 edited five lines of stand-in construction in
  `ComputeTargets/tests/test_quadsource_integral.py` (`Case.__init__`'s non-`exact`
  `Tk_builder`), which is **outside its "Files you may touch" list**. That module builds
  `TkSourceFunctions` objects from inputs captured out of `Fixture.exact_functions()` and then
  supplies its own `FakeModel(w)`; now that the friction comes from
  `model.functions.friction_F`, the two disagree by **5.6231e-06** in $F$ (the LG truncation of
  the exact envelope), so five of its 37 tests raised the new cross-check and two more failed on
  message text. The fix substitutes `Fq/Fr.exact_envelope_model()` per wavenumber, exactly as
  the file's own `exact` branch one line above already does; no tolerance, threshold or
  production file is touched. It cannot be done from inside the allowed files, because
  `Case.model` is shared with the Green's-function fixtures and `compute_QuadSource_integral`
  calls `Tk_functions_builder(model, k, …)` with its own model, so only the builder closure can
  substitute per $k$. With the cross-check temporarily disabled the module passes unedited — so
  the check is reporting a genuine fixture defect (its "realistic" transfer functions would
  silently become LG-amplitude rather than exact-envelope), not merely failing on a technicality.
  **Impact:** the prompt's §4 acceptance ("`test_quadsource_integral.py` passes") cannot be met
  without it; the campaign's own §4.3 file stop-list does not name this file. **Resolution (user, 2026-09-11):** the hunk is
  **accepted**. The orchestrator had confirmed the diagnosis independently by restoring the file
  to its pre-prompt content and re-running the module — four `RuntimeError`s from
  `_check_friction_samples` at `TkSourceFunctions.py:437` — so the alternative was to weaken a
  cross-check the prompt's own §1 requires in order to protect a fixture that was quietly
  inconsistent. Prompt 10's "Files you may touch" list was extended in place with a dated note
  giving this reasoning, and `wkb_reference.py` was added to it at the same time, that file having
  always been in scope in substance (§2 names both it and the `ClosedFormPrimitive` helper to put
  there). Both remain **stand-in construction only**.

- **[00-transfer-remedial-test-file-overlap]** *(planning, 2026-09-10; resolved by prompt 10,
  2026-09-11)* — prompt 10 edits the stand-in `ModelFunctions` fixtures in
  `ComputeTargets/tests/test_tk_source_functions.py` and `test_phase_groups.py`;
  `transfer-remedial` prompt 08 edits tolerance constants and comments in the same files, so
  either order risked a textual conflict. **Decision (user, 2026-09-10):** Workstreams A, B, C, E
  run in parallel with `transfer-remedial`; Workstream D waits for the merge (README §4.2 item 1).
  **Merge confirmed by the orchestrator, 2026-09-11 (Workstream C close-out):** `transfer-remedial`
  (all nine prompts) and `qsi-phase-groups` landed in `gktk-remedial` at **`e01c31d`**, whose
  message records that only `docs/OPEN_ISSUES.md` conflicted, resolved as the union; its prompt 08
  is **`8ba9159`** ("Tighten the Bessel tests to the new accuracy"), the last commit to touch
  either shared file.
  **Resolution (prompt 10).** No conflict of any kind arose, and the overlap turned out to be
  smaller than planned: **`test_phase_groups.py` needed no edit at all** — it imports `FakeModel`
  and `Fixture` from `test_tk_source_functions` and never builds a `ModelFunctions` itself, so
  upgrading `FakeModel` was enough, and `git diff` leaves that file byte-identical to `8ba9159`
  (18 tests OK). In `test_tk_source_functions.py`, `git blame` attributes 87 lines to `8ba9159`
  and **prompt 10 modified none of them**; every line it removed is `e3348e4`'s. All six
  assertions `8ba9159` set still pass, three of them with four extra orders of margin
  (`err_M` 1.272e-13 → 1.655e-15, `err_T` 3.021e-08 → 2.079e-12, `[grid refinement]` 6.090e-06 →
  1.776e-10). What did *not* survive is the accuracy of `8ba9159`'s *explanatory* text, which is
  now stale in five places — carried forward as `[10-transfer-remedial-tolerance-comments-stale]`
  (§3) rather than fixed, because editing it was a stop condition for prompt 10. A *third* module
  turned out to construct the consumer as well, which the planning pass did not anticipate:
  `[10-quadsource-fixture-model-substitution]` (§3).

- **[09-consumer-threshold-below-representation-floor]** *(opened by prompt 09, 2026-09-11;
  resolved by the orchestrator, 2026-09-11)* — prompt 09 §4 `test_primitive_phase.py` test 1 asked
  for a consumer phase error $\le10^{-8}$ rad on the geometry it names in the same sentence
  ($k=10^8$, $z_r=0.1$, $z_s\in[10,10^4]$, exact radiation, 100/decade), and called that
  "README §6's consumer row". It is not: README §6's row is $\le10^{-6}$ rad, and $10^{-8}$ rad is
  **below the double-precision floor of the quantity being asserted**. On that geometry $|\theta|$
  reaches 9.0899e7 rad, one ulp of which is 1.490e-8 rad and whose $\varepsilon k\tau$ floor is
  2.019e-8 rad, so the threshold was **0.67 ulp**. The measured error is **4.189e-8 rad = 2.81
  ulp** (3.681e-8 rad at the samples, where the spline contributes nothing); the companion ratio
  assertion passed as written at **1.739e5** against the required $10^5$. Prompt 09 asserted
  README §6's $10^{-6}$ rad plus a floor-aware 6-ulp bound instead and said so in the module
  docstring.

  **Resolution.** An independent review by Claude Fable 5.1
  ([`reviews/09-prompt-09-review-fable.md`](reviews/09-prompt-09-review-fable.md), commissioned by
  the user after the orchestrator stopped) confirmed the arithmetic, decomposed the achieved
  2.81 ulp into 2.39 ulp from `TablePrimitive.delta`'s double return and 0.50 ulp from the
  `k*delta` product, and traced the defect's origin: prompt 08's test 1 is the same $z_s$ band at
  $k=10^6$ (where $10^{-8}$ rad is 86 ulp) and README §6's prompt-06 radiation control is at a
  $10^7$ rad span (5.4 ulp), so both sources make $10^{-8}$ look right; prompt 09 scaled the $k$
  by 100 to reproduce the review's $8.26\times10^{-3}$ rad `phase_spline` figure and did not scale
  the tolerance. Its verdict was **accept, amend the prompt text**. The review also notes that
  "unreachable in double precision" overstates slightly — a correctly-rounded double would be
  0.5 ulp — but reaching that needs `delta` to return a double-double and a compensated product,
  which is prompt 03's file and would be chasing a number below the campaign's own declared
  $\varepsilon k\tau$ floor (§5 note 2). **Prompt 09's §4 test 1 text was therefore corrected in
  place**, with a dated note recording what it said before and why it changed. No code changed:
  the shipped test already asserts the corrected bounds. See
  [`logs/09-gk-consumer-primitive-phase.md`](logs/09-gk-consumer-primitive-phase.md) deviation 1
  and its orchestrator addendum.

- **[06-residual-table-per-object]** *(opened by prompt 06, 2026-09-11; resolved by prompt 14,
  2026-09-11)* — `WKB_phase_function` built the residual table on every call, although it depends
  only on `(model, k, sector)`: 5,536 of the 6,000 integrand evaluations and ~92 % of the 0.031 s
  per object at $k=3\times10^8$ on LambdaCDM, 5–7k evaluations and 82–255 ms on `QCD_Cosmology`,
  times ~1,700 source redshifts per $k$. **Resolution:** `residual_node_range` fixes the nodes
  from $(model, k, sector)$ alone — the background grid, cut at the top where the frequency stops
  keeping half its leading term (`[14-residual-range-top-margin]`) — and `cached_phase_residual`
  memoises one `CumulativeTable` per key in the worker (LRU, 256 entries, 0.395 MB each, ~40 MB
  for a production run's 100 keys). The object's anchor is off that grid and is reached by one
  `CumulativeTable.delta` partial **per object**, split at the nearest node, rather than per
  sample. Measured over 50 objects of one $k$ at $k=3\times10^8$: **142.5 residual-integrand
  evaluations per object on LambdaCDM against 6,924, and 163.0 on QCD against 7,908 — 49× on
  both**; the second and every later object adds nothing to the build (exactly 0 for an on-grid
  anchor, exactly 4 for an off-grid one), and wall time per object falls from 0.0309 s to
  **0.0010 s**. Nothing moved: $\theta$ is **bit-identical** at every sample of fifteen
  (model, $k$, sector) cases and $\rho$ moves by at most $1.4\times10^{-17}$ rad. Every §3.1
  accuracy, cost and sweep threshold of prompt 06 passes at its published number. Prompt 07
  inherits the reuse with no change. See
  [`logs/14-residual-table-reuse.md`](logs/14-residual-table-reuse.md).

- **[02-cosmology-break-point-api]** *(opened by prompt 02, 2026-09-10; resolved by prompt 03,
  2026-09-11)* — the table builders needed a public way to ask a cosmology for its break points.
  **Resolution:** `LambdaCDM_GenericEOS.integration_break_points(z_lo, z_hi) -> np.ndarray` returns
  the ascending $u=\log(1+z)$ values strictly inside the range at which any background quantity
  loses smoothness: the interior knots of the $T(z)$ spline (kept at build as
  `_T_z_spline_knots_log1pz`) plus the crossings of `GenericEOSBase.break_temperatures_GeV` (a new
  property, `()` by default; `QCD_EOS` returns `(T_LO, EOS_T_LO, T_120_MEV, T_HI)`), each solved in
  $u$ by `root_scalar` to `xtol=rtol=1e-15`. `compute_background` and `_create_functions` reach it
  duck-typed through `_cosmology_break_points(cosmology, z_lo, z_hi)`, empty when the cosmology has
  no such method (`LambdaCDM`, the stand-ins). On the production grid `QCD_Cosmology` reports 407
  points (404 knots + 3 crossings), each crossing within 1e-9 of log 02's values. See
  [`logs/03-tau-primitive.md`](logs/03-tau-primitive.md).

- **[00-tau-storage-decision]** *(planning, 2026-09-10; confirmed by the user 2026-09-10; resolved
  by prompt 03, 2026-09-11)* — README §7 D1 implemented: `BackgroundModelValue` gains
  `tau_lo_Mpc` (`Float(64)`, `nullable=False`) after `tau_Mpc`, which is now the high limb; both
  limbs round-trip exactly in `Mpc_units` (asserted for every node of both models). A datastore
  lacking the column is refused by `sqla_BackgroundModelFactory.build()` with a message naming the
  regeneration. The `cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F` columns are prompt 04's.

- **[01-qcd-eos-branch-boundaries]** *(opened by prompt 01, 2026-09-10; resolved by prompt 02,
  2026-09-10)* — no fixed-order Gauss–Legendre rule converged on the production intervals of
  `QCD_Cosmology` containing a `QCD_EOS.G(T)` branch boundary: 6.33e-5 relative per interval at
  order 4, falling only as $N^{-2}$, i.e. 4.78 rad at $k=3\times10^8$. **Resolution:** neither of
  the two options the issue named. Prompt 02 measured all three schemes and found (i) a third break
  point, the `EOS_T_LO` clamp in `QCD_EOS.w`, which kinks $c_s^2$ at $z=1.187\times10^{10}$;
  (ii) that **adaptive quadrature is not needed for $\rho$ at all** — $\rho$ is small enough that
  even an unconverged rule delivers it to $10^{-9}$ rad, and what actually fails under `plain` is
  the *leading* $\tau$ term; and (iii) that splitting each production interval at the three
  temperature crossings **and the 404 $T(z)$-spline knots** puts order 4 at the floor everywhere —
  $\tau$ 1.89e-14 relative, 0 of 1,731 intervals above $10^{-12}$ — for 24 % more integrand
  evaluations. Splitting at the temperatures alone is not enough (1.81e-13, 25× the floor); the
  knots are the load-bearing half. Orders: $N_\tau=N_{\tau_s}=N_F=N_\rho=4$.
  See [`logs/02-qcd-residual-convergence.md`](logs/02-qcd-residual-convergence.md) and
  [`docs/gktk-remedial/RESIDUAL-CONVERGENCE.md`](../../docs/gktk-remedial/RESIDUAL-CONVERGENCE.md).

---

## 5. Standing notes for implementers

1. **`RECONCILIATION.md` outranks the review** where they differ; the differences are listed in its
   §2 and each is carried by a named prompt. A *new* conflict must be recorded and, if
   load-bearing, stopped on.
2. **The floors are not targets.** $\varepsilon k\tau$ ($3\times10^{-7}$–$9\times10^{-4}$ rad),
   the LG truncation ($10^{-8}$ LambdaCDM, $10^{-3}$ QCD for $G_k$; $1.4\times10^{-4}$ of the
   envelope for $T_k$ at the hand-over), and the $T_k$ initial condition ($2.5\times10^{-6}$).
   A test asserting below a floor is asserting agreement between two errors.
3. **Never form $C=\omega^2-(k/H)^2$** by subtraction; use the correction functions prompt 05
   exposes — `Gk_omegaEff_sq_correction` / `Tk_omegaEff_sq_correction`, both $k$-independent, both
   in `ComputeTargets/WKB_{Gk,Tk}.py`. Never form $\Delta\tau$ as `tau(b) - tau(a)`; use
   `tau.delta(a, b)`.
4. **`*_omegaEff_sq` return values are stored columns** and must not move by a bit. Delivered by
   prompt 05, which kept the `A + B + C` summation order and asserts exact equality against
   verbatim copies of the pre-refactor bodies
   (`test_omega_eff_split.TestReturnValueUnchanged`). Note that `leading + correction` is **not**
   bit-equal to `*_omegaEff_sq` — floating-point addition is not associative and the two differ
   by one ulp on ~13 % of the production range (log 05, deviation 1) — so do not "simplify"
   `*_omegaEff_sq` to that sum.
5. **`ModelFunctions` stand-ins must keep constructing** with thirteen positional arguments.
   Delivered by prompt 04: `cs_tau` and `friction_F` are appended as fields 14 and 15 with
   namedtuple `defaults=(None, None)`, asserted by
   `test_background_cs_tau_friction.test_model_functions_still_constructs_with_thirteen_positional_arguments`.
   **Delivered by prompt 10 for the transfer-function fixtures:** `test_tk_source_functions.FakeModel`
   now supplies both as `wkb_reference.ClosedFormPrimitive` objects (`cs_tau = sqrt(w) tau`,
   `friction_F = (3/2)(1+w) log(1+z)`), and `test_phase_groups` inherits them by importing
   `FakeModel`. A `ModelFunctions` that leaves either at `None` is now refused **by name** by
   `TkSourceFunctions.__init__`, so a stand-in that reaches the transfer-function consumer must
   supply both. `ClosedFormPrimitive.delta` is `f(b) - f(a)`, which is acceptable only in a
   fixture — see its docstring and README §2 (c).
6. **The `GkSource` rectifier stays** (D5). Its logic is a stop condition.
7. **`phase_spline`'s signature is frozen** (D4); `bessel_phase` on `main` and three fixtures
   depend on it.
8. **Solver labels must be registered in `main.py`** (`:2810-2834` after the `transfer-remedial`
   merge) or `store()` raises `KeyError` in production only. Prompt 03 registers the τ table as
   `BackgroundModel.TAU_SOLVER_LABEL` = `"cumulative-GL-stepping4"` through the class attributes
   `TAU_SOLVER_LABEL_BASE` / `TAU_GAUSS_ORDER`, so `main.py` needs no new import.
9. **Tolerances are datastore lookup keys**; a missed `atol=` site makes lookups miss silently
   (prompt 12).
10. **Author conventions** (README §5 rule 6): $a_0$ absorbed; $\tau=a_0\eta$; unit-jump $\bar G_k$;
    $\theta<0$ decreasing with $\theta_{\rm mod}\in(-2\pi,0]$; `tau_init`'s radiation-era closed
    form; $c_s^2=$ `wPerturbations`. Do not "correct" any of these.
11. **`main.py` cannot be imported**; extract functions with `ast` (`test_main_plumbing.load_main_py_functions`).
12. **Redshift arithmetic** (`CLAUDE.md`): node lookup by exact `z`; a recovered $z$ from
    $\log(1+z)$ is only ever a quadrature endpoint.
14. **Wall-clock durations on the development machine are unreliable; CPU time and best-of-N
    are not.** It sleeps in transport, and `unittest`'s "Ran N tests in T s" is elapsed time: a
    `LiouvilleGreen/tests` run at Workstream C close-out reported 1654.9 s against ~1080 s of CPU
    (`ps -o pid,etime,time`), a 53 % overstatement. This bites **duration** figures — suite times,
    build times, the `compute_time` a payload records. It does **not** bite the per-object **cost**
    acceptance rows of README §6, which are best-of-N minima: a sleep can only inflate a sample,
    never shrink it, so it cannot manufacture a pass, and the $T_k$ figure was reproduced as two
    independent tight bands (0.0494–0.0516 s and 0.0505–0.0528 s). **Prompt 13 should take its
    timings as CPU time or best-of-N, and say which**; a lone elapsed figure from this machine is
    not evidence.

13. **The `transfer-remedial` campaign has landed** (merged at `e01c31d`; README §0.2, §4.2):
    still do not touch its files, and its `main.py` Bessel-stage comment is theirs. The two
    shared test files are no longer a live stop condition — prompt 10 left every `8ba9159` line
    in both of them untouched (`[00-transfer-remedial-test-file-overlap]`, §4) — but their
    tolerance *comments* are now stale and must not be trusted as calibration
    (`[10-transfer-remedial-tolerance-comments-stale]`, §3).
14. **A producer calls `cached_phase_residual`, never `build_phase_residual`** (prompt 14). The
    residual table is one per `(model, k, sector)`, built on `residual_node_range(model, k,
    leading_table.z_nodes, sector)` and memoised in the worker; pass the proxy's `store_id`, and
    nothing derived from the object. The object's anchor is off that grid: split it once at
    `nearest_table_node(rho, z_init)` and add `rho.delta(node, z)` per sample — calling
    `rho.delta(z_init, z)` per sample costs a Gauss panel each time. The *leading* table's anchor
    is **not** split this way, which is half of `[07-tk-per-object-cost-is-all-setup]`.
15. **The retired friction ODE lives in a test** (prompt 07). `friction_RHS` and `FRICTION_INDEX`
    are gone from `ComputeTargets/TkWKBIntegration.py`; the function is verbatim as
    `_friction_RHS` at `ComputeTargets/tests/test_background_cs_tau_friction.py:79`, used only by
    prompt 04's `TestFrictionODEComparison`. Producers read
    `friction_F.delta(z_init, z) = F(z) - F(z_init) < 0` from the background table, with **no sign
    flip**, and multiply the amplitude by `exp()` of it.
