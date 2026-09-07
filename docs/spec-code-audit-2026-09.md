# Spec → code audit: the SIGW pipeline up to the source time integral

**Subject.** The completed compute targets of `ComputeTargets/` — `BackgroundModel`,
`TkNumericIntegration`, `TkWKBIntegration`, `GkNumericIntegration`, `GkWKBIntegration`, `GkSource`,
`GkSourcePolicyData`, `QuadSource` — together with `QuadSourceIntegral` (known to need remedial work)
and the `OneLoopIntegral` stub, at commit `e9a5539` (branch `main`).

**Reference.** The signed-off typed specifications `docs/spec/01–05` (transcription campaign
`prompts/spec-transcription/`, all review-queue items closed 2026-09-07). Where the spec records an
author decision in a §0 block, that decision is taken as binding and is not re-litigated here.

**Method.** Four fresh-context agents each read one spec end-to-end and the code that implements
it, mapped every result-bearing formula (`R##`) to a code location, and verified each mapping
either by a sympy derivation (difference exactly zero) or by a numerical measurement against an
independent implementation of the spec formula. Their full reports, with the formula maps and all
measurements, are in `docs/spec-code-audit/`:

| Report | Spec | Code |
|---|---|---|
| [`TK-report.md`](spec-code-audit/TK-report.md) | 01 transfer function | `BackgroundModel`, `CosmologyModels/*`, `TkNumericIntegration`, `TkWKBIntegration`, `WKB_Tk`, `analytic_Tk` |
| [`GK-report.md`](spec-code-audit/GK-report.md) | 02 Green's function | `GkNumericIntegration`, `GkWKBIntegration`, `WKB_Gk`, `GkSource`, `GkSourcePolicyData`, `analytic_Gk` |
| [`QS-report.md`](spec-code-audit/QS-report.md) | 03 source term | `QuadSource`, `main.py` pair scheduling |
| [`QI-report.md`](spec-code-audit/QI-report.md) | 04 source integral, 05 one-loop | `QuadSourceIntegral`, `OneLoopIntegral`, `main.py` triple grid |

The 29 reproduction scripts are in `docs/spec-code-audit/scripts/` (§7). The orchestrator
re-checked every line-level defect below against the source. No repository file other than these
documents was modified.

---

## 0. Summary

### 0.1 Verdict

**The physics is implemented correctly, everywhere it is implemented.** Every result-bearing formula
of specs 01–04 that has a code counterpart reproduces the spec exactly (sympy) or to the expected
numerical floor. In particular:

- The redshift-space transfer-function ODE, its WKB frequency $\omega_{\rm eff}^2$ and the corrected
  $d\ln\omega_{\rm eff}/dz$ (the `641bb51` fix is present and exact), the Liouville–Green amplitude
  $P\,\omega^{-1/2}$ with $P=\sqrt{H_{\rm init}/H}\,e^{F}$, and the matching coefficients all agree
  symbolically. The analytic constant-$w$ oracle is the exact solution of the coded ODE.
- The Green's-function ODE, the unit-jump initial data (in $z$, slope $+1$), $\omega_{\rm eff}^2$ and
  the `NUM` 10 correction, the WKB matching ($2\times10^{-16}$ on 2000 random quadruples), the R38
  phase representation and the $(\mathrm{div}\,2\pi,\ \mathrm{mod}\,2\pi)$ carry all agree. The
  assembled `GkSource` function is the unit-jump $\bar G_k$ on both sides of the crossover, same sign
  and normalisation (residual is the intrinsic WKB error, $10^{-5}$–$10^{-4}$ at 5–6.6 e-folds
  sub-horizon).
- The source kernel `source_function` is *algebraically identical* to spec 03 R22, including which
  $w$ multiplies which term (`wBackground` at the source redshift) and the derivative variable
  (plain $dT/dz$). It is exactly symmetric in $q\leftrightarrow r$ and `main.py` schedules only
  unordered pairs.
- All three regions of `QuadSourceIntegral` implement the spec 03 R28 measure
  $\int dz'\,\bar G\,\frac{1+z}{1+z'}\,f/H^2$ with nothing missing ($4\times10^{-16}$ quadrature,
  $8\times10^{-14}$ Levin), and `analytic_integral` reproduces spec 04 R14 (with the signed-off
  $a_0^2$) to $10^{-8}$–$10^{-6}$ relative on seven configurations.
- **The normalisation question flagged in `docs/resonance-scaffolding/sigw-resonance-reconciliation.md`
  §4 item 4 is closed in the code's favour.** The code's $1/((3+2b)(2+b))$ against R31's
  $(2+b)/(3+2b)^3$ is exactly $1/c^2$ with $c=(2+b)/(3+2b)=3(1+w)/(5+3w)=c_*$, and that factor is
  entirely the normalisation of $f$: measured $f_{\rm code}/f_{\rm R28}=1/c^2$ to 14 digits at three
  values of $b$. `analytic_rad` is $I_s$ of spec 03 R28 with $Q_s/a_0^2$ stripped, precisely as
  specs 03/04/05 §0.1 assert.
- $a_0$ is absorbed, not set to one, and the covariance test passes *structurally*: momenta appear
  only as $k/a_0$ and time only as $z$ or $a_0\eta$, so there is no variable in which a stray power
  could hide.

**Remedial work is nevertheless needed**, and it falls into three groups: one genuine physics defect
in a cosmology class (§0.2, A1), the two sampling/representation defects that the reconciliation
document already predicted for `QuadSource`/`QuadSourceIntegral` (A2–A4, now measured), and a set of
stored-diagnostic and bookkeeping slips (B-group). Nothing above the source integral exists.

### 0.2 Findings that change a stored number or block a run

Ranked by consequence. IDs in parentheses are the detailed-report identifiers.

| # | Finding | Where | Magnitude | Severity |
|---|---|---|---|---|
| **A1** | `LambdaCDM_GenericEOS.wPerturbations` divides by the **total** density, $\rho_\Lambda$ included, contradicting its own comment, the author's "$\Lambda$ unperturbed" convention (spec 01 Tier 3, spec 03 §0.5) and the `LambdaCDM` sibling. This is $c_s^2$ of the transfer-function ODE, of $\omega_{\rm eff}^2$, of the LG friction integrand and (via the spline stack) of $w'$, $w''$. Likely a faithful transcription of spec 01 R16 *as literally written*, which is exactly the inconsistency spec 01 Q4 flagged and the sign-off resolved the other way. Plain `LambdaCDM` models are unaffected. (TK-1) | `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:290` | $c_s^2$ too small by $\times3.21$ at $z=0$, $\times1.28$ at $z=1$, $<1\%$ for $z\gtrsim5$ | **DEFECT, physics** |
| **A2** | `QuadSource._create_functions` fits a cubic spline through the *oscillating* source on the 100-per-decade grid. Measured on the exact radiation source: error reaches **100 % of the local envelope** once $kc_s\eta\gtrsim300$ (~95 cycles), pointwise relative error $10^4$–$10^7$ beyond. The shipped grid runs to $\sim4.5\times10^6$ cycles. Every `QuadSourceIntegral` region reads this spline, so it is the accuracy floor of the whole source integral sub-horizon — where the resonance lives. This is F1 of the reconciliation document, now quantified. (QS-5) | `ComputeTargets/QuadSource.py:294-309` | $O(1)$ of the envelope for $x\gtrsim300$ | **DEFECT, representation** |
| **A3** | `compute_quad_source` walks the **full** source grid against `TkNumericIntegration.z_sample`, tolerating a mismatch only at the leading (high-$z$) end. But `main.py` truncates the $T_k$ grid at **both** ends (`z_exit_suph_e5` … $0.85\,$`z_exit_subh_e6`, since 2025-03-21) and passes `mode="stop"`, so the $T_k$ grid always ends far above `z_end`. Reproduced with mock objects: unguarded `IndexError`. It fails loudly, so no bad data is at risk, but **the `--quad-source-queue` stage cannot complete with shipped settings.** `QuadSource.py` has had no substantive change since before the truncation landed. (QS-6) | `ComputeTargets/QuadSource.py:82-102` vs `main.py:505-507,528` | run-blocking | **DEFECT, regression** |
| **A4** | `WKB_Levin_integral` hands `adaptive_levin_sincos` the **Green's-function phase only**, with the oscillating source $f$ in the amplitude slot; and both Levin gates (`Levin_z` from $\lvert d\theta_G/d\log(1+z)\rvert$, and the `LEVIN_MIN_2PI_CYCLES = 10` *net* phase-difference test) consult nothing about $T_q$, $T_r$. The analytic branch `_three_bessel_Levin` does it correctly (four phase groups $\theta_G\pm\theta_q\pm\theta_r$) and is the template. This is F2 of the reconciliation document, confirmed with lines. (QI-5, QI-6) | `ComputeTargets/QuadSourceIntegral.py:1099-1112`, `:126-137`, `:158-186`; `GkSourcePolicyData.py:179-190` | loses Levin's frequency-independence; inherits A2's floor | **DEFECT, known** |
| **A5** | 92 % of scheduled `QuadSourceIntegral` work items are non-triangles: 63,750 $(k,q,r)$ triples, **5,133 (8.05 %)** satisfy $\lvert q-r\rvert\le k\le q+r$. A non-triangle is not a point of the spec 03 §0.3 integrand at all. F3 arithmetic re-derived and confirmed exactly (109 surviving pairs at mid-grid $k$, three nodes within $\pm0.05$ of $s=\sqrt3$). (QI-12) | `main.py:2488-2492` | waste + uncontrolled $s$ sampling | **DEFECT, known** |
| **A6** | `GkSourcePolicyData` classification band `"WKB_minimal"` tests `numeric_clearance` instead of `WKB_clearance`, so the last band never checks the WKB end and can pre-empt the intended fallback, changing the stored `crossover_z`/`Levin_z`. Reachable only when all nine earlier bands fail; needs a database run to exercise. (GK-2) | `ComputeTargets/GkSourcePolicyData.py:364` | corner case; changes region split, not $G$ | **DEFECT, policy** |
| **A7** | `BackgroundModel._build_derivative` spline stack is grid-end biased: on `main.py`'s actual grid, $\epsilon''$ relative error $3.0\times10^{-1}$ at the $z=0.1$ end vs $6.3\times10^{-5}$ interior; $w''$ $3.8\times10^{-1}$ at the end. Affects only cosmologies with no analytic derivatives (`LambdaCDM_GenericEOS`, QCD). Propagated impact is bounded — $\le9\times10^{-7}$ in $\omega_{\rm eff}^2$, $\le6\times10^{-5}$ in $d\ln\omega/dz$ — and confined to the outermost 2–3 grid points; the numeric→WKB handover sits deep in the interior. `_build_T_z_spline` pads its grid by 5 % for exactly this reason; `_build_derivative` does not. (TK-5) | `ComputeTargets/BackgroundModel.py:129-149` | ends only | **DEFECT, accuracy** |

### 0.3 Stored-diagnostic and bookkeeping slips (no physical number affected)

| # | Finding | Where |
|---|---|---|
| B1 | `TkWKBValue.analytic_T_w` / `analytic_Tprime_w` return the `_rad` members, so the persisted `analytic_*_w` columns of the TkWKB table hold the **radiation** oracle. Detectable as `analytic_T_w == analytic_T_rad` on every row. `TkNumericValue` is correct, so `QuadSource` is unaffected. (TK-2) | `ComputeTargets/TkWKBIntegration.py:632,636` |
| B2 | Same copy-paste slip in `GkWKBValue.analytic_G_w` / `analytic_Gprime_w`; propagates into `GkSourceValue` in the WKB-only region. ~15 % wrong at $w=0.2$. Any "WKB vs fixed-$w$ analytic" plot is silently comparing against $w=1/3$. (GK-1) | `ComputeTargets/GkWKBIntegration.py:583,588` |
| B3 | Pre-flight WKB-criterion warnings omit `fabs`; $d\ln\omega_{\rm eff}/dz<0$ throughout, so both warnings are dead code. The hard check in `WKB_phase_function.py:662` does use `fabs`, so the condition is guarded; the comments claiming `compute()` checks it are untrue. (TK-3, GK-3) | `TkWKBIntegration.py:356`, `GkWKBIntegration.py:312` |
| B4 | Attribute typo `_init_efolds_suph` where the property and `store()` use `_init_efolds_subh` → `AttributeError` instead of the intended `RuntimeError`. (TK-4, GK-8) | `TkWKBIntegration.py:115`, `GkWKBIntegration.py:81` |
| B5 | In `_three_bessel_Levin` the `Y3` call alone uses `LEVIN_ABSERR`/`LEVIN_RELERR` instead of the passed tolerances (100× tighter), making the four cancelling phase groups' error bars non-uniform. (QI-8) | `QuadSourceIntegral.py:622-623` |
| B6 | `analytic_integral`'s `atol`/`rtol` arguments are dead; `1e-21`/`1e-8` are hardwired, so the stored `atol_serial`/`rtol_serial` do not describe `analytic_rad`. (QI-9) | `QuadSourceIntegral.py:832-833,844-845` |
| B7 | No `b` column in the `QuadSourceIntegral` table; `analytic_rad` is stored with no record of the $b$ it used. Safe at HEAD (`b_value = 0.0` hardwired in `main.py:418`), unsafe for any future $b\ne0$ run. Related: the Levin branch uses caller-supplied Bessel-phase splines while `_three_bessel_quad` recomputes `jv(nu+b)`; they agree only because `main.py` builds the splines with the same `b` — an unguarded invariant. (QI-10, QI-1) | `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py:173` |
| B8 | `total` carries no error bound: only the Levin region's `abserr` reaches `metadata`; the two `quad` regions' errors are dropped (documented in a comment). (QI-11) | `QuadSourceIntegral.py:314-322` |
| B9 | The $\theta$ spline used to pick `Levin_z` is built with `chunk_logstep=None`, the one evaluated with `chunk_logstep=125`. Not measured. (GK-10) | `GkSourcePolicyData.py:170` vs `:669` |
| B10 | `ZSplineWrapper` in `QuadSource` is labelled `"T_k"`, so a source-spline range error reads `GkSource.function: evaluated T_k out of bounds`. (QS-10) | `QuadSource.py:304` |
| B11 | Region-nonempty guards in `compute_QuadSource_integral` test a ratio in $z$, not $1+z$, so the effective minimum interval width varies by orders of magnitude across the range and would divide by zero at $z_{\rm resp}=0$ (unreachable at `DEFAULT_ZEND = 0.1`). (QI-4) | `QuadSourceIntegral.py:194,220,246` |

### 0.4 What is not built

None of the spec 05 / spec 03 §0.3 outer layer exists, confirmed by grep: no $648\pi^2$, no
$\big(\tfrac{1+w^*}{5+3w^*}\big)^4$, no $w^*$, no spin-2 projector $Q_s$, no $\int dq/q\int d\theta\sin^5\theta$
measure, no $\mathcal P_\zeta$ object, no squaring, no per-polarisation label. `OneLoopIntegral` is a
stub that `main.py` never instantiates (its `compute()` guard is inverted and its body is empty).
Spec 03 §0.2 already labels the corrected prefactor chain a *build specification*; this audit
confirms the boundary: **everything below the loop integral is built and verified, everything above
it is unbuilt.** The exact ledger of which prefactor lives where is §3.1.

---

## 1. Stage map

| Stage | Spec | Compute target(s) | Formula agreement | Defects | Report |
|---|---|---|---|---|---|
| Background $H(z)$, $a_0\eta$, $w_0$, $c_s^2$, $\epsilon$ and derivatives | 01 §3.2–3.3 | `BackgroundModel`, `CosmologyModels/*` | exact (`LambdaCDM` analytic derivatives verified vs sympy) | **A1** (GenericEOS $c_s^2$), A7 (spline ends) | TK |
| Transfer function $T_k(z)$, numeric | 01 §3.1–3.3 | `TkNumericIntegration`, `analytic_Tk` | exact; oracle is the exact solution of the coded ODE; $T\to1$ normalisation error $2.5\times10^{-6}$ at the production $z_{\rm init}$ | — | TK |
| Transfer function, LG phase form | 01 §3.4–3.5 | `TkWKBIntegration`, `WKB_Tk` | exact ($\omega_{\rm eff}^2$, $d\ln\omega/dz$, $P$, matching); reconstruction $\le5.6\times10^{-3}$ of envelope at 3 e-folds sub-horizon (LG truncation) | B1, B3, B4 | TK |
| Green's function, numeric | 02 §3.1–3.2 | `GkNumericIntegration`, `analytic_Gk` | exact ODE; oracle to $1.5\times10^{-11}$ under $G_{\rm code}=-a_0H(z')\,{\rm Gr}_k$ | — | GK |
| Green's function, LG phase form | 02 §3.3–3.5 | `GkWKBIntegration`, `WKB_Gk` | exact ($\omega_{\rm eff}^2$, NUM 10 fix, R26/R36 matching, R38 $Q(u)$) | B2, B3, B4 | GK |
| Repackaging at fixed response time | 02 §3.5, 04 §0 | `GkSource`, `GkSourcePolicyData` | same unit-jump $G$ both sides of crossover | **A6**, B9; no WKB-validity criterion is applied (§4) | GK |
| Quadratic source $f(z'\mid q,r)$ | 03 R22 / §0.3 | `QuadSource` | exact (sympy), $2.3\times10^{-16}$ numeric | **A2**, **A3**, B10 | QS |
| Source time integral | 03 R28, 04 R1/R14 | `QuadSourceIntegral` | measure exact; analytic oracle to $10^{-8}$–$10^{-6}$ | **A4**, **A5**, B5–B8, B11 | QI |
| One-loop $P^h_{22,s}(k)$ | 03 §0.3, 05 R23/R31 | `OneLoopIntegral` | — | not built | QI |

---

## 2. Findings by stage (condensed)

Each item below is developed in full, with the measurement tables, in the linked report.

### 2.1 Background and transfer function (`TK-report.md`)

- **A1 — GenericEOS sound speed.** Spec 01 Tier 3: "$c_s^2$ means $w(z)$ of the *perturbed* fluid,
  $\Lambda$ unperturbed; in the code this is `wPerturbations(z)` (radiation + matter, $\Lambda$
  excluded)." `LambdaCDM.wPerturbations` (`LambdaCDM.py:215-224`) divides by
  $\Omega_m+\Omega_r(1+z)$, correctly. `LambdaCDM_GenericEOS.wPerturbations` divides by `self.rho(z)`
  $=\rho_m+\rho_r+\rho_\Lambda$ (`LambdaCDM_GenericEOS.py:250-257, 290`), while its own comment says
  $\Lambda$ is excluded. Measured with Planck 2018 values: ratio correct/code $=3.214$ at $z=0$,
  $1.656$ at $0.5$, $1.277$ at $1$, $1.082$ at $2$, $1.010$ at $5$, $1.000$ at $100$. The absolute
  size of $c_s^2$ at $z\lesssim2$ is $\sim10^{-4}$, so the absolute change in the ODE coefficients is
  small; but what is wrong is precisely the late-time radiation correction the quantity exists to
  describe. `wBackground` is correct in both classes.
- **Everything else agrees.** R5→R14/R21 (the coded ODE, integrated in $z$; the `wPerturbations*(k/H)²`
  term *is* R21's $c_s^2k^2/((1+z)^2a^2H^2)$ because $(1+z)^2a^2=a_0^2$); R11 (`analytic_Tk`);
  R23/R24/R26 (LG amplitude and the `raw_sin_coeff` matching, which expands to the spec's $\alpha$
  verbatim); R27/R29 (`Tk_omegaEff_sq`); R30 corrected (`Tk_d_ln_omegaEff_dz`; the pre-fix form
  differs by exactly $9(1+w)w'/(4(1+z)^2)$, confirming the `641bb51` fix); R18/R20 ($\tau$ integration
  and `tau_init`).
- **Conventions recorded.** (i) $\theta$ *decreases*: $d\theta/dz=+\omega$ integrated towards smaller
  $z$, so $\theta<0$; `WKB_mod_2pi`'s negative-remainder rule matches. (ii) The
  `sgn_sin_deltaTheta*sgn_T` factor in `store()` is provably always $+1$. (iii) Stored
  `theta_div_2pi` is rebased by an arbitrary integer per object; only `theta_mod_2pi` is used in
  `T_WKB`. (iv) `tau_init` is R20 (radiation-derived) applied to the total $\rho$: relative error
  $8.5\times10^{-4}$ at $z_{\rm init}=10^6$, $8.5\times10^{-8}$ at $10^{10}$, entering as a constant
  absolute offset in $a_0\eta$; $\tau$ feeds only the oracles, not the ODEs. (v) $T_k\to1$ is a
  normalisation of $\phi/\phi^*$; the $\zeta^*\to\phi^*$ constant $c_*=3(1+w^*)/(5+3w^*)$ is applied
  nowhere and belongs to the one-loop prefactor. Stored $T_k$, $T_k^{\rm WKB}$ and `QuadSource`
  quantities are in $\phi/\phi^*$ units.

### 2.2 Green's function (`GK-report.md`)

- **All formulas agree.** R17/R19/R32 (the ODE; sympy difference 0, derived with $a_0$ explicit and
  cancelling into $k_{\rm phys}$); R22/R32 and R31 ($\omega_{\rm eff}^2$ and the NUM 10 corrected
  $d\ln\omega/dz$, both 0); R24/R29 (`LambdaCDM` $d^n\ln H/dz^n$, 0); R26/R36 (matching, 2000
  random quadruples, $1.9\times10^{-16}$); R37 (single-sine rewriting, $3\times10^{-12}$); R38
  ($Q(u)$ verbatim, $2.9\times10^{-12}$). `analytic_Gk` reproduces the coded ODE to
  $1.5\times10^{-11}$ for $w=1/3,0.2,0$.
- **Unit jump confirmed.** Variable is $z$; initial slope $+1$ in $z$; $G\approx z-z'<0$ just after
  the source; the assembled `GkSource` function is $G_{\rm code}$, not $-G_{\rm code}$ and not
  ${\rm Gr}_k$.
- **GK-4 resolves spec 02 open question Q9 in the code's favour.** R37 as transcribed writes the
  amplitude with a *starred* $\omega_{\rm eff}$; the code keeps $\omega_{\rm eff}(z)$ unstarred
  (R34/R35/R46). With the starred form the reconstruction would drift by $(\omega^*/\omega(z))^{1/2}\approx4$
  over the test range; unstarred, it agrees to $3.8\times10^{-12}$ in radiation. The stars on
  `NUM` 11 p.5 are a slip. **Recommended spec edit:** close Q9 at spec 02 R37.
- **No WKB-validity criterion is applied** (GK-6). `crossover_z` is chosen purely from spline-clearance
  geometry (0.05/0.025/0.01 relative in $\log(1+z)$, `maximize-WKB`); `Levin_z` from
  `Levin_threshold` 1.5 or 5.0 on $\lvert d\theta_G/d\log(1+z)\rvert$. R38's $Q\to-1$ is recorded as
  metadata only, and `has_WKB_violation` is stored but never used to reject data. Consistent with the
  §0.2 item 2.8 sign-off; whether a violating mode should be dropped is open (§4). Also, the R38
  premise $\lvert Q\rvert\to1$ does not hold in practice ($\lvert Q\rvert\approx28$–$50$) because
  $\omega_{\rm eff}\approx k/H$ evolves strongly; harmless, but `stage_2_largest_Q`/`smallest_Q`
  should not be read as "close to $\pm1$" diagnostics.
- **The fixed-$w$ oracle uses $w$ at the response redshift** (GK-9) while the closed form assumes
  constant $w$ between source and response; a non-zero `G − analytic_G_w` across the matter/radiation
  transition is expected, not an error.

### 2.3 Source term (`QS-report.md`)

- **Kernel exact.** `source_function` $=$ spec 03 R22 identically; the only regrouping is
  $1+\frac{2}{3(1+w)}=\frac{5+3w}{3(1+w)}$, which moves the $\frac{2}{3(1+w)}T_qT_r$ cross term into
  the `undiff` column. So the `undiff`/`diff` split is a *reporting* convention and only
  `source = undiff + diff` is spec-defined; do not compare column-by-column against the spec.
- **Derivative variable** is plain $dT/dz$ (`solve_ivp` integrates in $z$;
  `compute_analytic_Tprime` $=-(1/H)\,dT/d\tau=dT/dz$, confirmed to 50 digits). The two wrong
  readings would differ by $(1+z)^{\pm1}$, i.e. factors of $10^1$–$10^5$ on the shipped grid.
- **A2, A3** as in §0.2. The spline-error table (QS-5): error/envelope $1.5\times10^{-9}$ at 1 cycle,
  $1.8\times10^{-4}$ at 10 cycles, $0.88$ at 95 cycles, $>1$ beyond.
- **Oracle hygiene.** `analytic_source_rad/_w` are the *same* `source_function` fed analytic $T$; a
  green `source ≈ analytic_source_rad` validates `TkNumericIntegration`, not $f$ (QS-8).
  `analytic_source_w` mixes $T$ from $w=$`wPerturbations` with coefficients from `wBackground`:
  identical above $z\approx100$, coefficient ratio 1.70 at $z=0.1$ (QS-7). `compute_analytic_Tprime`
  loses precision to cancellation deep super-horizon (relative error $\approx10^{-16}/x^2$), harmless
  inside $f$ (measured $f$ accuracy $\le10^{-13}$) but the `analytic_Tprime_*` columns should not be
  trusted on their own super-horizon (QS-11).

### 2.4 Source time integral and the one-loop layer (`QI-report.md`)

- **Measure exact, oracle exact.** See §0.1. `total` and `analytic_rad` are like-for-like: same
  range $[z_{\rm resp}, z_{\rm src,max}]$, same $f$ normalisation, same overall sign (`analytic_rad`
  carries R14's explicit minus; `total` computes R14's left-hand side). The three regions are the same
  integrand differing only in the $G$ representation, and the WKB `sin_coeff` is fixed from the
  numeric $G$, $G'$, so the $G$ normalisation is one object.
- **LG convention of the analytic branch.** `bessel_phase` supplies $J_\nu=m\sin\theta$,
  $Y_\nu=-m\cos\theta$, i.e. `NUM` 07 R22 with $\gamma=\Theta-\pi/2$ relative to `NUM` 06 p.10's
  $\cos\gamma$ — spec 04 Q7's undocumented identification, and the code is self-consistent in it.
  **Recommended spec edit:** record this at spec 04 Q7.
- **A4, A5** as in §0.2; **QI-7** (§4) is the one UNVERIFIED item: continuity of $G$ across
  `crossover_z` is never measured.
- **Nothing above the loop integral exists** (§0.4).

---

## 3. Cross-cutting conventions

### 3.1 Normalisation ledger: where each factor of the spec 03 §0.3 build form lives

$$P^h_{22,s}(k)=648\pi^2\Big(\tfrac{1+w^*}{5+3w^*}\Big)^4\int_0^\infty\frac{dq}{q}\int_0^\pi d\theta\,\sin^5\theta\;\mathcal P_\zeta(q)\frac{\mathcal P_\zeta(r)}{r^3}\Big\{\int dz'\,\bar G_k\frac{1+z}{1+z'}\frac{f}{H^2}\Big\}^2$$

| Factor | Spec | In the code? | Where it must be supplied |
|---|---|---|---|
| $f=T_qT_r+\frac{2}{3(1+w_0(z'))}(T-(1+z')T')_q(T-(1+z')T')_r$ | 03 R22 | **yes**, `QuadSource.py:35-45`, exact | — |
| $T_k\to1$ at $z_{\rm init}$ ($\phi/\phi^*$ units) | 01 §2.8 | **yes**, `TkNumericIntegration.py:376-377` | — |
| $\bar G_k$ unit-jump causal in $z$ | 02/04 §0 | **yes**, `GkNumericIntegration.py:343-344` | — |
| $dz'$, $(1+z)/(1+z')$, $1/H(z')^2$ | 03 R28, 04 R1 | **yes**, `QuadSourceIntegral.py:955-975` (as $\log(1+z')$ with the Jacobian absorbed) | — |
| $\frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma^2(\tfrac52+b)$, $c_s$ powers, finite $\eta_{\rm init}$ (oracle only) | 04 R11/R14 | **yes**, `QuadSourceIntegral.py:817,868-878` | — |
| $a_0$ powers | 03 R28/R29 | absorbed by construction ($k/a_0$, $a_0\eta$) | — |
| $Q_s(\mathbf k,\mathbf q)/a_0^2\to q_{\rm phys}^2\times$ spin-2 angular factor | 03 R21/R28/R33 | **no** | one-loop layer, inside the brace before squaring |
| $c_*=3(1+w^*)/(5+3w^*)$ ($\zeta^*\to\phi^*$), i.e. $36c_*^2/9\cdot$… $\to648\pi^2c_*^4/81$ | 03 R26/R35, §0.4 | **no**; $w^*$ appears nowhere | one-loop layer |
| $\int dq/q$, $\int d\theta\sin^5\theta$, $r=\sqrt{k^2+q^2-2kq\cos\theta}$, triangle filter | 03 §0.3 | **no** ($r$ is an independent grid point; A5) | one-loop layer |
| $\mathcal P_\zeta$, squaring, per-polarisation label $s$ | 03 §0.3–0.4, 05 §0.2 | **no** | one-loop layer |

Consequences: `QuadSourceIntegral.total` and `analytic_rad` are $I_s$ of spec 03 R28 **divided by
$Q_s/a_0^2$**, equivalently $-\tfrac{1}{c^2}\times$ spec 05 R31 in $a_0$-absorbed variables. The
minus is the relative sign between two *differently defined* intermediates — `NUM` 03/06's $\bar G_k$ with
$\int_z^{z_{\rm init}}dz'$ against `MAIN` 14's ${\rm Gr}_k$ with $\int d\eta'$ — not a sign error in either
chain. Followed consistently to $h_s$, each chain gives the same tensor field with the same sign; the code
follows the `NUM` 03 chain (R22 → R28) exactly as signed off, so its $h_s$ carries `NUM` 03's sign. The audit
did not re-trace the `MAIN` 14 prefactor chain sign-for-sign, and it does **not** rely on squaring to make
the result right.

### 3.2 Variables, signs, $a_0$

- **Independent variable is $z$** for every ODE (transfer function, Green's function, both phase
  functions, friction integral, $\tau$). $\log(1+z)$ appears only as spline abscissa, in the Levin
  frequency test, in the horizon-crossing solve, and as the integration variable of the three source
  integral regions (with $d\log(1+z')=dz'/(1+z')$ absorbing the spec's $1/(1+z')$). Every quantity
  named `d…_dz` is a genuine $z$-derivative (`ZSplineWrapper` divides the log-spline derivative by
  $1+z$).
- **$a_0$ absorbed.** `wavenumber.k` is $k/a_0$; `ModelFunctions.tau` is $a_0\eta$; each is
  individually invariant under $a_0\to\lambda a_0$, comoving $k\to\lambda k$, $\eta\to\eta/\lambda$.
  Measured: $G(z=5)$ bit-identical for $\lambda=1,3,0.1$. The model classes offer no dial for $a_0$
  because it never appears.
- **Signs that are conventions** (none changes a physical result): the $-1/(a_0H)$ source of spec 02
  R17/R18 vs the code's $+1$ unit jump; $\Theta'=+\omega_{\rm eff}$ (so $\theta<0$, $Q\to-1$ rather
  than $+1$); $G_{\rm code}<0$ just after the source; R14's explicit minus in `analytic_rad`.
- **Which $w$.** `wBackground` ($\Lambda$ included) inside $f$; `wPerturbations` ($\Lambda$
  unperturbed) in the $T_k$ equation, $\omega_{\rm eff}^2$ and the LG friction. Both as spec 03 §0.5
  prescribes, with the single departure A1.

---

## 4. Items that need a database run (UNVERIFIED)

1. **Continuity of $G$ at `crossover_z`** (QI-7). The crossover is picked on spline clearance, not on
   numeric/WKB agreement, and nothing checks $G_{\rm num}(\texttt{crossover\_z})\approx G_{\rm WKB}(\texttt{crossover\_z})$.
   A gross error would be caught by `GkSource.py:265-296`; a small step would not. Needed: evaluate
   both splines at `crossover_z` for a sample of $(k,z_{\rm resp})$ and report the relative difference.
2. **Reachability of A6** — whether any shipped $(k,z_{\rm resp})$ falls through to the `minimal`
   band.
3. **Should a mode with `has_WKB_violation` be rejected?** (GK-6). Currently used regardless.
4. **The A7 end bias in a real GenericEOS run** — measured here on `LambdaCDM` with the same code
   path; the QCD tables may behave differently near the crossover.

---

## 5. Consequences for the remedial-work and one-loop campaigns

This section records what the audit *implies* for Steps 2 and 3 of the delivery plan; it is not a
design. The design questions themselves are in `docs/resonance-scaffolding/sigw-resonance-reconciliation.md`
§3, which this audit leaves intact and, on its §4 item 4, resolves.

1. **A1 must be fixed before any GenericEOS/QCD production run**, independently of the refactor:
   one-line change to divide by $\rho_m+\rho_r$, plus a regression test that `LambdaCDM_GenericEOS`
   with the radiation EOS reproduces `LambdaCDM.wPerturbations`. Worth adding the `csSquared(z)`
   hook the reconciliation document proposes at the same time, so that $c_s^2$ and $w_{\rm pert}$
   become separable.
2. **A2 and A3 are the same fix**: `QuadSource` must consume `TkWKBIntegration` sub-horizon
   (amplitude and phase, never sampled values) and keep the numeric branch super-horizon, with its
   own validity boundary per factor — the binding constraint being the *smaller* of $q,r$. That is
   Tier 1 of the reconciliation document's §3.1, and it is a correctness fix, not a tidy-up.
3. **A4 has a working template in the same file.** `_three_bessel_Levin` already builds the four
   phase groups $\theta_G\pm\theta_q\pm\theta_r$ with per-group errors summed linearly; the numerical
   branch needs the analytic phase objects swapped for `phase_spline`s, with the $(\mathrm{div}\,2\pi,\mathrm{mod}\,2\pi)$
   carry done in integers. The Levin *decision* must then be made per phase group on total variation,
   not on $\theta_G$'s net change.
4. **A5 belongs to whoever owns the $(k,q,r)$ grid**, which after the refactor should be
   `OneLoopIntegral`. Adding the triangle filter now recovers 92 % of scheduled work at no design cost.
5. **The normalisation boundary is now known exactly** (§3.1), so the one-loop layer can be
   specified as: take `total`, multiply by $q_{\rm phys}^2\times$ the spin-2 angular factor, square,
   weight by $648\pi^2c_*^4/81\cdot\mathcal P_\zeta(q)\mathcal P_\zeta(r)/r^3$ with the
   $dq/q\,d\theta\sin^5\theta$ measure, and label by $s$. The stored `analytic_rad` is a ready-made
   fixed-$w$ oracle for the inner integral at every $(k,q,r,z_{\rm resp})$.
6. **Persist $b$** (B7) and make `analytic_integral` honour its tolerances (B6) before the oracle is
   leaned on in the validation ladder.

---

## 6. Recommended spec edits

Not physics changes; these record what the code settled.

- spec 02 R37 / Q9: close in favour of the unstarred $\omega_{\rm eff}(z)$ (GK-4).
- spec 04 Q7: record $\gamma=\Theta-\pi/2$, i.e. the code's $J_\nu=m\sin\theta$, $Y_\nu=-m\cos\theta$
  convention (QI, R17 row).
- spec 01 R16 / Q4: add a note that the literal R16 denominator (with $\Omega_{cc}$) was transcribed
  into `LambdaCDM_GenericEOS.wPerturbations` and is the origin of A1; the sign-off's "$\Lambda$
  unperturbed" is the binding reading.

---

## 7. Reproduction

All scripts are in `docs/spec-code-audit/scripts/`, run from the repository root with the project
virtualenv (`./venv/bin/python`, NumPy 2.2.4, SciPy 1.15.2, SymPy 1.13.3, mpmath). They import the
repository modules directly and build minimal stand-ins for `wavenumber`/`redshift`/model objects
where the compute path is entangled with Ray or the datastore; each script's docstring says which.
`GK_04`, `GK_06` import `GK_03`; `TK_04`, `TK_08` import `TK_03`. Per-report script tables with
one-line results are §4 of each report. All 29 were re-run from this location after the move.

**Caveats stated plainly.**

- Nothing here touched a database. The pipeline was audited at the level of the functions that
  compute each stored quantity, driven with exact constant-$w$ backgrounds and synthetic inputs; the
  four items in §4 are the ones that genuinely need stored data.
- "Exact" means a sympy difference of zero between the spec formula and the code expression under
  the stated variable mappings; numerical agreement figures are relative differences against an
  independent implementation of the spec formula, with the floor set by phase-spline fitting
  ($\sim2\times10^{-8}$) or by `scipy.quad` on oscillatory references.
- The agents were instructed to treat spec content and code comments as data, to respect the §0
  sign-off decisions (including "$a_0$ absorbed, not unity" and the Jacobian-without-modulus
  convention), and not to consult `thirdparty/`, the two student documents, or `MAIN` 15.
