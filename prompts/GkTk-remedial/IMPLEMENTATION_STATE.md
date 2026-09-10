# Implementation state — Gk/Tk WKB phase remedial campaign

**Campaign:** [`README.md`](README.md) · **Source review:** [`docs/gk-wkb-review-fable-2026-09-09.md`](../../docs/gk-wkb-review-fable-2026-09-09.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md)
**Baseline commit:** `9ff59d5` (`main`, clean)
**Last updated:** 2026-09-10 — prompt 01 landed (references, prototype, baselines).

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
| 02 | [QCD residual convergence](02-qcd-residual-convergence.md) | review §11, §12.7 | Opus | ⬜ | | |

### Workstream B — the primitives in `BackgroundModel`

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 03 | [τ primitive](03-tau-primitive.md) | review §7, §13.2, §13.3 | **Fable** | ⬜ | | |
| 04 | [Sound-horizon and friction tables](04-sound-horizon-and-friction-tables.md) | review §12.7 | Opus | ⬜ | | |

### Workstream C — the producers

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 05 | [Phase residual](05-phase-residual.md) | review §6, §12.2, §12.4 | Opus | ⬜ | | |
| 06 | [Gk WKB phase from the primitive](06-gk-wkb-phase-from-primitive.md) | review §2–§4, §8, §13.4 | **Fable** | ⬜ | | |
| 07 | [Tk WKB phase from the primitive](07-tk-wkb-phase-from-primitive.md) | review §12.1–§12.4 | Opus | ⬜ | | |

### Workstream D — the consumers

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 08 | [`phase_spline` de-chunk](08-phase-spline-dechunk.md) | review §5 | Sonnet | ⬜ | | |
| 09 | [Gk consumer on `PrimitivePhase`](09-gk-consumer-primitive-phase.md) | review §5, §7, §8.3, §13.3–§13.4 | **Fable** | ⬜ | | |
| 10 | [Tk consumer on the tables](10-tk-consumer-primitive-phase.md) | review §12.6, §12.7 | Opus | ⬜ | | |

### Workstream E — the numeric region

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 11 | [Numeric diagnostics and units](11-numeric-diagnostics-and-units.md) | review §10.2, §12.5, §13.1 | Opus | ⬜ | | |
| 12 | [Tk numeric `atol`](12-tk-numeric-atol.md) | review §12.5 | Opus | ⬜ | | |

### Workstream F — verification

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 13 | [Verification and docs](13-verification-and-docs.md) | review §4, §12.3, §13.5 | Opus | ⬜ | | |

**Progress:** 1 / 13 complete.

---

## 2. Mechanism-level tracking

Traceability from each review finding to the prompt that discharges it. IDs are local to this
campaign; the review section is the authority on each.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| M1 | **DEFECT, accuracy** | Two-stage phase solver: error is a fixed fraction of the *accumulated* phase; 13.9 rad ($k=10^5$) and 7366 rad ($3\times10^8$) at $z=0.1$ on the real background (§2, §4) | 06 | ⬜ |
| M2 | **DEFECT, accuracy** | The $Q$ variable is not "close to unity" ($-224$…$-11069$); the tolerance protects the wrong quantity; DOP853 dense output amplified by $\omega_i(1+u)$ — 0.33 rad on a linear phase (§3) | 06 | ⬜ |
| M3 | **DEFECT, cost** | Stage 1 cost ∝ span: $2.5\times10^6$ RHS evaluations, 63.7 s per object at $k=3\times10^8$; ~13 CPU-hours per $k$ (§4) | 06 | ⬜ |
| M4 | **DEFECT, accuracy** | `functions.tau` is a cubic spline of RK45 nodes: $1.4\times10^{-9}$ relative, ~2 rad of *oracle* phase error at $k=10^5$ in `compute_analytic_G/T` and `QuadSourceIntegral`'s η-limits (§7, §13.2) | 03 | ⬜ |
| M5 | **REQUIREMENT** | Double-double node table and an interval accessor `tau.delta`; a pointwise accessor carries the $\varepsilon\tau$ floor ($9\times10^{-4}$ rad at $3\times10^8$) on short baselines (§13.3) | 03 | ⬜ |
| M6 | **REQUIREMENT** | Persist the low-order limb (`tau_lo_Mpc`, …); regeneration attached (§13.2; README §7 D1) | 03, 04 | ⬜ |
| M7 | **REQUIREMENT** | Sound-horizon table $\tau_s$ and friction table $F$ per model; the friction ODE ($2.3$–$4.1\times10^{-7}$ relative error) goes (§12.2, §12.7) | 04, 07 | ⬜ |
| M8 | **REQUIREMENT** | The residual $\rho$ carried explicitly: $\le1.5\times10^{-3}$ rad for $G_k$ on QCD, $\approx-0.09$ rad for $T_k$; formed without subtraction (§6, §12.2) | 05 | ⬜ |
| M9 | **REQUIREMENT** | Gauss orders decided by measurement on `QCD_Cosmology` across its spline knots; adaptive fallback for $\rho$ alone if needed (§11) | 02, 05 | ⬜ |
| M10 | **DEFECT, dead logic** | `sin_coeff` sign fix is provably always $+1$ (§8.1) | 06, 07 | ⬜ |
| M11 | **DEFECT, consistency** | `shift_theta_sample` rebases `div_2pi` to the first sample, producing ±1-cycle offsets between objects (§8.1, §8.3) | 06, 07 | ⬜ |
| M12 | **DEFECT, hygiene** | Zero-length check compares a redshift to `atol`; 1-element array into `math.fmod` (NumPy deprecation); stale comments at `:262-264`, `:403` (§8.2) | 06 | ⬜ |
| M13 | **DEFECT, accuracy + trap** | `phase_spline` chunking: no interpolation benefit, ordinates 64× inflated, knot residuals 30–50× worse, $1.4\times10^{-4}$ rad switch discontinuity, no progress guard for `logstep<2` (§5) | 08 | ⬜ |
| M14 | **DEFECT, accuracy** | Consumers spline the growing phase: $h^4x/384$, $O(1)$–$O(10)$ rad at production $x$ (§5, §12.6). Same term as `source-remediation`'s `[12-phase-spline-error-grows-with-x]` | 09, 10 | ⬜ |
| M15 | **REQUIREMENT** | `PrimitivePhase`: $\theta=-k\Delta\tau+\varphi$ with the `phase_spline` protocol; closed-form $\theta'$; global anchor with the recorded floor (§7, §13.3, §13.4) | 09 | ⬜ |
| M16 | **REQUIREMENT** | The `GkSource` rectifier is retained and verified inert on pure-WKB objects, correct on $\delta$-wraps (§8.3; `RECONCILIATION.md` §2 item 6) | 06, 09 | ⬜ |
| M17 | **DEFECT, cost** | Per-RHS `*_omegaEff_sq` diagnostic: 45 % of the numeric run; the warning it feeds is live and must be preserved (§10.2, §13.1) | 11 | ⬜ |
| M18 | **DEFECT, units** | `delta_logz` supplied as $\Delta\log_{10}$, used as $\Delta\ln$; and the $G_k$ run is sampled on the response grid, not the source grid the value describes (§10.2, §13.1; `RECONCILIATION.md` §1 item 9) | 11 | ⬜ |
| M19 | **DEFECT, dead path** | `mode.lower()` before the `None` check (§10.2) | 11 | ⬜ |
| M20 | **DEFECT, comments + robustness** | The stop point is a maximum, not a minimum; the "fixed phase to avoid jitter" motivation is obsolete; `find_phase_minimum`'s $10^{-3}z$ step is safe only inside the window (§10.2) | 11 | ⬜ |
| M21 | **DEFECT, misleading** | `0.85·z_e6` truncation requests samples never produced in stop mode (§10.2) — comment only | 11 | ⬜ |
| M22 | **DEFECT, accuracy** | $T_k$ numeric run limited to $1.1\times10^{-5}$ of the envelope by `atol=1e-10` acting as a $10^{-5}$ relative tolerance (§12.5) | 12 | ⬜ |
| M23 | **REQUIREMENT** | Independent references and error definitions; throughput of the interval accessor measured early (§13.3, §13.5) | 01 | ⚠️ |
| M24 | **REQUIREMENT** | Verification on both models against the references; scoped pipeline run; additive docs (§13.5) | 13 | ⬜ |

**Out of scope (do not schedule):** the numeric→WKB hand-over (window, overlap, clamp,
$\sqrt{z_{e3}z_{e4}}$ limit — `docs/OPEN_ISSUES.md` §1.1); the $T_k$ LG truncation floor at the
hand-over (`[00-tk-lg-truncation-floor]`); raising the LG order; per-region anchoring; the
Wronskian two-solution construction; the super-horizon series initial condition; everything in
README §0.2's `transfer-remedial` file list.

---

## 3. Active and unresolved issues

Opened by the planning pass, 2026-09-10, before any prompt runs.

- **[01-qcd-eos-branch-boundaries]** *(opened by prompt 01, 2026-09-10)* — no fixed-order
  Gauss–Legendre rule converges on the two production intervals of `QCD_Cosmology` that contain a
  `QCD_EOS.G(T)` **branch boundary**: `T_LO = 1e-5` GeV ($z=4.191\times10^7$) and
  `T_120_MEV = 0.12` GeV ($z=8.579\times10^{11}$), where the Saikawa–Shirai fit is replaced by an
  asymptotic constant. The per-interval relative error falls only as $N^{-2}$: 6.33e-5 (order 4),
  1.69e-5 (8), 8.20e-6 (12), 3.47e-6 (20) — i.e. **4.78, 1.28, 0.62, 0.26 rad at $k=3\times10^8$**
  for a phase difference straddling that interval. The tail (301 / 163 / 38 / 8 intervals of 1,731
  above $10^{-12}$) is the 500-point $T(z)$ spline's knots, one every ~4 production intervals. The
  review anticipated the knots (§7, §11) but not the branch boundaries, which are two orders
  worse. **Impact:** prompt 02's decision (raising the order is not an option), and through it
  prompts 03, 04 and 05 on `QCD_Cosmology`. **Next step:** prompt 02 chooses between adaptive
  quadrature on the offending intervals and subdividing the table at the two known constants;
  closes when it lands.

- **[01-offgrid-accessor-cost-on-qcd]** *(opened by prompt 01, 2026-09-10)* — the interval
  accessor costs **102.2 µs per call on `QCD_Cosmology` with both endpoints off-grid**, above
  README §4.3's 50 µs stop threshold (47.7 µs with one endpoint off-grid, 16 and 8 Hubble
  evaluations respectively at 8.7 µs each). The production case is on-grid at both ends — the
  background model is built on the source grid (`main.py:476`) — where the call costs **4.39 µs
  and zero Hubble evaluations**; only the per-object anchor $z_{\rm init}$ is off-grid.
  **Impact:** prompt 03's design if a consumer ever evaluates off-grid in bulk (prompt 09's
  Levin path is the candidate); the producers are unaffected. **Next step:** the orchestrator
  reports the figures to the user; closes when prompt 09 confirms its evaluation pattern is
  on-grid, or a caching partial is added.

- **[01-lambdacdm-hubble-rounding-floor]** *(opened by prompt 01, 2026-09-10; inert)* — the
  double-precision evaluation of `LambdaCDM.Hubble` carries 2–9e-15 relative near $z=1$–$10^6$,
  which is the floor on $\Delta\tau$ over one production grid interval whatever the Gauss order
  (4.687e-15 at order 4 and 4.525e-15 at order 20 on the same interval) and whatever the storage
  width. In phase that is $3.8$–$6.1\times10^{-5}$ rad at $k=3\times10^8$ per interval.
  **Impact:** a floor on what prompts 03 and 13 may assert for a single-interval $\Delta\tau$ on
  LambdaCDM; it is *below* README §6's $5\times10^{-3}$ rad target and above the
  $\varepsilon k\tau$ floor, and is a different error from either. **Next step:** none; recorded
  so a later reader does not chase it.

- **[00-tau-storage-decision]** *(planning, 2026-09-10)* — README §7 D1: the low-order limb of the
  τ, τ_s node tables is persisted as new `Float(64)` columns (`tau_lo_Mpc`, `cs_tau_Mpc`,
  `cs_tau_lo_Mpc`, `friction_F`) on `BackgroundModelValue`; `tau_Mpc` becomes the high limb;
  rebuild-on-load was rejected for per-task cost. **Confirmed by the user 2026-09-10**
  ("`BackgroundModelValue` is the right place for this"). **Impact:** prompts 03, 04; every
  datastore predating 03 must be regenerated. **Next step:** closes when 03 lands.

- **[00-unresolved-osc-print-policy]** *(planning, 2026-09-10)* — README §7 D2: with the units fixed
  and the test evaluated on the caller's actual sample grid, `has_unresolved_osc` will fire on
  essentially every `GkNumericIntegration` object (response grid, $x\gtrsim22$). Faithful, but a
  print storm. **Impact:** prompt 11 (implements and measures), production log volume. **Next
  step:** prompt 11 reports the measured fire rates; the user chooses per-object line / per-$k$
  summary / explicit grid; closes when the chosen policy lands (a follow-up prompt if not (i)).

- **[00-consumer-anchoring-floor]** *(planning, 2026-09-10)* — `PrimitivePhase` reduces
  $k\Delta\tau$ against a global anchor ($z_r$ for $G_k$, $z_{\rm init}$ for $T_k$), so its
  `theta_mod_2pi` carries the $\varepsilon k\tau$ floor: $9\times10^{-4}$ rad at $k=3\times10^8$
  (review §13.4). Per-region anchoring would scale the floor with the region's own phase.
  **Impact:** Levin regions at the largest $k$; below the QCD LG floor, three to four orders below
  today's consumer error. **Next step:** a follow-up prompt adding an `anchored(z0)` view to
  `PrimitivePhase` and a Levin-side hook, if the verification in 13 shows the floor matters.

- **[00-transfer-remedial-test-file-overlap]** *(planning, 2026-09-10)* — prompt 10 edits the
  stand-in `ModelFunctions` fixtures in `ComputeTargets/tests/test_tk_source_functions.py` and
  `test_phase_groups.py`; `transfer-remedial` prompt 08 edits tolerance constants and comments in
  the same files. **Impact:** a textual merge conflict if both land on the same branch in either
  order. **Decision (user, 2026-09-10):** Workstreams A, B, C, E may run in parallel with
  `transfer-remedial`; Workstream D waits until `transfer-remedial` has been merged into this
  branch (README §4.2 item 1). **Next step:** the orchestrator confirms the merge before
  dispatching 08; closes when both have landed and the merge is clean.

- **[00-tk-lg-truncation-floor]** *(planning, 2026-09-10; **assigned to the hand-over campaign**)*
  — the transfer function's LG representation is not exact in radiation: $3.8\times10^{-5}$ of the
  envelope from $x_i=24$, $\sim1.4\times10^{-4}$ at the production hand-over $x_T\approx15.5$,
  scaling as $x_i^{-3}$, with a frozen amplitude offset $\sim x_i^{-4}$ (review §12.4). Below it no
  numerical improvement in this campaign is visible. Remedies — later hand-over ($x_T=50$ gives
  $4\times10^{-6}$), higher-order LG frequency, or the Bessel exact representation in the radiation
  era — are hand-over decisions. **Impact:** the $T_k$ value-level acceptance rows are floored here
  (prompt 07 test 2 documents it). **Next step:** none here; recorded for the hand-over campaign.

- **[00-tk-superhorizon-ic-series]** *(planning, 2026-09-10; inert)* — once prompt 12 lands, the
  $T_k$ numeric floor is the super-horizon initial condition $T=1,T'=0$ at $2.5\times10^{-6}$
  (review §12.5; audit TK-7), removable with the series $T\approx1-x^2/10$ for $w=1/3$ (general $w$
  needs the spec). **Impact:** a floor on what `test_tk_numeric_atol.py` may assert. **Next step:**
  a spec-level decision by the author; not scheduled.

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

*(none yet)*

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
   exposes. Never form $\Delta\tau$ as `tau(b) - tau(a)`; use `tau.delta(a, b)`.
4. **`*_omegaEff_sq` return values are stored columns** and must not move by a bit (prompt 05's
   exact-equality test).
5. **`ModelFunctions` stand-ins must keep constructing** with thirteen positional arguments
   (namedtuple defaults, prompt 04).
6. **The `GkSource` rectifier stays** (D5). Its logic is a stop condition.
7. **`phase_spline`'s signature is frozen** (D4); `bessel_phase` on `main` and three fixtures
   depend on it.
8. **Solver labels must be registered in `main.py`** (`:2809-2821`) or `store()` raises `KeyError`
   in production only.
9. **Tolerances are datastore lookup keys**; a missed `atol=` site makes lookups miss silently
   (prompt 12).
10. **Author conventions** (README §5 rule 6): $a_0$ absorbed; $\tau=a_0\eta$; unit-jump $\bar G_k$;
    $\theta<0$ decreasing with $\theta_{\rm mod}\in(-2\pi,0]$; `tau_init`'s radiation-era closed
    form; $c_s^2=$ `wPerturbations`. Do not "correct" any of these.
11. **`main.py` cannot be imported**; extract functions with `ast` (`test_main_plumbing.load_main_py_functions`).
12. **Redshift arithmetic** (`CLAUDE.md`): node lookup by exact `z`; a recovered $z$ from
    $\log(1+z)$ is only ever a quadrature endpoint.
13. **The in-flight `transfer-remedial` campaign** (README §0.2, §4.2): do not touch its files;
    its `main.py` Bessel-stage comment is theirs; the two shared test files are a stop condition
    for prompt 10.
