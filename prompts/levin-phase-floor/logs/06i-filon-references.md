# 06i — Filon references for Z3's calls (log)

## Predictions (copied verbatim from the prompt §5, NOT YET SCORED)

Copied at **2026-11-01T01:12:45Z** (laptop clock, UTC), before the method check and before any sample was run.

## 5. Predictions, fixed before the run

Copied into the log unchanged before any sample is run. Each is scored **holds**, **fails** or
**undecided** against the rule as written.

- **R0 — the method check.** **Rule:** every closed-form cell of §1 is met by both implementations,
  and the integrand comparison holds on both samples. **Holds** if all do. A failure is a stop.
- **R1 — Filon reproduces Gauss–Legendre where it resolved.** **Rule:** on each of samples 2, 3, 8,
  9, 14 and 15, the Filon reference counts, and differs from 06e's Gauss–Legendre reference by at
  most the sum of the two scatters plus $1/100$ of the minimum abserr. Scored per route: **Route A
  holds** if samples 8 and 9 meet it; **Route B holds** if samples 2, 3, 14 and 15 do.
- **R2 — Filon resolves where Gauss–Legendre did not.** **Rule:** for a route that holds R1, each of
  its dropped samples (Route A: 10, 22, 23, 25, 27; Route B: 1, 13) counts. **Holds** if all do.
- **R3 — Z3 re-scored.** **Rule:** for every reference that counts under a route holding R1,
  $|v_{06e} - {\rm ref}| \le {\rm abserr}_{06e}$. **Holds** if none violates it. Fewer than five
  newly counted is **undecided**. The 06c column is reported beside it.
- **R4 — the cost.** **Rule:** every Route A reading takes under 10 minutes on the laptop.
  **Holds** if all do. Descriptive beside it: Route B's times, and Gauss–Legendre's from 06e.


---

# levin-phase-floor 06i — Filon references for Z3's calls

**Subject:** Filon-type reference values for the 13 captured Levin calls of 06e's Z3 check.
**Commit:** none (records committed by the orchestrator). **Dispatch SHA:** `d7cb6d36`.
**Result so far: STOPPED at Gate R0** (prompt §8, "R0 fails"; the rule's own text, "A failure is a
stop"). Every closed-form cell is met by both implementations; the integrand comparison fails as
written on both samples, each time by a documented rounding the census tree does not declare. No
sample has been run: no reading exists, and `out/readings/` was never created.

## Timeline (UTC, laptop)

- 01:12:45 — predictions R0–R4 copied above, before anything ran. Gate checked: 13 inputs OK
  against `inputs/SHA256SUMS.expected`; main checkout clean at `d7cb6d36`; `runregistrykit list`
  shows nothing running or stale; integrand tree `6d81fad`, clean.
- 01:13–01:30 — survey of the 13 calls (spans, phase totals, `theta_deriv` sign, evaluation cost);
  no integral formed.
- 01:30–01:49 — the harness written; R0 closed forms; R0 integrand comparison; Route B jitter
  diagnostic. Scripts formatted with `black`, and all three R0 outputs **re-run** at the final
  script hashes (identical numbers), rather than re-stamped.

## What shipped (under `var/levin-phase-floor-06i/`)

- `scripts/filon.py` — the two implementations: FCC in `mpmath` (40 digits, moments by forward
  recurrence at 65 digits for |λ| ≥ 2N+10, by 384-point Gauss–Legendre below), and QAWO in double,
  panel by panel, with exact `mpmath` prefactors.
- `scripts/samples.py` — loading; Route A's integrand; Route B's segments and smooth phase map.
- `scripts/route_a.py`, `scripts/route_b.py` — one reading per call (`--impl fcc|qawo --res 1|2`),
  ≤ 6 processes. Ready; **not run** on any full sample.
- `scripts/r0_closed_forms.py`, `scripts/r0_integrand.py`, `scripts/r0_jitter.py` — Gate R0, and
  a diagnostic of the cosmological phase's jitter.
- `scripts/prov.py` (provenance header), `scripts/run.sh` (runs a script at the census tree).
- `out/r0_closed_forms.{json,txt}`, `out/r0_integrand.{json,txt}`, `out/r0_jitter.{json,txt}`.

## Predictions and verdicts

- **R0 — the method check. FAILS AS WRITTEN; the integrand cell is badly posed** (README §5 rule 3:
  recorded, not re-posed).
  - *Closed forms: met.* 32 cells (2 integrands × ω ∈ {10, 10³, 10⁶, 10⁹} × 4 spans, 1 to 10⁵
    wavelengths for (1), 1 to 10⁴ for (2)), each implementation at both settings. Worst
    |error| / (10⁻¹³ max|g|/ω): FCC N=32 0.015, FCC N=48 0.026, QAWO P=2¹² 0.031, QAWO P=2⁹ 0.087.
  - *Integrand, Route A (sample 9): fails.* 14 of 20 points exceed `theta_abserr`·|F|, by up to
    4.37×10⁴ (from u = −6.85, Kη ≈ 635, upward). The phase difference is accounted for by three
    roundings the census tree's `BesselPhaseGroup` does not declare (06h §0.1): (a) the product
    fl(K̂x̂), (b) K's rounding K_lo x̂ (K_lo = −2.91×10⁻¹¹ here), (c) the abscissa fl(eᵘ). After
    removing (a)+(b)+(c), each computed exactly, the residual is ≤ 0.061 of `theta_abserr` at all
    20 points. Route A excludes these by design (prompt §0.3).
  - *Integrand, Route B (sample 2): no declaration to compare with.* Cosmological groups declare no
    phase error (`[01-cosmological-group-declares-no-phase-error]`). Route B's integrand differs
    from the captured one by up to 9.6×10⁻¹⁰|F|: `theta_mod_2pi`'s white jitter about the smooth
    phase, σ ≈ 4.6×10⁻¹⁷|θ'| on all six Route B calls. That is the size of the rounding of
    z = expm1(x) (`primitive_phase.py`, "04's I1").
- **R1–R4:** not scored. No sample was run.

## Deviations from the prompt

1. **Scripts in `var/levin-phase-floor-06i/scripts/`, run from the main checkout at the dispatch
   SHA, not from a separate worktree.** STRUCTURALLY REQUIRED: the dispatch confines writes to
   `var/levin-phase-floor-06i/`. The main checkout is clean at `d7cb6d36`, so it is the same tree.
2. **QAWO with every internal product exact.** ω is replaced by ω_r, ω rounded to 40 significant
   bits, with (ω − ω_r)s folded into the amplitude. Panels are symmetric, [−h, h] with h = 2ᵉ, so
   QUADPACK's bisection yields only dyadic centres, and ω_r times any of them, down to h/2¹³, is
   exact. A segment's remainder is peeled into dyadic panels down to 2⁻¹² of a full panel, plus one
   last small panel. IMPLEMENTATION CHOICE: the prompt's panel shift removes the large product
   ωη_a but leaves ω·s rounded inside QAWO. That rounding is ~ε of ∫|g| per panel, and on sample 25
   ε∫|f| is 20× the counting threshold.
3. **Route A panels are aligned with each constituent's representation switches** (near-region
   `PiecewiseChebyshev` breakpoints and `x_star`, mapped by u = log x − log mᵢ). IMPLEMENTATION
   CHOICE: inside each panel the amplitude and residual are then analytic.
4. **Route B's t(x) is a least-squares smooth phase, not node-by-node correction by
   `theta_mod_2pi`.** On each segment, p is a degree-M polynomial fitted by least squares to the
   unwrapped captured phase raw + rem(θ_mod − raw) at K points, and refitted to the shared edge
   values, so t is continuous. IMPLEMENTATION CHOICE among the prompt's options, and **part of the
   question to the user.** `theta_mod_2pi` carries white jitter σ ≈ 4.6×10⁻¹⁷|θ'|. A node-corrected
   map would integrate that jitter at Filon's few nodes. The estimated jitter floor (r0_jitter,
   Gauss–Legendre's node density) is 12× the counting threshold on sample 1 and 161× on sample 13,
   and a Filon rule has ~10⁴ fewer nodes than Gauss–Legendre. The smooth map references the
   integrand with the jitter averaged out. That is the limit every dense quadrature of the captured
   integrand approaches, the same way Route A excludes (a)–(c).
5. **Route B segments are aligned with every switch in the integrand:** the φ spline knots (all
   three constituents), the Green's-function amplitude spline knots, and each CumulativeTable's nodes,
   nearest-node switch midpoints and breaks (3–5 tables). IMPLEMENTATION CHOICE.
6. **Route B corrects the abscissa to first order.** The callables can only be evaluated at
   x̂ = fl(x(t)), which is the sample at t̂ = p(x̂), not at t. The value is corrected by
   −(t̂ − t)g'. FCC takes g' from its own interpolant; QAWO takes it from a degree-24 interpolant of
   F on the segment. IMPLEMENTATION CHOICE. Measured on a 12-segment mechanics test, the correction
   is ≤ 10⁻¹⁵ of g.
7. **The R0 integrand cell was decomposed into (a), (b) and (c),** and a jitter diagnostic
   (`r0_jitter.py`) was added. IMPLEMENTATION CHOICE: neither is in the prompt. Both are evidence
   for the stop, not a re-posed rule.
8. **The R0 samples and points:** sample 9 for Route A (its order-5/2 factors exercise the
   near-region residual), with 20 points uniform in u; sample 2 for Route B, with one point in each
   twentieth of the span. The "beyond" spans are 10⁵ wavelengths for (1) and 10⁴ for (2). Reference
   (2) is `mpmath` Gauss–Legendre on two-wavelength pieces at 50 digits. IMPLEMENTATION CHOICE.
9. **Mechanics tests.** `route_a.py` was run on the first 6 FCC panels and the first 600 QAWO panels
   of sample 9, and `route_b.py` on the first 12 segments of sample 2, to check the code ran and to
   time it. The values were not kept or scored. IMPLEMENTATION CHOICE, recorded because they
   touched samples before R0 held.

## Verification performed

- Inputs: 13/13 equal `SHA256SUMS.expected` before and after.
- Main checkout and integrand tree: `git --no-optional-locks status` empty on both, and no
  `__pycache__` written in the tree (every run used `-B` and `PYTHONDONTWRITEBYTECODE=1`).
- Chebyshev moments checked against `mpmath.quad` at 50 digits for λ ∈ {0, 10⁻³, 0.7, 30, 105,
  106.5, 500, ±3000} and k up to 48, on both branches: worst 4×10⁻⁴².
- Every output header carries the scripts' SHA-256, both trees' HEAD and status, and the inputs'
  SHA-256 (`prov.py`).

## Observations not acted on

- Gauss–Legendre's dropping of Route B samples 1 and 13 is explained by the captured phase's own
  jitter, not by the quadrature. The estimated jitter floor over the threshold is 12 and 161,
  against 06e's observed scatter of 96 and 1,760, and the six calls rank the same way. Opened as
  `[06i-cosmological-phase-jitter-floors-z3-references]` in `issues.md`.
- Samples 8, 10, 22, 23 and 25 have all three orders ½, so R ≡ 0 and the amplitude is a power
  of η. Their Route A integral is then a closed form in Si and Ci, an independent check that was
  not used.
- QUADPACK flags ier ≠ 0 (roundoff) on most panels at the requested epsrel 10⁻¹³ and 1.2×10⁻¹⁴.
  The R0 errors stay ≤ 0.09 of tolerance. The counts are recorded per reading.

## State handed on

- Stopped at Gate R0, with the question in the handback. Nothing is running. The harness is ready.
- Estimated cost after resumption: all of §2–§3, both implementations at both resolutions,
  ~1 h wall on 6 processes.

---

## The user's ruling on the R0 stop (received 2026-11-01T01:54:00Z)

The question was relayed verbatim, with the orchestrator's view. The user: *"Yes, (A), and your two
additions"*. What changed:
- **R0 is recorded as fails as written; the integrand cell badly posed.** The closed forms are met.
  Route A's integrand agrees to 0.061 of `theta_abserr` once the three undeclared roundings are
  removed. Route B's agrees up to the jitter. §2–§3 run with the constructions as built: Route A
  with exact K, C and e^u, and Route B with the least-squares smooth phase. R1–R4 are scored as
  written.
- **Addition 1 (descriptive):** for every Route B call, the jitter's estimated contribution to the
  integral.
- **Addition 2 (descriptive):** a closed form in Si and Ci for samples 8, 10, 22, 23 and 25 (all
  orders ½), against each Route A reading.
- **Issues.** `[06i-z3-r0-integrand-rule-has-no-declaration-to-meet]` is dropped: the orchestrator
  records it as a prompt amendment. The jitter finding becomes a proposed **narrowing** of
  `[05a-t-phase-abscissa-rounding-remains]`, not a new issue.

---

## After the ruling: §2–§3 run, R1–R4 scored (2026-11-01, 01:52–02:40 UTC)

**Timeline.**
- 01:52 — addition 1 added to `route_b.py` (FCC path only); `closed_form_a.py` written and run
  (addition 2).
- 01:55–01:57 — Route A FCC, 14 readings.
- 01:56:54–02:27:57 — the other 38 readings, run in sequence by `scripts/batch.sh` from
  `out/jobs/rest.txt`, in one background job of this session; log in `out/run_log.txt`.
- 02:28 — `r0_integrand.py` and `r0_jitter.py` re-run at `route_b.py`'s final hash. The numbers are
  identical to the first run.
- ~02:35 — `diag_fit.py` on samples 1 and 13, and `score.py`.

### Result

R0 fails as written (ruled badly posed). **R1 holds** on both routes. **R2 holds** on both routes.
**R3 fails**, on sample 3 alone. **R4 holds.** All 13 references count.

**§8 flag:** on sample 8 the two implementations differ by 5.1× the larger implementation's own
scatter. That is 3.5×10⁻²⁵, or 1.3×10⁻⁸ of min(abserr) and 1.4×10⁻¹⁶ of the value, with the closed
form lying between them. It is reported in the handback for the user to rule on. Nothing hangs on
it: every verdict is the same with sample 8 counted or not counted, except that Route A's R1 needs
sample 8.

### Predictions and verdicts

- **R0 — fails as written; the integrand cell was badly posed (the user's ruling).** The closed
  forms are met. Route A's integrand agrees to 0.061 of `theta_abserr` once the three undeclared
  roundings are removed. Route B's agrees up to the jitter (≤ 9.6×10⁻¹⁰|F|).
- **R1 — holds on both routes.** Each sample's difference from Gauss–Legendre over the allowance
  (Filon scatter + GL scatter + m/100), with Filon's scatter over m:

  | sample | GL difference / allowance | Filon scatter / m |
  |---|---|---|
  | 8 | 0.043 | 1.6×10⁻⁸ |
  | 9 | 0.10 | 9.5×10⁻⁶ |
  | 2, 14 | 1.4×10⁻⁴ | 9.9×10⁻¹⁰ |
  | 3 | 0.22 | 2.4×10⁻⁶ |
  | 15 | 0.016 | 3.4×10⁻⁷ |

  All count.
- **R2 — holds on both routes.** Every dropped sample counts. Scatter / m: sample 10, 5.1×10⁻⁶;
  22, 3.1×10⁻⁵; 23, 2.2×10⁻⁵; 25, 1.0×10⁻³; 27, 3.1×10⁻⁴; 1, 1.0×10⁻³; 13, 4.5×10⁻³.
- **R3 — fails.** There are 13 counted references, 7 of them newly counted (10, 22, 23, 25, 27, 1,
  13). One violation: **sample 3, |v₀₆ₑ − ref|/abserr₀₆ₑ = 1.2410** (06c: 1.1932; 06e's
  Gauss–Legendre score was 1.245). The other ratios under 06e (06c in brackets):

  | sample | 06e | 06c |
  |---|---|---|
  | 8 | 0.99998 | 0.9998 |
  | 9 | 0.99954 | 0.9927 |
  | 10 | 0.9925 | 0.9576 |
  | 22 | 0.0015 | 0.0029 |
  | 23 | 0.9822 | 0.9607 |
  | 25 | 0.0287 | 0.00017 |
  | 27 | 0.9762 | 0.6689 |
  | 1 | 0.3326 | 0.0642 |
  | 2, 14 | 0.99848 | 0.99847 |
  | 13 | 0.0838 | 0.00093 |
  | 15 | 0.7814 | 0.7807 |

  The jitter does not explain sample 3's violation. Its excess is 0.241 abserr, 8.5×10⁻³⁵, while
  the jitter's contribution at Filon's nodes is 2.6–4.9×10⁻³⁹ and its floor at Gauss–Legendre's
  density is 6.8×10⁻³⁷. Shifting the reference by the measured jittered-minus-smooth leaves the
  ratio at 1.2410 (addition 1). It is Levin's error, as 06e's D3 found (a trusted leaf with
  r = 0.507).
- **R4 — holds.** The longest Route A reading on the laptop, with 6 processes, is 167 s (sample 10,
  QAWO res 2). Then sample 8 at 141 s, 23 at 68 s, 27 at 50 s, 22 at 36 s, 9 at 14 s, 25 at 5 s.
  Descriptive: Route B's longest is 203 s (sample 15, QAWO res 2). Gauss–Legendre's two runs per
  sample took 40–14,301 s each (`results.txt`).

### Additions 1 and 2 (descriptive)

- **Addition 1 — the jitter's contribution, Route B.**
  - *How it is estimated.* In each FCC reading, the same nodes and weights are applied to the
    captured, jittered phase: g_j·e^{iδ_j} with δ_j = rem(θ_mod(x̂_j) − p(x̂_j)). The reading
    reports the jittered value minus the smooth one.
  - *The floor beside it.* The 1σ floor at these nodes is √(Σ_j Im(h e^{ic} W_j g_j)² σ_seg²).
    `r0_jitter`'s floor at Gauss–Legendre's density is reported with it.
  - *Measured* (jittered − smooth at res 1 and res 2; floor at res 1 and res 2; GL-density floor;
    m/100):

    | sample | jittered − smooth | 1σ floor at these nodes | GL-density floor | m/100 |
    |---|---|---|---|---|
    | 1 | −2.7, +1.8 ×10⁻³⁸ | 2.9, 6.3 ×10⁻³⁸ | 1.1×10⁻³⁶ | 9.1×10⁻³⁸ |
    | 2, 14 | −7×10⁻⁴⁴, +3.6×10⁻⁴³ | 1.1, 2.2 ×10⁻⁴³ | 1.3×10⁻⁴¹ | 4.7×10⁻³⁷ |
    | 3 | −2.6, +4.9 ×10⁻³⁹ | 3.1, 6.4 ×10⁻³⁹ | 6.8×10⁻³⁷ | 3.5×10⁻³⁶ |
    | 13 | +1.2×10⁻³⁹, −1.05×10⁻³⁸ | 2.7, 6.5 ×10⁻³⁹ | 2.0×10⁻³⁷ | 1.2×10⁻³⁹ |
    | 15 | −3.5×10⁻⁴¹, −1.8×10⁻⁴⁰ | 1.0, 2.1 ×10⁻⁴⁰ | 2.5×10⁻³⁸ | 2.0×10⁻³⁷ |

    The differences are of the size of the floor, as white jitter should give.
  - *Sample 13:* at Filon's node count, the jittered integrand alone moves the value by 8.6× the
    counting threshold. That is why the smooth phase is the reference.
- **Addition 2 — the closed form for the five order-½ calls.**
  - *The closed form:* I = s_f A [Ci(|K|e^{u_b}) − Ci(|K|e^{u_a})], with A = (2/π)^{3/2}/√(m₁m₂m₃),
    b = 0 and C = 0. s_f = −1 is the sign of the cos-slot amplitude; the sin slot is 0.
  - *Its basis, checked on sample 8:* R ≡ 0 at 9 points; the captured amplitude equals −A to 1 ulp;
    and `mpmath.quad` over 100 wavelengths at the bottom of the span and about its geometric middle
    agrees with the Ci difference to relative 0 and 1×10⁻⁴⁷.
  - *Each reading minus the closed form:*

    | sample | readings − closed form | max \|diff\| / own scatter | / m |
    |---|---|---|---|
    | 8 | −2.5, −1.8, +2.1, +1.8 ×10⁻²⁵ | 0.54 | 8.9×10⁻⁹ |
    | 10 | ≤ 3.6×10⁻²⁹ | 1.07 | 5.5×10⁻⁶ |
    | 22 | ≤ 1.5×10⁻²⁹ | 0.81 | 2.5×10⁻⁵ |
    | 23 | ≤ 1.9×10⁻²⁹ | 0.62 | 1.4×10⁻⁵ |
    | 25 | ≤ 1.4×10⁻²⁹ | 0.66 | 6.8×10⁻⁴ |

    Every reading sits within about its own scatter of the closed form, and far below m/100.

### Route B's t(x): construction and bound, measured on every sample

- *Construction:* as deviation 4 above, with (M, K) = (20, 241) at res 1 and (24, 301) at res 2.
- *Monotone:* every segment, on all six calls.
- *Stationary points of θ:* none. No sign change of `theta_deriv` on a 20,001-point grid, and p' has
  `theta_deriv`'s sign on every segment.
- *The branch margin:* |rem(θ_mod − raw)| ≤ 1.4×10⁻⁷ rad (sample 15), far below π.
- *The jitter σ per segment:* ≤ 2.2×10⁻⁸ rad.
- *The abscissa:* |t̂ − t| ≤ 3.3×10⁻⁷ rad. The first-order correction is ≤ 10⁻¹⁵ of g, and it moves
  the integral by ≤ 2.2×10⁻⁴⁰ (sample 1), against m/100 = 9.1×10⁻³⁸.
- *The phase-error bound:* the map's own phase error is folded out at every node (mpmath at 40
  digits).
- *The fit-noise bound* (1σ, propagated through J_k):

  | sample | 1σ | m/100 |
  |---|---|---|
  | 1 | 7.4×10⁻³⁹ | 9.1×10⁻³⁸ |
  | 2, 14 | 2.6×10⁻⁴⁴ | 4.7×10⁻³⁷ |
  | 3 | 8.0×10⁻⁴⁰ | 3.5×10⁻³⁶ |
  | 13 | 5.6×10⁻⁴⁰ | 1.2×10⁻³⁹ |
  | 15 | 2.6×10⁻⁴¹ | 2.0×10⁻³⁷ |

  **Below 1/100 of the minimum abserr on every sample**, so §8's t(x) condition does not trigger.
- *Fit design measured directly* (`diag_fit.py`, four designs). Sample 1: spread 1.3×10⁻³⁸
  (0.15 of m/100). Sample 13: spread 6.1×10⁻⁴⁰ (0.50 of m/100). It is spread over the
  low-θ' segments at x ≈ 19–21.3, as fit noise, and is consistent with the 1σ bound.

### Deviations after the ruling

10. **Addition 1's estimator** (the jittered value at the same FCC nodes, and the 1σ floor formula),
    in `route_b.py`'s FCC path. IMPLEMENTATION CHOICE, under the user's ruling. This edit to
    `route_b.py` came after the R0 records, so those were re-run at the final hash (identical).
11. **Addition 2** (`closed_form_a.py`), as ruled. IMPLEMENTATION CHOICE: Ci only, since b = 0 and
    C = 0 on all five; s_f read off the captured callable.
12. **The readings were not registered with `runregistrykit`.** STRUCTURALLY REQUIRED: the dispatch
    confines writes to `var/levin-phase-floor-06i/`, and a registered run writes `var/runs/`. No
    reading exceeded 4 min. The sequence of 38 took 31 min, as one background job of this session,
    with ≤ 6 processes at any time.
13. **Two choices in the scoring.** IMPLEMENTATION CHOICE.
    - The reference value is FCC res 2.
    - §8's "the larger scatter" is read as the larger of the two implementations' own res-1/res-2
      scatters. The four-reading scatter is the counting rule's.
14. **`diag_fit.py`**, a diagnostic of the fit design's effect on samples 1 and 13. IMPLEMENTATION
    CHOICE, added to explain a design dependence on sample 1. It had first been misread at 10× its
    size (9×10⁻³⁸ instead of 9×10⁻³⁹) in a progress check. No record carries the misreading.
15. **Execution slips, none affecting a reading.** UNINTENDED DRIFT.
    - A first batch launch failed on shell quoting; `run_log.txt` shows it, and nothing ran.
    - A per-reading monitor woke the orchestrator on every reading. On their instruction it was
      replaced by one waiter.
    - The waiter's first version fired a false "over 30 min" on a GNU `date`, while the reading had
      run 28 s; it was relaunched with `/bin/date`.
    - The waiter's diagnostic step did not run (zsh does not word-split), and was run separately.

### Observations not acted on (after the ruling)

- **Sample 8, §8.** The implementations differ by 3.5×10⁻²⁵. That is 1.4×10⁻¹⁶ of the value and
  1.3×10⁻⁸ of min(abserr). Each implementation's own scatter is smaller: 6.9×10⁻²⁶ for FCC and
  3.2×10⁻²⁶ for QAWO. The closed form lies between them: FCC r2 is −1.8×10⁻²⁵ from it and QAWO r2
  +1.8×10⁻²⁵. FCC's offset is the systematic 1-ulp rounding of the captured amplitude (2.2×10⁻¹⁶ of
  A), and QAWO's is double-precision summation. Reported for a ruling (handback).
- **The design dependence.** Samples 25 and 13 now count at 1.0×10⁻³ and 4.5×10⁻³ of m. 06e's
  Gauss–Legendre references for them were 9.2 and 10.5 abserr away from these. That is consistent
  with Gauss–Legendre's scatter there (66 and 17.6 of m), and for 13 with the jitter.
- **Sample 27,** dropped by Gauss–Legendre, is 1.05 m from its reference (allowance ratio 1.35).
  Not an R1 sample.
- **QUADPACK reports ier ≠ 0** (roundoff) on most panels at the requested tolerances. The QAWO
  readings agree with FCC to ≪ m/100 throughout.
- **The census's own error on samples 1, 13 and 25** is 0.33, 0.084 and 0.029 of its abserr under
  06e, against Gauss–Legendre's 0.63, 10.4 and 9.2.

### Verification performed (after the ruling)

- Inputs 13/13 OK by SHA-256 at the end. The main checkout and the integrand tree are clean
  (`git --no-optional-locks status`), and no `__pycache__` was written in the tree.
- Every record's header: main checkout `d7cb6d36`, integrand tree `6d81fad` (0 status lines).
- Every record's producing scripts (`filon.py`, `samples.py`, `prov.py`, `route_a.py`,
  `route_b.py` and its own script) equal the final files by SHA-256. The only mismatch is
  `r0_closed_forms.json`'s inventory hash of `route_b.py`, which that script does not import.
- Formatting: `black --check` is clean on all scripts.

### State handed on

All 52 readings are in `out/readings/`. `out/scores.json` and `out/results.txt` hold the tables and
verdicts. `out/closed_form_a.json` holds addition 2, and `out/diag_fit_s01.*` and
`out/diag_fit_s13.*` the fit-design diagnostic. Nothing is running. §4 was not run; R1 holds on
both routes, so it is available to the user.

## The orchestrator's review and the records (2026-11-01)

**The review passed.** On the laptop:
- **Scope:** no commit and no code change; the inputs, 07's stashes and `wip/` unchanged by
  SHA-256; the integrand worktree clean; nothing under `var/` newer than the dispatch marker outside
  `var/levin-phase-floor-06i/`.
- **R0** re-checked against the orchestrator's own antiderivative at $\omega = 10^6$ and $10^9$,
  $10^5$ wavelengths: the errors as recorded here, at most 0.072 of tolerance.
- **Sample 8's closed form** re-derived from the captured inputs: all four readings within
  $3\times10^{-25}$.
- **Determinism:** sample 9's FCC reading re-run in a fresh process, value bit-identical; the
  original reading file restored.
- **R1–R4** re-scored by the orchestrator's own reader: the same verdicts and numbers.

**The user's rulings:** sample 8's reference accepted (*"I accept sample 8."*); the 17
never-referenced Z3 samples left for 06g. Both are in the prompt's §9 and the board's Decisions,
**06i rulings**.

**The records:** `scripts/`, `out/`, `verification.md` and `issues.md` copied to
`docs/levin-phase-floor/06i/`, with the inputs' SHA-256; this log; `VERIFICATION.md` §6i; the
board's 06i row and §3; `docs/OPEN_ISSUES.md`. The inputs stay in the laptop's
`var/levin-phase-floor-06i/inputs/`.
