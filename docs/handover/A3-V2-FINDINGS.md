# A3 v2 — what it shows, and the workplan that follows

**Written 2026-10-25**, after `handover` prompt 08 closed and `rpk-v030-adoption` landed, at `71962f6`.
This is an **analysis**, not a prompt. It ran nothing on the store and changed no code. It reads
[`A3-V2-VERIFICATION.md`](A3-V2-VERIFICATION.md) §3, prompt 08's committed depth census
([`a3-v2/after/out/depth_census_08.txt`](a3-v2/after/out/depth_census_08.txt)), and the earlier
measurements it names. One new computation was made, from the committed census alone:
[`a3-v2/findings/depth20_floor.py`](a3-v2/findings/depth20_floor.py), with its output at
[`a3-v2/findings/depth20_floor.txt`](a3-v2/findings/depth20_floor.txt). It uses no store and no Ray.
The output's header records the script's and the input's SHA-256 and the `git HEAD` it ran at. A
second run reproduces everything but that `HEAD` line.

**Status of the claims.** §1 is measurement. §2 is a **hypothesis**: it fits every number below,
and §2.6 states the test that would refute it. §3's recommendation on README §7 **D1** is a
recommendation, and D1 stays the user's decision. Where this document and a board disagree, the
board is right (`CLAUDE.md`).

---

## 1. What A3 v2 settled, and what it did not

**Settled** (verification §3, the boards):
- **A complete, verified production-shaped store exists:** 9280 `QuadSourceIntegral` rows at
  `(1e-32, 1e-7)`, fingerprint `0e32f24a…`, census and soundness clean. It is kept on the Mac
  Studio.
- **`rtol = 1e-7` did not remove the depth-20 cliff:**
  - 2265 rows reach `DEFAULT_LEVIN_MAX_DEPTH = 20`, carrying 285.8 of 286.3 CPU-hours;
  - 1179 of them are stored with `total_converged = 0`;
  - 249 CPU-hours fall in the low band ($z_{\rm response} < 9$), which the baseline never reached.
- **The geometry is the baseline's.** Of the 679 depth-20 rows over 60 s, 575 have `r == k`
  (85 of them `q == r == k`) and 104 do not. Expensive rows are absent at $k = 10^5$ and at
  $k = 3\times10^8$.
- **The seam band ($z_{\rm response} > 10^5$) is cheap:** 0.47 CPU-hours over 6080 rows. There is
  no cost problem at the hand-over.

**Not settled: anything about the hand-over.** Prompt 08 measured the build and its cost, and no
seam number came out of it. The `GkSourcePolicyData` summary block for A3 v2 is in the run's
`stdout.log` on the Mac Studio and was not transcribed. The hand-over's state is the one A2 left
(§3).

---

## 2. The depth-20 cliff: a phase-precision floor the integrator cannot see

### 2.1 The evidence

For each depth-20 row in the mid and low bands, `depth20_floor.py` compares the error the row
declares, `total_abserr / |total|`, with $\varepsilon k\eta(z_{\rm response})$. That quantity is
the double-precision floor of a phase of size $k\eta$. $\eta$ is a flat ΛCDM conformal time
written out in the script, and it agrees with the pipeline's own $|\theta_G|/k$ (`gktk-remedial`
Layer 1) to 0.09–0.13 % at five redshifts. Medians by $k$ and band:

| $k$ (/Mpc) | band | rows | median time | median abserr/\|total\| | median $\varepsilon k\eta$ | ratio |
|---|---|---|---|---|---|---|
| 1.00e5 | low | 85 | 1.4 s | 6.9e-8 | 2.4e-7 | 0.28 |
| 9.85e5 | low | 112 | 176 s | 5.2e-7 | 2.4e-6 | 0.22 |
| 3.09e6 | low | 110 | 920 s | 3.3e-6 | 7.8e-6 | 0.47 |
| 9.70e6 | low | 104 | 2935 s | 3.5e-5 | 2.4e-5 | 1.71 |
| 3.05e7 | low | 104 | 2849 s | 2.2e-4 | 7.4e-5 | 3.90 |
| 9.56e7 | low | 104 | 540 s | 6.6e-4 | 2.3e-4 | 3.22 |
| 1.00e5 | mid | 153 | 0.6 s | 3.7e-8 | 8.7e-9 | 5.34 |
| 3.09e6 | mid | 146 | 3.8 s | 4.0e-7 | 3.9e-7 | 2.32 |
| 9.70e6 | mid | 146 | 27 s | 2.9e-6 | 1.5e-6 | 2.87 |
| 3.05e7 | mid | 121 | 115 s | 2.1e-5 | 6.5e-6 | 3.37 |

Over the 1666 mid- and low-band rows with a finite ratio:
- the log–log slope of the declared error on $\varepsilon k\eta$ is **0.84**;
- the ratio's 10th, 50th and 90th percentiles are **0.14, 2.2 and 22**.

The declared error therefore tracks the floor over five decades of it, within an order of
magnitude for most rows. The full table, every $k$, is in the output file.

**The phase really is delivered at that floor.** `docs/gktk-remedial-verification.md` Layer 1
measured $\theta_G$ through the production functions. On LambdaCDM at $z = 0.1$:
- at $k = 10^7$ the error is **3.05e-5 rad** on a span of 1.37e11 rad;
- at $k = 3\times10^8$ it is **9.77e-4 rad**, exactly one ulp of 4.1e12.

That document concluded *"nothing downstream is limited by it"*. That was right about accuracy
against the review's targets. It did not look at the cost of a Levin driver asked for
`rtol = 1e-7` against this input, and A3 v2 is the first store to reach the $k\eta$ where the two
cross: $\varepsilon k\eta = 10^{-7}$ at $k\eta \approx 4.5\times10^8$.

### 2.2 The mechanism proposed

1. **The integrand's phase is built from rounded pieces.** Each phase group's phase is the signed
   sum of the factors' own `theta_mod_2pi` remainders
   ([`phase_groups.py:542`](../../ComputeTargets/phase_groups.py)). Each remainder comes from a
   `PrimitivePhase` reduced against a global anchor, so it carries rounding jitter of
   $\sim\varepsilon k\tau$ (`[00-consumer-anchoring-floor]`).
2. **On `r == k`, the large pieces should cancel and do not.** The $G\,T_r$ product has a slowly
   varying group, in which $-k\tau_s$ from $G$ and $+r\tau_s$ from $T_r$ cancel identically. Each
   was rounded on its own before the sum, so what reaches the driver is a slow phase plus
   non-smooth jitter of order $\varepsilon k\eta$. That is about 1e-5 rad at $k \sim 10^7$ and low
   $z$.
3. **Subdivision cannot resolve jitter.** A parent region's estimate and its two children's differ
   at the jitter level at every depth, and the per-region test asks for `rtol = 1e-7`.
4. **Every exit but the depth cap is shut:**
   - `local_atol` is `atol` by length share, about 1e-38 at 10⁶ regions, so that branch is dead
     (`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`);
   - `phase_limited` tests against `_roundoff_floor`
     ([`levin_quadrature.py:344`](../../AdaptiveLevin/levin_quadrature.py)), Chen et al.
     eq. (151), which bounds the **Levin solve's own** round-off from $\theta'$. It has no term for
     error **in the phase it is given**.
5. **So `depth_max` ends it.** Each extra level doubles the region count, which is how a row
   reaches 966,508 regions.

### 2.3 What it accounts for

- **The scaling of §2.1.** The declared error tracks $\varepsilon k\eta$ across $k$ and both bands.
- **The low-band plateau** that the tolerance sweep called representation-limited
  (`QUADSOURCE-TOLERANCE-SWEEP.md`, phase 3b). Four rungs agree to ~1e-5 on
  $k = q = r = 3.05\times10^7$ at $z = 0.1$, where $\varepsilon k\eta = 9.2\times10^{-5}$.
- **The cost's growth as $\eta_R^3$ at fixed triple,** with median cost flat: only rows whose
  floor exceeds `rtol` take the cliff.
- **The geometry.** `r == k` is where a slow group forms from two large cancelling phases.
- **The exact flavour's flat cost against the realistic flavour's growth** (KT §8; A2 Table 8).
  This is relevant to `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]`. A3 v2 uses
  production's `PrimitivePhase` for $G$ and still reaches 966k regions, so a stale $G$ phase is not
  needed to produce growth. A2 ran at `REF_RTOL = 1e-12`, which $\varepsilon x_{\rm resp}$ exceeds
  from $x_{\rm resp} \approx 4.5\times10^3$. Why the exact flavour, whose phases are also doubles
  of size $x$, stays flat is **not** explained here; one possibility is that its composed phases
  are exact. That is not checked.

### 2.4 What it does not account for

- **The edges in $k$.** At $k = 3\times10^8$ only two rows reach depth 20, and none costs more than
  6 s. A floor as large as 6e-4 may let `phase_limited` fire early, through its
  `phase_err > rtol * relerr_denom` arm, but that has not been checked.
- **The spread.** The ratio's 90th percentile is 22. A second, smaller mechanism may sit on top of
  the floor.
- **Whether the fast groups stall too.** In a sum group ($G + r$) the jitter is just as large, but
  the group oscillates at $\sim 2k$. Which group carries the depth-20 regions is not recorded in the
  census, so the census cannot say. §2.6 item 1 asks.

### 2.5 Two side findings

1. **`total_converged` is checked per group, against the group's own value.** In
   `phase_group_Levin_integral` ([`QuadSourceIntegral.py:1080`](../../ComputeTargets/QuadSourceIntegral.py)),
   each Levin call's `converged` compares its error with `rtol` times **its own** value, and the
   groups cancel against each other in `total`. So a converged flag says little about
   `abserr/|total|`:
   - of the 1086 depth-20 rows flagged converged, **257** declare `abserr/|total|` above 1e-6 and
     **132** above 1e-4;
   - the largest is **5.0e-2**.

   A `--require-converged` gate on this flag — the shape
   `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` proposed — would admit those
   rows. A gate belongs on `total_abserr / |total|`.
2. **Five depth-20 rows store `abserr/|total|` = +inf**, which is presumably `total_abserr = +inf`:
   the census writes `nan`, not `inf`, for a zero `total`. All five have $q = r = 9.56\times10^7$,
   and each cost under 2 s. This is presumably `levin-refactor`'s
   `[04-roundoff-floor-can-be-infinite]`, now reaching stored data. That board said *"any future
   code that sums … must not assume finiteness"*, and `total_abserr` is such a sum. It was not
   confirmed against the store.

### 2.6 The test that decides it

Read-only on A3 v2, or offline. It needs no rebuild.

1. **Instrument.** Take about six mid-band depth-20 rows of 100–800 s, `r == k ≫ q`; the
   $k = 3.09\times10^6$ and $9.70\times10^6$ mid rows are the candidates. Re-run each with
   per-group region counts and depths recorded, and the parent–child discrepancy at the deepest
   regions. The hypothesis predicts:
   - the depth-20 regions sit in the slow group;
   - the discrepancy there is at $\varepsilon k\eta$ times the region's envelope, and does not fall
     with depth.
2. **Inject.** On a cheap row, or on A2's realistic fixture at large $x$, add non-smooth noise of
   amplitude $\delta$ to the composed `theta_mod_2pi`. The hypothesis predicts depth 20 once
   $\delta$ exceeds roughly `rtol`, and the region count rising as $\delta$ does.
3. **Remove.** Compose the slow group's phase so its $\tau_s$ terms cancel before reduction (§2.7
   layer 1), and re-run the rows of item 1. The hypothesis predicts:
   - that group's depth falls well below 20;
   - its cost falls towards the ~1 s median;
   - `total` moves by less than the row's old declared error.

If item 1 finds the depth-20 regions in a group with no cancelling pair, or item 3 leaves the
depth at 20, the hypothesis is wrong.

### 2.7 The remedy, in three layers

1. **Compose phases from leading terms before reducing.** A group's phase would be built as
   $(s_G k + s_q q + s_r r)\,\tau_s$ plus constants plus the summed residuals, then reduced once.
   The constant response phase $k\tau_R$ would be taken out of the integrand. For `r == k` the
   $\tau_s$ coefficient of the slow group is then **exactly zero**, and its jitter goes with it.
   This is `[00-consumer-anchoring-floor]`'s follow-up (an `anchored(z0)` view on `PrimitivePhase`
   and a Levin-side hook), done at the group rather than the factor. It removes the floor only
   where a cancelling pair exists: a sum group keeps $\varepsilon(k+r)\tau_s$. The layer touches
   `PrimitivePhase`, `phase_groups.py` and README D2's surface.
2. **Tell the driver the input's floor.** `adaptive_levin_sincos` would take a declared
   phase-error floor. A region whose discrepancy is within that floor times its envelope is
   accepted as `phase_limited`: flagged, honest, and stopped. Layer 1 cannot remove every floor, so
   this layer is needed too; on its own it would stop the cost while certifying an answer good to
   only ~1e-4.
3. **Then decide what is left.** That means `DEFAULT_LEVIN_MAX_DEPTH`, which is owned by nobody;
   the per-region acceptance test with its dead `local_atol` branch; and what `total_converged`
   should mean (§2.5).

**Ownership: none.** `levin-refactor` is closed, `[00-consumer-anchoring-floor]` is on the closed
`GkTk-remedial` board, and the `handover` README §0.3 excludes `AdaptiveLevin/`. This document is
written to serve as the audit a new campaign is planned from.

---

## 3. The hand-over

### 3.1 Where it stands

Nothing has moved since A2 (2026-09-20):
- **Landed:** prompt 00, A1 and A2.
- **Written, not run:** A3 (the policy-geometry census) and A4 (the $G_k$ phase decision test).
- **Groupings only:** B1–B4, C1, C2, D1, D2, E1 and E2.
- **Still open:** README §7 decisions **D1** (which gap remedy), **D4** (whether D2 belongs here)
  and **D7** (what to do where the $G_k$ geometry is irrecoverable).

### 3.2 Why it governs accuracy, and the Levin floor does not

- **The seam's error is large.** A2 measured the clamp term at **3.9e-3 to 0.91** of `total`,
  independent of $x$ to $10^8$. `[12-handover-clamp-error-in-production]` measured
  **6.6e-2 to 4.6e-1** on real rows.
- **The Levin floor is small by comparison:** 1e-7 to 1e-3 (§2.1).
- **Almost every mid- and low-band row integrates across the $T_q$ and $T_r$ seams**, which sit
  near $x_T \approx 15$ and so at $z \gg z_{\rm response}$ for every $q, r \ge 10^5$/Mpc in the work
  list.
- **Self-convergence in `rtol` cannot see a representation bias.**

So A3 v2's stored `total`s are presumably good to a few percent, not to the ~1e-6 the sweep's
self-convergence suggests. **This is inferred from A2's figures, not measured on A3 v2.** §3.3's
second item measures the gap that drives it.

### 3.3 What A3 v2 makes possible, read-only

1. **A3, the policy-geometry census.** It "needs a datastore", and A3 v2's 1160
   `GkSourcePolicyData` rows on the shipping grid are the obvious one. It answers **D5** and is
   what D1 and B4 are scored on.
2. **The production gap at full coverage.** Every `QuadSourceIntegral` row stores `metadata` as
   JSON, including `partition` with `max_clamp_gap_log1pz`
   ([`QuadSourceIntegral.py:564`](../../ComputeTargets/QuadSourceIntegral.py)). The 2026-09 gap
   figures (median 1.2e-2, max 2.2e-2 in $\log(1+z)$) came from 1666/1798 sub-intervals of one run.
   A3 v2 gives them on all 9280 rows.
3. **The `GkSourcePolicyData` summary block** from A3 v2's `stdout.log`: a grep, which says whether
   any `fail`/`incomplete` row exists on the shipping grid.

### 3.4 D1 — the recommendation

**(b), overlap by construction:** continue the numeric $T$ integration two or more grid nodes past
the stop point, store them, and place the crossover at a grid node inside both representations.
- **It is the same design on both sides of the seam.** $G_k$ already has an engineered overlap, and
  B4 sizes it by construction; this gives $T_k$ the same.
- **It is the only option that leaves a stored diagnostic:** the continuity check of the follow-up
  §1.4, which nothing can run today.
- **It serves C1.** The WKB row can start at a shared grid node, rather than being keyed on a
  root-solved $z_{\rm init}$ matched as an exact float.

Against the others:
- **(c), the Taylor extension,** is confined to one file, but it keeps the gap and rests
  correctness on an extrapolation across it.
- **(a), an LG sample at $z_{\rm init}$,** needs a `redshift` row that is not on the grid.

The choice remains the user's.

### 3.5 Order

By epistemic dependency (README §4):
1. **B3** now; it has no prerequisite.
2. **A3** on A3 v2, then **D1**.
3. **B1 → B2.** Before B2 scores anything, its fixture's $G$ phase is rebuilt
   (`[02-realistic-fixture-Gk-phase-is-the-superseded-construction]`), unless **A4** shows it does
   not matter.
4. **E1**, then **E2** last.
5. **C1 and C2** are independent of all of it.

**The two halves are separate pieces of work.** The seam work happens at $x_T \approx 15$, and the
Levin floor at late source times. Every measurement on the list reads A3 v2 or runs offline, so
neither half needs a rebuild to make progress.

---

## 4. The workplan

| # | Item | Kind | Needs | Owner |
|---|---|---|---|---|
| 1 | §2.6 instrument / inject / remove | measurement | A3 v2, offline | unassigned |
| 2 | §2.7 layers 1–2 | remedy | item 1 | unassigned (new campaign) |
| 3 | §2.7 layer 3: depth cap, acceptance test, `total_converged` | decision + remedy | item 2 | unassigned |
| 4 | The A3 v2 `GkSourcePolicyData` block; the gap census from `metadata` | measurement | A3 v2 | `handover` |
| 5 | A3, the policy-geometry census | measurement | A3 v2 | `handover` prompt 03 |
| 6 | B3, the seam contract | remedy | — | `handover` |
| 7 | D1, decided; then B1 | decision, remedy | items 4–5 | user; `handover` |
| 8 | A4, then B2 | measurement | B1 | `handover` |
| 9 | E1, then E2 | measurement, decision | B2, D1 | `handover` |

The boards hold the status. This table is the plan as it stood on 2026-10-25, and later changes are
recorded on the boards.

## 5. Where this is recorded

The board entries narrowed in the same commit, each pointing here:
- on `handover`, `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`,
  `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` and
  `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]`;
- on `GkTk-remedial`, `[00-consumer-anchoring-floor]`;
- on `levin-refactor`, `[04-roundoff-floor-can-be-infinite]`.

The `handover` README §7 D1 carries a pointer to §3.4.
