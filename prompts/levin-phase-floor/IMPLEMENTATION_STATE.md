# levin-phase-floor campaign — implementation state

**Last updated:** 2026-10-28 · **Status: LAYER 1(a) AND THE STORED PHASE LANDED FOR $G$ AND $T$ — 01, 03, 04, 05, 05a and 06
landed, 02 stopped and superseded; gate G1 answered 2026-10-26; G2 written up and answered in
part on 2026-10-27 (§2.3); 06a's depth census landed the same day (C1 and C2 hold), and G2 is
answered: layer 2 is not needed at this code. Next are 07 (the stationary-point guard,
`[06-levin-accepts-a-region-across-a-stationary-point]`, the generic Levin integrator only) and 08
(fact-finding for gate G3). 07's Part A stopped before its §2 on 2026-10-27; on the user's decision
of 2026-10-28, 06b (the round-off-limited exit checks its premise,
`[07-roundoff-exit-assumes-bisection-cannot-help]`) lands first, and 07 resumes on it. 06b's code (§1–§3) landed on the laptop on 2026-10-28; its rehearsals R15/R16 and its census ran on the Mac Studio the same day and landed: W0, W1 and W3 hold, W2 (honesty where it moved) and W4 (the cap) fail, and its header issue stays open until W2 is fixed. 07's §2 then stopped again, no rung of $\Phi_w$ passing its rule because each region is accepted against its own value while the total cancels; on the user's decision the same day, 06c (global error control in the Levin driver, `[07-per-region-acceptance-misses-a-cancelling-total]`) lands after 06b and before 07's §2.** 01 ran on the Mac Studio on 2026-10-25, in parallel with
`handover` 11. Its review stopped at check 9, a records defect, and was corrected on committing by
the user's decision. **H-cancel fails P1 and P2; H-anchor holds P1–P3** (§2.1). 01 opened two
issues (§3). 03 ran on the Mac Studio the same day, Ray-free, and its review passed 9/9. **M1, M2,
M4 and M5 hold; M3 fails on its slope clause, so the mechanism as written does not hold**, and G1
decides how that is read (§2.2). H-anchor holds in every scored group. 03 opened three issues (§3).
**02 stopped at step 3 and is superseded by 03** (user, 2026-10-25):
- its noise sweep was infeasible and is not run in any form;
- N1–N4 are not scored, and N4 moves to 04.

**04 landed** (2026-10-25, on the Mac Studio at the pre-B1 tree `98e68d3`). It stopped once at step 5,
on two gates whose round-off bounds were incomplete; the user amended them (prompt §8), and its review
passed. **With the floor removed the cap goes:** R1 holds on 7 of 7 rows, with the region count down
to 0.20–0.49 % of production's. **Taking $k\tau_R$ out of $G$'s phase alone does the same, region for
region** (A1 holds, and A matched X on every row). **N4 fails on 3 of 7 rows:** the stored
`total_abserr` understates the jitter's own effect by 1.16–1.94×. R2 holds, and Q5's number is at most
6.7e-6. 04 opened two issues and narrowed four here and on other boards (§3). **Gate G1 was answered
on 2026-10-26** (Decisions below): H-anchor; layer 1(a) here, after the fingerprint fix; the
producers' stored phase after it.

**05 landed** (2026-10-26): layer 1(a) is code `120fdfe`, with the comparison tool
`tools/store_value_diff.py`.
- **$G$'s remainder is now formed node by node,** against $2\pi$ itself. $T$ is unchanged.
- **R9 and R10** (Mac Studio) both give `1b332ffa…`, and every value table is identical. Only
  `QuadSourceIntegral`'s values move: 542 of 864 totals, by a median 8.7e-5 of their error.
- **On A3 v2's depth-20 rows still capped under B1's code,** the change removes the cap on all 11:
  644 s → 4.4 s, and L3 holds on all 11.
- **Scored as written,** L1, L1q and L3 fail, only on rows B1 had already uncapped and on R5's
  55-region chain. L2 holds.
- **Two stops,** both answered by the user. The prompt's §8 wording counted timings and labels as
  values. The reading of "depth-20 row" was settled as the capped-under-$p$ reading.
- **The user closed `[00-consumer-anchoring-floor]` for $G$** on that reading (`GkTk-remedial` §4).
- 05 opened three issues and narrowed four (§3). The finding that matters beyond the campaign:
  **B1 moves 36 of 39 stored A3 v2 totals beyond their stored error**.

**05a written** (2026-10-26), on the user's decision the same day (Decisions, **05a**): $T$'s
remainder gets the same two corrections as $G$'s, the node anchor and reduction against $2\pi$.
Writing it opened `[05-t-remainder-keeps-the-anchoring-floor]` (§3). A planning measurement on 05's
run, `measure/convergence_kinds.py`, finds why 9 of 05's 11 uncapped rows stay unconverged: no
group reaches the cap. Every unconverged group either cancels across its regions, or is one region
at depth 0 whose declared error is the driver's round-off bound. It narrows `handover`'s
`[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]`, for G2.

**05a landed** (2026-10-26): $T$'s remainder is node-anchored and reduced against $2\pi$, code
`1fc349c`.
- **R11 and R12** (Mac Studio) both give `1b332ffa…`, and every value table is identical. Only
  `QuadSourceIntegral` moves: 424 of 864 totals, by at most $8.8\times10^{-5}$ of their error. No
  depth, region count or convergence flag changes.
- **On 05's 39 rows:** T0, T1, T2 and T3 hold, and T4 fails on 05's two L3 rows, unchanged.
  - Under $t$, $T$'s jitter sampled in $z$ falls from 0.20–0.37 to at most $1.3\times10^{-5}\,\varepsilon|\theta_T|$.
  - $T$'s two errors moved a total by at most $4.7\times10^{-6}$ of its declared error, and
    changed no Levin call.
- **T1 was amended before the run** (prompt §9). After the run its step rule was read by the
  true step, both times by the user. Read literally it fails on 36 aliased windows.
- **The user closed** `[05-t-remainder-keeps-the-anchoring-floor]` and
  `[04-g-phase-is-reduced-against-the-double-two-pi]` (§4).
- **05a opened** `[05a-t-phase-abscissa-rounding-remains]`: the rounding of $z = {\rm expm1}(u)$
  is what production's $T$ phase still carries at $z \gg 1$ (§3).

**06 written** (2026-10-26), G1 Q3's prompt for the producers' stored phase (README §6.2 C8).
Each `GkWKBIntegration` and `TkWKBIntegration` value stores its residual
$r = \Delta\theta - \rho.\delta$ from its small terms, and its remainder is reduced exactly
against $2\pi$. The consumers build $\varphi$ from $r$. It closes
`[02-consumer-phi-below-the-storage-granularity]` (`phase-representation`) when its §5–§6 hold.
Part B runs from worktrees if `handover` A4 holds the Mac Studio's main checkout.

**06's code landed** (2026-10-26, §1–§4, on the laptop; not yet pushed). Every stored remainder is
within half an ulp of $2\pi$ of the exact one; the consumers' $\varphi$ is noise-free on the test
geometry, window RMS at most $1.8\times10^{-10}\,{\rm ulp}(\theta)$ against 0.37–3.9 for the
old construction. The first dispatch stopped before any code on the payload's key set; the user
ruled on that and eight findings (prompt §9 point 6). §5 and §6 are pending; the header's issue
moves only in 06's records commit.

**06 landed** (2026-10-27): the producers' stored phase, code `dc9d28c`.
- **R13 and R14** (Mac Studio, from worktrees) both give `1b332ffa…`. Outside the five classes 06
  changes, every value table is identical. Inside them no cycle count changes, and the remainder
  moves by a median 0.3 $\varepsilon|\theta|$. 554 of 864 totals move, by a median
  $2.8\times10^{-11}$ relative.
- **On 05's 39 rows,** with the producers re-run in-process: U0 holds. **S1 holds**: $\varphi_G$'s
  storage noise goes, from production's quantised value to at most $8.8\times10^{-14}F$ in 6,700 of
  6,700 pure-WKB windows. S3 holds. S2 fails as written, where its metric measures $\varphi_T$'s own
  shape (W4). S4 fails on 5 rows and S5 on 3, both findings about the declared error.
- **Three stops**, each answered by the user: route (B) for reading A3 v2 under the new code, S1's
  pure-WKB reading, and check 4's reading of 05's classifier (Decisions, **06**).
- **The user closed** `[02-consumer-phi-below-the-storage-granularity]` (`phase-representation` §4).
- **P1, the finding that matters beyond the prompt.** Two rehearsal totals moved by up to
  $1.8\times10^4$ times their declared error. Investigated from the laptop: R13's are right and
  R14's are wrong, because `AdaptiveLevin` accepts a region containing an interior stationary point,
  the Kohri–Terada resonance, with an error estimate about $10^4$ too small. R13's noisier phase
  forced the bisection that found it by luck. Opened as
  `[06-levin-accepts-a-region-across-a-stationary-point]`, assigned to G2 with layer 3 (§3).
- 06 opened three issues (§3).

**06a landed** (2026-10-27): the depth census, G2's L2 (b). Every one of A3 v2's 9,280 rows was
re-run in-process at the current code, at `1e-7` and `1e-8`, on the Mac Studio in 45 minutes,
Ray-free (§2.3.1).
- **C1 and C2 hold:** no row has a region at depth ≥ 18 at either tolerance. The deepest row is at
  12 and 13 respectively.
- **C3 fails:** 19 rows (`1e-7`) and 29 (`1e-8`) exceed 06's largest depth, 11. Every one holds an
  interior stationary point, and its deep regions sit at that point.
- **C4 holds:** 06's 70 units are reproduced bit for bit.
- **Beyond the prompt:** 720 calls store-wide accept a region across a stationary point, 623 of
  them as a single region. B1's movement of A3 v2's stored totals reaches 6,187 of 8,722 rows.
- **The review** stopped on check 9 alone, a superseded script's hash with no kept file. The user
  accepted the record as it stands.

The plan is [`README.md`](README.md), written on 2026-10-25 at `effba46`. Its audit is
[`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §2.

The campaign owns the depth-20 cost cliff of the `QuadSourceIntegral` Levin integration, and the
honesty of the error and convergence a row stores. It first runs findings §2.6's test, extended to
two named hypotheses, H-cancel and H-anchor (README §0.2). Then it lands the remedy layers G1
admits.

**Campaign:** [`README.md`](README.md) ·
**Code:** none in 01–04; layer 1(a) in 05 (`120fdfe`, $G$) and 05a (`1fc349c`, $T$); the producers' stored phase in 06 (`dc9d28c`) ·
**Verification:** `docs/levin-phase-floor/VERIFICATION.md` (created by 01), with each prompt's
records in `docs/levin-phase-floor/NN/` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.24

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row is added, moved or deleted, and the count and date in its header are corrected. An issue held
> on another board is narrowed or closed **there**. See `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-25 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** | Open this campaign from findings §2: the §2.6 test first, then the remedy |
| **U2** | A3 v2 analysis runs on the Mac Studio; the laptop's Ray is left to ChamPBH and StochasticInstantons |

**Gate G1, answered by the user on 2026-10-26 (README §6.3):**

| | Answer |
|---|---|
| **Q1** | **H-anchor stands.** H-cancel is not excluded as a minor contribution, but re-anchoring $G$ alone removes the cliff (04's A1), so the remedy starts there |
| **Q2** | **Layer 1(a): $k\tau_R$ taken out of $G$'s phase.** It is not guaranteed for every future geometry, so the prompt that lands it re-measures a wider sample: low band and `Gqr`-dominated rows. The large constant is reduced against the true $2\pi$ in extended precision, not against `TWO_PI` (`[04-g-phase-is-reduced-against-the-double-two-pi]`) |
| **Q3** | **Here, and as early as possible**, before anything else is stored. Layer 1(a) is consumer-only and lands in this campaign, not as `handover`'s D2. The noise in the producers' stored phase samples is not removed by regenerating the store. It needs its own method, and this campaign takes it up for both `GkWKBIntegration` and `TkWKBIntegration` after layer 1, before the next production build (`[02-consumer-phi-below-the-storage-granularity]`, assigned here) |
| **M3** | **Its slope clause is badly posed** (`[03-m3-slope-clause-is-badly-posed-at-round-off]`). Its ratio clause and 04's R1 carry the comparison, so the mechanism is recorded as holding (§2.2). 03's verdict as written stays beside it |
| **Layer 2** | **Deferred to G2.** Whether the driver must be told its input's floor, and whether that must reach the Clenshaw–Curtis branch (`[03-depth20-regions-are-all-clenshaw-curtis]`), is decided after layer 1 and the wider re-measurement |
| **Order** | **05 now; no fingerprint change** (revised twice the same day). The fingerprint certifies identities, tags and value-row counts, never computed values, by design, so the issue was a misreading and is closed (`handover` board §4). A prompt that changes how values are computed certifies them directly, by a rehearsal pair compared by identity (`CLAUDE.md`, campaign conventions rule 7). 05 adds the tool, `tools/store_value_diff.py`. *(Withdrawn: first, the fingerprint before 05, on the orchestrator's mistaken premise that a value table would make it see values; then a DatastoreKit computed-values feature, judged out of proportion by the user.)* |

**Made by the user after 05, 2026-10-26:**

| | Decided |
|---|---|
| **05a** | **Rows 1 and 2 for $T$, "in exactly the same way" as 05 did them for $G$:** the node-anchored remainder and reduction against $2\pi$, in a prompt 05a before 06+. G1 Q2 had left $T$ unchanged, and 05's closing note recorded only that no sampled row needed it. $G$ and $T$ enter the integrand on an equal footing, and $T$ carries both known errors. The case is correctness, not cost: neither error has been seen to drive the bisection. Row 3, the producers' stored samples, stays G1 Q3's 06+ for both. *Orchestrator's note, writing 05a:* for $T$ the rounding of $z = {\rm expm1}(u)$ (04's I1) is the same size as the floor wherever $z \gg 1$, so row 1's gain in production is expected to be of order one there; 05a measures both (its §0.1) |
| **05a §9** | **T1 amended before §6 ran** (`2db3f9e`): the code's tests showed that a $10^{-6}$ window is not resolved at large $\lvert\theta_T\rvert$, and that between nodes the remainder keeps $\varepsilon k\lvert p_z\rvert$. Each window is narrowed until resolved; the bound under $t$ admits $\varepsilon(k\max\lvert p_z\rvert + 2\pi)$. The three stale comments the code's log named were corrected (`bd38e50`) |
| **05a T1** | **T1 is read by the true step**, $\lvert d\theta_T/dz\rvert(b-a)/64 < \pi/2$, not by `numpy.unwrap`'s folded step. The user chose it without having seen either reading's numbers, which already existed on the Mac Studio. The user then confirmed it once the timing had been corrected. T1 holds (70 of 70); the literal reading's 36 aliased failures stay beside it |
| **05a close** | **Both 05a issues closed** although T4 fails, since T4's failure is 05's L3 on the same rows and ratios, unchanged by $T$; X2 opened as an issue |

**Made by the user during and after 06, 2026-10-26 and 27:**

| | Decided |
|---|---|
| **06 §9** | **Six amendments to the prompt** (`59d4070`, `7b6c1cd`, `9f66622`): $\varphi_G$'s sign; R14 must give `1b332ffa…`; the cycle-change bound; S1 scored on pure-WKB windows only (a numerically seeded value's $\Delta\theta$ carries the numeric integration's own error); two source greps and the payload's key set named in §2; $\varphi_G$ and the agreement check in `_build_phase`, a disagreement warning rather than raising; closed-form fixtures forming the residual in double |
| **06 route (B)** | **$s$ reads a certified scratch copy of A3 v2** with the three nullable columns added, since no post-06 tree can open a pre-06 store: *"We have to be practical and pragmatic. The repo rules are there to prevent cheating, not to get in the way of honest work."* |
| **06 S1** | **Pure-WKB means `z_init == z_source`**, reading (I): *"Clearly (I)."* Ruled with no S1 number in existence. The prompt's "null" was the laptop's error |
| **06 check 4** | **Passes**: 05's classifier exempts `QuadSourceIntegral` alone, and its 185 `UNCLASSIFIED` all lie inside the five classes 06 changes, with 0 outside |
| **06 close** | **`[02-consumer-phi-below-the-storage-granularity]` closed** on S1 and S3, with S2's failure read as the metric's (W4). R13, R14 and the scratch copy of A3 v2 are kept |
| **06 P1** | **The stationary-point guard belongs here**, as a generic fix to the Levin integrator: *"We are supposed to deliver a function Levin integrator, and this is needed whether or not we plan to use the integrator as part of the 1-loop machinery. So we should fix this now, generically."* The stationary points are located in advance from the closed-form $\theta'$, and the driver refuses a region whose $\theta'$ changes sign or nearly vanishes. The time integrals must also run late enough to include the QCD crossover and $e^+e^-$ annihilation for the configurations that resonate during them, which the advance location makes possible |

**Made by the user at gate G2, 2026-10-27** (§2.3):

| | Decided |
|---|---|
| **G2 L2** | **A census first, then the decision:** *"L2 (b) first, then make a decision whether layer 2 is needed. As you say, it doesn't look to be required."* Prompt 06a re-runs A3 v2's rows at the current code for the depth distribution; the user decides layer 2 on it |
| **G2 Q4 (i)–(ii)** | **"Prompt 07 becomes the stationary point guard."** `[06-levin-accepts-a-region-across-a-stationary-point]`, fixed generically (Decisions, **06 P1**). The round-off-limited exit where the guard fires is part of it. Its open choices, (α)–(γ) in §2.3, are proposed in 07's prompt and settled when it is written |
| **G2 L2, decided** | **Layer 2 is not needed at this code; revisit after 07 or later.** On 06a's census no Levin call bisects to the cap: the deepest row is at 12 (`1e-7`) and 13 (`1e-8`), and what depth remains is at stationary points. *"What we need is layer 3 to fix the error estimates. When that happens we may find that some of these regions need to be bisected more aggressively, and then the layer 2 question becomes open again."* Layer 3 is 07 and what G3 decides on 08. `[04-x-at-1e-8-still-reaches-depth-20]`, `[03-depth20-regions-are-all-clenshaw-curtis]` and `[05-b1-uncaps-most-cheap-depth-20-rows]` are closed (§4) |
| **07 scope** | **07 is the generic Levin integrator only.** 06 P1's requirement that the time integrals run late enough to include the QCD crossover and $e^+e^-$ annihilation, for the configurations that resonate during them, is physics: *"Leave this out of 07 and we'll make that change in a separate prompt."* Opened as `[06-time-integrals-must-reach-late-resonances]` (§3) |
| **07 (α)–(γ), (ii)** | **Settled in writing 07** (prompt §0.2). **(α)** The location uses the closed-form phase rate, never physics, and lives in the integrator's generic code, `_adaptive_levin`: *"We want all callers to be protected"*; `adaptive_levin_sincos` *"is just a facade … the wrong place for the stationary phase check, which is generic and does not depend on the basis"*. The basis supplies its phases and rates through two new interface methods. **(β)** An explicit window of phase half-width $\Phi_w$ round each point, not a bisection cascade to $6\pi$, which *"may anyway be too wide or too narrow"*: at the window's edge $\theta'^2/|\theta''| = 2\Phi_w$ whatever $\theta''$ is, and the window spends no bisection depth. **(γ)** A minimum of $|\theta'|$ counts as a nearly coalescing pair when its Airy parameter is below $\Lambda_c$: *"I'd like to see how this works out numerically"*, so $\Phi_w$ and $\Lambda_c$ are measured by fixed rules at a planned stop. **(ii)** The round-off-limited exit is barred in window pieces and guarded regions |
| **A3 v2's scratch copy** | **Kept for now** (`var/levin-phase-floor-06/wider/state/a3v2_copy/` on the Mac Studio, 06's route (B)). It is how a tree after 06 reads A3 v2's rows, for 07's and 08's re-measurements. The user is to be reminded to retire it later; A3 v2 itself is likely to be rebuilt after 07, depending on `handover` |
| **06a check 9** | **The record is accepted as it stands:** *"Yes, just accept as it stands."* `out/cmd/17` names the first, superseded `c0.py` (`b33229d8…`), whose text was not kept (deviation 15); C0's verdict rests on `out/cmd/18` |
| **07 §2 reference** | **Any acceptable method** (2026-10-28), after Part A found `mpmath.quad` as written gives no correct digit at $10^7$ rad: *"we can compute the integral by any acceptable method"*. Family (R) is computed both by the complex error function and by contour integration, which must agree: *"I would like to be sure we are getting this part of the integrator right. It is the science-critical step."* Family (C) by contour. **$\omega$:** *"some cases where `omega` is zero, and some where it is not … A few cases with `omega=1` seem sufficient"*. 07 §9 points 2–3 |
| **07 LG stop** | **Fix the round-off-limited exit first** (2026-10-28). A seeded `test_JJJ` draw missed its tolerance under 07's §1 because the exit accepted the Levin range beyond a window at depth 0, on a floor bisection would have cut by ~$10^4$. The user: *"Fix it now. A prompt that runs before 07's §2."* That is 06b; `[07-roundoff-exit-assumes-bisection-cannot-help]` (§3) |
| **07 §2 stop** | **Global error control in the Levin driver, before 07's §2** (2026-10-28). 07's §2 picked no rung of $\Phi_w$: every (R) case honest at every rung, but 1–3 per rung unconverged, all at $x_s = 0.7236$, $\omega = 0$, where $f_{\cos}(x_s) + f_{\sin}(x_s)$ nearly cancels and $|I|$ is 14–17× smaller; declared/requested 1.0–6.7 while the true error met the request in every case. Each window piece is accepted at `rtol` against its own value, so the summed allowance is `rtol`$\sum|v_i|$ against a request of `rtol`$|\sum v_i|$. The parent passes the same cases only narrowly (declared/requested 0.88, 0.30, 0.74). Λ_c came out at 4 on every rung, by the supplement. The user: *"The fully global scheme sounds like the right problem to solve, for the CC algorithm around the stationary point, and the Levin scheme in general … Otherwise we are just treating the symptoms rather than dealing with the actual cause."* and *"Yes, please go ahead with 06c. I realise all existing computed values will move under this change."* That is 06c; `[07-per-region-acceptance-misses-a-cancelling-total]` (§3). The same defect one level up, in `QuadSourceIntegral`'s per-call shares, is `[07-quadsource-applies-rtol-per-call-against-a-cancelling-total]`, for G3's Q4 (v). 06c overtakes Q4 (iv): the length share goes |
| **06b close** | **06b's records land; its header issue stays open** (2026-10-28). The Mac Studio review stopped on check 10 alone (the census log has no What shipped section, and its deviation 19, no over-budget pass, is unclassified): *"If the only issue is that the agent didn't say whether anything went over budget, then I don't care. Just accept it."* `[07-roundoff-exit-assumes-bisection-cannot-help]`: *"seems that we should keep it open until we've fixed W2 at least."* W2's failure is opened as `[06b-moved-calls-declare-less-than-their-error]`. W4's three depth-20 calls narrow `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`: *"If we already have this on the board, there's no need to duplicate."* Not opened: C3 (06a's stationary-point calls moving, *"07 will presumably validate with its reference values"*), C4 (the analytic blocks as where the exit fired, *"more a downstream consequence of another issue"*), N1 (the exit now dead on the rehearsal scope, *"not actually a problem"*) and N2 (the analytic blocks' flags not stored, *"not a problem unless we need them later"*). The analytic-block reading of W1 and W4 had been accepted before the launch: *"For 06b, that reading is reasonable."* |
| **G2 Q4 (iii)–(v), Q5** | **Fact-finding after 07:** *"After that we have the next fact-finding exercise to answer Q4(iii)-(v) and Q5."* Prompt 08 measures them with 07's guard in place, and the user answers them at a new gate, G3 |

The orchestrator's choices of method, C1–C6, are in README §6.2. The open questions for the user,
Q1–Q4, are in README §6.3.

---

## 1. Prompts

| # | Prompt | Covers | Where | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|---|
| 01 | [The phase-group census of A3 v2](01-phase-group-census.md) | cost, regions and declared error per group; P1–P3; findings §2.5 on the store; the candidates for 03 | Mac Studio, read-only, no Ray | Opus | ✍️ 2026-10-25 | ✅ 2026-10-25 on the Mac Studio, reviewed there; checks 1–8 pass, check 9 (records) failed and was corrected on committing, by the user's decision; shares and $S$ re-checked from the laptop | records in the bookkeeping commit | [log](logs/01-phase-group-census.md) |
| 02 | [Phase-noise injection](02-phase-noise-injection.md) | findings §2.6 item 2 on A2's fixtures, three arms; N1–N4 | laptop, offline, no Ray | Opus | ✍️ 2026-10-25 | ⛔ stopped at step 3, superseded by 03 (user, 2026-10-25). Step 2's bit-identity at δ = 0 held on all eight cases; the δ = 1e-4 check never ran; the probe `levin-phase-floor-02-noise-probe-20261025T183809` was stopped by SIGTERM, `killed` at 16/19, after its first noised cell ran ~69 min unfinished; the sweep was never launched; N1–N4 not scored (N1, N3 answered by 03; N4 to 04; N2 dropped) | `87813ed` | [log](logs/02-phase-noise-injection.md) |
| 03 | [Instrument seven depth-20 rows](03-instrument-depth-20-rows.md) | findings §2.6 item 1 on seven of 01's candidates: depth per group by branch; the jitter in the composed phase; the discrepancy along a region's ancestry, real and with the jitter removed; what the driver can see; how far `G+r` cancels. M0 (bit-identity) gates M1–M5 | Mac Studio; Ray-free if a stand-in read-only pool works, else under the lock | Opus | ✍️ 2026-10-25 | ✅ 2026-10-25 on the Mac Studio, run `levin-phase-floor-03-instrument-20261025T194950`; route (a), Ray-free; M0 bit-identical 7/7; store unchanged, `0e32f24a…`; reviewed there, 9/9 pass, three readings confirmed by the user on 2026-10-25 (`elapsed` not gated, M5 inapplicable on R7, the pid in `status.json`); R1 re-checked from the laptop: stored and re-run agree in `float.hex` on `total` (`-0x1.631c5efe83a46p-78`), `total_abserr` (`0x1.2f222e3d4dd6dp-95`), `WKB_Levin` and `numeric_quad` (`0x0.0p+0`), `total_converged` 0 / False | `d5f5a2b` (records) | [log](logs/03-instrument-depth-20-rows.md) |
| 04 | [Remove the floor from seven rows](04-remove-the-floor.md) | findings §2.6 item 3 in a harness, on 03's seven rows. Arm **A**: $k\tau_R$ out of $G$'s phase (layer 1(a)). Arm **X**: every factor's remainder reduced exactly, the limit of any phase layer and N4's reference. Layer 1(b) is not an arm (README §6.2 C7). Gates E1, E2, M0; scored R1 (the cap goes under X), A1 (layer 1(a) is enough where $G$'s anchor is the floor), N4, R2 with Q5's number | Mac Studio, Ray-free, **at the pre-B1 tree `98e68d3`** in a worktree: `handover` B1's strict coverage check refuses A3 v2's `QuadSource` rows, so an A3 v2 re-run at `HEAD` raises (`[12-realistic-large-x-reads-the-retired-clamp-records]`) | Opus | ✍️ 2026-10-25; §8 added on the resume (`d170e6d`) | ✅ 2026-10-25 on the Mac Studio, run `levin-phase-floor-04-remove-20261025T231110`, 28/28 units, none over budget; route (a), Ray-free; M0 bit-identical 7/7; store unchanged, `0e32f24a…`. The first dispatch stopped at step 5 (E1 (c), E2 failed by round-off, 4e-16–6.5e-14 rad); the user amended those two gates only. The resume's review passed. R1 re-checked from the laptop: stored and P agree in `float.hex` on `total` and `total_abserr` | records in the bookkeeping commit | [log](logs/04-remove-the-floor.md) |
| 05 | [Layer 1(a): take $k\tau_R$ out of $G$'s phase](05-reanchor-g-phase.md) | $G$'s remainder formed node by node: an exact remainder per table node, against $2\pi$, in extended precision and memoised, plus a small variable part. $T$, `raw_theta`, `theta_deriv`, the producers and every schema unchanged. Tests (exactness against $2\pi$, no jitter, nothing else moves), with four breakages. Adds `tools/store_value_diff.py`, which compares two stores' values by physical identity (`CLAUDE.md` rule 7), with its tests. Rehearsals R9 (parent) and R10, compared with it. A wider re-measurement on A3 v2 (up to 24 depth-20 rows by class and band, 03's seven and 8 controls), with `QuadSource` recomputed in-process; gate L0; scored L1, L1q, L2, L3 | code on the laptop; R9, R10 and the re-measurement on the Mac Studio | Opus | ✍️ 2026-10-26 | ✅ 2026-10-26. **Code** (§1–§4) on the laptop: reviewed there (nine checks, all pass) and pushed; ComputeTargets 742 → 754, tools 49 → 55. **§5** on the Mac Studio: R9 `levin-phase-floor-05-r9-parent-20261026T014729` (worktree at `e648951`) and R10 `levin-phase-floor-05-r10-reanchor-20261026T015120` (`53084dc`), both `done`, `1b332ffa…`; every value table identical; QSI 542/864 totals moved, median 8.7e-5 of the error, one row 28×; readers exit 0; lock released. Stopped at §8 on timings, labels, `parents_digest` and work counters; the user ruled the prompt wrongly worded. **§6**: run `levin-phase-floor-05-wider-20261026T020005`, 109/109, none over budget; L0 holds; L1 fails (9/24), L1q fails (1/7), L2 holds (8/8), L3 fails (29/31), as written, all failures on rows B1 had already uncapped or R5; on the 11 rows capped under $p$ the cap goes on all, L3 holds on all. Mac Studio review: stop on check 10 (two unclassified deviations), classified on committing by the user's agreement; checks 1–9 pass. Laptop re-checks: the 28× row, and `QuadSource` at $p$ on `low-G-r-2`. R9's and R10's stores were retired after the records commit, on the user's word (sidecars kept; `05/rehearsal/out/retire_r{9,10}.txt`), and the worktree was removed | `120fdfe` (code); records in the records commit | [log](logs/05-reanchor-g-phase.md) |
| 05a | [Layer 1(a) for $T$](05a-reanchor-t-phase.md) | $T$'s remainder node-anchored and reduced against $2\pi$, by `PrimitivePhase`'s existing `exact_remainder`, which `TkSourceFunctions` now turns on (a keyword, default on; closed-form test fixtures turn it off). $T$'s anchor `z_init` is off the grid, so the anchor's end partial is non-zero in production for the first time. Tests with four breakages; 05's T-unchanged test replaced. R11 and R12 compared with `tools/store_value_diff.py`. 05's 39 rows re-run, $c$ against $t$, with a probe of $T$'s jitter at each sub-interval's low-$z$ end; gate T0; scored T1–T4 | code on the laptop; R11, R12 and the re-measurement on the Mac Studio | Opus | ✍️ 2026-10-26 | ✅ 2026-10-26. **Code** (§1–§4) on the laptop: `TkSourceFunctions(..., exact_remainder=True)`; 9 closed-form fixture sites pass `False`; `test_t_phase_reanchor` (10 methods) and 4(b) in place of 05's T-unchanged test; the four breakages caught; ComputeTargets 754 → 764, the other six roots equal; reviewed there (nine checks, all pass; breakage (iv) re-applied by the orchestrator) and pushed. Three stale comments corrected (`bd38e50`); **T1 amended before §6** (prompt §9, `2db3f9e`). **§5** on the Mac Studio: R11 `levin-phase-floor-05a-r11-parent-20261026T122041` (worktree at `c4eaefd`) and R12 `levin-phase-floor-05a-r12-reanchor-t-20261026T122434` (`149c393`), both `done`, `1b332ffa…`; every value table identical, only the four volatile kinds outside QSI; QSI 424/864 totals moved, at most 2.9e-11 relative and 8.8e-5 of the error, no depth, region or flag changed; readers exit 0; lock released. **§6**: run `levin-phase-floor-05a-wider-20261026T123047`, 109/109, none over budget; T0 holds (all 39 $c$ units equal 05's bit for bit); T1 holds by the true step (70/70), the reading of record by the user's decision, and fails read literally (36/70, all aliased); T2 holds (largest depth 11); T3 holds (largest 4.7e-6 of the error); T4 fails (29/31, 05's two L3 rows, same ratios). Mac Studio review: all ten checks pass. Laptop re-checks: the QSI row of largest change, and a $c$ unit against 05's. R11's and R12's stores were retired after the records commit, on the user's word (sidecars kept; `05a/rehearsal/out/retire_r1{1,2}.txt`), and the worktree was removed | `1fc349c` (code); `bd38e50` (comments); `2db3f9e` (T1 amended); records in the records commit | [log](logs/05a-reanchor-t-phase.md) |
| 06 | [The producers' stored phase](06-producers-store-the-residual.md) | `GkWKBIntegration` and `TkWKBIntegration`'s stored phase, by its own method (G1 Q3; README §6.2 C8): each value stores its residual $r = \Delta\theta - \rho.\delta$ from its small terms (a new `theta_residual` column, on `GkSource`'s values too); the stored remainder is reduced exactly against $2\pi$; `_build_phase` and `TkSourceFunctions` build $\varphi$ from $r$, $G$'s with its $2\pi$ wraps unwrapped and checked against the rectifier. Tests with five breakages. R13 and R14 compared with `tools/store_value_diff.py`. 05's 39 rows re-run with the producers re-run in-process; gate U0; scored S1–S5, with 04's $\varphi$ metric on R1–R7 | code on the laptop; R13, R14 and the re-measurement on the Mac Studio, from worktrees | Opus | ✍️ 2026-10-26; amended before dispatch (§9 points 1–5) and after Part A's first stop (§9 point 6, `9f66622`) | ✅ 2026-10-27. **Code** (§1–§4) on the laptop: the producers form each sample's phase exactly and reduce it against $2\pi$; `theta_residual` stored on `GkWKBValue`, `TkWKBValue` and `GkSourceValue`; $\varphi_T = r$, $\varphi_G = r + k\tau.\delta(z_s, z_{\rm init})$ unwrapped in `_build_phase` and checked against the rectifier; ComputeTargets 764 → 777, tools 55 → 58; reviewed there (nine checks, breakages (ii) and (iii) re-applied) after a first dispatch that stopped on the payload's key set. **§5** on the Mac Studio, from worktrees: R13 `levin-phase-floor-06-r13-parent-20261026T215550` (`9f66622`) and R14 `levin-phase-floor-06-r14-residual-20261026T215945` (`f5cee4c`), both `done`, `1b332ffa…`; every value table outside the five classes identical; no cycle count changed; no disagreement warning; readers exit 0; lock released. **§6**: run `levin-phase-floor-06-wider-20261026T223634`, 109/109, none over budget, $s$ on a certified scratch copy of A3 v2 (route B); U0 holds; S1 holds (6,700/6,700, $\le 8.8\times10^{-14}F$), S3 holds, S2 fails (2,115/12,862, the metric's floor), S4 fails (34/39), S5 fails (28/31). Mac Studio review: ten checks, check 4 on the user's ruling. Laptop re-checks: one `GkWKBValue` row, and `mid-G-1` against 05a. **P1** investigated from the laptop: an `AdaptiveLevin` defect at stationary points, R13 right and R14 wrong. The user closed the header issue | `dc9d28c` (code); `59d4070`, `7b6c1cd`, `9f66622` (prompt §9); records in the records commit | [log](logs/06-producers-store-the-residual.md) |
| 06a | [The depth census at the current code](06a-depth-census.md) | G2's L2 (b): all 9,280 A3 v2 rows re-run in-process at `HEAD` (code `f5cee4c`'s), Ray-free, on 06's certified scratch copy, at `1e-7` and `1e-8`, for the distribution of Levin depth, so that the user can decide whether layer 2 is needed. Each producer re-run once per $k$, not per row; gate C0 (06's units reproduced bit for bit); scored C1–C4; descriptive, a stationary-point scan for 07 | Mac Studio, in a worktree, no lock | Opus | ✍️ 2026-10-27 | ✅ 2026-10-27 on the Mac Studio, run `levin-phase-floor-06a-census-20261027T023143`, 18,560/18,560 units in 45.5 min, none over budget or in error; worktree at `e05c34f`. C0 holds; **C1 holds** (0 of 9,280 at depth ≥ 18 at `1e-7`, deepest 12), **C2 holds** (0 at `1e-8`, deepest 13), **C3 fails** (19 and 29 rows above 11, all stationary-point rows), **C4 holds** (70/70). A3 v2 and its copy unchanged, `0e32f24a…`. Mac Studio review: nine checks pass, check 9 (a superseded `c0.py`'s hash, file not kept) accepted as it stands by the user. Laptop re-checks: two more of 05's rows against 06, and one census line against its checkpoint | records in the records commit | [log](logs/06a-depth-census.md) |
| 06b | [The round-off-limited exit checks its premise](06b-roundoff-exit-premise.md) | `[07-roundoff-exit-assumes-bisection-cannot-help]`: `_adaptive_levin` accepts a region as round-off-limited only if bisection would not lower its eq. (151) floor by half ($\rho = \tfrac12$: the halves' floors, already computed on the Levin branch and computed on demand on the fallback branch, against the parent's). Strictly stronger than today's test, so a call with no phase-limited region is bit-identical. Tests with three breakages and 20 seeded LiouvilleGreen draws; R15/R16; 06a's census re-run, scored W1–W4, which becomes 07's "before" | code on the laptop; R15, R16 and the census on the Mac Studio | Opus | ✍️ 2026-10-28, on the user's decision the same day, before 07's §2 | ✅ 2026-10-28, W2 and W4 failing. **Code** (§1–§3) landed on the laptop 2026-10-28 (`4ddc3c2`, reviewed and pushed): `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 0.5` and `_roundoff_exit_premise_holds`; the Levin branch checks the premise from `dataL`/`dataR`, the fallback branch solves its halves only where the exit would fire and carries them forward if it bisects; a test-only bypass keyword on `_adaptive_levin`. `test_levin_roundoff_exit` (6 methods): test 1's parent accepts $\theta = 100e^x$ on $[0, 20]$ at depth 0 through the exit, relative error $7.5\times10^{-7}$ against `rtol` $10^{-7}$, and the new code converges at $1.7\times10^{-9}$; tests 2–4 hold, test 4 bit-identical on all 25 driven problems; the three breakages caught by tests 1, 2 and 3. 20 seeded LiouvilleGreen draws: no test moves; 1,026 of 4,320 calls phase-limited at the parent, 133 after; every other call bit-identical; **one call reaches depth 20** (seed 15, `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`). AdaptiveLevin 32 → 38, the other six roots equal. Opened two issues (§3). **§4 and §5 on the Mac Studio, 2026-10-28, from worktrees** (`da5c24d`, `92bc62f`): R15 `levin-phase-floor-06b-r15-parent-20261028T135349` and R16 `levin-phase-floor-06b-r16-premise-20261028T135745`, both `done`, `1b332ffa…`; all 7 value tables outside `QuadSourceIntegral` identical; 782 of 864 rows with no phase-limited flag in R15 bit-identical, the 82 others tabulated (`total` moved on 26, by at most 0.36 of R15's `total_abserr`); readers exit 0; lock released. Census `levin-phase-floor-06b-census-20261028T161212`, 18,560/18,560 in 51 min, none over budget: **W0, W1 and W3 hold; W2 fails** (17 of 24 moved calls outside their declared error, median 1.79×, max 19.7×; closer to the reference than 06a's), **W4 fails** (three analytic-block calls at `1e-8` reach depth 20). Mac Studio review: checks 1–9 pass, check 10 (the census log's form) accepted as it stands by the user. Laptop re-checks: one moved R15/R16 row and one census unit against 06a. The header issue stays open on the user's word; W2 opened `[06b-moved-calls-declare-less-than-their-error]` | `4ddc3c2` (code); `92bc62f` (Part B dispatch); records in the records commit | [log](logs/06b-roundoff-exit-premise.md) |
| 06c | [Global error control in the Levin driver](06c-global-error-control.md) | `[07-per-region-acceptance-misses-a-cancelling-total]`, found by 07's §2: `_adaptive_levin` keeps every leaf of its partition and stops when $\sum e_i \le \max(\texttt{atol}, \texttt{rtol}|\sum v_i|)$, bisecting the refinable leaf with the largest error (QUADPACK's QAG), instead of accepting each region against its own value; a leaf is not refinable at `depth_max`, when floor-limited under 06b's premise, or when stalled (two successive splits along its line each failing to halve its error, added before dispatch: *"We don't want refinements that are harmful"*); `_local_atol` removed. Also `[06b-resolved-region-keeps-a-floor-bisection-would-cut]`. Every computed value moves (the user, 2026-10-28). Tests with four breakages and 20 seeded LiouvilleGreen draws; R17/R18 with Y1; 06b's census re-run, scored X1–X5, which becomes 07's "before" | code on the laptop; R17, R18 and the census on the Mac Studio | Opus | ✍️ 2026-10-28, on the user's decision the same day, after 06b and before 07's §2 | ⬜ | — | — |
| 07 | [The stationary-point guard](07-stationary-point-guard.md) | G2's Q4 (i) and (ii): `[06-levin-accepts-a-region-across-a-stationary-point]`, fixed generically in the Levin integrator's generic code (`_adaptive_levin`), with the basis supplying its phases and their closed-form rates through two new interface methods; `adaptive_levin_sincos` stays a facade. Stationary points and nearly coalescing minima (Airy parameter $\lambda < \Lambda_c$) located in advance; a window of phase half-width $\Phi_w$ round each, integrated in nested Clenshaw–Curtis pieces of at most $2\pi$; the rest to Levin as now; a nodal sign-change guard as the safety net; the round-off-limited exit barred there. $\Phi_w$ and $\Lambda_c$ measured by fixed rules at a planned stop (§2), shown to the user before the tests. R15/R16; 06a's census re-run, scored V1–V5 | code on the laptop; R15, R16 and the census on the Mac Studio | Opus | ✍️ 2026-10-27, (α)–(γ) and (ii) settled by the user in writing it; amended 2026-10-28 (its §9): 06b first, §2's reference and $\omega$ settled, §6 against 06b's census; then (§9 point 6) 06c first, rehearsals R19/R20, §6 against 06c's census | ⏸ Part A stopped before §2 on 2026-10-27; §1 set aside until 06b landed, rebased onto it, and §2 run on 2026-10-28: it stopped again, no rung of $\Phi_w$ qualifying (Decisions, **07 §2 stop**); paused until 06c lands | — | — |
| 08 | Fact-finding for layer 3 and `rtol` | G2's Q4 (iii)–(v) and Q5, measured with 07's guard in place: the depth cap, the dead `local_atol` branch, what `total_converged` means, and `rtol` | Mac Studio | Opus | ⬜ not written | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · 🟡 partly landed, with what is pending · ✅ landed · ⛔ stopped and superseded.

**This is a staged campaign, not a half-planned one.** README §2 fixes the charter of every held
prompt. Only its method waits on the measurement before it (README §6.2 C6).

**Orchestrator notes** are written just before dispatch, under `orchestrator/`. The planning
commit is `6af798f`. [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-25) dispatches 01
on the Mac Studio, orchestrated there, in parallel with `handover` 11 in the same checkout. It ran
the same day; its records are `docs/levin-phase-floor/01/` and VERIFICATION §1.
[`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) (2026-10-25) dispatches 03 on the Mac
Studio, orchestrated there, alone in its checkout. It ran the same day; its records are
`docs/levin-phase-floor/03/` and VERIFICATION §3. **For 04:** a harness on a real store needs 03's
`docs/levin-phase-floor/03/scripts/harness.py`, with its cloudpickle stand-ins
(`[03-storefixtures-standin-pickle-and-exit-rows]`) and its `PhaseGroup` reducer (03's I4, a harness
note, not opened as an issue). 04 takes 03's scripts and runs their code from a worktree at
`98e68d3`. The laptop makes the worktree on the Mac Studio before dispatch; the records stay in the
main checkout's `var/`. [`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) (2026-10-25)
dispatches 04 on the Mac Studio, orchestrated there, with the code in the worktree
`SecondaryGWKit-lpf04-98e68d3`; `handover` A3 may share the main checkout, read-only. It ran the same
day, in parallel with `handover` A3; its §6 is the resume. Its records are `docs/levin-phase-floor/04/` and
VERIFICATION §4. The worktree is kept until the user agrees to remove it.

## 2. The gates

| Gate | Holds when | After | State |
|---|---|---|---|
| **G1** | the user has read 01–04 and answered README §6.3 Q1–Q3: which hypothesis stands, which layer 1, and where it lands | 04 | ✅ **answered 2026-10-26** (Decisions above): H-anchor; layer 1(a) here; the producers after it |
| **G2** | the user has answered L2, Q4 and Q5 (§2.3): whether layer 2 is needed; layer 3, the acceptance test with the stationary-point guard, the depth cap and `total_converged`; and `rtol` | 06 | ✅ **answered 2026-10-27** (Decisions, **G2**): L2, after 06a's census, not needed at this code, to be revisited after 07 or later; Q4 (i)–(ii) is 07; Q4 (iii)–(v) and Q5 move to G3, after 08 |
| **G3** | the user has answered Q4 (iii)–(v) and Q5 on 08's measurements, with 07's guard in place | 08 | ⬜ |

### 2.1 The hypotheses

| | Statement | Scored by | State |
|---|---|---|---|
| **H-cancel** | findings §2.2 as written: the floor matters where a group's large $\tau_s$ terms should cancel and do not (`G+r` on `r == k`) | 01 P1–P2; 03 (04 does not re-score it: its §5) | **fails P1** ($S$ = 0.4385 ≥ 0.25) **and P2** (the non-cancelling class tracks $F$ too) **as the whole account** (01, 2026-10-25). Its premise is weaker than stated too: `G+r` on `r == k` is ~3.7× slower than `G-r`, not stationary (`[01-slow-group-leading-terms-do-not-cancel]`), confirmed by 03's M5. **03 (2026-10-25): its M1 condition is met** in all 18 cancelling groups, **but equally** in all 11 non-cancelling groups (median σ_θ/F 0.270 against 0.262), **so it does not single them out** |
| **H-anchor** | $G$'s remainder, reduced against $z_{\rm response}$, carries $\varepsilon k\eta_R$ into every group containing $G$ | 01 P2–P3; 03 (02 N3 not run: superseded); 04 A1; 05 L1 | **holds P1–P3** (01, 2026-10-25). P2: both classes have slope 0.97 / 0.82 and median $e_g/F$ 2.12 / 1.49, a ratio of 1.42. P3 discriminates weakly, because $F$ is per row (01's I5). **03 (2026-10-25): M1 and M2 hold in every scored group** (29 of 29), `G−r` (R4, R5: σ_θ/F 0.256, 0.225) and the lone `G` (R6: 0.337) included. The mechanism it names is §2.2. **04 (2026-10-25): A1 holds.** Taking $k\tau_R$ out of $G$'s phase, with $T$'s remainders left as production forms them, leaves no region at depth 18 or more in any of the 14 covered groups (every scored lone `G` and `G±r`). On all seven rows A matched X, the exact limit, region for region, `Gqr` groups included. So $G$'s anchor is the floor on these rows. **05 (2026-10-26), in production code:** on all 11 sampled depth-20 rows still capped under B1's code (5 low-band, 6 of 03's), the node-anchored $G$ remainder leaves no region at depth 18 or more, and regions fall to 0.16–4.7 % (34.5 % on R5's 55-region cap). L1 as written fails, only on rows B1 had already uncapped (VERIFICATION §5.2) |

They are not exclusive.

### 2.2 The mechanism

Prompt 03 §3: "a non-smooth phase floor, invisible to the driver, drives the bisection to the cap"
**holds** if M1, M2, M3 and M4 hold. Scored by 03, 2026-10-25, on seven depth-20 rows of A3 v2
(VERIFICATION §3.3):
- **M1 holds**: σ_θ/F 0.224–0.342 over 29 of 29 scored groups;
- **M2 holds**: on 7 of 7 rows; slopes 0.027–0.089; D/σ_θ 0.078–0.113;
- **M3 fails on its slope clause**: the control slope is −0.020 to 0.148, below 2 on 7 of 7 rows,
  while the ratio clause, 3.1e6–2.4e8, holds on 7 of 7;
- **M4 holds**: 1611 / 189,196 = 0.0085; the Clenshaw–Curtis share at depth 20 is 1.000;
- **M5 holds** on R1–R5 (worst +0.34 %); R7 has no `Gr` sub-interval, so it is inapplicable there.

**So "the mechanism holds" does not hold as written**, and the clause "M2 holds and M3 fails ⇒
refuted" fires by its letter. *(G1, 2026-10-26: the user ruled M3's slope clause badly posed. The
mechanism is recorded as **holding**, on M1, M2, M4, M3's ratio clause and 04's R1.)*

**I1's argument, the agent's** (03's `issues.md` I1 and log note (a)): with the jitter removed, D
is at round-off, about 1e-16 to 1e-13, from depth 10 onward, so no slope can be measured; the ratio
clause carries the comparison the prompt intends, and it points toward the jitter. The orchestrator
recorded this beside the score and did not adjudicate it. **G1 decides** how the verdict is read
(`[03-m3-slope-clause-is-badly-posed-at-round-off]`).

**04 (2026-10-25) made the full re-run that M3's control stood in for** (VERIFICATION §4.2–4.3).
Every factor's remainder was reduced exactly (arm X), and nothing else changed.
- **R1 holds on 7 of 7 rows.** No scored group keeps 10 regions at depth 18 or more, and the region
  count falls to 0.20–0.49 % of production's (66–167 against 22,147–52,909). Findings §2.6's
  refutation clause, *"item 3 leaves the depth at 20"*, does not fire.
- **What it settles.** The jitter is what sent these rows to the cap, by removal and not by
  inference.
- **What G1 still decides.** How M3's verdict is recorded; the re-run is the evidence for that
  ruling.

### 2.3 Gate G2: the questions, and the evidence after 06

*Written up 2026-10-27 by the laptop orchestrator, at the user's request, after 06 landed. The
questions are the user's to answer. The orchestrator's recommendation is at the end, apart from
the evidence.*

05, 05a and 06 landed layer 1 for $G$ and $T$ and the producers' stored phase. G1 deferred layer
2 to this gate, "decided after layer 1 and the wider re-measurement" (Decisions, **Layer 2**). That
re-measurement is 06's §6, on 05's 39 rows of A3 v2. They are 31 stored at depth 20, of which 11
were still capped under B1's code, and 8 controls.

**The evidence.**

| tree, `rtol` | rows | largest depth | rows with a region at depth ≥ 18 | regions | unconverged |
|---|---|---|---|---|---|
| $t_5$ (05a's code), `1e-7` | 39 | 11 | 0 | 901 | 21 |
| $s$ (06's code), `1e-7` | 39 | 11 | 0 | 742 | 20 |
| $s$ (06's code), `1e-8` | 31 | 11 | 0 | 1,053 | 19 |

- **Before layer 1** these rows included 11 at the cap under B1's code. 03's seven reached
  22,147–52,909 regions each. The cap is now reached by no row at either tolerance, and
  `[04-x-at-1e-8-still-reaches-depth-20]` is not seen at the current code (§3).
- **The kinds of non-convergence** (`convergence_kinds.py`, under $s$):

  | `rtol` | resolution | bound only | phase-limited |
  |---|---|---|---|
  | `1e-7` | 14 | 32 | 23 |
  | `1e-8` | 20 | 25 | 32 |

  These are against 05a's $t$: 18, 32, 17 and 21, 24, 27. Removing the stored noise moved
  non-convergence toward the round-off-limited exit.
- **S5.** $|{\rm total}_s(10^{-7}) - {\rm total}_s(10^{-8})| \le$ `total_abserr`$_s(10^{-7})$
  holds on 28 of 31 rows. The ratio's median is $1.4\times10^{-3}$, its 90th percentile 0.82, and
  its largest 1.57 (`mid-G-3`).
- **S4.** On 5 of 39 rows the stored noise moved the total beyond its declared error, by up to 10.8×
  (`low-G-3`). The median relative change is $1.3\times10^{-6}$.
- **P1** (`[06-levin-accepts-a-region-across-a-stationary-point]`). A region containing an interior
  stationary point is accepted with an error estimate about $10^4$ too small. At tighter `rtol` it
  is accepted through the round-off-limited exit instead, because the eq. (151) floor with a small
  but finite $G_0 = \min|\theta'|$ exceeds the residual. Whether P1 causes any of S4's or S5's
  failures is not separated.
- **Infinite declared errors.** 26 of R13's and R14's 864 rows store `total_abserr` = $+\infty$
  (`[04-roundoff-floor-can-be-infinite]`, `levin-refactor`).
- **The caveat.** The 39 rows were drawn by A3 v2's stored depth, which describes the code before
  B1 (`[05-b1-uncaps-most-cheap-depth-20-rows]`). No census of depth has been taken at the current
  code.

**L2: is layer 2 needed?** Layer 2 tells the driver its input's round-off floor, so that it stops
bisecting there. To act where the cap was reached it must also reach the Clenshaw–Curtis branch
(`[03-depth20-regions-are-all-clenshaw-curtis]`).
- (a) **Not needed.** No row reaches depth 18 at `1e-7` or `1e-8` under 06's code. Close
  `[04-x-at-1e-8-still-reaches-depth-20]` as not reproduced, and close or narrow
  `[03-depth20-regions-are-all-clenshaw-curtis]` with it.
- (b) **A census at the current code first.** Re-run A3 v2's rows in-process, Ray-free, as 05, 05a
  and 06 did, for the distribution of depth, then decide.
- (c) **Layer 2 anyway,** as a safety net for inputs this sample does not cover.

**Q4: layer 3, the acceptance test.**
- **(i) The stationary-point guard.** Decided in principle (Decisions, **06 P1**): a generic fix,
  the points located in advance from the closed-form $\theta'$, and a guard in the driver. Open
  within it:
  - (α) where the location lives: in `adaptive_levin_sincos`, from the `theta_deriv` it is handed,
    so that the integrator is correct for any caller; in `QuadSourceIntegral`'s partition, which
    knows the physics; or both;
  - (β) the rule for the window around each point: which resolved rule, and how wide, by phase
    span;
  - (γ) the threshold below which a near-zero minimum of $|\theta'|$ counts as a coalescing pair,
    the QCD and $e^+e^-$ caustics.
- **(ii) The round-off-limited exit.** P1 shows it accepting a region at any `rtol` where
  $|\theta'|$ is small. Should it be barred wherever the guard of (i) fires?
- **(iii) `DEFAULT_LEVIN_MAX_DEPTH`**, 20, never chosen. No sampled row now reaches it. Is it a cost
  guard only, and should reaching it mark the row unconverged?
- **(iv) The per-region acceptance test's dead length-share `local_atol` branch.**
- **(v) What `total_converged` means,** and whether a consumer gate reads
  `total_abserr/|total|` (findings §2.5). It bears on the 20 rows unconverged under $s$ at `1e-7`,
  on the 23 phase-limited non-convergences `convergence_kinds.py` counts there, and on infinite
  declared errors.

**Q5: is `rtol = 1e-7` right for accuracy?** (README §6.3.)
- Under 06's code, going from `1e-7` to `1e-8` adds 42 % to the regions on these rows. Neither
  tolerance reaches the cap.
- The accuracy given up at `1e-7` is S5's: a median of $1.4\times10^{-3}$ of the declared error, at
  most 1.57×.
- Cost is not an argument either way (README §5 rule 4).
- Options: keep `1e-7`; move to `1e-8`; or decide after the guard, since the guard changes the
  declared errors S5 compares against.

**The user's answer, 2026-10-27** (Decisions, **G2**): the recommendation below, as given. L2 (b)
first, with the decision on layer 2 after the census; 07 is the stationary-point guard; then a
fact-finding prompt for Q4 (iii)–(v) and Q5.

**The orchestrator's recommendation** (not a decision):
1. **L2:** (b), then (a). On present evidence layer 2 is not needed; a census at the current code
   would show it beyond 05's sample.
2. **07: the stationary-point guard, Q4 (i) with (ii).** It is the one known correctness defect, it
   has a known remedy and regression cases (06's P1 rows and reference), and it changes the declared
   error of every resonant row.
3. **Then Q4 (iii)–(v) and Q5, in one prompt, measured with the guard in place,** on 05's 39 rows,
   or on the census if L2 (b) is taken. Each of those questions turns on declared errors the guard
   changes.
4. **Close `[04-x-at-1e-8-still-reaches-depth-20]` at G2.**

#### 2.3.1 The census at the current code (06a, 2026-10-27)

*Added after 06a landed. The evidence above, on 05's 39 rows, stands as it was. This is G2's L2 (b),
over the whole store.* `VERIFICATION.md` §6a.

| `rtol` | rows | rows with a region at depth ≥ 18 | largest row depth | rows deeper than 11 | regions |
|---|---|---|---|---|---|
| `1e-7` | 9,280 | 0 | 12 | 19 | 108,879 |
| `1e-8` | 9,280 | 0 | 13 | 29 | 157,229 |

A3 v2 stored 145,313,734 regions. 2,489 rows have no Levin call.
- **The 2,265 rows stored at depth 20** fall to depths 1–12 at `1e-7`, 1,248 of them to depth 7.
- **Every row deeper than 11 is a stationary-point row.** Each holds an interior stationary point
  of θ′ in its `G+q+r` call, and its regions at depth ≥ 11 lie within 1–3.5 grid spacings of the
  point. They are mostly equilateral ($q = r = k$) at $k = 10^5$ and $3.14	imes10^5$/Mpc, mid band.
- **Depth ≥ 12 by branch** (Levin / Clenshaw–Curtis): 13 / 25 at `1e-7`; 17 / 42, plus 0 / 2 at
  depth 13, at `1e-8`.
- **Convergence, the "before" for 08:**
  - `total_converged` is False on 2,578 rows at `1e-7` and 2,638 at `1e-8` (2,262 stored).
  - The unconverged calls at `1e-7` / `1e-8` are, by kind: bound only 4,014 / 3,745, resolution
    1,737 / 2,182, phase-limited 1,213 / 1,720, capped 0 / 0.
  - 634 rows have an infinite `total_abserr`.
- **`1e-7` against `1e-8`:** the change exceeds `total_abserr`(`1e-7`) on 341 of 8,646 rows. The
  median ratio is 0, the 90th percentile 0.34, and the largest $3.9\times10^4$.

**For L2:** C1 and C2 hold, so no row of A3 v2 needs layer 2 at either tolerance at this code. What
depth remains is at stationary points, which are 07's.

**The user's decision on L2, 2026-10-27** (Decisions, **G2 L2, decided**): not needed at this
code; revisit after 07 or later, once layer 3 has fixed the error estimates and may bisect some
regions harder.

## 3. Active and unresolved issues

- **[01-slow-group-leading-terms-do-not-cancel]** *(opened 2026-10-25 by prompt 01, its I1)* —
  $G$'s and $T$'s leading phases do not run at the same rate.
  - $G$'s leading phase runs on conformal time `tau`, at rate $1/H$
    (`GkSourcePolicyData.py:142`, `primitive_phase.py:291-310`).
  - $T$'s runs on the sound horizon `cs_tau`, at rate $c_s/H$ (`TkSourceFunctions.py:184`,
    `:205-236`).
  - So on `r == k` the "slow" group `G+r` has leading $d\Psi/dz = -k(1-c_s)/H$, not zero, and
    `G-r` has $-k(1+c_s)/H$. In radiation that is about $0.42$ against $1.58\,k/H$: slower by about
    3.7, not stationary. The exactly cancelling geometry would be $r = k/c_s$.

  **Impact:**
  - findings §2.2's *"cancel identically"* and README §0.2's *"cancel"* overstate this;
  - H-cancel's mechanism becomes a partial cancellation;
  - §2.7 layer 1(b), composing $(s_Gk + s_rr)\tau_s$, cannot give a zero coefficient while the
    factors use different primitives. That bears on G1 Q2.

  **Next step:** check `theta_deriv` of `G+r` and `G-r` numerically on one candidate row in 03.
  Then a correction note on the findings document, which is left as written.

  **Narrowed (2026-10-25, prompt 03, M5):** the numerical check is done, and the issue stands as
  written. At five abscissae of the `Gr` sub-interval on R1–R5, $|\theta'_{G+r}|/|\theta'_{G-r}|$
  is 0.26795–0.26885, against $(1-c_s)/(1+c_s) = 0.267949$ at $c_s^2 = 0.333333$: worst deviation
  +0.34 %. So `G+r` oscillates at about 0.27 of `G-r`'s rate, and the cancellation is not exact. R6
  and R7 (`q == r == k`) have no `Gr` sub-interval. What remains is the correction note on the
  findings document. VERIFICATION §3.2.
- **[03-storefixtures-standin-pickle-and-exit-rows]** *(opened 2026-10-25 by prompt 03, its I3)* —
  `StoreFixtures` cannot carry A3 v2's objects as they stand.
  - `standin_pool`'s `StandinRef` and `_copy` serialise with standard `pickle`, which cannot carry
    a `GkSource`'s `NumericData`, a namedtuple bound as `_NumericData`
    (`ComputeTargets/GkSource.py:22`). So a stand-in pool on a real store fails at its first
    `GkSource` lookup. Ray uses `ray.cloudpickle`, which can.
  - `real_store_fixtures`' default `wavenumber_exit_time` rows carry `z_exit` only, and
    `TkNumericIntegration`'s constructor reads `z_exit_suph_e3`
    (`ComputeTargets/TkNumericIntegration.py:152`).

  **Impact:** none on 03, which worked round both inside its own harness
  (`docs/levin-phase-floor/03/scripts/harness.py`, and `build_real_store(replicated=…)`). The next
  harness on a real store (04) needs the same two workarounds.

  **Next step:** 04 reuses 03's harness. A change to `StoreFixtures/`, which is test-support code,
  is outside 01–04's remit.
- **[01-depth20-floor-eta-is-wrong-above-1e12]** *(opened 2026-10-25 by prompt 01, its I4)* —
  `docs/handover/a3-v2/findings/depth20_floor.py`'s $\eta(z)$ integrates from $z$ to $10^{15}$.
  - So it departs from the stored `eta_response` from $z \approx 1.2\times10^{13}$, and is
    negative above $10^{15}$.
  - **Impact:** none on findings §2.1 or on 01, which have no records there. It would give a wrong
    $F$ to anyone reusing it on seam rows.
  - **Next step:** a docstring note, or read the row's `eta_response` instead. It is a findings
    script, not this campaign's code.

- **[05a-t-phase-abscissa-rounding-remains]** *(opened 2026-10-26 by prompt 05a, its §6 agent's
  X2)* — $T$'s phase is evaluated at $z = {\rm expm1}(u)$ rounded to a double, which puts about
  $\varepsilon\tfrac{z}{1+z}|d\theta_T/du|$ of jitter into it. For $T$ that is the size of the floor
  05a removed wherever $z \gg 1$ (prompt 05a §0.1), and 05a's node-anchored remainder does not
  touch it.
  - **Measured** on 05's 39 rows, sampled in $u$ as `T_WKB` and the phase groups call it: under
    05a, σ_θ is 0.14–0.29 of the abscissa term in all 70 windows. In 14 windows at $z \gtrsim 10^2$
    it exceeds $0.1\,\varepsilon|\theta_T|$, at 0.10–0.22. Under the parent it is
    0.21–0.40 $\varepsilon|\theta_T|$. At $z < 1$, under 05a, it falls to 0.0025–0.0096.
  - **The node-anchored remainder's own residual** is smaller than this everywhere for $T$.
    Between table nodes it keeps $\varepsilon k|p_z|$, measured at up to
    $7.5\times10^{-3}\,\varepsilon|\theta_T|$ midway (05a's code log, observation 2). §6's
    windows sit at table nodes, so there it never binds (X5).

  **Impact:** an accuracy floor in $T$ of about $0.1$–$0.2\,\varepsilon|\theta_T|$ at $z \gg 1$.
  It has not been seen to drive the bisection: on 05's rows no Levin call moved under 05a. **Next
  step:** whether a phase evaluated in $u$ without an `expm1` round trip is wanted. That means a
  leading term in $u$, a producer-side change, so it falls to G1 Q3's 06+. `docs/levin-phase-floor/05a/wider/issues.md`
  X2; [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §5a.2.

- **[05-store-value-diff-compares-volatile-columns]** *(opened 2026-10-26 by prompt 05, its §5
  rehearsal agent's P1 and P2)* — `tools/store_value_diff.py` compares every row column except a
  row's serial, its timestamp and the columns that name another row. So two builds of the **same**
  code cannot be reported value-identical, and prompt 05 §5.4 and §8 were worded as if they could.
  On R9 against R10, eight classes outside `QuadSourceIntegral` were reported not identical, while
  every value table of every class was bit-identical. Every differing column falls into one of four
  kinds:
  - wall-clock timings (`*compute_time`, `*RHS_time`);
  - `label`, which is the job name plus `datetime.now()`;
  - `GkSource.parents_digest`, a SHA-256 of the parents' **serials**
    (`ObjectFactories/GkSource.py`, `GkSource_parents_digest`);
  - the GkWKB residual-memo counters `rho_evals`, `rho_reused` and `stage_1_RHS_evaluations`, in 25
    of 3477 rows. The swaps go both ways within each $k$, which fits a per-worker memo.

  **Impact:**
  - the tool's exit code cannot certify `CLAUDE.md` rule 7's "every value outside the classes the
    change is meant to touch must be bit-identical". 05 needed a classifier script and the user's
    ruling (*"an incorrectly-worded prompt"*);
  - no same-code pair exists to show that the memo counters are scheduling noise and not the
    change. That is inferred from the pattern of swaps (P2).

  **Next step:** decide whether the tool reports such columns apart from computed values (named in
  `config/` or by the factories, keeping the infrastructure generic), or each use classifies its
  differences as 05's `classify_non_qsi.py` did. Then correct rule 7's wording if needed.
  [`VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §5.1;
  `docs/levin-phase-floor/05/rehearsal/issues.md`.

- **[05-b1-moves-stored-a3v2-totals-beyond-their-error]** *(opened 2026-10-26 by prompt 05, its §6
  agent's W1)* — `handover` B1's code ($p$: B1, without 05's change) moves A3 v2's stored
  `QuadSourceIntegral` totals well beyond their stored `total_abserr`. Under $p$ the row's
  `QuadSource` is recomputed in-process, and is bit-identical to the stored row at every stored node
  plus B1's 3 (L0, in all 109 units).
  - **36 of the 39 sampled rows** lie outside the stored error, by 5.2× (`ctrl-mid-2`) to 1.5e5×
    (`mid-G-r-3`).
  - **The relative change** runs from 5.7e-6 to 0.75 (R4). On `low-G-r-2` the total **changes
    sign**: stored −3.05e-24, $p$ +3.73e-23.
  - **03's rows** move by 84× (R6) to 1.21e4× (R1) their stored error.
  - **The three rows within error** are `low-Gqr-3`, whose stored error is infinite, and the two
    `NULL` controls, which are bit-identical.

  **Impact:** A3 v2's stored totals are pre-B1 values, and no row with a Levin sub-interval agrees
  with this tree within its declared error. It extends R8's change of up to 0.80 relative on one
  triple (`docs/handover/b1/r8/`) to A3 v2. It bears on `handover`, which owns A3 v2: whether A3
  v2 is rebuilt at B1 or later before any physics is read from it.

  **Next step:** that decision, for the user. The census-based findings (01, `A3-V2-FINDINGS.md`)
  read cost, not values, and are affected through the next issue instead.
  [`VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §5.2;
  `docs/levin-phase-floor/05/wider/issues.md` W1.

  **Widened to the whole store (2026-10-27, prompt 06a's census, issues X3).** At the current code
  and `1e-7`, the total moves beyond the stored `total_abserr` on 6,187 of the 8,722 rows with a
  finite, non-zero stored error. It moves beyond 10× on 5,677 rows and beyond 100× on 4,818. The
  median is 286× and the largest $8.6\times10^5$× (`2-1524`). The relative change has median
  $1.2\times10^{-3}$ and maximum 42.6. 2,377 rows are bit-identical, mostly rows with no Levin call.
  05's $p$ units give the same size on their 38 rows, so the effect is B1's, not the census's. The
  next step is unchanged. `VERIFICATION.md` §6a;
  `docs/levin-phase-floor/06a/issues.md` X3.

- **[06-levin-accepts-a-region-across-a-stationary-point]** *(opened 2026-10-27 by prompt 06, its
  rehearsal agent's P1, investigated from the laptop)* — `adaptive_levin_sincos` accepts a Levin
  region that contains an interior stationary point of its phase, with an error estimate far too
  small.
  - **Where seen.** Two R14 rows at $z_{\rm response} = 250$, $(k,q,r) = (3.162\times10^5)^3$ and
    $(10^5, 3.162\times10^5, 3.162\times10^5)$. Group G+q+r on $z \in [250, 5.42\times10^9]$ is
    stationary at $z_* = 10211.3$ ($c_s = 1/2$) and $z_* = 275.0$ ($2qc_s = k$). Each is accepted
    as one region of 3 evaluations. The totals are wrong by $1.28\times10^4$ and 555 times their
    declared error, the group by $6.8\times10^{-3}$ and $1.2\times10^{-3}$.
  - **The mechanism.** The region's estimate, and those of its two halves, all lack the
    stationary-phase term, so the step-(4) residual is about $10^5$ smaller than the true error. The
    miss equals the leading stationary-phase term to $10^{-5}$. The relative test can also be passed
    through another part of the region. At rtol $10^{-9}$ and $10^{-11}$ the eq. (151) floor, with a
    small but finite $G_0 = \min|\theta'|$, accepts the region as round-off-limited instead.
  - **Why R13 was right.** Its noisier phase lifted the root's residual over rtol, by factors 2.05
    and 1.008, through the unrelated high-$z$ half, and bisection then found the point. 06's
    integrand is the more accurate of the two; it removed the noise that had rescued these rows.
  - **Not closed by C2.** The C2 total-variation gate diverts only regions whose phase varies by less
    than $6\pi$ (`levin_quadrature.py`, `_adaptive_levin_subregion_impl`). Its regression case puts
    the stationary point at $x = 1/2$, the first bisection point. The one-loop reconciliation's
    "already closed" (`docs/one-loop-planning/resonance-scaffolding-v1/sigw-resonance-reconciliation.md`
    §1.5, §3.1) is wrong, and carries a dated correction note. The QCD review's Levin benchmark
    (§5.2) is in exact radiation, where no isolated stationary point exists.

  **Impact:**
  - **Structural, not an edge case.** The stationary point is the Kohri–Terada resonance, where the
    signal is expected to come from, and nearly every triangle with $\Sigma = (q+r)/k > \sqrt3$
    has one wherever $c_s(z_*) = 1/\Sigma$. A stored `total_abserr` on such a group is not
    trustworthy, in stores built at either tree, A3 v2 included.
  - 06 makes the failure likelier, by removing the noise that sometimes forced the bisection.
  - Any error budget built from the stored errors, the one-loop integral's included, waits on the
    fix.
  - It may bear on 06's S4 and S5 failures; not separated.

  **Next step:** prompt 07, a generic fix to the integrator, in this campaign, before the next
  production build (the user, 2026-10-27; Decisions, **06 P1** and **G2 Q4 (i)–(ii)**).
  - **Locate the stationary points before integrating,** from each group's closed-form $\theta'$:
    the sign changes, and the near-zero minima where two points are about to coalesce (the QCD and
    $e^+e^-$ caustics). Split the range there, and give a small window around each to a rule that
    resolves the phase directly.
  - **Guard the driver,** as a safety net: no Levin acceptance, and no round-off-limited exit, for a
    region whose $\theta'$ changes sign or nearly vanishes on its nodes.
  - *(Moved out of 07 on 2026-10-27, by the user: how late the time integral must run, for the
    configurations that resonate during the QCD crossover and $e^+e^-$ annihilation, is physics,
    and is `[06-time-integrals-must-reach-late-resonances]`.)*
  - **Regression cases:** the two R14 rows with 06's reference, and a synthetic case with the
    stationary point off any bisection point.
  [`VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §6.3;
  [`docs/levin-phase-floor/06/p1/report.md`](../../docs/levin-phase-floor/06/p1/report.md).

  **Measured across A3 v2 (2026-10-27, prompt 06a's scan, issues X1–X2).** At the current code and
  `1e-7`, θ′ was sampled on 4,097 points per Levin call.
  - **Where:** 720 of 37,325 calls hold an interior sign change, in 720 of the 9,280 rows, all in
    group `G+q+r`. 412 are in the low band, 308 in mid and none in seam.
  - **What is accepted across them:** in every one a region is accepted across the point. 623 of
    those regions are at depth 0, the whole call accepted as one region, P1's pattern. 677 are
    Levin regions and 238 are phase-limited.
  - **Depth:** every A3 v2 row deeper than 11 is one of these rows, with its deep regions at the
    point.
  - **The limit:** two points closer together than the grid spacing (median $2.5\times10^{-3}$ in
    $\log(1+z')$) are not seen.
  - **For 07:** `docs/levin-phase-floor/06a/out/scan_listed_regions.csv` lists every such region.
    The next step is unchanged.

- **[07-roundoff-exit-assumes-bisection-cannot-help]** *(opened 2026-10-28 by the orchestrator,
  from 07's Part A stop 2)* — `_adaptive_levin` accepts a region as round-off-limited
  (`phase_limited`) when its eq. (151) floor is finite and above tolerance and the residual is down
  at it, on the premise that *"bisecting cannot help -- the children inherit essentially the same
  round-off floor"*. Nothing checks the premise, and it fails wherever $G_1/G_0 =
  \max|\theta'|/\min|\theta'|$ is large.
  - **Where seen.** 07's Part A, `test_3bessel_analytic.py`'s `test_JJJ` at seed 10 of 20, $(k, q,
    s) = (3.302, 0.885, 2.651)$, under 07's uncommitted §1. With a root windowed off, the Levin
    range $\log x \in [6.08, 27.63]$ has $|\theta'|$ from 102 to $\sim10^{11}$ ($G_1/G_0 \approx
    10^9$). It is accepted at depth 0 on a floor of $9\times10^{-7}$, with residual
    $6.6\times10^{-8}$; the call is honest but misses the test's $10^{-7}$ relative (2.6e-7).
    The parent bisected only because the root, in the same region, made the floor infinite.
  - **The mechanism.** The $G_1/G_0$ term comes from fitting one region over many decades of
    frequency. Halving the region cuts the ratio roughly to its square root, so the halves' floors
    are far below the parent's. The step-(4) residual has already computed both halves, each with
    its own `"phase_err"`, and the test reads only the parent's.
  - **How far.** In 06a's census, 1,213 of 37,325 Levin calls are phase-limited at `1e-7` (585
    rows), 1,720 at `1e-8` (745 rows); how many against the premise is not known. Every
    three-Bessel call is exposed, its $\theta' \approx Kx$ growing exponentially in $\log x$.

  **Impact:**
  - A phase-limited row's stored value may be looser than its tolerance, though its declared error
    covers it. Where the floor is declared as the error, the error is inflated too.
  - It confounds 07's §2: the Levin range beyond a window passes or fails as the edge value of
    $|\theta'|$ moves the floor across the tolerance, so $\Phi_w$ would be measured partly on this
    defect.
  - Not 07's (ii), which bars the exit in window pieces and guarded regions only.

  **Next step:** prompt 06b, before 07's §2 (the user, 2026-10-28; Decisions, **07 LG stop**): the
  exit checks its premise, firing only if the halves' floors sum to at least half the parent's, on
  both branches; R15/R16 and 06a's census re-run.

  **Narrowed 2026-10-28** by 06b, landed: the exit now checks its premise (`4ddc3c2`). §4 holds,
  and in §5 W0, W1 and W3 hold: no region on A3 v2's census is accepted against the premise, and
  nothing else moved. **W2 fails:** where the check now bisects, the declared error understates
  the true error on 17 of 24 sampled calls (`[06b-moved-calls-declare-less-than-their-error]`).
  **Kept open** on the user's word (Decisions, **06b close**): *"seems that we should keep it open
  until we've fixed W2 at least."*

- **[06b-resolved-region-keeps-a-floor-bisection-would-cut]** *(opened 2026-10-28 by prompt
  06b, its observation 1)* — 06b checks the premise only on the round-off-limited exit. A region
  accepted as **resolved** still carries its eq. (151) floor into `abserr` (`total_err` is
  `max(abserr, phase_err)`), even where $G_1/G_0$ puts that floor orders of magnitude above its
  halves'.
  - **Where seen.** $\theta = 1000e^x$ on $[0, 20]$, $f = (1, 0)$, `rtol` $10^{-7}$: accepted
    resolved at depth 0 (relerr $3.7\times10^{-8}$), declared `abserr` $9.7\times10^{-7}$ (the
    floor) against a true error of $2.1\times10^{-11}$, `converged` False, under the parent and
    06b alike. In 06b's seed-15 LiouvilleGreen call (below), $\log x \in [14.84, 27.63]$ is accepted
    at depth 1 by `atol` (residual $3.8\times10^{-15}$, relerr $2.4\times10^{-7}$) and its floor,
    $2.63\times10^{-10}$, is the whole call's `abserr`.
  - **The mechanism.** The same as `[07-roundoff-exit-assumes-bisection-cannot-help]`'s: the
    $G_1/G_0$ term of a region fitted over many decades of $\theta'$. Resolution there does not
    need bisection, so the inflated floor is kept.

  **Impact:** an honest but loose declared error, and `converged` False on a result that met its
  tolerance. In production both are stored (`total_abserr`, `total_converged`). How often is not
  measured; §5's census does not separate it.

  **Next step:** for the user. 06b keeps `resolved` unchanged (its §0.3). A remedy would apply the
  same premise to a resolved region whose floor exceeds its tolerance share, or attribute the
  floor differently.

  **Assigned (2026-10-28):** prompt 06c, global error control (Decisions, **07 §2 stop**). Under
  one budget for the call, a resolved leaf whose floor dominates the sum is the leaf chosen, and
  06b's premise fails where its halves cut the floor, so it is bisected. 06c's §2 test 3 is this
  entry's case; it closes when 06c's §4 and §5 hold.

- **[06b-premise-check-can-chase-a-flat-residual-to-depth-max]** *(opened 2026-10-28 by prompt
  06b, its §2 test 5)* — under 06b one LiouvilleGreen call of the 1,026 that moved runs to
  `depth_max`, chasing a step-(4) residual that does not fall with $h$.
  - **Where seen.** `test_3bessel_analytic.py`'s `test_JJJ` at seed 15, its second draw (J110,
    $(k, q, s) = (0.8743, 4.9331, 0.1827)$), phase group 3, `rtol` $10^{-10}$, order 12
    (`docs/levin-phase-floor/06b/calibration/out/lg_seed15_trail.txt`). The parent accepts
    $\log x \in [8.44, 14.84]$ at depth 2 through the exit (residual $3.5\times10^{-15}$, floor
    $2.2\times10^{-13}$) against its premise (halves/own 0.042). Under 06b the call goes from 6
    regions, depth 5 and 23 solves to 602 regions, depth 20 and 2,295 solves, with 56
    Clenshaw–Curtis regions at depth 20 in $[10.6797, 10.6904]$. `abserr` moves
    $2.636\times10^{-10} \to 2.634\times10^{-10}$, the value by $5\times10^{-11}$ relative,
    `converged` False under both; the test passes under both.
  - **The mechanism, as far as measured.** Along the ancestry of $\log x = 10.6797$ the residual
    stays at $1.2$–$2.2\times10^{-15}$ from depth 2 to 17 while the eq. (151) floor falls from
    $2.2\times10^{-13}$ to $7.8\times10^{-20}$. The premise passes $\tfrac12$ only at depth 4,
    where the floor is already below the residual, so the exit cannot fire again. Elsewhere the
    same flat residual collapses to $\sim10^{-21}$ once a region stops containing some point
    (depth 13 and 16 on two other ancestries). So something localised, which eq. (151) does not
    model, sets the residual. What it is is not diagnosed. The parent's exit had been accepting
    that real floor because eq. (151) at depth 2 happened to exceed it.

  **Impact:** cost (×100 solves on that call) and the cap, with no change to the value or the
  declared error. It bears on 06b §5's W4 and G2's L2. One call in 4,320 on the laptop's draws;
  A3 v2's rate is §5's to measure.

  **Next step:** score W4 in 06b §5. If calls reach the cap there, diagnose what sets the flat
  residual on one of them before any remedy.

  **Narrowed 2026-10-28** by 06b's census (W4 fails): on A3 v2, three analytic-block calls at
  `1e-8` failed the premise on 4–7 regions and then bisected to depth 20: `2-1704` Y1 (1,136
  regions, 640 at depth 20), `2-13964` J1 (31,885 regions, 21,840 at depth 20, 127,539 solves, 159 s)
  and `3-34987` J2 (14,293 regions, 6,960 at depth 20). None at `1e-7` (deepest 18); no unit over
  budget. **All three calls end `converged`**, their summed error 0.03, 0.53 and 0.28 of the request
  $\max(\texttt{atol}, \texttt{rtol}|v|)$: the depth was spent meeting per-region tests, each region
  at `rtol` against its own value, not the call's. Whether the residual is flat in $h$ on these
  calls has not been measured. Prompt 06c's single budget for the call bears on it directly
  (`[07-per-region-acceptance-misses-a-cancelling-total]`).

- **[06b-moved-calls-declare-less-than-their-error]** *(opened 2026-10-28 by prompt 06b's census,
  its C1, on the user's word)* — where 06b's premise check now bisects a region the old exit
  accepted, the call's declared error understates its true error.
  - **Where seen.** 06b §5's W2 (`docs/levin-phase-floor/06b/census/out/cmd/35`, `37`–`39`): of 24
    moved group calls at `1e-7` with no sign change of $\theta'$, 17 have $|{\rm value}_b -
    S_{10}| > $ `abserr`$_b$, median ratio 1.79, maximum 19.7 (`b-1e-7-3-34455` call 1, `G+q+r`,
    mid band, $k = 10^5$). The reference converges ($|S_9 - S_{10}| \le 0.0066$ `abserr`$_b$ on
    all 24; $S_{11}$ agrees to 0.002 on three; the driver at `rtol` $10^{-11}$ agrees on one).
  - **The values improved; the declared error fell further.** Against each call's own request,
    $b$ misses on 4 of 24 (at most 3.42×) and 06a on 23 of 24. $b$'s `abserr` is 1/10 to
    1/16,500 of 06a's. 06a's covered its true error on 23 of 24, through the over-wide floor that
    06b removed. So the calls now report `converged` while 4 of 24 sampled are outside tolerance.
  - **The mechanism, as far as measured.** On two calls (#21, #4) the driver at `rtol` $10^{-11}$
    returns $b$'s value exactly, with the same 4 and 2 regions, each resolved: residuals
    $3\times10^{-39}$–$1.2\times10^{-34}$ and floors $7\times10^{-39}$–$1.6\times10^{-34}$, against
    errors of $4.4\times10^{-33}$ and $9.5\times10^{-34}$. The error is common to a region and its
    halves, so the step-(4) residual cannot see it, and eq. (151) does not bound it. What it is
    has not been diagnosed.

  **Impact:** stored values and declared errors that are dishonest by up to ~20× on calls the
  premise check bisects, flagged `converged`. It keeps
  `[07-roundoff-exit-assumes-bisection-cannot-help]` open (Decisions, **06b close**). Prompt 06c's
  single budget uses the same per-region error estimates, so it inherits this: a budget met on
  estimates that miss a common error is not met.

  **Next step:** for the user: diagnose what the common error is on one of the 17 calls, before
  any remedy, and decide its order relative to 06c.

- **[07-per-region-acceptance-misses-a-cancelling-total]** *(opened 2026-10-28 by the orchestrator,
  from 07's §2 stop)* — `_adaptive_levin` accepts each region against its own value (`relerr <
  rtol`, or `abserr` below its length share of `atol`) and compares the summed error with
  `max(atol, rtol |value|)` only afterwards, to report `converged`. When the regions' values cancel,
  $\sum|v_i| \gg |\sum v_i|$, the summed allowance `rtol`$\sum|v_i|$ exceeds the request
  `rtol`$|\sum v_i|$.
  - **Where seen.** 07's §2 (rtol $10^{-10}$, 07's uncommitted §1 rebased on 06b): family (R) at
    $x_s = 0.7236$, $\omega = 0$, where the stationary-phase term $\propto f_{\cos}(x_s) +
    f_{\sin}(x_s) = -0.566 + 0.485$ nearly cancels and $|I|$ is 14–17× smaller than at
    $x_s = 0.382$. 1–3 cases per rung of $\Phi_w$ unconverged, declared/requested 1.0–6.7, every
    case honest and its true error within the request (at $c = 10^4$, $\Phi_w = 2\pi$:
    $1.4\times10^{-14}$ against $1.16\times10^{-13}$). The window's $2\pi$ pieces do not cancel
    individually: over a finite piece $\int\cos\theta$ and $\int\sin\theta$ are different Fresnel
    increments.
  - **The parent too.** At 06b's code the same cases converge only narrowly: declared/requested
    0.88, 0.30, 0.74 at $x_s = 0.7236$ against 0.27, 0.02, 0.05 at $x_s = 0.382$ ($c = 10^4, 10^6,
    10^8$). A measurement made writing 05a found 18 of 67 unconverged production Levin calls at
    `1e-7` failing on resolution because their regions cancel (`handover`'s
    `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]`).
  - **The mechanism, structurally.** Besides physical cancellation, Levin regions telescope: each
    value is a difference of boundary terms, adjacent regions share an endpoint with opposite signs,
    and $n$ regions sum to about their outer ends while $\sum|v_i|$ grows like $n$.

  **Impact:** `converged` False on results that met their request, and, in principle, `converged`
  True on results that did not. It blocks 07's §2, whose rule cannot pick a rung.

  **Next step:** prompt 06c, after 06b and before 07's §2 (the user, 2026-10-28; Decisions, **07 §2
  stop**): one budget for the call, the refinable leaf with the largest error bisected until
  $\sum e_i \le \max(\texttt{atol}, \texttt{rtol}|\sum v_i|)$.

- **[07-quadsource-applies-rtol-per-call-against-a-cancelling-total]** *(opened 2026-10-28 by the
  orchestrator, writing 06c)* — `QuadSourceIntegral` splits one integral into sub-intervals and
  phase groups and makes one Levin call per group, each with a length share of `atol` and the full
  `rtol`, and sums them (`compute_QuadSource_integral`, `phase_group_Levin_integral`).
  `total_converged` is the AND of the calls' own flags. Each call meets `rtol` against its own
  value, so where the groups or sub-intervals cancel, the row's total does not meet `rtol`
  against itself. The code's own comment records `numeric_quad` and `WKB_Levin` cancelling by up
  to 5×.
  - **Why it is separate from 06c.** 06c makes one Levin call meet its request against its own
    total. The share across calls is the caller's, and the same defect one level up.

  **Impact:** a row's `total_converged` and `total_abserr/|total|` do not say what the row met. It
  is G3's Q4 (v), what `total_converged` means.

  **Next step:** for the user, at G3 (Q4 (v)), on 08's measurements.

- **[06-time-integrals-must-reach-late-resonances]** *(opened 2026-10-27 by the orchestrator,
  from the user's decision on 06's P1, Decisions **06 P1** and **07 scope**)* — the time integrals
  must run late enough to include the QCD crossover and $e^+e^-$ annihilation, for the
  configurations that resonate during them.
  - **Why it matters.** A triangle resonates where $c_s(z_*) = 1/\Sigma$. During the QCD crossover
    and $e^+e^-$ annihilation $c_s$ dips below $1/\sqrt3$, so some configurations resonate then,
    and the stationary points can coalesce in caustics there.
  - **Why not in 07.** It changes the physics range of the integral, not how the integrator
    handles it. 07's advance location of the stationary points makes it possible to see which
    configurations need it.

  **Impact:** an integral that stops before its configuration's resonance omits the term the
  signal is expected to come from. Not measured.

  **Next step:** a separate physics prompt, after 07 (the user, 2026-10-27). Its scope is set
  when it is written: which integrals, in which model, and how late is late enough.

- **[06-s2-window-metric-floors-on-phi-t-structure]** *(opened 2026-10-27 by prompt 06, its §6
  agent's W4)* — 04's φ metric, the RMS residual of a degree-4 fit over 9 consecutive samples,
  measures $\varphi$'s own shape wherever that exceeds the storage noise. So a rule bounding the
  new $\sigma$ by $10^{-3}\sigma_{t_5}$, which presumes $\sigma_{t_5}$ is noise, fails there.
  - S2 failed in 2,115 of 12,862 $T$ windows, all at $z \ge 1.75\times10^7$, where $|\theta_T|$ is
    0.05–1,500 rad. There $\sigma_s/\sigma_{t_5}$ = 0.952–1.04, identical in both trees, and
    $\sigma$ is 1.7e-12–1.7e-10 rad, 5 to $1.5\times10^7$ times $\varepsilon|\theta|$.
  - In the other 10,747 windows $\sigma_s/\sigma_{t_5}$ has median $3.8\times10^{-5}$.

  **Impact:** none on 06's code: the user closed `[02-consumer-phi-below-the-storage-granularity]`
  on S1 and S3 with S2 read this way. A later prompt that scores $\varphi$'s noise by this metric
  meets the same floor.

  **Next step:** a later noise metric differences against a reference $\varphi$, or bounds by the fit
  residual of an exact $\varphi$ on the same samples, rather than by $\sigma_{t_5}$.
  `docs/levin-phase-floor/06/wider/issues.md` W4.

- **[06-ray-task-output-does-not-reach-the-launch-log]** *(opened 2026-10-27 by prompt 06, its
  rehearsal agent's P2)* — `_build_phase`'s disagreement warning and `_check_phase_sign` run inside
  Ray tasks, and in both rehearsal builds the launch log and `stdout.log` hold only the `Datastore`
  actor's output. No task printed anything in either build, so whether a task's `print` would have
  reached the launch log was not tested. The rehearsal agent searched Ray's worker logs under
  `/tmp/ray-sgk/session_*/logs/` instead (every count 0), which are not durable records.

  **Impact:** a check that greps a build's log for a warning, as 06's §8 stop did, may pass because
  the warning never reaches the log. The stationary-point fix above, and any production build that
  relies on a warning, meet this.

  **Next step:** show once, on a rehearsal build, whether a task's `print` reaches the driver's
  log; if not, record warnings in a durable place (a stored metadata field or a run-directory
  file). `docs/levin-phase-floor/06/rehearsal/issues.md` P2.

**04's other findings, not opened as issues** (`docs/levin-phase-floor/04/issues.md`):
- **I1**, the incomplete gate bounds, was resolved by prompt §8 on the user's decision.
- **I4**: `runregistrykit.begin` takes `git_head` from the script's directory. That is the package's
  documented behaviour, and the user chose the provenance it gives.
- **I5**, a `black` cache written outside the allowed area, is recorded as drift.
- **I7**, arm A matching X region for region, is recorded in §2.1.
- **I8** narrows `handover`'s `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]`.
- **I9**: mpmath's `from_float` swallows exceptions. That is a third-party behaviour, recorded in the
  log for the next harness that times out by signal.

**05's other findings, not opened as issues** (`docs/levin-phase-floor/05/{rehearsal,wider}/issues.md`):
- **W3 and W4,** that L1's tenfold clause presumes a row capped under $p$ and that R5's cap is a
  55-region chain, are about the rule. They are recorded with the verdicts (the log, and §2.1 H-anchor).
- **W5** narrows `handover`'s `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`.
- **W6 and W7, and the one R9/R10 row that moved by 28× its error,** narrow `handover`'s
  `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]`. W7's cause was measured on
  2026-10-26, writing 05a: [`measure/convergence_kinds.py`](measure/convergence_kinds.py), output
  [`measure/out/convergence_kinds.txt`](measure/out/convergence_kinds.txt), recorded in that entry.
- **W8,** $c$'s regions on R1–R7 against 04's arm A, is descriptive. The trees differ in three
  ways that it does not separate.
- **W9:** 15 scipy `IntegrationWarning`s from production's numeric quadrature, 5 under $p$ and 10
  under $c$. No unit failed. Recorded in the log.

**06's other findings, not opened as issues** (`docs/levin-phase-floor/06/{rehearsal,wider}/issues.md`):
- **W1:** no tree after `dc9d28c` can open a store written before it, since the factories declare
  `theta_residual`. That is the standing rule (an old store is read at the SHA that wrote it); 06's
  §6 met it and the user chose route (B).
- **W2:** A3 v2's diagnostic `analytic_G_w`, `analytic_Gprime_w` and some `analytic_G_rad` values
  predate `1c19998`. No consumer reads them.
- **W3:** the prompt's "`z_init` null" premise was the laptop's error; the user ruled reading (I).
  Its open point is settled: `CumulativeTable.delta(z, z)` is exactly 0.0, so a pure-WKB value's
  second term in `_phi_samples` is exactly zero.
- **W5 and W6:** S4 fails on 5 of 39 rows (up to 10.8× `total_abserr`$_{t_5}$, `low-G-3`; `ctrl-low-1`
  moves by 2.5 % of its total) and S5 on 3 of 31 (`mid-G-3`, `mid-G-r-3` as in 05a's T4, and
  `mid-G-1`). Findings about the declared error, for G2's Q4 and Q5; whether
  `[06-levin-accepts-a-region-across-a-stationary-point]` causes any of them is not separated.
- **W7:** more phase-limited non-convergence under $s$ (`convergence_kinds.py`: at 1e-7 resolution
  14, bound only 32, phase-limited 23, against 05a's 18, 32, 17). For G2's Q4.
- **W8:** numerically seeded $G$ windows keep $\sigma_s$ of $2.1\times10^{-4}F$ at the median, about
  8× the window RMS of $\Delta\theta$ alone, against $0.27F$ before. Descriptive; S1 does not score
  them (prompt §9 point 4).
- **W9, W10, W11:** the re-run values against A3 v2 (17 single-cycle changes, all within
  $4\varepsilon|\theta|$ of a boundary), `Levin_z` moving on one row, and 15 scipy
  `IntegrationWarning`s. Recorded in the log.
- **W12:** `python -m runregistrykit store fingerprint` takes its fingerprinter from the main
  checkout's `runregistrykit.toml`, so it cannot fingerprint a store of a later schema from a
  worktree. Possibly RunRegistryKit's intended behaviour; recorded for its maintainers, not worked
  round.
- **The rehearsal agent's recurrences** of `[05-store-value-diff-compares-volatile-columns]`, of
  05's memo-counter swaps (25 rows), of `[04-roundoff-floor-can-be-infinite]` (26 rows with an
  infinite `total_abserr` in both stores) and of the readers' unseeded work lists, as in 05a.

**Assigned here on 2026-10-25, held on their own boards** (README §1, §6.2 C5):
- `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` — `handover` board §3;
- `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` — `handover` board §3;
- `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` — `handover` board §3;
- `[00-consumer-anchoring-floor]` — `GkTk-remedial` board; **closed there for $G$ on
  2026-10-26** by 05, on the user's ruling (its §4);
- `[02-consumer-phi-below-the-storage-granularity]` — `phase-representation` board, assigned
  2026-10-26 at G1 (Q3); **closed there on 2026-10-27** by 06, on the user's word (its §4).

**Measured here, not assigned:** `[04-roundoff-floor-can-be-infinite]` (`levin-refactor` board).
01 confirms whether it reaches A3 v2, and narrows it there.

## 4. Resolved issues

- **[04-x-at-1e-8-still-reaches-depth-20]** *(opened 2026-10-25 by prompt 04, its I6)* — with every
  remainder exact, arm X at `rtol = 1e-8` still sends one short chain of regions to
  `DEFAULT_LEVIN_MAX_DEPTH`.
  - It does so on R1 and R4–R7, always with exactly 2 regions at depth 20 and 4 at depth 18 or more
    in each affected group: up to five groups, and 4–10 depth-20 regions, per row.
  - At `1e-7` no group reaches depth 20.
  - The pattern is a bisection to the cap at a single point. The candidates are a kink in the
    integrand (a clamp edge, at this pre-B1 tree, or a sub-interval boundary) or the remaining
    abscissa-rounding jitter, at about $\varepsilon|d\theta/du|$. **Not investigated.**

  **Impact:** it costs nothing measurable here (1–3 s a row), and R2 holds with every ratio at most
  0.47. It bears on G2's Q4 (the depth cap) and on layer 2. **Next step:** locate the point on one
  row before layer 2 is written. Measurement: `issues.md` I6; [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §4.

  **Narrowed (2026-10-26, prompt 05 §6): not seen at the current code.**
  - **The measurement.** With B1's code and 05's change ($c$) at `1e-8`, the maximum depth is 2–11
    on every one of the 31 sampled depth-20 rows of A3 v2. On R1 and R4–R7 it is 9, 9, 5, 8 and 9,
    so no chain reaches the cap.
  - **The candidate.** This fits the pre-B1 clamp edge, which B1 removed. It is not separated from
    05's own change, since $p$ was not run at `1e-8`.
  - **What it means.** It is likely closable on a re-check at `HEAD`. It bears on G2's Q4.

  [`VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §5.2;
  `docs/levin-phase-floor/05/wider/out/wider_05.txt`.

  **Narrowed again (2026-10-27, prompt 06 §6): not seen at `HEAD`.** With 06's code ($s$) at `1e-8`,
  on the same 31 rows, the largest depth is 11, and no row has a region at depth 18 or more. So the
  re-check at the current code that the last narrowing asked for is made, by a tree that contains
  both B1 and 05. **Next step:** close it with the user's decision on layer 2, after 06a's census
  (G2, §2.3).
  `docs/levin-phase-floor/06/wider/state/analysis.json`.

  **Closed 2026-10-27 at G2, on the user's layer-2 decision, by prompt 06a's census:** not
  reproduced. On all 9,280 rows of A3 v2 at the current code, no row has a region at depth 18 or
  more at `1e-8`; the deepest is 13, at a stationary point. `VERIFICATION.md` §6a;
  `prompts/levin-phase-floor/logs/06a-depth-census.md`.

- **[03-depth20-regions-are-all-clenshaw-curtis]** *(opened 2026-10-25 by prompt 03, its I2)* —
  every depth-20 region on the seven rows is a Clenshaw–Curtis fallback region.
  - 189,196 of 189,196 in the 29 scored groups; no group, scored or not, has a Levin region at
    depth 20.
  - So the bisection to `DEFAULT_LEVIN_MAX_DEPTH` is entirely in `_adaptive_levin`'s `is_direct`
    branch (`levin_quadrature.py:1975-2064`). There `abserr` is the nested-pair estimate, at
    D/σ_θ ≈ 0.1, and `phase_err` (eq. 151) reaches it in 0.85 % of the regions (M4).

  **Impact:**
  - findings §2.7 layer 2 must act on the Clenshaw–Curtis branch's acceptance test; acting on the
    Levin branch alone would not reach these regions;
  - that branch is the subject of `[10-levin-wholesale-cc-fallback]` (`source-remediation`;
    OPEN_ISSUES §1.3), which README §0.1 puts out of scope;
  - 01's "most" depth-20 regions in the expensive groups being `simple_regions` is, on these rows,
    all of them.

  **Next step:** a scope question for G1.

  **Narrowed (2026-10-25, prompt 04):** with the floor removed (arm X or arm A), no group of the
  seven rows reaches depth 20 at `rtol = 1e-7`. So the Clenshaw–Curtis regions at the cap there
  were the floor's. Layer 2's scope question stands for what is left:
  - at `1e-8` a single short chain per group still reaches the cap
    (`[04-x-at-1e-8-still-reaches-depth-20]`);
  - A3 v2's other depth-20 rows are unmeasured.

  **G1 (2026-10-26): deferred to G2** by the user. Layer 1 lands first, then a wider re-measurement,
  and G2 decides whether layer 2 is needed and whether it must reach this branch.

  **Closed 2026-10-27 at G2, on the user's layer-2 decision, by prompt 06a's census:** no region of
  A3 v2 reaches depth 20 at the current code, at `1e-7` or `1e-8`. At depth ≥ 12 the split is 13
  Levin / 25 Clenshaw–Curtis at `1e-7` and 17 / 44 at `1e-8`, all at stationary points. The scope
  question, whether layer 2 must reach the Clenshaw–Curtis branch, reopens only if layer 2 does,
  after 07 or later. `VERIFICATION.md` §6a.

- **[05-b1-uncaps-most-cheap-depth-20-rows]** *(opened 2026-10-26 by prompt 05, its §6 agent's
  W2)* — under B1's code, 20 of the 31 sampled stored depth-20 rows no longer reach the cap.
  - **Which rows:** all 12 drawn mid-band rows, all 6 drawn `Gqr` rows, four low-band rows and 03's
    R2.
  - **How far they fall:** to maximum depth 2–10 and 9–72 regions, against 39–29,782 stored. R2
    falls from 29,782 regions to 15.
  - **What stays capped:** 11 rows, 5 in the low band and 6 of 03's seven.
  - **The likely mechanism,** not investigated, is the pre-B1 clamp at the hand-over: a kink in
    the integrand, which 04's I6 named as a candidate for a short chain to the cap.

  **Impact:** A3 v2's stored `WKB_Levin_max_depth`, and 01's census built from it, describe the
  pre-B1 code. A sample drawn from them by stored depth is mostly not capped at `HEAD`. That is why
  05's L1, L1q and L3 fail as written. It also changes which rows layer 2 and G2's Q4 (the depth
  cap) are about.

  **Next step:** any later census or sample is taken at the current code, not read from A3 v2's
  stored depth. G2 reads 05's capped-under-$p$ table, not the stored one.
  [`VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §5.2;
  `docs/levin-phase-floor/05/wider/issues.md` W2.

  **Closed 2026-10-27 at G2, on the user's layer-2 decision:** its next step, a census at the
  current code, is prompt 06a's. All 2,265 rows stored at depth 20 fall to depths 1–12 at `1e-7`,
  1,248 of them to depth 7, and no row of the store reaches depth 18 at either tolerance. Any later
  sample is to be drawn from 06a's census (`docs/levin-phase-floor/06a/out/census_rows.csv.gz`),
  not from A3 v2's stored depth. `VERIFICATION.md` §6a.

- **[04-g-phase-is-reduced-against-the-double-two-pi]** *(opened 2026-10-25 by prompt 04, its I3)* —
  `PrimitivePhase.theta_mod_2pi` reduces with `WKB_mod_2pi`, an exact `fmod` against the double
  `TWO_PI`, and the phase groups take `sin`/`cos` of the remainder.
  `LiouvilleGreen/range_reduce_mod_2pi.py`'s docstring warns against exactly this. The remainder is
  that of $\theta + n\delta$, with $\delta = 2\pi - \texttt{TWO\_PI} \approx 2.45\times10^{-16}$: a smooth
  phase error of about $0.176\,\varepsilon|\theta|$.
  - On 03's rows it is **0.175 F for $G$ on every row** (2.96e-7 rad on R1, up to 1.61e-5 on R3), and
    1.9e-4 to 0.051 F for $T$.
  - Production, A and X all carry it. 04 kept it deliberately, so that P − X isolates the jitter
    (README §6.2 C7).

  **Impact:** an accuracy floor at about a fifth of $F$, smooth and so invisible to the cost. A
  layer 1 that only reduces "exactly" against `TWO_PI` keeps it. **Next step:** at G1, whether a
  layer 1 reduces against $2\pi$ itself (in extended precision for a constant such as $c_G$), not
  against `TWO_PI`. Measurement: `docs/levin-phase-floor/04/issues.md` I3; [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §4.

  **Narrowed (2026-10-26, prompt 05): done for $G$; $T$ remains.**
  - **$G$'s remainder** is now reduced against $2\pi$ itself (`120fdfe`). Its per-node constant
    is reduced at 256 bits. The small variable part is reduced with `WKB_mod_2pi`; its modulus error
    of $0.18\,\varepsilon|v|$ is negligible.
  - **The test.** 05's test 2 holds the remainder to $4\varepsilon(k|p_z|+2\pi)$ of the exact
    remainder against $2\pi$. Its breakage (i), which reduces against `TWO_PI`, fails it by
    1.6e-4 rad at $k = 3\times10^8$.
  - **$T$'s remainders** still reduce against `TWO_PI`, by G1 Q2: 1.9e-4 to 0.051 $F$ on 04's rows.

  `prompts/levin-phase-floor/logs/05-reanchor-g-phase.md`.

  **Assigned (2026-10-26):** $T$'s half to prompt 05a, on the user's decision (Decisions, **05a**).
  It closes when 05a's §5 and §6 hold.

  **Closed 2026-10-26 by prompt 05a, on the user's word:** $T$'s remainder is now reduced
  against $2\pi$ too (`1fc349c`). Its test 2 holds it to 0.19–0.29 of
  $4\varepsilon(k|p_z|+2\pi)$ of the exact remainder at $k$ up to $3\times10^8$, `z_init` on and off
  a node. R11 against R12 moves 424 of 864 totals by at most $8.8\times10^{-5}$ of their error;
  05's 39 rows move by at most $4.7\times10^{-6}$ (T3). `prompts/levin-phase-floor/logs/05a-reanchor-t-phase.md`;
  `docs/levin-phase-floor/VERIFICATION.md` §5a.

- **[05-t-remainder-keeps-the-anchoring-floor]** *(opened 2026-10-26 by the orchestrator, writing
  05a, from 05's closing note on `[00-consumer-anchoring-floor]`)* — $T$'s remainder is still formed
  by reducing one double of size $k\tau_s(z)$, against its global anchor `z_init`.
  - **The error.** Every remainder carries rounding of about $\varepsilon k\tau_s(z)$, which is not
    smooth in $z$. It is small at early source times and largest at late ones.
  - **What is known of its size.** With $G$ already clean, 04's I7 put the predicted floor,
    $\varepsilon\sum|\theta_T|$ at the late end of the `Gqr` sub-intervals, at 1.1–1.3 $F$, from
    the two $T$ factors. No $\sigma_\theta$ of $T$ has been measured there.
  - **What has not been seen.** It has not been seen to drive the bisection: 04's arm A, which left
    $T$ as production forms it, matched the exact limit, and 05 removed the cap on all 11 capped
    rows without $T$.
  - **Why it was lost.** `[00-consumer-anchoring-floor]` (`GkTk-remedial`) was closed for $G$ by
    05, and $T$ was recorded only in its closing note. So nothing in the index carried it.

  **Impact:** an accuracy error in a factor on the same footing as $G$; its size in totals is
  unmeasured. **Next step:** prompt 05a, which measures it in place (T1) and in totals (T3).
  **Assigned (2026-10-26):** to 05a, on the user's decision (Decisions, **05a**).

  **Closed 2026-10-26 by prompt 05a, on the user's word:** $T$'s remainder is formed node by
  node (`1fc349c`), and its anchor's non-zero end partial is tested (breakage (iv)).
  - **Measured in place** (T1, by the true step, the reading of record): σ_θ sampled in $z$ is
    0.20–0.37 $\varepsilon|\theta_T|$ under the parent and at most $1.3\times10^{-5}$ under
    05a, in all 70 probe windows on 05's 39 rows.
  - **In totals** (T3): at most $4.7\times10^{-6}$ of the declared error. No Levin call's
    regions, depth or flag changed, so on these rows $T$'s floor was never what the integrator
    saw.
  - **T4 fails**, on 05's two L3 rows, unchanged by $T$. The user closed the issue regardless.
  - **What production's $T$ phase still carries,** sampled in $u$, is the abscissa's rounding:
    `[05a-t-phase-abscissa-rounding-remains]` (§3).

- **[03-m3-slope-clause-is-badly-posed-at-round-off]** *(opened 2026-10-25 by prompt 03, its I1)* —
  prompt 03's M3 slope clause cannot be met by the jitter-free control.
  - With `theta_mod_2pi` replaced on the depth-10 span by a degree-12 fit, and nothing else
    changed, the median D is about 2e-16 to 1e-13 at **every** depth from 10 to 20: the round-off
    of the 24/47-point nested Clenshaw–Curtis pair. The real phase gives 4e-8 to 5e-6.
  - So the control's slope over depths 12–20 is −0.020 to 0.148 per row, below 2 on 7 of 7, while
    the depth-20 ratio real/control is 3.1e6–2.4e8, at least 100 on 7 of 7.

  **Impact:**
  - as written, M3 fails, the mechanism verdict is not met, and "M2 holds and M3 fails ⇒ refuted"
    fires by its letter (§2.2);
  - the agent argues that the clause assumes algebraic convergence above a floor, which a
    discrepancy already at round-off at depth 10 cannot show, and that the measurement points the
    other way. Per README §5 rule 3 it is recorded as badly posed, not re-posed.

  **Next step:** the user decides at G1 how the verdict is read. Not adjudicated here.

  **Narrowed (2026-10-25, prompt 04, R1):** the full re-run that M3's control stood in for has now
  been made. With every remainder reduced exactly and nothing else changed, the cap goes on 7 of 7
  rows, and the region count falls to 0.20–0.49 % of production's. So the re-run shows what I1
  argued the ratio clause meant. What stays open is the record: how G1 reads M3's verdict.

  **Closed 2026-10-26 at gate G1, by the user's ruling:** the slope clause is badly posed, and the
  mechanism is recorded as holding on M1, M2, M4, M3's ratio clause and 04's R1 (§2.2). 03's verdict
  as written stays on record beside the ruling.
