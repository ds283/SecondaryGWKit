# levin-phase-floor campaign — implementation state

**Last updated:** 2026-10-25 · **Status: STARTED — 3 of 4 measurement prompts closed: 01 and 03
landed, 02 stopped and superseded.** 01 ran on the Mac Studio on 2026-10-25, in parallel with
`handover` 11. Its review stopped at check 9, a records defect, and was corrected on committing by
the user's decision. **H-cancel fails P1 and P2; H-anchor holds P1–P3** (§2.1). 01 opened two
issues (§3). 03 ran on the Mac Studio the same day, Ray-free, and its review passed 9/9. **M1, M2,
M4 and M5 hold; M3 fails on its slope clause, so the mechanism as written does not hold**, and G1
decides how that is read (§2.2). H-anchor holds in every scored group. 03 opened three issues (§3).
**02 stopped at step 3 and is superseded by 03** (user, 2026-10-25):
- its noise sweep was infeasible and is not run in any form;
- N1–N4 are not scored, and N4 moves to 04.

04 waited on 03, which has landed; it is not yet written. The remedy prompts are not written until gate G1.

The plan is [`README.md`](README.md), written on 2026-10-25 at `effba46`. Its audit is
[`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §2.

The campaign owns the depth-20 cost cliff of the `QuadSourceIntegral` Levin integration, and the
honesty of the error and convergence a row stores. It first runs findings §2.6's test, extended to
two named hypotheses, H-cancel and H-anchor (README §0.2). Then it lands the remedy layers G1
admits.

**Campaign:** [`README.md`](README.md) ·
**Code:** none in 01–04; the remedy's set after G1 ·
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

The orchestrator's choices of method, C1–C6, are in README §6.2. The open questions for the user,
Q1–Q4, are in README §6.3.

---

## 1. Prompts

| # | Prompt | Covers | Where | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|---|
| 01 | [The phase-group census of A3 v2](01-phase-group-census.md) | cost, regions and declared error per group; P1–P3; findings §2.5 on the store; the candidates for 03 | Mac Studio, read-only, no Ray | Opus | ✍️ 2026-10-25 | ✅ 2026-10-25 on the Mac Studio, reviewed there; checks 1–8 pass, check 9 (records) failed and was corrected on committing, by the user's decision; shares and $S$ re-checked from the laptop | records in the bookkeeping commit | [log](logs/01-phase-group-census.md) |
| 02 | [Phase-noise injection](02-phase-noise-injection.md) | findings §2.6 item 2 on A2's fixtures, three arms; N1–N4 | laptop, offline, no Ray | Opus | ✍️ 2026-10-25 | ⛔ stopped at step 3, superseded by 03 (user, 2026-10-25). Step 2's bit-identity at δ = 0 held on all eight cases; the δ = 1e-4 check never ran; the probe `levin-phase-floor-02-noise-probe-20261025T183809` was stopped by SIGTERM, `killed` at 16/19, after its first noised cell ran ~69 min unfinished; the sweep was never launched; N1–N4 not scored (N1, N3 answered by 03; N4 to 04; N2 dropped) | `87813ed` | [log](logs/02-phase-noise-injection.md) |
| 03 | [Instrument seven depth-20 rows](03-instrument-depth-20-rows.md) | findings §2.6 item 1 on seven of 01's candidates: depth per group by branch; the jitter in the composed phase; the discrepancy along a region's ancestry, real and with the jitter removed; what the driver can see; how far `G+r` cancels. M0 (bit-identity) gates M1–M5 | Mac Studio; Ray-free if a stand-in read-only pool works, else under the lock | Opus | ✍️ 2026-10-25 | ✅ 2026-10-25 on the Mac Studio, run `levin-phase-floor-03-instrument-20261025T194950`; route (a), Ray-free; M0 bit-identical 7/7; store unchanged, `0e32f24a…`; reviewed there, 9/9 pass, three readings for the user (`elapsed` not gated, M5 inapplicable on R7, the pid in `status.json`); R1 re-checked from the laptop: stored and re-run agree in `float.hex` on `total` (`-0x1.631c5efe83a46p-78`), `total_abserr` (`0x1.2f222e3d4dd6dp-95`), `WKB_Levin` and `numeric_quad` (`0x0.0p+0`), `total_converged` 0 / False | `d5f5a2b` (records) | [log](logs/03-instrument-depth-20-rows.md) |
| 04 | remove | findings §2.6 item 3 in a harness: $k\tau_R$ out of $G$'s phase and/or the group composed before reduction; 03's rows again; 02's N4 (README §2) | Mac Studio | — | ⏸ held on 03 (which construction to test) | ⬜ | — | — |
| 05+ | remedy | §2.7 layers 1–2, as G1 decides | laptop | — | ⏸ held on G1 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed · ⛔ stopped and superseded.

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
note, not opened as an issue).

## 2. The gates

| Gate | Holds when | After | State |
|---|---|---|---|
| **G1** | the user has read 01–04 and answered README §6.3 Q1–Q3: which hypothesis stands, which layer 1, and where it lands | 04 | ⬜ |
| **G2** | the user has answered Q4: the depth cap, the acceptance test and `total_converged` | the layer 1–2 prompts | ⬜ |

### 2.1 The hypotheses

| | Statement | Scored by | State |
|---|---|---|---|
| **H-cancel** | findings §2.2 as written: the floor matters where a group's large $\tau_s$ terms should cancel and do not (`G+r` on `r == k`) | 01 P1–P2; 03; 04 | **fails P1** ($S$ = 0.4385 ≥ 0.25) **and P2** (the non-cancelling class tracks $F$ too) **as the whole account** (01, 2026-10-25). Its premise is weaker than stated too: `G+r` on `r == k` is ~3.7× slower than `G-r`, not stationary (`[01-slow-group-leading-terms-do-not-cancel]`), confirmed by 03's M5. **03 (2026-10-25): its M1 condition is met** in all 18 cancelling groups, **but equally** in all 11 non-cancelling groups (median σ_θ/F 0.270 against 0.262), **so it does not single them out** |
| **H-anchor** | $G$'s remainder, reduced against $z_{\rm response}$, carries $\varepsilon k\eta_R$ into every group containing $G$ | 01 P2–P3; 03 (02 N3 not run: superseded) | **holds P1–P3** (01, 2026-10-25). P2: both classes have slope 0.97 / 0.82 and median $e_g/F$ 2.12 / 1.49, a ratio of 1.42. P3 discriminates weakly, because $F$ is per row (01's I5). **03 (2026-10-25): M1 and M2 hold in every scored group** (29 of 29), `G−r` (R4, R5: σ_θ/F 0.256, 0.225) and the lone `G` (R6: 0.337) included. The mechanism it names is §2.2 |

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
refuted" fires by its letter.

**I1's argument, the agent's** (03's `issues.md` I1 and log note (a)): with the jitter removed, D
is at round-off, about 1e-16 to 1e-13, from depth 10 onward, so no slope can be measured; the ratio
clause carries the comparison the prompt intends, and it points toward the jitter. The orchestrator
recorded this beside the score and did not adjudicate it. **G1 decides** how the verdict is read
(`[03-m3-slope-clause-is-badly-posed-at-round-off]`).

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

**Assigned here on 2026-10-25, held on their own boards** (README §1, §6.2 C5):
- `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` — `handover` board §3;
- `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` — `handover` board §3;
- `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` — `handover` board §3;
- `[00-consumer-anchoring-floor]` — `GkTk-remedial` board §3.

**Measured here, not assigned:** `[04-roundoff-floor-can-be-infinite]` (`levin-refactor` board).
01 confirms whether it reaches A3 v2, and narrows it there.

## 4. Resolved issues

None.
