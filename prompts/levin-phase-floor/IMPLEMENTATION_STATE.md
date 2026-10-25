# levin-phase-floor campaign — implementation state

**Last updated:** 2026-10-25 · **Status: STARTED — 1 of 4 measurement prompts landed.** 01 ran on
the Mac Studio on 2026-10-25, in parallel with `handover` 11. Its review stopped at check 9, a
records defect, and was corrected on committing by the user's decision. **H-cancel fails P1 and P2;
H-anchor holds P1–P3** (§2.1). 01 opened two issues (§3). 02 is written. 03 is released by 01 and
not yet written, and 04 is held on 03. The remedy prompts are not written until gate G1.

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
| 02 | [Phase-noise injection](02-phase-noise-injection.md) | findings §2.6 item 2 on A2's fixtures, three arms; N1–N4 | laptop, offline, no Ray | Opus | ✍️ 2026-10-25 | ⬜ | — | — |
| 03 | instrument | findings §2.6 item 1 on rows 01 picks: depth per group, the parent–child discrepancy against the floor | Mac Studio | — | ⬜ released by 01 (`docs/levin-phase-floor/01/out/candidates_01.txt`: 8 `G+r`, 8 `G-r`, 3 lone `G`, 8 `Gqr`); not yet written | ⬜ | — | — |
| 04 | remove | findings §2.6 item 3 in a harness: $k\tau_R$ out of $G$'s phase and/or the group composed before reduction; 03's rows again | Mac Studio | — | ⏸ held on 03 (which construction to test) | ⬜ | — | — |
| 05+ | remedy | §2.7 layers 1–2, as G1 decides | laptop | — | ⏸ held on G1 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**This is a staged campaign, not a half-planned one.** README §2 fixes the charter of every held
prompt. Only its method waits on the measurement before it (README §6.2 C6).

**Orchestrator notes** are written just before dispatch, under `orchestrator/`. The planning
commit is `6af798f`. [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) (2026-10-25) dispatches 01
on the Mac Studio, orchestrated there, in parallel with `handover` 11 in the same checkout. It ran
the same day; its records are `docs/levin-phase-floor/01/` and VERIFICATION §1.

## 2. The gates

| Gate | Holds when | After | State |
|---|---|---|---|
| **G1** | the user has read 01–04 and answered README §6.3 Q1–Q3: which hypothesis stands, which layer 1, and where it lands | 04 | ⬜ |
| **G2** | the user has answered Q4: the depth cap, the acceptance test and `total_converged` | the layer 1–2 prompts | ⬜ |

### 2.1 The hypotheses

| | Statement | Scored by | State |
|---|---|---|---|
| **H-cancel** | findings §2.2 as written: the floor matters where a group's large $\tau_s$ terms should cancel and do not (`G+r` on `r == k`) | 01 P1–P2; 03; 04 | **fails P1** ($S$ = 0.4385 ≥ 0.25) **and P2** (the non-cancelling class tracks $F$ too) **as the whole account** (01, 2026-10-25). Its premise is weaker than stated too: `G+r` on `r == k` is ~3.7× slower than `G-r`, not stationary (`[01-slow-group-leading-terms-do-not-cancel]`) |
| **H-anchor** | $G$'s remainder, reduced against $z_{\rm response}$, carries $\varepsilon k\eta_R$ into every group containing $G$ | 01 P2–P3; 02 N3; 03 | **holds P1–P3** (01, 2026-10-25). P2: both classes have slope 0.97 / 0.82 and median $e_g/F$ 2.12 / 1.49, a ratio of 1.42. P3 discriminates weakly, because $F$ is per row (01's I5). 03 tests the mechanism |

They are not exclusive.

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
