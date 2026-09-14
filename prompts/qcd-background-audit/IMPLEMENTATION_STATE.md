# Implementation state — the QCD background campaign

**Campaign:** [`README.md`](README.md) · **Source document:**
[`docs/qcd-background-audit-2026-09.md`](../../docs/qcd-background-audit-2026-09.md)
**Baseline commit:** `e8f746d` (`qcd-background-audit`, clean; identical to `main`)
**Last updated:** 2026-09-14 — **prompt 01 complete; 1 / 12.** Twelve prompts in four workstreams.
Every figure below is the audit's, and prompt 01 re-measured the representation, the branch joins,
the jump locations and the conformal-time error **from the test tree** on `2a5e0fa`: all of them
reproduce the audit to every digit it quotes. The audit's script
(`PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py`) remains
the reproduction for the figures prompt 01 did not take.

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Workstream A — the representation (prompts 01–06)

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 01 | [The background-against-background harness](01-background-reference-harness.md) | Opus | ⚠️ | *"Add a background-against-background test for the QCD temperature"* (SHA not embedded, per the campaign convention) | [`logs/01-background-reference-harness.md`](logs/01-background-reference-harness.md) |
| 02 | [Make the QCD reference fixture regenerable](02-regenerable-qcd-references.md) | Sonnet | ⬜ | | |
| 03 | [Key the `T(z)` representation](03-key-the-representation.md) | Opus | ⬜ | | |
| 04 | [Tighten `_solve_T_z` (T2)](04-tighten-node-solve.md) | Sonnet | ⬜ | | |
| 05 | [Spline the entropy factor (T3)](05-entropy-factor-representation.md) | Opus | ⬜ | | |
| 06 | [Segment at the jumps (T4)](06-segment-at-the-jumps.md) | Opus | ⬜ | | |

### Workstream B — the break-point set (prompts 07–08)

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 07 | [Re-derive `integration_break_points` (G1)](07-rederive-break-points.md) | Opus | ⬜ | | |
| 08 | [Re-measure the per-sector break-point policy](08-per-sector-policy-remeasure.md) | Opus | ⬜ | | |

### Workstream C — close-out (prompt 09)

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 09 | [The consumer tables under a corrected background](09-close-out-verification.md) | Opus | ⬜ | | |

### Workstream D — the source grid and the consumer spline (prompts 10–12) — **gated on README §7 D7**

| # | Prompt | Model | Status | Commit | Log |
|---|---|---|---|---|---|
| 10 | [`PrimitivePhase` on the 3-point break set](10-primitive-phase-break-points.md) | Opus | ⬜ | | |
| 11 | [A cosmology-aware source grid](11-cosmology-aware-source-grid.md) | Opus | ⬜ | | |
| 12 | [A measured grid-density criterion](12-grid-density-criterion.md) | Opus | ⬜ | | |

**Progress:** 1 / 12 complete (1 / 9 in the ungated chain 01–09).

**Prompt 01 landed `COMPLETE WITH DEVIATIONS`** — no production file changed; two new modules,
`CosmologyModels/tests/T_z_reference.py` and `CosmologyModels/tests/test_T_z_representation.py`,
seven new tests, `CosmologyModels` 11 → 18. The five deviations are all recorded in
[its log](logs/01-background-reference-harness.md) and none touches a README §2 design fact; the
two that matter to later prompts are (2) the jump height is **7.6229229003969e-04** in $F$ and
**7.625829e-04** in $T$, the audit's 7.614e-04 being §1's *linearised* figure, and (3) a
`root_scalar` bracket on $T(z)-T_{\rm break}$ reports `converged` and returns a **non-root** —
which is audit §2's claim confirmed, in the second of the two forms prompt 01 allowed for.

**The representation version.** `T_Z_REPRESENTATION_VERSION` is introduced by prompt 03 and bumped
by **04, 05, 06 and 07**. Its value at each prompt boundary is recorded here as the campaign runs,
because it is the only thing that tells a datastore that its QCD rows are stale.

| After prompt | `T_Z_REPRESENTATION_VERSION` | What changed |
|---|---|---|
| 01 | *(does not exist)* | no production file touched |
| 03 | 1 | nothing numerically; the key exists |
| 04 | *(to be recorded)* | node solve tightened |
| 05 | *(to be recorded)* | entropy-factor representation |
| 06 | *(to be recorded)* | segmented at the jumps |
| 07 | *(to be recorded)* | break-point set collapsed |

---

## 2. Mechanism-level tracking

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| **T1** | **DEFECT, critical** | 3.461e-08 relative in $\int\mathrm{d}z/H$ — of order 47.5 / 4.75e3 / 1.43e5 rad at $k=10^5/10^7/3\times10^8$ against 1-ulp floors of 3.05e-7 / 3.05e-5 / 9.15e-4 rad. **Common mode** between producer and consumer, so invisible to every test in the tree. | 01, 04, 05, 06 | 🟡 |
| **T2** | **DEFECT, high** | `_solve_T_z` root-solves each spline node to `rtol=1e-4`; neighbouring nodes carry uncorrelated errors up to 2.496e-05. The root of `[02-qcd-T-z-spline-node-tolerance]`, and (measured, not asserted) of the $\pm0.1$ scatter in $\omega^2/\omega_0^2$ that `RESIDUAL_WKB_REGION_MARGIN = 0.5` exists to survive (`ComputeTargets/phase_residual.py:220`). | 04 | ⬜ |
| **T3** | **DEFECT, medium** | $T$ is splined against $u$, spending resolution on the $(1+z)$ ramp known in closed form. Splining $F(u)=\log(T/[T_{\rm CMB}(1+z)])$ instead improves the median 400× at the same node count (1.071e-07 → 2.599e-10). | 05 | ⬜ |
| **T4** | **DEFECT, high** | One global spline across three points at which $T(z)$ genuinely **jumps** (7.614e-04 at $z_c=4.25337\times10^7$). The max error is pinned near the jump height at 500, 2,000 and 5,000 nodes alike. | 06 | ⬜ |
| **G1** | **DEFECT, high** | 404 of the 407 `BREAK_POINT_ALL` points are knots of the auxiliary interpolant: a Gauss panel split every 4.04 grid intervals throughout `BackgroundModel`, and the sole cause of `prompts/phase-representation` prompt 02's Schoenberg–Whitney failure. | 07, 08 | ⬜ |
| **P2** | **DEFECT, accuracy** | Inherited `[13-consumer-spline-crosses-eos-break-points]`: `PrimitivePhase` splines $\varphi$ with default knots across the declared break points. 1.907e-6 rad ($G_k$, 8 ulp) and 3.186e-6 rad ($T_k$, 428 ulp) at $z=4.24\times10^7$. Blocked until G1 is gone. | 10 | ⬜ |
| **G2** | **DESIGN** | The source grid never consults the cosmology: `populate_z_sample` is a bare `logspace`, `winnow` a blind stride `[::-n]`, and the tag `SourceRedshiftGrid_{len}` labels size only, so two different grids of equal length collide in the datastore. | 11, 12 | ⬜ |

**T1 is 🟡 because its guard exists but its fix does not.** Prompt 01 put
`CosmologyModels/tests/test_T_z_representation.py::test_conformal_time_matches_the_exact_background`
in the tree: $\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$ computed twice from the same cosmology,
once as shipped and once with the temperature replaced by the accurate root solve, so that nothing
cancels. It measures **3.4605051e-08**, against the audit's 3.461e-08, in 0.013 s. Prompt 06
tightens its threshold from 4.0e-08 to 1e-15.

**Out of scope (do not schedule here):** `[00-consumer-anchoring-floor]` and
`[02-consumer-phi-below-the-storage-granularity]` — per-region anchoring, untouched by background
accuracy (audit §9); the numeric→WKB hand-over (`docs/OPEN_ISSUES.md` §1.1); tolerances and Gauss
orders (`prompts/tolerance-convergence`); the `QCD_EOS` fitting coefficients and branch boundaries
(README §0.5, §7 D6); `AdaptiveLevin/` and the Levin consumers.

---

## 3. Active and unresolved issues

Opened by this campaign's planning, 2026-09-13:

- **[00-eos-branch-joins-do-not-match]** *(planning, 2026-09-13; **pinned in a test by prompt 01,
  2026-09-14**, which re-measured every figure below from the test tree and confirmed all of them
  to the digits quoted)* — `QCD_EOS`'s branch joins at
  $T=10^{16}$, 0.12 and $10^{-5}$ GeV jump by **+1.395e-02**, **−3.744e-04** and **−2.284e-03** in
  $g_s$ (and +1.454e-02, −2.075e-04, +8.876e-04 in $g$), forcing steps in $T(z)$ at fixed $z$ of
  −4.649e-03, +1.248e-04 and **+7.614e-04**. The join at 0.002 GeV matches to **1.751e-11**, and
  that asymmetry is the evidence that the other three are a defect of the transcription rather than
  the parametrisation's intent: if discontinuous joins were designed in, none of the four would
  match to 1.8e-11. **Impact:** the $10^{-5}$ GeV join is the origin of the $4.4\times10^{-4}$ jump
  in $H(z)$ at $z=4.24\times10^7$ that `GkTk-remedial` log 02 measured without attribution, and of
  the consumer's worst QCD error in `docs/gktk-remedial-verification.md` §3.5 in **both** sectors.
  **This campaign does not repair it** — the equation of state is an upstream data fixture and a
  segmented representation reproduces a discontinuous fixture exactly (audit §0.2). **Next step:**
  the question in README §7 D6, put to the authors of the Saikawa & Shirai transcription. Prompt
  01's `test_the_branch_joins_are_where_the_fixture_puts_them`
  (`CosmologyModels/tests/test_T_z_representation.py`) now **pins all four joins**, and also pins
  the *set* of `break_temperatures_GeV`, so a later correction announces itself as a test failure
  rather than as a silent change of cosmology. That test's docstring says in terms that a failure
  means the fixture changed — not that the code regressed — and that the campaign's segment edges
  must then be re-derived. Measurements: audit §1 and §2, reproducible in 1.0 s, and re-measured
  by prompt 01 in 0.11 s as part of the `CosmologyModels` suite.

Inherited, and **assigned to this campaign** (each is owned by the board named, which holds its
measurements and its history; the closure is recorded there):

| Issue | Owning board | Closed by | Note |
|---|---|---|---|
| `[13-consumer-spline-crosses-eos-break-points]` | GkTk-remedial | prompt 10 | Assigned 2026-09-13 by `prompts/phase-representation`'s close-out. Blocked until prompt 07 |
| `[02-qcd-T-z-spline-node-tolerance]` | GkTk-remedial | prompt 04 | T2 is this issue at its root; prompt 04 must also re-measure the $\omega^2$ scatter it causes |
| `[01-genericeos-tz-spline-floor]` | source-remediation | prompts 05, 06 | "Whether the `T(z)` spline grid is adequately defined." The answer is in audit §3 and §4 |
| `[19-cosmologymodels-docstrings-predate-per-sector-policy]` | GkTk-remedial | prompt 07 | The two docstrings are exactly the text prompt 07 rewrites |

Re-measured but **not owned** here (they stay where they are; a prompt that moves one says so):

| Issue | Owning board | Touched by |
|---|---|---|
| `[02-qcd-reference-floor]` | GkTk-remedial | prompts 02, 04, 05, 06 — the QCD references are regenerated and stop being circular |
| `[03-qcd-short-baseline-reference-endpoint-rounding]` | GkTk-remedial | prompts 02, 06 |
| `[20-wkb-gauss-orders-not-in-lookup-key]` | GkTk-remedial | prompt 04 — `RESIDUAL_WKB_REGION_MARGIN`'s reason for existing is measured, not changed |
| `[20-wkb-rows-consume-numeric-initial-data]` | GkTk-remedial | prompt 03 — the same class of defect one level up; prompt 03 does not fix that one |

---

## 4. Resolved issues

*(none yet — the campaign has not started)*

---

## 5. Standing notes for implementers

1. **A test that passes both before and after proves nothing.** README §0.2. The error this
   campaign removes is common mode between every producer and every consumer in the tree. **The
   harness exists as of prompt 01**: `CosmologyModels/tests/T_z_reference.py` (the reference) and
   `CosmologyModels/tests/test_T_z_representation.py` (seven cases). Every later prompt is scored
   against it, and every threshold in it is a named module constant carrying the prompt that
   tightens it.
2. **Never use the shipped `_solve_T_z` as a reference.** It is one of the things being measured.
   The reference is `T_z_reference.accurate_T(cosmology, z, rtol=1.0e-14)` — the defining equation
   (README §2 (a)).
3. **Bisect for a segment edge; never root-find on $T(z)-T_{\rm break}$.** README §2 (b). Use
   `T_z_reference.jump_locations(cosmology)`; its three values on the production QCD model, to 17
   digits, are in log 01's handover. Prompt 01 measured the trap: a `root_scalar` bracket on
   $T(z)-T_{\rm break}$ reports `converged=True` and returns a **non-root** whose relative residual
   is 8.844e-06 — 1.2 % of the jump height — because the difference changes sign inside one ulp of
   $u$ without passing through zero. Its position depends on the tolerance (`+1.126e-12`,
   `+3.304e-13` or `+1.421e-14` in $u$), and the first of those is larger than the `pad = 1e-12` the
   audit's segmented build uses, which is how a first attempt left the full 5.7e-04 error in place.
4. **The jump height is 7.6229229003969e-04 in $F$ and 7.625829e-04 in $T$.** The **7.614e-04** in
   audit §2's prose and README §2 (b) is §1's *linearised* $-\tfrac13\Delta g_s/g_s$: correct, but a
   different quantity, 0.15 % away. Log 01 deviation 2.
5. **The improved representation is cheaper per call**, not more expensive: 2.19–2.21 µs against
   2.26–2.44 µs. A reported regression means something other than the measured design was built.
6. **Every number this campaign moves is invisible to the datastore's lookup key** until prompt 03
   lands. That is why prompt 03 comes before prompt 04 and not after.
7. **The QCD half of `wkb_reference_data.json` is built from the shipped `T(z)`** and moves with it
   (README §2 (e)). Ten test modules assert against it. Prompt 02 is the map.
8. **`BREAK_POINT_ALL` is load-bearing today.** `GkTk-remedial` prompt 19 measured that the $T_k$
   numeric sector needs it: 3 of 50 QCD wavenumbers missed the criterion with jumps alone. That
   measurement was taken against knots carrying a $10^{-4}$-level defect; prompt 08 re-takes it, and
   an unmeasured collapse of the set is a stop condition (README §2 (f)).
9. **LambdaCDM has no `T(z)` spline** — `CosmologyModels/LambdaCDM/LambdaCDM.py:129` returns
   $T_{\rm CMB}(1+z)$ in closed form and the class declares no break points. Every LambdaCDM,
   `RadiationModel` and stand-in number is bit-identical across this entire campaign. So is a
   `LambdaCDM_GenericEOS` built on a constant-$g_s$ equation of state, for which the new
   representation is **exact** rather than merely accurate (README §2 (g)).
