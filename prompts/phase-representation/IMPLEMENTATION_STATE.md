# Implementation state — the phase-representation campaign

**Campaign:** [`README.md`](README.md) · **Source measurements:**
[`docs/gktk-remedial-verification.md`](../../docs/gktk-remedial-verification.md) §3.5, §3.6, §3.7
**Baseline commit:** `9daa2cb` (`gktk-remedial`, clean — the `GkTk-remedial` close-out)
**Last updated:** 2026-09-13 — campaign opened. Nothing dispatched.

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

| # | Prompt | Closes | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [`WKB_mod_2pi` cycle count](01-wkb-mod-2pi-cycle-count.md) | `[13-wkb-mod-2pi-cycle-count-inconsistent]` | Opus | ⬜ | | |
| 02 | [`PrimitivePhase` break-point knots](02-primitive-phase-break-point-knots.md) | `[13-consumer-spline-crosses-eos-break-points]` | Opus | ⬜ | | |

**Progress:** 0 / 2 complete.

---

## 2. Mechanism-level tracking

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| P1 | **DEFECT, accuracy** | `WKB_mod_2pi`'s cycle count is a rounded division while its remainder is an exact `fmod`, so at large $\|\theta\|$ the stored pair reconstructs $\theta-2\pi$. 1 of 77,975 production $G_k$ samples at $k=3\times10^8$ on LambdaCDM; 6.17 rad of consumer error against a 9.15e-4 rad floor; rate = the half-ulp width of $\|\theta\|/2\pi$, growing linearly with $k$. `simple_mod_2pi` shares the construction. | 01 | ⬜ |
| P2 | **DEFECT, accuracy** | `PrimitivePhase` splines $\varphi$ with `make_interp_spline`'s default knots, interpolating across `QCD_EOS`'s declared break points where $\varphi$ kinks. 1.907e-6 rad ($G_k$, 8 ulp) and 3.186e-6 rad ($T_k$, 428 ulp) at $z=4.24\times10^7$, against 1.00 ulp elsewhere; and `theta_deriv` misses $\omega$ by 2.3e-7–3.3e-4 relative across the QCD interior. | 02 | ⬜ |

**Out of scope (do not schedule):** the numeric→WKB hand-over (`docs/OPEN_ISSUES.md` §1.1); the
$T_k$ numeric tolerance and `rtol` (`prompts/tolerance-convergence`); raising the LG order;
per-region anchoring (`[00-consumer-anchoring-floor]`); the super-horizon series initial condition;
`[02-qcd-T-z-spline-node-tolerance]`, which prompt 02 must *separate out* but may not fix.

---

## 3. Active and unresolved issues

None yet. The two defects this campaign exists to fix live on the
[`GkTk-remedial` board](../GkTk-remedial/IMPLEMENTATION_STATE.md) §3, which owns their measurements
and their history; each carries an `**Assigned (2026-09-13):**` line naming this campaign
(README §0.2). Issues **opened** by a prompt here are recorded in this section, with a row in
[`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit.

---

## 4. Resolved issues

None yet. When a prompt here closes one of the two, the closure is recorded on the
`GkTk-remedial` board's §4 — that board opened them — and this section records which prompt did it
and what it measured.

---

## 5. Standing notes for implementers

1. **The remainder is exact; only the cycle count may move.** Every stored `theta_mod_2pi`, and so
   every stored $G$ and $T$, must be bit-identical across prompt 01 (README §2 (b)).
2. **A cosmology that declares no break points must be bit-identical across prompt 02.** LambdaCDM,
   `RadiationModel` and every stand-in take the unchanged code path (README §2 (e)).
3. **Splitting at declared break points is not chunking.** README §2 (d). Prompt 02's log must say
   which of the two it did, and prove the absence of the switch discontinuity that killed chunking.
4. **The floor is $\varepsilon k\tau$** — 3.05e-7 rad at $k=10^5$/Mpc to 9.15e-4 rad at
   $3\times10^8$. Ten of the twelve production consumer cases are already there. A test that
   asserts below that floor is asserting agreement between two errors
   (`GkTk-remedial` board §5 note 2).
5. **`theta_div_2pi` is in no lookup key.** Prompt 01 moves it at a handful of samples and a
   pre-01 datastore is served silently with the old value. README §7 D3; prompt 01 §5.
