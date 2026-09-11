# Implementation state — QuadSourceIntegral phase groups

**Campaign:** [`README.md`](README.md)
**Baseline commit:** `0020b2f` (`claude/workstream-a-orchestrator-4982a3`, clean) — the close of
`prompts/transfer-remedial`, which is what makes `theta_abserr` available to declare.
**Last updated:** 2026-09-11 — campaign opened.

> **Maintenance rule.** The prompt updates this file *in its own commit*: its row, the item table,
> and §3/§4. **Any change to §3 or §4 must also update
> [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit** (see `CLAUDE.md`).

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [Three-Bessel Levin phase groups](01-three-bessel-levin-phase-groups.md) | Q1, Q2, Q3 | Opus | ⬜ | | |

**Progress:** 0 / 1 complete.

---

## 2. Item-level tracking

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| Q1 | **DEFECT, accuracy** | `_three_bessel_Levin`'s eight `adaptive_levin_sincos` calls build each group by summing three `raw_theta` values and three `theta_mod_2pi` values, so a near-resonant group carries the absolute error of its largest constituent. Measured in the sibling module at 2.86e-5 rad against 1.42e-13 for the \(Kt+C+R(t)\) assembly | 01 | ⬜ |
| Q2 | **DEFECT, error bound** | The same eight calls supply no `theta_deriv`, so Levin spectrally differentiates the raw phase and sizes its subdivision from that (`levin_quadrature.py:1038`, `:1090`), and no `theta_abserr`, so the reported `abserr` is silent about the phase construction | 01 | ⬜ |
| Q3 | **DEFECT, stale documentation** | Two comments in `QuadSourceIntegral.py` describe the pre-`transfer-remedial` Bessel oracle: `:80`'s "~2e-8 of that envelope", and `_check_bessel_order`'s docstring, which says the returned dict carries `Q` and no `"nu"` — the reverse of what is now true | 01 | ⬜ |

**Out of scope (do not schedule):** `ComputeTargets/phase_groups.py`, which composes cosmological
phases with no analytic leading term and legitimately sums `raw_theta`
(`README.md` §2); `CHEBYSHEV_ORDER` and `DEFAULT_3BESSEL_CHEBYSHEV_ORDER`
(`[08-3bessel-chebyshev-order-is-now-the-limit]`); replacing `_check_bessel_order`'s numeric guard
with a direct `phase_data["nu"]` comparison, which is a behaviour change.

---

## 3. Active and unresolved issues

One entry is opened at campaign creation, because it is the reason the campaign exists and it is
inherited rather than created by the prompt.

- **[00-orphaned-handoff]** *(opened by the planning pass, 2026-09-11)* — the gap this campaign
  fixes was recorded twice and owned by nobody. `prompts/transfer-remedial`'s prompt 09 filed
  `[transfer-remedial-qsi-phase-groups]` into `prompts/source-remediation`'s §3 as a hand-off, but
  that campaign was already complete at 13/13, so no prompt remained to discharge it; and the row
  went into §2 of `docs/OPEN_ISSUES.md` ("Error-bound completeness") rather than §1 ("Assigned to a
  future campaign"), with no `**Assigned (date):**` line on the board entry — the mechanism
  `CLAUDE.md` defines for exactly this case. `transfer-remedial` then closed its own
  `[00-qsi-three-bessel-levin-excluded]` into §4 on the strength of that hand-off.
  **Impact:** none on the code, which this campaign fixes. The process lesson is that a hand-off to
  a campaign should check that the receiving campaign has a prompt left to receive it, and should
  use §1 plus an `Assigned` line. **Next step:** closed by prompt 01, which discharges the
  underlying defect; the hand-off entries are then resolved on both boards.

> Add an entry here whenever the prompt finishes with something unresolved. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them.

---

## 4. Resolved issues

*(none yet)*

---

## 5. Standing notes for implementers

1. **`ComputeTargets/phase_groups.py` is not the same thing.** It composes `phase_spline` objects
   over \(\log(1+z)\), which carry no analytic leading term, so summing `raw_theta` is the correct
   construction there. The Bessel leading term \(x\) is what makes the \(Kt+C+R(t)\) split possible
   here. Do not unify them.
2. **`_PhaseGroup` is tested where it lives.** `LiouvilleGreen/tests/test_three_bessel.py`'s
   `TestPhaseGroups` measures both routes against 60-digit `mpmath` and asserts the declared error
   bounds the measured one. Reuse the class; do not reimplement it, and do not change its
   behaviour.
3. **`analytic_rad` is a stored column and an acceptance oracle.** It is persisted for every work
   item (`QuadSourceIntegral.py:917`) and `test_quadsource_integral.py`'s `TestAnalyticOracle`
   scores `total` against it. A change here moves a stored number; measure how much and attribute
   it.
4. **The `J`/`Y` split is the convention \(J=A\sin\theta\), \(Y=-A\cos\theta\)**, carried by the
   `f` vectors. It is a convention, not a defect; preserve it exactly.
5. **`theta` remains a required Levin key** even though it is never evaluated when `theta_mod_2pi`
   and `theta_deriv` are both supplied (`levin_quadrature.py:948-952`, `:1038`).
   "Compatibility-only" is not "optional".
