# Implementation state — QuadSourceIntegral phase groups

**Campaign:** [`README.md`](README.md)
**Baseline commit:** `0020b2f` (`claude/workstream-a-orchestrator-4982a3`, clean) — the close of
`prompts/transfer-remedial`, which is what makes `theta_abserr` available to declare.
**Last updated:** 2026-09-11 — prompt 01 landed; campaign complete at 1 / 1.

> **Maintenance rule.** The prompt updates this file *in its own commit*: its row, the item table,
> and §3/§4. **Any change to §3 or §4 must also update
> [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit** (see `CLAUDE.md`).

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [Three-Bessel Levin phase groups](01-three-bessel-levin-phase-groups.md) | Q1, Q2, Q3 | Opus | ⚠️ | *(the commit that adds this row; a SHA cannot be self-embedded)* | [log](logs/01-three-bessel-levin-phase-groups.md) |

**Progress:** 1 / 1 complete.

⚠️ rather than ✅ for one reason, and it is not about the eight calls the campaign exists for:
README §5's acceptance sentence "every `adaptive_levin_sincos` call in `QuadSourceIntegral.py`
supplies all four phase keys" is true of the eight analytic calls and **not** of the ninth, in
`phase_group_Levin_integral`, which composes cosmological `phase_spline` phases that have no
declared fit accuracy to pass as `theta_abserr`. Fixing it means editing
`ComputeTargets/phase_groups.py`, which README §2 and the prompt's own file list forbid. Recorded
as `[01-cosmological-group-declares-no-phase-error]` in §3. The prompt's own executable statement
of the same requirement (§3 item 1, "every *analytic* Levin call") is met: all sixteen carry all
four keys.

---

## 2. Item-level tracking

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| Q1 | **DEFECT, accuracy** | `_three_bessel_Levin`'s eight `adaptive_levin_sincos` calls build each group by summing three `raw_theta` values and three `theta_mod_2pi` values, so a near-resonant group carries the absolute error of its largest constituent. Measured in the sibling module at 2.86e-5 rad against 1.42e-13 for the \(Kt+C+R(t)\) assembly | 01 | ✅ |
| Q2 | **DEFECT, error bound** | The same eight calls supply no `theta_deriv`, so Levin spectrally differentiates the raw phase and sizes its subdivision from that (`levin_quadrature.py:1038`, `:1090`), and no `theta_abserr`, so the reported `abserr` is silent about the phase construction | 01 | ✅ |
| Q3 | **DEFECT, stale documentation** | Two comments in `QuadSourceIntegral.py` describe the pre-`transfer-remedial` Bessel oracle: `:80`'s "~2e-8 of that envelope", and `_check_bessel_order`'s docstring, which says the returned dict carries `Q` and no `"nu"` — the reverse of what is now true | 01 | ✅ |

**Out of scope (do not schedule):** `ComputeTargets/phase_groups.py`, which composes cosmological
phases with no analytic leading term and legitimately sums `raw_theta`
(`README.md` §2); `CHEBYSHEV_ORDER` and `DEFAULT_3BESSEL_CHEBYSHEV_ORDER`
(`[08-3bessel-chebyshev-order-is-now-the-limit]`); replacing `_check_bessel_order`'s numeric guard
with a direct `phase_data["nu"]` comparison, which is a behaviour change.

---

## 3. Active and unresolved issues

- **[01-cosmological-group-declares-no-phase-error]** *(opened by prompt 01, 2026-09-11)* — the
  ninth `adaptive_levin_sincos` call in `ComputeTargets/QuadSourceIntegral.py`, in
  `phase_group_Levin_integral` (`:1062`), passes
  `group.levin_theta(include_deriv=LEVIN_USE_THETA_DERIV)` from `ComputeTargets/phase_groups.py`,
  whose `levin_theta` (`:194-202`) builds `{"theta", "theta_mod_2pi", "theta_deriv"}` and has no
  `theta_abserr` branch at all. So README §5's acceptance sentence — "every
  `adaptive_levin_sincos` call in `QuadSourceIntegral.py` supplies all four phase keys" — is met by
  the eight analytic calls and not by this one, and the prompt could not close the gap: that file
  is on its "Do not touch" list and on §2's out-of-scope list, for the good reason that its phases
  are cosmological `phase_spline` objects and there is nothing for them to declare — a
  `phase_spline` does not report its own fit accuracy. That is the root cause, and it is
  `prompts/levin-refactor`'s `[09-abserr-does-not-bound-phase-spline-floor]` seen from the other
  end. **Impact:** the `WKB_Levin` part of `total` — the production path, not the analytic oracle —
  still reports an `abserr` that cannot see its own phase construction, so a region there is
  subdivided against a phase it may not be able to resolve rather than being reported
  `phase_limited`. Unchanged by this campaign, which only ever touched the analytic branch.
  **Next step:** give `phase_spline` a declared absolute phase error, then a `theta_abserr` branch
  in `ComputeTargets/phase_groups.PhaseGroup.levin_theta`. That is a `phase_spline` change first
  and belongs with the hand-over campaign's representation work (`docs/OPEN_ISSUES.md` §1.1), not
  here.

- **[06-analytic-rad-is-computed-at-the-callers-tolerance]** *(opened 2026-09-18 by
  `prompts/tolerance-convergence` prompt 06, board item **T11**, which measures
  `QuadSourceIntegral` read-only and may not edit it — README §0.4 of that campaign; filed here
  because the quantity is built by the three-Bessel machinery this campaign owns)* —
  `evaluate_QuadSource_integral` computes the stored oracle column `analytic_rad` by calling
  `analytic_integral` with **the caller's own `atol` and `rtol`** (`QuadSourceIntegral.py:884-897`),
  which is deliberate and is recorded at `:1547-1552` as the repair of audit B6/QI-9. The
  consequence nobody has recorded is that `analytic_rad` is then **not a fixed oracle**: it is a
  function of the row's `atol_serial` and `rtol_serial`, and it is *more* sensitive to them than
  `total` is. Measured offline over the 18 cases of
  `ComputeTargets/tests/test_quadsource_integral.py` in the exact flavour
  (`docs/tolerance-convergence/QUADSOURCE-READONLY.md` §2.1): at the production pair `analytic_rad`
  sits up to **3.84e-09** of `scale = max(|numeric_quad|, |WKB_Levin|, |analytic_rad|)` from its own
  value at `(1e-45, 1e-12)`, which is up to **×1.25e+06** the quadrature error of `total` on the
  same case; at `rtol = 1e-5` it reaches 2.66e-06. **Impact:** two. (i) Any comparison of the form
  `|total - analytic_rad|` taken from a single run — which is what the fixture's own acceptance
  tests and `docs/source-remediation-verification.md` report — compares two quantities that moved
  together, so it cannot be *swept* over tolerances; prompt 06 had to hold the oracle at a reference
  pair to measure anything. (ii) Anyone reading a stored `analytic_rad` across rows computed at
  different tolerances is reading a column that is not comparable between them. **Next step:** decide
  whether `analytic_rad` should be computed at a fixed, declared pair of its own rather than at the
  row's — it is an oracle, not a payload, and its whole value is in being the same number for every
  row — or whether the dependence should simply be documented at `QuadSourceIntegral.py:1547-1552`
  and in the schema. Either is a change to a file `prompts/tolerance-convergence` may not touch.

> Add an entry here whenever the prompt finishes with something unresolved. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them.

---

## 4. Resolved issues

- **[00-orphaned-handoff]** *(opened by the planning pass, 2026-09-11; **closed by prompt 01**,
  2026-09-11)* — the gap this campaign fixes was recorded twice and owned by nobody.
  `prompts/transfer-remedial`'s prompt 09 filed `[transfer-remedial-qsi-phase-groups]` into
  `prompts/source-remediation`'s §3 as a hand-off, but that campaign was already complete at 13/13,
  so no prompt remained to discharge it; and the row went into §2 of `docs/OPEN_ISSUES.md`
  ("Error-bound completeness") rather than §1 ("Assigned to a future campaign"), with no
  `**Assigned (date):**` line on the board entry — the mechanism `CLAUDE.md` defines for exactly
  this case. `transfer-remedial` then closed its own `[00-qsi-three-bessel-levin-excluded]` into §4
  on the strength of that hand-off.
  **Impact:** none on the code, which this campaign fixed. The process lesson is that a hand-off to
  a campaign should check that the receiving campaign has a prompt left to receive it, and should
  use §1 plus an `Assigned` line.
  **Resolution:** prompt 01 discharged the underlying defect, and both hand-off entries are now in
  their own boards' §4 — `[transfer-remedial-qsi-phase-groups]` on `source-remediation`'s and
  `[05-quadsource-order-check-docstring-stale]` on `transfer-remedial`'s — each edited only to mark
  it resolved with a pointer here, as prompt 01 §5 required. `docs/OPEN_ISSUES.md` §1.2 is
  correspondingly empty.

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
