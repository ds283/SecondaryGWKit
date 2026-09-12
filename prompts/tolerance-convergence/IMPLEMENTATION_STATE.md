# Implementation state — tolerance and convergence campaign

**Campaign:** [`README.md`](README.md) · **Logs:** [`logs/`](logs/)
**Last updated:** 2026-09-12 · **Status: planned, not started.**

This campaign is **blocked on `prompts/GkTk-remedial`** (README §0.3, §4.2): its prompt 18 must
land before any tolerance is measured here, because until it does the reference on `QCDModel` does
not converge at four wavenumbers (`[17-qcd-reference-not-converged]`), and its prompt 13 is the
practical precondition. Nothing below may be dispatched before that campaign merges to `main`.

---

## 1. The board

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | The convergence harness | README §2 (g); the method of GkTk-remedial `[17-qcd-reference-not-converged]` | Opus | ⬜ | | |
| 02 | Audit the numeric sectors | README §2 (c), (d), (e); review §10.1, §12.5 | Opus | ⬜ | | |
| 03 | Audit the WKB sectors | README §2 (a) | Opus | ⬜ | | |
| 04 | Decouple the tolerances | README §2 (d), (f); §7 D1 | Opus | ⬜ | | |
| 05 | `QuadSourceIntegral` and close-out | README §0.4, §2 (b) | Opus | ⬜ | | |

Status key: ⬜ not started · 🔄 in flight · ✅ complete · ⚠️ complete with a recorded caveat ·
❌ blocked.

---

## 2. Item-level state

One row per thing the campaign claims to establish. Filled in as prompts land; a row whose evidence
is a single wavenumber or a single model is **not** ✅, which is the specific failure this campaign
was created by.

| Item | Kind | Statement | Prompt | Status |
|---|---|---|---|---|
| T1 | **MACHINERY** | One reusable convergence facility, in the test tree, covering all five targets and calibrated against the constant-$w$ anchors at every use | 01 | ⬜ |
| T2 | **MEASUREMENT** | `TkNumericIntegration` characterised over the production $k$-grid on three models in both `atol` and `rtol` | 02 | ⬜ |
| T3 | **MEASUREMENT** | `GkNumericIntegration` likewise — no grid sweep, no QCD figure and no drift figure exists today (README §6) | 02 | ⬜ |
| T4 | **MEASUREMENT** | Gauss orders $N_\tau$, $N_{c_s\tau}$, $N_F$, $N_\rho$ audited at every production $k$ rather than at the one each was chosen on | 03 | ⬜ |
| T5 | **DECISION** | The decoupled constants settled by the user (§7 D1) and shipped with the measurement that chose each, in `config/defaults.py` | 04 | ⬜ |
| T6 | **PLUMBING** | Every `object_get` of a retuned target carries its own tolerance, with an `ast` guard that fails on an unclassified site | 04 | ⬜ |
| T7 | **HAND-OFF** | `QuadSourceIntegral` measured read-only and reported to `levin-refactor` / `qsi-phase-groups` | 05 | ⬜ |

---

## 3. Active and unresolved issues

None yet — the campaign has not started. Issues opened here must be added to
[`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) **in the same commit** (`CLAUDE.md`), with the
count corrected.

The issues this campaign was created *from* live on GkTk-remedial's board and stay there until it
closes them:

- `[12-tk-numeric-atol-largest-k-excursion]` — narrowed by that campaign's prompt 17; the `rtol`
  finding inside it is this campaign's D1.
- `[17-qcd-reference-not-converged]` — assigned to that campaign's prompt 18, and a hard
  precondition here (README §0.3).

---

## 4. Resolved issues

None yet.

---

## 5. Standing notes

1. **A number without its reference's drift beside it is not a measurement** (README §5 rule 5).
   Every figure quoted against a converged reference carries that reference's drift, and no
   conclusion is drawn from a signal that does not exceed it.
2. **The floors are not targets** (README §2 (e)). An agent reporting an accuracy below a declared
   floor has made an error, and it is a campaign-wide stop — not a caveat, not a footnote.
3. **Counts, not wall time** (README §2 (h)): this machine's elapsed times overstate by up to 53 %.
4. **Two of the five targets have no tolerance to converge** (README §2 (a)). An agent proposing to
   tighten the WKB tolerance has misread the tree; the knob there is Gauss order.
5. **`QuadSourceIntegral` is read-only here** (README §0.4). Touching it, `QuadSource.py`,
   `phase_groups.py` or `AdaptiveLevin/` is a stop.
