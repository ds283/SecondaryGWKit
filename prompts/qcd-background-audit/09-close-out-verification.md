# Prompt 09 — Close-out: the consumer tables under a corrected background

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Implements:** audit §9's last bullet — *"Any re-measurement of the consumer tables of
`docs/gktk-remedial-verification.md` §3.5 under a corrected background. That is work for the
campaign, and it is the real acceptance test for T1."*
**Depends on:** 08. **This prompt closes the ungated campaign.**
**Recommended model:** **Opus** — a verification prompt, and the one document a later reader will
actually consult.

**Files you may touch:** `docs/qcd-background-verification.md` (new),
`docs/qcd-background-audit/` (any script it needs), `docs/gktk-remedial-verification.md` —
**append a dated §9 and edit nothing at or above §8** — plus this campaign's log and board and
`docs/OPEN_ISSUES.md`.

**Do not touch:** **any production file**. This is a verification prompt and a verification prompt
may not change production code — that rule is what created `prompts/phase-representation` in the
first place. If you find a defect, record it as a §3 issue and **stop**; do not fix it.
Also: `docs/gktk-remedial/verify_production_path.py` must be run **unedited** (see §2 item 1), and
nothing above §8 of `docs/gktk-remedial-verification.md` may be rewritten (`CLAUDE.md`: verification
documents are additive; `prompts/phase-representation` already used §8).

**Read first:** `docs/gktk-remedial-verification.md` §3.5, §3.6, §3.7, §5 and §8;
`docs/gktk-remedial/verify_production_path.py`; the logs of prompts 01–08 of this campaign;
board entries `[02-verify-script-builds-its-own-Gk-consumer]` and
`[02-consumer-phi-below-the-storage-granularity]`; README §6.4 and §0.2.

---

## 1. What this prompt establishes, and what it cannot

Prompt 06 closed T1 against **prompt 01's guard** — an independent reference. This prompt asks the
different question: **what did the corrected background do to the consumers?**

README §0.2 is the frame, and it must be restated in the document you write. §3.5 and §3.6 score a
consumer against a producer built from the same background, so the $3.461\times10^{-8}$ this
campaign removed **cancels in them**. The consumer numbers are therefore **not required to
improve**. What must be established is:

- that **nothing got worse**;
- that every LambdaCDM row is **bit-identical**, because LambdaCDM has no `T(z)` spline
  (README §2 (g)) and any movement there means a prompt in this campaign leaked;
- that whatever *did* move on QCD has a stated cause;
- and that prompt 01's guard, which *is* sensitive to the background, now reads at the floor.

There is also a standing limitation to restate rather than to fix:
`[02-verify-script-builds-its-own-Gk-consumer]` — `verify_production_path.py` constructs
`PrimitivePhase` directly at `:557` and `:1173` instead of going through
`GkSourcePolicyData._build_phase`, so six of §3.5's twelve rows and both `theta_deriv` $G_k$ columns
are blind to whatever the production $G_k$ call site passes. **That issue is not this prompt's to
fix** (it is another campaign's file, and this is a verification prompt), but the document must say
which rows it makes blind, so a reader does not over-read them.

## 2. What to run, and what to write

1. **`docs/gktk-remedial/verify_production_path.py`, unedited.** Take the run at the campaign's
   **base** (`e8f746d`) and at `HEAD`, and diff them. ~55 s each, no Ray, no datastore. Every moved
   number gets a cause. A moved LambdaCDM number is a **stop**.

2. **`docs/qcd-background-audit/measure_T_z_representation.py`, unedited.** Its six sections are the
   audit's own reproduction. Run it at `HEAD` and show, section by section, what the campaign moved:
   §0 and §3 (the representation), §4 (H(z), conformal time, the phase), §5 (the break-point set),
   §6 (cost per call). Its §1 and §2 — the equation of state — must be **unchanged**, which is the
   demonstration that README §0.5's boundary held.

3. **The three suites**, with counts at base and at `HEAD`, and the wall time of each.

4. **`docs/qcd-background-verification.md`** — the campaign's verification document, written once
   and additive thereafter. Structure it on `docs/gktk-remedial-verification.md`: a statement of
   what tree it was taken on, what was run, and then one section per claim, each with its
   reproduction command. It must contain, at minimum:

   - **§1 The representation**, the audit §4 table's four rows re-measured at `HEAD`, with the
     prompt that moved each.
   - **§2 T1 closed**, prompt 01's guard before and after, with the equivalent phases at the three
     wavenumbers against their 1-ulp floors. **This is the campaign's headline and it goes first
     among the results.**
   - **§3 The consumers**, §3.5's twelve rows and §3.6's `theta_deriv` columns, base against `HEAD`,
     each movement with a cause, and the `[02-verify-script-builds-its-own-Gk-consumer]` caveat
     naming the blind rows.
   - **§4 The break-point set and what it cost**, 407 → 3, the `BackgroundModel` build cost, and
     prompt 08's drift table.
   - **§5 Cost**, per call and per build, against README §2 (c)'s figures.
   - **§6 What the campaign did not establish** — the standing caveats, in the shape of
     `docs/OPEN_ISSUES.md` §5. At minimum: that the verification geometry still does not reach
     production $x$ (§5 of the index); that `[00-consumer-anchoring-floor]` and
     `[02-consumer-phi-below-the-storage-granularity]` are untouched and are what limit
     `theta_deriv` at $k\ge10^7$; that the equation of state's branch joins are unrepaired
     (`[00-eos-branch-joins-do-not-match]`); and that a pre-campaign datastore is refused, not
     migrated.

5. **A dated §9 appended to `docs/gktk-remedial-verification.md`**, short, saying what moved in its
   own §3.5/§3.6 tables and pointing at the new document. **Nothing at or above §8 is edited.**
   Say explicitly that its §3.5 figures were correct for the tree they were taken on and that the
   background underneath them has since changed — which is precisely the fact its §6 could not have
   seen.

## 3. Acceptance

| Quantity | Requirement |
|---|---|
| Every LambdaCDM row of §3.5, §3.6, §3.7 | **bit-identical** to the base run |
| `RadiationModel` and stand-in figures | **bit-identical** |
| Prompt 01's $\int\mathrm{d}z/H$ guard | **≤ 1e-15**, phases below the three floors |
| Every moved QCD number | **has a stated cause** |
| Audit script §1 and §2 (the equation of state) | **unchanged** |
| Three suite counts | **no fall** from base |

**If a LambdaCDM number moved, stop**, and report which prompt is the suspect — the campaign's
logs record which prompts touched the shared path.

## 4. Log and commit

Log to `logs/09-close-out-verification.md` per README §5.1. Update this campaign's board: every
row 01–09 to its final state, §2's mechanism table, and §4 with every issue the campaign closed.
Update `docs/OPEN_ISSUES.md` — the count, the date, the `Boards` line (this campaign's board is
added there by the planning commit; confirm it is right), and §1.7, which should now hold only what
workstream D still owns.

Set the board's header to **CLOSED at 9 / 12 (workstream D gated)** or to whatever the true state
is, and say in one sentence what a reader should conclude.

Commit subject, or something equally specific:
`Verify the QCD background remediation against both consumers`
