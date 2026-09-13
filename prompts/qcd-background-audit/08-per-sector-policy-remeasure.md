# Prompt 08 — Re-take prompt 19's measurement: does the $T_k$ sector still need `BREAK_POINT_ALL`?

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design facts:** README §2 (f) — **the whole prompt is that one fact** · **Decision:** README §7
**D5** — this prompt **reports and does not decide**
**Depends on:** 07. **Blocks 09.**
**Recommended model:** **Opus** — a measurement whose answer may be "stop and ask the user", and
which must be able to tell the difference between that and "nothing to do".

**Files you may touch:** `docs/qcd-background-audit/` (a measurement script and its output),
`ComputeTargets/TkNumericIntegration.py` and `ComputeTargets/GkNumericIntegration.py` — **only**
their `BREAK_POINT_KIND` comment blocks, and their *values* **only** if §3 says so and the
orchestrator has relayed the user's decision — plus this campaign's log and board and
`docs/OPEN_ISSUES.md`.

**Do not touch:** `numeric_with_phase_cut.py` (prompt 07 has already corrected its text; its logic
is out of bounds here as everywhere), `Datastore/SQL/ObjectFactories/{Gk,Tk}NumericIntegration.py`,
any tolerance, any Gauss order, `main.py`.

**Read first:** README §2 (f); `prompts/GkTk-remedial/19-per-sector-break-point-policy.md` §4 (the
acceptance test this prompt re-takes) and `prompts/GkTk-remedial/logs/19-per-sector-break-point-policy.md`;
`prompts/GkTk-remedial/20-key-the-break-point-policy.md` and its log — `BREAK_POINT_KIND` is in a
datastore lookup key because of it; `ComputeTargets/TkNumericIntegration.py:98-118` and
`GkNumericIntegration.py:95-107`; `docs/gktk-remedial/tk_numeric_atol_sweep.py` §§ around
`SECTOR_POLICY` (`:1746-1760`), which is the harness prompt 19 used and the cheapest thing to
adapt; `docs/OPEN_ISSUES.md` §1.5's cost note (a QCD $T_k$ numeric object is ~31.5k RHS evaluations
under `BREAK_POINT_ALL` against ~9.8k under jumps alone).

---

## 1. Why this prompt exists

`TkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_ALL` is **not a default**. `GkTk-remedial`
prompt 19 chose it on measurement: with the jumps alone, **3 of 50** QCD $T_k$ wavenumbers missed
the 3.4e-08 reference-convergence criterion, worst **1.97e-07**; with the 404 knots, **4.65e-09 or
better**. That is a factor of forty, and it is why the $T_k$ sector pays ~31.5k right-hand-side
evaluations per object instead of ~9.8k.

Prompt 07 has just removed those 404 knots. **So `BREAK_POINT_ALL` now returns 3 points where it
returned 407, and prompt 19's justification has silently evaporated.** There are exactly three
possible states of the world and this prompt must determine which:

- **(a) The knots were a proxy for the representation's own defect.** With an accurate background
  the $T_k$ sector converges at all 50 wavenumbers under either policy. The per-sector distinction
  becomes vestigial but **the constants stay** — they are in a lookup key, they cost nothing, and a
  future equation of state may declare many break points. Record, comment, move on.
- **(b) The $T_k$ sector still needs more split points than the cosmology declares.** Then either
  the new representation must declare its knots after all — which reopens G1 and narrows this
  campaign's claim — or `BREAK_POINT_ALL` was never the right mechanism. **Stop and ask.**
- **(c) The $G_k$ sector has changed too.** It converged at all 50 on all three models under
  `BREAK_POINT_DISCONTINUITY` (worst 8.41e-09). If that is no longer true, something in prompts
  04–07 did more than it was supposed to. **Stop.**

## 2. What to measure

Re-take **prompt 19 §4's acceptance test**, unchanged in form, on the tree as prompt 07 left it:

1. **QCD $T_k$, all 50 production wavenumbers**, reference-convergence drift under
   `BREAK_POINT_ALL` (now 3 points) and under `BREAK_POINT_DISCONTINUITY` (2), against the
   **3.4e-08** criterion. Report the worst of each and the count above the criterion.
2. **QCD $G_k$, all 50**, same, under both.
3. **`RadiationModel` and `LambdaCDMModel`, both sectors** — the control. These declare no break
   points, so both policies must be **bit-identical** and unchanged from prompt 19's figures. A
   moved number here is a stop.
4. **Cost.** Right-hand-side evaluations and wall time per object, per sector, per policy. Prompt
   19's figures to beat are ~31.5k ($T_k$ under `ALL`) and ~9.8k (under jumps); $G_k$ ~13.3k.
   Quantify what prompt 07 changed.

**Budget.** This is the most expensive prompt in the campaign: prompt 19's sweep was 50
wavenumbers × 2 sectors × 2 policies × 3 models, plus references. Adapt
`docs/gktk-remedial/tk_numeric_atol_sweep.py` rather than writing a new harness — **copy it into
`docs/qcd-background-audit/` rather than editing another campaign's file**, on the precedent of
`GkTk-remedial` prompt 13 and `[13-scoped-run-driver-k-grid-literal]`. Report the total wall time.
If the full matrix cannot be run in a reasonable time, run the QCD half in full and a reduced
control, **say exactly what you ran and what you did not**, and do not present a partial matrix as
a complete one.

## 3. What to change

**In state (a): nothing but comments.** Rewrite both `BREAK_POINT_KIND` comment blocks to carry
*this* prompt's measurement in place of prompt 19's — the old numbers stay in prompt 19's log,
which is their record. Say explicitly that the justification has changed: the policy is retained
not because it is needed on today's QCD cosmology but because it costs nothing now that the set is
3 points, and because the mechanism must exist for an equation of state that declares more. **Do
not change either value.**

**In state (b) or (c): stop.** Write the log with `Result: BLOCKED` or
`COMPLETE WITH DEVIATIONS` as appropriate, state the measurement, state the options, and **make no
production change**. `BREAK_POINT_KIND` is in a datastore lookup key (`GkTk-remedial` prompt 20), so
moving it has a regeneration attached and is README §7 D5 — the user's.

**In every state:** do not bump `T_Z_REPRESENTATION_VERSION` unless you changed a value. Say in the
log which state you found and what the version is.

## 4. Acceptance

| Quantity | Prompt 19's figure | This prompt |
|---|---|---|
| QCD $T_k$ worst drift, `BREAK_POINT_ALL` | 8.72e-09 (with 404 knots) | **measured** |
| QCD $T_k$ worst drift, `BREAK_POINT_DISCONTINUITY` | 1.97e-07, 3 of 50 above criterion | **measured** |
| QCD $G_k$ worst drift, `BREAK_POINT_DISCONTINUITY` | 8.41e-09 | **≤ 3.4e-08**, else stop |
| Radiation / LambdaCDM, both sectors, both policies | — | **bit-identical**, else stop |
| $T_k$ RHS evaluations per QCD object | ~31.5k | **measured**, both policies |

All three suites pass; counts quoted and not falling. `black` clean.

## 5. Log and commit

Log to `logs/08-per-sector-policy-remeasure.md` per README §5.1. It must state **which of the three
states (a)/(b)/(c) was found**, in those words, with the numbers that establish it; the full drift
table; the cost table; and exactly what was and was not run if the matrix was reduced. If state (b),
the log's "State handed to the next prompt" is the user's decision packet: the options, their costs,
and what each does to the datastore.

Commit subject, depending on the state found:
`Re-measure the numeric break-point policy on the corrected background`
