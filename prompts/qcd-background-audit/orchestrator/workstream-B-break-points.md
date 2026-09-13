# Orchestrator prompt — workstream B, the break-point set (prompts 07–08)

You are orchestrating workstream B of the QCD background campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `qcd-background-audit`). **You do not write
code.** Workstream A must be complete — prompts 01–06 landed, `T_Z_REPRESENTATION_VERSION` at 4, no
row missed.

Two prompts, and they are asymmetric. **Prompt 07 changes what every quadrature and every ODE in
the tree is told about the cosmology**: `BREAK_POINT_ALL` goes from 407 points to 3. **Prompt 08
then checks whether that was affordable**, by re-taking the measurement that put
`TkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_ALL` there in the first place.

**Prompt 08 may legitimately end in "stop and ask", and that is a success, not a failure.** Read
campaign README §2 (f) and §7 D5 before you dispatch anything.

## What to read

`../README.md` §0.2, §2 (f), §4, §6.3, §7 D5. `../IMPLEMENTATION_STATE.md` §5 note 7.
`docs/qcd-background-audit-2026-09.md` §7. `prompts/GkTk-remedial/19-per-sector-break-point-policy.md`
§4 and `prompts/GkTk-remedial/logs/19-per-sector-break-point-policy.md` — **the measurement prompt
08 re-takes**. `prompts/GkTk-remedial/20-key-the-break-point-policy.md` — why `BREAK_POINT_KIND` is
in a lookup key. `prompts/phase-representation/logs/02-primitive-phase-break-point-knots.md` — what
the 407 points blocked. `orchestrator/README.md`. `CLAUDE.md`.

## Preconditions

`git status` clean; workstream A's completion criterion met. **Baselines, taken at workstream A's
final commit:**

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py \
  > /tmp/qcdbg-audit-preB.txt 2>&1
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py \
  > /tmp/qcdbg-verify-preB.txt 2>&1
```

Record prompt 05's and 06's reported `BREAK_POINT_ALL` counts from their logs. Prompt 07's job is to
take that number to 3, and you need to know what it was.

## Reviewing prompt 07 — the break-point set

1. One commit; log per §5.1; this campaign's board, the `GkTk-remedial` board entry
   `[19-cosmologymodels-docstrings-predate-per-sector-policy]` moved to that board's §4, and
   `docs/OPEN_ISSUES.md` — all in the same commit, count and date correct.
2. **`T_Z_REPRESENTATION_VERSION` bumped to 5.**
3. **The counts.** Re-run the audit script and diff against `/tmp/qcdbg-audit-preB.txt`. Its §5
   table must read **3** for `BREAK_POINT_ALL`, **2** for `BREAK_POINT_DISCONTINUITY`, and **0**
   knots. Nothing in §1 or §2 (the equation of state) may have moved — if it has, something touched
   `QCD_EOS.py` and that is always a stop.
4. **The removal was measured, not assumed.** §2 item 1 of the prompt requires a number for the
   residual non-smoothness at a knot of the new representation, against the $10^{-4}$-level defect
   the old lattice carried. **If the log asserts "the knots are now smooth" without a number,
   stop.** That claim is the entire justification for the change.
5. **The tables did not lose accuracy.** QCD `tau`, `cs_tau` and `friction_F` against the
   regenerated references, before and after, all three quoted. This is what establishes that the
   404 splits were buying nothing — a *worse* table here is the outcome that sends this back to the
   user.
6. **The build got cheaper, or the log says it did not.** Wall time and integrand evaluations for
   the QCD `BackgroundModel` cumulative tables. An omitted row is a stop; a "no speed-up" row
   honestly reported is not.
7. **The knot vector constructs.** Prompt 07 §3 item 5 requires an assertion that a
   multiplicity-`spline_order` knot vector at the 3 break points constructs on all six production
   grids that `prompts/phase-representation` prompt 02 measured as singular. Check the test exists
   and passes. **It must not build a consumer spline** — that is prompt 10, and a `primitive_phase.py`
   hunk in this diff is a stop.
8. **`BREAK_POINT_KIND` was not touched**, in either integration class. Verify:
   ```bash
   git diff HEAD~1 -- ComputeTargets/TkNumericIntegration.py ComputeTargets/GkNumericIntegration.py
   ```
   must be empty. Changing either here, without prompt 08's measurement, is a campaign stop
   condition (README §2 (f)).
9. **`numeric_with_phase_cut.py` is docstrings and comments only.** `git diff HEAD~1 --
   Quadrature/integrators/numeric_with_phase_cut.py` must contain no executable change — in
   particular `_separated_boundaries`, `BREAK_POINT_STANDOFF` and `_solve_segmented` are untouched.
10. **Bit-identity.** Diff `verify_production_path.py` against `/tmp/qcdbg-verify-preB.txt`: every
    LambdaCDM row unchanged. QCD rows may move; each needs a cause in the log.
11. `git diff HEAD~1 --stat`: allowed files only. **No `main.py`, no `Datastore/`, no compute
    target.**

## Reviewing prompt 08 — the policy re-measurement

1. One commit; log per §5.1; board and `docs/OPEN_ISSUES.md` in it.
2. **The log names the state it found — (a), (b) or (c) — in those words** (prompt 08 §1), with the
   numbers that establish it. A log that reports a drift table without saying which state it
   implies has not done the prompt's job: **stop**.
3. **The control is bit-identical.** `RadiationModel` and `LambdaCDMModel`, both sectors, both
   policies, unchanged from prompt 19's figures. A moved control number means something in
   prompts 04–07 leaked into a path it should not have: **stop, and say which prompt is the
   suspect**.
4. **$G_k$ still converges** at all 50 QCD wavenumbers under `BREAK_POINT_DISCONTINUITY`, worst
   ≤ 3.4e-08 (prompt 19's figure was 8.41e-09). If not: **stop**.
5. **What was actually run.** This is the campaign's most expensive prompt and the prompt permits a
   reduced matrix. The log must say **exactly** what was run and what was not. A reduced matrix
   presented as complete is a stop; a reduced matrix declared as reduced is fine — record it and
   pass it to prompt 09.
6. **State (a) — nothing but comments changed.** `git diff HEAD~1 --
   ComputeTargets/TkNumericIntegration.py ComputeTargets/GkNumericIntegration.py` shows comment
   hunks only, and both `BREAK_POINT_KIND` **values are unchanged**. The rewritten comments carry
   *this* prompt's measurement, not prompt 19's, and say that the justification has changed.
7. **State (b) or (c) — no production change at all, and you stop.** Relay the log's decision packet
   to the user verbatim: the options, their costs, and what each does to the datastore.
   `BREAK_POINT_KIND` is in a lookup key, so a change has a regeneration attached, and it is
   README §7 D5 — the user's, not yours and not the agent's.
8. **The harness was copied, not edited in place.** `docs/gktk-remedial/tk_numeric_atol_sweep.py`
   must not be in the diff (`[13-scoped-run-driver-k-grid-literal]`'s lesson, and the reason
   `GkTk-remedial` prompt 13 copied rather than edited).

## Continue or stop

Continue to workstream C when both logs are `COMPLETE`-class, the counts are 3 and 2, the control
is bit-identical, and prompt 08 found **state (a)**.

**Stop and report** on state (b) or (c); on an unmeasured knot-removal claim; on a `BREAK_POINT_KIND`
value moved without the user's decision; on a control number that moved; or on any campaign
README §4 condition.

**Relay every subagent question verbatim.**

## Completion criterion

Rows 07–08 ✅/⚠️; two logs; `T_Z_REPRESENTATION_VERSION` at 5; the audit script's §5 reading 3 / 2
with 0 knots; `[19-cosmologymodels-docstrings-predate-per-sector-policy]` on the `GkTk-remedial`
board's §4; `[13-consumer-spline-crosses-eos-break-points]` **narrowed and still open**, with its
blocker recorded as removed.

Report: the 407 → 3 collapse; what the `BackgroundModel` build cost did; the three break points to
17 digits; prompt 08's state and its drift table; and — explicitly — **whether the $T_k$ sector's
`BREAK_POINT_ALL` policy still has a justification, or now exists only as a mechanism for a future
equation of state.**
