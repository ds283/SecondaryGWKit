# Prompt 08 — Refresh the stale agreement threshold in `test_wPerturbations.py`

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Workstream D — GATED.** Do not run this without the user's go-ahead (README §7 **D3**).
**Closes:** `[01-agreement-threshold-comment-predates-the-representation]`, opened by prompt 01
**Depends on:** **01** — its log holds the two measured figures you need.
**Recommended model:** **Sonnet** — one measurement, one comment, one constant, in one test file.
Prompt 01 has already taken the measurement; this prompt re-takes it, writes it down and tightens.

**Files you may touch:** `CosmologyModels/tests/test_wPerturbations.py` (**the module comment at
`:34-41`, the `AGREEMENT_RTOL` constant, and nothing else**), plus this campaign's log, board and
`docs/OPEN_ISSUES.md`.

**Do not touch:** any production file; any other test module; the `PureRadiationEOS` stand-in or
`lambdaCDM_gstar`, which three modules import
(`test_temperature_spline.py:36`, `test_T_z_representation.py:71`, `measure_rho_equality.py:21`);
any assertion body.

**Read first:** `CosmologyModels/tests/test_wPerturbations.py:1-50`;
`logs/01-equality-solve-characterisation.md` §4 item 3;
`RECONCILIATION.md` §9.3; and `LambdaCDM_GenericEOS.py:72-73`
(`DEFAULT_T_Z_SPLINE_SAMPLES`, `DEFAULT_T_Z_SPLINE_ORDER`).

---

## 1. What is wrong

`test_wPerturbations.py:34-41` says:

> The T(z) inversion inside LambdaCDM_GenericEOS is tabulated on a **500-point spline** in
> log(1+z) spanning [DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT, max_z]. That spline, not the physics, sets
> the floor on how exactly the GenericEOS model can reproduce the closed-form LambdaCDM
> expressions: measured **~1.3e-9** relative with max_z = 1e4 and **~4e-7** with the default
> max_z = 1e20. **1e-8** is therefore the tightest robust threshold at max_z = 1e4 …

with `AGREEMENT_RTOL = 1.0e-8`.

Since `qcd-background-audit` prompts 05 and 06 the representation is a **segmented entropy factor
over 3,000 nodes at order 5** — `DEFAULT_T_Z_SPLINE_SAMPLES = 3000`,
`DEFAULT_T_Z_SPLINE_ORDER = 5` — and what is tabulated is not $T$ at all. Both the description and
both figures predate the tree by two replacements. **Every assertion still passes**, and the
threshold is a ceiling, so nothing is wrong today; the comment simply describes a spline that no
longer exists and justifies a constant with numbers nobody can reproduce.

Same class as `[10-transfer-remedial-tolerance-comments-stale]`, which is the precedent for how
this is handled: measure, re-word, tighten only as far as the measurement supports.

## 2. The change

1. **Re-measure both figures** — the `PureRadiationEOS` model against `LambdaCDM`'s closed forms at
   `max_z = 1e4` and at `max_z = 1e20`, which is what the two quoted numbers are. Take the maximum
   relative departure over the same probe set the tests use, and say what that probe set is.
   Prompt 01's log has these; **re-take them anyway** — this prompt's whole content is that a number
   in a comment must be reproducible by the person reading it, and inheriting it would be the same
   fault in a new place.
2. **Re-word the comment** to describe what is actually there: a segmented entropy-factor
   interpolant at 3,000 nodes of order 5, in which the $(1+z)$ ramp is closed form and only the
   entropy factor is interpolated; the representation is segmented at the equation of state's
   branch temperatures. Name `qcd-background-audit` prompts 05 and 06 as what replaced it, and this
   campaign as what re-took the figures. **Keep the sentence that explains what the constant is
   *for*** — that the representation, not the physics, is the floor, and that this threshold is
   seven orders below the 0.69 relative discrepancy the A1 defect produced. That sentence is why
   the constant exists and it is still true.
3. **Tighten `AGREEMENT_RTOL` to what the measurement supports**, with margin. State the margin and
   why you chose it. The rule from `qcd-background-audit` README §5: a tolerance may only tighten
   or stay. **If the measurement says the threshold should *loosen*, that is a finding — stop and
   report it**, because it would mean a model agreement got worse across two campaigns that both
   claimed to improve the representation.
4. **`PureRadiationEOS` is an exact case.** $g_* = g_{s,*}$ constant, so $T(z) = T_{\rm CMB}(1+z)$
   exactly and the entropy factor $F$ is identically zero. If the interpolant reproduces a constant
   to the floor — which an order-5 B-spline through 3,000 equal values should — the departure may
   now be at the rounding floor rather than at any interpolation error, and the *shape* of the
   comment changes, not just its numbers. Say which regime you are in; that distinction is the most
   useful thing this prompt can record.

## 3. Acceptance

| Check | Threshold |
|---|---|
| Both figures | re-measured, quoted, with the command that reproduces them |
| The comment | describes the representation at this commit; every number in it is one you took |
| `AGREEMENT_RTOL` | tightened or unchanged, never loosened; the margin stated |
| Assertion bodies | **unchanged**; `git diff` shows only the comment and the constant |
| `CosmologyModels` suite | unchanged count, OK |
| `ComputeTargets` suite | **447 → 447**, OK |
| `black --check` | clean |

## 4. Stop conditions

- **The measurement says the threshold must loosen.** Stop and report.
- **An assertion fails at the tightened value.** Your margin was wrong; widen it to what the
  measurement supports and say so, or stop if the gap is large enough to be a finding.
- **You want to change an assertion, the stand-in, or another module.** You may not.

## 5. Deliverables

1. The re-worded comment and the refreshed constant.
2. `logs/08-refresh-agreement-threshold.md` per README §5.1, with both re-measured figures, the
   reproduction command, and §2 item 4's regime statement.
3. Board row 08, `[01-agreement-threshold-comment-predates-the-representation]` moved to §4 —
   `docs/OPEN_ISSUES.md` updated in the same commit, count and date corrected.
4. One commit, README §5 rule 2.
