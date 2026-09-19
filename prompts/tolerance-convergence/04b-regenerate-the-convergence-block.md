# Prompt 04b — regenerate the convergence block, and repair the two tests that read it

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** board item **T14**, and with it
`[01-convergence-block-has-a-separate-generator]` and
`[04-convergence-floor-used-as-a-test-threshold]`
**Depends on:** prompt 04, and on nothing else. Prompt 04 did the whole measurement and stopped
one step short of writing it down; this prompt writes it down and repairs what that exposed.
**Recommended model:** **Opus** — the measurement is done, but the test repair is a design
judgement about what two assertions are *for*, and the wrong repair looks exactly like the right
one.

**Files you may create or touch:**
`ComputeTargets/tests/wkb_reference_data.json` — the `convergence` key **only**, written by the
generator and never by hand;
`ComputeTargets/tests/test_background_tau.py` — the threshold of §3, and
`QCD_BREAK_POINT_ALIGNMENT_TOL`;
`ComputeTargets/tests/test_background_cs_tau_friction.py` — the threshold of §3, **and nothing
else in that module** (README §7 **D8**, which the user granted for this prompt after prompt 04
stopped on it);
`docs/gktk-remedial/residual_convergence.py` — only if the run needs it, and additively;
`docs/tolerance-convergence/ORDER-AUDIT.md` — a new §12 recording the landed run (README §5
rule 7: **additive**, never a rewrite of §§1–11, which were correct for the tree they were taken
on);
plus this campaign's log, board, the `qcd-background-audit` board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `config/defaults.py`; `main.py`; any file under `ComputeTargets/` that is not one
of the two test modules named above — in particular **not** `BackgroundModel.py`,
**not** `phase_residual.py` and **not** `ComputeTargets/tests/test_phase_residual.py`, which must
pass **unchanged**; any `Datastore/` file; the source grid, the band, `RESIDUAL_WKB_REGION_MARGIN`,
`BREAK_POINT_KIND` (README §0.5); `docs/gktk-remedial/RESIDUAL-CONVERGENCE.md`, which is another
campaign's published document.
**Do not change a production parameter** (README §5 rule 8). The three constants this prompt may
move — two thresholds and one alignment tolerance — all live in the test tree.

**Read first:** README §5 rules 5, 6, 7 and 9; §6.1's target rule, **rule 6 especially**; §7 **D5**
and **D8**; board item **T7** and the §3 issues
`[01-convergence-block-has-a-separate-generator]` and
`[04-convergence-floor-used-as-a-test-threshold]`;
**`docs/tolerance-convergence/ORDER-AUDIT.md` in full, and §§1, 9.2 and 9.4 twice** — it is prompt
04's measurement and this prompt does not re-derive any of it;
`prompts/tolerance-convergence/logs/04-order-governed-targets.md`, especially "State handed to the
next prompt"; the comment blocks at `ComputeTargets/tests/test_background_tau.py:56-110` and
`ComputeTargets/tests/test_background_cs_tau_friction.py:143`, which are four prompts of another
campaign explaining why they expect exactly this prompt to exist.

---

## 1. Why this prompt exists

Prompt 04 answered every question it was asked and then declined to write its answer into the
tree, because writing it makes two tests fail and one of the two modules was outside its grant.
**That was a scope boundary, not a finding.** The user has removed it (README §7 **D8**). Nothing
in prompt 04's measurement is in question here and **you must not re-take it**: the orders are 4,
the scheme is `branch`, the floors are what §9.2 records. Your job is to land it.

**What is stale in the tree right now.** `ComputeTargets/tests/wkb_reference_data.json`'s
`convergence` block was generated **2026-09-10**, against a $T(z)$ representation
`qcd-background-audit` prompts 04–06 replaced and a break-point set prompt 07 removed. Its
`recommended_scheme` names `branch+knots`, a knot split production has been unable to perform since
`integration_break_points` stopped returning knots. Every `*_GAUSS_ORDER` comment in the tree cites
it. Four separate comment blocks in `test_background_tau.py` say, in terms, that a later prompt
will re-run the generator and take a loosened constant back; this is that prompt, and the constant
comes back by ten orders rather than the one those comments expected.

**What landing it exposes, and what this prompt must actually decide.** Two tests bound the
production table's error by the *fixture's* error:

```
worst_production_error  <=  QCD_FLOOR_FACTOR * convergence...["json_vs_reference_max_rel"]
```

at `test_background_tau.py:333` ($\tau$) and `test_background_cs_tau_friction.py:620`
($c_s\tau$). The left-hand side is how far the production order-4 Gauss table sits from the JSON's
stored values. The right-hand side is how far those stored values sit from a converged adaptive
reference. **They are independent quantities**, and the test's own comment admits it — "a
floor-against-floor comparison, not an accuracy claim". The comparison held only while both sides
sat at ~2e-14 by coincidence.

Regenerating the block improves the right-hand side by 58× and 43× and leaves the left-hand side
unmoved to every digit (`ORDER-AUDIT.md` §1):

| | $\tau$ | $c_s\tau$ |
|---|---|---|
| floor, recorded 2026-09-10 | 1.878541e-14 | 1.886653e-14 |
| floor, regenerated | 3.223619e-16 | 4.354138e-16 |
| production error, unmoved | 2.254e-15 | 2.212e-15 |
| factor by which the present assertion then fails | 6.99 | 5.08 |

So the assertion has to change, and **§3 is about how**. Raising `QCD_FLOOR_FACTOR` past 7 is the
repair that looks right and is wrong: it keeps a comparison between two unrelated quantities alive
with a fresher number in it, and it will need raising again the next time the fixture improves.

## 2. Writing the block

### 2.1 The run

`docs/gktk-remedial/residual_convergence.py` is the **only** thing that may write the `convergence`
key. Prompt 04 left it able to do so and to write elsewhere for a dry run; use it as it stands.
`ORDER-AUDIT.md` §9 gives the two commands prompt 04 used. Run the generator against the fixture
this time, not a scratch path.

**Hand-editing the JSON is forbidden**, and the check that catches it is that `generated` and
`campaign` must both move. Record what they read before and after in your log.

### 2.2 What must survive

- **Only the `convergence` key may change.** `models`, `baselines`, `k_values`, `k_keys` and
  `rho_anchor_efolds_subh` are other campaigns' evidence and a diff that touches them is a stop.
- **The `branch+knots` key must survive and stay populated.** Two test modules index
  `convergence.schemes` and `convergence.models.*` by that name. It is demoted from candidate to
  **control** — production cannot execute the split — but it is not deleted. Prompt 04 measured
  what it now buys: nothing (`branch` 4.835e-16 against `branch+knots` 6.011e-16 on QCD's $\tau$ at
  order 4), which is the quadrature-level confirmation of `qcd-background-audit` prompt 07.
- **`decision.recommended_scheme` becomes `branch`.** If it comes back `branch+knots`, stop: that
  is a scheme production cannot perform and prompt 04 measured it as very slightly worse.
- **All four recommended orders must come back 4.** `test_phase_residual` pins
  `RHO_GAUSS_ORDER == decision.N_rho` and would catch only $N_\rho$; check all four yourself,
  against `ORDER-AUDIT.md` §9.1. **If any is not 4, stop and report** — the tree is then in a state
  no prompt in this campaign is chartered to repair, and it contradicts a measurement taken six
  days ago on the same tree.
- `rho_fixed_order_works_without_subdivision` goes `true` → `false`. Nothing reads it. Note it;
  do not act on it.

### 2.3 What not to do

Do **not** write the old floor into the new block to keep the tests green. It is the one repair
prompt 04 named as forbidden, and it would be a false statement about the reference.

## 3. The two thresholds

**The rule, and you apply it rather than choosing a number** (README §6.1's spirit, applied to a
test bound rather than a parameter): the bound on a production quantity must be a property of that
quantity, justified by a floor that has been measured, and it must not be the accuracy of the thing
it is compared against.

What that gives you, and the arithmetic must be in the comment:

- The quantity is the worst disagreement between the production order-4 Gauss table and the JSON's
  stored values: **2.254e-15** ($\tau$) and **2.212e-15** ($c_s\tau$), figures prompt 06 of
  `qcd-background-audit` recorded and prompt 04 reproduced to every digit.
- The floor under it is **double-precision accumulation over the grid**, which `ORDER-AUDIT.md`
  §§3.1 and 5 measure at **2.16e-16** ($\tau$) and **3.30e-16** ($c_s\tau$) — the level at which
  raising the Gauss order stops buying anything. The production figure sits about an order above
  it, which is what a cumulative over twenty decades costs.
- So the bound is an **absolute** one, stated in the module, with the measured floor and the
  measured production figure both named in the comment and the headroom stated as a factor.

Choose the number by that rule. State in the comment what it is, what it is a bound on, what the
floor under it is, where the floor was measured, and how much headroom the production figure has.
A bound with no headroom is a tripwire for the next unrelated change and a bound with four decades
of headroom asserts nothing; **prompt 04 §7's warning applies — a constant chosen to sit
comfortably above whatever was measured is the defect this campaign exists to undo.**

**Keep the history.** The comment blocks at `test_background_tau.py:56-110` and
`test_background_cs_tau_friction.py:143` are four prompts of another campaign recording why a
constant moved three times and predicting this repair. Append the resolution; do not delete the
record. Say plainly that the construction changed and why — that the two sides were never the same
quantity — because the next reader's question will be why the factor form was abandoned rather than
raised.

**`QCD_FLOOR_FACTOR` itself.** If your repair leaves the name with nothing to multiply, remove the
constant rather than leaving it defined and unused; if it survives in a different role, say so in
its comment. Either is acceptable; a constant left behind meaning something it no longer means is
not.

## 4. `QCD_BREAK_POINT_ALIGNMENT_TOL`

It is `1.5e-04` and it is measured to belong at just above **1.421085e-14**
(`ORDER-AUDIT.md` §9.4). Once the block is written, `test_qcd_break_points` compares the declared
break points against **the regenerated** branch boundaries, and the three offsets become 3.55e-15,
7.11e-15 and 1.421085e-14 — the floating-point agreement of two independent solves of the same
crossing. Take the constant there, with the three per-break figures in the comment and the
arithmetic that picks the headroom, and record that the whole of the former 1.418851e-04 was the
block's age, as that constant's comment has predicted since prompt 06.

This is ten orders of magnitude. Do not soften it because it looks like a large move; it is the
measurement, and the comment in the tree has been waiting for it through four prompts.

## 5. Acceptance

1. `ComputeTargets/tests/wkb_reference_data.json`'s `convergence` key is regenerated, `generated`
   and `campaign` have both moved, and **no other top-level key differs** from `HEAD~1`.
2. `decision` reads `4 4 4 4` and `recommended_scheme` reads `branch`; `branch+knots` is still a
   populated key under `schemes` and under each model that carried it.
3. Both threshold assertions are bounds on the production quantity, not multiples of the
   fixture's own agreement, each with its measured floor and headroom in the comment.
4. `QCD_BREAK_POINT_ALIGNMENT_TOL` is just above 1.421085e-14, with the three per-break figures
   behind it.
5. `ComputeTargets` **must not fall below 452** and should read **491**; `CosmologyModels` **39**;
   `ComputeTargets.tests.test_convergence_reference` **32**; and the three block readers —
   `test_background_tau`, `test_background_cs_tau_friction`, `test_phase_residual` — **41**,
   all green, with `test_phase_residual` **unedited**.
6. `black --check` clean on every `.py` in the diff.
7. `ORDER-AUDIT.md` gains a §12 recording the landed run: what the block says now, what the two
   thresholds became and why, and the alignment tolerance. §§1–11 are untouched, and §1's
   description of the stop stays as written — it was true of the tree it described.
8. Board row 04b, item **T14**, §3 and §4, the `qcd-background-audit` board entry for
   `[01-convergence-block-has-a-separate-generator]`, and `docs/OPEN_ISSUES.md` with its count and
   date corrected — all in the **same commit**.
9. The three published source-grid digests are unmoved: `3bef2c06`, `60a3205a`, `21ffc126`.

## 6. Stop conditions

Stop and report rather than working around any of these:

- **Any recommended order comes back other than 4**, or `recommended_scheme` comes back
  `branch+knots`.
- **A top-level key other than `convergence` differs.**
- **A reference fails to converge** in a way prompt 04 did not already record in `ORDER-AUDIT.md`
  §9.3. That section's 16 `NOT CONVERGED` cells are known, are the reference's order-32-vs-33 drift
  against cells whose reported differences are at the rounding floor, and carry no conclusion;
  anything beyond them is prompt 17's error and is a stop.
- **A test outside the three named modules changes its verdict.**
- **A source-grid digest moves.**
- **You find yourself wanting to edit a production module**, `test_phase_residual.py`, or any file
  not listed at the head of this prompt — including to make an assertion pass.
- **The repair of §3 cannot be made without changing what the tests measure.** If the honest
  conclusion is that one of them should test something else entirely, say so and stop; that is a
  charter question, not an implementation one.

## 7. The log

`logs/04b-regenerate-the-convergence-block.md`, on the campaign's template, classifying every
deviation. Beyond the template:

- The block's identity **before and after** — `generated`, `campaign`, the four orders, the scheme.
- The two thresholds: the old form, the new form, the number, and the arithmetic that chose it.
- `QCD_BREAK_POINT_ALIGNMENT_TOL`: old, new, the three per-break offsets, the headroom.
- **"State handed to the next prompt"** — prompt 05 is next and needs to know that the block is now
  current, that `QCD_FLOOR_FACTOR`'s role has changed or ended, and that
  `[01-convergence-block-has-a-separate-generator]` is closed. Prompt 04's five provenance fields
  (README §1.2) are already recorded in log 04 and **must not be restated here**; cite them.
