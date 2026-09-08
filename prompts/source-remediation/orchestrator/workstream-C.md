# Orchestrator prompt — Workstream C (prompts 07, 08, 09, 10)

You are orchestrating Workstream C of the source-remediation campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `main`). You do not write code yourself. You
dispatch one fresh-context subagent per prompt, review what it produced against fixed criteria,
and either continue or stop and report to the user.

Workstream C is the source time integral: the phase-group algebra (07), the region partition and
per-group Levin integration in `QuadSourceIntegral` (08), the error bound, `b` column and
tolerance plumbing (09), and the `main.py` wiring (10). Two of the four prompts run on Fable and
realise the design in README §6. This workstream has the campaign's one schema change (09) and its
one designed decision point (08's cost measurement, which decides 10's `QuadSourcePolicy`
question). Expect to stop at least once if that measurement comes out the unexpected way.

## What to read

Read these, in full, before dispatching anything:

- `prompts/source-remediation/README.md` — §1.1 (out of scope; in particular the Tier 2
  amplitude-and-phase output that no prompt here may pre-empt), §2 (a)–(c), §2.1 (why A2/A3 are
  split and why `QuadSourcePolicy` is not consumed), §4, §4.1 (your procedure and stop
  conditions), §5, §5.1, and **§6, the phase-group table**, which you will check 07 and 08 against.
- `prompts/source-remediation/IMPLEMENTATION_STATE.md` — the status board, §3 (active issues; you
  will see prompt 06's spline-residual observation, which sets a floor 08 must respect) and §5
  (standing notes; note 5 is the phase sign convention).
- `docs/spec-code-audit-2026-09.md` §0.2 A2, A4; §0.3 B5–B8, B11; §3 (author conventions).
- `docs/spec-code-audit/QI-report.md` in full — QI-2 (the measure that must be preserved), QI-4,
  QI-5, QI-6 (the gates 08 replaces), QI-7, QI-8 to QI-12, and §3 note 6 (the `bessel_phase`
  sign convention).
- The **"State handed to the next prompt"** sections and any deviations in
  `prompts/source-remediation/logs/05-tk-source-functions.md`,
  `logs/06-quadsource-regions.md` and `logs/04-triangle-filter.md`. These are the names and
  numbers 07–10 must build on; you will check the new logs against them.

Read `prompts/source-remediation/07-phase-group-algebra.md` only when you are about to dispatch
it, and likewise for 08, 09, 10. **Do not read prompts 11 and 12.** Do not read any log other than
04, 05, 06 and the ones 07–10 produce.

## Preconditions

Before the first dispatch, confirm: `git status` is clean; `IMPLEMENTATION_STATE.md` shows rows 05
and 06 as ✅ or ⚠️ and rows 07–10 as ⬜; `git log -1` is at or after prompt 06's commit. Before
dispatching **10**, additionally confirm row 04 is ✅ or ⚠️ (10 edits the same `main.py` stage and
must land on top of it) and rows 08 and 09 are ✅ or ⚠️. If 04 has not landed when you reach 10,
stop and report; do not run Workstream D yourself.

## Dispatching a prompt

For prompt NN, launch a subagent with **exactly** this context and nothing more:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/source-remediation/README.md`, `prompts/source-remediation/IMPLEMENTATION_STATE.md`,
> `docs/spec-code-audit-2026-09.md` §0–§3, then your prompt
> `prompts/source-remediation/NN-<name>.md`, the audit report sections it cites, and the earlier
> logs under `prompts/source-remediation/logs/` that your prompt tells you to read (their "State
> handed to the next prompt" sections are binding on you). Execute the prompt exactly. Do not read
> any other file under `prompts/source-remediation/`. Follow README §5 for the commit, the log and
> the board update. When you finish, reply with: the commit SHA, the **Result** line from your log,
> the "State handed to the next prompt" section verbatim, and a list of every deviation with its
> classification tag.

For **10**, append one of:
- if 08's log reports a §6 cost ratio ≤ 3×: "Prompt 08 measured the cost ratio at <value>; take the
  first branch of your §3."
- if the ratio was > 3× and the user has since decided: "Prompt 08 measured the cost ratio at
  <value>. The user's decision is: <the user's words, verbatim>. Take the branch of your §3 that
  matches it." (You will only be in this position if you stopped after 08 and were restarted.)

Model per prompt (README §3): **07 → Fable**, **08 → Fable**, **09 → Opus**, **10 → Opus**. Do not
substitute, and in particular do not downgrade 07 or 08.

Run them strictly in order 07 → 08 → 09 → 10. Each is a hard dependency of the next. Never in
parallel.

## Reviewing a prompt's result

When the subagent replies, check these five things yourself. Do not take the subagent's word for
any of them, and do not re-derive the physics — that is what the prompt's tests and sympy script are
for. Where a check below says "read", it means a targeted read of the named function, not a review
of the file.

1. **One new commit**, on top of the previous HEAD, whose message follows README §5 item 2, with
   the trailer naming the model you dispatched (`Claude Fable 5.1` for 07 and 08, `Claude Opus 5`
   for 09 and 10). Bodies: 07 states the number of groups per regime and that the module is pure
   and untested against a live pipeline; 08 states the regimes now handled, the oracle agreement
   achieved, and that the Green's-function-only Levin gate is gone; 09 lists the schema changes
   explicitly; 10 names the payload keys added and the `QuadSourcePolicy` decision.
2. **`prompts/source-remediation/logs/NN-<name>.md` exists**, is in that commit, follows the §5.1
   template, has a **Result** line, tags every deviation, and quotes numbers. Specifically:
   - **07:** a table of the measured Oracle-1 and Oracle-2 maxima per regime (seven oscillatory
     regimes, both $w$ values); a "State handed to the next prompt" with the `PhaseGroup` fields,
     the `build_phase_groups` signature, how a smooth factor is passed, and the **Oracle-2 floor**
     (08's acceptance arithmetic depends on it).
   - **08:** a table of `total` vs `analytic_rad` relative differences for every case (two $b$
     values × three shapes × three $z_{\rm resp}$); the arithmetic deriving the $10^{-5}$ threshold
     from the 05/06/07 floors; the **§6 cost ratio** with the raw timings and evaluation counts for
     (a) and (b); the `theta_deriv` decision tagged `IMPLEMENTATION CHOICE` with the with/without
     numbers; the tolerance-distribution scheme used per group; and a "State handed" naming the
     four payload keys, the redefinition of `WKB_quad`, the `LevinData` aggregate rule, and the
     ratio.
   - **09:** the error-bound ratio `|total − analytic|/abserr` per oracle case with an explanation
     for any case above ~1; the `analytic_rad` change and runtime under the pass-through
     tolerances (`1e-25` vs the old hardwired `1e-21`); whether `LEVIN_ABSERR`/`LEVIN_RELERR` were
     deleted or where they are still used; the `WKB_quad` column decision and the $\delta$ value,
     both tagged `IMPLEMENTATION CHOICE`.
   - **10:** which `QuadSourcePolicy` branch was taken and the ratio that decided it; where the
     structural payload test lives and why; how `_do_not_populate` was handled so the `Tk` objects
     in the payload are fully populated; a plain statement that the commit has not been exercised
     end to end.
3. **`IMPLEMENTATION_STATE.md`** in that commit has row NN filled in and the item-level rows
   updated (07: A4 1/3; 08: A4 2/3, A2 3/3; 09: B5, B6, B7, B8, B11; 10: A4 3/3), and the §3/§4/§5
   bookkeeping the prompt asks for: 08 opens a §3 issue "pipeline non-runnable until 10" and, if
   the ratio exceeded 3×, a cost issue; 09 adds the §5 note that `QuadSourceIntegral` tables must be
   rebuilt; 10 moves the non-runnable issue to §4.
4. **The prompt's tests pass when you run them.** Every prompt:
   `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .`, plus
   `-s CosmologyModels/tests` and `-s AdaptiveLevin/tests` for regressions. Time the
   `ComputeTargets` run; 08 says under ~3 min. Then per prompt:
   - **07:** `PYTHONPATH=. ./venv/bin/python ComputeTargets/tests/sympy_phase_groups.py` prints zero
     residuals for every coefficient. Confirm from the test file's assertion lines that the
     thresholds are $10^{-12}$ (Oracle 1), $10^{-8}$ (Oracle 2), $10^{-10}$ (phase composition at
     $|\Psi|\sim10^6$ and the `compute_analytic_G` form check), that all seven oscillatory regimes
     are exercised, that the boundary-consistency test exists, and that `(False, False, False)`
     is asserted to raise. Read the function that composes `theta_mod_2pi` and confirm it is a
     signed sum of the constituents' `theta_mod_2pi` values, not a reduction of summed `raw_theta`.
     Grep the module for `ray` and `Datastore` imports; there must be none.
   - **08:** also re-run `docs/spec-code-audit/scripts/QI_03_measure.py` against the new all-smooth
     path as the log describes and confirm the R28 measure is unchanged at the $10^{-15}$ level.
     Confirm from the test file that the regime-coverage assertion (every README §6 row with
     $n \ge 1$, including "$G$ smooth, both $T$ oscillatory") exists and passes, that the $10^{-10}$
     seam-split and old-regime regression tests exist, and that the missing-payload test asserts a
     `RuntimeError` naming the key. Grep `QuadSourceIntegral.py` for `Levin_z` and
     `LEVIN_MIN_PHASE_DIFF`: `Levin_z` may be *read* from the policy object but must not appear in a
     conditional that selects an integration route; the two constants must be defined-but-unused
     with the comment the prompt asks for.
   - **09:** `PYTHONPATH=. ./venv/bin/python -c "import Datastore.SQL.ObjectFactories.QuadSourceIntegral"`.
     Read the factory's table definition and confirm exactly these additions: `b` (Float,
     non-null), `total_abserr` (Float, nullable), `total_converged` (Boolean, nullable),
     `total_phase_limited` (Boolean, nullable); and that `WKB_quad` plus its six timing columns are
     either dropped or kept with a named reader in the log. Grep `QuadSourceIntegral.py` for the
     literals `1e-21` and `LEVIN_ABSERR`; the hardwired tolerances in `analytic_integral` must be
     gone.
   - **10:** `PYTHONPATH=. ./venv/bin/python -c "import ast; ast.parse(open('main.py').read())"`,
     and run the structural payload test where the log says it lives (`ComputeTargets/tests` is
     already covered; if it is `tests/test_main_plumbing.py` at the repository root, add
     `-s tests -t .`). If the ≤3× branch was taken, grep the tree for `LEVIN_MIN_2PI_CYCLES` and
     `LEVIN_MIN_PHASE_DIFF`; both must be gone. Confirm `MetadataConcepts/QuadSourcePolicy.py` still
     defines the same fields and that `GkSourcePolicyData` still persists `Levin_z`.
5. **`git diff HEAD~1 --stat` touches only** the allowed files plus the log and the board.
   - **07:** new `ComputeTargets/phase_groups.py`, `ComputeTargets/__init__.py`, new
     `ComputeTargets/tests/test_phase_groups.py`, new `ComputeTargets/tests/sympy_phase_groups.py`.
     The prompt also allows the fixture builders to be shared rather than duplicated, so a new
     `ComputeTargets/tests/fixtures.py` and a *move-only* edit of `test_tk_source_functions.py` are
     acceptable — check that 05's assertion lines are unchanged. `QuadSourceIntegral.py`,
     `AdaptiveLevin/`, `LiouvilleGreen/` untouched.
   - **08:** `ComputeTargets/QuadSourceIntegral.py`, new `ComputeTargets/tests/test_quadsource_integral.py`.
     `main.py`, everything under `Datastore/`, and `phase_groups.py` untouched. The return dict of
     `compute_QuadSource_integral` keeps its keys; `compute()`'s payload gains exactly `Tq_numeric`,
     `Tq_WKB`, `Tr_numeric`, `Tr_WKB`.
   - **09:** `ComputeTargets/QuadSourceIntegral.py`, `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`
     (the only `Datastore/` edit in the campaign), `ComputeTargets/tests/test_quadsource_integral.py`.
     `config/sharding.py` only if the log documents a referenced field name that changed (the
     prompt expects none — verify the claim if it is touched). Any other `Datastore/` file, anything
     in `LiouvilleGreen/`, or any `extract_*.py` script fails this check.
   - **10:** `main.py`, `ComputeTargets/QuadSourceIntegral.py` (payload contract only),
     `MetadataConcepts/QuadSourcePolicy.py` (the diff must contain only docstring/comment lines — no
     executable change), and the structural test file. Within `main.py`, hunks only in the
     QuadSourceIntegral stage (`build_QuadSourceIntegral_batch` and its lookup queues) and the
     policy-creation block; a hunk in any other stage fails.

## Continue or stop

**Continue to the next prompt** when all five checks pass and the log's Result is `COMPLETE`, or
`COMPLETE WITH DEVIATIONS` where every deviation is tagged `IMPLEMENTATION CHOICE` and its
justification is stated (you are judging that a reason is *given*, not that you agree with it).

**Stop and report to the user** — do not dispatch the next prompt, do not amend or revert
anything — when any of the following holds:

- Any of the five checks fails.
- Result is `PARTIAL` or `BLOCKED`.
- **07's sympy derivation disagrees with the formulas in its §3.** The prompt tells the agent to
  ship the derived formulas and tag the deviation `STRUCTURALLY REQUIRED`; that is correct
  behaviour by the agent, but README §4.1 sends any formula-level deviation to the user. Stop,
  and put the derived coefficients and the sympy residuals in your report so the user can check
  them. Likewise stop if the number of groups per regime differs from README §6 (1 / 1 / 2 / 2 / 4).
- Any `STRUCTURALLY REQUIRED` deviation in 08 concerns the measure (audit QI-2), the
  $(1+z_{\rm resp})$ prefactor, the linear summation of `abserr` across groups, or which regime a
  sub-interval is assigned; in 09, the meaning of `total`; in 10, the payload contract in a way that
  changes `compute_QuadSource_integral`'s behaviour.
- **08's §6 cost ratio exceeds 3×**, whatever the log's Result says. This is the campaign's
  designed decision point (README §4.1): report the ratio, the raw timings, and the two options
  prompt 10 §3 lays out, and wait for the user.
- 08's `total` vs `analytic_rad` misses $10^{-5}$ in any case, the $10^{-8}$ short-range `quad`
  comparison misses, or the regime-coverage assertion is missing a README §6 row. Any tolerance
  in any prompt's tests looser than the prompt states, even if the test passes.
- Any prompt other than 09 changes a schema; 09 changes the schema beyond the four columns named
  above and the `WKB_quad` drop.
- Any prompt stores, or adds a column or return key for, an amplitude-and-phase decomposition of
  the integral value (README §1.1). Per-group `value`/`abserr` in `metadata["WKB_Levin"]` is
  diagnostic and is what prompt 08 §5 asks for; a new column, or anything described as
  interpolable in $(\ln q, \ln r)$, is not.
- 08 still routes on `Levin_z` or `LEVIN_MIN_PHASE_DIFF`; 10 reinstates `WKB_quad_integral` of $G$
  alone as a route for any sub-interval; 10's branch does not match 08's measured ratio.
- 09 keeps the hardwired `atol=1e-21, rtol=1e-8` in `analytic_integral` (B6 must be a
  pass-through), or reports that `analytic_rad` changed by more than its own `abserr` under the
  pass-through tolerances (that means the oracle was tolerance-limited, which the user should
  know before 12 relies on it).
- 10 removes `QuadSourcePolicy`, changes its schema or signature, or removes `Levin_z` from
  `GkSourcePolicyData`.
- A deviation is tagged `UNINTENDED DRIFT` and was kept.
- The subagent touched `AdaptiveLevin/`, `LiouvilleGreen/`, `Datastore/` (other than 09's one
  factory), `ComputeTargets/OneLoopIntegral.py`, `thirdparty/`, any `extract_*.py`, or any file
  outside its allowed list.
- The subagent asks a question. Relay it verbatim; do not answer it yourself.

**Report but do not stop:** 09's B6 pass-through making the analytic branch more than 2× slower
with no value change beyond $10^{-10}$ (the prompt says keep the pass-through and record the cost);
08's test suite running well over ~3 min; 10's serialised-payload size measurement and its verdict
on the Ray object-store TODO. Put these in the first paragraph of your report.

Your report to the user, in either case, is: for each prompt run — its SHA, Result, the deviations
with tags, the quoted verification numbers (for 08 the oracle table, the §6 ratio and the
`theta_deriv` numbers even when you continued), the "State handed to the next prompt" section
verbatim, and which of the five checks you ran and their outcome. Then either "Workstream C
complete; the tree is at <SHA>; every physics-bearing defect in the campaign is now fixed;
`QuadSource` and `QuadSourceIntegral` rows in existing datastores must be rebuilt; ready for
Workstream E" or the specific stop condition that fired and what the user needs to decide. Do not
summarise the code changes in your own words; point at the logs.
