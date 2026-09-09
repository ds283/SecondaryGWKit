# Orchestrator prompts — Bessel amplitude and phase campaign

Four ready-to-use orchestrator prompts, one per workstream. Each restates the procedure and stop
conditions of [`../README.md`](../README.md) §4.3 with the prompt-specific checks — test commands,
allowed files, thresholds — filled in.

| Workstream | Prompts | Prompt file | Character |
|---|---|---|---|
| A — measurement | 01, 02 | [`workstream-A.md`](workstream-A.md) | No production code changes. Safe to run unsupervised; the output is what every later workstream is scored against |
| B — the construction | 03, 04, 05 | [`workstream-B.md`](workstream-B.md) | The campaign's substance. **Review closely**; 04 and 05 are the two prompts where a wrong choice propagates |
| C — evaluation and consumers | 06, 07 | [`workstream-C.md`](workstream-C.md) | Plumbing plus three judgement calls, two of which are stop conditions by design |
| D — revalidation and close-out | 08, 09 | [`workstream-D.md`](workstream-D.md) | Tests and documentation. 08 may surface findings that are not this campaign's to fix |

## Running the campaign

The workstreams are **strictly sequential**: A → B → C → D. This is a property of the work, not a
scheduling choice — every prompt from 03 onward consumes an interface the previous one defines, and
01 defines the references all of them are scored against (`../README.md` §4). Do not parallelise.

Start with:

> Read `prompts/bessel-remedial/orchestrator/workstream-A.md` and follow it.

and move on only when that prompt's own completion criterion is met and the board shows its rows
✅ or ⚠️.

## Rules that bind every orchestrator prompt

1. **You do not write code.** You dispatch one fresh-context subagent per prompt, review what it
   produced against fixed criteria, and either continue or stop and report to the user.
2. **One prompt, one subagent, one commit.** Never dispatch two prompts to one agent, and never let
   an agent amend or squash across prompts. The commit boundary is the rollback boundary.
3. **Do not re-derive the work.** Your five checks (`../README.md` §4.3) are about whether the
   prompt's own tests pass when *you* run them, whether the log classifies its deviations, and
   whether the diff stayed inside its allowed files. Judging the physics is what the tests are for.
4. **Give each subagent only its own prompt.** It reads `../README.md`, `../RECONCILIATION.md`,
   `../IMPLEMENTATION_STATE.md` and its own prompt file — plus the specific "State handed to the
   next prompt" sections its prompt names. It must not read the other prompts, and it reads
   `../DRAFT-PLAN.md` only through the sections its own prompt cites.
5. **Use the model the tables specify.** Do not substitute a cheaper one to save time. Prompts 04
   and 05 in particular are marked as the campaign's hardest.
6. **Relay questions verbatim.** If a subagent asks something, do not answer it yourself.
7. **Stop rather than repair.** If a check fails, do not fix it, do not revert it, and do not
   dispatch a follow-up agent to patch it. Report to the user with the specific check that failed
   and what the log says.

## The campaign-wide stop conditions

These apply in every workstream, in addition to each prompt's own. Repeated from
`../README.md` §4.3 so an orchestrator reading only this folder still has them.

- A log's **Result** is `PARTIAL` or `BLOCKED`.
- A deviation tagged `STRUCTURALLY REQUIRED` touches any of: the zero-point
  \(c_\nu=\pi/4-\pi\nu/2\); the convention \(J=A\sin\theta\), \(Y=-A\cos\theta\); the Wronskian
  \(\theta'=a^{-2}\); the exactness of the scaled-Hankel algebra; which series supplies the tail;
  the two-region structure itself; or the choice of \(\theta'=e^{-2\ell}\) over \(1+r_u/x\).
- A deviation tagged `UNINTENDED DRIFT` was kept rather than reverted.
- Any test the prompt says must pass fails, or a numerical acceptance threshold is missed **even
  narrowly**.
- `test_phase_derivative`'s \(10^{-6}\) contract (`LiouvilleGreen/tests/test_bessel_phase.py:139`)
  fails or is loosened, at any point from prompt 05 onward.
- An agent proposes to keep the phase ODE, reintroduce `phase_spline` inside `bessel_phase`, make
  the tail optional or deferred, or sample `hankel1e` above \(x_\star\).
- An agent proposes to widen the supported \((\nu,x_{\max})\) domain or extend below
  \(\sqrt{\nu^2-\tfrac14}\).
- An agent touches `ComputeTargets/QuadSourceIntegral.py`, `ComputeTargets/QuadSource.py`,
  `ComputeTargets/phase_groups.py`, `ComputeTargets/TkSourceFunctions.py`, `Datastore/`,
  `AdaptiveLevin/`, `LiouvilleGreen/phase_spline.py`, `thirdparty/`, or any `extract_*.py`.
- An agent writes "the residual never exceeds a cycle", or any equivalent, into code, a docstring
  or a commit message. It is false above \(\nu\approx630\)
  (`../IMPLEMENTATION_STATE.md` §5 note 2) and it is the campaign's most likely
  false-statement-in-the-codebase failure, because `../DRAFT-PLAN.md` §4.6 says it.
- An agent claims a construction *speed* improvement as the campaign's motivation
  (`../IMPLEMENTATION_STATE.md` §5 note 3).
- A rebase against the in-flight `source-remediation` campaign conflicts
  (`../README.md` §4.2).

## Reporting

Your report to the user, per prompt run: the SHA, the **Result** line, every deviation with its
tag, the quoted verification numbers, the "State handed to the next prompt" section verbatim, and
which of the five checks you ran with their outcome. Then either a one-line statement of what the
tree is now ready for, or the specific stop condition that fired and what the user needs to decide.

**Do not summarise the code changes in your own words.** Point at the logs. The logs exist so that a
later reader does not have to trust a summary.
