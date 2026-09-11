# Orchestrator prompts — Gk/Tk WKB phase remedial campaign

Six ready-to-use orchestrator prompts, one per workstream, plus one for a later interim prompt.
Each restates the procedure and stop conditions of [`../README.md`](../README.md) §4.3 with the
prompt-specific checks — test commands, allowed files, thresholds — filled in.

| Workstream | Prompts | Prompt file | Character |
|---|---|---|---|
| A — measurement and prototypes | 01, 02 | [`workstream-A.md`](workstream-A.md) | No production code. Safe unsupervised; two numbers it produces (throughput, the QCD fallback flag) can change the design |
| B — the primitives | 03, 04 | [`workstream-B.md`](workstream-B.md) | The foundation and a schema change. **Confirm D1 with the user first.** Review 03 closely |
| C — the producers | 05, 06, 07 | [`workstream-C.md`](workstream-C.md) | The campaign's substance; 06 is the production point of no return for $G_k$ |
| D — the consumers | 08, 09, 10 | [`workstream-D.md`](workstream-D.md) | The $h^4x/384$ cure. **Settle the `transfer-remedial` overlap before 10** |
| — interim: `PrimitivePhase` rate | 15 | [`prompt-15.md`](prompt-15.md) | Not a workstream — a single prompt closing `[10-primitive-phase-leading-rate-is-hardcoded]`, dispatched after D closed and before E, at the user's request |
| E — the numeric region | 11, 12 | [`workstream-E.md`](workstream-E.md) | Independent; can run at any point. **Always stops after 11** to put D2 to the user |
| — interim: D2 print policy | 16 | [`prompt-16.md`](prompt-16.md) | Not a workstream — the follow-up §7 D2 anticipated, enacting the user's choice of option (ii). Runs between 11 and 12 |
| F — verification | 13 | [`workstream-F.md`](workstream-F.md) | Measurements and documents; may surface findings that are not this campaign's to fix |

## Running the campaign

Recommended order **A → B → C → D → 15 → E (11 → 16 → 12) → F** (`../README.md` §4). E is independent and may be
run first as a warm-up or interleaved with any of A–D and 15; nothing else may be reordered —
every prompt from 03 onward consumes an interface the previous one defines.

Start with:

> Read `prompts/GkTk-remedial/orchestrator/workstream-A.md` and follow it.

and move on only when that prompt's completion criterion is met and the board shows its rows ✅
or ⚠️.

## Rules that bind every orchestrator prompt

1. **You do not write code.** You dispatch one fresh-context subagent per prompt, review what it
   produced against fixed criteria, and either continue or stop and report to the user.
2. **One prompt, one subagent, one commit.** Never dispatch two prompts to one agent; never let an
   agent amend or squash across prompts.
3. **Do not re-derive the work.** Your five checks (`../README.md` §4.3) are about whether the
   prompt's own tests pass when *you* run them, whether the log classifies its deviations, whether
   the board and `docs/OPEN_ISSUES.md` were updated together, and whether the diff stayed inside
   its allowed files.
4. **Give each subagent only its own prompt.** It reads `../README.md`, `../RECONCILIATION.md`,
   `../IMPLEMENTATION_STATE.md`, its prompt file, the review sections its prompt cites, and the
   "State handed to the next prompt" sections of the logs its prompt names. Not the other prompts.
5. **Use the model the tables specify.** Fable for 03, 06, 09 (Opus if Fable is unavailable, and
   then review those three most closely); do not substitute a cheaper model elsewhere.
6. **Relay questions verbatim.** If a subagent asks something, do not answer it yourself.
7. **Stop rather than repair.** If a check fails, do not fix it, revert it, or dispatch a follow-up
   agent to patch it. Report the specific check and what the log says.
8. **The orchestrator commits too, so an agent must never assume `HEAD` is its own.** Between
   prompts you land planning and correction commits on the same branch — Workstream C landed seven.
   Rule 2 forbids amending across *prompts*; this forbids amending across *agents*, which rule 2
   does not cover because an orchestration commit is not a prompt. Every dispatch tells the
   subagent: before any `git commit --amend`, `git reset` or `git rebase`, check that `HEAD` is the
   commit you created, and never rewrite one you did not author. An agent that finds it already has
   must say so and stop — a `reset --hard` to "undo" it can silently drop a commit made in the
   meantime, and reconstructing is your call, not the agent's. (Prompt 07's agent amended an
   orchestration commit, caught it, and reset; nothing was lost, but only because of the order in
   which the two commits fell.)

## The campaign-wide stop conditions

These apply in every workstream, in addition to each prompt's own (`../README.md` §4.3):

- A log's **Result** is `PARTIAL` or `BLOCKED`.
- A deviation tagged `STRUCTURALLY REQUIRED` touches any of the design facts `../README.md` §2
  (a) the split; (c) the double-double table, the never-a-difference-of-pointwise rule, the
  `delta` sign convention; (e) the negative-remainder convention or the never-rebased per-sample
  offset; (f) the retained `*_omegaEff_sq` values and $(B,\delta)$ algebra; (g) the consumer
  decomposition; (h) preserving the `has_unresolved_osc` warning.
- A deviation tagged `UNINTENDED DRIFT` was kept.
- Any test the prompt says must pass fails, or a numerical acceptance threshold in the prompt or
  `../README.md` §6 is missed **even narrowly**.
- An agent proposes to keep any part of the phase ODE, to spline $\tau$, to spline the full phase
  in a consumer, to reintroduce chunking, to move the hand-over window or the
  $\sqrt{z_{e3}z_{e4}}$ limit, or to raise the LG order.
- An agent proposes to change a `*_omegaEff_sq` return value, a `GkWKBValue`/`TkWKBValue`
  column, or the `GkSource` rectifier's logic.
- An agent touches a `transfer-remedial` file (`../README.md` §0.2), `AdaptiveLevin/`,
  `QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `thirdparty/`, or any `extract_*.py`.
- An agent writes into code, a docstring or a commit message that an accuracy below a floor of
  `../IMPLEMENTATION_STATE.md` §5 note 2 was achieved.
- A `main.py` hunk outside the ones the prompt names.

## Reporting

Per prompt run: the SHA, the **Result** line, every deviation with its tag, the quoted verification
numbers, the "State handed to the next prompt" section verbatim, and which of the five checks you
ran with their outcome. Then either what the tree is now ready for, or the stop condition that fired
and what the user must decide.

**Do not summarise the code changes in your own words.** Point at the logs.
