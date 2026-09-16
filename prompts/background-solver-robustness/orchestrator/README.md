# Orchestrator prompts — the background solver robustness campaign

Four prompts, one per workstream. Each dispatches one fresh-context subagent per campaign prompt,
reviews between them against fixed criteria, and either continues or stops and reports to the user.

| Workstream | Prompts | File | Character |
|---|---|---|---|
| **A** — the equality solve | 01–02 | [`workstream-A-the-equality-solve.md`](workstream-A-the-equality-solve.md) | The core fix. Two prompts, of which one changes production code. The review is whether the two redshifts moved — they must not — and whether prompt 02's new assertions really fail on `HEAD~1` |
| **B** — what the redshifts feed | 03–04 | [`workstream-B-what-the-redshifts-feed.md`](workstream-B-what-the-redshifts-feed.md) | Widest blast radius, and **prompt 03 ends in a decision you put to the user**. A datastore identity is downstream of the thing being measured |
| **C** — cost and close-out | 05–06 | [`workstream-C-cost-and-close-out.md`](workstream-C-cost-and-close-out.md) | One timing prompt with a stop-or-escalate rule, one close-out that may not touch production code — and that rule is the review |
| **D** — housekeeping | 07–08 | [`workstream-D-housekeeping.md`](workstream-D-housekeeping.md) | **Gated on README §7 D3.** Two orphaned one-liners. Do not start without the user's go-ahead |

## Running one

Start with, for example:

> Read `prompts/background-solver-robustness/orchestrator/workstream-A-the-equality-solve.md` and
> follow it.

**Take the baselines the workstream names before dispatching anything.** They cannot be
reconstructed after the fact, and every review in this campaign is a diff against one of them.

Run **A → B → C** in order. Each workstream's preconditions include the previous one's completion
criterion. D is independent and gated.

## The rules that bind the orchestrator

The same ones `prompts/qcd-background-audit/orchestrator/README.md` and
`prompts/phase-representation/orchestrator/README.md` set out, and they are not restated in each
workstream file:

1. **You do not write code.** Not a fix, not a test, not a docstring.
2. **One prompt, one subagent, one commit.**
3. **Do not re-derive the work.** Check that the prompt's own tests pass when *you* run them, that
   the log classifies every deviation, that the boards and `docs/OPEN_ISSUES.md` were updated in
   the same commit, and that the diff stayed inside its allowed files.
4. **Give each subagent only its own prompt.** Do not let it read the others in the campaign; a
   prompt that knows what comes next starts optimising for it.
5. **Relay every subagent question verbatim.** Do not answer it yourself.
6. **Stop rather than repair.** Do not fix a failed check, revert it, or dispatch a follow-up agent
   to patch it. Report the specific check and what the log says about it.
7. **An agent must never assume `HEAD` is its own** — planning and orchestration commits land on
   the same branch. Every dispatch says so.
8. **A test that passes both before and after proves nothing** (campaign README §2 (e)). Where a
   prompt says "show it fails on `HEAD~1`", **run that check yourself**. It is the single most
   important review step in this campaign, because the campaign's *other* stop condition is
   "nothing moved" — and "nothing moved" is exactly what a test that tests nothing also reports.

## The three checks that apply after every prompt in A, B and C

Run these yourself, at the subagent's commit, before dispatching the next one. They are cheap and
they are the campaign's whole safety net.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
./venv/bin/python -m black --check $(git diff --name-only HEAD~1 HEAD -- '*.py')
```

- `ComputeTargets` must read **447, OK** at every commit of this campaign. Nothing in that package
  reads `_find_rho_equality`; a move there means something changed that nobody intended.
- `CosmologyModels` may only **rise**.
- Plus: `grep -n "T_Z_REPRESENTATION_VERSION" CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`
  must read **6** at every commit.

## The dispatch template

For prompt NN, launch a subagent with **exactly** this context, with the model the campaign
README §3 names:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/background-solver-robustness/AUDIT.md`,
> `prompts/background-solver-robustness/RECONCILIATION.md`,
> `prompts/background-solver-robustness/README.md`,
> `prompts/background-solver-robustness/IMPLEMENTATION_STATE.md`, then your prompt
> `prompts/background-solver-robustness/NN-<name>.md` and the files it tells you to read first.
> Execute the prompt exactly. **Do not read the other prompts in this campaign.** Follow README §5
> for the commit, the log, this campaign's board, any other board your prompt closes an entry on,
> and `docs/OPEN_ISSUES.md`. Other commits may land on this branch while you work: make exactly one
> commit, and do not amend, reset or rebase anything you did not create — if you need to change a
> commit you already made and it is no longer `HEAD`, stop and say so rather than rewriting.
> **This campaign's central claim is that no computed quantity moves**; if you find yourself
> needing to move one to make the prompt pass, stop and say so rather than moving it. When you
> finish, reply with: the commit SHA, the **Result** line from your log, the **two equality
> redshifts** section verbatim, the "State handed to the next prompt" section verbatim, the value
> of `T_Z_REPRESENTATION_VERSION` at your commit, both suite counts, and every deviation with its
> classification tag.

## When to stop and ask the user

Beyond each workstream's own list:

- **README §7 D2** — prompt 03's report on the three copies of the equality closed form. This is a
  decision with a datastore attached and it is not yours or the subagent's.
- **README §7 D4** — prompt 05's cost row, if the hoist does not clear 2.5 µs.
- **README §7 D3** — before workstream D runs at all.
- Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.
