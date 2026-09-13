# Orchestrator prompts — the QCD background campaign

Four prompts, one per workstream. Each dispatches one fresh-context subagent per campaign prompt,
reviews between them against fixed criteria, and either continues or stops and reports to the user.

| Workstream | Prompts | File | Character |
|---|---|---|---|
| **A** — the representation | 01–06 | [`workstream-A-representation.md`](workstream-A-representation.md) | The core fix. Six prompts, of which three move numbers. The review is about whether each moved **only what it was supposed to** |
| **B** — the break-point set | 07–08 | [`workstream-B-break-points.md`](workstream-B-break-points.md) | Widest blast radius. Every quadrature and every ODE reads this set, and one of the two prompts may legitimately end in "stop and ask" |
| **C** — close-out | 09 | [`workstream-C-close-out.md`](workstream-C-close-out.md) | One verification prompt. It may not touch production code, and that rule is the review |
| **D** — the source grid | 10–12 | [`workstream-D-source-grid.md`](workstream-D-source-grid.md) | **Gated on README §7 D7.** Do not start it without the user's go-ahead |

## Running one

Start with, for example:

> Read `prompts/qcd-background-audit/orchestrator/workstream-A-representation.md` and follow it.

**Take the baselines the workstream names before dispatching anything.** They cannot be
reconstructed after the fact, and every review in this campaign is a diff against one of them.

Run A → B → C in order. Each workstream's preconditions include the previous one's completion
criterion; do not start B with a prompt-06 miss unresolved.

## The rules that bind the orchestrator

The same ones `prompts/GkTk-remedial/orchestrator/README.md` and
`prompts/phase-representation/orchestrator/README.md` set out, and they are not restated in each
workstream file:

1. **You do not write code.** Not a fix, not a test, not a docstring.
2. **One prompt, one subagent, one commit.**
3. **Do not re-derive the work.** Check that the prompt's own tests pass when *you* run them, that
   the log classifies every deviation, that the boards and `docs/OPEN_ISSUES.md` were updated in the
   same commit, and that the diff stayed inside its allowed files.
4. **Give each subagent only its own prompt.** Do not let it read the others in the campaign; a
   prompt that knows what comes next starts optimising for it.
5. **Relay every subagent question verbatim.** Do not answer it yourself.
6. **Stop rather than repair.** Do not fix a failed check, revert it, or dispatch a follow-up agent
   to patch it. Report the specific check and what the log says about it.
7. **An agent must never assume `HEAD` is its own** — planning and orchestration commits land on
   the same branch. Every dispatch says so.
8. **A test that passes both before and after proves nothing** (campaign README §0.2). Where a
   prompt says "show it fails on `HEAD~1`", **run that check yourself**; it is the single most
   important review step in this campaign, because every error it removes is common mode.

## The dispatch template

For prompt NN, launch a subagent with **exactly** this context, with the model the campaign
README §3 names:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/qcd-background-audit/README.md`,
> `prompts/qcd-background-audit/IMPLEMENTATION_STATE.md`, then your prompt
> `prompts/qcd-background-audit/NN-<name>.md` and the sections of
> `docs/qcd-background-audit-2026-09.md` it cites. Execute the prompt exactly. **Do not read the
> other prompts in this campaign.** Follow README §5 for the commit, the log, this campaign's
> board, any other board your prompt closes an entry on, and `docs/OPEN_ISSUES.md`. Other commits
> may land on this branch while you work: make exactly one commit, and do not amend, reset or
> rebase anything you did not create — if you need to change a commit you already made and it is no
> longer `HEAD`, stop and say so rather than rewriting. When you finish, reply with: the commit
> SHA, the **Result** line from your log, the "State handed to the next prompt" section verbatim,
> the value of `T_Z_REPRESENTATION_VERSION` at your commit, and every deviation with its
> classification tag.
