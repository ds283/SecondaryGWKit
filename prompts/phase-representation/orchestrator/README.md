# Orchestrator prompts — the phase-representation campaign

One prompt, [`campaign.md`](campaign.md), covering both of the campaign's prompts. It restates the
procedure and stop conditions of [`../README.md`](../README.md) §4 with the prompt-specific checks
— test commands, allowed files, thresholds — filled in.

| Prompts | Prompt file | Character |
|---|---|---|
| 01, 02 | [`campaign.md`](campaign.md) | Both change production code every stored WKB phase passes through. Small diffs, wide blast radius; the review is about what moved that should not have |

## Running it

Start with:

> Read `prompts/phase-representation/orchestrator/campaign.md` and follow it.

**Take the `verify_production_path.py` baseline before dispatching anything** — both prompts are
reviewed by diffing against it, and it cannot be reconstructed after the fact.

## The rules that bind the orchestrator

The same ones `prompts/GkTk-remedial/orchestrator/README.md` sets out, and they are not restated
here:

1. **You do not write code.**
2. **One prompt, one subagent, one commit.**
3. **Do not re-derive the work.** Check that the prompt's own tests pass when *you* run them, that
   the log classifies its deviations, that the boards and `docs/OPEN_ISSUES.md` were updated
   together, and that the diff stayed inside its allowed files.
4. **Give each subagent only its own prompt.**
5. **Relay questions verbatim.**
6. **Stop rather than repair.** Do not fix a failed check, revert it, or dispatch a follow-up agent
   to patch it. Report the specific check and what the log says.
7. **An agent must never assume `HEAD` is its own** — you land planning commits on the same branch.
   Every dispatch says so.
