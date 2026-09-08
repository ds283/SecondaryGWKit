# Orchestrator prompts — source remediation campaign

One orchestrator prompt per workstream of [`../README.md`](../README.md). Each one is given to an
orchestrating agent that dispatches fresh-context subagents per prompt, runs the five checks of
README §4.1 itself, and stops on the conditions listed in README §4.1 plus the prompt-specific
ones spelled out in the file. The orchestrator never edits code and never resolves a stop
condition on its own.

| File | Prompts | Run after | Notes |
|---|---|---|---|
| [`workstream-A.md`](workstream-A.md) | 01, 02, 03 | — | Complete as of `f8c75f5`. |
| [`workstream-D.md`](workstream-D.md) | 04 | — | Single prompt, `main.py` QuadSourceIntegral stage only. Must land before 10. |
| [`workstream-B.md`](workstream-B.md) | 05, 06 | A (soft) | Strictly 05 → 06. Hands `TkSourceFunctions` and the `QuadSource` region to C. |
| [`workstream-C.md`](workstream-C.md) | 07, 08, 09, 10 | B; D before 10 | Strictly serial. Expect a stop after 08 if its cost ratio exceeds 3×. |
| [`workstream-E.md`](workstream-E.md) | 11, 12 | everything | 12 needs Ray and a fresh datastore; runs for one to two hours. |

**Parallelism.** D and B may run concurrently with each other (and with A). They both edit
`main.py`, in disjoint stages (04: QuadSourceIntegral; 06: QuadSource), so a rebase is expected to
be clean — but the second to land must be rebased so each prompt stays one commit, and a conflict
is a stop, not something the orchestrator resolves. C waits for B in full and for D before its
last prompt. E waits for everything.

**What each orchestrator may read.** Only its own prompts, the logs it produces, and the upstream
logs its prompts are told to consume (B: none; C: logs 04, 05, 06; E: all logs). Prompts of later
workstreams are off limits so that the review of the current prompt is not biased towards what a
later prompt would find convenient.
