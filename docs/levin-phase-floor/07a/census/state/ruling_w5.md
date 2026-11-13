# levin-phase-floor 07a §6 — the user's ruling at the W5 stop (relayed by the laptop orchestrator), recorded 2026-11-12

The user, verbatim: "Yes, W5 gated only, and take references for both calls"

The orchestrator's reading, as relayed:
1. W5 is scored on gated solves only: its three counts are over gated (non-forced) region solves. Under that reading W5
   holds (0 node-partition sums, 0 Clenshaw-Curtis regions with a half sent to Levin, 0 guarded). The forced-solve count is
   descriptive (out/scores/w5_forced.json): 18,236 sums in 4,416 calls, 3-10 per call -- the window piece whose end node is
   the root, and its forced halves and quarters -- within the root-located calls (not "exactly" them: 388 root-located
   group calls, 194 at each rtol, have none). state/predictions.md part 5's wording ("every solve ... forced included") is
   superseded by this ruling; predictions.md is not edited.
2. References for `3-678` and `1-34599` (overriding §C.1's "no new references are taken" for these two calls only), by
   V4's reference method of 07's census (the harness and settings of levin-phase-floor-07-references-20261109T005328),
   copied into scripts/ with the diff saved; each call as the aux_only/own table identifies it, at each rtol it was flagged
   at; each capture checked bit for bit against the 07a census record and against 07's; Ray-free, at most six workers; a
   registered detached run, slug `references`, if over ten minutes. Report per call: the reference with its stated error;
   |v - ref|/abserr at h and at 07 with each abserr; whether the stated error is at least 10x below both abserrs (if not,
   say so rather than score). Findings, not W verdicts.
