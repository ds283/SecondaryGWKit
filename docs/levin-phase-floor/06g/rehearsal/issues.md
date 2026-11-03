# Issues from 06g §4 (rehearsal agent)

Two findings for the orchestrator and the user. Neither is a §7 stop. The orchestrator decides
whether each becomes a board entry or is folded into an existing one.

1. **Proposed: `[06g-r24-g+q+r-group-moves-past-both-errors]`** (V1 fails; log, Observation 1).
   - **Hook:** "R23/R24: 12 rows' sub-interval-1 `G+q+r` group moves 2,500–321,000× its summed
     errors".
   - **Measured.** V1 on `total` fails in 12 of 864 rows. The row ratio
     $\lvert\Delta T\rvert/(e_{23}+e_{24})$ runs from 224 to 13,101.5.
   - **Where.** In every row the move is in one call, sub-interval 1's `G+q+r` group, regime (True,
     True, True).
     - R23 (06h's code): 2–12 regions, converged.
     - R24 (06g): 39–208 regions, phase-limited, unconverged at depth 20 in 10 of the 12.
     - The group's value moves by 0.1–3.4 % against declared errors near $10^{-7}$ of it.
   - **Impact.** At least one of the two codes' declared errors understates the true error of
     these calls by $\ge 2{,}500\times$. The rehearsal has no reference, so it cannot say which.
     - If R23 is wrong, the parent trusted a leaf that was far from converged. 06g's (b) is then
       doing its job, and the remaining question is why R24 cannot converge there.
     - If R24 is wrong, 06g's refinement has produced a worse value with an equally small
       declaration.
   - **Next step.** Reference one or more of these calls with `tools/filon_reference.py`. They are
     rehearsal rows ($k = q = r \in \{10^5, 3.16\times10^5, 10^6\}$), not A3 v2 rows, so they are
     outside §5's sample and would need a capture from the rehearsal tree.
2. **Proposed, or folded into `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`:
   convergence lost in the rehearsal** (log, Observation 2).
   - **Hook:** "R24 loses convergence on 67 of 573 Levin rows, every one at depth 20".
   - **Measured.**
     - 67 rows (97 calls) went converged → unconverged, and none the other way.
     - 76 rows reach `depth_max` under 06g, against 0 at the parent.
     - Every lost row is at depth 20, and all the lost calls are `G±q±r` groups of sub-intervals 1
       and 2.
     - Regions rose 4.29× (31,216 → 134,011).
   - **Mechanism.** Reaching the cap rather than the node-limited stop, which ends below
     `depth_max`. This is inferred: the store keeps no per-leaf data, and 06g's `node_limited` is
     not persisted. It is a different mechanism from the open issue's, so it fits better as its own
     entry, or as a note on V8.
   - **What it bears on.** §5's V5 and V8, which will score it on A3 v2.

**Not opened,** for the record:
- **The readers' unseeded `random.sample`** makes their outputs differ file for file between runs
  (log, Observation 5). No store is affected, and nothing in this prompt depends on it.
