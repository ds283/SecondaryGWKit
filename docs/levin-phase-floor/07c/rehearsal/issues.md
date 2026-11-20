# Issues seen by the 07c rehearsal agent (R32), not acted on

For the orchestrator to put to the user at 07c's records commit. None is a stop. None blocks a verdict of §5.

1. **Two rows are unconverged in both R31 and R32, each with a group call at depth 20.** Neither is a row that §9.5 names.
   Neither is beyond S3's bound.
   - **(3.162e5, 3.162e5, 3.162e5), z 573.11, G−q−r** (`out/stop1_unconverged_look.txt`, `out/unconverged_calls_look.txt`).
     - The row's total is bit-identical in R31 and R32, and the group record is unchanged.
     - The replay at the tree, which reproduces R32 bit for bit, gives:
       - 70 regions, maximum depth 20, two leaves at depth 20;
       - 2 range-rule refusals;
       - **0 chase stops and 5 noise bisections**.
     - Read from the counts alone: the chase below its refused leaves made 5 noise-level checks, one short of
       `_LEVIN_CHASE_STOP_LIMIT = 6`, so the stop never fired.
     - The rehearsal cannot say whether each depth-20 leaf sits below a refused leaf: the harness here does not carry
       07c Part A's leaf instrument.
   - **(1e6, 3.162e5, 1e6), z 434.65, G−q−r.**
     - Regions 102 → 100, still at depth 20.
     - Its total moved by 9.6e-8 of e31 + e32.
     - It has no located point, so it is not among the replayed rows and its counts were not read.
   - **Bearing:** §9.5's Y9 says no census call stays unconverged with every depth-20 leaf below a leaf a rate rule
     refused, apart from five named calls. If either call is of that kind and turns up in the census, Y9 will see it. The
     census's own instrument decides; this rehearsal does not.
   - Next step: none here. Its relevance is to §6's Y9.

2. **Captured calls are not byte-identical across trees, even when the integrand is.**
   - The blob of a captured call holds three things that change between trees:
     - the unit name, which includes the store serial (R32 `3-122` against R31 `3-20363`);
     - the worktree's path, in the code object's filename;
     - pickled `str` hashes, the `_name_hash` of the units object, which are randomised per process.
   - So every rehearsal's `blocks.json` reports `capture_equals_r3x: False`: 07a's, 07b's and this one's.
   - S4 already compares the integrand at 1,001 samples. The S3 record is now compared the same way (deviation 24).
   - Recorded so that no later prompt reads a byte comparison of captures as a test of the integrand.

No other issue. The two `phase_limited` flips (`out/phase_limited_flips.txt`) are tabulated as descriptive, at the
orchestrator's note. Both converged records moved by 5.5e-8 of e31 + e32 and stay converged, so no issue is opened for
them.
