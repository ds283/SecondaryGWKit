# levin-phase-floor 07d Part A′: KT's threshold, fixed before batch 1's replays are read (2026-11-23T03:25Z)

KT is (K) at "the best (V93-T) threshold" (§9.2 item 6). The best is chosen from V93T8, V93T10 and V93T12 by these rules,
in order:

1. those with no referenced call converged and dishonest that is not so at the tree (table 7);
2. among them, the fewest calls unconverged, over all 4,220;
3. ties go to the lower T.

If none passes rule 1, the one with the fewest new converged-and-dishonest calls is taken, then rules 2 and 3.

No new call reference is needed for batch 1:
- every variant moves regions only from Levin to Clenshaw–Curtis, or not at all;
- Part A's (Rt) holds calls in all 16 Levin→CC strata;
- the reverse stratum stays empty by construction.

KT is checked the same way when it runs.

## The choice (2026-11-23T05:0xZ, from out/tables/t7_scores.json of batch 1, before KT is specified or built)

- Rule 1: V93T8 passes (no new converged-dishonest call); V93T10 passes; V93T12 fails (R:a:W2-0-12977-5, ratio 2.79,
  honest 0.0485 at the tree).
- Rule 2: unconverged over all 4,220: V93T8 2, V93T10 2 (tree 9). A tie.
- Rule 3: the lower T. **KT is (K) at 8 pi.**
