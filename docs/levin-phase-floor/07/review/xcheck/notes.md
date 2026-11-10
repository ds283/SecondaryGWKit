# Cross-check (beyond §6, on the user's request): orchestrator's checks

The user's words, relayed verbatim by the laptop session:

> Yes, please do the cross-check on `rtol`s and the list of calls where `06m` and `07` differ by more than the stated error. We can decide on a sample of references to sort of what these difference really show, later.

This is the census agent's deviation 30.

- **The rule was fixed before the results.** The rule file `census/state/xcheck_rule_20261109.md` (sha `e06a1058…`) has mtime 00:57:12Z, and the first result, `out/xcheck/xcheck.json`, 00:58:21Z.
- **My own reader:** `review/scripts/xcheck_recount.py`, sha `af759dfa…`. It was written separately from the agent's `xcheck_07.py` and run under `nice -n 19`; its output is `review/xcheck/recount.json` and `recount.txt`.
  - Both checkpoints were hashed before and after, and are unchanged.
  - 06m's lines were each checked by their SHA-256 and unit name.
  - Every comparison is exact, in Fraction arithmetic.
- **Results, all matching the agent's:**
  - 4,968 calls in the population and 2,484 pairs, with none unpaired.
  - r_g ≤ 1 in 2,482 pairs and r_06m ≤ 1 in 2,432.
  - The pairs with r_g > 1 are exactly the agent's two, 3-14501 (1.385) and 3-35006 (1.084), both `G+q+r` #1 and class (e).
  - 843 listed calls, all `G+q+r`: (a) 31 at 1e-7; (c) 390 at 1e-7 and 396 at 1e-8; (e) 6 at 1e-7 and 20 at 1e-8. None is in (b) or (d).
  - **Call by call against the agent's `B.calls`:** none is listed only by me, none only by the agent, and no class differs.
- **Not re-checked by me:** the 396 captures, the projection figures, and the row-total statistics.
