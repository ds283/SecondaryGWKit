# 07c Part C — orchestrator's checks at planned stop 3 (2026-11-20, ~08:35 UTC)

Census run `levin-phase-floor-07c-census-20261119T201419`. My background wait on driver pid 90737 ended
08:15:25Z; status `done`, exit 0, 18,560 of 18,560, last heartbeat 08:12:19Z, `stderr.log` 0 bytes, stdout
ends "units in error 0; problems []", over budget 0.

1. **Y1 and `converged`, my own reader** (`../final/scripts/stop3_y1_y8.py`, sha in `stop3_y1_y8.out`; result
   `stop3_y1_y8.json`). It streams both checkpoints (07b's `396784bc…` run directory and 07c's), and compares
   every Levin call by hex value, hex abserr, `converged`, `num_regions` and `max_depth`. Group = `calls`,
   analytic = `other_calls`, and no call's notify label disagrees with that class.
   - 18,560 units in each, the same set, no duplicates, every 07c unit `done`, identical call keys per unit.
     Every 07c call carries the three chase counts.
   - Untouched (`num_chase_stops` 0) / equal / touched / touched yet equal:
     1e-7 analytic 73,021 / 73,021 / 34,395 / 5,707; 1e-7 group 24,775 / 24,775 / 12,550 / 206;
     1e-8 analytic 84,152 / 84,152 / 23,264 / 4,684; 1e-8 group 25,362 / 25,362 / 11,963 / 203.
     Total untouched 207,310, all equal. These equal the census agent's table cell for cell.
   - 8,508 units are untouched in every call; all 8,508 equal 07b's in hex `total` and `total_abserr`.
   - `converged`, 07b → 07c: analytic, 107,416 True→True at each rtol and nothing else. Group 1e-7:
     37,146 T→T, 97 F→T, 82 F→F; group 1e-8: 37,133 T→T, 96 F→T, 96 F→F. **No call is newly unconverged**,
     and 193 group calls are newly converged. This agrees with the agent.
2. **Y8, my own reader.** Each (S) replay with `census_which` '07b census' (453; one more, `r31-3-20363-1`, is
   R31's and is not a census id) is located in its 07b unit by its recorded 07b (value hex, abserr hex).
   - 347 match a unique call, and 07c's call equals the replay in all five outputs (247 of them where the stop
     acted in the census).
   - 106 match two calls. These are twin calls, identical in 07b and identical in 07c, with the replay equal
     to both (`../final/scripts/stop3_y8_ambiguous.py`, `stop3_y8_ambiguous.out`).
   - So 453 of 453 hold. The distinct call counts differ from the agent's 448 only because I count both twins.
3. **Protected files** at stop 3 (`../checks/protected_stop3_*`, `extra_stop3.sha256`): the 160 recorded files
   OK, 150 files in 07b's census run directory, and the 17,277 further files byte-identical to the before-hashes.
4. `calls_07b.jsonl` (82 MB) and `calls_07c.jsonl` (211 MB) are compact per-call extracts for the final
   review. They are too large for git and go on the review's KEPT list.
