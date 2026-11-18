# 07b Part C — orchestrator's checks at planned stop 3 (2026-11-17, 13:33–13:45 UTC)

The census run `levin-phase-floor-07b-census-20261117T010200` is `done`, exit 0, 18,560 of 18,560 units, 0 in error,
0 over budget, stderr empty, heartbeat 13:32:16Z. I waited with one background `kill -0` loop on pid 45520. The census
agent's STOP 3 report says that every untouched U1 call equals 07a's census, and that U8 holds at 187 of 187. These
are my own checks, made with my own readers. Every script and output is in this directory, with its hash in `sha256.txt`.

1. **The checkpoints**, read by my own index (`scripts/index_ck.py`, a copy of 07a's `index07.py`).
   - The census checkpoint is `396784bc…`, 18,560 lines, with no row@rtol duplicated. It equals the agent's figure.
   - 07a's census checkpoint is still `461df2ec…`, 18,560 lines.
2. **The census against 07a's, call by call** (`scripts/stop3_compare_07b.py`, sha `47de0084…`; outputs
   `compare_07b_vs_07a.{json,txt}`).
   - It is 07a's stop-3 reader with these changes:
     - the before is 07a's census;
     - new keys are those absent at 07a whose names contain `07b`, or that are one of the two refusal counts;
     - the class is U1's: a call is untouched iff `num_edge_rule_refusals + num_range_rule_refusals == 0`. Both
       counts are asserted ints, and `rate_rule_07b.counts_equal_result` is asserted on every call;
     - the outputs compared are value, abserr, `converged`, `num_regions` and `max_depth`.
   - Every leaf of the data record and the region record is compared.
   - Units: 18,560; calls: 289,482. The call sets match 07a's in every unit (kind, index, label and span), and every
     unit is done in both.

   | rtol | kind | untouched | range only | edge |
   |---|---|---|---|---|
   | 1e-7 | group | 16,274 | 20,711 | 340 |
   | 1e-7 | analytic | 13,484 | 93,932 | 0 |
   | 1e-8 | group | 16,651 | 20,340 | 334 |
   | 1e-8 | analytic | 13,751 | 93,665 | 0 |

   - **All 60,160 untouched calls equal 07a's in all five outputs. None differs in any other path of its call record
     or region record** (no location-only differences either).
   - 5,688 units are untouched in every call. All have `total` and `total_abserr` equal to 07a's in hex. One of them,
     `0-33600@1e-7`, differs at the unit level only in `analytic_06h.parent_counterfactual.value`.
   - These figures equal the agent's: 60,160 untouched; 228,648 range only; 665 + 9 = 674 with an edge refusal;
     5,688 units, with one differing in the oracle.
   - **The stop-3 condition (an untouched call differing) is not met.**
   - Descriptive, not scored:
     - converged at 07b against 07a: 198 calls newly unconverged (194 range, 4 edge); 5 newly converged (range).
       The unconverged in both: 31 untouched, 138 range, 4 edge.
     - regions: untouched 958,592 in both; range 1,927,252 → 4,993,407; edge 23,290 → 87,640.
     - refusals: edge 438 (1e-7) and 430 (1e-8), group only; range 53,794 group / 351,789 analytic at 1e-7, and 53,162 /
       323,256 at 1e-8.
     - max depth: 53 range-class calls equal 07a's in all five outputs; one range-class call is shallower than at 07a.
     - summed unit wall: 1e-7 52,154 s against 24,952 s; 1e-8 54,667 s against 28,513 s.
3. **U8** (`scripts/u8_check.py`, sha `dec92008…`; output `u8_check.txt`).
   - `census_wanted_07b.json` (`33d0d89c…`) has 188 ids. The only one excluded is `i3-34599-07` (which `07`).
   - The 187 calls (134 group, 53 analytic) each equal `out/ext/replays/h/<id>.json`'s `replay` in value and abserr
     (hex), `converged`, `num_regions` and `max_depth`. **0 fail.**
4. **The protected files** (`protected_at_stop3.txt`): all unchanged by hash.
   - The 159 files are R30's store, 07a's census run directory and the scratch copy.
   - Part A's `measurement/` is 7,645 files.
   - R29 is 5 files.
5. **State** (`state_check.txt`, `lock_status.txt`):
   - The main checkout is at `28ebaf07` and clean.
   - The worktree is at `0a103b5a` and empty under `--ignored`.
   - No file under `var/` is newer than the dispatch marker outside §C.1's paths. The new run directories are the five
     07b runs.
   - The registry has 86 runs, all finished.
   - The lock is free.

Nothing has been scored.
