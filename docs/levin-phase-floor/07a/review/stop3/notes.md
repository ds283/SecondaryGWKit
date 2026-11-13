# 07a Part C — orchestrator's checks at planned stop 3 (2026-11-12, ~20:00–20:25 UTC)

The census run `levin-phase-floor-07a-census-20261112T132744` is `done`, exit 0, 18,560 of 18,560 units, stderr empty,
heartbeat 19:55:48Z (one background `kill -0` wait on pid 90766, which ended by 19:57:06Z). The census agent's
STOP 3 report says that no untouched call differs from 07's in the `call` category. These are my own checks, made
with my own readers. Every script and output is in this directory, with its hash in `sha256.txt`.

1. **The checkpoints**, read by my own index (`scripts/index07.py`; outputs `index07.txt`, `index07a.txt`).
   - 07's checkpoint is still `03dfe786…`, 18,560 lines.
   - 07a's census checkpoint is `461df2ec…`, 18,560 lines. It equals the agent's figure, and no row@rtol is duplicated.
2. **The census against 07's, call by call** (`scripts/stop3_compare_07a.py`, sha `821e3d31…`; outputs
   `compare_07a_vs_07.{json,txt}`). It is modelled on 07's own stop-3 reader.
   - **How it reads.** Each line is read at my index's offset and checked by its SHA-256. Each region record is
     checked against its line's `pickle_record_sha256`.
   - **Re-keying.** Each call's entries are re-keyed by its index, and every leaf is compared (floats by `float.hex`).
   - **What is set aside:**
     - timing, provenance and `cache_built`;
     - keys absent at 07 whose names carry `07a` (or `x1_inputs`);
     - `guard_07.location_skipped_phases`.
   - **Everything else differs:** each such difference is attributed to its call, or to the unit.
   - **Class, per call:**
     - `window`: a point or window at 07 or at h;
     - else `own`: h's `route_07a.own_differs`;
     - else `aux_only`: `aux_differs`;
     - else `untouched`.
     The class rests on the h worker's own `route_07a` instrumentation. I take that as given and do not recompute it.
   - Totals: 18,560 units, 289,482 calls; the call sets match 07's in every unit, and every unit is done in both.

   | class | calls | equal to 07 (strict) | location only | anything else |
   |---|---|---|---|---|
   | untouched | 258,824 | 245,442 | 13,382 | **0** |
   | aux_only | 15,847 | 0 | 0 | 15,847 |
   | own | 7,511 | 0 | 0 | 7,511 |
   | window_both | 4,940 | 4,452 | 0 | 488 |
   | window_h_only | 2,360 | 0 | 0 | 2,360 |
   | window_07_only | 0 | | | |

   - **The 13,382 untouched location-only calls.**
     - 07 has one Rule A entry `{phase, variation}`.
     - At h, 12,598 have none and 784 have `{phase, reason: zero_rate}`.
     - No untouched call has any guard count at either tree. The 28 calls with guard counts are all `window_h_only`.
   - **Value, abserr, `converged` and `num_regions`** equal 07's in all 258,824 untouched calls.
   - **Untouched-only units.** 9,726 units hold only untouched calls. All have `total` and `total_abserr` equal to 07's
     in hex, and none has a unit-level difference.

   **No untouched call differs from 07's in anything but Rule A's location record. The stop-3 condition is not met.**
   - Every figure equals the census agent's.
   - The agent splits window_both's 488 as 460 `call` and 28 `harness` only. My reader does not split them.
3. **The protected files** (`protected_at_stop3.txt`): all 13,596 are unchanged by hash. They include A3 v2, its scratch
   copy, the R29 store and run dir, 07's census run dir, `var/levin-phase-floor-07/`, Part A's `measurement/` and
   Part A's run dirs.
4. **State** (`state_check.txt`):
   - The main checkout is at `28ebaf07` and clean.
   - The worktree is at `4bf50d34` and empty under `--ignored`.
   - The registry has 76 runs, all finished.
   - The lock is free.
   - No file under `var/` is newer than the marker outside §C.1's paths and `review/`.

Nothing has been scored. Q1 and Q2's rulings are applied as relayed.
