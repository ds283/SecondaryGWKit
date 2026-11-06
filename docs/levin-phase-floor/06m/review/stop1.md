# 06m Part B — planned stop 1, the orchestrator's checks (Mac Studio, 2026-11-06, ~09:55–10:05 UTC)

The rehearsal agent stopped at planned stop 1 with the lock held, readers not run. Checks, each by
the orchestrator's own scripts in `review/scripts/`, outputs in `review/stop1/`:

1. **Fingerprints.** R25 `levin-phase-floor-06m-r25-parent-20261106T092420` and R26
   `levin-phase-floor-06m-r26-signedyardstick-20261106T093937`: both `done`, exit 0, digest
   `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, `problems {}`, taken at
   `5ff8a5e4` and `38f6a828` with `git_dirty` false; both `classes` objects equal R24's. **Pass.**
2. **R25 against R24.** `qsi_full_compare.py` (`34eded4d…`) over every `QuadSourceIntegral` column
   and every `metadata` leaf (bookkeeping, labels and timing aside): 864 rows, 143,831 fields, **0
   differing** (`qsi_full_r24_r25.txt`). `vdiff_values.py` (`abdd5501…`) on the agent's JSON
   (`d9445335…`): every value table of every class identical, no row in one store only or
   unmatched; all 6,012 label differences differ only by the job-name prefix `…-06g` → `…-06m` and
   the stamp (`vdiff_r24_r25.txt`). The agent's deviation 1 is confirmed. **Pass: reproducible.**
3. **R26 against R25 outside `QuadSourceIntegral`.** 05's `classify_non_qsi.py` (`26690e61…`, run
   from the committed 05 copy) on the agent's JSON (`d67e696f…`): 0 unclassified, exit 0
   (`classify_r25_r26.txt`); `vdiff_values.py`: every value table identical, no row in one store
   only (`vdiff_r25_r26.txt`). **Pass.**
4. **R1, re-scored** by `r1_orchestrator.py` (`ad82f95c…`) in exact rational arithmetic
   (`r1_orchestrator.txt`): 0 of the 864 rehearsal rows is an A3 v2 row, so none is among 06a's 720.
   - `total`: 864 scored, **0 violations**, 0 non-finite, largest ratio 4.93492e-06 at
     (1e6, 1e5, 1e6, z 573.11465); 413 bit-identical.
   - oracle: 864 scored, **0 violations**, largest 6.67731e-08 at (1e6, 1e6, 1e6, z 996.42016);
     860 bit-identical.
   - `total_converged` 797 → 845: 48 gained, 0 lost; `total_abserr` 403 bit-identical; regions
     summed 134,011 → 114,826 (442 rows changed); rows at max depth 20: 76 → 23.
   - `qsi_full_r25_r26.txt` counts agree with the agent's tabulation in every figure compared.
   **R1 holds**, the agent's scoring confirmed.

The agent is resumed to run the readers on R26 and release the lock.
