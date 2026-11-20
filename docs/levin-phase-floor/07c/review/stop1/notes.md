# 07c Part C — orchestrator's checks at planned stop 1 (2026-11-19, ~14:40 UTC)

Mac Studio orchestrator, own readers. Marker `review/MARKER_dispatch` touched 13:59:41Z. Before dispatch, the 160
protected input files (R31's store, 07b's census run directory, the scratch copy) checked against their recorded
lists, all OK, 150 files in the run directory, no `-wal`/`-shm` (`checks/protected_before_*`); 17,277 further files
hashed (`checks/extra_before.sha256`: R29, R30, 07a's census run directory, Part A's `measurement/`, 07b's
`compare_census_calls.calls.jsonl`, A3 v2).

1. **R32's fingerprint**, re-run without `--write` (`fingerprint_r32.txt`):
   `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`; R32's files hash the same before and after
   (`r32_files_before_fp.sha256`). Run `done`, exit 0, 14:04:17Z–14:18:01Z (status.json heartbeat).
2. **S0′**: the agent's `git diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'` lists only `CLAUDE.md`;
   matches §C.5.
3. **`tools.store_value_diff` R31 R32 --json** re-run: byte-identical to the agent's (`c42c6d77…`).
4. **Outside `QuadSourceIntegral`** (`outside_qsi.txt`, my own reading of the JSON): no row in one store only,
   unmatched, or column in one only; no unresolvable column. Every value table identical: 1,159 + 1,478 + 2,209 +
   35,874 + 147,642 + 169,653 + 2,447 = 360,462 rows, as the agent reports. Differences only in timing columns,
   `label`, `GkSource.parents_digest` (291) and `GkWKBIntegration` `metadata` (`rho_evals`, `rho_reused`) with
   `stage_1_RHS_evaluations` (26, rho reuse).
5. **Inside `QuadSourceIntegral`**, `scripts/qsi_compare_07c.py` (07b's orchestrator's stop-1 reader, copied
   unchanged; its labels "R29"/"R30" mean A = R31, B = R32) (`qsi_compare.txt`): 864 rows, keys equal; 559
   identical, 305 moved; `total` 251, `total_abserr` 252, `analytic_rad` 106, `WKB_Levin_num_regions` 233, max depth
   78, `total_phase_limited` 0; `total_converged` up 1, down 0 (861 → 862); largest |Δtotal| 1.050e-5 of either
   abserr, at k = 1e6, z = 437173.17; no row and no Levin record with |Δ| beyond the sum of the two abserrs; guard
   counts unchanged (44 / 562 / 0 / 0 group, 400 / 2,515 / 0 / 0 analytic, 202 rows located); summed regions over
   Levin records 219,776 → 211,594. Record `converged` flips: one group record False → True. `phase_limited` flips:
   2 group records False → True (not in the agent's report; descriptive). All agree with the agent's figures.
6. **Lock and Ray** (`lock_status.txt`): lock HELD by SecondaryGWKit, owner pid 19463 not alive (R32's driver,
   exited); `ray-processes` lists none. The worktree is at `cc3368ce` with `status --short --ignored` empty.
