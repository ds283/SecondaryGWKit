# 07b Part C — orchestrator's checks at planned stop 1 (2026-11-16, ~19:55 UTC)

Mac Studio orchestrator, own readers. Marker `review/MARKER_dispatch` touched 19:14:22Z; 159 protected
input files (R30's store, 07a's census run dir, the scratch copy) checked against their recorded lists
before dispatch (`checks/protected_before_check.txt`); Part A's `measurement/` (7,645 files) and R29 hashed.

1. **R31's fingerprint**, re-run without `--write`: `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`
   (`fingerprint_r31.txt`). Run `done`, exit 0, 19:18:49Z–19:32:27Z.
2. **S0′**: the agent's `git diff --stat 1ec734d7 7b82a0cd` and AST check; matches §C.5 (checked on the laptop).
3. **`tools.store_value_diff` R30 R31 --json** re-run: byte-identical to the agent's (`b7d508c1…`).
4. **Outside `QuadSourceIntegral`** (`outside_qsi.txt`): no row in one store only, unmatched, or column
   in one only. Differences only in timing columns, `label`, `GkSource.parents_digest` (291) and
   `GkWKBIntegration` `metadata`/`stage_1_RHS_evaluations` (23, rho reuse). Every value table identical.
5. **Inside `QuadSourceIntegral`**, `scripts/qsi_compare_07b.py` (07a's stop-1 reader copied, diff in
   `qsi_compare_07b.diff`): 864 rows, keys equal; 295 identical, 569 moved; `total` 379, `total_abserr`
   386, `analytic_rad` 548, `WKB_Levin_num_regions` 386, max depth 93, `total_phase_limited` 11;
   `total_converged` 0 up, 1 down (862 → 861; k=1e5, q=r=1e6, z=144627.70674226896);
   largest |Δtotal| 0.27804 of abserr₃₀, 0.23848 of abserr₃₁ (k=1e5, z=12004.58); rows with
   |Δtotal| > abserr₃₀+abserr₃₁: 0 (float; S3 itself is the agent's, after the stop);
   `phase_limited` flips False→True: 5,336 analytic, 348 group, none the other way; one group record
   converged→unconverged; 7 group records (all rtol 1e-7, G-q±r) with |Δv| > abserr₃₀+abserr₃₁, 1.11–1.53;
   guard counts identical in both stores (group 44/562/0/0, analytic 400/2,515/0/0; 202 rows located).
   Every figure the agent reported that I recomputed agrees.
6. **Protected files** unchanged (159 + R29 by hash). Main checkout `28ebaf07`, clean; worktree
   `status --short --ignored` empty. Nothing new under `var/` since the marker outside the rehearsal's
   paths and this review directory.
7. **Lock** (`lock_status.txt`): held by SecondaryGWKit, as planned through the stop.
