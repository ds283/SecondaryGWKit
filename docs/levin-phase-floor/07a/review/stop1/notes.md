# 07a Part C — orchestrator's checks at planned stop 1 (2026-11-12, ~12:05 UTC)

Mac Studio orchestrator, own readers. Marker touched 11:34:31Z (session scratchpad); 13,596 protected
files hashed before the first dispatch (A3 v2, its scratch copy, R29 store and run dir, 07's census
run dir, `var/levin-phase-floor-07/`, Part A's `measurement/` and run dirs).

1. **R30's fingerprint**, re-run without `--write`: `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`,
   problems none (`fingerprint_r30.txt`). Run `done`, exit 0, ~9.7 min.
2. **S0′**: `git diff --stat e182bc7a dadd63a0 -- . ':!docs' ':!prompts'` lists only `CLAUDE.md` (checked
   before dispatch). Holds.
3. **`tools.store_value_diff` R29 R30 --json** re-run: byte-identical to the agent's (`f6de2ef1…`).
4. **Outside `QuadSourceIntegral`** (`outside_qsi.txt`): every value table identical, 360,462 rows; no
   row in one store only, unmatched, or column in one only. Other differences: timing columns,
   `label` (job name), `GkSource.parents_digest` 291, `GkWKBIntegration` rho_evals/rho_reused and
   `stage_1_RHS_evaluations` 19. Agrees with the agent.
5. **Inside `QuadSourceIntegral`**, by `scripts/qsi_compare.py` (`qsi_compare.txt`): 864 rows, keys
   equal; 484 identical (timing/label aside), 380 moved; `total` 25, `total_abserr` 60,
   `analytic_rad` 168, regions 17, max depth 2; `total_converged` 0 up, 0 down (862/864 both);
   largest |Δtotal| 0.0470 of abserr₂₉, 0.0745 of abserr₃₀ at k = 3.162e5, z = 1.918e8; no row with
   |Δtotal| > abserr₂₉ + abserr₃₀. Guard counts: group 44/562/0/0 both; analytic 336/2,172/4/0 →
   400/2,515/0/0; rows with a located point 170 → 202. Levin records differing: 88 group, 1,146
   analytic. Every figure equals the agent's.
6. **The agent's finding** reproduced: 49 group records with value bit-equal, abserr changed, regions
   unchanged and nothing located in either store; relative change −0.192 to +1.8 (17 beyond 1e-3,
   listed in `qsi_compare.txt`). Of the 380 moved rows, 214 locate and guard nothing in either store
   (`qsi_moved_without_location.txt`): `total` moved in 10 of them, largest 0.00565 of abserr₂₉.
7. **Protected files** unchanged (13,596, by hash). Main checkout `28ebaf07`, clean; worktree
   `status --short --ignored` empty. New files under `var/` since the marker only in §C.1's paths and
   this review directory.
8. **Lock** held by SecondaryGWKit (owner pid 75161, the finished driver); `ray-processes` none.
