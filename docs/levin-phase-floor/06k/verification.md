## §6k — Three ways to stop refining the slow leaves (prompt 06k, a measurement; on the Mac Studio)

Run on mps026298, 2026-11-04 to 2026-11-05.
- **Tree:** the worktree at 06k's dispatch SHA `89370609`, whose production code is `1c5336e`'s (06g's `3ae6f3a`);
  `git diff --stat 1c5336e 89370609 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md` and `tools/filon_reference.py`
  with its test ([06k/out/cmd/00_gate_trees.txt](06k/out/cmd/00_gate_trees.txt)).
- **Store:** 06's certified scratch copy of A3 v2, read-only on a stand-in `ShardedPool`. No Ray, no lock, no store
  built.
- **The routes:** each is a scratch variant of `AdaptiveLevin/levin_quadrature.py` under
  [06k/scripts/variants/](06k/scripts/variants/), loaded in place of the tree's module by
  [06k/scripts/variant_loader.py](06k/scripts/variant_loader.py). Nothing in the tree changed.

### 6k.1 The Gate

Every clause holds, from [06k/out/cmd/00](06k/out/cmd/00_gate_trees.txt) to
[06k/out/cmd/07b](06k/out/cmd/07b_ray_processes_plain.txt):
- `6e8c2559` and `89370609` are on `origin/handover-remedial`;
- the main checkout is at `28ebaf0`, clean;
- `runregistrykit list` shows 48 runs, all finished;
- the scratch copy is EQUAL to its certification, with fingerprint `0e32f24a…`;
- 06g's census checkpoint is `03adcbba…`, with 16 region pickles, and `v8_calls.json` is `1e9a62b0…`;
- 89 GB were free.

### 6k.2 The variants

- **Built:** five variants by [06k/scripts/make_variants.py](06k/scripts/make_variants.py)
  ([06k/out/cmd/10b](06k/out/cmd/10b_make_variants_rebuilt.txt), [90](06k/out/cmd/90_make_variants_with_route2lit.txt)):

  | variant | SHA-256 |
  |---|---|
  | identity | `d89c959e…`, byte-identical to the tree's file |
  | route 1 | `23cd69e4…` |
  | route 2 | `017fae1c…` |
  | route 3 | `e139ec71…` |
  | route 2 literal | `2fd10b78…`, added on the user's ruling at planned stop 1 |

- **Diffs:** [06k/out/variant_route1_diff.txt](06k/out/variant_route1_diff.txt) and the others beside it, including
  [route 2 literal against route 2](06k/out/variant_route2lit_vs_route2_diff.txt).
- **AST checks:**
  - every variant differs from the tree only in `_adaptive_levin`, and every name it binds there ends in `_06k`
    ([11b](06k/out/cmd/11b_ast_check_variants.txt), [91](06k/out/cmd/91_ast_check_route2lit_vs_tree.txt));
  - route 2 literal differs from route 2 as built only in `_check_leaf`
    ([92](06k/out/cmd/92_ast_check_route2lit_vs_route2.txt)).

### 6k.3 Gate K0 — holds

The 100 units are [06k/out/k0/units.json](06k/out/k0/units.json): 06j's 46 calls' units, plus 54 drawn by the rule in
[06k/state/predictions_and_rules_20261104T0915Z.md](06k/state/predictions_and_rules_20261104T0915Z.md) §2, which was
written before any unit ran. Each was compared with 06g's census record on every key, floats by bit pattern
([06k/out/k0/](06k/out/k0/), [06k/out/k0/summary.json](06k/out/k0/summary.json)):

| configuration | units equal to 06g | compared |
|---|---|---|
| tree's module (control) | 100 of 100 | every key |
| identity variant | 100 of 100 | every key |
| route 1 | 100 of 100 | its 1,855 calls with no firing |
| route 2 | 100 of 100 | its 1,837 calls with no firing |
| route 2 literal | 100 of 100 | its 1,032 calls with no firing of any kind |
| route 3 | 100 of 100 | its 683 calls with no firing |

- **What "every key" covers:** value, abserr, regions, region arrays, `converged`, checked leaves and the harness's
  counters.
- **Excluded:** timing and provenance, and `cache_built`, by
  [06k/state/k0_comparison_correction_20261104T0911Z.md](06k/state/k0_comparison_correction_20261104T0911Z.md),
  written before any K0 unit.
- **Route 2 literal:** clause 2 does not apply, by the orchestrator's ruling, because the variant changes every checked
  Clenshaw–Curtis leaf's declaration. Its calls with no firing of any kind were compared all the same.

### 6k.4 Part L

- **The captures:** 06j's 46 calls were re-captured, and each reproduces 06g's census bit for bit
  ([06k/out/check_captures.json](06k/out/check_captures.json)).
- **The probes reproduce the census:** Δ1, Δ2, r and φP agree on 142 of 142 L-a leaves and 1,882 of 1,882 L-c leaves
  ([06k/out/partL_leaves.csv](06k/out/partL_leaves.csv), [06k/out/partL_summary.json](06k/out/partL_summary.json)).
- **Route 2 literal:** its withdrawals are listed in
  [06k/out/partL_route2lit_withdrawals.csv](06k/out/partL_route2lit_withdrawals.csv).
- **The LiouvilleGreen draws** are compared in [06k/out/lg_draws/compare.txt](06k/out/lg_draws/compare.txt). The
  identity variant equals the tree on all 4,320 Levin calls and every case.

### 6k.5 Part C

Four registered censuses, each `done` with 18,560 of 18,560 units, exit 0. Each started only after the orchestrator
confirmed the previous one.

| route | run | wall | checkpoint |
|---|---|---|---|
| 1 | `levin-phase-floor-06k-census-route1-20261104T103009` | 16,758 s | `36e3dd4c…` |
| 2 | `levin-phase-floor-06k-census-route2-20261104T151120` | 13,621 s | `8363ae82…` |
| 2 literal | `levin-phase-floor-06k-census-route2lit-20261104T185927` | 13,968 s | `7fb9c41e…` |
| 3 | `levin-phase-floor-06k-census-route3-20261104T225339` | 7,626 s | `89afd6da…` |

- **The comparisons** are in [06k/out/partC/](06k/out/partC/). Every call matched its 06g call by kind, index, label
  and span.
- **The references:** 124 are matched and scored, and 06g's own ratio is reproduced on all 124
  ([06k/out/partC/h1.json](06k/out/partC/h1.json)). V1's 12 are unscored, on the user's ruling.
- **Results:** the verdicts are in [06k/out/results.md](06k/out/results.md) and
  [06k/log.md](06k/log.md).

### 6k.6 After

- **The scratch copy** is EQUAL to its certification after, and again after the final fingerprint, which is
  `0e32f24a…` ([110](06k/out/cmd/110_copy_files_after.txt)–[114](06k/out/cmd/114_copy_after_fp_vs_certified.txt)).
- **06g's census checkpoint, its 16 pickles, `v8_calls.json`, `v8_slow_leaves.json`, `rows.json` and 06h's
  checkpoint** are unchanged by SHA-256 ([115](06k/out/cmd/115_checkpoints_and_inputs_after.txt),
  [115b](06k/out/cmd/115b_v8_slow_leaves_after.txt)).
- **The `var/datastores/` listing** is unchanged.
- **No Ray:** every census and K0 worker reports Ray not initialised and no foreign module
  ([120](06k/out/cmd/120_runs_no_ray_after.txt), [121](06k/out/cmd/121_k0_workers_no_ray.txt)). The lock is free, and
  no Ray process is listed.
- **The trees:** the worktree is at `89370609`, clean including ignored files, and the main checkout is at
  `28ebaf0`, clean ([116](06k/out/cmd/116_trees_after.txt)).
- **The registry:** 55 runs, all finished ([117](06k/out/cmd/117_registry_after.txt)).
