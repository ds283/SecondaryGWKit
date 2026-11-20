### 7c.6 — the census (Part C §6), verification by the census agent

Records: `07c/census/` (this directory's `log.md`, `out/`, `scripts/`, `state/`). Links are in the `07c/…` form.

**The trees.**
- h is the worktree `SecondaryGWKit-lpf07c-cc3368c` at `cc3368ce`, clean with `--ignored`, no `__pycache__`, at the start
  and at stop 2 (`07c/census/out/cmd/01_gate_trees.txt`, `29_state_at_stop2.txt`). Its `levin_quadrature.py` is
  `d141a80b…`, `67b5af74`'s; `git diff --stat 67b5af74 cc3368ce -- . ':!docs' ':!prompts'` is empty.
- The main checkout is at `28ebaf07`, clean, not moved; git was read with `--no-optional-locks` only.

**Gate Y0.**
1. S0′: `git diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'` lists `CLAUDE.md` alone; the parent's module is
   `5e19c4b6`'s byte for byte (`07c/census/out/y0/y0a_ast.txt`, `07c/census/out/cmd/03_parent_module.txt`). The K0
   workers at the parent loaded no file of h that differs from the parent or from 07b's census tree, and not h's own
   module (`07c/census/out/y0/y0a_imports.txt`).
2. K0 at the parent: 100 of 100 units bit-identical to 07b's census (`07c/census/out/k0/compare_y0b.json`).
3. The instrument: two units IDENTICAL on and off; the instrument's counts equal the result's on every call
   (`07c/census/out/y0c3/compare_instr.json`).
4. Determinism: one unit where the fix acts, twice, IDENTICAL (`07c/census/out/y0c4/compare_det.json`).
5. The projection: 43,312 s central; 55,392 s at the highest margin times 07b's shortfall; within 48 h
   (`07c/census/out/census_projection.json`).

**Read only and unchanged** (before: `07c/census/out/cmd/04`–`07`; at stop 2: `27`–`28b`).
- The scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`: EQUAL to its certification and to its before-hash;
  `shasum -c` against 06e's after-hashes OK; no journal, `-wal` or `-shm`.
- 07b's census run directory: 150 files, equal to 07b's `KEPT_ON_MAC_STUDIO.txt`, checkpoint `396784bc…`; 07b's K0 run
  directory: 5 files, equal; 07b's census working records: the 54 kept files equal, 517 files hashed; Part A's
  `var/levin-phase-floor-07c/measurement/`: 17,111 files. Before and at stop 2, every hash EQUAL
  (`07c/census/out/runs_hashed_before.txt`, `07c/census/out/runs_hashed_stop2.txt`).

**The registry.** Before K0: 94 runs, all finished. At stop 2: 95, all finished (`levin-phase-floor-07c-k0-20261119T151009`
`done`, 11/11). No lock was taken; nothing used Ray (every worker's `bye`: `ray_initialized` false).

**Predictions.** `07c/census/state/predictions.md`, SHA-256 `cad4fd7f…`, mode 0444, mtime 2026-11-19T15:30:24.709924Z,
before any census launch; its four prompt blocks are substrings of the prompt (`07c/census/out/cmd/25_write_predictions.txt`).

**At planned stop 3** (`07c/census/out/cmd/40`–`49`).
- The census `levin-phase-floor-07c-census-20261119T201419`: `done`, 18,560/18,560, exit 0, 0 in error, 0 over budget,
  wall 43,080 s.
- Every unit's evidence holds: module `d141a80b` from the tree, no variant, h clean at `cc3368ce`, no foreign module, no
  Ray. 16 workers (`07c/census/out/stop3/compare_census_calls.json`).
- Y1's untouched calls: 207,310 of 207,310 equal 07b's in the five outputs; 8,508 of 8,508 all-untouched units equal 07b's
  in both totals (`07c/census/out/stop3/compare_census_calls.json`).
- Y8: 453 of 453 equal (S)'s replay records (`07c/census/out/stop3/y8.json`).
- The stop's instrument: its counts equal the result's on all 289,482 calls (`07c/census/out/cmd/48b_stop3_instrument_counts.txt`).
- After the census, every hash is unchanged: the scratch copy (certification, before-hash, `shasum -c`), 07b's runs and
  records, and Part A's records (`07c/census/out/runs_hashed_after.txt`). The trees are clean.

**The scoring** (`07c/census/out/cmd/50`–`55`; `07c/census/out/scores/summary.json`).
- **Y1–Y9 all hold.**
- **Y4** is scored on 312 scorings: 07b's U4's 190 (`07c/census/out/scores/y4_v4_scores.json`,
  `07c/census/out/scores/y4a_y6_scores.json`) and Part A's 122 (`07c/census/out/scores/y4b_scores.json`). The only
  converged dishonest scorings are R2-28's four, at 1.1434.
- **Y6:** `07c/census/out/scores/y4a_y6_scores.json`. **Y7:** `07c/census/out/xcheck/y7_vs_07b_pop07/xcheck.json`
  (Mac Studio only; summarised in `summary.json`).
- **Y9** reads 07b's `07b/census/out/scores/summary.json` list of 198.
- **Arithmetic:**
  - Y4 (b), the value moves and the abserr ratios are exact Fractions.
  - Y4 (a) and Y6 use 07b's scripts' mpmath at 40 digits.
  - R2-28 agrees between the two.

**At the end** (`07c/census/out/cmd/60`–`66`):
- The scratch copy is EQUAL to its certification and to its before-hash; `shasum -c` gives 5 OK.
- 07b's census and K0 runs, 07b's records and Part A's records: before against end, EQUAL
  (`07c/census/out/runs_hashed_end.txt`).
- The trees are clean, with no `__pycache__`.
- The registry holds 96 runs, all finished.
- `state/predictions.md` is still `cad4fd7f…`, mode 0444.
