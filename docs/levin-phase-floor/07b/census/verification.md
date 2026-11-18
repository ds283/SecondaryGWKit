### 7b.C The census at 07b's code (prompt 07b §6)

This ran on the Mac Studio (mps026298), 2026-11-16 to 2026-11-18, by Part C's census agent. The log is
[`07b/census/log.md`](07b/census/log.md).
- **Tree and store:** the census ran from the worktree `SecondaryGWKit-lpf07b-0a103b5`, at Part C's dispatch SHA
  `0a103b5a`, whose code is `5e19c4b6`'s (`AdaptiveLevin/levin_quadrature.py` `8956c7b6…`).
  - Gate U0's K0 run at the parent ran in the same tree with the parent's module installed in place of h's by
    `variant_loader.py`. That module is `7b82a0cd`'s, `4ea1948b…`, blob `0701d2a6`, byte-identical to `1ec734d7`'s and 07a's
    census tree's.
  - Both read 06's certified scratch copy of A3 v2, read-only.
- **Constraints:** Ray-free, no lock, no store built, nothing written in a tree.

#### The Gate and the inputs

Commands and outputs are `07b/census/out/cmd/00`–`10`.
- **The harness:** 07a's census `scripts/`. h's docs copy, 07a's Mac Studio copy and this copy before any change hash
  alike (214 files, list `b284121e…`; `out/cmd/00`).
- **The trees:** h is clean with `--ignored`. The main checkout is at `28ebaf07`, clean.
  - `git diff --stat 1ec734d7 7b82a0cd` (outside docs and prompts) lists `CLAUDE.md` and 89c368f2's four files.
  - `5e19c4b6..0a103b5a` lists nothing.
- **AST check:** 89c368f2's four files are AST-equal without docstrings between `1ec734d7` and `7b82a0cd`
  ([`07b/census/out/u0/u0a_ast.txt`](07b/census/out/u0/u0a_ast.txt)).
- **The files the K0 workers loaded**
  ([`07b/census/out/u0/u0a_imports.txt`](07b/census/out/u0/u0a_imports.txt)):
  - nothing changed between the parent and h;
  - of the 07a-tree → h diff, only the two AST-equal LiouvilleGreen modules;
  - h's own `levin_quadrature.py` was never loaded.
- **The registry:** 84 runs, all finished, before k0.
- **The scratch copy:** EQUAL to `copy_files_certified.txt`, and OK against 06e's `a3v2_copy_after.sha256`
  (`out/cmd/04b`, `05b`, `06`).
- **07a's census:** 149 of 149 files equal 07a's `KEPT_ON_MAC_STUDIO.txt`, checkpoint `461df2ec…`. Its 653 working-record
  files and Part A's 7,645 files are hashed
  ([`07b/census/out/runs_hashed_before.txt`](07b/census/out/runs_hashed_before.txt)).

#### Gate U0 and the projection

The k0 run is `levin-phase-floor-07b-k0-20261116T204402`, 11 of 11 commands, all exit 0.

| clause | result | record |
|---|---|---|
| (1) the parent's code is 07a's | holds | [`out/u0/`](07b/census/out/u0/) |
| (2) 07a's census reproduced at the parent | 100 of 100 K0 units bit-identical; 2,110 calls; 07b's new keys set aside, none in 07a's records | [`out/k0/compare_u0b.json`](07b/census/out/k0/compare_u0b.json) |
| (3) the instrument changes nothing | `h-1e-8-3-14501` (the fix acts) and `h-1e-7-1-20582` (untouched) IDENTICAL on and off | [`out/u0c3/compare_instr.json`](07b/census/out/u0c3/compare_instr.json) |
| (4) deterministic | `h-1e-8-3-14501` twice: IDENTICAL | [`out/u0c4/compare_det.json`](07b/census/out/u0c4/compare_det.json) |

- **The projection:** 37,616 s central; widest margin 65,989 s; no unit over budget
  ([`out/census_projection.json`](07b/census/out/census_projection.json)).
- **The predictions:** `state/predictions.md` (`0c6055ac…`), written 2026-11-16T21:03:31Z, before the launch. Its blocks
  are copied by text and checked by substring (`out/cmd/25`).

#### The census

- **The run:** `levin-phase-floor-07b-census-20261117T010200`, `done`:
  - 18,560 of 18,560 units, 0 in error, 0 over budget;
  - exit 0, stderr empty;
  - wall 45,016 s;
  - checkpoint `396784bc…`.
- **Every unit's module evidence holds:**
  - `levin_quadrature.py` is the tree's 8956c7b6, with no variant;
  - nothing foreign was loaded and Ray was never initialised;
  - all 16 workers' evidence holds.
- **Stop 3:**
  - U1's 60,160 untouched calls all equal 07a's in the five outputs
    ([`out/stop3/compare_census_calls.json`](07b/census/out/stop3/compare_census_calls.json));
  - U8's 187 calls all equal (h)'s replay records ([`out/stop3/u8.json`](07b/census/out/stop3/u8.json)).
- **The scores:**
  - [`out/scores/summary.json`](07b/census/out/scores/summary.json);
  - [`out/scores/u4_v4_report.md`](07b/census/out/scores/u4_v4_report.md);
  - [`out/scores/u4_u6_scores.json`](07b/census/out/scores/u4_u6_scores.json);
  - [`out/xcheck/u7_vs_07a_pop07/xcheck.json`](07b/census/out/xcheck/u7_vs_07a_pop07/xcheck.json).
- **U1–U8 all hold** (the log's table).

#### After the census and at the end

- **The scratch copy:** EQUAL to its certification and to its before-hash after the census and at the end
  (`out/cmd/41`–`43b`, `60`–`62`). There are no journal, `-wal` or `-shm` files.
- **07a's census, its working records and Part A's records:** byte-identical before, after and at the end
  (`out/cmd/45`, `64`, EQUAL).
- **The trees:** h is at `0a103b5a`, clean with `--ignored`, with no `__pycache__`. The main checkout is at `28ebaf07`,
  clean.
- **The registry:** 86 runs, all finished. No lock was taken.
