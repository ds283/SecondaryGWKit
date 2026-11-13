### 7a.C The census at 07a's code (prompt 07a §6)

On the Mac Studio (mps026298), 2026-11-12, by Part C's census agent. The log is
[`07a/census/log.md`](07a/census/log.md).
- **Tree and store:** the census ran from the worktree `SecondaryGWKit-lpf07a-4bf50d3`, at Part C's dispatch SHA
  `4bf50d34`, code `1ec734d7`'s (`AdaptiveLevin/levin_quadrature.py` `4ea1948b…`).
  - Gate W0's K0 run ran in the same tree with the parent's module (`dadd63a0`'s, `4e5da371…`, byte-identical to 07's
    census tree `e182bc7a`'s) installed in place of h's by `variant_loader.py`.
  - Both read 06's certified scratch copy of A3 v2, read-only.
- **Constraints:** Ray-free, no lock, no store built, nothing written in a tree.

#### The Gate and the inputs

Commands and outputs are `07a/census/out/cmd/00`–`10`.
- **The harness:** 07's census `scripts/`, byte-identical to the copy that ran on the Mac Studio (`out/cmd/00`).
- **The trees:** h is clean with `--ignored`. The main checkout is at `28ebaf07`, clean.
  - `git diff --stat e182bc7a dadd63a0` (outside docs and prompts) lists `CLAUDE.md` only.
  - `1ec734d7..4bf50d34` lists nothing.
- **The parent's module** (`out/cmd/08`): `git show dadd63a0` gives `4e5da371…`, blob `9ea30e2b…`, equal at `e182bc7a`.
- **The LiouvilleGreen modules:** AST-equal without docstrings between `dadd63a0` and h
  ([`07a/census/out/w0/w0a_ast_lg.txt`](07a/census/out/w0/w0a_ast_lg.txt)).
- **The registry:** 74 runs, all finished.
- **The scratch copy:** EQUAL to `copy_files_certified.txt`, and OK against 06e's after-hashes.
- **07's census:** 150 of 150 files equal its `KEPT_ON_MAC_STUDIO.txt`, checkpoint `03dfe786…`. Its 850 working-record files
  are hashed ([`07a/census/out/runs_hashed_before.txt`](07a/census/out/runs_hashed_before.txt)).

#### The harness

07's census `scripts/`, copied, changed where marked "prompt 07a" (diff:
[`07a/census/out/scripts_diff_at_end.txt`](07a/census/out/scripts_diff_at_end.txt)).
- **The tree:** h, with the parent as a variant.
- **The diagnostics `route_07a`:** every subregion solve of every Levin call, with:
  - h's V and 07's estimate on the same nodes;
  - the C2 route and LU choice under each;
  - the solve's role (own, trust-check quarter, comparison half, fallback half);
  - the node-partition sum;
  - Clenshaw–Curtis regions with a half sent to Levin.
- **The comparison:** 07's diff, with 07a's keys set aside. Each differing path is put in a category (`call`, `location`,
  `harness`).
- **V4's re-scoring:** on 07's readings.

#### Gate W0 — holds

- **(1)** The K0 workers loaded no file that S0′'s diff lists. Of the parent→h diff they loaded only the two AST-equal
  LiouvilleGreen modules, and never h's `levin_quadrature.py`
  ([`07a/census/out/w0/w0a_imports.txt`](07a/census/out/w0/w0a_imports.txt)).
- **(2)** Run `levin-phase-floor-07a-k0-20261112T130135`: 100 of 100 K0 units at the parent equal 07's census bit for bit.
  That is 2,110 calls and 32,956 leaves ([`07a/census/out/k0/compare_w0b.json`](07a/census/out/k0/compare_w0b.json)).
- **(3)** `h-1e-8-1-2174` (a window) and `h-1e-7-1-11214` (an own C2 route change), diagnostics on and off: identical
  ([`07a/census/out/w0c3/compare_instr.json`](07a/census/out/w0c3/compare_instr.json)).
- **(4)** `h-1e-8-1-2174` twice: identical ([`07a/census/out/w0c4/compare_det.json`](07a/census/out/w0c4/compare_det.json)).

#### The predictions and the rulings

- [`07a/census/state/predictions.md`](07a/census/state/predictions.md), written 2026-11-12T13:16:44Z, after W0 and before the
  launch, SHA-256 `3ec7b53d…`.
- The user's rulings at STOP 2 (Q1: reading A; Q2: the call's outputs) are in
  [`07a/census/state/ruling_stop2.md`](07a/census/state/ruling_stop2.md).

#### The census

Run `levin-phase-floor-07a-census-20261112T132744`: `done`, 18,560 of 18,560, exit 0, 0 errors, 0 over budget.
- Wall 23,284 s against a projection of 23,353 s. Checkpoint `461df2ec…`.
- All 16 workers used `4ea1948b…`, with no variant and no Ray.

#### W1–W7

[`07a/census/out/scores/summary.json`](07a/census/out/scores/summary.json),
[`07a/census/out/stop3/compare_census_calls.json`](07a/census/out/stop3/compare_census_calls.json),
[`07a/census/out/scores/w4_report.md`](07a/census/out/scores/w4_report.md),
[`07a/census/out/xcheck/`](07a/census/out/xcheck/).
- **W1 holds** (reading A; the call's outputs):
  - 258,824 untouched calls, 0 differing in the call's outputs;
  - 13,382 differ only in the skipped-phase record;
  - 9,726 rows made only of untouched calls have `total` and `total_abserr` equal.
- **W2 holds:** 0 Levin leaves listed.
- **W3 holds:** 0 violations. The 784 skipped phases are all `zero_rate`.
- **W4 fails, as predicted:** v08 3.16, v09 2.21, v12 3.08, v17 2.38, v22 2.53. All 33 calls are bit-identical to 07's.
- **W5 holds** (gated solves, by the user's ruling, [`07a/census/state/ruling_w5.md`](07a/census/state/ruling_w5.md)):
  0 node-partition sums, 0 Clenshaw–Curtis regions with a Levin half, 0 guarded regions.
  - Descriptive: 18,236 forced solves took the node-partition sum, within the root-located calls (4,416 of 4,804).
- **W6 fails, as predicted:** `3-14501` r_g 1.39 and `3-35006` r_g 1.08. The other 2,482 pairs are ≤ 1.
- **W7 fails:** 25 lone-`G` minima-window calls at 1e-8 move by 0.0066–0.145 of abserr_07, beyond 1e-5. At 1e-7 the largest
  move is 1.80e-7.

#### The user's references (findings, not W verdicts)

On the user's word at the W5 stop: rows `3-678` and `1-34599`, group calls 2 and 3, at 1e-7 and 1e-8, by 07's V4 method. The
results are in [`07a/census/out/refs/report.md`](07a/census/out/refs/report.md).
- **The captures:**
  - 8 of 8 h captures reproduce 07a's census bit for bit;
  - 8 of 8 captures with the parent's module reproduce 07's census;
  - the integrands are equal by bit pattern between the two captures and across the rtols.
- **The run:** `levin-phase-floor-07a-references-20261112T203555`: 16 of 16 readings, all usable. FCC r2 is the reference.
- **The stated error** (the readings' spread) is 1.3e-6 (`1-34599`) and 9.2e-8 (`3-678`) of the smaller abserr, so every call
  is scored.
- **`1-34599`:** h honest at 0.020 (1e-7) and 0.015 (1e-8) of its abserr; 07 not honest, at 4.03.
- **`3-678`:** h honest at 0.491 of its abserr, which is 11× smaller than 07's; 07 at 0.135.

#### After

- **The scratch copy:** EQUAL to its certification after the census and at the end, and equal to its before
  (`out/cmd/41`–`43`, `60`–`62`).
- **07's census:** all 1,000 files (run directory and working records) equal before, after the census and at the end
  (`out/cmd/44`–`45`, `63`–`64`).
- **After the references run:** the scratch copy is again EQUAL to its certification, and 07's census is equal to its before (`out/cmd/83`–`87`).
- **Trees:** h is clean with `--ignored`; the main checkout is clean at `28ebaf07`.
- **The registry:** 77 runs, all finished.
