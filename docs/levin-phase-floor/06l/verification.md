## 6l. The trust yardstick with the halves' node errors signed (route B), measured on A3 v2

Prompt 06l, on the Mac Studio (mps026298), 2026-11-05.
- **Tree and store:** from the worktree `SecondaryGWKit-lpf06l-af4c4dc` at `af4c4dc1`, whose production code is
  `1c5336e`'s (06g's). It read 06's certified scratch copy of A3 v2, read-only.
- **Constraints:** Ray-free, no lock, no store built, no change to the tree.
- **Records:** `06l/`. The log is [`06l/log.md`](06l/log.md), and the results are
  [`06l/out/results.md`](06l/out/results.md).

### 6l.1 The Gate

Every clause holds (`06l/out/cmd/00`–`07`, `06b`).
- **Trees:**
  - the worktree is at `af4c4dc1`, clean with `--ignored`;
  - `34437951` and `af4c4dc1` are on `origin/handover-remedial`;
  - `git diff --stat 1c5336e af4c4dc1 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md`, `tools/filon_reference.py` and
    its test;
  - the main checkout is at `28ebaf07`, clean.
- **Registry and lock:** 55 runs, none `running` or stale; the lock is free, and no Ray process is listed.
- **The scratch copy:** EQUAL to its certification by SHA-256, with fingerprint `0e32f24a…`.
- **06g's census:** checkpoint `03adcbba…` with its 16 region pickles; `census/out/v8_calls.json` is `1e9a62b0…`.
- **Route 1's census:** checkpoint `36e3dd4c…` with its 16 region pickles. The pickles and both extracts
  `var/levin-phase-floor-06k/state/extract/g.pkl` and `route1.pkl` match 06k's `KEPT_ON_MAC_STUDIO.txt`.
- **Disk:** 85 GB free.

### 6l.2 The variants

Built by 06k's `make_variants.py`, extended for B (`06l/out/cmd/10c`).
- **The identity variant** (`d89c959e…`) is the tree's file byte for byte; its diff is empty.
- **Route B** (`5aa6178a…`) differs only inside `_adaptive_levin` (AST check `06l/out/cmd/11c`: CONFINED). There it
  changes:
  - the records list;
  - `_check_leaf`'s trust statement: ν_B and B's record;
  - the return key `route_06l`.
- **Its diffs:** against the tree, [`06l/out/variant_routeB_diff.txt`](06l/out/variant_routeB_diff.txt); against 06k's
  route 1, [`06l/out/variant_routeB_vs_route1_diff.txt`](06l/out/variant_routeB_vs_route1_diff.txt).
- **06k's four routes,** rebuilt in memory by the same script, have 06k's recorded SHA-256.

### 6l.3 Gate K0 — holds

Run `levin-phase-floor-06l-k0partL-20261105T091942`, on 06k's 100 units
([`06l/out/k0/`](06l/out/k0/summary.json)).
- **The identity variant:** all 100 units equal 06g's census records on every key, counters included.
- **Route B:** all 1,862 calls with no firing are equal to 06g's (16 units compared whole).
- **Firings:** 248 calls fire, and there are 0 withdrawals.
- **Workers:** each K0 worker reports Ray not initialised and no foreign module.

### 6l.4 Part L, Gate L3 and the draws

- **The re-capture.** Run `k0partL`'s capture check refused all 46 captures, on the harness-script hashes alone
  (log deviation 11). It was re-run, with slicing and probes, as `levin-phase-floor-06l-partL-20261105T095407`:
  - 46 of 46 captures reproduce 06g's census bit for bit;
  - the probes reproduce the census's Δ1, Δ2, r and φ_P on all 142 L-a and 1,882 L-c leaves;
  - 06k's 42 columns equal 06k's committed table on all 2,046 rows
    ([`06l/out/partL_leaves.csv`](06l/out/partL_leaves.csv), [`06l/out/partL_summary.json`](06l/out/partL_summary.json)).
- **Gate L3 holds on all 2,046 leaves.** ν_B ≤ ν1·(1 + 10⁻¹²) everywhere, at most 1 + 2.2 × 10⁻¹⁶, and B trusts no
  leaf that route 1 does not.
- **L1 holds:** 0 of 17 carrying leaves trusted.
- **L2 fails:** 63 of 75 first-population leaves trusted.
  - Each of the 12 others is Clenshaw–Curtis over Clenshaw–Curtis, and on each E_P and E_L + E_R have opposite signs.
  - On those 12, Δ1/ν_B is 1.0004–1.0041 ([`06l/out/partL_l2/account.json`](06l/out/partL_l2/account.json)).
- **The LiouvilleGreen draws** (240 units, [`06l/out/lg_draws/compare.txt`](06l/out/lg_draws/compare.txt)):
  - every test passes;
  - the identity variant equals the tree on all 4,320 Levin calls;
  - route B fires in 122 calls, changes no convergence, and moves no case between honest and dishonest.

### 6l.5 The census

Run `levin-phase-floor-06l-census-routeB-20261105T100126`: `done`, 18,560/18,560, exit 0, 18,340 s against a
projection of 17,347 s. The backstop reported twice and stopped nothing. Checkpoint `5cb3d3c0…`.

Against 06g's census ([`06l/out/partC/routeB.json`](06l/out/partC/routeB.json)):
- **C1 holds:** 749/801 (93.5 %) and 773/830 (93.1 %).
- **C2 holds:** 4 losses, all `node_limited` at 1e-8.
- **S1 holds:** 0 calls outside the 720.
- **K1 holds:** 9,839 Levin-parent firings outside the 720, the largest Δ1/φ_P among them 2.85.
- **H1 holds:** 115/115, worst v03 at 1.182 ([`06l/out/partC/h1.json`](06l/out/partC/h1.json)).
- **Calls at `depth_max`:** 72 and 76 outside the 720.
- **Cost:** unit wall 0.86 of 06g's, and group-call regions 0.84.

Against route 1's census:
- route 1 converges 91 calls outside the 720 that B does not, and B converges 6 that route 1 does not;
- no call moves past both errors.

Against A ([`06l/out/partC/against_A.json`](06l/out/partC/against_A.json)): of the 91 calls,
- 14 rest only on Levin parents with a Clenshaw–Curtis half;
- 2 rest only on Clenshaw–Curtis parents;
- 75 rest on both.

### 6l.6 After

- **The scratch copy:** EQUAL to its certification after the run and after the final fingerprint; the fingerprint is
  still `0e32f24a…` (`110`–`114`).
- **The other inputs, 39 files equal by SHA-256 before and after** (`115`, `115c`):
  - 06g's census checkpoint and its 16 pickles;
  - `v8_calls.json`, `v8_slow_leaves.json` and `rows.json`;
  - route 1's checkpoint and its 16 pickles;
  - both 06k extracts.
- **No Ray:** all 16 census workers report Ray not initialised and no foreign module (`120`). The lock is free, and no
  Ray process is listed (`118`, `119`).
- **The trees:** the worktree is clean with `--ignored`, with no `__pycache__`, and the main checkout is clean (`116`).
- **The registry:** 58 runs, all finished (`117`).
- **The datastores:** `var/datastores/` is unchanged.

**The verdicts:** L1, C1, C2, H1, S1 and K1 hold, and L2 fails. B does not pass as a candidate remedy by §4's rule as
written. It proposes no code.
