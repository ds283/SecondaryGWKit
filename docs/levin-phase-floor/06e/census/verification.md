### 6e.5 The census: 06c's census re-run under exact end nodes (Mac Studio, 2026-10-30/31)

**What was run.** Run `levin-phase-floor-06e-census-20261030T081318` re-ran every A3 v2
`QuadSourceIntegral` row (9,280) at `rtol` 1e-7 and 1e-8, in-process and Ray-free.
- **Tree:** $g$, the worktree at `6d81fad`, whose code is 06e's `f4d8e17`.
- **Store:** 06's certified scratch copy of A3 v2, read-only.
- **Comparison:** against 06c's run `levin-phase-floor-06c-census-20261029T130214`.
- **Outcome:** 18,560 of 18,560 units `done`, none in error, none over budget, none past 300 s.
  The run took 08:13–10:00Z, against 06c's 2 h 30 min.
- **Harness:** 06c's, changed as the log's deviations 1–10 say. The diff is
  [`06e/census/out/scripts_diff_against_06c_final.txt`](06e/census/out/scripts_diff_against_06c_final.txt).

**Gate Z0 holds** ([`06e/census/out/cmd/20`](06e/census/out/cmd/20_Z0_check.txt),
[`21`](06e/census/out/cmd/21_Z0_clause1.txt)):
1. **The parent's code is 06c's census's.** `e5764a6..50277f3` outside `docs/` and `prompts/`
   changes `CLAUDE.md` only.
2. **The harness reproduces 06c.** At `50277f3`, `low-G+r-1` and `ctrl-low-1` at 1e-7, and
   `0-1971` at 1e-8, equal 06c's units in every value 06c recorded.
3. **The new code is deterministic.** `g-1e-7-0-14164` is identical in two fresh workers, on every
   key, the new ones included.

**The predictions.** [`06e/census/state/predictions.md`](06e/census/state/predictions.md), sha256
`f27771cc…`, was written at 08:12:47Z, before the launch. Part A is the prompt's Z2–Z5 byte for
byte. Part B fixes the readings, Z3's sample and its matching rule.

| | Measured | Verdict |
|---|---|---|
| **Z2** | Calls outside the 720 rows at depth 20: **0** at 1e-7 and **1** at 1e-8 (`g-1e-8-0-33995` call 5, G−q+r, unconverged, floor above its request), against 06c's 109 and 153. No unit over budget. | **fails** (1e-8) |
| **Z3** | 13 referenced (`orch_ref.py`, three resolutions); 6 count (2, 3, 8, 9, 14, 15), 7 dropped on scatter (1, 10, 13, 22, 23, 25, 27). Of the 6, $\|v_g-{\rm ref}\|/{\rm abserr}_g$ = 0.78–0.9995 on five; **sample 3 at 1.245** (06c 1.197). | **fails** |
| **Z4** | 8,560 rows scored at each rtol, 0 violations, exact. | **holds** |
| **Z5** | Calls outside the 720 that 06c reported unconverged: 85 (1e-7) and 173 (1e-8); not converged under $g$: 0 and **1** (the Z2 call). | **fails** (1e-8) |

**Descriptive.**

| | 1e-7, $g$ | 1e-7, 06c | 1e-8, $g$ | 1e-8, 06c |
|---|---|---|---|---|
| Unit wall summed | 2.28 h | 3.21 h | 2.97 h | 5.77 h |
| Solves | 4.86 M | 7.26 M | 6.21 M | 12.44 M |
| Regions | 0.63 M | 0.97 M | 0.80 M | 2.09 M |
| Calls at depth 20 | 11 (all in the 720) | 109 | 18 (17 in the 720) | 153 |
| Rows unconverged | 9 | 13 | 15 | 45 |
| Rows unconverged, outside the 720 | 0 | 9 | 1 | 33 |

- Every call past depth 12 outside the 720 (14 and 24) has ends that tile at every level, and all
  but the Z2 call converge at depth 13–15.
- Clenshaw–Curtis floors are 99.4 % the declared-phase term; the node term is 0.06 %.
- `*-1-2174` call 1 at 1e-8 converges at depth 7 on depth20-tiling's pinned value, 2.3e3 ×
  06c's abserr from 06c's (07's class).

**The store is unchanged.** It is byte-identical to its certification before, at the launch and
after, and its fingerprint is `0e32f24a…`. 06c's checkpoint is `b61cb034…` throughout. Records:
[`06e/census/log.md`](06e/census/log.md) and [`06e/census/issues.md`](06e/census/issues.md).
