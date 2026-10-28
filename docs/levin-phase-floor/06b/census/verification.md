### 6b.5 The census: 06a's census re-run at the code commit (Mac Studio, 2026-10-28)

**What was run.** Run `levin-phase-floor-06b-census-20261028T161212` re-ran every A3 v2
`QuadSourceIntegral` row (9,280) at `rtol` 1e-7 and 1e-8, in-process and Ray-free.
- **Tree:** $b$, the worktree at `92bc62f`, whose code is `4ddc3c2`'s.
- **Store:** 06's certified scratch copy of A3 v2, read-only.
- **Comparison:** against 06a's run `levin-phase-floor-06a-census-20261027T023143`.
- **Outcome:** 18,560 of 18,560 units `done`, none over budget, none in error. 16:12–17:03Z.
- **Harness:** 06a's, changed as the log's deviations 1–8 say. The diff is
  [`06b/census/out/scripts_diff_against_06a_final.txt`](06b/census/out/scripts_diff_against_06a_final.txt).

**Gate W0 holds** ([`06b/census/out/cmd/14`](06b/census/out/cmd/14_W0_check_rerun.txt),
[`15`](06b/census/out/cmd/15_W0_clause1.txt)):
1. **The parent's code is 06a's.** `e05c34f..da5c24d` changes:
   - `CLAUDE.md`;
   - three files whose ASTs without docstrings are equal: `primitive_phase.py`, `main.py` and
     `test_primitive_phase.py`;
   - four test modules that the harness does not import.
2. **The harness reproduces 06a.** At `da5c24d`, `low-G+r-1` and `ctrl-low-1` at 1e-7, and `0-1971`
   at 1e-8, are identical to 06a's units in every computed value.
3. **The new code is deterministic.** `b-1e-7-1-34599` is identical in two fresh workers.

**The predictions.** [`06b/census/state/predictions.md`](06b/census/state/predictions.md), sha256
`8df51796…`, was written at 14:13:55Z, before the launch. Part A is the prompt's W1–W4 byte for
byte. Part B fixes W2's selection rule and the readings, including the analytic-block reading the
user accepted.

| | Measured | Verdict |
|---|---|---|
| **W1** | 36,112 / 35,605 group calls (1e-7 / 1e-8) not phase-limited in 06a: all equal to 06a's bit for bit (value, abserr, regions, depth). 6,777 / 6,177 rows all of whose calls are such: all equal in `total` and `total_abserr`. | **holds** |
| **W2** | 24 calls (12 low, 12 mid). Every re-run from the saved inputs reproduces b. $\|S_9-S_{10}\| \le 0.0066$ `abserr`$_b$, so none is dropped. $\|{\rm value}_b - S_{10}\| \le$ `abserr`$_b$ on 7 and fails on 17, by up to 19.7×. | **fails** |
| **W3** | 44 regions accepted as phase-limited at 1e-7, all in analytic-block calls. Recomputed floors: own equals stored on all 44. Halves' sum / own is 0.504–0.891. 0 violate the rule. | **holds** |
| **W4** | Three moved analytic-block calls at 1e-8 reach depth 20 (up to 31,885 regions; unit wall up to 159 s). 0 units over budget. | **fails** |

**The store.** The copy's five files equal `copy_files_certified.txt` before W0, before the launch,
after the run and after the fingerprint, with no journal file. It gives `0e32f24a…` before and
after.

**What W2's failure means** ([`out/cmd/38`](06b/census/out/cmd/38_w2_crosscheck.txt),
[`39`](06b/census/out/cmd/39_w2_request.txt)):
- **The reference holds up.** $S_{11}$ agrees with $S_{10}$. On one call the adaptive driver at
  `rtol` 1e-11 agrees too.
- **On two others the driver at `rtol` 1e-11 returns b's value unchanged,** with the same regions.
  Their residuals and floors sit well below the true error.
- **b's values are better than 06a's, but its error is understated.** Against the request, b's
  error exceeds it on 4 of 24 calls and 06a's on 23 of 24. b's declared `abserr` is below its true
  error on 17 of 24.

**Descriptive items** ([`out/cmd/37`](06b/census/out/cmd/37_analyse06b_report_rerun.txt)):
- **Phase-limited group calls** fall from 1,213 (on 585 rows) to 0 at 1e-7, and from 1,720 (745)
  to 0 at 1e-8. Every one of 06a's phase-limited regions fails the premise.
- **Analytic blocks:** 12,633 / 17,722 of their calls reached the old exit under 06a. 32 / 800 are
  still phase-limited under b.
- **Row depth:** largest 14 under b, against 12 / 13 under 06a; with the analytic blocks, 18 / 20.
- **`total_converged` false:** 2,578 → 2,349 rows at 1e-7, and 2,638 → 2,388 at 1e-8.
- **Wall time:** 51 min against 45.5.

**Records:**
- [`06b/census/log.md`](06b/census/log.md) and [`06b/census/issues.md`](06b/census/issues.md)
  (C1–C4);
- W3's per-region floors: [`06b/census/out/w3_regions.csv`](06b/census/out/w3_regions.csv);
- W2's sample: [`06b/census/state/w2_selection.json`](06b/census/state/w2_selection.json);
- W2's references: [`06b/census/out/w2_reference.json`](06b/census/out/w2_reference.json).

The checkpoint, the region pickles and the 975 captured calls' inputs stay on the Mac Studio, in
`var/runs/levin-phase-floor-06b-census-20261028T161212/` (checkpoint sha256 `f934ea24…`).
