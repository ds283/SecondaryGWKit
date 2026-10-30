### 6c.5 The census: 06b's census re-run under the global budget (Mac Studio, 2026-10-29)

**What was run.** Run `levin-phase-floor-06c-census-20261029T130214` re-ran every A3 v2
`QuadSourceIntegral` row (9,280) at `rtol` 1e-7 and 1e-8, in-process and Ray-free.
- **Tree:** $g$, the worktree at `e5764a6`, whose code is 06c's `ad9a5f4` plus 06d's `3d70db9`.
- **Store:** 06's certified scratch copy of A3 v2, read-only.
- **Comparison:** against 06b's run `levin-phase-floor-06b-census-20261028T161212`.
- **Outcome:** 18,560 of 18,560 units `done`, none in error, none over budget by the dispatcher's
  test. 13:02–15:33Z, against 06b's 51 min. Nine units ran past 300 s (log, deviation 22).
- **Harness:** 06b's, changed as the log's deviations 1–14 say. The diff is
  [`06c/census/out/scripts_diff_against_06b_final.txt`](06c/census/out/scripts_diff_against_06b_final.txt).

**Gate X0 holds** ([`06c/census/out/cmd/09`](06c/census/out/cmd/09_X0_check.txt),
[`14`](06c/census/out/cmd/14_X0_clause1.txt)):
1. **The parent's code is 06b's census's.** `92bc62f..fef401c` outside `docs/` and `prompts/`
   changes `CLAUDE.md` only.
2. **The harness reproduces 06b.** At `fef401c`, `low-G+r-1` and `ctrl-low-1` at 1e-7, and
   `0-1971` at 1e-8, equal 06b's units in every value 06b recorded, analytic blocks and W3
   included.
3. **The new code is deterministic.** `g-1e-7-0-14164` is identical in two fresh workers.

**The predictions.** [`06c/census/state/predictions.md`](06c/census/state/predictions.md), sha256
`2672dd5c…`, was written at 12:58:35Z, before the launch. Part A is the prompt's X1–X5 byte for
byte. Part B fixes X1's selection rule and the readings.

| | Measured | Verdict |
|---|---|---|
| **X1** | 24 drawn, every capture equal to the census call bit for bit; 1 dropped ($\|S_9 - S_{10}\|$ = 0.0215 `abserr`$_g$). Of 23 left, 22 within `abserr`$_g$ (0.002–0.99); `g-1e-7-0-14127` call 3 at **1.375**. Reference sound there ($S_{10}$–$S_{12}$ agree to $3\times10^{-5}$ `abserr`$_g$). | **fails** |
| **X2** | 06b's X2-type calls outside the 720 rows: 20,425 (1e-7) and 22,442 (1e-8); unconverged under g: **53** and **127**, almost all at depth 20 (with an untrusted leaf: 49, 63) or floor-limited (2, 37). | **fails** |
| **X3** | No call under g declares an infinite or NaN abserr, at either rtol. | **holds** |
| **X4** | 8,092 rows scored per rtol (468 with a non-finite `total_abserr` counted apart); **263** and **302** violate, up to 29.9× and 71.5×, evaluated exactly. In every violating row a group call disagrees beyond both declared errors. | **fails** |
| **X5** | Moved calls reaching depth 20: **109** (1e-7) and **153** (1e-8), mostly analytic blocks; none over budget by the dispatcher. | **fails** |

**Examined further** (descriptive, after the scores):
- **X1's failing call:** the driver at `rtol` $10^{-11}$ returns g's value again, so its error is
  common to a leaf and its descendants, which the trust check cannot reach. Its error is
  $1.3\times10^{-4}$ of the call's own request.
- **X4's violations, on the four worst rows' most-violating calls**, referenced as X1 is: g's error
  is 0.11–0.61 of `abserr`$_g$; 06b's is 116–445 of `abserr`$_{06b}$
  ([`06c/census/out/x4_reference.json`](06c/census/out/x4_reference.json)).

**Descriptive** ([`06c/census/out/cmd/44`](06c/census/out/cmd/44_report.txt)):
- **Convergence.** Rows with `total_converged` false: 13 and 45, against 06b's 2,349 and 2,388.
- **Depth.** Rows at depth 20: 63 and 81, against 0 and 3.
- **Cost.** Solves ×8.2 and ×8.4 over 06b. The trust check's own solves are 55–86 % of all solves.
- **The trust check.** Untrusted leaves found during calls: 0.70 M and 0.88 M. Untrusted leaves
  returned, all at depth 20: 1,115 and 1,406.
- **The widening.** abserr / unwidened $E$ has median 1 and p99 11–13.
- **Flat residuals.** 217 of 262 depth-20 lines are flat over their last five levels, with
  $(e_L + e_R)/e_P$ often 2.
- **06b's depth-20 calls.** All three now converge at depth 1–5.

**The store** is byte-identical to its certification before, during and after, and gives
`0e32f24a…`. 06b's checkpoint is unchanged (`f934ea24…`). No census process initialised Ray.
