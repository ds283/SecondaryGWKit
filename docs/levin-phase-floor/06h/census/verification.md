### 6h.x The census: 06e's census re-run under 06h's group phase (Mac Studio, 2026-11-01)

**What was run.** Run `levin-phase-floor-06h-census-20261101T005037` re-ran every A3 v2
`QuadSourceIntegral` row (9,280) at `rtol` 1e-7 and 1e-8, in-process and Ray-free.
- **Tree:** $h$, the worktree at `c1ec3b2`, whose code is 06h's `7172a8f` + `ac1c6f4`.
- **Store:** 06's certified scratch copy of A3 v2, read-only.
- **Comparison:** against 06e's run `levin-phase-floor-06e-census-20261030T081318`.
- **Outcome:** 18,560 of 18,560 units `done`; none in error, none over budget, none past 300 s.
  The run took 00:50–03:24Z, against 06e's 1 h 47 min. The difference is post-timed
  instrumentation; the unit wall is ×0.97 and ×0.98 of 06e's.
- **Harness:** 06e's, changed as the log's deviations 1–9, 15 and 16 say. The diff is
  [`06h/census/out/scripts_diff_against_06e_final.txt`](06h/census/out/scripts_diff_against_06e_final.txt).

**Gate H holds** ([`06h/census/out/cmd/20b`](06h/census/out/cmd/20b_gate_h_check_2.txt),
[`21b`](06h/census/out/cmd/21b_gate_h_git_clause_2.txt); the first run, on the worker before the
oracle was added, [`20`](06h/census/out/cmd/20_gate_h_check.txt) and
[`21`](06h/census/out/cmd/21_gate_h_git_clause.txt), also holds):
1. **The parent reproduces 06e.**
   - `6d81fad..2c325f5` outside `docs/` and `prompts/` changes `CLAUDE.md` only.
   - At `2c325f5`, `low-G+r-1` and `ctrl-low-1` at 1e-7, and `0-1971` at 1e-8, equal 06e's units
     bit for bit in every value 06e recorded. The totals are `0x1.5e6b398fe7c40p-101`,
     `-0x1.e3272e29c8b71p-102` and `0x1.6ad248e2d3ab7p-66`.
2. **The new code is deterministic.** `g-1e-7-0-14164` is identical in two fresh workers on every
   key, 06h's new ones included. Its 16 analytic-block calls each carry a reproduced, non-zero
   declared term.

**The predictions.** [`06h/census/state/predictions.md`](06h/census/state/predictions.md), sha256
`6ff4f05b…`, was written at 00:49:57Z, before the launch. Part A is the prompt's H1–H4 byte for
byte. Part B fixes the readings, H3's second reading and H4's split by kind.

| | Measured | Verdict |
|---|---|---|
| **H3** (as written, on `total`) | 8,560 rows scored at each rtol, 0 violations, exact. Every `total` and `total_abserr` is bit-identical to 06e's: 06h's code feeds only the analytic oracle. | **holds** |
| **H3, second reading** (the analytic oracle) | 8,560 rows scored at each rtol, 0 violations. 06e's oracle abserr comes from a counterfactual recomputed with the parent's `BesselPhaseGroup`, whose value equals 06e's recorded `analytic_rad` bit for bit on every row. Max $\|a_h-a_{06e}\|/(e_{06e}+e_h)$: 0.0756 (1e-7), 1.35e-4 (1e-8). | **holds**; agrees with the first reading |
| **H4** | Scored per rtol outside the 720: 95,896 analytic-block and 33,725 group calls. Lost: 0 at 1e-7; **1 at 1e-8**, `g-1e-8-3-1072` call 13 "analytic J1": 06e converged at depth 1, $h$ unconverged at depth 0, abserr 2.36e-26 against a request of 1.63e-26. The round-off exit's premise held at the root under $h$, and failed there under 06e. Gained: 0. | **fails** (1e-8) |

**Descriptive** ([`06h/census/out/cmd/46`](06h/census/out/cmd/46_analyse06h.txt),
[`47`](06h/census/out/cmd/47_h4_detail.txt), [`49`](06h/census/out/cmd/49_nearsingular.txt)):
- **Group calls:** all 37,325 per rtol are bit-identical to 06e's.
- **Analytic calls:**
  - regions ×0.876 and ×0.870, solves ×0.872 and ×0.866;
  - shallower: maximum depth 8 against 06e's 9 at 1e-7, and 6 (unchanged) at 1e-8;
  - unconverged: 0 / 0 at 1e-7 and 1 / 0 at 1e-8 ($h$ / 06e).
- **The declared term**, per analytic call, against its request:
  - median 5.3e-6 and 6.3e-5, max 0.034 and 0.341;
  - none at or above the request;
  - up to 1.9e4 times what the constituents alone (06e's declaration) would charge on the same
    leaves; the abscissa part's median share is 0.26.
- **On the Clenshaw–Curtis branch** (06f's method): declared-dominated calls 2,493 → 5,137 (1e-7) and
  2,785 → 5,939 (1e-8); D ≥ 0.1 request on 0 → 0 and 1 → 2 calls.
- **The 720 stationary-point rows:** no total moved; the oracle moved on 462 and 421 rows, all within
  the bound; no convergence change.
- **The near-singular class is reached** (Part B's rule): 435 rows, all (+,+,−) with q = r ≫ k. The
  oracle moved by at most 5e-14 of its bound there. The oracle is unconverged in 302 of these rows
  under both codes.

**Store and before:** the copy is byte-identical to its certification throughout (`0e32f24a…`).
06e's checkpoint is `a0dc4d7f…` throughout. No worker started Ray. The run (2.3 GB, checkpoint
`3104c18e…`) is kept in full on the Mac Studio for 06g.
