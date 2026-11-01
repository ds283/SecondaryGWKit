#### §4 — Rehearsals R21 and R22 (Mac Studio, 2026-11-01)

**Trees.**
- **R21** ran at `2c325f5`, the parent of 06h's code commit `7172a8f`. Its code is 06e's
  (`f4d8e17`). It ran in the worktree `SecondaryGWKit-lpf06h-2c325f5`.
- **R22** ran at `c1ec3b2`, Part B's dispatch, whose code is `7172a8f` + `ac1c6f4`. It ran in the
  worktree `SecondaryGWKit-lpf06h-c1ec3b2`.
- **The command** in both: `RunDrivers/quadsource_atol_sweep.py --build --rehearsal --atol 1e-32
  --rtol 1e-7 --cpus 6`, as R19 and R20 ran it, with the same launcher (`b0135218…`).
- **Under the shared lock.** Records:
  [`06h/rehearsal/log.md`](06h/rehearsal/log.md) and [`06h/rehearsal/out/`](06h/rehearsal/out/).

**The builds** ([`r21_status.json`](06h/rehearsal/out/r21_status.json),
[`r22_status.json`](06h/rehearsal/out/r22_status.json)):

| run | id | tree | ended | fingerprint |
|---|---|---|---|---|
| R21 | `levin-phase-floor-06h-r21-parent-20261101T001852` | `2c325f5` | `done`, exit 0, 00:23:36Z | `1b332ffa…` |
| R22 | `levin-phase-floor-06h-r22-groupphase-20261101T002423` | `c1ec3b2` | `done`, exit 0, 00:28:57Z | `1b332ffa…` |

- **The fingerprints:** each equals R8's class by class (and R19's), and R22's `classes` equal
  R21's. `problems {}` in both. **§4.2 holds.**

**Outside `QuadSourceIntegral`** ([`classify_non_qsi.txt`](06h/rehearsal/out/classify_non_qsi.txt),
05's script unchanged, `26690e61…`):
- **Value tables:** all 7 identical, 360,462 rows.
- **Other differences:** timing, label, `parents_digest` (291 `GkSource`) and `rho_reuse` (24
  `GkWKBIntegration` rows). 0 unclassified.
- **Rows in one store only, or unmatched:** none. **Holds.**

**Inside `QuadSourceIntegral`, every row**
([`qsi_tabulation.txt`](06h/rehearsal/out/qsi_tabulation.txt),
[`qsi_rows.csv`](06h/rehearsal/out/qsi_rows.csv)), 864 rows, all at (atol, rtol) = (1e-32, 1e-7):

| | R21 | R22 |
|---|---|---|
| `total` bit-identical (`float.hex`) | 864 / 864 | |
| `total_abserr` bit-identical | 864 / 864 | |
| every value field bit-identical | 205 / 864 (all with no three-Bessel Levin block) | |
| `total_converged` | 864 | 864 |
| `total_phase_limited` | 195 | 195 |
| WKB Levin calls (rows with one) | 3,105 (573) | 3,105 (573) |
| `WKB_Levin_num_regions`, summed | 31,216 | 31,216 |
| `WKB_Levin_max_depth`, deepest; rows at 20 | 19; 0 | 19; 0 |

- **Region counts and depths.** `WKB_Levin_num_regions` is the sum of the calls' `num_regions`.
  `WKB_Levin_max_depth` is their maximum. Both are integer columns of the shard table
  `QuadSourceIntegral`, NULL with no WKB call. Per-call counts are in `metadata` at
  `WKB_Levin.subintervals[i].groups[j].regions`.
- **The analytic oracle's per-block counts** are at
  `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.regions`. No depth is stored for them.
- **Work counters,** not values: the regions, evaluations, simple-region, SVD, order-change,
  Chebyshev-order and max-depth columns, and the `regions`, `simple_regions` and `evaluations`
  leaves.

**Why `total` does not move.** `total` is `numeric_quad + WKB_Levin`. The WKB Levin groups are
`phase_groups.py`'s, which 06h does not change. `BesselPhaseGroup` is used only by
`_three_bessel_Levin`, the analytic oracle, which feeds `analytic_rad` and `analytic.*`, not
`total`.

**What moved** ([`qsi_analytic_summary.txt`](06h/rehearsal/out/qsi_analytic_summary.txt),
descriptive). 659 rows have a three-Bessel Levin block, 10,072 blocks in all:

| | R21 | R22 |
|---|---|---|
| `analytic_rad` bit-identical | 400 / 864 (moved in 464) | |
| max $\lvert\Delta\,{\rm analytic\_rad}\rvert / (e_{21}+e_{22})$, $e$ = `analytic.abserr` | $2.485\times10^{-5}$ (0 rows above 1) | |
| `analytic.abserr` R22/R21, by row | median 1, range 0.258–3.07 | |
| blocks converged | 10,072 | 10,072 |
| blocks phase-limited | 3,832 | 3,896 (F→T 584, T→F 520) |
| block regions, summed | 28,785 | 24,428 (−15.1%) |
| largest regions in one block | 14 | 11 |

**H0 — consistency.**
- **The rule:** every row outside 06a's 720 stationary-point rows with finite `total_abserr` in
  both stores has $\lvert T_{22}-T_{21}\rvert \le e_{21}+e_{22}$.
- **Stationary-point rows in the store:** 0. No rehearsal row is an A3 v2 row, by physical key
  against 06a's `rows.json` and its census's `scan`.
- **Scored:** 864 rows. **Non-finite:** 0.
- **Violations: 0.** $\lvert\Delta T\rvert = 0$ in every row, so the largest
  $\lvert\Delta T\rvert/e_{21}$ and $\lvert\Delta T\rvert/e_{22}$ are both 0.
- **H0 holds.** It holds trivially here: `total` cannot move under 06h in this table
  (log, Observation 1).

**R21 reproduces 06e's R20** ([`r21_vs_06e_r20_csv_check.txt`](06h/rehearsal/out/r21_vs_06e_r20_csv_check.txt)):
on all 864 rows, by `float.hex`, in `total`, `total_abserr`, convergence, regions, depth,
evaluations, `analytic_rad`, and the analytic regions and abserr.

**The readers on R22** ([`readers.txt`](06h/rehearsal/out/readers.txt),
[`readers_check.txt`](06h/rehearsal/out/readers_check.txt)): 6 of 6 exit 0. The store is
byte-identical before and after by SHA-256, size and mtime, with no journal. **Holds.**

**The lock** ([`lock.log`](06h/rehearsal/state/lock.log)): released. `status` reports free, and
`ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing. **Holds.**

**The stores are kept,** `levin-phase-floor-06h-r21*` and `…-r22*`, unchanged since the
comparison ([`stores_sha256_final.txt`](06h/rehearsal/state/stores_sha256_final.txt)).
