#### §4 — Rehearsals R23 and R24 (Mac Studio, 2026-11-01)

**Trees.**
- **R23** ran at `4f4f9ed`, the parent of 06g's code commit `3ae6f3a`. Its code is 06h's
  (`7172a8f` + `ac1c6f4`). It ran in the worktree `SecondaryGWKit-lpf06g-4f4f9ed`.
- **R24** ran at `1c5336e`, Part B's dispatch, whose code is `3ae6f3a`'s. It ran in the worktree
  `SecondaryGWKit-lpf06g-1c5336e`.
- **The command** in both: `RunDrivers/quadsource_atol_sweep.py --build --rehearsal --atol 1e-32
  --rtol 1e-7 --cpus 6`, as R21 and R22 ran it, with the same launcher (`b0135218…`).
- **Under the shared lock.** Records:
  [`06g/rehearsal/log.md`](06g/rehearsal/log.md) and [`06g/rehearsal/out/`](06g/rehearsal/out/).

**The builds** ([`r23_status.json`](06g/rehearsal/out/r23_status.json),
[`r24_status.json`](06g/rehearsal/out/r24_status.json)):

| run | id | tree | ended | fingerprint |
|---|---|---|---|---|
| R23 | `levin-phase-floor-06g-r23-parent-20261101T110641` | `4f4f9ed` | `done`, exit 0, 11:11:17Z | `1b332ffa…` |
| R24 | `levin-phase-floor-06g-r24-leafvalues-20261101T111215` | `1c5336e` | `done`, exit 0, 11:22:45Z | `1b332ffa…` |

- **The fingerprints:** each `classes` object equals R8's, R21's and R22's, and R24's equals
  R23's. `problems {}` in both. **Holds.**
- **R23 reproduces 06h's R22** on every tabulated column of all 864 `QuadSourceIntegral` rows
  ([`r23_vs_06h_r22_csv_check.txt`](06g/rehearsal/out/r23_vs_06h_r22_csv_check.txt)).

**Outside `QuadSourceIntegral`** ([`classify_non_qsi.txt`](06g/rehearsal/out/classify_non_qsi.txt),
05's script unchanged, `26690e61…`):
- **Value tables:** all 7 identical, 360,462 rows.
- **Other differences:** timing, label, `parents_digest` (291 `GkSource`) and `rho_reuse` (31
  `GkWKBIntegration` rows). 0 unclassified.
- **Rows in one store only, or unmatched:** none. **Holds.**

**Inside `QuadSourceIntegral`, every row**
([`qsi_tabulation.txt`](06g/rehearsal/out/qsi_tabulation.txt),
[`qsi_rows.csv`](06g/rehearsal/out/qsi_rows.csv),
[`qsi_analytic_summary.txt`](06g/rehearsal/out/qsi_analytic_summary.txt),
[`qsi_analytic_rows.csv`](06g/rehearsal/out/qsi_analytic_rows.csv)), 864 rows, all at (atol, rtol)
= (1e-32, 1e-7):

| | R23 | R24 |
|---|---|---|
| `total` bit-identical (`float.hex`) | 294 / 864 (291 rows with no WKB Levin call + 3) | |
| `total_abserr` bit-identical | 294 / 864 | |
| `analytic_rad` bit-identical | 228 / 864 (205 rows with no three-Bessel block + 23) | |
| `analytic.abserr` bit-identical | 378 / 864 | |
| `total_converged` | 864 | 797 |
| WKB Levin calls (rows with one) | 3,105 (573) | 3,105 (573) |
| WKB Levin calls unconverged | 0 | 97 |
| `total_phase_limited` | 195 | 478 |
| `WKB_Levin_num_regions`, summed | 31,216 | 134,011 |
| `WKB_Levin_max_depth`, deepest; rows at 20 | 19; 0 | 20; 76 |
| analytic-block regions, summed; blocks converged | 24,428; 10,072 / 10,072 | 29,228; 10,072 / 10,072 |

- **Where the quantities are stored.** Everything is in each shard's table `QuadSourceIntegral`:
  - `WKB_Levin_num_regions` is the sum of the calls' `num_regions`, and `WKB_Levin_max_depth` is
    their maximum. Both are integer columns, NULL with no WKB call.
  - The per-call counts are in `metadata` (a JSON string) at
    `WKB_Levin.subintervals[i].groups[j].regions`. **No per-call depth is stored.**
  - The oracle's value is the column `analytic_rad` (`Float(64)`), and its error is `metadata` at
    `analytic.abserr` (a JSON number).
  - The per-block counts are at `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.regions`. No depth is
    stored for them.
  - 06g's new result keys (`node_limited`, the signed node sums, the per-leaf trust) are not
    persisted.
- **Work counters,** not values: the regions, evaluations, simple-region, SVD, order-change,
  Chebyshev-order and max-depth columns, and the `regions`, `simple_regions` and `evaluations`
  leaves. 06g leaves these keys and their assignments untouched; it moves only the partition.
- **The stationary-point rows:** 0 of the 864. They are identified from 06a's census checkpoint
  (`c13685b4…`) by `scan06a`'s rule (720 asserted), mapped through 06a's `rows.json` (`aa523d28…`)
  by physical key. No rehearsal row is an A3 v2 row.

**V1 — consistency** (prompt 06g §4):

| | rows scored | violations | largest $\lvert\Delta\rvert/(e_{23}+e_{24})$ | non-finite | stationary-point rows |
|---|---|---|---|---|---|
| `total` against `total_abserr` | 864 | **12** | **13,101.5** ($k=10^5$, $z=5236.553$) | 0 | 0 |
| `analytic_rad` against `analytic.abserr` | 864 | 0 | 0.4986 ($k=316228$, $z=6904.72$) | 0 | 0 |

- **V1 FAILS** (on `total`). A violation is a finding for the user, not a stop.
  - The 12 rows, with both totals and both errors in `float.hex`, are in
    [`06g/rehearsal/log.md`](06g/rehearsal/log.md), "Predictions and verdicts".
  - In every one of them the whole move is in sub-interval 1's `G+q+r` phase group
    ([`v1_violations_detail.txt`](06g/rehearsal/out/v1_violations_detail.txt)).
  - In R23 that group has 2–12 regions and is converged. In R24 it has 39–208 regions, is
    phase-limited, and is unconverged at depth 20 in 10 of the 12.
  - The group's value moves by 0.1–3.4 % against declared errors near $10^{-7}$ of it, so at least
    one store's error understates the true error by $\ge 2{,}500\times$.
  - This rehearsal has no reference, so it cannot say which store is right.
- **Lost convergence (descriptive).** 67 rows (97 calls) went converged → unconverged, all in the
  `G±q±r` groups of sub-intervals 1 and 2. Every one of those rows reaches depth 20 in R24
  ([`qsi_lost_convergence_crosscount.txt`](06g/rehearsal/out/qsi_lost_convergence_crosscount.txt),
  [`qsi_unconverged_groups.txt`](06g/rehearsal/out/qsi_unconverged_groups.txt)).

**The readers** ([`readers_check.txt`](06g/rehearsal/out/readers_check.txt)):
- They ran on R24 from the code worktree, under the lock: six of six exit 0, with 0 Tracebacks.
- R24's five files are identical by SHA-256, size and mtime before and after, with no journal.
  **Holds.**

**The lock** ([`lock.log`](06g/rehearsal/state/lock.log)): released. `status` is free, and
`ray-processes` lists nothing before and after the release. **Holds.**

**The stores are kept:** `var/datastores/levin-phase-floor-06g-r23*` (primary `88cc355f…`) and
`…-r24*` (primary `6e4815c8…`), unchanged since the comparison.
