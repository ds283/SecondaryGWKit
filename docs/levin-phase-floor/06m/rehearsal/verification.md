#### §4 — Rehearsals R25 and R26 (Mac Studio, 2026-11-06)

**Trees.**
- **R25** ran at `5ff8a5e4`, the parent of 06m's code commit `48cf87c0`. Its production code is
  R24's (06g's `3ae6f3a`; `levin_quadrature.py` `d89c959e…`). It ran in the worktree
  `SecondaryGWKit-lpf06m-5ff8a5e`.
- **R26** ran at `38f6a828`, Part B's dispatch, whose code is `48cf87c0`'s (`levin_quadrature.py`
  `beaf0e23…`). It ran in the worktree `SecondaryGWKit-lpf06m-38f6a82`.
- **The command** in both: `RunDrivers/quadsource_atol_sweep.py --build --rehearsal --atol 1e-32
  --rtol 1e-7 --cpus 6 --job-name levin-phase-floor-06m`, as R23 and R24 ran it, with the same
  launcher (`b0135218…`).
- **Under the shared lock.** Records:
  [`06m/rehearsal/log.md`](06m/rehearsal/log.md) and [`06m/rehearsal/out/`](06m/rehearsal/out/).

**The builds** ([`r25_status.json`](06m/rehearsal/out/r25_status.json),
[`r26_status.json`](06m/rehearsal/out/r26_status.json)):

| run | id | tree | ended | fingerprint |
|---|---|---|---|---|
| R25 | `levin-phase-floor-06m-r25-parent-20261106T092420` | `5ff8a5e4` | `done`, exit 0, 09:36:10Z | `1b332ffa…` |
| R26 | `levin-phase-floor-06m-r26-signedyardstick-20261106T093937` | `38f6a828` | `done`, exit 0, 09:49:23Z | `1b332ffa…` |

- **The fingerprints:** each `classes` object equals R24's and R23's. `problems {}` in both.
  **Holds.**

**R25 against R24 — reproducibility**
([`classify_non_qsi_r24_r25_labels_normalised.txt`](06m/rehearsal/out/classify_non_qsi_r24_r25_labels_normalised.txt),
[`qsi_identity_r24_r25.txt`](06m/rehearsal/out/qsi_identity_r24_r25.txt)):
- **Value tables:** all 7 identical, 360,462 rows.
- **`QuadSourceIntegral`:** all 864 rows bit-identical (`float.hex`) in every value field (118,776)
  and every counter field (27,163). That includes `total`, `total_abserr`, `analytic_rad`,
  `analytic.abserr` and `total_converged`.
- **Other differences:** timings, labels, `parents_digest` (291) and `rho_reuse` (27 rows). No row
  is in one store only.
- **The labels.** R24 and R25 have different job names (`levin-phase-floor-06g` and
  `levin-phase-floor-06m`).
  - 05's classifier, unchanged, reports them UNCLASSIFIED on the raw JSON
    ([`classify_non_qsi_r24_r25.txt`](06m/rehearsal/out/classify_non_qsi_r24_r25.txt)).
  - All 6,012 differ only by that prefix and the stamp. With the prefix normalised, 0 differences
    are unclassified (log, deviation 1).
- **R25 equals R24. Holds:** the rehearsal is reproducible.

**Outside `QuadSourceIntegral`, R26 against R25**
([`classify_non_qsi.txt`](06m/rehearsal/out/classify_non_qsi.txt), 05's script unchanged on the raw
JSON, `26690e61…`):
- **Value tables:** all 7 identical, 360,462 rows.
- **Other differences:** timing, label, `parents_digest` (291 `GkSource`) and `rho_reuse` (24
  `GkWKBIntegration` rows). 0 unclassified.
- **Rows in one store only, or unmatched:** none. **Holds.**

**Inside `QuadSourceIntegral`, every row**
([`qsi_tabulation.txt`](06m/rehearsal/out/qsi_tabulation.txt),
[`qsi_rows.csv`](06m/rehearsal/out/qsi_rows.csv)), 864 rows, all at (atol, rtol) = (1e-32, 1e-7):

| | R25 | R26 |
|---|---|---|
| `total` bit-identical (`float.hex`) | 413 / 864 (291 rows with no WKB Levin call + 122) | |
| `total_abserr` bit-identical | 403 / 864 | |
| `analytic_rad` bit-identical | 860 / 864 | |
| `analytic.abserr` bit-identical | 848 / 864 | |
| `total_converged` | 797 | 845 |
| WKB Levin calls (rows with one) | 3,105 (573) | 3,105 (573) |
| WKB Levin calls unconverged | 97 | 20 |
| `total_phase_limited` | 478 | 464 |
| `WKB_Levin_num_regions`, summed | 134,011 | 114,826 |
| `WKB_Levin_max_depth`, deepest; rows at 20 | 20; 76 | 20; 23 |
| analytic-block regions, summed; blocks converged | 29,228; 10,072 / 10,072 | 29,199; 10,072 / 10,072 |

- **`total_converged` changes:** **48 rows unconverged → converged, 0 converged → unconverged.**
  Each row is listed in `qsi_tabulation.txt`.
  - All 48 were at depth 20 in R25 (73 unconverged calls between them).
  - In R26 they are at depth 11–20, 2 of them still at 20, and all their calls converge.
- **Regions:** fewer in 442 rows, more in none.
- **Maximum depth:** shallower in 275 rows, deeper in none.
- **Where the quantities are stored.** Everything is in each shard's table `QuadSourceIntegral`.
  - **Region counts:** `WKB_Levin_num_regions` is the sum of the calls' `num_regions`, and
    `WKB_Levin_max_depth` is their maximum. Both are integer columns, NULL with no WKB call.
  - **Per call:** the counts are in `metadata` (a JSON string) at
    `WKB_Levin.subintervals[i].groups[j].regions`. **No per-call depth is stored.**
  - **`total_converged`** is a Boolean column (SQLite 0/1), the AND of the group flags.
  - **The oracle:** its value is the column `analytic_rad` (`Float(64)`), and its error is
    `metadata` at `analytic.abserr` (a JSON number).
  - **The oracle's blocks:** the per-block counts are at
    `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.regions`. No depth is stored for them.
  - $\nu_B$ and per-leaf trust are not persisted.
- **Work counters,** not values:
  - the regions, evaluations, simple-region, SVD, order-change, Chebyshev-order and max-depth
    columns;
  - the `regions`, `simple_regions` and `evaluations` leaves.

  06m's diff leaves these keys and their assignments untouched (checked on its +/- lines). It moves
  only the partition.
- **The stationary-point rows:** 0 of the 864.
  - They are identified from 06a's census checkpoint (`c13685b4…`) by `scan06a`'s rule (720
    asserted), mapped through 06a's `rows.json` (`aa523d28…`) by physical key, as 06g's rehearsal
    agent identified them.
  - No rehearsal row is an A3 v2 row.

**R1 — consistency** (prompt 06m §4):

| | rows scored | violations | largest $\lvert\Delta\rvert/(e_{25}+e_{26})$ | non-finite | stationary-point rows |
|---|---|---|---|---|---|
| `total` against `total_abserr` | 864 | **0** | **$4.93\times10^{-6}$** | 0 | 0 |
| `analytic_rad` against `analytic.abserr` | 864 | **0** | **$6.68\times10^{-8}$** ($k=q=r=10^6$, $z=996.42$) | 0 | 0 |

- **R1 HOLDS** on both.
- **The largest single-store ratios:**
  - $\lvert\Delta T\rvert/e_{25} = 9.87\times10^{-6}$, at $k=10^6$, $q=10^5$, $r=10^6$,
    $z=573.11$;
  - $\lvert\Delta\,{\rm rad}\rvert/e = 1.34\times10^{-7}$.
- $\lvert\Delta T\rvert/\lvert T_{25}\rvert \le 2.3\times10^{-10}$ over all rows.

**The readers** ([`readers_check.txt`](06m/rehearsal/out/readers_check.txt)):
- They ran on R26 from the code worktree, under the lock: six of six exit 0, with 0 Tracebacks.
- R26's five files are identical by SHA-256, size and mtime before and after, with no journal.
  **Holds.**

**The lock** ([`lock.log`](06m/rehearsal/state/lock.log)): released. `status` is free, and
`ray-processes` lists nothing before and after the release. **Holds.**

**The stores are kept:** `var/datastores/levin-phase-floor-06m-r25*` (primary `d74ac31d…`) and
`…-r26*` (primary `e90e26c2…`), unchanged since the comparison. R24's files are unchanged
throughout (`06m/rehearsal/state/stores_sha256_final.txt`).
