#### §5 — Rehearsals R27 and R28 (Mac Studio, 2026-11-08)

**Trees.**
- **R27** ran at `91472dbe`, the parent of 07's code commit `ad732887`. Its production code is 06m's
  (`levin_quadrature.py` `beaf0e23…`, `QuadSourceIntegral.py` `962424c0…`); against R26's tree `38f6a828` only
  `CLAUDE.md` differs. Worktree `SecondaryGWKit-lpf07-91472db`.
- **R28** ran at `74600350`, Part B's dispatch, whose code is `ad732887`'s (`levin_quadrature.py` `dc05596c…`,
  `QuadSourceIntegral.py` `33cb161e…`). Worktree `SecondaryGWKit-lpf07-7460035`.
- **The command** in both: `RunDrivers/quadsource_atol_sweep.py --build --rehearsal --atol 1e-32 --rtol 1e-7
  --cpus 6 --job-name levin-phase-floor-07`, as R25 and R26 ran it, with the same launcher (`b0135218…`).
- **Under the shared lock.** Records: [`07/rehearsal/log.md`](07/rehearsal/log.md) and
  [`07/rehearsal/out/`](07/rehearsal/out/).

**The builds** ([`r27_status.json`](07/rehearsal/out/r27_status.json),
[`r28_status.json`](07/rehearsal/out/r28_status.json)):

| run | id | tree | ended | fingerprint |
|---|---|---|---|---|
| R27 | `levin-phase-floor-07-r27-parent-20261108T000634` | `91472dbe` | `done`, exit 0, 00:15:41Z | `1b332ffa…` |
| R28 | `levin-phase-floor-07-r28-guard-20261108T001916` | `74600350` | `done`, exit 0, 00:35:54Z | `1b332ffa…` |

- Each `classes` object equals R26's and R25's; `problems {}` in both. **Holds.**

**S0 — R27 against R26** ([`classify_non_qsi_r26_r27_labels_normalised.txt`](07/rehearsal/out/classify_non_qsi_r26_r27_labels_normalised.txt),
[`qsi_identity_r26_r27.txt`](07/rehearsal/out/qsi_identity_r26_r27.txt),
[`s0_labels_and_qsi_tool_report.txt`](07/rehearsal/out/s0_labels_and_qsi_tool_report.txt)):
- **Value tables:** all 7 identical, 360,462 rows (BackgroundModel 1,159; TkNumericIntegration 1,478;
  TkWKBIntegration 2,209; GkNumericIntegration 35,874; GkWKBIntegration 147,642; GkSource 169,653; QuadSource 2,447).
- **`QuadSourceIntegral`:** all 864 rows bit-identical (`float.hex`) in every value field (118,776) and every
  counter field (27,163): every column and every flattened `metadata` leaf, `total`, `total_abserr`,
  `total_converged`, `analytic_rad`, `analytic.abserr` and every group and analytic-block record included.
- **Other differences:** timings, labels, `parents_digest` (291 `GkSource`), `rho_reuse` (26 `GkWKBIntegration`).
  No row is in one store only.
- **The labels.** The job names differ; 05's classifier, unchanged (`26690e61…`), reports 5,148 UNCLASSIFIED on
  the raw JSON ([`classify_non_qsi_r26_r27.txt`](07/rehearsal/out/classify_non_qsi_r26_r27.txt)). All 6,012
  differing labels differ only by the prefix (`levin-phase-floor-06m-` / `-07-`) and the stamp; normalised in a
  copy, 0 are unclassified.
- **Holds:** the rehearsal is reproducible.

**R28 against R27, outside `QuadSourceIntegral`** ([`classify_non_qsi.txt`](07/rehearsal/out/classify_non_qsi.txt),
05's script unchanged on the raw JSON):
- **Value tables:** all 7 identical, 360,462 rows. **Other differences:** timing, label, `parents_digest` (291),
  `rho_reuse` (26). 0 unclassified; no row in one store only. **Holds.**

**Inside `QuadSourceIntegral`** ([`qsi_tabulation.txt`](07/rehearsal/out/qsi_tabulation.txt),
[`qsi_rows.csv`](07/rehearsal/out/qsi_rows.csv), [`qsi_guard_records.csv`](07/rehearsal/out/qsi_guard_records.csv),
[`qsi_rows_summary.txt`](07/rehearsal/out/qsi_rows_summary.txt)):
- **The classification.** A row is untouched iff every one of R28's four counts is 0 in every Levin record of the
  row: the group records `metadata` `WKB_Levin.subintervals[i].groups[j]` and the analytic blocks
  `metadata` `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data`, leaves `stationary_points` (the number located),
  `window_pieces`, `guarded_regions`, `guard_at_depth_max`, each a JSON int. They are asserted present in all
  3,105 group records and 10,072 analytic blocks of R28, absent from R27, and absent elsewhere.
- **Untouched rows: 640, all 640 bit-identical** to R27 in every value field (and every counter field), the four
  counts set aside. **Holds** (no stop).
- **The other 224 rows**, tabulated: a group call located a point in 70 (44 `G+q+r` records, 1 point each;
  26 `q-r` records with `q = r`, 4,097 each); an analytic block in 200 (400 blocks, the `2pt5` J4 and Y4, 1 each).
  **No leaf was guarded** and none reached `guard_at_depth_max`, anywhere.
- **Bit-identical, all 864 rows:** `total` 796, `total_abserr` 794, `analytic_rad` 672, `analytic.abserr` 664.
- **The largest change against `total_abserr`:** 7.38e4 (R27) and 6.42e4 (R28), at (1e6, 1e6, 1e6, z 2284.25).
  16 rows move by more than both errors, all G+q+r rows; S1 and S2 show R28 right on the referenced ones.
- **`total_converged`:** 845 → 862; **17 rows unconverged → converged, 0 the other way** (listed in
  `qsi_tabulation.txt`). Rows where a group call located a point: 52 of 70 converged in R27, 69 in R28.
- **Where the quantities are stored** (each shard's table `QuadSourceIntegral`): `total`, `total_abserr`,
  `analytic_rad` (`Float(64)` columns); `total_converged` (Boolean, SQLite 0/1, the AND of the group records'
  `converged`); `WKB_Levin_num_regions` (Integer, the sum of the group calls' `num_regions`, NULL with none) and
  `WKB_Levin_max_depth` (Integer, their maximum; no per-call depth is stored, none for the analytic blocks);
  per call, `regions`, `abserr`, `converged`, `phase_limited` (and `value` for groups) at the metadata paths
  above; the oracle's error `metadata` `analytic.abserr`.
- **Work counters**, not values: the regions, evaluations, simple-region, SVD, order-change, Chebyshev-order and
  max-depth columns, and the `regions`, `simple_regions`, `evaluations` leaves (unchanged in meaning: no +/- line
  of the `QuadSourceIntegral` diff names them).

**The located points, kind and z: the replay** ([`replay_check.txt`](07/rehearsal/out/replay_check.txt),
[`replay_calls.csv`](07/rehearsal/out/replay_calls.csv), [`replay_summary.txt`](07/rehearsal/out/replay_summary.txt)):
- 06g's `references/v1/scripts/` capture harness, copied, changed where marked (the trees, the stores through
  symbolic links, and `guard_07`: each call's own guard record; [`capture_harness_diff.txt`](07/rehearsal/out/capture_harness_diff.txt)),
  re-ran all 224 touched rows at R28's tree, Ray-free, the stores read-only.
- **470 of 470 records reproduce R28 bit for bit** (value and abserr in stored form, converged, phase_limited,
  regions, the four counts; every row's `total` and `analytic_rad`).
- **G+q+r:** 44 roots, at z = 10211.3178915 (42 at k = q = r) and z = 275.0085917 (2).
  **Analytic:** 400 roots, at eta = 8.13e-5, 2.57e-5, 8.13e-6 (k = 1e5, 3.162e5, 1e6). No minima; no interior end.
- **The 26 `q-r` calls with `q = r`:** each 4,097 points, every one a sample of the scan grid; theta' exactly 0.0 at
  all 4,097 samples, and theta exactly 0.0 (issue `[07-location-on-an-identically-zero-rate]`).

**The `q-r` references** ([`qr_scores.txt`](07/rehearsal/out/qr_scores.txt), [`qr_reference/`](07/rehearsal/out/qr_reference/)):
- **Method.** theta = 0, so the integrand does not oscillate: composite 20-point Gauss–Legendre on 2^m x-panels
  (m = 6–14, fsum), GL14 the reference, with QAGS (epsrel 1.2e-14) as a second rule; stated error
  max(|GL14 − GL12|, |GL14 − QAGS|). Each capture replays its record bit for bit; R27's and R28's integrands are
  bit-identical at 1,001 points.

| call | reference | stated error | R27 \|v−ref\|/abserr | R28 \|v−ref\|/abserr |
|---|---|---|---|---|
| (1e5, 1e6, 1e6, z 6.985e9) | `-0x1.a95ce1e6e5fdep-75` | 4.56e-36 | 0.00215 | **0.179** |
| (3.162e5, 1e6, 1e6, z 1.363e10) | `-0x1.3166237fddf59p-77` | 2.64e-38 | 0.00142 | **0** |
| (1e5, 3.162e5, 3.162e5, z 3.047e9) | `0x1.025f739eec8c8p-74` | 2.23e-37 | 0.000618 | **0.0695** |

- **On the worst call, R28's verdict rests on the Gauss–Legendre sequence, not on the stated error**
  ([`qr_worst_call_detail.txt`](07/rehearsal/out/qr_worst_call_detail.txt)). R28 declares 7.89e-37.
  - GL at m = 8, 10, 12, 14 agree with one another to 5.9e-39 (GL12 = GL14 exactly), and R28 is 0.186, 0.186,
    0.179, 0.179 of its abserr from them.
  - The stated error, 4.56e-36 = 5.8 × R28's abserr, is set entirely by QAGS, which is 4.9 times its own
    estimate (9.29e-37) away from GL14. Against QAGS, R28 is 5.6 × its abserr out.
  - So the verdict "honest" holds against GL14 and its converged sequence. Against the stated error it is
    undecided: |v − ref| plus the reference's error exceeds R28's abserr. On the other two calls the stated error
    is below R28's abserr (by 6.5× and 45×), and the verdict is honest outright. R27 is honest on all three by
    any reading.

**S1 — V1's twelve** ([`s1_scores.txt`](07/rehearsal/out/s1_scores.txt)): each w call (sub-interval 1, `G+q+r`)
locates one root at z = 10211.31789148–10211.31789150, within 8.5e-6 of 10211.3179 (one scan spacing there:
31–44 in z), and |v28 − ref|/abserr28 against 06g's `ref_30` is 0.036, 0.128, 0.035, 0.073, 0.00037, 0.019, 0.067,
0.0030, 0.088, 0.126, 0.092, **0.218** (w11); all 12 converged. **Holds.**

**S2 — P1's two rows** ([`s2_scores.txt`](07/rehearsal/out/s2_scores.txt), run
`levin-phase-floor-07-p1-references-20261108T103132`, `done`):
- captured at R27's tree, each capture bit for bit against R27's stored group ([`captures.json`](07/rehearsal/out/captures.json));
  four readings by `tools/filon_reference.py`, Route B, counting at min(abserr_R27, abserr_R28)/100.

| row | located (z) | \|x − x_s\| (spacing) | scatter / tolerance | reference (FCC r2) | R28 ratio | R27 ratio |
|---|---|---|---|---|---|---|
| 1, (3.162e5)^3, z 250 | root, 10211.3178915 | 7.1e-9 (0.0041) | 6.1e-37 / 6.6e-33 | `-0x1.059fce809a3c4p-74` | **0.020** | 5.6e5 |
| 2, (1e5, 3.162e5, 3.162e5), z 250 | root, 275.0085917 | 5.6e-9 (0.0041) | 1.2e-36 / 9.2e-33 | `0x1.2f1af86ca7aa5p-75` | **0.263** | 4.1e4 |

- **Holds** on both rows, both rules. R27 reproduces P1's failure (5.6e5 and 4.1e4 of its abserr).

**The readers** ([`readers_check.txt`](07/rehearsal/out/readers_check.txt)): on R28 from the code worktree under
the lock, six of six exit 0, 0 Tracebacks; R28's five files identical by SHA-256, size and mtime before and after,
no journal. **Holds.**

**The lock** ([`lock.log`](07/rehearsal/state/lock.log)): released; `status` free, `ray-processes` empty before and
after. **Holds.**

**The stores are kept:** R27 (primary `d6fd3023…`) and R28 (primary `00468d55…`), unchanged since the comparison
through the replay, the captures and the readers ([`stores_sha256_final.txt`](07/rehearsal/state/stores_sha256_final.txt));
R26 unchanged throughout.
