#### §5 resumed (§10, Rule A) — Rehearsal R29 (Mac Studio, 2026-11-08)

**Tree.** R29 ran at `e182bc7a`, Part B's resumed dispatch, whose code is `adffa21b` (Rule A on 07's `ad732887`;
`levin_quadrature.py` `4e5da371…`, `QuadSourceIntegral.py` `33cb161e…`, unchanged from R28's), in the worktree
`SecondaryGWKit-lpf07-e182bc7`, with R28's command line (launcher `b0135218…`). R27 and S0 stand; R28's store and
worktree are read-only records of the zero-rate case. Records: [`07/rehearsal/r29/log.md`](07/rehearsal/r29/log.md),
[`07/rehearsal/r29/out/`](07/rehearsal/r29/out/).

**The build** ([`r29_status.json`](07/rehearsal/r29/out/r29_status.json)):
`levin-phase-floor-07-r29-guard-rule-a-20261108T123429`, `done`, exit 0, 12:44:05Z, fingerprint `1b332ffa…`,
`problems {}`, `classes` equal to R28's, R27's and R26's. **Holds.**

**Rule A's key** `location_skipped_phases` does not reach the store (`QuadSourceIntegral.py` is unchanged); a skipped
phase shows in the store only as its record's four counts at 0.

**R29 against R28** ([`r29_vs_r28.txt`](07/rehearsal/r29/out/r29_vs_r28.txt),
[`r29_vs_r28_records.csv`](07/rehearsal/r29/out/r29_vs_r28_records.csv),
[`classify_non_qsi_r28_r29.txt`](07/rehearsal/r29/out/classify_non_qsi_r28_r29.txt)):
- Outside `QuadSourceIntegral`: all 7 value tables identical (360,462 rows), 0 unclassified.
- Inside it, record by record over 13,177 Levin records: 13,087 bit-identical to R28 (the four counts included);
  **90 changed**, every one with R28 `stationary_points` > 0 and R29 `stationary_points` 0:
  - **86 equal R27 bit for bit** in value, abserr, converged, phase_limited and regions, with R29's four counts 0: the
    26 `q-r` calls with q = r (V = 0) and 60 analytic `2pt5` J4/Y4 blocks (V 0.49–15.9, from the replay);
  - **4 are guarded** (`guarded_regions` 1): the J4 and Y4 `2pt5` blocks at (3.162e5)³, z 1.3291e9, and (1e6)³,
    z 4.0176e9 — Rule A skipped the phase (V 15.11, 15.90) while the C2 gate's estimate on the root leaf (19.16,
    20.20) exceeded SIX_PI, so the root leaf went to the Levin solve and the guard caught the root. Ruled findings, not
    a stop (the user, at R29's stop).
- 54 rows differ; each holds a changed record; R29's `total` and `total_abserr` equal R27's in all 54.

**The four blocks** ([`block_scores.txt`](07/rehearsal/r29/out/block_scores.txt), [`blocks.json`](07/rehearsal/r29/out/blocks.json)):
each block replayed at R27's, R28's and R29's trees, each replay reproducing its store's record (abserr, converged,
phase_limited, regions, four counts) and its capture's value bit for bit; the three trees' integrands bit-identical at
1,001 points. Reference: composite 20-point Gauss–Legendre on 2^m x-panels each side of the root (m = 6–14), GL14
the reference, QAGS (epsrel 1.2e-14, split at the root) a second rule; stated error max(|GL14 − GL12|, |GL14 − QAGS|).

| block | reference | stated error | R27 v, e, ratio | R28 v, e, ratio | R29 v, e, ratio |
|---|---|---|---|---|---|
| (3.162e5)³ J4 | -0x1.9017c8e4681f2p-30 | 0x1.7p-77 | -0x1.9017c8e468225p-30, 0x1.4fd2c038b843dp-63, 7.4e-5 | -0x1.9017c8e4681f6p-30, 0x1.65dacd1543ce8p-66, 4.4e-5 | -0x1.9017c8e468206p-30, 0x1.9778b3736d5e1p-59, 1.5e-6 |
| (3.162e5)³ Y4 | 0x1.2018ed31b4477p-27 | 0x1.0p-77 | 0x1.2018ed31b4474p-27, 0x1.79ab800000000p-62, 1.6e-5 | 0x1.2018ed31b447ep-27, 0x1.78af331f037f8p-66, 5.8e-4 | 0x1.2018ed31b4472p-27, 0x1.aa161bfb153afp-66, 3.7e-4 |
| (1e6)³ J4 | -0x1.e6f859d2f7c57p-33 | 0x1.3p-81 | -0x1.e6f859d2f7bf8p-33, 0x1.c680c05f3602ep-66, 1.0e-4 | -0x1.e6f859d2f7c72p-33, 0x1.ff3cae87aca98p-69, 2.1e-4 | -0x1.e6f859d2f7c38p-33, 0x1.2c82764608ef0p-62, 3.1e-6 |
| (1e6)³ Y4 | 0x1.9714e016a7978p-30 | 0x1.8p-80 | 0x1.9714e016a7978p-30, 0x1.1045c00003c2dp-63, 0 | 0x1.9714e016a7982p-30, 0x1.faa21d2682f6bp-69, 6.2e-4 | 0x1.9714e016a7979p-30, 0x1.332dd100e35e5p-68, 5.1e-5 |

- **All three stores honest on all four blocks, outright:** the stated error is below every store's abserr (by at least
  1,990×). V (trapezoid on the 4,097-point scan, equal to R29's `location_skipped_phases`): 15.112984154245268 and
  15.896291130748331; the gate's span estimate on the root leaf (mean|θ′| on its 24-node Chebyshev grid × width): 19.159572454953185
  and 20.195797752642036, against SIX_PI 18.84955592153876.

**R29 against R27** ([`qsi_tabulation_r27_r29.txt`](07/rehearsal/r29/out/qsi_tabulation_r27_r29.txt),
[`qsi_rows_r27_r29.csv`](07/rehearsal/r29/out/qsi_rows_r27_r29.csv)): outside `QuadSourceIntegral` all value tables
identical; **692 untouched rows, all bit-identical**; 172 tabulated (44 with a group call that located, 168 with an
analytic block that located, 2 with a guarded block); `total` bit-identical in 820 / 864; largest move 7.38e4 × R27's
`total_abserr` (as R28); `total_converged` 845 → 862, **17 gained, 0 lost** (R28's 17).

**Cost:** wall time R27 about 9 min, R28 about 16.5 min, R29 about 9.6 min; summed `WKB_Levin_num_regions` (= summed
group `regions`) 114,826 / 227,883 / **114,460**.

**The replay** ([`replay_check.txt`](07/rehearsal/r29/out/replay_check.txt), [`replay_calls.csv`](07/rehearsal/r29/out/replay_calls.csv),
[`replay_summary.txt`](07/rehearsal/r29/out/replay_summary.txt)): 224 rows at R29's tree; **470 of 470 records
reproduce R29 bit for bit** (the 384 with a nonzero count — 44 G+q+r roots at z 10211.318 / 275.009, 336 analytic J4/Y4
roots, 4 guarded blocks — and the 86 others that changed against R28). 423 replayed calls report a skipped phase.

**S1** ([`s1_scores.txt`](07/rehearsal/r29/out/s1_scores.txt)): **holds**, 12 of 12 located at z 10211.3179 ± 9e-6
and honest, largest 0.218 (w11); R29's w calls equal R28's bit for bit.
**S2** ([`s2_scores.txt`](07/rehearsal/r29/out/s2_scores.txt)): **holds** against the standing `p1-references`
readings: row 1 0.020, row 2 0.263; R29's values equal R28's bit for bit.

**The readers** ([`readers_check.txt`](07/rehearsal/r29/out/readers_check.txt)): six of six exit 0; R29 byte-identical
before and after. **The lock** ([`lock.log`](07/rehearsal/r29/state/lock.log)): released; `status` free;
`ray-processes` empty. **The stores** R27, R28, R29 (primary of R29 in `state/stores_sha256_final.txt`) unchanged since
R29's comparison; R26 unchanged.
