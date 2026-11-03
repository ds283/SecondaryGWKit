# 06g §5, the references — verification (Mac Studio, 2026-11-01 to 2026-11-03)

For `docs/levin-phase-floor/VERIFICATION.md` §6g. Links are to `06g/references/…`. Every command and
its output is in [`out/cmd/`](06g/references/out/cmd/) and [`out/commands.log`](06g/references/out/commands.log).

## The harness and the trees

- **The census harness is the census's, byte for byte:** the nine files of `worker.py`'s `SCRIPTS` in
  `scripts/` equal the single `scripts_sha256` set recorded in all 18,560 records of the g census
  `levin-phase-floor-06g-census-20261101T114315` (`run_census.py` `5f21f072…`)
  ([`out/cmd/12`](06g/references/out/cmd/12_harness_equals_census.txt)); every capture worker recorded the
  same set ([`out/check_captures.json`](06g/references/out/check_captures.json)).
- **Trees** ([`out/cmd/10`](06g/references/out/cmd/10_state_stage2.txt), [`97`](06g/references/out/cmd/97_final_state.txt)):
  `1c5336e`, `4f4f9ed` and `b5626ec` clean with `--ignored` before and after; main checkout `28ebaf0`
  clean. `ff3e839` (`ff3e8394`, clean at the orchestrator's check before the batch) now holds 16
  `__pycache__/` directories (71 `.pyc`, mtime 2026-11-02T23:12:03–05Z), not deleted (log, observation 3);
  at the same second `scripts/__pycache__/harness.cpython-313.pyc` appeared in this records directory,
  also left in place.
- **Ray never initialised:** every capture worker reported `ray_initialized` False; 479 of 479 reading
  files record `[false, false]`.
- **Census checkpoints** ([`out/cmd/15`](06g/references/out/cmd/15_checkpoints.txt)): g `03adcbba…`
  (18,560 lines, `done`); 06h `3104c18e…`.

## The captures, bit for bit

- **V3's 90 and V8's 12** at g ([`out/cmd/30`](06g/references/out/cmd/30_capture.txt),
  [`31`](06g/references/out/cmd/31_check_captures.txt)): **102 of 102** reproduce the g census record:
  value and abserr by hex, num_regions, max_depth, converged, label; every key of the call's region
  arrays by dtype, shape and bytes; the harness's hashes. Sliced, each blob's SHA-256 equal to its
  capture record ([`out/cmd/32`](06g/references/out/cmd/32_slice.txt)).
- **V1's 12 at the parent against R23 and at g against R24** ([`out/cmd/44`](06g/references/out/cmd/44_v1_capture.txt)):
  **24 of 24**: the row's total equals the store's bit for bit, and exactly one call equals the store's
  sub-interval-1 G+q+r group bit for bit in its stored form, (1 + z_response) × the call's value and
  abserr; the pass-2 capture's own record again. The two captures' groups are equal on 12 of 12
  ([`out/cmd/45`](06g/references/out/cmd/45_v1_compare_groups.txt)).
- **z00 and z12** ([`out/cmd/81`](06g/references/out/cmd/81_compare_z00_z12.txt)): their captured
  integrands are equal by the 06i-13 criterion (route, span, signs, each factor's spline and table by
  bytes) and the integrand at 65 points with the default edges; only `kw.rtol` differs.

## 06i's 13

- **Inputs:** 13 of 13 equal 06i's `inputs_SHA256SUMS.txt` ([`out/cmd/32`](06g/references/out/cmd/32_slice.txt)).
- **Groups:** 13 of 13 equal, criterion and integrand ([`out/cmd/33`](06g/references/out/cmd/33_compare13.txt));
  descriptive only, the 7 Route A inputs' groups lack `K_lo` (pickled before 06h).
- **The tool against 06i's readings** ([`out/cmd/35`](06g/references/out/cmd/35_tool_vs_06i.txt),
  [`out/tool_vs_06i.json`](06g/references/out/tool_vs_06i.json)): **13 of 13 agree** under the fixed rule
  (D ≤ s_tool + s_06i + m/100, reading by reading). 44 of 52 readings bit-identical; the other 8 are FCC
  r1 and r2 on Route A samples 9, 22, 25, 27, at D/m 2.8e-6 to 4.0e-5. Sample 8's §8 statistic is 5.14
  for both, reported and not used. **The command line** on z25 and z13 is bit-identical to
  `one_reading.py`'s four readings each.

## Re-reads

- **Part 2's progress wrapper** ([`out/cmd/47`](06g/references/out/cmd/47_check_progress_wrapper.txt)):
  6 of 6 readings bit-identical to part 1's (z25 four, z13 qawo r1 and fcc r1).
- **Under `ff3e8394` with the out-of-order wrapper**, the rerun batch's stage 1 (its checkpoint,
  `out/rereads/`): **6 of 6 bit-identical** on value_hex, value_30 and imag_30: z13 qawo r1, r55 qawo r2,
  z13 fcc r1, r56 fcc r1, z25 qawo r2, r09 fcc r1.

## Cross-host

- **z00's four readings under `ff3e8394`**, the Mac Studio's against the laptop's: **4 of 4
  bit-identical** (qawo r1 `0x1.0c5f18f86f93bp-86`, qawo r2 `0x1.0c5f18f86f3e1p-86`, fcc r1
  `0x1.0c5f18f4cb423p-86`, fcc r2 `0x1.0c5f18f86f178p-86`) ([`out/scores_after_stage2/report_20261102T204420Z.md`](06g/references/out/scores_after_stage2/report_20261102T204420Z.md)).
- **The laptop's batch at `b5626ecb`** (the orchestrator's checks, `review/checks/fcc_laptop_check.txt`):
  w06 qawo r1 and r55 fcc r1, fcc r2 and qawo r1 bit-identical to the Mac Studio's.

## The usable-clause tally

In the final scoring ([`out/scores_stage3/scores_20261103T065158Z.json`](06g/references/out/scores_stage3/scores_20261103T065158Z.json)),
426 readings scored over V3's 90, V8's 12 and V1's 12: **Route A 184; clause (i) all_monotone 186;
clause (ii) `ff3e8394` with direct_segments 1, 28** (the QAWO readings of w00–w11, z00, z12); **clause
(iii) `b5626ecb` FCC with every non-monotone segment direct, 28** (their FCC readings). No unusable
reading is scored. Unscored, with reasons: the 14 samples' part-2 readings (tool `1c5336e`, inversion
through a non-monotone segment) and their stage-2 FCC readings (superseded by `b5626ecb`).

## The stores

| Store | Before | After |
|---|---|---|
| A3 v2's scratch copy (`var/levin-phase-floor-06/wider/state/a3v2_copy/`, 5 files) | EQUAL to its certification, 2026-11-01T17:48:59Z ([`out/cmd/13`](06g/references/out/cmd/13_copy_files_before.txt), [`14`](06g/references/out/cmd/14_copy_vs_certified.txt)) | EQUAL, 2026-11-03T07:02Z ([`out/cmd/95`](06g/references/out/cmd/95_copy_files_after.txt), [`96`](06g/references/out/cmd/96_copy_after_vs_certified.txt)); the orchestrator's ~07:10Z: EQUAL |
| R23 and R24 (10 files) | EQUAL to `rehearsal/state/stores_sha256_final.txt`, 2026-11-01T19:59Z ([`out/cmd/40`](06g/references/out/cmd/40_v1_stores_before.txt), [`41`](06g/references/out/cmd/41_v1_stores_before_vs_baseline.txt)) | EQUAL after the captures ([`46`](06g/references/out/cmd/46_v1_stores_after_capture.txt)) and at the end ([`97`](06g/references/out/cmd/97_final_state.txt)) |

Hashes (both before and after): primary `1542e05c…`, shards `bb007a4c…`, `aae63ccd…`, `5e2bda9b…`,
`ecc3330d…`. No journal, `-wal` or `-shm` file beside either store; nothing created under
`var/datastores/`.
