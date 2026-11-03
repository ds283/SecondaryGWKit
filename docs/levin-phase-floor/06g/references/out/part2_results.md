# Part 2 of the references (run levin-phase-floor-06g-references-20261101T201415): results

Run `done`, exit 0, 380 of 380 units, first record 2026-11-01T20:14:25Z, last 2026-11-02T18:46:25Z
(driver pid 14269, sh pid 14264). Driver wall times by stage (sum of readings): V1 2,013 s, V8 2,811
s, the rest of V3 48,856 s, the ten long QAWO r1 27,447 s; 81,127 s in all against the plan's 48,670
s. Raw summary: `out/part2_results_raw.txt`; per reading: the run's checkpoint, `out/readings/`,
`out/reading_logs/{v1,v8,ref,ref_long}/`, `out/progress/`, `out/reading_monitor.log`.

| Status | Readings |
|---|---|
| done | 363 (V1 43, V8 48, rest of V3 262, long QAWO r1 10 of 10) |
| error | 9: QAWO `ZeroDivisionError` (tool line 1063, a Newton step at dp/dy = 0): w03 r2, w04 r1 and r2, w08 r1, w10 r2, z00 r1 and r2, z12 r1 and r2 |
| stopped | 8, by part 2's backstop: r24 r1 (b) and r2 (a, ordered progress projecting 182,652 s); r26 r1 and r2 (b); r47 r2 (b); r52 r2 (b); r53 r1 (b); r53 r2 (a, 290,167 s) |

Every (b) stop came at 3,905 s, the first 5-minute check past max(6 x projection, projection + 1 h)
for readings projected under 12 minutes; the projections of large-phase-span Route B QAWO readings
were 30-100x too low (the model scaled Route B by segment count, not phase span; the orchestrator's
`review/refs_part2_r52_report.md`).

**Monotone check:** of the 207 Route B readings written (parts 1's 06i13 readings excluded), 47
have `tmap.all_monotone` False, every one a reading of w00-w11 or z00, z12, each with one
non-monotone segment (addendum 4's hold). Every other Route B reading is all monotone.

Not scored yet (the rerun batch first: addenda 4 and 5).
