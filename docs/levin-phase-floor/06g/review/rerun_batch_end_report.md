# The rerun batch's end and the stage-3 recheck (2026-11-03, about 07:00Z)

This is the message for the laptop orchestrator. Two attempts to send it failed: the PreToolUse
hook timed out, and the host client may be unreachable.

**The run:** `levin-phase-floor-06g-references-rerun-20261102T185324`, done, exit 0, 67 of 67, at
06:47Z on 3 Nov: 64 done, 0 errors, 3 stopped.

## The recheck

It is recorded beside the stage-2 and addendum-6 scores; neither was rewritten. The agent's report
is `references/out/scores_stage3/report_20261103T065158Z.md`.

**What landed** (both monotone):
- **r24 qawo r1** gives r24 three readings, with a spread of 5.9e-5 of the tolerance.
- **r53 qawo r1** gives r53 three readings, with a spread of 1.8e-3 of the tolerance.
- Both move from FCC-only to two implementations: 82 two-implementation references, 8 FCC-only.
- Their ratios are unchanged: r24 0.307 (06h 0.221), r53 0.022 (06h 0.088).

**What the recheck changes:** no verdict.
- **V3** is unchanged under all three readings of the rule: it holds as written (66 counted), and
  fails with clauses (ii) and (iii) and as ruled, on z12 alone.
- **V4** holds, unchanged.
- **V1's table and the V8 table** are unaffected.

## What did not land

Three readings were stopped by the batch's day-or-more backstop:
- r47 qawo r2: r47 stays at three readings;
- r53 qawo r2: r53 has no r2;
- r26 qawo r1: r26 stays FCC-only.

**Finding: those three stops were wrong.** The driver projected elapsed × total / done. At the first
chunk back, that ignores the other five chunks in flight, counts the setup time, and treats very
uneven chunks as equal. The orchestrator's addendum asked for progress "counted out of order" but
not for the chunks in flight, so that part of the slip is the orchestrator's.

The agent recomputed each projection (`references/out/stage3_projection.json`). Its model costs each
chunk by its phase span, and was checked against the two finished readings to within 2 chunks at
every 5-minute check.

| Reading | Backstop's projection | Recomputed |
|---|---|---|
| r47 r2 | 453,016 s | **3.0 h** |
| r53 r2 | 290,165 s | **8.9 h** |
| r26 r1 | 2,296,531 s | **13.7 h** |

- **The model's rates hold up:** r26's cost per radian per worker, 0.342 µs/rad, equals that of the
  finished r24 and r53 r1 readings (0.343, 0.344). The r2 readings come out at about 4.1×.
- **Even a cruder correction keeps two of them under a day.** Taking chunks as uniform but six in
  flight gives r47 r2 19.6 h and r53 r2 12.5 h; only r26 r1, at 105 h, goes over.

The draft issue is `[06g-rerun-backstop-projection-ignores-chunks-in-flight]`. It is a harness
defect, not production code. No scored verdict depends on the three readings.

## The question for the user

Should these three QAWO readings be rerun, with a corrected projection? Run one at a time, that is
about 26 h in all by the recomputed estimates, or 26–80 h at the estimates' ranges. Nothing is
running now, and nothing has been launched.
