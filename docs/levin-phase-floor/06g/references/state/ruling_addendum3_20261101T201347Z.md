# levin-phase-floor 06g §5, the references — addendum 3, the user's ruling on part 2 (recorded before any part-2 reading)

> Source: `review/dispatch_references_addendum3.md` (SHA-256 `d5f71f07…`), the user's ruling of
> 2026-11-01 relayed by the laptop orchestrator. The predictions file
> (`state/predictions_20261101T175120Z.md`) is not edited; V3 and V4 are scored both ways (below).
> For V1's calls this ruling supersedes `state/v1_rule_20261101T175144Z.md`'s "stopped at 30 minutes"
> and its four-readings count; its fields and question stand. Read-only from here.

**The user's words, in part (quoted in the addendum):**
- "Quibbling over ~8 hours of compute that only has to be run once strikes me as a false economy."
- "A runtime that exceeds 30 minutes means that we should stop and check, not guillotine it
  immediately. If it is projected to finish in the next 5 minutes, we should finish it. If it's not
  projected to finish for a day, then we would kill it. In between these extremes is the judgement
  call."
- On scoring: "We are iteratively improving on our algorithm, and we don't know what to expect."

## What runs (`out/part2_plan.json`, `scripts/make_part2_plan.py`: 380 readings, projected 48,670 s = 13.5 h)

1. V1's 12 (w00–w11, the captures at g against R24): QAWO and FCC at resolutions 1 and 2.
2. V8's 12 (v00–v11): the same four.
3. The rest of V3's sample (Z3's 17 not in 06i, r00–r59): the same four, except on the 17
   large-span Route A calls z04–z07, z11, z16–z21, z24, z26, z28, z29, r07, r30: FCC r1 and r2
   only here, no QAWO r2 on any of them.
4. Last: QAWO r1 on z20, r30, z19, z26, z05, z17, z11, z16, z28, r07 (in that order); none on z04,
   z07, z06, z21, z29, z18, z24.

## The 30-minute cap is a check (driver `run_refs.py --part 2`)

- No reading is stopped for passing 30 minutes. From max(30 min, its projection) on, every 5 min,
  its progress (chunks done of total, `one_reading.py --progress`) and projected finish
  (start + elapsed × total/done) go to the run's heartbeat and `out/reading_monitor.log`.
- **The backstop, chosen before launch:** a reading is stopped (SIGTERM to its process group, SIGKILL
  after 30 s), recorded `stopped` with its reason and reported, only if
  (a) after 30 minutes its progress projects a total of 24 h or more ("a day"), or
  (b) its elapsed time reaches its ceiling, max(6 × projection, projection + 1 h), or
  (c) 24 h. The largest ceiling is r07's QAWO r1, 39,075 s (projection 6,512 s).
- The projections (`make_part2_plan.py`'s docstring): part 1's per-radian (Route A QAWO), per-sample
  (Route A FCC) and per-segment (Route B) rates, plus 1.6 s per reading.

## Scoring

**As written** (the predictions file, unchanged): a reference counts only when its four readings are
all present and agree within min(abserr_06h, abserr_g)/100 (Route B: all monotone); V3 is undecided
if fewer than 45 of the 90 count; V4 needs all four of z02, z08, z09, z14 counted.

**As the ruling scores them** (labelled so, beside the above):
- every reference is scored that has at least FCC's two readings: four readings, three (FCC two and
  QAWO r1) or FCC's two alone. Its value is the FCC res-2 reading, as in every case.
- each row states its readings (methods, resolutions), their spread (largest minus smallest), the
  tolerance min(abserr_06h, abserr_g)/100 (V1: min(e23, e24)/100) and whether the spread is within
  it. A spread outside the tolerance is reported, not dropped. Route B's monotone flag is reported.
- each reference is classed **two implementations** (QAWO and FCC readings present) or **FCC only**.
  V3 as ruled: a violation is |v_g − ref| > abserr_g; **fails** if any scored reference violates
  it, **holds** otherwise; the same verdict restricted to two-implementation references is given
  beside it, and every FCC-only violation is listed as a finding. V4 as ruled: the same ratio rule
  on z02, z08, z09, z14 (all four have 06i's four readings).
- findings to stop and look at, never reasons to drop: an FCC-only reference far from Levin's value
  (|v_g − ref| > abserr_g), and FCC's two resolutions disagreeing (|FCC r1 − FCC r2| over the
  tolerance).
- a sample with fewer than FCC's two readings (a reading stopped by the backstop or failed) has no
  reference and is reported as such.
- the V1 and V8 tables use every reference the same way, with their readings and spread.
