## Addendum 4 to the references agent's brief — the hold on non-monotone Route B readings, and the rerun of stopped QAWO readings after part 2

**The source.** The user's ruling of 2026-11-02, relayed by the laptop orchestrator after the
orchestrator's r52 report (`review/refs_part2_r52_report.md`), with the laptop's amendments of the
same day. The user's words:
- "Leave the backstop alone for the rest of part 52. We are getting enough good results." Part 2
  runs to its end unchanged.
- On a rerun batch of the stopped QAWO readings: "Happy with this".

### 1. The hold: non-monotone Route B readings (until the user rules; not a stop)

**What was found.**
- **The error:** the laptop reproduced the nine QAWO `ZeroDivisionError` readings' error
  (`tools/filon_reference.py:1063`) on z00 r1. It occurs in a Route B segment where the fitted
  phase's derivative changes sign (`monotone` False), and Newton runs onto the turning point.
- **The tool's own rule:** its docstring says, "A segment must not contain a stationary point of the
  phase; one that does is reported (`all_monotone` False) and the reference must not be used." On
  such a segment the map from t back to x is two-valued, so every reading through it is unsound,
  FCC as well as QAWO.
- **Which samples:** the orchestrator checked all 205 Route B readings
  (`review/checks/refs_monotone.txt`). Every Route B reading of **w00–w11** (all of V1's 12) and
  of **z00 and z12** has exactly one non-monotone segment and `tmap.all_monotone` False. No other
  sample has one.

**What you do.**
- These 14 samples are **not scored as references**: not in V1's table, V3, V4 or anywhere else.
- Each is reported as "Route B, non-monotone segment: reference unusable by the tool's own rule",
  with its readings.
- **Every reading you do score** must have `all_monotone` True. Confirm that in your records.
- **The nine error readings are not rerun,** and no tool change is made or awaited for them. How
  the tool should handle a stationary point awaits the user's ruling.

### 2. The rerun of the stopped QAWO readings

**When:** after part 2 (`levin-phase-floor-06g-references-20261101T201415`) has ended.

**Where:** on the **existing tool at `1c5336e`** (the code worktree). The tool is sound for monotone
calls, and no new worktree is needed. g's census harness goes on `PYTHONPATH` as before.

**How:** one registered, detached run with slug `references-rerun` (`--campaign levin-phase-floor
--prompt 06g`), at most 6 processes, one reading at a time with 6 workers.

**Before launching, list each candidate** in `out/rerun_plan.md`, with:
- its call's actual phase span, read from its FCC readings' segment records (the sum of `t_span`);
- its estimate, at 0.06–0.10 µs/rad for r1 and about 4× that for r2, giving both ends;
- **`all_monotone`** from its existing readings, which must be True for the reading to run;
- whether it runs.

**Candidates:**
- r47 qawo r2;
- r26 qawo r1 and r2;
- r24 qawo r1, if its estimate is under a day;
- r53 qawo r1 and r2, if part 2 stops them.

**Skip, each listed with its reason:**
- r52 qawo r2 (it already has three readings agreeing to 4.4e-5 of the tolerance);
- r24 qawo r2 (about 2 days);
- any reading estimated at a day or more;
- any call whose FCC readings and QAWO r1 already agree as tightly as r52's;
- any call with `all_monotone` False.

**The backstop** follows the user's earlier ruling (addendum 3), not part 2's mechanism:
- no reading is stopped for passing 30 minutes or for exceeding a multiple of its estimate;
- from 30 minutes, its progress and projected finish are logged every 5 minutes;
- it is stopped only when the projected finish is **a day or more**, or when it has run 24 h.

Part 2's `imap` progress counter counts only the leading run of chunks. **Base the day-or-more
projection on a measure that sees work done out of order:**
- **preferred:** completed chunks counted out of order, with `imap_unordered` used for progress
  only, and the result reassembled in chunk order;
- **otherwise:** the estimate from phase span.

**If the progress wrapper changes,** the change is a numbered deviation. Before the batch, re-read
two cheap Route B QAWO readings that finished cleanly in part 2. Each must reproduce **bit for bit**
(`value_hex` and the imaginary part), as deviation 18's check did. A reading that does not reproduce
is a stop.

### 3. Records and scoring

- Every rerun reading goes in `out/readings/` in the same form, with a `rerun` marker.
- Scoring (V3, V4, the V1 and V8 tables) uses every available reading with `all_monotone` True,
  under the rules of addenda 1–3. The 14 held samples are reported, not scored.

**Stop and ask** only if something here cannot be done as described.
