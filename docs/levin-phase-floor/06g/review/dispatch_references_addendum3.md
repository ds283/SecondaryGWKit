## Addendum 3 to the references agent's brief — the user's ruling on part 2 (2026-11-01)

**The source.** The user's ruling, relayed by the laptop orchestrator, on the cost-gate question of
`references/out/cost_projection_part2.txt` and `cost_projection_part2_wall.txt`. The cost gate is
lifted for the plan below, whose total wall time the laptop projects at about 13 h.

The user's words, in part:
- "Quibbling over ~8 hours of compute that only has to be run once strikes me as a false economy."
- "A runtime that exceeds 30 minutes means that we should stop and check, not guillotine it
  immediately. If it is projected to finish in the next 5 minutes, we should finish it. If it's not
  projected to finish for a day, then we would kill it. In between these extremes is the judgement
  call."
- On scoring: "We are iteratively improving on our algorithm, and we don't know what to expect."

### 1. What runs

**Every reading of part 2 as planned**, except QAWO on the 17 large-span Route A calls: z04–z07,
z11, z16–z21, z24, z26, z28, z29, r07 and r30. On those 17:
- **FCC at both resolutions** on all of them;
- **QAWO r1** only where its projection is under about 2 h: z20, r30, z19, z26, z05, z17, z11, z16,
  z28 and r07;
- **no QAWO r2** on any of them, and **no QAWO r1** on z04, z07, z06, z21, z29, z18 or z24.

**The order:**
1. V1's 12;
2. V8's 12;
3. the rest of V3's sample;
4. the long QAWO r1 readings last, so they hold nothing else up.

**V1's capture** against the rehearsal stores goes ahead as the first addendum already allows: a
numbered deviation, the stores read-only and hashed before and after, and v23/e23 and v24/e24
reproduced bit for bit.

### 2. The 30-minute cap becomes a check, not a cutoff

- **No reading is killed for passing 30 minutes.**
- A reading past 30 minutes, or past its own projection, is **logged** with its progress (QAWO's
  panels done out of total, or the equivalent for FCC) and its projected finish.
- **It finishes** if its projected finish is reasonable. A few minutes more always counts, and so
  does anything up to a few times its projection.
- **It is stopped** only if its projection is plainly absurd (a day or more). It is then reported,
  never silently dropped.
- **A backstop** must ensure nothing runs for ever. Choose one before launch and record it, for
  example a hard ceiling per reading at several times its projection, set from QAWO's panel
  progress, with the projected finish logged once it passes 30 minutes. Each harness change is a
  numbered deviation.
- The orchestrator checks once, at part 2's projected end, and uses judgment on anything still
  running then. Write that projected end into your launch report.

### 3. Scoring

- **Every reference is scored,** whether it has four readings, three, or only FCC's two. V3, V4,
  the V1 table and the V8 table all use all of them.
- **Each row states its readings:** which methods, which resolutions, and their spread.
- **Weight:** references with two independent implementations carry the most weight. A reference
  read by FCC alone is marked as such and is still scored.
- **Findings to stop and look at:** an FCC-only reference far from Levin's value, or FCC's two
  resolutions disagreeing. Neither is a reason to drop the reference.
- **The tolerance:** 1/100 of the minimum abserr stays as the measure of whether the readings
  agree. A reference outside it is still reported, with its spread, not dropped.

**The predictions as written** are also kept. Your timestamped predictions file is not edited.
Give V3's and V4's verdicts both ways:
- **by their rules as written:** a reference counts only when its four readings agree within the
  tolerance, and V3 is undecided if fewer than 45 of the 90 count;
- **beside that, as the user's ruling scores them:** over every reference, with its readings and
  spread.

Label each clearly. The review reports both. Write this ruling into a new timestamped `state/`
file before part 2's readings, and copy the user's words into `log.md`.

### 4. Records

Record the ruling in your log, quoting the user's words, and number every harness change as a
deviation. **Stop and ask only if something here cannot be done as described.**
