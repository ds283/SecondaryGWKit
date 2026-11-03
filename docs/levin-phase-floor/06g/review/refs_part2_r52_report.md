# References part 2 at 09:08Z on 2026-11-02: r52, the backstop, and the rest of the run

**The run:** `levin-phase-floor-06g-references-20261101T201415`, driver pid 14269, 339 of 380 units
done. This is the orchestrator's read-only reading of its records. Nothing in the run was changed.

**Sources:** the readers `review/scripts/refs_part2_timing.py` and the inline readers whose output is
in `review/checks/`:
- `refs_part2_timing_0905.txt`
- `refs_routeB_profile_0905.txt`
- `refs_routeB_phase_span.txt`
- `refs_part2_ratio_summary_0910.txt`

## 1. What r52 is, and why its QAWO readings overran

**Identity:**
- r52 is one of **V3's random draw** (seed 20261101), in the 1e-8 group stratum.
- Its call is `g-1e-8-3-34925` call 1, **group G+q+r**, with k = 1e5, q = r = 3.05e7 Mpc⁻¹ and
  z_response = 2.7567.
- It is on **Route B**, with 2,270 segments and 3 tables.

**The cause is a large phase span inside a Route B call.** It is not a hard integrand, nor anything
else.
- **Phase span:** the segments' fitted phase spans (`t_span` in the reading) add to **3.18e10 rad**.
  That is about 14 million rad per segment.
- **How QAWO works:** it splits each segment into panels of at most 2¹² rad (r1) or 2⁹ rad (r2). So
  r52's r1 reading needed 7.77 million panels; a typical Route B call needs 4–10 panels per segment.
- **The projection model was wrong for this:** it scaled Route B by segment count only (0.0139 s per
  segment for r1), calibrated on 06i's six Route B samples. Their phase spans were small.
- **The measured cost tracks phase span:** across all 43 Route B QAWO r1 readings done,
  - every call above about 1e8 rad costs **0.057–0.10 µs per rad**: r47 0.057, r52 0.057, r17
    0.058, r29 0.059, r25 0.059 and v00 0.062;
  - calls below that cost a floor of about 5–10 s.
- **The overrunning calls all have very late response times or large k:** G±q±r groups at
  z_response between 1.6 and 11 (r24, r26, r47, r52, r17), or 1,589 to 8,343 with k ≈ 3e7–1e8 (r25,
  r29).
- QUADPACK's `ier` is non-zero on most panels of these calls, but the cost per rad is the same as
  for the well-behaved large-span calls. The panel count, not the integrand's difficulty, sets the
  time.
- **r2 against r1:** r2 costs 3.8–4.0 times r1 on every Route B call that completed both: r17,
  r25, r29, v00 and v03.

## 2. What happened at the ceiling

**The mechanism as built:** backstop (b) stops a reading at **max(6 × projection, projection + 1 h)**.
For any reading projected under 12 minutes that is projection + 3,600 s. The monitor checks every 5
minutes after 30 minutes, so a reading is actually killed at the first check past its ceiling, at
about **3,905 s**.

It was recorded in `state/ruling_addendum3_20261101T201347Z.md` before launch, as the user's "several
times its projection" taken literally.

**Against the ruling's intent:**
- The ruling said a reading is stopped only when its projected finish is **absurd (a day or more)**.
- (b) instead stopped readings whose true cost was hours, because their projections were 30–100
  times too low.
- The orchestrator accepted the agent's launch report without checking (b) against that intent.
  That is the orchestrator's slip.

**Stopped so far,** each recorded `stopped` with its reason in `out/reading_monitor.log` and the
reading's log in `out/reading_logs/ref/`:

| Reading | Backstop | Note |
|---|---|---|
| r24 qawo r1 | (b) at 3,905 s | |
| r24 qawo r2 | (a) at 30 min | ordered progress projected 182,652 s |
| r26 qawo r1 | (b) at 3,905 s | |
| r26 qawo r2 | (b) at 3,905 s | |
| r47 qawo r2 | (b) at 3,905 s | its r1 finished in 2,594 s |
| r52 qawo r2 | (b), due at the 3,905 s check, about 09:08:14Z | its r1 finished in 1,796 s |

**The progress counter understates.** The reading's `pool.imap` is ordered, so "chunks done" counts
only the leading run of chunks. r26's readings showed 0 of 139 for an hour. The progress-based
projections (26,000–43,000 s for r52 r2) are therefore unreliable in both directions.

## 3. The completed readings: wall time divided by projection

There are 325 reading files written.

| Readings | n | Median | p90 | Max |
|---|---|---|---|---|
| Projection ≥ 1 s | 286 | **1.01** | 1.61 | **101** |

The 39 Route A readings with projection < 1 s took 1–2 s. The rate model gives about 0 s for tiny
spans and omits start-up, so their ratios are meaningless.

Every one over 5× is Route B QAWO:

| Ratio | Wall s | Projection s | Reading |
|---|---|---|---|
| 101.0× | 2,594 | 26 | r47 g-1e-8-3-1916 #4 r1 |
| 56.8× | 1,796 | 32 | r52 g-1e-8-3-34925 #1 r1 |
| 30.7× | 1,687 | 55 | r29 g-1e-7-1-35525 #4 r2 |
| 26.2× | 1,839 | 70 | r17 g-1e-7-1-2229 #3 r2 |
| 20.5× | 422 | 21 | r29 r1 |
| 17.4× | 849 | 49 | r25 g-1e-7-0-1887 #3 r2 |
| 17.4× | 455 | 26 | r17 r1 |
| 11.8× | 215 | 18 | r25 r1 |
| 7.2× | 299 / 298 | 41 | v06 / v00 (g-1e-8-2-1380 #4 / g-1e-7-2-1380 #4) r2 |
| 5.0× | 78 | 15 | v06 / v00 r1 |

Route A's long-span QAWO projections were good: 0.7–1.4× on r00, r10, r36, r37 and r38. Every FCC
reading, Route B included, came within 0.85–1.0×.

**Also found: 9 QAWO readings ended in an error** (exit 1, status `error`, no value), all with
`ZeroDivisionError: float division by zero` in `tools/filon_reference.py` line 1063, `gfun`:
`yn = min(1.0, max(-1.0, y - fval / dpy))`, a Newton step with dpy = 0 inside Route B's QAWO
integrand.
- **Which:** w03 r2, w04 r1 and r2, w08 r1, w10 r2 (all V1's 12), and z00 r1 and r2, z12 r1 and r2
  (Z3's).
- **Consequence:** these samples have QAWO's readings missing, in part or in full. Under the
  user's ruling they are scored with what they have, FCC-only where both QAWO readings are missing.
- **Whose:** this is a defect in this repository's tool, at the code commit. It is not reported to
  any package. Nothing was fixed.

## 4. The remaining readings, and a revised finish

40 readings remain after r52 qawo r2.

| Sample | Unit | Group | Route B segments | QAWO r1 / r2 / FCC r1 / r2 projections (s) | Expectation |
|---|---|---|---|---|---|
| r52 | g-1e-8-3-34925 #1 | G+q+r | 2,270 | — / — / 41 / 157 | FCC is unaffected; about 3 min |
| r53 | g-1e-8-3-14492 #4 | G+q−r, k 9.7e6, q 9.85e5, z_r 0.174 | 2,204 | 31 / 82 / 39 / 153 | **Like r47** (same k, q, r; z_r 10.98 gave 4.6e10 rad). Its later response suggests a larger span, so **both QAWO readings will very likely hit (b)**: about 2 × 3,905 s |
| r54 | g-1e-8-0-34014 #0 | G | 900 | 13 / 34 / 16 / 62 | normal |
| r55 | g-1e-8-2-34671 #2 | G−r | 298 | 4 / 11 / 5 / 21 | normal |
| r56 | g-1e-8-1-10794 #5 | G−q+r, z_r 1.75e10 | 204 | 3 / 8 / 4 / 14 | normal |
| r57 | g-1e-8-3-34394 #3 | G−q+r, k 9.7e6, q = r 3e8, z_r 8,343 | 1,774 | 25 / 66 / 32 / 123 | **Possibly like r29** (about 7e9 rad): r1 about 400 s, r2 about 1,700 s, both under ceiling |
| r58 | g-1e-8-3-34453 #3 | G−q+r, k 1e5, q = r 9.56e7, z_r 399 | 1,938 | 27 / 72 / 35 / 134 | probably moderate |
| r59 | g-1e-8-0-1823 #0 | G | 998 | 14 / 37 / 18 / 69 | normal |

The ten long Route A QAWO r1 readings, in order:

| Reading | Projection s |
|---|---|
| z20 | 554 |
| r30 | 1,178 |
| z19 | 1,740 |
| z26 | 1,802 |
| z05 | 2,002 |
| z17 | 2,607 |
| z11 | 5,466 |
| z16 | 6,333 |
| z28 | 6,333 |
| r07 | 6,512 |

They total 34,544 s (9.6 h). Their ceilings are 6 × projection, and Route A projections held to
0.7–1.4×.

**Revised finish:**

| Stage | Ends about |
|---|---|
| r52 FCC | 09:12Z |
| r53 | 11:25Z |
| r54–r59 | 12:10Z |
| The ten long readings | **about 21:45Z on 2 November** (range about 18:50Z to 01:30Z on 3 November) |

That is about 12 h later than the 09:45Z first projected. The orchestrator's single check, set for
09:50Z, will find it mid-run; the wait on the driver's exit stays in place.

## 5. Options for r52 qawo r2 (stopped), and the other stopped readings

Nothing has been run. Each would be a separate registered reading after part 2 ends: one at a time,
6 workers, ceiling set from phase span. The estimates use cost = phase span × 0.057 µs/rad for r1,
and 4× that for r2.

| Reading | Phase span | Estimated cost |
|---|---|---|
| **r52 qawo r2** | 3.18e10 rad | **about 2.0 h** (about 7,300 s) |
| r47 qawo r2 | 4.59e10 rad | about 2.9 h |
| r26 qawo r1 and r2 | unknown; r1 > 3,905 s implies > 6.8e10 rad | r1 > 1.1 h, r2 > 4.3 h |
| r24 qawo r1 and r2 | unknown; similar | similar or more |
| r53 qawo r1 and r2, if stopped | span from its FCC reading's segments | r1 likely 2–3 h, r2 8–12 h |

The span can be read from the FCC readings' segment records before deciding.

**A rerun of r52 r2 alone** would give r52 its fourth reading. Its r1 and both FCC readings already
exist, so whether it counts under the four-reading rule depends on that fourth reading.
