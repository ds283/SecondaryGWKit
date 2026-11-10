# Planned stop 4: levin-phase-floor 07 Part B (Mac Studio orchestrator)

Written 2026-11-09. **A §6 stop:** V4's references project beyond a day. No reading has been taken, no lock is held,
and nothing is running.

## The user's word at stop 3, verbatim (relayed by the laptop session)

> Yes, please do that

That word cleared step 7. On the 28 O9 calls, the laptop session's note: "nothing is needed now. I'll add them to the
6π gate issue in the laptop records commit."

## The census agent's step 7, as it reports it

All under `var/levin-phase-floor-07/census/`. The detail is in `out/scores/summary.json` and `out/scores/extra_descriptive.json`.

- **V1 holds.** At each rtol: 9,280 units, and 142,257 calls that locate nothing and guard nothing, all equal to 06m's. The 8,052 quiet-only units also have `total` and `total_abserr` equal in hex.
- **V2 holds.**
  - 182,066 calls scanned: groups at both rtols, analytic at 1e-7. None of them returns a Levin leaf holding a sign change.
  - The listed leaves are window leaves, or 168 Clenshaw–Curtis leaves in calls whose phase Rule A skipped.
- **V3 holds.**
  - 0 violations among 170,582 scored calls.
  - 3,584 have a sign change or a located root, and each is matched within the scan spacing h.
  - §10's set-aside is 11,484 scanned calls. Of these, 168 analytic calls at 1e-7 (J4 and Y4 in 84 rows) show a sign change: 14 of them guarded (the O9 calls) and 154 not.
  - Deviation 25: analytic calls at 1e-8 are not scanned, so they are not scored.
- **V5 fails: 23 calls.**
  - All are `G+q+r` (#1) calls that locate a root and hold leaves at `depth_max` 20: 10 at 1e-7 and 13 at 1e-8.
  - A finding, not a fault: it reopens G2's L2. 0 units were over budget.
- **V4's captures:** 24 of 24 reproduce the census bit for bit, plus a 25th, z12's, taken for the projection only. Every one of the 24 locates a root.
- **The projection:**
  - 775,468 s for all four readings on all 24 calls: QAWO 691,209 s and FCC 84,259 s. That is 1,163,202 s with the ×1.5 margin.
  - The heaviest call is v13, `g-1e-8-2-34841`, at 373,093 s.
  - 29 readings are projected past 30 minutes.
- **Deviations 16–25.** Four are unintended drift:
  - 16 and 17: two scoring readers crashed and were fixed and re-run;
  - 20: `out/v4/` was created while the capture was running, before it wrote there;
  - 21 and 22: the projection's first version failed, its output kept, and it was started with a plain `&`.

## My checks

1. **V5, recounted from the census checkpoint myself.**
   - **23 violations, the same units as the agent's list.**
   - Its prose has two slips; its JSON list is right:
     - the counts at `depth_max` run from **2 to 54**, not 2 to 14: `g-1e-7-0-38999` and `g-1e-8-0-38999` hold 54 each;
     - **three** are converged at g, not two: `g-1e-7-1-34766`, `g-1e-8-1-34766` and `g-1e-8-0-14167`.
   - Locating or guarding calls: 956 group and 1,528 analytic per rtol. The 1,528 are 1,514 locating plus 14 guard-only.
2. **The capture checks** (`out/v4/check_captures.json`): 25 entries, all `reproduces_census` true and `scripts_sha256_equal` true.
3. **The projection's file** (`out/v4/refs_projection.json`):
   - `total_s` is 775,467.9 and `total_with_1p5_s` 1,163,201.9. `beyond_a_day` is true, and 29 readings are over 1,800 s.
   - The model's scatter is large: on 06g's readings, QAWO r1's largest observed-to-predicted ratio is 3.33 and its median 1.29. The projection is uncertain by about that factor.
4. **State:**
   - All 181 protected files are unchanged (`checks/protected_at_stop4.txt`).
   - The lock is free, with no Ray processes of `/tmp/ray-sgk`.
   - The registry holds 68 runs, all finished.
   - All three worktrees are clean with `--ignored`.
   - Nothing is committed.
