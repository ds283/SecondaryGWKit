# levin-phase-floor 06g §5, the census: issues for the board

These are proposals; the orchestrator names them and decides what is opened. Each has its measurements in this
directory's `log.md` and `out/`. **The note's §5 stop on V8 was reached, put to the user, and the user ruled to
score the rest** (`var/levin-phase-floor-06g/review/stop_v8.md`).

**C1. V8 fails: under 06g's trust test, slow leaves are refined to depth 20 and returned untrusted, so group
calls that meet their request report unconverged.**
- 801 calls in 470 rows (1e-7) and 827 calls in 474 rows (1e-8), outside the 720. All are `G±q±r`-type group
  calls.
- Every one meets its request on abserr (median abserr/request 0.035; max 0.987), and returns untrusted leaves at
  depth 20 (median 6, max 469).
- Per V8's pre-fixed classification, the cause is "(b), refined to the depth cap".
- The addendum (`out/v8_slow_leaves.md`) finds the scoped slow leaves' r clustered at 1/2 (41 % within 0.01).
  In 223 and 227 of the calls, the untrusted leaves returned are r ≥ 1, not in the (b) class.
- Bears on `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]` and
  `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`.

**C2. V5 fails: 966 (1e-7) and 996 (1e-8) group calls outside the 720 reach depth 20 under 06g, against 0 and 1
under 06h.**
- 165 and 168 of them are still converged. Every call is in `out/v5_depth_max_calls.csv`.
- Same mechanism as C1.

**C3. V2 fails on `total`: 5 rows at each rtol move by more than the sum of the two errors (ratio up to 2.16).**
- In each of these rows, the `G+r` call (index 1) moves by 7.1–7.2e-33 against abserrs of 3–7e-34, under 06h on 2
  leaves and under 06g on 12–16.
- At call level, 62 calls in 59 rows (1e-7) and 80 in 75 rows (1e-8) move beyond both errors, all group calls, all
  on a different partition. At least one of the two codes is dishonest on each.
- The oracle reading of V2 holds (max ratio 0.499).
- The references agent's V3 may say which code is wrong.

**C4. V6 fails: Z2's call (`g-1e-8-0-33995` #5) ends `node_limited` but at depth 20.**
- It has 32 leaves at the cap, 15 of them untrusted. Abserr 7.16e-34 against a request of 6.69e-34; 963 regions
  (06h: 8,706); ΣE −5.71e-35, Σ|E| 6.51e-34.
- The node-limited flag is set, since no untrusted leaf is refinable, even though the call reached the cap.

**C5. Cost: the census took 5 h 40 min against 06h's 2 h 33 min.**
- Unit wall ×3.43 (1e-7: 2.22 → 7.62 h) and ×2.96 (1e-8: 2.92 → 8.64 h).
- Group regions ×4.95 and ×3.81, group solves ×5.03 and ×3.91; analytic ×1.16.
- (b) refinement alone adds 776,371 regions and 6.21 M solves at 1e-7.
- No unit went over budget; the largest took 47 s.

**C6. The node-limited stop lost three calls at 1e-8, each just above its request** (`0-12783` #4, `0-33997` #4,
`2-34662` #2). These are instances of `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`; I
propose adding them to that entry rather than opening a new one.

**C7 (descriptive, for the laptop).** The slow leaves behind V8 have r clustered at 1/2. Their Δ1/φ_P has median
2.3 and p90 10. Their r wanders along the recorded ancestor chains and settles only in the last few levels. They
sit in a few clusters per call (median 2), none at the call's interval ends. This is reported, not interpreted
(`out/v8_slow_leaves.md`).

**Not proposed:** W3's `premise_holds` is still 06b's formula on whole floors. It is a harness record, not a code
defect; 06g's premise is recorded in `premise_06g`.
