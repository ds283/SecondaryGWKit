# 07a Part C — stop on W5's reading (2026-11-12, ~21:00 UTC)

The census agent scored W1–W4, W6 and W7 and wrote its records. It asks one question, on W5's reading, which changes
W5's verdict. The question is relayed verbatim, with no recommendation.

## My own checks on W5's numbers (`scripts/w5_forced.py`, outputs `w5_forced.txt`, and `w_rescore.{json,txt}`)

- Gated (non-forced) solves taking the node-partition sum: **0**. Clenshaw–Curtis regions with a half sent to the
  Levin solve: **0**. `num_guarded_regions` summed over every call at h: **0**.
- Forced solves taking the node-partition sum: **18,236**, in 4,416 calls. That is 3–10 per call: 3 in 2,080 calls,
  4 in 776, 5 in 655, 6 in 771, 7 in 93, 8 in 31, 9 in 8 and 10 in 2.
- Every one of those 4,416 calls locates a root.
- Of the 4,804 calls that locate a root, 388 have no forced partition sum: 194 group calls at each rtol, each with one
  root and a window.
  - So "exactly the 4,416 root-located calls" in the agent's report holds as "all within root-located calls".
  - It does not hold as "every root-located call".
- The post-unit auxiliary solves also take forced partition sums (`route_07a_post`):
  - cc_floor terms 4,416;
  - declared 06h terms 3,364;
  - w3 1,920;
  - deep lines 1,480.

## The rest, by my own readers so far, all agreeing with the agent's figures

- **W1** holds: 0 untouched calls differ in the call's outputs, at both rtols (`w1_compare.*`).
- **W2** holds: 0 Levin leaves listed (`w_rescore.*`).
- **W3** holds: 0 violations; 784 skipped phases, all `zero_rate`.
- **W4** fails, as §9.5 predicted: the five at 3.16, 2.21, 3.08, 2.38 and 2.53; all 33 calls bit-equal to 07's
  (`w4.*`).
- **W6** fails, as predicted: 2,482 of 2,484 pairs pass; `3-14501` is at 1.385 and `3-35006` at 1.084.
- **W7 fails, not predicted.**
  - 25 of 4,774 covered calls move beyond 1e-5 of 07's abserr, at 0.0066–0.145, all at 1e-8, all group `G` call 0.
  - They are exactly the 25 windows that stopped at 07 and do not stop at h (`stops.*`).
  - `converged` is unchanged in all 25.
- **S4** holds by my reading of R30's store and the agent's replays (`s4_check.txt`). R30's fingerprint is
  `1b332ffa…` again (`fingerprint_r30.txt`).
  - Two earlier fingerprint attempts, from the main checkout's cwd, were refused by the main checkout's code with
    nothing read or written.
  - The store's mtimes are unchanged and the worktree is clean.

The §C.2 review (records, predictions, scope at the end) and the §C.3 report follow the user's word on W5.
