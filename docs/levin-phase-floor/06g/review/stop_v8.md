# The note's §5 stop on V8 — 2026-11-01, ~17:35Z

The note's §5 says: "Also stop if the noise clause or the slow threshold costs convergence on more
than a handful of rows (V8), before scoring the rest." That condition is met, so I stopped. V2,
V5–V7 and the references are unscored, and the references agent has not been resumed.

**The census:** `levin-phase-floor-06g-census-20261101T114315` ended done, exit 0, with 18,560 of
18,560 units. No unit ended in error and none went over budget. It took about 5 h 40 min against
06h's 2 h 33 min. The store and 06h's checkpoint are unchanged.

**V8 fails at both rtols.** These counts are outside the 720 stationary-point rows. I recounted
them from the census agent's `census/out/v8_calls.json` and they match its report.

| rtol | (b) slow leaf, refined to depth 20 | node_limited |
|---|---|---|
| 1e-7 | 801 calls / 470 rows | 0 |
| 1e-8 | 827 calls / 474 rows | 3 calls / 3 rows |

- Every lost call is a `G±q±r` group call; no analytic-block call was lost.
- Every lost call still meets its request on abserr: abserr/request has a median of 0.035 and a
  maximum of 0.987. Each is unconverged only because untrusted slow leaves (1/4 ≤ r < 1, Δ1 > φ_P)
  are returned at depth 20.
- Regions per call: the median rises from 7 to 135 at 1e-7, and from 9 to 140 at 1e-8.
- One call gained convergence: `g-1e-8-3-1072` call 13, V7's call.
- Inside the stationary-point rows, not scored: 316 calls in 212 rows at 1e-7, and 340 calls in
  228 rows at 1e-8.

The question for the user was sent to the laptop orchestrator `local_87bf4c4b-…`.
