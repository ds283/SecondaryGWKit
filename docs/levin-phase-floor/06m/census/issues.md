# levin-phase-floor 06m §5, the census agent: issues

**None to open.** Gate M0 and Gate M hold, and no unit ended in error or over budget.

## For the user at the records commit

Prompt 06m's header proposes `[06m-calls-flagged-unconverged-within-their-request]`. Whether to open it, or to fold
it into 08's question of what `total_converged` means, is the user's decision. The census at $m$ gives these numbers
for it (`out/descriptive/descriptive_06m.json`):
- **Outside the 720:** 114 unconverged group calls in 84 units and 46 rows.
  - 104 report an `abserr` within their request; each of them holds an untrusted leaf at `depth_max`, and none is
    `node_limited`.
  - 10 are `node_limited`, at most 1.048× over the request.
  - The largest share of a row's `total_abserr` is 4.73 %.
- **In the 720:** 311 unconverged group calls, 309 within their request.

## Observations

These are recorded in the log; none is proposed as an issue.
- **Producer build order:** in 7 units, the census harness's producer cache recorded two builds in the opposite order
  to route B's census. Every value is identical. It is a harness record, not production code.
- **`[06m-levin-floor-misses-node-rounding-far-from-the-origin]` is not measured here.**
  - The census's records give the count of checked Levin leaves with $\Delta_1/\phi_P > 10$, by r band.
  - They show that no such leaf is trusted on the noise clause, and that none outside the 720 is returned untrusted.
  - They do not hold what would separate node rounding from resolution. The issue stays with 07.
