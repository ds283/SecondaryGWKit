## Addendum 2 to the references agent's brief — the V8 sample, referenced and scored apart

**The source.** The user's ruling of 2026-11-01 on the note's §5 stop on V8, relayed by the laptop
orchestrator, is to score the rest of Part B for the record and to add a reference sample of the
calls V8 lost. The user's words: "(b) with the V8 sample, please". The stop stands in the record and
V8's verdict (fails) does not change.

**What does not change.**
- V3's sample, its draw, its seed, its count rule and its scoring are exactly as your brief says.
- The first addendum (V1's 12 group calls) stands as written.
- **If this sample would change how V3's own sample is drawn or scored in any way, stop and ask.**
  Do not adjust.
- A V8 call may also be drawn into V3's random sample. Overlap changes neither: the call is scored
  in both tables, independently, and you say so.

### The calls: fixed before any reference was computed

The orchestrator selected them with `review/scripts/select_v8_sample.py` (SHA-256 `92ff6122…`)
from the census agent's `census/out/v8_calls.json` (`1e9a62b0…`). The output is
`review/v8_sample.json` (`874e17db…`), and its transcript is `review/checks/select_v8_sample.txt`.

**The rule:**
- At each rtol, take the lost calls outside the 720 stationary-point rows whose class is (b). The
  node_limited calls are excluded.
- Sort them by abserr_g / request_g, ascending, breaking ties by (unit, call index).
- With n calls, take positions round(q(n−1)) for q = 0, 0.2, 0.4, 0.6, 0.8 and 1.

They are not reselected after any result.

| rtol | q | unit | call | group | abserr_g / request_g |
|---|---|---|---|---|---|
| 1e-7 | 0.0 | g-1e-7-2-1380 | 4 | G+q−r | 5.709e-09 |
| 1e-7 | 0.2 | g-1e-7-0-12617 | 5 | G−q+r | 2.404e-04 |
| 1e-7 | 0.4 | g-1e-7-3-34212 | 4 | G+q−r | 1.063e-02 |
| 1e-7 | 0.6 | g-1e-7-1-34521 | 4 | G+q−r | 1.353e-01 |
| 1e-7 | 0.8 | g-1e-7-3-34296 | 1 | G+q+r | 4.693e-01 |
| 1e-7 | 1.0 | g-1e-7-3-34090 | 4 | G−q−r | 9.867e-01 |
| 1e-8 | 0.0 | g-1e-8-2-1380 | 4 | G+q−r | 5.709e-09 |
| 1e-8 | 0.2 | g-1e-8-0-12606 | 5 | G−q+r | 2.773e-04 |
| 1e-8 | 0.4 | g-1e-8-3-1192 | 5 | G−q+r | 1.346e-02 |
| 1e-8 | 0.6 | g-1e-8-1-35480 | 2 | G+q−r | 2.437e-01 |
| 1e-8 | 0.8 | g-1e-8-3-38023 | 4 | G−q−r | 8.071e-01 |
| 1e-8 | 1.0 | g-1e-8-0-12655 | 3 | G−q+r | 9.963e-01 |

### The captures

These are census calls on the A3 v2 scratch copy, so capture them as you capture V3's: at $g$, from
the code worktree with $g$'s census harness. Check each bit for bit against the $g$ census record
of the same call (value, abserr and regions). A capture that does not reproduce its record is a
stop.

**The "before" code.** The laptop's ruling names R23. R23 is the rehearsal store and holds none of
these calls. For census calls the parent-code record is **06h's census**,
`var/runs/levin-phase-floor-06h-census-20261101T005037/`. The orchestrator reads the ruling's "R23"
as 06h's census record of the same call, and this table uses that record. `v8_sample.json` carries
both records' values and abserrs.

### The route and the count rule

- These are cosmological groups, so use **Route B** (the least-squares smooth phase).
- The count rule is the brief's, unchanged: four readings (QAWO and FCC, each at two resolutions)
  agreeing to within 1/100 of $\min({\rm abserr}_{06h}, {\rm abserr}_g)$.
- **Never loosen the tolerance.** A call that cannot be counted is recorded with its four readings
  and their scatter.
- Readings over 30 minutes are stopped, as for V3.
- At most 6 processes.

### The order and the cost

- Run these as a stage after V3's 77 new samples and V1's 12, in the same registered `references`
  run or a second registered run with slug `references-v8sample`.
- Write the fields and the rule into a timestamped file under `state/` before any of these readings
  is computed.
- **Stop and ask** if the references' total wall time looks like it will exceed the census's, about
  5 h 40 min, for all four parts together: 06i's 13, V3's 77, V1's 12 and V8's 12. Once 06i's 13
  have been run, project from their wall times. If the projection already exceeds 5 h 40 min, end
  your turn with it before launching the rest.

### What you record, in a table of its own

Put the table in `out/v8_sample.md` and its data in `out/v8_sample.json`. It is separate from V3's
tables, and V3's verdict does not use it.

For each of the 12 calls:
- its four readings, their scatter and wall times, and whether it counted;
- for each counted call:
  - the reference value;
  - 06h's value, declared error and converged flag;
  - $g$'s value, declared error and converged flag;
  - $|v - {\rm ref}|/{\rm declared\ error}$ for each code.

Give every value in `float.hex`.

**The question it answers, per call:** does $g$'s value, returned from untrusted slow leaves, lie
within its declared error of the reference, and does 06h's? Report the numbers per call.
Interpretation stays with the laptop.
