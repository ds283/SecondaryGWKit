## Addendum to the references agent's brief — V1's 12 group calls, referenced and scored apart

**The source.** The user's ruling of 2026-11-01, relayed by the laptop orchestrator: add V1's 12
failing group calls to the Filon reference sample, but score them apart from V3. The user's words:
"Happy to have the Filon references added – that sounds like a good idea."

**What does not change.** V3's sample (Z3's 30 plus the 60 drawn), its draw, its seed, its count
rule and its scoring are exactly as your brief says. These 12 calls come from the rehearsal stores
R23 and R24 (the R8 configuration), not from the census, so they cannot enter V3's draw. **If
including them would change how V3's own sample is drawn or scored in any way, stop and ask.** Do
not adjust.

### The calls

These are the 12 rows of R23/R24 where V1 fails on `total`. Each row's whole move is in one group
call: sub-interval 1, group `G+q+r`. The identities and both stores' values come from the rehearsal
agent's records, which you read only:
- `var/levin-phase-floor-06g/rehearsal/out/qsi_rows.csv`, rows with `V1 == VIOLATES`;
- `var/levin-phase-floor-06g/rehearsal/out/v1_violations_detail.txt`, which gives each call's
  `v23`, `v24`, `e23` and `e24` in `float.hex`, and its convergence, phase limit and regions;
- its script `var/levin-phase-floor-06g/rehearsal/scripts/v1_violations_detail.py`, which shows
  where those fields are stored.

All 12 rows are at `log10_rtol = -7` and `log10_atol = -32`, with 5 Levin calls in each store:

| k | z_response |
|---|---|
| 1e5 | 3971.4125591760326, 5236.5530126517315, 6904.718924492998 |
| 316227.7660168379 | 434.65132503921666, 573.1146466584687, 996.4201622779478, 6904.718924492998 |
| 1e6 | 329.64045755914753, 573.1146466584687, 996.4201622779478, 1732.3813753196057, 6904.718924492998 |

### The captures

Take each call's integrand and endpoints exactly as R23 and $g$ give them.
- **At $g$:** capture from the code worktree (`1c5336e`) against the R24 store. The capture must
  reproduce R24's stored value and error for that group call bit for bit (`v24`, `e24`).
- **At the parent:** capture from the parent worktree (`4f4f9ed`) against the R23 store. The
  capture must reproduce R23's stored value and error bit for bit (`v23`, `e23`).
- **The two captures' groups** (coefficients, orders, signs and spans) must be bit-identical,
  because the integrand is 06h's (Gate V's second part). Say what you compared and how. A capture
  that does not reproduce its stored record, or two captures whose groups differ, is a stop.

The rehearsal stores are read only:
- open every one with `mode=ro`;
- create nothing beside them (no journal, `-wal` or `-shm`);
- hash all ten files (SHA-256) before and after. The baseline is the rehearsal agent's
  `var/levin-phase-floor-06g/rehearsal/state/stores_sha256_final.txt`, and a file that differs is a
  stop.

No process initialises Ray, and you take no lock.

If the census harness cannot point at a rehearsal store, the change that lets it do so is allowed:
- it is a numbered deviation;
- it must not change any computed value;
- the bit-for-bit reproduction above is the proof.

If no such change works without going out of scope, stop and ask.

### The route and the count rule

- These are cosmological groups, so use **Route B** (the least-squares smooth phase), as for V3's
  cosmological-group calls.
- The count rule is unchanged: four readings (QAWO and FCC, each at two resolutions). A reference
  counts only when all four agree to within 1/100 of the minimum declared error. Here that minimum
  is $\min(e_{23}, e_{24})$ for that group call.
- **Never loosen the tolerance** to make one count. A call that cannot be counted is recorded as
  uncounted, with its four readings and their scatter.
- Readings over 30 minutes are stopped, as for V3.
- At most 6 processes, and never while the census runs.

### The order

Run these as a stage after V3's 77 new samples, in the same registered `references` run or in a
second registered run with slug `references-v1groups`. Write the fields and the rule above into a
timestamped file under `state/` **before** any of these readings is computed, and do not edit it
afterwards. This table carries no prediction to score: it is descriptive.

### What you record, in a table of its own

Put the table in `out/v1_groups.md` and its data in `out/v1_groups.json`. It is separate from V3's
tables, and V3's verdict does not use it.

For each of the 12 calls:
- its four readings, their scatter and wall times, and whether it counted;
- for each counted call:
  - the reference value;
  - R23's value and declared error;
  - $g$'s (R24's) value and declared error;
  - $|v - {\rm ref}|/{\rm declared\ error}$ for each code.

Give every value in `float.hex`.

**The question it answers, per call:** which of the two codes is nearer the reference, and does
that code's declared error honestly cover its distance from it? Report the answer per call.
Interpretation stays with the laptop.
