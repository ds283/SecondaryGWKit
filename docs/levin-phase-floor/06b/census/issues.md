# levin-phase-floor 06b §5 (the census): observations to open on the board

Each observation below gives its numbers and where they are recorded. The paths are relative to
this directory (`06b/census/` once committed). The run is
`levin-phase-floor-06b-census-20261028T161212`.

## C1 — W2 fails: where the premise check bisects, b's `abserr` understates its error

**Measured** (`out/cmd/35`, `37`, `38`, `39`; `out/w2_reference.json`; `state/w2_selection.json`):
- **The failures.** On 17 of the 24 sampled calls, $|{\rm value}_b - S_{10}| >$ `abserr`$_b$. The
  ratio has median 1.79 and maximum 19.7 (`b-1e-7-3-34455` call 1, G+q+r, mid, k 1e5). For all 24:
  - the reference converges: $|S_9 - S_{10}| \le 0.0066$ `abserr`$_b$;
  - each call re-runs from its saved inputs bit for bit.
- **The reference holds up** on three calls cross-checked (`38`):
  - $S_{11}$ agrees with $S_{10}$ to 0.002 `abserr`$_b$;
  - on #1, the adaptive driver at `rtol` 1e-11 (215 regions) agrees with $S_{10}$.
- **The error is invisible to the residual.** On #21 and #4, the driver at `rtol` 1e-11 returns
  b's value exactly, with the same 4 and 2 regions, each accepted as resolved. Their residuals
  (3×10⁻³⁹–1.2×10⁻³⁴) and floors (7×10⁻³⁹–1.6×10⁻³⁴) sit far below the error, which is 4.4×10⁻³³
  and 9.5×10⁻³⁴. A region and its halves carry the same error, so the step-(4) residual cannot see
  it, and the eq. (151) floors of the bisected regions do not bound it.
- **b's values are better than 06a's** (`39`).
  - Against each call's own request max(atol, rtol|value|): b exceeds it on 4 of 24 (max 3.42×,
    `b-1e-7-3-34455` call 1); 06a exceeded it on 23 of 24.
  - b's `abserr` is 1/10 to 1/16,500 of 06a's, which on 17 of 24 calls is below the true error.
  - 06a's `abserr` covered the true error on 23 of 24 (median 0.068 of it; max 1.13).

**Impact.**
- **The fix itself:** the premise check removes 06a's over-wide floors (no group call is
  phase-limited under b at either rtol). The regions it bisects then report errors smaller than
  their true error. The calls are flagged `converged` while some are outside tolerance: 4 of 24
  sampled, at up to 3.4× the request.
- **Closure:** this bears on whether `[07-roundoff-exit-assumes-bisection-cannot-help]` closes as
  "§4 and §5 hold". §5's W2 does not hold.

**Next step:** the user's decision. Possible directions are a floor model for the bisected regions,
or an estimate of the shared error. Not acted on here.

## C2 — W4 fails: three analytic-block calls chase to depth 20 at 1e-8

**Measured** (`out/cmd/37`): three analytic-block calls at 1e-8 failed the premise on 4–7 regions,
then subdivided to `depth_max = 20`:

| Unit | Call | Regions | Evaluations | Unit wall | Row `total_converged` |
|---|---|---|---|---|---|
| `b-1e-8-2-1704` | 16, analytic Y1 | 1,136 (640 at depth 20) | 4,543 | | true |
| `b-1e-8-2-13964` | 5, analytic J1 | 31,885 (21,840 at depth 20) | 127,539 | 159 s | true |
| `b-1e-8-3-34987` | 9, analytic J2 | 14,293 (6,960 at depth 20) | 57,171 | 69 s | false |

Under 06a no call deeper than 13 was recorded at either rtol: the group calls, which were all 06a
recorded. No unit was over budget. At 1e-7 the largest depth under b is 18 (analytic) and 14
(group).

**Impact.** This is the mechanism the B.5 addendum foresaw from LiouvilleGreen seed 15 `test_JJJ`
(`[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`): once the premise fails, nothing
stops the bisection short of the cap. It bears on G2's L2. The cost is real (159 s for one unit
against 06a's 1 s maximum), and two of the three calls still end `converged`.

**Next step:** a finding for the user (W4's own text).

## C3 — 06a's stationary-point calls move by up to 6.2×10⁴ of their 06a abserr (07's)

**Measured** (`out/cmd/37`, `out/moved_calls.csv`):
- **How far rows move.** 32 rows at 1e-7 (36 at 1e-8) move by more than 06a's `total_abserr`, up
  to 2.2×10⁴ of it.
- **Which calls.** At 1e-7, 30 of the 38 group calls that move by more than their 06a abserr have
  a sign change of θ′ in 06a's scan; at 1e-8, 35 of 47.
- **The largest moves.** These are G+q+r calls that 06a accepted phase-limited at depth 0, on a
  region holding a stationary point, and that b bisects to depth 11–12. Two examples:
  - `b-1e-8-0-1867` call 1: 6.2×10⁴;
  - `b-1e-7-3-38029` call 1: 2.5×10⁴.
- **The 720 stationary-point rows:** 323 move at 1e-7 and 370 at 1e-8.
- **Not verified:** these calls are outside W2's sample (which excludes sign changes), so neither
  side has been checked against a reference.

**Impact.** 06a's (and production's) values on these rows may have been wrong by far more than
their declared error, and b's are not yet verified. They are 07's rows (the stationary-point
guard), and this census is 07's "before".

**Next step:** 07 §6 compares against this run.

## C4 — the analytic blocks are where the exit fired most, and 06a never recorded them

**Measured** (`out/cmd/37`):
- **Under 06a:** 12,633 analytic-block Levin calls at 1e-7 and 17,722 at 1e-8 reached the old
  exit, by the premise-counter reading the user accepted. The group calls reached it only 1,213
  and 1,720 times.
- **Under b:** 32 analytic-block calls at 1e-7 (800 at 1e-8) are still phase-limited, and all of
  W3's 44 regions are theirs.
- **Not recorded:** 06a's census and A3 v2's stored `total_phase_limited` cover the group calls
  only.

**Impact.**
- **Descriptive statistics:** any statistic over "phase-limited calls" that comes from 06a's
  records, or the stored flag, leaves out most of the exit's firings.
- **W2:** the sample covers group calls only. No analytic block's moved value has been checked
  against a reference.

**Next step:** record, so that 07's census includes the analytic blocks (this run's
`other_calls` and `other_regions` hold them).
