# levin-phase-floor 06e §5 (the census): observations to open or amend on the board

Each observation below gives its numbers and where they are recorded; paths are relative to this
directory (`06e/census/` once committed). The census run is
`levin-phase-floor-06e-census-20261030T081318` and the "before" is 06c's
`levin-phase-floor-06c-census-20261029T130214`. The orchestrator decides whether each is a new
issue or an amendment.

## D1 — Z2 and Z5 fail on one call whose floor exceeds its request

**Measured** (`out/cmd/55`, `58`): `g-1e-8-0-33995` call 5 (G−q+r; k = 3.14e5, q = 1e5,
r = 3.14e5 /Mpc; z_response 4.809e6). The row is outside the 720: 06a's scan finds no sign change,
min|θ′|/max|θ′| = 0.0028.
- **Under $g$ at 1e-8** the call is unconverged at depth 20:
  - 8,706 regions (8,600 Clenshaw–Curtis), 5,642 floor-limited and 3,064 at depth 20, unchecked;
  - 30,707 solves; the unit's wall rose from 9.2 s to 105 s;
  - its floor sums to 1.33e-33 against atol 6.7e-34.
- **Under 06c** the call stopped at depth 13: all 688 leaves floor-limited, floor 4.99e-32,
  unconverged.
- **At 1e-7 under $g$** it converges at depth 11.

**Impact:** 06e's node term lowered this call's floor 37×, but not below the request. The loop
then splits the floor-limited and near-floor leaves to `depth_max` without lowering the floor sum.
This is the flat-floor chase of `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`, now
on a floor that exceeds the request.

**Suggested:** amend `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`, or
`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`'s family, with this call. It is the only
depth-20 call outside the 720 rows at either rtol.

## D2 — Z3 fails on sample 3; 06c was dishonest on the same call

**Measured** (`out/cmd/57`, `references/out/z3_scores.json`, `references/out/ref-s03-*`):
- **The call:** `g-1e-7-3-34370` call 4 (G−q−r), converged at depth 2 with 4 regions.
- **The failure:** $|v_g - {\rm ref}| = 1.245$ `abserr`$_g$.
- **The reference is sound:** its three resolutions agree to 0.0083 of min(abserr).
- **06c** was at 1.197 `abserr`$_{06c}$ on the same reference.
- **The other five counted samples:** 0.78, 0.998, 0.998, 0.9985, 0.9995.

**Impact:** 06e does not make this call honest. It was already dishonest by about the same margin
under 06c's census code (06c plus 06d), so the failure is not 06e's introduction. It is a further instance of
`[06b-moved-calls-declare-less-than-their-error]`.

**Suggested:** narrow `[06b-moved-calls-declare-less-than-their-error]` with this call.

## D3 — |v − ref| / abserr sits just under 1 in both codes

**Measured** (`out/cmd/57`): on samples 2, 8, 9 and 14, $|v - {\rm ref}|/{\rm abserr}$ is
0.998–0.9995 in both $g$ and 06c. It is 0.95–0.985 on 10 and 23 (dropped), 1.2–1.25 on 3, and
0.78 on 15.
- The relative difference value − reference is about 1.2e-6 on samples 2 (rtol 1e-7) and 14 (rtol 1e-8), 12× and 120× `rtol`.

**Impact:** the declared error appears to track a systematic difference of nearly its own size
rather than to bound a smaller one. On these calls the floor's declared-phase term dominates (see
D4). This is unexplained here; it is not a Z3 failure except at sample 3.

**Suggested:** open an issue for 07 or a follow-up: what sets |v − ref| ≈ abserr on these calls.

## D4 — Clenshaw–Curtis floors are 99.4 % the declared-phase term

**Measured** (`out/cmd/55`): over the Clenshaw–Curtis leaves whose floor carries their error
(5,628 calls at 1e-7, 7,326 at 1e-8), the floor sums break down as:

| Term | Share |
|---|---|
| `_declared_endpoint_phase_err`, the crude `f_scale × width/2` proxy | 99.4 % |
| (n+4)uS | 0.46 % |
| 06e's node term $\phi_{\rm node}$ | 0.06 % |
| argument term | 0.03 % |

Per call, the declared term is the largest floor term on 2,493 and 2,785 calls.

**Impact:** with 06d's node term replaced, the Clenshaw–Curtis floor on production calls is
whatever the declared `theta_abserr` proxy says. This bears on D1 and D3, and on any later
tightening.

**Suggested:** open an issue (the CC branch's declared-term proxy), for the orchestrator to
assign.

## D5 — `*-1-2174` call 1 converges at 1e-8 on depth20-tiling's pinned value

**Measured** (`out/cmd/55`):
- At 1e-8, `g-1e-8-1-2174` call 1 converges at depth 7 with 18 regions. Its value,
  `0x1.0c6c2cd926acfp-86`, is depth20-tiling's pinned counterfactual value bit for bit, and
  2,295 × `abserr`$_{06c}$ (1,678 × `abserr`$_g$) from 06c's.
- At 1e-7 it stays at depth 20, unconverged, close to 06c's value.

**Impact:** a stationary-point row (07's class) reported converged at one rtol and probably
dishonest, as depth20-tiling suspected (unverified).

**Suggested:** amend 07's stationary-point issue with this row.

## D6 — Y1's 12 equilateral rows are not A3 v2 rows

**Measured** (`out/cmd/56`): y1-references' Y01–Y12 have (k, q, r, z_response) values that match
no row of the census's `rows.json`. They come from the R17/R18 rehearsal stores.

**Impact:** the prompt's descriptive item "by name Y1's 12 equilateral rows" cannot be reported
from this census. They appear only in §4's R19/R20 comparison.

**Suggested:** none beyond noting it; the rehearsal agent's records carry them.
