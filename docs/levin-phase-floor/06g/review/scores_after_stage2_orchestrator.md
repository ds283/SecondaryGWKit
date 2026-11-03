# References scored after stage 2 of the rerun batch: the orchestrator's summary and checks (2026-11-02, about 20:55Z)

**Sources:**
- the references agent's `references/out/scores_after_stage2/report_20261102T204420Z.md` and
  `scores_20261102T204420Z.json`, made by `references/scripts/score_refs.py`;
- the orchestrator's own checks, `review/checks/rerun_stage2_check.txt` and
  `review/checks/stage2_recheck.txt`, from `review/scripts/stage2_recheck.py`.

**Stage 3 of the rerun batch is still running and was not touched.**

## What the orchestrator checked itself

**The new tool's readings:**
- the six re-reads under `ff3e8394` (tool SHA-256 `5ed56fc2…`) are bit-identical to their old
  readings;
- all 56 stationary-point readings meet the usable rule, clause (ii): `direct_segments` 1, equal
  to their one non-monotone segment.

**z00 and z12, recomputed from the readings:**
- z00's four new readings agree to 0.978 of their tolerance;
- z00's g value is 0.02 of its abserr from them;
- z12's g value is **1,681** of its abserr from them (06h's, 1,678).

**Why "as written" excludes z00 and z12.** The references agent's predictions file
(`references/state/predictions_20261101T175120Z.md`, written 17:51Z on 1 November, before any
reading) requires that "on Route B, every reading reports `tmap.all_monotone` true". The
stationary-point readings meet only clause (ii), so z00 and z12 cannot count as written. The
exclusion follows a rule fixed beforehand; it is not a choice made after the result.

**z00 and z12's readings are bit-identical.** Their captured pickles differ (`1b870b17…` and
`7b8792e5…`), as the two rtols' records differ. The orchestrator has **not yet checked** that the
two captured integrands are equal. That check is open in the review: their equality is the
premise behind "the same integral, so z12's Levin value is wrong".

## The scores

| Prediction | As written | As the ruling scores it |
|---|---|---|
| **V3** | **holds**: 66 of 90 count; no violation | **fails**: all 90 scored (80 with two implementations, 10 FCC-only); one violation, z12 at 1,681 × abserr_g. Restricted to two-implementation references, also fails, on z12 |
| **V4** | **holds**: z02 0.00121, z08 2.1e-7, z09 1.44e-4, z14 0.00121 | **holds** |

## V1's 12 (the R23/R24 group calls)

**R24, the new code, is nearer the reference than R23 on all 12, against every reading.** R23 is
4.5e3 to 6e5 of its abserr from every reading.

**Finding: the four readings disagree far beyond the tolerance on every call.**
- FCC r1 is the outlier, 117 to 11,900 × e24 from the other three.
- FCC r2 and QAWO also part on w07–w11.
- The ruling makes FCC's two resolutions disagreeing "a finding to stop and look at".
- The scorer takes FCC r2 as the reference.

R24's abserr against each reading:

| Call | vs FCC r2 (the scorer's reference) | vs QAWO r1 and r2 |
|---|---|---|
| w00 | 0.27 | 0.005 |
| w01 | 0.28 | 0.82 |
| w02 | 0.23 | 0.003 |
| w03 | **1.53** | 0.57 |
| w04 | 0.60 | 0.002 |
| w05 | 0.23 | 0.06 |
| w06 | 0.33 | 0.12 |
| w07 | **329** | 0.94 |
| w08 | **79.5** | 0.01 |
| w09 | **12.6** | 0.06 |
| w10 | **32.7** | 0.39 |
| w11 | **70.8** | 0.05 |

- **Against both QAWO readings,** which agree with each other closely, R24 is within its abserr on
  all 12.
- **Against FCC r2,** it is within on 6.

## The V8 sample (g unconverged, 06h converged, on every call)

| Call | g: \|v − ref\|/e | 06h: \|v − ref\|/e |
|---|---|---|
| v00 | 0.075 | 0.058 |
| v01 | 0.366 | 0.566 |
| v02 | 0.0053 | 0.0090 |
| v03 | **1.203** | **2.311** |
| v04 | 0.0002 | 0.255 |
| v05 | 0.0045 | 0.163 |
| v06 | 0.075 | 0.058 |
| v07 | 0.043 | 0.605 |
| v08 | 0.0073 | **4.293** |
| v09 | 0.0061 | 0.311 |
| v10 | 0.0011 | 0.373 |
| v11 | 0.429 | **1.074** |

- **g** is within its abserr on 11 of 12.
- **06h** is within on 9 of 12.
- v00 and v06 are the same call at two rtols.

## The other results

**The correction −ΣE** (25 references with a Clenshaw–Curtis leaf):
- it is nearer the reference on 12, and about neutral on most, since ΣE is tiny;
- it helps on r51 (0.35), r43 (0.47), r33 (0.54) and z15 (0.93);
- it is worse on r09 (1.345) and r23 (1.015).

**z00 across hosts:** all four readings are bit-identical to the laptop's.

**Marked readings:**
- r24, r26 and r53 are FCC-only for now;
- r47 has three readings;
- r52 and the ten long-span Route A calls have three readings each.
