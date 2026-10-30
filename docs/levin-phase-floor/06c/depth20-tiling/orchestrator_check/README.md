# Orchestrator's re-check of depth20-tiling

- **Host:** mps026298
- **Time:** 2026-10-29, 22:29–22:55Z
- **Tree:** `e5764a6`, clean. Nothing runs from any other tree.
- **Writes:** read-only throughout; nothing is written outside this directory.
- **Inputs:** `inputs/sNN.pkl` are the agent's captured inputs, sliced out of `state/capture/` and checked against `captured.json`'s SHA-256.

## The checks

**`01-pin_replay.txt`: my own end-node pin, run on samples 1, 2, 8, 9, 12, 13, 22 and 25.** The script is `pin_replay.py` (sha256 `006e4238…`).
- It swaps out a module attribute in this process. Only x[0] := b and x[−1] := a change; on 4 spans × 2 orders the interior nodes and D stay bit-identical.
- Every as-is replay equals its capture.
- The pinned depth, regions and |Δv|/abserr equal the agent's counterfactual table on all eight.

**`03-cheap_recount.txt`: my own lineage reconstruction over all 262 lines.** The script is `cheap_recount.py` (sha256 `b67b87fd…`).
- All 262 reproduce the recorded leaf.
- Over the last five levels, 1,179 of the 1,222 Levin levels with residual > 10× floor do not tile. All 17 with residual ≤ 10× floor tile.
- Over all levels, 3,777 of 3,867 flagged levels do not tile. But so do 1,284 of the 1,564 levels that are not flagged.
- **So not tiling is the base rate**, at 93% of all Levin levels. The cheap pass is consistent with the mechanism but does not by itself attribute anything. The attribution rests on the pinned re-solves and the counterfactual.

**`02-ref-*.txt` and `04-ref-*-P64.txt`: my own Gauss–Legendre references for samples 1, 9 and 25.** The script is `orch_ref.py` (sha256 `8a0d291d…`). It ran at three resolutions: 64/96 nodes per 128 rad, then 96/128 per 64 rad.

**`05-honesty.txt`: the results.**

| sample | references converge? | census | pinned |
|---|---|---|---|
| 9 (Y2, 1e-7) | yes, to 0.0045 of min(abserr) | 0.991 | 0.998 |
| 1 (G+q-r, 1e-7) | no: scatter 0.4–0.6 of the pinned abserr | 0.12 | 0.63 |
| 25 (Y4, 1e-8) | no (below) | 4.9e-5 | 9.2 |

The census and pinned columns are |v − ref| / abserr. Sample 9 is honest both ways. Sample 1 is consistent with honesty both ways, but that is not established.

**Sample 25:**
- The references scatter by about 1.4e-24 across resolutions, without converging. That is 65× the pinned run's declared abserr of 2.1e-26, and 1.6e-3 of the census's 8.9e-22.
- This is well above the double-precision noise floor, about 1e-16 × Σ|terms| = 3e-27. It suggests an integrand that is not smooth at the 1e-24 level, for example interpolant knots.
- **The census value is honest.** The pinned run's declared error, 4e4× smaller than the census's, can be neither confirmed nor refuted here.
