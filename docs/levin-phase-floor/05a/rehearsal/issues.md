# Issues found by the rehearsal agent, levin-phase-floor 05a §5 (Mac Studio)

These are proposals for the board. The orchestrator decides whether to open them.

**No new issue is proposed.** Every finding below falls under an issue that is already open, and
each is recorded there for the board to weigh.

## Recurrences of open issues (observations)

- **`[05-store-value-diff-compares-volatile-columns]`** recurs as 05 described it. R11 against R12
  differ outside `QuadSourceIntegral` only in timings, labels, `GkSource.parents_digest` and the
  GkWKB residual-memo counters ([`out/classify_non_qsi.txt`](out/classify_non_qsi.txt): 8607,
  5148, 291 and 62 column-differences; 0 unclassified). Every value table is bit-identical.
- **05's P2, run-to-run nondeterminism of the residual-memo counters.** The counters differ in 31
  GkWKB rows here and in 25 in 05's R9/R10. This supports the attribution to scheduling, but still
  does not prove it: no same-code pair exists.
- **`[04-roundoff-floor-can-be-infinite]`.** 26 `QuadSourceIntegral` rows have `total_abserr` = inf
  in both R11 and R12, at depths 0–8 ([`out/qsi_extra_counts.txt`](out/qsi_extra_counts.txt),
  [`out/qsi_rows.csv`](out/qsi_rows.csv)). 05a neither causes this nor changes it.
- **`[10-the-readers-draw-their-work-lists-unseeded]`.** The readers' file counts differ from 05's
  R10 (Gk 387 against 450, GkSource 162 against 142, QSI 73 against 74), with the same lookups.
