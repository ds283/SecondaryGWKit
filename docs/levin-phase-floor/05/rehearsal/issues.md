# Issues found by the rehearsal agent, levin-phase-floor 05 §5 (Mac Studio)

These are proposals for the board. The orchestrator decides whether to open them.

## P1 `[05-store-value-diff-compares-volatile-columns]` (proposed)

`tools/store_value_diff.py` compares every column of a row except its serial, its timestamp and
the columns that name another row. Two builds of the **same** code cannot pass that test. On
R9 against R10 ([`out/classify_non_qsi.txt`](out/classify_non_qsi.txt)), every difference outside
`QuadSourceIntegral` falls into one of four classes, and none of them is a computed value:

- **Wall-clock timings.** Every `*compute_time` and `*RHS_time` column, in 7 classes.
- **`label`.** `main.py` writes the job name followed by `datetime.now()`. Once the timestamp is
  removed the two labels are equal (6 classes).
- **`GkSource.parents_digest`** (291 of 291 rows). It is the SHA-256 of the parents' **serials**
  (`ObjectFactories/GkSource.py`, `GkSource_parents_digest`). It names other rows, but through
  their serials, which the tool cannot resolve. Every row is matched by identity, so the parent
  sets agree by identity; only the serials differ.
- **The residual memo's warmness.** In `GkWKBIntegration`, 25 of 3477 rows differ in `metadata`
  `rho_reused`/`rho_evals` and in `stage_1_RHS_evaluations`
  ([`out/gkwkb_reuse_rows.txt`](out/gkwkb_reuse_rows.txt)). In each row one store reused the
  per-process residual table and the other built it. The swaps go both ways within each $k$ (A
  reused / B built: 11 rows; the reverse: 14 rows). This is consistent with whichever Ray worker
  first meets a $k$ building that worker's memo. Every `GkWKBValue` row is identical.

**Impact:** the tool alone cannot certify "everything outside the touched classes is identical" by
its exit code, or by a class being reported value-identical. Prompt 05 §5.4 and §8 assume it can.
**Next step:** decide whether the tool should report volatile columns (timings, labels,
serial-derived digests, cache counters) separately from computed values, or whether each use of
the tool should classify its differences, as `scripts/classify_non_qsi.py` does here.

## P2 — no same-code pair separates run-to-run nondeterminism from the change (observation)

The `rho_reuse` attribution in P1 is an inference from the pattern of swaps. Producing direct
evidence would take a second build at one tree, compared with this tool. No such pair exists: the
retired R3, R4 and R8 stores are gone. It is recorded here and not acted on.
