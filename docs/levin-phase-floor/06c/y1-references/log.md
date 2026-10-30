# Y1 references: log

Brief: `dispatch.md` (this directory), from the Mac Studio orchestrator, 2026-10-29T17:12Z. Host
`mps026298.phys.susx.ac.uk`. Trees: R17 `fef401c` (`SecondaryGWKit-lpf06c-fef401c`), R18
`e5764a6` (`SecondaryGWKit-lpf06c-e5764a6`), both detached, clean, HEAD as expected in every
output header. Main checkout at `28ebaf0`, clean. Python: the main checkout's venv, always with
`-B` and an explicit `PYTHONPATH=<tree>:<scripts>` (the login shell carries a `PYTHONPATH` of
`/opt/local/lib/python3.11/site-packages:`, which every command here overrides).

Every command's output is under `out/`, numbered in the order run. Working state (captures, pickles,
per-call reference files) is under `state/`.

## Sequence

| out | what |
|---|---|
| 00 | SHA-256 of the ten original store files, before anything else (equal to the review's `check3_r1{7,8}.before.sha256`). |
| 01 | the five files of each store copied (`cp -p`) to `store_copies/`; every copy hashed equal to its original; originals unchanged. |
| 02 | `rows.py`: the 30 Y1-violating rows from the copies (sqlite3 `mode=ro`), keyed by physical identity; same 864 keys, 838 scored, 30 violations, same order and ratios as `review/checks/check4_y1.txt`. Each row's shard and serial in each store. |
| 03 | development: capture of Y01 at R18 (found that the analytic oracle's 16 Bessel calls also pass through the wrapped name; the capture now keeps only calls whose notify_label names a group). |
| 04, 05 | `capture.py`, every row, each tree, on its own copy: total and total_abserr recomputed bit-identical to the store for all 30 rows in both trees; every group call's (1+z)·value and (1+z)·abserr and its region count bit-identical to the store's metadata. |
| 06 | `moved.py`: 154 moved calls in 30 rows (value, abserr, num_regions or max_depth not bit-identical); every pair of calls has the same label, span, notify_label and keyword arguments in both trees. |
| 07, 08 | second capture of the top 12 rows in each tree, pickling the 56 moved calls' inputs (span, f, theta dict, kwargs); again bit-identical to the store. |
| 09 | development probe of Y01. |
| 10, 11 | `probe.py`, top 12, each tree on its own pickles: every replay reproduces its capture and its store bit for bit (value, abserr; regions and depth too). Oscillation counts and theta' profiles. |
| 12-15 | development of `refquad.py` on Y01's calls 0 and 1 (panels, integrate, R17 comparison). |
| 16-19 | capture with pickles and probe for the other 18 rows, both trees: every replay reproduces its capture and store bit for bit (98 calls each tree). |
| 20- | reference batches (`batch.py`); see below. |

## Deviations

1. **IMPLEMENTATION CHOICE: own copies of rebuild.py and harness.py.** `scripts/rebuild.py` is
   the census's, changed only (each change marked "Y1") to the rehearsal scope
   (`quadsource_atol_sweep.build_scope` with `--rehearsal`: k = logspace(5, 6, 3), `--zend 250`,
   everything else as A3 v2's), to the store copies (`use_store`), and, in `rebuild_row`'s plain
   path, to the stored `QuadSource` in place of prompt 05's in-process recompute (R17 and R18 each
   hold the `QuadSource` rows their own tree computed, which main.py's stage reads back).
   `scripts/harness.py` is the census's, unchanged. Every check `rebuild_row` makes against the
   stored row passed for all 30 rows in both trees, and the recomputed rows equal the stores bit for
   bit, which is the evidence that the rebuild is main.py's.
2. **IMPLEMENTATION CHOICE: two captures.** The first capture records every call's result only, to
   find the moved calls; the second (same code, same rows) pickles only the moved calls' inputs.
   Both reproduce the store bit for bit.
3. **IMPLEMENTATION CHOICE: the replay is in a fresh process from the pickle**, in the call's own
   tree (probe.py, and again at the start of every `refquad.py` step), not in the capturing process.
   The reference integrates exactly the unpickled callables whose replay reproduced the store.
4. **IMPLEMENTATION CHOICE: the reference method.** Composite Gauss-Legendre (scipy
   `roots_legendre`) on panels, n = 192 and n = 384 nodes per panel on the same panels (the brief's
   "doubling the nodes per panel"). Panels are equal steps of W(x) = Phi(x)/512 + (x-a)/0.005, Phi
   the integral of |theta'| (theta' the call's own `theta_deriv`, trapezoid on 1e-3 cells), so each
   panel spans at most ~512 rad of phase (~81 oscillations; at 192 nodes, ~2.4 nodes per oscillation
   at the low resolution and ~4.7 at the high, where Gauss-Legendre with 2n-1 >= 1.5 * (phase/2)
   converges like exp(-0.2 (2n-1)) for an oscillatory factor) and at most 0.005 in x. The
   oscillation count was taken first (probe.py) and sized the runs. Terms summed exactly (Shewchuk
   partials merged across workers, one final rounding). This is not the Levin driver.
5. **IMPLEMENTATION CHOICE: the cross-tree comparison is on a dense sample**, every 16th panel
   (thinned to at most 3000 sampled panels per call), at both node sets: f_sin, f_cos,
   theta_mod_2pi, theta and theta_deriv, plus theta_mod_2pi at the sampled panels' ends, compared
   by bit pattern. R18's tree computes the reference; R17's tree evaluates its own unpickled
   callables at the same nodes.
6. **IMPLEMENTATION CHOICE: the reference total.** The reference total is formed exactly, in
   rationals, as the store forms its total: numeric_quad + Σ (1+z_response)·value. Each moved call
   with a reference enters at its reference value, each unmoved call at its shared value. A moved
   call without a reference enters at R17's value and, separately, at R18's. The verdict is
   reported because it is the same both ways in every row. Errors in the unmoved calls, which are
   identical in both codes, are not referenced.
7. **IMPLEMENTATION CHOICE: the `--dedupe` twin check (batches B1).** It looked for a G+q-r / G-q+r
   pair with the same span, so that one reference could serve both. It never found one bit-identical
   on the dense sample. (q ≠ r in most of those rows; and even with q = r, (θ_G + θ_q) − θ_r and
   (θ_G − θ_q) + θ_r round differently.) So every call got its own reference. The equal references
   in equilateral rows are therefore independent computations that agree.
8. **STRUCTURALLY REQUIRED: the 1/100 accuracy target is not reached on 15 of 137 calls**
   (0.010–0.099 on 14, 0.76 on Y22-04). All 15 are `G±q±r` calls whose R17 abserr is near the
   double-precision noise floor of the integrand (out/26: a 384/768 re-run of Y16-06 does not
   lower the difference). No verdict depends on them. Reported in results.md.
9. **IMPLEMENTATION CHOICE: dev outputs 03, 09 and 12–15** were made while the scripts were being
   written. capture.py was changed after 03 (group-call filter); refquad.py's panel scheme was
   changed after the first 12 run (12 was then re-run with the final script, then 13–15). Every
   result used is from the batches 20–29 (refquad.py sha256 0e292159…), whose Y01 rows supersede
   the dev ones.
10. **UNINTENDED DRIFT (cleaned up): two scratch files written to `/tmp`.** Provisional assembly
    outputs (`dev_results.json`, `dev_results.md`) were moved to `/tmp` and then deleted there.
    They were superseded by out/30 and held nothing else. Nothing else was written outside
    `y1-references/`.

## Reference batches

| out | calls | wall |
|---|---|---|
| 20 | A1: the sub-0 `G` call of each of the top 12; `G+q+r` of Y01, Y02, Y05, Y06, Y09, Y10, Y12 | ~15 min |
| 21 | A2: `G+q+r` of Y03, Y04 | ~23 min |
| 22 | A3: `G+q+r` of Y07, Y08, Y11 | ~25 min |
| 23 | B1: every moved call of Y15–Y18, Y20–Y30 | ~10 min |
| 24 | B2: every moved call of Y13, Y19 | ~30 min |
| 25 | C1: the other moved calls of Y01, Y12, Y10, Y09 | ~37 min |
| 26 | 384/768 re-runs of Y22-04 and Y16-06 (accuracy floor check) | ~1 min |
| 27 | C2: the other moved calls of Y06; Y02 `G±q∓r` | ~36 min |
| 28 | C3: `G-q-r` of Y02, Y08 | ~35 min |
| 29 | D: Y14's `G`, `G+q-r`, `G-q+r` | ~8 min |

At most six worker processes at any time; each launched computation was under 45 min. No Ray was
initialised (every capture reports `ray.is_initialized()` False at its end); the rehearsal lock and
the run registry were not touched (no registered run was needed).

## Not done (17 moved calls; estimated wall at 6 workers, from the measured rate)

- Y03: `G+q-r` 38 min, `G-q+r` 38 min, `G-q-r` 62 min.
- Y04: `G+q-r` 36 min, `G-q+r` 30 min, `G-q-r` 50 min.
- Y05: `G+q-r` 21 min, `G-q+r` 21 min, `G-q-r` 37 min.
- Y07: `G+q-r` 27 min, `G-q+r` 27 min, `G-q-r` 47 min.
- Y11: `G+q-r` 51 min, `G-q+r` 51 min, `G-q-r` 92 min.
- Y14: `G+q+r` 38 min, `G-q-r` 48 min.

That is about 11.5 h of 6-way wall in all. Four of these calls would each run over 45 min (Y03,
Y04, Y07 and Y11 `G-q-r`), and Y11's two `G±q∓r` calls are just over it. Each of those would need
a registered run. None of the 17 has a stationary point. In these rows each moves by ≤ 1e-3 of the
row's `G+q+r` movement, and the verdicts hold with them at either code's value.

## Store hashes and clean-up

- out/31: all ten originals are unchanged against out/00, and all ten copies against out/01.
- out/32: the copies were then deleted, and the directory `store_copies/` removed.
- `state/` holds 2.6 GB: the pickled inputs (262 MB), and the reference samples and logs
  (`state/ref`, 2.3 GB, mostly the `*.sample.npz` the R17 comparisons read). Kept for re-checking.
  They can be deleted.
