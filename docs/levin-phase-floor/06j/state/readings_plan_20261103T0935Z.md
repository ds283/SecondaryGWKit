# levin-phase-floor 06j — Part W: the plan for the reference readings (for the orchestrator's check)

Written 2026-11-03T09:35Z, after run 1 (`levin-phase-floor-06j-run1-20261103T091504`, done, exit 0) and
**before any reading**. Nothing in this plan has been launched. Not rewritten afterwards; a change is a new file.
(§7's two context lines were corrected at 09:38Z, before this plan was delivered, against `out/cmd/51`.)

## 1. The W sample (drawn at 09:06Z by `scripts/draw_06j.py`, rules in `predictions_and_rules_20261103T0905Z.md` §4)

- **W-5:** rows 0-12778, 0-12959, 0-1873, 0-2108 and 0-33730, call index 1 (`G+r`), at both rtols: f00–f04
  (1e-7), f05–f09 (1e-8).
- **The rtol equality** (`out/w_prep_v2/summary.json`, `out/cmd/45`): for each of the five rows, the 1e-7 and
  1e-8 captures are equal by 06g's criterion (route, span, signs, each factor's phase spline and leading table by
  bytes) and the integrand (f_sin, f_cos, theta, theta_mod_2pi, theta_deriv at 65 points; production_edges). Only
  the keyword `rtol` differs. 06g's and 06h's results are also identical between the two rtols on every one.
  **So one reference per row is read, on the 1e-7 capture (f00–f04), and serves f05–f09.**
- **W-12:** c00–c11 (`out/sample/w_sample.json`). Every W call is a Route B group call.

## 2. Routes and phase spans (`out/w_prep_v2/`, `out/w_segments.json`)

- Every call's default edges come from production_edges, and every segment's theta_deriv keeps one sign at 17
  points (no flagged stationary point). The tool makes its own 161-point monotonicity test when it reads. A
  reading is usable when `tmap.all_monotone`, or when every non-monotone segment is integrated directly.
- The phase span is the sum over segments of |theta(X_r) − theta(X_l)|, which here equals the net change.

## 3. Cost, projected from phase span

The model is fitted on 06g's 258 Mac Studio Route B readings that finished (`scripts/cost_06j_v2.py`,
`out/cmd/48`, `out/cost_projection_v2.json`). The fit is non-negative least squares,
wall = a × segments + b × phase span:

| reading | a (s/segment) | b (µs/rad) | observed/fitted (min, median, max) |
|---|---|---|---|
| QAWO r1 | 0.00555 | **0.0571** | 0.86, 1.14, 2.62 |
| QAWO r2 | 0.00870 | **0.233** | 0.64, 1.01, 1.93 |
| FCC r1 | 0.0218 | 0 | 0.63, 0.89, 3.56 |
| FCC r2 | 0.0868 | 0 | 0.60, 0.85, 4.05 |

- The fitted b reproduces 06g's QAWO rate (0.057 µs/rad at r1, about 4× at r2). FCC does not grow with span.
- The worst FCC ratios (about 4) are z00/z12, whose stationary-point segments were integrated directly.
- **Projection = fit × the largest observed ratio** (the "envelope"). The fit is shown beside it.
- A first form, `cost_06j.py` (`out/cmd/47`), took the largest single per-radian rate. That rate came from
  readings dominated by per-segment overhead, so it projected 105 h. It is superseded and kept as a record.

Per reading, fit / envelope, in seconds (`out/cmd/50_plan_table.txt`):

| call | unit #index | phase span (rad) | segments / chunks | qawo r1 | qawo r2 | fcc r1 | fcc r2 |
|---|---|---|---|---|---|---|---|
| f00 | g-1e-7-0-12778 #1 | 345.5 | 306 / 20 | 2 / 4 | 3 / 5 | 7 / 24 | 27 / 108 |
| f01 | g-1e-7-0-12959 #1 | 345.5 | 306 / 20 | 2 / 4 | 3 / 5 | 7 / 24 | 27 / 108 |
| f02 | g-1e-7-0-1873 #1 | 345.5 | 306 / 20 | 2 / 4 | 3 / 5 | 7 / 24 | 27 / 108 |
| f03 | g-1e-7-0-2108 #1 | 345.5 | 306 / 20 | 2 / 4 | 3 / 5 | 7 / 24 | 27 / 108 |
| f04 | g-1e-7-0-33730 #1 | 345.5 | 306 / 20 | 2 / 4 | 3 / 5 | 7 / 24 | 27 / 108 |
| c00 | g-1e-7-1-2142 #5 | 1.031e+10 | 1854 / 116 | 598 / 1570 | 2414 / 4663 | 40 / 144 | 161 / 651 |
| c01 | g-1e-7-2-33416 #6 | 2067 | 234 / 15 | 1 / 3 | 2 / 4 | 5 / 18 | 20 / 82 |
| c02 | g-1e-7-1-35497 #5 | 4.514e+07 | 1110 / 70 | 9 / 23 | 20 / 39 | 24 / 86 | 96 / 390 |
| c03 | g-1e-7-0-12727 #4 | 2.209e+05 | 654 / 41 | 4 / 10 | 6 / 11 | 14 / 51 | 57 / 230 |
| c04 | g-1e-7-1-35052 #6 | 1.419e+10 | 2122 / 133 | 822 / 2155 | 3320 / 6413 | 46 / 165 | 184 / 746 |
| c05 | g-1e-7-0-33892 #1 | 345.5 | 306 / 20 | 2 / 4 | 3 / 5 | 7 / 24 | 27 / 108 |
| c06 | g-1e-8-3-34272 #4 | 1.714e+08 | 1376 / 86 | 17 / 46 | 52 / 100 | 30 / 107 | 119 / 483 |
| c07 | g-1e-8-3-37323 #1 | 1.038e+09 | 1640 / 103 | 68 / 179 | 256 / 494 | 36 / 127 | 142 / 576 |
| c08 | g-1e-8-1-35136 #3 | 1.302e+04 | 562 / 36 | 3 / 8 | 5 / 9 | 12 / 44 | 49 / 197 |
| c09 | g-1e-8-1-35446 #4 | 1.271e+07 | 1076 / 68 | 7 / 18 | 12 / 24 | 23 / 83 | 93 / 378 |
| c10 | g-1e-8-0-12965 #4 | 2.776e+07 | 1086 / 68 | 8 / 20 | 16 / 31 | 24 / 84 | 94 / 382 |
| c11 | g-1e-8-0-33648 #1 | 345.5 | 306 / 20 | 2 / 4 | 3 / 5 | 7 / 24 | 27 / 108 |

**68 readings in all.** Run one after another, they total **2.55 h by the fit and 6.06 h by the envelope**.
- The largest single reading is c04 QAWO r2: fit 55 min, envelope 1.8 h.
- No reading projects anywhere near a day.

## 4. How each reading is made

- 06g's `one_reading.py` makes one reading per process: the tool's `reading()` with the command line's exact
  arguments, six workers, the pickle loaded by `_load_pickle(..., "ray.cloudpickle")`.
- It runs with `--progress FILE --progress-order unordered`, 06g's addendum-4 wrapper. The chunks come back by
  `imap_unordered(chunksize=1)` and are put back in order before the tool sums them. 06g checked this bit for bit
  on six re-reads.
- **One marked change:** the progress file also records the job list ((i0, i1) per chunk) and the indices of the
  finished chunks. This changes no value.
- The tree is `dfa19d98`, the tool `b5626ecb`'s, with PYTHONPATH set to the tree then `scripts/`.
- One reading runs at a time, so there are six computing processes plus an idle parent and an idle driver.
  06g's deviation 8 read the six-process limit the same way.

## 5. The backstop

The driver checks every reading's progress file every 30 s.

**The weighted progress.**
- Each Route B chunk has a weight w_c = a × (its segments) + b × (its phase span), with the (impl, res)
  coefficients above. Its segments and span come from `out/w_segments.json`, the same edges and the same 16-segment
  chunks the tool forms.
- Weighting matters here because the phase span is not uniform: on f00–f04 and c01–c03 and c05–c11 the first
  chunk alone holds 15–18 % of the span.
- **done** is the weight of the finished chunks. **In flight** is the weight of the unfinished chunks among the
  first (finished + 6) in index order. `imap_unordered` with `chunksize=1` hands chunks out in index order, so
  these are the ones running.

**The projections.**
- **Central:** P_c = elapsed × W / (done + ½ in flight).
- **Optimistic:** P_o = elapsed × W / (done + in flight), every chunk in flight counted as finished.
- The plain chunk-count forms of both are logged beside them.

**The rules.**
- **Report:** once a reading is past 30 minutes or past its envelope projection, a line goes to
  `out/reading_monitor.log` and the heartbeat, then every 10 minutes. The line gives the reading, elapsed, chunks
  done and in flight, P_c and P_o with their projected finish times, and the count-based forms. The reading goes on.
- **Stop:** only when **P_o ≥ 86 400 s**. That means a projected day or more even if every chunk in flight
  finished now, so the chunks in flight are counted, not only those finished.
  - Before the first chunk returns, P_o uses the in-flight weight alone. A stop then means that even if the first
    six chunks finished at that moment, the reading would take a day.
  - A stopped reading is recorded with its progress file, and the run moves on.
  - A stop is a §7 condition ("a reading's projection is a day or more"). I would report it on resume, not repair
    it.
- **Nothing else stops a reading.** There is no 30-minute cut-off and no multiple-of-projection rule. A reading
  that raises is recorded, and the next one starts.

## 6. Grouping into runs

- **One registered run**, slug `readings` (`--campaign levin-phase-floor --prompt 06j`), launched detached by
  `launch_06j.py readings`, manifest first. The driver is `run_06j.py`, its monitor added for reading commands.
- It holds the 68 readings in ascending order of envelope projection: the W-5 rows and the cheap W-12 calls
  first, c00 and c04's QAWO readings last. Most references are therefore complete early.
- Outputs go to `out/readings/<sid>-<impl>-r<res>.json`, `out/progress/`, `out/reading_monitor.log` and
  `out/cmd/readings/`.
- After it ends, in session:
  - `score_w.py` applies the usable clause per reading and W1–W4 per the rules file §6. Each reference is its FCC
    r2 reading, with all four readings and their spread against min(e_g, e_06h)/100.
  - The f-references are scored against both rtols' f calls.
  - The §3 tables are filled from `out/w_prep_v2/`.

## 7. Already measured, for context (no reading involved)

- **06h's leaf values reconstructed:** solving 06h's returned leaves' spans with the tree's subregion solve and
  adding them in the code's own ascending order reproduces 06h's value bit for bit on all 22 W calls. 06g's
  returned values reproduce its value the same way.
- **The constant move on the `G+r` calls** (f00–f09, c05, c11) is carried entirely (99.7–100 %) by 06h's left
  depth-1 leaf, [23.539, 25.255]. 06h trusted that leaf at r = 0.27–0.67 (its test was r < 1). On f00–f04, 06g
  refines it to 11–13 leaves at depths 3–7, all trusted, with r ≤ 0.234 (`out/cmd/51`).
- On the other W-12 calls one 06h leaf carries 97.6–116.7 % of the move (c06's 117 % is offset by its other cells).
- Which code is right waits for the references.
