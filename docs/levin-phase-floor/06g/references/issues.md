# Issues from 06g Part B §5, the references (for the board's §3/§4 and `docs/OPEN_ISSUES.md`, applied by the orchestrator)

Proposed names; the orchestrator chooses. Each points at its records under `06g/references/`.

## 1. `[06g-filon-qawo-zerodivision-at-stationary-point]` — closed by `ff3e8394`

- **What:** `tools/filon_reference.py` at `1c5336e`, Route B QAWO: `ZeroDivisionError` in `gfun`'s
  Newton step (`yn = … y - fval / dpy`, line 1063) at dp/dy = 0, in a segment whose fitted phase has a
  stationary point (`monotone` False). Every reading through such a segment, FCC too, inverted a
  two-valued map: unsound by the tool's own rule.
- **Where:** w00–w11 and z00, z12: exactly one non-monotone segment each (the orchestrator's
  `review/checks/refs_monotone.txt`).
- **Fix:** `ff3e8394` integrates such a segment directly in y and refuses the inversion. The rerun
  batch's 56 readings of the 14 samples ran under it; 6 re-reads were bit-identical.
- **Status:** closed for QAWO (FCC, see 2).

## 2. `[06g-filon-fcc-unresolved-near-stationary-point]` — closed by `b5626ecb`

- **What:** at `ff3e8394`, FCC's readings of the 14 samples, though usable by clause (ii), disagreed with
  QAWO: on V1's 12, FCC r1 was 117 to 11,900 of R24's abserr from the other readings, and FCC r2 up to
  329 (w07) on w07–w11 (`out/scores_after_stage2/report_20261102T204420Z.md`).
- **Fix:** `b5626ecb` integrates directly any segment whose slope ratio exceeds 2. The laptop's batch
  (32 readings): the four readings of each of the 14 now agree within 3.1e-6 to 3.5e-4 of the
  tolerance (`out/scores_addendum6/report_20261103T020815Z.md`).
- **Status:** closed.

## 3. `[06g-refs-part2-backstop-b-stops-on-low-projections]`

- **What:** part 2's projections scaled Route B by segment count, not phase span, so large-span QAWO
  readings were projected 30–100× too low; backstop (b), max(6 × projection, projection + 1 h), then
  stopped them at about 3,905 s, against the ruling's "only at a day or more". Six readings were stopped
  by (b) and two by (a) on ordered progress (`out/part2_results.md`; the orchestrator's
  `review/refs_part2_r52_report.md`).
- **Impact:** the rerun batch reran five of them (r24 r1 and r53 r1 done; r52 r2 and r24 r2 skipped on
  the user's word; three stopped again, see 4).
- **Status:** a harness defect of this campaign's scripts (log, deviation 32); for any later backstop,
  estimate Route B by phase span.

## 4. `[06g-rerun-backstop-projection-ignores-chunks-in-flight]`

- **What:** `run_rerun.py` projected `elapsed × total / done`; at done = 1 that ignores five chunks in
  flight, counts setup and treats chunks as equal. All three stops fired at the first chunk back: r47
  qawo r2 (projected 453,016 s), r53 qawo r2 (290,165 s), r26 qawo r1 (2,296,531 s).
- **Measured** (`out/stage3_projection.json`): a six-worker schedule with chunk cost proportional to phase
  span, calibrated on the first chunk, reproduces the finished r24 and r53 r1 readings' monitor counts at
  every check, and projects 10,826 s, 31,889 s and 49,162 s (3.0, 8.9 and 13.7 h): none a day.
- **Impact:** no verdict depends on them (the stage-3 recheck: none changes).
- **Next step:** for the user, whether to rerun the three; any later backstop should project by phase
  span, or not before a few chunks per worker are back.

## 5. `[06g-z12-levin-1681x-abserr-while-converged]`

- **What:** Z3's sample 12, `g-1e-8-1-2174` #1 (G+q+r, rtol 1e-8): the census reports it converged with
  abserr `0x1.fdbb1bc1a92b8p-110`, and its value `0x1.0c6c2cc40e857p-86` is **1,681.35** abserrs from the
  reference `0x1.0c5f18f86f178p-86` (06h's value: 1,678.07 of its own). The only V3 violation, as ruled.
- **Why the reference is right:** two implementations, four readings within 2.6e-4 of the tolerance; the
  captured integrand equals z00's (`g-1e-7-1-2174` #1, the same call at rtol 1e-7) bit for bit by the
  06i-13 criterion (`out/compare_z00_z12.json`), and z00's Levin value lies 0.0183 of its abserr from the
  same reference; z00's readings are bit-identical across hosts.
- **Context:** the row is one of 06a's 720 stationary-point rows (07's); Z3 flagged this sample at 2,295×
  at selection (06c's depth20-tiling, class c). At 1e-8 the driver stops converged at depth 18 with 135
  regions; at 1e-7 it is unconverged at depth 20 with 249.
- **Next step:** for the board; 07's stationary-point work.

## 6. `[06g-v1-r23-dishonest-r24-honest]` — V1's 12 group calls

- **What:** on all 12 of V1's failing rows (sub-interval 1, G+q+r), against references whose four
  readings agree (addendum 6): **R24 (06g's code) is nearer and honest on all 12** (|v24 − ref|/e24 at
  most 0.942: w07 0.942, w01 0.817, w03 0.573, the rest at most 0.39); **R23 (the parent, 06h's code) is
  dishonest on all 12**, 4.5e3 to 6.0e5 of its abserr away (`out/scores_addendum6/report_20261103T020815Z.md`).
- **Impact:** V1's 12 violations are 06h's code's error, which 06g removes.
- **Status:** a finding for V1's entry on the board.

## 7. `[06g-refs-part2-nine-qawo-error-readings]`

- **What:** part 2's nine QAWO readings that ended in the `ZeroDivisionError` of 1: w03 r2, w04 r1 and
  r2, w08 r1, w10 r2, z00 r1 and r2, z12 r1 and r2 (exit 1, no value).
- **Status:** not rerun at `1c5336e` (addendum 4); superseded by the rerun batch's readings of those
  samples at `ff3e8394`. Their logs stay in `out/reading_logs/` and the part-2 checkpoint.

## 8. Other findings

- **V8's sample** (`out/scores_stage3/report_20261103T065158Z.md`): every call is unconverged under g and
  converged under 06h. g's value is within its abserr on 11 of 12 (v03 at 1.20×); 06h's on 9 (v03 2.31×,
  v08 4.29×, v11 1.07×, while reported converged).
- **06i's Route A FCC offsets** (samples 9, 22, 25, 27): the port's FCC readings differ from 06i's by a
  constant at both resolutions, 2.8e-6 to 4.0e-5 of m, within the agreement rule; not investigated.
- **The `ff3e839` worktree's `__pycache__/`** (71 `.pyc`, mtime 2026-11-02T23:12:03–05Z): written by a
  process not established (log, observation 3); not deleted. At the same second
  `scripts/__pycache__/harness.cpython-313.pyc` appeared in this records directory: the process imported
  this directory's `harness.py` (as a capture's unpickling does) with the `ff3e839` tree on its path.
