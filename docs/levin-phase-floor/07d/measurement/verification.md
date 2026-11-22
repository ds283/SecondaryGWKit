# levin-phase-floor 07d, Part A — the measuring agent's part of VERIFICATION.md §7d

Mac Studio (mps026298), Python 3.13.16. The worktree is `SecondaryGWKit-lpf07d-49ce964`, at the dispatch SHA `49ce9642`,
whose code is 07c's (`67b5af74`; `AdaptiveLevin/levin_quadrature.py` `d141a80b…`; `tools/filon_reference.py`
`d37e633b…`). Every command is in `out/cmd/` and `out/commands.log`, under the record number given here. These
measurements were taken on this tree. A later re-run adds a subsection and leaves this one as written.

## 7d.A.1 The tree and the gate

- **The worktree was unchanged.** `git --no-optional-locks status --short --ignored` gave no lines at every check, the
  last at about 13:25Z on 22 Nov. The main checkout stayed at `28ebaf07`, with an empty status.
- **The gate held before and after.** Record 01 (09:06Z) and record 55 (end) are identical once start and end times,
  load, disk and the registry's run count are set aside. They cover:
  - the trees and the code diff `67b5af74..HEAD`, which is empty in `AdaptiveLevin ComputeTargets LiouvilleGreen tools`;
  - the module's SHA-256 and lines;
  - `SIX_PI`'s code readers by `ast`, only line 3055, the gate;
  - `CHEBYSHEV_ORDER` 24;
  - 07c's census checkpoint `33d94ce2…`;
  - the A3 v2 scratch copy's five files, each MATCH.
- **No Ray started and the rehearsal lock was never taken.** No store was opened but the scratch copy, read only, for
  captures.

## 7d.A.2 Section 2.1: captures and replays at the tree

- **Captures** each equal their census record (value, abserr, regions, depth, converged, label):
  - `-capture-calls-` 306/306;
  - `-capture-r1x-ra-` 1,559/1,559 for the analytic top-up with ride-along, and 30/30 for (Ra).
- **Replays at the tree:** 4,220 of 4,220 reproduce their census record in hex, R31's record against its call-level
  record included (deviation 4). They match on value, abserr, regions, depth, converged, both refusal counts, the chase's
  three counts, the region arrays and the guard.
- **The stop instrument** recorded every stopped leaf: 16,047, none unrecorded.

## 7d.A.3 Sections 2.2–2.4: the leaf sample

- **Strata:** 44, holding 2,873 leaves measured. Every stratum reaches 40 scored leaves except `chase|analytic|1e-8`,
  exhausted at 8. Each stratum is filled in its seeded order (`state/selection_leaves_07d.json` `c4c01873…`). Seeds were
  written at 09:06:35Z (`state/seeds_07d.json`), before any draw.
- **The leaf reference** is v3 (`state/refs_rule_leaf_07d.txt` `6fcce794…`, 09:58:15Z), written before its
  re-measurement. Transitions from v2: 0 leaves between bounded and understates.
- **The candidates** were fixed at 10:28:45Z (`state/candidates_07d.json` `d73b208e…`), after D-N and before D-E scored
  anything.
- **07c's 176 truncation-dominated leaves:** every Clenshaw–Curtis value is reproduced against 07c's in hex
  (`out/trunc176/`).

## 7d.A.4 Section 2.5: the variants

- **Files:** each variant's diff is `out/variants/<v>.diff`, and its SHA-256 is in `out/variants/variants.json`.
- **Cross-check (record 56; deviation 19):** on Ea's first four Clenshaw–Curtis leaves of `R:c-1e-7-0-13757-5`, the
  variant's abserr and phase_err equal leafcc's nest3 + 4s and floor + 4s within 9.0e-4 relative. Eg, ET20 and ET12 share
  Ea's estimate block, but this check was not run on them.
- **Smoke replays (record 30):** under Ea, `d:h-1e-7-0-12577|g4` reproduces its census record bit for bit, although 34 of
  its Clenshaw–Curtis solves were re-declared: the larger declarations changed no decision in that call.
- **The replay run** (`-variants-`, records 31, 40): every variant wrote 4,220 of 4,220 with rc 0. Every call it leaves
  untouched reproduces 07c's census bit for bit: Eg 972/972, Ea 813/813, ET20 407/407, ET12 533/533, Ep 813/813 and
  T20 720/720.

## 7d.A.5 Section 2.6: the references

- **Rules written before any reading:**
  - The rule is `state/refs_rule_calls_07d.txt` (`e6cd65f5…`, 10:37:01Z).
  - The (Rt) selection is `state/selection_rt_07d.json` (`c7bd5c64…`, 11:49Z), drawn at the RT seed 707404.
  - The (Ra) selection is `state/selection_ra_07d.json`.
- **Projection** (record 48; `out/refs/refs_projection_07d.json` `e6176893…`): 1.59 h, sent to the orchestrator before any
  reading.
- **The run** (`-refs-`, record 52; plan `state/plan_refs.json` `3e37173a…`): 175 of 175 exit 0, in 50 min. Every
  reading records `ray_initialized` false, false, the tool's SHA-256 `d37e633b…` and its pickle's SHA-256.
- **Smoke readings** (records 50, 51) equal the run's readings of the same integrands in every value field.
- **Scoring** (record 53): 425 referenced calls, 417 scored. (Ra) at 07c's code: 60 of 60 scored, converged and honest.

## 7d.A.6 Not verified here

- Table 2's cause rule (`causes_07d.py`) is written as it was first applied (deviation 18); it was not fixed before the
  data were read.
- Route A's projection model is this Part's own (deviation 15) and has no fit against earlier readings. The run took
  0.52 of its elapsed projection.
