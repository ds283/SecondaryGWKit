# levin-phase-floor 06h §5, the census: working draft (stage 1, preparation)

Written 2026-11-01, while R21/R22 hold the six cores. Nothing computed: no store or checkpoint hash,
no fingerprint, no unit, no Gate H. Becomes `log.md` (README §5.1's form) at the end.

## Done at preparation (out/cmd/00–04)

- `00`: 06e's census harness copied from the code worktree's `docs/levin-phase-floor/06e/census/scripts/`
  (62 files) into `scripts/` and, unchanged, into `state/scripts_06e_reference/`
  (`state/scripts_06e_reference.sha256`). The source is identical to the run's own copy in the main
  checkout's `var/levin-phase-floor-06e/census/scripts/` (`diff -r`).
- `01`: `state/rows.json` copied from 06e's census state, `aa523d28…` both.
- `03`: `out/scripts_diff_against_06e_at_prep.txt` (`diff -ruN state/scripts_06e_reference scripts`) and
  `out/scripts_diff_new_files_against_06e_sources_at_prep.txt` (the three new scripts against the 06e
  scripts they were copied from); every script's SHA-256.
- `04`: `ast.parse` of the ten scripts changed or added.
- `state/drafts/predictions_partB_draft.md`, `state/launch_plan.txt`.
- The code worktree stayed clean (`status --short --ignored` empty, no `__pycache__`).

## Where each per-call field lives (planned)

Checkpoint `data.calls` (cosmological group calls) and `data.other_calls` (analytic-block calls); per call:

| Need | Field(s) |
|---|---|
| H4, matching | `index`, `notify_label`, `x_span` (hex), `converged` |
| H4 / descriptive, the call | `value`, `abserr` (repr + hex), `num_regions`, `num_subregion_solves`, `max_depth`, `depth_hist_levin_cc` (per depth: [Levin, CC] leaves), `rtol`, `atol` |
| Declared term, CC branch (06e and h; 06f's method) | `cc_floor.sums_floor_attributed_cc_leaves.declared`, `cc_floor.sums_all_cc_leaves.declared`, `cc_floor.n_floor_attributed` |
| Declared term, by branch, h only (new) | `declared_06h`: `request`, `has_parts`, `n_leaves`, `n_reproduced`, `n_errors`; per branch `levin`, `levin_floor_attributed`, `cc`, `cc_floor_attributed`: `n`, `n_reproduced`, sums of `declared`, `declared_constituents`, `declared_abscissa`, `declared_phi_eval`, `floor`, `phase_err_resolve`, max endpoint `theta_abserr` |

Per unit (checkpoint): H3's `result.total`, `result.total_abserr`, `result.total_converged`; `status`,
`wall_s`; `tree_head`, `tree_status_short`, `scripts_sha256`; `modules` (counts by root, foreign
modules, the tree's files, all_in_tree); the registry's `run_census.py` stamp on each record.

Per leaf (worker pickle `regions_<worker>.pkl`, at `pickle_offset`/`pickle_length`, SHA-checked):
`regions`/`other_regions` arrays (start, end, depth, type 0 Levin / 1 CC, abserr, phase_err, …,
06e's); `cc_floor_terms` (06e's); new `declared_terms_06h`: per call supplying theta_abserr, per leaf
`branch`, `floor_attributed`, `declared`, `floor`, `phase_err_resolve`, `theta_abserr_a/_b`,
`p_l1_a/_b`, the three parts at each end and the declared term split by part, `reproduces`,
`captured`, `errors`.

## Deviations so far (numbered, classified)

1. **`prov.py`:** trees `p` = `…-lpf06h-2c325f5` and `g` = `…-lpf06h-c1ec3b2`; 06e's run, records,
   rows and checkpoint SHA-256 (`RUN06E`, `REC06E`, `ROWS06E`, `CHECKPOINT06E_SHA256`, named as 06f's
   prov.py names them); `WIDER` = `var/levin-phase-floor-06h/census`. STRUCTURALLY REQUIRED.
2. **The tree h keeps the scripts' key `g`,** so units are `g-<rtol>-<shard>-<serial>` as in 06e's census
   and run_census.py's and gate_h.py's code is unchanged in that respect. IMPLEMENTATION CHOICE.
3. **`harness.py`:** `job_name="levin-phase-floor-06h"` (the brief), stored and never read.
   STRUCTURALLY REQUIRED.
4. **`run_census.py`:** docstring only. STRUCTURALLY REQUIRED.
5. **`launch.py`:** prompt `06h`, purpose, scope, imports from prov.py. STRUCTURALLY REQUIRED.
6. **`worker.py`, tree evidence:** `LiouvilleGreen.three_bessel_integrals`, `LiouvilleGreen.bessel_phase`
   and the file QSI's `BesselPhaseGroup.theta_abserr` comes from are among the files that must be in
   the tree; whether the group has `_declared_error_parts`, the module `_two_product_error` and
   `EXP_RELERR`; both modules' SHA-256. STRUCTURALLY REQUIRED (the brief: evidence of where the modules
   were loaded from).
7. **`worker.py`, `_declared_terms_06h`:** after the timed section, for every call whose phase supplies
   `theta_abserr`, every returned leaf (both branches) re-solved once with pass-throughs at
   `_declared_endpoint_phase_err`, `_roundoff_floor` and `_cc_roundoff_floor`, each leaf checked to
   reproduce bit for bit; per call `declared_06h`, per leaf `declared_terms_06h`. The harness as copied
   records the declared term only on fallback leaves (`cc_floor`), so the Levin branch the brief and
   prompt §0.2 ask about could not be read. Nothing in the timed section changes, and no 06e key
   changes; Gate H's reproduction at the parent, on 06e's keys, is the check that it changes no
   computed value. IMPLEMENTATION CHOICE in form (allowed by the brief).
8. **`gate_h.py`** (from 06e's `z0.py`): the before is 06e's census; 06e's `deep_lines_06e`,
   `cc_floor_terms` and per-call `cc_floor` are compared in the reproduction too, every pickle key 06e
   recorded is checked to be compared; the determinism unit is `g-1e-7-0-14164` (06e's Z0 unit; two
   analytic blocks, 16 calls each handed a group's `theta_abserr`), with a qualification test that
   fails "deterministic" if no analytic-block call carries a reproduced, non-zero declared term.
   IMPLEMENTATION CHOICE.
9. **`gate_h_clause1.py`** (from `z0_clause1.py`: commits `6d81fad` and `2c325f5`), `scripts_equal.py`
   (`state/H`), `check_predictions.py` (06h's prompt, lines 334–347), `checks_after_06h.py` (06h's
   declared-term counts, over-budget and error units). IMPLEMENTATION CHOICE.
10. **`out/cmd/00` was recorded by an inline equivalent of rec.sh** (rec.sh does not exist in the target
    before the copy), with its cwd the census directory. IMPLEMENTATION CHOICE.
11. **A stray `cp -p out/cmd/00_harness_copied.txt /dev/null`** in the same shell as `00` failed
    ("preserving times for '/dev/null': Permission denied"); it wrote nothing anywhere. UNINTENDED DRIFT.
12. **The Levin branch has no 06e column.** 06e's checkpoint does not separate a Levin leaf's declared
    term from its eq. (151) floor; the descriptive item reports the CC branch under both and the Levin
    branch under h only, with h's constituents part as a labelled counterfactual. STRUCTURALLY REQUIRED.
13. **The launcher's dry import (06e's `02`) is deferred to stage 2,** with the store and checkpoint
    hashes, under the six-core rule. IMPLEMENTATION CHOICE.
14. **Scripts are not `black`-formatted;** they are records. IMPLEMENTATION CHOICE.

## Stage 2 (2026-11-01 00:41–00:51Z): checks, Gate H, predictions, launch

- 10–18: trees clean at 28ebaf0 / 2c325f5 / c1ec3b2; 38 runs, all finished; lock free, no Ray process;
  copy EQUAL to its certification, no journal; fingerprint 0e32f24a… at h and p, no problems; 06e's
  checkpoint a0dc4d7f…. 05: the orchestrator's two facts checked against the harness.
- 19–21 (first Gate H, state/H): holds on all parts. 19b–21b (state/H2, on the final worker): holds
  on all parts; 06e's oracle counterfactual reproduces 06e's analytic_rad bit for bit on the G unit.
- 22/22b smoke (4 units, done, all declared leaves reproduced, counterfactual present); 23/24 copy EQUAL;
  25–27 predictions (Part A byte-equal, sha256 6ff4f05b…, mtime 00:49:57Z), scripts equal to H2's and
  the smoke's, 06e's checkpoint a0dc4d7f…; 28–34 pre-launch all good.
- 35 launch: run levin-phase-floor-06h-census-20261101T005037, sh 73326, driver 73329; 36 one check.

## Deviations added at stage 2

15. **The oracle's abserr and 06e's oracle as a counterfactual** (`worker.py`, `analytic_06h`): the
    orchestrator's fact 1 asks H3 on the analytic oracle's value and abserr; 06e's census recorded
    `analytic_rad` but not its abserr. A pass-through at `QSI_MOD.analytic_integral` in the timed section
    keeps its arguments; after the timed section, at g, `analytic_integral` is re-run with the parent's
    `BesselPhaseGroup` (source from git at 2c325f5, SHA-256 862a18fe… checked, exec'd into an unregistered
    module). Counted as 06e's only where its value equals 06e's analytic_rad bit for bit. Gate H re-run
    (state/H2) on this worker. IMPLEMENTATION CHOICE (the orchestrator's request; the method mine).
16. **Gate H was run twice** (state/H on the worker before deviation 15, state/H2 on the final one); both
    hold; H2 certifies the run. `scripts_equal.py` points at state/H2. `out/worker_diff_gate_h_1_to_final.txt`
    is the change between. IMPLEMENTATION CHOICE.
17. **H3's second reading and H4's split by kind** are in Part B of `state/predictions.md` (the
    orchestrator's request); a different verdict between H3's two readings is a stop. IMPLEMENTATION CHOICE.
18. **`out/cmd/11` piped rec.sh's output through `sed`,** so the lock's listing shows that sed (as 06e's
    deviation 18). 29 and 30 do not. UNINTENDED DRIFT.
19. **launch.py's purpose does not mention the oracle counterfactual** (launch.py was not edited after
    Gate H, to keep the scripts equal to those Gate H ran); the worker's docstring and this log carry it.
    IMPLEMENTATION CHOICE.
20. **The run's wall time includes every post-timed pass** (06e's and 06h's new ones); the unit-wall
    comparison is on `wall_s`, which excludes them, and the post-timed seconds are reported beside the
    run's wall. STRUCTURALLY REQUIRED by the instrumentation.
