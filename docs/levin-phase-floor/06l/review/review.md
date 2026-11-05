**Verdict: PASS — all seven checks of the note's §B.3 pass; two observations change no verdict (06l: K0 and L3 hold; L1, C1, C2, H1, S1, K1 hold; L2 fails, 63 of 75, so route B does not pass as a candidate remedy by prompt §4's rule).**

# Review of levin-phase-floor 06l on the Mac Studio — the Mac Studio orchestrator, 2026-11-05

The agent ran from the worktree `SecondaryGWKit-lpf06l-af4c4dc`, detached at the dispatch SHA `af4c4dc1`. Its
records are in `var/levin-phase-floor-06l/`. This review's scripts and outputs are in `review/checks/`, its preflight
is in `review/preflight/`, and its marker is `review/dispatch_marker` (2026-11-05T09:06:04Z). The agent's brief is
`review/dispatch.md`, and the record of planned stop 1 is `review/stop1.md`.

## The run

| Run | State | Note |
|---|---|---|
| `levin-phase-floor-06l-k0partL-20261105T091942` | `failed` (exit 2) | Its capture check refused all 46 captures, on the `run_census.py` hash alone. Every capture's values equalled 06g's census (`review/checks/` reread `out/check_captures.json`: 0 region or call keys differing on 46 of 46). Its capture, K0 and draws commands exited 0. This is the log's deviation 11. |
| `levin-phase-floor-06l-partL-20261105T095407` | `done` 48/48 | The check, the slicing and the 46 probes re-run. |
| `levin-phase-floor-06l-census-routeB-20261105T100126` | `done` 18,560/18,560, exit 0, stderr empty | 18,340 s, against a projection of 17,347 s. The backstop reported twice (last P_o 18,033 s) and stopped nothing. Checkpoint `5cb3d3c0e6e628720aa122a1d23d4a90f140b116ef9e58e2f66a3bffb9dd0d39`. |

The orchestrator waited on each run's pid with a background `kill -0` loop. Planned stop 1 was checked before the
census (`review/stop1.md`): the AST check, ν_B read against §0.2, K0, Gate L3 and L2's account by the orchestrator's
own reader, and the census plan. Planned stop 2 (`done`, 18,560/18,560) was checked before scoring. The agent asked
no question. No §7 stop arose.

## The seven checks

1. **Scope — passes.**
   - Under `var/`, nothing outside `levin-phase-floor-06l/` and the 06l runs is newer than the marker, except the
     directory entry `runs/` itself, whose mtime moved when the run directories were created
     (`c1_newer_outside.txt`).
   - **Unchanged by SHA-256:** the scratch copy's five files match the certification, and its listing is unchanged
     (`c1_a3v2_after.txt`, `c1_a3v2_ls_after.txt`). 06g's checkpoint and its 16 pickles: 17/17. Route 1's checkpoint
     and its 16 pickles: 17/17. `v8_calls.json`, `g.pkl` and `route1.pkl`: 3/3 (`c1_*_after.txt`).
   - **The worktree** is clean with `--ignored` at `af4c4dc1`. The main checkout is clean at `28ebaf07`
     (`c1_trees.txt`).
   - **No Ray:** the lock is free; `ray-processes` lists none; `/tmp/ray-sgk` holds the same 97 entries as before. No
     raylet or GCS process is running, and every worker of the census and of this review's re-runs reported
     `ray_initialized False`.
   - **`var/datastores/`:** its listing is unchanged (`c1_datastores_after.txt`).
2. **The variant and K0 — passes.**
   - **The AST comparison** (`c2_ast_compare.txt`, the orchestrator's own script, the same output as at stop 1):
     - identity `d89c959e…` is byte-identical to the tree's module;
     - route B `5aa6178a…` differs only in `_adaptive_levin`: `route_records_06l = []`, `_check_leaf` (its trust
       statement replaced by ν_B and B's record), and the return dict, which gains `route_06l` (every other key's
       value is AST-equal);
     - the signature and decorators are unchanged.
   - **Two K0 units re-run by the orchestrator** under the identity variant and under B (`k0rerun/`,
     `c2_k0_rerun_compare.txt`): `g-1e-7-0-1873` (k `0:202`) and `g-1e-8-2-1626` (k `2:2`), chosen as units on which B
     fired nowhere in K0 (`c2_choose_units.txt`). Under both variants every call record (`elapsed`, `route_06k` and
     `route_06l` aside) and the whole region pickle (timing aside) equal 06g's census: calls 23 and 21, 0 differing,
     0 firings.
3. **The predictions — passes.**
   - `predictions.md` holds prompt §4's L1–K1 block verbatim. It is dated 2026-11-05T09:07:47Z, before the first
     command output (`out/cmd/00`, 09:09:41Z) and before any number.
   - The log carries the same block verbatim as a quotation (`> ` prefixes stripped) under "Predictions and verdicts"
     (`c3_verbatim.txt`).
4. **Part L, re-checked — passes** (`c4_partL_recheck.txt`).
   - **Method:** for six leaves, the seven solves P, L, R, LL, LR, RL and RR were re-made by the orchestrator with the
     tree's `_adaptive_levin_subregion` on the agent's captured call. ν_B was formed in exact arithmetic from the
     halves' own `is_direct`, `phase_err`, `phase_err_less_node` and `node_error_signed`.
   - **The six leaves:**
     - first population: s02 (B trusts, Δ1/ν_B 0.200), s04 (B trusts, 0.99967) and s09 (B does not, 1.00408, one of
       L2's 12);
     - second population: s01 (B trusts, 0.690) and s01 (B does not where route 1 does, 1.0113);
     - the 06h carrying leaf f02 (Levin over Levin, Δ1/ν_B 4.7 × 10⁶).
   - **Result:** on all six, Δ1, r, φ_P, the kinds, ν_B (relative difference 0) and the decisions of 06g, route 1 and
     B equal the agent's table.
   - **B's census at these spans:** it holds no trust record at any of the six (its refinement does not reach them),
     so the comparison with the census record could not be made at leaf level here. Check 5's re-run compares a call
     with the census instead.
5. **Part C, re-checked — passes** (`c5_counts.py` / `.txt`, the orchestrator's own reader of the three raw
   checkpoints, not the agent's extracts).
   - **The 720** are taken from 06a's scan (720 rows; v8 flags agree).
   - **C1 holds:** 749/801 (93.5 %) at 1e-7, 773/830 (93.1 %) at 1e-8.
   - **C2 holds:** 4 calls, all at 1e-8, all `node_limited`.
   - **S1 holds:** 0 calls, 0 units, 0 shard-serial rows.
   - **K1 holds:** 9,839 Levin-parent firings outside the 720, all at Δ1/φ_P < 5.8, the largest 2.854.
     Clenshaw–Curtis-parent firings: 158,073 below 5.8 and 64,869 at or above it (descriptive).
   - **Against route 1:** route 1 converges and B does not, 43 at 1e-7 and 48 at 1e-8 (91). B converges and route 1
     does not, 6 at 1e-8. All of these equal the agent's.
   - **One recovered call re-run by the orchestrator** under B (`recovered_rerun/`, `c5_recovered_compare.txt`):
     `g-1e-8-3-825` #0, V8-lost under 06g (85 regions, unconverged, 8 at `depth_max`).
     - The re-run equals B's census record: value `0x1.32cd3d792ae78p-97`, abserr `0x1.1f25b6709afb5p-119`, 61
       regions, converged, 15 firings, not `node_limited`. The whole call record (but `elapsed`) is equal as JSON.
     - Its move against 06g is 2.2 × 10⁻⁶ of e_B + e_g.
6. **The references — passes** (`c6_refs.txt`).
   - **Method:** for five calls (three as the note asks, and two more because z01 and c01 are moved by no route),
     the ratios |v − ref|/e for 06g, route 1 and B were recomputed exactly from the raw checkpoints and the
     references' own sources (06g's `scores.json` `rows` `ref_30`; 06j's `w_calls.csv`).
   - **Results:**
     - v03: 1.2027 / 1.1854 / 1.1818;
     - z01: 0.3203 for all three;
     - c01: 0.001382 for all three;
     - r17: 0.5458 / 0.5455 / 0.5454;
     - z15: 3.95e-6 / 5.56e-6 / 5.54e-6.
   - All equal the agent's `h1_calls.csv` to 1e-12, and H1 holds on each.
7. **The records — passes.**
   - **The log** has every §5.1 heading:
     - the subject, commit ("records committed by the orchestrator") and result;
     - What shipped;
     - Predictions and verdicts;
     - Deviations from the prompt;
     - Verification performed;
     - Observations not acted on;
     - State handed on.
   - **Deviations:** 21, each classified once (2 `STRUCTURALLY REQUIRED`, 13 `IMPLEMENTATION CHOICE`, 6
     `UNINTENDED DRIFT`; `c7_classification.txt`).
   - **`out/results.md` proposes no code.** It says so in its header and its last section, and ends "Which route, if
     any, becomes code is for the user."
   - **Provenance:** 65 of 77 outputs under `out/` (`cmd/` aside) carry a header (`c7_provenance.txt`). The per-route
     tables `out/partC/*.json` carry theirs, so 06k's deviation 30 is not repeated. For the 12 without one, see
     observation 1.
   - **Rows** are counted shard-serial with units apart (S1: "0 calls, in 0 units and 0 rows (shard-serial)").
   - **`KEPT_ON_MAC_STUDIO.txt`:** 338 entries, all matching by SHA-256 and size. Every pickle and every file over
     2 MB under `var/levin-phase-floor-06l/` (`review/` aside) is listed (`c7_kept_check.txt`).

## Observations (change no verdict)

1. **Twelve harness outputs carry no provenance header of their own.**
   - The files: `out/capture.json`, `check_captures.json`, `check_captures_b.json`, `inputs.json`, `inputs_b.json`,
     `k0partL_results.json`, `partL_results.json`, `sample/selection.json`, `commands.log` and the three
     `variant_*_diff.txt`.
   - They are the same kinds of file that 06k wrote without headers, which 06k's review passed. The diffs' `---`/`+++`
     lines name both paths and times.
   - The log does not list them as a deviation. For the laptop: accept them as they are, or record them as a
     deviation when committing.
2. **The agent's final report miscounts its deviations** by class (3 / 12 / 6 against the log's 2 / 13 / 6). The log
   is the record, and each of its items is classified once.

## For the user (the orchestrator's reading, not a ruling)

- **L2 fails**, so by prompt §4's rule as written route B does not pass as a candidate remedy. Every safety rule
  holds.
- **§0.2's guarantee is first-order only.** On the 12 untrusted first-population leaves, E_P and E_L + E_R have
  opposite signs, and Δ1 exceeds |E_P − (E_L + E_R)| by 0.2–0.6 %. The rest of ν_B is 0.1–0.3 % of it, so the leaves
  miss by 0.04–0.41 %.
- **Route 1 trusts the same 63 of 75.**
- **The laptop's estimate split the 85 wrongly.** It gave B 85 of the 142 slow leaves as all 75 of the first
  population and 10 of the second; the measured split is 63 and 22.
- **What B gives up against route 1:** about 5 points of C1, and 72/76 group calls left at `depth_max` against 14/16.
- **What A would keep:** at most 77 of route 1's 91 extra calls (by inference from route 1's census, not measured).
