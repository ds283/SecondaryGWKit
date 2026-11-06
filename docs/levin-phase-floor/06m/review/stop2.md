# 06m Part B — planned stop 2, the orchestrator's checks (Mac Studio, 2026-11-06, ~10:17–10:22 UTC)

The census agent stopped at planned stop 2 after Gate M0 (`levin-phase-floor-06m-k0-20261106T101118`,
`done`, exit 0), with no census launched. It reported M0 holding: 100 of 100 units, 2,110 calls and
2,110 region records equal to route B's census.

**The orchestrator's checks** (scripts in `review/scripts/`, outputs in `review/stop2/`):

1. **Two K0 units re-run by the orchestrator** at $m$, chosen as units where route B *fired*
   (`choose_units.txt`, from route B's `route_06l` records), so that the new yardstick's path is
   exercised on both parent kinds:
   - `g-1e-7-3-34738` (k `3:1`): 257 firings in route B, 239 on Clenshaw–Curtis parents, 18 on Levin
     parents;
   - `g-1e-8-2-34906` (k `2:302`): 81 firings, 57 on Clenshaw–Curtis parents, 24 on Levin parents.

   Run with the agent's documented command (`run_k0.py tree units.json rerun --workers 2`, PYTHONPATH
   the code worktree): both `done` in 78 s, `ray_initialized False`, no foreign module, 102 own-tree
   modules (`rerun_run.txt`). The code worktree stayed clean (`status --short --ignored` empty).
2. **Compared with route B's census by the orchestrator's own comparator**,
   `gate_m_orchestrator.py` (`9f5d1c20…`), written independently of the agent's `compare_06m.py`:
   every key of the unit record and the whole region-pickle record (record SHA-256 checked), floats
   by `float.hex`, arrays by dtype/shape/bytes, lists positionally; excluded and counted apart:
   timing (`elapsed`, `compute_time`, `seconds`, `*_s`), the instruments `route_06k`/`route_06l`,
   and provenance (tree, worker, pickle pointers, `modules`, scripts, `cache_built`).
   **2 of 2 units, 42 calls, 2 region records: 0 differing** (`rerun_compare.txt`). Every unit names
   the code worktree's `levin_quadrature.py` and no variant.
3. **The agent's whole K0 output re-compared** by the same comparator
   (`k0_agent_output_compare.txt`): **100 of 100 units, 2,110 calls, 100 region records, 0 differing**,
   all `done`, all with the tree's module. The excluded differences are only timing, the instrument
   (`route_06l`: present at route B, `None` at $m$) and provenance. **Gate M0 holds, confirmed.**

The agent's eight deviations are classified (`STRUCTURALLY REQUIRED` for the harness's tree, depth and
index changes and the new comparison; `IMPLEMENTATION CHOICE` otherwise, two `UNINTENDED DRIFT`,
cosmetic or caught before launch). Deviation 7, the exclusion set being compare_k0.py's whole set
(provenance and `cache_built` too), is accepted: those keys are not call or region records, and the
orchestrator's comparator counts them apart.

**The census plan** is accepted as reported: `launch_census_06m.py 17640` (projection 17,640 s from
K0's ratio 0.962 to route B's walls), 300 s per unit, six workers, run_census.py's backstop
(stops only at a projected day, counting units in flight). Load at 10:21: 6.1 / 7.0 / 9.6.
