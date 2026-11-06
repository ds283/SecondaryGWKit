# 06m Part B — planned stop 3, the orchestrator's checks (Mac Studio, 2026-11-06, from ~15:30 UTC)

**The census** `levin-phase-floor-06m-census-20261106T102153` (driver pid 62309 under sh 62304),
waited on by one background `kill -0` loop; the pid had exited by 15:30:19.

- `status.json`: `state done`, `exit_code 0`, `units_done 18560` of `units_total 18560`, heartbeat
  `2026-11-06T15:29:38+00:00`.
- `checkpoint.jsonl`: 18,560 lines, 18,560 unique units, 0 duplicates, status tally `{done: 18560}`:
  **0 in error, 0 over budget**. `stdout.log` ends "18560 of 18560 units recorded; units in error 0;
  problems []"; all 16 workers exit 0; `stderr.log` empty.
- `monitor.log`: past the projection (17,640 s) twice, 15:15:55 (18,466 recorded, 1 in flight,
  projected finish 15:17) and 15:25:55 (18,538, 1 in flight, 15:26); far below the one-day backstop.
- Wall, manifest `created` 10:21:53 to last heartbeat 15:29:38: **18,465 s** (route B 18,340 s; 06g
  20,361 s; projection 17,640 s).
- Load at 15:30: 37.5 / 19.2 / 11.6 after the census had exited, from ChronoSync, fileproviderd,
  CategoriesService and WindowServer (`ps -r`), not from this work.

**Stop 3 cleared**; the census agent was resumed to score Gate M first and alone. The orchestrator
runs its own comparator, `gate_m_orchestrator.py` (`9f5d1c20…`), over every unit
(`stop3/gate_m_compare.txt`, `.json`).

**Gate M, the orchestrator's re-score** (`stop3/gate_m_compare.txt`, exit 0): **18,560 of 18,560
units, 289,482 calls (74,650 group, 214,832 analytic), 18,560 region records: 0 differing; VERDICT
ALL EQUAL.** All units `done`, none missing in route B, every unit with the tree's
`levin_quadrature.py` and no variant. Excluded and counted apart: the instrument `route_06l`
(present at route B only); timing (`elapsed`, `*_s`, `seconds`, `compute_time`); provenance
(`modules` paths, `tree_head`, `worker_pid`, pickle pointers and record hashes, `scripts_sha256`'s
`prov.py`, and `cache_built[].mode_serial` in 14 units, which records which units shared a worker's
producer cache).

**The flag-cost table, the orchestrator's reader** `flag_cost_orchestrator.py` (`f0e7ce48…`,
`stop3/flag_cost.txt`), every unconverged call; the 720 by 06a's rule (a 1e-7 record in 06a's census
checkpoint with any scan entry `n_sign_changes > 0`), asserted 720; request max(atol, rtol·|v|) in
exact rational arithmetic; share = abserr / the unit's `result.total_abserr`:

| census | rows | unconverged group calls | units | within request | node_limited | max abserr/request | max share |
|---|---|---|---|---|---|---|---|
| m | outside 720 | 114 (1e-7 52, 1e-8 62) | 84 | 104 | 10 | 1.048 | 4.735 % |
| m | in 720 | 311 (144, 167) | 285 | 309 | 2 | 1.004 | 52.99 % |
| 06g | outside 720 | 1,632 (801, 831) | 946 | 1,628 | 4 | 1.071 | 34.22 % |
| 06g | in 720 | 679 (325, 354) | 461 | 677 | 2 | 1.000 | 52.99 % |

No analytic call is unconverged in either census. Route B's census gives m's table line for line
(as Gate M requires). m's line outside the 720 equals the count taken from 06l's extract (114 calls
in 84 units, 104 within, at most 1.05× over, share at most 4.7 %).

**The census agent's report agrees** with both re-scores: Gate M 18,560 units, 289,482 calls,
289,482 region records, 0 differing (its `out/gate_m/compare_m.json`); `cache_built` differing by
order in 7 units (the orchestrator's 14 `mode_serial` positions); the flag-cost table identical
in every figure; m's checkpoint `ea5bc3ab6b618d562b9dee0a7ed4bdba5c1999e8348b2ad33bdadb6135e63c2e`
(the orchestrator's own `shasum`, `hashes_after.log`). Its 15 deviations are classified in
`census/log.md` (4 STRUCTURALLY REQUIRED script groups, 3 UNINTENDED DRIFT — a broken line caught
by `ast.parse`, three log times, `descriptive_06m.py` changed after a first run with the first output
kept in `state/superseded/` and everything recomputed — the rest IMPLEMENTATION CHOICE); accepted.
`issues.md` opens none and leaves `[06m-calls-flagged-unconverged-within-their-request]` to the user.

## The end (~16:00 UTC)

- `sh scripts/hash_inputs.sh after`: every list in `hashes_after/` is byte-identical to
  `hashes_before/` (sha256, size and mtime of every file): the scratch copy (5 files), R23/R24 stores
  (10) and run directories (10), route B's census (150), 06g's census (149); 0 journal files.
- Registry: 62 runs, all finished. Lock: free, `lock_end.txt`. Main checkout at `28ebaf07`, clean;
  both code worktrees clean with `--ignored`.
- Nothing is left running. Nothing was committed on this host.
