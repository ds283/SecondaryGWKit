# 07b Part C — orchestrator's checks at planned stop 2 (2026-11-16, ~21:05–21:10 UTC)

The census agent's k0 run `levin-phase-floor-07b-k0-20261116T204402` is `done`, exit 0, 11/11 (one background
`kill -0` wait on pid 24239, ended 21:00:39Z). The agent reports U0 (1)–(4) holding and a projection of 10.4 h central,
18.3 h with every margin. These are my own checks; the re-runs used the agent's `run_k0.py` (cwd `census/`,
PYTHONPATH the worktree); every comparison is by my `scripts/stop2_compare.py` (07a's stop-2 reader, the "before"
path changed to 07a's census; diff in `stop2_compare.diff`).

1. **Two K0 units re-run by me at the parent** (07a's module by the variant loader): `g-1e-7-1-35492` and
   `g-1e-8-0-12617`, neither among the agent's named units. Both exit 0, Ray never initialised, no foreign module,
   101 own-tree files + the variant (`reruns_p.log`). Against 07a's census checkpoint lines (`ck07a_lines.jsonl`,
   `grep -F` from `461df2ec…`), `p_vs_07a.txt`: every differing path is timing, provenance (worker, pid, tree head,
   scripts' hashes, module paths and counts by root, pickle pointer, `worker_tree`), `cache_built` (only mine) or a
   07b key present only in mine (`num_edge_rule_refusals`, `num_range_rule_refusals`, `rate_rule_07b`,
   `instrument_07b`). Region records differ only in timing and `rate_rule_07b`. **0 value differences. Pass.**
2. **One unit where the edge rule acts, at h, run twice by me** (`g-1e-8-2-34906`, one of the agent's five edge-rule
   units in K0 but not its U0 unit; its call #1 has 1 edge and 1 range refusal, `h_refusals.txt`). The two runs differ
   only in timing, the pid and the pickle hash (`h_det.txt`). **Deterministic. Pass.**
3. **The predictions** (`predictions_check.txt`): `state/predictions.md` SHA-256 `0c6055ac…`, mtime 21:03:31.28Z,
   after the k0 run's end (21:00:33Z heartbeat) and every U0 output; files newer than it are only stop-2 bookkeeping
   (`cmd/25–27`, the scripts diff, `commands.log`). Prompt §6's predictions block (15 non-blank lines), §9.5 (21) and
   §9.7 (4) are each a verbatim substring of the file, 0 lines missing. No census run directory exists.
4. **State** (`state_check.txt`): the 159 protected files, R29 and Part A's 7,645 `measurement/` files unchanged by
   hash; the scratch copy has its five files only; main checkout `28ebaf07`, clean; worktree `--ignored` empty;
   nothing new under `var/` since the marker outside the rehearsal's, census's and review's paths, R31's store and the
   07b r31/k0 run directories.

Not re-derived by me: the projection's arithmetic (`census/out/census_projection.json`), U0 (2)'s 100-unit comparison
and (3) (the agent's outputs); my item 1 samples (2).
