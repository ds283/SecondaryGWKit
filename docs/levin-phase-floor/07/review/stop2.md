# 07 Part B — planned stop 2, the orchestrator's checks (Mac Studio, 2026-11-08 ~14:15–14:30 UTC)

Census agent dispatched 13:34:50Z (marker `dispatch_marker_census`; brief `dispatch_census.md`
e62c4acb…, with `preflight/B6_B7_verbatim.md` d44f3b2c… and `dispatch_common.md` 27bf93a6…;
protected 181/181 OK immediately before). The k0 run `levin-phase-floor-07-k0-20261108T135240`:
done, exit 0, 7/7, 13:52:40–14:05:41Z. The agent reports V0 (a), (b), (c) all hold and asks one
question (V4 over eight k, not six), relayed verbatim.

My checks (outputs in `stop2/`, script `scripts/stop2_compare.py`, which reads 06m's checkpoint
line at the agent's index offset but accepts it only if its SHA-256 matches and it names the unit,
and 06m's region record only if its bytes match the line's own `pickle_record_sha256`):

1. **Two K0 units re-run by me at the parent** (`g-1e-8-0-2106`, in the 720, and `g-1e-7-1-14039`,
   outside; neither the agent's examples), by the agent's `run_k0.py --tree p`, exit 0
   (`reruns.log`), compared with 06m's census by my reader (`p_vs_06m.txt`,
   `p_vs_06m_newkeys_check.txt`): every differing leaf of the record and the region record is a
   07 addition absent from 06m's record (none present in 06m's), timing, provenance (unit/tree
   names, worker, pid, tree_head, pickle pointer, module lists, and `worker_tree`, absent from the
   new records), or `cache_built` (06k's known K0 correction). **0 unclassified value
   differences. Pass.**
2. **One stationary-point unit at $g$ run twice by me** (`g-1e-7-0-33997`, whose group call 0
   locates 6 minima; not the agent's V0(c) unit), exit 0 each (`g_det.txt`): the two runs differ
   only in timing (incl. `scan_07…seconds`) and the provenance that follows from it (pid, pickle
   SHA). **Deterministic. Pass.** My run also equals the agent's K0 record of the same unit at $g$,
   timing and provenance aside.
3. **Predictions** (`predictions_and_crosscheck.txt`): `state/predictions.md` SHA-256
   `92e22911…`, mtime 14:10:10Z, after V0's outputs (13:58:32Z, 14:05:40Z); prompt §6 lines 765–803
   and §10 lines 1017–1018 are in it verbatim. No census run directory exists.
4. **State:** protected 181/181 OK; all three worktrees `status --short --ignored` empty; main
   checkout clean; nothing running or stale; nothing newer than the census marker outside
   `census/`, `review/` and the k0 run directory.
