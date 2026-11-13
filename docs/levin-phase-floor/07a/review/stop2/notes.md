# 07a Part C — orchestrator's checks at planned stop 2 (2026-11-12, ~13:18–13:25 UTC)

Census agent dispatched ~12:37Z, after the rehearsal agent finished and the lock was released
(`../lock_after_rehearsal.txt`: free, no Ray process). Its k0 run `levin-phase-floor-07a-k0-20261112T130135`
is `done`. The agent reports that W0 (1)–(4) hold and asks two questions (Q1, Q2), which are relayed verbatim.

My checks. The re-runs were made with the agent's `run_k0.py` (cwd `census/`, PYTHONPATH the worktree). Every
comparison is by my own `scripts/stop2_compare.py`, which flattens every leaf (floats by `float.hex`) and groups the
differing paths. A region record is read at its offset only if its bytes match `pickle_record_sha256`.

1. **Two K0 units re-run by me at the parent** (the parent's module installed by the harness's variant loader in h).
   `g-1e-8-3-34739` is in the 720 and `g-1e-7-2-21980` is outside it; neither is one of the agent's examples. Both
   exit 0 (`reruns.log`). Against 07's census checkpoint lines (`ck07_lines.jsonl`, taken with `grep -F` from
   `03dfe786…`), in `p_vs_07.txt`:
   - every differing path is timing, provenance (unit and tree names, worker, pid, tree_head, scripts' hashes,
     module file paths and counts by root, the pickle pointer, `worker_tree`), `cache_built` (06k's known K0
     effect, present only in mine), or a 07a key (`route_07a*`, `instrument_07a`);
   - every 07a key is present only in my run (`p_vs_07_newkeys_check.txt`: 494 groups, 0 otherwise);
   - in the region records, only timing and `unit` differ.

   **0 value differences. Pass.**
2. **One windowed unit at h run twice by me** (`g-1e-7-3-34969`, four windowed calls at h in the agent's K0; not
   the agent's W0 unit). The two runs differ only in timing, the pid and the pickle hash that follows from them
   (`h_det.txt`). **Deterministic. Pass.** My run also equals the agent's K0 h record of the same unit, apart from
   timing, provenance and `cache_built` (`h_vs_agent_k0.txt`).
3. **The predictions** (`predictions_check.txt`): `state/predictions.md` has SHA-256 `3ec7b53d…` and mtime
   13:16:44.807Z. That is after every W0 output (latest 13:16:21Z) and the k0 run's end (13:16:06Z). Prompt §6's
   lines 525–556 (31 non-blank), §9.5's lines 716–731 (15) and §10.4's lines 823–825 (3) are all in it verbatim.
   No census run directory exists.
4. **State** (`state_check.txt`): the protected files (13,596) are unchanged; the main checkout is at `28ebaf07`
   and clean; the worktree is empty under `--ignored`; 75 runs, all finished. Nothing is newer than the marker
   outside §C.1's paths and `review/`.

**The two open readings (Q1, Q2)** are the user's. They are relayed verbatim, with no recommendation.
