# 06h Part B, the phase agent: the plan for stage 2 (written in stage 1, 2026-11-01, before any computation)

Every command from `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06h/phase`, with
`PY=/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python` and `PYTHONDONTWRITEBYTECODE=1`, each
recorded by `bash scripts/rec.sh NN_name ...` into `out/cmd/` (and `out/commands.log`).

## Before the launch
1. `touch state/marker_stage2`; trees, registry, lock-free/no-Ray check (read only) -> `10_state_before`.
2. Hash the scratch copy: `$PY -B scripts/hash_store.py "06h phase before" .../levin-phase-floor-06/wider/state/a3v2_copy`
   -> `11_copy_files_before`; `bash scripts/compare_hashes.sh .../copy_files_certified.txt out/cmd/11_copy_files_before.txt`
   -> `12_copy_vs_certified` (must print EQUAL).
3. Write the predictions: prompt 06h section 5's H1 and H2, verbatim (lines 336-341 of the prompt at c1ec3b2),
   into `state/predictions_<UTC>.md` and into `log.md`'s Predictions section. Not edited afterwards.
4. The census inputs (read only): `ls` and `status.json` of `var/runs/levin-phase-floor-06h-census-*` (must be
   exactly one, `done`); `shasum -a 256` of its `checkpoint.jsonl` -> `13_census06h_checkpoint`.
5. `$PY -B scripts/select_06h.py` -> `20_select` (writes `out/sample/selection_{p,h}.json`; checks 06f's selection
   against its committed copy, 06e's checkpoint against a0dc4d7f, and 06e's records against 06f's selection; stops
   on any call missing or disagreeing).
6. Scripts at launch: `diff -ruN` against `state/scripts_06f_reference/` -> `21_scripts_diff_against_06f_at_launch`
   (copied to `out/scripts_diff_against_06f_at_launch.txt`); `shasum -a 256 scripts/*` -> `out/scripts_sha256_at_launch.txt`.
7. `runregistrykit list` -> `22_registry_list_prelaunch`.
8. Launch, detached and registered (slug `phase`): `$PY -B scripts/launch_06h.py > state/launch_phase.txt 2>&1`.
9. One check: the run's `status.json`/heartbeat and `kill -0` of the recorded pid -> `23_launch_check`. End the turn.

## The run (run_06h.py; at most six child processes)
capture p (x1_capture.py --tree p, 06e's census record) -> capture h (--tree h, 06h's census record) -> STOP if any
of the 40 captures does not match -> method check (method_check_06f.py on h's capture of h-1e-7-0-12689 #19, at
the exact e^u; STOP if it fails) -> 40 replays (replay06f.py --m2, each tree's own) -> 40 cross scorings
(cross06h.py, p at h's points and h at p's) -> the 50-digit reference check (orch_m2.py worst h 5).

## After the run (on resume)
10. Hash the scratch copy again -> `40_copy_files_after`, `41_copy_after_vs_certified` (EQUAL); 06h's census
    checkpoint hashed again (unchanged); `find var/datastores -newer state/marker_stage2` empty; both worktrees
    clean, `--ignored` empty; no `__pycache__` under `phase/`; registry list.
11. `$PY -B scripts/analyse06h.py` -> `43_analyse` (H1, H2, the parent's columns, the split by dominant term;
    `out/phase_calls.csv`, `out/phase_leaves.csv`, `out/scores.json`).
12. Records: `log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt` (every pickle under
    `state/capture_{p,h}/` and every file over 2 MB, with SHA-256 and size).
