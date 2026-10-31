# levin-phase-floor 06e §5 — the census (Mac Studio)

(Preparation section drafted 2026-10-30 ~08:00Z, while R19/R20 held the cores, before any
computation; to be completed after Z0, the run and the references.)

## Before the launch — preparation (no computation)

1. **Harness copied** (`out/cmd/00`): 06c's 56 census scripts from the code worktree's
   `docs/levin-phase-floor/06c/census/scripts/` (equal, `diff -r`, to
   `var/levin-phase-floor-06c/census/scripts/`) into `scripts/`, and unchanged into
   `state/scripts_06c_reference/` (`state/scripts_06c_reference.sha256`).
2. **Rows** (`01`): 06c's `state/rows.json` copied; sha256 `aa523d28…`, equal.
3. **Launch text** (`02`): a dry import of `launch.py` (PURPOSE, SCOPE, trees, 9,280 rows; Python
   3.13.16). No Ray, no store, no unit.
4. **Diff at preparation:** `out/scripts_diff_against_06c_at_prep.txt` (9 files changed or new;
   `references/orch_ref.py` byte-identical to 06c's, sha256 `8a0d291d…`).
5. **Plans:** `state/launch_plan.txt` (Z0 to the launch, every command); `state/drafts/
   predictions_partB_draft.md` (Part B's readings, Z3's sample and matching rule; finalised into
   `state/predictions.md` after Z0).

## Deviations from the prompt (numbered and classified) — so far

1. `prov.py`: trees `p` = `…-lpf06e-50277f3`, `g` = `…-lpf06e-6d81fad`; 06c's run, records and
   checkpoint SHA-256; depth20-tiling's directory; `WIDER` = `var/levin-phase-floor-06e/census`.
   STRUCTURALLY REQUIRED.
2. `harness.py`: `job_name="levin-phase-floor-06e"`. IMPLEMENTATION CHOICE (stored, never read).
3. `worker.py`, tree evidence: 06e's `_chebyshev_node_displacements` and `_cc_node_term_parts`,
   06d's `_cc_node_error_bound`, and `levin_quadrature.py`'s SHA-256. STRUCTURALLY REQUIRED.
4. `worker.py`, the Clenshaw–Curtis floor's terms (`_cc_floor_terms`): each returned fallback leaf
   re-solved after the timed section with pass-throughs at `_cc_node_term_parts`,
   `_cc_roundoff_floor`, `_declared_endpoint_phase_err`; the floor's first two terms read from the
   code's own function on zeroed inputs; each leaf checked to reproduce bit for bit. IMPLEMENTATION
   CHOICE (the result does not return the terms; after the timed section so the budget is 06c's).
5. `worker.py`, deep lines past depth 12 (`deep_lines_06e`, with per-level tiling at N and 2N − 1,
   `_line_tiling`); 06c's `deep_lines` unchanged (depth_max only). IMPLEMENTATION CHOICE.
6. Nothing in the timed section changes and no 06c key changes; every addition is a new key.
   IMPLEMENTATION CHOICE (Z0's clause 2 compares on 06c's keys).
7. `run_census.py`: docstring only. STRUCTURALLY REQUIRED (the trees move through `prov.py`).
8. `launch.py`: prompt `06e`, purpose, scope, imports. STRUCTURALLY REQUIRED.
9. New `z0.py` (06c's `x0.py`): P units against 06c's `g-` units on 06c's keys, deep lines
   included; determinism unit 06c's X0 unit `g-1e-7-0-14164`; no capture. IMPLEMENTATION CHOICE.
10. New `z0_clause1.py`; `scripts_equal.py` (state/Z0) and `check_predictions.py` (06e, lines
    345–375) adapted. IMPLEMENTATION CHOICE.
11. Z3: `references/orch_ref.py` unchanged; run with `GIT_OPTIONAL_LOCKS=0` (its `git status`
    would otherwise refresh the worktree's index: a write in a tree), `PYTHONDONTWRITEBYTECODE=1`
    and one BLAS/OpenMP thread per process. STRUCTURALLY REQUIRED (the no-write rule) for the
    first; IMPLEMENTATION CHOICE (six cores) for the rest.
12. Z3: the eight short references in one detached sequential script (`run_short.sh`), never two
    at once. IMPLEMENTATION CHOICE (the brief allows it if recorded) — pending the orchestrator.
13. Z3: three resolutions = S96(P128), S96(P64), S128(P64); reference S128(P64); "agree" = range
    ≤ min(abserr)/100, exact; the orchestrator's reading reported beside. IMPLEMENTATION CHOICE.
14. Z3: a capture stands for g's call only if g's driver replay on it reproduces g's census call bit
    for bit (`replay_g.py`). IMPLEMENTATION CHOICE.
15. Scripts not `black`-formatted (records). IMPLEMENTATION CHOICE.
16. 06b's W3 evidence and 06a's scan kept in each 1e-7 unit, as 06c. IMPLEMENTATION CHOICE.
17. `out/cmd/00` written by hand (rec.sh did not exist before the copy); its `# cwd` is the shell's
    then directory, the orchestrator's `review/`, where nothing was written. UNINTENDED DRIFT.

## Step 2 (2026-10-30, 08:07–08:13Z): checks, Z0, predictions, launch

- **State** (`10`, `29`): all three trees clean at `28ebaf0`, `50277f3` and `6d81fad`; 33 runs, all finished; the lock is free (`11`, `30`); `ray-processes` lists none (`12`, `31`).
- **The copy** (`13`/`14`, `23`/`24`, `32`/`33`): EQUAL to its certification before Z0, after Z0 and the smoke test, and at the launch, with no journal file beside it. Fingerprint `0e32f24a…`, no problems, at g (`15`) and at p (`16`).
- **06c's checkpoint** (`17`, `28`): `b61cb034…` before Z0 and after preparation.
- **The job name** (`18`): datastorekit 0.5.0 stores `job_name` and reads it nowhere.
- **Gate Z0 holds, all three clauses.**
  - **Clause 1** (`21`): the diff `e5764a6..50277f3` touches CLAUDE.md only. The harness imported 102 modules from p.
  - **Clause 2** (`20`): P1 `p-1e-7-2-38302`, P2 `p-1e-7-2-2095` and P3 `p-1e-8-0-1971` are identical to 06c's `g-` units in every value 06c recorded, on 06c's keys. These units have no deep lines, in 06c's run or here.
  - **Clause 3** (`20`): `g-1e-7-0-14164` is identical in G1 and G2 on every key. Its 6 fallback leaves reproduce bit for bit.
  - No worker initialised Ray.
- **Development checks.**
  - The tiling check at g and at p (`21b`): every level tiles at g; at p some do not.
  - The smoke test (`22`, `22b`): 2 rows × 2 rtols, 4 units done; 64 of 64 fallback leaves reproduced; no call deeper than 12.
- **Predictions:** `state/predictions.md`, sha256 `f27771cc…`, mtime 08:12:47Z. Part A is byte-equal to the prompt (`26`, `35`). Every script is equal to what Z0 and the smoke test ran (`27`, `34`).
- **Launch** (`36`), with **one check** (`37`): `levin-phase-floor-06e-census-20261030T081318`, sh 3892, driver 3896; state `running`; six 1e-7 workers started; stderr empty; no checkpoint line yet at 08:13:23Z.

18. **`out/cmd/11`** piped rec.sh's output through `sed` in the same shell. The lock's process listing therefore showed that `sed` (pid 2919), the shell's own child. `12` and the pre-launch `30` and `31` use `>>` only, and show nothing. UNINTENDED DRIFT.
19. **`out/cmd/25`** failed: a non-ASCII bytes literal in my writer meant `state/predictions.md` was never written, so the check found no file. It was rewritten correctly and checked as `26`. UNINTENDED DRIFT.
20. **`out/cmd/22b`** (the smoke summary) is an inline read of the smoke ledger, not run through rec.sh; its command is noted at its head. A new `scripts/dev_line_tiling.py` exercises `_line_tiling`, because Z0 had no call deeper than 12. IMPLEMENTATION CHOICE.
21. **Over budget: more than twelve units is a stop, asked before the pass.** This is my reading, not the prompt's. Its basis is 06c's Part B, which fixed the same threshold for 06a's pass. IMPLEMENTATION CHOICE. Deviation 12 is confirmed by the coordinator as within the brief, and stays recorded.

## Step 3 (2026-10-30, 10:00–10:02Z): after the run; the references begin

- **Status** (`38`): the run is `done`, exit 0, with 18,560 of 18,560 units recorded. The driver's heartbeat was 09:59:15Z and it exited at 10:00:05Z (reported by the coordinator). Wall time: launch 08:13:18Z to exit, about 1 h 47 min, against 06c's 2 h 30 min. `stderr.log` is empty. All 16 workers' `bye` messages report `ray_initialized` false and no foreign modules. The run directory is 1.6 GB, 149 files.
- **Records** (`39`, `checks_after_06e.py`):
  - every record is at `6d81fad` with empty status, carries `run_census.py` `a6d9bac4…`, and has scripts equal to `scripts/` now;
  - no unit is in error or over budget, and **no unit's wall_s exceeds 300 s**, so no over-budget pass is needed;
  - 278,863 of 278,863 fallback leaves' floor terms reproduce;
  - 200 calls go deeper than 12, every line reproducing, none with an error. 29 calls reach depth_max.
- **The store** (`40`–`44`): the copy is EQUAL to its certification after the run and after the fingerprint, with no journal file. Fingerprint `0e32f24a…`, no problems. **06c's checkpoint** (`45`): `b61cb034…`.
- **Z3 inputs** (`46`): the 13 captures, sliced and SHA-checked against `captured.json`, are equal to the orchestrator's copies where it has them (1, 2, 8, 9, 13, 22, 25).
- **The replay** (`48`): on all 13 samples, g's driver replayed on the captured input gives g's census value, abserr, regions, depth and converged, bit for bit, with label and span equal. Every capture stands for its g call.
- **Short references launched** (`49`; one check `50`): `run_short.sh 02 08 10 13 14 22 23 27`, sh pid 50499, bash 50504. s02 at P=128 began 10:02:21Z (746,786 sub-panels).

22. **`out/cmd/47` failed.** Unpickling the first capture raised `ModuleNotFoundError: harness`: the group calls' pickles name `harness._rebuild_lru_cache`, from 06c's census harness. I re-ran it as `48` with the census `scripts/` after the tree on PYTHONPATH. That is the precedent of depth20-tiling's own commands (`PYTHONPATH=<tree g>:<scripts>`). Our `harness.py` differs from 06c's only in its docstring and `job_name`. `run_short.sh` and `run_refs.py` were changed the same way before their launch, and `orch_ref.py` is unchanged. UNINTENDED DRIFT, corrected; `out/references_scripts_sha256_at_short_launch.txt` records the SHA-256s.
23. **New `scripts/checks_after_06e.py`** (the after-run checks of every record). IMPLEMENTATION CHOICE.

## The short references (10:02–19:25Z) and the registered run's launch (19:27Z)

- **Short batch** (`51`): sh 50499 exited after `short_progress.txt` recorded "19:25:40Z all done, exit 0". All 16 passes (s02, s08, s10, s13, s14, s22, s23, s27, each at P=128 and P=64) exited 0. Walls per pass ran from 305 s (s13 P128) to 6,095 s (s10 P64); the coordinator notes this is about 2–3.5× depth20-tiling's estimates. Each output's replay line equals g's census call (value and abserr), and each says "equals captured False", as expected at g, since the capture holds 06c's values. Scoring comes later.
- **Samples 3 and 15** (`52`; one check `53`): registered run `levin-phase-floor-06e-references-20261030T192730`, sh 87997, driver 88002. s03 at P=128 began 19:27:31Z with 2,548,199 sub-panels (3.4× s02's). The references scripts are unchanged since the short launch (`out/references_scripts_sha256_at_refs_launch.txt`).
