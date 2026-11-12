# levin-phase-floor 07a Part A — verification (the measuring agent's part of VERIFICATION.md §7a)

As of stop 1, 2026-11-10, Mac Studio (mps026298); worktree `SecondaryGWKit-lpf07a-f93aafd` at `f93aafd0` (code 07's);
venv the main checkout's (Python 3.13.16). Every figure below is in `out/` with the SHA-256 of the script that wrote
it.

## 7a.1 The Gate (Mac Studio clauses)

- `git --no-optional-locks diff --stat adffa21b f93aafd0 -- AdaptiveLevin ComputeTargets LiouvilleGreen tools`: empty.
  `AdaptiveLevin/levin_quadrature.py` SHA-256 `4e5da371...` (07's g module), 5,719 lines.
- `ea9bf326` and `b5405eb0` are on `origin/handover-remedial` (`git branch -r --contains`).
- `runregistrykit list` (Mac Studio): 69 runs, all finished, none running or stale, before launch.
- `out/gate_hashes_before.json` and `out/gate_hashes_after_interim_1.json` (`scripts/gate_hashes.py`): 78 of 78 lines
  of 07's `KEPT_ON_MAC_STUDIO.txt` under `state/v4/capture/` and `state/xcheck/capture/` match; the scratch copy's
  five files match `copy_files_certified.txt` (SHA-256, size, mtime), no extra file. 07's census checkpoint
  `03dfe786...` before and after.

## 7a.2 Section 2.1 item 2's captures

`capture_pairs_07a.py` (a copy of 07's `xcheck_capture_07.py`, merging a unit's calls) re-ran `g-1e-7-3-14501`,
`g-1e-8-3-14501`, `g-1e-7-3-35006` and `g-1e-8-3-35006` on the scratch copy (07's `worker.py`, Ray not initialised):
9 of 9 calls (the two pairs at both rtols, five no-window calls) equal 07's census by 07's `check_captures.py`
(value, abserr, label, converged, every region array, the call record, the harness). `out/pairs_capture/`.

## 7a.3 Replays at 07's code

430 of 430 calls (V4's 25, the four pair calls, the five no-window calls, the 396 at 1e-8) at the tree's own module
and at variant (a), each in its own process: value and abserr by hex, converged, num_regions, max_depth, every region
array of 07's `_region_arrays_06g` and `_region_arrays_07`, and section 1.5's result entries, all equal to 07's census
record (`out/tables_interim_1.md`). The 390 of the 396 captured at 1e-7 were replayed with the 1e-8 call's `rtol` and
`atol` and reproduce the 1e-8 record.

## 7a.4 The no-window check (section 2.3)

Five calls (n0-n4), each variant against the tree, bit for bit (value, abserr, converged, regions, depth, region
arrays' digest, windows stopped): (a), (c), (d), (e), (f) 5 of 5. (b2), (g2): running at stop 1.

## 7a.5 The attribution's closure

v22 (pilot), v21, v03: |sum_i ref_i - ref| / tol = 1.5e-4, 1.8e-3, 1.0e-4; holds on all three
(`out/attr/attr_tables_interim_1.md`).

## 7a.6 Nothing written where it must not be

The worktree's `git status --short --ignored` empty at every check; the main checkout at `28ebaf07`, `status --short`
empty; R29's store and 07's run directories not opened; `var/levin-phase-floor-07/` read only (its captures hashed
before and after).

## At the section 2.5 stop (2026-11-11), added

Measured after the user's rulings on stop 1 (log deviations 16, 17, 20, 21). The subsections above stay as they were
taken.

### 7a.1 The Gate, after
`out/gate_hashes_after_final.json` (`scripts/gate_hashes.py`):
- 78 of 78 of 07's capture lines match;
- the scratch copy's five files match their certification, with no extra file;
- 07's census checkpoint is `03dfe786f69343a0...`, hashed read-only;
- `runregistrykit list`: 07a's four runs are all `done`, and nothing is running or stale.

### 7a.4 The no-window check, (b2) and (g2)
(b2) and (g2) pass, 5 of 5 each, bit for bit against the tree. Every variant now passes.

### 7a.5 The attribution's closure, all 24 calls
`out/attr/attr_tables_final.md`: |sum_i ref_i - ref| / tol runs from 6.2e-6 to 3.3e-3. It holds on every call, so
none is reported unattributed.

### 7a.6 Nothing written where it must not be, after
- The worktree's `git status --short --ignored` is empty.
- The main checkout is at `28ebaf07`, and its `status --short` is empty.
- R29's store was not opened. 07's census run directory was read only, for its checkpoint's hash.

### 7a.7 The O9 check (added at the user's word)
For v08, v09, v12, v17 and v22, every leaf, parent and sibling was reached by `break_point` descent from its seed
region at exactly its recorded depth. Each region's `phase_span` was recomputed by 07's own `build_Levin_data`, on 07's
module (sha256 `4e5da371...`), with no variant (`out/o9/`).
