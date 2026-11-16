# levin-phase-floor 07b Part A — verification (the measuring agent's part of VERIFICATION.md §7b)

As of the section 2.6 stop, 2026-11-14, on the Mac Studio (mps026298). The worktree is
`SecondaryGWKit-lpf07b-436ed5d`, at `436ed5d4` (code 07a's). The venv is the main checkout's (Python 3.13.16). Every
figure below is in `out/`, with the SHA-256 of the script that wrote it. Every command is in `out/cmd/`.

## 7b.1 The gate (Mac Studio clauses)

- **The code** (`scripts/gate_code_07b.py`; `out/gate/gate_code_before.json` `3ac346a2…`, `gate_code_after.json`
  `729e3fb2…`, both `holds`):
  - `git --no-optional-locks diff 1ec734d7 436ed5d4` touches only `89c368f2`'s four files, each AST-equal with
    docstrings stripped;
  - `AdaptiveLevin/levin_quadrature.py` is `4ea1948b…`;
  - 07a's census checkpoint (`var/runs/levin-phase-floor-07a-census-20261112T132744/checkpoint.jsonl`) is `461df2ec…`;
  - the trees are clean.
- **The hashes** (`scripts/gate_hashes.py`, 07a's with `TREE` changed; `out/gate/gate_hashes_before.json` `40b3092f…`,
  `gate_hashes_after.json` `235aa5c5…`):
  - 78 of 78 of 07's kept capture lines match;
  - the scratch copy's five files match their certification, with no extra file.
- **The registry** (`runregistrykit list`): no run was running or stale before launch. After: 80 runs, all finished, the
  three `07b` runs `done`.

## 7b.2 Section 2.1: captures and replays at 07a's code

- **Captures** (`scripts/x1_capture_07b.py`, from 07a's `x1_capture_07a.py`; `out/captures/captures_h.json`
  `1f160136…`): 46 calls on 45 units from A3 v2's scratch copy (Ray not initialised), 46 of 46 equal to 07a's census by
  x1's own check. This covers item 4's four, R2 and R3; the pairs come from 07a Part A's captures, R1 from 07's
  cross-check captures and V4 from 07's inputs.
- **Replays** (`scripts/replay_07b.py`, `run_replays_07b.py`; `out/replays/tree/`, `tree_summary_all.json`): 187 of 187
  calls equal 07a's census record bit for bit.
  - Compared: value and abserr by hex, converged, regions, depth, region arrays and section 1.5's entries.
  - Every capture also reproduces at the other rtol's (rtol, atol) where the census has both (deviation 8).
  - `i3-34599-07` (07's capture) reproduces 07's census at 07's module (`out/replays/07/`). At 07a's code it returns
    07a's census value (`out/replays/not_a_test/`).
- **06b's W2** (`scripts/w2_rescore_07b.py`; `out/w2/w2_rescore.json` `efdd8914…`): 24 of 24 scored against S₁₀.
  Dishonest at 07a's code: 4 (2.20, 1.36, 6.01, 1.39), all converged. At 06b's code: 17.

## 7b.3 Section 2.2: probes, leaf references, closure

- **Probes** (`scripts/leafprobe_07b.py`): every probed leaf's V_P, V_L, V_R and quarters re-solved by
  `_adaptive_levin_subregion` equal the replay's leaf by hex. This covers the eighths where solved, and the
  perturbations.
- **Leaf references** (`scripts/leafref_07b.py`, Route B FCC r1 and r2 on split edges, deviation 14): the pilot on v08
  agrees with the production-edge reading to at most 3.4×10⁻⁴³ per piece, against e′ 1.7×10⁻³⁶.
- **Closure** (`scripts/tables_07b.py`, `out/tables/tables_07b.md` § Closure):
  - item 3's and item 4's summed leaf references against the call's reference: 5.6×10⁻⁸, 1.72×10⁻⁴, 4.0×10⁻⁶,
    9.4×10⁻⁵ and 2.5×10⁻⁵ of abserr;
  - V4's leaf references against 07a Part A's on the same span: 49 leaves, at most 0.0039 e′.
- **M3 recomputed** (`scripts/m3_theta_07b.py`; `out/m3/theta_prime_check.json` `caf310da…`): on 15 leaves, θ′
  against the captured phase's own derivative.
  - Integrated per segment, the mismatch is at most 4.0×10⁻⁹. The δ* that would account for each carrier's error is
    1.5×10⁻⁸ to 7.9×10⁻³: ratio at most 2.3×10⁻³ on the carriers.

## 7b.4 Section 2.4: references

- **Seeds and selection:** `state/seeds_07b.json` (`6d99b42e…`) was written before any draw, and
  `state/selection_07b.json` (`4b94c181…`) saved before any reading.
- **The readings:** 200 of 200 exit 0 (`out/refs_results.json`; run `levin-phase-floor-07b-refs-20261113T160925`).
- **Agreement:** every reference's readings agree to at most min(abserr)/100, but R2-27 (spread 0.107 of abserr) and
  R3a-0-33666-a13 (0.014). Those two are reported and not scored.

## 7b.5 Section 2.3: variants

- **The variants** are built by `scripts/make_variants_07b.py` by exact replacement, each with its diff in
  `out/variants/` and its SHA-256 in `out/variants/variants.json`. Each is loaded by 07's `variant_loader.py` before any
  import.
- **The no-window check:** n0–n4 are bit-identical to the tree under (b4), (b8), (c), (d) and (e), 5 of 5. Under (f)
  only n0 is, 1 of 5, by design: (f) is not tied to windows.

## 7b.6 Nothing written where it must not be

- The worktree's `git --no-optional-locks status --short --ignored` was empty at every check, the last on
  2026-11-14 after the runs.
- The main checkout is at `28ebaf07`, and git was never written.
- A3 v2's scratch copy was read only, and hashed before and after.
- At most six worker processes, but for two stretches with a seventh (deviations 9 and 15).
- No Ray, and no rehearsal lock.

## Extension (§2.7)

As of §2.7.4's stop, 2026-11-15, on the Mac Studio (mps026298); the worktree at `436ed5d4` throughout; the prompt read at
`a510e4ad`. Every figure is in `out/ext/`, with the SHA-256 of its script; every command in `out/ext/cmd/`.

### 7b.7 Before, and the projection

- **Before:** `runregistrykit list`: 80 runs, all finished; `~/.raypoolkit/` held no lock; the worktree's
  `git --no-optional-locks status --short --ignored` empty; the main checkout at `28ebaf07`.
- **The projection** (`scripts/ext/leafproj_ext.py`; `out/ext/x1/leafproj_x1.json` `d5da9581…`): 07a's refitted FCC
  model with its margin, on the reading's own segments after the 1.8 split, every segment classified by 07's 33-sample
  rule: 0 stationary, 0 steep, γ D_f = 0; 0.27 h of job CPU. Calibrated on five of Part A's split readings (actual job
  CPU 5.2–7.6× the projection): 2.1 h of job CPU, 0.6 h wall. Reported to the orchestrator before the run. Actual: 16 min.

### 7b.8 The leaf references (X1)

- **The run:** `levin-phase-floor-07b-ext-leafrefs-20261115T225140`, 6/6 exit 0, 0 direct segments in any reading
  (`out/ext/cmd/ext-leafrefs/`).
- **Probes:** every probed leaf's V_P, halves, quarters and the trust check's δ₁, δ₂, r reproduce the (e) replay's leaf
  by hex (Part A's `leafprobe_07b.py` for the pairs; `scripts/ext/leafprobe_ext.py` for the other-rtol W2 calls).
- **Closure** (`out/ext/x1/x1_table.md`): on the seven no-window calls the leaf references' sum against the call's
  reference is at most 1.8×10⁻⁴ of abserr (reference accuracy at most 3.1×10⁻⁵ of abserr). On the three pair calls the
  window leaves' error is by subtraction and at most 1.5×10⁻⁷ of abserr. v09 is Part A's carrier reading only.

### 7b.9 The variants

- **Built** by `scripts/ext/make_variants_ext.py` by exact replacement (each old text once), with diffs and SHA-256 in
  `out/ext/variants/variants.json` (`045681c4…`); each loaded by Part A's `variant_loader.py` before any import, its
  `_VARIANT_07B` checked.
- **Replays:** g1, g2, h, g2h, i and i2 on the 187 calls, 186 + item 3, each with its instrument
  (`out/ext/replays/<v>/`); fdiag on (f)'s 61 unconverged calls, equal to Part A's (f) on all 61 (regions_sha256,
  value, abserr and every leaf record).
- **The no-window check** (`out/ext/g/g_windows.md`): n0–n4 bit-identical to (a) under g1 and g2, 5 of 5; under h,
  g2h, i and i2 n0 only (not tied to windows).
- **Scores** (`out/ext/tables/scores_ext.json` `508e946a…`) use Part A's references and rule through Part A's
  `score_07b.py` (imported, unchanged); the extended Table 4 reproduces Part A's qualification, unconverged, regions
  and wall for (a)–(f).
- **Family (R)** (`out/ext/family_r/family_r.md` `49a41f4a…`): 40 calls, all honest and converged.

### 7b.10 Nothing written where it must not be

- At the end the worktree's status was empty and the main checkout at `28ebaf07`; git was never written.
- `find` under `measurement/` for files newer than the extension's start, outside `out/ext/`, `scripts/ext/` and
  `state/ext/`: none before the three records were appended.
- A3 v2's scratch copy was not opened (no new capture was needed). No Ray, no lock.
- At most six worker processes: the leafrefs run (three readings of two workers) ran alone. The replays ran with 5
  workers. X2's two single processes and the probes (at most 6) ran when nothing else did.
