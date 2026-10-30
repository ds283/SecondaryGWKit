# depth20-tiling: log (Mac Studio, mps026298, 2026-10-29)

Brief: `dispatch.md`. Tree g: `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6` at
`e5764a6`, detached, clean. Every command and its output: `out/NN_*.txt`, also `out/commands.log`.

## Step 0: the sample

- `out/00_inventory.txt`: the 262 depth-20 calls' identities, tolerances, blocks and unit wall
  times (`state/inventory.json`). Nothing else was read before the rule was written.
- **`sampling_rule.md` fixed at 2026-10-29T21:54Z (file mtime; the draw, out/01, ran at 21:54:25Z), SHA-256
  `d47a35db9b11330a1b4e7000a9af06ab77fe25442f5a5bc5d367b4e1b2772edc`.**
- Before the rule was written, one entry of `deep_lines_06c.json` (`g-1e-7-0-12616` call 3, its
  first and last two levels) was printed to learn the file's format; that call was not drawn, and
  nothing else was read. The orchestrator's check9 totals were known from the brief.
- `out/01_draw_sample.txt`: 30 calls on 30 units (`state/sample.json`), every cell had an
  eligible call (unit `wall_s` ≤ 120 s); fill: 1e-8 J1, Y4, Y1, J4; 1e-7 J1, J3.

## Step 1: the cheap pass (`out/02_cheap_pass.txt`, `state/cheap_pass.json`)

All 262 spans were reconstructed from the checkpoint's `x_span` (no capture needed): the
reconstruction reproduces every recorded leaf bit for bit (262/262), and every level's (start, end)
equals the census pickle's full deep line (262/262), which also gave each half's branch.

## Step 2: captures

- `out/03`–`05`: the scratch copy before the first capture: EQUAL to
  `copy_files_certified.txt`, no journal file; fingerprint `0e32f24a…`, problems none.
- `out/06_capture.txt`: **failed at once**, every unit `ModuleNotFoundError: No module named
  'rerun'` — I had not copied the census's `rerun.py`, which `rebuild.py` imports. No call was
  computed (each worker stopped in `rebuild_row` before any Levin call). Kept as
  `state/capture_attempt1_failed/` and `state/captured_attempt1_failed.json`.
- `scripts/rerun.py` copied (SHA-256 `6efcfa11…`, the checkpoint's); re-run as `out/07_capture.txt`.
- `out/07_capture.txt`: 30 of 30 captures equal the census call bit for bit (value, abserr by hex,
  num_regions, max_depth, converged); every worker `ray_initialized` false.
- `out/08`–`12`: the copy after the last capture and again after the fingerprint: EQUAL to its
  certification, no journal file; fingerprint `0e32f24a…`, problems none. **Unchanged.**
- `out/13_replay.txt` (`state/runs/replay/`): each capture replayed in its own fresh process at
  e5764a6: **30 of 30 reproduce the census call bit for bit** (value, abserr, num_regions,
  max_depth, converged). These are the captures that count.

## Step 3 and 4: rules fixed before running

The classification rules are in `scripts/classify.py`'s docstring, written before it ran (SHA-256
of the script at its first run is in its output header). In short: deep levels are depths 16–20;
a deep level is flagged if its parent is Levin with residual > 10 × floor; it is attributed to
tiling if (i) P, L, R's computed end nodes do not tile, (ii) with the ends pinned P's double
residual ≤ 10 × its pinned floor, (iii) |ΔS_meas − ΔS_pred| ≤ 0.1 |ΔS_pred| + floor. A call is (a)
if every flagged deep level is attributed; (b) if θ′ changes sign in, or |θ′| falls to 1e-6 of the
span's max|θ′| on, the depth-16 region of the line.
- `out/14_pinned.txt` (`state/runs/pinned/`): each call re-run in a fresh process with
  `chebyshev_matrices` swapped (module attribute, this process only) for a wrapper that sets
  x[0] := b, x[−1] := a on a copy. Before each run the wrapper was compared element by element
  with the original on 22 spans × orders 24 and 47 (the call's span, the line's spans at depths
  0, 5, 10, 15, 16, 18, 20 and their halves): every interior node and every D bit-identical, the
  ends exactly a and b, the original untouched — 30 of 30.
- `out/15_classify.txt` (`state/classify/`): every line level's computed-end double re-solve
  reproduces the census line's residual and floor bit for bit (30 of 30 calls, every level).
- `out/16_ref_size.txt`: Y1's method sized per call. 11 calls under 45 min; 2 at 2.0–2.6 h; 17 at
  20 h to ~1 year (Φ up to 1.6e12 rad on the analytic spans).
- `out/17_ref_01.txt`: the first reference, sample 1, **did not converge** (|S384 − S192| = 0.30 of
  the smaller abserr). **Stopped under the brief's stop rule**; no other reference launched.
  `scripts/ref_batch.sh` stopped itself there (`out/ref_batch_progress.txt`).
- `out/18_assemble.txt`: `results.json`, `state/tables.md` (in `results.md`), from records only.

## Deviations (classified)

1. **The first capture failed** (`out/06`): `rerun.py` not copied. No Levin call was computed;
   re-run as `out/07` after copying it. UNINTENDED DRIFT.
2. **`x1_capture.py`'s PYTHONPATH check** now requires `<tree g>:<scripts>`, and the workers get the
   same (the census gave them the tree alone; each worker inserts its own directory anyway).
   STRUCTURALLY REQUIRED (the brief's rule for every Python command).
3. **`prov.py`, `worker.py`, `harness.py`, `rebuild.py`, `rerun.py`, `dispatch.py`, `spscan.py`,
   `hash_store.py`, `fingerprint_copy.py`, `compare_hashes.sh` copied unchanged** (SHA-256 equal to
   the checkpoint's where it records them); `prov.py`'s WIDER therefore resolves to this task's
   directory, and `state/rows.json` is a copy of the census's (equal SHA-256). IMPLEMENTATION CHOICE.
4. **`rec.sh`** writes `out/NN_name.txt` and adds the tree's HEAD and `status --short`. The
   census's own scripts (`hash_store.py`, `fingerprint_copy.py`, `compare_hashes.sh`) and
   `ref_batch.sh`'s progress file do not print their own SHA-256 in their outputs; rec.sh's header
   gives host, UTC and tree, and the scripts' SHA-256 are in `out/02`'s inputs or listed here
   (`compare_hashes.sh` b8c57a5a…, `hash_store.py` 881355fd…, `fingerprint_copy.py` 348ccce9…).
   `out/ref_batch_progress.txt` has no header. UNINTENDED DRIFT.
5. **One deep-lines entry was printed before the rule** (format discovery; not drawn). See Step 0.
   UNINTENDED DRIFT (no effect on the draw, which is mechanical).
6. **The mp re-solve** (`dcommon.solve_mp`) uses the exact D (mp_base / exact half-width, as the
   floor-breakdown scheme), not the code's double D; f, θ′ and θ_mod_2pi are the call's own double
   values at the grid's double nodes; sin/cos in mp of the double phase; the p-mode gate from the
   double solve. Changed from `q3_decompose.py`: no closed form exists, so its resolution/true-error
   components are dropped; the decomposition is computed vs pinned ends and double vs mp, plus the
   first-order offset prediction g·δ (Levin) or w_end·Δg (CC). IMPLEMENTATION CHOICE.
7. **The classification's thresholds** (deep levels 16–20; "matches" = within 10 % + floor) are
   mine, fixed in `classify.py` before it ran. IMPLEMENTATION CHOICE. Consequence, read after:
   10 (c) calls fail only on a bottom Levin-parent/CC-halves level or on the 10 % match.
8. **(b)'s test** read as |θ′| ≤ 1e-6 × max_span|θ′|, on the line's depth-16 region. IMPLEMENTATION
   CHOICE. Consequence, read after: 3 analytic calls pass it with θ′ ≈ 6e5–9e5 (span max ~1e12), and
   the G+q+r row with a true θ′ zero 1.8e-3 outside its depth-16 region fails it.
9. **Tiling per solve's own order** (24 Levin, 47 CC) as well as at 24, as the brief words it:
   identical on every level. IMPLEMENTATION CHOICE.
10. **References: only one attempted**, and it failed the convergence test; the stop rule was
    applied. Two more (samples 3, 15) were over 45 min and 17 far beyond; none launched, none
    registered. STRUCTURALLY REQUIRED.
11. **`refq.py`** adapts Y1's `refquad.py` (copy kept unchanged as `refquad_y1.py`): this task's
    capture format, no in-script replay (each capture's replay is `out/13`), no cross-tree sample,
    a `size` sub-command. IMPLEMENTATION CHOICE.
12. **`refq.py size` built the full panel-edge array** for the analytic calls (up to 3.1e9 panels,
    ~25 GB transiently) before reporting the estimate; it completed, but a size check should not
    need to. UNINTENDED DRIFT.
13. **`assemble.py` was edited once** after its first run to exclude unconverged references from
    the honesty totals and mark them; only `out/18` (the edited script's run) is kept.
    IMPLEMENTATION CHOICE.

## Observations not acted on

- The pinned run of `g-1e-8-1-2174` call 1 (G+q+r; θ′ zero at x = 5.63671, z ≈ 279.5) converges at
  depth 7 with a value 1.7e3 × its own abserr from the census value, which the census at 1e-7,
  the census at 1e-8 and the pinned run at 1e-7 all agree on: very likely the Y1 stationary-point
  dishonesty, uncovered once tiling no longer drives refinement near it. Unverified.
- At a line's bottom, a Levin parent whose halves fall below the 6π gate is compared with
  Clenshaw–Curtis halves; that residual (1e3–3e4 × floor) is not an end-node effect.
- Y1's 192/384 Gauss–Legendre pair at D = 512 rad may be resolution-limited at 192 nodes.

## State handed on

Nothing running; no Ray, no lock, no registry run. Tree g clean at `e5764a6`. The scratch copy
unchanged (`out/08`–`12`). Nothing written outside `var/levin-phase-floor-06c/depth20-tiling/`.
