### 6m.C The census at the code commit (prompt 06m §5)

On the Mac Studio (mps026298), 2026-11-06, by Part B's census agent.
- **Tree and store:** from the worktree `SecondaryGWKit-lpf06m-38f6a82` at Part B's dispatch SHA `38f6a828`, whose
  code is the code commit `48cf87c0`'s (`AdaptiveLevin/levin_quadrature.py` `beaf0e23…`), with no variant installed.
  It read 06's certified scratch copy of A3 v2, read-only.
- **Constraints:** Ray-free, no lock, no store built, nothing written in the tree.
- **Records:** `06m/census/`. The log is [`06m/census/log.md`](06m/census/log.md).

#### The Gate and the inputs

Commands and outputs are `06m/census/out/cmd/01`–`09`.
- **Trees:** the worktree is at `38f6a828`, clean with `--ignored`. Its module equals `git show 38f6a828:` and is
  unchanged since `48cf87c0`. The main checkout is at `28ebaf07`, clean.
  - `git diff --stat 5ff8a5e4 38f6a828` (outside docs and prompts) touches the module, the two AdaptiveLevin test
    modules and `CLAUDE.md`.
  - Against 06l's census tree `af4c4dc1` it also touches `.gitattributes`.
- **Registry and lock:** 60 runs, none running or stale. The lock is free, and no Ray process is listed.
- **The scratch copy:** EQUAL to `copy_files_certified.txt` (SHA-256, size and mtime of all five files), with no
  journal, `-wal` or `-shm` file.
- **Route B's census** (`levin-phase-floor-06l-census-routeB-20261105T100126`): checkpoint `5cb3d3c0…`, and all 150
  files equal 06l's `KEPT_ON_MAC_STUDIO.txt`.
- **06g's census** (`levin-phase-floor-06g-census-20261101T114315`): checkpoint `03adcbba…`, and its 16 region pickles
  equal 06g's list. All 149 files are hashed ([`06m/census/out/runs_hashed_before.txt`](06m/census/out/runs_hashed_before.txt)).
- **`rows.json`:** 06g's, copied, `aa523d28…`.
- **The predictions:** Gates M0 and M were copied verbatim into [`06m/census/predictions.md`](06m/census/predictions.md)
  at 10:07:44Z, before any unit ran.

#### The harness

06l's `scripts/`, copied, with no variant installed: the tree's module is the code.
- The changes, each marked "prompt 06m", are the tree and its paths (`prov.py`, `refs_common.py`) and a title
  (`run_k0.py`).
- New scripts: route B's index, the comparison, the launchers and the readers.
- The diff is [`06m/census/out/scripts_diff_final.txt`](06m/census/out/scripts_diff_final.txt).
- `worker.py` and `run_census.py` are unchanged from 06l's.

#### Gate M0 — holds

Run `levin-phase-floor-06m-k0-20261106T101118` (`done`, 2 of 2) ran 06k's 100 K0 units at the tree, six workers,
in 330 s ([`06m/census/out/k0/compare_m0.json`](06m/census/out/k0/compare_m0.json)).
- **Equality:** all 100 units equal route B's census.
  - That is 2,110 calls and 2,110 region records (32,518 returned leaves), 0 differing.
  - It includes the 248 calls route B fired in (4,393 firings).
- **The comparison:**
  - Every call is matched by unit, kind, index, label and span (hex).
  - Then every key of the unit record and of the pickle record is compared by 06l's `compare_k0.diff`: floats by
    `float.hex`, arrays by dtype, shape and bytes.
  - The exclusions are `compare_k0.py`'s. Inside call records they are `elapsed` (timing), `route_06l` (route B's
    instrument) and `route_06k` (None on both sides).
  - The other exclusions are unit-level timing and provenance (the log lists each).
- **Workers:** all 15 used the tree's module `beaf0e23…`, with no variant, no foreign module and no Ray.
- **The orchestrator's check:** it re-ran two units with route B firings itself and found them equal to route B.

#### The census

Run `levin-phase-floor-06m-census-20261106T102153`: `done`, 18,560 of 18,560 units, exit 0.
- No unit is in error or over budget.
- Wall: 18,465 s against a projection of 17,640 s. The backstop reported twice and stopped nothing.
- All 16 workers used `beaf0e23…`, with no variant and no Ray.

#### Gate M — holds

[`06m/census/out/gate_m/summary.json`](06m/census/out/gate_m/summary.json); the full comparison, 41 MB, is kept on the
Mac Studio.
- **Equality:** all 18,560 units equal route B's census by the same comparison.
  - That is 289,482 calls (74,650 group, 214,832 analytic) and 289,482 region records (2,871,152 returned leaves), 0
    differing.
  - It includes the 21,264 calls route B fired in (271,030 firings).
- **Of the excluded keys,** only timing, provenance and `route_06l` differ, apart from one: `cache_built` lists the
  same producer builds in another order in 7 units (log, observation 1).
- **So 06l's verdicts are the code's, by identity and not re-scored:** L1, C1, C2, H1, S1 and K1 hold, and L2 fails as
  scored (§6l).

#### Descriptive (not scored)

[`06m/census/out/descriptive/descriptive_06m.json`](06m/census/out/descriptive/descriptive_06m.json)

- **The flag's cost** at $m$ (route B's, by identity), outside the 720:
  - 114 unconverged group calls in 84 units (46 rows).
  - 104 are within their request max(atol, rtol |v|). Each of the 104 holds an untrusted leaf, and none is
    `node_limited`.
  - 10 are over the request, all `node_limited`, at most 1.048×.
  - The largest share of a row's `total_abserr` is 4.73 %.
  - In the 720: 311 calls, 2 over the request (`node_limited`, at most 1.0038×), largest share 53.0 %.
- **06g, the same measure:**
  - Outside the 720: 1,632 calls in 946 units; 1,628 within the request; 4 over, `node_limited`, at most 1.071×;
    largest share 34.2 %.
  - In the 720: 679 calls.
- **No analytic call** is unconverged in either census.
- **Cost:**

  | | census wall | unit wall | group-call regions |
  |---|---|---|---|
  | $m$ | 18,465 s | 51,040 s | 1,993,637 |
  | route B | 18,340 s | 50,495 s | 1,993,637 |
  | 06g | 20,361 s | 58,535 s | 2,380,310 |

  - The regions are route B's exactly, 0.838 of 06g's.
  - Load: 5–6 at launch, about 25–37 at the end, from system services.
- **Checked Levin leaves with $\Delta_1/\phi_P > 10$** (for `[06m-levin-floor-misses-node-rounding-far-from-the-origin]`):
  - Outside the 720 there are 1,564,785 of 1,917,822 in group calls. Most have r < 1/4 (trusted by that clause) or
    r ≥ 1.
  - None is trusted on the noise clause.
  - None is returned untrusted.
  - All 471,378 with 1/4 ≤ r < 1 were refined.
  - The records cannot isolate node rounding, so this is left for 07's census.

#### After

- **The scratch copy:** EQUAL to its certification, and before equals after (`06m/census/out/cmd/40`–`42`).
- **Route B's and 06g's census run directories:** all 299 files byte-identical before and after (`43`, `44`).
- **The trees:** the worktree is clean with `--ignored`, with no `__pycache__`; the main checkout is clean.
- **Registry and lock:** 62 runs, all finished. The lock is free, and no Ray process is listed (`45`).
