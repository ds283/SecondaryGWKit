# levin-phase-floor 06m §5 — the census at the code commit (Mac Studio)

**Subject:** prompt 06m §5 (`prompts/levin-phase-floor/06m-signed-yardstick-in-the-driver.md`) on the Mac Studio
(`mps026298`), 2026-11-06, by the census agent of Part B. It ran Gate M0, the census $m$, Gate M and the descriptive
tables, Ray-free and with no lock, on 06's certified scratch copy of A3 v2.
- **The tree $m$:** the worktree `SecondaryGWKit-lpf06m-38f6a82` at Part B's dispatch SHA `38f6a828`. Its code is
  the code commit `48cf87c0`'s, and `AdaptiveLevin/levin_quadrature.py` is `beaf0e23…`. No variant was installed.
- **The "before":** route B's census, `levin-phase-floor-06l-census-routeB-20261105T100126` (checkpoint
  `5cb3d3c0…`).
- **Scope:** §4 (R25, R26) is the rehearsal agent's.

**Commit:** none here. The orchestrator commits the records from the laptop as `docs/levin-phase-floor/06m/census/`.

**Result:**
- **Gate M0 holds:** all 100 K0 units equal route B's census bit for bit (2,110 calls, 2,110 region records).
- **Gate M holds:** all 18,560 units are `done` and each equals route B's census bit for bit.
  - That is 289,482 calls and 289,482 region records, with 2,871,152 returned leaves. Nothing differs.
  - It includes the 21,264 calls where route B fired (271,030 firings).
- **So 06l's verdicts are the code's,** by identity and not re-scored: L1, C1, C2, H1, S1 and K1 hold, and L2 fails
  as scored (`VERIFICATION.md` §6l).
- **The store and both earlier censuses are unchanged.** The scratch copy, route B's census (150 files) and 06g's
  census (149 files) are byte-identical before and after.

## What shipped

No code. The records are in `var/levin-phase-floor-06m/census/`:
- **The records themselves:** `log.md` (this file), `predictions.md`, `verification.md`, `issues.md` and
  `KEPT_ON_MAC_STUDIO.txt`.
- **`scripts/`:** 06l's harness copied from `docs/levin-phase-floor/06l/scripts/`, changed where marked "prompt 06m"
  (deviations 1–3).
  - The diffs against 06l's are `out/scripts_diff_at_copy.txt` (empty), `out/scripts_diff_before_k0.txt`,
    `out/scripts_diff_at_stop2.txt` and `out/scripts_diff_final.txt`.
  - Changed: `prov.py`, `refs_common.py` and `run_k0.py`.
  - New: `index_routeB_06m.py`, `compare_06m.py`, `hash_runs_06m.py`, `launch_06m.py`, `launch_census_06m.py`,
    `census_projection_06m.py`, `descriptive_06m.py`, `gate_m_summary_06m.py` and `kept_list_06m.py`.
- **`out/`:**
  - every command and its output (`out/cmd/`, `out/commands.log`);
  - M0's comparison, `out/k0/compare_m0.json`;
  - Gate M's summary, `out/gate_m/summary.json` (the full comparison, `out/gate_m/compare_m.json`, 41 MB, is kept on
    the Mac Studio);
  - the descriptive tables, `out/descriptive/descriptive_06m.json` with `unconverged_outside_720_m.csv` and
    `unconverged_outside_720_06g.csv`;
  - `out/census_projection.json`;
  - the before and after hashes, `out/runs_hashed_before.txt`, `out/runs_hashed_after.txt` and `out/cmd/04`, `40`–`42`.
- **`state/`:** `rows.json` (06g's, `aa523d28…`), `index_routeB.json`, `plan_k0.json`, K0's workers (`state/k0/tree/`),
  `gate_trees.sh`, the development self-test, the `cache_built` reader, and `superseded/` (deviation 12).
- **Two registered runs, both `done`:**
  - `levin-phase-floor-06m-k0-20261106T101118`: 2 of 2 commands.
  - `levin-phase-floor-06m-census-20261106T102153`: 18,560 of 18,560 units, exit 0, wall 18,465 s, checkpoint
    `ea5bc3ab…`.
  - **The census's checkpoint and region pickles stay on the Mac Studio in full; 07 compares against them.**

## Predictions and verdicts

Copied verbatim from the code log's "Predictions and verdicts" at 2026-11-06T10:07:44Z, before Gate M0's first unit.
The timestamped copy is `predictions.md` (SHA-256 `a57843d5…`). It was not rewritten.

- **§5, Gate M0** — copied verbatim:

  > **Gate M0 — K0 first.** At $m$ the harness reproduces route B's census on 06k's 100 K0 units
  > (`docs/levin-phase-floor/06l/out/k0/units.json`), bit for bit by the comparison above. It is one
  > registered run with slug `k0`. **Holds** if all 100 are equal. A failure is a stop: the code is not
  > what 06l measured.

- **§5, Gate M** — copied verbatim:

  > **Gate M — the census is route B's.** **Rule:** all 18,560 units are `done`, and each equals route
  > B's census by the comparison above. **Holds** if all are equal. A failure is a stop, with the first
  > differing units and fields reported.

| Prediction | Verdict | The number it was scored on |
|---|---|---|
| **Gate M0** | **holds** | 100 of 100 units equal: 2,110 calls (566 group, 1,544 analytic) and 2,110 region records (32,518 returned leaves); 0 differing. 248 of the calls are ones route B fired in (4,393 firings). |
| **Gate M** | **holds** | 18,560 of 18,560 units `done`, 0 in error, 0 over budget; all 18,560 equal. 289,482 calls (74,650 group, 214,832 analytic) and 289,482 region records (2,871,152 returned leaves); 0 differing. 21,264 calls are ones route B fired in (271,030 firings). |

**How the comparison was made** (`scripts/compare_06m.py`, the same for M0 and M):
- **Matching:** every unit is matched by its id to route B's checkpoint record (the line's SHA-256 checked) and its
  region-pickle record (SHA-256 checked). Every call is matched by unit, kind (group `data.calls`, analytic
  `data.other_calls`), index, `notify_label` and `x_span` (both ends by `float.hex`). Every region record (pickle
  `regions` and `other_regions`) is matched by kind, index and span.
- **What is compared:** the whole unit record and the whole pickle record, by 06l's `compare_k0.diff`, imported
  unchanged. Floats are compared by `float.hex`, arrays by dtype, shape and bytes.
- **Module evidence:** each unit and each worker shows the tree's module `beaf0e23…`, with no variant, no foreign
  module and no Ray.

**What is excluded, and why.** The excluded set is `compare_k0.py`'s `EXCLUDED` plus every key ending `_s`, reported
apart and never counted. The inventory of every excluded key, with its present/differing counts, is in
`out/gate_m/summary.json` (`excluded_keys_inventory_units_present_differing`).
- **Inside a call record:**
  - `elapsed` (timing);
  - `route_06l` (route B's instrument; None at $m$);
  - `route_06k` (None on both sides; differs in 0 units).
- **Inside a region record:** nothing.
- **Elsewhere in the pickle:**
  - `result_full.compute_time`, `result_full.WKB_Levin_subintervals[].groups[].elapsed` and `scan[].seconds` (timing);
  - the lists `route_06l[]` and `route_06k[]` (the instruments).
- **Unit level, timing:** `wall_s`, `rebuild_s`, `scan_s`, `cc_floor_s`, `checked_06g_s`, `declared_06h_s`,
  `analytic_06h.seconds`, `deep_lines.seconds`, `w3.seconds`, `result.compute_time`, and `stored.compute_time` (the
  stored row's own value, differing in 0 units).
- **Unit level, not timing, and none a computed value:**
  - **Provenance:**
    - `tree_head` differs (`af4c4dc1` against `38f6a828`); `tree_status_short` differs in 0 units.
    - `scripts_sha256` differs in `prov.py` only.
    - `worker`, `worker_pid` and `worker_tree` (in the census, `worker` and `worker_tree` differ in 0 units).
    - The pointers `pickle`, `pickle_offset`, `pickle_length` and `pickle_record_sha256`; the records they point to are
      compared whole.
    - `modules`: the module paths and counts by root.
    - `rows_sha256` and `w2_capture_sha256` differ in 0 units.
  - **Run plan:** `cache_built`. In K0 it differs in 95 of 100 units, because a K0 worker runs a subset of its k's units
    (06k's correction, already in `compare_k0.py`). In the census it differs in 7 units, by order only (observation 1).
  - **Configuration:** `budget_s` is 300 in every unit, differing in 0; it is excluded only by the `_s` rule.

## Deviations from the prompt

1. **The copied harness's tree and paths** (`prov.py`, `refs_common.py`): the tree `g` is `38f6a828`; the records
   directory is one level deeper than 06l's (`assert HERE.parents[3]`); route B's and 06g's census runs, their
   checkpoint SHA-256 and the module's SHA-256 are named. **STRUCTURALLY REQUIRED.**
2. **`run_k0.py`'s output title names 06m.** **IMPLEMENTATION CHOICE.**
3. **New scripts:**
   - **STRUCTURALLY REQUIRED:**
     - `index_routeB_06m.py`, from `index_06g.py`: 06l's index was deleted, so route B's checkpoint is indexed afresh.
     - `compare_06m.py`: `compare_k0.py` compares against 06g, so a new comparison against route B was needed; it
       reuses `compare_k0`'s exclusions and diff unchanged.
     - `launch_census_06m.py`, from `launch_census_06l.py`: 06l's launcher requires a variant. This one passes none,
       uses slug `census` and prompt 06m, and refuses to launch unless the module is `beaf0e23…`.
   - **IMPLEMENTATION CHOICE:** `hash_runs_06m.py`, `launch_06m.py`, `census_projection_06m.py`, `descriptive_06m.py`,
     `gate_m_summary_06m.py` and `kept_list_06m.py`.
4. **M0's units and comparison ran as one registered run** through 06l's plan driver `run_06j.py`, unchanged, as 06l
   ran K0. **IMPLEMENTATION CHOICE.**
5. **My first edit to `run_k0.py` broke a line:** a comment swallowed `, inputs):`. `ast.parse` caught it, and I fixed
   it before the launch. **UNINTENDED DRIFT** (a failed attempt).
6. **Three hand entries in `out/commands.log` first had estimated times,** which I corrected to the real ones.
   **UNINTENDED DRIFT** (cosmetic).
7. **No store fingerprint of the scratch copy:** byte equality (SHA-256, size, mtime) with `copy_files_certified.txt`
   before and after stands in for it. **IMPLEMENTATION CHOICE.**
8. **`worker.py` is unchanged,** so at $m$ each call summary holds `route_06l: None` rather than no key. It is excluded
   under either reading. **IMPLEMENTATION CHOICE.**
9. **The exclusions are `compare_k0.py`'s whole set,** which also holds unit-level provenance and run-plan keys, not
   only timing (the list above).
   - Read as the brief states it (call records and region records), those keys lie outside both, and inside them only
     `elapsed`, `route_06l` and `route_06k` are excluded.
   - Read literally as "only timing", `tree_head`, the pickle offsets or the worker names can never be equal across two
     runs.
   - Accepted by the orchestrator at stop 2. **IMPLEMENTATION CHOICE.**
10. **`launch_census_06m.py` shortens 06l's unused `ROUTES` text.** **IMPLEMENTATION CHOICE** (cosmetic).
11. **Gate M's comparison and the descriptive reader ran with five processes, not six,** so that with the
    orchestrator's one-process Gate M reader at most six ran. **IMPLEMENTATION CHOICE.**
12. **`descriptive_06m.py` was changed after its first run.** I added the r bands of the $\Delta_1/\phi_P > 10$ count
    and four cross-counts of the flag's cost.
    - The first run's output and script are kept in `state/superseded/`.
    - The tables were recomputed with the final script, not re-stamped (`CLAUDE.md` rule 7).
    - Every number the first run shared with the second is equal.
    - **UNINTENDED DRIFT** (a failed attempt).
13. **The cost and the 06g column were computed from the checkpoints and manifests** of all three censuses, rather
    than copied from 06l's tables. They reproduce 06l's census walls (18,340 s, 20,361 s), unit walls and region
    counts. Route B's flag's cost is also tabulated, for the record; by Gate M it is $m$'s. **IMPLEMENTATION CHOICE.**
14. **The flag's cost counts group and analytic calls apart.** No analytic call is unconverged in any of the three
    censuses. The 720 are found by `refs_common.scan06a_sp` on 06a's census (read only; 720 rows). **IMPLEMENTATION
    CHOICE.**
15. **A development self-test of the comparison** ran on three of route B's units against themselves, outside the
    scored set (`out/cmd/10`). **IMPLEMENTATION CHOICE.**

## Verification performed

`verification.md` has the full text. In brief:
- **The Gate checks** (`out/cmd/01`–`09`):
  - The trees are at `38f6a828` (clean with `--ignored`), `5ff8a5e4` (clean) and `28ebaf07` (clean).
  - The module is `beaf0e23…`, equal to `git show 38f6a828:` and unchanged since `48cf87c0`.
  - `git diff --stat` against `5ff8a5e4` and against `af4c4dc1` (06l's census tree) touches only the module, the two
    AdaptiveLevin test modules, `CLAUDE.md` and `.gitattributes`.
  - The registry held 60 runs, none running or stale. The lock is free, with no Ray process.
  - `rows.json` is `aa523d28…`.
- **The scratch copy** is EQUAL to its certification before and after, and before equals after (`04`, `05`, `40`–`42`).
- **Route B's census** has 150 files matching 06l's `KEPT_ON_MAC_STUDIO.txt`. **06g's census** has its checkpoint and
  16 pickles matching 06g's list. Both directories are byte-identical before and after, all 299 files (`06`, `43`,
  `44`).
- **No variant and no Ray:**
  - In each of the census's 16 workers, `levin_quadrature_sha256` is `beaf0e23…` (the file and the module) and
    `variant_06k` is None.
  - Every unit's `AdaptiveLevin.levin_quadrature` is the tree's file, with no `variant` root and no foreign module.
  - In every `bye`, Ray was not initialised. The same holds for K0's 15 workers.
- **The trees after:** the code worktree is clean with `--ignored`, with no `__pycache__`; the main checkout is clean.
  The registry holds 62 runs, all finished. The lock is free (`45`).
- **The orchestrator checked independently:**
  - It re-ran K0 units `g-1e-7-3-34738` and `g-1e-8-2-34906` (both with route B firings) with the command given at
    stop 2, and found them equal to route B with its own comparator.
  - Its comparator also found all 100 of M0's units equal.

### The descriptive tables (prompt §5; not scored)

**The flag's cost**, unconverged group calls (`out/descriptive/descriptive_06m.json`). The request is
max(atol, rtol |v|) with the call's own atol and rtol, and the share is abserr / the unit's `total_abserr`. The 720
are found by 06a's rule (06a's 1e-7 scan shows a sign change of θ′). No analytic call is unconverged in either census.
At $m$ the flag's cost is route B's, by identity.

| | $m$ (= route B), outside the 720 | 06g, outside the 720 | $m$, in the 720 | 06g, in the 720 |
|---|---|---|---|---|
| unconverged calls (units; rows) | **114 (84; 46)** | 1,632 (946; 476) | 311 (285; 163) | 679 (461; 251) |
| at 1e-7 / 1e-8 | 52 / 62 | 801 / 831 | — | — |
| abserr within the request | **104** | 1,628 | 309 | 677 |
| over the request | **10, all `node_limited`**, at most **1.048×** (1e-7: 1, 1.0057×; 1e-8: 9) | 4, all `node_limited`, at most 1.071× | 2, `node_limited`, at most 1.0038× | 2, `node_limited`, at most 1.00026× |
| `node_limited` in all | 10 | 4 | 2 | 2 |
| holding an untrusted leaf (at `depth_max`) | 105: the 104 within the request and 1 of the 10 | 1,629 | 309 | 677 |
| share of the row's `total_abserr`: largest | **4.73 %** | 34.2 % | 53.0 % | 53.0 % |
| share ≤ 1 % / (1 %, 5 %] / (5 %, 50 %] / > 50 % | 104 / 10 / 0 / 0 | 1,530 / 71 / 31 / 0 | 198 / 51 / 61 / 1 | 486 / 114 / 78 / 1 |

These are the orchestrator's counts from 06l's extract exactly: 114 calls in 84 units; 104 within their request; at
most 1.05× over; share at most 4.7 %. Every one of the 104 within its request holds an untrusted leaf, and none is
`node_limited`: each is flagged by the leaf, not by its error.

**Cost** (census wall from manifest `created` to the last heartbeat; unit walls and group-call regions from the
checkpoints):

| | census wall | unit wall 1e-7 / 1e-8 / all | group-call regions 1e-7 / 1e-8 / all | load seen |
|---|---|---|---|---|
| 06g | 20,361 s | 27,419 / 31,116 / 58,535 s | 1,124,662 / 1,255,648 / 2,380,310 | (06g's records) |
| route B | 18,340 s | 24,063 / 26,432 / 50,495 s | 939,882 / 1,053,755 / 1,993,637 | about 15 (06l) |
| **$m$** | **18,465 s (5 h 8 min)** | **23,970 / 27,070 / 51,040 s** | **939,882 / 1,053,755 / 1,993,637** | 5–6 at launch (10:21Z); 24.8 / 17.9 / 11.4 at 15:31Z; about 37 at the end by the orchestrator (ChronoSync, fileproviderd, WindowServer) |

- **Regions:** $m$'s group-call regions are route B's exactly, 0.838 of 06g's.
- **Unit walls:** 1.011 of route B's and 0.872 of 06g's.
- **Projection:** 17,640 s, from ρ = 0.962 of route B's walls on K0. The run took 4.7 % more. The backstop reported
  twice (15:15:55Z and 15:25:55Z, one unit in flight each time; P_o 17,731 s and 18,263 s) and stopped nothing. Cost is
  not an argument (README §5 rule 4).

**Route B's firings** (from route B's census, `route_06l.n_firings`), which $m$ reproduces:

| | outside the 720 | in the 720 |
|---|---|---|
| group calls firing / firings | 18,352 / 232,587 | 2,690 / 38,221 |
| analytic calls firing / firings | 194 / 194 | 28 / 28 |

The totals, 232,781 outside and 38,249 in, are 06l's.

**Checked Levin leaves with $\Delta_1/\phi_P > 10$**, for `[06m-levin-floor-misses-node-rounding-far-from-the-origin]`.
- **Source:** each unit's `checked_leaves_06g` in $m$'s pickles. These are every checked leaf, returned or bisected,
  with type Levin, its $\Delta_1$ and $\phi_P$ (`phase_err`), its r and its final trust.
- **Bands:** r is split by 06g's rule (b): r < 1/4, 1/4 ≤ r < 1, and r ≥ 1.

| | Levin leaves checked | $\Delta_1/\phi_P > 10$ | of which r < 1/4 (all trusted by r < 1/4) | 1/4 ≤ r < 1 | r ≥ 1 | returned untrusted | 1/4 ≤ r < 1 and returned | largest ratio in 1/4 ≤ r < 1 |
|---|---|---|---|---|---|---|---|---|
| outside the 720, group | 1,917,822 | 1,564,785 | 661,112 | 471,378 | 432,295 | **0** | **0** | 1.0 × 10¹⁰ |
| outside the 720, analytic | 1,067,208 | 86,294 | 83,911 | 1,462 | 921 | 0 | 0 | 513 |
| in the 720, group | 329,064 | 248,610 | 101,923 | 75,892 | 70,795 | 368 | 81 | 1.8 × 10¹² |
| in the 720, analytic | 223,136 | 17,760 | 17,348 | 265 | 147 | 0 | 0 | 1,041 |

$\phi_P \le 0$ on no Levin leaf. Read this table with three points in mind:
- **The count is mostly resolution, not noise:** a leaf with r < 1/4 is converging and is trusted by that clause, and a
  leaf with r ≥ 1 is not resolved.
- **No Levin leaf is trusted on the noise clause at $\Delta_1/\phi_P > 10$.** Every trusted one has r < 1/4, which
  agrees with K1's maximum of 2.85.
- **Outside the 720, every Levin leaf with $\Delta_1/\phi_P > 10$ and 1/4 ≤ r < 1 was refined.** None is returned.

The records do not hold what the issue needs: the halves' solves, or the node-rounding term a Levin floor would need,
to tell which of these leaves carry node rounding far from the origin. **It is left for 07's census.**

## Observations not acted on

1. **The order of two producer builds differs in 7 units.**
   - In these 7 of the census's 18,560 units, `cache_built` lists the same five builds as route B, but two
     `producers_T` builds (mode serials 1 and the worker's k serial) appear in the other order (`out/cmd/32`).
   - The units are `g-1e-7-2-32814`, `g-1e-7-1-20254`, `g-1e-7-0-10168`, `g-1e-8-2-32862`, `g-1e-8-2-32814`,
     `g-1e-8-1-20254` and `g-1e-8-1-20086`.
   - Every computed value in those units is bit-identical to route B's.
   - It is a harness record (the producer cache's build order, apparently not deterministic across runs), not a value.
     I have not looked further.
2. **The $\Delta_1/\phi_P > 10$ count** is as given above.
   - 1,564,785 of the 1,917,822 checked Levin leaves in group calls outside the 720 exceed 10. Most are unresolved
     leaves, not noise-level ones.
   - In the 720, 368 Levin leaves with $\Delta_1/\phi_P > 10$ are returned untrusted at `depth_max`, 81 of them with
     1/4 ≤ r < 1. These are stationary-point rows, 07's.
3. **The run took 4.7 % longer than projected,** with the host's load rising to about 37 at the end from system
   services. The region count, which does not depend on load, is route B's exactly.
4. **`compare_k0.py`'s `_s` rule also excludes `budget_s`,** a configuration value. It is equal in every unit (300).

## State handed on

- **Nothing is running.** The lock is free, and no Ray process is listed.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`, with SHA-256 and size), and not copied to the laptop:
  - the census run directory in full: checkpoint, 16 region pickles, worker logs and events;
  - K0's run directory;
  - K0's 15 worker pickles and `units.jsonl` (15 MB) under `state/k0/tree/`;
  - `out/gate_m/compare_m.json` (41 MB), `state/index_routeB.json` (10 MB) and `state/rows.json` (7.8 MB).
  - The census checkpoint is `ea5bc3ab6b618d562b9dee0a7ed4bdba5c1999e8348b2ad33bdadb6135e63c2e` (2,077,206,324 bytes,
    18,560 records); the run directory holds 150 files, 4,413,262,193 bytes. The list is 185 lines, SHA-256
    `2efec35d…`, written by `scripts/kept_list_06m.py`.
  - **The census's checkpoint and unit records stay in full; 07 reads them as its "before".**
- **Read only and unchanged:** the scratch copy, route B's census and 06g's census.
- **Issues:** `issues.md` proposes no new issue. Its numbers bear on the narrower issue the prompt proposes,
  `[06m-calls-flagged-unconverged-within-their-request]`, which is for the user at the records commit.
