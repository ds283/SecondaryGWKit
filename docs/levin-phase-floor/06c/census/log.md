# levin-phase-floor 06c §5 — the census (Mac Studio)

The preparation sections were drafted before the launch (2026-10-29, 12:59Z) and completed after
the run; the run, the scores and the post-run sections were written after it.

**Subject:** prompt 06c §5 on the Mac Studio (mps026298), Ray-free, no lock. 06b's census re-run
under 06c's global error budget and 06d's Clenshaw–Curtis floor, against 06b's run as the "before".
- **Trees:** $g$ is the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6`,
  detached at Part B's dispatch SHA `e5764a6` (code `ad9a5f4` + `3d70db9`). $p$ is the worktree
  `…-lpf06c-fef401c` at 06c's code commit's parent `fef401c` (06b's code), used for Gate X0's
  reproduction only. 06b's census tree `92bc62f` is read through git history only.
- **Run:** `levin-phase-floor-06c-census-20261029T130214`. Launched 13:02:14Z (sh 1752, driver
  1756); the run ended `done`, exit 0, at about 15:32:44Z: about 2 h 30 min, against 06b's 51 min.
  All 18,560 units are `done`, none in error, none over budget by the dispatcher's test (but see
  deviation 22: nine units ran longer than 300 s).
- **Commit:** none here; the laptop commits this directory.

**Result:**
- **X0 holds**, all three clauses.
- **X1 fails.** 24 drawn; 1 dropped (its $S_9$, $S_{10}$ differ by 0.0215 of `abserr`$_g$); of the 23
  left, 22 are within their declared error and **one is not: `g-1e-7-0-14127` call 3, 1.375 ×
  `abserr`$_g$** (true error $9.4\times10^{-8}$ relative, $1.3\times10^{-4}$ of the call's request).
  The reference is sound there ($S_{10}$, $S_{11}$, $S_{12}$ agree to $3\times10^{-5}$ of
  `abserr`$_g$); the driver at `rtol` $10^{-11}$ returns g's value again, so the error is shared by
  a leaf and its descendants, outside the trust check's reach (prompt §8, "What it does not do").
- **X2 fails.** Of 06b's 20,425 (1e-7) and 22,442 (1e-8) X2-type calls outside the 720 rows, 53
  and 127 are unconverged under g: 49 and 63 with an untrusted leaf at depth 20, 2 and 27 at depth
  20 with every leaf trusted, 2 and 37 floor-limited.
- **X3 holds.** No call under g declares an infinite (or NaN) abserr, at either rtol.
- **X4 fails.** 263 of 8,092 rows at 1e-7 and 302 of 8,092 at 1e-8 violate it, by up to 29.9× and
  71.5× the summed `total_abserr`. In every violating row a group call itself disagrees with 06b's
  beyond both calls' declared errors, and g reports that call converged. On the four worst rows'
  calls, referenced as X1 is (descriptive), g is within 0.11–0.61 of its `abserr`; 06b is off by
  116–445 of its own.
- **X5 fails.** 109 moved calls at 1e-7 and 153 at 1e-8 reach depth 20 (14 and 12 group calls, the
  rest analytic blocks); the largest has 74,941 regions and 277,395 solves.
- **The store:** the scratch copy is byte-identical to its certification before, after X0, at the
  launch, after the run and after the fingerprint, and gives `0e32f24a…`. 06b's checkpoint is
  `f934ea24…` throughout. No worker initialised Ray.

## What shipped

No code. Records only, in `var/levin-phase-floor-06c/census/` (for the laptop to commit as
`docs/levin-phase-floor/06c/census/`) and the run directory (kept in full on the Mac Studio):
- the harness, 06b's census copy changed as §5 allows (`scripts/`; the diff in
  `out/scripts_diff_against_06b_at_prep.txt` and `…_final.txt`), and 06b's copy unchanged in
  `state/scripts_06b_reference/`;
- `state/predictions.md` (fixed before the launch), Gate X0's runs (`state/X0/`), the X1 sample and
  captures (`state/x1_selection.json`, `state/x1_captured.json`, `state/X1capture/`), X4's
  descriptive captures (`state/x4_worst_calls.json`, `state/x4_captured.json`, `state/X4capture/`);
- every command and its output (`out/cmd/00`–`52`, `out/commands.log`), the X1 and X4 references
  (`out/x1_reference.json`, `out/x4_reference.json`), the per-unit table
  (`out/census_rows_06c.csv.gz`), every depth-20 call's recomputed line
  (`out/deep_lines_06c.json`) and `out/convergence_kinds_g.txt`;
- the run `var/runs/levin-phase-floor-06c-census-20261029T130214/` (1.3 GB, 149 files): checkpoint
  (`b61cb034…`, 775 MB), 16 region pickles, the events and the worker logs. 07's census reads it.

## Before the launch (in order; every command and its output in `out/cmd/`)

1. **Harness copied** (`00`): 06b's 48 census scripts from the code worktree's
   `docs/levin-phase-floor/06b/census/scripts/`, which equals (`diff -r`)
   `var/levin-phase-floor-06b/census/scripts/`, into `scripts/` and unchanged into
   `state/scripts_06b_reference/` (`state/scripts_06b_reference.sha256`). The diff against 06b's
   copy at preparation is `out/scripts_diff_against_06b_at_prep.txt`.
2. **Rows** (`01`): 06b's `state/rows.json` copied; sha256 `aa523d28…`, equal.
3. **The store** (`02`–`05`, `19`, `20`): the scratch copy's five files equal
   `copy_files_certified.txt` (EQUAL) before X0 and again after X0 and the smoke test, with no
   journal, `-wal` or `-shm` file. `fingerprint_copy.py` gives `0e32f24a…`, problems none, at $g$
   and at $p$.
4. **06b's checkpoint** (`06`, `21`): `f934ea24…` before and after preparation.
5. **06b's pool** (`07`, `explore06b_pool.py`): 06b's X2-type calls (unconverged, no region at
   depth 20, none phase-limited, finite abserr) are 4,151 group and 16,274 analytic-block calls at
   1e-7 outside the 720 stationary-point rows (719 and 489 inside them); 4,474 and 17,968 at 1e-8
   (794 and 1,063 inside). X1's first stratum has 4,757 candidates before the run (1e-7 group calls,
   X2-type in 06b, no sign change in 06a's scan), in all 17 (band, k) strata. X0's determinism unit
   by rule: `1e-7-0-14164`.
6. **Gate X0** (`08`, `09`, `14`): **holds**, all three clauses.
   - **Clause 1** (`14`): `git diff --stat 92bc62f fef401c -- . ':!docs' ':!prompts'` shows
     `CLAUDE.md` only. The harness imported 102 modules from $p$ in X0's parent workers
     (`state/X0/p_tree_modules.json`); none needs the AST test.
   - **Clause 2** (`09`): at $p$, `p-1e-7-2-38302` (`low-G+r-1`), `p-1e-7-2-2095` (`ctrl-low-1`) and
     `p-1e-8-0-1971` are identical to 06b's `b-` units in every value 06b recorded, on 06b's keys:
     group calls and analytic-block calls, the slim and full results, `GkPolicy`, the scan, all
     region arrays (group and analytic), and W3's numbers. Totals `0x1.5e6b3999d1ab2p-101`,
     `-0x1.e3272e29a4c5dp-102`, `0x1.6b3058bb2e6a9p-66`. Dispatcher counts equal (descriptive).
   - **Clause 3** (`09`): `g-1e-7-0-14164` in two fresh workers, identical in every computed value
     the worker records (06c's per-call keys, budget counters and solves by purpose included; every
     region array; W3; the captured call's value, abserr, regions, depth, converged).
   - No worker initialised Ray.
7. **Development checks** (`10`–`13`, `15`): `dev_worker_functions.py` on 06c's tests 5 and 12d at
   $g$ (the deep line reproduces the leaf's residual and floor bit for bit; the trust numbers match);
   a capture of X0's g call 3 by `x1_capture.py` (matches the record bit for bit) and
   `x1_reference.py` on it; the driver smoke test, 2 rows × 2 rtols, 4 units done.
8. **Predictions** (`16`): `state/predictions.md`, sha256 `2672dd5c…`, mtime
   2026-10-29T12:58:35Z. Part A is the prompt's lines 363–392 byte for byte; Part B is X1's
   selection rule and the readings.
9. **Scripts** (`17`, `18`): every run script equals those X0 and the smoke test ran.
10. **Pre-launch** (`24`–`30`), after the coordinator reported R17/R18 done and the lock released:
    trees clean, 30 runs all finished, lock free, `ray-processes` lists none, the copy EQUAL to its
    certification with no journal file, every script equal, Part A byte-equal.
11. **Launch** (`31`) and **one check** (`32`): run `levin-phase-floor-06c-census-20261029T130214`,
    sh 1752, driver 1756, `running`, stderr empty, no checkpoint line yet at 13:02:20Z.
    The coordinator gave no ruling on D1: X1 is scored by its rule as written, every dropped call
    reported and counted.

## After the run (`out/cmd/33`–`52`)

- **Status** (`33`): `done`, 18,560 of 18,560, exit 0, stderr empty. Every record is at `e5764a6`
  with empty status, carries `run_census.py` `ae11f6bc…`, and lists no module from a foreign root.
  All 16 workers reported `ray_initialized` false.
- **The store** (`34`–`38`): EQUAL to the certification after the run and after the fingerprint,
  no journal; `0e32f24a…`, problems none. **06b's checkpoint** (`39`): `f934ea24…`.
- **Over-budget pass:** none run; no unit is over budget by the dispatcher's test. Nine units took
  longer than 300 s (deviation 22).
- **X1** (`40`–`43`): `select` (24 drawn, no shortfall), `x1_capture.py` (24 of 24 captures equal
  the census call bit for bit), `x1_reference.py`, and a descriptive cross-check on three calls.
- **Scores and descriptive items** (`44`, `50`), `convergence_kinds_06a.py --tree g` (`45`), X4's
  call-level breakdown and descriptive references (`46`–`49`).
- **Final state** (`51`, `52`): trees clean, 31 runs all finished, `state/predictions.md` unchanged
  (`2672dd5c…`, mtime 12:58:35Z).

## Predictions and verdicts

`state/predictions.md` (sha256 `2672dd5c…`, mtime 12:58:35Z, before the launch at 13:02:14Z) holds
§5's X1–X5 unchanged. They are copied here unchanged from the prompt:

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
"Moved" means the call's value, abserr, region count or maximum depth differs from 06b's.
- **X1 — honest where it moved.**
  - **The sample:** 24 calls at `1e-7` that moved, with no sign change of $\theta'$ in 06a's scan
    (`spscan.py`): 12 that 06b reported unconverged with no region at `depth_max`, none
    phase-limited and a finite abserr (or all such, if fewer), and the rest stratified by band
    (low, mid) and by $k$. The selection rule is fixed in `state/predictions.md` before the run.
  - **The reference:** W2's, Levin forced to $2^9$ and to $2^{10}$ equal pieces, with each piece's
    own floor reported. The two must agree to within a hundredth of $g$'s `abserr`; a call where
    they do not is reported, and dropped from the rule.
  - **The rule:** $|{\rm value}_g - {\rm ref}| \le$ `abserr`$_g$ for every call left in the sample.
- **X2 — budget misses converge.**
  - **The rule:** every Levin call that 06b reported unconverged, with no region at `depth_max`, no
    phase-limited region and a finite abserr, is `converged` under $g$, outside 06a's stationary-
    point rows.
  - **Holds** if true at both rtols.
- **X3 — no infinite declared error.**
  - **The rule:** no call under $g$ declares an infinite abserr unless one of its leaves is at
    `depth_max`.
  - **Holds** if true at both rtols.
- **X4 — the rows agree.**
  - **The rule:** outside 06a's 720 stationary-point rows, every row with finite `total_abserr`
    under both has $|{\rm total}_g - {\rm total}_{06b}| \le {\rm total\_abserr}_{06b} + {\rm
    total\_abserr}_g$.
  - **Holds** if true at both rtols.
- **X5 — the cap.**
  - **The rule:** at both rtols, no call that moved reaches `depth_max`. An over-budget unit counts
    as false.
  - A failure is a finding for the user, not a fault. It bears on G2's L2 (Decisions, **G2 L2,
    decided**).

| | Number | Verdict |
|---|---|---|
| **X0** | see "Before the launch", 6 | holds |
| **X1** | 24 drawn (12 category 1, 6 low + 6 mid category 2; `state/x1_selection.json`). Every capture and every re-run from the reloaded inputs equals the census call bit for bit. 1 dropped: `g-1e-7-3-38436` call 5, $\|S_9 - S_{10}\|$ = 0.0215 `abserr`$_g$ (its error 0.479 `abserr`$_g$, descriptive). 23 left: $\|{\rm value}_g - S_{10}\| \le$ `abserr`$_g$ on 22, ratio 0.002–0.99 (median 0.22); **fails on `g-1e-7-0-14127` call 3 (category 2, low, `0:202`), 1.375**. (`42`, `44`) | **fails** |
| **X2** | 1e-7: 20,425 X2-type 06b calls outside the 720 rows; 53 unconverged under g (49 untrusted leaf at the cap, 2 at depth 20 trusted, 2 floor-limited). 1e-8: 22,442; 127 unconverged (63, 27, 37). No unit over budget. (`44`) | **fails** |
| **X3** | 0 calls with an infinite abserr, 0 with NaN, at both rtols. (`44`) | **holds** |
| **X4** | 1e-7: 8,092 rows scored (468 with a non-finite `total_abserr` in 06b or g, counted apart), 263 violate, worst 29.9×. 1e-8: 8,092 scored, 302 violate, worst 71.5×. Evaluated exactly. (`44`, `47`) | **fails** |
| **X5** | 1e-7: 124,543 calls moved; 109 reach depth 20 (14 group, 95 analytic; 8 converged; 99 with an untrusted leaf). 1e-8: 125,860 moved; 153 reach depth 20 (12, 141; 7; 117). No unit over budget by the dispatcher; nine ran past 300 s (deviation 22), which does not change the verdict. (`44`) | **fails** |

**X1, examined further** (descriptive, after the score; `43`):
- On the failing call, $S_{10}$, $S_{11}$ and $S_{12}$ agree to $2.6\times10^{-5}$ of `abserr`$_g$. The
  adaptive driver at `rtol` $10^{-11}$ returns g's value to 16 digits, 1.37 `abserr`$_g$ from
  $S_{12}$. g's seven leaves are all trusted or floor-limited; the leaf $[14.72, 28.12]$ at depth 1
  ($r = 0.079$) carries the declared error, $1.74\times10^{-37}$, against a true $2.39\times10^{-37}$.
  An error common to a leaf and its descendants: the trust check cannot see it (prompt §8).
  06b's abserr for the call was $+\infty$.
- Against each call's own request max(atol, rtol|value|), all 24 are well inside it (the failing
  call's error is $1.3\times10^{-4}$ of it).
- 06b's value against the same reference (descriptive): `err_06b/abserr_06b` exceeds 1 on 5 of the
  23 calls with a finite 06b abserr (1.04–5.64).

**X4, examined further** (descriptive, after the score; `46`–`49`): in all 263 (302) violating rows
some group call violates the call-level test $|v_g - v_{06b}| > e_g + e_{06b}$, and g reports it
converged. The violating rows are mostly seam (179, 190) and mid (80, 106). On the most-violating call
of the two worst rows at each rtol, referenced as X1 is: g's error is 0.45, 0.61, 0.31 and 0.11 of
`abserr`$_g$; 06b's is 121, 116, 445 and 332 of `abserr`$_{06b}$. On these four, the disagreement is
06b's declared error, `[06b-moved-calls-declare-less-than-their-error]`.

## Descriptive items (`out/cmd/44` unless noted)

- **Convergence.**
  - `total_converged` false on 13 rows under g against 2,349 under 06b at 1e-7; 45 against 2,388 at
    1e-8.
  - Unconverged group calls 15 (06b 5,788) and analytic-block calls 89 (17,735) at 1e-7; 49 (6,186)
    and 151 (19,851) at 1e-8.
  - `convergence_kinds_06a.py --tree g` (`45`, `out/convergence_kinds_g.txt`), group calls: at 1e-7
    cap 13, phase-limited 2; at 1e-8 phase-limited 37, cap 12. Resolution and bound-only: 0.
- **Depth** (row depth over every Levin call; histograms in `44`). At 1e-7, rows at depth 20: 63
  under g, 0 under 06b; at ≥ 18: 131 against 1. At 1e-8: 81 against 3 at depth 20. Shallow rows
  move deeper: 06b's mode at depth 2 and 7 becomes a broad spread over 4–14.
- **Cost.**
  - Solves: 7.26 M against 0.88 M at 1e-7 (×8.2), 12.44 M against 1.48 M at 1e-8 (×8.4). Regions:
    0.97 M against 0.29 M, and 2.09 M against 0.47 M.
  - Wall time 2 h 30 min against 51 min; unit walls summed 3.21 h + 5.77 h against 0.69 h + 0.94 h.
    The largest unit is 426 s at 1e-7 and 1,185 s at 1e-8. Cost is not an argument (README §5 rule
    4).
- **Where the budget went** (per call, the largest of residual, floor and truncation in $E$): group
  calls at 1e-7: residual 31,159, floor 6,166, truncation 0; analytic 66,540, 40,854 and **22**. At
  1e-8: group 31,163 / 6,162 / 0; analytic 59,991 / 47,425 / 0. Summed over calls the residual
  dominates by about three orders.
- **The trust check** (`44`, `50`):
  - Leaves found untrusted during the calls: 170,761 (group) and 528,991 (analytic) at 1e-7;
    189,672 and 686,328 at 1e-8. Returned untrusted (at depth 20): 110 and 1,005; 122 and 1,284.
  - Its cost: the quarters and the fallback halves it solved are 86 % (group) and 80 % (analytic) of
    all solves at 1e-7, 78 % and 55 % at 1e-8.
  - Its widening, abserr / $E$ unwidened per call: median 1, p90 1.7, p99 11–13, max $3.5\times10^6$.
    No call is unconverged only because of the widening.
  - $r$ over every returned leaf, by branch and band, is tabulated in `50`. The trust check was taken
    on 633 k Levin and 99 k fallback leaves at 1e-7 (760 k and 157 k at 1e-8); floor-limited leaves
    keep the $r$ measured before they were exempted, so the "≥ 1" column counts them too.
  - The running test and the returned flag never disagreed; no resum rejection.
- **Flat residuals** (`44`, `out/deep_lines_06c.json`): 262 calls reach depth 20 (109 at 1e-7, 153
  at 1e-8); every recomputed line reproduces its leaf's residual and floor bit for bit. On 217 the
  residual falls by less than a factor 10 over the line's last five levels (the reading fixed in
  `analyse06c.py`). Along the deep lines $(e_L + e_R)/e_P$ is from depth 4–5 on often exactly 2.0 and mostly 1–3:
  bisection does not lower the line's error, and often doubles it, the signature of an error at a
  point that both halves carry.
- **06b's three depth-20 calls, by name:** `2-13964` call 5 (J1) converges at depth 5 with 71
  solves (06b: 31,885 regions, 127,539 solves); `2-1704` call 16 (Y1) at depth 2 with 23; `3-34987`
  call 9 (J2) at depth 1 with 15.
- **Totals** (outside the 720 rows), by (band, k), in `44`: median $|\Delta{\rm total}|$ /
  `total_abserr`$_{06b}$ 0–0.78; relative moves mostly below $10^{-3}$; the largest are 0.29 (mid `2:302`),
  0.15 (seam `2:302`), 0.04 (low `2:302`) and 0.0145 (seam `1:301`). `total_abserr`$_g$/`total_abserr`$_{06b}$ medians 0.006–2.4.
- **Stationary-point rows** (07's; not scored): totals moved on 711 (1e-7) and 710 (1e-8) of the 720;
  by more than `total_abserr`$_{06b}$ on 160 and 172, by up to $9.4\times10^4$ and
  $2.6\times10^5$ of it.

## Deviations from the prompt (numbered and classified)

1. **`prov.py`:** trees `p` (`fef401c`) and `g` (`e5764a6`), 06b's run and records, `WIDER` at
   `var/levin-phase-floor-06c/census`. STRUCTURALLY REQUIRED.
2. **`worker.py`, tree evidence:** whether `levin_quadrature` has the trust check, 06d's
   Clenshaw–Curtis floor and `_local_atol`, and its default depth_max. STRUCTURALLY REQUIRED (the
   brief: evidence of which tree a unit ran).
3. **`worker.py`, per call** (`_summarise_call_06c`): converged, `num_untrusted_regions`, the
   result's error breakdown and solve counts, depth_max and leaves at it, abserr finite, the share
   of $E$ from residual, floor and truncation, $E$ without the trust check's widening, leaves by
   branch and by trust, $r$'s maximum. STRUCTURALLY REQUIRED (prompt §5: "what X1–X5 need
   recorded per call"; the brief adds the descriptive items).
4. **`worker.py`, the budget loop's counters** from its one DEBUG record per call, captured by a
   logging filter on the module's logger. The logger is opened to DEBUG; the filter passes only
   records at or above the level that reached the module's handlers before (WARNING; only a
   `NullHandler` is attached), so what the module emits is unchanged. IMPLEMENTATION CHOICE: the
   result dict does not return the leaves found untrusted during the call, which the trust
   check's descriptive item asks for.
5. **`worker.py`, solves by purpose:** a pass-through at `_adaptive_levin_subregion` reads its
   caller's `description` (and, for a fallback leaf's halves, who asked for them). Installed for the
   timed section only, like 06b's premise counter, so its small overhead falls inside the 300 s
   budget. IMPLEMENTATION CHOICE (the cost of the check).
6. **`worker.py`, region arrays** (`_region_arrays_06c`): 06b's eight plus truncation, total_err,
   delta1, delta2, convergence_ratio, trusted, declared_resolution. STRUCTURALLY REQUIRED (the
   brief).
7. **`worker.py`, the deepest line** (`_deep_line`) of every call that reaches depth_max,
   recomputed after the timed section with the module's own solver. IMPLEMENTATION CHOICE: the
   flat-residual item needs ancestors the result does not return. Checked against the returned leaf
   on each line, and on two test problems (`10`).
8. **X1's inputs captured after the run, not during it:** the run captures nothing
   (`run_census.py`); `x1_capture.py` re-runs each selected call's unit at $g$ and keeps the capture
   only if it equals the census call bit for bit. IMPLEMENTATION CHOICE: capturing every candidate
   in-run, as 06b's W2 did for 975 calls, costs about 1.35 MB per call over tens of thousands of
   candidates; X0's clause 3 is the ground for expecting a match. The captured inputs are therefore
   those of a re-run of the call at $g$, not of the census call itself; they stand for it only where
   the bit-for-bit match holds, and a call whose capture does not match is reported and its X1
   score is undecided. Recorded as a deviation on the coordinator's instruction before the launch.
9. **`run_census.py`:** tree `g` by default, no capture list. STRUCTURALLY REQUIRED.
10. **`launch.py`:** prompt `06c`, purpose and scope. STRUCTURALLY REQUIRED.
11. **`harness.py`:** `job_name="levin-phase-floor-06c"`, as the brief asks; datastorekit 0.5.0
    stores it and reads it nowhere (`22`). IMPLEMENTATION CHOICE.
12. **06b's W3 evidence and 06a's scan are kept** in each 1e-7 unit, unchanged, though no 06c
    prediction scores W3: §5 allows changes only of the listed kinds. IMPLEMENTATION CHOICE.
13. **New `x0.py` and `x0_clause1.py`:** the determinism row by rule (`07`); one parent worker per
    unit; clause 1's import list from X0's parent workers (the harness's run at the parent);
    clause 2 on every value 06b recorded, analytic blocks and W3 included; clause 1 with no
    test-module exemption, as the prompt words it. IMPLEMENTATION CHOICE.
14. **New `explore06b_pool.py`, `dev_worker_functions.py`, `x1_capture.py`; `x1_reference.py`**
    (06b's `w2_reference.py`, changed for X1; prompt §5 allows "the reference for X1").
    `x1_reference.py` STRUCTURALLY REQUIRED; the rest IMPLEMENTATION CHOICE.
15. **`out/cmd/11` stopped** on reading back the development capture (`ModuleNotFoundError`): the
    record must be unpickled with `PYTHONPATH` the tree $g$, as 06b's deviation 11. The worker had
    finished; `x1_capture.py` now refuses to run without that `PYTHONPATH`, and the capture was
    re-run as `12`. UNINTENDED DRIFT.
16. **`out/cmd/17` stopped** on a syntax error that my edit of `scripts_equal.py` introduced; fixed,
    re-run as `18`. UNINTENDED DRIFT.
17. **`check_predictions.py` and `scripts_equal.py`** adapted to 06c's prompt lines and X0's
    directory. IMPLEMENTATION CHOICE.
18. **X1's reading** (`state/predictions.md` Part B): the first stratum is drawn from all three
    bands (the prompt's "(low, mid)" is read as qualifying "the rest"); the rest excludes the first
    stratum; no replacement is drawn for a dropped call. **X3:** a NaN abserr is reported beside
    the verdict. **X4:** the inequality is evaluated exactly on the recorded doubles.
    IMPLEMENTATION CHOICE.
19. **Scripts are not `black`-formatted:** they are records of the run. IMPLEMENTATION CHOICE.
20. **New `analyse06c.py`** (select, report, trust_r), **`x1_crosscheck.py`** and **`x4_detail.py`**.
    The cross-check (`43`) and X4's call-level breakdown and references (`46`–`49`) are descriptive,
    run after the scores. IMPLEMENTATION CHOICE.
21. **`out/cmd/46` picked X4's descriptive calls by the largest absolute move**, which across calls
    of different scale named calls that do not violate; `x4_detail.py` was changed to pick each
    row's most-violating call and re-run as `47`. Nothing was scored from `46`. UNINTENDED DRIFT.
22. **Nine units ran longer than 300 s** (426–1,185 s: `g-1e-7-0-38939`, `g-1e-8-0-38939`,
    `-1-35477`, `-1-38737`, `-2-1574`, `-2-1578`, `-2-1605`, `-2-38298`, `-3-34271`) and are `done`:
    06b's dispatcher raises only at the next `theta_mod_2pi` entry past the deadline, and these
    units' long Levin calls made none until they finished. Under the dispatcher's test no unit is
    over budget and no pass was run. Read as over budget in the run, 06a's pass (one re-run with
    3,600 s; nine is fewer than twelve) would admit all nine, whose full runs took at most 1,185 s,
    and the new code is deterministic (X0). No verdict depends on the reading: X5 fails on its
    depth-20 calls either way, and X2–X4 do not use it once the pass is done. IMPLEMENTATION CHOICE
    (no pass run).
23. **`out/cmd/51` printed two errors** from `stat` with BSD flags on a GNU `stat` (06b's deviation 24
    again); the mtime is recorded by `52` instead. UNINTENDED DRIFT.
24. **X1's inputs were captured after the run** (deviation 8), by re-running each selected call's
    unit at g: 24 of 24 captures equal the census call bit for bit (`41`). The coordinator asked
    that this be recorded as the deviation it is: the reference is computed from a re-run's inputs,
    not from the census call's own. IMPLEMENTATION CHOICE.

Deviations 15, 16, 21 and 23 are UNINTENDED DRIFT. None of them touched a store, a tree, or any
record outside `var/levin-phase-floor-06c/census/` and the run directory.

## Verification performed
- **The trees:** g at `e5764a6`, `status --short --ignored` empty at X0, at the launch, on all 18,560
  records and at the end (`51`); p at `fef401c`, likewise. The main checkout is at `28ebaf0`,
  status empty, throughout.
- **The copy:** byte-identical to `copy_files_certified.txt` before X0, after X0, at the launch,
  after the run and after the fingerprint; `0e32f24a…` at g and p before, at g after.
- **06b's run was only read:** its checkpoint is `f934ea24…` before, after preparation and after the
  run (`06`, `21`, `39`).
- **The run:** 18,560 records, every unit `done`, every record's pickle matched its SHA-256 where
  read (`44`, `45`, `50`, all 18,560), `stderr.log` empty, every worker `ray_initialized` false.
- **X1's inputs** equal the census calls bit for bit; every re-run from reloaded inputs reproduces
  value and abserr.
- **The deep lines** reproduce their leaves bit for bit (262 of 262).

## Observations not acted on

All are in `issues.md`:
- **D1 (preparation): X1's reference might drop many calls.** It did not: one of 24 was dropped.
  The coordinator gave no ruling; X1 was scored as written.
- **C1:** X1's failure: one call 1.375× its declared error, an error common to a leaf and its
  descendants.
- **C2:** X2's failure: 53 and 127 calls chased to depth 20 or floor-limited.
- **C3:** X4's failure, which on the calls checked is 06b's understated error, not g's.
- **C4:** X5's failure and the cost (×8 solves); 217 of 262 depth-20 lines are flat.
- **C5:** the census budget is not enforced inside a long Levin call (deviation 22).

## State handed on
- **Nothing is running.** The run is `done`. No lock was taken. Nothing was written in either
  worktree or the main checkout outside `var/levin-phase-floor-06c/census/` and the run directory.
- **The scratch copy is unchanged** (`0e32f24a…`), and is still kept for the user to delete.
- **The run directory** `var/runs/levin-phase-floor-06c-census-20261029T130214/` (1.3 GB) is kept in
  full: checkpoint, region pickles, events, worker logs, for 07's census.
- **The header issues** of 06c and 06d close "if 06c's §4 holds and X0–X4 hold; otherwise ask the
  user" (the dispatch brief, B.6): X1, X2 and X4 fail, so **the user is to be asked**.
- **For the orchestrator:** copy `log.md`, `verification.md`, `issues.md`, `scripts/`, `out/` and
  `state/` to `docs/levin-phase-floor/06c/census/`. Leave out `out/cmd/45` (3.1 MB; its summary is
  `out/convergence_kinds_g.txt`), `out/deep_lines_06c.json` if over 2 MB, and the pickles under
  `state/X0/`, `state/X1capture/`, `state/X4capture/`, `state/dev/` and `state/smoke/` (about 50 MB
  of captured inputs), listing them as kept on the Mac Studio.
