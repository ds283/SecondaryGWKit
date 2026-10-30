# levin-phase-floor 06c Part B — the orchestrator's review (Mac Studio)

**Verdict: pass.** All ten review checks pass, and no prompt 06c §7 stop condition was met. **But
§4's Y1 fails, and X1, X2, X4 and X5 fail.** So under the orchestrator note's §B.6, the header
issues of 06c and 06d **do not close without the user's word**: *"close … if 06c's §4 holds and
X0–X4 hold; otherwise ask the user."* The prediction failures are findings, not faults (§5 X5;
README §5).

Written 2026-10-29 on mps026298 by the Mac Studio orchestrator of 06c Part B, which measures 06c's
code `ad9a5f4` and 06d's `3d70db9` together. I wrote no repository code and committed nothing. Every
command named below was run by me. Its output is in `review/checks/`, beside my scratch readers;
the agents' briefs are in `review/`, and the pre-flight records in `review/preflight/`.

| | |
|---|---|
| Trees | `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-fef401c` (parent, `fef401c`, R17 and X0's reproduction) · `…-lpf06c-e5764a6` (g, `e5764a6` = `ad9a5f4`+`3d70db9`, R18 and the census) · main checkout `28ebaf0` (venv, stores, `var/`, registry; ran nothing) |
| Agents | rehearsal agent (§4) · census agent (§5), each a fresh Opus agent briefed by `review/dispatch_rehearsal.md` (`2be08067…`) and `review/dispatch_census.md` (`df2df8d7…`) |
| Marker | `marker` touched 2026-10-29T12:35:36Z, before any agent ran |
| Runs | `levin-phase-floor-06c-r17-parent-20261029T123720` (12:37–12:40Z) · `levin-phase-floor-06c-r18-global-20261029T124058` (12:41–12:45Z) · `levin-phase-floor-06c-census-20261029T130214` (13:02:14–15:32:44Z) |

## The results in one place

**R17/R18 fingerprints:** both `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`,
with no problems. I re-took both, and their listings are identical.

**Value comparison (R17 vs R18):**
- **Outside `QuadSourceIntegral`:** 7 value tables, 360,462 rows, bit-identical. The only other
  differences are timing, `label`, `parents_digest` and `rho_reuse`. Nothing is unclassified and
  no row is in one store only.
- **Inside it,** 864 rows:
  - `total` bit-identical in 295;
  - `total_converged` 516 → 863;
  - regions 13,577 → 34,892;
  - maximum row depth 12 → 20.
- **Y1 fails: 30 of the 838 rows scored violate** |T18−T17| ≤ e17+e18. Another 26 rows have a
  non-finite e in R17 only and are scored apart.
  - Twelve violate by 336–16,763×, all equilateral (k=q=r at 1e5, 3.16e5 and 1e6). In each, the
    `G+q+r` call of the all-oscillatory sub-interval moves by 0.03–1%, and both codes call it
    converged in 11 of the 12.
  - Worst row: k=q=r=1e5, z=5236.55. T17 `-0x1.5ff6eb7403dbcp-55`, T18 `-0x1.60d489de69404p-55`,
    e17 `0x1.1eff05d8001dfp-79`, e18 `0x1.21b46d538f125p-78`. A relative move of 2.5×10⁻³ at rtol
    1e-7.
  - The other 18 violate by 1.007–20.4×.

**X0 holds**, all three clauses. I re-checked each independently (check 6):
- `git diff --stat 92bc62f fef401c -- . ':!docs' ':!prompts'` shows `CLAUDE.md` only.
- At the parent, `low-G+r-1` and `ctrl-low-1` at 1e-7 and `0-1971` at 1e-8 equal 06b's
  checkpoint call for call in `float.hex`.
- The two g runs of `g-1e-7-0-14164` are identical.

**X1–X5** (re-scored by my own readers; every number agrees with the census agent's):

| | rtol 1e-7 | rtol 1e-8 | verdict |
|---|---|---|---|
| **X1** honest where it moved | 24 drawn; 1 dropped (\|S9−S10\| = 0.0215 abserr); 22 of the remaining 23 within abserr (0.002–0.99); **`g-1e-7-0-14127` call 3 at 1.375×** | — | **fails** |
| **X2** budget misses converge | 20,425 X2-type calls, **53** unconverged (49 untrusted leaf at depth 20, 2 depth 20 all trusted, 2 below depth 20) | 22,442, **127** (63, 27, 37) | **fails** |
| **X3** no infinite declared error | 0 non-finite abserr, 0 NaN | 0, 0 | holds |
| **X4** the rows agree | 8,092 scored (468 non-finite apart), **263** violate, worst 29.9× | 8,092 (468), **302**, worst 71.5× | **fails** |
| **X5** the cap | 124,543 moved calls, **109** at depth 20 (14 group, 95 analytic) | 125,860, **153** (12, 141); largest `g-1e-8-0-38939` call 5, 74,941 regions | **fails** |

**The trust check** (descriptive, the census agent's `out/cmd/44`, `50`):

| | 1e-7 | 1e-8 |
|---|---|---|
| Leaves found untrusted during calls (group / analytic) | 170,761 / 528,991 | 189,672 / 686,328 |
| Untrusted leaves returned, all at depth 20 (group / analytic) | 110 / 1,005 | 122 / 1,284 |
| The check's solves as a share of all solves (group / analytic) | 86% / 80% | 78% / 55% |

- **The widening's effect on abserr:** median 1, p90 1.7, p99 11–13, maximum 3.5×10⁶.
- No call is unconverged because of the widening alone.

**Depth distribution against 06b** (rows by maximum depth over their Levin calls; my
`checks/check9_depth.txt`):

| depth | 06b 1e-7 | g 1e-7 | 06b 1e-8 | g 1e-8 |
|---|---|---|---|---|
| no Levin call | 2,215 | 2,215 | 2,215 | 2,215 |
| 0 | 538 | 259 | 503 | 253 |
| 1–4 | 3,062 | 1,119 | 2,770 | 1,040 |
| 5–9 | 3,372 | 4,006 | 3,594 | 3,807 |
| 10–14 | 92 | 1,316 | 172 | 1,515 |
| 15–19 | 1 | 302 | 23 | 369 |
| **20** | **0** | **63** | **3** | **81** |

**Cost:**

| | 1e-7 | 1e-8 |
|---|---|---|
| Regions | ×3.33 | ×4.48 |
| Evaluations (solves) | ×8.25 | ×8.42 |
| Unit wall, summed | 0.69 h → 3.21 h | 0.94 h → 5.77 h |

The census took 2 h 30 min against 06b's 51 min. 06b's three depth-20 calls now converge at depth
1–5, with 15–71 solves.

**Over-budget units:** none by the dispatcher's test, so 06a's pass was not run. **But nine units
ran past 300 s**: 314–1,185 s, one at 1e-7 and eight at 1e-8, all `done`. 06b's maximum was
159 s. The dispatcher raises only at the next `theta_mod_2pi` entry after the deadline, which a
long Levin call never makes (census deviation 22, issue C5). No verdict depends on it:
- X5 fails anyway;
- had the nine gone to 06a's 3,600 s pass, all would finish, and by X0's determinism clause their
  records would equal these, so X2–X4 stand.

## Findings for the user (beyond the verdicts)

1. **The census agent's explanation of X1's failure is not right.** It says the error is "shared
   by a leaf and all its descendants", because the driver at rtol 1e-11 returned g's value again.
   - But the call's atol (1.86×10⁻³³) dominates rtol·|V| (2.5×10⁻³⁷, and ~10⁻⁴¹ at rtol 1e-11).
     Re-running at rtol 1e-11 alone asks the driver for nothing tighter.
   - With the request actually tightened (`checks/check9_x1_tight.out`), the driver refines to the
     reference:
     - atol 1e-42, rtol 1e-9: 48 regions, |v−S10| = 2.9×10⁻⁴ abserr_g;
     - atol 1e-45, rtol 1e-11: 1,378 regions, 2.1×10⁻⁵.
   - So the error is reachable by refinement. **What fails is the 7-leaf answer's declared error,
     1.375× short, which the trust check passed (0 untrusted).** Its error is 1.3×10⁻⁴ of the
     call's request.
   - Issue C1's wording should change before it reaches the board.
2. **Y1's 12 large violations are unresolved.** Both codes call the `G+q+r` call converged, and they
   disagree by up to 1%, so at least one of them is converged and dishonest by 10³–10⁵×. No
   reference was computed for these calls, so it is not known which. The census's X4 references
   (four calls) found 06b's declared error the understated one, but none of those calls is of this
   class. **This is the strongest candidate in Part B for 06d's "near-singular converged-dishonest"
   class.** Neither agent addressed that class directly, although both briefs asked them to.
3. **06d's node term does not bind on the deep lines.** On the last five levels of the 262 depth-20
   lines (`checks/check9_deep_lines.txt`):
   - the residual dominates e in 1,179 of 1,310 levels, the floor in 81 and truncation in 50;
   - the residual sits a median 2×10⁴ above the floor (p10 861);
   - its mantissa carries a median of only ~24 significant bits;
   - (e_L+e_R)/e_P is exactly 2.0 on 27 levels.

   These lines are chased by a residual that bisection does not reduce, and the floor is well below
   it. The node term's worst-case overstatement (06d, Observations 2) is not what drives them. The
   37 calls at 1e-8 that end unconverged below depth 20 (X2's third kind) are where a floor could
   bind. They were not examined further.
4. **The trust check is most of the cost:** 55–86% of all solves. Cost is not an argument (README §5
   rule 4); it is recorded for G2's L2.

## Check 1 — Scope: **pass**

`checks/check1.txt` holds each tree's HEAD and `git --no-optional-locks status --short`:
- the main checkout is `28ebaf0`, `e5764a6` is the g worktree and `fef401c` the parent;
- all three are clean (0 lines).

`find <worktree> -newer marker -not -path '*/__pycache__/*' -not -path '*/.git/*'` lists nothing
in either worktree. Outside `var/`, the main checkout lists only `var` and `.git`.

The `.git` entries are the directories `.git/worktrees/SecondaryGWKit-lpf06c-{fef401c,e5764a6}`:
- their own mtimes are 12:40:08Z and 12:45:47Z, the instants the R17 and R18 drivers took their
  fingerprints (`taken.when`);
- no file inside either changed (`index` is 12:32, before the marker);
- this is the fingerprinter's `git status` probe making and removing a lock file there. Not an
  agent's write.

`find var -newer marker -type f`, grouped (`checks/check1.txt`), finds only allowed paths:
- the stores `var/datastores/levin-phase-floor-06c-r1{7,8}*.sqlite` (5 files each);
- `var/levin-phase-floor-06c/` (3,512 files);
- the run directories of R17 (4 files), R18 (4) and the census (149);
- the two launch logs.

The protected set is unchanged:
- `LC_ALL=C shasum -a 256 -c review/preflight/protected_before.sha256` gives 14,608 of 14,608 OK
  (`checks/check1_protected.txt.gz`);
- it covers `var/levin-phase-floor-0*` (including A3 v2's scratch copy and 06b's records),
  `var/runs/levin-phase-floor-*`, `var/datastores/handover-A3-v2-lambdacdm*` and
  `var/datastores/levin-phase-floor-*` as they were at pre-flight.

## Check 2 — Runs: **pass**

`./venv/bin/python -m runregistrykit list` (`checks/check2_registry.txt`):
- the three 06c runs are `done`, the census at 18,560/18,560;
- nothing is `running` or stale.

The manifests (`checks/check2_manifests.txt`) all carry campaign `levin-phase-floor`, prompt `06c`
and git_dirty false:

| run | git_head | script SHA-256 | exit |
|---|---|---|---|
| R17 | `fef401c` | `5b984c75…` | 0, pid 96791 |
| R18 | `e5764a6` | `b0135218…` (a docstring-only launcher change, rehearsal deviation 1) | 0, pid 97470 |
| census | `28ebaf0` | `run_census.py` `ae11f6bc…` | 0, pid 1756, 18,560 units |

The census manifest's `git_head` is the main checkout's, as in 06b's (`28ebaf0` there too), since
its script lives under `var/`. Each record carries its tree (check 7).

## Check 3 — Fingerprints: **pass**

`runregistrykit store fingerprint <primary>`, without `--write`, run from the g worktree with
`PYTHONPATH=<g>` (`checks/check3_r17.json`, `check3_r18.json`):
- **R17:** `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, problems none.
- **R18:** the same, problems none. The two listings are identical below the header.

Both stores' SHA-256s are unchanged across it (`check3_*.before.sha256`).

**My own deviation:** I first ran the command from the main checkout. Python put the cwd ahead of
`PYTHONPATH`, the code at `28ebaf0` refused the stores (`theta_residual` columns it does not
declare) and exited 1 with nothing read. Store hashes were unchanged. I re-ran it from the
worktree, as the rule requires.

## Check 4 — Values: **pass** (Y1 fails as a finding)

- **The diff, re-run.** `python -B -m tools.store_value_diff <R17> <R18> --json`, from the g
  worktree:
  - exit 1 (differences found, as expected);
  - its JSON is byte-identical to the agent's `out/store_value_diff.json` (`57313b01…`,
    `checks/check4_svd.json.sha256`).
- **The classifier.** 05's `classify_non_qsi.py` is unchanged (`26690e61…`, equal to
  `docs/levin-phase-floor/05/rehearsal/scripts/`). Run on my JSON, its output is byte-identical to
  the agent's, with 0 unclassified (`checks/check4_classify.txt`).
- **My own reader,** `checks/y1_reader.py`:
  - it opens every shard of both stores with sqlite3 `mode=ro`;
  - it keys rows by physical identity (k, q, r, z_response, atol, rtol, all by `float.hex`) and
    tests Y1 exactly in `Fraction`.
  - Output (`checks/check4_y1.txt`): 864 rows in each, the same key set; `total` bit-identical 295;
    `total_converged` 516/863; maximum depth 12/20; regions 13,577/34,892; non-finite e in R17
    only 26; **Y1 scored 838, violations 30**.
  - It lists all 30 rows, in hex, matching the agent's `out/qsi_rows.csv`.
- **Spot-check.** The worst row's T17, T18, e17 and e18 are, in hex, the same in my reader and in
  the agent's CSV.
- **"11 of 12 converged in both"** (rehearsal log) refers to the `G+q+r` group call, not the row.
  Several of those rows are R17-unconverged through other calls. Consistent with my rows.
- **No `-wal`, `-shm` or journal file** beside either store.

## Check 5 — Readers and lock: **pass**

Each command was a shell command of its own, with its output appended by `>>`:
- `python -m raypoolkit.rehearsal_lock status --ray-temp-dir /tmp/ray-sgk --ray-address in-process`
  prints `lock: free (/Users/ds283/.raypoolkit/rehearsal-lock does not exist)` and
  `Ray processes of /tmp/ray-sgk: none`;
- `… ray-processes --ray-temp-dir /tmp/ray-sgk` prints nothing, exit 0.

The rehearsal's lock transcript (`rehearsal/state/lock.log`):
- acquire;
- set-pid 96791 (R17), then 97470 (R18);
- release, then status free, ray-processes none, every command exit 0.

**The readers:** 6 of 6 exit 0, with no traceback (`rehearsal/out/readers.txt`,
`readers_check.txt`).
- R18's five files hash identically before and after them. The after-file carries two more
  lines, `journal/wal/shm: none` (rehearsal deviation 8).
- Each reader prints the warning "number of shards read from database (=4) does not match specified
  number of shards (=10)", which is descriptive.

**Rehearsal deviation 3** (the readers call `ray.init` under the lock) comes from **my brief's
wording**, which said no Ray session may be started by anything but R17's and R18's drivers. That
conflicts with §4.4's readers. The agent classified it STRUCTURALLY REQUIRED and asked me to
confirm; the readers are what §4.4 asks for. As 06b's deviation 17, the error is the brief's.

## Check 6 — X0 and the store: **pass**

- **Clause 1:** I re-ran `git --no-optional-locks diff --stat 92bc62f fef401c -- . ':!docs'
  ':!prompts'`. It gives `CLAUDE.md | 8 +++++---`, 1 file.
- **Clauses 2 and 3**, `checks/x0_independent.py` (`checks/check6_x0.txt`):
  - each X0 parent unit's per-call (index, value, abserr, converged, regions, depth), in hex,
    equals 06b's checkpoint unit:
    - `p-1e-7-2-38302` = `b-1e-7-2-38302` (`low-G+r-1`), 7 calls;
    - `p-1e-7-2-2095` = `b-1e-7-2-2095` (`ctrl-low-1`), 7 calls;
    - `p-1e-8-0-1971` = `b-1e-8-0-1971`, 5 calls;
  - the two g runs of `g-1e-7-0-14164` are equal over 7 calls;
  - the `names05` labels were confirmed in 06b's checkpoint.
- **The store:**
  - A3 v2's scratch copy is unchanged against pre-flight, and its five SHA-256s equal the agent's
    certification after the run (`census/out/cmd/35`, `38`: EQUAL).
  - My own `fingerprint_copy.py` at g gives `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`,
    problems none (`checks/check6_copy_fp.txt`).
  - No journal, `-wal` or `-shm` file is beside it.
- **06b's checkpoint** is still `f934ea24…`.

## Check 7 — Trees: **pass**

Every one of the 18,560 census records (`checks/census_rescore.py`, `checks/check9_rescore.txt`):
- carries `tree_head` `e5764a6…`, an empty `tree_status_short`, `script_sha256` `ae11f6bc…` and
  status `done`;
- has an empty `modules.foreign`;
- loads `AdaptiveLevin.levin_quadrature` from
  `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6/AdaptiveLevin/levin_quadrature.py`;
- loads no repository module from outside g.

All 16 workers' `bye` records report `ray_initialized` false, with no foreign module. X0's parent
units load from `…-lpf06c-fef401c` (`census/out/cmd/09`).

## Check 8 — Predictions: **pass**

`state/predictions.md` is `2672dd5c…`, unchanged.

Its mtime is 2026-10-29T12:58:35Z. That is later than X0's last output (`out/cmd/09`, 12:50:22Z)
and earlier than the census's `manifest.json` (13:02:14Z).

Prompt 06c lines 363–392 (§5, "Predictions…" through X5), at `e5764a6`, appear in it verbatim
(`checks/prompt_x1x5.txt`, a substring test). Part B fixes X1's selection mechanically, and the
readings of X2–X5.

**The harness diff** (`census/out/scripts_diff_against_06b_final.txt`) against
`state/scripts_06b_reference/`, which is identical to `docs/levin-phase-floor/06b/census/scripts/`
at `e5764a6`:

| change | deviation |
|---|---|
| `prov.py` | 1 |
| `worker.py` | 2–7 |
| `run_census.py` | 9 |
| `launch.py` | 10 |
| `harness.py` | 11 |
| `x0.py`, `x0_clause1.py` | 13 |
| `explore06b_pool.py`, `dev_worker_functions.py`, `x1_capture.py`, `x1_reference.py` | 14 |
| `check_predictions.py`, `scripts_equal.py` | 17 |
| `analyse06c.py`, `x1_crosscheck.py`, `x4_detail.py` | 20 |

`convergence_kinds.py`, `spscan.py`, `rerun.py`, `dispatch.py` and `rebuild.py` are unchanged.

## Check 9 — Verdicts re-scored: **pass** (each agrees with the agent)

- **X2–X5,** by `checks/census_rescore.py`:
  - my own streaming reader of both checkpoints, with 06a's 720 stationary-point rows from 06a's
    1e-7 scans;
  - it uses the prompt's definitions of "moved", X2-type and depth_max (20), and X4 is exact in
    `Fraction`.
  - The output is in the table above. It agrees with the agent on every count and on the breakdown
    by kind. There are no call label mismatches.
- **X1,** by `checks/x1_ref_mine.py`:
  - modelled on 06b's `w2_ref_mine.py`, run at g, with `PYTHONPATH=<g>:<census scripts>` (the
    pickles name the harness module);
  - it reloads the agent's captured inputs, checking each blob's SHA-256.
  - On three calls (`checks/check9_x1_ref_mine.out`), every re-run equals the census call in hex:

    | call | \|S9−S10\| / abserr_g | \|value_g − S10\| / abserr_g | result |
    |---|---|---|---|
    | `g-1e-7-0-14127` call 3 | 2.5×10⁻⁴ | **1.375** | fails; S11 and S12 within 3×10⁻⁵ of S10 |
    | `g-1e-7-3-38436` call 5 | 0.0215 | 0.479 | dropped, correctly |
    | `g-1e-7-3-39140` call 2 | 1.1×10⁻⁵ | 0.9915 | holds |

  - All match the agent's `out/x1_reference.json`. The tightened-request diagnostic is
    `checks/x1_tight.py`, discussed in finding 1.
- **Depth and cost:** `checks/depth_dist.py` (`check9_depth.txt`).
- **The deep lines:** `checks/deep_lines_look.py` (`check9_deep_lines.txt`).

## Check 10 — Records: **pass**

**Both logs carry README §5.1's headings:** subject and result, What shipped, Predictions and
verdicts, Deviations from the prompt, Verification performed, Observations not acted on, State
handed on.

**Every deviation is classified** (a regex over each numbered item):
- **Rehearsal:** 9 deviations. 1–3 STRUCTURALLY REQUIRED; 4–6 and 9 IMPLEMENTATION CHOICE; 7–8
  UNINTENDED DRIFT.
- **Census:** 24 deviations, each classified.
  - UNINTENDED DRIFT: 15, 16, 21, 23, and the drift part of 24.
  - None of the drift touched a store or a tree.

**Both** directories carry `verification.md` and `issues.md`. The issues proposed are rehearsal N1
and census C1–C5. **C1's diagnosis needs rewording** (finding 1).

**Sizes:**

| directory | size |
|---|---|
| `var/levin-phase-floor-06c/rehearsal/` | 257 MB, of which `extract/` is 177 MB |
| `var/levin-phase-floor-06c/census/` | 63 MB |
| the census run directory | 1.3 GB |
| R17's and R18's run directories | 28 KB each |
| `var/levin-phase-floor-06c/review/` | 3.4 MB |

**For the laptop copy, leave out these, and list them as kept on the Mac Studio:**
- `rehearsal/extract/`;
- `census/out/cmd/45` (3.1 MB);
- the captured-input pickles under `census/state/{X0,X1capture,X4capture,dev,smoke}`;
- the census run directory: checkpoint `b61cb034…`, 775 MB, and its unit pickles, kept in full for
  07's §6;
- the stores `var/datastores/levin-phase-floor-06c-r1{7,8}*`, kept, not retired.

## Noted, not acted on

At pre-flight, idle Python 3.14 multiprocessing workers (pids 20870–20878, 96360, 96471–96478, 13
days old) were present. They were not this project's, and were left alone.

Pre-existing in the registry: `py313-venv-02-reference-20261025T163202` shows `failed`.
