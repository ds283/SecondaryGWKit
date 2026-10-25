# Orchestrator review — levin-phase-floor prompt 04

**Verdict: STOP.** The agent stopped at step 5. The gates E1 and E2 fail as written (prompt §5:
"a gate that fails is a stop, not a verdict"; §7: "E1, E2 or M0 fails"). Nothing was launched.
The agent's question goes to the user verbatim (§4 below). Nothing has been repaired.

- Orchestrated on the Mac Studio (`mps026298`), 2026-10-25, in the main checkout at the dispatch
  SHA `c8ca1559fcc53fdb88998f170257635431fb85ca`. The code ran from the worktree
  `/Users/ds283/Documents/Code/SecondaryGWKit-lpf04-98e68d3`, at `98e68d3`.
- Executing agent: one fresh-context Opus subagent, dispatched 22:33 GMT. It stopped at about
  22:55 GMT.
- The `var/` marker is
  `/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/258741ed-1a2e-409a-96c9-2e41a3ff5091/scratchpad/var-marker`,
  touched 2026-10-25 22:32:59 GMT.

## 1. The Gate (note §1), run by the orchestrator before dispatch, 22:32 GMT

| Item | Result |
|---|---|
| Main checkout `HEAD` | `c8ca1559fcc53fdb88998f170257635431fb85ca`; `git status --short` empty; `refs/remotes/origin/handover-remedial` at the same SHA; `d5f5a2b` is an ancestor |
| Worktree `HEAD` | `98e68d3e1b60e6de4378b7d9aded3a120d6088a9`, detached; `git status --short` empty; `98e68d3` is an ancestor of `HEAD` |
| `runregistrykit list` | 12 runs, "12 finished; 0 with no host recorded"; nothing `running` or stale |
| `rehearsal_lock status` | `lock: free (/Users/ds283/.raypoolkit/rehearsal-lock does not exist)`; `~/.raypoolkit/` is empty; `~/ScienceData/.rehearsal-lock` does not exist |
| `store show` | `problems: none`, digest `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56` |
| 03's run directory | `row_R1.pkl` to `row_R7.pkl`, `checkpoint.jsonl`, `manifest.json`, `status.json`, `stdout.log`, `stderr.log` |
| `var/levin-phase-floor-04/` | did not exist |

Load was 12.6 / 7.5 / 5.6, from system services (`mds_stores`, ChronoSync, OrbStack). 115 GiB
free. **The Gate passed.**

## 2. The stop

From `out/phases_04.txt` (header `# levin-phase-floor prompt 04 step 5: phases_04 …`):

- **E1 FAILS** (line 197). Bound (c), $|{\rm wrap}(\rho^A-\rho^X)| \le 4\varepsilon\Phi^A_i + 10^{-15}$
  for $G$, fails on the lone-`G` sub-interval `sub0` of R1 (45 abscissae of 2000), R2 (160), R4
  (23), R5 (136), R6 (148) and R7 (23). The worst ratio is 1.776, and 1.332 on R5. R3's lone `G`
  passes, at 0.666. Bounds (a) and (b) hold everywhere: (a) has a worst ratio of 1.000, (b) 0.870.
- **E2 FAILS.**
  - Under P every scored group holds, at ratios 0.22–0.34, and the reproduction of 03's medians
    `HOLDS` (line 229).
  - Under A, the R1 lone `G` fails at ratio 3.86 and the R4 lone `G` at 5.45. The agent counts
    5 of 29 groups in all.
  - Under X, the "$\le \varepsilon\sum\Phi^X$" clause fails in most groups, for example ratio 21.4
    on R3 `G-r`. The "$\le 10^{-2}F$" clause holds everywhere.
- **M0** was not reached as a gate of the run. The agent reports route (a) re-proved on R6 at
  `98e68d3`, with M0 bit for bit in 114.7 s (`state/route_a_proof.txt`). I have not re-checked
  this, because the run was not launched.

The size of the failures: the excesses are round-off, $\sim10^{-15}$–$10^{-14}$ rad, against
$F$ = 1.7e-6 to 9.2e-5. The agent attributes them to two round-off terms the bounds omit: the
ulp of A's unreduced value in $(-4\pi, 0]$, and the rounding of $z = {\rm expm1}(x)$. It
supports the second with `out/diag_abscissa_rounding_04.txt`. **I did not adjudicate this.**
Whether the gates should be re-posed is the user's decision.

## 3. Checks that apply at a stop (note §3)

**3.1 Scope** — holds, with one exception, recorded under 3.4 below.
- Main checkout: `HEAD` `c8ca155…`, `git status --short` empty.
- Worktree: `HEAD` `98e68d3…`, `git status --short --ignored` empty.
- `find var -newer <marker> -maxdepth 2` lists `var`, `var/levin-phase-floor-04/…` and
  `var/handover-03/…`. The second is `handover` A3, which the note allows. No store file, no
  sidecar, no run directory.
- `find <worktree> -newer <marker> -not -path '*/__pycache__/*'` lists nothing, and no
  `__pycache__` is newer either.
- No `var/runs/levin-phase-floor-04-*` exists, so nothing was registered.

**3.2 The store** — holds.
- `state/store_files_before.txt` (22:34Z) and `state/store_files_at_stop.txt` (22:53Z) differ only
  in their two header lines. The SHA-256, size and mtime of the primary, the four shards and the
  sidecar are identical.
- `runregistrykit store fingerprint` without `--write`, run by me: digest `0e32f24a2dc2…d56`,
  `problems: none`, "recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00".
- There is no `-wal`, `-shm` or `-journal` beside any store file.
- `ray-processes --ray-temp-dir /tmp/ray-sgk`, run as a command of its own, printed nothing.
  `~/.raypoolkit/` is empty: no lock was taken.

**3.3 Checks 3–8** (the trees, arms, M0, predictions, budget, analysis and run) are not reviewed.
The run was not launched, and `state/predictions.md` was correctly not written, because step 5's
gates did not pass.

**3.4 The records, and a write outside the allowed area.**
- `log.md` and `issues.md` (I1–I5) exist.
- `verification.md` was not written: the prompt stopped before step 10.
- The agent's `log.md` records one **UNINTENDED DRIFT**. A `black` run wrote its cache to
  `~/Library/Caches/black/25.1.0/`, outside `var/levin-phase-floor-04/` and the run directory.
  The note says "A write anywhere else is a stop". I relay it rather than judge it. It touched
  no store file, no run directory and neither tree.

**3.5 For whoever resumes.** `issues.md` I4: `runregistrykit.begin` takes `git_head` from the
script's directory. Recording `98e68d3` therefore needs `script=None` with the launcher's working
directory in the worktree, and that drops the registry's own script hash. This bears on note §3
check 8 and prompt step 7. Nothing was done about it.

## 4. The agent's question, verbatim

> **Question for the user:** E1 (c) and E2 fail only at round-off. The bounds leave out two round-off terms:
> - the ulp of arm A's value, which prompt §2 leaves unreduced in (−4π, 0];
> - the rounding of the abscissa, z = expm1(x) in `PrimitivePhase._get_x`, which enters every arm as about ε|dθ/du|.
>
> Should I:
> 1. go ahead to the run as written, treating these as failures below the scale the gates protect;
> 2. have E1 (c) and E2 re-posed with those two terms first; or
> 3. stop here?

## 5. State left

- Both trees are unmoved and clean. Nothing is committed.
- No run is registered, no lock was taken, and Ray was never initialised.
- The worktree is still in place for a resume.
- The executing agent's context is kept, and it can be resumed with the user's answer.

---

# The resume under prompt §8 (commit `d170e6d`), reviewed under note §3 as amended by note §6

**Verdict: PASS.** Every check of note §3, as §6 amends it, holds. Under §8's amended bounds:
- **E1, E2 and M0 hold.**
- **R1 holds, A1 holds, N4 fails, R2 holds.**
- **No unit went over budget.**
- **The fingerprint is `0e32f24a…`.**

N4's failure is a scored verdict, not a failed check.

- **The resume.** The laptop's message of 2026-10-25 gave the user's decision: amend gates E1 (c)
  and E2, then run, and keep the registry's script hash. It carried prompt §8 and note §6
  verbatim; the commit adding them is `d170e6d` on `origin/handover-remedial`. This checkout was
  **not** fetched or moved: `handover` A3 ran in it at `c8ca155`. I passed §8 to the same
  executing agent verbatim, and resumed it with its context.
- **The run.**
  - Run id `levin-phase-floor-04-remove-20261025T231110`, launched 23:11:10Z, detached pid 79510.
    There is no `sh` pid: `nohup` execs Python in the same process.
  - I waited on the pid once, in the background (`while kill -0 79510; do sleep 60; done`). It
    exited at 23:34:59Z.
  - Final `status.json`: `state: done`, `exit_code: 0`, `units_done: 28`, `pid: 79510`.
  - I then resumed the agent for steps 8–10.

## R.1 Scope — holds

- **The trees.**
  - Main checkout: `HEAD` `c8ca1559fcc53fdb88998f170257635431fb85ca`, `git status --short` empty.
  - Worktree: `HEAD` `98e68d3e1b60e6de4378b7d9aded3a120d6088a9`, `git status --short --ignored`
    empty.
- **`find var -newer <marker> -maxdepth 2`** lists:
  - `var/levin-phase-floor-04/…`;
  - `var/runs/levin-phase-floor-04-remove-20261025T231110`;
  - `var/handover-03/…`, which is `handover` A3 and allowed.

  Nothing under `var/datastores` is newer than the marker, and nothing in 03's run directory.
- **`find <worktree> -newer <marker>`** lists nothing, `__pycache__` included.
- **Outside the repository.**
  - `~/Library/Caches/black/25.1.0/cache.-.88.1.0.0.0.1.0.0.@.pickle` has mtime
    2026-10-25 22:54:59Z. That is the first dispatch's write, `issues.md` I5, and §6 says it is not
    a stop. Nothing there is later.
  - `/tmp/ray-sgk` was last modified 21:14, before this prompt began.
  - `~/.raypoolkit/` is empty.

## R.2 The store — holds

- `diff` of `state/store_files_before.txt` against `state/store_files_after.txt`, with the
  `#` header lines removed: **identical**. That covers the SHA-256, size and mtime of the primary,
  the four shards and the sidecar.
- `runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite`, without
  `--write`, re-run by me after the run:
  - `digest:   0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`
  - `problems: none`
  - `recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00 by a person, at b90752d…`
- There is no `-wal`, `-shm` or `-journal` file in `var/datastores/`.
- `rehearsal_lock ray-processes --ray-temp-dir /tmp/ray-sgk`, run as a command of its own, printed
  nothing (rc 0).
- `state/step3_module_origins.txt` records `ray initialised False`, and `run_remove.py` checks
  `ray.is_initialized()` at its end.

## R.3 The trees, read — holds, with §6's provenance

- **`PYTHONPATH` is the worktree.**
  - `scripts/launch.py:77` sets `env["PYTHONPATH"] = str(WORKTREE)`.
  - `state/step3_module_origins.txt` shows `ComputeTargets.primitive_phase`,
    `QuadSourceIntegral`, `phase_groups`, `cumulative_table`, `AdaptiveLevin.levin_quadrature`,
    `StoreFixtures.standin_pool` and `main.py` all resolving under
    `/Users/ds283/Documents/Code/SecondaryGWKit-lpf04-98e68d3/`. The pinned packages resolve from
    the main checkout's venv.
- **The manifest** (§6):
  - `git_head` is `c8ca1559…`, with `git_dirty: false`;
  - `script` is `var/levin-phase-floor-04/scripts/run_remove.py`, with `script_sha256`
    `3c43db3f51ccd8a59fd3237067a17875eaf42dc56b8f29c2de1b5753b9280da7`;
  - `purpose` names the worktree path and `98e68d3e1b60e6de4378b7d9aded3a120d6088a9`.
- **The checkpoints** (§6): all 28 records carry the registry's `script_sha256` `3c43db3f…`,
  `git_head` `c8ca155`, and `git_dirty` false. Every payload has `worktree_head` `98e68d3…` and
  `worktree_status_short` `''`, and records every script's SHA-256.
- **The output headers.** Each records the writing script's SHA-256, the worktree's `HEAD` and
  the main checkout's: `out/phases_04.txt`, `out/remove_04.txt` and `state/store_files_*.txt`.
  `phases.py`'s SHA-256 `9f046157…` in `phases_04.txt`'s header equals the one in the run's
  checkpoints.

## R.4 The arms, read (`scripts/phases.py`, `scripts/run_remove.py`) — holds

- **Where the patch sits.** `install()` replaces `PrimitivePhase.theta_mod_2pi` at the class
  attribute and restores `ORIG_THETA_MOD_2PI` in a `finally`, then asserts the restore.
  - It is entered only inside the wrapper `w_eval` around
    `QuadSourceIntegral.evaluate_QuadSource_integral`, whose module attribute is itself restored
    in a `finally`.
  - After each unit, `run_remove.py` asserts that all three attributes are the originals.
- **P.** `Dispatcher.__call__` returns `ORIG_THETA_MOD_2PI(phase, x, x_is_log)` with its arguments
  unchanged. Any `other` kind is passed through the same way and counted. `dispatcher_counts`
  shows `other: 0` in all 28 units.
- **A.** It changes G only: `kind_of` requires `sign == -1` and leading label `tau`.
  - `z_top` is `min(z in table._z_list if z >= max_x)`, a node of the `tau` table.
  - `c_G` is `reduce_exact(s·k·Δ^ext(z_top, z_response))` at `MP.prec = 512` bits (about 154
    digits, at least 40), rounded once.
  - The variable part is `WKB_mod_2pi(s·k·delta(z, z_top) + φ)[1]`, and the sum is not
    re-reduced.
  - T is production's.
- **X.** `delta_ext` sums the node limbs exactly, `(hi_b − hi_a) + (lo_b − lo_a)`, plus
  production's own end partials from `_integral` at production's `_locate` node choice.
  `reduce_exact` reduces against `TWO_PI_MP = mpf(TWO_PI)` into `(−TWO_PI, 0]`, and
  `mp_to_float` rounds once to nearest.
- **Nothing else is patched.** `raw_theta`, `theta_deriv`, the amplitudes, the partition, the
  driver and the tolerances are untouched. The only other module attributes replaced are 03's
  transparent recorder on `adaptive_levin_sincos` and the `evaluate_QuadSource_integral` wrapper
  that installs the arm. Both call the original with their arguments unchanged and return its
  value.

## R.5 The gates — hold, against §8's amended bounds

- **E1** (`out/phases_04.txt` line 185: `E1 HOLDS`). Line 138 states the amended (c),
  `4 EPS Phi^A + 4 x 2^-49`. Every sub-interval line ends `-> ok`. The worst ratios are 1.000 for
  (a), 0.870 for (b) and 0.379 for (c).
- **E2** (line 278: `E2 HOLDS`). Line 219 states §8's bound:
  `EPS·median ΣΦ + EPS·median Σ|dθ/du| + 2^-48`.
  - All 29 scored groups are `-> ok` under P, A and X. The ratios are P 0.22–0.34, A 0.10–0.77
    and X 0.05–0.20.
  - Under X, σ_θ is at most 6.53e-14, against `1e-2 F` of at least 1.69e-08.
  - The reproduction of 03's medians under P `HOLDS` (line 217). I compared R1's five groups, R2's
    three, R3's two and R4's first three by eye against
    `docs/levin-phase-floor/03/out/instrument_03.txt`: 4.74e-07, 4.8e-07, 4.77e-07, 4.84e-07,
    4.75e-07, … They match to 03's printed digits.
- **The first dispatch's output** is kept as `out/phases_04_first.txt`: mtime 22:52:10, its
  header written 22:51:05Z, line 197 `E1 FAILS`, unchanged.
- **M0.**
  - `run_instrument.py:183–207` compares with `==`, adds the `struct.pack("<d")` bit pattern for
    floats, and uses `is` for `None`. There is no `isclose`.
  - `out/remove_04.txt` reports `M0 HOLDS on all seven rows`, with 0 gated fields differing and
    `analytic_rad EQUAL`. Only `elapsed` differs, and it is ungated.
  - **Independently, for R1:** `sqlite3` with `mode=ro` on `handover-A3-v2-lambdacdm-shard0002.sqlite`,
    serial 37527, gives `total` `-0x1.631c5efe83a46p-78` and `total_abserr`
    `0x1.2f222e3d4dd6dp-95`, with 34147 regions at depth 20. These are equal in `float.hex` to the
    `P-1e-7-R1` checkpoint's `total` and `total_abserr`, and that record's `m0_holds` is `true`.

## R.6 The predictions — hold

- **The order of the mtimes**, which §6 amends to use the new `phases_04.txt`:
  1. `out/phases_04.txt`, 23:10:04.08Z;
  2. `state/predictions.md`, 23:10:41.15Z;
  3. the run's `manifest.json`, 23:11:10.42Z.
- **Prompt §5 copied unchanged.** I extracted it from `prompts/levin-phase-floor/04-remove-the-floor.md`
  at `c8ca155`, between its `## 5.` and `## 6.` headings. Its body occurs **verbatim** in both
  `log.md` and `state/predictions.md`, checked by substring test.
- **A1's list.** I recomputed it from `out/phases_04.txt`'s every-group table: 45 groups, 29
  scored, and 14 scored with `F^A ≤ 1e-8`. The set **equals** the 14 groups listed in
  `state/predictions.md`.
- **The scores, re-scored by me from `out/remove_04.txt`'s totals.**
  - N4 on R3: |7.026853300557323e-29 − 7.024951258016924e-29| = 1.902e-32, against
    `abserr_P` 1.638e-32, a ratio of 1.161.
  - R2 on R6: |4.9262712373466785e-24 − 4.9262716419434e-24| = 4.046e-31, against 8.532e-31,
    a ratio of 0.474.

  Both agree with the agent's numbers. The verdicts against §5 as written:

| Prediction | Verdict | Numbers |
|---|---|---|
| R1 | **holds**, 7/7 | X/P regions R1–R7: 0.0039, 0.0030, 0.0020, 0.0032, 0.0049, 0.0040, 0.0038. The largest count at depth ≥ 18 in any scored group under X: 0, 0, 0, 2, 3, 3, 3 |
| A1 | **holds**, 14/14 | Every covered group has 0 regions at depth ≥ 18 under A, with a maximum depth of 13–17 |
| N4 | **fails** on R3, R6, R7 | \|P − X(1e-8)\| / `total_abserr_P`, R1–R7: 0.440, 0.0795, **1.161**, 0.245, 0.445, **1.533**, **1.942**. The relative differences on the failing rows are 2.7e-4, 6.6e-7 and 7.3e-7 |
| R2 | **holds**, 7/7 | \|X7 − X8\| / `abserr_X7`, R1–R7: 0.405, 2.1e-4, 0, 0.305, 8.9e-7, 0.474, 2.5e-3 |
| Q5 | reported for every row | \|X7 − X8\| / \|X8\|, R1–R7: 5.36e-7, 3.16e-9, 0, 6.73e-6, 1.41e-11, 8.21e-8, 3.04e-10. The 2026-09 sweep gave 1.9e-6 (mid band), 2.84e-6 (low) and 8.3e-7 (seam) |

- **A reading I record and do not adjudicate:** `log.md` deviation 11, which is UNINTENDED DRIFT.
  The agent's R6 smoke tests of the budget mechanism ran under X and A before
  `state/predictions.md` was saved, so the agent had seen R6's X and A results first.
  - The predictions file holds §5 verbatim (checked above) and A1's list.
  - That list is fixed mechanically by step 5's rule, and I recomputed it.
  - So there was no discretionary content for those results to influence.
  - Check 6's evidence, the mtime order, holds as written.

## R.7 The budget and the analysis — holds

- **The budget is enforced by the run itself.**
  - `Dispatcher.__call__` raises `OverBudget` on the first `theta_mod_2pi` call past
    `t0 + 3 × p_wall[row]`, and `run_remove.py` records the unit `over budget` and continues.
  - This replaces `setitimer`, which `log.md` deviation 10 records as IMPLEMENTATION CHOICE under
    the prompt's "or equivalent", with `issues.md` I9 giving the reason.
  - The budgets were 372.8–964.7 s, and the X and A units took 0.4–3.0 s, so **no unit went over
    budget**.
- **`analyse.py`'s inputs.** It reads the checkpoints, the run's `manifest.json` and
  `status.json`, the sidecar's digest (read-only), and step 5's `state/phases_04.json` for each
  group's scored status and its `F^arm_g`. The last is a step-5 product, not a checkpoint. It
  supplies only what §5 sets *beside* the depths, and `log.md` records it as IMPLEMENTATION
  CHOICE. No production code is imported and nothing is re-run.
- **The re-run.** I re-ran `analyse.py` (cwd `var/levin-phase-floor-04`, `-B`,
  `PYTHONDONTWRITEBYTECODE=1`) into my scratchpad (`remove_04.rerun.txt`), with empty stderr and
  exit 0. Its `diff` against `out/remove_04.txt` differs **only in line 2**, the `# written`
  timestamp.
- **Coverage.**
  - Every one of the 45 groups is reported, with all four units each.
  - The depth histograms are split by type (`depth:Levin/CC`).
  - The descriptive items are in `out/phases_04.txt`:
    - `F^P_g`, `F^A_g` and `F^X_g` for every group (line 280);
    - `σ_φ`, production and rebuilt, for every factor (line 351);
    - the modulus against F (line 439).

## R.8 The run — holds

- `runregistrykit list` shows `levin-phase-floor-04-remove-20261025T231110 done 28/28`.
- The manifest has `campaign: levin-phase-floor`, `prompt: 04`, and a run id carrying the slug
  `remove`.
- `status.json` holds `pid: 79510`.
- `stdout.log` (17401 bytes) and `stderr.log` (0 bytes) are in the run directory, beside
  `checkpoint.jsonl` (28 lines) and the 21 X and A unit pickles.

## R.9 The records — hold

- **`log.md`** follows README §5.1. Its sections are Predictions and verdicts, What shipped,
  Deviations from the prompt, Verification performed, Observations not acted on, and State
  handed on.
  - It names `d170e6d` as the source of §8, and quotes §8's amended gates verbatim.
  - Every deviation is classified. Step 1 is item 8, **STRUCTURALLY REQUIRED**. The `black` cache
    (7) and the smoke tests before the predictions (11) are **UNINTENDED DRIFT**. The rest are
    **IMPLEMENTATION CHOICE**.
- **`verification.md`** is headed `## 4. The floor removed`. Every link is relative to
  `docs/levin-phase-floor/` in the `04/out/…`, `04/scripts/…` or `04/state/…` form.
- **`issues.md`** holds I1–I9. I6–I9 are new at the resume:
  - I6: X at 1e-8 still sends one short chain to depth 20;
  - I7: A matched X everywhere, so `F^A_g` did not predict the depth;
  - I8: N4's three rows;
  - I9: mpmath `from_float`'s bare `except:`.

  I5's title still reads "The registry run was never begun", which was true at the first stop.

## R.10 State left

- Both trees are unmoved and clean. Nothing is committed.
- No lock was taken and no Ray process exists.
- The store is byte-identical, with fingerprint `0e32f24a…`.
- The worktree remains, for the laptop to remove once the records are committed and the user
  agrees (note §4).
- The checkpoints and pickles stay in the run directory.
