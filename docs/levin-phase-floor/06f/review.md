**Verdict: PASS. Prompt 06f is complete, all six review checks hold, and no §7 stop condition and no agent question arose. F0 holds with no replacement; F1, F2 and F3 hold; F4, F5 and F6 fail.**

# levin-phase-floor 06f — the orchestrator's review (Mac Studio)

Written 2026-10-31 on mps026298 by the orchestrator session that dispatched the 06f agent. The
note's §B.3 is the checklist. The failures of F4–F6 are scored results under prompt §6 acceptance
item 2, not stops; F4's violation is "a finding for the user, not a stop" (prompt §4).

**The trees:** the worktree `SecondaryGWKit-lpf06f-f4770d1`, detached at the dispatch SHA
`f4770d13`, whose code is `f4d8e17`'s; the main checkout, `handover-remedial` at `28ebaf0`, never
moved.

**Where things are** (all under `var/levin-phase-floor-06f/review/`):
- the brief is `dispatch.md`, SHA-256 `0e02a2e1…` (`dispatch_sha256.txt`). The agent was dispatched
  on Opus, once, with §B.0 verbatim;
- the preflight is `preflight/`: marker 2026-10-31 20:27:17.525 UTC, `var/` listing of 66,760 files;
- every check's command output is in `checks/`;
- my readers are `scripts/` (`orch_z2_replay.py`, `orch_m1.py`, `orch_m2.py`, `orch_score.py`,
  `orch_population.py`), with their SHA-256 in `scripts_sha256.txt`.

Every float comparison below is exact on the recorded doubles (hex), unless a relative difference
is quoted.

## The run

`levin-phase-floor-06f-measure-20261031T204803`: launched detached at 20:48:03Z (driver pid 54404,
`sh` 54397), done at 21:01:03Z, 44/44 units, exit 0. I waited on the pid with one background
`kill -0` loop and resumed the agent once with the final `status.json`. The manifest records
`git_head` `28ebaf0`, `git_dirty` false, cwd `var/levin-phase-floor-06f` (the main checkout, as for
06e's census); every replay's header records the tree $g$ at `f4770d13`, status empty.

## 1. Scope — holds

| Check | Result | Record |
|---|---|---|
| `var/` against the preflight listing | 0 files gone; 1 changed, the preflight listing itself (written while it listed); 584 new, all under `var/levin-phase-floor-06f/` or the 06f run directory | `checks/scope.txt` |
| Files newer than the marker outside the 06f directory and run | none (twice: 21:09Z and 21:21Z) | `checks/scope.txt`, `checks/scope_final.txt` |
| `var/datastores/` | nothing newer than the marker | `checks/scope_2.txt` |
| The scratch copy | SHA-256 of its five files identical to the preflight's (= `copy_files_certified.txt`), listing identical, no journal, `-wal` or `-shm` | `checks/a3v2_copy_after.sha256`, `checks/scope_final.txt` |
| 06e's census checkpoint | `a0dc4d7f5c47eb28…` before and after | `checks/scope_final.txt` |
| The worktree | `f4770d13`, `status --short --ignored` empty, no `__pycache__`, nothing newer than the marker | `checks/scope_2.txt`, `checks/scope_final.txt` |
| The main checkout | `28ebaf0`, `status --short` empty, nothing newer than the marker outside `var/` and `.git/` | `checks/scope_2.txt` |
| `.git/` | only the directory's own mtime moved, at 20:48:03.397 (the run's `begin`, whose git probe ran in the main checkout); no file under it is newer than the marker; refs unchanged | `checks/scope_2.txt` |
| 07's work | no `var/levin-phase-floor-07*` touched (only `-07-leaves` exists here, nothing in it newer than the marker); no stash touched | `checks/scope_final.txt` |
| Ray, lock, registry | lock free throughout, no Ray processes of `/tmp/ray-sgk`; all 41 capture workers report `ray_initialized` false; registry 36 runs, all finished | `checks/scope_final.txt` |

## 2. F0 — holds; Z2's call re-run by me

- **The agent's comparisons** (`out/f0.csv`, `out/cmd/43`): all 41 sampled calls hold every clause
  (capture equals census; replay equals census on value, abserr, regions, maximum depth and
  converged; summed terms equal the checkpoint's `cc_floor`; every leaf reproduced; leaves sum to
  value and abserr). No replacement. Z2's call in two fresh processes (pids 55078, 55415): identical
  digest, `leaves.json.gz` and `solves_cc.json.gz` (`out/cmd/42`).
- **My re-run** (`scripts/orch_z2_replay.py`, `checks/z2_replay.txt`): Z2's capture loaded (its
  SHA-256 equal to `capture.json`'s), replayed in one process **with nothing wrapped**, 93.6 s:
  - value `-0x1.0e0aae8560300p-85`, abserr `0x1.badaaed87034dp-110`, 8,706 regions, max depth 20,
    not converged — equal to the census checkpoint;
  - every one of the 8,706 regions equal to 06e's region pickle (record SHA-256 checked against the
    checkpoint) in start, end, depth, type, abserr, phase_err, phase_limited and total_err: 0
    mismatches;
  - the agent's per-leaf table has the same 8,706 spans.
  - A first attempt stopped on a comparison error in my own reader (`type` is `"Levin"`/`"direct"`
    on the region, 0/1 in the pickle); fixed in the reader and re-run. Kept as
    `checks/z2_replay_attempt1_type_compare_error.txt`.

## 3. M1, re-scored with my own reader — holds

`scripts/orch_m1.py` (`checks/m1_rescore.txt`), on Z2's call, `g-1e-8-0-14312` #5 and
`g-1e-7-1-11219` #2. From the per-leaf tables, every CC leaf's `phase_err` and every leaf's
`total_err` rebuilt in the code's order (0 mismatches on 8,706 + 176 + 133 leaves), and the sum of
leaf errors left to right equal to the call's abserr bit for bit on all three.

| Call | $N_{\rm CC}$ | $\sum\lvert E\rvert$ | $\lvert\sum E\rvert$ | ratio | signed (leafwise) / request | literal / request |
|---|---|---|---|---|---|---|
| `g-1e-8-0-33995` #5 (Z2) | 8,600 | 1.3123e-33 | 2.4313e-35 | 53.98 | 0.263 | 0.067 |
| `g-1e-8-0-14312` #5 | 125 | 1.0362e-34 | 1.2566e-35 | 8.25 | 0.977 | 0.936 |
| `g-1e-7-1-11219` #2 | 133 | 1.6611e-35 | 7.9721e-37 | 20.84 | 0.954 | 0.926 |

All four quantities equal the agent's `result.json` bit for bit. On Z2's call, $\sum\lvert E\rvert$
over its returned CC leaves (fsum) is `0x1.b4186452c4a69p-110`, equal to the checkpoint's
`cc_floor.sums_all_cc_leaves.abs_E` bit for bit.

## 4. M2, re-checked — holds

- **δθ at five nodes, in `mpmath`, by another formula** (`scripts/orch_m2.py nodes`,
  `checks/m2_dtheta_nodes.txt`). On `g-1e-7-0-12689` #19 (J4): the three nodes with the largest
  ratio and two seeded others. The reference is each constituent's phase as $\arg H^{(1)}_\nu(y) +
  \pi/2$ (`mp.hankel1`), not the agent's `atan2(J, −Y)`, at 30 and 50 digits. Production's
  `theta_mod_2pi` and `theta_abserr` recomputed: equal to the agent's at every node. My δθ at the
  agent's argument convention equals the agent's to all printed digits (relative difference 0 or
  $3\times10^{-15}$), and 30 → 50 digits changes nothing. The worst node: δθ $= -3.943\times10^{-13}$
  against `theta_abserr` $2.665\times10^{-15}$, ratio 148.0, at $K\hat\eta = -6849$.
- **δI by another method** (`scripts/orch_m2.py dI`, `checks/m2_dI_driver.txt`): the Levin driver
  itself on $[-s\,\texttt{ta}\,f_1]\sin\theta + [s\,\texttt{ta}\,f_0]\cos\theta$, with the phase dict
  less `theta_abserr`, split at the recorded kinks, at rtol $10^{-10}$ and $10^{-12}$:
  - `g-1e-7-0-12689` #19: $\delta I(+) = -8.19167373\times10^{-21}$, agent's Gauss–Legendre
    $-8.19167376\times10^{-21}$, relative difference $4.5\times10^{-9}$; $D/\max_s\lvert\delta I\rvert$
    2.6216317 (mine) against 2.6216317 (agent's);
  - `g-1e-8-3-914` #16: relative difference $8\times10^{-16}$; ratio 1.80922 both.
- **The method check** (`out/cmd/31`): the ν = 1/2 phase-pair error is $9.1\times10^{-18}$ and
  $5.9\times10^{-17}$, under the 1.1e-16 `bessel_phase.py` documents (`:196`, `:625`); 30- and
  50-digit references agree to $10^{-31}$. Passes as written.
- Item 3's δI converged to $1.9\times10^{-7}$ relative or better on every call (`m2_calls.csv`).

## 5. The predictions — holds

- **Copied verbatim:** `state/predictions_20261031T203333Z.md` (SHA-256 `fc084439…`, mtime
  20:33:33Z) holds prompt §4 identical to the prompt at `f4770d13` (prompt SHA-256 `9af76379…`);
  each rule line appears verbatim in `log.md` (`checks/predictions.txt`).
- **Before the numbers:** the first output with a measured number is `out/cmd/20` (the sample,
  20:36:46Z), then the development capture `21` (20:38:07Z); the registered run began 20:48:03Z.
- **Re-scored by me** from the agent's tables (`scripts/orch_score.py`, `checks/f_rescore.txt`),
  against the rules as written:

| | Number | Mine | Agent's |
|---|---|---|---|
| F1 | Z2's call: $\sum\lvert E\rvert/\lvert\sum E\rvert$ = 53.98 (rule ≥ 10) | **holds** | holds |
| F2 | Z2's call: signed-combination error $1.759\times10^{-34}$ (literal form $4.47\times10^{-35}$) ≤ request $6.689\times10^{-34}$ | **holds** | holds |
| F3 | 11 sampled node-dominated calls with $N_{\rm CC}\ge100$: ratios 7.20–80.2, median 20.84 (19.19 without Z2's call) | **holds** | holds |
| F4 | 49 of 1,177 nodes with $\lvert\delta\theta\rvert >$ `theta_abserr`, on 2 of 20 calls (`g-1e-7-0-12689` #19: 45 nodes, up to 148×; `g-1e-7-3-1054` #20: 4, up to 1.83×) | **fails** | fails |
| F5 | median $D/\max_s\lvert\delta I\rvert$ = 6.01 (1.07–21.2) | **fails** | fails |
| F6 | median $\sum_\ell\lvert\delta I_\ell\rvert/\max_s\lvert\delta I\rvert$ = 1.00 (1.00–1.57) | **fails** | fails |

- **The sample** (`scripts/orch_population.py`, `checks/population.txt`): my own count from the
  checkpoint gives 491 / 1,323 node-dominated and 2,493 / 2,785 declared-dominated calls, equal to
  06e's report; at 1e-7 only 4 node-dominated calls have $N_{\rm CC}\ge100$ at all, so the agent's
  "all 4 taken" is right; all 41 sampled calls are in their stated stratum with their stated
  $N_{\rm CC}$.

## 6. The records — holds

- **`log.md`** has every heading of campaign README §5.1 (subject, commit and result; What shipped;
  Predictions and verdicts; Deviations; Verification performed; Observations not acted on; State
  handed on). **17 deviations, each classified**: 2 STRUCTURALLY REQUIRED (`prov.py`;
  `x1_capture.py`'s `ROWS_JSON`), 13 IMPLEMENTATION CHOICE, 2 UNINTENDED DRIFT (a `/tmp/x` scratch
  file, written and deleted at once; a `BrokenPipeError` in `out/cmd/46`). F2's leafwise reading
  (4) and F3's inclusion of Z2's call (5) are among them, each with the other reading reported
  beside it; neither changes a verdict. Item 1's reference convention is 7 (see Observation 1).
- **`out/results.md` proposes no code.** Each term's remedy section opens "No code is proposed
  here" and states what a remedy would have to do in terms of the measured quantities.
- **Provenance:** every script prints a header with its own SHA-256, the tree's HEAD and status and
  its inputs' SHA-256 (e.g. `out/cmd/24`, `31`, `43`), and the replays' `result.json` carry a
  `header`. The CSV and JSON tables carry none in the file; their SHA-256 are printed in
  `out/cmd/43` as written, and the current `m1_calls.csv` and `m2_calls.csv` hash to them
  (`checks/f_rescore.txt`). I read that as meeting README §5 rule 5 through the command record.
- **Kept on the Mac Studio:** `KEPT_ON_MAC_STUDIO.txt` lists 90 files with SHA-256 and size; every
  hash and size re-checked, 0 mismatches; every pickle and every file over 2 MB under the 06f
  directory (outside `review/`) is listed, and nothing else (`checks/kept.txt`).
- `verification.md` (§6f text) and `issues.md` (I1 new, I2 and I3 narrowing, O1) are present; M1's
  and M2's tables are complete for every sampled call (empty cells only in the runs-test columns of
  calls with too few CC leaves for it).

## Observations of the review (not acted on)

1. **F4 also fails with the rounding of `exp(u)` counted.** The agent's reference (deviation 7)
   evaluates the constituents at the double $\hat\eta$ = `np.exp(u)`, so δθ leaves out the
   rounding of `exp(u)`. With the exact $e^u$ instead, the three worst nodes give ratios 246, 17.9
   and 29.2 (against 148, 125 and 87.8): different per node, still violations. So F4's verdict does
   not depend on the convention. Whether the `exp` rounding belongs to the phase's error or to the
   node's displacement is a question for whoever takes I1.
2. **One physical call is counted twice in F3.** `g-1e-7-1-11219` #2 and `g-1e-8-1-11219` #2 are
   bit-identical (value, abserr, digest: atol binds at both rtols), each drawn in its own rtol
   stratum. Counted once, the median is 19.19, and 17.55 with Z2's call also left out: F3 holds on
   every reading.
3. **F6 could hardly hold on this population.** 18 of the 20 declared-dominated calls have one or
   two leaves. The agent says so too.
4. **My own stray file.** In my scope check, one redirect wrote `/tmp/claude-35086/b.txt`, outside
   my scratchpad. It was removed in the next command, and nothing read it.

## Kept on the Mac Studio

- `var/levin-phase-floor-06f/`: the 90 files of `KEPT_ON_MAC_STUDIO.txt` (82 capture pickles under
  `state/capture/C00…C40/`, 6 under `state/dev/`, and Z2's two `leaves.json.gz`, 2.4 MB each,
  `8741e4fa…`).
- The run directory `var/runs/levin-phase-floor-06f-measure-20261031T204803/` (44 KB).
- `review/checks/var_files_after.txt.gz` and `review/preflight/var_files_before.txt.gz` are
  listings, about 1.1 MB each, and may be copied.
