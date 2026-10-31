# levin-phase-floor 06f — how two Clenshaw–Curtis floor terms add up across a call's leaves

**Subject:** prompt 06f, a measurement on the Mac Studio (mps026298), 2026-10-31.
- It ran Ray-free, with no lock, no store built and no code changed.
- Tree $g$ = worktree `SecondaryGWKit-lpf06f-f4770d1` at `f4770d13`, whose code is `f4d8e17`'s.
- **Commit:** none. The records are committed by the orchestrator on the laptop.

**Result:** F0 holds with no replacement. F1, F2 and F3 hold. F4, F5 and F6 fail.
- **The node errors.** Combined signed, they cancel 54× on Z2's call, and the call would then meet
  its request on its returned partition. The median factor on the larger sampled calls is 21.
- **The declared phase term** is a median 6× the call's response to a phase error of the declared
  size. That factor is cancellation within a leaf: it is neither the proxy (1.05×) nor the
  combination across leaves (1.0×).
- **The declaration** is about 2,000× the phase error actually committed, except on tail Levin
  leaves. There it is up to 148× too small: `theta_abserr` omits the rounding of $K\eta$.
- Across 06e's census the declared term never reaches a call's request.

*The predictions below were copied in at 2026-10-31T20:33Z, before any number was computed
(`state/predictions_20261031T203333Z.md`, SHA-256 `fc084439…`).*

## Predictions (copied verbatim from prompt §4 before the run)

**M1:**
- **F1 — the node errors cancel on Z2's call.** **Rule:** $\sum|E| \ge 10\,|\sum E|$ over its
  returned Clenshaw–Curtis leaves.
- **F2 — Z2's call meets its request with signed node errors.** **Rule:** on its returned partition,
  the signed-combination error is at most $\max(\texttt{atol}, \texttt{rtol}\,|v|)$.
- **F3 — the cancellation is general.** **Rule:** over the sampled node-dominated calls with
  $N_{\rm CC} \ge 100$, the median of $\sum|E|/|\sum E|$ is at least 10. Undecided if fewer than five
  such calls exist.

**M2:**
- **F4 — the declaration is honest.** **Rule:** on every M2 call, $|\delta\theta(x)| \le
  \texttt{theta\_abserr}(x)$ at every node sampled. A violation is a finding for the user, not a
  stop.
- **F5 — the declared term overstates the response.** **Rule:** the median over M2 calls of
  $D / \max_s|\delta I|$ is at least 10.
- **F6 — combining per leaf is part of it.** **Rule:** the median over M2 calls of
  $\sum_\ell|\delta I_\ell| / \max_s|\delta I|$ is at least 10.

## What shipped

Nothing shipped in code. The records are under `var/levin-phase-floor-06f/`:
- **Write-ups:**
  - `log.md` (this file);
  - `verification.md`, the text of `VERIFICATION.md` §6f;
  - `issues.md`: I1 opens one issue, I2 and I3 narrow two, and O1 is one observation;
  - `KEPT_ON_MAC_STUDIO.txt`.
- **Results:** `out/results.md` (the verdicts, the numbers, and what a remedy would have to do for
  each term). The tables are `out/f0.csv`, `m1_calls.csv`, `m1_z2_by_depth.csv`,
  `m1_z2_subtrees.csv`, `m2_calls.csv`, `m2_leaves.csv` and `scores.json`.
- **Per call:** `out/replay/<unit>_<index>/` holds `result.json`, `leaves.json.gz`, and
  `nodes.json.gz` on M2 calls. Z2's call also has `solves_cc.json.gz`.
- **The sample:** `out/sample/`.
- **The capture record:** `out/capture.json`.
- **Every command and its output:** `out/cmd/NN_*.txt`, also in `out/commands.log`.
- **Scripts:** `scripts/`, 06e's census scripts as copied, with the diff saved in
  `out/scripts_diff_against_06e_at_launch.txt`. Seven are new:
  - `draw_sample_06f.py`, `replay06f.py`, `method_check_06f.py`;
  - `run_06f.py`, `launch_06f.py`;
  - `analyse06f.py`, and the descriptive scripts `f4_attribution_06f.py` and
    `declared_vs_request_06f.py`.
- **`state/`:**
  - the predictions;
  - 06e's reference copies, in `state/scripts_06e_reference/`;
  - the capture pickles, in `state/capture/`;
  - the development captures and replays, in `state/dev/`;
  - the launch transcript, `state/launch_measure.txt`.
- **The registered run:** `var/runs/levin-phase-floor-06f-measure-20261031T204803`, done, 44 of 44
  units, exit 0.

## Predictions and verdicts

| | Number scored | Verdict |
|---|---|---|
| F0 | 41 of 41 captures and replays equal 06e's census record bit for bit (value, abserr, regions, max depth, converged); the summed terms over every call's CC leaves equal `cc_floor` (all and floor-attributed, fsum, hex); every leaf's floor rebuilt in the code's order; Z2's call twice in fresh processes (pids 55078, 55415) with identical digest, per-leaf table and every CC solve's $E$. No replacement | holds |
| F1 | Z2's call: $\sum\lvert E\rvert = 1.3123\times10^{-33} \ge 10\lvert\sum E\rvert = 2.431\times10^{-34}$ (ratio 54.0) | **holds** |
| F2 | Z2's call: signed-combination error $1.759\times10^{-34}$ ≤ request $6.689\times10^{-34}$. The linear form $\texttt{abserr} - \sum\lvert E\rvert + \lvert\sum E\rvert = 4.47\times10^{-35}$ also holds | **holds** |
| F3 | 11 calls with $N_{\rm CC}\ge100$ (Z2's included): median 20.8. Without Z2's call, 19.2 over 10 | **holds** |
| F4 | 49 of 1,177 nodes violate, on 2 of 20 calls, up to 148× | **fails** (a finding, not a stop) |
| F5 | median $D/\max_s\lvert\delta I\rvert$ = 6.0 (1.07–21.2) | **fails** |
| F6 | median $\sum_\ell\lvert\delta I_\ell\rvert/\max_s\lvert\delta I\rvert$ = 1.0 (18 of 20 calls at 1.000) | **fails** |

No rule turned out badly posed. One observation on F6: 18 of the 20 declared-dominated calls
sampled have one or two leaves, so the rule could hardly have held on this population. The
population's declared-dominated calls are small calls.

## Deviations from the prompt

1. **`prov.py` changed.** It names the one tree $g$ (`f4770d13`) and asserts its depth as
   `parents[2]`, since the scripts sit in `var/levin-phase-floor-06f/scripts`. It also adds 06e's
   census run, checkpoint hash and `rows.json` as read-only inputs. **STRUCTURALLY REQUIRED**:
   06e's worker refuses to run unless the tree's HEAD equals the expected SHA.
2. **`x1_capture.py`'s `ROWS_JSON` points at 06e's `census/state/rows.json` (aa523d28), read only.**
   **STRUCTURALLY REQUIRED.**
3. **The terms were captured during the replay's own driver call**, filed by span, rather than by
   re-solving each returned leaf afterwards as `_cc_floor_terms` does. This also gives the trial
   solves that §2's subtree comparison needs. F0 shows the two routes agree bit for bit.
   **IMPLEMENTATION CHOICE.**
4. **F2's signed-combination error is computed leafwise.** Each CC leaf's floor is taken without its
   $\lvert E\rvert$, max'd with its resolution term as the code does, plus truncation; these are
   summed, and $\lvert\sum E\rvert$ is added once. The literal $\texttt{abserr} - \sum\lvert E\rvert +
   \lvert\sum E\rvert$ is reported beside it. The leafwise form is the larger, and both hold.
   **IMPLEMENTATION CHOICE.**
5. **F3 is scored with Z2's call among the "sampled node-dominated calls".** That is the literal
   reading. Without it the median is 19.2, and the verdict is the same. **IMPLEMENTATION CHOICE.**
6. **Item 1's points.** On CC leaves, all 47 fine nodes. On Levin leaves, the 24 collocation nodes,
   endpoints included, since a Levin leaf evaluates θ only at its endpoints. A seeded dense sample
   of 1,000 points per call is reported descriptively. **IMPLEMENTATION CHOICE.**
7. **Item 1's reference.** It is evaluated at the double $\hat\eta = $ `np.exp(u)` that production
   evaluates at, with the products $c_i\hat\eta$ exact (`bessel_phase.py`, "Accuracy of the supplied
   argument"), and $\theta_\nu = $ `atan2(J, −Y)`. **IMPLEMENTATION CHOICE.**
8. **Item 3's route.**
   - It uses Gauss–Legendre per returned leaf, split at the kinks of `theta_abserr` (where $c_i\eta
     = x_\star$), doubled until the relative change is ≤ $10^{-6}$. It does not use the driver.
   - $s = -1$ is computed separately, and is exactly the negative.
   - **IMPLEMENTATION CHOICE.**
9. **Item 4's meaning.** It is the rule's first-order response to the measured errors at the points
   where it evaluates θ.
   - On CC leaves, all fine nodes.
   - On Levin leaves, the endpoints, with $P$ from a re-solve with `build_p_sample=True`, whose value
     equals the leaf's bit for bit. Errors in θ′ are not included.
   - **IMPLEMENTATION CHOICE.**
10. **The "honest proxy"** replaces $\tfrac12 f_{\rm scale}h$ per endpoint by
    $\tfrac12\sum_k\lvert w_k\rvert\lvert f_k\rvert$. **IMPLEMENTATION CHOICE.**
11. **The method check** compares the ν = 1/2 phase-pair error with `EVALUATION_FLOOR`'s documented
    1.11e-16, at the call's argument and at $10^3$, with a 50-digit reference beside it.
    **IMPLEMENTATION CHOICE.**
12. **Development runs.** Before the registered run, three calls were captured and replayed under
    `state/dev/` (`out/cmd/21`, `22`). They are kept as records. **IMPLEMENTATION CHOICE.**
13. **Formatting.**
    - The new scripts were black-formatted after `draw_sample_06f.py` had first run (`out/cmd/20`).
    - That run was repeated (`out/cmd/24`) and gave byte-identical output. The pre-format output is
      kept in `state/dev/sample_before_black_dir/`.
    - The copied 06e scripts are left as copied (issues.md O1).
    - **IMPLEMENTATION CHOICE.**
14. **Two descriptive measurements beyond the prompt.** The attribution of F4's violations
    (`out/cmd/44`) and the census-wide declared term against each call's request (`out/cmd/45`).
    Neither is scored. **IMPLEMENTATION CHOICE.**
15. **`harness.py`'s inert `job_name`** (stored, never read) is left as `levin-phase-floor-06e`.
    **IMPLEMENTATION CHOICE.**
16. **A scratch file in `/tmp`.** A shell redirection wrote `/tmp/x` (a file listing) during
    record-keeping, against the no-`/tmp` rule. It was deleted at once, and nothing depends on it.
    **UNINTENDED DRIFT.**
17. **A broken pipe in `out/cmd/46`.** That record holds a `BrokenPipeError` traceback because
    `runregistrykit list` was piped to `head`. The registry is not affected; the full listing is in
    `out/cmd/11` and `26`. **UNINTENDED DRIFT.**

## Verification performed

- **The Gate.**
  - Worktree clean at `f4770d13`, its code `f4d8e17` + `CLAUDE.md`.
  - Main checkout `28ebaf0`, clean.
  - Board 06e row ✅.
  - Registry: 35 runs, all finished.
  - Lock free, no Ray processes.
  - Scratch copy EQUAL to its certification. Fingerprint at $g$ `0e32f24a…`. Checkpoint `a0dc4d7f…`.
  - Records: `out/cmd/10`–`19`.
- **The sample** reproduces 06e's per-call counts exactly (`out/cmd/20`).
- **F0** as above (`out/cmd/43`, `out/cmd/42`).
- **The method check passes** (`out/cmd/31`).
- **Item 3 is converged** to $1.9\times10^{-7}$ relative or better on every call.
- **After the run:**
  - the scratch copy is EQUAL to its certification, with no journal, `-wal` or `-shm` (`out/cmd/40`,
    `41`);
  - nothing was created or modified under `var/datastores/` (`out/cmd/46`);
  - all 41 capture workers report `ray_initialized` false;
  - lock free, no Ray processes;
  - worktree clean with nothing ignored, no `__pycache__` anywhere;
  - main checkout clean.
- **Records:** every command and its output is under `out/cmd/`. Each script writes its SHA-256,
  the tree's HEAD and status and its inputs' SHA-256 into its header. The scripts' hashes at launch
  are in `out/scripts_sha256_at_launch.txt`.

## Observations not acted on

- **F4: `theta_abserr` omits the $K\eta$ product rounding**, up to 148× (issues.md I1).
- **The declared term never reaches a request.** Across 06e's census the CC declared term over
  floor-attributed leaves reaches the request on 0 of 11,279 calls, and 0.1 of it on one (I3). The
  Levin leaves' declared terms are not in the checkpoint's sums and were not counted.
- **On Z2's call the signs of $E$ run** ($z = -25$) while the total still cancels 54×. A $\sqrt N$
  model is not evidence of a bound.
- **On Z2's call, $\sum\lvert E\rvert$ under splitting** varies by orders across subtrees. Most of it
  sits on leaves at depths 10–16, not at depth 20 (`out/m1_z2_by_depth.csv`).
- **Item 3's rule-response cross-check** on Levin leaves differs from the integral by 1.5–10 %. The
  endpoint formula has no θ′ response. This is descriptive only.
- **The copied 06e scripts are unformatted** (O1).

## State handed on

- **The run:** `levin-phase-floor-06f-measure-20261031T204803` is done, 44/44, exit 0. Nothing is
  running.
- **The lock** is free and was never taken.
- **The scratch copy** is unchanged.
- **The worktree** `SecondaryGWKit-lpf06f-f4770d1` is clean, and may be removed by the orchestrator.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`, 90 files with SHA-256):
  - the capture pickles under `state/capture/` and `state/dev/`;
  - Z2's two `leaves.json.gz` (2.4 MB each).
- **For the user:** whether either term's remedy joins 06g (prompt preamble), and I1.
