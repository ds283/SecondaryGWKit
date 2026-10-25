# levin-phase-floor 04 — remove the floor from seven depth-20 rows of A3 v2

**Subject:** prompt 04, run on the Mac Studio (mps026298).
- The main checkout `/Users/ds283/Documents/Code/SecondaryGWKit` was at the dispatch SHA `c8ca155`.
  It was not moved for the resume.
- The code ran from the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf04-98e68d3` at
  `98e68d3`, by route (a), with no Ray and no lock.
- The first dispatch stopped at step 5. The resume ran under prompt §8, which was added in commit
  `d170e6d` and reached the agent verbatim through the orchestrator.
- The registered run was `levin-phase-floor-04-remove-20261025T231110`. It ended `done`, 28/28
  units, exit 0, at 23:34:59Z.

**Commit:** none here. The records are committed by the orchestrator.

**Result:**
- **The gates.** E1 and E2 hold, as amended by §8. M0 holds on all seven rows, bit for bit.
- **R1 holds, on 7 of 7 rows.** Under X at 1e-7 the region count falls to 0.20–0.49 % of P's: 66–167
  regions against 22,147–52,909. No scored group has more than 3 regions at depth 18 or deeper.
- **A1 holds in all 14 covered groups**, each with 0 regions at depth 18 or deeper.
  - A also matched X region for region on every row, `Gqr` groups included.
  - The totals of A and X differ by at most 1.2e-12 relative.
- **N4 fails on 3 of 7 rows.** On R3, R6 and R7, |P − X(1e-8)| exceeds the stored `total_abserr`, by
  1.16, 1.53 and 1.94 times.
- **R2 holds on 7 of 7 rows**, with ratios 0–0.47.
- **Q5's number** is between 0 and 6.7e-6 per row.
- **No unit went over budget.**
- **The store is byte-identical**, fingerprint `0e32f24a…`.

## Predictions and verdicts

Prompt §5 is copied unchanged below. The same text, with §8's amended gates and A1's 14 covered
groups, is `state/predictions.md`.
- Its mtime is 23:10:41Z. That is after step 5's gates passed under §8 (23:03Z, then 23:10:04Z for
  the final `phases.py`) and before the launch at 23:11:10Z.
- The first dispatch copied §5 into this log at its stop, about 22:55Z, before any re-run.
- The gates of §5 that failed in the first dispatch (E1 (c), E2) were replaced by §8's amended
  gates, quoted after §5. The scored predictions are §5's, unchanged.

## 5. Predictions, fixed before the run (README §5 rule 3)

Copy these into the log unchanged, and score them there. The gates come first. A gate that fails is
a stop, not a verdict.

**Gates:**
- **E1 — the arms compute the same phase.** Every bound of §4 step 5 holds at every abscissa.
- **E2 — the floor model is right.** In every scored group, under each of P, A and X, the median
  $\sigma_\theta$ over 03's sampled regions is at most $\varepsilon \sum_i \Phi^{\rm arm}_i$, taken
  as a median at the same regions' midpoints. Under X it is also at most $10^{-2} F$.
- **M0 — P is production.** On all seven rows, the P unit equals the stored row **bit for bit**, by
  `==` and by IEEE-754 bit pattern. The fields are 03's: `total`, `total_abserr`,
  `total_converged`, `WKB_Levin`, `numeric_quad`, `WKB_Levin_data.max_depth` and `num_regions`, and
  every field of every group and sub-interval record of `metadata["WKB_Levin"]["subintervals"]`
  except `elapsed`. `analytic_rad` is compared and reported, but not gated.

**Scored:**
- **R1 — remove the floor and the cap goes** (findings §2.6 item 3).
  - **The rule:** under X at `rtol = 1e-7`, every one of the row's scored groups has fewer than 10
    regions at depth 18 or more, and the row's region count is at most a tenth of its P unit's.
  - **Holds** if true on at least six of the seven rows. An over-budget X unit counts as false on
    its row.
  - If R1 fails, findings §2.6's own clause fires: *"item 3 leaves the depth at 20"*, so the
    hypothesis is wrong.
- **A1 — layer 1(a) is enough where $G$'s anchor is the floor** (H-anchor's prediction).
  - **The rule:** under A at `rtol = 1e-7`, every scored group with $F^A_g \le 0.1 \times 10^{-7}$
    has fewer than 10 regions at depth 18 or more.
  - **Holds** if true in every such group. **Undecided** if there is no such group. An over-budget
    A unit counts as false for each such group on its row.
  - Report the other scored groups too, unscored: their depth beside X's, and their $F^A_g$.
- **N4 — the stored error is honest about the jitter.**
  - **The rule:** $|{\rm total}_P - {\rm total}_{\rm ref}| \le$ `total_abserr`$_P$, with
    ${\rm total}_{\rm ref}$ from X at `rtol = 1e-8`. If that unit was over budget, use X at `1e-7`
    and say so. If both were, the row is undecided.
  - **Holds** if true on all seven rows. Otherwise list each failing row with the ratio of the two
    sides. Each such row is one whose stored error understates the jitter's effect.
- **R2 — with the floor gone, the declared error is honest.**
  - **The rule:** $|{\rm total}_X(10^{-7}) - {\rm total}_X(10^{-8})| \le$ `total_abserr`$_X(10^{-7})$.
  - **Holds** if true on all seven rows. A row with either unit over budget is undecided.
  - **Q5's number,** reported on every row whichever way R2 goes:
    $|{\rm total}_X(10^{-7}) - {\rm total}_X(10^{-8})| / |{\rm total}_X(10^{-8})|$. Set it beside
    the 2026-09 sweep's 1.9e-6 (mid band), 2.84e-6 (low) and 8.3e-7 (seam).

**Not scored, and why.** H-cancel is not re-scored. 03 showed that its condition holds in the
non-cancelling groups as well, so it fails as the whole account. Both hypotheses predict R1, and
neither A nor X can tell them apart. Nothing here predicts which groups still reach depth 20 under A
when $F^A_g$ is large. In P, 16 of 45 groups were left unscored although their rows' $F$ was the
same, so a large floor is not enough to send a group to the cap. Those groups are descriptive.


### The amended gates (prompt §8, commit `d170e6d`, verbatim)

- **E1 (c):** $|{\rm wrap}(\rho^A - \rho^X)| \le 4\varepsilon\Phi^A_G(z) + 4\times2^{-49}$, for
  $G$. E1 (a) and (b) are unchanged.
- **E2:** in every scored group, under each of P, A and X, the median $\sigma_\theta$ over 03's
  sampled regions is at most
  $$\varepsilon\sum_i \Phi^{\rm arm}_i + \varepsilon\sum_i \Big|\frac{d\theta_i}{du}\Big| + 2^{-48},$$
  each term a median at the same regions' midpoints. $d\theta_i/du$ is
  `theta_deriv(x, x_is_log=True, log_derivative=True)`. Under X, the median is also at most
  $10^{-2}F$, unchanged.

### Verdicts

| | Number | Verdict |
|---|---|---|
| **E1 (a)** | worst \|float(Δ^ext) − delta()\| / 2 ulp = 1.000 over 80,000 abscissae | holds |
| **E1 (b)** | worst ratio 0.870 | holds |
| **E1 (c)**, amended | worst ratio 0.379 (first dispatch, bound 1e-15: failed at 23–160 of 2000 on six lone-`G` sub-intervals) | holds |
| **E2**, amended | 29/29 scored groups under each arm; ratio to the bound 0.224–0.341 (P), 0.104–0.772 (A), 0.046–0.203 (X); X ≤ 1e-2 F everywhere (σ_θ^X ≤ 6.5e-14) | holds |
| (E2's precondition) | P reproduces 03's printed medians in 29/29 groups, to 03's digits | holds |
| **M0** | 7/7 rows: every gated field equal by `==` and bit pattern; `analytic_rad` equal; only `elapsed` differs | holds |
| **R1** | 7/7 rows. X regions / P regions: R1 0.0039, R2 0.0030, R3 0.0020, R4 0.0032, R5 0.0049, R6 0.0040, R7 0.0038. Largest count at depth ≥ 18 in a scored group: 0, 0, 0, 2, 3, 3, 3 | **holds** |
| **A1** | 14 covered groups (every scored lone `G`, `G+r`, `G−r` with F^A ≤ 1e-8): depth ≥ 18 count 0 in all 14 | **holds** |
| **N4** | \|P − X(1e-8)\| / `total_abserr`_P: R1 0.440, R2 0.0795, R3 **1.161**, R4 0.245, R5 0.445, R6 **1.533**, R7 **1.942** | **fails** on R3, R6, R7 |
| **R2** | \|X(1e-7) − X(1e-8)\| / `total_abserr`_X(1e-7): R1 0.405, R2 2.1e-4, R3 0, R4 0.305, R5 8.9e-7, R6 0.474, R7 2.5e-3 | **holds** (7/7) |
| **Q5** | \|X(1e-7) − X(1e-8)\| / \|X(1e-8)\|: R1 5.36e-7, R2 3.16e-9, R3 0, R4 6.73e-6, R5 1.41e-11, R6 8.21e-8, R7 3.04e-10. The 2026-09 sweep: 1.9e-6 (mid band), 2.84e-6 (low), 8.3e-7 (seam) | reported |

**Reading the scores.**
- **R1.** Findings §2.6 item 3's refutation clause ("item 3 leaves the depth at 20") does not fire.
  - With every remainder exact, no scored group stays at the cap.
  - The few regions at depth 18–19 are in `Gqr` groups on R4–R7.
  - The cost falls from 124–322 s to 0.8–2.1 s per row.
- **A1.** Layer 1(a) alone does what X does on these rows: the same 66–167 regions on every row.
  - That includes the `Gqr` groups, whose predicted F^A_g is 1.1–1.3 F.
  - So F^A_g, a maximum over the sub-interval, overstates the floor where those groups' deep
    regions lie. It is descriptive, not a score (`issues.md` I7).
- **N4.** The stored error does not cover the jitter's effect on three rows:
  - R3: 1.16×, with \|ΔP\|/\|P\| = 2.7e-4;
  - R6: 1.53×, with \|ΔP\|/\|P\| = 6.6e-7;
  - R7: 1.94×, with \|ΔP\|/\|P\| = 7.3e-7.

  X is not the truth, because P and X share φ and the modulus (step 5: φ_G is noisy at 0.15–0.78 F,
  I2). But P − X isolates the jitter's effect, and on these rows it exceeds the declared error.
- **R2 and Q5.** Q5 measures the self-consistency of X at the two rungs: R4's 6.7e-6 is the
  largest. It is not a distance from the truth.
  - Both rungs share the same φ error (I2), so neither number bounds it.
  - On R1, R4, R5, R6 and R7, X at 1e-8 still sends a short chain of regions to the cap: 2 regions
    at depth 20 per affected group, 4–10 per row (I6).

## What shipped

Nothing is committed. Everything is under `var/levin-phase-floor-04/`, and the checkpoints are in
the run directory.

**Scripts** (`scripts/`):
- `03_*.py`: prompt 03's eight scripts, as `git show c8ca155:docs/levin-phase-floor/03/scripts/…`,
  untouched.
- `prov.py`: both trees' `HEAD`s, for every output header.
- `hash_store.py`: 03's, with `prov.py`'s header.
- `harness.py`: 03's, with `job_name` `levin-phase-floor-04`.
- `rebuild.py`: 03's, plus an optional `rtol` argument to `rebuild_row`. The 1e-8 rung uses the
  store's own 1e-8 `tolerance` row.
- `run_instrument.py`: 03's, with both trees' provenance. Used for the route-(a) proof.
- `phases.py`: step 4. It holds:
  - the dispatcher at `PrimitivePhase.theta_mod_2pi`, with an optional deadline for the budget;
  - Δ^ext in mpmath at 512 bits;
  - the exact reduction against `TWO_PI`;
  - arms P, A and X;
  - Φ^arm and F^arm;
  - an exact wrap.
- `check_phases.py`: step 5. Its gates are amended per §8 (E1C_ADD = 4×2⁻⁴⁹; E2_ADD = 2⁻⁴⁸ plus
  ε·median Σ|dθ/du|).
- `diag_abscissa_rounding.py`: the diagnostic behind §8.
- `run_remove.py`: the registered run.
- `launch.py`: its detached launch.
- `analyse.py`: step 8, from the checkpoints and step 5's json.

**Outputs** (`out/`):
- `phases_04.txt`: step 5 under §8, with the final `phases.py`;
- `phases_04_first.txt`: the first dispatch's step 5;
- `phases_04_amended_phases_v1.txt`: §8's gates before the budget edit to `phases.py`, with an
  identical body;
- `diag_abscissa_rounding_04.txt`;
- `remove_04.txt`: step 8;
- `fingerprint_04.txt`.

**State** (`state/`): every command and its output.
- Step 1: `step1_trees.txt` and `pip_freeze.txt`.
- The store:
  - `store_files_before.txt`, `_at_stop`, `_resume`, `_prelaunch`, `_after` and
    `_after_fingerprint`.
- Steps 3 and 5:
  - `step3_*`;
  - `route_a_proof*`;
  - `phases_04*_cmd.txt` and `phases_04*_stderr.txt`;
  - `phases_04*.json`.
- Tests:
  - `trial/`;
  - `smoke*/` with their `.txt` outputs.
- The run:
  - `predictions.md` and `prompt_section5.raw`;
  - `registry_before_launch.txt` and `registry_after.txt`;
  - `scripts_sha256_prelaunch.txt`;
  - `launch.txt` and `launch_check.txt`;
  - `analyse_cmd.txt` and `analyse_stderr.txt`.
- `log_before_step10.md`: this log as it stood before step 10.

**The run directory:** `var/runs/levin-phase-floor-04-remove-20261025T231110/`.
- `manifest.json`, `status.json`, `checkpoint.jsonl` (28 records), `stdout.log`, and `stderr.log`
  (empty).
- `unit_{X,A}-*.pkl`, 21 pickles of 13–20 kB each. Each holds the result dict and every Levin
  call's region list (start, end, depth, type, abserr). All were kept: the region lists are small.

## Deviations from the prompt

**The first dispatch (stopped at step 5):**
1. **The phase objects are rebuilt through route (a), and the regions come from 03's pickles.**
   **IMPLEMENTATION CHOICE.** The worker forms `TkSourceFunctions`, `build_partition` and
   `build_phase_groups` exactly as `evaluate_QuadSource_integral` and `phase_group_Levin_integral`
   do. The rebuilt x_spans and labels are checked equal to 03's recorded calls.
2. **E2's reproduction of 03 uses 03's own population**, all depth-20 regions, because that is
   what 03 printed medians over. The gate itself uses the 20 sampled regions. **IMPLEMENTATION
   CHOICE.**
3. **The wrap in E1 is evaluated exactly** (`fractions`) on the two doubles. **IMPLEMENTATION
   CHOICE.**
4. **The exactness check was re-posed during the trial** (not a gate). It now checks that Δ^ext is
   exact and that ρ^X is the correctly rounded exact remainder. The first form failed where φ is
   subnormal. **IMPLEMENTATION CHOICE.**
5. **Descriptive columns in E2** (ε Σ|dθ/du|, |composed phase|) and `diag_abscissa_rounding.py`.
   **IMPLEMENTATION CHOICE.**
6. **The route-(a) proof used 03's `run_instrument.py`** on R6, into `state/route_a_proof/`.
   **IMPLEMENTATION CHOICE.**
7. **`black` wrote its cache** under `~/Library/Caches/black/25.1.0/`. **UNINTENDED DRIFT.** Per
   §8 this is not a stop, and `black` was not run again.
8. **Step 1 was a check, not a move.** **STRUCTURALLY REQUIRED.**

**The resume (§8):**
9. **The gates E1 (c) and E2 were amended**, exactly as §8 states. This is the user's decision, not
   a deviation of the agent's. It is recorded here because it changes the prompt's text after a
   measurement, as §8 itself says.
10. **The budget is enforced inside the dispatcher, not by `setitimer`.** **IMPLEMENTATION CHOICE**
    (the prompt says "or equivalent").
    - A SIGALRM handler's exception was swallowed in the first smoke test. mpmath's
      `libmpf.from_float` wraps `math.frexp` in a bare `except:` that returns NaN.
    - The dispatcher now raises `OverBudget` at the start of the first `theta_mod_2pi` call past
      the deadline.
    - Step 5 was re-run with the final `phases.py`, and its output body is identical.
    - No unit came near its budget: X and A took 0.4–3.0 s against budgets of 373–965 s.
11. **The R6 smoke tests ran before `state/predictions.md` was saved.** **UNINTENDED DRIFT.**
    - They ran between 23:04Z and 23:08Z, so X and A on R6 were seen first.
    - The predictions are §5's text unchanged, and A1's coverage list is fixed by step 5. So
      nothing in them could be influenced.
12. **The run's working directory is `var/levin-phase-floor-04`**, with the code from the worktree
    through `PYTHONPATH` and with no bytecode written. **IMPLEMENTATION CHOICE.** §8 made the
    registry's start directory immaterial, and this keeps any stray relative write out of the
    worktree.
13. **The provenance is §8's.**
    - The manifest's `git_head` is `c8ca155` and its `script_sha256` is `run_remove.py`'s
      (`3c43db3f…`), stamped on every record.
    - Every payload carries the worktree's `HEAD` `98e68d3` and its empty `git status --short`, plus
      the SHA-256 of all six scripts the run imports.
    - The purpose text names the worktree and `98e68d3`.
14. **A heartbeat thread runs every 60 s**, because a unit could outlast the 900 s staleness
    window. The manifest's `heartbeat_means` says so. **IMPLEMENTATION CHOICE.**
15. **`analyse.py` also reads step 5's `state/phases_04.json`** for scored status and F^arm_g, which
    §5 places beside the depths. Nothing is re-run. **IMPLEMENTATION CHOICE.**

## Verification performed

- **The trees.**
  - Main `HEAD` = `origin/handover-remedial` = `c8ca155` throughout, and `git status --short` is
    empty.
  - Worktree `HEAD` = `98e68d3`, and its status is empty, also with `--ignored`, after every run.
    Every checkpoint records the same.
- **The store.**
  - All six files' SHA-256, sizes and mtimes are identical at step 2, at the stop, at the resume,
    before launch, after the run and after the fingerprint.
  - `store fingerprint` without `--write` gives the digest
    `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, `problems: none`,
    "matches the fingerprint taken … at b90752d" (`out/fingerprint_04.txt`).
- **The run.**
  - It ended `done`, 28/28, exit 0, with `stderr.log` empty.
  - Its units are in §3's order.
  - The dispatcher saw no `other` instance in any unit.
  - Ray was never initialised: the run checks this at the end.
  - No lock was taken, and `~/.raypoolkit/` is empty.
- **The registry.** Nothing was `running` or stale before launch. After, this run is `done`
  (`state/registry_after.txt`).
- **M0.** It holds on all seven P units, which are also the budget base. P walls were 124–322 s,
  0.89–0.93 of stored `compute_time`.
- **Smoke tests on R6.**
  - P: M0 holds.
  - X and A, loose budget: the same totals as the run.
  - X and A, 0.3 s budget: recorded `over budget` with the calls completed so far.

## Observations not acted on

All of these are in `issues.md`:
- **I1:** the first gates' missing round-off terms. Resolved for this prompt by §8's amendment;
  kept as the record.
- **I2:** φ_G carries noise at 0.15–0.21 F (median) in the stored samples, and production's
  quantised φ_G reaches 0.46–0.78 F. Every arm keeps this error.
- **I3:** G's modulus error is 0.175 F.
- **I4:** `runregistrykit` takes `git_head` from the script's directory. Settled by §8.
- **I5:** the `black` cache.
- **I6:** X at 1e-8 still sends one short chain of regions to depth 20 in up to five groups per
  row.
- **I7:** A matched X in `Gqr` groups too, so F^A_g does not predict the depth.
- **I8:** N4 fails on R3, R6 and R7.
- **I9:** mpmath's bare `except:` in `from_float` returns NaN for a finite input on any exception.
  This is a hazard for any signal-based timeout around mpmath.

## State handed on

- **Nothing is running and no lock is held.** The run is `done`.
- **The checkpoints and pickles stay in the run directory**
  `var/runs/levin-phase-floor-04-remove-20261025T231110/`.
- **The store is unchanged**, fingerprint `0e32f24a…`.
- **For the orchestrator:**
  - copy `out/`, `scripts/`, `state/` (except the pickles under `state/route_a_proof/` and
    `state/smoke*/`), `log.md`, `verification.md` and `issues.md` to `docs/levin-phase-floor/04/`;
  - append `verification.md` as `VERIFICATION.md` §4.
- **For G1:**
  - R1 and A1 hold;
  - layer 1(a) alone reproduced the exact limit on all seven rows;
  - N4 fails on three rows;
  - φ_G's stored noise (I2) is an O(F) error that no consumer-side phase layer removes. That bears
    on Q3.
