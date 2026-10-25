# levin-phase-floor 04 — remove the floor from seven depth-20 rows of A3 v2

**Subject:** prompt 04, on the Mac Studio (mps026298). The main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit` was at the dispatch SHA `c8ca155`. The code ran from
the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf04-98e68d3` at `98e68d3`, by route (a),
with no Ray and no lock.

**Commit:** none here. The records are for the orchestrator.

**Result: STOPPED at step 5. Gates E1 and E2 fail by their letter, at round-off.** Prompt §5 says
*"A gate that fails is a stop, not a verdict"*, and §7 says *"E1, E2 or M0 fails"* is a stop. So
the following were not done:
- step 6 (`state/predictions.md`);
- step 7 (the registered run);
- steps 8–10, beyond these records.

What was found:
- **Route (a) works at `98e68d3`.** R6 was re-run and M0 holds, bit for bit, in 114.7 s.
- **E1 (a) and (b) hold everywhere.**
- **E1 (c) fails** on the lone-`G` sub-interval of six rows. The largest excess is 1.776e-15,
  which is one ulp of the arm-A value, against a bound of about 1e-15.
- **E2 holds under P** in all 29 scored groups.
- **E2 fails under A** in 5 groups, by ratios of 1.28–6.57. These are the lone `G` groups and R6's
  `G+q+r`.
- **E2 fails under X** ("≤ ε ΣΦ^X") in 19 groups, by ratios of 1.00–21.4. Its clause "≤ 1e-2 F"
  holds in all 29: σ_θ^X is at most 6.5e-14, which is under 1e-6 of that bound.
- **The X jitter is the abscissa's own rounding**, z = expm1(x). Its correlation with the
  prediction is 0.94–1.00. Φ^X leaves this term out.
- **P reproduces all 29 of 03's printed medians** to 03's digits.
- **A1 would cover 14 groups.**
- The store is byte-identical at the stop.

The question for the user is at the end of this log (State handed on).

## Predictions and verdicts

Prompt §5, copied unchanged below. It was copied after step 5 ran, because the gates it contains
had failed. **No verdict is scored**: the gates failed, so the scored predictions (R1, A1, N4, R2)
were never tested. `state/predictions.md` was **not** written; it is due only after step 5's gates
pass.

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


### Gate outcomes (step 5, `out/phases_04.txt`)

| Gate | Number | Outcome |
|---|---|---|
| **E1 (a)** | worst \|float(Δ^ext) − delta()\| / 2 ulp = 1.000 over 80,000 abscissae | holds |
| **E1 (b)** | worst \|wrap(ρ^X − ρ^P)\| / (4εΦ^P + 1e-15) = 0.870 | holds |
| **E1 (c)** | lone-`G` sub-interval of R1, R2, R4, R5, R6, R7: 45, 160, 23, 136, 148, 23 of 2000 abscissae over; largest lhs 1.776e-15 = 2⁻⁴⁹ at \|ρ^A\| ≥ 8; every other G factor ≤ 0.666 | **fails** |
| **E2, P** | σ_θ / ε ΣΦ^P = 0.224–0.341 in 29/29 | holds |
| **E2, A** | over in 5/29: R1, R4, R6, R7 lone `G` (3.86, 5.45, 6.57, 2.57), R6 `G+q+r` (1.28); others 0.21–0.72 | **fails** |
| **E2, X, ≤ ε ΣΦ^X** | over in 19/29, ratios 1.00–21.4; others 0.43–0.94 | **fails** |
| **E2, X, ≤ 1e-2 F** | σ_θ^X ≤ 6.5e-14 everywhere; largest σ_θ^X/(1e-2 F) = 9.6e-7 | holds |
| **E2's precondition, P reproduces 03** | 29/29 groups: median σ and σ/F to 03's 3 digits, and M1's σ/F to 4 | holds |
| **M0** | not reached (step 7), except R6 in the route-(a) proof: holds | — |

**Why the failures sit at round-off** (`issues.md` I1; `out/diag_abscissa_rounding_04.txt`):
- E1 (c)'s bound is below one ulp of the arm-A value. That value is the sum
  `WKB_mod_2pi(·)[1] + c_G`, which prompt §2 leaves unreduced in (−4π, 0].
- E2's model ε ΣΦ omits two terms:
  - the rounding of the abscissa, z = expm1(x) in `PrimitivePhase._get_x`, which enters every arm
    as about ε|dθ/du|. Predicted from the exact expm1 alone, it matches the X residual with
    correlation 0.9996–1.0000 on R3 and R4 `G−r`, and 0.94–0.97 on R1 `G+q+r`;
  - the granularity of a remainder sum of magnitude up to 4π, under A.
- Every failure is between 4e-16 and 6.5e-14 rad, against an F of 1.7e-6 to 9.2e-5.

## What shipped

Nothing is committed. Everything is under `var/levin-phase-floor-04/`.

**Scripts** (`scripts/`):
- `03_*.py`: prompt 03's eight scripts, as `git show c8ca155:docs/levin-phase-floor/03/scripts/…`,
  untouched.
- `prov.py`: both trees' `HEAD`s and the output header. It replaces 03's one-tree `git_head()`.
- `hash_store.py`: 03's, with the header from `prov.py`.
- `harness.py`: 03's, with `job_name` `levin-phase-floor-04`.
- `rebuild.py`: 03's, plus an optional `rtol` argument to `rebuild_row`, for the 1e-8 rung (the
  stored tolerance row 1e-8 is used).
- `run_instrument.py`: 03's, with both trees' provenance. Used for the route-(a) proof.
- `phases.py`: step 4. It holds:
  - the dispatcher, patched at `PrimitivePhase.theta_mod_2pi` and restored in a `finally`;
  - Δ^ext in mpmath at 512 bits;
  - the exact reduction against `TWO_PI`;
  - arms P, A and X;
  - Φ^arm and F^arm;
  - an exact wrap.
- `check_phases.py`: step 5, one spawned worker per row.
- `diag_abscissa_rounding.py`: a diagnostic of what the X jitter is.

**Outputs:**
- `out/phases_04.txt`: E1, the reproduction, E2, F^arm per group, A1's list, σ_φ and the modulus;
- `out/diag_abscissa_rounding_04.txt`.

**State:**
- step 1: `step1_trees.txt` and `pip_freeze.txt`;
- step 2: `store_files_before.txt`;
- step 3: `step3_copy_03_scripts.txt` and `step3_module_origins.txt`;
- the route-(a) proof: `route_a_proof.txt` and `route_a_proof/row_R6.pkl`;
- step 5: `phases_04_cmd.txt`, `phases_04_stderr.txt` and `phases_04.json`;
- `trial/`: the first single-row trial and the exactness debug that led to two changes in
  `check_phases.py`;
- at the stop: `store_files_at_stop.txt`;
- `prompt_section5.raw`.

## Deviations from the prompt

1. **The worker reads the phase objects from route (a), and the regions from 03's pickles.**
   **IMPLEMENTATION CHOICE.** Step 5 says "from the rebuilt inputs alone".
   - Each worker rebuilds the row through route (a). It then forms `TkSourceFunctions`,
     `build_partition` and `build_phase_groups` on Ray-style copies, exactly as
     `evaluate_QuadSource_integral` and `phase_group_Levin_integral` do.
   - E2's regions come from 03's pickles, as the prompt says.
   - The rebuilt sub-intervals' x_spans and group labels are checked equal to 03's recorded calls.
2. **E2's reproduction check is made on 03's population.** **IMPLEMENTATION CHOICE.**
   - 03 printed its medians over **all** of a group's depth-20 regions, not over the 20 sampled.
   - So the reproduction is computed on all of them under P, and compared with 03's printed
     strings. The gate itself uses the 20 sampled regions, as written.
3. **The wrap in E1 is evaluated exactly** (`fractions`), on the two doubles. **IMPLEMENTATION
   CHOICE.** A floating-point wrap adds a rounding of its own (≤ 4.4e-16) that belongs to neither
   arm. The first trial used the float wrap; the failures persist with the exact one.
4. **The exactness check against `fractions` was re-posed during the trial.** **IMPLEMENTATION
   CHOICE**, and not a gate.
   - The first form required θ itself to be exact at 512 bits. It failed where φ is subnormal
     (φ = 5e-324 at a sample, as a quantised φ_G allows), which needs about 1100 bits.
   - It now checks that Δ^ext is exact, and that ρ^X equals the correctly rounded exact remainder.
     Both hold at 80,000 of 80,000 abscissae.
5. **Descriptive columns were added to E2** (ε Σ|dθ/du|, and the median |composed phase| per arm),
   plus `diag_abscissa_rounding.py`. **IMPLEMENTATION CHOICE.** Neither enters a gate; they are
   there to state the stop's question precisely.
6. **The route-(a) proof used 03's `run_instrument.py`** with `--out-dir state/route_a_proof
   --rows R6`. **IMPLEMENTATION CHOICE.** This is "prove route (a) again, on one row". It wrote a
   5 MB pickle under `state/`.
7. **`black` wrote its cache** in `~/Library/Caches/black/25.1.0/`, outside the allowed area, when
   run once on four scripts. **UNINTENDED DRIFT.** It is harmless, and no further `black` run was
   made.
8. **Step 1 was a check, not a move**, as the orchestrator's note says. **STRUCTURALLY REQUIRED.**
   Both trees were as the Gate states (`state/step1_trees.txt`).

## Verification performed

- **Step 1.**
  - Main `HEAD` = `origin/handover-remedial` = `c8ca1559…`, and `git status --short` is empty.
  - Worktree `HEAD` = `98e68d3e…`, and its status is empty (also with `--ignored`, before and
    after every run).
  - `98e68d3` and `d5f5a2b` are ancestors of `HEAD`.
  - Python is 3.13.16.
- **Step 2, and at the stop.** The store files' SHA-256, sizes and mtimes are equal to each other
  and to 03's `store_files_after.txt`.
- **Imports resolve in the worktree.** `ComputeTargets`, `AdaptiveLevin`, `StoreFixtures` and
  `main.py` (through `test_main_plumbing.MAIN_PY`) all resolve there; the packages resolve in the
  venv (`state/step3_module_origins.txt`).
- **Route (a), R6.** M0 holds on every gated field, with `analytic_rad` equal too. 21 Levin calls
  were matched.
- **Step 5.**
  - Ray was never initialised: each worker checks this.
  - There was no Levin call.
  - The dispatcher saw no `other` instance: every `PrimitivePhase` call was classified G or T.
- **No lock, and no registry write.** No lock was taken. `runregistrykit begin` was never called.

## Observations not acted on

All of these are in `issues.md`:
- I1: the gates' missing round-off terms.
- I2: φ_G at the floor's own scale. It is in the stored samples, at 0.15–0.21 F (median, rebuilt
  in mpmath). Production's quantised φ_G reaches 0.46–0.78 F. That is a smooth phase error which X
  keeps.
- I3: G's modulus error is 0.175 F on every row.
- I4: `runregistrykit` takes `git_head` from the script's directory, which bears on step 7's
  provenance.
- I5: the `black` cache.

## State handed on

- **Nothing is running.** No registry run exists for this prompt. The lock was never taken.
- **The store is unchanged.** `state/store_files_at_stop.txt` equals `state/store_files_before.txt`.
- **The run is ready to write but not written.** The run script, its launcher and the analysis do
  not exist yet; step 7 was not reached. The design intended:
  - one process;
  - the dispatcher installed around `QuadSourceIntegral.evaluate_QuadSource_integral` (a wrapper at
    that module attribute);
  - a `setitimer` budget raising a `BaseException` subclass;
  - 28 units, each with a ledger record;
  - per-unit pickles for X and A.
  For provenance: `begin(script=None)`, with the launcher's working directory in the worktree, so
  that the manifest records `98e68d3`; the scripts' SHA-256 go in every record's data (I4).

**The question for the user.** E1 (c) and E2 fail by their letter. Every failure is at round-off,
4e-16 to 6.5e-14 rad, against an F of 1.7e-6 to 9.2e-5. The causes:
- the arm-A value's own ulp (E1 (c));
- the abscissa rounding z = expm1(x), which the Φ model omits (E2 under X and A).

Should the prompt:
1. proceed to the run as written, treating these as gate failures below the scale the gates
   protect; or
2. have E1 (c) and E2 re-posed with those two terms first; or
3. stop here?

---

## Resume under prompt §8 (2026-10-25, 23:00Z onward)

**Where §8 came from.** Prompt §8 was added in commit `d170e6d` on `origin/handover-remedial`. It
reached the agent verbatim, through the orchestrator. Neither checkout was moved and nothing was
fetched: the main checkout stays at `c8ca155`, because `handover` A3 runs in it.

**What the resume did, in order:**
1. **Checked the store again.** `state/store_files_resume.txt` is identical to the step-2 hashes.
2. **Kept the first dispatch's step-5 records** under new names:
   - `out/phases_04_first.txt`;
   - `state/phases_04_first.json`;
   - `state/phases_04_first_cmd.txt` and `state/phases_04_first_stderr.txt`.
3. **Amended the gates** in `check_phases.py` as §8 states, and changed nothing else there:
   - E1 (c)'s additive term is now 4×2⁻⁴⁹;
   - E2's bound is now ε·med ΣΦ^arm + ε·med Σ|dθ_i/du| + 2⁻⁴⁸, with "≤ 1e-2 F" under X unchanged.
4. **Re-ran step 5.**
   - **E1 holds.** The worst ratios are 1.000 for (a), 0.870 for (b) and 0.379 for (c).
   - **E2 holds.** The bound ratios are 0.224–0.341 under P, 0.104–0.772 under A and 0.046–0.203
     under X.
   - **The reproduction of 03 holds.**
   - **A1 covers the same 14 groups.** The F^arm section is identical to the first dispatch's.
5. **Moved the budget check into `phases.py`.** The dispatcher now takes an optional deadline, and
   step 5 was run once more with the final file:
   - that is `out/phases_04.txt`;
   - its body is identical to the run before the edit, which is kept as
     `out/phases_04_amended_phases_v1.txt`.
6. **Smoke-tested `run_remove.py`** with `--out-dir` under `state/`, on R6 only:
   - `state/smoke*`: P (M0 holds) and X and A at 1e-7;
   - `state/smoke_v2_*`: the tight-budget path and the loose-budget path, for X and A at 1e-7 and
     X at 1e-8.
7. **Saved `state/predictions.md`** at 23:10:41Z. It holds §5 unchanged, §8's amended gates and
   A1's 14 groups.
8. **Checked the registry** (`state/registry_before_launch.txt`): nothing `running` or stale.
9. **Checked the store again before launch** (`state/store_files_prelaunch.txt`): identical.
10. **Launched** at 23:11:10Z (`state/launch.txt`), and made one check (`state/launch_check.txt`).

**The run:**
- **Run id:** `levin-phase-floor-04-remove-20261025T231110`.
- **Run directory:** `var/runs/levin-phase-floor-04-remove-20261025T231110/`.
- **Detached Python pid:** 79510. `nohup` execs Python in the same process; there is no `sh`.
- **Manifest:** `git_head` `c8ca155` and `script_sha256` `3c43db3f…`, per §8. Every checkpoint
  payload carries the worktree's `HEAD` `98e68d3` and its empty `git status --short`.

**Deviations of the resume:**
- **The budget is enforced inside the dispatcher, not by `setitimer`.** **IMPLEMENTATION CHOICE**
  (the prompt allows "or equivalent").
  - In the first smoke test, the SIGALRM handler's exception was swallowed in one unit: X on R6
    ran 1.4 s against a 0.3 s budget and reported "done". The other unit's alarm, A on R6, was
    caught as intended.
  - The cause is mpmath's `libmpf.from_float`. It wraps `math.frexp` in a bare `except:`, which on
    any exception returns NaN for a finite input. An alarm landing there is swallowed silently,
    and can corrupt a value.
  - So the dispatcher now checks `time.perf_counter()` against the deadline at the start of every
    `theta_mod_2pi` call, and raises `OverBudget` from there.
  - The X/A smoke results with and without that first alarm are identical.
- **The R6 smoke tests ran before `state/predictions.md` was saved.** **UNINTENDED DRIFT.**
  - They ran between the gates passing (23:03Z) and the predictions being saved (23:10:41Z), so
    X and A on R6 were seen first.
  - The predictions text is §5 unchanged, and A1's list is fixed by step 5. So nothing in
    `predictions.md` could be, or was, influenced.
- **The run's working directory is `var/levin-phase-floor-04`, not the worktree.** **IMPLEMENTATION
  CHOICE.** The code is the worktree's, through `PYTHONPATH`, with `-B` and
  `PYTHONDONTWRITEBYTECODE`. §8's provenance makes the registry's starting directory irrelevant,
  and this keeps any stray relative write out of the worktree.
- **A heartbeat thread runs every 60 s.** **IMPLEMENTATION CHOICE.** A unit may outlast the
  registry's 900 s staleness window. The manifest's `heartbeat_means` says so.

Steps 8–10 follow when the run ends.
