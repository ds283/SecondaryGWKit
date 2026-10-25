# levin-phase-floor 03 — instrument seven depth-20 rows of A3 v2

> **Written on the Mac Studio, committed on the laptop** by the orchestrator (README §4.1).
>
> **Commit:** `d5f5a2b`, the records commit for prompt 03.
>
> A path `scripts/…`, `out/…` or `state/…` below is under `var/levin-phase-floor-03/` there, and is
> committed at `docs/levin-phase-floor/03/`, except for what stays on the Mac Studio:
> - the fixture-proof store `state/fixture_proof_store/` (three SQLite files), which
>   `.git/info/exclude` keeps out of git (`*.sqlite`);
> - the smoke pickles, and the run's pickles `row_R1.pkl` … `row_R7.pkl` in
>   `var/runs/levin-phase-floor-03-instrument-20261025T194950/`, which were not copied back.
>
> `verification.md` is `docs/levin-phase-floor/VERIFICATION.md` §3, unchanged. The text below is
> the agent's, except that its two links are re-pointed to resolve from this file. The
> orchestrator's review is appended at the end.

**Subject:** prompt 03, run on the Mac Studio (mps026298) in `/Users/ds283/Documents/Code/SecondaryGWKit` at the dispatch SHA `98e68d3`. Route (a), with no Ray and no lock, and a registered run `levin-phase-floor-03-instrument-20261025T194950`.
**Commit:** none here; records committed by the orchestrator.
**Result:**
- **M0 holds** on all seven rows, bit for bit.
- **M1, M2, M4 and M5 hold**; M5 is inapplicable on R7. **M3 fails, on its slope clause alone.** The control discrepancy is at round-off at every depth, so the clause cannot be met. Its ratio clause holds, with a factor of 3e6 to 2e8.
- **The mechanism verdict as written therefore does not hold**, and the prompt's refutation clause fires by its letter. Note (a) gives the reasons to take that clause as badly posed.
- **H-anchor holds in every scored group.** H-cancel's condition is met, but the non-cancelling groups carry the same floor.
- **Every depth-20 region is Clenshaw–Curtis.**
- The store is byte-identical, fingerprint `0e32f24a…`.

## Predictions and verdicts

The prompt's §3, copied unchanged before step 6's launch, is below. It is also in `state/predictions.md`, whose mtime of 19:48:38Z precedes the 19:49:50Z launch. This section was first written at 19:48:46Z with the same text, and the verdicts follow it in their own subsection.

## 3. Predictions, fixed before the re-runs (README §5 rule 3)

Copy these into the log unchanged, and score them there.

**The groups scored.** A group is **scored** if it has at least 10 regions at depth 18 or more.
Report every group, scored or not.

- **M1 — the jitter is there, at the floor.**
  - **The rule:** in every scored group of every row, the median $\sigma_\theta$ over its depth-20
    regions, divided by the row's $F$, lies in $[0.1, 10]$.
  - **H-anchor** needs M1 in the `G-r` groups (R4, R5) and the lone `G` (R6) as well as in `G+r`.
  - **H-cancel** needs it only in the cancelling groups.
- **M2 — the discrepancy does not fall with refinement.**
  - **The rule, part 1:** over the ancestry spans at depths 12 to 20, the least-squares slope of
    $\log D$ on $\log h$, real phase, lies in $[-0.5, 0.5]$ for the median scored group of each row.
  - **The rule, part 2:** at depth 20, the median $D/\sigma_\theta$ lies in $[0.01, 10]$.
  - **Holds** if both parts hold on at least six of the seven rows.
- **M3 — remove the jitter and it converges.**
  - **The rule:** in the jitter-removed control, the same slope is **at least 2**, and the median
    $D$ at depth 20 is at least 100 times smaller than with the real phase.
  - **Holds** if true on at least six of the seven rows.
  - If M2 holds and M3 fails, the discrepancy is not the jitter's, and the floor mechanism is
    **refuted**.
- **M4 — the driver cannot see it.**
  - **The rule:** among depth-20 regions of scored groups, the share with
    `phase_err` $\ge$ `abserr`, so that the region would have been accepted as `phase_limited`, is
    below 0.1.
  - **Also report, descriptive:** the share of depth-20 regions that are Clenshaw–Curtis. §2.7
    layer 2 must reach whichever branch carries them.
- **M5 — how far `G+r` cancels** (`[01-slow-group-leading-terms-do-not-cancel]`).
  - **The rule:** on R1–R5 and R7, $|\theta'_{G+r}|/|\theta'_{G-r}|$ matches $(1-c_s)/(1+c_s)$ to
    within 5 % at all five abscissae.
  - If it holds, the issue stands as written. If the ratio is near 0 instead, the cancellation is
    exact and the issue is wrong.

**The verdict for the mechanism.** "A non-smooth phase floor, invisible to the driver, drives the
bisection to the cap" **holds** if M1, M2, M3 and M4 hold. Score H-anchor and H-cancel from M1–M2
group by group, as the rules above say.


### Verdicts

Scored against the text above, as written, from [`out/instrument_03.txt`](../../../docs/levin-phase-floor/03/out/instrument_03.txt). The operational choices the text leaves open were fixed in `scripts/analyse.py`'s docstring before anything was computed (deviation 13). Every group is reported there, scored or not: Of 45 groups, 29 are scored and 16 are not. Every unscored group has fewer than 10 regions at depth ≥ 18 and none at depth 20; 13 of them have one region.

| | Rule (prompt §3) | Number | Verdict |
|---|---|---|---|
| **M0** (§2 step 5) | bit-identity on the listed fields | 7 of 7 rows equal under `==` and IEEE-754 bit pattern; `analytic_rad` equal on all 7; only the 45 wall-clock `elapsed` fields differ (deviation 6) | **holds** |
| **M1** | median σ_θ(depth 20)/F in [0.1, 10] in every scored group of every row | 29 of 29 scored groups in range; values 0.224–0.342 (medians: cancelling 0.270, n = 18; non-cancelling 0.262, n = 11; ratio 1.03) | **holds** |
| **M2** part 1 | median over scored groups of the real-phase slope of ln D on ln h, depths 12–20, in [−0.5, 0.5] | per row 0.027 (R6) to 0.089 (R3); every group's own slope lies in [−0.005, 0.174] | in range on 7/7 rows |
| **M2** part 2 | median D/σ_θ at depth 20 in [0.01, 10] | per row 0.078 (R3) to 0.113 (R6); groups 0.066–0.148 | in range on 7/7 rows |
| **M2** | both parts on ≥ 6 of 7 rows | 7 of 7 | **holds** |
| **M3** | control slope ≥ 2 **and** median D(depth 20) real/control ≥ 100, on ≥ 6 of 7 rows | slope, per row: −0.020 to 0.148, so < 2 on 7/7 rows. Ratio, per row: 3.1e6 to 2.4e8, so ≥ 100 on 7/7 rows. Both together: 0/7 | **fails**, on the slope clause alone. See note (a): the clause is badly posed. It is recorded as such and not re-posed |
| **M4** | among depth-20 regions of scored groups, the share with `phase_err` ≥ `abserr` below 0.1 | pooled 1611 / 189,196 = **0.0085**; per row 0.0026 (R3) to 0.021 (R6) | **holds** |
| M4, descriptive | the share of depth-20 regions that are Clenshaw–Curtis | **189,196 / 189,196 = 1.000**, in every scored group of every row | — |
| **M5** | on R1–R5 and R7, \|θ'_{G+r}\|/\|θ'_{G−r}\| matches (1−c_s)/(1+c_s) within 5 % at all five abscissae | R1–R5: 0.26795–0.26885 against 0.267949 (c_s² = 0.333333), worst deviation +0.34 %. R7 has no `Gr` sub-interval, so the rule cannot be applied there (deviation 7) | **holds** on the five rows it applies to; inapplicable on R7. The issue stands as written. The ratio is not near 0, so the cancellation is not exact |

**The mechanism verdict**, as written: "holds if M1, M2, M3 and M4 hold". **M3 fails, so the verdict as written is that it does not hold.** The prompt's own clause also applies by its letter: "If M2 holds and M3 fails, the discrepancy is not the jitter's, and the floor mechanism is **refuted**." I record both as the rule gives them. Note (a) says why I take the slope clause to be badly posed. Whether the clause stands is for the orchestrator and the user; I do not re-score it.

**(a) Why the M3 slope clause cannot be met here.** All 6380 ancestry spans are Clenshaw–Curtis spans, under both the real and the control phase. They are 580 sampled depth-20 regions, each with its 11 aligned spans at depths 10–20. With the jitter removed, the nested-pair discrepancy is already at round-off at depth 10: the median D is about 2e-16 to 1e-13 at every depth 10–20 in every scored group (the ancestry table in `instrument_03.txt`). A discrepancy at the round-off floor at every depth cannot fall with h at any slope. So "slope ≥ 2" tests something that is already settled before depth 12. It cannot separate "converges once the jitter is removed" from "does not converge". The clause assumes algebraic convergence above a floor. A 24/47-point nested Clenshaw–Curtis pair on a smooth integrand over these spans is spectrally converged by depth 10.

The ratio clause carries the comparison the prompt intends. Removing the jitter, and nothing else, lowers the depth-20 discrepancy by a factor of 3e6 to 2e8 on every row. The control's degree-12 fit residual is 0.5e-6 to 3e-5 rad, the same size as σ_θ, so the control removes exactly the non-smooth part and leaves the rest unchanged. The Levin/Clenshaw–Curtis routing is unchanged on all 6380 spans. This is evidence *for* the discrepancy being the jitter's, which is the opposite of what the literal clause concludes. Recorded as `issues.md` I1.

**H-anchor and H-cancel, group by group** (the last table of `instrument_03.txt`). M1 and both parts of M2 hold in **every** scored group:
- in the cancelling groups (18): `G+r` on R1–R5, `G±q+r` on R1, R2, R4 and R5, and `G+q+r`, `G+q−r` and `G−q+r` on R6 and R7;
- in the non-cancelling groups (11): the lone `G` on R1, R4, R6 and R7, `G−r` on R1–R5, and `G−q−r` on R6 and R7.

The verdicts that follow:
- **H-anchor needs M1 in `G−r` (R4, R5), the lone `G` (R6) and `G+r`: holds.** R4 `G−r` 0.256, R5 `G−r` 0.225, R6 `G` 0.337, `G+r` 0.225–0.314. M2 holds in each of them too.
- **H-cancel needs M1 only in the cancelling groups. Its condition is met**, since it holds there. But it is met equally in every non-cancelling group. The jitter is the same size in both classes: median σ_θ/F 0.270 against 0.262, a ratio of 1.03, and within a row the groups agree to a few per cent. So the cancellation adds no jitter beyond what every group containing G carries. **H-cancel does not account for the non-cancelling groups**, which carry the same floor at the same depth. As 01 found, it fails as the whole account.

## What shipped

Nothing in the tree. Records only, under `var/levin-phase-floor-03/`, for the orchestrator to copy to `docs/levin-phase-floor/03/`.

- **`scripts/`**:
  - `harness.py`: route (a), and the pickler;
  - `rebuild.py`: step 4;
  - `run_instrument.py`: the wrappers, M0 and the pickles;
  - `launch.py`: the registered, detached launch;
  - `analyse.py`: step 7;
  - `fixture_proof.py`: step 3 (a);
  - `smoke_check_pickle.py`, `hash_store.py`.

  All are formatted with `black`. Every script the registered run used was formatted *before* the run, so each copy is byte-identical to the file whose SHA-256 its outputs record.
- **`out/`**:
  - `instrument_03.txt`: step 7, every group;
  - `m0_03.txt`: M0, field by field, for all seven rows;
  - `fingerprint_03.txt`: `store fingerprint` without `--write`.
- **`state/`**:
  - the Gate, `pip freeze`, and the store hashes before, at launch, after, and after the fingerprint;
  - the fixture proof and its transcripts;
  - the smoke tests (`smoke/`, `smoke2/`);
  - the predictions, the launch and the launch check;
  - the analysis command, and the registry and lock state after.
- **The registered run** `levin-phase-floor-03-instrument-20261025T194950`, in `var/runs/` (pickles `row_R1.pkl` … `row_R7.pkl`, 4.9–9.2 MB each, kept there):
  - launched 19:49:50Z and ended `done`, exit 0, at 20:11:08Z, with 7/7 units and an empty `stderr.log`;
  - the per-row compute wall time was 114.9–299.7 s, against 134.4–355.2 s in production.

## Deviations from the prompt

1. **Step 1 was a check, not a move.** `STRUCTURALLY REQUIRED`: the laptop pins the tree.
   - The orchestrator had already fast-forwarded the checkout to `98e68d3`. No git command that writes was run.
   - `HEAD` = `98e68d3e…` = `origin/handover-remedial`, `git status --short` was empty, and `pip freeze` was recorded once (`state/gate_03.txt`, `state/pip_freeze.txt`).
2. **The first store-hash command's output was lost.** `UNINTENDED DRIFT`.
   - The first attempt at step 2 used `stat -f`. This host's PATH finds GNU `stat`, which rejects it.
   - That transcript was overwritten when `state/store_files_before.txt` was rewritten by `scripts/hash_store.py`, so it is not kept.
   - The hashes themselves were taken by `hash_store.py` before anything opened the store, and they equal 01's.
3. **Fixture proof: efold columns supplied.** `STRUCTURALLY REQUIRED`.
   - The default `real_store_fixtures` exit-time rows carry `z_exit` only. `TkNumericIntegration`'s constructor reads `z_exit_suph_e3`, so the default store cannot return a `QuadSourceIntegral` input.
   - The efold columns were supplied through `build_real_store`'s own `replicated=` argument, in the form `standin_pool.make_exit_time` uses.
   - The proof looks up one input class, `TkNumericIntegration` (a Tq/Tr input), together with the replicated classes the keys need. The other input classes are proven on A3 v2 itself, by M0.
   - The fixture was built under `state/fixture_proof_store/`, not a `TemporaryDirectory`, and kept for re-hashing. Its transcripts are `state/fixture_proof.txt` and `state/fixture_proof_final.txt`, the latter re-run on the final scripts.
4. **What the harness stands in, beyond `standin_pool`.**
   - (i) **`StandinRef` and `_copy` use `ray.cloudpickle`.** `STRUCTURALLY REQUIRED`: standard `pickle` cannot serialise a `GkSource`'s `NumericData`, a namedtuple bound under another name, and Ray uses cloudpickle.
   - (ii) **`ray.put` returns a cloudpickled ref.** `STRUCTURALLY REQUIRED`: `ModelProxy`, `GkSourceProxy` and `BesselPhaseProxy` call it.
   - (iii) **`compute_QuadSource_integral.remote` runs the undecorated production `_function` in-process,** on cloudpickle copies of its arguments as Ray ships them. `QuadSourceIntegral.compute()` itself is called, so all of production's pre-compute checks run. `IMPLEMENTATION CHOICE` (the form); the substitution itself is `STRUCTURALLY REQUIRED` by route (a).

   All are `mock.patch`es of module attributes, scoped by a context manager. Ray was never initialised, and the harness checks this.
5. **`OMP_NUM_THREADS=1` for the run.** `IMPLEMENTATION CHOICE`.
   - Ray sets this for a worker with the default one CPU, which is how `compute_QuadSource_integral` ran in production.
   - numpy here uses Accelerate, so the setting probably has no effect. M0 shows the result is bit-identical either way.
6. **M0 does not gate the group records' `elapsed`.** `STRUCTURALLY REQUIRED`.
   - The prompt lists "every group record", and `elapsed` is a wall-clock time, which no re-run reproduces. This was stated before the launch.
   - Every other field of every group record is gated (`label`, `value`, `abserr`, `converged`, `phase_limited`, `regions`, `simple_regions`, `evaluations`), and so are the sub-interval records' own fields. All are equal under `==` and by bit pattern.
   - `elapsed` is reported. All 45 differ: the re-run took 0.76–0.96 of the stored time, median 0.83.
   - `analytic_rad`, `total_phase_limited`, `b` and the other `LevinData` counters are reported but not gated. All are equal.
7. **M5 is inapplicable on R7, and R6 is not scored either.** `STRUCTURALLY REQUIRED`.
   - Both R6 and R7 are `q == r == k`. Their partition is `G` then `Gqr`, with **no `Gr` sub-interval**, as 01's records already showed.
   - M5's rule names "R1–R5 and R7" and "its `Gr` sub-interval", so it cannot be applied to R7 as written. That part of the rule is recorded as badly posed, not re-posed.
   - M5 is scored on R1–R5. §2.7's "every row with `r == k`" likewise reaches R6 only in name. Stated before the launch.
8. **Rebuild: what is `main.py`'s own code and what is mirrored.** `STRUCTURALLY REQUIRED` for the closures; `IMPLEMENTATION CHOICE` for the direct calls.
   - `main.py`'s module-level functions are used through `load_main_py_functions`: `cosmology_feature_redshifts`, `pre_grid_background_proxy`, `source_grid_spacing_profile`, `build_grid_tag_labels`, `closes_triangle` and `build_QuadSourceIntegral_payload`. Their globals are `main.py`'s own import block, executed from its source.
   - `run_pipeline`'s closures cannot be extracted, so they are mirrored statement for statement, with line citations. That covers the set-up, `resolve_GkSource_parents`, `resolve_QuadSourceIntegral_parents`, the existence query and the full lookups.
   - Each `RayWorkPool` becomes the same pool call, made directly and in the same item order.
   - A3 v2's arguments come from its build manifest and `quadsource_atol_sweep.py --build`.
   - **The work object is not got by `object_get`.** For a stored row, `object_get` returns the stored object. The work object is constructed as the factory's miss branch constructs it (`QuadSourceIntegral(payload=None, …, **parents)`) with `main.py`'s arguments.
   - Every check the rebuild makes passed on all seven rows (`stdout.log` of the run):
     - the grid-tag labels equal the stored tags;
     - every rebuilt z equals the stored z exactly;
     - the six parents, the z serials, the model, tolerance and policy serials all equal the stored row's;
     - the existence query returns the row by its serial.
9. **The wrappers.** `IMPLEMENTATION CHOICE` (the identity check).
   - Each wrapper calls the original with exactly its arguments and returns its return object, recording references only.
   - Both attributes are restored in a `finally`, and asserted restored after each row.
   - The group label is parsed from `notify_label` by regex.
   - Each Levin call is matched to its `PhaseGroup` by (regime, label), and the match is **confirmed by identity**: the call's `theta["theta_mod_2pi"]` is that group's function. This held on every group call of every row (the 23 or 21 calls per row include the eight `analytic J/Y` calls, kept with label kind `other`).
10. **The `PhaseGroup` pickles.** `STRUCTURALLY REQUIRED`.
    - `f_sin` and `f_cos` close over `build_phase_groups`' local `lru_cache`-wrapped `terms_at`, which `ray.cloudpickle` cannot pickle.
    - A reducer pickles such a wrapper as its `__wrapped__` function plus its cache size, and rebuilds an empty cache on load. The function is pure, so an empty cache returns what a full one does.
    - Proven on the smoke rows (`state/smoke2/smoke_check_pickle.txt`). On the real run, the recomputed driver discrepancy equals the recorded `region.abserr`, type included, at all 580 sampled depth-20 regions.
11. **Smoke tests before the launch.** `IMPLEMENTATION CHOICE`.
    - The whole pipeline ran on two cheap rows outside the seven, (1, 1844) and (2, 21681), once before and once after `black` (`state/smoke/`, `state/smoke2/`).
    - M0 held bit for bit on both, `analytic_rad` included.
12. **The launch was made from Python.** `IMPLEMENTATION CHOICE`.
    - `launch.py` called `runregistrykit.begin` first (manifest, `checkpoint=True` ledger, no `results`: the store is input, not results). It then ran `subprocess.Popen(["nohup", python, run_instrument.py, "--run-dir", …], start_new_session=True)`, with stdout and stderr appended to the run directory and stdin from /dev/null, and called `Run.heartbeat(pid=child)`.
    - No `sh -c` was used. `nohup` execs Python, so pid 87581 was the Python process (ppid 1).
    - One check that it started (`state/launch_check.txt`), then the turn ended.
13. **The analysis's operational definitions**, fixed in `analyse.py`'s docstring before it ran. `IMPLEMENTATION CHOICE`.
    - F uses 01's method (`depth20_floor.eta`). The `F` column of `instrument_03.txt` agrees with the prompt's table to its three figures.
    - Ancestry samples are up to 20 depth-20 regions per group, evenly spaced through the list sorted by start. The spans are replayed from the call's `x_span` with the driver's own `c = (b + a)/2.0`, and the depth-20 span equals the region's endpoints exactly (asserted).
    - The slope is one least-squares fit per group, pooled over its samples at depths 12–20, with spans where D = 0 excluded and counted. Per row, M2 and M3 use the median over scored groups.
    - D/σ_θ: per group, the median over all depth-20 regions; per row, the median over scored groups.
    - The M3 ratio is per group (median of the real D at depth 20 over the sample, divided by the median of the control's), then the median over the row's scored groups.
    - M4 is scored on the pooled share, with each row reported.
    - M5 uses abscissae at j/6 of the `Gr` x_span, j = 1..5, with c_s = √`wPerturbations`.
    - The control's degree-12 fit uses numpy's Chebyshev basis (a degree-12 polynomial), domain the depth-10 span, on N ≥ 4097 equispaced, unwrapped samples. N was 4097 everywhere, and the largest unwrapped step was 4.6e-3 rad.
    - `theta`, `theta_deriv` and f are unchanged in the control.
    - σ_θ is computed on **all** 189,196 depth-20 regions of scored groups, not a sample.
14. **The ancestry used the driver's own single-region routines.** `IMPLEMENTATION CHOICE`, the preferred option of §2.7.
    - `AdaptiveLevin.levin_quadrature._adaptive_levin_subregion` was called on each span with the group's f and theta dict, at `CHEBYSHEV_ORDER` = 24 and the call's rtol.
    - On a Levin span the discrepancy is |v₀ − v_L − v_R| from three calls. On a Clenshaw–Curtis span it is the routine's own `abserr_direct`.
    - It can be called cleanly, as the exact reproduction of every recorded `abserr` shows.
    - In the event, all 6380 spans were Clenshaw–Curtis.
15. **`analyse.py` re-implements 01's `cancelling` rule** (`group_census.py:105-115`), with the same logic. `IMPLEMENTATION CHOICE`. On R6 and R7 (`q == r == k`) it marks `G+q−r` and `G−q+r` cancelling, as 01 did.
16. **The analysis was not a registered run.** `IMPLEMENTATION CHOICE`. It took 314 s, under the ten-minute threshold. It ran in the background with its command, timing and exit code in `state/analyse_cmd.txt` and `state/analyse_stderr.txt`.

17. **The analysis was run twice.** `UNINTENDED DRIFT`, corrected.
    - The first run's `analyse.py` docstring cited the driver's discrepancy lines wrongly, as `levin_quadrature.py:1946-1954, 2117-2131`; they are `:1984` and `:2129-2131`.
    - The docstring was corrected, and the analysis re-run so that its output records the hash of the script that is kept (CLAUDE.md rule 7: recompute, never re-stamp).
    - The two outputs differ only in that hash line. The first is kept in `state/superseded_run1/`, with its command and timing.

## Verification performed

- **Gate:** every item as stated (`state/gate_03.txt`).
- **Store:**
  - hashed at step 2, at launch, after the run, and after the fingerprint. All four are identical in SHA-256, size and mtime_ns, and equal to 01's.
  - `store fingerprint` without `--write`: digest `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, problems none, "recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00" (`out/fingerprint_03.txt`).
  - The pool was read-only: writes raise `ReadOnlyWrite`, misses raise `ReadOnlyMiss`, and every connection is `mode=ro`.
- **Route (a):**
  - the fixture proof passed, twice;
  - Ray was never initialised, and the lock was never taken. It is free before and after (`state/registry_after.txt`). `ray-processes` does not apply.
- **M0:** 7 of 7 rows bit-identical (`out/m0_03.txt`); M0 also held on the two smoke rows.
- **Harness fidelity in step 7:**
  - all 580 sampled depth-20 regions reproduce their recorded `abserr` and type exactly from the pickles alone;
  - the replayed spans reproduce the regions' endpoints exactly.
- **Unwrap validity:** no depth-20 region has max|θ'|h/64 > π/2. σ_θ at degree 8 is within 7 % of degree 4 in 27 of 29 scored groups, so there σ_θ measures jitter, not a poorly fitted smooth phase. The exceptions are R3's two groups (k = 9.56e7), where degree 8 is 16–26 % lower; a part of their degree-4 residual is smooth, and M1 holds there either way.
- **Registry:** the run is `done`, 7/7 units, exit 0. No other run was touched.
- **Tree:** `git status --short` was empty before and after. No production code, test file or `AdaptiveLevin/` file changed.

## Observations not acted on

Recorded in [`issues.md`](../../../docs/levin-phase-floor/03/issues.md) for the orchestrator to place.

1. **I1, the M3 slope clause.** It is badly posed: the control's discrepancy is at round-off at every depth 10–20, so a slope of 2 cannot occur. By its letter, the clause "M3 fails ⇒ refuted" fires. Note (a) above.
2. **I2, every depth-20 region is Clenshaw–Curtis**: 189,196 of 189,196 in the 29 scored groups. Across all groups, in all seven rows, no depth-20 region is a Levin region.
   - The bisection that reaches the cap is entirely in the `is_direct` branch of `_adaptive_levin`.
   - There, `phase_err` (eq. 151) exceeds `abserr` in only 0.85 % of regions, so `phase_limited` is out of reach in more than 99 % of them.
   - §2.7 layer 2 must act on the Clenshaw–Curtis branch. That branch is also where `[10-levin-wholesale-cc-fallback]` (OPEN_ISSUES §1.3) lives, which campaign README §0.1 puts out of scope. A scope question for G1.
3. **I3, `StoreFixtures`.** `standin_pool` serialises with plain `pickle`, which cannot carry A3 v2's `GkSource`. `real_store_fixtures`' default exit-time rows lack the efold columns a `TkNumericIntegration` needs. Both are worked round here, inside the harness, without editing either file.
4. **I4, `PhaseGroup` is not picklable**, because of the `lru_cache` on a local function. A future harness (04) that keeps groups needs the reducer in `harness.py`.
5. **I5, the prompt's "Read first" item 4 is inaccurate.** It says each accepted `used_interval` records "the `abserr_history` and `relerr_history` of its ancestors". It does not: only the internal `_levin_interval` carries them, and they are dropped on acceptance. Step 7's ancestry recomputes them, so nothing was lost.
6. **Descriptive.**
   - σ_θ/F is 0.22–0.34 in every scored group. D/σ_θ is 0.07–0.15. The real-phase discrepancy at depths 10–20 is flat to within the scatter of the sample, a slope of −0.005 to 0.17.
   - On R6 and R7, `G+q−r` and `G−q+r` have identical records, because their composed phases coincide when q = r.
   - The read-only pool prints a shard-count warning (10 against 4) because, like `extract_QuadSourceIntegral_data.py`, it passes no `shards`. It is cosmetic.

## State handed on

- The store is unchanged, fingerprint `0e32f24a…`. The lock is free, and the registry holds 11 runs, all finished.
- The pickles stay in `var/runs/levin-phase-floor-03-instrument-20261025T194950/`. Some bulk stays in `var/levin-phase-floor-03/state/` and is not for copying:
  - `fixture_proof_store/` (three SQLite files);
  - the smoke pickles in `smoke/` and `smoke2/`, about 1.8 MB each;
  - `predictions_section3.raw` and the two `log_*.md.part` fragments this log was assembled from.
- **For the orchestrator:**
  - M0 holds; M1, M2, M4 and M5 hold; M3 fails on its slope clause only, which note (a) and I1 argue is badly posed;
  - the mechanism verdict as written does not hold, and the literal refutation clause fires;
  - H-anchor holds in every group it names; H-cancel's condition is met, but it does not account for the non-cancelling groups;
  - I2 (Clenshaw–Curtis at depth 20) bears on §2.7 layer 2's scope at G1.

## Orchestrator's review

**PASS, 9 of 9 checks** (`orchestrator/prompt-03.md` §3), reviewed on the Mac Studio at `98e68d3`,
2026-10-25, 20:15–20:45 GMT. The full review is
[`docs/levin-phase-floor/03/review.md`](../../../docs/levin-phase-floor/03/review.md). It wrote no
code, repaired nothing and ran no git command that writes.

Three checks pass on a reading the user is asked to confirm:
- **Check 5 (M0): `elapsed` is not gated.** It is each group record's wall-clock time, which no
  re-run reproduces. The agent said so before the launch (deviation 6). All 45 values differ; the
  re-run took 0.76–0.96 of the stored time. Every other field is gated and bit-equal.
- **Check 6 (the predictions): M5 is inapplicable on R7.** M5's rule names "R1–R5 and R7", but R7
  (`q == r == k`) has no `Gr` sub-interval. The agent scores M5 on the five rows it applies to
  (deviation 7). Strictly, R7's part is undecided.
- **Check 8 (the run): the pid is in `status.json`, not the manifest.** RunRegistryKit `v0.2.0`'s
  manifest schema has no pid field; the manifests of `handover-07-a3-v2` and
  `py313-venv-02-resumed-c` have the same keys. The detached pid (87581) is recorded by
  `Run.heartbeat(pid=…)` in `status.json`, as `CLAUDE.md` rule 3 says.

The mechanism verdict as written does not hold, because M3 fails on its slope clause. The review
records that as a result of the prompt, not a failed check, and leaves its reading to the user at
gate G1, beside I1's argument.

**Re-checked from the laptop.** After the review, the laptop orchestrator re-ran check 5's
independent comparison on R1 itself, over `ssh macstudio-tunnel`: the stored row, read through
`sqlite3` with `mode=ro`, against the pickled re-run. They agree in `float.hex` on
- `total`: `-0x1.631c5efe83a46p-78`;
- `total_abserr`: `0x1.2f222e3d4dd6dp-95`;
- `WKB_Levin` and `numeric_quad`: `0x0.0p+0`;

with `total_converged` 0 in the store and `False` in the re-run.

**Where `issues.md` went** (board §3):
- I1 opens `[03-m3-slope-clause-is-badly-posed-at-round-off]`;
- I2 opens `[03-depth20-regions-are-all-clenshaw-curtis]`;
- I3 opens `[03-storefixtures-standin-pickle-and-exit-rows]`;
- **I4 is not opened.** It is a note for 04's harness, and the agent proposed no tag; the reducer
  is in `docs/levin-phase-floor/03/scripts/harness.py`, and the board's §1 points 04 at it.
- **I5 is not opened.** It corrects prompt 03's "Read first" item 4, a reading list, and nothing
  was lost by it. It is recorded here, as observation 5 above, and nowhere else.

`[01-slow-group-leading-terms-do-not-cancel]` is narrowed by M5 on this board, and
`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` (`handover`) and
`[00-consumer-anchoring-floor]` (`GkTk-remedial`) on theirs.
