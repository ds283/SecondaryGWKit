# Prompt 03 — instrument seven depth-20 rows of A3 v2

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, after 01 landed (`3f28791`). 01 chose the rows and fixed the two classes.
**Where it runs:** on the Mac Studio, orchestrated there, as 01 did. **It commits nothing there.**
Everything it writes is under `var/levin-phase-floor-03/` (README §4.1).
**Uses:** no store write. The preferred route uses no Ray; Ray, if needed, runs under the Mac
Studio's lock (§2 step 3). The re-runs are a registered, detached run.
**Closes:** nothing. **Narrows:**
- `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` (`handover`);
- `[00-consumer-anchoring-floor]` (`GkTk-remedial`);
- this campaign's `[01-slow-group-leading-terms-do-not-cancel]`, which M5 decides.

**Recommended model:** **Opus**.

**Read first:**
1. [`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §2.
2. This campaign's [`README.md`](README.md) §0.2 (with its two correction notes) and §5, and the
   board's §2.1 and §3.
3. [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §1,
   01's census, and `docs/levin-phase-floor/01/issues.md` I1.
4. `AdaptiveLevin/levin_quadrature.py`:
   - `used_interval` (`:450-706`). Each accepted region records `depth`, its `type` (Levin, or the
     Clenshaw–Curtis fallback), `abserr`/`relerr`, `phase_err`, and the `abserr_history` and
     `relerr_history` of its ancestors, keyed by depth.
   - `_adaptive_levin`'s acceptance and bisection (`:1694-2384`).
   - `_roundoff_floor` (`:344`).
   - What `adaptive_levin_sincos` returns (`:2836-2870`): `regions` is the list of accepted
     `used_interval`s.
5. `ComputeTargets/QuadSourceIntegral.py`:
   - `evaluate_QuadSource_integral` (`:734-962`), which runs without Ray;
   - `phase_group_Levin_integral` (`:1010-1096`);
   - line 9, where `adaptive_levin_sincos` is imported by name, and line 15, where
     `build_phase_groups` is. Both are wrapped at those module attributes.
6. `main.py`:
   - `build_QuadSourceIntegral_payload` (`:358-440`);
   - the Bessel set-up (`:1240-1296`), including `b_value` and `largest_x_with_clearance`;
   - the stage's lookups (`:3815-4045`), with their tags.

   Extract any function you need with `ComputeTargets/tests/test_main_plumbing.load_main_py_functions`.
   `main.py` cannot be imported.
7. `StoreFixtures/standin_pool.py`, a real `ShardedPool` on in-process stand-in actors, and
   `datastorekit`'s `ShardedPool(read_only=True)`, which the readers use
   (`extract_QuadSourceIntegral_data.py:1320-1330`).

**Gate:**
- 01 landed and recorded (`3f28791` or later);
- the dispatch commit pushed;
- on the Mac Studio, `runregistrykit list` shows nothing `running` or stale;
- the lock is free;
- `git status --short` is empty;
- `store show` gives A3 v2 live, fingerprint `0e32f24a…`.

---

## 0. The question

01 found that the declared error of groups with and without a cancelling pair both track
$F = \varepsilon k\eta_R$. It found that most of the depth-20 regions in the expensive groups are
**Clenshaw–Curtis fallback** regions (`simple_regions`), not Levin regions. And it found that the
"slow" group on `r == k` does not cancel exactly.

This prompt looks inside the bisection, on seven rows:
- Is there non-smooth jitter in the composed phase, at about $F$?
- Does the discrepancy that drives the bisection stay at that level, however small the region?
- Does it fall away when the jitter is removed and nothing else changes?
- Can the driver see it?
- How far does `G+r` actually cancel?

## 1. The rows

Taken from `docs/levin-phase-floor/01/out/candidates_01.txt`; all are mid band and at depth 20.

| # | (shard, serial) | top group | $k$ | $q$ | $r$ | $z_{\rm response}$ | stored s | $F$ |
|---|---|---|---|---|---|---|---|---|
| R1 | (2, 37527) | `G+r` | 3.09e6 | 1.00e5 | 3.09e6 | 33.16 | 200 | 1.69e-6 |
| R2 | (0, 1797) | `G+r` | 3.05e7 | 3.09e6 | 3.05e7 | 302.6 | 220 | 4.60e-6 |
| R3 | (1, 37670) | `G+r` | 9.56e7 | 9.85e5 | 9.56e7 | 10.98 | 240 | 9.19e-5 |
| R4 | (2, 37720) | `G-r` | 3.09e6 | 1.00e5 | 3.09e6 | 14.47 | 355 | 2.60e-6 |
| R5 | (0, 37966) | `G-r` | 3.05e7 | 3.09e6 | 3.05e7 | 174.1 | 200 | 6.49e-6 |
| R6 | (2, 34728) | lone `G` | 3.09e6 | 3.09e6 | 3.09e6 | 25.16 | 134 | 1.96e-6 |
| R7 | (2, 34820) | `G+q+r` | 3.09e6 | 3.09e6 | 3.09e6 | 14.47 | 150 | 2.60e-6 |

That is about 25 minutes of compute for the re-runs alone. **Instrument every group of every
Levin sub-interval of each row, not only the top group.** The other groups are the comparison.

## 2. Steps

Make `var/levin-phase-floor-03/{out,scripts,state}`. Save every command and its output. Run
commands with variable arguments through `bash`.

1. **Check out the dispatch commit.**
   - `git fetch origin && git merge --ff-only origin/handover-remedial`;
   - `HEAD` is the dispatch SHA, and `git status --short` is empty;
   - `pip freeze` is unchanged by the move.

   If the orchestrator note says the checkout was moved for you, this step is a check, and no git
   command that writes is run.
2. **Hash the store.** The SHA-256, size and mtime of the primary, the four shards and the sidecar:
   `state/store_files_before.txt`.
3. **Choose the route to the inputs, and record why.**
   - **(a) Ray-free, preferred.** A `ShardedPool(read_only=True)` with the actors stood in by
     `StoreFixtures.standin_pool`. First prove it on a store built by
     `StoreFixtures.real_store_fixtures` in a temporary directory: it opens, `object_get` returns
     a `QuadSourceIntegral`'s inputs, and the fixture's files are byte-identical afterwards. Only
     then open A3 v2 with it.
   - **(b) Ray, if (a) cannot work.** A `ShardedPool(read_only=True)` on a local Ray instance
     (`address="local"`, `_temp_dir` from `config/ray_local.py`), under the Mac Studio's lock as
     `CLAUDE.md` gives it. `acquire` comes first, then `set-pid` with the harness's pid, then
     `release` after.
   - **If neither works, stop and ask.**
4. **Rebuild each row's inputs** as `main.py` does, by the same keys and tags:
   - the model;
   - $k$, $q$, $r$, and $z_{\rm response}$;
   - the `GkSourcePolicyData` for $(k, z_{\rm response})$;
   - the `QuadSource` for $(q, r)$;
   - the four $T$ objects;
   - `b`, and the two Bessel phase objects at the run's `largest_x_with_clearance`;
   - `z_source_max`, which the stored row's `metadata["partition"]["z_source_max"]` records.

   Write `scripts/rebuild.py`.
5. **M0, the gate on everything else: bit-identity.** Run `evaluate_QuadSource_integral` once for
   each row with the wrappers of step 6 in place. The following must equal the stored row
   **bit for bit**:
   - `total`, `total_abserr`, `total_converged`;
   - `WKB_Levin`, `numeric_quad`;
   - `WKB_Levin_data.max_depth` and `num_regions`;
   - every group record of `metadata["WKB_Levin"]["subintervals"]`.

   The wrappers must be transparent: they record, and change nothing. **A row that is not
   bit-identical is a stop**, since the harness would not then be measuring production. Since
   `b90752d`, only `ComputeTargets/analytic_Gk.py` has changed in this path's imports, and it does
   not reach the Levin integration. `analytic_rad` is compared too; a difference there is recorded,
   not a stop.
6. **The wrappers.**
   - **`adaptive_levin_sincos`, at `ComputeTargets.QuadSourceIntegral.adaptive_levin_sincos`.**
     Keep each call's full return, its `regions` list included, keyed by the group label parsed
     from `notify_label`.
   - **`build_phase_groups`, at `ComputeTargets.QuadSourceIntegral.build_phase_groups`.** Keep the
     returned `PhaseGroup` objects, so that `theta`, `theta_mod_2pi`, `theta_deriv` and `envelope`
     can be evaluated afterwards.

   This is a registered run (`campaign levin-phase-floor`, `prompt 03`, slug `instrument`),
   launched detached. Check once that it has started, and **end the turn**. Pickle what you keep
   into the run directory, row by row, so that the analysis in step 7 needs no re-run.
7. **The analysis**, in `scripts/analyse.py`, from the pickles alone, into `out/instrument_03.txt`.
   Fix the definitions before computing anything:
   - **$E$:** the group's `envelope` at a region's midpoint.
   - **$h$:** the region's width in $\log(1+z')$.
   - **$D$, the normalised discrepancy:** `abserr` $/(E\,h)$. For a Levin region `abserr` is
     $|v_0 - v_L - v_R|$; for a Clenshaw–Curtis region it is the nested-pair estimate.
   - **$\sigma_\theta$, the jitter:** for a region, evaluate `theta_mod_2pi` at 65 equispaced points
     across it. Unwrap with `numpy.unwrap`, fit a degree-4 polynomial in the local coordinate,
     and take the RMS residual.

   Measure:
   - **Per group:** a histogram of the regions' depths, split by type (Levin or Clenshaw–Curtis).
   - **The ancestry of a depth-20 region:** for up to 20 depth-20 regions per group, take the
     aligned spans at depths 10 to 20 that contain it. On each span, compute the discrepancy the
     driver would compute for that span's type:
     - the nested Clenshaw–Curtis pair at `CHEBYSHEV_ORDER` and $2N-1$;
     - or $|v_0 - v_L - v_R|$ through the driver's own single-region routines, if they can be
       called cleanly. Otherwise use the Clenshaw–Curtis pair for every span, and say so.

     Do this twice: with the real phase, and with the **jitter removed**, where `theta_mod_2pi` is
     replaced on the depth-10 span by a degree-12 polynomial fit to it there. The control changes
     nothing else.
   - **The slow group:** on every row with `r == k`, at five abscissae in its `Gr` sub-interval,
     $|\theta'_{G+r}|/|\theta'_{G-r}|$ against $(1-c_s)/(1+c_s)$, with $c_s$ from the model's
     `wPerturbations` there.
8. **Hash the store again** (`state/store_files_after.txt`), identical to step 2. Run
   `store fingerprint`. If Ray was used: `ray-processes` is empty, `release` is done, and the lock
   is free.
9. **Write the records.**
   - `log.md`, in README §5.1's form, with **Predictions and verdicts** first, written before
     step 5;
   - `verification.md`: the text of `docs/levin-phase-floor/VERIFICATION.md` **§3 "Seven rows,
     instrumented"**, with links relative to `docs/levin-phase-floor/` in the form `03/out/…`;
   - `issues.md`.

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

## 4. Acceptance

1. M0 holds on all seven rows.
2. The store is byte-identical and its fingerprint is `0e32f24a…`. If Ray was used, the lock was
   taken and released.
3. M1–M5 are each scored holds, fails or undecided, against §3 as written, with their numbers.
4. Records under `var/levin-phase-floor-03/`: `out/`, `scripts/`, `state/`, `log.md`,
   `verification.md` and `issues.md`. The pickles stay in the run directory.

The orchestrator copies the records, but not the pickles or any CSV, to `docs/levin-phase-floor/03/`,
appends §3 to `VERIFICATION.md`, writes `logs/03-instrument-depth-20-rows.md`, records the verdicts
on the board and the narrowings on the boards that hold them, and commits.

## 5. Stop conditions — stop and ask

- Any Gate item not as stated.
- Neither route of step 3 works.
- M0 fails on any row.
- Any store file changes.
- The lock is held, or will not release.
- The run would need a change to production code, a test file or `AdaptiveLevin/`.
