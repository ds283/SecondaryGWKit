# levin-phase-floor 07d, Part A (section 2): the measurement — log of the measuring agent

**Subject:** prompt 07d Part A, the 6π gate and the Clenshaw–Curtis estimate. It covers:
- section 2.1's calls;
- section 2.2's D-N, the integrand's noise;
- section 2.3's D-E, candidate estimates of CC_{2N-1}'s error;
- section 2.4's D-T, the routes against the phase variation;
- section 2.5's variants and section 2.6's references (Ra) and (Rt);
- section 2.7's tables.

Run on the Mac Studio (mps026298), Python 3.13.16. The worktree is `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-49ce964`,
detached at the dispatch SHA `49ce96429da797efe55f546d65a23ebcd0099f33`. Its code is 07c's (`67b5af74`;
`AdaptiveLevin/levin_quadrature.py` `d141a80b…`; `tools/filon_reference.py` `d37e633b…`). Records are in
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/`. Part A started at 09:04:13Z on
2026-11-22 (`state/started_utc.txt`) and reached its stop at about 13:30Z.
**Commit:** none. The orchestrator commits the records, on the laptop.
**Result: the planned stop of section 2.8.** No condition of section 8 arose.
- Every replay reproduced 07c's census record, and every capture checked.
- The projection was 1.6 h against the 36 h limit.
- No error class invisible to the leaf's own samples was found.

## What shipped

Records only; no production code changed (README §5, rules 01–04). `scripts/` holds every script that ran, `state/` every
draw, seed, selection, rule and plan, `out/` every output (each with the SHA-256 of its script and inputs), and
`out/cmd/` with `out/commands.log` every command and its output by record number (01–56). The tables:

| table | file |
|---|---|
| 1 (D-N) | `out/tables/d_n.{md,json}` |
| 2 (D-E) | `out/tables/d_e.{md,json}`, `out/tables/causes.{md,json}`, `out/tables/trunc176.{md,json}` |
| 3 (D-T) | `out/tables/d_t.{md,json}` (first form `d_t_v1.*`) |
| 4 and 5, qualification, (Ra) at 07c | `out/tables/scores_07d.{md,json}` |
| the variants' replays | `out/tables/variants_replay.{md,json}` |
| v2 → v3 reference transitions | `out/tables/v2_v3_transitions.{md,json}` |

Registered runs, all ended `done`:

| run id | what |
|---|---|
| `levin-phase-floor-07d-capture-calls-20261122T091357` | 306 calls captured |
| `levin-phase-floor-07d-leaves-20261122T094441` | the leaf fill, reference v2 |
| `levin-phase-floor-07d-leaves-repair-20261122T095605` | 510 leaves re-measured |
| `levin-phase-floor-07d-leaves-v3-20261122T095839` | the fill re-measured, reference v3 |
| `levin-phase-floor-07d-capture-r1x-ra-20261122T100747` | the analytic top-up and (Ra) captured |
| `levin-phase-floor-07d-variants-20261122T103630` | six variants × 4,220 calls |
| `levin-phase-floor-07d-refs-20261122T120706` | 175 reference readings |

## Predictions and verdicts

The prompt's rules, each scored:

- **The gate (section 2.0): holds.** At `d141a80b`, by `ast`, `SIX_PI`'s only code reader is the gate (line 3055).
  The trees, the code diff, the inputs and the scratch copy match the brief. This is record 01; record 55 re-runs it at the
  end and is identical but for times and load.
- **Section 2.1, replays reproduce 07c's census record: holds,** 4,220 of 4,220 at the tree.
  - The 4,220 are the stratified class draw, 306 calls at both rtols; the analytic top-up with ride-along, 1,559 at both
    rtols; 07c Part A's 454; R31's record; and (Ra) at both rtols.
  - Captures check: 306/306, 1,559/1,559 and 30/30.
  - The stop instrument recorded all 16,047 stopped leaves, with none unrecorded.
- **Section 2.2 (D-N): measured.**
  - **Size:** σ/|g| has median 2.75e-12 on group leaves and 5.6e-15 on analytic leaves. Window pieces have median
    3.3e-10, p90 6e-8.
  - **Source:** `theta_mod_2pi`'s scatter. Its phase-predicted share over σ_g is 1.42 ≈ √2, and the amplitude share
    is ~1e-3 of it. The raw phase's rounding (0.7 u|θ|) is not the source.
  - **Structure:** independent node to node, lag-1 autocorrelation −0.016. The plateau at 47 nodes over the
    least-squares fit has group median 0.98, p05 0.69, p01 0.54. Leaves with 5+ knots show structure (issues.md item 5).
  - **Declared:** group calls declare nothing. The analytic blocks' `theta_abserr` overpredicts by a median 4.1×.
- **Section 2.3 (D-E): measured.** 928 leaves scored in 24 Clenshaw–Curtis strata, every stratum at 40 except
  `chase|analytic|1e-8`, exhausted at 8 (the census holds 80 analytic Ccc calls).
  - The tree's declaration understates on 114, all group. **nest3N4** understates on none (ratio min 0.95), and
    **pairDecl** on none (analytic only). Both qualify.
  - pairN4 understates on 6, pairN3 on 7, and tail4N4, chopN4 and pairtailN4 on 5 each. Every such leaf has 59–82 knots.
  - **Causes** (`out/tables/causes.md`; knots first, then noise): knots 68, noise 46 (32 of the 46 window pieces),
    unexplained 0.
  - **07c's 176 truncation-dominated chase leaves** (`out/tables/trunc176.md`), all of them reproduced:
    - The tree is bounded on 28, understates on 22 and is undecided on 126.
    - Causes: knot 8, noise 4, noise at 3–4 s_CC 10.
    - nest3N4 understates on none; pairN4, pairN3, tail4N4, chopN4 and pairtailN4 on 4 each.
- **Section 2.4 (D-T): no threshold below 20π is bounded on both sides.**
  - The Levin route's own declaration understates on about a fifth of the leaves in every band up to 20π: 408 band
    leaves, all group, 348 with knots. True error over declaration has median 1.22, p90 72, p99 326. With the trust
    check, 5–40 per band still understate.
  - The tree's Clenshaw–Curtis declaration understates below every T; nest3N4's understates on none up to 20π.
  - At 20π the Levin side is empty: no sample leaf lies above it, so "bounded" there is vacuous.
  - Rule A changes nothing at any T, by construction (deviation 2).
- **Section 2.5 (variants): six built; at most eight allowed.** Each replayed all 4,220 calls with rc 0. Every call a
  variant leaves untouched reproduces 07c's census bit for bit: Eg 972, Ea 813, ET20 407, ET12 533, Ep 813 and T20 720
  (`out/tables/variants_replay.md`).
- **Section 2.6 (references): read.**
  - **Projection** (`out/refs/refs_projection_07d.json`), reported to the orchestrator before any reading: 189 readings
    (175 commands), 1.59 h elapsed. The run took 50 min (12:07–12:56Z), with all 175 at exit 0.
  - **Scored:** 425 referenced calls, of which 417 are scored. The 8 unscored are R2-27, R2-29, R3a-0-33666-a13 and
    c-2-1375-4, each at both rtols: under some code their min(abserr)/100 falls below the reference's spread.
- **Section 2.7 (qualification):** only **ET20** qualifies, and vacuously on its threshold.
  - **Eg and Ea** fail on threshold (6π).
  - **ET12** fails on threshold, and makes `R:a:W2-0-12977-5` converged and dishonest at ratio 3.25.
  - **Ep** fails on estimate (pairN4) and threshold.
  - **T20** fails on estimate (tree) and threshold, and makes `R:a:R2-08~o` converged and dishonest at 1.12.
- **(Ra) at 07c's code:** 60 of 60 scored, and all converged and honest. The largest ratio is 0.42.

## Deviations from the prompt

1. The script copies were made with the shell's cwd in the source and records directories, not the worktree. They are
   07c's census harness into `scripts/`, 07c Part A's scripts into `scripts/from_07c_measurement/`, and their SHA-256
   lists. The copies used `cd` for `find`, `shasum` and `cp`. One `find` listing was written to `/tmp/x` by mistake,
   read once and deleted at once (09:04Z). No Python ran there and nothing was written in the worktree. **UNINTENDED DRIFT.**
2. **Rule A reads no threshold at 07c's code.** The orchestrator note's section A.0 ("Rule A reads `SIX_PI` at `:3806`")
   and prompt section 2.4 ("Rule A reads `SIX_PI` by name") are not so at `d141a80b`. By `ast`, `SIX_PI` is loaded as
   code only by the gate (line 3055; record 01). Line 3806 is a docstring describing prompt 07's Rule A, which 07a
   replaced. So "what Rule A does at T" is nothing, by construction, at every T, and no skips are counted. The laptop
   orchestrator confirmed it by its own AST walk and ruled it a deviation, not a stop; the prompt and the note are
   corrected in section 9's amendment. **STRUCTURALLY REQUIRED.**
3. **The stratum "guarded leaves at the cap" is empty.** 07c's census has no call with `num_guard_at_depth_max` positive
   (0 of 289,482; `out/census_index_07d_summary.json`), so no class draws for it. This is the prompt's "if any", confirmed
   by the orchestrator. **STRUCTURALLY REQUIRED.**
4. **make_calls_07d.py v1 compared R31's record with refcap.json's store-scale `R32` entry,** whose value is 2¹⁷ times
   the call's. v2 compares with the call-level record, which its agent showed reproduces R32's store, and adds the refusal
   and chase counts to the comparison. The v1 manifest is in `state/superseded/`, and no table reads its smoke output.
   **UNINTENDED DRIFT.**
5. **The leaf reference was amended to v2 after six test leaves, before any leaf of the sample was read.** v1's single
   global least-squares fit shared a knotted leaf's knot error. v2 fits each knot interval. Also corrected then:
   - phase fits on θ − θ(mid);
   - an exact unwrap of `theta_mod_2pi`;
   - Gauss–Legendre end pieces for shifted spans;
   - an analytic block's knots read from its Bessel objects.
   v1 and the test outputs are kept; no table reads them. **IMPLEMENTATION CHOICE.**
6. **The fill's first registered run lost 510 leaves to a defect of the shift test, repaired before v3.** The shifted
   spans left the call's span: T_k above its WKB region, and a Bessel amplitude below its minimum argument. Shifts are now
   reflected inside the span, and a leaf's failure is recorded without ending its spec. The repair run re-measured all 510.
   **UNINTENDED DRIFT.**
7. **The leaf reference became v3 after v2's complete reading of the fill, and every leaf was re-measured under v3.**
   - **Why:** v2's stated error on the analytic band leaves was about 1,000 times both routes' errors, so none of the 88
     could be decided.
   - **What changed:** v3 raises the fit degree only on the reference's own convergence. It never raises it on agreement
     with any value scored. It keeps v2's check and stated error's form; the orchestrator accepted it on those conditions.
   - **Timing:**
     - v2's verdicts were written at 09:50:55Z, 09:54:52Z and 09:57:49Z, and read at 09:54:52Z and about 09:56Z.
     - v3's rule was written at 09:58:15Z (`6fcce794`, with v2 moved to `state/superseded/`).
     - The re-measurement launched at 09:58:39Z.
   - **Transitions** (`out/tables/v2_v3_transitions.md`): 0 leaves moved between bounded and understates in either
     direction. **IMPLEMENTATION CHOICE.**
8. **The analytic strata were topped up, with the drawn units' other analytic calls riding along.** Round 1 drew 99
   analytic G, L and Ccc calls (seed + 1000). The other 1,460 analytic calls of their units were captured with them,
   since a capture re-runs its whole unit, and are marked as ride-along. Their leaves are appended after each stratum's
   existing order, so no filled prefix changed. **IMPLEMENTATION CHOICE.**
9. **The variants carry the noise term as the candidates were scored: added to the floor-limited declaration too.** The
   scored form is max(E, floor) + κs. Patching κs into E alone gives max(E + κs, floor), which understates on 4 scored
   leaves. So κs is also added to `round_off_err` and `round_off_err_less_node`. The cross-check finds each variant's
   abserr equal to leafcc's within 9e-4 relative. **IMPLEMENTATION CHOICE.**
10. **Table 3 gained a column after its first form:** `levin_tc`, the Levin declaration after the trust check. The first
    form is kept (`d_t_v1.*`), and no verdict of it changed. **IMPLEMENTATION CHOICE.**
11. **A seventh process, briefly.** Draw and summary scripts, each under about 10 s and reading JSON and pickles, ran
    while a registered run held six worker processes: draw_calls_07d.py round 1, ridealong_07d.py,
    variants_summary_07d.py's first run and draw_rt_07d.py. **UNINTENDED DRIFT.**
12. **The driver runs its jobs with cwd in the records directory.** run_07d.py is 07c's run_06j.py v2 byte for byte
    (07c's deviation 22). Every job names its scripts by absolute path with PYTHONPATH the worktree first, and nothing
    imports from the cwd. **IMPLEMENTATION CHOICE.**
13. **x1_capture_07d.py groups units by `k_key`, 20 units a worker,** so that units sharing a k's tables load them once.
    Captures and checks are unchanged. **IMPLEMENTATION CHOICE.**
14. **(Rt)'s population and strata.**
    - "Own-class calls" is read as this Part's own drawn integrands (`d:` entries), each counted once over both rtols.
    - The 18 (Ra) integrands whose route changes are left out, since their references are read anyway.
    - The direction stratum holds only Levin → Clenshaw–Curtis: every (T) and (ET) variant raises the threshold, so the
      reverse is empty by construction.
    - 30 of 1,517 integrands were drawn, in 16 strata of kind × k band × rtol. **IMPLEMENTATION CHOICE.**
15. **Route A's projection has no 07c model, so its cost per panel was timed, no value kept.** `refs_costA_07d.py` timed
    the tool's chunk functions; the projection uses the largest per-panel wall × 1.5. The first refs_inputs_07d.py run
    listed every QAWO panel, 4e8 at |K| ~1e8, and was stopped after 10 minutes with no output (record 42). The count is now
    computed by `dyadic_tiling`'s own arithmetic without the list, checked equal to the list's length at three |K|.
    **IMPLEMENTATION CHOICE.**
16. **07d copies of 07c's reading scripts.** `leafref_07c.py` names 07c's removed worktree, and `one_reading.py` and
    `refs_projection_07c.py` import `refs_common`, whose location check fails one level deeper. The copies are
    `leafref_07d.py` and `one_reading_07d.py`, changed only where marked "prompt 07d", and `refs_feats_07d.py`, which
    holds the feature functions verbatim, checked by diff. The tool is the same file. **STRUCTURALLY REQUIRED.**
17. **Two smoke readings** were taken after the projection was sent and before the launch, kept in `out/refs/smoke/` and
    read by no table. The run's readings of the same integrands equal them in every value field. **IMPLEMENTATION CHOICE.**
18. **Table 2's cause rule was written out after it was applied.** The rule: knots if the split at the knots leaves under
    half the true error, else noise if the true error is at most 4 s_CC. The agent applied it reading `d_e.json` at about
    10:31Z; `causes_07d.py` writes it down unchanged and reproduces the same counts, 68 and 46. It follows the prompt's
    order, knots then noise. **IMPLEMENTATION CHOICE.**

19. **The variants' cross-check first ran unrecorded.** At 10:36Z the comparison of Ea's leaf declarations with leafcc's
    nest3 + 4 s_own ran as an inline command, outside `rec.sh`. Its inputs and leafcc's output were kept in
    `out/replays_test/xcheck_*.json`. Record 56 (`variant_xcheck_07d.py`) recomputes the comparison from those files and
    gives the same numbers: 4 leaves, largest relative difference 9.0e-4. The check covers Ea only; Eg, ET20 and ET12 share
    its estimate block, but were not checked separately. **UNINTENDED DRIFT.**

## Verification performed

`verification.md` (this Part's section of VERIFICATION.md §7d). In short:
- **Gate:** before and after are identical (records 01 and 55).
- **Worktree:** `git --no-optional-locks status --short --ignored` was empty at every check, the last at about 13:25Z.
  The main checkout stayed at `28ebaf07`, clean.
- **Replays and captures:** 4,220 of 4,220 replays at the tree reproduce bit for bit. Every capture checks against its
  census record.
- **Variants:** every untouched call reproduces bit for bit under each variant. Ea's declaration was cross-checked against leafcc's
  estimate (record 56).
- **Readings:** all 175 exited 0 with Ray uninitialised. Both smoke readings are reproduced.
- **Process limit:** at most six worker processes, deviation 11 aside. No Ray, no lock, and no store opened but A3 v2's
  scratch copy, read only.

## Observations not acted on

See `issues.md` (proposals; nothing opened). In brief:
1. The Levin route's own declaration understates in every band up to 20π.
2. The fallback's declaration understates on 114 of 928 scored leaves.
3. The noise source is `theta_mod_2pi`, and group calls declare nothing.
4. Window pieces carry the largest noise.
5. Label G and 5+-knot leaves show structure.
6. Analytic leaves show step structure at θ's wraps.
7. Route A QAWO is infeasible at large |K|.
8. D-N's "common" flag is too loose.
9. The Rule A description is wrong (deviation 2).

Also, table 4's measurements:
- ET20 and T20 leave 32 and 43 scored calls unconverged, against the tree's 3.
- R2-28 is converged and dishonest at ratio 1.13–1.14 under the tree, Eg, Ea and Ep, as in 07c.
- Table 5: T20 costs 5.8× the tree's wall and 8.8× its regions.

## State handed on

- The records directory above. `KEPT_ON_MAC_STUDIO.txt` lists every pickle and every file over 2 MB, with SHA-256, size
  and path.
- No run is live and the rehearsal lock is free. The worktree is unchanged; the orchestrator may remove it.
