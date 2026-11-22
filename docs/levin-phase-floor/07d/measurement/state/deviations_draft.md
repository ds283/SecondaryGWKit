# 07d Part A — deviations, drafted as they happen (folded into log.md at the end)

1. The script copies (07c's census harness into scripts/, 07c Part A's scripts into scripts/from_07c_measurement/,
   their SHA-256 lists) were made with the shell's cwd in the source and records directories (`cd` for `find`, `shasum`,
   `cp`), not the worktree. One `find` listing was also written to `/tmp/x` by mistake, read once and deleted at once
   (09:04Z). No Python ran there and nothing was written in the worktree. UNINTENDED DRIFT.
2. **Rule A reads no threshold at 07c's code.** The orchestrator note's section A.0 ("Rule A reads `SIX_PI` at `:3806`")
   and prompt section 2.4 ("Rule A reads `SIX_PI` by name"; a variant moving the threshold moves Rule A with it) are not
   so at `d141a80b`: by `ast`, `SIX_PI` is loaded as code only by the gate in `_adaptive_levin_subregion_impl` (line 3055;
   out/cmd/01_gate.txt); line 3806 is `_locate_stationary_points`'s docstring describing prompt 07's Rule A, which 07a
   replaced (a phase is skipped only when its rate is exactly zero at every finite scan sample). So "what Rule A does at
   T" is nothing, by construction, at every T, and no skips are counted. Put to the laptop orchestrator, who confirmed it
   (its AST walk: reads at 296, the definition, and 3055) and ruled: a deviation, not a stop; the prompt and note are
   corrected in section 9's amendment. STRUCTURALLY REQUIRED.
3. **The stratum "guarded leaves at the cap" is empty:** 07c's census has no call with `num_guard_at_depth_max` positive
   (0 of 289,482; out/census_index_07d_summary.json), so no class draws for it and no leaf stratum exists (the prompt's
   "if any"). Confirmed by the orchestrator. STRUCTURALLY REQUIRED.
4. **make_calls_07d.py v1 compared R31's record with refcap.json's store-scale ``R32`` entry** (value 2^17 times the
   call's), so the smoke replay of ``R:r31`` (record 09) reported a difference in value and abserr. refcap.json also holds
   the call-level record, which its agent showed to reproduce R32's store (``reproduces_store``). v2 compares with that
   (and adds the refusal and chase counts to the comparison in replay_07d.py); the v1 manifest is in state/superseded/
   (the census records wanted are identical), and the smoke output under v1 is out/replays_test/tree/R_r31.v1manifest.json.
   No table reads either. UNINTENDED DRIFT (a failed first form).
5. **The leaf reference was amended (v2) after six test leaves, before any leaf of the sample was read**
   (state/refs_rule_leaf_07d.txt; v1 in state/superseded/; the test outputs in out/leaves_test/, which no table reads).
   On the tests: v1's single global least-squares fit could not represent a leaf with interior knots (20 knots: the
   residual's lag-1 autocorrelation 0.999, sigma(d_lo)/sigma(d_hi) 9.4), and on a one-knot gate leaf it agreed with
   CC_{2N-1} to 2e-40 while the knot-split Gauss-Legendre rule differed by 2.2e-38: the fit shared the rule's knot error.
   v2 fits each knot interval. With it the same leaf's reference agrees with the second rule, and CC_{2N-1} errs by
   1.8e-38. Also corrected in leafcc_07d.py before the sample: (a) the phase fits are made on theta - theta(mid), since a
   fit to |theta| up to 1e11 has a rounding floor of its own; (b) theta_mod_2pi is unwrapped exactly (an integer count
   of 2 pi, the offset in 40 digits), since theta_raw + (theta_mod - theta_raw) in double carries theta_raw's rounding
   back; (c) the shifted spans' integrals are ref plus Gauss-Legendre end pieces, not a fit on a widened span (on an
   analytic test leaf that fit gave a bimodal offset); (d) an analytic block's knots are read from its Bessel objects
   (production_edges reads group calls only). IMPLEMENTATION CHOICE.
6. **The fill's first registered run lost 510 leaves to a defect of the shift test, repaired before v3.** In
   `levin-phase-floor-07d-leaves-20261122T094441` 94 of its specs exited nonzero: the shift test's shifted spans, at a
   call's own ends, evaluated the integrand beyond the call's span (T_k above its WKB region; a Bessel amplitude below its
   minimum argument), and one failure ended the spec. leafcc_07d.py (v3 of the script, shift handling only) keeps a shifted
   span inside the call's span (a shift that would leave it is reflected; one that cannot be kept is skipped) and records a
   leaf's failure without ending its spec; `levin-phase-floor-07d-leaves-repair-20261122T095605` re-measured all 510 in
   place (all measured, no nonzero exit). UNINTENDED DRIFT (a defect found by its run).
7. **The leaf reference became v3 after v2's complete reading of the fill; every leaf was re-measured under v3.** Why: on
   the analytic Levin-band leaves v2's stated error was about 1,000 times both routes' errors (e.g. eps_ref 2.5e-22 against a
   Levin error of 1.1e-25 and a declaration of 4.4e-25), so 0 of the 88 analytic band leaves could be decided: v2's
   starting degree (ceil(V/2) + 32) under-resolves a leaf whose theta' grows across it, as an analytic block's does in its
   log variable (sigma(d_lo)/sigma(d_hi) 429 on a test leaf). v3 raises both degrees together only on the reference's own
   convergence (until the d_lo fit's residual RMS is at most 1.5 times the d_hi fit's, the residual at its plateau; never on
   agreement with CC_{2N-1}, CC_N or the Levin value), and keeps v2's check (step 3) and stated error's form (step 4); the
   orchestrator accepted it on those conditions. **Timing:** v2's fill wrote its verdicts at 09:50:55Z (round 1), 09:54:52Z
   (round 2) and 09:57:49Z (repair); the agent read their counts at 09:54:52Z and the analytic band leaves' numbers at about
   09:56Z; v3's rule was written at 09:58:15Z (state/refs_rule_leaf_07d.txt `6fcce794`, v2 moved to state/superseded/) and
   its re-measurement launched at 09:58:39Z (`levin-phase-floor-07d-leaves-v3-20261122T095839`) from the start, in the same
   seeded order. v2's fill is kept in out/leaves_v2_superseded/; the v2 to v3 transitions are tabulated per stratum and route
   in out/tables/v2_v3_transitions.{json,md}, every leaf moving between bounded and understates named. IMPLEMENTATION CHOICE.
8. **The analytic strata were topped up, with the drawn units' other analytic calls riding along.** After v3's fill, every
   group stratum held 40 scored leaves but 16 analytic strata were exhausted short (gate q2 at 1e-7, q3 and q4, the chase
   and every Levin band; e.g. gate q4 4 leaves, band 8-10 pi 3, at each rtol). The top-up rule fixed in
   draw_calls_07d.py drew round 1 (seed + 1000, 5 calls a cell, excluding the first sample) restricted to the cells of the
   classes that hold those leaves, analytic G, L and Ccc (`--kinds analytic --classes G,L,Ccc`; 99 calls in 99 units,
   the analytic Ccc cells exhausted). Since a capture re-runs its whole unit, every other analytic call of those 99 units
   (1,460) was captured with the drawn ones (ridealong_07d.py): a sample of units by the rule with all their analytic
   calls, the ride-along calls marked. Their leaves are appended after each stratum's existing order in a seeded order of
   their own (leaf_population_07d.py --extend), so no prefix already filled changes. IMPLEMENTATION CHOICE.
9. **The variants carry the noise term as the candidates were scored: added to the floor-limited declaration too.** The
   D-E candidates are max(E, floor) + kappa s (state/candidates_07d.json, fixed before scoring), so the noise term stands
   outside the max. Patching it into `_adaptive_levin_subregion_cc` only where the estimate E is formed (abserr = E +
   kappa s with the floors unchanged) gave a solve whose declaration is max(E + kappa s, floor), which on 4 scored leaves
   is below the scored form and understates. make_variants_07d.py therefore adds kappa s to abserr and to the two floor
   terms (`round_off_err`, `round_off_err_less_node`), and its cross-check confirms each variant's abserr and phase_err
   equal leafcc's nest3 + 4 s and floor + 4 s within 9e-4 relative. IMPLEMENTATION CHOICE.
10. **Table 3 gained a column after its first form:** the Levin route's declaration after the trust check (`levin_tc`:
    a trusted leaf returning its quarters' sum Q with the trust check's resolution), beside its own declaration. Its first
    form is kept as out/tables/d_t_v1.{json,md}; no verdict of the first form changed. IMPLEMENTATION CHOICE.
11. **A seventh process, briefly.** The draw and summary scripts (draw_calls_07d.py round 1, ridealong_07d.py,
    variants_summary_07d.py's first run and draw_rt_07d.py; each under about 10 s, reading JSON and pickles) ran while a
    registered run held six worker processes. UNINTENDED DRIFT.
12. **The driver runs its jobs with cwd in the records directory** (run_07d.py is 07c Part A's run_06j.py v2, byte for
    byte; 07c's deviation 22), not in the worktree. Every job names its scripts by absolute path with PYTHONPATH the
    worktree first, and writes only into the records directory; nothing imports from the cwd. Kept byte-identical for its
    provenance. IMPLEMENTATION CHOICE.
13. **x1_capture_07d.py groups the units it captures by `k_key`, 20 units a worker** (07c's x1_capture_07c.py gave each
    worker a slice of the unit list), so that units sharing a k's tables load them once; the captures and their checks are
    unchanged (every capture matched its census record). IMPLEMENTATION CHOICE.
14. **(Rt)'s population and strata.** "Own-class calls" is read as this Part's own drawn integrands (the replay manifest's
    `d:` entries), each counted once over its two rtols, not 07c Part A's regression entries, whose references 07c read.
    The 18 (Ra) integrands whose route changes are left out of the pool, since their references are read in any case.
    The direction stratum holds only Levin to Clenshaw-Curtis: every (T) and (ET) variant raises the threshold above
    6 pi, so the reverse is empty by construction (recorded in state/selection_rt_07d.json). Of 1,517 integrands in 16
    strata (kind x k band x rtol), 30 were drawn, BASE 1 then round-robin. IMPLEMENTATION CHOICE.
15. **Route A's projection has no 07c model, so its cost per panel was timed, no value kept.** 07c read no analytic block.
    refs_costA_07d.py ran the tool's Route A chunk functions on 6 (FCC) and 40 (QAWO) panels of four drawn integrands
    spanning |K| 5e5 to 3e8 and kept only the wall time (out/refs/costA.json); the projection uses the largest per panel
    times 1.5. The first run of refs_inputs_07d.py listed every Route A QAWO panel to count them (`RouteA.panels`), which at
    |K| ~1e8 is 4e8 entries: it was stopped after 10 minutes having written one input pickle and no output
    (out/cmd/42_refs_inputs_ra.txt). The panel count is now computed by the tool's own `dyadic_tiling` arithmetic without
    the list (`qawo_panel_count`), checked equal to the list's length at |K| 37, 1,234.5 and 98,765. IMPLEMENTATION CHOICE.
16. **07d copies of 07c's reading scripts.** 07c Part A's leafref_07c.py names 07c's worktree (`-lpf07c-2fd979f`), which
    no longer exists, as TREE, and its one_reading.py and refs_projection_07c.py import refs_common, whose location check
    fails one level deeper (scripts/from_07c_measurement/). leafref_07d.py and one_reading_07d.py are copies changed only
    where marked "prompt 07d" (TREE; the worker's loader module; refs_common's four names defined in place), and
    refs_feats_07d.py holds `features` (07's) and `feats` (07c's) copied verbatim (checked by diff). The tool is the same
    file (d37e633b). STRUCTURALLY REQUIRED.
17. **Two smoke readings** (one Route A FCC r1 of rt-h-1e-8-1-10776-a12 and one Route B split FCC r1 of
    rt-h-1e-8-1-10922-g3) were taken after the projection was sent and before the launch, to check the copied scripts;
    they are kept in out/refs/smoke/ and no table reads them (the run reads both integrands again). IMPLEMENTATION CHOICE.
