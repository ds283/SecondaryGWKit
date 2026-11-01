## Your brief: the phase agent — prompt 06h §5, the phase re-measurement (H1 and H2)

**Your scope is prompt 06h §5's "The phase re-measurement"**, with H1 and H2 scored from it. It is
06f's M2 item 1 and item 4, repeated at $h$ on 06f's 20 M2 calls. These parts are not yours:
- §5's census, Gate H, H3, H4 and the descriptive items belong to the census agent;
- §4, R21 and R22, belongs to the rehearsal agent.

Read 06f's prompt (`prompts/levin-phase-floor/06f-cc-floor-terms-across-leaves.md`, §3 "M2" and
§4's F4–F5) for what items 1 and 4 measure.

**Where you write:**
- only `var/levin-phase-floor-06h/phase/` (with `out/`, `scripts/` and `state/`) and your own run
  directory in `var/runs/`, if you register one;
- you take no lock;
- no process of yours initialises Ray;
- nothing under `var/datastores/` is created.

**The scripts** are 06f's, from `docs/levin-phase-floor/06f/scripts/` in the code worktree, and the
reviewer's `docs/levin-phase-floor/06f/review/scripts/orch_m2.py` where you use its method.
- Copy them into `var/levin-phase-floor-06h/phase/scripts/` (prompt §5).
- Save the diff against 06f's copies in `out/`. Each change is a numbered deviation.
- **H1's reference is at the exact $e^u$:** `mpmath.exp` of the float $u$, then each constituent's
  phase at $m_i e^u$ exactly, composed with the group's signs (prompt §2). 06f's own scripts scored
  at the float $\hat x$, except its review's Observation 1, so the copied scripts must change there,
  and each change is recorded.
- Use the prompt's precision, 30 digits, and check the reference at 50 digits at one call's worst
  node.
- Run the method check (06f's `method_check_06f.py`, adapted to the exact-$e^u$ reference) before
  any scoring.

**The calls** are 06f's 20 M2 calls, `docs/levin-phase-floor/06f/out/sample/`. 06f's capture
pickles were removed, so capture each call afresh:
- **at $h$** (the code worktree), as `x1_capture.py` captures. Check each capture bit for bit
  against the $h$ census's checkpoint record of the same call (value, abserr, regions). The census
  agent's run, `var/runs/levin-phase-floor-06h-census-*`, will be complete before you compute; read
  it only. A capture that does not reproduce its census record is a stop;
- **at the parent** (the parent worktree, `2c325f5`), for H1's parent column. Check it bit for bit
  against 06e's census checkpoint.

**The measurement** (prompt §5):
- $\delta\theta$ at every fine node of every returned Clenshaw–Curtis leaf, and at every collocation
  node of every returned Levin leaf, plus 06f's seeded dense sample of 1,000 points per call (06f's
  seed, recorded);
- for each point, $|\delta\theta(u)|$ against `theta_abserr`$_h(u)$, with the reference at the exact
  $e^u$;
- **the parent's declaration is scored on the same points beside it.**
  - Score it on the same $u$ values, with the parent's $\hat\theta$ and the parent's
    `theta_abserr`.
  - Where $h$'s and the parent's returned leaves differ, the node sets differ. Score each code on
    its own returned leaves' nodes, and on the common dense sample. Report both, and say which
    points H1 is scored on;
- per returned leaf, the first-order response to the measured phase error (06f's item 4), against
  that leaf's declared term.

**The predictions:**
- Copy prompt §5's H1 and H2 verbatim into `log.md`, and into a timestamped file under `state/`,
  **before** any number is computed.
- Do not edit that file afterwards.

**Long runs.** Anything that may exceed about ten minutes is registered with `runregistrykit`
(`--campaign levin-phase-floor --prompt 06h`, slug `phase`) and launched detached, as 06f's
`launch_06f.py` did. Make one check that it started, then end your turn with the launch report. At
most six worker processes.

**Scoring:**
- **H1:** holds, fails or undecided against its rule as written. Report:
  - the number of points scored;
  - the worst $|\delta\theta|/\texttt{theta\_abserr}_h$, with its call, leaf, $u$ and which term of
    the declaration dominates there;
  - the parent's count of violations and its worst ratio, on the same points.
- **H2:** holds, fails or undecided against its rule as written. Report:
  - the leaves scored;
  - the worst response over declared term, with its call and leaf;
  - the median.
- **Descriptive:** test 4's split by which term dominates, from the
  `_declared_error_parts` helper of `three_bessel_integrals.py`, if it serves.

**Sequencing:**
1. **Now, while the rehearsals and the census run:**
   1. read;
   2. copy the scripts;
   3. write their diff;
   4. write your changes for the exact-$e^u$ reference;
   5. plan the captures.

   **Compute nothing:** no capture, no replay, no reference evaluation, no hashing of the store.
   Other stages hold the six cores. End your turn reporting that you are ready, with the commands
   you will run.
2. **When the orchestrator resumes you,** after the census is over:
   1. hash the scratch copy;
   2. write the predictions;
   3. run the method check;
   4. capture, replay and measure;
   5. hash the scratch copy again;
   6. score H1 and H2;
   7. write your records;
   8. report in full.

   End your turn at each registered launch, as above.

**Your final report** gives:
- the method check;
- the captures' bit-for-bit checks at $h$ and at the parent;
- H1 and H2, each holds, fails or undecided, with numbers, and the parent's column;
- the 50-digit check of the reference;
- the store's hashes before and after;
- your deviations;
- anything in `issues.md`.
