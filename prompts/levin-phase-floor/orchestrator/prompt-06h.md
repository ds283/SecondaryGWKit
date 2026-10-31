# Orchestrator — prompt 06h, the Bessel group phase's rounding

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06h-group-phase-rounding.md`](../06h-group-phase-rounding.md)
**Board item:** 06h · **Closes, in its records commit, if prompt §2's tests and §5's H1 and H2
hold:** `[06f-theta-abserr-omits-the-product-rounding]`. Otherwise ask the user. H0, H3 or H4
failing is a finding for the user, not by itself a reason to keep it open.
**Model:** Opus.

This note follows [`prompt-06e.md`](prompt-06e.md), whose machinery it shares: Part A as 06e's
note §A, Part B as 06e's note §B. It was written 2026-10-31, at dispatch: the prompt's commit
(`664def1`) carried no note.

## 0. The order

1. **07's work stays set aside.** It is `stash@{0}` (07 rebased on 06d, set aside for 06e) and
   `var/levin-phase-floor-07/wip/`, as 06e's note §0 left them; the checkout is clean, so nothing
   is set aside again. The four stashes are kept.
2. **Part A** is the code on the laptop, one commit, reviewed here.
3. **Part B**, R21, R22, the census and the phase re-measurement, runs on the Mac Studio after
   Part A is pushed, on the user's word.
4. **06g** follows, its census against 06h's; **07** resumes after it.

## A. Part A — the code, on the laptop

### A.0 Facts, checked by the orchestrator at the dispatch SHA

- **The code is 06e's** (`f4d8e17`); 06f added no code. The seven roots at the dispatch SHA, each
  run on its own: AdaptiveLevin 68, ComputeTargets 780, CosmologyModels 39, LiouvilleGreen 148
  (1 skipped), ObjectFactories 280, RunDrivers 127, tools 58, all OK.
- **The group** is `LiouvilleGreen/three_bessel_integrals.py`'s `BesselPhaseGroup`. `sin_cos`
  forms `big = self.K * x` and `small = self.C_reduced + self.residual(x)` with `x = np.exp(log_x)`,
  and takes `math.sin`/`math.cos` on a 0-d argument and `np.sin`/`np.cos` otherwise.
  `theta_abserr` is the plain sum of the three `theta_abserr_at(m_i x)`. `K`, `C` and `C_reduced`
  are `math.fsum` of the signed terms (`_coefficient_sum`).
- **The tests the prompt names:** `test_three_bessel.py:322`
  (`test_the_group_phase_survives_cancellation_of_its_leading_term`, the `+ 2.0 * EPS * abs(K) * x`
  bound at :354), the definition test at :640–:663 (`assertAlmostEqual(theta_abserr, sum,
  delta=1e-18)` at :659); `test_quadsource_integral.py:1850` (`TestThreeBesselPhaseGroups`, the
  bound at :2004). The `4.0 * EPS * abs(K) * x_max` assertions at :385 and :2047 are not of
  test 9's kinds.
- **`math.fma`** is in the venv's Python 3.13.16; numpy is 2.2.4. `np.exp` on a 0-d argument and on
  an array may take different code paths, so $\varepsilon_{\exp}$'s basis must cover both.
- **06f's records** are under `docs/levin-phase-floor/06f/`; its two violating calls are
  `g-1e-7-0-12689` #19 (J4) and `g-1e-7-3-1054` #20 (Y4).
- **The known flake** is `test_tk_wkb_phase … test_wall_time_per_object`; re-run alone, not a stop.
  Run the roots one at a time (06e's note §A.5).

### A.1 Before you dispatch

1. 06f's records commit has landed (the board's 06f row is ✅); `git status --short` is empty;
   `runregistrykit list` shows nothing `running` or stale.
2. Commit this note and its row in `README.md`, and push. That is the dispatch SHA.
3. `touch` a marker in the scratchpad, and record the SHA-256 of the four stashes' commits and of
   `var/levin-phase-floor-07/wip/`.

### A.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. It is given the prompt, the dispatch
SHA, the baselines and §A.0, and it does not read `orchestrator/`. As 06c's note §A.2 (`HEAD` is
not its own; never `git stash`; scope §1–§3, §4's H0 and §5's predictions copied unchanged and not
scored; one commit staged by explicit path, not pushed; no Ray, no store, nothing under `var/`;
every deviation classified; stop at prompt §7), with these paths:
- `LiouvilleGreen/three_bessel_integrals.py`;
- `LiouvilleGreen/bessel_phase.py`, **docstring only** (prompt §0.2 item 3);
- `ComputeTargets/QuadSourceIntegral.py`, **comment only**, and only if a comment describes the
  group's phase error;
- `LiouvilleGreen/tests/test_group_phase_rounding.py`, new;
- `LiouvilleGreen/tests/test_three_bessel.py` and `ComputeTargets/tests/test_quadsource_integral.py`,
  **only** for assertions of the kinds prompt §2 test 9 allows, each listed;
- `docs/levin-phase-floor/06h/calibration/`;
- `prompts/levin-phase-floor/logs/06h-group-phase-rounding.md` and the board's 06h row;
- `docs/OPEN_ISSUES.md` and the board's §3, only if it opens an issue.

Any other path is a stop. It never touches `var/levin-phase-floor-07/`.

### A.3 The review — nine checks, on the laptop

1. **Scope:** `git show --stat` lists only §A.2's paths; the checkout is clean; the four stashes and
   `var/levin-phase-floor-07/wip/` are unchanged by SHA-256; nothing under `var/` is newer than the
   marker.
2. **Nothing else moved,** by AST without docstrings:
   - in `three_bessel_integrals.py` only `BesselPhaseGroup.__init__`, `sin_cos` and `theta_abserr`
     change, beside new private helpers and constants; `theta`, `theta_deriv`, `residual`,
     `arguments`, `theta_mod_2pi`, `levin_theta` and everything outside the class are unchanged;
   - `bessel_phase.py` and `QuadSourceIntegral.py` are AST-identical;
   - `AdaptiveLevin/` and `ComputeTargets/phase_groups.py` are untouched.
3. **The construction, read** against prompt §0.2:
   - $K_{\rm lo}$ by `math.fsum` of the signed terms and $-\hat K$, once, in `__init__`;
   - $p_{\rm lo}$ by TwoProduct, `math.fma` on scalars and Dekker's split on arrays, and the split's
     range of exactness stated against the production range of $\hat K\hat x$;
   - the small angle is $C_{\rm reduced} + R + p_{\rm lo} + K_{\rm lo}\hat x$, and the big angle
     ${\rm fl}(\hat K\hat x)$, unreduced;
   - `theta_abserr` is the constituents' sum plus $\varepsilon_{\exp}$ times a bound on
     $|d\theta/du|$ built from $|\hat K|\hat x + \sum_i|dr_i/d\log y|$ (not `theta_deriv`), plus
     $\phi_{\rm eval}$, combined linearly;
   - $\varepsilon_{\exp}$ derived, or measured on at least $10^6$ seeded arguments over both ranges,
     **on the scalar and the array path**, and taken at a stated multiple of at least 2;
   - each term of $\phi_{\rm eval}$ has its derivation in a comment; no rounding is counted twice
     against a constituent's `EVALUATION_FLOOR`, and none is omitted.
4. **The roots, re-run here,** one at a time. LiouvilleGreen rises by the new module's methods; the
   other six equal the baselines.
5. **The tests, read and run:**
   - tests 1–9 are named in the log;
   - **re-run tests 1, 2, 3 and 6 yourself at the parent and at the commit;**
   - on 06f's two cells, **recompute $|\hat\theta(u) - \theta(e^u)|$ yourself** in `mpmath` at 50
     digits at two points each, the reference at the exact $e^u$, against `theta_abserr` at both
     codes;
   - if $\varepsilon_{\exp}$ is measured, re-run its measurement;
   - read every changed assertion against prompt §2 test 9;
   - re-run the LiouvilleGreen draws in a throwaway worktree and compare with the committed output,
     timings aside.
6. **The breakages:** (i)–(iii), each named with its catching test; **re-apply (i) and (ii)
   yourself**, in a throwaway worktree.
7. **Determinism:** `sin_cos`, `theta_abserr` and one `adaptive_levin_sincos` call on 06f's
   `g-1e-7-0-12689` group, each twice in fresh processes, bit-identical.
8. **The records:**
   - the log has README §5.1's headings, with every deviation classified;
   - §4's H0 and §5's Gate H and H1–H4 are copied verbatim and marked not yet scored;
   - `black` holds outside `calibration/scripts/`;
   - the commit message follows `CLAUDE.md`.
9. **Nothing else landed unaccounted:** `git log --oneline <dispatch SHA>..HEAD`.

### A.4 Landing

If the review passes, report to the user. On their word, push the code commit, then commit and push
Part B's dispatch addendum (§B.1), whose commit is R22's and $h$'s tree.

## B. Part B — R21, R22, the census and the phase re-measurement, on the Mac Studio

As 06e's note §B, through an SSH session on the Mac Studio (`ssh macstudio-tunnel`, `--cpus 6`),
with these differences:
- **The scope** is prompt §4 and §5. The worktrees are `SecondaryGWKit-lpf06h-<parent>` and
  `SecondaryGWKit-lpf06h-<dispatch>`.
- **H0 is re-scored** by your own `mode=ro` reader over every `QuadSourceIntegral` row.
- **Gate H's first clause** is against 06e's Part B dispatch tree `6d81fad`. **06e's census run**
  `levin-phase-floor-06e-census-20261030T081318` is the "before", its checkpoint `a0dc4d7f…`.
- **H1 and H2 are re-scored** from the phase re-measurement's tables by a scratch reader, with
  $\delta\theta$ recomputed by `mpmath` at the exact $e^u$ on one call's worst node.
- **H3 and H4 are re-scored** from the two checkpoints by a scratch reader.
- **The census's checkpoint and unit records stay on the Mac Studio** in full; 06g's census reads
  them.
- **A3 v2's scratch copy** is read-only and hashed before and after. Nothing in its directory is
  moved, retired or deleted.

### B.1 Part B's dispatch addendum

Written after Part A lands. It records the code commit, its parent, the trees by SHA, and what
Part B must carry.

## 5. Stop and ask the user

As 06e's note §5, with prompt 06h's §7 in place of 06e's. **Never touch 07's stashes or saved
work.**
