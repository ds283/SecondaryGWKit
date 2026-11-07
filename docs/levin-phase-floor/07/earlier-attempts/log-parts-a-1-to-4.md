# levin-phase-floor 07 — the stationary-point guard: locate them first, integrate their windows directly

> **§1–§4 only** (the code, the calibration, the tests and these records), written by the
> executing agent on the laptop. §5 (rehearsals R19 and R20, renamed by prompt §9 points 4 and 6)
> and §6 (the re-measurement) run later on the Mac Studio. The orchestrator appends their record to this
> log.
>
> **Status of this log:** written at the first stop (2026-10-27: §2's reference, and a
> LiouvilleGreen test moving), updated at the second (2026-10-28: §2 run on the rebased tree;
> no Phi_w rung qualifies, a stop under §2 itself), and at the third (2026-10-29: §1 rebased onto
> 06c's global budget; §2 cannot run on it, and a LiouvilleGreen draw moves), and at the fourth
> (2026-10-29: §1 rebased onto 06d's Clenshaw-Curtis floor; §2 still cannot complete). No §3 test
> is written. Sections marked
> *pending* are filled on the resume.

**Subject:** prompt 07 §1–§4, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `93d5cb1`. Two commits landed on the branch while this work ran, neither
  touching a file of this prompt: `4cef62a` and `9afcfd0` (`handover` B2's issue 13: its probe, its
  board and one row of `docs/OPEN_ISSUES.md`).
- **The gate.** `git status --short` was empty at dispatch. `runregistrykit list` showed nothing
  `running` or stale: 61 runs, 59 finished, 2 `unknown` (pre-registry). The seven roots' baselines
  are the orchestrator's, at `93d5cb1`.
- **The resume** (2026-10-28) is on `cf3efac`, which carries prompt 06b's code (`4ddc3c2`). This
  work was set aside before 06b ran (by the orchestrator: a patch, a tarball and `stash@{0}`) and
  `git stash apply`'d onto `cf3efac`. Its baselines are the roots at `cf3efac`: AdaptiveLevin 38,
  ComputeTargets 780, CosmologyModels 39, LiouvilleGreen 148 (skipped 1), ObjectFactories 280,
  RunDrivers 127, tools 58, all OK. Nothing landed after `cf3efac` before the second stop.
- No Ray was started, no run was launched, no store was opened, and nothing was written under
  `var/`.

**Commit:** *pending* (§4; one commit, not pushed).

**Result so far:**
- **First stop (2026-10-27),** on two §8 conditions: the prompt's §2 reference was impractical as
  written, and a LiouvilleGreen test moved on 1 of 20 seeded draws. The user ruled on both (prompt
  §9), and prompt 06b ran first.
- **Second stop (2026-10-28).** §1 is rebased onto 06b's code. No LiouvilleGreen draw moves over
  seeds 0–19. §2 has run. **Under the Phi_w rule no rung qualifies:** every (R) case is honest at
  every rung, but every rung leaves one to three (R) cases unconverged. §2 says that is a stop. The
  Lambda_c rule therefore did not run as written; a supplementary measurement gives (C) at every
  rung, for the user's decision.

## What shipped (so far, uncommitted)

**`AdaptiveLevin/levin_quadrature.py`**
- **§1.0, the basis interface.** Two optional methods, `independent_phases()` and
  `independent_phase_rates()`, each returning a list of callables, one per independent phase
  (deviation 1). `_Basis_SinCos` returns `[theta["theta"]]` and `[theta["theta_deriv"]]`, or `None`
  for the rates when `theta_deriv` is absent. A basis without the methods is treated as unable to
  supply rates. The module docstring gains a section, THE BASIS INTERFACE, listing every method the
  driver uses.
- **§1.1, the location,** `_locate_stationary_points()`, called by `_adaptive_levin()` when the
  basis supplies rates:
  - each rate is sampled through `_sample_vectorized()` on 06a's grid, `numpy.linspace` over
    `x_span` with 4,097 points (`_LEVIN_STATIONARY_SCAN_POINTS`);
  - adjacent opposite-signed samples are refined by `brentq`, with xtol = eps max(|x_a|, |x_b|) and
    rtol = 4 eps; an exact zero sample is a root as it stands;
  - every strict interior minimum of |theta'| with no sign change beside it is refined by bounded
    Brent minimisation. theta''' there is a central second difference of the rate (deviation 13),
    and the minimum is located if lambda < `_LEVIN_COALESCENCE_LAMBDA`;
  - ends within half a scan spacing, merging within 4 eps |x|, and each point's phase, as written.
- **§1.2, the windows.** Each edge is found by walking the scan outward and refining by `brentq` on
  the raw phase. Overlapping windows are merged. Each window is cut into pieces of at most 2 pi in
  every phase (deviations 2 and 3). The pieces enter the loop as regions of kind `"window"` at
  depth 0. Each is integrated by the nested Clenshaw–Curtis pair (`_adaptive_levin_window_piece()`)
  and bisected if it misses its tolerance, with `phase_limited` barred. The initial region list is
  `x_span` less the windows; with nothing located it is today's single region.
- **§1.3, the guard.** `_adaptive_levin_subregion_impl()` reports whether any phase rate fails to
  be of one strict sign on the region's nodes (`_nodal_guard()`; deviations 4 and 9). A guarded
  region is bisected on the Levin branch below `depth_max`, never accepted there. On the fallback
  branch `phase_limited` is barred. At `depth_max` it is integrated by the nested pair, flagged
  `guard_at_depth_max`, and the call's `converged` is False (deviation 5).
- **§1.4, the result.** `stationary_points`, `unlocated_minima`, `num_window_pieces`,
  `num_guarded_regions` and `num_guard_at_depth_max` (deviations 6, 7 and 10).
  `used_interval` gains the type `"window"` and the flags `guarded` and `guard_at_depth_max`.
  Docstrings are updated as §1.4 asks: the module, `_roundoff_floor`, `_adaptive_levin` (it had
  none), `_adaptive_levin_subregion_impl`, the basis interface, and `adaptive_levin_sincos`, which
  is otherwise unchanged.
- **The constants:** `_LEVIN_WINDOW_HALF_PHASE` and `_LEVIN_COALESCENCE_LAMBDA` are provisional
  (32 pi and 64) until §2 and the user's decision (deviation 15). `_LEVIN_WINDOW_PIECE_PHASE` is
  2 pi.
- **The test-only switch:** `_adaptive_levin(..., _test_only_bypass_location=False)` (deviation 16).

**`AdaptiveLevin/tests/test_levin_quadrature.py`:** `_TwoIndependentSinCosBasis` gains the two
interface methods, and nothing else.

**`ComputeTargets/QuadSourceIntegral.py`:** each group record gains `stationary_points`,
`window_pieces`, `guarded_regions` and `guard_at_depth_max`, and nothing else. No test in
`ComputeTargets/tests/` pins a group record's key set (searched: `simple_regions`, `"groups"`,
`keys()`).

**`docs/levin-phase-floor/07/calibration/`** (scripts formatted with `black` before each recorded
run; every output's header records the SHA-256 of its script and of `levin_quadrature.py`, `HEAD`,
and that the tree was dirty):
- `scripts/prov.py`, the provenance header;
- `scripts/time_reference.py` → `out/time_reference.json`, the orchestrator's item 3;
- `scripts/reference.py`, the proposed references;
- `scripts/validate_reference.py` → `out/validate_reference.json`, and
  `scripts/validate_reference_precision.py` → `out/validate_reference_precision.json`;
- `scripts/lg_location.py` → `out/lg_location.json`, the orchestrator's item 1;
- `scripts/lg_repeat.py` → `out/lg_repeat.json`, and `scripts/lg_seed10.py` →
  `out/lg_seed10.json`: the LiouvilleGreen analytic tests over 20 seeded draws, and the one draw
  that moved;
- `scripts/parent_compare.py` → `out/parent_compare.json`, the new driver against the parent on
  `test_levin_quadrature.py`'s problems;
- `scripts/calibrate.py`, §2's runner. **Not yet run as a record.**

## First stop (2026-10-27): §2's reference (the orchestrator's item 3)

*The records of the first stop are kept as they ran in `docs/levin-phase-floor/07/calibration/
stop1/` (scripts and outputs; each output's script hash matches its file there). The live
`scripts/` and `out/` were then re-targeted at the new parent and re-run (deviation 24).*

**The prompt's reference,** `mpmath.quad` at 30 digits split at every zero of theta' and at x_0, on
the largest case of each family (`stop1/out/time_reference.json`):

| case | total variation (rad) | as written: wall, mpmath's own error estimate | subdivided to converge (pi pieces): s/piece, pieces, projected wall |
|---|---|---|---|
| (R) c = 1e8, x_s = 0.7236… | 3.0e7 | 0.06 s, **0.11** (not converged) | 3.0e-3 s, 9.5e6, **8.0 h** |
| (C) b = 1e8, lambda = 1024 | 9.8e6 | 0.05 s, **0.2** (not converged) | 3.4e-3 s, 3.1e6, **2.9 h** |

As written it returns a number with no correct digit. Made to converge, §2 needs about 15 h for
(R) at c = 1e8 and about 20 h for (C) at b = 1e8 (seven cases of about 3 h). That is a long run,
which README §5 rule 7 would register under `var/`, which this dispatch may not write.

**Proposed in its place** (`scripts/reference.py`; *not used for any recorded §2 number*):
- **(R):** a closed form. Completing the square in the complex error function gives J(alpha) =
  int_0^1 exp(alpha x) exp(i theta) dx, and I = Im J(-1) + Re [J(3i) + J(-3i)]/2, evaluated by
  `mpmath.erf` at 40 digits.
- **(C):** Cauchy's theorem. The integrand is entire, so [0, 1] is replaced by the rectangle
  through the upper saddle point u = i sqrt(a/b), on which |exp(i theta)| <= 1 everywhere. Each side
  is a smooth decaying or Gaussian integral, done by `mpmath.quad` at 40 digits. The part of the
  horizontal side not integrated is bounded by 1e-69 or less.
- **Validated** against the prompt's reference, made to converge, on every case where that takes
  minutes (`out/validate_reference.json`): (R) at c = 1e4 (both x_s) and c = 1e6 (x_s = 0.38…; 144,803
  pieces, 7.8 min), and (C) at b = 1e4 (all seven lambda). They agree to at most 1.9e-29 relative,
  except (C) b = 1e4, lambda = 0.25, at 1.4e-26. There the 30-digit literal reference is the one in
  error: at 45 digits on pi/4 pieces it equals the contour to 7e-44
  (`out/validate_reference_precision.json`).
- **Cost:** at most 7.6 s per case, all 20 cases in under 40 s.

## First stop, continued: a LiouvilleGreen test moves (prompt §8, "a test in `LiouvilleGreen/tests/` moves")

`test_3bessel_analytic.py` draws its wavenumbers from `random.uniform`, **unseeded**, on every run,
as its module docstring says. So which of its Levin calls hold a stationary point varies from run
to run. `lg_location.py` saw 21 such calls in two tests on one run, and 19 in three on the next,
every test passing both times.

`lg_repeat.py` runs that module over seeds 0–19, each seed under the new driver and under the
parent, on the same draw (`stop1/out/lg_repeat.json`). All 20 seeds locate points: 428 calls in all, 19–23
per seed, every point a root.
- The parent fails nothing.
- **The new driver fails `test_JJJ` on seed 10**, at the provisional constants (Phi_w = 32 pi).

**What moved** (`stop1/out/lg_seed10.json`). The triangle is (J2, J2, J0) at k = 3.302, q = 0.885,
s = 2.651. Its relative error is 2.57e-7 against the test's REL_TOLERANCE of 1e-7; the parent's is
1.15e-12.
- One of the four groups has a root at log x = 1.095, near the lower end of the span
  [1.018, 27.631]. Its window, clipped at that end, runs to 6.076, where theta' = -102.
- The Levin range beyond it, [6.076, 27.631], runs from |theta'| = 102 to about 1e11 at
  x = 1e12. Its eq. (151) floor, inflated by G1/G0 of about 1e9, is 9.0e-7, above its step-(4)
  residual of 6.6e-8.
- So the range is accepted at depth 0 **through the round-off-limited exit**. The exit is barred
  only in window pieces and guarded regions (§0.2 (ii)), and this range is neither.
- The group is off by 6.7e-8, inside its declared 9.0e-7, so the call is honest. The test asserts
  accuracy, not honesty, and fails.
- **The parent's region holding the root had G0 near 0,** so an infinite floor and no such exit.
  It bisected the whole span, depth 7, and was right to 5e-14.

**The same draw across §2's Phi_w ladder:**

| Phi_w | relative error | declared abserr | test | the Levin range after the window |
|---|---|---|---|---|
| parent | 1.15e-12 | 1.06e-10 | passes | (bisected, depth 7) |
| 2 pi | 2.31e-12 | 8.69e-11 | passes | bisected; four regions from 5.00, depths 1–4, one phase-limited with floor 4.6e-14 |
| 8 pi | 3.05e-6 | 6.51e-7 | **fails** | [4.732, 27.631], depth 0, phase-limited, floor 3.7e-6, theta' = -26.6 at its start |
| 32 pi | 2.57e-7 | 1.60e-7 | **fails** | [6.076, 27.631], depth 0, phase-limited, floor 9.0e-7, theta' = -102 |
| 128 pi | 1.23e-8 | 3.79e-8 | passes | [7.451, 27.631], depth 0, phase-limited, floor 2.1e-7, theta' = -404 |
| 512 pi | 5.36e-10 | 8.91e-9 | passes | [8.835, 27.631], depth 0, phase-limited, floor 5.0e-8, theta' = -1610 |

Every rung is honest on this draw, and the passes at 128 pi and 512 pi carry the same mechanism
with a smaller floor. Whether another draw fails at a given rung is not measured.

## The resume (2026-10-28) and the second stop

**The user's rulings** (prompt §9, quoted there): 06b ran first, fixing the exit that moved the
LiouvilleGreen draw (`[07-roundoff-exit-assumes-bisection-cannot-help]`); §2's reference may be
any acceptable method, with (R) computed both by the error function and by contour integration,
and (C) by contour integration; (R) adds cases at omega = 1.

**The rebase onto 06b** (deviation 22). Five conflict hunks in `levin_quadrature.py`, all where 07
meets 06b's premise check, resolved keeping 06b intact:
- `_adaptive_levin`'s signature carries both test-only keywords; 06b's comment on its own keyword
  is kept as it was, below 07's docstring, whose loop description now names the premise check.
- **The fallback branch.** 07's bar (no round-off-limited exit in a window piece or a guarded
  region) is applied *before* 06b's premise check, so a barred region never solves the two halves
  the check would need. Where the check does solve them, they are solved with 07's `guard_rates`,
  and the bisected children carry both 06b's `estimate=` and 07's `kind=`.
- **The Levin branch.** 06b's premise check is kept as written; 07's `not guarded and (...)` is the
  acceptance condition after it.

**Bit-identity at the new parent** (`out/parent_compare.json`, against `cf3efac`): 21 problems --
`test_levin_quadrature.py`'s kind with and without `theta_deriv`, the generic m = 4 basis, and
06b's two premise problems (tests 1 and 3 of `test_levin_roundoff_exit.py`). Value, abserr and
region list are bit-identical on 20; the one that moves is C2 with `theta_deriv`, as before (root
located at 0.5; abserr +inf -> 6.8e-13).

**The LiouvilleGreen re-test** (`out/lg_repeat.json`, seeds 0–19, rebased tree against `cf3efac`
on the same draws): **no seed fails under either.** Seed 10, the draw that moved, passes. All 20
seeds locate points, 429 calls in all. One unseeded draw in full (`out/lg_location.json`): 148
tests pass (skipped 1); 23 of 290 calls locate a point (22 roots, 1 minimum) in four tests; the
guard never fires; all 267 calls that locate nothing are bit-identical to the same call with the
location bypassed; on the 23 the value moves from 06b's by at most 0.16 of 06b's abserr and 0.39
of the new; two calls that 06b reported converged are now unconverged, and one the reverse.

**The references** (`out/validate_reference.json`, `out/validate_reference_precision.json`):
- (R) by the error function and by contour, on all ten cases (six at omega = 0, four at omega = 1):
  **largest relative disagreement 2.19e-37.** The error function at 40 against 60 working digits:
  also at most 2.19e-37, so mpmath's erf is stable at this precision.
- Against the prompt's reference made to converge, where that takes minutes -- (R) at c = 1e4 (all
  three cases, omega = 1 included) and c = 1e6 (x_s = 0.38, omega = 0, 144,803 pieces, 7.9 min), (C)
  at b = 1e4 (all seven lambda): at most 1.9e-29 relative, except (C) b = 1e4, lambda = 0.25 at
  1.4e-26, where the 30-digit literal is the one in error (at 45 digits on pi/4 pieces it equals
  the contour to 7e-44).

**§2's run** (`out/calibrate.json`, `out/calibrate_tables.md`; `levin_quadrature.py` SHA-256
`7d6921ac…`, HEAD `cf3efac`, tree dirty; 35 s).

*Phi_w on (R)* -- every case honest at every rung; unconverged cases by rung (abserr over the
request max(atol, rtol |value|), and the share of it in the window pieces' nested-pair estimates,
`abserr_fallback`):

| Phi_w | unconverged (R) cases | abserr / request | of which window pieces |
|---|---|---|---|
| 2 pi | c = 1e4, 1e6, 1e8 at x_s = 0.7236, omega = 0 | 1.01, 1.10, 1.48 | 0.83, 0.90, 1.05 |
| 8 pi | c = 1e8, x_s = 0.7236, omega = 0 | 1.84 | 0.71 (Levin residual 0.96) |
| 32 pi | c = 1e4 and 1e8, x_s = 0.7236, omega = 0 | 1.13, 1.77 | 1.07, 1.05 |
| 128 pi | c = 1e4, 1e6, 1e8, x_s = 0.7236, omega = 0 | 1.44, 1.18, 2.22 | 1.36, 1.08, 1.89 |
| 512 pi | c = 1e4, 1e6, 1e8, x_s = 0.7236, omega = 0 | 1.47, 1.50, 6.66 | 1.41, 1.43, 6.14 |

- **Every unconverged case is x_s = 0.7236 at omega = 0,** whose |I| (1.2e-3, 1.4e-4, 1.4e-5) is
  10–16 times smaller than at x_s = 0.38: the integral cancels. Each window piece is accepted on its
  own relative test, rtol against its own value, and the pieces' summed estimates then exceed rtol
  against the cancelling total. The parent converges on all ten (R) cases.
- **The location** finds every root exactly where it is: point - (x_s - omega/c) = 0 to the
  printed digit on every case and rung, and at omega = 1 point - x_s = -1/c (1e-4, 1e-6, 1e-8).
- theta'^2/theta'' at the window edges is 2 Phi_w (12.57, 50.27, 201.1, 804.2, 3217) except where
  the window is clipped at an end of [0, 1].
- **Phi_w by the rule: none.** §2: "If no rung qualifies under either rule, stop and ask."

*Lambda_c on (C):* not run under the rule, which needs a Phi_w. **Supplementary, not the rule**
(`scripts/calibrate_supplement.py`, `out/calibrate_supplement.md`): (C)'s procedure repeated at
every Phi_w rung. With Lambda_c = 0 nothing is located on (C), so those runs equal the parent's
bit for bit, and every (C) case is honest at every lambda (err/abserr 0.012–0.97). So **the
Lambda_c rule would pick 4 at every rung**, and the check -- lambda = 0.25 and 1 at both b,
re-run with the minimum located -- **holds at every rung**: honest and converged in all 20 runs.

## The third stop (2026-10-29): §2 on 06c's global budget

**The rebase onto 06c** (deviation 26) is on `ad63e8a` (06c's records commit; its module is the code
commit `ad9a5f4`'s). AdaptiveLevin: 53 tests OK on the rebuilt module, 06c's included.

**Bit-identity at 06c** (`out/parent_compare.json`, against `ad63e8a`, the same 21 problems):
- 19 bit-identical, 06b's two premise problems and the generic m = 4 basis among them.
- **C2 without `theta_deriv`:** no longer identical, as expected now that a guarded leaf is
  split and never floor-limited inside 06c's loop. The guard fires on 25 leaves. Honest: true
  error 2.9e-15 against the oracle (-6.87907971687041e-4, mpmath to 1e-31) with abserr 1.35e-10,
  97,084 leaves, unconverged; 06c's own: 5.6e-15 with abserr 4.3e-9, 96,941 leaves.
- **C2 with `theta_deriv`:** root located at 0.5; 71 leaves, abserr 4.678e-11 against a true
  error of 4.666e-11 (honest, err/abserr 0.9975), converged at atol 1e-10. 06c's own: abserr +inf,
  4,798 leaves.

**§2 cannot run** (`out/stop3_diagnose.json`). `calibrate.py`, as written, reached the fourth (R)
case at Phi_w = 2 pi -- c = 1e6, x_s = 0.7236, omega = 0 -- and the call raised
`ZeroDivisionError` in 06c's depth-18 diagnostic in `_make_leaf`
(`abs_improvement = prev_abs / this_abs`), where an ancestor's abserr is exactly 0. The record
re-runs that case at bounded depth:

| depth_max | window leaves | their summed floor | their summed nested pair | worst window leaf's floor (holds x_s) | true error | abserr |
|---|---|---|---|---|---|---|
| 0 | 3 | 1.4e-16 | 2.4e-11 | 1.2e-16 (yes) | 1.1e-5 | 5.4e-6 |
| 1 | 6 | 1.3e-15 | 9.1e-15 | 4.6e-17 (no) | 5.6e-6 | 3.7e-6 |
| 2 | 12 | 6.7e-15 | 1.4e-17 | 6.6e-15 (yes) | 1.9e-6 | 1.6e-6 |
| 4 | 48 | 3.0e-15 | 4.6e-18 | 2.3e-15 (yes) | 1.5e-8 | 1.4e-8 |
| 6 | 192 | 9.3e-15 | 1.9e-18 | 5.8e-15 (yes) | 1.7e-11 | 1.7e-11 |
| 8 | 768 | 2.6e-13 | 4.2e-19 | 2.2e-13 (yes) | 9.4e-16 | 1.2e-10 |
| 10 | 3,072 | 4.6e-13 | 2.8e-19 | 3.7e-13 (yes) | 1.1e-15 | 1.2e-9 |

The call's request is max(atol, rtol |V|) = 1.4e-14.
- **The mechanism.** A window leaf's error is max(nested pair, eq. (151) floor). Once the nested
  pair has fallen below round-off, the floor dominates. The floor of the leaf that holds the root
  grows as the leaf shrinks: its G0 is theta' at the node nearest the root, which shrinks with the
  leaf. By depth 8 the window leaves' floors alone (2.6e-13) exceed the request 18 times over.
- **Why the loop cannot stop.** §9 point 6 makes a window leaf never floor-limited, so it stays
  refinable until `depth_max`. The budget is unmet, so the loop keeps choosing window leaves. It
  stops only when none is refinable: every window leaf at depth 20, about 3 x 2^20 leaves per
  call. The 06b driver accepted each window piece on rtol against its own value, so it never
  reached this.
- **Where the crash comes from.** A window leaf's nested-pair estimate is often exactly 0 at depth
  (0.0 at depths 8 and 10 above). The depth-18 diagnostic, reached on the way, divides by it.
  Neither the diagnostic nor its division is 07's code; 07's window leaves are what reach it.
- The other leaves' error also stops falling: from 1.7e-11 at depth 6 to 1.2e-9 at depth 10.
  This is the Levin leaves near the window, at depth_max, and is not separated here.

**The LiouvilleGreen draws.** `lg_repeat.py` (seeds 0–19 against 06c) was stopped after seed 0,
which took about 20 minutes: `test_JJJ` errors under the rebased tree and passes under 06c. The
record re-runs seed 0's `test_JJJ` alone. The error is the same `ZeroDivisionError`, in a call
that reached depth 18; it is an error, not an assertion failure (`lg_repeat` reports the two
together). It takes 89 s under 06c's driver, and 31 s up to the crash under the rebased tree. The
other 19 seeds were not run.

**This is §8's stop twice over:** §2 cannot pass, and a LiouvilleGreen test moves. Both come from
§9 point 6's "never floor-limited" meeting an eq. (151) floor that grows under bisection at a
stationary point inside 06c's loop. The decision is the user's.

## The fourth stop (2026-10-29): §2 on 06d's Clenshaw–Curtis floor

**The rebase onto 06d** (deviation 29) is on `e5764a6` (06d's records commit; its code commit is
`3d70db9`). One conflict hunk, where 06d's `_history_improvement` and `_log_depth_history` sit
after `_adaptive_levin_subregion_cc`, beside 07's helpers: both kept, 06d's first. The window
pieces call `_adaptive_levin_subregion_cc`, so they now carry 06d's floor, `_cc_roundoff_floor()`;
`_Basis_SinCos` has 06d's `eval_basis_and_arguments()`; the depth-18 call site is 06d's
`_log_depth_history`. The docstring passage that justified placing no node on a root by eq. (151)
now says that reason has gone with 06d, and the placement is kept. AdaptiveLevin: 60 OK, the
baseline.

**Bit-identity at 06d** (`out/parent_compare.json`, against `e5764a6`): 20 of 21 bit-identical,
now including C2 without `theta_deriv` (the guard fires on 13 leaves, and the result is still
06d's). The one that moves is C2 with `theta_deriv`: root located at 0.5, error 4.666e-11 against
abserr 4.707e-11 (honest, inside atol 1e-10), 71 leaves; 06d's own is error 5.7e-15, abserr
2.26e-11, 26 leaves.

**§2 does not complete.** `calibrate.py`, as written, ran under a 25-minute limit. It finished
the 2 pi rung's first five (R) cases and the parent's 24, and was stopped inside the sixth: the
fourth, c = 1e6, x_s = 0.7236, omega = 0, alone took about 20 minutes (honest, abserr 7.95e-13,
unconverged), and the sixth, c = 1e8 at the same x_s, did not finish. No output was written.

**Why** (`out/stop4_diagnose.json`):
- **06d alone cannot converge three (R) cases at §2's tolerances**, all of them honest:

  | (R) case | abserr / request | leaves |
  |---|---|---|
  | c = 1e6, x_s = 0.7236, omega = 0 | 2.61 | 31 |
  | c = 1e8, x_s = 0.7236, omega = 0 | 94.4 | 63 |
  | c = 1e8, x_s = 0.7236, omega = 1 | 13.3 | 69 |

  The other seven converge (abserr/request 0.25–1.0). On these three the request rtol |I| is
  below the round-off the driver declares: the total cancels. So no rung of Phi_w can make
  "every case of (R) honest and converged" true; §2's rule is not satisfiable at its own
  tolerances on the parent's code.
- **Window leaves are chased to depth 20 whenever the budget is unmet.** With 06d's floor, the
  window leaves' summed error is flat in depth -- 5.3e-15, 2.1e-14, 8.5e-14, 3.4e-13, 1.4e-12 at
  2, 8, 32, 128, 512 pi, against a request of 1.4e-14 -- so bisection does not lower it, and
  06b's premise would hold. But §0.2 (ii) keeps window leaves never floor-limited, so each stays
  refinable to `depth_max`. At depth_max 12 the stuck case already has 12,288 window leaves at
  2 pi and 2,101,248 at 512 pi (495 s); at depth 20 that is 256 times more.
- Outside the windows, Levin leaves with a residual flat in h above their floor (7–64 of them
  at depths 6–9) are chased too: 06c's `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.

**The LiouvilleGreen draws** (`lg_repeat.py`, seeds 0–19 against `e5764a6`) **did not get past
seed 0.** Two copies ran -- the first, piped through `grep`, which I misread as having ended, and a
duplicate started 32 minutes later -- for 4 h 16 min and 3 h 44 min, CPU-bound at 40–50 MB, and
were ended by pid on the orchestrator's prompt. No seed completed and no output was written. The
draw was then reproduced (`out/stop4_lg_seed0.json`):
- seed 0's first test, `test_JJJ`: calls 1–14 finish in at most 6 s each (13 and 14, which
  locate nothing, are identical under 06d and the rebased tree);
- **call 15 does not finish.** It locates one root, at log x = 0.17684, whose window
  [-0.0492, 4.6525] (clipped at the span's lower end) has 17 pieces. The window leaves' summed
  error is flat in depth -- 4.564e-12, 4.552e-12, 4.549e-12 at depth_max 6, 8, 10 -- against the
  call's request of 2.59e-12, and every other leaf together carries 8.4e-16. So the budget is
  never met, and the window leaves, never floor-limited, are chased to depth 20: 1,088, 4,352 and
  17,408 leaves at depth_max 6, 8, 10; 69,653 in 181 s at 12.
- 06d on the same call: 161 leaves (0.3 s) at depth_max 8, 697 (1.2 s) at 12, and 3,081 leaves in
  7.6 s at the call's own depth_max, unconverged (abserr 7.5e-12), its leaves floor-limited.
- **This is the fourth stop's mechanism inside a LiouvilleGreen call:** the call cannot finish,
  which is §8's "a test in `LiouvilleGreen/tests/` moves".

## Predictions and verdicts

**§2's two rules** (prompt §2, fixed before any number exists):
- **Phi_w:** the smallest rung of {2 pi, 8 pi, 32 pi, 128 pi, 512 pi} on which every case of (R) is
  honest and converged. — **No rung qualifies** (`out/calibrate_tables.md`): every case honest at
  every rung; unconverged 3, 1, 2, 3, 3 of 10 at the five rungs. A stop under §2. *The user's
  decision: pending.*
- **Lambda_c:** the smallest rung L of {4, 16, 64, 256, 1024} such that every case of (C) with
  lambda >= L, at both b, is honest, run with no minimum located; the check, the cases with
  lambda < L re-run with their minimum located, must be honest and converged. — **Not run under
  the rule** (no Phi_w). Supplementary: L = 4 at every rung, the check holding at every rung.
  *The user's decision: pending.*

**§6's predictions, copied from the prompt unchanged. Not yet scored** (they are scored on the Mac
Studio):

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **V1 — nothing else moves.**
  - **The rule:** every Levin call in which $g$ locates nothing and guards nothing equals 06a's
    call bit for bit, in value, abserr, region count and maximum depth. Every row all of whose
    calls are such equals 06a's row in `total` and `total_abserr`.
  - **Holds** if true at both rtols.
- **V2 — no region is accepted across a stationary point.**
  - **The rule:** at `1e-7`, 06a's scan (`spscan.py`, the same grid) finds no accepted **Levin**
    region holding a sign change of $\theta'$, in any row.
  - **Holds** if no row has one, against 06a's 720.
- **V3 — the location finds what 06a's scan found.**
  - It runs on 06a's grid, so the two should agree exactly.
  - **The rule:** every call 06a's scan flagged has at least one located `root`, and every located
    `root` lies within one scan spacing of a sign change 06a's scan recorded.
- **V4 — honest on a sample.**
  - **The sample:** 24 stationary-point calls at `1e-7`, stratified by band (low, mid), by $k$ (all
    six that carry them) and by 06a's depth, with the deepest four rows included. The selection
    rule is fixed in `state/predictions.md` before the run.
  - **The reference** is P1's method (P1 report §5):
    - Gauss–Legendre pieces of at most 0.75 rad over $|\theta - \theta(x_s)| \le \Phi$;
    - Levin forced to $2^9$ pieces outside;
    - its error measured as P1 measured it, by $\Phi = 10^3$ against $10^4$.
  - **The rule:** $|{\rm value}_g - {\rm ref}| \le$ `abserr`$_g$ for every call in the sample.
- **V5 — the cap is not reached where a point was handled.**
  - **The rule:** at both rtols, no region reaches `depth_max` in a call that locates a point or
    guards a region. An over-budget unit counts as false.
  - A failure is a finding for the user, not a fault. It reopens G2's L2 (Decisions, **G2 L2,
    decided**).

**Prompt §9 points 4–5, copied unchanged** (they amend §5 and §6):

4. **The rehearsals are renamed.** 06b takes R15 and R16. §5's rehearsals are **R17** (at the code
   commit's parent, which now carries 06b's code) and **R18**, with slugs `r17-parent` and
   `r18-guard`. Both fingerprints must still equal `1b332ffa…`.
5. **§6's "before" is 06b's census,** not 06a's. 06b's code changes values wherever the exit fired
   against its premise, so 06a's run is no longer the parent's.
   - **V0's first clause:** `git diff --stat <06b's code commit> <07's code commit's parent> --
     . ':!docs' ':!prompts'` shows nothing but `CLAUDE.md`. So 06b's census is the parent's.
   - **V0's second clause:** the harness reproduces **06b's** census units bit for bit, on the same
     three rows.
   - **V1** compares with 06b's calls and rows. **V4's** "errors before" are 06b's.
   - **V2 and V3** keep 06a's scan. The scan reads $\theta'$ alone, which 06b does not change.
   - The harness is copied from 06b's census copy, and its diff is saved against that copy.

**Prompt §9 point 6, its bullets for §5 and §6, copied unchanged:**

   - **The rehearsals are renamed again.** 06c takes R17 and R18. §5's rehearsals are **R19** (at
     the code commit's parent, which now carries 06c's code) and **R20**, with slugs `r19-parent`
     and `r20-guard`. Both fingerprints must still equal `1b332ffa…`.
   - **§6's "before" is 06c's census,** not 06b's: point 5 holds with 06c's code commit, 06c's
     census and its harness copy in place of 06b's. V1's bit-identity is against 06c's calls.

V1–V5: **not yet scored.**

## Deviations from the prompt

1. **The interface methods return callables** (`independent_phases()` and
   `independent_phase_rates()`, each a list of callables) rather than values at x. The scan must go
   through `_sample_vectorized()`, which takes a callable. The prompt leaves names and signatures to
   the agent. `IMPLEMENTATION CHOICE`.
2. **A window's phase span is measured from the phase, not estimated on the scan; and every
   phase's cuts are united rather than taking the largest phase's count.** A window can be a few
   scan spacings wide: at c = 1e8 and Phi_w = 2 pi its half-width is sqrt(2 Phi_w / c) = 3.5e-4
   against a spacing of 2.4e-4. A trapezoid estimate on three points cannot place cuts that
   guarantee test 7's "every piece's phase span is at most Delta phi_w". On each monotone segment
   (the window split at that phase's interior roots) the span is |Delta theta|, exact, and each cut
   is found by `brentq` on theta - level. Uniting every phase's cuts makes no piece exceed 2 pi in
   any phase. `STRUCTURALLY REQUIRED`.
3. **No cut at a root: a root's central piece holds it a third of the way through in phase.**
   With the eq. (151) formula unchanged (§0.3), a Chebyshev node on a zero of theta' gives G0 = 0,
   so `_roundoff_floor()` returns +inf, and the call's abserr with it. Two attempts failed, both on
   the C2 problem `theta = 1e6 (x - x^2)` with `theta_deriv` supplied (the scratch comparison,
   not a record):
   - **(a) cuts at the root:** abserr = +inf;
   - **(b) a central piece symmetric in phase about the root,** a cut half a piece above and below:
     at rtol 1e-10 the central piece missed its tolerance and was bisected. For a simple root the
     phase midpoint is the x midpoint, so the root became an endpoint node of both halves: abserr
     +inf again.
   - **As built:** a third of the central piece's phase below the root and two thirds above
     (`_CENTRAL_BELOW`, `_CENTRAL_ABOVE`). For a quadratic phase the root then sits at
     1/(1 + sqrt 2) = 0.414… of the piece in x, an irrational fraction no bisection lands on. The
     same problem then gave value -6.879079710e-4, abserr 6.8e-13 (honest against the test's
     oracle) and `converged` True; the parent gave abserr +inf.
   - The rest of each segment is cut into pieces of equal phase. Two roots closer than one piece
     share a cut that divides the segment between them 1 : 2.
   - The need is `STRUCTURALLY REQUIRED`; the ratio 1 : 2 is an `IMPLEMENTATION CHOICE`.
4. **"Changes sign on its nodes" is read as "is not of one strict sign":** a zero on a node, or a
   non-finite sample, counts as guarded. `IMPLEMENTATION CHOICE`.
5. **When `guard_at_depth_max` is set.** Always, for a guarded region at `depth_max` on the Levin
   branch, which can be accepted no other way. On the fallback branch, only when it is accepted
   *because of* the cap, i.e. unresolved; a guarded fallback region resolved at the cap is accepted
   on resolution as §1.3 says and is not flagged. `IMPLEMENTATION CHOICE`.
6. **`num_guarded_regions` counts every region found guarded when popped**, accepted or bisected,
   so that it says whether the guard fired at all. Accepted regions carry their own `guarded` flag.
   `IMPLEMENTATION CHOICE`.
7. **Window pieces are counted in `num_simple_regions` and `abserr_fallback`** as well as in
   `num_regions` and `abserr`: they are direct quadrature by the same nested pair.
   `used_interval.abserr_fallback` is defined for type `"window"`. `IMPLEMENTATION CHOICE`.
8. **Non-finite scan samples are skipped,** as 06a's scan skipped them, rather than raised.
   `IMPLEMENTATION CHOICE`.
9. **The guard for a basis with several phases.** It samples every phase's closed-form rate on the
   region's nodes, as well as `theta_prime_Cheb`. Those rates reach the subregion functions through
   a new parameter, `guard_rates`, of `_adaptive_levin_subregion` and
   `_adaptive_levin_subregion_impl`. For a basis with one phase, `theta_prime_Cheb` is that phase's
   rate and is used alone, at no extra evaluation. `IMPLEMENTATION CHOICE`.
10. **Diagnostics beyond §1.4.** Each point also carries `found_as`, `theta`, `own_window`, and
    `own_window_edges` (theta', theta'' by a central difference, and theta'^2/|theta''| at each edge
    of its own window). The result carries `unlocated_minima` (each minimum found whose lambda was
    not below Lambda_c). They are for §2's table and §6's descriptive measures.
    `IMPLEMENTATION CHOICE`.
11. **Merging is within a phase.** Two phases' points that coincide are kept as one point each, and
    their windows overlap and merge. `IMPLEMENTATION CHOICE`.
12. **The precision constants.** eps is float64's, 2.22e-16 (the module's `MACHINE_EPSILON` is the
    approximate 1e-16). `brentq` uses xtol = eps max(|x_a|, |x_b|), finer than §1.1's 4 eps, and
    rtol = 4 eps, scipy's least. A root is then within about 5 eps |x| of the computed rate's zero.
    `IMPLEMENTATION CHOICE`.
13. **theta''' at a minimum:** a central second difference of the rate,
    (theta'(c + s) - 2 theta'(c) + theta'(c - s)) / s^2, with step s = h/2 (half the scan spacing).
    The centre c is the refined minimum, moved inward if need be so that the stencil stays in
    `x_span`. Truncation error s^2 |theta^(5)| / 12, zero for the cubic family (C); rounding about
    4 eps max|theta'| / s^2. On (C) at b = 1e4, lambda = 1024 (a = 2189, s = 1.2e-4) that is 6.6e-5
    against theta''' = 2e4, 3e-9 relative. The bounded-Brent tolerance is xatol = 4 eps max|x|,
    which scipy floors at sqrt(eps)|x|. `IMPLEMENTATION CHOICE`.
14. **The new direct paths keep a reversed span's sign.** `_adaptive_levin_subregion_cc()` ignores
    the orientation of `x_span` (`_cc_weights` takes |b - a|), so on a reversed span its value has
    the wrong sign. That is a defect of the existing fallback (observation O1). A window piece, and
    a guarded region at `depth_max`, call it through `_oriented_cc()`, which integrates on the
    ascending span and negates the value. Without this a reversed span with a located point gave
    +0.0017451 against the forward span's +0.0019426 ((R) c = 1e6). With it, the two sum to 2e-19.
    The existing fallback is not touched. `STRUCTURALLY REQUIRED`.
15. **Provisional constants.** Phi_w = 32 pi and Lambda_c = 64 stand in the module until the user
    settles them. Every §2 run sets its rung explicitly. `IMPLEMENTATION CHOICE`.
16. **The test-only switch** is a private keyword of `_adaptive_levin`,
    `_test_only_bypass_location`, off by default. It is not a parameter of the facade, which changes
    only in its docstring. It skips the location; the nodal guard still runs.
    `IMPLEMENTATION CHOICE`.
17. **`scipy.optimize` is imported inside `_locate_stationary_points`,** not at module scope, to
    keep this module's import cost (a concern its docstring records). `IMPLEMENTATION CHOICE`.
18. **(R)'s omega is read as 0 for every case.** §2 lists c and x_s, and says omega = 0 puts the
    root at x_s; no other omega is named. `IMPLEMENTATION CHOICE` (a reading; the stop report asks).
19. **§2's phases are handed over as `theta` and `theta_deriv` only,** with no `theta_mod_2pi`. Both
    families vanish at their stationary point (theta(x_s) = 0, theta(x_0) = 0), so the raw phase is
    small inside every window, and outside one the Levin endpoint error eps |theta| |p| is about
    eps |theta / theta'|, at most about eps. `IMPLEMENTATION CHOICE`.
20. **§2's tables carry the dispatch SHA's result on every case, for context.** It is not part of
    either rule. `IMPLEMENTATION CHOICE`.
21. **Records overwritten and scripts fixed before a kept run.** `out/lg_location.json` was first
    written without the parent comparison, then by the extended script; and it is to be re-run on
    the final code. `reference.py`'s first `literal()` read x_s from decimal text at mpmath's
    default 15 digits, not the exact double, and differed from the closed form by 6e-18. That was
    found and fixed before `validate_reference.py` ran. `calibrate.py` was exercised for mechanics
    only, on a scratch copy writing to the session scratchpad: three cases, Phi_w = 8 pi only, no
    record kept. `IMPLEMENTATION CHOICE`.
22. **§1 rebased onto 06b's code** (prompt §9 point 1). Five conflict hunks, resolved as **The
    resume** above says: 07's bar on the round-off-limited exit applied before 06b's premise check,
    06b's halves solved with 07's `guard_rates`, children carrying both `estimate=` and `kind=`.
    `STRUCTURALLY REQUIRED`.
23. **The references as the user ruled** (§9 points 2–3). `reference.py` gains omega in (R)
    (`closed_R(c, x_s, omega)` as J_0(alpha + i omega)) and `contour_R`, a second (R) reference by
    Cauchy's theorem along the steepest-descent line through the saddle. The omega = 1 cases are
    the agent's choice, recorded in `reference.OMEGA_ONE_CASES`: c = 1e4, 1e6, 1e8 at x_s = 0.38…
    and c = 1e8 at x_s = 0.72…, so the shift 1/c spans 1e-4 to 1e-8 and both x_s are covered.
    `calibrate.py` computes both (R) references for every case, uses the closed form, and records
    their disagreement; its table gains the point's offset from x_s - omega/c beside that from
    x_s. `IMPLEMENTATION CHOICE`.
24. **The first stop's records kept; the live scripts re-targeted.** The scripts and outputs of
    the first stop are copied as they ran into `calibration/stop1/`, byte for byte; they are
    records and do not run in place (`prov.py` locates the repository by its depth). The live `lg_location.py`,
    `lg_repeat.py`, `parent_compare.py` and `calibrate.py` name the parent `cf3efac` (06b's code);
    `parent_compare.py` gains 06b's two premise problems; `prov.py`'s header also records the
    SHA-256 of every script beside it (a script's inputs, such as `reference.py`);
    `time_reference.py` and `lg_seed10.py` live only in `stop1/`, being records of that stop.
    The LiouvilleGreen scripts now run with their working directory in the session scratchpad,
    because those tests write plots to it (O6). `IMPLEMENTATION CHOICE`.
25. **A supplementary (C) measurement** (`calibrate_supplement.py`) beyond §2, run because §2's
    rule picked no Phi_w and so never reached (C): (C)'s exact procedure repeated at every rung,
    labelled as not the rule, for the user's decision. `IMPLEMENTATION CHOICE`.
26. **§1 rebased onto 06c's code** (prompt §9 point 6). 06c rewrote `_adaptive_levin` as a global
    budget over a partition of leaves; `git stash apply` left ten conflict regions. Rather than
    edit them in place, the module was rebuilt from 06c's file (`ad63e8a`): the 22 hunks of 07's
    diff against 06b that lie outside the driver applied cleanly (one landed by fuzz on
    `_write_progress_data`'s call and was reverted), and 07's driver changes were written afresh
    for the leaf model:
    - `_levin_leaf` gains `kind` ("levin" or "window"); `_solve`, `_make_leaf`, a fallback leaf's
      halves, the trust check's grandchildren and a bisected leaf's children all carry it, so a
      window leaf, its halves and its grandchildren are solved by the nested pair
      (`_adaptive_levin_window_piece`), and a Levin leaf's solves pass `guard_rates`;
    - the location runs before the first leaf is made; the initial partition is the window pieces
      and the ranges between windows, each a leaf at depth 0, oriented as `x_span`; with nothing
      located it is 06c's single leaf;
    - **window leaves and guarded leaves are never floor-limited** (§9 point 6's restatement of
      (ii)), so they stay refinable until `depth_max`;
    - **a guarded Levin leaf is chosen like an untrusted one:** refinable whatever the budget,
      before every trusted leaf, and the loop does not stop while one remains below `depth_max`
      (07 §1.3, "never accepted on the Levin branch", in the global loop);
    - **a guarded leaf at `depth_max`** is integrated by the nested pair when made (a Levin leaf's
      estimate replaced by the pair's), flagged `guard_at_depth_max`, exempt from the trust check,
      and the call is not converged while one is returned.
    The diff against 06c's file removes only lines 07 replaces. `STRUCTURALLY REQUIRED`.
27. **`guard_at_depth_max` on the fallback branch** is now set on every guarded fallback leaf at
    `depth_max`, not only one accepted unresolved (deviation 5): the global loop has no per-leaf
    "accepted on resolution". `IMPLEMENTATION CHOICE`.
28. **The scripts re-targeted at 06c** (`ad63e8a`, whose module equals the code commit
    `ad9a5f4`'s): `PARENT_SHA` and the docstrings naming the parent. The second stop's scripts and
    outputs are kept as they ran in `calibration/stop2/` (each output's script hash matches its
    file there), and the live copies of outputs this tree has not reproduced were removed, being
    identical to `stop2/`'s. `calibrate_supplement.py` lives only in `stop2/`. A new record,
    `stop3_diagnose.py`, measures the third stop. `IMPLEMENTATION CHOICE`.
29. **§1 rebased onto 06d's code** (prompt §9 point 7): one conflict hunk, both sides kept, as
    **The fourth stop** says; one docstring passage of 07's corrected, its eq. (151) reason having
    gone. The third stop's scripts and outputs kept as they ran in `calibration/stop3/`; the live
    scripts name `e5764a6` as the parent; two new records, `stop4_diagnose.py` and
    `stop4_lg_seed0.py`. `STRUCTURALLY REQUIRED`.
30. **Two long LiouvilleGreen runs left unattended.** `lg_repeat.py` was launched in the
    background twice, unregistered (this dispatch may not write `var/`), and I ended my turn
    waiting on it; the first I misread as finished, so the second duplicated it. Both ran for
    hours without completing a seed until the orchestrator found them; both were ended by pid.
    Nothing was written by either. `UNINTENDED DRIFT`.

## Verification performed (so far)

- **At the resume:** AdaptiveLevin 38 OK after the rebase (06b's six tests included); the final
  seven-roots run is *pending*. Everything below this item, to the end of this list, is the first
  stop's, on the tree before the rebase (`stop1/`); the resume's checks are in **The resume** above.
- **The seven roots, midway** (after §1, before deviation 14's orientation fix): AdaptiveLevin 32 OK, ComputeTargets 780 OK (266.8 s), CosmologyModels 39 OK,
  LiouvilleGreen 148 OK (skipped=1), ObjectFactories 280 OK, RunDrivers 127 OK, tools 58 OK. All
  equal the baselines. Load averages 5–9 (`uptime` beside each, in the session's record).
- **Against the parent** (`stop1/out/parent_compare.json`): the parent's driver (`git show
  93d5cb1:…`) and the new one on 19 problems of `test_levin_quadrature.py`'s kind, with and without
  a closed-form `theta_deriv`, and the generic m = 4 basis through `_adaptive_levin`.
  - Value, abserr and region list are bit-identical on 18.
  - On C2 without `theta_deriv` the guard fired on 13 regions, and the result is still
    bit-identical.
  - The one that moved is C2 with `theta_deriv`, where the root at 0.5 is located: abserr goes from
    +inf to 6.8e-13 (deviation 3).
- **LiouvilleGreen** (`stop1/out/lg_location.json`, the orchestrator's item 1; at the provisional
  constants, on one unseeded draw). Every test passes (148, skipped 1), over 290 Levin calls, 282 of
  them with rates.
  - The location fires on 19 calls, all `root`, in three tests of `test_3bessel_analytic.py`:
    `test_YJJ_log_singularity` (17), `test_JJJ` (1) and `test_abserr_bounds_truth` (1). An earlier
    run of the same script, on a different draw, saw 21 in two.
  - No minimum was found on any call, and the guard never fired.
  - All 271 calls that located nothing are bit-identical to the same call with the location
    bypassed.
  - On the 19, the value moved from the parent's by up to 0.95 of the new abserr, and by up to 184
    times the parent's. One call that the parent reported converged is now unconverged.
  - Over 20 seeded draws (`stop1/out/lg_repeat.json`), one test fails on one draw. That was the
    first stop.

## Observations not acted on

- **O1. The Clenshaw–Curtis fallback ignores a reversed span.** `_adaptive_levin_subregion_cc()`
  weights by |b - a|, so on `x_span = (b, a)` with b > a every fallback region contributes with the
  wrong sign. At the parent, GRZ at lambda = 10 (all fallback) gives +0.2 over (-1, 1) and +0.2 over
  (1, -1). `test_reversed_span` does not see it, since theta = 100x makes every region Levin. Every
  production `x_span` is ascending (`QuadSourceIntegral`, `three_bessel_integrals`). Prompt 07's
  new paths avoid it (deviation 14). *To open on the board's §3 at the commit.*
- **O2. Carving out a window can leave outer Levin ranges that bisect to the cap.** A scratch smoke
  test, not a record: (C), b = 1e8, lambda = 64, atol 1e-18, rtol 1e-10, minimum located. Far from
  the window, at x in [0.70, 1.00], regions bisected to depth 20 (14,814 regions, `converged`
  False). The value is honest, 7.6e-15 off against abserr 3.3e-10. The parent accepted the whole
  span in 12 regions at depth 8, honest with error 6.2e-15 against abserr 7.7e-15, but not
  converged. Neither residual nor floor reaches rtol |value| there. The parent's region holding the
  slow zone had a floor inflated by its small G0, and so exited through `phase_limited`.
  *At §2's tolerances on the rebased tree:* the (C) runs with a minimum located reach depth 11 at
  most; on (R), c = 1e8, x_s = 0.38, omega = 1 reaches depth 19 (454 regions) at 32 pi and depth 20
  (3,590 regions) at 512 pi, still honest and converged. The parent reaches 14 on that case. This
  bears on V5.
- **O3.** `handover` B2's probe for issue 13 ran in this main checkout while this prompt's
  uncommitted edits to `levin_quadrature.py` were present
  (`docs/handover/b2/probe13/README.md`). That session re-ran in a clean worktree and found every
  figure identical.
- **O4.** The scratchpad held files this session did not write (`baselines`, `marker-07a`). They
  were not touched.
- **O5.** `test_3bessel_analytic.py` draws its wavenumbers unseeded on every run, so its
  assertions sample a different triangle each time. That is how a move can appear on one draw in
  twenty (the first stop), and why a single passing run of the root does not show that nothing
  moves.
- **O6.** Those tests also write diagnostic plots into the working directory
  (`three_bessel_debug/`). The first stop's LiouvilleGreen scripts ran from `calibration/scripts/`
  and left three such directories there; they were removed, and the scripts now run from the
  scratchpad (deviation 24).
- **O7. Per-region relative acceptance against a cancelling total.** Every unconverged (R) case of
  §2 is the smaller, cancelling integral (x_s = 0.7236, omega = 0). Each window piece is accepted
  on rtol against its own value; their summed nested-pair estimates then exceed rtol against the
  total, by 1.01–6.7 times. The per-region `resolved` test is outside 07's reach (§0.3), and this
  is what keeps every Phi_w rung from qualifying. It bears on G3's Q4 (v), what `converged`
  means.

## State handed on

*Pending.* At the third stop: §1 rebased onto 06c (`ad63e8a`) and uncommitted, the module staged as
resolved; the orchestrator's stashes and `var/levin-phase-floor-07/wip/` untouched. §2 not run on
06c (it cannot complete); the second stop's tables stay in `calibration/stop2/`. No §3 test
written. The constants in the module are still the provisional 32 pi and 64.
