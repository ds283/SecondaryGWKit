# levin-phase-floor 07 — the stationary-point guard: locate them first, integrate their windows directly

> **Part A, §1–§4** (the code, the measurement, the tests and these records), written by the
> executing agent on the laptop, on the rewritten prompt (2026-11-07). Part B, §5 (rehearsals R27
> and R28) and §6 (the census), runs later on the Mac Studio; the orchestrator appends its record.
>
> **Status of this log:** written at §2's planned stop (2026-11-07, its first part, kept as written,
> with §2's first run), and completed on the resume after the user's rulings the same day (prompt
> §9, commit `91472dbe`): §1.3 built, §2 re-run in full at `rtol` $10^{-9}$, the constants set by its
> rules, §3's tests written, and the code committed (**The resume**, below). The set-aside work's own log is kept, byte for byte, as
> [`docs/levin-phase-floor/07/earlier-attempts/log-parts-a-1-to-4.md`](../../../docs/levin-phase-floor/07/earlier-attempts/log-parts-a-1-to-4.md).

**Subject:** prompt 07 §1–§4 (Part A), on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `bd21bf23` (pushed). The parent module is `git show
  bd21bf23:AdaptiveLevin/levin_quadrature.py`, 06m's code (SHA-256 `beaf0e23…`).
  `git log --oneline bd21bf23..HEAD` was empty at the start and at the stop.
- **The Gate held:** `git status --short` empty; `runregistrykit list` 82 runs, 80 finished,
  2 `unknown` (pre-registry), nothing `running` or stale; `git stash list` shows the four stashes,
  `19eb784e`, `4e6db654`, `1278f4e5`, `a89ac9be`; the eight files of
  `var/levin-phase-floor-07/wip/` each match `SHA256SUMS.txt`; the 14 captures and `r55.pkl` are
  present in `var/levin-phase-floor-06g-fcc-laptop/inputs/`, each matching 06g's recorded SHA-256
  (checked again by `replay.py`); the seven roots at the dispatch SHA equal the orchestrator's
  (Verification).
- **The set-aside work** was read only: `git stash list` and `git stash show -p 'stash@{0}'`
  (saved to the session scratchpad); the latest tarball extracted into the scratchpad. No other
  stash subcommand was run, and `var/levin-phase-floor-07/wip/` was not written.
- No Ray was started (`ray.cloudpickle` was imported to read the captures; every replay records
  `ray.is_initialized()` False), no store was opened, no run was registered, and nothing was
  written under `var/`. Read from `var/`: the captures, 06g's capture harness (for its pickler's
  reducer) and the set-aside work.

**Commit:** the code commit of this log (one commit, not pushed); dispatched at `bd21bf23`, resumed at
`91472dbe` (two commits had landed: `945c8668`, which opens
`[07-cc-interior-nodes-shift-with-a-rounded-midpoint]`, and `91472dbe`, the prompt's amendments; the
board and `docs/OPEN_ISSUES.md` merged by reading).

**Result at the end (after the resume):**
- **§1.3 is QUADPACK's `iroff1`, counted per window** (prompt §9 point 2), built on §1 with the
  provisional constants before any §2 run.
- **§2 re-run in full at `rtol` $10^{-9}$** on that code (§9 points 1 and 4), its outputs in
  `docs/levin-phase-floor/07/calibration/out/run2/`, the first run's kept in `out/run1_rtol1e-10/`.
  **Both rules pick a rung at $10^{-9}$: $\Phi_w = 2\pi$ and $\Lambda_c = 4$**, so the fallback to
  $10^{-8}$ was not run. **All 14 of §2.3's calls are honest and converged at $2\pi$, z12 first**
  ($|v - {\rm ref}|/{\rm abserr} = 0.718$, against 1,681 at the parent). No LiouvilleGreen draw fails;
  3,897 of 3,897 calls that locate nothing are bit-identical to the parent. So, under §9 point 4,
  the agent went on to §3 without a further stop.
- **The constants** are set in the module, each comment citing the measurement.
- **§3:** `AdaptiveLevin/tests/test_levin_stationary.py`, 12 tests (1–10, with 10 (a)–(c)); test 11 in
  `ComputeTargets/tests/test_quadsource_levin_guard_counts.py`; one existing assertion changed, the
  key set of `test_levin_trust_yardstick.py`'s test 2, under §3 test 12's amended allowance. All
  nine breakages caught. The seven roots: AdaptiveLevin 91 → 103 and ComputeTargets 780 → 781, the
  rises equal to the tests added; the other five equal (Verification).
- **A finding for the user, not a stop** (§1.3, §9 point 4): at the chosen rung $2\pi$ the rule stops
  the window of eight of §2.3's fourteen calls (z00, z12, w00, w01, w07, w09, w10, w11) while some of
  its leaves' errors fell at their last bisection; every one is converged and honest.
- **Issues opened:** `[07-cc-fallback-ignores-a-reversed-span]` and
  `[07-cc-node-term-takes-one-phase-per-node]` (Observations O1 and O7).

**Result at the first stop:**
- **§1 is built** on the dispatch SHA's driver (What shipped). On every call where nothing is
  located and no leaf is guarded it is bit-identical to the parent: 3,803 of 3,803 such
  LiouvilleGreen calls, test 2's nine cells of `test_levin_trust_yardstick.py`, and r55.
- **§2.2: no rung of $\Phi_w$ qualifies.** Every (R) case is honest at every rung, and the
  location finds each root exactly; but one case, $c = 10^8$, $x_s = 0.7236$, $\omega = 0$ (the
  cancelling integral, $|I| = 1.4\times10^{-5}$, request $1.4\times10^{-15}$), is unconverged at
  every rung (node-limited at 1.24, 1.20, 1.02 times its request at $2\pi$, $8\pi$, $32\pi$; did not
  finish in an hour at $128\pi$ and $512\pi$). The parent converges it. Every other (R) case is
  honest and converged at $2\pi$, $8\pi$ and $32\pi$. $\Lambda_c$ (supplementary): 4 at every rung,
  the check holding at every rung.
- **§2.3:** the replay check reproduces all 15 captures bit for bit at 06g's code, and z00, z12 and
  r55 at the parent. V1's twelve are honest at every rung (largest $|v - {\rm ref}|/{\rm abserr}$
  0.733). **z12 and z00 do not finish** at any rung within 900 s; at bounded depth 12, z12 is
  honest at $2\pi$–$128\pi$ (0.72, 0.0012, 0.76, 0.91).
- **§2.4:** five `test_JJJ` draws (seeds 0, 4, 11, 13, 17) do not finish under §1's code within
  1200 s; they pass under the parent. Every other test passes under both.
- **§2.5: window leaves are chased to `depth_max` in two ways**, both reached on real calls:
  (A) their summed Clenshaw–Curtis floor is above the request and flat in $h$, and they are never
  floor-limited; (B) their nested-pair differences sit at the integrand's own noise, the trust
  check finds them untrusted, and an untrusted leaf is refined whatever the budget says. Lifting
  the bar on the floor-limited exit for window leaves ends (A) on every case measured and (B) on
  V1's twelve, not on z00 or z12.
- **Four ComputeTargets tests do not finish** under §1's code (mechanism (A), on three-Bessel
  blocks at rtol $10^{-10}$ to $10^{-12}$), and `test_levin_trust_yardstick.py`'s test 2 fails on
  the new result keys. Both are §8 conditions, reported at this stop.

## What shipped (uncommitted at §2's stop)

**`AdaptiveLevin/levin_quadrature.py`** (built afresh on the dispatch SHA's driver; the set-aside work
read as a reference, §0.5; what was taken from it is listed under **Code taken from the set-aside
work**):
- **§1.0, the basis interface.** `_Basis_SinCos.independent_phases()` returns `[theta["theta"]]`;
  `independent_phase_rates()` returns `[theta["theta_deriv"]]`, or `None` without it (deviation 1).
  A basis without the methods is treated as unable to supply rates. The module docstring gains a
  section, THE BASIS INTERFACE, listing every basis method the driver calls.
- **§1.1–§1.2, location and windows:** `_locate_stationary_points()`, run by `_adaptive_levin()`
  before its loop when the basis supplies rates and `x_span` is not empty. The scan, roots, minima,
  ends and merging are §1.1's (deviations 10–12); the windows are §1.2's, cut on the phase itself
  (deviations 2, 3). Helpers: `_scalar_callable`, `_second_difference`, `_first_difference`.
- **The forced path** (§1.2 item 4): `force_direct` on `_adaptive_levin_subregion()` and
  `_adaptive_levin_subregion_impl()`, which skips the C2 gate and calls `_oriented_cc()`
  (deviation 4).
- **Seeding** (§1.2 item 5): with windows, the initial partition is the ranges of `x_span` between
  windows (ordinary leaves) and the window pieces (window leaves, `forced_direct`), all at depth 0,
  oriented as `x_span`; with nothing located, today's root leaf, unchanged.
- **Window leaves in the loop** (§1.2 item 6): their halves and grandchildren are solved by the
  forced path; they are never floor-limited (the floor-limited test is not taken on them), stay
  refinable until `depth_max`, and keep their Clenshaw–Curtis estimate there.
- **§1.4, the guard:** `_nodal_guard()`, computed in every Levin solve from `theta_prime_Cheb` (and
  each phase's rate for a basis with several, `guard_rates`; deviation 9), carried as `guarded` in
  the solve's dict. A guarded leaf below `depth_max` has error $+\infty$ and is never floor-limited
  (deviation 5); at `depth_max` it is re-solved by the forced path, flagged `guard_at_depth_max`,
  exempt from the trust check, and the call is not converged while one is returned (deviation 6).
- **§1.5, the result:** `stationary_points`, `stationary_windows`, `unlocated_minima`,
  `num_window_pieces`, `num_guarded_regions`, `num_guard_at_depth_max` (deviations 7, 8); each
  `used_interval` gains `window`, `guarded` and `guard_at_depth_max` (shown in `__str__` only when
  set). Docstrings: the module docstring's account of stationary points corrected (the C2 item no
  longer calls gate-and-bisect "the correct structural response") and a new item for the guard;
  `_roundoff_floor()`, `_adaptive_levin()` (a new section, *The stationary-point guard*),
  `_adaptive_levin_subregion_impl()` (where the C2 gate's account was incomplete), the basis interface,
  and `adaptive_levin_sincos()` (that `theta_deriv` now drives the location, and the new keys).
  `adaptive_levin_sincos()` is otherwise unchanged.
- **The constants:** `_LEVIN_STATIONARY_SCAN_POINTS = 4097`, `_LEVIN_WINDOW_PIECE_PHASE = 2*pi`, and
  the provisional `_LEVIN_WINDOW_HALF_PHASE = 32*pi` and `_LEVIN_COALESCENCE_LAMBDA = 64`
  (deviation 14).
- **The test-only switch** `_test_only_bypass_location` of `_adaptive_levin()`, off by default, read
  only at the location; the facade does not expose it.

**`ComputeTargets/QuadSourceIntegral.py`** (§1.6): `phase_group_Levin_integral`'s per-group record and
`_three_bessel_Levin`'s per-call record (its `metadata`, one entry per J/Y block) each gain
`stationary_points` (the number located), `window_pieces`, `guarded_regions` and
`guard_at_depth_max`. Nothing else changes; no schema changes (both records are JSON metadata).

**`docs/levin-phase-floor/07/`:**
- `calibration/scripts/` (formatted by `black` before every recorded run): `prov.py` (06m's,
  re-targeted), `reference.py`, `validate_reference.py`, `validate_reference_precision.py` (the
  set-aside work's, re-validated), `analysis07.py`, `calibrate.py`, `replay.py`, `lg_seeded_draws.py`
  and `variant_loader.py` (06m's harness, changed where marked), `lg_compare_07.py`, `ct_stuck_call.py`,
  `ct_stuck_tests.py`, `lg_stuck_units.py`, `z12_untrusted_leaves.py`, `check_earlier_attempts.py`;
  `calibration/out/`, their outputs, with `scripts_diff.txt` (the copied scripts against their
  sources);
- `earlier-attempts/`: the latest tarball's `docs/levin-phase-floor/07/calibration/` (as
  `earlier-attempts/calibration/`) and its log (as `log-parts-a-1-to-4.md`), byte for byte, with
  `SHA256_CHECK.txt` (61 files, all equal).

**Not yet:** no test of §3 is written; §1.3 is not built; the board, `CLAUDE.md` and
`docs/OPEN_ISSUES.md` are untouched; nothing is committed.

### Code taken from the set-aside work (§0.5)

Read from `stash@{0}`'s patch (`git stash show -p 'stash@{0}'`, equal in content to
`var/levin-phase-floor-07/wip/07-part-a-rebased-on-06d-tracked.patch`) and from the latest tarball
(`07-part-a-rebased-on-06d-untracked.tar.gz`). Each piece was re-fitted to today's driver by hand:
- **`AdaptiveLevin/levin_quadrature.py` (from the patch):** `_strictly_one_sign`, `_nodal_guard`
  (unchanged in substance), `_oriented_cc` (changed: it now samples $\theta'$ afresh on the ascending
  span and negates the signed node error too, which 06d–06g added), `_scalar_callable`,
  `_second_difference`, `_first_difference`, `_locate_stationary_points` (changed: the edge search
  pre-screens on the phase sampled on the scan; a root is a cut, without the 1 : 2 central piece,
  deviation 3; `found_as` set before the cut test), `_Basis_SinCos.independent_phases` /
  `independent_phase_rates`, the provisional constants and their names, the result keys
  `stationary_points`, `unlocated_minima`, `num_window_pieces`, `num_guarded_regions`,
  `num_guard_at_depth_max`, and the module-docstring paragraphs on the C2 gate and on location
  (rewritten). Not taken: its loop changes (`kind` on leaves, `_must_split`, its window-piece
  solver), written for the loop of 06c–06d; the driver changes here are new.
- **`ComputeTargets/QuadSourceIntegral.py` (from the patch):** the four keys of the group record.
  The same four in `_three_bessel_Levin`'s per-call record are new.
- **`docs/levin-phase-floor/07/calibration/scripts/` (from the tarball's
  `docs/levin-phase-floor/07/calibration/scripts/`):** `reference.py`, `validate_reference.py` and
  `validate_reference_precision.py`, changed only in their docstrings' section numbers and their
  provenance header (diff in `out/scripts_diff.txt`); `calibrate.py`'s families, phase dicts and
  table layout (the script is rewritten).

## Section 2, first run: the measurement on §1's code without §1.3 (at the first stop)

*Kept as written at the stop. Its outputs, cited below as `out/…`, were moved, unchanged, to
`out/run1_rtol1e-10/` before the second run (deviation 29); its scripts are the current ones' earlier
versions, each output's header naming the SHA-256 it ran with.*

Every output in `docs/levin-phase-floor/07/calibration/out/` opens with its provenance: the
SHA-256 of its script, of `prov.py`, of `AdaptiveLevin/levin_quadrature.py` as measured
(`13294b2a…` for every run below) and of each input; `HEAD` `bd21bf23`; and the tree dirty
(`git status --short`: the two modified files and `docs/levin-phase-floor/07/`). Every script was
formatted by `black` before the run recorded. The machine was shared and heavily loaded (load
averages 15–125, another session's `tools/compare_stores.py` among the processes); wall times
are recorded, and every wall-clock limit below is a limit on the measurement, not a result.

### §2.1 The references

`out/validate_reference.json`, `out/validate_reference_precision.json`:
- **(R) by the error function and by contour integration, all ten cases** (six at $\omega = 0$,
  four at $\omega = 1$): **largest relative disagreement $2.19\times10^{-37}$**; the error function at
  40 against 60 working digits likewise at most $2.19\times10^{-37}$. Not a stop.
- **Against `mpmath.quad` subdivided to convergence** (pieces of at most $\pi$ of phase, split at
  every zero of $\theta'$ and at $x_0$), where that takes minutes: (R) at $c = 10^4$ (all three
  cases) and at $c = 10^6$, $x_s = 0.38$, $\omega = 0$ (144,803 pieces, 506 s); (C) at $b = 10^4$ (all
  seven $\lambda$). At most $1.9\times10^{-29}$ relative, except (C) $b = 10^4$, $\lambda = 0.25$ at
  $1.4\times10^{-26}$, where the 30-digit literal is the one in error: at 45 digits on $\pi/4$ pieces
  it equals the contour to $5.3\times10^{-48}$ at 60 working digits.
- **The (C) contour's omitted tail** is bounded by at most $10^{-60}$ on every case.

### §2.2 The two constants (`out/calibrate.json`, `out/calibrate_tables.md`; 8,559 s)

Every run: `rtol` $10^{-10}$, `atol` $10^{-14}|{\rm ref}|$, `chebyshev_order` 16, `depth_max` 20;
each call in its own process with a one-hour limit.

**$\Phi_w$ on (R).** Honest and converged, by rung (true error / abserr / abserr ÷ request in the
tables):

| $\Phi_w$ | honest | converged | the cases not converged |
|---|---|---|---|
| $2\pi$ | 10 / 10 | 9 / 10 | $c=10^8$, $x_s=0.7236$, $\omega=0$: node-limited, abserr/request 1.24, depth 20 (40 ordinary Clenshaw–Curtis leaves at depth 20, 22 untrusted) |
| $8\pi$ | 10 / 10 | 9 / 10 | the same case: node-limited, 1.20, depth 20 |
| $32\pi$ | 10 / 10 | 9 / 10 | the same case: node-limited, 1.02, depth 20, 168,780 window leaves (283 s) |
| $128\pi$ | 9 / 9 finished | 7 / 10 | the same case did not finish in an hour; $c=10^6$, $x_s=0.7236$, $\omega=0$ node-limited (1.00, 15,706 window leaves); $c=10^8$, $x_s=0.7236$, $\omega=1$ node-limited (1.00, depth 20) |
| $512\pi$ | 7 / 7 finished | 5 / 10 | three did not finish in an hour ($c=10^6$ and $10^8$ at $x_s=0.7236$, $\omega=0$; $c=10^8$, $x_s=0.7236$, $\omega=1$); $c=10^8$, $x_s=0.382$, $\omega=0$ and $\omega=1$ node-limited (1.00, depth 20, about 2.3e5 window leaves) |

- **The parent** (06m's code) is honest and converged on all ten (R) cases, the stuck one included
  (abserr $1.35\times10^{-15}$ against a request of $1.41\times10^{-15}$, 68 regions).
- **The location** finds every root exactly: point $- (x_s - \omega/c) = 0$ to the printed digit on
  every case and rung; at $\omega = 1$, point $- x_s = -1/c$.
- **$\theta'^2/|\theta''|$ at the window's edges** is $2\Phi_w$ (12.57, 50.27, 201.1, 804.2, 3217)
  except where the window is clipped at an end of $[0, 1]$.
- **Node-limited** is the only way a finished (R) case fails, every time with abserr within
  1.0–1.24 times the request.
- **$\Phi_w$ by the rule: none** (a stop under §2.2 itself). Setting aside the one case, the rule's
  criterion holds at $2\pi$, $8\pi$ and $32\pi$, so it would pick $2\pi$.

**$\Lambda_c$ on (C): not run under the rule,** which needs a $\Phi_w$. Supplementary, not the
rule: (C)'s procedure at every rung of $\Phi_w$.
- With no minimum located nothing is located on (C), so the result is the same at every rung,
  and equal to the parent's: **every case honest** at both $b$ (err/abserr 3.7e-5 to 0.057).
  Unconverged: $b = 10^8$ at $\lambda = 16$, 64, 256, 1024 (abserr/request 361, 55, 854, 336, the
  parent's too: a request below the round-off the driver declares).
- **The rule would pick $\Lambda_c = 4$ at every rung.** The check, $\lambda = 0.25$ and 1 at both
  $b$ with the minimum located: **honest and converged at every rung** (20 of 20 runs; e.g. at
  $2\pi$ err/abserr 5.7e-5 to 2.1e-3).

### §2.3 The real calls (`out/replay.json`, `out/replay_tables.md`)

Each call in its own process; a 900 s limit (a first run with a one-hour limit was stopped and its
two finished units kept, deviation 17); a call that did not finish was re-run at `depth_max` 8
and 12. The scratch worktree at `3ae6f3a4` was made with `git worktree add --detach`, its `venv`
symlinked (the main `.git/info/exclude` already lists `venv`), status clean.

**The replay check: all 15 reproduce.** At `3ae6f3a4` every capture reproduces its 06g record bit
for bit: value and abserr by `float.hex`, regions, maximum depth and `converged` (z00: 249 regions,
depth 20, unconverged; z12: 135, 18, converged; r55: 1, 0). At the parent, z00, z12 and r55
reproduce route B's records in `06l/out/partC/h1_calls.csv` (ratio, `converged`, regions: z00
0.018276897852472537, False, 225; z12 1681.2986886703777, True, 112), once the ratio is formed as
06l formed it (deviation 18).

**At each rung** (|v − ref|/abserr against 06g's 30-digit reference):

| | parent | $2\pi$ | $8\pi$ | $32\pi$ | $128\pi$ | $512\pi$ |
|---|---|---|---|---|---|---|
| **z12** | 1681, converged | did not finish (depth 12: 0.718) | did not finish (12: 0.0012) | did not finish (12: 0.762) | did not finish (12: 0.908) | did not finish (8: 0.930) |
| **z00** | 0.0183, unconverged | did not finish (12: 0.718) | did not finish (12: 0.0012) | did not finish (12: 0.762) | did not finish (12: 0.908) | did not finish (12: 0.930) |
| **w00–w11, honest** | 12 | 12 | 12 | 12 | 12 | 11 (w07 did not finish; 12: 0.0015) |
| **w00–w11, largest ratio** | 0.942 | 0.218 | 0.141 | 0.113 | 0.224 | 0.733 |
| **w00–w11, converged** | 3 | 8 | 1 | 0 | 0 | 0 |
| **w00–w11, at depth 20** | 9 | 4 | 12 | 12 | 12 | 11 |
| **r55** (control) | 0.0267 | nothing located; value and abserr bit-identical to the parent's at every rung |||||

Each w call locates one root, near $z \approx 10211$. **The verdict at the chosen rung cannot be
given: no rung was chosen.** At every rung **z12 does not finish**, so whether it is honest is
undecided; at bounded depth it is honest at $2\pi$–$512\pi$ (depth 8 at $512\pi$), against 1,681 at
the parent. V1's twelve are honest at every rung where they finish.

### §2.4 The LiouvilleGreen draws (`out/lg_seeded_draws.jsonl`, `out/lg_seeded_draws_report.txt`, `out/lg_compare_07.txt`)

06m's harness, modes `parent` and `commit`, seeds 0–19, four tests (160 units, each in its own
process, 1200 s limit; 06m's longest unit took 35.5 s), at the provisional $\Phi_w = 32\pi$,
$\Lambda_c = 64$.
1. **Tests failing under the code that pass under the parent: five,** all `test_JJJ`, seeds 0, 4,
   11, 13 and 17, each **did not finish** in 1200 s. Every other unit passes under both.
2. **Bit-identity:** of 4,220 paired calls, **3,803 locate nothing and guard nothing, and all
   3,803 are bit-identical** to the parent's (value and abserr by hex, every returned region's
   fields, every other result key). The five unfinished units' calls are not in the pairing.
3. **What location does:** 417 calls locate a point (416 roots, one minimum: seed 19 `test_JJJ`
   call 5, $\lambda = 1.92$, 17 window pieces); no call guards a leaf; no point is at an end. By test:
   `test_YJJ_log_singularity` 392, `test_abserr_bounds_truth` 20, `test_JJJ` 5. On the 417 the value
   moves from the parent's by at most 0.032 of the parent's abserr (0.022 of the new); 416 are
   converged under both, one goes from converged to unconverged (seed 1 `test_YJJ_log_singularity`
   call 83, depth 20).

### §2.5 What still does not finish

From `out/calibrate_tables.md` and `out/replay_tables.md` (each unfinished or unconverged call's
leaves by class and their ancestries; QUADPACK's counters), `out/ct_stuck_call.json`,
`out/ct_stuck_tests.txt`, `out/lg_stuck_units.txt`. The last three re-run each stuck call at
`depth_max` 10, and at its own `depth_max` by the parent's code and by the **"bar lifted"
variant**: the tree's module with the one condition that bars the floor-limited exit for window
and guarded leaves removed (`calibrate.BAR_TEXT`), loaded as a module of its own.

**(A) Window leaves on a flat floor.** Where a window's summed Clenshaw–Curtis floor is above the
call's request, bisection does not lower it, and window leaves are never floor-limited, so they
are chased to `depth_max`: $3\cdot2^{20}$ to $17\cdot2^{20}$ leaves, hours per call.
- **The stuck ComputeTargets call** (`test_short_range`'s three-Bessel block "analytic J4", rtol
  $10^{-10}$; its window is the whole span, three pieces): the window leaves' summed error is
  $2.363, 2.210, 2.182, 2.176, 2.174, 2.174, 2.174\times10^{-17}$ at depth 0, 2, …, 12, against a
  request of $9.0\times10^{-18}$; all of it the floor less the node term (the nested pair is
  $10^{-22}$). 12,288 leaves at depth 12 took 342 s. QUADPACK's `ier = 2` would have been reached
  at the 12th bisection.
- **Every stuck call found** in three of the four ComputeTargets tests (`out/ct_stuck_tests.txt`:
  six, twelve and six locating three-Bessel calls, at rtol $10^{-10}$, $10^{-12}$, $10^{-12}$; the
  fourth test was not diagnosed) and in the five LiouvilleGreen units (`out/lg_stuck_units.txt`,
  one or two calls per unit, rtol $10^{-10}$) is of this kind, but one: LiouvilleGreen seed 0, call 11,
  where one window leaf is untrusted at the cap (kind (B); the parent and the bar-lifted variant
  both converge it). **The parent** returns each of the others quickly, unconverged, its leaves
  floor-limited (1–61 regions). **With the bar lifted** the window leaves are floor-limited and the
  call returns in under a second, unconverged, with abserr within about 10 % of the parent's.
- **On (R),** the cases that did not finish at $128\pi$ and $512\pi$ are this kind with a
  near-flat error: at bounded depth the window leaves' summed error falls slowly or not at all
  (e.g. $c = 10^6$, $x_s = 0.7236$ at $512\pi$: $2.60, 1.43, 1.44, 1.46\times10^{-14}$ at depth 6, 8,
  10, 12 against a request of $1.4\times10^{-14}$). With the bar lifted, all eight (R) calls that
  left window leaves at depth 20 (or did not finish) finish, honest, unconverged, seven of them
  node-limited (abserr/request 1.02–1.27; 3.64 for one).
- **Today's floor-limited exit would fire on them**: the bar-lifted runs show it firing on 34 to
  1,989 window leaves per call.
- **Whether the call ended node-limited:** at $32\pi$ the stuck (R) case did (168,780 window leaves
  at depth 20 first); the ComputeTargets and LiouvilleGreen calls did not reach an end.

**(B) Window leaves at the integrand's noise, untrusted.** On the real calls (z00, z12, V1's
twelve) the windows' error is negligible against the request ($10^{-4}$ to $10^{-6}$ of it), but
hundreds to thousands of window leaves are returned untrusted at depth 20 (e.g. w00 at $8\pi$: 712
window leaves, 112 at depth 20; z12 at $8\pi$, depth 12: 1,014 of 1,461 window leaves at the cap,
790 untrusted). They hold values of $10^{-32}$ whose nested-pair differences are $10^{-43}$ to
$10^{-42}$ (relative $10^{-10}$ to $10^{-11}$), $r$ between 0.26 and 3.9, $\Delta_1$ above $\nu_B$:
the trust check reads the QSI integrand's own small-scale structure, at relative $10^{-10}$, as
non-convergence (`out/z12_untrusted_leaves.txt`), and an untrusted leaf is
refined whatever the budget says. This is why z00 and z12 do not finish, and why V1's twelve end
unconverged at $8\pi$ and above. **With the bar lifted**, V1's twelve converge at every rung from
$8\pi$ to $128\pi$ (11 of 12 at $512\pi$), all honest, the same values to the printed digits: the
untrusted window leaves become floor-limited and so exempt. z00 and z12 still do not finish
(z00 finished at $8\pi$ and $128\pi$ in 850–890 s, unconverged, 18,618–19,172 regions).

**Outside the windows**, the stuck (R) case at $2\pi$ and $8\pi$ holds its error in ordinary
Clenshaw–Curtis leaves beside the window (40 at depth 20 at $2\pi$, 22 untrusted) and in the call's
node sum; it is the only finished (R) case not converged at those rungs.

**QUADPACK's counters** along the unconverged calls (`dqage`'s `iroff1` and `iroff2`, global, in
bisection order, from the leaves' errors as made): `ier = 2` would have been reached in every
unconverged §2.2 and §2.3 call that locates a point (98 calls), within its first 9–141 bisections,
and in the stuck ComputeTargets call at its 12th.

**The agent's reading for §1.3, stated as a reading.** A rule is needed: without one, real calls
(z00, z12, three ComputeTargets tests, five LiouvilleGreen draws) do not finish, at every rung. Two
mechanisms reach `depth_max`, and the two candidate remedies reach them differently:
1. **(A) is the bar itself.** §0.2 (ii)'s bar was written when a window piece carried the Levin
   solve's eq. (151) floor, whose $1/\min|\theta'|$ grew at the stationary point; since 06d a window
   piece is a Clenshaw–Curtis leaf with `_cc_roundoff_floor()`, which has no $\theta'$ in a
   denominator and is flat under bisection. On every case measured, lifting the bar for window
   leaves (keeping it for guarded Levin leaves) ends (A) at once, with the parent's result.
2. **(B) is the trust check on window leaves** at the integrand's noise; lifting the bar ends it on
   V1's twelve (their untrusted leaves are within their floor), not on z00 or z12.
3. **QUADPACK's `iroff1`** (a bisection whose halves change the value by at most $10^{-5}$ relatively
   while their summed error stays above 0.99 of the parent's) counts on both kinds: a per-window
   counter of that form, stopping a window's refinement after six, would end both, at the cost of
   returning some window leaves unconverged that the bar-lifted exit would return floor-limited.

The ruling is the user's (§1.3; Decisions, **07 fourth stop**: *"Let's go for (b)"*, QUADPACK's
rule, if §2 shows window leaves chased to `depth_max` on a flat error — it does).

### §1.6 The analytic blocks

`QuadSourceIntegral` keeps one record per three-Bessel Levin call: `_three_bessel_Levin`'s
`metadata`, one entry per block `J1_data` … `Y4_data` (`elapsed`, `regions`, `abserr`,
`converged`, `phase_limited`), stored under `analytic_rad`'s metadata (`0pt5`, `2pt5`). The four
counts join each entry there, as they join the group record; `LiouvilleGreen/` is not touched.
§5's rule can read the analytic blocks' counts directly.

## The resume (2026-11-07): §1.3, §2 re-run, §3

**The user's rulings**, relayed verbatim by the orchestrator and written into the prompt as its §9
(`91472dbe`): $\Phi_w$ and $\Lambda_c$ from §2.2 re-run at `rtol` $10^{-9}$, $10^{-8}$ as the
fallback (*"At the moment it's looking like we should re-run §2.2 with `rtol = 1e-9` and proceed for
there. If that fails, then relax to `rtol = 1e-8`, but hopefully `1e-9` works across the board."*);
§1.3 is QUADPACK's `iroff1` counter (*"did we not plan to implement the QUADPACK-style `iroff1`
counter for this issue? … Is that really a problem?"*); the key-set assertion admits the new keys
(*"It's absurd not to allow new keys that were added specifically because they were needed."*); and
the order, §1.3 first, then all of §2 once, then §3 without a further stop if both rules find a rung
and all 14 of §2.3's calls are honest at the chosen rung (*"All looks good. Option 1, please."*).

### §1.3 as built

In `_adaptive_levin()`, with three module constants, QUADPACK's numbers
(`_LEVIN_WINDOW_IROFF1_VALUE_RTOL = 1e-5`, `_LEVIN_WINDOW_IROFF1_ERROR_RATIO = 0.99`,
`_LEVIN_WINDOW_IROFF1_LIMIT = 6`), their comment giving the basis:
- each window leaf carries the index of its merged window (`_levin_leaf.window_index`, inherited by
  its halves);
- when the loop bisects a window leaf $P$ into $L$, $R$ (its own solves, `value_own`; $e'$ the error
  the budget sums for each, `_levin_leaf.error`), the bisection **counts** against the window if
  $|(V_L + V_R) - V_P| \le 10^{-5}|V_L + V_R|$ and $e'_L + e'_R \ge 0.99\,e'_P$;
- **at the sixth** the window is stopped (`_stop_window()`): every leaf of it, and any made later, is
  made floor-limited by `_floor_limit_stopped_window_leaf()` -- not refinable, `phase_limited`,
  trusted `None` (exempt from the trust check); a leaf the check had found untrusted declares
  $\max(\epsilon_P, \Delta_1)/(1 - r)$ if $r < 1$ and $\max(\epsilon_P, \Delta_1 + \Delta_2)$ if not;
  a trusted leaf keeps its widened declaration and returns $V_P$ (deviation 31); its halves and
  grandchildren are dropped;
- each child records whether its bisection lowered the error (`fell_at_last_bisection`), so that a
  stopped window reports how many of its leaves were still falling;
- the result's `stationary_windows` entries gain `iroff1_count`, `stopped` and, for a stopped window,
  `stopped_after_split`, `leaves_at_stop`, `leaves_falling_at_stop` and `max_depth_at_stop`; no new
  result key (deviation 30);
- guarded leaves are not counted; the bar on the floor-limited exit is unchanged for every window
  leaf not stopped and every guarded leaf.
Docstrings: `used_interval.window` and `phase_limited`, `_levin_leaf`, and `_adaptive_levin()`'s
section on the guard (item 2a).

### §2, second run (`out/run2/`; module SHA-256 as measured below)

Run on §1 plus §1.3. §2.2 ran on the module with the provisional constants (`calibrate.py` sets every
rung explicitly); the constants were then set by its rules, and §2.3 and §2.4 ran on the module as
committed (`dc05596c…`). The two differ only in the constants' values and comments (§4). The
scripts that changed for the second run are named in deviation 29.

**§2.1, the references** (`validate_reference.json`, `validate_reference_precision.json`): as in the
first run, to the digit: (R) erf against contour at most $2.19\times10^{-37}$ relative; against
mpmath.quad at most $1.9\times10^{-29}$, except (C) $b = 10^4$, $\lambda = 0.25$ at $1.4\times10^{-26}$,
where the literal is in error.

**§2.2 at `rtol` $10^{-9}$** (`calibrate_tables_rtol1e-09.md`; 49 s):

| $\Phi_w$ | honest | converged | not converged | windows stopped by §1.3 |
|---|---|---|---|---|
| $2\pi$ | 10 / 10 | **10 / 10** | -- | none |
| $8\pi$ | 10 / 10 | 10 / 10 | -- | 3 |
| $32\pi$ | 10 / 10 | 9 / 10 | $c = 10^8$, $x_s = 0.7236$, $\omega = 0$ (abserr/request 1.74) | 10 |
| $128\pi$ | 10 / 10 | 9 / 10 | the same (3.23) | 10 |
| $512\pi$ | 10 / 10 | 9 / 10 | the same (5.95) | 10 |

- At $2\pi$ every case is honest (err/abserr at most 0.030) and converged (abserr/request at most
  0.885), the location exact (point $-$ $(x_s - \omega/c) = 0$), $\theta'^2/|\theta''| = 12.57 = 2\Phi_w$
  at every edge. **$\Phi_w$ by the rule: $2\pi$.**
- The parent is honest and converged on all ten (context).
- Every call finished within the unit limit; nothing reached `depth_max` in (R) (deepest 18).
- **$\Lambda_c$ at $2\pi$:** every (C) case honest with no minimum located (err/abserr at most 0.057);
  unconverged $b = 10^8$, $\lambda \ge 16$, as at the parent (abserr/request 5.5–85). **The rule picks
  $\Lambda_c = 4$**; the check, $\lambda = 0.25$ and 1 at both $b$ with the minimum located: honest
  (err/abserr at most 0.0016) and converged, all four. **The fallback to $10^{-8}$ was not run.**

**§2.3** (`replay.json`, `replay_tables.md`; 105 units, each in its own process, none over its 900 s
limit). The replay check reproduces all 15 again: bit for bit at `3ae6f3a4` (worktree re-made,
removed after), and z00, z12 and r55 at the parent against route B. At each rung (|v − ref|/abserr
against the 30-digit reference; C converged):

| call | parent | $2\pi$ | $8\pi$ | $32\pi$ | $128\pi$ | $512\pi$ |
|---|---|---|---|---|---|---|
| **z12** | 1681 C | **0.718 C** d13 n70 | 0.00121 C | 0.762 C | 0.908 C | 0.930 C |
| z00 | 0.0183 U | 0.718 C | 0.00121 C | 0.762 C | 0.908 C | 0.930 C |
| w00–w11, honest | 12 | **12** | 12 | 12 | 12 | 12 |
| w00–w11, largest ratio | 0.942 | 0.218 | 0.141 | 0.113 | 0.224 | 0.733 |
| w00–w11, converged | 3 | **12** | 12 | 12 | 12 | 12 |
| r55 (control) | 0.0267 C | nothing located; bit-identical to the parent at every rung |||||

**At the chosen rung, $2\pi$, all 14 are honest and converged; z12 first, at 0.718.** No call reaches
`depth_max` at $2\pi$ (deepest 16).

**§2.4** (`lg_compare_07.txt`; 160 units, none over its 1200 s limit), at $\Phi_w = 2\pi$,
$\Lambda_c = 4$: **no test fails under the code** (every unit passes in both modes); of 4,320 paired
calls, **3,897 locate nothing and guard nothing, all bit-identical to the parent**; 423 locate a point
(419 roots, 4 minima; no guard, no point at an end); on them the value moves by at most 0.057 of the
parent's abserr (0.115 of the new); convergence equal under both codes on all 423 (419 converged, 4
unconverged under both: seeds 0, 4, 11, 17, `test_JJJ`, the calls that did not finish in the first
run, now finishing within the unit limit).

**§2.5** (the tables' §2.5 sections; `run2/calibrate_tables_rtol1e-09.md`, `run2/replay_tables.md`,
and the LiouvilleGreen run's per-call `section_2_5_07`):
- **At the chosen rung nothing is chased to `depth_max`:** every (R) and §2.3 call converges; the (C)
  calls at $b = 10^8$, $\lambda \ge 16$ that do not are unconverged at the parent too, nothing located,
  their error in Levin leaves on a request below the declared round-off (abserr/request 5.5–85);
  QUADPACK's `ier = 2` would have been reached on one of them ($\lambda = 16$, at bisection 51).
- **Above it**, the one unconverged (R) case ($c = 10^8$, $x_s = 0.7236$, $\omega = 0$ at $32\pi$–$512\pi$)
  holds its error in its stopped window (2.4e-14 to 8.2e-14 against a request of 1.4e-14), depth 7–18,
  no leaf at `depth_max`; `[07-cc-interior-nodes-shift-with-a-rounded-midpoint]` is the orchestrator's
  account of that floor.
- **Windows stopped while a leaf's error was still falling** (§1.3's report; a finding for the user,
  not a stop):
  - in §2.2, none at $2\pi$; 3 at $8\pi$; 10 of 10 at $32\pi$ and at $128\pi$, and at $512\pi$ (the
    tables list them), with 12–112 of their leaves falling, stopped after 18–71 bisections at depth
    1–3; every such call is honest, and all but the cancelling case converged;
  - **in §2.3 at $2\pi$: the windows of z12, z00, w00, w01, w07, w09, w10 and w11**, each stopped after
    32–78 bisections at depth 3–11 with 2 to 12 of its 11–20 leaves falling; every one of the eight
    calls is converged and honest (0.0030 to 0.718). At $8\pi$ and above every w and z call's window
    stops so.

### §2.6's report (written into the log, as §9 point 4 asks)

- **§2.2's tables:** above; both rules pick a rung at $10^{-9}$: $\Phi_w = 2\pi$, $\Lambda_c = 4$; the
  check on (C)'s cases below $L$ holds.
- **§2.3 at every rung:** above; at $2\pi$ all 14 honest, **z12 first: honest (0.718) and converged.**
- **§2.4:** no test moves; bit-identity holds on every call that locates nothing; what location does is
  listed in `out/run2/lg_compare_07.txt`.
- **§2.5:** above. **The agent's reading:** the rule ends both kinds of chase §2's first run found, at
  every rung, at the cost the user accepted: windows stop, on real calls at the chosen rung, while some
  of their leaves are still falling, and the calls stay honest and converged.
- **§1.6:** unchanged (the analytic blocks keep a per-call record; the counts join it).
- **Deviations:** 1–28 at the first stop; 29–37 on the resume.

### §3, the tests

**`AdaptiveLevin/tests/test_levin_stationary.py`** (12 tests, 3.5 s; references from
`scripts/test_references.py`, `out/run2/test_references.json`, each at two precisions, agreeing to at
most $1.5\times10^{-30}$ relative; the closed form in the incomplete gamma function checked against
mpmath.quad to $3.4\times10^{-31}$):
1. **One stationary point of P1's shape:** $\theta = A e^x - a x$ on $[0, 16]$, $A = 169$, root at the
   non-dyadic $x_s = 9.381966011250105$, variation about $1.5\times10^9$ rad. **The parent returns one
   Levin leaf across the point, floor-limited at depth 0: value $-3.1261800297467865\times10^{-7}$,
   abserr $2.542\times10^{-11}$, 1 region, `converged` False, true error $1.411\times10^{-3}$ =
   $5.55\times10^7$ times its abserr** (`out/run2/parent_cases.json`). New: located at $x_s$ exactly
   (within $4\varepsilon|x_s|$), one window, honest.
2. **Two points between adjacent nodes** of the root leaf ($\theta' = -0.25 + 10^4(x - x_0)^2$, roots
   $x_0 \mp 0.005$; no sign change on the root's 16 nodes, asserted): both located, honest.
3. **Family (C) at $\lambda = 1 = \Lambda_c/4$,** $b = 10^4$ and $10^8$: the minimum located with
   $\lambda < \Lambda_c$ and windowed; honest.
4. **No `theta_deriv`** on test 1's problem: nothing located, the guard fires, no Levin leaf returned
   guarded, honest.
5. **The exit barred:** P5 ($x_s = 10.0918$), where **the parent marks its one Levin leaf across the
   point floor-limited** (1.25e8 times its abserr out). New: no window leaf and no guarded leaf
   floor-limited (and no window stopped), honest.
6. **The cap:** test 4 with `depth_max` 4: `guard_at_depth_max` set, `converged` False, each capped leaf
   a Clenshaw–Curtis leaf whose abserr and value equal a fresh forced solve's, honest.
7. **The window's edges** in test 1: $|\theta - \theta(x_s)| = \Phi_w$ at each edge within
   $5\varepsilon|x||\theta'| + {\rm ulp}(|\theta(x)|) + {\rm ulp}(|\theta(x_s)|)$; every piece at most $2\pi$
   to the same precision.
8. **Nothing else moves:** ten problems of `test_levin_quadrature.py` without a stationary point (sin,
   cos, sinc, the lstsq and direct paths, the Gaussian bump, $\omega = 10^6$, GRZ at 10, 100, 1000),
   each with its closed-form `theta_deriv`: nothing located, nothing guarded, value, abserr and every
   returned leaf's fields bit-identical to the call with `_test_only_bypass_location`.
9. **The generic path:** a subclass of `_TwoIndependentSinCosBasis` with a quadratic second phase
   ($c = 10^5$, $x_s = 0.3819\ldots$) and the Clenshaw–Curtis and interface methods, through
   `_adaptive_levin()`: the point located on phase 2 (index 1) and windowed, phase 1 nothing, honest
   against the closed form.
10. **§1.3's rule:** (a) family (R), $c = 10^6$, $x_s = 0.7236$, `rtol` $10^{-12}$, `depth_max` 10: the
    window stops at its sixth counted bisection, its leaves no deeper than 3, all floor-limited, the
    call unconverged and honest; (b) family (R), $c = 10^6$, $x_s = 0.3819$, `rtol` $10^{-9}$, with
    $f_{\cos}$ quantised to multiples of $2^{-40}$ (a deterministic noise of about $10^{-12}$; without
    the rule 88 window leaves are returned untrusted at `depth_max` 14, their summed error
    $1.7\times10^{-14}$ against a request of $1.9\times10^{-12}$, unconverged): the window stops, the
    call is converged with no leaf untrusted, honest with the quantisation's bound $2^{-40}$ added to
    the true error; (c) on test 1's window no bisection counts.

**Test 11,** `ComputeTargets/tests/test_quadsource_levin_guard_counts.py`: one group call through
`phase_group_Levin_integral` on `test_quadsource_integral.py`'s exact-ingredient stand-in; each group
record's key set is pinned, and its four counts equal the driver's.

**Test 12, existing tests:** every test in the seven roots passes. **One existing assertion changed**,
under the amended allowance: `test_levin_trust_yardstick.py`'s `test_bit_identical_to_the_variant`
now admits the six new result keys (`GUARD_KEYS_07`) in its key-set comparisons with the variant and
the parent, asserts they are present, and compares every other key and every region field as before.
No number changed: under both codes the nine cells' every other field is bit-identical. No other
assertion of any test changed; `test_levin_global_budget.py`'s test 12c and the other tests with a
stationary point pass unchanged.

**The breakages** (`scripts/breakages.py`, `out/run2/breakages.txt`; each a text substitution in a copy
of the module, installed in a fresh process, the tree untouched; each named test passes on the tree as
a control):

| | breakage | test | result |
|---|---|---|---|
| (i) | location, windows and guard all off | 1 | **fails** (no point located) |
| (ii) | location off, the guard alone | 2 | **fails** (0 of 2 points located; the value stays honest) |
| (iii) | the floor-limited exit allowed on window and guarded leaves | 5 | **fails** (window leaves floor-limited) |
| (iv) | $\Lambda_c = 0$ | 3 | **fails** (no minimum located; the value stays honest) |
| (v) | a guarded leaf at `depth_max` returned as a Levin leaf | 6 | **fails** (no capped leaf) |
| (vi) | a plain split at $x_s$, no guard | 1 / 7 | **test 1 fails** (no window pieces); test 7 passes (it reads the location's window record, which the split leaves in place) |
| (vii) | location for the first phase only | 9 | **fails** (no point on phase 2) |
| (viii) | the four counts out of the group record | 11 | **fails** (KeyError) |
| (ix) | the counter never reaching its limit | 10 (a) | **fails** (the window not stopped) |

## Predictions and verdicts

**§2's rules, the second run** (`rtol` $10^{-9}$, on §1 and §1.3; prompt §2.1 as amended, §9 point 1;
the rules unchanged):
- **$\Phi_w$:** the smallest rung on which every case of (R) is honest and converged. — **Holds at
  $2\pi$:** 10 of 10 honest and converged (err/abserr at most 0.030). **$\Phi_w = 2\pi$.**
- **$\Lambda_c$:** the smallest $L$ such that every (C) case with $\lambda \ge L$, at both $b$, is honest
  with no minimum located; the check below $L$ honest and converged. — **$L = 4$** (every case honest);
  **the check holds** (4 of 4). **$\Lambda_c = 4$.**
- **§2.3: at the chosen $\Phi_w$, all 14 are honest; z12 first.** — **Holds:** at $2\pi$ all 14 honest
  (and converged); **z12 at 0.718**, z00 0.718, w00–w11 at most 0.218.
- **The constants as the user settled them:** by the rules, as the user's §9 point 4 directs:
  `_LEVIN_WINDOW_HALF_PHASE = 2 pi`, `_LEVIN_COALESCENCE_LAMBDA = 4`.

**§2's rules, the first run** (prompt §2.2–§2.3 as first written, `rtol` $10^{-10}$, on §1 without §1.3):
- **$\Phi_w$:** the smallest rung of $\{2\pi, 8\pi, 32\pi, 128\pi, 512\pi\}$ on which every case of
  (R) is honest and converged, those at $\omega = 1$ included. — **No rung qualifies**
  (`out/calibrate_tables.md`): honest everywhere it finished; converged 9, 9, 9, 7, 5 of 10; at
  every rung $c = 10^8$, $x_s = 0.7236$, $\omega = 0$ is unconverged (node-limited, or did not finish
  in an hour). A stop under §2.2 itself. *Ruled at the stop: re-run at rtol 1e-9 (prompt §9 points 1 and 4).*
- **$\Lambda_c$:** the smallest rung $L$ of $\{4, 16, 64, 256, 1024\}$ such that every case of (C)
  with $\lambda \ge L$, at both $b$, is honest (run with no minimum located); the check, the cases
  with $\lambda < L$ re-run with their minimum located, honest and converged. — **Not run under the
  rule** (no $\Phi_w$). Supplementary, at every rung: $L = 4$, the check holding (20 of 20).
  *Ruled at the stop: re-run at rtol 1e-9 (prompt §9 points 1 and 4).*
- **§2.3, at the chosen $\Phi_w$, all 14 are honest; z12 first.** — **Undecided: no rung was
  chosen.** At every rung z12 and z00 do not finish within 900 s (honest at bounded depth 12, or 8
  at $512\pi$); V1's twelve are honest at every rung where they finish (59 of 60).
- **The constants:** settled by the second run (above).

**§5's rules S0–S2, copied from the prompt unchanged.** Scored in Part B on the Mac Studio on
2026-11-08; each verdict follows its rule, and the Part B record at the end of this log has the
detail. Under §10, R29 (at `e182bc7a`, Rule A's code) takes R28's place:

3. **S0 — R27 against 06m's R26.** R26 (`var/datastores/levin-phase-floor-06m-r26*`, kept on the
   Mac Studio) was built at 06m's code, the parent's. `tools.store_value_diff` on R26 and R27: every
   value table identical, every `QuadSourceIntegral` value field identical, and every other
   difference of the four volatile kinds. **Holds** if so. A difference is a finding for the user,
   reported before R28 is scored.
   — **Holds.** R27 equals R26 in all seven value tables (360,462 rows) and in all 864
   `QuadSourceIntegral` rows, every value field (118,776) and counter field (27,163) bit-identical;
   the labels differ only by the job-name prefix and the stamp (normalised in a copy).
4. **R28 against R27.** `tools.store_value_diff` with `--json`, with 05's `classify_non_qsi.py`.
   - **Outside `QuadSourceIntegral`:** every value table identical, every other difference of the
     four volatile kinds, no row in one store only.
   - **Inside it, each row is classed** by R28's four counts in every Levin record of the row,
     group and analytic (or as §2's stop settled for the analytic blocks):
     - **a row in which nothing was located or guarded is bit-identical** in `total`,
       `total_abserr`, the analytic columns and every Levin record, with the four counts set
       aside;
     - **any other row is tabulated:** the change against R27's and R28's `total_abserr`, the
       located points (kind, $z$), window pieces, regions, maximum depth and `total_converged` in
       each.
   - **The stationary-point rows'** convergence in each store, and the rows whose `total_converged`
     changes, in each direction.

   — **Holds, and tabulated.** R28 against R27: outside `QuadSourceIntegral` 360,462 rows identical,
   0 unclassified; 640 untouched rows, all bit-identical; 224 tabulated, with the kind and $z$ of
   every located point from the replay (470 of 470 records bit for bit). **R29 against R27** (§10):
   692 untouched rows, all bit-identical; 172 tabulated; the largest move $7.38\times10^4$ of R27's
   `total_abserr`, at $(10^6)^3$, $z = 2284.25$; `total_converged` 845 → 862, 17 rows gained and none
   lost. R29 against R28 (the note's §B.7 rule) fails as written on the 4 guarded analytic blocks,
   which the user ruled findings; the rule was the orchestrator's error (the note's §B.8).
5. **S1 — V1's twelve.** w00–w11 are R24's rows, so R27's and R28's too. Each w call, the `G+q+r`
   group of its row's sub-interval 1, is read from R28's group record. **Rule:** each locates its
   point at $z \approx 10211$, and $|v_{28} - {\rm ref}| \le {\rm abserr}_{28}$ against 06g's
   reference (`scores.json`, `V1`). **Holds** if all twelve do. A violation is a finding for the
   user, not by itself a stop.
   — **Holds**, on R28 and on R29: each of the twelve locates one root at
   $z = 10211.3179 \pm 9\times10^{-6}$, and each is honest, the worst 0.218 (w11); R29's twelve
   equal R28's bit for bit.
6. **S2 — P1's two rows.** P1's row 1, $(3.162\times10^5)^3$ at $z_{\rm response} = 250$, and row 2,
   $(10^5, 3.162\times10^5, 3.162\times10^5)$, each hold a point of `G+q+r`, at $z_* = 10211.3$ and
   $275.0$.
   - **The rule, location:** each locates its point within one scan spacing.
   - **The rule, honesty:** each `G+q+r` call is within its abserr of a reference. P1's references
     were made at 06's tree, whose integrand differs from today's by more than these errors. So the
     reference is taken afresh: the call is captured at R27's tree, as 06g's `v1/` captured R24's,
     and referenced by `tools/filon_reference.py` with four readings. A call that is one of
     w00–w11 is scored by S1 instead.
   - **Holds** if both hold on both rows. A violation is a finding, not by itself a stop.

   — **Holds**, on R28 and on R29: rows 1 and 2 locate their points within $7.1\times10^{-9}$ and
   $5.6\times10^{-9}$ in $x$ (spacing 0.0041), and are 0.020 and 0.263 of their abserr from four
   agreeing readings (`levin-phase-floor-07-p1-references-20261108T103132`); R27 is $5.6\times10^5$
   and $4.1\times10^4$ out, P1's failure reproduced at the parent.

**§6's Gate V0 and predictions V1–V5, copied from the prompt unchanged.** Scored in Part B on the
Mac Studio, 2026-11-08 to 2026-11-10, with V3 as §10 sets it; each verdict follows its rule:

**Gate V0, before the run:**
- **The parent's code is 06m's.** `git diff --stat 48cf87c0 <code commit's parent> -- . ':!docs'
  ':!prompts'` shows nothing the harness imports.
- **The harness, as changed, reproduces 06m's census at the parent,** bit for bit by 06m's
  comparison, on 06k's 100 K0 units (`docs/levin-phase-floor/06l/out/k0/units.json`). It is one
  registered run with slug `k0`.
- **The new code is deterministic.** Two runs of one $g$ unit on a stationary-point row are
  bit-identical.

— **V0 holds**, all three clauses: `git diff --stat 48cf87c0 91472dbe` touches a test module and
`CLAUDE.md`, none of the 102 files the K0 workers loaded; `levin-phase-floor-07-k0-20261108T135240`
reproduces 06m's census on 100 of 100 units (2,110 calls, 32,518 leaves, 0 differing);
`g-1e-8-1-2174` run twice at $g$ is identical.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into
`state/predictions.md` unchanged.
- **V1 — nothing else moves.**
  - **The rule:** every Levin call, group or analytic, in which $g$ locates nothing and guards
    nothing equals 06m's call bit for bit, in its record and its region record, by 06m's
    comparison with the new keys set aside. Every row all of whose calls are such equals 06m's row
    in `total` and `total_abserr`.
  - **Holds** if true at both rtols.
  - — **Holds** at both rtols: 284,514 calls that locate and guard nothing (142,257 per rtol) equal
    06m's bit for bit, and so do the 16,104 units made only of such calls, `total` and
    `total_abserr` equal in hex.
- **V2 — no Levin leaf is returned across a stationary point.**
  - **The rule:** the scan finds no returned **Levin** leaf holding a sign change of $\theta'$, in
    any group call at either rtol or any analytic call at `1e-7`.
  - **Holds** if none does. At 06a's code there were 677 such regions, at `1e-7` alone.
  - — **Holds:** 182,066 calls scanned (37,325 group calls at each rtol, 107,416 analytic calls at
    `1e-7`), and no returned Levin leaf holds a sign change. The leaves the scan lists are window
    leaves and 168 ordinary Clenshaw–Curtis leaves of analytic calls whose phase Rule A skipped.
- **V3 — the location finds what the scan finds.**
  - It runs on the scan's grid, so the two should agree exactly.
  - **The rule:** every call in which the scan finds a sign change has a located `root` within one
    scan spacing of each, and every located `root` lies within one scan spacing of a sign change
    the scan found.
  - — **Holds:** 0 violations in 170,582 scored calls, 2,954 of them with a sign change; no point
    of kind `end`. Set aside under §10: 11,484 scanned calls with a phase Rule A skipped, 168 of
    them (all analytic) with a scan sign change, 14 of those guarded (O9's).
- **V4 — honest where it is referenced.**
  - **The set:**
    - z00 and z12, against 06g's reference (`rows`);
    - Z3's seven analytic calls in the 720 (z07, z16, z18, z21, z24, z28, z29), against 06g's;
    - 24 more stationary-point group calls, stratified by band (low, mid), by $k$ (all six that
      carry them), by rtol and by 06m's depth, with the deepest four included. The selection rule
      is fixed in `state/predictions.md` before the run.
  - **The 24's references** are `tools/filon_reference.py`'s, four readings (QAWO and FCC at two
    resolutions), FCC at resolution 2 the reference, from captures at $g$'s tree, as 06g's were
    made. Each capture is checked bit for bit against the census record. They run as one registered
    run with slug `references`, after the census, at most six workers. The 30-minute reading limit
    is a check, not a cutoff (06g's ruling). A projection beyond a day is a stop.
  - **The rule:** $|{\rm value}_g - {\rm ref}| \le$ `abserr`$_g$ for every call in the set whose
    four readings agree within 06g's tolerance, here $\min({\rm abserr}_{06m}, {\rm abserr}_g)/100$.
    A call whose readings do not agree is reported, not scored.
  - **Holds** if every scored call is honest.
  - — **Fails.** 33 calls scored, every call's readings agreeing (the widest spread 0.0039 of the
    tolerance); 28 honest. z00 and z12 are at 0.718 of their abserr and Z3's seven at most 0.0021,
    on 06g's readings; 19 of the 24 are honest. **Not honest:** v08 3.16, v09 2.21, v12 3.08, v17 2.38
    and v22 2.53 times $\text{abserr}_g$ (06m: 3.0e4, 639, 9.1e3, 497 and 1.67e4). Each is a converged
    `G+q+r` call whose one window was stopped by §1.3's `iroff1` at 6 with 4–13 leaves still falling
    (`max_depth_at_stop` 5–6): `[07-iroff1-window-stop-can-leave-a-call-dishonest]`. The 24's basis,
    on the user's ruling at stop 4: four readings on 13, FCC at both resolutions on 11. 06m is
    honest on 17 of the 33 and $356$–$5.2\times10^5$× out on the other 16 (z12 1,681×).
- **V5 — the cap is not reached where a point was handled.**
  - **The rule:** at both rtols, no leaf reaches `depth_max` in a call that locates a point or
    guards a leaf. An over-budget unit counts as false.
  - A failure is a finding for the user, not a fault. It reopens G2's L2 (Decisions, **G2 L2,
    decided**).
  - — **Fails, as written.** 23 `G+q+r` calls that locate their root (10 at `1e-7`, 13 at `1e-8`)
    hold 2–54 leaves at `depth_max` 20; 3 are converged; no unit is over budget. **The rule's
    premise was stale.** V5 was written on 2026-10-27 from 06a's census, where no Levin call
    outside the stationary points reached the cap. 06g–06m's trust-test changes left calls at the
    cap outside the 720 (06m's census: 114 group calls unconverged there, 105 of them holding an
    untrusted leaf at `depth_max`), and the orchestrator did not re-check the premise at Part B's
    dispatch: the orchestrator's error. Inside the 720, 62 of the 64 unconverged group calls carry
    the signature of those outside (an untrusted leaf at the cap, the call's error within its
    request), so V5's failure says nothing specific about stationary points. The user: *"I don't
    see what value `V5` has if there are other regions that go down to the depth cap."* G2's L2 is
    reopened as one question for 08: whether any call should reach the cap, with the depth-cap
    experiment (Decisions, **G2 L2, reopened**).

## Deviations from the prompt

1. **The two interface methods return callables.** `independent_phases()` and
   `independent_phase_rates()` each return a list of callables, one per independent phase (the
   rates `None` when the basis cannot supply them), rather than values at $x$: the scan must go
   through `_sample_vectorized()`, which takes a callable. The names and signatures are left to the
   agent (§1.0). As the set-aside work had them (its deviation 1). `IMPLEMENTATION CHOICE`.
2. **A window's pieces are cut on the phase itself, not on the scan, and every phase's cuts are
   united.** §1.2 item 3 says the span is "estimated on the scan, taking the largest over the
   basis's phases". A window can be narrower than a few scan spacings (at $c = 10^8$, $\Phi_w = 2\pi$
   its half-width is $\sqrt{2\Phi_w/c} = 3.5\times10^{-4}$ against a spacing of $2.4\times10^{-4}$),
   so a scan estimate cannot place cuts that keep every piece within $2\pi$ (§3 test 7). Each phase's
   variation is measured exactly on its monotone segments (the window split at that phase's
   located roots), each segment cut into $\lceil|\Delta\theta|/2\pi\rceil$ pieces of equal phase by
   `brentq` on $\theta - $level, and the cuts of every phase united. The set-aside work's deviation 2.
   `STRUCTURALLY REQUIRED`.
3. **A root is a cut.** The set-aside work kept a root a third of the way through its central piece
   (its deviation 3), because a node on a zero of $\theta'$ made the eq. (151) floor the window pieces
   then carried infinite. Since 06d a Clenshaw–Curtis solve carries `_cc_roundoff_floor()`, with no
   $\theta'$ in a denominator, so that reason has gone, and a root is simply a cut between two
   monotone segments. `IMPLEMENTATION CHOICE`.
4. **The forced Clenshaw–Curtis path keeps the span's orientation.** §1.2 item 4's path is a keyword
   `force_direct` of `_adaptive_levin_subregion()` and `_adaptive_levin_subregion_impl()`, which
   skips the C2 gate and calls the new `_oriented_cc()`. `_adaptive_levin_subregion_cc()` weights by
   $|b - a|$, so on a reversed span its value has the wrong sign (observation 1); `_oriented_cc()`
   solves the ascending span, with $\theta'$ sampled afresh on its nodes, and negates the value and
   the signed node error. The existing fallback is not changed. `STRUCTURALLY REQUIRED` (a forced
   path that inherited the defect would return wrong window values on a reversed span).
5. **The guard's mechanism.** A Levin solve reports `guarded` when a phase rate is not of one
   *strict* sign on its own nodes: a zero on a node, or a non-finite sample, counts. A guarded leaf
   below `depth_max` has its error read as $+\infty$ (`_levin_leaf.error`), the mechanism §1.4
   offers: it is chosen before every finite leaf, the budget cannot be met while it is in the
   partition (so the trust check never runs on it, and it is never trusted), and it is barred from
   the floor-limited exit explicitly. "Guarded" is a property of a Levin leaf only: the
   Clenshaw–Curtis halves the C2 gate makes of it are ordinary fallback leaves. `IMPLEMENTATION CHOICE`.
6. **A guarded leaf at `depth_max`** is re-solved by the forced path when it is made, flagged
   `guarded` and `guard_at_depth_max`, marked checked so that the trust check is not taken on it
   (its errors are that solve's, unwidened, and it is never trusted), and the call's `converged` is
   False while one is returned. `IMPLEMENTATION CHOICE`.
7. **`num_guarded_regions` counts every leaf found guarded when made,** bisected or returned, so that
   it says whether the guard fired at all; a returned leaf's own flag says whether it was returned.
   `IMPLEMENTATION CHOICE`.
8. **Results beyond §1.5.** Each point also carries `found_as`, `theta`, `own_window` and
   `own_window_edges` ($\theta'$, $\theta''$ by a central difference, $\theta'^2/|\theta''|$, and
   $\theta - \theta(x_*)$ at each edge of its own window); the result also carries
   `stationary_windows` (each merged window's edges and piece cuts, from which a leaf's ancestry
   can be rebuilt) and `unlocated_minima` (each minimum found whose $\lambda$ was not below
   $\Lambda_c$). For §2's tables and §6's descriptive measures. `IMPLEMENTATION CHOICE`.
9. **The guard for a basis with several phases** samples each phase's closed-form rate on a Levin
   leaf's nodes, through a new keyword `guard_rates` of the two subregion functions; for a basis
   with one phase `theta_prime_Cheb` is its rate, at no extra evaluation. `IMPLEMENTATION CHOICE`.
10. **Merging is within a phase.** Two phases' points that coincide are kept as one point each; their
    windows overlap and merge. `IMPLEMENTATION CHOICE`.
11. **Precisions.** $\varepsilon = 2^{-52}$. Root refinement: `brentq` with
    xtol $= \varepsilon\max(|x_a|,|x_b|)$, finer than §1.1's "no coarser than $4\varepsilon\max$",
    and rtol $= 4\varepsilon$ (scipy's least). Minimum refinement: bounded Brent with
    xatol $= 4\varepsilon\max|x|$, which scipy floors at about $\sqrt\varepsilon|x|$; a refined
    value above the grid's is replaced by the grid point. $\theta_i'''$: the central second
    difference of the rate with step $s = h/2$, $h$ the scan spacing, its centre moved inward to keep
    the stencil in `x_span`; truncation error $s^2|\theta^{(5)}|/12$ (zero for family (C)), rounding
    about $4\varepsilon\max|\theta'|/s^2$. Window edges: `brentq` on
    $|\theta - \theta(x_*)| - \Phi_w$, bracketed by the first scan point outward at which the bound is
    reached; at a returned edge the residual is at most about
    $5\varepsilon|x||\theta'| + {\rm ulp}(|\theta(x)|) + {\rm ulp}(|\theta(x_*)|)$, of order
    $10^{-8}$ rad for a raw phase of $10^8$ (§1.2 item 1's "the log states the size").
    `IMPLEMENTATION CHOICE`.
12. **Non-finite scan samples are skipped,** as 06a's scan skipped them. `IMPLEMENTATION CHOICE`.
13. **`scipy.optimize` is imported inside `_locate_stationary_points()`,** not at module scope, for the
    module's import cost (its LOGGING note). `IMPLEMENTATION CHOICE`.
14. **Provisional constants.** `_LEVIN_WINDOW_HALF_PHASE` $= 32\pi$ and `_LEVIN_COALESCENCE_LAMBDA`
    $= 64$ stand in the module, each comment saying so, until the user settles them. Every §2.2 run
    sets its rung explicitly; §2.3's replays set $\Phi_w$ to each rung and leave $\Lambda_c$ at 64;
    §2.4's draws run at both provisional values. `IMPLEMENTATION CHOICE`.
15. **Two §8 conditions met during §1, reported at §2's planned stop rather than as stops of their
    own.** (a) `test_levin_trust_yardstick.py`'s `TestCodeIsTheVariants.test_bit_identical_to_the_variant`
    (06m's test 2) asserts that the result's key set equals 06l's variant's; §1.5's new keys make
    it fail on all nine cells (and its firing count, which the failed subtests no longer reach).
    Every other field is bit-identical to the variant and, where the variant does not fire, to the
    parent (a scratch check of the nine cells with the six new keys set aside). The test holds no
    stationary point, so §3 test 12's allowance does not cover changing it. (b) Three
    ComputeTargets tests do not finish (§2.5 (A)). Neither changes any number §2 measures, and §2
    writes no test, so §2 was run first and both are put to the user here.
    `IMPLEMENTATION CHOICE`.
16. **How §2 was run.** Each call in its own process with a wall-clock limit (one hour in §2.2,
    900 s in §2.3, 1200 s in §2.4); a call over it is recorded as "did not finish" and counts as
    neither honest nor converged, and (§2.2, §2.3) is re-run at bounded `depth_max` for §2.5. Added
    for §2.5: the parent's code on every case, for context; the "bar lifted" variant on every call
    that left window leaves at `depth_max` or did not finish; and three diagnostic records of the
    calls that do not finish (`ct_stuck_call.py`, `ct_stuck_tests.py`, `lg_stuck_units.py`) and one
    of (B) (`z12_untrusted_leaves.py`). Since no $\Phi_w$ rung qualified, (C)'s procedure was run at
    every rung, labelled supplementary (as the set-aside work's second stop did). (C) with
    $\Lambda_c = 0$ was run at every rung in one stage, though nothing is located on it, so the five
    copies are equal. `IMPLEMENTATION CHOICE`.
17. **§2.3's first run was stopped** after 20 minutes, with a one-hour limit per call and three z12
    calls each past 18 minutes, and re-run with a 900 s limit and bounded depths 8 and 12 (the
    first had 6, 8, 10, 12). Its two finished units are kept as `out/replay_run1_stopped_*`.
    `IMPLEMENTATION CHOICE`.
18. **`replay.py`'s parent check was corrected after its run.** As run it formed
    $|v - {\rm ref}|/{\rm abserr}$ from the double nearest the reference (`ref_hex`), which differs
    from route B's ratios in `h1_calls.csv` in the tenth digit; 06l formed it from the 30-digit
    reading (`ref_30`) in `mpmath`. `replay.py tables` now recomputes the check, and every ratio in
    the tables, from the stored records in 06l's arithmetic; no call is re-run; the run's own check
    is kept in `replay.json` as `replay_check_as_run` (False, on that one key) and the recomputation
    has its own header (`recheck_header`, the script's new SHA-256). The values compared are the
    run's. `UNINTENDED DRIFT` (a defect of my script, corrected).
19. **The captures need 06g's capture harness on `sys.path`.** Each pickle names the harness's
    lru_cache reducer (`harness._rebuild_lru_cache`), so `var/levin-phase-floor-06g-fcc-laptop/harness/`
    is put after the tree on `sys.path`, read only, as 06g's laptop FCC batch had it.
    `STRUCTURALLY REQUIRED`.
20. **The w calls' 06g regions and depth** come from `references/v1/out/v1_capture.json` ("g", R24's
    capture), since `scores.json`'s `V1` records value, abserr and `converged` only.
    `STRUCTURALLY REQUIRED`.
21. **The parent replays** load `git show bd21bf23:AdaptiveLevin/levin_quadrature.py` into a
    temporary file in place of the module, by 06m's `variant_loader.py`, not a second worktree (the
    dispatch's choice). `IMPLEMENTATION CHOICE`.
22. **06m's LiouvilleGreen harness, changed where marked "prompt 07"** (`out/scripts_diff.txt`): modes
    `parent` and `commit`; the guard's keys and a result digest without them in each call record;
    §2.5's account under `commit`; each unit a subprocess with a 1200 s limit, in place of 06m's
    pool; and, after the run, its `report` changed to leave out the five unfinished (seed, test)
    pairs (it raised on their records), so the report's header carries the script's later SHA-256
    and the run file its earlier one. 06m's copies are not `black`-clean; 07's are, so the recorded
    diff is against 06m's after `black`. `IMPLEMENTATION CHOICE`.
23. **§2.3 and §2.4 ran at the provisional $\Lambda_c = 64$, and §2.4 at the provisional
    $\Phi_w = 32\pi$** (deviation 14): §2 fixes no rung for them before §2.2's rule picks one, and it
    picked none. `IMPLEMENTATION CHOICE`.
24. **The long runs are not registered.** They ran in the background of this session, unregistered,
    because this dispatch forbids writing under `var/` (where `runregistrykit` records), as the
    set-aside work's did. `STRUCTURALLY REQUIRED`.
25. **Exploratory runs before the recorded ones,** on scratch copies of the scripts and in the
    session scratchpad, not kept as records: (R) at each rung, the stuck ComputeTargets call at
    bounded depth, z12 at bounded depth, sanity checks of the guard, the reversed span and the
    switch. `IMPLEMENTATION CHOICE`.
26. **`ct_stuck_tests.py` was run twice:** first on the two tests then known, then on three, after the
    root's re-run found the third; the first run's outputs are kept as
    `out/ct_stuck_tests_run1_two_tests*`. `IMPLEMENTATION CHOICE`.
27. **The tarball's AppleDouble entries** (`._` names, 74 of them) are listed and skipped by
    `check_earlier_attempts.py`; they are extended attributes `tar` stored beside each entry, not
    records, and extraction makes no file of them. `IMPLEMENTATION CHOICE`.
28. **The ComputeTargets root at the stop** was run three ways: as the root (stopped by me after 31
    minutes inside `test_short_range`), and twice by a scratch runner that skips the tests that do
    not finish, with each test's wall time and a 600 s limit per test (Verification). The roots
    after §3 are *pending*. `IMPLEMENTATION CHOICE`.

29. **The second run's records sit beside the first's.** The first run's outputs were moved unchanged
    from `out/` to `out/run1_rtol1e-10/` (their headers keep their scripts' SHA-256 of the time), and
    `prov.py`'s `OUT` became `out/run2/`. For the second run `calibrate.py` gained `--rtol` (outputs
    named by it) and records each window's stop; `analysis07.py` records the windows' iroff1 fields;
    `replay.py` and the LiouvilleGreen harness are otherwise as in the first run. The second run
    re-ran §2.1's validation too. `IMPLEMENTATION CHOICE`.
30. **§1.3's report is carried in `stationary_windows`,** not in a new result key (`iroff1_count`,
    `stopped`, `stopped_after_split`, `leaves_at_stop`, `leaves_falling_at_stop`,
    `max_depth_at_stop`), so that §3 test 12's key-set allowance stays the six keys of §1.5.
    "Falling" is read per bisection: a leaf whose making bisection lowered the summed error,
    $e'_L + e'_R < 0.99\,e'_P$ (`fell_at_last_bisection`). `IMPLEMENTATION CHOICE`.
31. **What a stopped window's leaf declares.** §1.3: "its $e'$, except that a leaf already found
    untrusted declares as an untrusted leaf at `depth_max` does". A leaf the trust check had trusted
    keeps its widened declaration $\max(\epsilon_P, \Delta_1)/(1 - r)$ (its $e'$ as the budget summed it)
    and, being floor-limited, returns its own value $V_P$, as `_exempt_floor_limited()` does for a
    floor-limited leaf; an unchecked leaf declares its raw $e'$. `IMPLEMENTATION CHOICE`.
32. **The constants were set by the rules without a further stop,** as §9 point 4 directs; their
    comments cite `out/run2/calibrate_tables_rtol1e-09.md`. `IMPLEMENTATION CHOICE` (as directed).
33. **The tests' references are held as float hex in the test module,** computed and checked by a
    recorded script (`test_references.py`), so that the tests run in seconds; the parent's numbers for
    tests 1 and 5 come from a recorded script too (`parent_cases.py`), and the tests do not run the
    parent. Test 1's and 5's problem is not one of §2's families: (R) at the parent is honest and
    converged on every case (O3), so a P1-shaped family with a closed form was found
    ($\theta = A e^x - a x$, the incomplete gamma function), by a scratch search, not kept, over 300
    random members, of which 25 showed the parent's failure. `STRUCTURALLY REQUIRED`.
34. **Test 10 (b)'s noise is a quantised amplitude,** $f_{\cos}$ rounded down to multiples of $2^{-40}$,
    a closed-form-referenced stand-in for z12's integrand noise; its honesty is asserted with the
    quantisation's bound added to the true error. Test 10 (a) runs at `depth_max` 10, so that
    breakage (ix) fails in minutes rather than chasing to depth 20. `IMPLEMENTATION CHOICE`.
35. **Test 9's basis hands the fallback's node term one $\theta'$ per node,** the larger $|\theta_i'|$:
    `_cc_node_term_parts()` takes one phase rate per node, which a two-phase basis cannot supply
    exactly (O7, opened as an issue). `STRUCTURALLY REQUIRED`.
36. **Test 11 is a new module,** `ComputeTargets/tests/test_quadsource_levin_guard_counts.py`, reusing
    `test_quadsource_integral.py`'s stand-in; and test 7's piece tolerance doubles the edge precision
    (two refined ends per piece). `IMPLEMENTATION CHOICE`.
37. **Breakage (vi) is caught by test 1, not test 7.** Test 7 reads the window's record from the
    location, which the plain split leaves in place, so it passes; the prompt asks test 1 or test 7.
    `IMPLEMENTATION CHOICE`.

## Verification performed

### On the resume (2026-11-07, at `91472dbe`, the final module)

- **The seven roots on the final module before §3's tests were written** (22:07–22:17, `uptime`
  load 4.2 / 7.1 / 8.3 at the start, 3.8 / 4.5 / 6.2 at the end): AdaptiveLevin 91 run, 10 failures,
  all `test_levin_trust_yardstick.py`'s key-set assertion (test 12's change not yet made);
  **ComputeTargets 780 OK in 412 s** (the four tests that did not finish on §1's code all pass:
  §1.3 ends mechanism (A)); CosmologyModels 39 OK; LiouvilleGreen 156 OK (skipped=1); ObjectFactories
  280 OK; RunDrivers 127 OK; tools 64 OK.
- **The seven roots on the final tree** (the commit's tree; 22:17–22:28, `uptime` load
  3.4 / 4.4 / 6.1 at the start, 6.3 / 7.7 / 7.4 at the end): **AdaptiveLevin 103 OK** (91 + `test_levin_stationary`'s 12); **ComputeTargets 781 OK** in 425 s
  (780 + test 11); CosmologyModels 39 OK; LiouvilleGreen 156 OK (skipped=1); ObjectFactories 280 OK;
  RunDrivers 127 OK; tools 64 OK. The rises equal the tests added; the other five equal the
  dispatch SHA's. The wall-clock tests (`test_gk_wkb_phase`, `test_tk_wkb_phase`) passed.
- **§2's second run** (`out/run2/`): each output's header records its script's SHA-256, the module's
  (`dc05596c…`), `91472dbe` and a dirty tree. §2.1's validation re-run passes; the replay check
  reproduces the 15 captures at 06g's code and z00, z12, r55 at the parent.
- **§3's tests:** `test_levin_stationary.py` 12 tests in 3.5 s; test 11 passes; the nine breakages
  each caught by their named test, every named test passing on the unbroken tree as a control
  (`out/run2/breakages.txt`). The references for the tests by `test_references.py`
  (`out/run2/test_references.json`): each value's 40- and 60-digit readings agree to
  $1.5\times10^{-30}$ or better (the two-roots quadrature; the closed forms to $10^{-36}$), the exp
  family's closed form agrees with `mpmath.quad` on its small member to $3\times10^{-31}$, and (R)'s
  closed forms agree with their contours to $10^{-38}$.
- **`black --check`** on the changed modules, the two test modules and `calibration/scripts/`: clean.
- `git log --oneline 91472dbe..HEAD` empty before staging.

### At the first stop (kept as written)


- **The seven roots at the dispatch SHA** (`bd21bf23`, before any edit; run serially, 08:25–08:34,
  `uptime` 18.66 / 18.19 / 21.43 at the start and 12.02 / 20.70 / 22.64 at the end):
  AdaptiveLevin 91 OK; ComputeTargets 780 OK (291 s); CosmologyModels 39 OK; LiouvilleGreen 156 OK
  (skipped=1); ObjectFactories 280 OK; RunDrivers 127 OK; tools 64 OK. They equal the
  orchestrator's.
- **The seven roots on §1's code** (08:36–09:12, load 12–29):
  - AdaptiveLevin: 91 run, **10 failures, all in `test_levin_trust_yardstick.py`'s
    `test_bit_identical_to_the_variant`** (nine cells and the firing count; deviation 15).
  - ComputeTargets: **did not finish;** stopped after 31 minutes inside
    `test_quadsource_integral.TestDirectQuadOracle.test_short_range`. Re-run by a scratch runner
    skipping the tests that do not finish, each test timed and limited to 600 s (12:09–12:25,
    load 4.6–6.4): 777 run, 776 pass, **one over 600 s**
    (`TestErrorBound.test_abserr_bounds_the_quadrature_error`); skipped as not finishing:
    `test_short_range`, `test_domenech_oracle.TestNormalisation.test_N_is_constant_over_the_nine_b_02_fixture_cases`
    (stopped after 18 minutes in an earlier pass) and
    `test_kohri_terada_oracle.TestPipeline.test_total_is_minus_nine_eighths_of_I_RD` (over 600 s
    in a second). So **four ComputeTargets tests do not finish** and 776 pass. The two wall-clock
    tests (`test_gk_wkb_phase`, `test_tk_wkb_phase` `test_wall_time_per_object`) passed.
  - CosmologyModels 39 OK; LiouvilleGreen 156 OK (skipped=1; one unseeded draw); ObjectFactories
    280 OK; RunDrivers 127 OK; tools 64 OK.
- **Bit-identity where nothing is located** (§0.4): 3,803 of 3,803 LiouvilleGreen calls (§2.4);
  `test_levin_trust_yardstick.py`'s nine cells, every field but the new keys, to the variant and
  (where it does not fire) to the parent (scratch check); r55 at every rung (§2.3).
- **Scratch sanity checks** (not records): on (R) $c = 10^6$, $x_s = 0.382$ at $8\pi$ the reversed span
  gives the negated value to $2\times10^{-19}$; without `theta_deriv` the guard fires on 8 leaves and
  the value is honest; with `depth_max` 4 one guarded leaf is re-solved at the cap,
  `guard_at_depth_max`, `converged` False, honest; with the location bypassed and `theta_deriv`
  supplied the guard alone handles the point, honest.
- **The earlier attempts' records:** 61 files copied, each equal to the tarball's member by
  SHA-256 (`earlier-attempts/SHA256_CHECK.txt`, by `check_earlier_attempts.py`); the tarball's
  SHA-256 is the one `SHA256SUMS.txt` lists.
- **`black --check`** on the two changed modules and on `calibration/scripts/`: clean.
- `git log --oneline bd21bf23..HEAD` empty at the start and at the stop; the scratch worktree at
  `3ae6f3a4` clean throughout (`git status --short` empty, recorded in `replay.json`).

## Observations not acted on

- **O1. The Clenshaw–Curtis fallback ignores a reversed span** (the set-aside work's O1, still true
  at the parent): `_adaptive_levin_subregion_cc()` weights by $|b - a|$, so with
  `x_span = (1.0, 0.0)` a fallback region returns the value of `(0.0, 1.0)`; checked at the parent
  on $f = (e^{-x}, \cos 3x)$, $\theta = 2x$: both give 0.7191774453594725. No production `x_span`
  is reversed. 07's forced path avoids it (deviation 4). **Opened** on the board's §3 as
  `[07-cc-fallback-ignores-a-reversed-span]`.
- **O2. The trust check reads a QSI integrand's own small-scale structure as non-convergence on
  Clenshaw–Curtis leaves** (§2.5 (B)): window leaves whose values are $10^{-32}$ carry nested-pair
  differences of $10^{-43}$–$10^{-42}$, $\Delta_1$ above $\nu_B$, $r$ 0.26–3.9
  (`out/run1_rtol1e-10/z12_untrusted_leaves.txt`). §1.3 now stops such a window; outside a window
  the behaviour is the parent's. It bears on 08's flag question.
- **O3. Family (R) does not show P1's failure at the parent:** the parent is honest and converged
  on all ten (R) cases. Tests 1 and 5 therefore use the exp-phase family (deviation 33), on which
  the parent fails as P1 did (`out/run2/parent_cases.json`).
- **O4 (withdrawn).** At the first stop this said `CLAUDE.md` does not list 06m's calibration
  scripts, which are not clean under `black --check`. It does list them (06m's `calibration/scripts/`,
  present at `bd21bf23`); the first stop misread the list. Nothing to act on.
- **O5.** Location costs about 4,097 scalar evaluations of `theta_deriv` per call whose rate does
  not vectorise (every QSI group); the ComputeTargets root took 412 s and 425 s on the final module
  against 291 s at the dispatch SHA (at different loads). Cost is not an argument (README §5 rule 4).
- **O6.** Another session ran `tools/compare_stores.py` on this machine during §2's first run; it
  touched none of 07's files.
- **O7. The fallback's node term takes one $\theta'$ per node** (`_cc_node_term_parts()`, through
  `theta_prime_on_fine_grid()`), so a basis of more than one phase cannot hand it its node term
  exactly; test 9's basis passes the larger $|\theta_i'|$ (deviation 35). No production basis has
  more than one phase (`_Basis_SinCos` alone). **Opened** on the board's §3 as
  `[07-cc-node-term-takes-one-phase-per-node]`.
- **O8. Windows stopped while leaves were still falling** (the finding of §2.6, not a stop under §9
  point 4): at $2\pi$ eight of §2.3's fourteen calls (z00, z12, w00, w01, w07, w09, w10, w11) have a
  window stopped with 2–12 of its leaves' errors having fallen at their last bisection; all eight
  are honest and converged. Part B's census measures whether this holds over the wider set; not
  opened as an issue, since nothing measured is wrong.

## State handed on

At the end of Part A:
- **Committed** (one commit, not pushed): the module (SHA-256 `dc05596c…`),
  `ComputeTargets/QuadSourceIntegral.py` (`33cb161e…`), the two new test modules, the changed
  yardstick test, `docs/levin-phase-floor/07/calibration/` (scripts, `out/run1_rtol1e-10/`,
  `out/run2/`) and `earlier-attempts/`, this log, the board, `docs/OPEN_ISSUES.md` (two issues
  opened, 128 → 130) and `CLAUDE.md` (the two directories added to the `black` list).
- **The constants:** $\Phi_w = 2\pi$, $\Lambda_c = 4$, from the second run at `rtol` $10^{-9}$; the
  iroff1 constants are §1.3's ($10^{-5}$, 0.99, 6).
- **Part B is pending** (R27, R28 and the census, on the Mac Studio). The census should read O8's
  windows over the wider set.
- **Not staged:** `docs/levin-phase-floor/07/investigations/` (the orchestrator's), `__pycache__`,
  and the session scratchpad.
- **The set-aside work** is as it was: four stashes, `var/levin-phase-floor-07/wip/` unchanged.
- The header's issue `[06-levin-accepts-a-region-across-a-stationary-point]` does not move.

At §2's first stop (kept as written):

At §2's stop:
- **Uncommitted in the main checkout:** `AdaptiveLevin/levin_quadrature.py` (SHA-256 `13294b2a…`),
  `ComputeTargets/QuadSourceIntegral.py` (`33cb161e…`), and `docs/levin-phase-floor/07/`
  (`calibration/scripts/`, `calibration/out/`, `earlier-attempts/`). This log, written at the stop.
  Nothing staged.
- **The constants** in the module are the provisional $32\pi$ and 64.
- **§1.3 is not built;** no test of §3 is written; the board, `CLAUDE.md` and
  `docs/OPEN_ISSUES.md` are untouched.
- **The scratch worktree** at `3ae6f3a4` was removed after §2.3 (its `venv` symlink first, then
  `git worktree remove`); `git worktree list` shows the main checkout alone. A re-run of §2.3 on the
  final code makes it afresh.
- **The set-aside work** is as it was: four stashes, `var/levin-phase-floor-07/wip/` unchanged.
- **Questions for the user:** $\Phi_w$ and $\Lambda_c$ (no rung qualifies under §2.2's rule); §1.3,
  for mechanisms (A) and (B); `test_levin_trust_yardstick.py`'s key-set assertion; and the four
  ComputeTargets tests that do not finish, which §1.3 decides.

## Amendment §10 (Rule A)

> Written by the executing agent on the laptop, 2026-11-08, for prompt 07 §10 (Decisions, **07 Part
> B stop 1**): a second code commit for 07, before Part B goes on. The sections above are kept as
> they were written.

**Subject:** prompt 07 §10 (Rule A), on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `2ddd3306` ("Amend levin-phase-floor 07 with Rule A at Part B's first stop"),
  whose code is `ad732887`'s (07's Part A code commit; module SHA-256 `dc05596c…`).
  `git log --oneline 2ddd3306..HEAD` was empty at the start and before staging.
- **The Gate held:** `git status --short` empty; `HEAD` the dispatch SHA; `runregistrykit list` 82
  runs, 80 finished, 2 `unknown` (`a3-pilot`, `a3-pilot-resume`, pre-registry), nothing `running`
  or stale; `git stash list` shows 07's four stashes (`stash@{0}`–`stash@{3}`). The stashes and
  `var/levin-phase-floor-07/wip/` were not otherwise touched.
- No Ray was started (the replay imports `ray.cloudpickle` to read the captures; every replay records
  `ray.is_initialized()` False), no store was opened, no run was registered, nothing was written
  under `var/`.

**Commit:** this amendment's code commit (one commit, not pushed; review pending).

**Result:**
- **Rule A is built** in `_locate_stationary_points`: each phase's total variation
  $V_i = \int|\theta_i'|\,dx$ by the trapezoidal rule on the 4,097 scan samples, intervals with a
  non-finite end left out, compared with `SIX_PI` by name; below it steps 2–4 do not run for the
  phase. **The new result key is `location_skipped_phases`**: one dict per skipped phase,
  `{"phase": i, "variation": V_i}`, empty when nothing is skipped or the location did not run.
- **The guard**, for a basis passing several rates to it: a skipped phase is tested for a sign change
  (samples of both strict signs on the nodes); `theta_prime_Cheb` and every located phase keep §1.4's
  test.
- **§2 re-run on the amended code** (`docs/levin-phase-floor/07/calibration/rule-a/out/`): **no call
  of §2.2, §2.3 or §2.4 skips a phase, and every call is bit-identical to run 2** (148 §2.2 units,
  105 §2.3 units, 8,640 LiouvilleGreen calls). The rungs do not move: $\Phi_w = 2\pi$,
  $\Lambda_c = 4$, the check holding.
- **Test 13 (a)–(d)** added to `test_levin_stationary.py`; **breakages (x) and (xi) caught**.
- **The seven roots:** AdaptiveLevin 103 → 107 (the four tests added), the other six equal.
- **Closed:** `[07-location-on-an-identically-zero-rate]` (board §4; `docs/OPEN_ISSUES.md` 135 → 134).

### What shipped

**`AdaptiveLevin/levin_quadrature.py`** (SHA-256 `4e5da371…`):
- `_locate_stationary_points`: Rule A as the first step for each phase, after the scan and before
  the roots; the returned dict gains `"skipped_phases"`. Its docstring states the rule and the
  estimate's error.
- **The estimate's error.** For a rate smooth between its roots, on the scan's spacing
  $h = |x_{\rm span}|/4096$: the trapezoidal rule's $|x_{\rm span}|\,h^2\max|\theta_i'''|/12$ on the
  smooth stretches; plus, at each root, where $|\theta_i'|$ has a kink, an overestimate of at most
  $h^2|\theta_i''|/4$ (on an interval $[0, h]$ with the root at $t$, the rule's excess is
  $|\theta''|\,t(h - t)$); plus the sum's rounding, about $n\varepsilon V_i$. Measured on test 13
  (b) ($\theta' = 70(x - x_0)$ on $[0, 1]$, linear, so only the kink term): $V = 18.475242613612156$
  against the exact $18.475241575014722$, an excess of $1.039\times10^{-6}$ against the bound
  $70h^2/4 = 1.043\times10^{-6}$. The estimate can move $V_i$ across `SIX_PI` only for a phase that
  close to it, and either side is covered (above, the location; below, the gate, and the guard
  where the gate still sends a region to the Levin solve).
- `_changes_sign` (new) and `_nodal_guard`, with a new keyword `guard_sign_change_phases` (the
  indices of skipped phases), threaded from `_adaptive_levin` through `_solve`,
  `_adaptive_levin_subregion` and `_adaptive_levin_subregion_impl`. It is set only when the basis
  passes several rates to the guard and the location ran; empty when nothing is skipped, so every
  call that skips nothing takes exactly its old path.
- The result's `"location_skipped_phases"`.
- **Docstrings:** the module docstring (the stationary-point item and the basis interface),
  `_adaptive_levin_subregion_impl` (its parameter and its account of the gate), `_nodal_guard`,
  `_locate_stationary_points`, `_adaptive_levin` (location item 1, guard item 4, the result keys)
  and `adaptive_levin_sincos` (`theta_deriv`, and the new key).

**`AdaptiveLevin/tests/test_levin_stationary.py`:** class `TestRuleA`, test 13 (a)–(d), with its
references as float hex and a region-record helper (test 8's fields); the module docstring names
Rule A. **`AdaptiveLevin/tests/test_levin_trust_yardstick.py`:** `GUARD_KEYS_07` admits
`location_skipped_phases` (and its comment says so); nothing else.

**`docs/levin-phase-floor/07/calibration/rule-a/`:** `scripts/` (copies of run 2's `prov.py`,
`calibrate.py`, `analysis07.py`, `reference.py`, `replay.py`, `variant_loader.py`,
`lg_seeded_draws.py`, `lg_compare_07.py`, changed as `out/scripts_diff.txt` shows; and three new
scripts, `test_references_rule_a.py`, `compare_run2.py`, `breakages_rule_a.py`), formatted by
`black` before the recorded run, and `out/`. Every output's header records its script's SHA-256,
`prov.py`'s, the module's (`4e5da371…`), `HEAD` `2ddd3306` and a dirty tree.

### §3 test 13 (`rule-a/out/test_references_rule_a.json`; every reference a closed form at 40 and 60 digits, agreeing to $\le 1.1\times10^{-41}$, and against `mpmath.quad` to $\le 1.7\times10^{-41}$)

All at `rtol` $10^{-9}$, `atol` $10^{-14}|{\rm ref}|$, on $[0, 1]$, $f = (e^{-x}, \cos 3x)$ unless
said.

| | case | located | skipped | regions | guarded | converged | $|v - {\rm ref}|$ / abserr |
|---|---|---|---|---|---|---|---|
| (a) | $\theta \equiv 0$, `theta_deriv` $\equiv 0$, ref $\sin 3/3$ | nothing | phase 0, $V = 0$ | 1 | 0 | yes | $1.39\times10^{-17}$ / $2.47\times10^{-15}$ = 0.0056 |
| (b) | $\theta = 35(x - x_0)^2$, $V = 18.475 = 0.980\cdot6\pi$ | nothing | phase 0, $V = 18.4752426$ | 5 | 1 (the root leaf) | yes | $2.78\times10^{-17}$ / $1.96\times10^{-13}$ = $1.4\times10^{-4}$ |
| (c) | $\theta = 36(x - x_0)^2$, $V = 19.003 = 1.008\cdot6\pi$ | root at $x_0$ exactly, one window, 6 pieces | none | 7 | 0 | yes | $1.39\times10^{-16}$ / $7.21\times10^{-12}$ = $1.9\times10^{-5}$ |
| (d) | test 9's basis, $\theta_1 = 200x$, $\theta_2 \equiv 0$; $f = (e^{-x}, 0, 0, \cos 3x)$ | nothing | phase 1, $V = 0$ | 1 (Levin) | 0 | yes | $1.53\times10^{-16}$ / $1.97\times10^{-14}$ = 0.0078 |

- (a) and (b) are bit-identical, in value, abserr, `converged` and every returned leaf's fields, to
  the same call with `_test_only_bypass_location`; (b)'s reported $V$ is within the estimate's
  stated error of the exact one.
- **(b) reaches the guard.** The root leaf's own span estimate, from its 16 nodes, is 23.69, above
  $6\pi$, so the C2 gate sends it to the Levin solve with the root unlocated; the nodal guard catches
  it, and it is bisected into Clenshaw–Curtis halves (one guarded region, under Rule A and under the
  bypass alike). This is §10's "a region it still sends to the Levin solve … is the nodal guard's to
  catch", met on the first case built (deviation 42).
- (c) at $c = 72$ is located and windowed as before Rule A.
- (d)'s zero block in the Levin system does not trouble the solve: one Levin leaf, converged.

### Breakages (`rule-a/out/breakages_rule_a.txt`; each a text substitution in a copy of the module installed in a fresh process, the tree untouched; each named test passes on the tree as a control)

| | breakage | test | result |
|---|---|---|---|
| (x) | Rule A off (`variation < SIX_PI` → `variation < 0.0`) | 13 (a) | **fails**: 4,097 points located (`stationary_points` not empty); the same call then makes one window and 4,102 pieces, R28's shape (honest, 17.7 s) |
| (xi) | a skipped phase tested for "not of one strict sign" (`_changes_sign(sample)` → `not _strictly_one_sign(sample)`) | 13 (d) | **fails**: 15 guarded regions, not 0 |

## Amendment §10: predictions and verdicts

§10's "What must not move", scored on `rule-a/out/compare_run2.txt` (`compare_run2.py`; set aside:
wall-clock times, the module's SHA-256, the replay's `evidence`, Rule A's key, and for a
LiouvilleGreen call under `commit` 06m's `result_sha256` and `result_keys`, which count the result's
keys; `result_sha256_07`, which sets the guard's keys aside, Rule A's among them, is compared):

| rule | measured | verdict |
|---|---|---|
| §2.2 at `rtol` $10^{-9}$: every call with no phase skipped bit-identical to run 2 | 148 of 148 units paired; none skips a phase; 148 identical | **holds** |
| §2.3, the 14 calls (and r55): the same | 105 of 105 units (g06, parent, five rungs, each sid); none skips; 105 identical; the replay check reproduces all 15 again | **holds** |
| §2.4, the draws: the same | 160 of 160 units; 8,640 calls (4,320 per code); none skips; 8,640 identical, and every unit's outcome, three-Bessel results and output identical | **holds** |
| every call with a skipped phase listed with both results | none | **holds** (nothing to list) |
| a rung of $\Phi_w$ or $\Lambda_c$ that moves is a stop | $\Phi_w$ $2\pi$ → $2\pi$; $\Lambda_c$ 4 → 4; the check holds in both | **holds** |
| the seven roots: a fall is a stop, a rise equals the tests added | AdaptiveLevin 103 → 107 (+4, test 13 (a)–(d)); the others equal | **holds** |
| any other test whose assertion moves is a stop | none moved; one key-set allowance (`GUARD_KEYS_07`) | **holds** |

`lg_compare_07.txt` on the re-run (parent against the amended code) repeats run 2's: no test fails;
3,897 of 4,320 paired calls locate nothing and guard nothing, all bit-identical to the parent; 423
locate a point (419 roots, 4 minima), none guards.

## Amendment §10: deviations from the prompt

38. **Rule A's estimate uses the scan's own spacings** (`numpy.diff` of the `linspace` grid, not one
    $h$), and is compared with `SIX_PI` inside `_locate_stationary_points`, not passed in as a
    parameter. `IMPLEMENTATION CHOICE`.
39. **The key is `location_skipped_phases`**, a list of `{"phase", "variation"}`, always present in
    `_adaptive_levin`'s result (empty when nothing is skipped, or the location bypassed or not run);
    the location's own dict gains `"skipped_phases"`. `IMPLEMENTATION CHOICE`.
40. **The guard's mechanism:** a keyword `guard_sign_change_phases` (a frozenset of indices into
    `guard_rates`) threaded to `_nodal_guard`, and a helper `_changes_sign`. A zero or a non-finite
    sample of a skipped phase does not count, following "samples of both strict signs" literally;
    for a located phase a non-finite sample still guards. `IMPLEMENTATION CHOICE`.
41. **A skipped phase still bounds the window pieces** of a multi-phase basis whose other phase is
    located (§1.2 item 3, "in every phase of the basis", unchanged): Rule A removes its points,
    windows and minima, not its share in the piece cuts. `IMPLEMENTATION CHOICE`.
42. **Test 13 (b) asserts that the guard catches its root leaf** (one guarded region, under Rule A
    and under the bypass), beyond §10's list. The first draft asserted no guarded region and failed:
    the root leaf's node estimate of its span is 23.69, above $6\pi$, though $V = 18.475$. §10
    foresees this path; the test now pins it. `IMPLEMENTATION CHOICE`.
43. **Test 13's problems:** (a), (b) and (d) are driven through `_adaptive_levin` (test 8's switch
    is there; (d) as test 9 is), (c) through `adaptive_levin_sincos`; $x_0 = 0.3819660112501051$;
    $c = 70$ and 72 ($V/6\pi$ = 0.980 and 1.008); (d) is `_TwoPhaseQuadraticBasis(200, 0, x_0)`,
    whose second rate is $\pm0$ on every node, with $f = (e^{-x}, 0, 0, \cos 3x)$. Each test also
    asserts what it reports: (a), (b) `converged` equal to the bypass, (c) no phase skipped.
    `IMPLEMENTATION CHOICE`.
44. **Test 13's references** come from a new recorded script, `test_references_rule_a.py` (closed
    forms at two precisions, checked against `mpmath.quad`; (b), (c) by `reference.closed_R` with
    `contour_R` beside it), held as float hex in the test module, as 07's tests hold theirs.
    `IMPLEMENTATION CHOICE`.
45. **The copied scripts change beyond their paths** (`out/scripts_diff.txt`):
    - paths: `prov.py`'s `REPO` (`HERE.parents[5]`) and `OUT` (`rule-a/out/`), and
      `lg_seeded_draws.py`'s `VARIANT_06M` (`HERE.parents[3]`, unused: route B is not a mode);
    - `analysis07.summary` records `location_skipped_phases` in each record;
    - `lg_seeded_draws.py` sets the key aside in its `GUARD_KEYS_07`, so that `result_sha256_07`
      stays comparable with run 2's, and records it per call (`location_skipped_phases_07`).
    Without the last two no call with a skipped phase could be listed, and every call's digest would
    differ from run 2's by the new key. `black` changed nothing in the copies but the wrapping of my
    own edits. `calibrate.py`, `reference.py`, `replay.py`, `variant_loader.py` and
    `lg_compare_07.py` are unchanged. `STRUCTURALLY REQUIRED`.
46. **Three new scripts:** `compare_run2.py` (the comparison with run 2, its set-asides as stated
    above), `breakages_rule_a.py` (07's `breakages.py`'s mechanism, for (x) and (xi)) and
    `test_references_rule_a.py`. `IMPLEMENTATION CHOICE`.
47. **§2 was re-run whole,** the modes that do not depend on the amended code included (the parent,
    and §2.3's `g06` replay check in a scratch worktree at `3ae6f3a4`, made, found clean, and removed
    with its `venv` symlink), so that the harnesses run as they ran in run 2 and every unit pairs.
    Run 2's settings: §2.2 six workers, one-hour unit limit; §2.3 four workers, 900 s; §2.4 three
    workers. `IMPLEMENTATION CHOICE`.
48. **A first recorded pass was discarded.** After §2.2 and §2.3 had run and §2.4 had started, I
    edited docstrings of the module, so those outputs named a module SHA-256 other than the
    committed one. §2.4's run was stopped (its processes by pid), every output in `rule-a/out/` but
    `scripts_diff.txt` deleted, and the whole re-run made again on the final module (`4e5da371…`,
    11:51–12:05). No record of the first pass is kept; its §2.2 and §2.3 comparisons were not made.
    `UNINTENDED DRIFT` (corrected).
49. **The re-runs are not registered.** Each is under ten minutes (§2.2 61 s, §2.3 208 s, §2.4
    537 s); README §5 rule 7 asks registration above about ten minutes. `IMPLEMENTATION CHOICE`.

## Amendment §10: verification performed

- **The seven roots at the dispatch SHA** (before any edit; the seven run in parallel):
  AdaptiveLevin 103 OK; ComputeTargets 781 OK (492 s); CosmologyModels 39 OK; LiouvilleGreen 156 OK
  (skipped=1); ObjectFactories 280 OK; RunDrivers 127 OK; tools 64 OK. They equal the dispatch's.
- **The seven roots on the commit's tree** (in parallel, 12:06–12:13, after §2's re-run; ComputeTargets 449 s): AdaptiveLevin **107** OK; ComputeTargets 781 OK;
  CosmologyModels 39 OK; LiouvilleGreen 156 OK (skipped=1); ObjectFactories 280 OK; RunDrivers 127
  OK; tools 64 OK.
- Before the key-set allowance, `test_levin_trust_yardstick.py`'s `test_bit_identical_to_the_variant`
  failed on every cell on the new key alone; after it, it passes, every other field compared as
  before.
- **§2's re-run and its comparison** as above; the references and the breakages as above.
- **`black --check`** on the module, the two test modules and `rule-a/scripts/`: clean.
- `git log --oneline 2ddd3306..HEAD` empty before staging; `git worktree list` the main checkout
  alone after §2.3.

## Amendment §10: observations not acted on

- **O9. A phase Rule A skips can still reach the Levin solve** (test 13 (b)): the gate's per-region
  estimate, $\overline{|\theta'|}$ on the Chebyshev nodes times the width, weights the ends, where
  $|\theta'|$ of a quadratic phase is largest, and reads 23.69 against $V = 18.475$ on $[0, 1]$.
  The guard caught the root there, as §10 intends; it would miss a pair of sign changes between two
  adjacent nodes of such a region. No production phase is known to be in that band; it bears on
  `[07-c2-gate-threshold-was-never-measured]`, which owns the gate's estimate. Not opened.
- **O10.** None of §2's calls has a phase below `SIX_PI` (no family, real call or draw skipped one),
  so §2 does not exercise Rule A at all; test 13 and R29 (the 26 `q-r` calls) are where it is
  measured.
- `[07-location-cuts-a-flat-stretch-at-every-zero-sample]` and
  `[07-c2-gate-threshold-was-never-measured]` are untouched, as §10 says.

## Amendment §10: state handed on

- **Committed** (one commit, not pushed; review pending): the module (`4e5da371…`), the two test
  modules, `docs/levin-phase-floor/07/calibration/rule-a/`, this log, the board (the issue moved to
  §4, the 07 row), `docs/OPEN_ISSUES.md` (135 → 134) and `CLAUDE.md` (the `black` list).
- **Part B after it** (§10, the note's §B.6): R29 at this commit in R28's place, against R27 and R28;
  S1 and S2 re-scored on R29; the census at this code, V3 setting aside a phase Rule A skipped.
- **The set-aside work** is as it was: four stashes, `var/levin-phase-floor-07/wip/` unchanged.

## Part B: the Mac Studio record (appended by the orchestrator, 2026-11-10)

Part B ran from 2026-11-08 to 2026-11-10 on the Mac Studio (`mps026298`), from three worktrees the
laptop made:
- R27, S2's captures and V0's K0 run at the parent `91472dbe`, whose production code is 06m's;
- R28 at `74600350`, Part B's first dispatch, whose code is `ad732887`'s;
- after stop 1 and Rule A, R29, the census and V4's references at `e182bc7a`, Part B's resumed
  dispatch, whose code is `adffa21b`'s.

The Mac Studio orchestrator dispatched a rehearsal agent (R27 and R28, then R29), then a census
agent. It held each at the note's planned stops 1, 1′, 2, 3 and 4 and checked each with readers of
its own, then reviewed Part B in the nine checks of the note's §B.3, all passing. The records are in
`docs/levin-phase-floor/07/`:
- `rehearsal/`, `rehearsal/r29/` and `census/`: each agent's log, verification, issues, scripts and
  outputs. The three logs are appended below, as written on the Mac Studio; only the census log's
  headings are moved down one level to sit under this log's own;
- `review/` and `review.md`: the orchestrator's dispatch briefs, stop records, scripts and hashes,
  and its review.

What stays on the Mac Studio is in `07/KEPT_ON_MAC_STUDIO.md`; `VERIFICATION.md` §7 is the summary;
the orchestrator note's §B.8 is the story of Part B after stop 1′.

**The verdicts on the predictions above** (each entered against its rule there too):

| | Verdict | Numbers |
|---|---|---|
| §5, the fingerprints | **holds** | R27, R28 and R29 `1b332ffa…` |
| §5, S0 | **holds** | R27 equals R26 in all seven value tables (360,462 rows) and all 864 `QuadSourceIntegral` rows, every value and counter field |
| §5, R28 (then R29) against R27 | **holds** | untouched rows bit-identical: R28 640 of 640, R29 692 of 692; the rest tabulated; 17 rows gain convergence, none loses |
| §5, S1 | **holds** | 12 of 12 locate one root at $z = 10211.3179 \pm 9\times10^{-6}$ and are honest, worst 0.218 (w11), on R28 and R29 |
| §5, S2 | **holds** | both rows located within one spacing, 0.020 and 0.263 of their abserr, on R28 and R29; R27 $5.6\times10^5$ and $4.1\times10^4$ |
| §6, Gate V0 | **holds** | (a) no file the harness loads differs; (b) 100 of 100 K0 units equal 06m's; (c) deterministic |
| §6, V1 | **holds** | 284,514 calls and 16,104 units equal 06m's, at both rtols |
| §6, V2 | **holds** | 0 returned Levin leaves hold a sign change, 182,066 calls scanned |
| §6, V3 | **holds** | 0 violations in 170,582 calls; 11,484 set aside under §10 |
| §6, V4 | **fails** | 28 of 33 honest; v08 3.16, v09 2.21, v12 3.08, v17 2.38, v22 2.53 × $\text{abserr}_g$ |
| §6, V5 | **fails, as written** | 23 `G+q+r` calls with 2–54 leaves at `depth_max`, 3 converged; its premise was stale (Predictions and verdicts above) |
| §B.7, R29 against R28 | **fails as written on 4 records** | 86 of 90 changed records equal R27; the 4 guarded analytic blocks are findings, the rule the orchestrator's error (the note's §B.8) |

**The census.** `levin-phase-floor-07-census-20261108T171137`, checkpoint `03dfe786…`, 18,560 of
18,560 units in 22,708 s (06m 18,465 s), none over budget. Rows at depth 20 fall from 192 to 77 at
`1e-7` and from 213 to 83 at `1e-8`. V4's references: `levin-phase-floor-07-references-20261109T005328`,
74 readings in about 25.6 h, on the user's subset at stop 4.

**The cross-check,** beyond §6, on the user's word at stop 4: 2,484 pairs, $r_g \le 1$ in 2,482
(the exceptions `3-14501` at 1.39 and `3-35006` at 1.08, both converged `G+q+r` calls at
$k = 10^5$) and $r_{06m} \le 1$ in 2,432. 843 calls disagree beyond $\max$(abserr): (a) settled for
07 31, (b) 0, (c) undecided 786 (396 rows), (d) 0, (e) 26.

**Deviations in Part B.** The rehearsal agent logged 15 (R27 and R28) and 9 more (16–24, R29); the
census agent logged 40. The Mac Studio orchestrator accepted each:
- `STRUCTURALLY REQUIRED`: the job-name prefix normalised for S0; `p1-references`' manifest naming
  the main checkout; the stop on R29's four records under §B.7's rule as written; the census
  harness's changes within §6's list; K0 at the parent computing 06h's counterfactual; V4's capture
  scripts; and the reading subset and its scoring rule, by the user's ruling at stop 4;
- `UNINTENDED DRIFT`, none with an effect: a scratch file in `/tmp` and a mismatched grep at the
  gate; two mistyped `pgrep -P` pids; the `q-r` references failing twice before they ran; a gate
  check printing the main checkout's ignored files; a broken `run_k0.py` line found by reading; log
  times corrected; a superseded projection; two scoring readers that crashed and were re-run; an
  output directory made while a capture ran; a projection that failed on a missing input and was
  started with `&`; a capture chain started with `&`; two figures misstated in the stop-4 report and
  one in a message, corrected in the log; a before-against-after comparison re-run under `bash`;
- the rest `IMPLEMENTATION CHOICE`.

**Observations not acted on, and not opened as issues** (the agents' `issues.md`, and the user's
word at the records commit):
- **The 786 undecided calls** of the cross-check (class (c)), where each code agrees with itself
  across rtols and the two disagree beyond both errors. They are undecided for want of references.
  The user: *"these are undecided because we don't have references for them? If so, then yes, we
  attribute them to 06m's errors for the time being, on this evidence."* V4 referenced 12 of them:
  06m fails all twelve by 356–$5.2\times10^5$×, and 07 is honest on 7 and 2.2–3.2× out on 5.
- **The lone-`G` minima:** 236 group calls per rtol locate 6 or 19 minima of $|\theta'|$, λ about
  $10^{-7}$–$10^{-6}$; all converged, nothing measured wrong.
- Rule A skips 64 analytic blocks that R28 located, not only the 26 `q-r` calls; it skips phases in
  423 replayed calls in all, of every group and analytic label.
- The rehearsal agent's recurrences: 05's classifier and job names; the registry recording the
  launcher's repository; QAGS 4.9× its own estimate from the Gauss–Legendre sequence on the worst
  `q-r` call.

**On the laptop (2026-11-10)** (`VERIFICATION.md` §7.7): the copied files and the 29 run-directory
files match the Mac Studio's SHA-256 lists. S1 was recomputed exactly from `s1_scores.json` (worst
0.217573), and its twelve `ref_30` strings found verbatim in 06g's committed V1. V4 was re-scored in
exact Fractions on v00, v08 and z12 from both census checkpoints. A moved and an untouched rehearsal
row were read from the R27 and R29 stores in `float.hex`, and all 692 untouched rows compared. Every
figure is the same as the agents' and the Mac Studio orchestrator's.

**The user's rulings** (the board's Decisions, **07 stop 1′**, **07 Part B stops 2–4**, **07
results** and **G2 L2, reopened**):
- `[06-levin-accepts-a-region-across-a-stationary-point]` closes, on V2: *"I didn't address this,
  but yes, please close it"*;
- `[07-iroff1-window-stop-can-leave-a-call-dishonest]` is opened from the census agent's proposal,
  and assigned to prompt 07a with O9 of `[07-c2-gate-threshold-was-never-measured]`, which O9's
  census calls narrow (*"I think so"*): *"I agree we fix the early roundoff stop and the O9 problem.
  These are issues where the problem and the fix are both clear."*;
- V5 is scored fails as written; G2's L2 is reopened as one question for 08, with the depth-cap
  experiment: *"Include the cap experiment in prompt 08."*

## §5 — rehearsals R27 and R28 on the Mac Studio

**Prompt:** [`07-stationary-point-guard.md`](../07-stationary-point-guard.md) §5 ("As 06m §4"), under the
orchestrator's Part B dispatch (`74600350`, addendum B.5), with the user's addendum at planned stop 1. The code
measured is 07's code commit `ad732887` (location, windows, the nodal guard and §1.3's per-window `iroff1` rule in
`_adaptive_levin`; the four counts in `QuadSourceIntegral`'s Levin records), against its parent `91472dbe`, whose
production code is 06m's.

**Host:** mps026298, the Mac Studio.
- **R27, S2's capture and the q-r captures at p** ran in the worktree `SecondaryGWKit-lpf07-91472db`, detached at
  `91472dbe` (`levin_quadrature.py` `beaf0e23…`, `QuadSourceIntegral.py` `962424c0…`; against R26's tree
  `38f6a828` only `CLAUDE.md` differs; `bd21bf23..91472dbe` outside `docs/` and `prompts/` is empty).
- **R28, the comparisons, the replay, the q-r references, S1, the readers and S2's references** ran in the
  worktree `SecondaryGWKit-lpf07-7460035`, detached at `74600350` (`levin_quadrature.py` `dc05596c…`,
  `QuadSourceIntegral.py` `33cb161e…`). `git diff --stat 91472dbe 74600350` outside `docs/` and `prompts/` touches
  exactly the six files the dispatch lists; `ad732887..74600350` outside `prompts/` is empty.
- **The same file in both trees:** the launcher `RunDrivers/quadsource_atol_sweep.py` (`b0135218…`), `main.py`
  (`a5d47f5f…`), `LiouvilleGreen/three_bessel_integrals.py` (`fea67d56…`), `tools/store_value_diff.py`
  (`7997af85…`), `ComputeTargets/phase_groups.py` (`795a0119…`), `config/fingerprint.py` (`9fb94d77…`).
- **The main checkout** stayed at `28ebaf07`, clean; it ran no tree code.

**Model:** Claude Opus 5.5, the rehearsal subagent of the Mac Studio orchestrator, in six turns: (1) the gate, the
lock, R27's launch; (2) R27's checks, S0, R28's launch; (3) R28's checks, R28 against R27, the tabulation
(planned stop 1, lock held); (4) the replay's launch; (5) the replay's checks, the q-r references, S1, the readers,
the release, S2's references' launch; (6) S2's scoring and these records.

**Date:** 2026-11-08, 00:05–10:50 UTC.

**Commit:** none. The records are committed on the laptop, in 07's records commit.

**Result:** COMPLETE. **S0, S1 and S2 HOLD**; outside `QuadSourceIntegral` every value table is identical; inside it
every untouched row is bit-identical and the rest are tabulated. No §8 stop condition fired. One finding for the
user: the location on an identically zero rate (the q-r groups with q = r), where R28 is honest but does about
4,100–5,500 Clenshaw–Curtis pieces per call.
- **The fingerprints.** R27 and R28 both `done`, exit 0, `1b332ffa…`; `classes` equal R26's and R25's; `problems {}`.
- **S0 — R27 against R26: value-identical.** 7 value tables, 360,462 rows, identical; all 864 `QuadSourceIntegral`
  rows bit-identical in every value and counter field. Labels differ only by the job-name prefix and stamp
  (normalised in a copy, as 06m did).
- **R28 against R27, outside `QuadSourceIntegral`:** every value table identical (360,462 rows); every other
  difference a timing, label, `parents_digest` (291) or `rho_reuse` (26); 0 unclassified; no row in one store only.
- **Inside it:** 640 rows untouched (every count 0 in every Levin record), **all 640 bit-identical**; 224 rows
  tabulated (70 with a group call that located a point, 200 with an analytic block that did). No leaf guarded.
  `total_converged` 845 → 862: 17 rows gained, none lost. The largest move is 7.4e4 × R27's `total_abserr`.
- **The replay** (the user's approval): 470 of 470 touched Levin records reproduced bit for bit at R28's tree, giving
  every located point's kind and location: G+q+r roots at z = 10211.32 (42) and 275.009 (2); analytic J4/Y4 roots
  (400); each q-r call with q = r 4,097 points, every scan sample, θ′ ≡ 0.
- **The q-r references** (the user's addendum): on three of the 26 calls, the worst included, R28 is honest (0.179,
  0, 0.069 of its abserr) and so is R27 (0.0021, 0.0014, 0.0006). On the worst call R28's verdict rests on the
  Gauss–Legendre sequence; against the reference's stated error, set by QAGS, it is undecided (below).
- **S1 holds:** w00–w11 each locate one root at z = 10211.3179 ± 9e-6, and each is honest (largest 0.218, w11).
- **S2 holds:** both P1 rows locate their point (z 10211.318, 275.009; |x − x_s| ≤ 7e-9 against a spacing of 0.0041)
  and are honest against four agreeing readings (0.020, 0.263). R27 reproduces P1's failure (5.6e5, 4.1e4).
- **The readers** exit 0, six of six; R28 byte-identical before and after. **The lock** is released.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md` §7.

### What shipped

No code. The records are under the main checkout's `var/levin-phase-floor-07/rehearsal/`, for the laptop to
commit as `docs/levin-phase-floor/07/rehearsal/`:
- `log.md` (this), `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`.
- **`scripts/`:**
  - `rec.sh` (`b0952204…`): 06m's (`0a13627b…`), the record path changed by `sed`;
  - `classify_non_qsi.py` (`26690e61…`): 05's, unchanged, byte-identical to 05's and 06m's copy;
  - `normalise_jobname_labels.py` (`b02fd3fd…`): 06m's (`08dc45a8…`), its prefixes and docstring changed
    (`out/normalise_jobname_labels.diff`);
  - `qsi_identity_r26_r27.py` (`244840d0…`): 06m's `qsi_identity_r24_r25.py` (`e63aae0c…`), R24/R25 → R26/R27
    and a provenance header (`out/qsi_identity_r26_r27.diff`);
  - `s0_labels_and_qsi_tool_report.py` (`88357082…`), new;
  - `tabulate_qsi_guard.py` (`a1eaba31…`), `qsi_guard_records.py` (`e327f3f9…`), `qsi_rows_summary.py`
    (`aa317123…`): new; the store reader and field classification copied verbatim from 06m's (06g's);
  - `capture/`: 06g's `references/v1/scripts/` harness (`worker.py`, `dispatch.py`, `harness.py`, `rebuild.py`,
    `prov.py`, `run_census.py`, `launch.py`, `rerun.py`, `spscan.py`), copied (hashes equal to 06g's,
    `state/capture_harness_06g_v1_sha256.txt`); `prov.py` (`f84a1771…`) and `worker.py` (`ba140a73…`) changed
    where marked "prompt 07 rehearsal" (`out/capture_harness_diff.txt`); `replay07.py` (`68f985eb…`) and
    `run_replay_pipeline.sh` (`11139d79…`), new;
  - `replay_summary.py` (`8c978d0b…`), `s1_score.py` (`21cfa56f…`), `qr_reference.py` (`add90893…`),
    `run_qr_reference.sh`, `qr_score.py` (`f6d6282a…`), `qr_worst_detail.py` (`a2e7131e…`), `p1_references.py`
    (`d3770e03…`), `s2_score.py` (`a2bcd258…`): new;
  - `run_readers_worktree.sh` (`ec96cf09…`): 06m's (= 06g's), unchanged; its diff against the original,
    `out/run_readers_worktree.diff`, equals 06m's.
- **`out/`:** `commands.log`; the gate (`00-gate-trees.txt`, `01-gate-registry-lock.txt`, `01-registry-list.txt`,
  `02-r26-vs-06m-kept.txt`); each build's launch and end checks and `status.json`; S0's (`classify_non_qsi_r26_r27*`,
  `normalise_jobname_labels.*`, `qsi_identity_r26_r27.*`, `s0_*`); R28 against R27 (`classify_non_qsi.*`,
  `qsi_tabulation.*`, `qsi_rows.csv`, `qsi_guard_records.*`, `qsi_rows_summary.txt`, `stop1_check.txt`,
  `depth_max_and_counters.txt`); the replay (`replay_rows.txt`, `replay_dev1*`, `replay_pipeline.log`,
  `replay_run.txt` … `capture_check.txt`, `replay_calls.csv`, `replay_calls.checks.json`, `captures.json`,
  `replay_summary.txt`, `replay_store_check.txt`, `replay_launch_check.txt`); the q-r references
  (`qr_reference/` with its two failed attempts kept, `qr_scores.*`, `qr_worst_call_detail.txt`); S1
  (`s1_scores.*`); S2 (`p1_references/` small files, `p1_references_status.json`, `p1refs_*`, `s2_scores.*`); the
  readers (`readers.txt`, `readers_check.txt`); `final_check.txt`, `final_registry_list.txt`.
- **`state/`:** `lock.log`; the launch lines (`launch_r27.txt`, `launch_r28.txt`, `launch_replay.txt`,
  `launch_p1_references.txt`); the markers; every store-hash file (R26 at the gate; R26/R27 before and after S0;
  R27 at R28's launch; R27/R28 before and after the comparison, after the replay; R28 before and after the readers;
  `stores_sha256_final.txt`); the replay's row list and specs.
- **Not for copying back** (`KEPT_ON_MAC_STUDIO.txt`): the three diff JSONs, the readers' `extract/`, `state/capture/`
  and `state/inputs/` (the pickles), the tool's two reading JSONs (2.5 MB each) and their `.log` copies.
- **The stores are kept:** `var/datastores/levin-phase-floor-07-r27*` and `…-r28*`, five files each.

### Predictions and verdicts

| Rule (prompt 07 §5, as written) | Scored on | Verdict |
|---|---|---|
| R27 and R28 with 06m's command line, slugs `r27-parent`, `r28-guard`, `--prompt 07`; **both fingerprints `1b332ffa…`** | both `done`, exit 0, `1b332ffa…`; `classes` equal R26's and R25's; `problems {}` | **holds** |
| **S0 — R27 against R26:** every value table identical, every `QuadSourceIntegral` value field identical, every other difference of the four volatile kinds | 7 tables, 360,462 rows identical; 864 / 864 rows bit-identical in 118,776 value and 27,163 counter fields; other: timing, label (prefix and stamp, normalised in a copy), `parents_digest` 291, `rho_reuse` 26 | **holds** |
| R28 against R27, outside `QuadSourceIntegral`: every value table identical, every other difference of the four volatile kinds, no row in one store only | 360,462 rows identical; 0 unclassified; 0 only-in-one, unmatched or unresolvable | **holds** |
| Inside: a row in which nothing was located or guarded is bit-identical in `total`, `total_abserr`, the analytic columns and every Levin record, the four counts set aside | 640 untouched rows; 640 bit-identical (value and counter fields) | **holds** |
| Inside: any other row tabulated (change against both `total_abserr`, located points (kind, z), window pieces, regions, maximum depth, `total_converged`) | 224 rows in `qsi_rows.csv`; 470 records in `qsi_guard_records.csv`; kind and location for each from the replay (`replay_calls.csv`, 470 / 470 bit for bit) | **done** |
| The stationary-point rows' convergence in each store, and the rows whose `total_converged` changes | rows with a group call that located: 52 / 70 converged (R27), 69 / 70 (R28); all rows: 845 → 862; 17 gained, 0 lost (listed) | **done** |
| **S1 — V1's twelve:** each locates its point at z ≈ 10211 and \|v28 − ref\| ≤ abserr28 against 06g's reference | 12 / 12 one root at z = 10211.31789148–50 (within 8.5e-6 of 10211.3179; spacing 31–44); ratios 0.00037–0.218 | **holds** |
| **S2 — P1's two rows:** each locates its point within one scan spacing; each G+q+r call within its abserr of a reference taken afresh (captured at R27's tree, four readings) | row 1: \|x − x_s\| 7.1e-9 (spacing 0.0041), ratio 0.020; row 2: 5.6e-9, 0.263; readings agree (scatter 6.1e-37, 1.2e-36 against 6.6e-33, 9.2e-33) | **holds** |
| The readers run on R28, exit 0, the store byte-identical | 6 / 6 exit 0; five files identical by SHA-256, size, mtime; no journal | **holds** |
| The lock is released, `ray-processes` lists nothing | released; `status` free; `ray-processes` empty before and after | **holds** |
| The stores are kept | R27 and R28 in place, unchanged since the comparison | **holds** |

**The user's addendum at planned stop 1** (not a prompt rule; reported): **R28 is honest on the three referenced
q-r calls**, and so is R27:

| call (q = r) | R27 \|v−ref\|/abserr | R28 \|v−ref\|/abserr | reference's stated error / R28 abserr |
|---|---|---|---|
| (1e5, 1e6, 1e6, z 6.985e9), the worst | 0.00215 | 0.179 | 5.8 |
| (3.162e5, 1e6, 1e6, z 1.363e10) | 0.00142 | 0 | 0.15 |
| (1e5, 3.162e5, 3.162e5, z 3.047e9) | 0.000618 | 0.0695 | 0.022 |

**On the worst call R28's "honest" rests on the Gauss–Legendre sequence alone.** GL at m = 8, 10, 12, 14 agree to
5.9e-39 (GL12 = GL14), and R28 is 0.179–0.186 of its abserr (7.89e-37) from each. The reference's stated error,
4.56e-36, is |QAGS − GL14|: QAGS sits 4.9 times its own estimate (9.29e-37) from GL14, and R28 is 5.6 of its abserr
from QAGS. Against the stated error the verdict is undecided. On the other two calls it is honest outright. R27 is
honest on all three under any reading.

### Deviations from the prompt

1. **The reading for S1 and S2 was deferred until after planned stop 1** (06g's `scores.json` and `report.md`, `v1/`
   scripts, the capture harness, `tools/filon_reference.py`). The brief listed them among what to read; nothing
   before the stop depended on them. IMPLEMENTATION CHOICE. 06g's and 06h's §4 were not read: 06m's records gave the
   launch's exact form.
2. **At the gate one read-only check wrote and removed a scratch file `/tmp/x_never`** (outside the write scope),
   and its diff of R26's hashes exited 1 (the grep also matched R26's run-directory line). Redone properly
   (`out/02-r26-vs-06m-kept.txt`: equal). UNINTENDED DRIFT.
3. **S0's labels were normalised in a copy of the JSON** (06m's deviation 1): R26's job name is
   `levin-phase-floor-06m`, R27's `levin-phase-floor-07`; the raw classifier reports 5,148 UNCLASSIFIED, the
   normalised copy 0. The verdict is the same either way. STRUCTURALLY REQUIRED.
4. **S0's identity inside `QuadSourceIntegral`** used 06m's identity script, adapted, and a new corroboration script
   (the labels; the tool's own `QuadSourceIntegral` report). IMPLEMENTATION CHOICE.
5. **A mistyped parent pid in `pgrep -P`** when finding R28's driver (read-only; printed nothing). UNINTENDED DRIFT.
6. **The tabulation inside `QuadSourceIntegral` is new scripts**, not 06m's `tabulate_qsi_signedyardstick.py`: the
   classification by R28's four counts and the untouched-row test are new; the reader and field classification are
   06m's (06g's), verbatim. IMPLEMENTATION CHOICE.
7. **Two inspections were printed and not saved** (R28's metadata structure at stop 1; a tally over
   `qsi_guard_records.csv`), and two more later (a q-r pickle's keys and timing; `scores.json`'s structure). Every
   figure used is in a saved output. IMPLEMENTATION CHOICE.
8. **A two-unit development run of the replay** (`dev1`), to check the harness on 07's code and time it; kept.
   IMPLEMENTATION CHOICE.
9. **The replay ran every touched row at R28's tree only** (enough for kind and z); R27's tree ran only P1's rows and
   the q-r rows, for the captures. IMPLEMENTATION CHOICE.
10. **The harness lives in `scripts/capture/`** (so `prov.py`'s `parents[4] == MAIN` holds unchanged), the stores are
    reached through symbolic links in `state/stores/` (as 06g's `v1/stores/`), and a worker runs several units (one
    k cache per process). Kind and location come from `guard_07`, a record the worker reads from each call's own
    result after the timed section; for a call of more than 64 points it records a summary (counts, x range, whether
    every point is a grid sample) and re-samples θ′ and θ on the grid. IMPLEMENTATION CHOICE.
11. **The q-r references failed twice before they ran:** first a zsh loop did not split its arguments (six processes
    failed on a path); then scipy refused QAGS at epsrel 1e-14 (it must exceed 50 eps), fixed to 1.2e-14. Both
    attempts' outputs are kept under `out/qr_reference/failed_attempt_*`; all six runs were redone on the fixed
    script. UNINTENDED DRIFT.
12. **The q-r reference method** (the user's terms leave it to the agent): direct composite Gauss–Legendre at five
    resolutions with QAGS as a second rule, on the integrand captured at R28's tree; each capture replayed bit for
    bit at its tree, and the two trees' integrands compared bit for bit at 1,001 points. IMPLEMENTATION CHOICE.
13. **`p1-references`' manifest records the main checkout's `git_head` (28ebaf07)**, because its launcher lives under
    `var/` there; its cwd and the tool it ran are the code worktree's (74600350). STRUCTURALLY REQUIRED (the
    registry records the script's repository; the scripts' records live in `var/`).
14. **Two readings the prompt leaves open:** S2's count tolerance is min(abserr_R27, abserr_R28)/100 (06g's form, as §6
    V4 states it for the census); S1's "at z ≈ 10211" is read as one root within one scan spacing (in z, at the
    point) of 10211.3179, P1's value. Either verdict is the same at any reasonable reading (|z − 10211.3179| ≈ 9e-6).
    IMPLEMENTATION CHOICE.
15. **S2's references ran at the R27-tree captures from the code worktree** (as 06g read captures across trees); the
    integrand code is the same in both trees (`QuadSourceIntegral.py` differs only in the four counts; `phase_groups.py`
    and `LiouvilleGreen/` are equal). IMPLEMENTATION CHOICE.

### Verification performed

**The gate** (`out/00-gate-trees.txt`, `out/01-gate-registry-lock.txt`): the trees as stated above; 62 runs, all
finished; `~/.raypoolkit` empty, no old lock; `auth_token` mode 600; 135 GB free; venv 3.13.16 with datastorekit 0.5.0,
raypoolkit 0.3.0, runregistrykit 0.2.0, ray 2.59.0, SQLAlchemy 2.0.55, numpy 2.2.4, scipy 1.16.3, mpmath 1.3.0; no 07
store or run; R26's five files equal to 06m's record.

**R27 and R28** (`out/r2{7,8}_{launch,end}_check.txt`): R27 launched 00:06:33Z (sh 83000, driver 83008), `done`
00:15:41Z (about 9 min); R28 launched 00:19:15Z (sh 83821, driver 83825), `done` 00:35:54Z (about 16.5 min); each
manifest at its tree with `git_dirty` false and the launcher `b0135218…`; no Traceback and no `!! WARNING`; each
driver's cwd its worktree.

**S0 and R28 against R27:** `tools.store_value_diff --json` from the code worktree (16.9 s each); the stores
byte-identical before and after each comparison (`state/s0_store_sha256_*`, `state/store_sha256_*_diff.txt`).

**The tabulation's assertions** (`out/qsi_tabulation.txt`): equal key sets, columns and tolerances; the four counts
present as ints exactly in R28's Levin records; the group regions summing to `WKB_Levin_num_regions` and
`total_converged` the AND of the group flags, in every row of both stores; `DEFAULT_LEVIN_MAX_DEPTH` 20 in both trees.

**The replay and captures** (`out/replay_check.txt`, `out/capture_check.txt`, `out/replay_store_check.txt`): 470 / 470
records; 8 / 8 captures (S2's two at p against R27; three q-r at p against R27 and at g against R28); Ray never
initialised; R27 and R28 unchanged; both worktrees clean, nothing newer than the markers.

**The q-r references** (`out/qr_scores.txt`): the replays reproduce their captures; the two trees' integrands
bit-identical at 1,001 points with equal x_span and kw; θ′ and θ exactly 0 at every sample.

**S2's references** (run `levin-phase-floor-07-p1-references-20261108T103132`, `done`, exit 0, 2 / 2; row 1 191 s, row
2 425 s): four readings each, Route B, one non-monotone segment direct (QAWO) and three (FCC), as the tool's rule says
beside a stationary point; the readings agree within the tolerance by four orders of magnitude.

**The readers** (`out/readers_check.txt`): six of six exit 0; R28 identical before and after; 0 Tracebacks; six
`!! WARNING … shards (=4) … (=10)`, as 06m's.

**The lock** (`state/lock.log`), ten commands, each its own, appended by `>>`: ray-processes (empty); acquire;
set-pid 83008; ray-processes (empty); set-pid 83825; ray-processes (empty); ray-processes (empty); release; status
(free, no Ray processes); ray-processes (empty). Each exited 0.

**Final** (`out/final_check.txt`): R26, R27 and R28 equal to their earlier hashes; 65 runs, all finished;
`~/.raypoolkit` empty; the main checkout and both worktrees clean and unmoved, nothing written in either; no
`__pycache__` in the records.

### Observations not acted on

1. **The location on an identically zero rate** (issue `[07-location-on-an-identically-zero-rate]`): the 26 q-r
   calls with q = r locate every scan sample; they cost about 20 times a G+q+r call and double the rehearsal's
   regions; R28 stays honest where referenced.
2. **17 rows gained convergence, none lost;** the one row still unconverged among the touched (3.162e5, 3.162e5,
   3.162e5, z 573.11) has one unconverged call (R27: two), at depth 20 in both.
3. **R27 is wrong where 07 says it should be:** 16 rows move by more than both stores' errors (up to 7.4e4); on the
   referenced ones (S1's w rows, S2's rows) R28 is honest and R27 is 4.1e4–5.6e5 of its abserr out (S2) — P1's
   defect reproduced at the parent's code.
4. **No minimum was located and no leaf guarded** anywhere in the rehearsal (`guarded_regions` 0 throughout, as §1.5
   expects with `theta_deriv` supplied).
5. **The analytic blocks locate a root in the 2pt5 J4 and Y4 blocks** of 200 rows (k = q = r), at a fixed η per k;
   `analytic_rad` moved by at most 1.5e-5 of its abserr.
6. **QAGS on the worst q-r call** is 4.9 times its own estimate from the Gauss–Legendre sequence; the reference rests
   on GL there (Predictions and verdicts).
7. **The tool writes its JSON to stdout too,** so each `p1_row*.log` duplicates its `.json` (kept on the Mac Studio).

### State handed on

- **R27** `levin-phase-floor-07-r27-parent-20261108T000634` and **R28** `levin-phase-floor-07-r28-guard-20261108T001916`:
  both `done`, `1b332ffa…`. **`p1-references`** `levin-phase-floor-07-p1-references-20261108T103132`: `done`, exit 0.
- **Their stores are kept** (`state/stores_sha256_final.txt`): `var/datastores/levin-phase-floor-07-r27*`, primary
  `d6fd3023…`; `…-r28*`, primary `00468d55…`. R26 (06m's, read only) unchanged.
- **The lock is released;** `~/.raypoolkit` is empty and `ray-processes` lists nothing. Nothing of this agent runs.
- **The trees** are clean and unmoved; nothing was written in either worktree.
- **Held, for the user:** the zero-rate case, before the census (the orchestrator's hold). Nothing of prompt §6 was done.
- **Kept on the Mac Studio, not for copying back:** see `KEPT_ON_MAC_STUDIO.txt` (the diff JSONs, `extract/`,
  `state/capture/`, `state/inputs/`, the tool's reading JSONs).

## §5 resumed under §10 (Rule A) — rehearsal R29 on the Mac Studio

**Prompt:** [`07-stationary-point-guard.md`](../07-stationary-point-guard.md) §10 (Part B after Rule A) and §5, under the
orchestrator's resumed dispatch (`e182bc7a`), with the user's word at R29's stop. The code measured is `adffa21b`
("Skip locating a phase whose variation is below the C2 gate"), Rule A on 07's `ad732887`. R29 takes R28's place in §5;
R28 (`74600350`) is kept as the record of the zero-rate case.

**Host:** mps026298. **Trees:** R29, the comparisons, the replay, S1, S2 and the readers in the worktree
`SecondaryGWKit-lpf07-e182bc7` (detached at `e182bc7a`; `levin_quadrature.py` `4e5da371…`; `QuadSourceIntegral.py`,
the launcher, `main.py` and `store_value_diff.py` unchanged from R28's tree); the four-block replays also at
`SecondaryGWKit-lpf07-91472db` (R27's) and `SecondaryGWKit-lpf07-7460035` (R28's), read-only replays. `74600350..e182bc7a`
outside `docs/` and `prompts/` touches `levin_quadrature.py`, the two test modules and `CLAUDE.md` only.

**Model:** Claude Opus 5.5, the rehearsal subagent, in four turns (7–10 of this agent): the gate and R29's launch; R29's
checks and the comparisons (a stop on 4 records); the user's word, the replay and the four-block captures; the
references, S1, S2, the readers, the release and these records.

**Date:** 2026-11-08, 12:33–13:35 UTC. **Commit:** none; the laptop commits these records.

**Result:** COMPLETE (planned stop 1′). R29 `done`, `1b332ffa…`. **S1 and S2 hold on R29.** Against R28, 90 Levin
records changed, every one a call R28 located and R29's Rule A skipped: 86 equal R27 bit for bit (26 `q-r`, 60
analytic `2pt5` J4/Y4 blocks), and 4 analytic blocks were caught by the nodal guard instead (a stop, ruled findings by
the user). **On those 4 blocks all three stores are honest** against references with stated errors 1,990× or more below
every abserr. Against R27, 692 untouched rows are all bit-identical. Regions are back to R27's level (114,460 against
114,826; R28 227,883), wall time 9.6 min (R27 9, R28 16.5).

### What shipped

No code. Records under `var/levin-phase-floor-07/rehearsal/r29/` (committed by the laptop as
`docs/levin-phase-floor/07/rehearsal/r29/`): `log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`;
- **`scripts/`:** `rec.sh` (b5279dd9, the stop-1 copy with the record path changed; `out/rec_sh.diff`);
  `classify_non_qsi.py` (26690e61, 05's, unchanged); `tabulate_qsi_guard.py` (a1eaba31, the stop-1 script unchanged)
  and `tabulate_qsi_guard_r29.py` (47e18cbe; `out/tabulate_qsi_guard_r29.diff`); `compare_r29_r28.py` (f6ba1f2e),
  `stop_detail.py` (93517fee), `block_reference.py` (fc2c20df), `run_block_reference.sh` (0a5372da), `block_score.py`
  (e62946d1), `replay_summary_r29.py` (49097a96), `s2_score_r29.py` (e7430b57): new; `s1_score_r29.py` (6f60c8d1; the
  stop-1 `s1_score.py` by sed, `out/s1_score_r29.diff`); `run_readers_worktree.sh` (ec96cf09, unchanged);
  `capture/`: the stop-1 harness (hashes equal, `state/capture_harness_stop1_sha256.txt`) with `prov.py` (1ec5f339)
  and `worker.py` (6bcb4a2d) changed where marked "prompt 07 r29" (`out/capture_harness_r29_diff.txt`), and the new
  `replay07_r29.py` (192c9c42) and `run_r29_pipeline.sh` (984105f7).
- **`out/`:** `commands.log`; the gate and launch/end checks; `r29_status.json`; both diff JSONs and their classifier
  outputs; `qsi_tabulation_r27_r29.*`, `qsi_rows_r27_r29.csv`; `r29_vs_r28.*`, `r29_vs_r28_records*.csv`,
  `stop_detail.txt`; the replay (`replay_rows.txt`, `r29_pipeline.log`, stage outputs, `replay_calls.csv`,
  `replay_calls.checks.json`, `replay_summary.txt`, `blocks.json`); `block_reference/`, `block_reference_run.txt`,
  `block_scores.*`; `s1_scores.*`, `s2_scores.*`; `readers.txt`, `readers_check.txt`; `final_check.txt`,
  `final_registry_list.txt`.
- **`state/`:** `lock.log`, the launch lines and markers, every store-hash file, the replay's rows and specs.
- **Not for copying back:** see `KEPT_ON_MAC_STUDIO.txt` (the diff JSONs, `extract/`, `state/capture/`, `state/inputs/`).
- **The store is kept:** `var/datastores/levin-phase-floor-07-r29*`.

### Predictions and verdicts

| Rule (as the resumed dispatch and prompt §5/§10 write it) | Scored on | Verdict |
|---|---|---|
| R29 with R28's command line, slug `r29-guard-rule-a`; fingerprint `1b332ffa…` | `done`, exit 0, `1b332ffa…`, `classes` equal R28's/R27's/R26's | **holds** |
| R29 against R28, outside `QuadSourceIntegral`: every value table identical | 360,462 rows identical; 0 unclassified | **holds** |
| R29 against R28, inside: a record may differ only if R28 located and R29 did not; such a record equals R27 (value, abserr, converged, phase_limited, regions) with R29's four counts 0; every other record bit-identical to R28 | 13,087 of 13,177 bit-identical; 90 changed, all R28-located / R29-skipped; **86 equal R27**; **4 do not** (guarded) | **fails as written on 4 records** — the user ruled them findings; the laptop corrects the rule (§10's guard clause) |
| Rows that differ hold such a record; tabulated | 54 rows, each holding a changed record; `total`, `total_abserr` equal R27's in all 54 | **holds** |
| R29 against R27 (§5 item 4): untouched rows bit-identical; the rest tabulated; `total_converged` changes listed | 692 / 692; 172 tabulated; 17 gained, 0 lost | **holds** |
| The four-block check (the user's addition): each store's value against a reference | 12 of 12 store-blocks honest outright (largest 6.2e-4 of abserr) | **reported** |
| S1 on R29 | 12 / 12 located at z 10211.3179 ± 9e-6, honest, largest 0.218 | **holds** |
| S2 on R29 (the standing `p1-references` readings) | located within one spacing on both; 0.020 and 0.263 | **holds** |
| The readers on R29, exit 0, store byte-identical | 6 / 6; identical | **holds** |
| The lock released, `ray-processes` empty | released; free; empty | **holds** |

**The four blocks** (`out/block_scores.txt`; float.hex in `verification.md`): the stated error of each reference is
set by QAGS's distance from GL14 (QAGS within its own estimate), and GL's sequence agrees to 0–2e-25. Ratios
|v − ref|/abserr: J4 at (3.162e5)³ R27 7.4e-5, R28 4.4e-5, R29 1.5e-6; Y4 there 1.6e-5, 5.8e-4, 3.7e-4; J4 at (1e6)³
1.0e-4, 2.1e-4, 3.1e-6; Y4 there 0, 6.2e-4, 5.1e-5. V = 15.113 and 15.896 (below SIX_PI); the gate's span estimate on
the root leaf = 19.160 and 20.196 (above it). R29's J4 declarations are 19× and 11× R27's; its Y4 declarations smaller.

### Deviations from the prompt

(Numbered on from the stop-1 rehearsal's 15.)

16. **Rule A's key is not in the store**, so V_i comes from the replay (`guard_07` records `location_skipped_phases`).
    IMPLEMENTATION CHOICE.
17. **The gate check printed the main checkout's `status --short --ignored` listing** (read-only; `status --short` is
    empty). UNINTENDED DRIFT.
18. **A mistyped parent pid in `pgrep -P`** when finding R29's driver (read-only). UNINTENDED DRIFT.
19. **`tabulate_qsi_guard.py` adapted for R29 by two sed passes** (R28 → R29, then the `_28` key suffixes); and
    `s1_score_r29.py` by one, which also changed the word in its own new comment ("R29 -> R29") and left one key,
    `store_value28`, named for R28 (it holds R29's stored value). Cosmetic. IMPLEMENTATION CHOICE.
20. **The record-by-record comparison is a new script** (`compare_r29_r28.py`), which also reports `simple_regions` and
    `evaluations`. IMPLEMENTATION CHOICE.
21. **The replay also covers the 90 records that changed against R28** (for V_i): 224 rows, 470 records (384 with a
    nonzero count, the dispatch's "about 444" being the addendum's oversight). IMPLEMENTATION CHOICE.
22. **The four-block check ran as stages of the replay pipeline** (pass 1 and pass 2 through the harness at each of the
    three trees), then each capture replayed in its own process at its tree. IMPLEMENTATION CHOICE.
23. **The four-block references:** composite 20-point Gauss–Legendre on 2^m x-panels each side of the root (located by
    brentq on θ′ sampled at 40,001 points; one root each), m = 6–14, with QAGS split at the root as a second rule, on
    R29's tree's captures; V by the trapezoidal rule on the driver's 4,097-point scan (equal to R29's
    `location_skipped_phases`, bit for bit); the gate's estimate as `build_Levin_data` forms it on the root leaf. The
    user's terms left the method to the agent. IMPLEMENTATION CHOICE.
24. **A stop was taken on the 4 guarded records** (the dispatch's rule as written), reported before anything else, and
    resumed on the user's word. STRUCTURALLY REQUIRED (the rule as written).

### Verification performed

- **The gate** (`out/00-gate.txt`): the four trees at their SHAs, clean; 65 runs, all finished; lock free.
- **R29** (`out/r29_launch_check.txt`, `out/r29_end_check.txt`): launched 12:34:28Z (sh 14524, driver 14528), `done`
  12:44:05Z, manifest at `e182bc7a`, clean, launcher `b0135218…`; no Traceback or WARNING.
- **The comparisons** from the new worktree, `tools.store_value_diff` (16.5–16.7 s each); R27, R28, R29 byte-identical
  before and after (`state/store_sha256_{before,after}_diff.txt`), and after the replays (`state/after_replay_body.txt`).
- **The replay and captures** (`out/replay_check.txt`, `out/blockcheck.txt`): 470 / 470 records; the 12 block calls
  reproduce their stores in pass 1 and pass 2, each capture's replay reproduces its value and abserr; Ray never
  initialised (224 + 6 + 6 workers, 12 block processes).
- **The readers** (`out/readers_check.txt`): 6 / 6 exit 0; R29 identical before and after; 0 Tracebacks.
- **The lock** (`state/lock.log`), eight commands each its own: ray-processes, acquire, set-pid 14528, ray-processes,
  ray-processes, release, status (free, no Ray processes), ray-processes. Each exited 0.
- **Final** (`out/final_check.txt`): R27, R28, R29 and R26 unchanged; 66 runs, all finished; `~/.raypoolkit` empty; the
  three worktrees and the main checkout clean, nothing written in any; the stop-1 records outside `r29/` untouched.

### Observations not acted on

1. **The gate's root-leaf estimate exceeds V on the four blocks** (19.16, 20.20 against 15.11, 15.90), which §10
   anticipates; the guard catches the root, at the cost of a wider declaration on J4 (`issues.md`).
2. **Rule A skips phases widely** — 423 replayed calls across every group and analytic label, V 0–18.70 — but only the
   90 records that R28 located differ from R28.
3. **R29's values on S1's and S2's calls equal R28's bit for bit;** the G+q+r roots are unchanged by Rule A.

### State handed on

- **R29** `levin-phase-floor-07-r29-guard-rule-a-20261108T123429`, `done`, `1b332ffa…`; its store kept
  (`state/stores_sha256_final.txt`). R27, R28 (read-only record) and R26 unchanged.
- **The lock is released;** nothing of this agent runs. All trees clean and unmoved.
- **Planned stop 1′:** the census agent is not dispatched; nothing of prompt §6 was done.

## levin-phase-floor 07 §6 — the census at 07's code (Mac Studio), the census agent

**Subject:** prompt 07 §6, as amended by §10 (V3 sets aside a phase Rule A skipped), on the Mac Studio (`mps026298`),
2026-11-08 to 2026-11-10, by Part B's census agent. Gate V0, the census, V1–V5, V4's references, the descriptive items,
and a cross-check beyond §6 that the user asked for at stop 4. Ray-free; no store built.
- **The tree g:** the worktree `SecondaryGWKit-lpf07-e182bc7` at Part B's resumed dispatch SHA `e182bc7a`, whose code is
  `adffa21b`'s (07's `ad732887` with Rule A): `AdaptiveLevin/levin_quadrature.py` `4e5da371…`,
  `ComputeTargets/QuadSourceIntegral.py` `33cb161e…`. No variant was installed.
- **The parent p:** the worktree `SecondaryGWKit-lpf07-91472db` at `91472dbe` (06m's code, `beaf0e23…`), for V0's K0 run.
- **The "before":** 06m's census, `levin-phase-floor-06m-census-20261106T102153` (checkpoint `ea5bc3ab…`).
- **The store:** 06's certified scratch copy of A3 v2, read only.

**Commit:** none here. The orchestrator commits these records from the laptop as `docs/levin-phase-floor/07/census/`.

**Result:**
- **Gate V0 holds**, all three clauses.
- **The census** `levin-phase-floor-07-census-20261108T171137` is `done`: 18,560 of 18,560 units, 0 in error, 0 over budget,
  wall 22,708 s (06m: 18,465 s).
- **V1 holds:** all 284,514 calls that locate nothing and guard nothing equal 06m's bit for bit, and all 16,104 units
  made only of such calls equal 06m's whole, `total` and `total_abserr` included.
- **V2 holds:** no returned Levin leaf holds a sign change of θ′ (182,066 calls scanned).
- **V3 holds:** 0 violations in 170,582 scored calls; 11,484 set aside under §10.
- **V4 fails:** 28 of 33 scored calls are honest. z00, z12 and Z3's seven are honest, and 19 of the 24. Not honest:
  v08, v09, v12, v17 and v22, at 2.2–3.2× their abserr.
- **V5 fails:** 23 calls that locate a point hold leaves at `depth_max`.
- **The store and 06m's census are unchanged**, before against after.

### What shipped

No code. The records are in `var/levin-phase-floor-07/census/`:
- **The records:** `log.md` (this file), `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`, and
  `state/predictions.md` (SHA-256 `92e22911…`, mtime 2026-11-08T14:10:10Z, not rewritten).
- **`scripts/`:** 06m's census harness, copied from g's `docs/levin-phase-floor/06m/census/scripts/`, changed where marked
  "prompt 07". The diffs are `out/scripts_diff_at_copy.txt` (empty), `out/scripts_diff_before_k0.txt`,
  `out/scripts_diff_at_stop2.txt` and `out/scripts_diff_at_stop4.txt`.
  - **Changed:**
    - `prov.py` and `refs_common.py`: the trees and the "before";
    - `worker.py`: §1.5's diagnostics, Rule A's key, the scan;
    - `run_k0.py`: `--tree`;
    - `run_census.py`: 06a's over-budget pass;
    - `run_06j.py`: the references' run backstop;
    - `x1_capture.py` and `check_captures.py`: V4's captures, against 07's census.
  - **New:**
    - `compare_07.py`, `hash_runs_07.py`, `index_06m_07.py`, `v0a_imports_07.py`, `select_v4_07.py`;
    - `census_projection_07.py`, `launch_07.py`, `launch_census_07.py`, `stop3_guarded_07.py`;
    - `score_07.py`, `summarise_07.py`, `extra_descriptive_07.py`, `v4_capture_selection_07.py`, `slice_v4_07.py`;
    - `refs_projection_07.py`, `score_v4_07.py`;
    - the cross-check's `xcheck_07.py`, `xcheck_capture_07.py`, `xcheck_projection_07.py` and `xcheck_report_07.py`;
    - `kept_list_07.py`.
- **`out/`:** every command and its output (`out/cmd/`, `out/commands.log`), with these results:
  - V0: `out/k0/compare_v0b.json`, `out/v0c/compare_det.json`, `out/v0/v0a_imports.txt`, `out/k0/compare_k0_g_calls.json`;
  - stop 3: `out/stop3/compare_census_calls.json`, `out/stop3/guard_only.json`;
  - the scores: `out/scores/summary.json`, `out/scores/extra_descriptive.json`;
  - V4: `out/v4/` (captures, checks, inputs, projection, readings, `v4_scores.json`, `report.md`, `scripts_run.txt`);
  - the cross-check: `out/xcheck/`;
  - before and after: `out/runs_hashed_before.txt`, `out/runs_hashed_after.txt`, `out/census_projection.json`.
- **Three registered runs, all `done`:**
  - `levin-phase-floor-07-k0-20261108T135240`: 7 of 7 commands;
  - `levin-phase-floor-07-census-20261108T171137`: 18,560 of 18,560, exit 0, checkpoint `03dfe786…`;
  - `levin-phase-floor-07-references-20261109T005328`: 74 of 74 readings, exit 0, checkpoint `40a9d871…`.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`): the census's checkpoint and region records in full (08 compares
  against them), the references and k0 run directories, and every pickle and every file over 2 MB under `census/`.

### Predictions and verdicts

`state/predictions.md` holds prompt §6's predictions (lines 765–803) and §10's last bullet (lines 1017–1018), verbatim,
then V4's selection and how each rule is read. It was written after V0 and before the census.

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **V0 (a)** the parent's code is 06m's | **holds** | `git diff --stat 48cf87c0 91472dbe` (outside docs and prompts) touches `AdaptiveLevin/tests/test_levin_leaf_values.py` and `CLAUDE.md`; none of the 102 files of p the K0 workers loaded is among them |
| **V0 (b)** K0 reproduces 06m at the parent | **holds** | 100 of 100 units equal: 2,110 calls (566 group, 1,544 analytic), 2,110 region records, 32,518 returned leaves; 0 differing |
| **V0 (c)** the new code is deterministic | **holds** | `g-1e-8-1-2174` (z12's row, which locates one root at g) run twice in two fresh workers: identical on every key but timing and provenance, 07's new keys included |
| **V1** nothing else moves | **holds** at both rtols | At each rtol: 9,280 of 9,280 units; 142,257 calls that locate and guard nothing, all equal; 8,052 units made only of such calls, `total` and `total_abserr` equal in hex |
| **V2** no Levin leaf across a stationary point | **holds** | 182,066 calls scanned (37,325 group at each rtol; 107,416 analytic at 1e-7), 0 scan errors. Listed leaves: 0 Levin. The listed leaves are window leaves (526 group at each rtol, 1,514 analytic) and 168 ordinary Clenshaw–Curtis leaves of analytic calls whose phase Rule A skipped |
| **V3** the location finds what the scan finds (§10's set-aside) | **holds** | 170,582 scored calls; 3,584 with a sign change or a located root; 0 violations; no point of kind `end`. Set aside: 11,484 scanned calls with a phase Rule A skipped (V from 0 to 18.81): 3,260 group calls at each rtol, none with a sign change; 4,964 analytic calls at 1e-7, of which 168 show a scan sign change (J4 and Y4 of 84 rows; 14 of them guarded, O9's calls). In all, counting the calls not scanned, 16,448 calls have a skipped phase |
| **V4** honest where referenced | **fails** | 33 scored. Honest: z00 (0.718), z12 (0.718), Z3's seven (at most 0.0021), and 19 of the 24. Not honest: v08 3.16, v09 2.21, v12 3.08, v17 2.38, v22 2.53, |v_g − ref| / abserr_g. Every call's readings agree, the widest spread being 0.0039 of the tolerance. Basis: 13 calls on four readings, 11 on FCC r1 and r2 (the ruling at stop 4) |
| **V5** the cap is not reached where a point was handled | **fails** | 23 calls (10 at 1e-7, 13 at 1e-8), each a `G+q+r` call #1 that locates its root, hold 2–54 leaves at `depth_max` 20 (`g-*-0-38999` holds 54). Three of them are converged (`g-1e-7-1-34766`, `g-1e-8-1-34766`, `g-1e-8-0-14167`); 20 are not. 0 units over budget |

**The five V4 failures.**
- Each is a converged `G+q+r` call of class (c) in the cross-check.
- Each call's maximum region depth at g is 6, 7, 9, 11 or 14 (v17, v12, v08, v09, v22).
- Each one's window was stopped by §1.3's `iroff1` rule while 4–13 of its leaves were still falling; the windows'
  `max_depth_at_stop` was 5–6.
- 06m was 497–30,400× its abserr from the reference on the same calls.
- This is the case O8 and the note's B.5 anticipated (windows stopped while leaves still fell). On these five calls the
  value is not within its declared error.

**The comparison** with 06m is `scripts/compare_07.py`: 06l's `compare_k0.diff`, applied unchanged.
- **Set aside:** 06m's exclusions (timing and provenance) and 07's new keys, each named and why:
  - `guard_07`, `window_07`, `guarded_07`, `guard_at_depth_max_07`: §1.5's diagnostics;
  - `scan_07`, `scan_07_error`, and `scan_07_s` (timing): §6's scan;
  - QuadSourceIntegral's four counts (`stationary_points`, `window_pieces`, `guarded_regions`, `guard_at_depth_max`),
    which reach the pickle's `result_full…groups[]` at g.
- No 07 key name occurs in any 06m record.

### Descriptive (not scored)

- **Depth of the rows:**

  | | rows at depth 20, g / 06m | rows at depth ≥ 18, g / 06m |
  |---|---|---|
  | 1e-7 | 77 / 192 | 283 / 384 |
  | 1e-8 | 83 / 213 | 297 / 411 |

  Of the 720 rows, 24 are at depth 20 at g against 139 at 06m (1e-7), and 27 against 157 (1e-8).
- **Locations:** 4,940 calls locate, the same 2,470 at each rtol.
  - Group calls: 720 roots (one per row of the 720), and 2,404 minima in lone-`G` calls (160 calls with 6 minima and 76
    with 19, at each rtol), λ from 1.6e-7 to 2.3e-6.
  - Analytic calls: 1,514 roots per rtol (757 J4, 757 Y4).
  - No point at an end of a span; no minimum left unlocated.
  - `num_guarded_regions`: 28 in all, O9's calls, one region each; `num_guard_at_depth_max` 0.
- **Windows:** 2 to 32 pieces per call; widths in x from 4.7e-5 to 14.6 (median 2.58).
  - θ′²/|θ″| at the own window edges: group roots 12.52–12.62 (median 12.566 = 2Φ_w); analytic roots 1.18–8.09;
    minima up to 2π.
- **§1.3's firings (B.5):**
  - 1e-7: 1,157 of 2,470 windows stopped; 1,153 with a leaf still falling; 1,146 of those calls converged.
  - 1e-8: 1,195 of 2,470 stopped; 1,192 falling; 1,183 of those calls converged.
  - `leaves_falling_at_stop` 0–29. Every stopped window is listed in `out/scores/summary.json`.
- **Convergence:**
  - Located calls: 262 went from unconverged (06m) to converged, and 15 the other way.
  - Unconverged group calls in the 720: 64 at g (62 within their request, each with an untrusted leaf at `depth_max`;
    2 over, `node_limited`, at most 1.0038×; largest share of a row's `total_abserr` 18.7 %) against 311 at 06m
    (largest share 53.0 %).
  - Outside the 720: 114, identical to 06m's.
  - No analytic call is unconverged.
- **Totals in the 720:** 643 of 1,440 row totals moved by more than 06m's `total_abserr`; the largest move is 2.6e5 of
  it (`g-1e-8-0-2065`); relative moves at most 0.86 %.
- **Cost:**

  | | g | 06m |
  |---|---|---|
  | census wall | 22,708 s | 18,465 s |
  | unit walls, 1e-7 / 1e-8 | 24,203 / 27,479 s | 23,970 / 27,070 s |
  | group-call regions, 1e-7 / 1e-8 | 952,275 / 1,065,288 | 939,882 / 1,053,755 |

  - The median unit-wall ratio is 1.000.
  - 254 tiny units (under 0.3 s) take more than twice 06m's time; 0 units are over budget.
- **The stationary-point calls before:** 06m is honest on 17 of the 33 V4 calls. On the other 16 it is 356–5.2e5× its
  abserr from the reference (z12 1,681×).
- **V4's references:** 74 readings, all usable.
  - Run wall about 25.6 h, against 89,230 s projected.
  - The longest readings were FCC r2 on v18 (20,509 s) and on v13 (19,373 s).

#### The cross-check (beyond §6, interim; the user's request at stop 4)

The rule is `state/xcheck_rule_20261109.md` (SHA-256 `e06a1058…`), fixed before any result. The results are
`out/xcheck/xcheck.json` and `out/xcheck/report.md`. Both checkpoints were unchanged before and after.

- **(A) The population and the cross-rtol ratios:**
  - 4,968 calls that locate or guard, i.e. 2,484 pairs; all paired.
  - r_g ≤ 1 in 2,482 pairs and r_06m ≤ 1 in 2,432.
  - Two pairs have r_g > 1: `3-14501` (1.39) and `3-35006` (1.08), both `G+q+r`, band low, k 1e5, class (e).
- **(B) The disagreement list**, |v_g − v_06m| > max(abserr): 843 calls, all `G+q+r`.
  - Classes: (a) 31, (b) 0, (c) 786, (d) 0, (e) 26.
  - 3-38439's `G+q+r` is class (c), the known case.
- **The cost of referencing class (c):** one capture per row; all 396 reproduce the census bit for bit.
  - All four readings: 2.6e7 s; FCC only: 4.47e6 s.
  - 145 rows project at 30 min or less by FCC alone.
- **These are interim results.** The user is to choose a sample of references from them.

### Deviations from the prompt

1. **The harness changes,** each within §6's list: the tree and its paths; §1.5's diagnostics per call and per returned
   leaf, with Rule A's `location_skipped_phases`; the scan on every group call at both rtols and every analytic call at
   1e-7; the comparison with the new keys set aside. STRUCTURALLY REQUIRED.
2. **K0 at p runs as `p-<rtol>-…`.** `compare_07.py whole` compares its `unit` and `tree` under 06m's names (`g-…`, `g`)
   rather than excluding them. IMPLEMENTATION CHOICE.
3. **06h's analytic counterfactual is computed at p as at g,** so that K0 reproduces 06m's records. STRUCTURALLY REQUIRED.
4. **My first `run_k0.py` edit broke the spec line** (a comment swallowed `"units": units,`). I found it by reading and
   fixed it before any run. UNINTENDED DRIFT (a failed attempt).
5. **Development self-tests outside the scored set** (`out/cmd/10`–`13`, `21`; `state/smoke/`). IMPLEMENTATION CHOICE.
6. **K0 at g, its call-by-call comparison and V0 (c) ran inside the k0 run,** so that the census projection had unit
   timings at g. IMPLEMENTATION CHOICE.
7. **Hand entries in `out/commands.log` first had estimated times,** later corrected. UNINTENDED DRIFT (cosmetic).
8. **06a's over-budget pass lives in `run_census.py`,** in the census's own run. It was added after K0, so K0's records
   name the earlier `run_census.py`. IMPLEMENTATION CHOICE.
9. **V4 is stratified over all eight k,** not "all six that carry them": all eight carry stationary-point calls. The
   user ruled at stop 2: "Please keep all 8 `k` and launch the census". IMPLEMENTATION CHOICE.
10. **`state/predictions.md` gained a part 4,** how each rule is read, beyond the brief's three parts. IMPLEMENTATION
    CHOICE.
11. **`census_projection_07.py`'s first version counted the producer rebuild** in K0's ratio. It was superseded, and its
    output is kept in `state/superseded/`. UNINTENDED DRIFT (a failed attempt).
12. **The predictions check wrote and removed an unused `/tmp/x765.txt`.** UNINTENDED DRIFT (cosmetic).
13. **The census held the shared rehearsal lock,** on the orchestrator's direction, though the brief said no lock was
    needed. IMPLEMENTATION CHOICE.
14. **The lock commands and the hashing are hand entries in `out/commands.log`,** each lock command a shell command of its
    own. IMPLEMENTATION CHOICE.
15. **`stop3_guarded_07.py`,** a new reader for stop 3. IMPLEMENTATION CHOICE.
16. **`score_07.py`'s first full run crashed after 17 s** (an unpacking bug). Fixed, checked on five units, then re-run.
    UNINTENDED DRIFT (a failed attempt).
17. **`summarise_07.py`'s first run crashed** on a `None` row depth in a sort. Fixed and re-run. UNINTENDED DRIFT (a failed
    attempt).
18. **V4's capture scripts:** `x1_capture.py` (its title) and `check_captures.py` (07's census) were edited, and
    `refs_common.py` gained `CENSUS07`. STRUCTURALLY REQUIRED (§6's "V4's references").
19. **A 25th capture, z12's call,** made for the projection only. IMPLEMENTATION CHOICE.
20. **`out/v4/` was made after `x1_capture.py` had started,** but before it wrote there. UNINTENDED DRIFT (caught before
    it failed).
21. **`refs_projection_07.py`'s first version failed:** 06g's captures are no longer on this host. Rewritten to take the
    readings' own segment tables; the features were checked equal on z12's and P1's rows. UNINTENDED DRIFT (a failed
    attempt).
22. **That first attempt was started with a plain `&`;** it failed after 3 s. UNINTENDED DRIFT (cosmetic).
23. **The projection model's choices:** γ is the largest calibration value; the margin is ×1.5; the laptop's FCC times
    are scaled to this host by r55's readings. IMPLEMENTATION CHOICE.
24. **New readers and launchers for scoring and V4.** IMPLEMENTATION CHOICE.
25. **V3 covers the scanned calls only,** as §6's scan does: analytic calls at 1e-8 locate (1,514) but are not scanned.
    IMPLEMENTATION CHOICE (recorded as read).
26. **The reading subset.** FCC r1 and r2 on all 24; QAWO r1 and r2 on the 13 whose QAWO readings each project at 30
    minutes or less (`state/v4_reading_list_20261109.json`, `ce594095…`); 74 readings in place of 96. The user: "Take
    your recommended option of FCC at both resolutions plus QAWO where it is projected to be 30 minutes or less. It
    costs only a few hours for an independent check of many calls. Calls without QAWO should be scored on whatever
    references we do have, either one or two FCC readings." STRUCTURALLY REQUIRED (by the ruling).
27. **V4's scoring rule, as the ruling states it.** Four readings: as written. Without QAWO: on the FCC readings it has,
    FCC r2 the reference and the two in agreement within the same tolerance, or one reading with no agreement test.
    Each call's basis is reported; readings that disagree are reported, not scored. STRUCTURALLY REQUIRED (by the
    ruling).
28. **`run_06j.py` gained the run backstop:** it reports past 30 minutes and stops only at a projected finish of 48 h. It
    was tested on a scratch copy with a stub registry. IMPLEMENTATION CHOICE.
29. **`launch_07.py`'s `references` plan was written for the ruled subset.** IMPLEMENTATION CHOICE.
30. **The cross-check, beyond §6.** The user: "Yes, please do the cross-check on `rtol`s and the list of calls where
    `06m` and `07` differ by more than the stated error. We can decide on a sample of references to sort of what these
    difference really show, later." IMPLEMENTATION CHOICE.
31. **Class (e) added to the cross-check's four classes,** named in the rule before any result so that the classes are
    exhaustive. IMPLEMENTATION CHOICE.
32. **The cross-check's captures were grouped one worker per (rtol, k),** one at a time under `nice`, one row per
    class-(c) pair (the integrand is the same at both rtols). IMPLEMENTATION CHOICE.
33. **The cross-check's capture chain was started with `&` inside a background shell,** and awaited by an until-loop on
    its last command's record. UNINTENDED DRIFT.
34. **Development runs of the cross-check on two rows** (`state/dev/`). IMPLEMENTATION CHOICE.
35. **A reading-usability check in V4's scorer:** the tool's and the capture's SHA-256, Route B, Ray not initialised,
    every non-monotone segment integrated directly. All 74 readings pass. IMPLEMENTATION CHOICE.
36. **z00, z12 and Z3's seven are scored on 06g's own readings, two to four of them,** with this rule's tolerance and
    agreement test. IMPLEMENTATION CHOICE.
37. **Readings are compared as exact Fractions of their 30-digit strings** (06g used mpmath at 50 digits). IMPLEMENTATION
    CHOICE.
38. **Two figures in my stop-4 report were misstated;** the JSON was right throughout. The leaves at `depth_max` in V5's
    23 calls number 2–54, not 2–14 (`g-*-0-38999` holds 54). Three of the 23 are converged, not two (`g-1e-7-1-34766`,
    `g-1e-8-1-34766`, `g-1e-8-0-14167`). Corrected here. UNINTENDED DRIFT.
39. **My V4 report to the orchestrator said "3–14 regions deep"** of the five failures. Their maximum region depth at g
    is 6, 7, 9, 11 and 14, and their windows' `max_depth_at_stop` is 5–6. The phrase was in the report message only; it
    is in no file. Corrected here. UNINTENDED DRIFT.
40. **The first before-against-after comparison of 06m's run directory failed:** `sh` has no process substitution. Its
    record is kept in `state/superseded/`, and it was re-run with `bash` (`out/cmd/99b`). UNINTENDED DRIFT (a failed
    attempt).

### Verification performed

`verification.md` has the full account. In brief:
- **The trees, before and after:** g at `e182bc7a`, p at `91472dbe` and `-7460035` at `74600350`, all clean with
  `--ignored` and with no `__pycache__`. The main checkout is at `28ebaf07`, clean.
- **The module hashes:** g `4e5da371…` (equal to `git show` at `e182bc7a` and `adffa21b`), p `beaf0e23…`. Every census
  worker loaded g's module from the tree, with no variant, no foreign module and no Ray.
- **The scratch copy:** EQUAL to its certification and to 06e's after-hashes, before and after; before equals after.
- **06m's census:** 150 of 150 files equal its kept list, before and after.
- **Captures:** 25 + 396 captures reproduce 07's census bit for bit (values, regions, call records, harness).
- **The registry:** 69 runs, all finished. The lock is free.

### Observations not acted on

1. **Lone-`G` group calls locate minima of |θ′|** with λ about 1e-7–1e-6, 6 or 19 per call, 236 calls at each rtol.
   Every one is converged. Not looked into further.
2. **The class-(c) calls of the cross-check** (786 calls, 396 rows) are where both codes are self-consistent across
   rtols and disagree beyond their errors. All five V4 failures are among them, and so is 3-38439. Which code
   is right there is undecided without references.
3. **The references run took about 25.6 h** against a projection of 89,230 s (+3 %). The backstop reported and stopped
   nothing.

### State handed on

- **Nothing is running.** The lock is free; all three of my registered runs are `done`.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory (150 files, 4.8 GB), the references and k0 run directories;
  - every pickle and every file over 2 MB under `census/`: the K0, V0 (c), V4 and cross-check captures (`state/`,
    917 MB), `state/score_07_units.json`, the comparison outputs.
- **The census's checkpoint and region records stay in full; 08 compares against them.**
- **Read only and unchanged:** the scratch copy and 06m's census.
- **Issues:** proposed in `issues.md`, for the laptop and the user to decide.
