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

**§5's rules S0–S2, copied from the prompt unchanged. Not yet scored** (Part B, on the Mac Studio):

3. **S0 — R27 against 06m's R26.** R26 (`var/datastores/levin-phase-floor-06m-r26*`, kept on the
   Mac Studio) was built at 06m's code, the parent's. `tools.store_value_diff` on R26 and R27: every
   value table identical, every `QuadSourceIntegral` value field identical, and every other
   difference of the four volatile kinds. **Holds** if so. A difference is a finding for the user,
   reported before R28 is scored.
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
5. **S1 — V1's twelve.** w00–w11 are R24's rows, so R27's and R28's too. Each w call, the `G+q+r`
   group of its row's sub-interval 1, is read from R28's group record. **Rule:** each locates its
   point at $z \approx 10211$, and $|v_{28} - {\rm ref}| \le {\rm abserr}_{28}$ against 06g's
   reference (`scores.json`, `V1`). **Holds** if all twelve do. A violation is a finding for the
   user, not by itself a stop.
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

**§6's Gate V0 and predictions V1–V5, copied from the prompt unchanged. Not yet scored** (Part B):

**Gate V0, before the run:**
- **The parent's code is 06m's.** `git diff --stat 48cf87c0 <code commit's parent> -- . ':!docs'
  ':!prompts'` shows nothing the harness imports.
- **The harness, as changed, reproduces 06m's census at the parent,** bit for bit by 06m's
  comparison, on 06k's 100 K0 units (`docs/levin-phase-floor/06l/out/k0/units.json`). It is one
  registered run with slug `k0`.
- **The new code is deterministic.** Two runs of one $g$ unit on a stationary-point row are
  bit-identical.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into
`state/predictions.md` unchanged.
- **V1 — nothing else moves.**
  - **The rule:** every Levin call, group or analytic, in which $g$ locates nothing and guards
    nothing equals 06m's call bit for bit, in its record and its region record, by 06m's
    comparison with the new keys set aside. Every row all of whose calls are such equals 06m's row
    in `total` and `total_abserr`.
  - **Holds** if true at both rtols.
- **V2 — no Levin leaf is returned across a stationary point.**
  - **The rule:** the scan finds no returned **Levin** leaf holding a sign change of $\theta'$, in
    any group call at either rtol or any analytic call at `1e-7`.
  - **Holds** if none does. At 06a's code there were 677 such regions, at `1e-7` alone.
- **V3 — the location finds what the scan finds.**
  - It runs on the scan's grid, so the two should agree exactly.
  - **The rule:** every call in which the scan finds a sign change has a located `root` within one
    scan spacing of each, and every located `root` lies within one scan spacing of a sign change
    the scan found.
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
- **V5 — the cap is not reached where a point was handled.**
  - **The rule:** at both rtols, no leaf reaches `depth_max` in a call that locates a point or
    guards a leaf. An over-budget unit counts as false.
  - A failure is a finding for the user, not a fault. It reopens G2's L2 (Decisions, **G2 L2,
    decided**).

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
