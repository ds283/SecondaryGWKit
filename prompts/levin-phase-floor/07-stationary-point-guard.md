# Prompt 07 — the stationary-point guard: locate them first, integrate their windows directly

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-27, after 06a landed and G2 was answered, on the user's decision the same day.
Its charter is G2's Q4 (i) and (ii) (the board's Decisions, **06 P1**, **G2 Q4 (i)–(ii)** and
**07 scope**):
- *"We are supposed to deliver a function Levin integrator, and this is needed whether or not we
  plan to use the integrator as part of the 1-loop machinery. So we should fix this now,
  generically."*
- *"For now, 07 is just about the generic Levin integrator."*

The late-time range of the integrals is physics, and is a separate prompt
(`[06-time-integrals-must-reach-late-resonances]`).

**Where it runs:** in three parts, as 05, 05a and 06 did.
- **§1–§4, the code:** on the laptop, by a fresh-context subagent, in the main checkout at `HEAD`.
  **§2 is a planned stop:** the agent reports its two measurements to the user before the tests
  are written. One commit.
- **§5, rehearsals R15 and R16:** on the Mac Studio, after the code commit is pushed, under the
  lock. Both run from worktrees: R15 at the code commit's parent, R16 at the code commit.
- **§6, the re-measurement:** on the Mac Studio, Ray-free and with no lock. 06a's census is
  re-run at the code commit, on 06's certified scratch copy of A3 v2, against 06a's run as the
  "before".

The records of §5 and §6 are committed on the laptop afterwards, in 07's records commit.

**Closes, when §5 and §6 hold:** `[06-levin-accepts-a-region-across-a-stationary-point]`.

**Recommended model:** **Opus**.

**Amended 2026-10-28, before §2 runs (§9).** 06b, the round-off-limited exit's premise, lands
first. §2's reference and its $\omega$ are settled by the user. The rehearsals become R17 and R18.
§6's "before" is 06b's census.

**Read first:**
1. The board's §3 entry `[06-levin-accepts-a-region-across-a-stationary-point]`, with its 06a
   measurement. Then [`docs/levin-phase-floor/06/p1/report.md`](../../docs/levin-phase-floor/06/p1/report.md)
   in full: the mechanism; the reference method (its §5), whose window this prompt adopts; and the
   root diagnosis (its §6). Its scripts are in `docs/levin-phase-floor/06/p1/scripts/`, among them
   `reference.py`.
2. `docs/levin-phase-floor/VERIFICATION.md` §6.3 and §6a, then
   [`docs/levin-phase-floor/06a/issues.md`](../../docs/levin-phase-floor/06a/issues.md) X1 and
   X2. 06a's scan is `docs/levin-phase-floor/06a/scripts/spscan.py`.
3. The board's §2.3, Q4 (i) and (ii), and its Decisions for **06 P1**, **G2** and **07 scope**.
4. `AdaptiveLevin/levin_quadrature.py`:
   - the module docstring, in particular its paragraph on the total-variation gate;
   - `_levin_G0_G1`, `_roundoff_floor` and the constants above them;
   - `_adaptive_levin_subregion_impl` (the gate at `phase_span < SIX_PI`) and
     `_adaptive_levin_subregion_cc`;
   - `_adaptive_levin`, the generic bisection loop, written for a basis of any number of
     components: the seeding of `regions`, both acceptance branches (`resolved`,
     `phase_limited`, `depth_max`), and the aggregate `converged`;
   - the basis interface it uses: `build_Levin_data`, `eval_basis`, `theta_abserr_at` and
     `supports_complexified_solve`;
   - `_Basis_SinCos`, for how `theta`, `theta_mod_2pi` and `theta_deriv` are held;
   - `adaptive_levin_sincos`, the public facade that builds a `_Basis_SinCos` and calls
     `_adaptive_levin`: its docstring and result dict.
5. `AdaptiveLevin/tests/test_levin_quadrature.py`, in particular:
   - `test_stationary_phase_gate_uses_total_variation`,
     `test_fallback_region_bisects_on_missed_tolerance` and the two `test_roundoff_floor_G0_*`
     tests;
   - `_TwoIndependentSinCosBasis` and `test_generic_m_component_basis_two_decoupled_phases`: the
     generic path, driven through `_adaptive_levin` directly.
6. `ComputeTargets/phase_groups.py`, for `theta_deriv`: the closed-form sum each group hands to the
   integrator. Then `ComputeTargets/QuadSourceIntegral.py`, `phase_group_Levin_integral`, for what a
   group's record stores.
7. The other production caller, which this prompt must not move unless one of its phases has a
   stationary point: `LiouvilleGreen/three_bessel_integrals.py` (and, through it,
   `QuadSourceIntegral`'s analytic columns).

**Gate:**
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale on the laptop;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defect

`adaptive_levin_sincos` accepts a Levin region that contains an interior stationary point of its
phase, with an error estimate far too small (P1).
- **Why the estimate cannot see it.** The step-(4) residual compares a region's estimate with its
  two halves'. Near a stationary point the Levin antiderivative $p \sim f/(i\theta')$ is singular.
  The parent and the half that holds the point both lack the stationary-phase term, so the
  residual is blind to it. On 06's two R14 rows it was about $10^5$ smaller than the true error.
- **The relative test can still pass.** Another part of the region can dominate its denominator.
- **So can the round-off-limited exit.** At tighter `rtol` the eq. (151) floor, with a small but
  finite $G_0 = \min|\theta'|$, accepts the region as phase-limited instead. Tightening `rtol`
  cannot fix it.
- **The total-variation gate does not catch it.** That gate (C2) routes a region to the nested
  Clenshaw–Curtis fallback only when its phase varies by less than $6\pi$. Its regression test puts
  the stationary point at $x = 1/2$, the first bisection point, where bisection exposes it by
  construction.

**How far it reaches** (06a, at the current code, `1e-7`):
- 720 of A3 v2's 9,280 rows hold a call with an interior sign change of $\theta'$, all in group
  `G+q+r`.
- Every one of those calls accepts a region across the point, 623 of them as a single region of
  the whole call.
- Every A3 v2 row deeper than 11 is one of them.

The stationary point is the Kohri–Terada resonance, where the signal is expected to come from. A
stored `total_abserr` on such a group is not trustworthy.

### 0.2 The remedy, and the user's decisions on it

All of it lives in `AdaptiveLevin/levin_quadrature.py`, in the generic code: `_adaptive_levin`, the
bisection loop, which is written for a basis of any number of components. That covers every caller
of `adaptive_levin_sincos`, and any future basis.
- **The basis supplies what is basis-specific** through two new methods of the basis interface: its
  independent phases, and their rates.
  - `_Basis_SinCos` implements them from `theta` and `theta_deriv`. It has one phase.
  - A basis with several independent phases, such as the tests' `_TwoIndependentSinCosBasis`,
    reports each of them.
  - A basis that cannot supply rates reports none. Then there is no location, and the nodal guard
    of step 4 is the only protection.
- **A stationary point** is a zero of any of the basis's phase rates.
- **`adaptive_levin_sincos` stays a facade.** Only its docstring changes. The user, 2026-10-27: it
  *"is just a facade that sets up the basis functions used by the Levin step. It seems the wrong
  place for the stationary phase check, which is generic and does not depend on the basis."*

The steps, each generic:
1. **Locate** the points before integrating, from the basis's closed-form phase rates. For
   `_Basis_SinCos` that is the caller's `theta["theta_deriv"]`. Two kinds of point are located:
   - every sign change of $\theta'$, which is a stationary point;
   - every interior local minimum of $|\theta'|$ deep enough to act as a nearly coalescing pair.
     That covers the QCD and $e^+e^-$ caustics and any like them.
2. **Carve out a window** round each located point. The window is where the phase stays within
   $\Phi_w$ of its value at the point. It is integrated directly, in pieces, by a rule that
   resolves the phase.
3. **Hand the rest to Levin as it stands.** The ranges between windows go to the existing bisection
   loop, which bisects them as it needs to.
4. **Guard what the location missed.** A region whose $\theta'$ changes sign among its own Chebyshev
   nodes is never accepted by the Levin rule.

**The decisions** (board §2.3, (α)–(γ) and (ii); the user, 2026-10-27):
- **(α) The location uses `theta_deriv` and lives in the integrator's generic code, not in any
  caller and not in the facade.** *"From `theta_deriv` is the right way to locate stationary phase
  points, not by injecting physics knowledge. We want all callers to be protected."*
  - `QuadSourceIntegral` already supplies `theta_deriv` (`LEVIN_USE_THETA_DERIV = True`). Its only
    change is to record the integrator's new diagnostics in each group's record.
  - **Without `theta_deriv`** there is no location. The nodal guard is then the only protection,
    and the docstring says what it misses.
- **(β) An explicit window of phase half-width $\Phi_w$, not a bisection cascade.** The user's
  argument:
  - bisection to $6\pi$ builds a logarithmically dense cascade of regions round the point;
  - $6\pi$ is a criterion for a different question, whether a generic region is weakly
    oscillatory, and may be too wide or too narrow for the stationary region;
  - and the width wanted is known.

  **What fixes the width.** For a simple root, $\theta' \approx \theta''(x - x_s)$. At the window's
  edge the parameter $\theta'^2/|\theta''|$, which governs whether Levin's non-oscillatory solution
  exists, equals $2\Phi_w$ whatever $\theta''$ is. So one dimensionless number sets how well
  conditioned the Levin regions next to the window are. The width in $x$ follows from it.

  **Why not the cascade.**
  - It leaves $\theta'^2/\theta'' \approx 4\pi$ at its innermost Levin region, by accident rather
    than by design.
  - It spends about $\log_2(L\sqrt{\theta''/12\pi})$ levels of bisection depth, so the cap is
    reached first where $\theta''$ is largest. A window costs no depth.

  **What already validates it.** P1's reference used this construction (P1 report §5) and agreed to
  about $10^{-12}$ relative for $\Phi$ from $10^3$ to $3\times10^4$. $\Phi_w$ is measured in §2.
- **(γ) A minimum of $|\theta'|$ counts as a nearly coalescing pair by its Airy parameter.**
  - Near a local minimum $x_m$, $\theta' \approx a + \tfrac12\theta'''(x_m)(x - x_m)^2$. The pair
    matters when the phase accumulated across its slow zone is small. That phase is
    $\lambda = |\theta'(x_m)|^{3/2}/\sqrt{|\theta'''(x_m)|/2}$.
  - A minimum is located when $\lambda < \Lambda_c$, and then gets a window like a root.
  - $\Lambda_c$ is measured in §2 by a rule fixed here. The user will see the numbers before it is
    fixed: *"I'd like to see how this works out numerically."*
  - *Rejected:* a threshold on $\min|\theta'|/\max|\theta'|$. $|\theta'|$ spans many decades along
    calls with no stationary point at all; P1's control groups run from 27 to $1.5\times10^8$.
- **(ii) The round-off-limited exit is barred** in every window piece and every guarded region.
  - **At `depth_max`, a guarded region does not get the Levin estimate.** It is integrated by the
    nested Clenshaw–Curtis pair whatever its span, and flagged. Its call is reported not
    converged.
  - Reaching the cap there is the case in which the user said the layer-2 question reopens
    (Decisions, **G2 L2, decided**).

### 0.3 What this changes, and what it does not

- **It changes** `AdaptiveLevin/levin_quadrature.py`, and `QuadSourceIntegral`'s per-group record,
  which gains the integrator's diagnostic counts. A row's values change only where a point is
  located or a region is guarded.
- **It does not change:**
  - `phase_groups.py`, the partition, or the physics range of any integral;
  - the eq. (151) floor's formula or the per-region `resolved` test;
  - `_local_atol`, `DEFAULT_LEVIN_MAX_DEPTH` or `rtol`;
  - what `total_converged` means.

  Those belong to G3 (08) or to the separate physics prompt.
- **A call in which nothing is located and no region is guarded takes exactly today's path**, and
  its result is bit-identical (§3 test 8, §5, §6 V1).

## 1. The code

### 1.0 The basis interface

Two methods are added to the basis interface `_adaptive_levin` uses. Their names and signatures are
the agent's choice, recorded in the log.
- **The phases:** the basis's independent phases at $x$.
- **Their rates:** $d\theta_i/dx$ at $x$, in closed form, or a statement that the basis cannot
  supply them.

What each basis does:
- **`_Basis_SinCos`** has one phase. Its phase is `theta["theta"]`, and its rate is
  `theta["theta_deriv"]` when supplied, and "cannot" otherwise.
- **The tests' `_TwoIndependentSinCosBasis`** gains both methods, for its two phases.
- **Any other basis** that does not implement them is treated as unable to supply rates. Its
  behaviour is then unchanged except for the nodal guard of §1.3.

Everything below runs in `_adaptive_levin` and the generic subregion functions, once for each
independent phase the basis reports.

### 1.1 Location (in `_adaptive_levin`, when the basis supplies its rates)

1. **Scan.** Sample each rate $\theta_i'$ on $N_{\rm scan} = 4097$ points uniform in $x$ over
   `x_span`, endpoints included. This is 06a's grid. Sample through `_sample_vectorized`, so that a
   vectorised rate is used as such.
2. **Roots.** Each adjacent pair of samples of opposite sign, and each exact zero, is refined by
   `scipy.optimize.brentq` on that rate, with `xtol` no coarser than
   $4\varepsilon\max(|x_a|, |x_b|)$.
3. **Minima.** Each strict interior local minimum of $|\theta_i'|$ on the grid, with no sign change
   within a scan spacing of it, is refined by bounded Brent minimisation of $|\theta_i'|$.
   - $\theta_i'''$ there comes from a central difference of the rate at the refined point. The step
     and its stencil are the agent's choice, recorded with their error in the log.
   - The minimum is located if $\lambda < \Lambda_c$.
4. **Ends and merging.**
   - A point within half a scan spacing of an end of `x_span` is treated as at that end.
   - Points within $4\varepsilon|x|$ of one another are merged.
   - Each point records which phase it belongs to.

### 1.2 Windows

1. **The edges.** Round each located point $x_*$ of phase $i$, the window is the largest interval
   containing it on which $|\theta_i(x) - \theta_i(x_*)| \le \Phi_w$, clipped to `x_span`. Each
   edge is found by `brentq` on the basis's phase $\theta_i$; for `_Basis_SinCos` that is
   `theta["theta"]`, the raw phase. Its precision of about ${\rm ulp}(|\theta|)$ is negligible
   against $\Phi_w$; the log states the size.
   - **For a root,** $\theta - \theta(x_*)$ has one sign on both sides, so each edge is where it
     first reaches $\Phi_w$ in magnitude.
   - **For a minimum,** $\theta$ is monotone through $x_*$, so the edges are where it has changed
     by $\Phi_w$ either way.
2. **Overlap.** Windows that overlap or touch are merged into one.
3. **Inside a window.** The window is cut into pieces whose phase span is at most
   $\Delta\phi_w = 2\pi$. The span is $\int|\theta_i'|\,dx$, estimated on the scan, taking the
   largest over the basis's phases. Each piece is integrated by the existing
   nested Clenshaw–Curtis pair. A piece whose nested-pair estimate misses its tolerance is bisected,
   as C7 bisects a fallback region now. `phase_limited` is barred in every piece.
4. **Seeding the loop.** The initial region list is `x_span` with every window removed, each
   remaining range at depth 0. The window pieces enter the same accumulation, carrying their own
   region type, so that `num_regions`, `abserr` and the per-source breakdowns cover them. With
   nothing located, the list is today's single region, unchanged.
5. **Tolerance.** `_local_atol` applies to window pieces and Levin regions alike, by length share,
   unchanged.

### 1.3 The guard: the safety net

A region of the bisection loop is *guarded* if any phase rate changes sign on its own Chebyshev
nodes: `theta_prime_Cheb`, or the basis's equivalent for each of its phases. That means the
location missed a point, or the basis could not supply its rates. A guarded region:
- **is never accepted on the Levin branch.** While its phase span is at least $6\pi$ it is bisected.
- **goes to the nested Clenshaw–Curtis fallback** once its span is below $6\pi$, as any region does
  now. There it is accepted only on nested-pair resolution, with **`phase_limited` barred**.
- **at `depth_max`** is integrated by the nested Clenshaw–Curtis pair whatever its span. Its abserr
  is that pair's estimate. It is flagged `guard_at_depth_max`, and the call's `converged` is False.

An unguarded region outside every window behaves exactly as now.

### 1.4 What the result reports

- **`stationary_points`:** one entry per located point, giving:
  - its $x$ and its kind (`root`, `minimum` or `end`);
  - $\theta'$ there, and $\lambda$ for a minimum;
  - its window's edges and number of pieces.
- **Counts:**
  - `num_window_pieces`;
  - `num_guarded_regions`, which with `theta_deriv` supplied is expected to be 0;
  - `num_guard_at_depth_max`.
- **`used_interval`** gains a region type for window pieces, and a `guarded` flag.
- **The result comes from `_adaptive_levin`.** `adaptive_levin_sincos` returns it unchanged, as it
  returns everything else.
- **Docstrings.** Those of the module, `_roundoff_floor`, `_adaptive_levin`,
  `_adaptive_levin_subregion_impl` and the basis interface say what the location, the windows and
  the guard do, and where the C2 gate's account of a stationary point was incomplete.
  `adaptive_levin_sincos`'s docstring says that `theta_deriv` now also drives the location.

### 1.5 `QuadSourceIntegral`

`phase_group_Levin_integral`'s per-group record gains four counts: `stationary_points`,
`window_pieces`, `guarded_regions` and `guard_at_depth_max`. Nothing else in `QuadSourceIntegral`
changes, and no schema changes.

## 2. $\Lambda_c$ and $\Phi_w$, measured — a planned stop

**Run on §1's code, before §3's tests are written.** Both rules are fixed here, before any number
exists (README §5 rule 3).

**The two families,** on $x \in [0, 1]$, with smooth amplitudes $f_{\sin} = e^{-x}$ and
$f_{\cos} = \cos 3x$:
- **(R), a simple root:** $\theta(x) = \omega x + \tfrac{c}{2}(x - x_s)^2$. Here $\omega = 0$ puts
  the root at $x_s$, and $\omega \ne 0$ moves it to $x_s - \omega/c$.
  - $c \in \{10^4, 10^6, 10^8\}$;
  - $x_s \in \{0.3819660112501051, 0.7236067977499790\}$, neither of them dyadic.
- **(C), a nearly coalescing pair:** $\theta(x) = a(x - x_0) + \tfrac{b}{3}(x - x_0)^3$, with
  $x_0 = 0.3819660112501051$ and $\lambda = a^{3/2}/b^{1/2}$.
  - $b \in \{10^4, 10^8\}$;
  - $a$ chosen so that $\lambda \in \{0.25, 1, 4, 16, 64, 256, 1024\}$.

**The reference:** `mpmath.quad` at 30 digits, split at every zero of $\theta'$ and at $x_0$.

**The runs:** every case at `rtol` $10^{-10}$, with `atol` $10^{-14}$ times the reference.
**"Honest"** means $|{\rm value} - {\rm ref}| \le$ `abserr`. **"Converged"** means the call reports
`converged`.

**$\Phi_w$** first, on family (R).
- **The ladder:** $\Phi_w \in \{2\pi, 8\pi, 32\pi, 128\pi, 512\pi\}$.
- **The rule:** $\Phi_w$ is the smallest rung on which every case of (R) is honest and converged.
- **Also recorded,** for every case and rung: the true error, abserr, regions, window pieces,
  maximum depth, and $\theta'^2/|\theta''|$ at the window's edges.

**$\Lambda_c$** next, on family (C), with that $\Phi_w$.
- **The runs:** every case with no minimum located, as if $\Lambda_c = 0$.
- **The rule:** $\Lambda_c$ is the smallest rung $L$ of $\{4, 16, 64, 256, 1024\}$ such that every
  case with $\lambda \ge L$, at both values of $b$, is honest.
- **The check:** the cases with $\lambda < L$ are re-run with their minimum located. They must be
  honest and converged.

**Then stop and report.** Report both tables, the rung each rule picks, and the check, to the
orchestrator for the user. The agent continues to §3 only on the user's word, with the two
constants as the user settles them. **If no rung qualifies under either rule, stop and ask.**

Each constant becomes a named module constant, whose comment cites the measurement.

## 3. Tests

A new module, `AdaptiveLevin/tests/test_levin_stationary.py`, with no Ray and no datastore.
References are closed forms or `mpmath.quad` at 30 digits, split at every stationary point. Every
test asserts both that the value is within its declared `abserr` of the reference, and that
`abserr` is at least the true error.
1. **One stationary point, off every bisection point, of P1's shape.**
   - The phase has total variation of at least $10^5$ rad and one interior root at a non-dyadic
     $x_s$.
   - The parameters are chosen so that **the parent code accepts the whole span as one Levin
     region**, wrong beyond its abserr. The agent shows this at the parent, and records the
     parent's value, abserr and region count in the test's comment.
   - Under the new code: the point is located within $4\varepsilon|x_s|$, it has one window, and
     the value is honest.
2. **Two stationary points between two adjacent Chebyshev nodes** of the root region, so that
   $\theta'$ has no sign change on the root's nodes. Both are located by the scan, and the value is
   honest.
3. **The nearly coalescing pair,** §2's family (C) at $\lambda = \Lambda_c/4$, at both values of
   $b$. The minimum is located and windowed, and the value is honest.
4. **No `theta_deriv`.** Test 1's problem with `theta` only. Nothing is located, the nodal guard
   fires, and the value is honest.
5. **The round-off-limited exit is barred.** A case built like P1's row 1 at `rtol` $10^{-11}$,
   where the parent accepts the root as phase-limited across the point. Under the new code no
   window piece or guarded region is phase-limited, and the value is honest.
6. **The cap.** Test 4 with `depth_max` small enough that a guarded region reaches it. Assert:
   - the flag is set;
   - `converged` is False;
   - the region's abserr is its nested pair's.
7. **The window's edges.** In test 1, $|\theta - \theta(x_s)|$ at each edge equals $\Phi_w$ to
   within the precision §1.2 states. Every piece's phase span is at most $\Delta\phi_w$.
8. **Nothing else moves.**
   - Run every problem in `test_levin_quadrature.py` that has no stationary point, with a
     closed-form `theta_deriv` where one exists. Nothing is located and no region is guarded.
   - Value, abserr and region list are bit-identical to the same call with location bypassed. That
     needs a private switch in the integrator: test-only, named as such, and off by default.
9. **The generic path.** `_TwoIndependentSinCosBasis`, or a subclass of it in the new module, with
   one phase linear and the other quadratic, $\theta_2 = \tfrac{c}{2}(x - x_s)^2$ at a non-dyadic
   $x_s$, driven through `_adaptive_levin` directly.
   - The point is located on phase 2 and windowed, and the value is honest against the closed form
     (a Fresnel integral and an elementary one).
   - Phase 1 locates nothing.
10. **Existing tests.** Every test in `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/` passes
   unchanged. That includes C2's stationary-point tests, which supply no `theta_deriv` and so
   exercise the guard.

**Deliberate breakages,** each recorded in the log with the test that catches it, then reverted:
- **(i)** location, windows and guard all off: test 1 must fail;
- **(ii)** location off, leaving only the nodal guard: test 2 must fail;
- **(iii)** `phase_limited` allowed in window pieces and guarded regions: test 5 must fail;
- **(iv)** minima never located ($\Lambda_c = 0$): test 3 must fail;
- **(v)** the Levin estimate kept at `depth_max` in a guarded region: test 6 must fail;
- **(vi)** the window replaced by a plain split at $x_s$, with the Levin regions on either side
  accepted unguarded: test 1 or test 7 must fail;
- **(vii)** location done only for the first phase a basis reports: test 9 must fail.

**`ComputeTargets/tests/`:** a test that pins a group record's exact key set gains the four counts
of §1.5. That is the only existing assertion that may change; any other change is a stop.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 4. Records of the code

- `black` on the files changed.
- `logs/07-stationary-point-guard.md`, in README §5.1's form. Its **Predictions and verdicts**
  holds:
  - §2's two rules, with their tables and the constants as the user settled them;
  - §6's predictions, copied before §6 runs and marked not yet scored.
- The board's 07 row.
- **One commit.** The header's issue moves only in 07's records commit, after §5 and §6.

## 5. Rehearsals R15 and R16 — on the Mac Studio, after the code commit is pushed

As 06 §5. Records go under `var/levin-phase-floor-07/rehearsal/`.
1. **The lock:** `acquire`, then `set-pid` at each launch, and `release` after the readers. Each
   lock command is a shell command of its own (`CLAUDE.md`).
2. **R15** runs at the code commit's parent and **R16** at the code commit, each in a worktree.
   - Each runs `RunDrivers/quadsource_atol_sweep.py --build --rehearsal` as R13 and R14 ran it.
   - Each is registered with `--campaign levin-phase-floor --prompt 07`, with slugs `r15-parent` and
     `r16-guard`.
   - **Both fingerprints must equal `1b332ffa…`.**
3. **The value comparison** is `tools.store_value_diff` on R15 and R16, with `--json`, with 05's
   `classify_non_qsi.py` copied unchanged.
   - **Outside `QuadSourceIntegral`:** every value table is identical, every other difference is of
     the four volatile kinds of `[05-store-value-diff-compares-volatile-columns]`, and no row is in
     one store only.
   - **Inside `QuadSourceIntegral`, per row:** group records are compared with the four new counts
     set aside.
     - **A row in which R16 located nothing and guarded nothing is bit-identical** in `total`,
       `total_abserr`, the analytic columns and every group record.
     - **Any other row is tabulated:** the change against R15's and R16's `total_abserr`, the
       points located, the window pieces, the guarded regions and the depth.
4. **P1's two rows**, R14's rows at $z_{\rm response} = 250$: $(k,q,r) = (3.162\times10^5)^3$ and
   $(10^5, 3.162\times10^5, 3.162\times10^5)$.
   - **The rule:** under R16, $|{\rm total} - {\rm ref}| \le$ `total_abserr`.
   - **The references** are the totals P1 corrected on R14's integrand:
     $5.918585406\times10^{-20}$ and $-3.3961904975\times10^{-19}$.
   - **The location:** each row's `G+q+r` group locates its point, at $z = 10211.3$ and $275.0$.
5. **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`) run on R16. Every one
   exits 0, and the store is byte-identical before and after.
6. **Release the lock.** `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.
7. **The stores are kept** until the user says otherwise.

## 6. The re-measurement — on the Mac Studio, Ray-free, on A3 v2's scratch copy

06a's census, re-run at the code commit, against 06a's run as the "before".
- **The harness** is 06a's, from `docs/levin-phase-floor/06a/scripts/`, copied into
  `var/levin-phase-floor-07/census/scripts/`. Change only:
  - the tree and its paths;
  - the diagnostics §1.4 adds, recorded per call;
  - the reference for V4.

  Save the diff against 06a's copy, and record each change as a deviation.
- **The tree** $g$ is a worktree at Part B's dispatch SHA, whose code is the code commit's.
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification and fingerprint, read-only, and hashed before and after.
- **The units:** every row at `1e-7` and at `1e-8`, as in 06a, in one registered, detached run
  with slug `census`.
- **The budget:** 300 s per unit, with 06a's over-budget pass.

**Gate V0, before the run:**
- **The parent's code is 06a's.** `git diff --stat e05c34f <code commit's parent> -- . ':!docs'
  ':!prompts'` shows only `CLAUDE.md`, or files whose ASTs without docstrings equal `e05c34f`'s. So
  06a's run is the parent's.
- **The harness reproduces 06a.** At the parent it reproduces 06a's units **bit for bit** on
  `low-G+r-1` and `ctrl-low-1` at `1e-7`, and on one stationary-point row, 06a's `0-1971`, at
  `1e-8`.
- **The new code is deterministic.** Two runs of one $g$ unit on a stationary-point row are
  bit-identical.

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

**Descriptive, not scored:**
- **Depth:** the distribution of row depth under $g$ at both rtols, against 06a's.
- **Windows:** window pieces per call, the windows' widths in $x$, and $\theta'^2/|\theta''|$ at
  their edges.
- **The sample's errors before:** for V4's sample, $|{\rm value}_{06a} - {\rm ref}|$ against 06a's
  abserr. That is how wrong the stored rows were.
- **Totals:** how they moved against 06a, relative and against `total_abserr`$_{06a}$, by band and
  $k$.
- **Locations:** minima located and their $\lambda$; points at an end of a span; and
  `num_guarded_regions`, expected 0 since `theta_deriv` is supplied.
- **Convergence:** `convergence_kinds.py` (06a's changed copy) under $g$, and the
  `total_converged` counts.
- **Cost:** wall time against 06a. Cost is not an argument (README §5 rule 4).

## 7. Acceptance

1. **§1–§4 as written:**
   - §2's tables reported, and $\Lambda_c$ and $\Phi_w$ settled by the user;
   - every test passing and the seven breakages caught;
   - the seven roots rising only by the tests added.
2. **§5:**
   - R15 and R16 both `done` with `1b332ffa…`;
   - outside `QuadSourceIntegral`, every value table identical;
   - inside it, every row that locates and guards nothing bit-identical, and the rest tabulated;
   - P1's two rows honest;
   - the readers exit 0, and the lock is released.
3. **§6:** V0 holds; V1–V5 are each scored holds, fails or undecided against the rule as written,
   with their numbers; and the store is unchanged.
4. **Records:**
   - `logs/07-stationary-point-guard.md`, with §5 and §6's Mac Studio record appended by the
     orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §7, "The stationary-point guard";
   - the records in `docs/levin-phase-floor/07/`.

## 8. Stop conditions — stop and ask

- **The Gate** fails.
- **§2 is reached:** the planned stop, always taken. Separately, no rung qualifies under either
  of §2's rules.
- **A test cannot be built as written:** the parent does not show test 1's or test 5's failure.
- **The construction reaches outside its files.** It needs a change outside
  `AdaptiveLevin/levin_quadrature.py` and §1.5's record: to `phase_groups.py`, the partition, the
  eq. (151) formula, `resolved`, `_local_atol`, `DEFAULT_LEVIN_MAX_DEPTH` or `rtol`. Or it needs
  the location in `adaptive_levin_sincos` rather than in the generic code.
- **A test moves:**
  - an existing assertion has to change outside §3's one exception;
  - a test in `LiouvilleGreen/tests/` moves;
  - a seven-roots count falls.
- **The rehearsals:**
  - R15 or R16 does not give `1b332ffa…`;
  - outside `QuadSourceIntegral` a value table differs;
  - inside it, a row that locates and guards nothing is not bit-identical;
  - P1's two rows are not honest under R16.
- **The re-measurement:** V0 fails.
- **Any store file** changes.
- **The lock** is held, or will not release.

## 9. Amendments, 2026-10-28, before §2 runs

Part A stopped before §2 on 2026-10-27, on two of §8's conditions, with §1 built and uncommitted.
The user's rulings, quoted, and what follows from them. **Where this section and the text above
differ, this section holds.**

1. **A new prompt runs first** (the board's Decisions, **07 LG stop**). A `LiouvilleGreen` draw
   moved because the round-off-limited exit accepted the Levin range beyond a window at depth 0,
   on a floor that bisection would have cut by about $10^4$
   (`[07-roundoff-exit-assumes-bisection-cannot-help]`). The user: *"Fix it now. A prompt that runs
   before 07's §2."* That is [`06b-roundoff-exit-premise.md`](06b-roundoff-exit-premise.md).
   - **§1's uncommitted work** is set aside before 06b runs, with nothing lost, and restored after
     06b's code commit. The agent rebases it onto 06b's code. That rebase is a deviation, recorded
     as `STRUCTURALLY REQUIRED`.
   - **§2 runs on 06b's code and 07's §1 together.**
   - **The LiouvilleGreen stop is re-tested before §2's report.** The agent re-runs the 20 seeded
     draws, seeds 0–19, at the rebased tree. Any draw that fails under the new code and passes at
     06b's code is still §8's stop.
2. **§2's reference** (the user, 2026-10-28): *"we can compute the integral by any acceptable
   method"*, and *"Testing (R) using contour integration also: yes, please. I would like to be sure
   we are getting this part of the integrator right. It is the science-critical step."*
   - **Family (R)** is computed twice, by the closed form in the complex error function and by
     contour integration. The two must agree, to well below the precision §2's rules need. The
     log records the largest disagreement, and a case where they disagree is a stop.
   - **Family (C)** is computed by contour integration.
   - Both are validated against `mpmath.quad`, subdivided until it converges, wherever that takes
     minutes, as Part A did.
   - "`mpmath.quad` at 30 digits" in §2 and §3 means a reference of this kind.
3. **$\omega$ in family (R)** (the user, 2026-10-28): *"we should have some cases where `omega` is
   zero, and some where it is not … With this many digits precision, it should be possible to pick
   out the shift in the root against the value of `x_s`. A few cases with `omega=1` seem
   sufficient, since the amplitude of `omega` otherwise trades off against the value of `c`."*
   - Every case of §2's (R) grid runs at $\omega = 0$.
   - A few run also at $\omega = 1$, chosen by the agent and recorded. Their root lies at
     $x_s - 1/c$, and the location must find it there, not at $x_s$.
   - §2's rule for $\Phi_w$ covers every (R) case, those at $\omega = 1$ included.
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
