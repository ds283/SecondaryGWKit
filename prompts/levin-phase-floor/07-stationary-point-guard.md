# Prompt 07 — the stationary-point guard: locate them first, integrate their windows directly

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-27, after 06a landed and G2 was answered. **Rewritten 2026-11-07,** after 06m,
for the driver as it now stands (the board's Decisions, **07 rewritten**). The user: *"As far as I
can see, there is now no reason why we should not make another attempt at the long-delayed prompt
07"*, then *"Please draft a rewritten 07 and orchestration note"*. The text before the rewrite is
`50277f3e`'s, with its amendments in its §9. **This text replaces it. Where the two differ, this one
holds.**

**The charter is unchanged:** G2's Q4 (i) and (ii) (the board's Decisions, **06 P1**, **G2 Q4
(i)–(ii)**, **07 scope** and **07 (α)–(γ), (ii)**):
- *"We are supposed to deliver a function Levin integrator, and this is needed whether or not we
  plan to use the integrator as part of the 1-loop machinery. So we should fix this now,
  generically."*
- *"For now, 07 is just about the generic Levin integrator."*

The late-time range of the integrals is physics, and is a separate prompt
(`[06-time-integrals-must-reach-late-resonances]`).

**Why it was rewritten.** Part A of the first text stopped four times, each time on a defect in the
driver beneath it, and each defect was fixed first by its own prompt:
- 06b, the round-off exit's premise;
- 06c, global error control;
- 06d, a round-off floor for the Clenshaw–Curtis rule;
- 06e, exact end nodes and a node term that measures the displacement.

06h, 06g and 06m followed, on what 06e's census found. The loop the first text was written for has
gone: there is no region list, no per-region `resolved` test and no `_local_atol`. Since the work was
last set aside, `AdaptiveLevin/levin_quadrature.py` has gained about 1,150 lines and lost about 200,
in 06e, 06g and 06m. **So §1 is built afresh on the current driver.** The set-aside work is a
reference, not a base (§0.5).

**Where it runs:** in two parts, as 06g and 06m did.
- **Part A, §1–§4, the code:** on the laptop, by a fresh-context subagent, in the main checkout at
  `HEAD`. **§2 is a planned stop:** the agent reports its measurements to the user before any test
  is written. One commit.
- **Part B, §5 and §6:** on the Mac Studio, after the code commit is pushed.
  - **§5, rehearsals R27 and R28,** under the lock, each in a worktree: R27 at the code commit's
    parent, R28 at the code commit.
  - **§6, the census:** Ray-free, with no lock, on 06's certified scratch copy of A3 v2, against
    06m's census as the "before". Then references for a sample of stationary-point calls.

The records of Part B are committed on the laptop afterwards, in 07's records commit.

**Amended 2026-11-08, at Part B's first stop (§10):** Rule A, a phase whose variation is below the
C2 gate's threshold is not located. It is a second code commit on the laptop, made before Part B
goes on, with a rehearsal R29 in R28's place.

**Closes, in its records commit, on the user's word:** `[06-levin-accepts-a-region-across-a-stationary-point]`,
if §5 and §6 hold. It also answers two of the user's questions, from 06g's Part B:
- *"What needs to be true is that z12 works after 07."* (§2.3, §6 V4);
- on V1's twelve rehearsal rows: *"V1 succeeds but we think that could be accidental; 07 should
  decide that"* (§2.3, §5 S1).

**Recommended model:** **Opus**.

**Read first:**
1. **The board's §3 entry** `[06-levin-accepts-a-region-across-a-stationary-point]`, with every
   narrowing: 06a's scan, 06c's Y1, 06e's D5 and 06g's references. Then
   [`docs/levin-phase-floor/06/p1/report.md`](../../docs/levin-phase-floor/06/p1/report.md) in full:
   the mechanism; the reference method (its §5), whose window this prompt adopts; and the root
   diagnosis (its §6).
2. **The board's Decisions** for **06 P1**, **G2**, **07 scope**, **07 (α)–(γ), (ii)**, **07 §2
   reference**, **07 LG stop**, **07 §2 stop**, **06c stall withdrawn**, **07 third stop**, **07
   fourth stop**, **06g Part B** and **07 rewritten**.
3. **`AdaptiveLevin/levin_quadrature.py`**, the whole of `_adaptive_levin`'s docstring first. Then:
   - **the loop in `_adaptive_levin`:**
     - the root leaf;
     - `_refinable`;
     - the candidate heap and its ordering;
     - the stop test and its confirming re-sum;
     - the floor-limited exit with `_roundoff_exit_premise_holds` and `_split_could_meet_budget`;
     - an infinite eq. (151) floor taken first;
     - the `node_limited` stop;
     - the final `converged`;
   - `_check_leaf` and `_trust_noise_yardstick`;
   - `_adaptive_levin_subregion_impl` (the C2 gate at `phase_span < SIX_PI`) and
     `_adaptive_levin_subregion_cc` and their result dicts;
   - `_roundoff_floor`, `_levin_G0_G1`, `_cc_roundoff_floor` and the constants above them;
   - `used_interval` and its region types, and `_sample_vectorized`;
   - `_Basis_SinCos` and every basis method the driver calls;
   - `adaptive_levin_sincos`, the facade, and its result dict;
   - the test-only switch `_test_only_bypass_roundoff_premise`, the precedent for §3's.
4. **`AdaptiveLevin/tests/`:**
   - `test_levin_quadrature.py`'s `_TwoIndependentSinCosBasis`, which has no Clenshaw–Curtis methods,
     and `test_generic_m_component_basis_two_decoupled_phases`;
   - the existing tests that put a stationary point in a call:
     - `test_stationary_phase_gate_uses_total_variation`;
     - `test_levin_cc_floor.py`'s `TestFloorDoesNotGrow.test_nested_leaves_round_a_stationary_point`,
       its `TestNoInverseThetaPrime` and its `TestStationaryPhaseOracle`;
     - `test_levin_end_nodes.py`'s `TestStationaryPhaseOracle`;
     - `test_levin_global_budget.py`'s test 6 and test 12c
       (`test_trust_check_stationary_point_line`, which supplies `theta_deriv`).
5. **The callers.**
   - `ComputeTargets/QuadSourceIntegral.py`: `phase_group_Levin_integral` and its per-group record,
     and the analytic blocks' Levin calls and what is recorded of them.
   - `ComputeTargets/phase_groups.py`, for `theta_deriv`, the closed-form sum each group hands to
     the integrator.
   - `LiouvilleGreen/three_bessel_integrals.py`, whose `levin_theta` also supplies `theta_deriv`, so
     location runs there too.
6. **The earlier attempts' records** (§0.5): their log, and §2's tables at stops 2–4, from
   `var/levin-phase-floor-07/wip/07-part-a-rebased-on-06d-untracked.tar.gz`.
7. **The references this prompt scores against:**
   - 06g's, `docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json` (`V1` for w00–w11,
     `rows` for z00, z12 and Z3's others), with its `report.md`;
   - 06c's Y1 references, `docs/levin-phase-floor/06c/y1-references/results.json`;
   - `tools/filon_reference.py` and its tests.

**Gate:**
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale on the laptop;
- `git stash list` shows 07's four stashes, and `var/levin-phase-floor-07/wip/` matches its
  `SHA256SUMS.txt`;
- the 14 captures in `var/levin-phase-floor-06g-fcc-laptop/inputs/` are present (§2.3);
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defect

`adaptive_levin_sincos` returns a Levin leaf that contains an interior stationary point of its phase,
with an error estimate far too small (06's P1).
- **Why the estimate cannot see it.** The step-(4) residual compares a leaf's estimate with its two
  halves'. Near a stationary point the Levin antiderivative $p \sim f/(i\theta')$ is singular. The
  parent and the half that holds the point both lack the stationary-phase term, so the residual is
  blind to it. On 06's two R14 rows it was about $10^5$ smaller than the true error. 06g's trust
  check one level down (Δ1, Δ2, $r$) is built from the same solves, and is blind in the same way.
- **The round-off floor can admit it.** The eq. (151) floor carries $1/G_0$, $G_0 = \min|\theta'|$
  on the leaf's nodes. When a node falls near the point, $G_0$ is at noise and the floor is
  infinite; the loop then chooses that leaf first and bisects it. That is the only stationary-point
  handling in the driver today, and it works only when a node falls close enough.
- **The C2 total-variation gate does not catch it.** It routes a leaf to the Clenshaw–Curtis rule
  only when its phase varies by less than $6\pi$. Its regression test puts the stationary point at
  $x = 1/2$, the first bisection point, where bisection exposes it by construction.

**How far it reaches:**
- **06a's scan** (at 06a's code, `1e-7`): 720 of A3 v2's 9,280 rows hold a call with an interior
  sign change of $\theta'$, all in group `G+q+r`. In every one a region was accepted across the
  point, 623 of them a single region of the whole call. Every A3 v2 row deeper than 11 was one of
  them.
- **06c's Y1:** 12 equilateral rehearsal rows hold the point of `G+q+r` at $z \approx 10211$, where
  $c_s = 1/2$. On each, 06b's or 06c's code reported converged while wrong by
  $4.5\times10^3$–$1.6\times10^6$× its declared error.
- **06g's references:**
  - **z12** (`g-1e-8-1-2174` call 1) is reported converged at depth 18 and is 1,681× its declared
    error from the reference. That is unchanged under 06m (1,681.30). z00, the same integrand at
    `1e-7`, is unconverged at depth 20 and 0.018 from it.
  - **V1's twelve** (w00–w11, R24's rows): 06h's code was $4.5\times10^3$–$6.0\times10^5$× its error
    out. 06g's is within on all twelve (at most 0.942), but not by design.
- **06m's census, in the 720:** 311 group calls unconverged, 309 of them within their request and
  holding a leaf at `depth_max`. Their largest share of a row's `total_abserr` is 53 %.

The stationary point is the Kohri–Terada resonance, where the signal is expected to come from. A
stored `total_abserr` on such a group is not trustworthy.

### 0.2 The remedy, and the user's decisions on it

All of it lives in `AdaptiveLevin/levin_quadrature.py`, in the generic code `_adaptive_levin`, which
is written for a basis of any number of components. That covers every caller of
`adaptive_levin_sincos`, and any future basis.
- **The basis supplies what is basis-specific** through two new methods of the basis interface: its
  independent phases, and their rates. `_Basis_SinCos` has one phase. A basis with several, such as
  the tests' `_TwoIndependentSinCosBasis`, reports each of them. A basis that cannot supply rates
  reports none; then there is no location, and the nodal guard of step 4 is the only protection.
- **A stationary point** is a zero of any of the basis's phase rates.
- **`adaptive_levin_sincos` stays a facade.** Only its docstring changes. The user, 2026-10-27: it
  *"is just a facade that sets up the basis functions used by the Levin step. It seems the wrong
  place for the stationary phase check, which is generic and does not depend on the basis."*

**The four steps, each generic:**
1. **Locate** the points before integrating, from the basis's closed-form phase rates. For
   `_Basis_SinCos` that is the caller's `theta["theta_deriv"]`. Two kinds of point are located:
   - every sign change of $\theta'$, which is a stationary point;
   - every interior local minimum of $|\theta'|$ deep enough to act as a nearly coalescing pair.
     That covers the QCD and $e^+e^-$ caustics and any like them.
2. **Carve out a window** round each located point. The window is where the phase stays within
   $\Phi_w$ of its value at the point. It is integrated directly, in pieces, by the Clenshaw–Curtis
   rule.
3. **Hand the rest to the loop as it stands.** The ranges between windows are leaves of the same
   loop, with the window pieces, in one budget for the call.
4. **Guard what the location missed.** A leaf whose $\theta'$ changes sign among its own Chebyshev
   nodes is never returned as a Levin leaf.

**The decisions** (the board's §2.3 and Decisions; the user, 2026-10-27):
- **(α) The location uses `theta_deriv` and lives in the integrator's generic code, not in any
  caller and not in the facade.** *"From `theta_deriv` is the right way to locate stationary phase
  points, not by injecting physics knowledge. We want all callers to be protected."* So it runs for
  `QuadSourceIntegral`'s group calls and for the analytic blocks (`LiouvilleGreen`'s `levin_theta`
  supplies `theta_deriv` too). Without `theta_deriv` there is no location; the nodal guard is then
  the only protection, and the docstring says what it misses.
- **(β) An explicit window of phase half-width $\Phi_w$, not a bisection cascade.** The user's
  argument:
  - bisection to $6\pi$ builds a logarithmically dense cascade of leaves round the point;
  - $6\pi$ answers a different question, whether a generic leaf is weakly oscillatory;
  - the width wanted is known.

  **What fixes the width.** For a simple root, $\theta' \approx \theta''(x - x_s)$. At the window's
  edge the parameter $\theta'^2/|\theta''|$, which governs whether Levin's non-oscillatory solution
  exists, equals $2\Phi_w$ whatever $\theta''$ is. One dimensionless number sets how well
  conditioned the Levin leaves next to the window are. P1's reference used this construction (P1
  report §5) and agreed to about $10^{-12}$ relative for $\Phi$ from $10^3$ to $3\times10^4$.
  $\Phi_w$ is measured in §2.
- **(γ) A minimum of $|\theta'|$ counts as a nearly coalescing pair by its Airy parameter.** Near a
  local minimum $x_m$, $\theta' \approx a + \tfrac12\theta'''(x_m)(x - x_m)^2$. The pair matters when
  the phase accumulated across its slow zone is small:
  $\lambda = |\theta'(x_m)|^{3/2}/\sqrt{|\theta'''(x_m)|/2}$. A minimum is located when
  $\lambda < \Lambda_c$, and then gets a window like a root. $\Lambda_c$ is measured in §2. The
  user: *"I'd like to see how this works out numerically."* *Rejected:* a threshold on
  $\min|\theta'|/\max|\theta'|$, since $|\theta'|$ spans many decades along calls with no stationary
  point at all.
- **(ii) The round-off-limited exit is barred** in every window piece and every guarded leaf. In
  today's loop that is the floor-limited exit: such a leaf is never marked floor-limited, and stays
  refinable until `depth_max`. At `depth_max` a guarded leaf does not keep a Levin estimate (§1.4).
  Reaching the cap there is the case in which the user said the layer-2 question reopens (Decisions,
  **G2 L2, decided**).
- **A stopping rule for window leaves whose error does not fall** (Decisions, **07 fourth stop**;
  the user, *"Let's go for (b)"*). If window leaves are still chased to `depth_max` on a flat error,
  the rule is 07's, written after §2 measures them (§1.3, written at §2's stop). Its basis is QUADPACK's own (§2.5). The driver's
  general stall rule stays withdrawn: *"Let's remeasure the stall question after 07"* (Decisions,
  **06c stall withdrawn**). It is not this prompt's.

### 0.3 The driver it lands in

Read the code; this is a map, not a substitute.
- **One loop, one budget.** The call starts from one root leaf over `x_span`. Each leaf carries a
  budget error $e'_i$. The loop stops when $\sum e'_i \le \max(\texttt{atol}, \texttt{rtol}|\sum
  v_i|)$, confirmed by an ascending re-sum. While the sum is above the request it pops the
  refinable leaf from a heap ordered untrusted first, then by largest $e'$, and bisects it.
- **Refinable** means below `depth_max` and not floor-limited. A leaf is found floor-limited only
  when popped: its eq. (151) or Clenshaw–Curtis floor is finite, its residual is at most that floor,
  and 06b's premise holds as 06g states it (the halves' floors less their declared terms sum to at
  least half its own, and replacing it by its halves could not meet the budget). A Levin leaf whose
  $G_0$ is at noise has an infinite floor, and is taken first.
- **The trust check** (06c, 06g, 06m) runs once the budget is met: each leaf is solved one level
  down. It is trusted iff $r < 1/4$, or $r < 1$ and $\Delta_1 \le \nu_B$. A trusted leaf returns its
  quarters sum $Q$; an untrusted one is refined first.
- **The node errors** of Clenshaw–Curtis leaves are carried as a signed sum $\sum E$. The call stops
  `node_limited` when $\sum e'$ meets the request, every leaf has been checked, no untrusted leaf is
  refinable, and $\sum e' + |\sum E|$ is still above the request.
- **`converged`** means the budget is met with $\sum e' + |\sum E|$, and no returned leaf is
  untrusted.
- **No stall rule.** A leaf whose error does not fall with $h$ is chased to `depth_max` if it holds
  the total back.
- **Leaf types:** `used_interval` has `INTERVAL_TYPE_LEVIN` and `INTERVAL_TYPE_DIRECT`. No path
  forces a Clenshaw–Curtis solve on a span of $6\pi$ or more. `theta_prime_Cheb` is a local of the
  Levin solve; leaves keep no sample of $\theta'$.

### 0.4 What this changes, and what it does not

- **It changes** `AdaptiveLevin/levin_quadrature.py`, and `QuadSourceIntegral`'s records of Levin
  calls, which gain the integrator's four counts (§1.6). A call's result changes only where a point
  is located or a leaf is guarded.
- **It does not change:**
  - `phase_groups.py`, the partition, the physics range of any integral, or `LiouvilleGreen/`;
  - the eq. (151) formula, `_cc_roundoff_floor`, the trust rule, $\nu_B$ or the `node_limited`
    stop;
  - `DEFAULT_LEVIN_MAX_DEPTH` or `rtol`;
  - what `total_converged` means. That is 08's (`[06m-calls-flagged-unconverged-within-their-request]`).
- **A call in which nothing is located and no leaf is guarded takes exactly today's path**, and its
  result is bit-identical (§3 test 8, §5, §6 V1).

### 0.5 The set-aside work, and the earlier records

07's Part A was set aside four times. **It stays exactly as it is**, until the user decides its
fate after 07's records commit.
- **The stashes,** newest first:
  - `stash@{0}`: rebased on 06d at the fourth stop (base `f5def2f`, onto `e5764a6`);
  - `stash@{1}`: rebased on 06c at the third stop;
  - `stash@{2}`: rebased on 06b, §2 run;
  - `stash@{3}`: §1 uncommitted, before 06b.
- **The tarballs and patches** are in `var/levin-phase-floor-07/wip/`, listed in its
  `SHA256SUMS.txt`.

**How the agent uses them:**
- **Read only.** `git stash list` and `git stash show -p stash@{N}` are allowed. No other `git
  stash` subcommand is run: no `apply`, `pop`, `drop`, `clear` or `push`.
- **The tarballs** are extracted only into the session scratchpad, never into the tree.
- **Code may be taken from them by hand.** §2's references, families and harness
  (`calibration/scripts/reference.py`, `calibrate.py`, `validate_reference*.py`, `lg_location.py`,
  `parent_compare.py`) are the obvious candidates. Each piece taken is named in the log, with the
  file it came from. The construction is the agent's, on today's driver.

**The earlier records are kept.** The latest tarball's `docs/levin-phase-floor/07/calibration/`
(stops 1–3 and the fourth stop's `out/` and `scripts/`) and its
`prompts/levin-phase-floor/logs/07-stationary-point-guard.md` are copied byte for byte into
`docs/levin-phase-floor/07/earlier-attempts/`, the log renamed `log-parts-a-1-to-4.md`. Each file
is checked against the tarball by SHA-256, with the list saved. They are records of the trees they
ran on, and are not edited (`CLAUDE.md` rule 6).

## 1. The code

Every change is in `AdaptiveLevin/levin_quadrature.py`, except §1.6's.

### 1.0 The basis interface

Two methods are added to the basis interface `_adaptive_levin` uses. Their names and signatures are
the agent's choice, recorded in the log.
- **The phases:** the basis's independent phases at $x$.
- **Their rates:** $d\theta_i/dx$ at $x$, in closed form, or a statement that the basis cannot
  supply them.

**What each basis does:**
- **`_Basis_SinCos`** has one phase. Its phase is `theta["theta"]`. Its rate is
  `theta["theta_deriv"]` when supplied, and "cannot" otherwise.
- **The tests' `_TwoIndependentSinCosBasis`** gains both methods for its two phases (§3 test 9).
- **Any other basis** without them is treated as unable to supply rates. Its behaviour is then
  unchanged except for the nodal guard of §1.4.

Everything below runs once for each independent phase the basis reports.

**Amended 2026-11-08 (§10):** a phase whose total variation over `x_span` is below `SIX_PI` is not
located (Rule A), and §1.4's guard tests such a phase of a multi-phase basis for a sign change.

### 1.1 Location (in `_adaptive_levin`, before the loop, when the basis supplies its rates)

1. **Scan.** Sample each rate $\theta_i'$ on $N_{\rm scan} = 4097$ points uniform in $x$ over
   `x_span`, endpoints included. This is 06a's grid. Sample through `_sample_vectorized`, so that a
   vectorised rate is used as such.
   - **Rule A (§10).** If the phase's total variation, estimated from these samples, is below
     `SIX_PI`, steps 2–4 do not run for it.
2. **Roots.** Each adjacent pair of samples of opposite sign, and each exact zero, is refined by
   `scipy.optimize.brentq` on that rate, with `xtol` no coarser than
   $4\varepsilon\max(|x_a|, |x_b|)$.
3. **Minima.** Each strict interior local minimum of $|\theta_i'|$ on the grid, with no sign change
   within a scan spacing of it, is refined by bounded Brent minimisation of $|\theta_i'|$.
   - $\theta_i'''$ there comes from a central difference of the rate at the refined point. The step
     and its stencil are the agent's choice, recorded with their error in the log.
   - The minimum is located if $\lambda < \Lambda_c$.
4. **Ends and merging.**
   - A point within half a scan spacing of an end of `x_span` is treated as at that end, and recorded
     as kind `end`.
   - Points within $4\varepsilon|x|$ of one another are merged.
   - Each point records which phase it belongs to.

### 1.2 Windows

1. **The edges.** Round each located point $x_*$ of phase $i$, the window is the largest interval
   containing it on which $|\theta_i(x) - \theta_i(x_*)| \le \Phi_w$, clipped to `x_span`. Each edge
   is found by `brentq` on the basis's phase $\theta_i$; for `_Basis_SinCos` that is
   `theta["theta"]`. Its precision, about ${\rm ulp}(|\theta|)$, is negligible against $\Phi_w$; the
   log states the size.
   - **For a root,** $\theta - \theta(x_*)$ has one sign on both sides, so each edge is where it
     first reaches $\Phi_w$ in magnitude.
   - **For a minimum,** $\theta$ is monotone through $x_*$, so the edges are where it has changed
     by $\Phi_w$ either way.
2. **Overlap.** Windows that overlap or touch are merged into one.
3. **Pieces.** Each window is cut into pieces whose phase span is at most $\Delta\phi_w = 2\pi$. The
   span is $\int|\theta_i'|\,dx$, estimated on the scan, taking the largest over the basis's phases.
4. **Each piece is a Clenshaw–Curtis leaf** of the loop, solved by `_adaptive_levin_subregion_cc`
   **whatever the C2 gate's own span estimate says**. That needs a path that forces the
   Clenshaw–Curtis solve; the agent adds it, and the log describes it. A window leaf's halves are
   window leaves too.
5. **Seeding.** With windows, the root leaf is replaced by:
   - one leaf for each range of `x_span` between windows, at depth 0, an ordinary leaf of today's
     loop;
   - the window pieces, at depth 0, marked as window leaves.

   All of them are in the one budget, the one heap, the trust check and the node sum, so that
   `num_regions`, `abserr` and the per-source breakdowns cover them. **With nothing located the
   root leaf is today's, unchanged.**
6. **Window leaves in the loop:**
   - the floor-limited exit never fires on them (§0.2 (ii)); they stay refinable until `depth_max`
     or until §1.3's rule stops their window;
   - their trust check, node errors and declared terms are a Clenshaw–Curtis leaf's, unchanged;
   - **at `depth_max`** a window leaf keeps its Clenshaw–Curtis estimate, as any leaf there does.

### 1.3 The stopping rule for window leaves (written at §2's stop, §9 point 2)

§0.2's last decision, written from §2.5's measurement. QUADPACK's `iroff1`, counted per window.
- **The count.** Each window, after merging (§1.2 item 2), keeps a counter. A bisection of one of
  its leaves $P$ into $L$ and $R$ **counts** if both hold:
  - $|(V_L + V_R) - V_P| \le 10^{-5}\,|V_L + V_R|$: the halves change the value by at most $10^{-5}$
    relatively, with $V$ each leaf's own solve (`value_own`);
  - $e'_L + e'_R \ge 0.99\,e'_P$: their summed error is at least 0.99 of the parent's, with $e'$
    the error the loop's budget sums for each leaf.
- **At 6 counted bisections the window is round-off-limited.** Every leaf of it, those already
  made and any made later, is then not refinable. Each is returned **floor-limited**
  (`phase_limited`), and is exempt from the trust check as a floor-limited leaf is. It declares
  what it carries: its $e'$, except that a leaf already found untrusted declares as an untrusted
  leaf at `depth_max` does, $\max(\epsilon_P, \Delta_1)/(1-r)$ if $r < 1$ and
  $\max(\epsilon_P, \Delta_1 + \Delta_2)$ if not. The call's `converged` then follows the budget,
  as for any floor-limited leaf.
- **The basis.** `dqage` counts the same bisection with the same two thresholds, and gives
  `ier = 2` at 6. QUADPACK stops the whole call; here one window stops, because the rest of the
  call may still be resolving. The three numbers are QUADPACK's, unchanged. `iroff2` (a rise in
  error) is not adopted: §2.5 found both mechanisms flat, not rising.
- **What it ends,** on §2.5's measurement: (A), window leaves on a Clenshaw–Curtis floor flat in
  $h$ and above the request; and (B), window leaves whose nested-pair differences sit at the
  integrand's noise, found untrusted and refined whatever the budget says. On (A) a window ends
  floor-limited, as the parent's leaves do; on (B) its untrusted leaves stop holding the call
  unconverged when their declared error is within the request.
- **Where it could fire falsely:** a window that still has a leaf whose error is falling when six
  flat bisections have counted elsewhere in it. That leaf then declares what it carries, so the
  value stays honest, and the call may end unconverged where refinement would have converged it.
  §2's re-run on the final code reports every window that stops while one of its leaves' errors
  fell at its last bisection.
- **Unchanged:** the bar on the floor-limited exit (§0.2 (ii)) for window leaves the rule has not
  stopped, and for every guarded leaf; guarded leaves are not counted (none was guarded in §2).

### 1.4 The guard: the safety net

A Levin leaf is *guarded* if any phase rate changes sign on its own Chebyshev nodes: on
`theta_prime_Cheb`, or the basis's equivalent for each of its phases. That means the location missed
a point, or the basis could not supply its rates. The Levin solve already holds those samples; the
guard is computed there and carried in the leaf's result dict.

**A guarded leaf:**
- **is never returned as a Levin leaf.** It is never trusted, never floor-limited, and is chosen and
  bisected before any unguarded leaf, while it is refinable. Below $6\pi$ the C2 gate sends its
  halves to the Clenshaw–Curtis rule, as now. Treating its $e'$ as infinite, as the loop already
  does for a Levin leaf whose $G_0$ is at noise, is one way to do this; the mechanism is the agent's
  choice, and the log names it.
- **at `depth_max`, if still a Levin leaf,** is re-solved by the Clenshaw–Curtis rule whatever its
  span (§1.2 item 4's path). Its errors are that solve's. It is flagged `guard_at_depth_max`, and
  the call's `converged` is False.

An unguarded leaf outside every window behaves exactly as now. A phase that Rule A skipped (§10) is
tested by the guard, for a basis of several phases, for a sign change rather than for one strict
sign.

### 1.5 What the result reports

- **`stationary_points`:** one entry per located point, giving:
  - its $x$, its phase, and its kind (`root`, `minimum` or `end`);
  - $\theta'$ there, and $\lambda$ for a minimum;
  - its window's edges and number of pieces.
- **Counts:**
  - `num_window_pieces`, the window leaves returned;
  - `num_guarded_regions`, which with `theta_deriv` supplied is expected to be 0;
  - `num_guard_at_depth_max`.
- **`used_interval`** gains a window flag and a `guarded` flag, read by the census's region records
  (§6).
- **The result comes from `_adaptive_levin`.** `adaptive_levin_sincos` returns it unchanged, as it
  returns everything else.
- **Docstrings:**
  - the module docstring's account of stationary points, which calls gate-and-bisect "the correct
    structural response", is corrected;
  - `_roundoff_floor`, `_adaptive_levin`, `_adaptive_levin_subregion_impl` and the basis interface
    say what the location, the windows and the guard do, and where the C2 gate's account of a
    stationary point was incomplete;
  - `adaptive_levin_sincos`'s docstring says that `theta_deriv` now also drives the location.

### 1.6 `QuadSourceIntegral`

Each record `QuadSourceIntegral` keeps of a Levin call gains four counts: `stationary_points` (the
number located), `window_pieces`, `guarded_regions` and `guard_at_depth_max`.
- **The group calls:** `phase_group_Levin_integral`'s per-group record.
- **The analytic blocks:** the record `QuadSourceIntegral` keeps of each three-Bessel Levin call, if
  it keeps one per call. If it keeps none that the counts can join without touching `LiouvilleGreen/`,
  that is reported at §2's stop with what §5's rule would then have to read instead. It is not a
  stop.

Nothing else in `QuadSourceIntegral` changes, and no schema changes.

## 2. The measurement — a planned stop

**Run on §1's code, before §3's tests are written.** The rules are fixed here, before any number
exists (README §5 rule 3).

### 2.1 The families and their references

**The two families,** on $x \in [0, 1]$, with smooth amplitudes $f_{\sin} = e^{-x}$ and
$f_{\cos} = \cos 3x$:
- **(R), a simple root:** $\theta(x) = \omega x + \tfrac{c}{2}(x - x_s)^2$, with:
  - $c \in \{10^4, 10^6, 10^8\}$;
  - $x_s \in \{0.3819660112501051, 0.7236067977499790\}$, neither of them dyadic.

  Every case runs at $\omega = 0$. A few run also at $\omega = 1$, chosen by the agent and
  recorded; their root lies at $x_s - 1/c$, and the location must find it there, not at $x_s$. The
  user: *"we should have some cases where `omega` is zero, and some where it is not … A few cases
  with `omega=1` seem sufficient."*
- **(C), a nearly coalescing pair:** $\theta(x) = a(x - x_0) + \tfrac{b}{3}(x - x_0)^3$, with
  $x_0 = 0.3819660112501051$ and $\lambda = a^{3/2}/b^{1/2}$:
  - $b \in \{10^4, 10^8\}$;
  - $a$ chosen so that $\lambda \in \{0.25, 1, 4, 16, 64, 256, 1024\}$.

**The references** (the user, 2026-10-28: *"we can compute the integral by any acceptable method"*,
and *"Testing (R) using contour integration also: yes, please … It is the science-critical step."*):
- **Family (R)** is computed twice, by the closed form in the complex error function and by contour
  integration. The two must agree, to well below the precision the rules need. The log records the
  largest disagreement, and a case where they disagree is a stop.
- **Family (C)** is computed by contour integration.
- Both are validated against `mpmath.quad`, subdivided until it converges, where that takes
  minutes.

The earlier attempts' `reference.py` and `validate_reference*.py` did this (§0.5). They may be
reused, re-validated on this tree.

**The runs:** every case at `rtol` $10^{-9}$, with `atol` $10^{-14}$ times the reference
(amended at §2's stop, §9 point 1; the first run was at $10^{-10}$). If at $10^{-9}$ no rung of
$\Phi_w$ qualifies under §2.2's rule, or no rung of $\Lambda_c$ under its rule, §2.2 is run again in
full at `rtol` $10^{-8}$ and its rules applied there.
**"Honest"** means $|{\rm value} - {\rm ref}| \le$ `abserr`. **"Converged"** means the call reports
`converged`.

### 2.2 The two constants

**$\Phi_w$** first, on family (R).
- **The ladder:** $\Phi_w \in \{2\pi, 8\pi, 32\pi, 128\pi, 512\pi\}$.
- **The rule:** $\Phi_w$ is the smallest rung on which every case of (R) is honest and converged,
  those at $\omega = 1$ included.
- **Also recorded,** for every case and rung:
  - the true error, abserr, regions, window pieces and maximum depth;
  - $\theta'^2/|\theta''|$ at the window's edges;
  - whether the call ended `node_limited`.

**$\Lambda_c$** next, on family (C), with that $\Phi_w$.
- **The runs:** every case with no minimum located, as if $\Lambda_c = 0$.
- **The rule:** $\Lambda_c$ is the smallest rung $L$ of $\{4, 16, 64, 256, 1024\}$ such that every
  case with $\lambda \ge L$, at both values of $b$, is honest.
- **The check:** the cases with $\lambda < L$ are re-run with their minimum located. They must be
  honest and converged.

Each constant becomes a named module constant, whose comment cites the measurement.

### 2.3 The real calls: z00, z12 and V1's twelve

The 14 captures in `var/levin-phase-floor-06g-fcc-laptop/inputs/` (`z00`, `z12`, `w00`–`w11`) are
06g's captured inputs of these calls: their span, integrand, phase and arguments, checked bit for bit
against 06g's census or R24's store when they were made. Their references are 06g's
(`scores_stage3/scores.json`; FCC at resolution 2, with four readings agreeing). Each is
replayed through `adaptive_levin_sincos` with the captured arguments, in its own process, at the
parent and at §1's code with each rung of $\Phi_w$.
- **First, the replay check.** In a scratch worktree at `3ae6f3a4` (06g's code), each replay
  reproduces its 06g record bit for bit: value and abserr in `float.hex`, regions, maximum depth and
  `converged` (`scores.json`: `V1`'s `R24` for the w calls, `rows`' `g` for z00 and z12). At the
  parent, z00 and z12 are compared with route B's records in
  `docs/levin-phase-floor/06l/out/partC/h1_calls.csv`, which 06m's census equals. A replay that
  does not reproduce is a stop. The integrand has not changed since 06g's tree: 06m changed only
  the driver.
- **Recorded for each call and rung:** value, abserr, $|v - {\rm ref}|/{\rm abserr}$, `converged`,
  `node_limited`, regions, maximum depth, the located points and their windows.
- **The rule, at the chosen $\Phi_w$:** all 14 are honest. **z12 is the user's own test**: *"What
  needs to be true is that z12 works after 07."* z12 not honest at the chosen rung is reported at
  the stop as a failure, with the rung at which it would be.

`r55.pkl` in the same directory is not a stationary-point call; it may be replayed as a control,
where nothing should be located.

### 2.4 The LiouvilleGreen draws

06g's 20 seeded draws of `test_3bessel_analytic.py`, seeds 0–19, run by 06m's calibration harness
(`docs/levin-phase-floor/06m/calibration/scripts/`, copied into this prompt's
`calibration/scripts/` with the diff recorded), in two modes: the parent and §1's code.
- no test fails under the code that passes under the parent;
- every Levin call in which nothing is located and no leaf is guarded is bit-identical to the
  parent's;
- **what location does on these calls is recorded:** every call that locates a point, with the
  point's kind, $\lambda$ for a minimum, and the call's value and abserr under both codes. Location
  may fire here, where $K$ and the residual phase terms have opposite signs.

### 2.5 What still does not finish

For every call of §2.1–§2.4 that does not converge, or reaches `depth_max`:
- which leaves hold the unmet error: window, guarded, or ordinary Levin or Clenshaw–Curtis leaves;
- each such leaf's error over its ancestry, depth by depth, and whether it falls with $h$;
- for each window leaf chased to `depth_max`: whether today's floor-limited exit would have fired on
  it if it were not barred, and whether the call ended `node_limited`;
- what QUADPACK's counters would have said along that ancestry. `dqage`'s `iroff1` counts a
  bisection whose halves change the value by at most $10^{-5}$ relatively while their summed error
  is at least 0.99 of the parent's, and gives `ier = 2` at 6 such. `iroff2` counts a rise in error
  after 10 subintervals and gives up at 20. QUADPACK stops the whole call; it does not mark a leaf.

**This is evidence for §1.3, not a rule.** If no window leaf is chased to `depth_max` on a flat
error, the report says so, and §1.3 may need nothing.

### 2.6 Stop and report

Report to the orchestrator for the user:
- both tables of §2.2, the rung each rule picks, and the check on (C)'s cases below $L$;
- §2.3's table at every rung, and the chosen rung's verdict on all 14, z12 first;
- §2.4's results, and what location did there;
- §2.5's account, and the agent's reading of what it implies for §1.3, stated as a reading;
- §1.6's analytic-block finding;
- every deviation so far, numbered and classified.

**The agent continues to §3 only on the user's word,** with the two constants as the user settles
them, and §1.3 as the user rules it. **If no rung qualifies under either rule, stop and ask.**

## 3. Tests

A new module, `AdaptiveLevin/tests/test_levin_stationary.py`, with no Ray and no datastore.
References are closed forms, contour integrals or `mpmath.quad` subdivided to convergence, split at
every stationary point. Every test asserts both that the value is within its declared `abserr` of
the reference, and that `abserr` is at least the true error.
1. **One stationary point, off every bisection point, of P1's shape.**
   - The phase has total variation of at least $10^5$ rad and one interior root at a non-dyadic
     $x_s$.
   - The parameters are chosen so that **the parent returns a Levin leaf across the point, wrong
     beyond its abserr**. The agent shows this at the parent, and records the parent's value,
     abserr, region count and `converged` in the test's comment.
   - Under the new code: the point is located within $4\varepsilon|x_s|$, it has one window, and
     the value is honest.
2. **Two stationary points between two adjacent Chebyshev nodes** of the root leaf, so that
   $\theta'$ has no sign change on the root's nodes. Both are located by the scan, and the value is
   honest.
3. **The nearly coalescing pair,** §2's family (C) at $\lambda = \Lambda_c/4$, at both values of
   $b$. The minimum is located and windowed, and the value is honest.
4. **No `theta_deriv`.** Test 1's problem with `theta` only. Nothing is located, the guard fires,
   and the value is honest.
5. **The floor-limited exit is barred.** A case where the parent marks a leaf across the point
   floor-limited (P1's row 1 at tight `rtol` is the model). Under the new code no window leaf or
   guarded leaf is floor-limited, and the value is honest. If no such case can be built on today's
   driver, the agent says why at §2's stop; the test then asserts the bar directly on a window leaf
   whose floor the exit's other conditions would accept.
6. **The cap.** Test 4 with `depth_max` small enough that a guarded leaf reaches it. Assert:
   - the flag is set;
   - `converged` is False;
   - the leaf is a Clenshaw–Curtis leaf, and its abserr is that solve's.
7. **The window's edges.** In test 1, $|\theta - \theta(x_s)|$ at each edge equals $\Phi_w$ to
   within the precision §1.2 states. Every piece's phase span is at most $\Delta\phi_w$.
8. **Nothing else moves.**
   - Run every problem in `test_levin_quadrature.py` that has no stationary point, with a
     closed-form `theta_deriv` where one exists. Nothing is located and no leaf is guarded.
   - Value, abserr and every returned leaf are bit-identical to the same call with location
     bypassed. That needs a private switch in the integrator: test-only, named as such, off by
     default, and read nowhere but the location (as `_test_only_bypass_roundoff_premise` is).
9. **The generic path.** A subclass of `_TwoIndependentSinCosBasis` in the new module, with the
   Clenshaw–Curtis methods the driver calls (`eval_basis_and_arguments`, `theta_prime_on_fine_grid`
   and `basis_derivative`) and §1.0's two. One phase is linear and the other quadratic,
   $\theta_2 = \tfrac{c}{2}(x - x_s)^2$ at a non-dyadic $x_s$. It is driven through
   `_adaptive_levin` directly.
   - The point is located on phase 2 and windowed, and the value is honest against the closed form
     (a Fresnel integral and an elementary one).
   - Phase 1 locates nothing.
10. **§1.3's rule.**
    - **(a) A flat window stops.** A call whose window leaves sit on a Clenshaw–Curtis floor above
      the request (§2.5 (A); a family (R) case at a tight `rtol` is the model). The window stops
      after six counted bisections, far short of `depth_max`; its leaves are floor-limited; the
      call is unconverged and honest.
    - **(b) Untrusted leaves at noise stop.** A call whose window leaves are found untrusted while
      their error is far inside the request (§2.5 (B); z12's shape is the model, built on a closed
      form). The window stops, the call is converged, and the value is honest.
    - **(c) It does not fire where leaves converge:** on test 1's window leaves no bisection
      counts.
11. **The records of `QuadSourceIntegral`.** A test in `ComputeTargets/tests/` that runs a group
    call through `phase_group_Levin_integral` on a stand-in, as the module's existing tests do, and
    asserts that the group record carries the four counts. No existing test pins the record's key
    set; this one does.
12. **Existing tests.** Every test in the seven roots passes.
    - **An existing assertion may change** only in a test whose problem holds a stationary point and
      supplies `theta_deriv`, so that location now fires: `test_levin_global_budget.py`'s test 12c
      is one. Only its asserted value, abserr, region, depth or solve counts may change, and each
      change is listed with its numbers under both codes.
    - An honesty assertion that would fail is a stop. A convergence assertion that would go from
      converged to unconverged is a stop.
    - The tests that put a stationary point in a call without `theta_deriv` exercise the guard; they
      pass unchanged.
    - **An assertion on the result's key set** may admit §1.5's new keys, and nothing else
      (amended at §2's stop, §9 point 3): `test_levin_trust_yardstick.py`'s
      `test_bit_identical_to_the_variant` is one. Every other field it compares stays
      bit-identical.

**Deliberate breakages,** each recorded in the log with the test that catches it, then reverted:
- **(i)** location, windows and guard all off: test 1 must fail;
- **(ii)** location off, leaving only the guard: test 2 must fail;
- **(iii)** the floor-limited exit allowed on window and guarded leaves: test 5 must fail;
- **(iv)** minima never located ($\Lambda_c = 0$): test 3 must fail;
- **(v)** a guarded leaf at `depth_max` returned as a Levin leaf: test 6 must fail;
- **(vi)** the window replaced by a plain split at $x_s$, with the leaves on either side unguarded:
  test 1 or test 7 must fail;
- **(vii)** location done only for the first phase a basis reports: test 9 must fail;
- **(viii)** the four counts left out of the group record: test 11 must fail;
- **(ix)** §1.3's counter never reaching its limit: test 10 (a) must fail.

**Amended 2026-11-08 (§10):** test 13 (a)–(d), Rule A's, and breakages (x) and (xi).

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 4. Records of the code

- `black` on the files changed, outside `calibration/scripts/` and `earlier-attempts/`.
- **§2's records** in `docs/levin-phase-floor/07/calibration/`: its scripts in `scripts/`, formatted
  by `black` before the run that is recorded, and its outputs in `out/`. Each output records the
  SHA-256 of its script and of `AdaptiveLevin/levin_quadrature.py` as it was measured, the `HEAD`,
  and that the tree was dirty. **If `levin_quadrature.py` changes after §2's recorded run** in
  anything but the two constants' values, their comments and §1.3's rule, §2 is re-run on the final
  code. A rung that then moves is a stop.
- **The earlier attempts' records,** copied as §0.5 says.
- `logs/07-stationary-point-guard.md`, in README §5.1's form. Its **Predictions and verdicts**
  holds:
  - §2's rules, with their tables and the constants as the user settled them, and §2.3's verdict;
  - §5's rules S0–S2 and §6's V0–V5, copied before Part B runs and marked not yet scored.
- The board's 07 row.
- **One commit.** The header's issue moves only in 07's records commit, after Part B.

## 5. Rehearsals R27 and R28 — on the Mac Studio, after the code commit is pushed

As 06m §4, with records under `var/levin-phase-floor-07/rehearsal/`, slugs `r27-parent` and
`r28-guard`, `--prompt 07`, and both fingerprints `1b332ffa…`.
1. **The lock:** `acquire`, then `set-pid` at each launch, and `release` after the readers. Each
   lock command is a shell command of its own (`CLAUDE.md`).
2. **R27** runs at the code commit's parent and **R28** at the code commit, each in a worktree, with
   06m's command line.
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
7. **The readers** run on R28, exit 0, and the store is byte-identical before and after.
8. **The lock** is released, and `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.
9. **The stores are kept** until the user says otherwise.

## 6. The census — on the Mac Studio, Ray-free, on A3 v2's scratch copy

06m's census re-run at the code commit, against 06m's run as the "before".
- **The harness** is 06m's, from `docs/levin-phase-floor/06m/census/scripts/`, copied into
  `var/levin-phase-floor-07/census/scripts/`. Change only:
  - the tree and its paths (`prov.py`, `refs_common.py`);
  - the diagnostics §1.5 adds, recorded per call and per returned leaf;
  - the scan: run on every group call at both rtols, and on every analytic call at `1e-7`, against
    the returned leaves;
  - the comparison with 06m's census, which sets the new keys aside;
  - V4's references.

  Save the diff against 06m's copy, and record each change as a deviation.
- **The tree** $g$ is a worktree at Part B's dispatch SHA, whose code is the code commit's.
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification, read-only, and hashed before and after. **06m's census**
  (`levin-phase-floor-06m-census-20261106T102153`, checkpoint `ea5bc3ab…`) is read only, and hashed
  before and after.
- **The units:** every row at `1e-7` and `1e-8`, in one registered detached run with slug `census`.
- **The budget:** 300 s per unit, with 06a's over-budget pass (a re-run once at 3,600 s, after both
  sets).

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

**Descriptive, not scored:**
- **Depth:** the distribution of row depth under $g$ at both rtols, against 06m's.
- **Windows:** window pieces per call, the windows' widths in $x$, and $\theta'^2/|\theta''|$ at
  their edges.
- **The stationary-point calls before:** for V4's set, $|{\rm value}_{06m} - {\rm ref}|$ against
  06m's abserr. That is how wrong the census at 06m's code was.
- **Totals:** how they moved against 06m in the 720, relative and against `total_abserr`$_{06m}$, by
  band and $k$.
- **Locations:** minima located and their $\lambda$; points at an end of a span; analytic calls that
  located a point; and `num_guarded_regions`, expected 0 since `theta_deriv` is supplied.
- **Convergence and the flag:** unconverged calls, in the 720 and outside it, with 06m's flag-cost
  table beside them (`[06m-calls-flagged-unconverged-within-their-request]`, 08's).
- **§1.3's rule,** if one is in: where it fired, and on which calls.
- **Cost:** group-call regions and wall time against 06m's. Cost is not an argument (README §5
  rule 4).

**The census's checkpoint and region records stay on the Mac Studio in full;** 08 compares against
them.

## 7. Acceptance

1. **§1–§4 as written:**
   - §2's tables reported; $\Lambda_c$, $\Phi_w$ and §1.3 settled by the user;
   - every test passing and the eight breakages caught;
   - every changed existing assertion of §3 test 12's kind, and listed;
   - the seven roots rising only by the tests added.
2. **§5:**
   - R27 and R28 both `done` with `1b332ffa…`;
   - S0 scored;
   - outside `QuadSourceIntegral` every value table identical;
   - inside it every row that located and guarded nothing bit-identical, and the rest tabulated;
   - S1 and S2 scored;
   - the readers exit 0, and the lock is released.
3. **§6:** V0 holds; V1–V5 are each scored holds, fails or undecided against the rule as written,
   with their numbers; the store and 06m's census are unchanged.
4. **Records:**
   - `logs/07-stationary-point-guard.md`, with Part B's Mac Studio record appended by the
     orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §7, "The stationary-point guard";
   - the records in `docs/levin-phase-floor/07/`.

## 8. Stop conditions — stop and ask

- **The Gate** fails.
- **§2 is reached:** the planned stop, always taken. Separately: no rung qualifies under either of
  §2.2's rules; §2.1's two references for (R) disagree; or §2.3's replay check does not reproduce a
  capture, unexplained.
- **A test cannot be built as written:** the parent does not show test 1's failure, and §3 test 5's
  fallback does not apply.
- **The construction reaches outside its files.** It needs a change outside
  `AdaptiveLevin/levin_quadrature.py`, the test modules, `docs/levin-phase-floor/07/` and §1.6's
  records: to `phase_groups.py`, `LiouvilleGreen/`, the partition, the eq. (151) formula,
  `_cc_roundoff_floor`, the trust rule, $\nu_B$, `DEFAULT_LEVIN_MAX_DEPTH` or `rtol`. Or it needs
  the location in `adaptive_levin_sincos` rather than in the generic code.
- **The set-aside work** would have to be changed, applied or dropped (§0.5).
- **A test moves** beyond §3 test 12's allowance; a test in `LiouvilleGreen/tests/` fails; a seven-roots
  count falls.
- **The rehearsals:**
  - R27 or R28 does not give `1b332ffa…`;
  - outside `QuadSourceIntegral` a value table differs between R27 and R28;
  - inside it, a row that located and guarded nothing is not bit-identical.
- **The census:** V0 fails; a unit ends in an error (not over budget); V4's references project
  beyond a day.
- **Any store file** changes, or 06m's census does.
- **The lock** is held, or will not release.

## 9. Amendments at §2's stop

1. **§2.1's runs move from `rtol` $10^{-10}$ to $10^{-9}$, with $10^{-8}$ as the fallback**
   (2026-11-07). At $10^{-10}$ no rung of $\Phi_w$ qualified. The only failing (R) case,
   $c = 10^8$, $x_s = 0.7236\ldots$, $\omega = 0$, is honest at every rung but ends `node_limited`
   at $2\pi$ and $8\pi$. Its request, $1.4\times10^{-15}$ against $|I| = 1.4\times10^{-5}$, is about
   $6\varepsilon$ of the integrand's scale. The orchestrator measured it at rtol $10^{-7}$ to
   $10^{-10}$, at the parent and at $2\pi$, $8\pi$ and $32\pi$
   (`docs/levin-phase-floor/07/investigations/cc-midpoint-shift/out/rtol_sweep.txt`): all 16
   calls are honest, and every one converges except the three windows at $10^{-10}$. What sets the
   floor there is a Clenshaw–Curtis node shift that 07's non-dyadic window edges expose and the
   parent's dyadic leaves on $[0, 1]$ escape (`[07-cc-interior-nodes-shift-with-a-rounded-midpoint]`).
   The user: *"For `Phi_w`: At the moment it's looking like we should re-run §2.2 with `rtol =
   1e-9` and proceed for there. If that fails, then relax to `rtol = 1e-8`, but hopefully `1e-9`
   works across the board."*
   - **What is re-run:** all of §2, on the code with §1.3 built (point 4): §2.2 at the new rtol,
     both families and both rules, with the parent's runs for context; §2.3 and §2.4 at their own
     captured arguments and tolerances, which this does not change; and §2.5 for every call that
     does not converge or reaches `depth_max`. The first run's outputs are kept beside the new ones,
     not overwritten.
   - **Unchanged:** the ladders, the rules, "honest" and "converged", and §8's stop when no rung
     qualifies at the last rtol tried.
2. **§1.3's rule is QUADPACK's `iroff1`, counted per window** (2026-11-07). The user had ruled it in
   on 2026-10-30 (Decisions, **07 fourth stop**: *"Let's go for (b)"*), to be written after §2
   measured the window leaves. §2.5 found them chased to `depth_max` two ways, (A) on a flat
   Clenshaw–Curtis floor and (B) untrusted at the integrand's noise, and found that `iroff1`
   counts on both. (B) is what keeps z00 and z12 from finishing at every rung. The user, on the
   agent's reading that the rule would return some window leaves unconverged: *"§1.3: did we not
   plan to implement the QUADPACK-style `iroff1` counter for this issue? The cost is that we have
   some leaves marked unconverged (if I understood). Is that really a problem?"* It is not: a
   stopped window's leaves are returned floor-limited, declaring what they carry, as the first
   text of this prompt (its §9 point 8) said. On (A) the call is unconverged because its request is
   below the floor, as at the parent; on (B) it converges when its budget is met. §1.2 item 6, §1.3,
   §3 test 10 and breakage (ix) say so.
3. **§3 test 12 admits §1.5's new keys in a key-set assertion** (2026-11-07).
   `test_levin_trust_yardstick.py`'s `test_bit_identical_to_the_variant` (06m's test 2) asserts that
   the result's key set equals 06l's variant's, so §1.5's keys fail it on all nine cells; every
   other field is bit-identical (the log's deviation 15). The user: *"It's absurd not to allow new
   keys that were added specifically because they were needed."* The test admits exactly those
   keys, and keeps every other comparison.
4. **The order after §2's stop: §1.3 first, then all of §2 once, then §3 without a further stop**
   (2026-11-07). The orchestrator offered two orders: build §1.3, re-run §2 once on that code and go
   on; or re-run §2.2 alone first and stop again before §1.3. The user: *"All looks good. Option 1,
   please."* So:
   - **§1.3 is built** on §1's code, with the provisional constants, before any §2 run.
   - **§2 is re-run in full** on that code, as point 1 says, and its rules set $\Phi_w$ and
     $\Lambda_c$. Each becomes a named module constant whose comment cites the measurement (§2.2).
     That run is the recorded §2; the module's SHA-256 in its outputs is the committed file's, but
     for the two constants' values and comments.
   - **The agent goes on to §3 without a further stop** if a rung qualifies under each of §2.2's
     rules and all 14 of §2.3's calls are honest at the chosen rung, z12 first. Otherwise it stops
     and reports, as §2.6 says. §2.6's report is written into the log either way.
   - §8's other stop conditions stand, and §2.5's report of windows stopped while a leaf's error
     was still falling (§1.3) is a finding for the user, not a stop.

## 10. Amendment at Part B's first stop

1. **Rule A: a phase whose variation is below the C2 gate's threshold is not located**
   (2026-11-08). This is a second code commit for 07, on the laptop, before Part B goes on.
   - **What R28 found** (the Mac Studio orchestrator's stop-1 report; the board's
     `[07-location-on-an-identically-zero-rate]`):
     - **The case.** 26 `q-r` group calls, in rows with $q = r$, each "located" 4,097 points, one
       for every scan sample. They come from the regime in which $G$ is smooth and $T_q$, $T_r$
       oscillate (`ComputeTargets/phase_groups.py`: phases $\theta_q \pm \theta_r$). On $q = r$,
       $\theta_q - \theta_r$ is exactly 0.0, and so is its rate, at every sample.
     - **The cause.** §1.1 step 2 takes each exact-zero sample as a root. With $\theta$ constant,
       every window is the whole span; they merge into one window, cut at every root. So each call
       was integrated as 4,102–5,505 forced Clenshaw–Curtis pieces, about 110 s per call, against
       1–68 regions at the parent. Summed regions in the store went from 114,826 to 227,883.
     - **It cost work, not honesty.** The integrand is $f_{\cos}$, smooth and not constant, of
       order $G M_q^2/H^2$. On three referenced calls, by composite Gauss–Legendre at five
       resolutions with QAGS as a second rule, R28 is at 0.179, 0 and 0.069 of its declared
       error. R27 is at 0.0021, 0.0014 and 0.0006 of its own, which is $10^4$–$10^6$ times larger.
     - **Why the C2 gate did not catch it.** The gate would send the whole span to the
       Clenshaw–Curtis rule, since its phase span is 0. But location runs first, before the loop,
       and does not consult the phase's variation.
   - **The user,** after the orchestrator explained the case and two candidate rules: *"Yes, rule A
     alone, please."* Then: *"Please make the fix as an amendment of 07."* The second candidate, a
     run of exact-zero samples treated as one flat stretch, was dropped. The partly flat phase it
     addressed is recorded instead as `[07-location-cuts-a-flat-stretch-at-every-zero-sample]`.
   - **Rule A (§1.1, a new first step for each phase).**
     - Estimate the phase's total variation over `x_span` from the scan's samples of its rate:
       $V_i = \int|\theta_i'|\,dx$ by the trapezoidal rule on the 4,097 samples, skipping an
       interval with a non-finite end, as step 2 skips such samples. This costs no evaluation
       beyond the scan. The log states the estimate's error.
     - **If $V_i <$ `SIX_PI`,** the C2 gate's own constant named as such, never a literal, phase
       $i$ is not located: steps 2–4 do not run for it, and it gives no points, no windows and no
       unlocated minima.
     - **Why the gate's threshold.** Below it, the gate already sends a region of that phase to
       the Clenshaw–Curtis rule, which needs no protection from a stationary point. The gate
       estimates each region's span from its own nodes, which can exceed $V_i$. A region it still
       sends to the Levin solve, holding a point this rule left unlocated, is the nodal guard's to
       catch (§1.4), as for a basis without rates.
     - **Reported.** The result gains one key listing each phase Rule A skipped, with its index and
       $V_i$. The key's name is the agent's choice, recorded in the log. `stationary_points` and the
       four counts are unchanged in meaning.
   - **The guard (§1.4) for a skipped phase.**
     - **One phase.** For `_Basis_SinCos`, and any basis whose guard reads only
       `theta_prime_Cheb`, the guard is unchanged.
     - **Several phases.** For a basis that passes several rates to the guard, a phase Rule A
       skipped is tested for a sign change, meaning samples of both strict signs on the nodes,
       rather than for "not of one strict sign". Otherwise a rate identically zero would mark every
       Levin leaf guarded and drive it to `depth_max`, while a root the rate holds would still be
       caught.
     - **Every phase located** keeps §1.4's test exactly.
   - **New tests (§3 test 13),** in `test_levin_stationary.py`. Each asserts honesty as §3 says,
     against a closed form.
     - **(a) A zero rate.** `_Basis_SinCos` with $\theta \equiv 0$, `theta_deriv` $\equiv 0$, and a
       smooth, varying $f_{\cos}$ of the shape of R28's calls. Nothing is located, no window piece is
       made, and the phase is reported skipped. Value, abserr and every returned leaf are
       bit-identical to the same call with location bypassed (test 8's switch).
     - **(b) A small phase with a root.** $\theta = \tfrac{c}{2}(x - x_s)^2$ at a non-dyadic $x_s$,
       with $c$ chosen so that $V < 6\pi$. Nothing is located. The result is bit-identical to the
       bypassed call, and the value is honest.
     - **(c) The same phase with $V$ just above $6\pi$.** The root is located and windowed as
       before.
     - **(d) Several phases.** Test 9's two-phase basis with phase 2 identically zero and phase 1
       linear with $V_1 \ge 6\pi$. Phase 2 is reported skipped, no leaf is guarded, and the call
       converges and is honest.
   - **New breakages,** each recorded with the test that catches it, then reverted:
     - **(x)** Rule A off (the threshold 0): test 13 (a) must fail;
     - **(xi)** a skipped phase tested by the guard for "not of one strict sign": test 13 (d) must
       fail.
   - **Test 12's key-set allowance** (§9 point 3) admits Rule A's key too, and nothing else.
   - **What must not move.**
     - **§2, re-run on the amended code:** §2.2's families at the recorded rtol, §2.3's 14 calls and
       §2.4's draws.
       - Every call with no phase skipped is bit-identical to the recorded run
         (`docs/levin-phase-floor/07/calibration/out/run2/`).
       - Every call with a skipped phase is listed, with its result under both codes.
       - A rung of $\Phi_w$ or $\Lambda_c$ that moves is a stop. The outputs go in a new directory
         beside run 2's; nothing is overwritten.
     - **The seven roots,** before and after: a fall is a stop, and a rise equals the tests added.
       Any other test whose assertion moves is a stop.
   - **Records.**
     - The log gains a section for this amendment, in README §5.1's form, with its own deviations,
       numbered on from Part A's.
     - The board's 07 row, its §3 entry for the issue (moved to §4 as closed by this commit), and
       `docs/OPEN_ISSUES.md`.
     - **One commit.**
   - **Part B after it** (the orchestrator note's §B.6).
     - The R28 that stopped is kept, as the record of the zero-rate case. A rehearsal R29 at the
       amended code takes R28's place in §5, compared with R27 as §5 item 4 says, and with R28.
     - Against R28, R29 should differ only in Levin calls whose phase Rule A skipped and which
       located a point at R28. Those calls should equal R27's.
     - S1 and S2 are re-scored on R29, and the census runs at the amended code.
     - **V3 sets aside a phase Rule A skipped:** its calls are counted and listed, with any sign
       change the scan finds in them; V2 still covers every returned Levin leaf.
