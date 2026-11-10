# Prompt 07a — the window's early stop, and the phase variation the C2 gate routes on

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-10, on the user's decisions after 07's records landed (the board's Decisions,
**07 results** and **07a**):
- on what follows 07: *"I agree we fix the early roundoff stop and the O9 problem. These are issues
  where the problem and the fix are both clear."*;
- on the early stop: *"Yes, replay the calls on the Mac Studio and then stop so that we can assess
  what to do."*;
- on O9: *"Yes, please write the O9 fix using the method you describe above: (B) for location, then
  route on |theta(b) - theta(a)| evaluated on regions where theta is monotone."*;
- on the census: *"If we need a full census for 07a, then we are going to be running it anyway. Is
  your question "can we re-use that as the 'before' for prompt 08"? If so, then I would think so,
  provided the code doesn't move significantly between."*

**Where it runs:** in three parts.
- **Part A, §2, the measurement:** on the Mac Studio, Ray-free, with no lock, in a worktree at the
  dispatch SHA. It replays captured calls at 07's code and at variants of the window's stop. **It
  ends at a planned stop:** the user chooses the stop's fix with its tables in front of them, and
  the orchestrator writes the choice into §1.2 as an amendment (§9), committed and pushed before
  Part B.
- **Part B, §1, §3 and §4, the code:** on the laptop, by a fresh-context subagent, in the main
  checkout at `HEAD`. One commit.
- **Part C, §5 and §6:** on the Mac Studio, after the code commit is pushed.
  - **§5, rehearsal R30,** under the lock, at the code commit, compared with 07's R29, which was
    built at the code commit's parent's code.
  - **§6, the census:** Ray-free, with no lock, on 06's certified scratch copy of A3 v2, against
    07's census as the "before". **This census becomes 08's "before"**, in place of 07's.

The records of Parts A and C are committed on the laptop afterwards: Part A's with §9's amendment,
Part C's in 07a's records commit.

**Closes, in its records commit, on the user's word:**
`[07-iroff1-window-stop-can-leave-a-call-dishonest]`, if §6's W4 and W6 hold and §5 holds. It also
**narrows** `[07-c2-gate-threshold-was-never-measured]`: O9 is removed from it, if §6's W3 and W5
hold. Whether $6\pi$ itself is right stays in that issue, which this prompt does not touch.

**Recommended model:** **Opus**, for every agent.

**Read first:**
1. **The board's §3 entries** `[07-iroff1-window-stop-can-leave-a-call-dishonest]` and
   `[07-c2-gate-threshold-was-never-measured]`, with every narrowing; and the Decisions rows
   **07 amendment lands**, **07 stop 1′**, **07 results**, **G2 L2, reopened** and **07a**.
2. **07's prompt,** [`07-stationary-point-guard.md`](07-stationary-point-guard.md): §1 (the
   location, the windows, §1.3's stopping rule, the guard), §9 point 2 and §10 (Rule A).
3. **`AdaptiveLevin/levin_quadrature.py`:**
   - `SIX_PI` and `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`, and the comments above them;
   - `_LEVIN_WINDOW_IROFF1_VALUE_RTOL`, `_ERROR_RATIO` and `_LIMIT`, and the comment above them;
   - `_Basis_SinCos.build_Levin_data`, which computes `phase_span`;
   - `_adaptive_levin_subregion_impl`: the C2 gate (`phase_span < SIX_PI`) and the choice of LU
     over least squares (`phase_span > _LEVIN_DIRECT_SOLVE_PHASE_SPAN`);
   - `_strictly_one_sign`, `_changes_sign`, `_nodal_guard` and `_oriented_cc`;
   - `_locate_stationary_points`, Rule A first;
   - in `_adaptive_levin`: the docstring's items on location, windows and §1.3; `_stop_window`;
     `_floor_limit_stopped_window_leaf`; the counting of bisections in a window; and the final
     `converged`;
   - `adaptive_levin_sincos`'s docstring on Rule A.
4. **QUADPACK's `dqage`** (netlib, `quadpack/dqage.f`), its counters `iroff1` and `iroff2`, the
   test `defab1.eq.error1.or.defab2.eq.error2` that skips them, and `ier = 2`; and `dqk21.f`, for
   when its error estimate equals `resasc`.
5. **The tests:** `AdaptiveLevin/tests/test_levin_stationary.py`, tests 10 (a)–(c) and 13 (a)–(d),
   and its `_TwoPhaseQuadraticBasis`; `test_levin_quadrature.py`'s
   `test_stationary_phase_gate_uses_total_variation` and its `_TwoIndependentSinCosBasis`.
6. **07's records,** `docs/levin-phase-floor/07/`:
   - `census/out/v4/report.md` and `v4_scores.json`, V4's five failures and their readings;
   - `census/out/xcheck/report.md`, the cross-check, with its two pairs at $r_g > 1$;
   - `census/log.md` and `verification.md`, and `VERIFICATION.md` §7;
   - `rehearsal/r29/`, R29 and its four guarded blocks;
   - `census/scripts/`, the harness: `make_capture.py`, `check_captures.py`,
     `xcheck_capture_07.py`, `worker.py` and `variant_loader.py` among them.

**Gate (Part A):**
- 07's records (`ea9bf326`) and the housekeeping (`b5405eb0`) are on `origin/handover-remedial`;
- the laptop's main checkout is clean, and `runregistrykit list` shows nothing `running` or stale on
  either machine;
- **the code is 07's:** `git diff adffa21b <dispatch SHA> -- AdaptiveLevin ComputeTargets
  LiouvilleGreen tools` is empty;
- on the Mac Studio, `var/levin-phase-floor-07/census/state/v4/capture/C00`–`C24` and
  `xcheck/capture/X00`–`X09` match the SHA-256 in 07's `census/KEPT_ON_MAC_STUDIO.txt`, and A3 v2's
  scratch copy matches its certification.

---

## 0. What this prompt is

Two defects that 07's Part B found, each with its cause known and its fix in reach. They are
independent, and they land in one commit.

### 0.1 The window's early stop

07's §1.3 ports QUADPACK's `iroff1` to the windows round each stationary point. A bisection of a
window leaf $P$ into $L$ and $R$ counts if $|(V_L + V_R) - V_P| \le 10^{-5}|V_L + V_R|$ and
$e'_L + e'_R \ge 0.99\,e'_P$. At six counted bisections the window stops: every leaf of it is
returned floor-limited, declaring its $e'$, and the call's `converged` follows the budget.

**What went wrong** (`[07-iroff1-window-stop-can-leave-a-call-dishonest]`): five of V4's 24
stationary-point calls are converged and 2.21–3.16× their declared error from agreeing readings.
Each stopped its one window while 4–13 of its leaves were still falling. Over the census, 1,153
windows at `1e-7` and 1,192 at `1e-8` stopped with a leaf still falling.

**Where the port departs from `dqage`.** The orchestrator checked it against the netlib source:
1. **`dqage` skips the count** for a bisection in which either half's error estimate is unresolved
   (`defab1.eq.error1.or.defab2.eq.error2`). 07 has no such skip.
2. **`dqage` counts on its raw error estimates.** 07 counts on $e'$, the error the budget sums,
   which can be a trusted parent's widened declaration (inference from the code, not measured).
3. **The value tolerance is $10^{-5}$ whatever the request.** It is `dqage`'s number, but the five
   calls are wrong by $1.5\times10^{-8}$ to $1.8\times10^{-5}$ relative, so a test this coarse may
   not see their errors.
4. **`dqage` reports failure:** at `iroff1` 6 it sets `ier = 2`, "roundoff error is detected, which
   prevents the requested tolerance from being achieved". 07 reports the call converged if the
   budget allows; all five failures are.

**What is not known** is where the five calls' excess error lies:
- **(H1)** in the stopped window's falling leaves, whose error estimates understate their error;
- **(H2)** in window leaves declared from a trusted parent's widened error;
- **(H3)** in the Levin leaves beside the window, which the stop does not touch.

So the fix is chosen at a planned stop, after §2 measures it. The default, if the measurement does
not argue otherwise, is to follow `dqage`: its skip, its raw estimates, and a call left unconverged
when a window is stopped with its request unmet.

### 0.2 O9: the phase variation the gate routes on

**The C2 gate** sends a region to nested Clenshaw–Curtis if its phase's total variation is below
$6\pi$, and to the Levin solve otherwise. The total variation, not the net change
$\theta(b) - \theta(a)$, because the two differ when $\theta'$ changes sign inside the region (prompt
03's C2). It is *estimated*, as the plain mean of $|\theta'|$ on the region's 16 Chebyshev–Lobatto
nodes times its width. In production $\theta$ itself is not evaluated on the nodes; `theta_deriv` is.

**Rule A** (07 §10) estimates the same integral a second way: by the trapezoidal rule over the whole
call on the 4,097-point scan. If that is below $6\pi$ the phase is not located, on the reasoning that
every region of it goes to Clenshaw–Curtis, which needs no protection from a stationary point.

**What is wrong:**
- **The gate's estimate is biased, by an amount that depends on the region's shape.** The nodes
  crowd to the ends, so the mean leans to $|\theta'|$ there. Measured on 16 nodes: 1.27–1.32× high
  for a near-centred root (test 13 (b) 1.2823; R29's two blocks 1.268, 1.270); up to 8.43× high with
  $|\theta'|$ weighted to one end; 0.47 of the truth on a narrow bump in the middle. So $6\pi$ has
  not meant one thing.
- **It is not monotone under bisection.** For a bump of $|\theta'|$ in the middle ($V = 10\pi$) the
  region reads $4.73\pi$ (Clenshaw–Curtis) while its half [0, 0.5] ($V = 5\pi$) reads $11.59\pi$
  (Levin). The gate decides afresh on every region the driver visits, so **a half of a region the
  gate sent to Clenshaw–Curtis can be sent to Levin.** Synthetic; how often in production is not
  measured.
- **Rule A's shortcut holds only if the gate agrees with it.** Where it does not, a region holding
  a root that nobody located goes to the Levin solve, and only the nodal guard stands between it and
  a wrong value. R29: 4 analytic J4/Y4 blocks ($V$ 15.11, 15.90; gate 19.16, 20.20). 07's census:
  28 guarded calls, and 154 with a skipped phase and a sign change, routed to Clenshaw–Curtis.

**Two requirements, independent** (the user: *"These seem completely orthogonal to actually
computing the right phase variation across the region."*):
- **(R1) Routing uses the right variation.**
- **(R2) Every region that goes to the Levin solve has its stationary points located and cut out,**
  whatever its variation.

**The fix, as the user chose it:**
1. **(R2) Location no longer depends on any estimate of the variation.** Rule A is replaced: a phase
   is not located only when there is nothing to locate, its rate exactly zero at every finite scan
   sample (the $q = r$ case Rule A was written for, `[07-location-on-an-identically-zero-rate]`).
   Every other phase is located: roots, exact zeros and minima, as 07's §1.1 steps 2–4.
   - **Not "no strict sign change".** A phase of one sign can hold a nearly coalescing pair, a
     minimum of $|\theta'|$ with $\lambda < \Lambda_c$, which 07 locates and windows (the 236
     lone-`G` calls per rtol locate 6 or 19 minima each). Skipping every phase without a sign change
     would drop them.
2. **(R1) The gate routes on the variation computed from $\theta$ itself.**
   - **On a region where $\theta$ is monotone,** $V = |\theta(b) - \theta(a)|$: exact, from two
     evaluations of the raw phase.
   - **After step 1 every region outside a window is monotone,** except where the scan missed a
     pair of roots between two of its samples. Located roots are window centres; window pieces are
     cut at them.
   - **Halves cannot read more than their parent.** On a monotone region the variation is additive,
     so a half of a Clenshaw–Curtis region always goes to Clenshaw–Curtis.
   - **Where the region's own nodes show $\theta'$ changing sign,** $\theta$ is not monotone there
     and $|\theta(b) - \theta(a)|$ can read near zero. There $V$ is the variation on the node
     partition, $\sum_j |\theta(x_{j+1}) - \theta(x_j)|$ over its nodes: exact where $\theta$ is
     monotone between adjacent nodes, a lower bound otherwise. A lower bound sends a region towards
     Clenshaw–Curtis, which is right whatever the stationary points do, at worst at the cost of
     bisections. Predicted never to fire in production with `theta_deriv` supplied (§6 W5).
3. **The same $V$ replaces the estimate everywhere** the estimate was used: the gate, the choice of LU
   over least squares at $20\pi$, and the `phase_span` reported in the result dicts.

**Unchanged:** `SIX_PI` and $20\pi$; the nodal guard (still the safety net for a pair between scan
samples, and for a basis without rates); the windows, $\Phi_w$, $\Lambda_c$ and the pieces.

**The precision.** $|\theta(b) - \theta(a)|$ for a raw phase near $10^{8}$ carries about
${\rm ulp}(\theta) \approx 1.5\times10^{-8}$ rad, and near $10^{12}$ about $1.2\times10^{-4}$:
negligible against $6\pi$. The log states the largest raw phase in production and its ulp.

### 0.3 What this changes, and what it does not

- **§0.1's fix** moves values, errors and convergence only in calls that have a window, by
  construction.
- **§0.2's fix moves every region whose route changes** between the old estimate and $V$, or whose
  Levin solve changes between LU and least squares. Calls with such a region move, with or without
  a window. **A call with no window and no such region is bit-identical** to 07's, with
  `phase_span` set aside (§6 W1).
- **Calls whose phase Rule A skipped** at 07 are now located, as R28 located them; their windows are
  new.
- **The `converged` flag** keeps its meaning, unless §9's amendment changes what a stopped window
  does to it.
- **No other module changes:** `LiouvilleGreen/`, `ComputeTargets/` and the datastore layer are
  untouched. No schema changes.

### 0.4 What it does not fix

- **Whether $6\pi$ is right** (`[07-c2-gate-threshold-was-never-measured]`). With $V$ exact the
  question can be measured; this prompt does not measure it.
- **The calls at the depth cap** (G2's L2, the board's **G2 L2, reopened**): 08's experiment.
- **The 786 undecided calls** of 07's cross-check: a reference sample, later.
- **A partly flat phase** (`[07-location-cuts-a-flat-stretch-at-every-zero-sample]`): unchanged.
- **The gate's sum over phases for a basis of several.** It stays as each basis combines its
  phases now, and is recorded in the 6π issue.

## 1. The code (Part B)

Every change is in `AdaptiveLevin/levin_quadrature.py`.

### 1.1 O9

1. **The location's first step,** in `_locate_stationary_points`, in place of Rule A: a phase whose
   rate is exactly 0.0 at every finite scan sample is not located, and is reported in
   `location_skipped_phases` with the reason. Every other phase runs steps 2–4 unchanged. The
   entry's form is the agent's choice; it no longer carries a variation.
2. **The guard's handling of a skipped phase** (07 §10, several phases): unchanged; it now applies
   to an identically zero rate only.
3. **$V$, in each basis's `build_Levin_data`,** per phase:
   - if the phase's rate on the region's nodes has no two samples of opposite strict sign
     (`_changes_sign` false), $V_i = |\theta_i(b) - \theta_i(a)|$, with $\theta_i$ the raw phase at
     the region's two end nodes;
   - otherwise $V_i = \sum_j |\theta_i(x_{j+1}) - \theta_i(x_j)|$ over the region's nodes in order;
   - the phases combined as the basis combines them now; the log states how, for each basis.

   For `_Basis_SinCos` the raw phase is `theta["theta"]`; it is sampled on the nodes only in the
   second case, and at the ends otherwise, through `_sample_vectorized`. The rate's samples are
   `theta_prime_Cheb`, whether from `theta_deriv` or by spectral differentiation. The two test
   bases, `_TwoIndependentSinCosBasis` and `_TwoPhaseQuadraticBasis`, follow the same rule.
4. **`phase_span` is $V$** wherever it was the estimate: the gate, the LU choice, `_oriented_cc` and
   the result dicts. The agent lists every use with its line, and says what each now receives.
5. **Docstrings and comments:** the module docstring's account of the C2 gate; `build_Levin_data`;
   `_adaptive_levin_subregion_impl` at the gate; `_locate_stationary_points`; `_adaptive_levin`'s
   items on location and the guard; `adaptive_levin_sincos`'s on Rule A; the comment above
   `SIX_PI`. Each says what is computed, why the old estimate was replaced (§0.2), and that $6\pi$
   is unchanged and unmeasured.

### 1.2 The window's stop

**Written at §2's stop, as §9's amendment, from the user's choice.** Until then this section is
empty, and Part B is not dispatched. Whatever is chosen:
- it touches only calls with a window, and only window leaves;
- it states its basis in `dqage` and §2's measurement, with its constants named and commented;
- a call whose window stops with its request unmet is not reported converged while its declared
  error understates what it carries.

### 1.3 What the result reports

No new keys are required. If §9's amendment adds one (for example a flag that a window stopped with
a leaf falling), its name is fixed there.

## 2. The measurement — Part A, on the Mac Studio, a planned stop

**Run at the dispatch SHA, whose code is 07's.** Nothing in the tree changes. Variants are copies of
`levin_quadrature.py` in the agent's `scripts/variants/`, loaded by 07's `variant_loader.py` pattern,
each with its diff against the tree saved. Records go in `var/levin-phase-floor-07a/measurement/`,
in `scripts/`, `state/` and `out/`, every output carrying the SHA-256 of its script and of the module
or variant it ran.

### 2.1 The calls

1. **V4's 25 captures,** `var/levin-phase-floor-07/census/state/v4/capture/C00`–`C24`: v00–v23 and
   z12 (`cal_z12`), each with its readings from 07's `census/out/v4/` (z12's is 06g's,
   `0x1.0c5f18f86f3efp-86`). The five failures are v08, v09, v12, v17 and v22; the other twenty are
   the controls.
2. **The cross-check's two pairs at $r_g > 1$,** `3-14501` and `3-35006`, index 1, `G+q+r`, at both
   rtols: four calls, captured afresh from A3 v2's scratch copy at the dispatch tree by 07's
   `make_capture.py`, each checked bit for bit against 07's census record by its
   `check_captures.py`. They have no reference; they are scored by the pair's
   $r = |v_7 - v_8|/(\epsilon_7 + \epsilon_8)$, as the cross-check scored them.
3. **The 396 class-(c) captures,** `xcheck/capture/X00`–`X09`: the stationary-point calls at `1e-8`
   where 07 and 06m disagree beyond their errors. Scored against variant (b) of §2.3 as a proxy
   (§2.4).

Each call is replayed in its own process. **A replay of 07's code that does not reproduce its census
record bit for bit** (value, abserr, `converged`, regions, depth) **is a stop.**

### 2.2 What is recorded, at 07's code

For every window of every call in §2.1 items 1 and 2:
- **each bisection of a window leaf:** $V_P$, $V_L + V_R$ and their relative change; the raw error
  estimates $e_P$, $e_L$, $e_R$ and the declared $e'_P$, $e'_L$, $e'_R$; whether it counted at 07;
  whether it would count under `dqage`'s skip and on raw estimates (§2.3 (c)); and the relative
  value change against $10^{-5}$ and against the call's request divided by $|v|$;
- **at the stop:** each leaf, whether it fell at its last bisection, its value, raw estimate and
  $e'$.

**The attribution (H1–H3), for the five failures and for every control with a stopped window.**
Split the call's error $v - {\rm ref}$ into three parts: the stopped window's falling leaves, its
other leaves, and the Levin leaves outside every window. Each returned leaf is referenced on its own
span by a method independent of the code under test, with a stated accuracy at least 100× finer
than the leaf's declared error. The method is the agent's choice and is recorded with its basis:
for example `mpmath` at extended precision on window leaves, which span at most $2\pi$ of phase, and
`tools/filon_reference.py`'s routines, called through its Python interface on a sub-span, for Levin
leaves. **The closure check:** the leaves' references sum to the call's reference within the
reading's tolerance. A call that fails it is reported, not attributed. For each part: its true error
against its declared error. **The cost is projected first,** from a pilot on one call; a projection
beyond twelve hours is reported to the orchestrator before the rest runs, with the calls it would
drop to fit.

### 2.3 The variants

Each changes only what happens in a window; the agent shows from the diff that a call with no window
runs the tree's code, and checks it on five such calls.
- **(a) 07 as is** (the replay).
- **(b) No stop:** the window leaves are refined like any leaf, to `depth_max`.
- **(c) `dqage`'s count:** a bisection is not counted if either half's error estimate is
  unresolved, and the count compares raw estimates, not $e'$. **The analogue of
  `defab.eq.error`** (in `dqk21`, the error estimate set to `resasc`, not yet in its asymptotic
  regime) is the agent's to define for a Clenshaw–Curtis leaf, with its basis.
- **(d) (c), with the value tolerance tied to the request:** the form, and its basis, are the
  agent's.
- **(e) 07's count, but `dqage`'s outcome:** a window stopped while its request is unmet leaves the
  call unconverged (`ier = 2`), values and errors unchanged.
- **(f), (g)** at most two more, each with a stated basis: for example, no count while a leaf of the
  window fell at its last bisection; or a stopped falling leaf declaring a widened error.

### 2.4 The tables, fixed before the run

For every call in §2.1 and every variant: value, abserr, `converged`, regions, maximum depth,
windows stopped, leaves falling at the stop, and wall time. Then:
1. **Honesty on V4's 25:** $|v - {\rm ref}|/{\rm abserr}$, against readings that agree (07's rule).
2. **The pairs:** $r$ for each pair.
3. **The 396:** $|v - v_b|/{\rm abserr}$, scored only where (b) converges with
   ${\rm abserr}_b \le {\rm abserr}/10$, the rest counted undecided.
4. **A variant qualifies** if no call is converged and dishonest (items 1 and 3 above 1), and each pair has
   $r \le 1$ or an unconverged member. Among those that qualify, the tables give the cost: calls
   unconverged, regions and wall time, against (a).
5. **The attribution** of §2.2, as a table per call.

### 2.5 Stop and report

The agent reports §2.4's tables, which variants qualify, the attribution, and anything it found
that §2.3 does not cover, and stops. **It recommends nothing.** Its log, in README §5.1's form,
is written before it reports, with every deviation numbered and classified. The user chooses; the
orchestrator writes §1.2 from that choice as §9's amendment.

## 3. Tests (Part B)

In `AdaptiveLevin/tests/test_levin_stationary.py`, and a new module
`AdaptiveLevin/tests/test_levin_phase_variation.py`, with no Ray and no datastore. References are
closed forms, contour integrals or subdivided `mpmath`, split at every stationary point. Every
honesty assertion checks both that the value is within its `abserr` of the reference and that
`abserr` is at least the true error.

**O9:**
1. **$V$ exactly, on a monotone region.** For `_Basis_SinCos` with and without `theta_deriv`, and
   each test basis, $V$ equals $|\theta(b) - \theta(a)|$ computed in the test, bit for bit, on
   several regions.
2. **The end-weighted shape.** $\theta = e^{\alpha x}$, with $\alpha$ chosen so that the true $V$ is
   just below $6\pi$ and the parent's estimate is above it. Under the parent the region goes to the
   Levin solve; under the commit to Clenshaw–Curtis. The parent's estimate and route are recorded
   in the comment.
3. **Monotone under bisection.** On the bump of §0.2 and two other shapes, every region the driver
   visits: no half of a Clenshaw–Curtis region goes to the Levin solve. At the parent, the case of
   §0.2 is shown to violate it, and recorded.
4. **A non-monotone region.** A region whose nodes show a sign change of $\theta'$, with no rates
   supplied (test 4's problem): $V$ is the node-partition sum, bit for bit, and the call is honest.
5. **O9's case.** Test 13 (b)'s phase, $\theta = 35(x - x_0)^2$ on [0, 1]: true $V = 18.475$, the
   parent's gate 23.69. Under the commit the root is located and windowed, no leaf is guarded, and
   the value is honest. At the parent it was skipped and guarded.
6. **Test 13, rewritten for the new rule:** (a) the zero rate is still skipped, with the reason;
   (b) becomes test 5 above; (c) unchanged; (d) the zero phase is skipped and no leaf is guarded.
7. **The minima stay located.** Test 3's nearly coalescing pair, whose rate has one sign: located
   and windowed, as at the parent.
8. **The LU choice** follows $V$: a region with true $V$ just above $20\pi$ and the parent's estimate
   below it (or the reverse) takes LU under the commit.

**The window's stop:** the tests §9's amendment specifies. At least one reproduces, on a synthetic
window, a stop with a leaf still falling that leaves the call converged and dishonest at the parent,
or, if no such case can be built, says why and asserts the rule directly.

**The LiouvilleGreen draws.** `test_3bessel_analytic.py` over seeds 0–19 by 07's harness, under the
parent and the commit: no test fails under the commit that passes under the parent; the number of
calls with a changed route or a window, and their honesty, tabulated.

**Existing tests.** Every test in the seven roots passes. An existing assertion may change only in its
value, abserr, region, depth or solve counts, in a test whose problem has a region whose route or LU
choice changes, or a window; each change is listed with its numbers under both codes. **An honesty
assertion that would fail is a stop; so is a convergence assertion that would go from converged to
unconverged,** unless §9's amendment makes that the intended outcome for that problem.

**Deliberate breakages,** each recorded with the test that catches it, then reverted:
- **(i)** the gate back on the mean-of-nodes estimate: tests 2 and 3 must fail;
- **(ii)** Rule A back (skip below `SIX_PI` on the scan): test 5 must fail;
- **(iii)** a phase skipped when its rate has no strict sign change: test 7 must fail;
- **(iv)** the node-partition sum replaced by $|\theta(b) - \theta(a)|$ on every region: test 4 must
  fail;
- **(v)** the LU choice left on the old estimate: test 8 must fail;
- the breakages §9's amendment lists for §1.2.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 4. Records of the code (Part B)

- `black` on the files changed.
- `logs/07a-window-stop-and-phase-variation.md`, in README §5.1's form, holding Part A's record (§2,
  with the user's choice) and Part B's. Its **Predictions and verdicts** holds §2.4's tables and
  the choice, and §5's and §6's predictions copied before they run, marked not yet scored.
- The board's 07a row.
- **One commit.** The issues move only in 07a's records commit, after Part C.

## 5. Rehearsal R30 — on the Mac Studio, after the code commit is pushed

As 07 §5, with records under `var/levin-phase-floor-07a/rehearsal/`, slug `r30-07a`, `--prompt 07a`,
`--job-name levin-phase-floor-07a`, and the fingerprint `1b332ffa…`. There is no new parent
rehearsal: **R29** (`levin-phase-floor-07-r29-guard-rule-a-20261108T123429`, the store
`var/datastores/levin-phase-floor-07-r29*`, kept) was built at `e182bc7a`, whose code is the code
commit's parent's.
1. **S0′ — R29 is the parent.** `git diff --stat e182bc7a <code commit's parent> -- . ':!docs'
   ':!prompts'` lists nothing the build imports. **Holds** if so; if not, a parent rehearsal R31 is
   run first and this prompt's comparisons use it.
2. **The lock:** `acquire`, `set-pid` at the launch, `release` after the readers, each a shell
   command of its own.
3. **R30 against R29,** by `tools.store_value_diff` with `--json` and 05's `classify_non_qsi.py`
   (with the job names normalised in a copy, as 07 did):
   - **outside `QuadSourceIntegral`:** every value table identical, every other difference of the
     four volatile kinds, no row in one store only;
   - **inside it, every moved row tabulated:** the change against R29's and R30's `total_abserr`,
     regions, maximum depth, located points, window pieces, guarded regions and `total_converged`;
     and the rows whose `total_converged` changes, in each direction.
4. **S3 — consistency.** Every row with finite `total_abserr` in both has
   $|{\rm total}_{30} - {\rm total}_{29}| \le {\rm total\_abserr}_{29} + {\rm total\_abserr}_{30}$.
   **Holds** if no row violates it. A violation is a finding, not by itself a stop.
5. **S1 and S2** re-scored on R30 as 07 §5 items 5 and 6, against 06g's references and 07's
   `p1_references`.
6. **S4 — O9's four blocks.** R29's four guarded analytic J4/Y4 blocks: in R30 each locates its
   root, guards no leaf, and is honest against R29's block references
   (`docs/levin-phase-floor/07/rehearsal/r29/out/block_reference/`).
7. **The readers** run on R30, exit 0, and the store is byte-identical before and after. **The
   lock** is released and `ray-processes` lists nothing. **R29 and R30 are kept.**

## 6. The census — on the Mac Studio, Ray-free, on A3 v2's scratch copy

07's census re-run at the code commit, against 07's run as the "before".
- **The harness** is 07's, from `docs/levin-phase-floor/07/census/scripts/`, copied into
  `var/levin-phase-floor-07a/census/scripts/`, the diff saved and each change a deviation. Change
  only: the tree and its paths; the comparison with 07's census, which sets `phase_span` aside; the
  diagnostics below; and V4's re-scoring.
- **The diagnostics,** an instrument that wraps the region solve and changes no computed value. For
  each call: the regions visited whose route (Levin or Clenshaw–Curtis) or LU choice would differ
  under 07's estimate, computed from the same nodes; the regions whose $V$ took the node-partition
  sum; and each Clenshaw–Curtis region with a half sent to the Levin solve.
- **The tree** $h$ is a worktree at Part C's dispatch SHA, whose code is the code commit's.
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification, read-only, hashed before and after. **07's census**
  (`levin-phase-floor-07-census-20261108T171137`, checkpoint
  `03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10`) is read only, and hashed before
  and after.
- **The units:** every row at `1e-7` and `1e-8`, in one registered detached run with slug `census`.
  The budget is 300 s per unit, with 06a's over-budget pass.

**Gate W0, before the run:**
- **The parent's code is 07's:** as S0′.
- **The harness reproduces 07's census at the parent,** bit for bit by 07's comparison, on 06k's
  100 K0 units. One registered run, slug `k0`.
- **The instrument changes nothing:** one unit with a window and one with a route change, at $h$,
  run with and without it, bit-identical.
- **The new code is deterministic:** two runs of one $h$ unit with a window, bit-identical.

**Predictions, fixed before the run** (README §5 rule 3), copied into `state/predictions.md`
unchanged, with §9's additions:
- **W1 — nothing else moves.** Every call with no window at 07 or at $h$, and no region whose route
  or LU choice differs, equals 07's call bit for bit, with `phase_span` set aside. Every row all of
  whose calls are such equals 07's in `total` and `total_abserr`. **Holds** if true at both rtols.
- **W2 — no Levin leaf is returned across a stationary point** (07's V2): the scan finds no returned
  Levin leaf holding a sign change of $\theta'$.
- **W3 — every phase with something to locate is located.** The location agrees with the scan both
  ways (07's V3), with no phase set aside except one whose rate is identically zero; every skipped
  phase is listed with its call, and each is of that kind.
- **W4 — honest where it is referenced.** V4's 33 calls (z00, z12, Z3's seven and the 24),
  re-scored from the census against the readings 07 scored them on, by 07's rule. **Holds** if
  every scored call is honest, or is unconverged under a rule of §9's amendment.
- **W5 — the gate is consistent, and the guard is a backstop.** In every call: no Clenshaw–Curtis
  region has a half sent to the Levin solve; no region takes the node-partition sum; no region is
  guarded (`num_guarded_regions` 0). **Holds** if all three are zero. A non-zero count is a finding,
  with its calls listed.
- **W6 — the pairs.** The cross-check's $r$ re-computed at $h$ over all 2,484 pairs: every pair has
  $r \le 1$, or a member unconverged under §9's rule. `3-14501` and `3-35006` are listed.

**Descriptive, not scored:**
- **Route changes:** the regions and calls that changed route or LU choice, by kind, band, $k$ and
  rtol, in which direction, and how far the old estimate stood from $V$.
- **Moves:** totals and calls against 07's, relative and against `total_abserr`$_{07}$.
- **The newly located:** calls whose phase Rule A skipped at 07, with their points and window
  pieces; any call locating more than 64 points.
- **The windows:** windows stopped, leaves falling at the stop, and calls left unconverged by the
  stop, against 07's 1,157/1,153 and 1,195/1,192.
- **Convergence and the flag,** in the 720 and outside, against 07's; and the depth distribution.
- **The disagreement list:** 07's cross-check list re-made at $h$ against 06m and against 07, by
  class.
- **Cost:** regions and wall against 07's. Cost is not an argument (README §5 rule 4).

**The census's checkpoint and region records stay on the Mac Studio in full: they are 08's
"before".**

## 7. Acceptance

1. **Part A:** §2's tables reported; the user's choice recorded; §9's amendment committed and
   pushed.
2. **Part B:** §1 as written and amended; every test passing and the breakages caught; every
   changed existing assertion listed; the LiouvilleGreen draws with no new failure; the seven roots
   rising only by the tests added.
3. **§5:** R30 `done` with `1b332ffa…`; S0′ holds; outside `QuadSourceIntegral` every value table
   identical; S1–S4 scored; the readers exit 0; the lock released.
4. **§6:** W0 holds; W1–W6 each scored holds, fails or undecided against the rule as written, with
   their numbers; the store and 07's census unchanged.
5. **Records:** the log, with Parts A and C appended by the orchestrator;
   `docs/levin-phase-floor/VERIFICATION.md` §7a; the records in `docs/levin-phase-floor/07a/`.

## 8. Stop conditions — stop and ask

- **The Gate** fails.
- **§2.5 is reached:** the planned stop, always taken. Separately: a replay at 07's code does not
  reproduce its census record; a capture of §2.1 item 2 does not check; a variant changes a call with
  no window.
- **The construction reaches outside its files:** beyond `AdaptiveLevin/levin_quadrature.py`, the
  test modules, `docs/levin-phase-floor/07a/` and the records. In particular `phase_groups.py`,
  `LiouvilleGreen/`, `QuadSourceIntegral.py`, `SIX_PI`, $20\pi$, $\Phi_w$, $\Lambda_c$, the trust
  rule, $\nu_B$, `DEFAULT_LEVIN_MAX_DEPTH` and `rtol` are not touched.
- **$V$ cannot be built as §1.1 says** for a basis: it lacks the raw phase, say. Report what is
  missing.
- **A test cannot pass as written,** or a breakage is not caught; an existing test moves beyond §3's
  allowance; a LiouvilleGreen test fails under the commit that passes under the parent; a
  seven-roots count falls.
- **The rehearsal:** R30 does not give `1b332ffa…`; outside `QuadSourceIntegral` a value table
  differs from R29's.
- **The census:** W0 fails; a unit ends in an error (not over budget).
- **Any store file** changes, or 07's census does, or R29 does.
- **The lock** is held, or will not release.

## 9. Amendments at §2's stop

*Empty until the user chooses.*
