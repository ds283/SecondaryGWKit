# Prompt 06k — three ways to stop refining the slow leaves, measured on A3 v2 (a measurement)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-04, on the user's decisions after 06j (the board's Decisions, **06k**):
- after the analysis of what the slow leaves do to a call's value and error bar: *"I don't see any
  reason not to make all three measurements"*;
- on its scale: *"Let's do Tier 2 for all three routes, and see where that gets us"*.

It runs **before any remedy prompt** for
`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`, and before 07 resumes. Its
results decide which route, if any, a remedy prompt implements.

**Where it runs:** on the Mac Studio, Ray-free, with no lock, on 06's certified scratch copy of
A3 v2, read-only. **It changes no code in the tree** and builds no store. Each route is a scratch
variant of `AdaptiveLevin/levin_quadrature.py` kept outside the tree, as 06c measured its (b′) (06c's
log, deviation 41). One agent, dispatched and reviewed by an orchestrator session on the Mac Studio;
the records are committed on the laptop.

**Closes:** nothing. It measures three remedies for
`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`, and states for each what it
would do to convergence, honesty and cost.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entries for `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`
   (with its 06j narrowing), `[06c-budget-can-stop-on-a-pre-asymptotic-residual]` and
   `[06e-node-errors-add-in-absolute-value]`, and §4's `[06g-g-plus-r-rows-move-past-both-errors]`.
2. `AdaptiveLevin/levin_quadrature.py`:
   - `_LEVIN_TRUST_MAX_RATIO` and the comment above it;
   - `_adaptive_levin`: `_check_leaf` (the trust check one level down: $\Delta_1$, $\Delta_2$, $r$,
     rule (b), the declarations), the stop test and `untrusted_refinable`, the round-off exit and
     its budget clause, `node_error_signed` and the running sums.
3. 06j's records, `docs/levin-phase-floor/06j/`: `out/results.md`, `out/partS_leaves.csv`,
   `out/partS_nodes.json`, `out/w_calls.csv`, `out/cmd/63_w_carrying_leaves.txt`, and `scripts/`
   (`probe_s.py`, `probe_s_nodes.py`, `replay_06j.py`, `x1_capture.py`, `check_captures.py`).
4. 06g's records, `docs/levin-phase-floor/06g/`:
   - `census/scripts/` (`harness.py`, `worker.py`, `run_census.py`, `launch.py`, `v8_06g.py`,
     `values_beyond06g.py`, `v8_slow_leaves.py`) and `census/out/` (`v8.md`, `v8_slow_leaves.md`);
   - `references/out/scores_stage3/scores.json` and `report.md` (the references for V1, V3 and V8);
   - `review.md`, for the census's cost, and `KEPT_ON_MAC_STUDIO.md`.
5. 06c's log, deviations 41–46, for how a scratch variant of the module was loaded and run.

**Gate:**
- 06j's records commit (`6e8c2559`) is on `origin/handover-remedial`;
- the Mac Studio's main checkout is clean, and `runregistrykit list` shows nothing `running` or
  stale;
- the scratch copy matches its certification by SHA-256, and its fingerprint is `0e32f24a…`;
- 06g's census checkpoint is `03adcbba…` with its 16 region pickles, and `census/out/v8_calls.json`
  is `1e9a62b0…`;
- the Mac Studio has at least 40 GB free (06g's census run directory is 4.5 GB; three more are
  made).

---

## 0. What this prompt is

### 0.1 The problem, and why it matters

06g's rule (b) trusts a checked leaf if $r < 1/4$, or if $r < 1$ and $\Delta_1 \le \phi_P$. Outside
the 720 stationary-point rows, 801 calls at `1e-7` and 830 at `1e-8` lose convergence because slow
leaves at $r \approx 1/2$ are refined to `depth_max` and returned untrusted. 06j found what they are:
**rounding that scales with the leaf's width**, not a non-smooth point.
- On 75 of 142 sampled leaves ($\Delta_1/\phi_P < 2$), $\Delta_1 = |E_P - (E_L + E_R)|$ to 0.6 %:
  the difference of three solves' signed node errors. The noise clause compares it with $\phi_P$,
  which declares only $|E_P|$.
- On the other 67 ($2 \le \Delta_1/\phi_P \le 25$), the node errors explain about half of
  $\Delta_1$. The rest is inside the integrand's sensitivity to its inputs at the nodes, which no
  floor declares.

**What is at stake is cost and the `converged` flag, not accuracy.** The orchestrator read this from
06j's replays of its 24 lost calls, after 06j:
- the sampled slow leaves' declarations sum to $5\times10^{-10}$–$1.7\times10^{-3}$ of their call's
  abserr (median about $4\times10^{-8}$);
- on 06g's 12 referenced lost calls, 06g is within its error on 11; v03's 1.20 is not the slow
  leaves' doing.

So the calls' answers are right and their flags say otherwise. The census took 2.2× 06h's wall,
with 4–5× the group-call regions.

**The constraint any remedy must meet** (06j's Part W): 06h's trust test ($r < 1$) passed
pre-asymptotic leaves at $0.26 \le r \le 0.67$, with $\Delta_1/\phi_P$ from 5.8 to
$1.2\times10^7$. On 22 referenced calls it was 2.4–20× its declared error. Rule (b) refines every
one of those leaves, and a remedy must still refine them. They overlap the slow leaves fully in $r$;
on the $\Delta_1/\phi_P$ axis they meet only at c07's 5.8.

### 0.2 The three routes

Each route changes one thing in `_adaptive_levin` and nothing else. Each is measured alone, against
06g's code (the tree's).

- **Route 1 — the noise yardstick covers the three solves.** In the noise clause, $\phi_P$ is
  replaced by $\nu_1 = \phi_P + \phi_L + \phi_R$, each the solve's own whole floor (`phase_err`,
  its $|E|$ included). The leaf is trusted if $r < 1/4$, or if $r < 1$ and $\Delta_1 \le \nu_1$.
  - **Basis:** $\Delta_1 = |V_P - (V_L + V_R)|$ is a difference of three solves, and each solve's
    rounding is bounded by its own floor. So the rounding in $\Delta_1$ is bounded by their sum, not
    by $\phi_P$ alone.
  - Unchanged: the value returned, the declarations and everything else.
  - 06j's tables read with $\phi_L + \phi_R \approx \phi_P$ trust all 75 of the first population
    and none of the second. Part L measures it with the halves' floors exactly.
- **Route 2 — the differences are taken from node-corrected values.** Each of the seven solves'
  values is corrected by $-E$ (its `node_error_signed`; a solve without one is corrected by 0)
  before $\Delta_1$, $\Delta_2$ and $r$ are formed. The leaf is then trusted under rule (b) as
  written: $r_c < 1/4$, or $r_c < 1$ and $\Delta_{1c} \le \phi_P$.
  - A trusted leaf declares $\max(\text{residual}, \Delta_{1c})/(1 - r_c)$.
  - Unchanged: the value returned ($Q$ or $V_P$, uncorrected) and the call's signed node sum.
  - **Basis:** the correction removes the first-order node error that fixes $r$ at ½. What remains
    is measured against the floor 06g uses.
  - In 06j's tables the correction removes 99.8 % of $\Delta_1$ on the first population, but $r_c$
    still wanders (median 0.64). Read against $\phi_P$, 45 of 75 would be trusted.
- **Route 3 — an untrusted leaf too small to matter is left alone.** At each stop test, the
  untrusted refinable leaves are ranked by their declaration $D$, smallest first:
  - $D = \max(\text{residual}, \Delta_1)/(1 - r)$ if $r < 1$, else $\max(\text{residual},
    \Delta_1 + \Delta_2)$ — what the code declares today for an untrusted leaf at `depth_max`.

  The longest prefix whose $D$ sum to at most $\eta \times$ the call's request is **left**:
  - its leaves are not refined and do not block the stop;
  - each carries its $D$ in the call's error, and is reported as left.

  Every other untrusted leaf blocks the stop as before, and the call is converged if its budget is
  met. **$\eta = 10^{-3}$.**
  - **Basis:** a left leaf's declaration would have to understate its error by a factor of 1,000
    before the call's error reached its request.
  - The sampled slow leaves sit at a median $4\times10^{-8}$ of the call's abserr, which is at most
    its request.
  - The leaf that carried V2's error carried 2.4–20× the call's whole declared error.
  - It is a cut-off that ends refinement, so it starts generous and is reported in full. For every
    call, the smallest $\eta$ that would converge it is recorded, so that another $\eta$ can be read
    off the record without a rerun.

  This route needs no account of the second population.

**What none of them changes:**
- the leaf rules, $\phi$, the node sum;
- the round-off exit and its premise, `depth_max`;
- 06g's trust for any leaf with $r < 1/4$;
- anything outside `_adaptive_levin`.

### 0.3 What it does not do

- **No change to the tree.** Nothing under `AdaptiveLevin/`, `ComputeTargets/`, `LiouvilleGreen/`,
  `CosmologyModels/`, `ObjectFactories/`, `config/`, `tools/` or `main.py` changes, and no test file
  is edited. The variants live under `var/levin-phase-floor-06k/scripts/variants/` and are loaded
  in place of the tree's module by the harness, in its own processes.
- **No remedy.** The results state what each route does. Which one becomes code is for the user.
- **Not the stationary-point rows.** They are reported apart, and not scored (07).
- **No new references.** It scores against the references 06g and 06j already made.

## 1. The harness

Records go under `var/levin-phase-floor-06k/{out,scripts,state}`, with `log.md`, `verification.md`
and `issues.md` beside them (README §4.1).
- **The tree** is a worktree at this prompt's dispatch SHA. Its production code is 06g's
  (`3ae6f3a`): `git diff --stat 1c5336e <dispatch SHA> -- . ':!docs' ':!prompts'` shows only
  `tools/filon_reference.py`, its test and `CLAUDE.md`.
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification and fingerprint, read-only, and hashed before and after.
- **The scripts** are copied from `docs/levin-phase-floor/06g/census/scripts/` and
  `docs/levin-phase-floor/06j/scripts/` into `var/levin-phase-floor-06k/scripts/`. Save the diff
  against their sources, and record each change as a deviation.
- **The variants.** Each route is a copy of the tree's `AdaptiveLevin/levin_quadrature.py` with its
  change:
  - **The diff is confined.** Each variant's diff against the tree's file is saved, and an AST
    comparison shows that only `_adaptive_levin` differs, and within it only what §0.2 names.
  - **Each variant counts its route's firings:** a leaf trusted, or left, that 06g's rule would
    have refined. It reports the count per call in the result, under a new key that the
    comparison ignores.
  - **The identity variant** is a byte-identical copy of the tree's file, loaded the same way. It is
    what Gate K0 runs.
  - **The loading must be transparent.** The census harness wraps module functions (the premise
    counter among them), and the wrappers must still see every call under a variant. Gate K0
    checks this, counts included.
- **Long runs.** Anything that may exceed about ten minutes is registered with `runregistrykit`
  (`--campaign levin-phase-floor --prompt 06k`) and launched detached (README §5 rule 7). At most
  six workers.
- **A census run is checked, not cut off.** A run past its projection is reported with its progress
  and projected finish. It is stopped only if the projection is a day or more, counting the units
  in flight.

**Gate K0, before anything is scored:**
- **The identity variant is transparent.** On 06j's 46 calls and 54 more units drawn by a seeded
  rule written beforehand (100 units in all), the census worker under the identity variant
  reproduces 06g's census records bit for bit. That covers value, abserr, regions, region arrays,
  `converged`, checked leaves and the harness's counters.
- **Each route variant fires only where it is meant to.** On the same units, every call on which
  the route's firing count is 0 is bit-identical to 06g's record.

A K0 failure is a stop (§7).

## 2. Part L — the routes at leaf level (tier 1)

On 06j's 46 calls, re-captured as 06j captured them, each checked against 06g's census bit for bit.
06j's capture pickles were deleted on 2026-11-03 and are reproducible.

For each of the three routes, evaluate its trust decision in isolation, as 06j's `probe_s.py`
evaluated rule (b):
- **L-a:** on 06j's 142 sampled slow leaves, with the halves' floors $\phi_L$, $\phi_R$ computed
  exactly. For route 3, take the leaf's $D$ against $\eta$ times the call's request, alone.
- **L-b:** on the 06h leaves that carried each W call's error (`06j/out/cmd/63`: 17 distinct
  leaves on 22 calls), evaluated at their own span in 06g's code with each route's rule.
- **L-c:** on every returned leaf of the 24 S calls that 06g trusted, whether the route changes its
  decision (it should not, since the routes only add trust).

Report each leaf's $\Delta_1$, $\nu_1$, $\Delta_{1c}$, $r$, $r_c$, $D$ and decision per route, as
CSV.

**The LiouvilleGreen draws.** Run 06g's 20 seeded LiouvilleGreen draws (its Part A's test 9,
`docs/levin-phase-floor/06g/calibration/scripts/lg_seeded_draws.py`) under each variant, against
06g's code. Report per route:
- the Levin calls whose convergence changes;
- the cases that move between honest and dishonest, the near-singular
  `test_YJJ_log_singularity` cases ($\epsilon \le 10^{-4}$) apart from the rest.

## 3. Part C — the census under each route (tier 2)

Three registered census runs, one per route, each 06g's census (all 9,280 rows at `1e-7` and
`1e-8`, 18,560 units) re-run at the tree with the route's variant loaded. They run one after
another. The cost is projected before each, from 06g's (5 h 40 min) and 06h's (2 h 33 min).

**Against 06g's census, per route** (the comparisons are 06g's own analyses, `v8_06g.py`,
`values_beyond06g.py`, rerun with the route in the place of 06g's code):
- **Convergence:** of V8's lost calls outside the 720, how many converge, and what holds back
  those that do not: an untrusted leaf, `node_limited`, or something else. Report also calls that
  were converged under 06g and are not now.
- **Depth:** calls at `depth_max` (V5's count).
- **Moves:** every call outside the 720 whose value moves by more than $e_{\rm route} + e_g$ (V2's
  measure), with its row, group and the leaves that carry the move. Rows whose `total` moves past
  the sum of the two errors.
- **Firings:** per route, the number of leaves trusted or left that 06g would have refined, by
  depth, by $\Delta_1/\phi_P$ and by $r$.
- **Cost:** census wall, unit wall per rtol and group-call regions, against 06g's and 06h's.
- **The 720 stationary-point rows,** with the same counts, reported apart.

**Against the references, per route.** Score every referenced call, outside the 720 and apart
within it, as $|v_{\rm route} - {\rm ref}|/e_{\rm route}$ beside 06g's
$|v_g - {\rm ref}|/e_g$. The references are:
- 06g's 102 (V3's 90 and V8's 12, `references/out/scores_stage3/scores.json`, `rows`);
- V1's 12 (the same file, `V1`);
- 06j's 22 (`06j/out/w_calls.csv`).

Each reference is used as its source scored it, with its readings and class.

## 4. Predictions, fixed before the run

README §5 rule 3. Copy them into the log unchanged. Each is scored per route, **holds**, **fails**
or **undecided**, against the rule as written. None is a stop. In each rule, "outside the 720"
means outside the 720 stationary-point rows.

- **L1 — the route does not trust 06h's kind of leaf.** **Rule:** of the 17 06h carrying leaves
  (L-b), the route trusts or leaves none.
- **L2 — the route reaches the slow leaves.** **Rule:** the route trusts or leaves at least 80 % of
  06j's 142 sampled slow leaves (L-a).
- **C1 — convergence comes back.** **Rule:** at least 90 % of V8's lost calls outside the 720, at
  each rtol, converge under the route.
- **C2 — nothing converged is lost.** **Rule:** no call outside the 720 that 06g converges is
  unconverged under the route.
- **H1 — no referenced call gets worse.** **Rule:** on every referenced call outside the 720,
  $|v_{\rm route} - {\rm ref}|/e_{\rm route} \le \max(1,\ |v_g - {\rm ref}|/e_g)$.
- **S1 — no move past both errors.** **Rule:** no call outside the 720 moves by more than
  $e_{\rm route} + e_g$ against 06g's census.
- **K1 — routes 1 and 2 fire only on noise.** **Rule:** every leaf the route trusts that 06g
  refined has $\Delta_1/\phi_P < 5.8$ (c07's carrying leaf, the smallest of 06h's kind). Scored for
  routes 1 and 2. Route 3 decides by size, not by noise. Its firings are reported by
  $\Delta_1/\phi_P$ and by $D$ against the request, and K1 is not scored for it.

**The basis.**
- **L1, H1, S1 and K1** are the safety constraint of §0.1, stated four ways: at leaf level on the
  known bad leaves, on the references, census-wide, and on every firing. Route 3's safety rests on
  L1, H1 and S1 alone, and on $\eta$'s margin.
- **L2 and C1** are what the remedy is for. The 80 % follows 06j; 90 % leaves room for calls held
  by something other than slow leaves.
- **C2** holds by construction if the route only adds trust, and is a check that it does.

A route passes as a candidate remedy if L1, C2, H1 and S1 hold, and K1 for routes 1 and 2. L2 and
C1 say how much it recovers. Every number behind each verdict is reported, so that a different
reading can be made from the record.

## 5. Records

- **`var/levin-phase-floor-06k/log.md`** in README §5.1's form, with every deviation classified;
  `verification.md`; `issues.md` with any observation proposed as an issue. The laptop orchestrator
  copies the log to `prompts/levin-phase-floor/logs/06k-slow-leaf-trust-routes.md`.
- **`out/`:**
  - the variants' diffs and AST checks;
  - K0's checks;
  - Part L's tables and the draws' tables;
  - each census's comparison tables (CSV or JSON) and its scores against the references;
  - a `results.md`.
- **`results.md`** states, per route, each verdict with its numbers, and what the route would do
  as a remedy. It compares the three routes side by side, and proposes no code.
- **Provenance** in every output header (README §5 rule 5).
- **Kept on the Mac Studio:** the capture pickles, the three census run directories (each about
  4.5 GB, with checkpoints and region pickles), and anything over 2 MB, listed with SHA-256.

## 6. Acceptance

1. Gate K0 holds.
2. L1, L2, C1, C2, H1 and S1 scored for each route, and K1 for routes 1 and 2, against the rule as
   written, with their numbers.
3. Part L complete for every leaf; three census runs complete, each 18,560 units; §3's tables
   complete for each route.
4. The store and 06g's census checkpoint unchanged by SHA-256; the tree clean, with `--ignored`;
   nothing created under `var/datastores/`; no Ray started.
5. **Records:** the log, `verification.md`, `issues.md` and `out/results.md`;
   `docs/levin-phase-floor/VERIFICATION.md` §6k, written by the laptop orchestrator; the records in
   `docs/levin-phase-floor/06k/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **Gate K0** fails: the identity variant is not bit-identical on any unit, or a route differs
  from 06g where it never fired.
- **A variant cannot be confined:** a route cannot be written as a change inside `_adaptive_levin`
  alone, or needs a change to a leaf rule, to $\phi$ or to the call's result beyond its counter.
  Report what was tried.
- **A measurement cannot be made as written:** a leaf's route decision cannot be evaluated in
  isolation, or a reference cannot be matched to its call.
- **A census run's projection is a day or more,** counting the units in flight.
- **Any store file** changes, a change to the tree looks necessary, or Ray is needed.
