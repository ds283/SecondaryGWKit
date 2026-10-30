# Prompt 06e — exact end nodes, and a node term that measures the displacement

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-30, on the user's decisions (the board's Decisions, **07 fourth stop**), after
07's Part A, rebased onto 06d, stopped a fourth time at §2, and after a leaf study of its stuck cases
and a check of 06c's production depth-20 calls:
- on the remedies: *"My first impression is that we probably want to land all of your 1, 2, and 3.
  But let's not make a definite decision until the Y1 reference job has finished"*;
- on their shape: *"No, let's just do a single prompt."*;
- on the third remedy, a QUADPACK-style rule for leaves whose error does not fall: *"Let's go for
  (b)"*, that is, **it is not in this prompt**. It becomes 07's answer to its window-leaf decision,
  where its cases are (§0.4).

It lands after 06c's records commit and **before 07 resumes**. 07's work, rebased onto 06d and
uncommitted, is set aside a fourth time, and rebased onto 06e afterwards. 07's rehearsals become
R21 and R22 (07 §9, to be amended).

**Where it runs:** in three parts, as 06c does.
- **§1–§3, the code:** on the laptop, by a fresh-context subagent, in the main checkout at `HEAD`.
  One commit.
- **§4, rehearsals R19 and R20:** on the Mac Studio, after the code commit is pushed, under the
  lock. R19 runs at the code commit's parent and R20 at the code commit, each in a worktree.
- **§5, the census:** on the Mac Studio, Ray-free, with no lock. 06c's census is re-run at the code
  commit, on 06's certified scratch copy of A3 v2, with 06c's run as the "before". **This run is
  also 07's "before".**

The records of §4 and §5 are committed on the laptop afterwards, in 06e's records commit.

**Closes, when §4 and §5 hold:** `[06c-chebyshev-end-nodes-miss-the-span]` and
`[06d-cc-node-term-is-pessimistic]`.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entries `[06c-chebyshev-end-nodes-miss-the-span]`,
   `[06d-cc-node-term-is-pessimistic]` and
   `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`.
2. `AdaptiveLevin/levin_quadrature.py`:
   - `_chebyshev_base` and `chebyshev_matrices`, and its docstring's claim that x[0] is the upper
     endpoint and x[-1] the lower;
   - every caller of `chebyshev_matrices`, in particular the Levin endpoint extraction in
     `_adaptive_levin_subregion_impl` and the fine grid in `_adaptive_levin_subregion_cc` ("node
     reuse": `fine_grid[::2]` is the coarse grid);
   - the comment above `_CC_NODE_ROUNDING_COUNT`, `_cc_node_error_bound` and `_cc_roundoff_floor`,
     and how `_adaptive_levin_subregion_cc` forms their arguments (`theta_prime_bound`,
     `f_prime_sample`);
   - `_adaptive_levin`: how a leaf is bisected (the midpoint, and whether both halves receive the
     same float), the step-(4) residual, the floor-limited test and the trust check.
3. `prompts/levin-phase-floor/06d-cc-roundoff-floor.md` §5 and `logs/06d-cc-roundoff-floor.md`, for
   the node term this prompt tightens and the tests it must keep passing.
4. The records this prompt rests on, committed with 06c's records:
   - `docs/levin-phase-floor/06e/investigations/07-leaves/out/tables.md`: the leaf study of 07's stuck cases;
   - `docs/levin-phase-floor/06c/depth20-tiling/results.md` and its `orchestrator_check/README.md`:
     the production depth-20 calls, the end nodes pinned by hand, and three references;
   - `docs/levin-phase-floor/06e/investigations/floor-breakdown/tables.md`: how 06d's floor divides between its terms.

**Gate:**
- 06c's records commit has landed (the board's 06c row is ✅);
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale on the laptop;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defects

**(A) The grid's end nodes miss the span.** `chebyshev_matrices` forms
$x = (a+b)/2 + (b-a)/2\cdot x_{\rm base}$. With $x_{\rm base} = \pm 1$ exactly, the end nodes
should be $b$ and $a$, but in floating point they miss by one ulp in about a third of spans
(1,249 of 4,000 random spans, checked by the orchestrator). The docstring says they are the
endpoints. So:
- **A Levin leaf integrates a shifted interval.** Its value carries
  $g(b)\,\delta b - g(a)\,\delta a$, about $|f|\,{\rm ulp}(x)$.
- **The step-(4) residual sees it** wherever the parent's ends and its halves' fail to tile, and the
  eq. (151) floor (about $10^{-19}$ on these leaves) does not. So the leaf is never floor-limited,
  its residual is flat in $h$, and 06c's loop chases it to `depth_max`.
- **Where the ends do tile, the offset is hidden**, and a floor-limited leaf declares up to
  $10^3\times$ less than its true error, at the $10^{-17}$ level (the leaf study).
- **In production** (`depth20-tiling`): 06c's census has 262 calls at depth 20 (109 at `1e-7`, 153
  at `1e-8`). Over their last five levels, 1,179 of the 1,222 Levin levels whose residual exceeds
  10× its floor have end nodes that do not tile; all 17 below that tile. On 30 sampled calls,
  setting only x[0] := b and x[−1] := a brings the residual within 10× of the floor on 127 of 138
  flagged levels, and the change matches the offset formula within 10% on 122:

  | 30 sampled calls | census (06c + 06d) | end nodes pinned |
  |---|---|---|
  | maximum depth | 20 on all 30 | 1–7 on 29; 20 on 1 |
  | regions / solves | 10,181 / 55,694 | 274 / 2,162 |
  | converged | 1 | 29 |
  | wall time | 91.6 s | 3.5 s |

  The one left at depth 20 is a `G+q+r` call whose span holds a zero of $\theta'$ at
  $z \approx 279.5$: 07's class, not this prompt's.
- **In 07's stuck cases** the same offset holds Levin leaves 50 to 35,000 radians from the
  stationary point flat at a true error near $10^{-16}$ (the leaf study, 301 of the 302 flat-residual
  leaves at $c = 10^6$).

**(B) 06d's node term is a worst case, 20–1000× above the error it bounds.** 06d's third term
$\sum_k |w_k|\,\delta\,\sum_i(\max|\theta'|\,|f_{ik}| + |f'_{ik}|)$ takes:
- $\delta = 5u(\max(|a|,|b|) + h)$ for every node, where the displacement measured is at most
  about $0.5\,{\rm ulp}(x)$ (at most $1.4u|x|$);
- the region's largest $|\theta'|$ at every node;
- absolute values throughout, where the displacements' signs are effectively random.

On the leaf study's Clenshaw–Curtis leaves, the term exceeds the true node error by 20–1000×, and
in 06d's floor breakdown it is 98.8–99.7% of the declared error. It is flat under splitting, so a
leaf it dominates is floor-limited at a level the true error lies far below, and the call's
declared error overstates its true error by as much.

### 0.2 The remedy

**1. Exact end nodes.** `chebyshev_matrices` sets x[0] := b and x[−1] := a after the affine map.
- The interior nodes and the differentiation matrix are unchanged, bit for bit.
- The docstring's claim becomes true. It says so, and why.
- Both rules get it: the Levin grid, and the Clenshaw–Curtis fine grid, whose `[::2]` stays the
  coarse grid.
- **The halves must share their parent's midpoint as one float**, so a parent and its halves tile
  exactly. The agent checks how `_adaptive_levin` bisects, and records it. If the halves do not
  receive the same float, stop (§7).

**2. A node term that measures the displacement.** For a Clenshaw–Curtis solve, replace the worst
case by the displacement actually committed.
- **$\Delta_k = \hat x_k - x_k$, signed, per node**: the computed node against the exact one,
  $x_k = (a+b)/2 + (b-a)/2\cdot\cos(\pi k/(n-1))$ in exact arithmetic. It is computed to a relative
  accuracy of a few $u$ of $\Delta_k$ itself, by error-free transformations (TwoSum, and TwoProduct
  by `math.fma` or Dekker's split) against the base nodes held in double-double. The base nodes'
  corrections are computed once per order and cached as `_chebyshev_base` caches its nodes. With
  the end nodes exact, $\Delta_0 = \Delta_{n-1} = 0$.
- **The first-order error the displacement commits**, with $g = \sum_i f_i\,b_i(\theta)$:
  $$E = \sum_k w_k\, g'(\hat x_k)\,\Delta_k,
  \qquad g' = \sum_i \bigl(f'_i\, b_i(\theta) + f_i\,\theta'\, b_i'(\theta)\bigr),$$
  with $\theta'$ evaluated **at each fine node**, not the region's maximum, and $f'$ the fine grid's
  Chebyshev derivative of the samples, as 06d has it.
- **The node term stays a bound:**
  $$\phi_{\rm node} = |E| + R_g + R_2 + R_c .$$
  - $R_g = \sum_k |w_k|\,|\Delta_k| \sum_i(\eta_\theta\,|\theta'_k|\,|f_{ik}| + \eta_f\,|f'_{ik}|)$
    bounds the error from using approximate $\theta'_k$ and $f'_{ik}$.
    - $\eta_\theta$ and $\eta_f$ are bounds on their relative errors, **derived** from how each is
      obtained.
    - If one cannot be derived, it is measured on named cells (§2 test 4) and taken at a stated
      conservative multiple of the worst measured. Where neither can be stated, it is 1: a
      triangle-inequality bound on the exact displacements.
    - The log states each $\eta$ and its basis.
  - $R_2$ bounds the second-order remainder,
    $\tfrac12\sum_k |w_k|\,\Delta_k^2 \max|g''|$, with $|g''|$ bounded from the same data.
  - $R_c$ bounds the rounding in computing $E$ itself.
- **No $1/\theta'$**, as 06d. $\phi_{\rm node}$ replaces 06d's third term in $\phi_{\rm CC}$; the
  first two terms are unchanged.

**3. Docstrings and comments**, per §0.3:
- `chebyshev_matrices`;
- the comment above `_CC_NODE_ROUNDING_COUNT` (or its replacement);
- `_cc_roundoff_floor`;
- the module docstring where it describes the fallback's floor.

### 0.3 What this changes, and what it does not

- **It changes:**
  - `chebyshev_matrices`'s end nodes;
  - the node term of `_cc_roundoff_floor`, and how `_adaptive_levin_subregion_cc` forms its
    arguments;
  - new private helpers for $\Delta_k$ and the cached corrections;
  - the docstrings and comments above.
- **It does not change:**
  - eq. (151) or `_roundoff_floor` on the Levin branch;
  - 06d's first two terms;
  - the declared-phase term;
  - 06b's premise or $\rho$;
  - 06c's loop, budget or trust check;
  - the C2 gate and its $6\pi$;
  - the nested Clenshaw–Curtis estimate (§0.4);
  - `depth_max`;
  - the result dict's keys;
  - any caller.
- **What moves: every computed value.** Every Levin and Clenshaw–Curtis solve on a span whose end
  nodes missed now integrates the span itself, so its value moves at the round-off level, and
  through the loop so do region lists, depths, `abserr` and `converged`. Every `QuadSourceIntegral`
  row moves (§4), and so does every LiouvilleGreen three-Bessel call.

### 0.4 What it does not fix

- **A leaf whose error does not fall under refinement** (`[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`)
  is not given a stopping rule here. On the evidence, once the end nodes are exact, production's
  remaining case is a stationary point inside a Levin span, and 07's stuck case is 07's own window
  leaves, which 07's §0.2 (ii) keeps from being floor-limited. **QUADPACK's rule**, round-off
  detected when bisection leaves the error flat a set number of times, is assigned to 07 as its
  answer for window leaves (Decisions, **07 fourth stop**).
- **Stationary points** inside a Levin span, including Y1's 12 equilateral rows at $z \approx 10211$
  (each reported converged and wrong by up to $1.6\times10^6$ × its error, by 06b's code or 06c's)
  and the `*-1-2174` call above, are 07's.
- **The nested Clenshaw–Curtis estimate** $|{\rm CC}_{31} - {\rm CC}_{16}|$ measures CC₁₆'s error
  while the value returned is CC₃₁'s. At a $6\pi$ span that is $2.7\times10^{-6}$ against
  $5.4\times10^{-17}$, so a leaf just under the gate overstates by many decades. Its own issue
  (`[06d-cc-nested-estimate-measures-the-coarse-rule]`), not this prompt's.
- **The Levin floor near a stationary point** (eq. (151) short per leaf through node rounding in
  $\theta'$) and **raw-phase endpoint rounding** ($u|\theta|$ at a leaf's ends, on calls without
  `theta_mod_2pi`) stay as they are, on their issues.

## 1. The code

1. **The end nodes**, as §0.2 item 1, in `chebyshev_matrices`.
2. **The displacement:** a private helper returning $\Delta_k$ for a span and an order, with the base
   nodes' double-double corrections cached per order.
3. **The node term:** $\phi_{\rm node}$ in `_cc_roundoff_floor` or a helper of its own, from
   $\Delta_k$, per-node $\theta'$, $f'$ and the basis's derivatives. How $\theta'$ reaches the fine
   nodes is the agent's choice, recorded with its cost per solve. Where it costs an evaluation, it
   costs at most one evaluation of $\theta'$ per fine node.
4. **Constants** named, each with its derivation in a comment, as 06d's are. A replaced constant is
   removed, not left unused.
5. **Docstrings and comments**, per §0.2 item 3.

## 2. Tests

A new module, `AdaptiveLevin/tests/test_levin_end_nodes.py`, no Ray, no datastore. References are
closed forms or `mpmath` at 30 digits. **Every test on an integral asserts honesty: the value within
its declared `abserr` of the reference.** The parent is 06d's module (`git show <dispatch
SHA>:AdaptiveLevin/levin_quadrature.py`, imported under another name). A test that quotes the
parent's numbers records them in its comment.

1. **The end nodes are exact.** 4,000 seeded random spans, with $|a|$ and $|b|$ from $10^{-3}$ to
   $10^{15}$ and relative widths from $10^{-12}$ to 1, at orders 2, 16, 24, 31 and 64:
   - x[0] == b and x[−1] == a;
   - the interior nodes and D are bit-identical to the parent's.

   The parent's miss rate is recorded.
2. **The leaves tile.** Over a seeded set of spans, bisected as `_adaptive_levin` bisects them: the
   parent's ends and its halves' outer ends are equal, and the halves share their inner end, bit
   for bit.
3. **The offset-driven chase is gone.** A problem with a closed form and **no stationary point** on
   which the parent's Levin residual is flat above its floor and the parent reaches depth 12 or
   more. An example is a non-stationary phase at $|x|$ of order 1 to $10^3$ with $\theta'$ of order
   $10^6$, as the leaf study's outside leaves.
   - Under 06e it converges below depth 8.
   - Both codes are honest.
   - The log records the parent's residual over floor along its deepest line.
   - If no such problem can be found on the parent, stop (§7).
4. **The node term bounds the node error, and is tighter.** The cells:
   - 06d's test 2, 20 cells raw and reduced;
   - 06d's test 3, the nested fallback leaves round a stationary point;
   - **window-like cells:** single Clenshaw–Curtis solves on spans within a few windows of a
     stationary point, at $c = 10^6$ and $10^8$, at the leaf study's depths and widths.

   In every cell, the solve is honest, and $\phi_{\rm node}$ is at least the true node error. The
   true node error is the rule evaluated in `mpmath` at the float nodes against the exact nodes, as
   06d §5 measured it.

   The log tables, per cell:
   - the true node error;
   - 06d's third term and $\phi_{\rm node}$;
   - $|E|$, $R_g$, $R_2$ and $R_c$ separately;
   - both ratios to the true error.
5. **The stationary-phase oracle** (06d's test 4): honest and converged. Leaves, depth and solves
   are recorded against 06d's 76 leaves.
6. **Every problem `test_levin_quadrature.py` drives:** honest under both codes, and
   $|v_{06e} - v_{06d}| \le {\rm abserr}_{06d} + {\rm abserr}_{06e}$. The log tables each.
7. **The LiouvilleGreen draws.** `test_3bessel_analytic.py` over seeds 0–19, as 06d's test 7, at
   06d's code and 06e's.
   - No test fails under 06e that passes under 06d.
   - The log carries the honesty table (dishonest, dishonest-and-converged, worst ratio) under both,
     with the `test_YJJ_log_singularity` near-singular cases ($\epsilon \le 10^{-4}$) reported
     separately, and converged counts and depth-20 calls under both.
   - The harness lives in `docs/levin-phase-floor/06e/calibration/scripts/`.
8. **Existing tests.** Every test in `AdaptiveLevin/tests/` and `LiouvilleGreen/tests/` passes. An
   existing assertion may change only if it is of 06d §2 test 8's kinds, or **a bit-identity or a
   quoted value that this prompt's end nodes move by no more than the declared errors allow**.
   Never an honesty or accuracy assertion. A convergence assertion may change only from unconverged
   to converged, with the result honest. Each change is listed with its numbers under both codes.

**Deliberate breakages,** each recorded with the test that catches it, then reverted:
- **(i)** the end nodes unpinned: tests 1, 2 and 3 must fail;
- **(ii)** $\Delta_k$ set to zero: test 4 must fail on a reduced-phase cell;
- **(iii)** $\theta'_k$ replaced by the region's minimum $|\theta'|$: test 4 must fail on a cell. If
  no cell can be made to fail, the log records why, and (iii) is caught by a direct test of
  $\phi_{\rm node}$ against the true node error on a constructed cell instead.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 3. Records of the code

- `black` on the files changed, outside `calibration/scripts/`.
- `logs/06e-exact-end-nodes.md`, in README §5.1's form. Its **Predictions and verdicts** holds §4's
  and §5's predictions, copied before they run and marked not yet scored, and P1 below, scored by
  test 4.
  - **P1 — the node term tightens.** Over test 4's window-like cells, the median of 06d's third
    term over $\phi_{\rm node}$ is at least 10. A failure is a finding, not a stop.
- The board's 06e row.
- **One commit.** The header's issues move only in 06e's records commit, after §4 and §5.

## 4. Rehearsals R19 and R20 — on the Mac Studio, after the code commit is pushed

As 06c §4. Records go under `var/levin-phase-floor-06e/rehearsal/`.
1. **The lock:** `acquire`, `set-pid` at each launch, `release` after the readers. Each lock
   command is a shell command of its own (`CLAUDE.md`).
2. **R19** runs at the code commit's parent and **R20** at the code commit, each in a worktree.
   - Each runs `RunDrivers/quadsource_atol_sweep.py --build --rehearsal` as R17 and R18 ran it.
   - Each is registered with `--campaign levin-phase-floor --prompt 06e`, with slugs
     `r19-parent` and `r20-endnodes`.
   - **Both fingerprints must equal `1b332ffa…`.**
3. **The value comparison** is `tools.store_value_diff` on R19 and R20, with `--json`, with 05's
   `classify_non_qsi.py` copied unchanged.
   - **Outside `QuadSourceIntegral`:** every value table is identical, every other difference is of
     the four volatile kinds, and no row is in one store only.
   - **Inside `QuadSourceIntegral`, every row is tabulated:** $|{\rm total}_{20} - {\rm
     total}_{19}|$ against `total_abserr` in each, the region counts, maximum depths and
     `total_converged` in each.
   - **Z1 — consistency.** **Rule:** every row outside 06a's 720 stationary-point rows with finite
     `total_abserr` in both stores has $|{\rm total}_{20} - {\rm total}_{19}| \le {\rm
     total\_abserr}_{19} + {\rm total\_abserr}_{20}$. Rows with an infinite `total_abserr` in either
     are counted separately. The stationary-point rows are tabulated, not scored. **Holds** if no
     row violates it. A violation is a finding for the user, not by itself a stop.
4. **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`) run on R20. Every one
   exits 0, and the store is byte-identical before and after.
5. **Release the lock.** `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.
6. **The stores are kept** until the user says otherwise.

## 5. The census — on the Mac Studio, Ray-free, on A3 v2's scratch copy

06c's census, re-run at the code commit, against 06c's run as the "before".
- **The harness** is 06c's census copy, from `docs/levin-phase-floor/06c/census/scripts/`, copied
  into `var/levin-phase-floor-06e/census/scripts/`. Change only:
  - the tree and its paths;
  - what Z2–Z5 need recorded per call.

  Save the diff against 06c's copy, and record each change as a deviation.
- **The tree** $g$ is a worktree at Part B's dispatch SHA, whose code is the code commit's.
- **The store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its
  certification and fingerprint, read-only, and hashed before and after. Nothing else in that
  directory is touched.
- **The units:** every row at `1e-7` and at `1e-8`, as in 06a–06c, in one registered, detached run
  with slug `census`.
- **The budget:** 300 s per unit, with 06a's over-budget pass.
- **Its checkpoint and per-unit records are kept in full on the Mac Studio.** 07's §6 compares
  against them.

**Gate Z0, before the run:**
- **The parent's production code is 06c's census's.** `git diff --stat <06c's Part B dispatch
  commit> <code commit's parent> -- . ':!docs' ':!prompts'` shows only `CLAUDE.md` and files whose
  ASTs without docstrings equal the dispatch commit's.
- **The harness reproduces 06c.** At the parent it reproduces 06c's units **bit for bit** on
  `low-G+r-1` and `ctrl-low-1` at `1e-7`, and on `0-1971` at `1e-8`.
- **The new code is deterministic.** Two runs of one $g$ unit are bit-identical.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "The
stationary-point rows" are 06a's 720 (`spscan.py`).
- **Z2 — the cap.**
  - **The rule:** outside the stationary-point rows, no call under $g$ reaches `depth_max`, at
    either rtol. An over-budget unit counts as false.
  - **Holds** if true at both rtols. 06c's census had 109 and 153 such calls among those that
    moved.
- **Z3 — honest where referenced.**
  - **The sample** is `depth20-tiling`'s 30 calls, matched by row, rtol and call index. Their
    references are computed by `orchestrator_check/orch_ref.py`'s method at three resolutions, as
    copied into `var/levin-phase-floor-06e/census/references/`. The samples:
    - 1, 9 and 25, already run, re-scored against $g$;
    - 2, 8, 10, 13, 14, 22, 23 and 27, each estimated at under 45 minutes;
    - 3 and 15, estimated at 2–2.6 h, in one registered run with slug `references`.

    The 18 estimated at 20 h or more are not run.
  - **A reference counts** only if its resolutions agree to within 1/100 of $\min({\rm abserr}_{06c},
    {\rm abserr}_g)$. Those that do not are reported with their scatter and dropped.
  - **The rule:** $|v_g - {\rm ref}| \le {\rm abserr}_g$ for every counted reference. The census
    side, $|v_{06c} - {\rm ref}|/{\rm abserr}_{06c}$, is reported beside it.
  - **Holds** if no counted reference violates it. Fewer than five counted is **undecided**, not
    holds.
- **Z4 — the rows agree.**
  - **The rule:** outside the stationary-point rows, every row with finite `total_abserr` under both
    has $|{\rm total}_g - {\rm total}_{06c}| \le {\rm total\_abserr}_{06c} + {\rm
    total\_abserr}_g$.
  - **Holds** if true at both rtols.
- **Z5 — budget misses converge.**
  - **The rule:** every Levin call outside the stationary-point rows that 06c reported unconverged
    is `converged` under $g$.
  - **Holds** if true at both rtols.

**Descriptive, not scored:**
- **Convergence and depth:** converged calls and rows, `total_converged`, and the depth distribution
  under 06c and $g$, at both rtols.
- **Cost:** regions, solves and wall time against 06c. Cost is not an argument (README §5 rule 4).
- **Where the budget went:** per call, the share of $E$ from residual, floor (and, for
  Clenshaw–Curtis leaves, each of its terms) and truncation.
- **Anything still deep:** every call outside the stationary-point rows past depth 12, with its
  residual over floor along its deepest line and whether its ends tile.
- **The stationary-point rows:** how they moved, and by name Y1's 12 equilateral rows and the
  `*-1-2174` call. They are 07's, and not scored here.

## 6. Acceptance

1. **§1–§3 as written:**
   - every test passing, the three breakages caught;
   - every changed existing assertion of the kinds §2 test 8 allows, and listed;
   - the LiouvilleGreen draws with no new failure;
   - the seven roots rising only by the tests added.
2. **§4:** R19 and R20 both `done` with `1b332ffa…`; outside `QuadSourceIntegral`, every value table
   identical; inside it every row tabulated and Z1 scored; the readers exit 0; the lock released.
3. **§5:** Z0 holds; Z2–Z5 are each scored holds, fails or undecided against the rule as written,
   with their numbers; the store is unchanged.
4. **Records:**
   - `logs/06e-exact-end-nodes.md`, with §4 and §5's Mac Studio record appended by the
     orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §6e, "Exact end nodes";
   - the records in `docs/levin-phase-floor/06e/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **The halves do not share their parent's midpoint** as one float.
- **A test cannot be built as written:**
  - test 3 finds no problem on which the parent chases;
  - $\phi_{\rm node}$ fails to bound a cell of test 4;
  - no basis can be given for an $\eta$ below 1 and the term with $\eta = 1$ is not tighter than
    06d's anywhere (report the numbers).
- **The construction reaches outside its file,** beyond test files and comments: to
  `ComputeTargets/`, `LiouvilleGreen/`, eq. (151) on the Levin branch, 06d's first two terms, the
  declared term, 06b's premise, 06c's loop or trust check, the C2 gate, the nested estimate, or the
  result dict's keys.
- **A test moves beyond §2 test 8's allowance,** a LiouvilleGreen test fails under 06e that passes
  under 06d, or a seven-roots count falls.
- **The rehearsals:** R19 or R20 does not give `1b332ffa…`, or outside `QuadSourceIntegral` a value
  table differs.
- **The census:** Z0 fails, or a unit ends in an error (not over budget).
- **Any store file** changes.
- **The lock** is held, or will not release.
