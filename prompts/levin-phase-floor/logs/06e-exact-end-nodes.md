# levin-phase-floor 06e — exact end nodes, and a node term that measures the displacement

> **§1–§3 only** (the code, the tests and these records), written by the executing agent on the
> laptop. §4 (rehearsals R19 and R20) and §5 (the census) run later on the Mac Studio; the
> orchestrator appends their record to this log.

> **One stop, resolved.** Part A stopped before committing under §7 ("a test moves beyond §2 test 8's
> allowance"): `test_levin_roundoff_exit`'s test 3 (b) asserted that the premise check at least halves
> the declared error, which no longer holds under 06e and is of no allowed kind. The user ruled the
> same day: *"Looks like (a) to me. The halving assertion looks to me like it was just made up, so no
> reason to worry about it."* — option (a), restate (b) as 06d restated (a), bit-identical across the
> bypass with honesty kept, and withdraw the halving assertion (Verification, test 8, item 3;
> deviation 18).

**Subject:** prompt 06e §1–§3, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `50277f3`; the code there is 06d's (`3d70db9`). `git log --oneline 50277f3..HEAD`
  was empty when checked. The stashes were not touched, and `var/levin-phase-floor-07/` was not read.
- **The gate** held at dispatch (checked by the orchestrator, re-checked here): the 06c row ✅,
  `git status --short` empty, `runregistrykit list` nothing `running` or stale (61 runs: 59 finished,
  2 `unknown`, pre-registry). The seven roots at `50277f3` are the orchestrator's (Verification).
- No Ray was started, no run was registered, no store was opened, and nothing was written under
  `var/`.

**Commit:** `f4d8e17`, reviewed by the orchestrator and pushed 2026-10-30, landed by the user
(the board's Decisions, **06e lands**).

**Result:** `chebyshev_matrices` returns the endpoints exactly as its end nodes, so every solve
integrates its own region and every region tiles with its halves; the Clenshaw–Curtis node term is
the signed first-order error of each node's actual displacement, with per-node $\theta'$, plus
bounded remainders. Tests 1–7 hold; P1 **holds** (median 848); the three breakages are caught; the
20 seeded LiouvilleGreen draws pass under both codes, with every depth-20 call gone (101 → 0) and no
case moving from honest to dishonest. Three existing assertions changed: two of allowed kinds, and
06b's fallback-branch test 3 (b) restated on the user's ruling after the one stop.

## What shipped

**The midpoint (§0.2 item 1), confirmed by reading `_adaptive_levin` at the dispatch SHA.** Every
path takes the same float for both halves:
- the Levin step-(4) halves in `_make_leaf`: `c = (b + a) / 2.0`, then `_solve((a, c))` and
  `_solve((c, b))`;
- a fallback leaf's halves (`_solve_halves`) and every bisection: `c = leaf.break_point`, the
  property `(self.end + self.start) / 2.0`, the same expression (IEEE addition commutes), and the
  children are made on `(a, c)` and `(c, b)` with the halves' solves carried forward;
- the trust check's grandchildren: `cL = (c + a) / 2.0`, `cR = (b + c) / 2.0`, solved on `(a, cL)`,
  `(cL, c)`, `(c, cR)`, `(cR, b)`; when the leaf is bisected they become its children's halves, and
  each child's own `break_point` is `(c + a) / 2.0 = cL` and `(b + c) / 2.0 = cR`, bit for bit.

So a region and its halves tile exactly once the end nodes are the endpoints. No stop on item 1.

**The end nodes** (`AdaptiveLevin/levin_quadrature.py`, `chebyshev_matrices`): after the affine map,
`x[0] = b` and `x[-1] = a`. The interior nodes and D are the map's, bit for bit (test 1). Its
docstring now says the end nodes are the endpoints, and why.

**The displacement** (`_chebyshev_node_displacements(x_span, N)` → `(Delta, err_abs)`):
$\Delta_k = \hat x_k - x_k$, signed, with $x_k$ the exact node. By error-free transformations
(`_two_sum`, Knuth; `_two_product`, Dekker with Veltkamp's split, vectorised) against the base nodes
in double-double: `_chebyshev_base_corrections(N)` holds $\mathrm{fl}(\cos(\pi k/(N-1)) - \hat t_k)$,
computed once per order in a private mpmath context at 160 bits and cached as `_chebyshev_base` is.
The exact node expands into $\hat x_k$ and eight small terms, summed by Sum2 (Ogita–Rump–Oishi).
Stated bound $|\hat\Delta_k - \Delta_k| \le 2u|\hat\Delta_k| + u(u(M+h) + h\max|{\rm corr}|)$,
checked against mpmath at 80 digits on 300 seeded spans (a test) and 8,105 nodes during development
(worst relative error 6.5 u, at a node whose $\Delta$ is near the absolute part). $\Delta_0 =
\Delta_{n-1} = 0$. Measured $\max_k|{\rm corr}_k|$ is at most 1.26 u for orders 2–64.

**The node term** (`_cc_node_term_parts`, returning $(E, R_g, R_2, R_c)$; $\phi_{\rm node} = |E| +
R_g + R_2 + R_c$ replaces 06d's third term in `_cc_roundoff_floor`, whose first two terms are
unchanged, bit for bit):
- $E = \sum_k w_k \Delta_k \sum_i (f'_{ik} b_{ik} + f_{ik}\theta'_k b'_{ik})$, $b' = $
  `_Basis_SinCos.basis_derivative` = $(\cos, -\sin)$, exact.
- $R_g = \sum_k |w_k||\Delta_k| \sum_i(\eta_\theta|\theta'_k||f_{ik}| + \eta_f|f'_{ik}|)$.
- $R_2 = \tfrac12 \cdot 2 \cdot \max_k \sum_i(|f''| + 2|f'||\theta'| + |f||\theta''| + |f|\theta'^2)
  \cdot \sum_k |w_k|\Delta_k^2$, $f''$ and $\theta''$ the fine grid's Chebyshev derivatives
  (`_CC_NODE_SECOND_ORDER_SAFETY = 2`).
- $R_c = \gamma_K \sum_k|w_k||\Delta_k|\sum_i(|f'b| + |f\theta'b'|)$ with $K = n + 2m + 5$
  (`_CC_NODE_TERM_ROUNDING_COUNT = 5`, counted term by term in its comment), plus the basis values'
  error $\min(2, u(|a_{ik}| + 2))$ against $\sum|w||\Delta|\sum(|f'| + |f\theta'|)$, plus $\Delta$'s
  absolute error against $\sum|w|\sum(|f'b| + |f\theta'b'|)$.
- No $1/\theta'$.

**The η values and their basis** (each a named constant with its comment):

| | value | basis |
|---|---|---|
| $\eta_\theta$, theta' supplied (`theta_deriv`) | $2^{-10}$ (`_CC_THETA_PRIME_REL_ERROR_SUPPLIED`) | **measured** (§2 test 4): the module only evaluates the caller's function at the node. Over the 102 test-4 cells whose $\theta'$ is supplied, the worst relative error against mpmath at the same float node is **0.997 u**. Taken at $2^{-10}$, a margin over $10^{12}$, to cover a $\theta'$ that is a fit good to three digits; a worse $\theta'$ also corrupts the Levin branch's operator and belongs in `theta_abserr`. |
| $\eta_\theta$, spectral (no `theta_deriv`) | 1 (`_CC_THETA_PRIME_REL_ERROR_SPECTRAL`) | none statable: resolution of $\theta$ and the raw phase's rounding amplified by D, neither bounded relative to $\theta'$ |
| $\eta_f$ | 1 (`_CC_F_PRIME_REL_ERROR`) | none statable: how well the samples resolve $f$ (worst near a singularity, where $f'$ matters) |

With $\eta = 1$ the term is still tighter than 06d's in every cell (T2 cells, 4.3–58×), so §7's third
test-condition does not arise.

**How θ' reaches the fine nodes, and its cost** (`_Basis_SinCos.theta_prime_on_fine_grid`):
- with `theta_deriv` (both production callers supply it): the even fine nodes are the coarse grid,
  bit for bit, whose $\theta'$ `build_Levin_data` already holds; the odd ones are evaluated, **N − 1
  evaluations of the caller's $\theta'$ per fallback solve** (15 at the default order);
- without it: the fine grid's spectral derivative of the raw $\theta$ at the fine nodes — no
  evaluation when no `theta_mod_2pi` is supplied (the basis's arguments are that $\theta$), else
  **2N − 1 evaluations of $\theta$**.
- A non-finite $\theta'$ at a fine node logs and raises `ValueError`, as `build_Levin_data` does.
- Wall time per fallback solve (`parent_demo.txt` section G, machine under other load): 97 → 186 µs
  with $\theta'$ supplied (102 cells), 237 → 317 µs on the spectral path (20 cells, whose reduced
  phase is an mpmath function in both codes). Recorded, not an argument (README §5 rule 4).

**Removed:** `_CC_NODE_ROUNDING_COUNT` and `_cc_node_error_bound` (§1 item 4). `_cc_roundoff_floor`
now takes the node term as one argument.

**Docstrings and comments** (§0.2 item 3): `chebyshev_matrices`; the comment block that replaces the
one above `_CC_NODE_ROUNDING_COUNT`; `_cc_roundoff_floor`; the module docstring's fallback-floor
paragraph; `_adaptive_levin_subregion_cc`'s docstring (its cost, and what `theta_prime_Cheb` now
feeds) and its comment above the floor.

**Unchanged** (§0.3): eq. (151) and `_roundoff_floor`; 06d's first two terms; the declared-phase
term; 06b's premise and $\rho$; 06c's loop, budget and trust check; the C2 gate; the nested estimate;
`depth_max`; the result dict's keys; every caller.

**Tests.** `AdaptiveLevin/tests/test_levin_end_nodes.py`, 8 methods: tests 1–6 and two more under test
4 (the measurement behind $\eta_\theta$, and $\Delta$ against mpmath). The parent is
`git show 50277f3:AdaptiveLevin/levin_quadrature.py` executed as a module of another name.

**Records.** `docs/levin-phase-floor/06e/calibration/`:
- `scripts/prov.py` (06d's, the parent moved to `50277f3`); `scripts/parent_demo.py` →
  `out/parent_demo.txt` (sections A–G); `scripts/lg_seeded_draws.py` → `out/lg_seeded_draws.jsonl`
  and `out/lg_seeded_draws_report.txt`; `scripts/breakages.py` → `out/breakages.txt`;
  `scripts/exit_search.py` → `out/exit_search.txt` (the stop's search).
- Commands, from the repository root: `PYTHONPATH=. ./venv/bin/python
  docs/levin-phase-floor/06e/calibration/scripts/<script>.py` (`lg_seeded_draws.py run --processes 8`,
  then `report`).

## Predictions and verdicts

**P1 — the node term tightens.** *Over test 4's window-like cells, the median of 06d's third term
over $\phi_{\rm node}$ is at least 10. A failure is a finding, not a stop.* Scored on the 90 window
cells of test 4 (`parent_demo.txt` section C): **median 848.3** (min 75.7, max $2.07\times10^4$).
**Holds.**

The following are §4's and §5's, copied before they run. **Not yet scored.**

§4, the value comparison:

   - **Z1 — consistency.** **Rule:** every row outside 06a's 720 stationary-point rows with finite
     `total_abserr` in both stores has $|{\rm total}_{20} - {\rm total}_{19}| \le {\rm
     total\_abserr}_{19} + {\rm total\_abserr}_{20}$. Rows with an infinite `total_abserr` in either
     are counted separately. The stationary-point rows are tabulated, not scored. **Holds** if no
     row violates it. A violation is a finding for the user, not by itself a stop.

§5, the census:

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

## Deviations from the prompt

1. **θ' at the fine nodes:** the caller's `theta_deriv` at the odd nodes, the coarse sample at the
   even ones; the spectral derivative where none is supplied. §1 item 3 left the route to the agent.
   **IMPLEMENTATION CHOICE.**
2. **η values:** $\eta_\theta = 2^{-10}$ on the supplied path (measured, a stated multiple), 1 on the
   spectral path; $\eta_f = 1$. Per §0.2's rules. **IMPLEMENTATION CHOICE.**
3. **$R_c$ also covers the module's own inputs to $E$** (the basis values' error and $\Delta$'s
   absolute error), not only the arithmetic of $E$. §0.2 names only "the rounding in computing E";
   the inputs' rounding is of the same kind and has to be bounded somewhere. **IMPLEMENTATION CHOICE.**
4. **$R_2$** uses the maximum over the fine nodes with a factor 2. **IMPLEMENTATION CHOICE.**
5. **TwoProduct by Dekker's split**, vectorised, rather than `math.fma`; the base corrections from a
   private mpmath context at 160 bits, imported lazily. **IMPLEMENTATION CHOICE.**
6. **$\Delta$'s accuracy** is $2u|\Delta|$ plus an absolute part of order $u^2(M+h)$: "a few u of
   $\Delta_k$ itself" except at a node whose $\Delta$ is itself near $u^2(M+h)$ (worst measured
   6.5 u). **IMPLEMENTATION CHOICE.**
7. **New methods on `_Basis_SinCos`** (`basis_derivative`, `theta_prime_on_fine_grid`), and
   `_cc_roundoff_floor`'s signature takes the node term. Within the file. **IMPLEMENTATION CHOICE.**
8. **A non-finite θ' at a fine node raises**, as on the coarse grid. **IMPLEMENTATION CHOICE.**
9. **`test_levin_cc_floor.py`'s import of `_cc_node_error_bound` replaced** by the same formula
   inline, in test 1's upper bound; the asserted bound is numerically 06d's, unchanged. Required by
   §1 item 4 (the replaced helper removed), so not an assertion change. **STRUCTURALLY REQUIRED.**
10. **Test 1's spans** have both ends of one sign (|a| from $10^{-3}$ to $10^{15}$, |b| up to
    $2\times10^{15}$); the parent misses on 2,046 of 4,000, against the orchestrator's 1,249 on its own
    sampling. **IMPLEMENTATION CHOICE.**
11. **Test 2** bisects through `_levin_leaf.break_point` (checked equal to `(b + a) / 2.0`), three
    levels, at orders N and 2N − 1. **IMPLEMENTATION CHOICE.**
12. **Test 3's problem** is a linear phase with an exact (mpmath) `theta_mod_2pi`, so the Levin value
    carries no phase rounding and what the parent chases is the offset alone. **IMPLEMENTATION CHOICE.**
13. **Test 4 also asserts $\phi_{\rm node}$ below 06d's third term in every cell**, and its window
    cells are the dyadic leaves of [0, 1] at the leaf study's depths nearest the root of $\theta'$,
    up to four each side, with total variation under $5.5\pi$. **IMPLEMENTATION CHOICE.**
14. **The test module runs `git show` at test time** to load the parent, so the AdaptiveLevin root
    needs a git checkout holding `50277f3`. **IMPLEMENTATION CHOICE.**
15. **The draws ran unregistered** (about 25 min each run, in the background): the dispatch forbids
    writing under `var/`. **STRUCTURALLY REQUIRED** by the dispatch.
16. **The draws were run twice:** once before a docstring-only edit of the module, then again on the
    final module, for provenance; the report kept is the second. **IMPLEMENTATION CHOICE.**
17. **Breakage (iii)** sets every fine node's $\theta'$ to the minimum $|\theta'|$ over the fine
    sample. **IMPLEMENTATION CHOICE.**
18. **`test_levin_roundoff_exit` test 3 (b)'s halving assertion withdrawn**, beyond §2 test 8's
    allowance, after a stop under §7 and on the user's ruling (*"Looks like (a) to me. The halving
    assertion looks to me like it was just made up, so no reason to worry about it."*); its other
    assertions changed to 06e's outcome, of allowed kinds (Verification, test 8, item 3).
    **STRUCTURALLY REQUIRED** by 06e's node term, which is nearly additive under bisection (the
    search below), and authorised by the user.

## Verification performed

**Test 1** (`test_end_nodes_exact`): 4,000 seeded spans × orders 2, 16, 24, 31, 64: `x[0] == b`,
`x[-1] == a`, interior and D bit-identical to the parent's. **The parent misses on 2,046 of 4,000**
(the same spans at every order: the end nodes are midpoint ± half-width whatever the order).

**Test 2** (`test_halves_tile`): 500 spans, 3 levels, 3,500 bisections, orders 16 and 31: tile bit
for bit. At the parent 1,841 of 3,500 fail.

**Test 3** (`test_offset_chase_gone`): $\theta = 10^6(x - 100)$ on $[100.1, 100.8]$, $f =
(e^{-(x-100)}, 0)$, exact `theta_mod_2pi`, $\theta'$ supplied, atol $10^{-30}$, rtol $10^{-9}$ (request
$1.3\times10^{-15}$). **Parent: depth 16**, 639 leaves (512 fallback, below its deepest Levin level
13), 2,555 solves, unconverged, abserr $2.33\times10^{-10}$, true $1.78\times10^{-13}$. **06e: depth
0**, 1 leaf, 7 solves, converged, abserr $9.5\times10^{-17}$, true below the reference's resolution.
Both honest. The parent's residual over floor along its deepest line (`parent_demo.txt` section B):

| depth | type | residual | floor | residual/floor | halves tile |
|---|---|---|---|---|---|
| 0 | Levin | 3.6e-15 | 9.5e-17 | 38 | no |
| 1 | Levin | 1.37e-14 | 4.8e-17 | 288 | no |
| 2 | Levin | 1.37e-14 | 2.4e-17 | 576 | no |
| 3 | Levin | 3.6e-15 | 1.2e-17 | 299 | no |
| 5 | Levin | 2.54e-14 | 3.0e-18 | 8.6e3 | no |
| 7 | Levin | 2.35e-14 | 7.4e-19 | 3.2e4 | no |
| 9 | Levin | 2.48e-14 | 1.9e-19 | 1.3e5 | no |
| 11 | Levin | 2.41e-14 | 5.4e-20 | 4.5e5 | no |
| 13 | Levin | 2.47e-14 | 3.1e-20 | 8.0e5 | no |
| 14 | Levin | 2.47e-14 | 2.7e-20 | 9.2e5 | no |
| 15 | Levin | 1.67e-14 | 2.5e-20 | 6.6e5 | no |
| 16 | fallback | 9.7e-15 (pair) | 5.4e-13 | 0.018 | no |

The residual is flat at $|f|\,{\rm ulp}(100) \approx 1.4\times10^{-14}$ while the floor halves; the
call ends floor-limited at depth 16 on 06d's node term. The search found the family also dishonest
at the parent where the root's ends happen to hide the offset: at $X_0 = 1000$ the parent accepts
the root at depth 0 with abserr $9.5\times10^{-17}$ against a true error of $3.7\times10^{-15}$
(Observations 2).

**Test 4** (`test_node_term_bounds_node_error`, `parent_demo.txt` section C tables all 122 cells):
every solve honest (true error / max(pair, floor) at most 0.98); **$\phi_{\rm node}$ bounds the true
node error in every cell** and is below 06d's third term in every cell.

| cells | n | true/$\phi_{\rm node}$ max | 06d/$\phi_{\rm node}$ min / median / max | 06d/true |
|---|---|---|---|---|
| T2 (06d test 2, spectral θ', η = 1) | 20 | 0.210 | 4.34 / 36.4 / 58.2 | 29–1,130 |
| T3 (06d test 3, nested leaves) | 12 | 0.992 | 174 / 283 / 2,974 | 175–4,300 |
| W (window-like, c = 1e6, 1e8) | 90 | 0.997 | 75.7 / 848 / 20,660 | 76–1.26e5 |

Parts: $|E|$ carries $\phi_{\rm node}$ where θ' is supplied ($R_g$ up to 95 % of it only where $E$
cancels, e.g. W c=1e8 ω=1 depth 14 leaf +2: true $3.0\times10^{-21}$, $|E|$ $3.0\times10^{-21}$, $R_g$
$1.5\times10^{-20}$); $R_g$ carries it on the T2 cells ($\eta = 1$); $R_2$ at most $1.9\times10^{-4}$
and $R_c$ at most $1.1\times10^{-4}$ of $\phi_{\rm node}$. Per cell, the first-order $E$ reproduces
the true node error to two or three digits. `test_supplied_theta_prime_within_eta`: worst 0.997 u.
`test_displacements_exact`: 300 spans, every node within its bound.

**Test 5** (`test_stationary_phase_oracle`): under 06d and 06e alike, 76 leaves (12 fallback), depth
14, 607 solves, converged, abserr $2.52\times10^{-11}$, true $1.9\times10^{-14}$ (06d's own 76).
Honest, converged.

**Test 6** (`test_driven_problems`, section E): all 25 problems honest under both; $|v_{06e} -
v_{06d}|/({\rm abserr}_{06d} + {\rm abserr}_{06e})$ at most 0.29; the two that raise raise under both.
Bit-identical in value and abserr: 11 (SinIntegral, CosIntegral, SincIntegral, vectorized_sinc,
input_validation_order_clamp, lstsq_path, direct_solve_path, converged_flag_bump, both reversed
spans, generic_m4_basis). Moved:

| problem | 06d: leaves, depth, solves, conv, abserr, true | 06e |
|---|---|---|
| GRZIntegral(10) | 11, 4, 87, T, 3.00e-13, 5.6e-17 | 12, 4, 95, T, 2.98e-13, 5.6e-17 |
| GRZIntegral(100) | 92, 7, 367, F, 4.06e-14, 5.2e-17 | 100, 7, 399, F, 1.20e-14, 1.1e-16 |
| GRZIntegral(1000) | 228, 10, 911, F, 1.53e-13, 4.6e-16 | 238, 10, 951, F, 4.30e-14, 3.0e-16 |
| stationary_phase | 76, 14, 607, T, 2.52e-11, 1.9e-14 | the same; value equal, abserr differs in the last digits |
| roundoff_floor_raw(1e6) | 1, 0, 3, F, 2.15e-16, 9.9e-17 | 1, 0, 3, F, 2.15e-16, 6.6e-19 |
| roundoff_floor_reduced(1e6) | 1, 0, 3, F, 2.15e-16, 1.05e-16 | …, 4.9e-18 |
| roundoff_floor_raw(1e12) | …, 1.23e-16 | …, 2.0e-18 |
| roundoff_floor_reduced(1e12) | …, 1.27e-16 | …, 2.2e-18 |
| theta_abserr_scalar / _callable | 1, 0, 3, F, 8.35e-15, 9.9e-17 | …, 6.6e-19 |
| theta_mod_2pi_path_only | …, 2.15e-16, 1.05e-16 | …, 4.9e-18 |
| theta_deriv_path_only | …, 2.15e-16, 9.9e-17 | …, 6.6e-19 |
| mode_filter_loose | 1, 0, 7, T, 7.36e-13, 2.13e-13 | 8, 3, 63, T, 6.15e-12, 2.12e-13 |
| mode_filter_tight | 32 (fallback), 5, 127, F, 7.35e-14, 5.1e-16 | 1 (Levin), 0, 7, T, 6.9e-17, 2.6e-17 |

The single-region Levin problems' true errors fall 20–200× with the ends exact.

**Test 7, the LiouvilleGreen draws** (`lg_seeded_draws_report.txt`; 160 units; parent = `50277f3`'s
module, 06d's code):
- **All 80 (seed, test) pairs pass under both. No stop.**
- Calls 4,320 under each; converged (parent, new) T,T 3,810, **T,F 0**, F,T 20, F,F 490 (3,810 →
  3,830). **Calls at depth_max: 101 → 0.** Regions 804,043 → 68,772; solves 3,375,592 → 468,216.
  Running-test disagreements 0; leaves returned untrusted 0.
- **The honesty table** (printed cases against their closed forms):

  | test | cases | 06d: dishonest; and converged; worst ratio | 06e |
  |---|---|---|---|
  | `test_YJJ_log_singularity`, near-singular ($\epsilon \le 10^{-4}$) | 560 | 438; 431; 8.13e5 | 438; **438**; 8.13e5 |
  | `test_YJJ_log_singularity`, the rest | 240 | 0; 0; 0.93 | 0; 0; 0.93 |
  | `test_JJJ` | 100 | 0; 0; 0.34 | 0; 0; 0.28 |
  | `test_YJJ` | 40 | 0; 0; 0.41 | 0; 0; 0.41 |

  **No case honest at 06d is dishonest under 06e.** The 438 near-singular dishonest cases are the
  same set; seven of them that 06d left unconverged are now flagged converged (input
  conditioning, `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`).

**Test 8, existing tests.** Every test in `LiouvilleGreen/tests/` passes. In `AdaptiveLevin/tests/`
every test passes. Three changed (numbers in `parent_demo.txt` sections E and F):
1. `test_levin_quadrature.test_mode_filter_truncation_visible_in_abserr`: `loose["num_regions"]`
   1 → **8** (a region count). Under 06d the root's step-(4) residual, $2.2\times10^{-16}$, was the
   end-node offset, and the trust check passed it (ratio 0.19); with the ends exact it is
   $2.1\times10^{-17}$ of noise against $5.9\times10^{-17}$ (ratio 2.8), so the root is refined: depth
   3, converged, abserr $6.15\times10^{-12}$ (06d $7.36\times10^{-13}$), true $2.12\times10^{-13}$
   (06d $2.13\times10^{-13}$). The thesis assertions (jump > 0, truncation > 0, abserr ≥ jump,
   tight truncation 0) hold unchanged under both. The comment records the tight call: 1 region,
   converged, under 06e (06d: 32, unconverged).
2. `test_levin_cc_floor.TestNothingElseMoves` for `mode_filter_tight`: `num_simple_regions > 0` →
   `== 0` (which rule its regions take): 06d 32 fallback regions, unconverged, abserr
   $7.35\times10^{-14}$, true $5.1\times10^{-16}$; 06e one Levin region, converged, abserr
   $6.9\times10^{-17}$, true $2.6\times10^{-17}$; honest under both (the honesty assertion
   unchanged).
3. **The stop, and its resolution.** `test_levin_roundoff_exit.test_exit_against_its_premise_fallback_branch`, problem
   (b) ($\theta = 0.025\,e^{x - X_0}$, $f = (e^{-2(x - X_0)}, 0)$ on $[X_0, X_0 + 6]$, $X_0 = 10^6$,
   rtol $10^{-16}$):

   | | bypass | premise check |
   |---|---|---|
   | 06d | 1 region, depth 0, phase-limited, abserr 3.35e-9, true 9.7e-14 | 3 regions, depth 2, abserr 6.96e-10 (0.21), true 2.1e-13 |
   | 06e | 4 regions, depth 3, all phase-limited, abserr 2.51e-11, true 2.1e-13 | **bit-identical to the bypass** (bar the evaluation count) |

   06e's root floor ($2.5\times10^{-11}$ on the left half; the root's pair estimate $1.8\times10^{-9}$
   is above it) does not bind at the root, so the bypass bisects too. Failing: `parent["num_regions"]
   == 1`, `regions[0].depth == 0` (both of allowed kinds) and **`new["abserr"] < 0.5 *
   parent["abserr"]`, which is of no allowed kind**. `exit_search.py` searched the family ($X_0 \in
   \{10^3, 10^4, 10^6, 10^8\}$, $A \in \{0.025, 0.04, 0.3\}$, $\gamma \in \{1,2,3\}$, $L \in \{4, 6\}$,
   72 problems): under 06d the premise check changes 18 calls and halves the declared error on 12;
   **under 06e it changes 19 calls and halves none** (best ratio 0.92). Over the 3,214 06e fallback
   regions whose halves were solved, the premise ratio (halves' floors / own) has median 1.01 and
   falls below 1/2 on 481: the check still fires, but 06e's floor is nearly additive under
   bisection, so where it fires the call's declared error moves by at most 8 %.

   **The user's ruling** (option (a) of the stop): *"Looks like (a) to me. The halving assertion
   looks to me like it was just made up, so no reason to worry about it."* Changed, on (b):
   - the bypass's `num_regions == 1` → `== 4`, and `num_simple_regions == 4` added (region count);
     `regions[0].type == "direct"` → every region direct (which rule); `regions[0].depth == 0` →
     `max_depth == 3` (depth); `regions[0].phase_limited` → `num_phase_limited_regions == 4` (the
     phase-limited count on a fallback region) — 06d: 1 region, depth 0, phase-limited, abserr
     3.35e-9, true 9.7e-14; 06e: 4 regions, depth 3, all phase-limited, abserr 2.51e-11, true
     2.1e-13;
   - `new["abserr"] < 0.5 * parent["abserr"]` (06d: 6.96e-10 against 3.35e-9, 0.21) **withdrawn**,
     replaced by the call's signature equal to the bypass's but for the evaluation count (bit-
     identity across the bypass; 06e: 15 solves against 7);
   - kept unchanged: both honesty assertions, and the new call's depth, rule, phase-limited and
     per-region premise assertions. The docstring records the change and the ruling.

**Breakages** (`out/breakages.txt`; module restored and its SHA-256 checked after each):
- **(i)** end nodes unpinned: caught by **tests 1, 2 and 3** (and `test_displacements_exact`).
- **(ii)** $\Delta_k = 0$: caught by **test 4** on 122 cells, **every reduced-phase T2 cell among
  them**, and by `test_displacements_exact`.
- **(iii)** $\theta'_k$ → the region's minimum $|\theta'|$: caught by **test 4** on 82 cells, all 12 T3
  cells and 70 of the 90 window cells (and by the θ' measurement test, which checks η's value).

**The seven roots** (before: the orchestrator's at `50277f3`; after: this tree, each run alone):

| root | before | after |
|---|---|---|
| AdaptiveLevin | 60 | 68 |
| ComputeTargets | 780 | 780 |
| CosmologyModels | 39 | 39 |
| LiouvilleGreen | 148 (1 skipped) | 148 (1 skipped) |
| ObjectFactories | 280 | 280 |
| RunDrivers | 127 | 127 |
| tools | 58 | 58 |

All seven OK. AdaptiveLevin rises by 8, the new module's methods; it was re-run after the
resolution of the stop, and the other six re-run with it on the final tree. No timing flake
occurred.

`black` is clean on the changed files outside `calibration/scripts/`.

## Observations not acted on

1. **06b's premise on the fallback branch is now nearly inert** (the stop's search): it still fails
   on ~15 % of fallback regions whose halves are solved, but no call's declared error falls by half.
   No issue opened: the user ruled the halving assertion out of concern.
2. **The parent was dishonest where the offset hid**: test 3's family at $X_0 = 1000$ (and $c = 10^7$,
   $X_0 = 10$) accepts the root floor-limited at depth 0, abserr $9.5\times10^{-17}$ against true
   $3.7\times10^{-15}$ (and $5.6\times10^{-16}$); 06e is honest on both. Bears on Z3.
3. **The trust check now meets noise where the offset used to stand**: `mode_filter_loose`'s root
   is untrusted on a residual of $2\times10^{-17}$ (ratio 2.8) and refined to 8 leaves. Honest; cost
   and abserr up (abserr ×8.4).
4. **$\eta_\theta = 2^{-10}$ is measured on closed-form θ'**, not on production's spline θ'; its
   multiple is meant to cover those, and is not measured on them here.
5. **`_TwoIndependentSinCosBasis`** (a test stand-in) lacks the two new basis methods; it never
   reaches the fallback (06d Observation 5 already).
6. **Cost** per fallback solve roughly doubles with θ' supplied (97 → 186 µs). Not an argument.

## State handed on

- **This commit** holds the code, the tests, the calibration records and this log, and the board's
  06e row (code landed; §4–§5 pending). The header issues `[06c-chebyshev-end-nodes-miss-the-span]`
  and `[06d-cc-node-term-is-pessimistic]` are not moved: they move in 06e's records commit, after
  §4 and §5. No issue opened.
- **For §4 and §5 on the Mac Studio:** every value moves at round-off. Expect the census's depth-20
  calls outside stationary points to go (the draws: 101 → 0), calls whose floor-limited leaves hid
  the offset to declare more honestly (Observations 2), and fallback-heavy calls to cost about twice
  per solve. The AdaptiveLevin root runs `git show 50277f3` at test time (deviation 14).
