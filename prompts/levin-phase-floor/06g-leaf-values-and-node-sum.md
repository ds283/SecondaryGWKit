# Prompt 06g — the trusted leaf's value, the node errors signed, and the round-off exit's premise

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-01, on the user's decisions (the board's Decisions, **06e follow-ups**, **06f
results**, **06i rulings**, **06h Part B lands** and **06g scope**):
- on 06e's D3 and Z3: *"D3: before 07"*; *"Z3: OK, fold into D3's prompt."*;
- on the node errors: *"Yes, I agree with 06h first and the design in point 2."*;
- on 06i: *"I think we should leave the 17 Z3 samples that could not be computed for 06g"*, then
  *"I'd like to widen the sample, and fold item 3 into 06g"* (item 3 is a Filon reference tool
  under `tools/`, with tests);
- on 06h's H4: *"Fold the test into 06g."*

**Where it runs:** in three parts, as 06h does.
- **§1–§3, the code:** on the laptop, by a fresh-context subagent, in the main checkout at `HEAD`.
  One commit.
- **§4, rehearsals R23 and R24:** on the Mac Studio, after the code commit is pushed, under the
  lock. R23 runs at the code commit's parent and R24 at the code commit, each in a worktree. 07's
  rehearsals become R25 and R26.
- **§5, the census and the references:** on the Mac Studio, Ray-free, with no lock, on 06's
  certified scratch copy of A3 v2. 06h's census is re-run at the code commit, with 06h's run as the
  "before", and a widened sample of calls is referenced with §1's Filon tool. **This census
  becomes 07's "before".**

The records of §4 and §5 are committed on the laptop afterwards, in 06g's records commit.

**Closes, when its conditions hold:**
- `[06e-trusted-leaf-returns-its-coarsest-value]`: §2's tests 1–2, and §5's V3 and V4;
- `[06e-node-errors-add-in-absolute-value]`: §2's tests 4–6, and §5's V5 and V6.

**Narrows:** `[06c-budget-can-stop-on-a-pre-asymptotic-residual]` (the slowly converging leaf, §2
test 3) and `[07-roundoff-exit-assumes-bisection-cannot-help]` (H4's mechanism, §2 test 7 and §5's
V7). Neither closes here: the first keeps its noise question, which waits for after 07; the second
stays open until W2 is fixed (Decisions, **06b close**).

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entries `[06e-trusted-leaf-returns-its-coarsest-value]`,
   `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, `[06e-node-errors-add-in-absolute-value]`
   and `[07-roundoff-exit-assumes-bisection-cannot-help]` (its narrowing of 2026-11-01), and the
   Decisions rows named above.
2. `AdaptiveLevin/levin_quadrature.py`:
   - `_adaptive_levin`: its docstring, `_make_leaf`, `_check_leaf`, `_exempt_floor_limited`, the
     running sums (`_ExactSum`, `_add_leaf`, `_remove_leaf`, `_running_sums`), the stop test, the
     floor-limited test and the bisection, and the result dict it returns;
   - `_roundoff_exit_premise_holds` and the comment above `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO`;
   - `_adaptive_levin_subregion_cc`'s floor assembly, `_cc_node_term_parts` and `_cc_roundoff_floor`;
   - the Levin branch's floor assembly in `_adaptive_levin_subregion_impl`, and
     `_declared_endpoint_phase_err`;
   - `used_interval`: `total_err`, `declared_resolution`, `trusted`, `delta1`, `delta2`,
     `convergence_ratio`.
3. The tests: `AdaptiveLevin/tests/test_levin_global_budget.py` (the trust check),
   `test_levin_roundoff_exit.py` (06b's premise), `test_levin_cc_floor.py` and
   `test_levin_end_nodes.py` (the node term), `test_levin_quadrature.py`.
4. The evidence:
   - 06e's leaf breakdown, `docs/levin-phase-floor/06e/investigations/d3-leaf-values/` (its
     README and `s02`, `s03`, `s08`, `s09`);
   - 06f's M1, `docs/levin-phase-floor/06f/out/results.md` and `m1_calls.csv`, and the definition
     of its F2 combination there;
   - 06h's H4, `docs/levin-phase-floor/06h/census/issues.md` D1 and `out/cmd/46`–`48`.
5. 06i's Filon references, `docs/levin-phase-floor/06i/`: `scripts/filon.py`, `route_a.py`,
   `route_b.py`, `r0_closed_forms.py`, `closed_form_a.py`, `samples.py`; `verification.md`; and
   the prompt's §0.2–§0.3 and §9.

**Gate:**
- 06h's records commit has landed (the board's 06h row is ✅);
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale on the laptop;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

Four defects in `_adaptive_levin`, each found by a census and each in the loop that 06c's global
budget built, and the reference tool that showed two of them. They share a theme: the driver
treats a quantity it has computed as if it were a different one. The fixes are small; the
census and the references are what certify them.

### 0.1 The defects

1. **A trusted leaf returns its coarsest value.** 06c's trust check solves a leaf's halves and
   quarters, forms $\Delta_1 = |V_P - H|$, $\Delta_2 = |H - Q|$ and $r = \Delta_2/\Delta_1$, and
   declares $\max(\epsilon_P, \Delta_1)/(1-r)$, which bounds $V_P$'s error if the error falls
   geometrically at ratio $r$ from here on. It then returns $V_P$, the coarsest of the three. On
   Z3's samples 2, 8 and 9 one such leaf carries the call's error, at 0.998–0.9995 of the
   declaration; $Q$ is 650–2,200× closer (06e's D3).
2. **A slowly converging leaf is trusted on extrapolation.** The bound $\Delta_1/(1-r)$ holds only
   if the observed $r$ is not exceeded later. On sample 3 the leaf has $r = 0.507$, and $V_P$ is
   1.245× its declaration (06e's Z3; 06i's Filon reference, 1.241×): the leaf was not yet in the
   regime where one ratio predicts the next.
3. **The node errors add in absolute value.** Each Clenshaw–Curtis leaf's floor carries $|E_i|$,
   its node term's signed first-order error, and the call sums the floors. $E$'s sign varies from
   leaf to leaf: on 06e's Z2 call $\sum|E| / |\sum E| = 54$, and the signed sum would meet the
   request at 0.26 of it; the absolute sum held the call above its request to depth 20 (06f's F1,
   F2). Splitting does not lower $\sum|E|$: $\Delta_k$ is set by ${\rm ulp}(x)$, not by a leaf's
   width.
4. **The round-off exit's premise is tested against the wrong quantity.** 06b's premise holds when
   the halves' floors sum to at least half the parent's. On 06h's H4 call (`g-1e-8-3-1072` #13,
   analytic J1) the parent's floor is eq. (151)'s $1.94\times10^{-26}$ plus the declared endpoint
   term $4.14\times10^{-27}$. The declared term integrates the caller's phase error over the
   region, so the halves' declared terms sum to about the parent's: it does not fall under
   bisection, and on both sides of the comparison it dilutes the ratio. The premise held, the exit
   fired at depth 0 and the call stopped unconverged at $2.36\times10^{-26}$ against a request of
   $1.63\times10^{-26}$. Under 06e's code it split once and both halves converged, their floors
   summing to $9\times10^{-27}$. The test also compares the halves with the parent, not with what
   the call needs: a split that lowers a floor by 30 % fails the halving test, though it may be all
   the call needs to meet its request.

### 0.2 The remedy

**(a) The finest value.** A leaf trusted by the check returns $Q = (V_{LL} + V_{LR}) + (V_{RL} +
V_{RR})$, summed in that order, the order `_check_leaf` sums it in now. **Its declared resolution
error is unchanged,** $\max(\epsilon_P, \Delta_1)/(1-r)$. The reason is measured, not assumed: on
sample 8's carrying leaf $\Delta_1 = 2.77\times10^{-17}$, $\Delta_2 = 4.6\times10^{-21}$, and $Q$'s
error against 06i's reference is about $1.3\times10^{-20}$, nearly $3\Delta_2$
(`d3-leaf-values/s08.txt`). So the error does not keep falling at ratio $r$ below $Q$, and any
declaration built from $\Delta_2$ ($\Delta_2/(1-r)$, or $r\Delta_2/(1-r)$) would understate it.
Under the same assumption that makes $\Delta_1/(1-r)$ a bound on $V_P$'s error, it is also a
bound on $Q$'s, so the change cannot make an honest leaf dishonest; it changes no error, no
convergence and no partition, only values. Tightening the declaration is not in this prompt.
- An untrusted leaf at `depth_max`, a floor-limited leaf, and a leaf never checked keep $V_P$.
- The leaf's region records which value it returned, and $V_P$ beside it.
- The running value sum is updated by $Q - V_P$ when the check trusts a leaf, exactly
  (`_ExactSum`), and back if a later check or the floor-limited test changes that.

**(b) Slow leaves are not trusted.** A checked leaf is **trusted** if
- $r < r_{\rm trust}$, or
- $r < 1$ and $\Delta_1 \le \phi_P$, the leaf's own round-off floor (`phase_err`). Then $\Delta_1$
  and $\Delta_2$ are differences of rounding noise, $r$ means nothing, and the floor the leaf
  declares bounds them. Without this clause a leaf of noise-level differences, far below the
  budget, would be forced through refinement to `depth_max` (sample 8's second leaf has
  $\Delta_1 = 1.5\times10^{-28}$ against the call's $2.8\times10^{-17}$, and $r = 1/3$).

Otherwise it is **untrusted**, as an $r \ge 1$ leaf is now: refinable whatever the budget says,
chosen before every trusted leaf, and at `depth_max` returned flagged, with the call not converged.
At `depth_max` an untrusted leaf with $r < 1$ declares $\max(\epsilon_P, \Delta_1)/(1-r)$, which is
larger than $\Delta_1 + \Delta_2$; one with $r \ge 1$ keeps $\max(\epsilon_P, \Delta_1 + \Delta_2)$.

**$r_{\rm trust} = 1/4$,** a named constant with this basis in its comment:
- every rule here converges, in its asymptotic regime, by far more than 4× per halving: a Levin
  leaf of Chebyshev order $n \ge 8$ by about $2^{-n}$, and the nested Clenshaw–Curtis pair
  spectrally. A leaf at $r \ge 1/4$ is not in that regime, and one ratio does not predict the next
  there (sample 3, $r = 0.507$);
- at $r < 1/4$ the geometric tail adds at most a third to $\Delta_1$, so the bound leans little on
  the extrapolation;
- per the campaign's practice, a cut-off that ends refinement starts at the conservative end: 1/4
  trusts fewer leaves than 1/2 would. §2 test 3 and §5 report every checked leaf's $r$, so the
  value can be revisited on the evidence.

**(c) The node errors, signed.** Each Clenshaw–Curtis solve returns its node term's parts beside
its floor: $E$ (signed) and $R_g + R_2 + R_c$. Then:
- **The call carries $|\sum E|$.** The driver keeps $\sum E$ over the current Clenshaw–Curtis leaves
  as a running exact sum (`_ExactSum`), adding a leaf's $E$ when it is added and removing it when it
  is bisected. A leaf's contribution to the call's error, $e'_i$, is its `total_err` with $|E_i|$
  taken out of its floor: $\max(\text{resolution}, \phi_i - |E_i|)$ plus truncation, as 06f's F2
  formed it. The call's error is $\sum e'_i + |\sum E|$, in the budget and in the returned
  `abserr`; both sums are confirmed in ascending $x$ before a stop, as now.
- **The leaf keeps $|E_i|$** in the floor its floor-limited test compares its residual against: the
  residual of one leaf sees its own node error, not the sum's.
- **A call held above its request by $|\sum E|$ alone stops.** When $\sum e'_i$ meets
  $\max(\texttt{atol}, \texttt{rtol}|V|)$ and $\sum e'_i + |\sum E|$ does not, and the stop's other
  conditions hold (every leaf checked or floor-limited, no refinable leaf untrusted), the call
  stops **unconverged**, flagged `node_limited`. Bisection does not lower $\sum E$ systematically
  (06f), so refining further chases it.
- **The correction $-\sum E$ is measured, not applied.** The result carries $\sum E$ and $\sum|E|$;
  §5 measures $|V - \sum E - {\rm ref}|$ against $|V - {\rm ref}|$.
- `abserr_roundoff` and the other breakdown keys stay sums of the leaves' terms as now; a new key
  carries $|\sum E|$ and the per-leaf $\sum|E_i|$ beside it.

**(d) The round-off exit's premise.** The exit fires on a chosen leaf only if both of these hold:
1. **The halving test, on the part of the floor bisection can lower.** Each solve returns its
   declared endpoint term (`_declared_endpoint_phase_err`'s value) beside `phase_err`. The premise
   compares $\phi - \phi_{\rm decl}$: the halves' sum against $\rho = 1/2$ of the parent's.
   $\phi_{\rm decl}$ is excluded from both sides, since it is the caller's phase error integrated
   over the region and the halves' terms sum to about the parent's whatever the region's
   frequency range. Nothing else changes in the test: on the Levin branch it now compares eq.
   (151)'s floors, the term 06b's argument for $\rho$ is about; on the fallback branch it compares
   the Clenshaw–Curtis floors with their node terms, as 06b did.
2. **The split could not meet the budget.** The call's error is recomputed with the leaf's $e'$
   replaced by its halves' floors as each would enter it (less $|E|$ on a Clenshaw–Curtis half,
   per (c)) and $\sum E$ updated by the halves' $E$ in place of the leaf's; the floors are lower
   bounds on the halves' errors. If that is still above the request, the clause holds. A leaf whose
   split might be all the call needs is bisected. It cannot chase: where floors do not fall, the
   recomputed error is the call's error now, which is above the request, so the exit fires as
   before; every split it allows deepens a leaf, and `depth_max` bounds that.

An infinite half floor makes the premise hold, as now. 06b's test-only bypass keeps its meaning.

**(e) The Filon reference tool.** 06i's references become a module, `tools/filon_reference.py`,
with tests in `tools/tests/test_filon_reference.py`. It takes a captured call (06c's
`x1_capture.py` format: `x_span`, `f`, `theta`, `kw`) and returns the reference, by:
- **Route A** for the analytic Bessel blocks, in $\eta$, with $K$ taken exactly from the group's
  coefficients ($\omega = K$, the remainder $R$ and $C$ in the amplitude);
- **Route B** for the cosmological groups, in the phase itself, through 06i's continuous
  least-squares smooth phase, with its fit-noise bound;
- **two implementations,** QUADPACK's QAWO with 06i's exact internal products, and the
  Filon–Clenshaw–Curtis rule with moments in `mpmath`; each at two resolutions.

It reports each reading as a hex double with its scatter and the route's own error terms, and a
verdict: a reference **counts** when its four readings agree to within a caller-given tolerance.
It imports neither Ray nor a datastore. 06i's scripts are its source; its design choices (06i's
deviations) are kept unless a test shows one wrong. It is a tool, not production code: nothing in
`ComputeTargets/`, `LiouvilleGreen/` or `AdaptiveLevin/` imports it.

### 0.3 What this changes, and what it does not

- **Every converged call's value can move,** by at most its trusted leaves' $\Delta_1$, and most by
  far less. Calls whose leaves were all trusted with $r < 1/4$ and have no Clenshaw–Curtis leaf and
  no exit firing keep their errors, convergence and partition bit for bit; only values move.
- **Errors fall** where $\sum|E|$ was a material part of a call's error, and convergence may be
  gained there. **Refinement rises** where leaves at $1/4 \le r < 1$ above their floor were trusted.
  **Calls H4-like** split where they stopped.
- **`total` moves**, unlike 06h: the driver serves every Levin call in `QuadSourceIntegral`.
- **No other module changes:** `LiouvilleGreen/`, `ComputeTargets/` and the datastore layer are
  untouched, so the integrands, and the captured inputs of 06i and 06e, are those of 06h's code.

### 0.4 What it does not fix

- **A flat residual chased to `depth_max`** (`[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`):
  07's QUADPACK-style rule, which counts bisections that leave the error flat. (d) handles one
  level; floors that fall slowly over several are 07's.
- **The declared-phase proxy's tightness** (`[06e-cc-declared-phase-term-is-a-crude-proxy]`).
- **Shipping the $-\sum E$ correction.** Measured here; for the user after §5.
- **A tighter declaration for $Q$.** (a)'s evidence says $\Delta_2$ does not bound it; a better
  bound is a question for later.
- **Filon in production** for the analytic Bessel blocks: a separate prompt (the user,
  2026-11-01: *"That presumably requires a separate prompt"*), planned after this one as a
  measurement first.
- **The noise question of `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`** for $r \ge 1$
  leaves: (b)'s noise clause applies to $r < 1$ only; how many $r \ge 1$ leaves are noise is
  counted (§2 test 3, §5), not changed.

## 1. The code

1. **(a)** in `_check_leaf`: $Q$ returned by a trusted leaf, $V_P$ recorded, the value sum updated
   exactly; `_exempt_floor_limited` restores $V_P$ if it un-trusts a leaf. `used_interval` gains the
   returned value's kind and $V_P$.
2. **(b)** in `_check_leaf`: the trust test and $r_{\rm trust}$ as a named constant, its basis in
   its comment; the `depth_max` declaration for an untrusted $r < 1$ leaf.
3. **(c)**: `_adaptive_levin_subregion_cc` returns $E$ and $R_g + R_2 + R_c$; `used_interval`
   carries $E$ and a property for $e'_i$; the running signed sum; the budget, the stop and
   `_sum_leaves_ascending` on $\sum e'_i + |\sum E|$; the `node_limited` stop; the result keys
   (`node_error_signed_sum`, `node_error_abs_sum`, `abserr_node_signed`, `node_limited`).
4. **(d)**: both branches return their declared endpoint term; `_roundoff_exit_premise_holds`
   takes it and the budget clause; the constant's comment and the docstrings say what the premise
   now tests and why.
5. **(e)**: `tools/filon_reference.py`, its docstring stating the method, the routes, the count
   rule and what it does not do.
6. **Docstrings and comments**: `_adaptive_levin`'s docstring (the trust check, the error formula,
   the stops), `_check_leaf`'s ("Its value is unchanged" goes), `used_interval.total_err`'s.

## 2. Tests

New modules `AdaptiveLevin/tests/test_levin_leaf_values.py` (tests 1–3) and
`AdaptiveLevin/tests/test_levin_node_sum.py` (tests 4–6), additions to `test_levin_roundoff_exit.py`
(test 7), and `tools/tests/test_filon_reference.py` (test 8). No Ray, no datastore. The parent is
`HEAD`'s module (`git show <dispatch SHA>:AdaptiveLevin/levin_quadrature.py`, imported under another
name). References are closed forms or `mpmath` at 30 digits. A stand-in that wraps
`_adaptive_levin_subregion` to fix solves' values is allowed where a case cannot be built from a
real integrand, and says so. A test that quotes the parent's numbers records them in its comment.

1. **The trusted leaf returns $Q$.** On at least six cells with known integrals (smooth and
   oscillatory, Levin and fallback, one with a leaf at depth ≥ 2): every returned trusted leaf's
   value is its quarters sum, bit for bit, and records $V_P$; the call's value is the sum of the
   returned values in ascending $x$; `abserr`, `converged`, the regions' spans, depths and errors
   are bit-identical to the parent's on every cell where (b), (c) and (d) do not apply (cells are
   chosen with the error away from $\texttt{rtol}|V|$, which the value's move could cross); and the
   true error is at most the parent's on every cell, and smaller on at least one by a factor
   recorded.
2. **The value sum follows the leaves.** On a cell where a checked leaf is later found
   floor-limited or bisected, the running value sum equals the ascending sum of the returned
   values exactly, at the stop.
3. **Slow leaves.** On a cell with a leaf at $1/4 \le r < 1$ and $\Delta_1 > \phi_P$, the leaf is
   untrusted and refined before every trusted leaf, and the call's error covers its true error;
   at the parent the same cell trusts it (the parent's ratio recorded). On a cell with a leaf at
   $1/4 \le r < 1$ and $\Delta_1 \le \phi_P$, the leaf is trusted. 06c's
   `test_converged_flag` Gaussian bump is honest. Descriptive: over every checked leaf of tests 1,
   3 and 9's draws, the distribution of $r$, and the counts at $r < 1/16$, $[1/16, 1/4)$,
   $[1/4, 1)$ and $\ge 1$, each split by $\Delta_1$ above or below $\phi_P$.
4. **The signed node sum.** On a cell with at least 100 Clenshaw–Curtis leaves whose $E$ vary in
   sign: `node_error_signed_sum` equals the exact sum of the returned regions' $E$
   (`math.fsum`), `node_error_abs_sum` the sum of $|E_i|$, and `abserr` equals
   $\sum e'_i + |\sum E|$ recomputed from the regions; the parent's `abserr` on the same partition
   is recorded with the ratio.
5. **The node-limited stop.** On a cell where $|\sum E|$ alone keeps the call above its request:
   the call stops unconverged with `node_limited` True, below `depth_max`; the parent on the same
   input runs deeper (its depth and regions recorded). On every other cell of tests 1–7,
   `node_limited` is False.
6. **The floor-limited test keeps $|E_i|$.** A leaf whose residual is at most its full floor but
   above $\phi - |E|$ is found floor-limited.
7. **The premise.**
   - **The declared term is excluded.** On a cell in the shape of 06b's test 1 ($\theta = 100e^x$ on
     $[0, 20]$), with a caller `theta_abserr` sized so the declared term is 15–30 % of the root's
     floor: at the parent the exit fires and the call misses its request; under 06g the root is
     bisected and the call converges. With no `theta_abserr` the two codes give identical results.
   - **The budget clause.** On a cell whose root's halves' floors sum to between 1/2 and 1 of its
     own and fit the request: the parent fires the exit and misses; 06g bisects and converges.
   - **Termination.** On a cell whose floors do not fall under bisection and lie above the request,
     the exit fires under 06g on the same leaf as the parent's, and the two results are
     identical.
   - 06b's existing tests pass unchanged.
8. **The Filon tool.**
   - **The closed forms:** 06i's R0 cells at $\omega = 10^6$ and $10^9$, both implementations at
     both resolutions, within the tolerance 06i used.
   - **The Ci closed form:** a stand-in all-orders-½ `BesselPhaseGroup`, whose phase is exactly
     $K\eta$, against $s_f A[{\rm Ci}(|K|\eta_b) - {\rm Ci}(|K|\eta_a)]$ in `mpmath`.
   - **Route B's fit:** a stand-in cosmological-shaped phase with known smooth part and added jitter
     of 06i's size, recovered within the declared fit-noise bound.
   - **The count rule:** four readings agreeing count; one perturbed past the tolerance does not.
   - **Determinism:** one reading bit-identical in a fresh process.
9. **The LiouvilleGreen draws.** `test_3bessel_analytic.py` over seeds 0–19, at the parent and at
   06g, as 06h's test 8: no test fails under 06g that passes under the parent; the honesty table
   (dishonest, dishonest-and-converged, worst ratio), converged counts, depth-20 calls,
   `node_limited` calls and the exit's firings under both, the near-singular cases separately. The
   harness lives in `docs/levin-phase-floor/06g/calibration/scripts/`.
10. **Existing tests.** Every test in the seven roots passes. Allowed changes to an existing
    assertion, each listed with its numbers under both codes:
    - an asserted **value** that was a trusted leaf's $V_P$ and is now its $Q$;
    - an asserted **`abserr`** that carried $\sum|E|$ and now carries $|\sum E|$, or a declared
      resolution of a leaf now untrusted;
    - a **count** of exit firings, regions or solves that (b), (c) or (d) moves.

    A convergence assertion that would go from converged to unconverged is a stop (§7), unless the
    cause is (b) on a leaf that is not noise, which is reported, not changed.

**Deliberate breakages,** each recorded with the test that catches it, then reverted:
- **(i)** a trusted leaf returns $V_P$: test 1 must fail;
- **(ii)** the call carries $\sum|E|$ again: test 4 must fail;
- **(iii)** the declared term left in the premise's ratio: test 7's first case must fail;
- **(iv)** $r_{\rm trust} = 1$: test 3 must fail;
- **(v)** Route A with $K$ rounded once: test 8's Ci case must fail.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 3. Records of the code

- `black` on the files changed, outside `calibration/scripts/`.
- `logs/06g-leaf-values-and-node-sum.md`, in README §5.1's form. Its **Predictions and verdicts**
  holds §5's predictions, copied before they run and marked not yet scored.
- The board's 06g row.
- **One commit.** The header's issues move only in 06g's records commit, after §4 and §5.

## 4. Rehearsals R23 and R24 — on the Mac Studio, after the code commit is pushed

As 06h §4, with records under `var/levin-phase-floor-06g/rehearsal/`, slugs `r23-parent` and
`r24-leafvalues`, `--prompt 06g`, and both fingerprints `1b332ffa…`.
- **Outside `QuadSourceIntegral`:** every value table is identical, every other difference is of
  the four volatile kinds, and no row is in one store only.
- **Inside it, every row is tabulated:** $|{\rm total}_{24} - {\rm total}_{23}|$ against
  `total_abserr` in each, and the same on the analytic oracle (`analytic_rad` against
  `analytic.abserr`); region counts, maximum depths and `total_converged` in each.
- **V1 — consistency.** **Rule:** every row outside 06a's 720 stationary-point rows with finite
  `total_abserr` in both stores has $|{\rm total}_{24} - {\rm total}_{23}| \le {\rm
  total\_abserr}_{23} + {\rm total\_abserr}_{24}$, and the same on the oracle. Rows with an infinite
  error in either are counted separately; the stationary-point rows are tabulated, not scored.
  **Holds** if no row violates either. A violation is a finding for the user, not by itself a stop.
- **The readers** run on R24, exit 0, and the store is byte-identical before and after. **The
  lock** is released and `ray-processes` lists nothing. **The stores are kept** until the user
  says otherwise.

## 5. The census and the references — on the Mac Studio, Ray-free

**The tree** $g$ is a worktree at Part B's dispatch SHA, whose code is the code commit's. **The
store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its certification,
read-only, and hashed before and after.

**The census.** 06h's census re-run at $g$, with 06h's run
(`levin-phase-floor-06h-census-20261101T005037`, checkpoint `3104c18e…`) as the "before".
- **The harness** is 06h's, copied from `docs/levin-phase-floor/06h/census/scripts/` into
  `var/levin-phase-floor-06g/census/scripts/`, the diff saved, each change a deviation. It records,
  per call, beside 06h's fields: $\sum E$, $\sum|E_i|$, `node_limited`, and the exit's firings and
  premise checks; and in its region records, per checked leaf, $r$, $\Delta_1$, $\phi_P$, trust
  and the returned value's kind.
- **The units:** every row at `1e-7` and `1e-8`, one registered detached run with slug `census`, a
  budget of 300 s per unit with 06a's over-budget pass. Its checkpoint and per-unit records are
  kept in full on the Mac Studio: 07 compares against them.

**The references.** One registered run with slug `references`, at most 6 processes, using §1's
`tools/filon_reference.py` at $g$. The sample, fixed before any reading:
- **Z3's 30 calls** (`docs/levin-phase-floor/06c/depth20-tiling/`, matched by row, rtol and call
  index). The 13 referenced by 06i keep their references
  (`docs/levin-phase-floor/06i/out/readings/`), if the captured groups at $g$ equal those 06i read
  (coefficients, orders, signs and spans, bit for bit). The 17 others are captured at $g$ as
  `x1_capture.py` captures, and referenced.
- **A random sample of 60 calls,** drawn with a seed fixed and recorded before the draw, from 06h's
  census outside the 720 stationary-point rows and Z3's 30: 30 at each rtol, at each rtol 15
  analytic-block calls and 15 cosmological-group calls, each drawn uniformly among calls with
  finite `abserr`. Captured at $g$ and referenced.
- **A reference counts** when its four readings agree to within 1/100 of
  $\min({\rm abserr}_{06h}, {\rm abserr}_g)$. Those that do not are reported with their scatter and
  dropped. A reading over 30 minutes is stopped, reported, and the sample's reference dropped.

**Gate V, before scoring:**
- **The parent reproduces 06h.** At the parent the harness reproduces 06h's units **bit for bit**
  on `low-G+r-1` and `ctrl-low-1` at `1e-7`, and on `0-1971` at `1e-8`.
- **The integrand is 06h's.** `git diff --stat <06h's census tree c1ec3b2> <code commit> -- .
  ':!docs' ':!prompts'` touches only `AdaptiveLevin/`, `tools/`, test files and `CLAUDE.md`.
- **The new code is deterministic.** Two runs of one $g$ unit are bit-identical.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "The
stationary-point rows" are 06a's 720 (`spscan.py`).
- **V2 — the rows agree.** **Rule:** outside the stationary-point rows, every row with finite
  `total_abserr` under both has $|{\rm total}_g - {\rm total}_{06h}| \le {\rm
  total\_abserr}_{06h} + {\rm total\_abserr}_g$, and the same on the oracle. **Holds** if true at
  both rtols.
- **V3 — honest where referenced.** **Rule:** $|v_g - {\rm ref}| \le {\rm abserr}_g$ for every
  counted reference, Z3's and the random sample's. $|v_{06h} - {\rm ref}|/{\rm abserr}_{06h}$ is
  reported beside it. **Holds** if no counted reference violates it; fewer than 45 of the 90 counted is
  **undecided**.
- **V4 — the trusted leaf's value.** **Rule:** on Z3's samples 2, 8, 9 and 14,
  $|v_g - {\rm ref}|/{\rm abserr}_g \le 0.01$ (06e's breakdown, with the quarters: 0.0012, 0.00046,
  0.0015). **Holds** if true on all four.
- **V5 — the cap.** **Rule:** outside the stationary-point rows no call under $g$ reaches
  `depth_max`, at either rtol. An over-budget unit counts as false. **Holds** if true at both.
- **V6 — Z2's call.** **Rule:** `g-1e-8-0-33995` call 5 ends below `depth_max`, converged or
  `node_limited`. **Holds** if so; its $\sum E$, $\sum|E|$, regions and depth are reported.
- **V7 — H4's call.** **Rule:** `g-1e-8-3-1072` call 13 converges. **Holds** if so.
- **V8 — no convergence lost.** **Rule:** outside the stationary-point rows, no call that 06h
  reported converged is unconverged under $g$. **Holds** if true at both rtols. A failure is a
  finding for the user: each such call is reported with its cause ((b) on a non-noise leaf,
  `node_limited`, or other).

**Descriptive, not scored:**
- **The correction $-\sum E$:** on every counted reference with a Clenshaw–Curtis leaf,
  $|v_g - \sum E - {\rm ref}|$ against $|v_g - {\rm ref}|$ and against ${\rm abserr}_g$.
- **The values:** per call, $|v_g - v_{06h}|$ against ${\rm abserr}_g$, split by cause.
- **Slow leaves:** the counts of test 3's classes over the census, and the cost they add.
- **The node sum:** per call, $\sum|E|/|\sum E|$ where $\sum|E|$ is at least 1 % of the error.
- **The exit:** firings and premise checks under 06h and $g$; calls whose exit changed.
- **Cost:** regions, solves and wall time against 06h. Cost is not an argument (README §5 rule 4).
- **The stationary-point rows:** how they moved. They are 07's, and not scored here.

## 6. Acceptance

1. **§1–§3 as written:** every test passing, the five breakages caught, every changed existing
   assertion of the three kinds §2 test 10 allows and listed, the LiouvilleGreen draws with no new
   failure, the seven roots rising only by the tests added.
2. **§4:** R23 and R24 both `done` with `1b332ffa…`; outside `QuadSourceIntegral` every value table
   identical; V1 scored; the readers exit 0; the lock released.
3. **§5:** Gate V holds; V2–V8 each scored holds, fails or undecided against the rule as written,
   with their numbers; the store unchanged.
4. **Records:** `logs/06g-leaf-values-and-node-sum.md`, with §4 and §5's Mac Studio record appended
   by the orchestrator; `docs/levin-phase-floor/VERIFICATION.md` §6g; the records in
   `docs/levin-phase-floor/06g/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **A design point cannot be built as written:** (a)'s exact value sum, (c)'s running signed sum,
  or (d)'s budget clause needs information the loop does not hold at that point. Report what is
  missing.
- **A test cannot pass as written,** or a deliberate breakage is not caught.
- **The construction reaches outside its files:** beyond `AdaptiveLevin/levin_quadrature.py`,
  `tools/filon_reference.py`, test files and comments.
- **An existing test moves beyond §2 test 10's allowance,** a LiouvilleGreen test fails under 06g
  that passes under the parent, or a seven-roots count falls.
- **The rehearsals:** R23 or R24 does not give `1b332ffa…`, or outside `QuadSourceIntegral` a value
  table differs.
- **The census:** Gate V fails, or a unit ends in an error (not over budget).
- **The references:** fewer than half of the 77 new samples count, or a reading disagrees with 06i's
  on one of its 13.
- **Any store file** changes.
- **The lock** is held, or will not release.
