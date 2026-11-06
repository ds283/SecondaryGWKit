# Prompt 06m — the signed trust yardstick in the driver (route B, as 06l measured it)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-11-06, on the user's decisions after 06l (the board's Decisions, **06l**, **06l
results** and **06m**):
- before 06l: *"OK, measure B first."*;
- on 06l's results: *"I think B as measured is the right choice, as far as I can see."*, then, told
  what the remaining unconverged flags cost (§0.3): *"Good. Please draft 06m"*.

**Where it runs:** in two parts, as 06g did.
- **Part A, §1–§3, the code:** on the laptop, by a fresh-context subagent, in the main checkout at
  `HEAD`. One commit.
- **Part B, §4 and §5:** on the Mac Studio, after the code commit is pushed.
  - **§4, rehearsals R25 and R26**, under the lock: R25 at the code commit's parent and R26 at the
    code commit, each in a worktree. 07's rehearsals become R27 and R28.
  - **§5, the census**, Ray-free, with no lock, on 06's certified scratch copy of A3 v2: 06l's census
    re-run at the code commit, which must reproduce route B's census bit for bit. **This census
    becomes 07's "before"**, in place of 06g's.

The records of Part B are committed on the laptop afterwards, in 06m's records commit.

**Closes, in its records commit, on the user's word:**
`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`, if §2's tests 1–5, Gate M
(§5) and R1 (§4) hold. **Its residual is proposed then as a narrower issue,**
`[06m-calls-flagged-unconverged-within-their-request]`: the calls B leaves unconverged outside the
720, most of them within their request, and what the flag should mean to the one-loop stage
(§0.4). The user decides whether it is opened, or folded into 08's question of what
`total_converged` means.

**Recommended model:** **Opus**.

**Read first:**
1. The board's §3 entry for `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`,
   with every narrowing (06j, 06k, after 06k, 06l), and the Decisions rows **06k results**, **06l**,
   **06l results** and **06m**.
2. `AdaptiveLevin/levin_quadrature.py`:
   - `_LEVIN_TRUST_MAX_RATIO` and the comment above it;
   - `_adaptive_levin`'s docstring (its account of the trust check), `_check_leaf`, `_solve_halves`,
     `_levin_leaf` (its `is_direct`) and the halves' data dicts;
   - `_adaptive_levin_subregion_cc`'s return dict: `phase_err`, `node_error_signed` ($E$),
     `node_error_bound` and `phase_err_less_node` ($\phi^-$).
3. 06l's records, `docs/levin-phase-floor/06l/`:
   - `out/results.md`, `log.md` and `review/review.md`;
   - the variant `scripts/variants/levin_quadrature_routeB.py` (`5aa6178a…`) and its diff against
     the tree, `out/variant_routeB_diff.txt`. **This prompt puts that diff's trust test into the
     tree, and nothing else of it**;
   - `scripts/calibration/` (the LiouvilleGreen draws), `scripts/variant_loader.py`,
     `scripts/run_census.py`, `scripts/run_k0.py` and `scripts/compare_k0.py`.
4. 06g's `test_levin_leaf_values.py`, for the pattern of a test against the parent's module
   imported under another name, and its stand-in for fixed solves.

**Gate:**
- 06l's records commit (`3076ba0e`) is on `origin/handover-remedial`;
- the main checkout's `git status --short` is empty, and `runregistrykit list` shows nothing
  `running` or stale on the laptop;
- nothing under `AdaptiveLevin/` has changed since 06g's code (`3ae6f3a4`), so the variant's base
  is the tree's module: `git diff 3ae6f3a4 HEAD -- AdaptiveLevin/` is empty;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

06g's rule (b) trusts a checked leaf if $r < 1/4$, or if $r < 1$ and $\Delta_1 \le \phi_P$. Outside
the stationary points it refines slow leaves, where $r \approx 1/2$ on rounding noise, to
`depth_max`. 06j showed that on half of them $\Delta_1$ is three solves' node-error difference,
which $\phi_P$ alone does not cover. 06k measured three remedies, and 06l measured a fourth, route B.
The user chose B as measured. This prompt puts it into the driver, **exactly as 06l measured it**,
so that 06l's census certifies the code by bit identity.

### 0.1 The change

In `_check_leaf`, the noise clause compares $\Delta_1$ with $\nu_B$ in place of $\phi_P$:

$$\text{trusted} \iff r < \tfrac14 \;\text{ or }\; \big(r < 1 \text{ and } \Delta_1 \le \nu_B\big),$$

$$\nu_B = \phi_P + \sum_{\text{Levin halves}} \phi_h
+ \Big[\text{CC parent only:}\; \sum_{\text{CC halves}} \phi^-_h
+ \Big|\sum_{\text{CC halves}} E_h\Big|\Big].$$

Here $\phi_P$ is the parent region's `phase_err`, and $\phi_h$ a Levin half's `phase_err`. For a
Clenshaw–Curtis half, $\phi^-_h$ is its `phase_err_less_node` and $E_h$ its `node_error_signed`.
A half is Clenshaw–Curtis if its data dict's `is_direct` is true. The parent is Clenshaw–Curtis if
`leaf.is_direct` is.

| Parent | Halves | $\nu_B$ |
|---|---|---|
| Levin | Levin + Levin | $\phi_P + \phi_L + \phi_R$ (route 1's $\nu_1$) |
| Levin | one or two Clenshaw–Curtis | $\phi_P$ + the Levin half's $\phi$, if any; **a Clenshaw–Curtis half adds nothing** (the user's modification) |
| Clenshaw–Curtis | Clenshaw–Curtis + Clenshaw–Curtis | $\phi_P + \phi^-_L + \phi^-_R + \lvert E_L + E_R\rvert$ |
| Clenshaw–Curtis | one Clenshaw–Curtis, one Levin | $\phi_P + \phi_{\rm Levin} + \phi^-_{\rm CC} + \lvert E_{\rm CC}\rvert$ |

**The arithmetic is 06l's, in its order.** Bit identity with 06l's census depends on it:
1. $\nu \leftarrow \phi_P$;
2. for each half in the order L, R: if it is Levin, $\nu \leftarrow \nu + \phi_h$, with a `None`
   $\phi_h$ read as 0.0;
3. if the parent is Clenshaw–Curtis and at least one half is:
   1. for each Clenshaw–Curtis half in the order L, R: $\nu \leftarrow \nu + \phi^-_h$;
   2. then $\nu \leftarrow \nu + |S|$, where $S$ is the one half's $E$, or $E_L + E_R$ if both halves
      are Clenshaw–Curtis.

On a Clenshaw–Curtis parent, a Clenshaw–Curtis half without `phase_err_less_node` or
`node_error_signed` raises, as the variant does. Every Clenshaw–Curtis solve returns both, so this
is a guard against an internal inconsistency, not a path.

**Nothing else changes:**
- the cut-off $1/4$;
- the declaration $\max({\rm residual}, \Delta_1)/(1 - r)$ for a trusted leaf;
- the value returned ($Q$);
- the `depth_max` declarations;
- the floor-limited exemption, the budget, the stops and the result dict.

The variant's firing records and its `route_06l` key are 06l's instruments and are not ported.

**The basis** (06l, `VERIFICATION.md` §6l): on a Levin parent $\nu_B$ holds only Levin floors. On a
Clenshaw–Curtis parent it holds the declared floors less the node term, plus the halves' node errors
combined signed, as the call's own sum combines them. $\nu_B \le \nu_1$ to the rounding of the sum,
so B trusts no leaf route 1 does not.

### 0.2 What 06l measured (route B against 06g, on A3 v2's census)

- **Safety:**
  - B trusts none of 06h's carrying leaves;
  - it loses 4 calls 06g converges, all `node_limited` at `1e-8`;
  - it leaves no matched reference worse (worst v03 at 1.182, 06g's 1.203);
  - it moves no call past $e_B + e_g$;
  - its Levin-parent firings reach $\Delta_1/\phi_P = 2.85$ at most.
- **Recovery:** 93.5 % and 93.1 % of V8's lost calls outside the 720 converge. Calls at `depth_max`
  outside the 720 fall from 966 and 996 to 72 and 76.
- **Cost:** group-call regions 0.84 of 06g's.
- **K0 and L3:** B equals 06g on every call where it never fires; $\nu_B \le \nu_1(1 + 10^{-12})$.
- **L2:** the prediction was wrong as written. B trusts the same 63 of 75 first-population slow
  leaves that route 1 does.

### 0.3 What this changes, and what it does not

- **Values, errors, partitions and convergence** move exactly as between 06g's census and route B's.
  They move only in calls where some leaf is trusted under $\nu_B$ and not under $\phi_P$. Elsewhere
  every result is bit-identical to 06g's code.
- **`total` moves** in `QuadSourceIntegral`, where the driver serves every Levin call.
- **No other module changes:** `LiouvilleGreen/`, `ComputeTargets/` and the datastore layer are
  untouched.
- **The `converged` flag keeps its meaning:** a call is converged only if its error meets the
  request and no leaf it returns is untrusted.

### 0.4 What it does not fix

- **The calls left flagged.** In route B's census, 114 group calls outside the 720 stay unconverged,
  in 84 of 18,560 units. 104 of them already report an `abserr` within their request
  ($\max(\texttt{atol}, \texttt{rtol}\,|v|)$); the other 10 are `node_limited`, at most 1.05× over.
  Their share of their row's `total_abserr` is at most 4.7 %. They are flagged by an untrusted leaf
  at `depth_max`, not by their error. `OneLoopIntegral` does not read `total_converged` yet, so what
  the flag should mean to it is open. This is the narrower issue the header proposes. The board
  already gives 08 the question of what `total_converged` means.
- **A margin for the node term's remainder.** On 12 first-population leaves $\Delta_1$ exceeds the
  first-order node-error difference by 0.2–0.6 %, and B misses them by at most 0.41 %. A margin would
  need a stated basis and a census of its own (the user chose B as measured).
- **`[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`**, which owns the
  `node_limited` losses.
- **The 720 stationary-point rows:** 07.
- **Recording $\nu_B$ on `used_interval`.** It is not stored. A later census can recompute it from
  a capture, as 06l's probes did.

## 1. The code

Every change is in `AdaptiveLevin/levin_quadrature.py`.
1. **A module-level function**, `_trust_noise_yardstick(phi_P, parent_is_direct, data_L, data_R)`,
   returns $\nu_B$ by §0.1's arithmetic in §0.1's order. It is pure: it reads the two halves' data
   dicts and returns a float. Its docstring states the formula, the four cases, the modification on
   a Levin parent, the basis, and the order and why it is fixed.
2. **`_check_leaf`'s trust test** calls it with `region.phase_err`, `leaf.is_direct`, `dataL` and
   `dataR`, and compares $\Delta_1$ with the result. Nothing else in `_check_leaf` changes.
3. **The comments and docstrings:**
   - the comment above `_LEVIN_TRUST_MAX_RATIO`: the noise clause now reads $\Delta_1 \le \nu_B$; why
     $\phi_P$ alone was too small (06j); why the halves' Clenshaw–Curtis node errors are combined
     signed and kept out of a Levin parent's test (06k, 06l); and that $1/4$ is unchanged;
   - `_check_leaf`'s docstring;
   - `_adaptive_levin`'s docstring where it states the trust test.

## 2. Tests

A new module, `AdaptiveLevin/tests/test_levin_trust_yardstick.py`. No Ray, no datastore.
- **The parent** is `HEAD`'s module (`git show <dispatch SHA>:AdaptiveLevin/levin_quadrature.py`),
  imported under another name.
- **06l's variant** is `docs/levin-phase-floor/06l/scripts/variants/levin_quadrature_routeB.py`,
  imported from that path under another name. The test checks its SHA-256, `5aa6178a…`, before
  using it.
- **A stand-in** that wraps `_adaptive_levin_subregion` to fix solves' values is allowed where a
  case cannot be built from a real integrand, and says so. A test that quotes the parent's numbers
  records them in its comment.

1. **The yardstick, exactly.** `_trust_noise_yardstick` is checked on each of §0.1's four cases, and
   on the Levin parent with one Clenshaw–Curtis half on either side. Each result is checked bit for
   bit against the formula written out in the test in §0.1's order. (The variant computes $\nu_B$
   inline in `_check_leaf`, so test 2 is its comparison.) The cases' $E$ include both signs, and one
   case is chosen so that adding the terms in another order changes the result's last bit.
   $\nu_B \le \nu_1$ in each.
   It raises on a Clenshaw–Curtis parent whose Clenshaw–Curtis half lacks $\phi^-$ or $E$.
2. **The code is the variant's.** On at least eight cells, run through `adaptive_levin_sincos` under
   the commit and under the variant:
   - the cells span smooth and oscillatory integrands, Levin and fallback leaves, and leaves at
     depth ≥ 2;
   - across them, the variant's `route_06l` records at least one firing on a Clenshaw–Curtis parent
     and at least one on a Levin parent;
   - **the check:** every result key the commit returns, `elapsed` aside, is bit-identical to the
     variant's, and so is
     every returned region's span, depth, value, errors, `trusted`, `delta1`, `delta2` and
     `convergence_ratio`;
   - **against the parent:** on every cell where the variant records no firing, the commit is also
     bit-identical to the parent, `elapsed` aside.
3. **A leaf trusted on cancelling node errors.** The case is a Clenshaw–Curtis parent with two
   Clenshaw–Curtis halves, $r$ near 1/2, and $\phi_P < \Delta_1 \le \nu_B$. The halves' $E$ are
   large and nearly equal and opposite, so $|E_L + E_R| \ll |E_L| + |E_R|$. Under the commit the leaf
   is trusted. At the parent it is untrusted and refined; the parent's $r$, $\Delta_1$ and $\phi_P$
   are recorded.
4. **A Levin parent's Clenshaw–Curtis halves are kept out.** The case is a Levin parent with two
   Clenshaw–Curtis halves, $r$ near 1/2, and $\phi_P < \Delta_1 \le \phi_P + \phi_L + \phi_R$. Under
   the commit the leaf is untrusted, where route 1 would trust it; $\nu_1$ and $\nu_B$ are recorded.
5. **06h's kind of leaf stays untrusted.** The case is a Levin-over-Levin leaf with
   $\Delta_1 > \phi_P + \phi_L + \phi_R$ and $r < 1$, untrusted under both codes. 06l's f02 is the
   model: $\Delta_1/\nu_B = 4.7\times10^6$.
6. **The LiouvilleGreen draws.** `test_3bessel_analytic.py` is run over seeds 0–19 by 06l's harness,
   copied from `docs/levin-phase-floor/06l/scripts/calibration/` into
   `docs/levin-phase-floor/06m/calibration/scripts/` with the diff recorded. It is run in three
   modes: the parent, the commit, and the variant. Checks:
   - no test fails under the commit that passes under the parent;
   - **the commit equals the variant bit for bit** on all 4,320 Levin calls and every case;
   - the honesty table, converged counts and `depth_max` calls under the parent and the commit,
     with the near-singular cases separately (06l: 122 calls fire, and no convergence or honesty
     changes).
7. **Existing tests.** Every test in the seven roots passes. An existing assertion may change only
   where a leaf the parent left untrusted is now trusted, and only in its asserted value, `abserr`,
   region, depth or solve counts. Each such change is listed with its numbers under both codes. A
   convergence assertion that would go from converged to unconverged is a stop (§7).

**Deliberate breakages,** each recorded with the test that catches it, then reverted:
- **(i)** $|E_L| + |E_R|$ in place of $|E_L + E_R|$: test 1 must fail;
- **(ii)** a Clenshaw–Curtis half's $\phi^-$ counted on a Levin parent: tests 1 and 4 must fail;
- **(iii)** the $|\sum E|$ term dropped: tests 1 and 3 must fail;
- **(iv)** a Clenshaw–Curtis half's full $\phi_h$ in place of $\phi^-_h$: test 1 must fail;
- **(v)** the $|\sum E|$ term added before the $\phi^-$ terms: test 1 must fail, and test 2's
  comparison with the variant is run and its result recorded.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 3. Records of the code

- `black` on the files changed, outside `calibration/scripts/`.
- `logs/06m-signed-yardstick-in-the-driver.md`, in README §5.1's form. Its **Predictions and
  verdicts** holds §4's R1 and §5's Gates M0 and M, copied before they run and marked not yet scored.
- The board's 06m row.
- **One commit.** The header's issue moves only in 06m's records commit, after Part B.

## 4. Rehearsals R25 and R26 — on the Mac Studio, after the code commit is pushed

As 06g §4, with records under `var/levin-phase-floor-06m/rehearsal/`, slugs `r25-parent` and
`r26-signedyardstick`, `--prompt 06m`, and both fingerprints `1b332ffa…`.
- **Outside `QuadSourceIntegral`:** every value table is identical, every other difference is of
  the four volatile kinds, and no row is in one store only.
- **Inside it, every row is tabulated:**
  - $|{\rm total}_{26} - {\rm total}_{25}|$ against `total_abserr` in each, and the same on the
    analytic oracle;
  - region counts, maximum depths and `total_converged` in each;
  - the rows whose `total_converged` changes, in each direction.
- **R1 — consistency.** **Rule:** every row outside 06a's 720 stationary-point rows with finite
  `total_abserr` in both stores has $|{\rm total}_{26} - {\rm total}_{25}| \le {\rm
  total\_abserr}_{25} + {\rm total\_abserr}_{26}$, and the same on the oracle. Rows with an infinite
  error in either are counted separately; the stationary-point rows are tabulated, not scored.
  **Holds** if no row violates either. A violation is a finding for the user, not by itself a stop.
- **R25 against R24.** 06g's R24 (`levin-phase-floor-06g-r24`, kept on the Mac Studio) was built at
  the same production code as R25. `tools.store_value_diff` on R24 and R25: every value table
  identical, and every other difference of the four volatile kinds. A difference is a finding for
  the user (the rehearsal is not reproducible), reported before R26 is scored.
- **The readers** run on R26, exit 0, and the store is byte-identical before and after. **The lock**
  is released and `ray-processes` lists nothing. **The stores are kept** until the user says
  otherwise.

## 5. The census — on the Mac Studio, Ray-free

**The tree** $m$ is a worktree at Part B's dispatch SHA, whose code is the code commit's. **The
store** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked against its certification,
read-only, and hashed before and after.

**The harness** is 06l's, copied from `docs/levin-phase-floor/06l/scripts/` into
`var/levin-phase-floor-06m/census/scripts/`, with the diff saved and each change a deviation. It runs
**with no variant installed**: the tree's own module, as 06l's harness records it. The "before" is
route B's census, `levin-phase-floor-06l-census-routeB-20261105T100126` (checkpoint `5cb3d3c0…`),
which is hashed before and after.

**The comparison** is made call by call, every call matched by unit, kind, index, label and span:
- in each call record, every key but `route_06l` (the variant's, absent at $m$) and the timing
  fields `compare_k0.py` already excludes;
- in each unit's region records, every field, timing aside.

**Gate M0 — K0 first.** At $m$ the harness reproduces route B's census on 06k's 100 K0 units
(`docs/levin-phase-floor/06l/out/k0/units.json`), bit for bit by the comparison above. It is one
registered run with slug `k0`. **Holds** if all 100 are equal. A failure is a stop: the code is not
what 06l measured.

**The census:** every row at `1e-7` and `1e-8`, in one registered detached run with slug `census`.
The budget is 300 s per unit, with 06a's over-budget pass, as 06l's. The checkpoint and the
per-unit records are kept in full on the Mac Studio; 07 compares against them.

**Gate M — the census is route B's.** **Rule:** all 18,560 units are `done`, and each equals route
B's census by the comparison above. **Holds** if all are equal. A failure is a stop, with the first
differing units and fields reported.

**If Gate M holds, 06l's verdicts are the code's.** They are L1, C1, C2, H1, S1 and K1, and L2 as
scored: by identity, not re-scored. The records say so, and cite `VERIFICATION.md` §6l.

**Descriptive, not scored:**
- **The flag's cost,** recomputed at $m$ from the census by the agent's own reader. For every
  unconverged call, outside and in the 720:
  - its `abserr` against its request, $\max(\texttt{atol}, \texttt{rtol}\,|v|)$;
  - whether it is `node_limited`;
  - its share of its unit's `total_abserr`.

  The same for 06g's census, beside it. (The orchestrator's count from 06l's extract: outside the
  720, 114 calls in 84 units; 104 within their request; at most 1.05× over; share at most 4.7 %.)
- **Cost:** census wall and group-call regions against 06g's and route B's. Cost is not an
  argument (README §5 rule 4).

## 6. Acceptance

1. **Part A as written:**
   - every test passing, and the five breakages caught;
   - every changed existing assertion of §2 test 7's kind, and listed;
   - the LiouvilleGreen draws equal to the variant's, with no new failure;
   - the seven roots rising only by the tests added.
2. **§4:** R25 and R26 both `done` with `1b332ffa…`; outside `QuadSourceIntegral` every value table
   identical; R1 scored; the readers exit 0; the lock released.
3. **§5:** Gates M0 and M hold; the descriptive tables made; the store and route B's census
   unchanged.
4. **Records:**
   - `logs/06m-signed-yardstick-in-the-driver.md`, with Part B's Mac Studio record appended by the
     orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §6m;
   - the records in `docs/levin-phase-floor/06m/`.

## 7. Stop conditions — stop and ask

- **The Gate** fails.
- **The change cannot be built as written:** $\nu_B$ needs a quantity `_check_leaf` does not hold.
  Report what is missing.
- **A test cannot pass as written,** or a deliberate breakage is not caught. **Test 2 or test 6
  finds the commit differs from the variant,** even in one bit.
- **The construction reaches outside its files:** beyond `AdaptiveLevin/levin_quadrature.py`, test
  files, `docs/levin-phase-floor/06m/calibration/` and comments.
- **An existing test moves beyond §2 test 7's allowance,** a LiouvilleGreen test fails under the
  commit that passes under the parent, or a seven-roots count falls.
- **The rehearsals:** R25 or R26 does not give `1b332ffa…`, or outside `QuadSourceIntegral` a value
  table differs.
- **The census:** Gate M0 or M fails, or a unit ends in an error (not over budget).
- **Any store file** changes, or route B's census does.
- **The lock** is held, or will not release.
