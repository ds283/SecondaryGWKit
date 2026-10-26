# levin-phase-floor 05 — layer 1(a): take $k\tau_R$ out of $G$'s phase

> **§1–§4 only** (the code, the comparison tool, their tests and these records), written by the
> executing agent on the laptop. §5 (rehearsals R9 and R10) and §6 (the wider re-measurement) run
> later on the Mac Studio. The orchestrator appends their record to this log.

**Subject:** prompt 05 §1–§4, on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `e648951` (`e6489519ee662434d1f8e430e1c172246678ac8c`). It differs from `a130831`
  only by an orchestrator note under `prompts/`.
- **The gate.** `git status --short` was empty. `runregistrykit list` showed nothing `running` or
  stale: 60 runs, 58 finished, 2 `unknown` (pre-registry). The seven roots were re-run at the
  dispatch SHA and reproduced the orchestrator's baselines exactly (Verification, below).
- No Ray was started, no run was launched, no production store was opened, and nothing was written
  under `var/`. The tests write stores only through `tempfile`.

**Commit:** `120fdfe` (`120fdfe320bbc5eaef2890a9a5055337b75c8125`), this one commit (§4). Reviewed by the orchestrator on the laptop (note §A.3, nine checks, all pass) and pushed by it.

**Result:**
- **§1.** `CumulativeTable` has four node accessors: `end_partial`, `delta_parts`, `node_term` and
  `node_limbs`. `TablePrimitive` passes them through. `delta` is built from them and is bit-identical
  to its old formula at 1000 random pairs.
- **§2.** `PrimitivePhase(..., exact_remainder=False)`. With the keyword on, `theta_mod_2pi` is
  $c_{n_z} + {\rm rem}(v(z))$.
  - $c_n$ comes from the four limbs and $k$ in exact rational arithmetic. It is reduced against
    $2\pi$ itself at 256 bits in a private `mpmath` context, rounded once, and memoised per node.
  - $v$ is formed in double and reduced with `WKB_mod_2pi`.
  - `_build_phase` turns the keyword on for $G$. `TkSourceFunctions` is untouched. The module
    docstring's "Anchoring" now says what holds for each.
- **§3.** 12 test methods, all passing. Every exactness error is within 0.22–0.32 of its bound.
  σ_θ falls from 0.297 F with the keyword off to 8.8e-10 F with it on.
- **The breakages.** All four were caught, and each was reverted:
  - (i) by test 2, in all 16 cases;
  - (ii) by test 3 (and test 2);
  - (iii) by test 4;
  - (iv) by (t4), and by (t2) and (t3).
- **§3A.** `tools/store_value_diff.py`, with tests (t1)–(t5) in 6 methods.
  - Every class of the production schema can be matched by identity. On the full fixture store,
    all 21 classes compare, every table is accounted for, and no column fails to resolve.
- **The seven roots.** Five are equal to the baseline. `ComputeTargets` rose by 12 and `tools` by
  6, the methods added. (Verification, below.)
- **No stop condition of §8 arose.**

## What shipped

**`ComputeTargets/cumulative_table.py` (§1).**
- `end_partial(z) -> (j, p)`: the node `_locate` assigns to `z`, and the partial
  $p = \int_z^{z_j} f\,dz$ as `delta` forms it. It is `0.0` on a node, with no integrand call.
- `delta_parts(z_a, z_b) -> (j_a, j_b, p_a, p_b)`.
- `node_term(j_a, j_b)`: `(hi_b − hi_a) + (lo_b − lo_a)`, as `delta` rounds it.
- `node_limbs(j) -> (hi, lo)`. It refuses an index outside the table.
- `delta` is now `node_term(j_a, j_b) + (p_b − p_a)` from `delta_parts`: the same operations in
  the same order.

**`ComputeTargets/BackgroundModel.py`.** `TablePrimitive` only: the four accessors, passed through.

**`ComputeTargets/primitive_phase.py` (§2).**
- **The constructor keyword** `exact_remainder=False`.
  - With it on, the constructor checks that `leading` has `end_partial` and `node_limbs`. If not, it
    raises `ValueError` naming what is missing, with no fallback.
  - It then forms $T(n_R) + p_R$ exactly, as a `Fraction`.
- **`theta_mod_2pi`, keyword on:**
  - $(z, u)$ are taken through `_get_x` and `_clamped_u`, as `raw_theta` takes them.
  - $(j, p_z)$ = `end_partial(z)`.
  - $c_j$ is memoised.
  - $v = -{\rm sign}\,k\,p_z + \phi(u)$, in double.
  - The result is $r = c_j + $ `WKB_mod_2pi(v)[1]`, plus one exact add of `TWO_PI` if
    $r \le -$`TWO_PI`.
- **`_node_constant(j)`:**
  - forms ${\rm sign}\cdot k\cdot\big[(T(n_R)+p_R) - ({\rm hi}_j + {\rm lo}_j)\big]$ as an exact
    `Fraction`;
  - reduces it in `_remainder_against_two_pi`: a private `mpmath.MPContext` at 256 bits, against
    `2*ctx.pi`, into $(-2\pi, 0]$;
  - rounds once with `to_float(..., round_nearest)`.
- **Unchanged:** the default path, `raw_theta`, `theta_deriv` and `build_phi_samples`.
- **The `exact_remainder` property.**
- **The module docstring's "Anchoring, and the floor it implies"**, rewritten as §2 asks. It now
  states both modes, who uses which, and why, citing G1.

**`ComputeTargets/GkSourcePolicyData.py`.** `_build_phase` only: `exact_remainder=True`, and a
docstring paragraph. Both its callers (`_classify_Levin`, which reads only `theta_deriv`, and
`_create_functions`) get the node-anchored remainder. The constants are lazy, so
`_classify_Levin` computes none.

**`ComputeTargets/tests/test_g_phase_reanchor.py` (§3), new: 12 methods.**
- **The background.** A `CumulativeTable` of $1/H$ at `TAU_GAUSS_ORDER`, shifted by the
  radiation-era `tau_init`, as `compute_background` builds `functions.tau`.
  - $H$ is a closed-form flat ΛCDM with radiation, in 1/Mpc, with τ(0.1) = 1.371e4 Mpc.
  - The grid has 100 nodes per decade of $z$, from 1e12 to 0.1; $z = 0.1$ is a node.
  - $\phi(u) = 0.3\sin u + 0.05u$, sampled at the nodes.

| test (§3) | method(s) |
|---|---|
| 1 | `TestNodeAccessor.test_delta_is_bit_identical_to_todays_formula`, `.test_exact_node_term_rounds_to_the_table_term`, `.test_end_partial_is_delta_s_own_end` |
| 2 | `TestExactness.test_remainder_against_two_pi` (16 subtests: 4 k × 2 $z_R$ × 2 signs) |
| 3 | `TestNoJitter.test_sigma_theta_off_and_on` (both signs) |
| 4 | `TestNothingElseMoves.test_raw_theta_and_theta_deriv_are_unchanged`, `.test_transfer_function_remainders_are_unchanged` |
| 5 | `TestDefault.test_keyword_off_is_WKB_mod_2pi_of_raw_theta`, `.test_without_the_accessor_it_raises` |
| 6 | `TestEndToEnd.test_build_phase_turns_the_node_anchored_remainder_on`, `.test_evaluate_sum_against_the_direct_product`, `.test_composed_remainder_moves_by_at_most_the_old_floor` |

**`tools/store_value_diff.py` (§3A), new.**
- **Reading.** Each store is read as `read_inventory` reads it: `open_read_only`, then
  `inventory_classes` order, then `read_records` per class per shard, with the same
  `ShardContext` bookkeeping.
  - The difference is that each row's serial is kept beside its identity (`Record.identity()`).
  - Rows are matched by that identity alone. Value rows are matched by their parent's identity and
    the reference digests of every other row they name, never by serial.
- **What is compared.** All of a row's own columns except `serial`, `timestamp` and the key's
  parent columns. A foreign key outside the key (e.g. `z_min_serial`) is compared through the
  named row's reference digest.
- **How.** Floats by `float.hex`. A string holding a JSON object or array in both stores is
  compared as JSON, by key. Anything else by `==`.
- **What it reports:**
  - per class: identical, differ, only in A or B, and cannot be matched;
  - per column: the count, and the maximum $|\Delta|$ and $|\Delta|/|a|$;
  - for JSON, the keys that differ;
  - each differing row by its reference digest and resolved key.
- **Output.** Text by default; `--json`; `--classes`; `--max-rows`.
- **Exit status.** 0 if value-identical, 1 otherwise (including when anything cannot be matched), 2
  on a usage or open error.

**`tools/tests/test_store_value_diff.py` (§3A), new: 6 methods**, on `build_full_store` stores in
temporary directories:
- **(t1)** `TestIdentical.test_t1_two_builds_of_one_fixture_are_value_identical`. It also checks
  that the comparison leaves both stores' files unchanged (`file_state`).
- **(t2)**, in two methods:
  - `TestOneUlp.test_t2_one_ulp_in_a_row_s_own_column`: `QuadSourceIntegral.total`;
  - `TestOneUlp.test_t2_one_ulp_in_a_value_table_row`: `TkNumericValue.T`.
- **(t3)** `TestOnlyInOne.test_t3_a_deleted_row_is_in_one_store_only`.
- **(t4)** `TestNeverBySerial.test_t4_renumbered_serials_are_value_identical`.
- **(t5)** `TestJSONColumns.test_t5_json_key_order_is_immaterial_and_a_changed_value_is_named`.
- **How identities are checked.** The expected identity is found without the tool. It is the
  inventory record that disappears when the row is deleted. For the value row, it is the parent
  that `StoreInventory.find` and `.resolve` give.

## Predictions and verdicts

**Not yet scored.** §6 runs on the Mac Studio after this commit. Its predictions are copied below
from prompt 05 §6, word for word, before §6 runs, as §4 asks. This prompt (§1–§4) scores nothing.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "Scored
group" is 03's definition: at least 10 regions at depth 18 or more, here under $p$.
- **L1 — the cap goes where $G$'s anchor was the floor.**
  - **The rule:** on every depth-20 row whose top group is `G+r`, `G−r` or the lone `G`, under $c$
    at `1e-7`, every group under $p$'s scored set has fewer than 10 regions at depth ≥ 18, and the
    row's regions are at most a tenth of $p$'s.
  - **Holds** if true on every such row; an over-budget unit counts as false.
- **L1q — the same in the `Gqr` rows.**
  - **The rule:** L1's rule, on the `Gqr`-topped rows.
  - **Holds** if true on every such row. A failure is evidence that $T$'s phase carries a floor
    that matters there, which is the case for re-anchoring $T$. Report the depth-18 regions'
    $z'$ against $z_{\rm response}$ on any row that fails.
- **L2 — the controls do not move beyond their error.**
  - **The rule:** on every control row, $|{\rm total}_c - {\rm total}_p| \le$ `total_abserr`$_p$.
- **L3 — the declared error is honest after the change.**
  - **The rule:** on every depth-20 row with both $c$ units in budget,
    $|{\rm total}_c(10^{-7}) - {\rm total}_c(10^{-8})| \le$ `total_abserr`$_c(10^{-7})$.

| | Verdict |
|---|---|
| L1 | not yet scored (§6) |
| L1q | not yet scored (§6) |
| L2 | not yet scored (§6) |
| L3 | not yet scored (§6) |

## Deviations from the prompt

1. **The accessor's shape** (§1). **IMPLEMENTATION CHOICE.**
   - §1 suggests `delta_parts(z_a, z_b) -> (j_a, j_b, ends)` and `node_limbs(j)`, and leaves the
     name and shape to the agent.
   - `delta_parts` returns the two partials, `(j_a, j_b, p_a, p_b)`, and `delta` forms
     `ends = p_b − p_a` from them as before.
     - `PrimitivePhase` needs them apart: $p_R$ belongs in the per-node constant, and $p_z$ in
       the variable part.
     - `end_partial(z)` gives one end alone, so `theta_mod_2pi` does not recompute $p_R$ on each
       call.
   - `node_term(j_a, j_b)` gives `delta`'s rounded table term, for test 1.
2. **Exact arithmetic for the constant** (§0.2, §2). **IMPLEMENTATION CHOICE.**
   - The node term and the product with $k$ are exact `fractions.Fraction`s.
   - The reduction is in a private `mpmath.MPContext` at 256 bits, about 77 digits, against
     `2*ctx.pi`. §0.2 asks for at least 40 digits.
   - The one conversion of the exact rational into the context rounds relative $2^{-256}$. At
     $|X| \lesssim 10^{13}$ that is about $10^{-63}$ rad.
   - The global `mp.dps` is never touched.
3. **The remainder's range.** **IMPLEMENTATION CHOICE.**
   - $c_n$ is rounded to a double in $[-$`TWO_PI`$, 0]$, and the result is wrapped into the same
     interval.
   - A true remainder in $(-2\pi, -$`TWO_PI`$]$ rounds to $-$`TWO_PI`, which lies in §0.2's
     $(-2\pi, 0]$, since `TWO_PI` $< 2\pi$.
   - `WKB_mod_2pi`'s own range is $(-$`TWO_PI`$, 0]$. The closed end $-$`TWO_PI` is reachable only
     by that rounding, and is the same point of the circle as 0 to within $2.4\times10^{-16}$ rad.
4. **Test 3's window straddles a node switch**, and its threshold takes the strict reading.
   **IMPLEMENTATION CHOICE.**
   - The window is $10^{-6}$ wide in $u$, centred on the midpoint, in $u$, between the two table
     nodes nearest $z' = 10^{10}$: $z' = 9.886\times10^{9}$, where `_nearest_node` switches. So
     §0.2's switch between nodes is inside it. The test asserts that two nodes occur in the window.
   - That is also what lets test 3 see breakage (ii). A constant formed from a rounded double is
     still constant within one node's cell, so its error shows only as a step at a switch.
   - "At most $10^{-3}$ of that" is read as $10^{-3} \times 0.1\,\varepsilon k\eta_R$, the stricter
     of the two readings. $\eta_R$ is `tau(z_R)` from the table.
5. **Test 2's geometry** (left open by §3). **IMPLEMENTATION CHOICE.**
   - $z_R = 0.1$ is a node, the production case. $z_R = 30$ is not, so $p_R \ne 0$ is exercised
     too.
   - The abscissae: 200 log-uniform in $u$ up to $z = 10^{11}$, plus every 25th node, which is
     passed as $z$ so that the exact node lookup finds it.
6. **Test 4's `TkSourceFunctions`.** **IMPLEMENTATION CHOICE.**
   - It is `test_tk_source_functions`' exact fixture (w = 1/3), with `cs_tau` replaced by a real
     `TablePrimitive`: a `CumulativeTable` of $c_s/H$ on the WKB grid, shifted to the closed form
     at $z_{\rm init}$.
   - The stand-in's `ClosedFormPrimitive` has no node accessors. So breakage (iii) would only make
     the constructor raise, not change a remainder.
   - "Bit-identical before and after this prompt" is tested as equality, in `float.hex`, with
     `WKB_mod_2pi(raw_theta)[1]`, which is what `theta_mod_2pi` returned before this prompt. It is
     tested at all 173 WKB samples and the 172 midpoints between them.
7. **Test 6's $G$** is built by `GkSourcePolicyData._build_phase` itself. **IMPLEMENTATION
   CHOICE.**
   - The stand-ins are duck-typed: `source.model_proxy.get()`, `.k.k`, `.z_response.z`, and value
     objects carrying `z_source.z` and `WKB.theta_div_2pi`/`theta_mod_2pi`.
   - The τ table is on the realistic $G$ fixture's own grid plus $z_{\rm response}$ as a node.
   - The "old" phase is the same `PrimitivePhase` with the keyword off.
   - It runs at one configuration: w = 1/3, regime `G q r`. The bound for `evaluate_sum` is
     `TestOracle1DirectProduct.TOLERANCE`, 1e-12.
8. **Tests beyond §3's list.** **IMPLEMENTATION CHOICE.** These three methods are added, and
   counted in the roots' rise:
   - `test_end_partial_is_delta_s_own_end`;
   - `test_without_the_accessor_it_raises` (§2: "without the accessor it raises");
   - `test_build_phase_turns_the_node_anchored_remainder_on`.
9. **The tool drives `read_records` itself.** **STRUCTURALLY REQUIRED.**
   - `read_inventory` returns records without their serials. Value rows name their parent by
     serial, so matching them by identity needs the serial→identity map.
   - The tool therefore runs `read_inventory`'s own driver loop with the package's public pieces:
     `open_read_only`, `inventory_classes`, `inventory_specs`, `read_records`, `ShardContext` and
     `reference_digest`. It keeps the serials. `datastorekit` is not changed or patched.
   - The cost is about 40 lines that mirror the package's driver. If `read_inventory`'s ordering
     or context rules change in a later release, this tool must follow.
10. **References outside the key are compared through identity.** **IMPLEMENTATION CHOICE.**
    - §3A excludes "the columns that name another row … compared through that row's identity,
      which the key already holds".
    - For a key parent that is what happens. Some foreign keys are not in the key: `z_min_serial`,
      `solver_serial`, `version`, `z_max_serial` and others. They are compared through the named
      row's reference digest rather than dropped.
    - A `*_serial` column with no foreign key and no parent would be named as uncomparable. The
      production schema has none.
11. **Replicated classes are compared on the lowest-serial shard**, the shard whose records the
    inventory reports. **IMPLEMENTATION CHOICE.** Divergence between shards is the inventory's and
    the fingerprint's to report.
12. **How a JSON column is detected.** **IMPLEMENTATION CHOICE.** The `metadata` columns are
    `String`, not `sqla.JSON`. So a string that parses to a JSON object or array in both stores is
    compared as JSON. Array elements are compared by index.
13. **Exit status 2** for a usage or open error, beside §3A's 0 and 1. **IMPLEMENTATION CHOICE.**
14. **(t2) is two methods**, and (t1) also checks that the files are unchanged. **IMPLEMENTATION
    CHOICE.**
15. **Breakage (iv) was run twice.** **IMPLEMENTATION CHOICE.**
    - The first version keyed value rows by a different shape. The tool's own describer then
      raised `TypeError: 'int' object is not iterable` in (t4) and (t2), so the tests errored
      rather than failing on an assertion.
    - The second keeps the key's shape and matches every row and value row by serial. (t4) then
      fails on its assertion.
    - Both runs are recorded under Verification.
16. **`TablePrimitive`'s new annotations use the builtin `tuple[...]`.** **IMPLEMENTATION
    CHOICE.** `BackgroundModel.py` imports no `Tuple`. Adding one would have changed a line outside
    `TablePrimitive`, which the dispatch allows alone.

No deviation is `UNINTENDED DRIFT`.

## Verification performed

**The seven roots.** These are the counts from `PYTHONPATH=. ./venv/bin/python -m unittest
discover -s <root>/tests -t .`.

| root | dispatch SHA `e648951` | after | change |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | equal |
| ComputeTargets | 742 OK | 754 OK | +12 (the 12 methods of `test_g_phase_reanchor`) |
| CosmologyModels | 39 OK | 39 OK | equal |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) | equal |
| ObjectFactories | 280 OK | 280 OK | equal |
| RunDrivers | 127 OK | 127 OK | equal |
| tools | 49 OK | 55 OK | +6 (the 6 methods of `test_store_value_diff`) |

- `uptime` before the baseline run: `1:02 up 20 days, 23:47, load averages: 6.74 10.19 8.22`; after it: `1:07 up 20 days, 23:52, load averages: 5.39 7.26 7.43`.
- `uptime` before the run after the change: `1:23 up 21 days, 9 mins, load averages: 8.67 8.91 7.77`; after it:
  `1:27 up 21 days, 12 mins, load averages: 4.71 6.64 7.01`.
- The known flake, `test_tk_wkb_phase … test_wall_time_per_object`: did not fire in either run (`ComputeTargets` OK at 742 and at 754), so it was not re-run alone.

**§3, measured** (each test prints its numbers):

| test | number |
|---|---|
| 1 | `delta` bit-identical to the old formula (`float.hex`) at 1000 pairs, 1022 of 2000 endpoints on a node; exact node term against the table term at worst 0.972 ulp |
| 2 | worst err/bound 0.224–0.318 over the 16 cases, i.e. about $1.3$–$1.8\times10^{-15}$ rad, except one point at $2.0\times10^{-12}$ rad where $k\lvert p_z\rvert$ is large ($z_R$ = 0.1, $k$ = 3e8, sign +1: 0.263 of its bound) |
| 3 | $F = \varepsilon k\eta_R = 9.133\times10^{-6}$ rad, at $z' = 9.886\times10^{9}$; σ_θ off $2.713\times10^{-6}$ rad = 0.2971 F (both signs); on $8.02\times10^{-15}$ rad = $8.8\times10^{-10}$ F (sign −1) and $7.58\times10^{-15}$ rad = $8.3\times10^{-10}$ F (sign +1), against the bound $10^{-4}$ F |
| 4 | `raw_theta` and `theta_deriv` (both forms) bit-identical, keyword on and off, at 992 abscissae × 2 signs × 2 k; `TkSourceFunctions` remainders 345 of 345 bit-identical to `WKB_mod_2pi(raw_theta)[1]` |
| 5 | keyword off: 960 remainders bit-identical to `WKB_mod_2pi(raw_theta)[1]`; on, with a `ClosedFormPrimitive`: `ValueError` |
| 6 | Oracle 1 max relative residual $4.25\times10^{-16}$ (bound $10^{-12}$); composed remainder new against old at worst $5.38\times10^{-13}$ rad, 800 of 800 points moved, against the bound $4\varepsilon k\eta_R + 10^{-14} = 1.95\times10^{-12}$ rad ($k\eta_R$ = 2188 rad) |

**The deliberate breakages.** Each was made in the working tree, run, recorded, and reverted by
copying back a saved original. Each reverted file was then checked with `cmp` against the saved
copy. `grep BREAKAGE` finds nothing in the tree.

- **(i) $c_n$ reduced against `TWO_PI` instead of $2\pi$.** The edit, in
  `_remainder_against_two_pi`: `two_pi = 2 * ctx.pi` → `two_pi = ctx.mpf(TWO_PI)`.
  - **Caught by test 2**, `TestExactness.test_remainder_against_two_pi`, in all 16 subtests
    (`FAILED (failures=16)`), $k = 3\times10^{8}$ included:
    ```
    FAIL: ... (z_response=0.1, k=300000000.0, sign=-1)
    AssertionError: 28731139371.007282 not less than or equal to 1.0 : (97723722095.58112, False, 0.00016033669747845094)
    ```
  - That is $1.60\times10^{-4}$ rad at $k$ = 3e8, $z_R$ = 0.1, which is $0.176\,\varepsilon k\eta_R$:
    04's I3 modulus error. At $k$ = 1e5 it is $5.3\times10^{-8}$ rad. Nothing else failed.
- **(ii) The node term from `delta`'s rounded double instead of the limbs.** The edit, in
  `_node_constant`: `node_term = Fraction(self._leading.node_term(j, self._anchor_node)) + p_R`.
  - **Caught by test 3**, `TestNoJitter.test_sigma_theta_off_and_on`, at both signs:
    ```
    FAIL: ... (sign=-1)
    AssertionError: 2.836180081212333e-07 not less than or equal to 9.132973144490524e-10
    ```
    That is σ_θ on = 0.0311 F, against 0.2971 F off and the bound $10^{-4}$ F. Two nodes were in
    the window.
  - **Also caught by test 2**, in all 16 subtests: up to $4.4\times10^{-4}$ rad at $k$ = 3e8.
    `FAILED (failures=18)`.
- **(iii) The keyword on for $T$ as well.** The edit, in `TkSourceFunctions._build_WKB`'s
  `PrimitivePhase(...)`: `exact_remainder=True`.
  - **Caught by test 4**, `TestNothingElseMoves.test_transfer_function_remainders_are_unchanged`.
    Only 34 of the 345 remainders stayed bit-identical; the first difference was at $z$ = 4.7735:
    ```
    [05 test 4] TkSourceFunctions remainders: 34 of 345 bit-identical to WKB_mod_2pi(raw_theta)[1]; first difference at z=np.float64(4.773502691896256): -2.545332484907412 against -2.545332484907405
    AssertionError: Lists differ: [(np.float64(4.773502691896256), -2.545332[23059 chars]635)] != []
    ```
  - **Also:** `TestEndToEnd.setUpClass` errored, because `test_phase_groups`' realistic $T$ fixtures
    use a `ClosedFormPrimitive` `cs_tau`:
    ```
    ValueError: PrimitivePhase[T_k WKB phase]: exact_remainder needs the leading primitive's node accessors end_partial and node_limbs (CumulativeTable, through BackgroundModel.TablePrimitive); ClosedFormPrimitive lacks end_partial, node_limbs
    ```
    `FAILED (failures=1, errors=1)`.
- **(iv) Match by serial instead of identity** (`tools/store_value_diff.py`).
  - **The first attempt** keyed rows by `mapping["serial"]` and value rows by
    `(parent serial, serial)`. That changed the value key's shape, and the describer raised
    `TypeError: 'int' object is not iterable` in (t4) and (t2b). (t2a) and (t3) failed on their
    identity assertions. This is deviation 15.
  - **The second attempt** keyed rows by `str(serial)` and value rows by
    `(str(parent serial), (("serial", "serial", str(serial)),))`.
  - **Caught by (t4)**, `TestNeverBySerial.test_t4_renumbered_serials_are_value_identical`:
    ```
    AssertionError: 1 != 0 : store_value_diff: A = .../store1/full-store.sqlite
    21 classes compared by identity: 0 value-identical, 21 not
    version (replicated): NOT value-identical
      rows: 0 identical, 0 differ, 2 only in A, 2 only in B, 0 cannot be matched by identity
    ```
  - **Also caught by (t2a)** (`AssertionError: '1' != '49f1232149b0…'`), by (t3), whose expected
    identity was not found, and by (t2b), which raised `ValueError` on unpacking.
    `FAILED (failures=3, errors=1)`.

**Other checks.**
- **The existing suites of the touched modules pass unchanged:** `test_cumulative_table`,
  `test_primitive_phase`, `test_gk_source_policy`, `test_gk_source_primitive_phase` and
  `test_quadsource_integral` (87 tests), before the roots were run.
- **Cost.** At $k$ = 3e8, 20,000 random abscissae, `theta_mod_2pi` takes 6.44 µs a call with the
  keyword off, and 6.21 µs warm with it on: 7.22 µs on the first pass, which computes 1169 node
  constants. `raw_theta` takes 6.02 µs.
- **Pickling.** A `PrimitivePhase` with the keyword on survives a `ray.cloudpickle` round trip with
  bit-identical remainders, and its memo is carried. Ray's runtime is not started.
- **`black --check`** passes on the seven `.py` files changed.

## Observations not acted on

1. **`test_primitive_phase.py` sets `mp.mp.dps = 50` globally at import.** That is an existing
   test-module side effect. This prompt's code and tests use private contexts and do not depend on
   it. Not opened as an issue.
2. **The production response redshift is a node of the τ table.** The response grid is a subset of
   the source grid (`main.py:476`), so $p_R = 0$ in production. The construction also handles an
   off-node anchor, and test 2's $z_R = 30$ exercises it.
3. **`_classify_Levin` gets the keyword too.** `_build_phase` has two callers, and only
   `_create_functions` reads the remainder. Because $c_n$ is lazy, the classification computes no
   constant and pays only the constructor's one `end_partial` at the anchor.
4. **The tool mirrors `read_inventory`'s driver loop** (deviation 9). A `datastorekit` API that
   returned each record's serial would remove the duplication. That is a possible request for
   DatastoreKit, not a defect. Not opened.
5. **Every production class can be matched by identity, structurally.**
   - On the full fixture store, which holds every class of the inventory, (t1) finds every table
     accounted for. No `*_serial` column fails to resolve, and no row or value row is unmatchable.
   - A production store could still hold rows that share an identity (a `duplicate` problem in the
     inventory). The tool would then name them as unmatchable, and §5's run would see it.
   - §8's stop condition "a production class whose value rows cannot be matched by identity" did
     not arise.

## State handed on

- **One commit**, not pushed. It holds:
  - `ComputeTargets/{cumulative_table,BackgroundModel,primitive_phase,GkSourcePolicyData}.py`;
  - `ComputeTargets/tests/test_g_phase_reanchor.py`;
  - `tools/store_value_diff.py` and `tools/tests/test_store_value_diff.py`;
  - this log;
  - the board's 05 row.
- `docs/OPEN_ISSUES.md` is untouched: no issue was opened.
- **The header's issues do not move in this commit** (§4):
  - `[00-consumer-anchoring-floor]`;
  - `[04-g-phase-is-reduced-against-the-double-two-pi]`;
  - `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`.

  They move in 05's records commit, after §5 and §6.
- **For §5:** R9 at this commit's parent and R10 at this commit, compared with
  `PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff R9.sqlite R10.sqlite --json`.
  - The JSON gives, for each differing `QuadSourceIntegral` row, its identity and each differing
    column's two values (`float.hex` and number), $\Delta$ and $|\Delta|/|a|$.
  - For `metadata`, it gives the JSON keys that differ.
- **For §6:** nothing here bears on it beyond the code.
