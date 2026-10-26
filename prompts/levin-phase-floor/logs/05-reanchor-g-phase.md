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

**Scored 2026-10-26 on the Mac Studio** by §6's agent, and re-scored from the checkpoints by the
Mac Studio orchestrator (review check 9), with the same numbers. The rules are below, copied from
prompt 05 §6 word for word before §6 ran, as §4 asks. §1–§4 scored nothing. The numbers and every
row are in the §6 record appended below, and in
[`docs/levin-phase-floor/05/wider/out/wider_05.txt`](../../../docs/levin-phase-floor/05/wider/out/wider_05.txt).

| | Verdict, as written | On the 11 rows still capped under $p$ (not scored; the user's reading) |
|---|---|---|
| **Gate L0** | **holds**: `QuadSource` at $p$ bit-identical at all stored nodes plus B1's 3; two runs identical | — |
| **L1** | **fails**, 9 of 24 rows. Clause 1 holds on all 24; clause 2 fails on 14 rows not capped under $p$ ($c/p$ 0.73–1.09) and on R5 (0.345) | the rule holds on 9 of 10; R5 has a 55-region cap. The cap goes on all 11 |
| **L1q** | **fails**, 1 of 7. R7 holds (0.0016); the six drawn `Gqr` rows are not capped under $p$ ($c/p$ 1.00–1.35), with no region at depth ≥ 18 under $p$ or $c$ | holds on 1 of 1 (R7) |
| **L2** | **holds**, 8 of 8; largest ratio 0.652 | — |
| **L3** | **fails**, 29 of 31: `mid-G-r-3` 1.29 and `mid-G-3` 1.70, neither capped under $p$ | holds on 11 of 11; largest 0.166 |

Every failure as written is on a row B1 had already taken off the cap, or R5's short chain. The
user said the capped-under-$p$ reading is the one that matters, and closed
`[00-consumer-anchoring-floor]` for $G$ on it (2026-10-26).

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


---

# The Mac Studio record: §5 and §6

> **Appended by the laptop orchestrator** in 05's records commit (orchestrator note §B.4). Part B
> ran on the Mac Studio on 2026-10-26, from the dispatch SHA `53084dc` (main checkout, code
> `120fdfe`) and the worktree `SecondaryGWKit-lpf05-e648951` (detached at `e648951`, the code
> commit's parent). Its orchestrator dispatched two agents, whose logs follow as they wrote them,
> with two exceptions: deviations 12 and 14 of §6, which carried no classification, are classified
> here (marked), on the user's agreement.
>
> A path `scripts/…`, `out/…` or `state/…` below is under `var/levin-phase-floor-05/rehearsal/` or
> `…/wider/` there, and is committed at `docs/levin-phase-floor/05/rehearsal/` or `…/wider/`. Not
> copied back: the run directories' checkpoints and unit pickles, the readers' `extract/` (177 MB),
> the comparison's full `store_value_diff.json` (41 MB) and `.txt` (10 MB), and `qsi_rows.csv`.
> They stay in the Mac Studio's `var/`.

**Two stops, both answered by the user on 2026-10-26.**
1. **§5, at prompt §8** ("any value outside `QuadSourceIntegral` differs"): the differences were
   timings, timestamped labels, the serial-derived `GkSource.parents_digest`, and GkWKB work
   counters in 25 rows, with every value table bit-identical. The user: *"Those columns cannot be
   expected to be identical. So I think that's just an incorrectly-worded prompt."* Opened as
   `[05-store-value-diff-compares-volatile-columns]`.
2. **§6, on reading "depth-20 row"** after B1 had uncapped many stored depth-20 rows: *"Carry on as
   you propose. … I guess we only care about rows that are capped under the new system"*. $p$ is
   the code commit's parent, B1's code without 05's change, as the orchestrator told the agent.

**The Mac Studio orchestrator's review** (`docs/levin-phase-floor/05/review.md`): **stop, on check
10 alone**, the two unclassified deviations above; checks 1–9 pass. The user agreed that they be
classified on committing.

**The laptop's own re-checks** (orchestrator note §B.4), over `ssh macstudio-tunnel`, read-only:
- **check 4's spot-check, on another row:** the one row that moved by 28× its declared error,
  $(k,q,r,z_{\rm resp}) = (10^6, 3.162\times10^5, 10^6, 329.64)$, read with `sqlite3` `mode=ro`.
  R9 shard 1 serial 20124 gives `total` `-0x1.14101c8c8a2a3p-71` and R10 shard 1 serial 20072 gives
  `-0x1.140d42f3c6f3fp-71`, equal bit for bit to the tabulation (`total_A`, `total_B`);
- **check 6's independent comparison, on another row:** `low-G-r-2` (shard 0, serial 38940), where
  B1 changes the stored total's sign. The orchestrator's `review/check6_qs.py` with only the row
  changed, run at $p$ from the worktree with `-B`: 666 common nodes × 9 columns, 0 differing in
  `float.hex`, recomputed-only z serials `[667, 668, 669]` (B1's), stored-only none. A3 v2's six
  files hashed identical before and after; no journal file; the worktree still empty under
  `git status --short --ignored`.

## §5 — rehearsals R9 and R10 on the Mac Studio

**Prompt:** [`05-reanchor-g-phase.md`](../05-reanchor-g-phase.md) §5, under the orchestrator's
dispatch, §B.1 and §B.5.
**Host:** mps026298 (Mac Studio). The main checkout was at the dispatch SHA `53084dc`, whose code
is the code commit `120fdfe`'s. The worktree `SecondaryGWKit-lpf05-e648951` was detached at
`e648951`, the code commit's parent.
**Model:** Claude Opus 5.5, as the Mac Studio orchestrator's rehearsal subagent, in three turns:
- the gate, the lock and R9's launch;
- R9's checks and R10's launch;
- the comparison, the stop and question, then after the answer the tabulation, the readers and
  the release.

**Date:** 2026-10-26, 01:46–02:10 UTC.
**Commit:** none. The records are committed on the laptop in 05's records commit.
**Result:** COMPLETE WITH DEVIATIONS.
- R9 and R10 both ended `done` with exit 0 and fingerprint `1b332ffa…`, equal to R8's class by
  class.
- Every value table of every class is bit-identical between them.
- `QuadSourceIntegral` differs: 542 of 864 `total`s moved, all through `WKB_Levin`.
- The readers exited 0, and the stores were left byte-identical.
- The lock was released.
- §8's "any value outside QuadSourceIntegral differs" fired as written, on wall-clock, label,
  serial-digest and work-counter columns. The agent stopped and asked. The user ruled the prompt
  wrongly worded (deviation 1).

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md` §5
"Layer 1(a)".

### What shipped

Nothing in the repository: no code, and no commit. Written on the Mac Studio, all in the main
checkout:
- `var/levin-phase-floor-05/rehearsal/`, holding `out/`, `scripts/` (four read-only scripts),
  `state/`, `extract/` (the readers' output, not for copying back), `log.md`, `verification.md`
  and `issues.md`;
- the stores `var/datastores/levin-phase-floor-05-r9*.sqlite` and `-r10*.sqlite`, five files
  each;
- the runs `levin-phase-floor-05-r9-parent-20261026T014729` and
  `levin-phase-floor-05-r10-reanchor-20261026T015120`, and their launch logs
  `var/runs/levin-phase-floor-05-r{9,10}.launch.log`.

The stores are the user's to keep or retire.

### Predictions and verdicts

§5 fixes no scored predictions; §6 holds them. The acceptance checks of §7.2:

| check (§7.2) | number | verdict |
|---|---|---|
| R9 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`, 21/21 class digests = R8's | holds |
| R10 ends `done` with `1b332ffa…` | `done`, exit 0, `1b332ffa…`; `classes` object = R9's | holds |
| every non-`QuadSourceIntegral` value identical | every value table bit-identical; 0 rows in one store only, 0 unmatched. Row-column differences only of the four kinds the user ruled out of scope | holds, on the user's ruling (deviation 1) |
| `QuadSourceIntegral` differences tabulated | 542/864 `total`s; median \|Δ\|/abserr 8.7e-5, max 28.2 (one row) | done |
| the readers exit 0 | 6 × exit 0; store byte-identical | holds |
| the lock is released | `release` exit 0; `status` free, no Ray processes | holds |
| (dispatch) R9 ≠ R10 in `QuadSourceIntegral` | not value-identical, so each tree ran its own code | holds |

### Deviations from the prompt

#### 1. §8's stop on "any value outside QuadSourceIntegral" fired as written, and was ruled not to apply — STRUCTURALLY REQUIRED

`store_value_diff` reported eight classes besides `QuadSourceIntegral` as not value-identical. All
eight differ only in row columns that cannot agree between two builds of any code:
- wall-clock timings;
- `label`, which is the job name plus `datetime.now()`;
- `GkSource.parents_digest`. This is the SHA-256 of the compact JSON of the sorted (z_source
  serial, GkNumericIntegration serial or None, GkWKBIntegration serial or None) triples of the
  row's parent set (`ObjectFactories/GkSource.py` `GkSource_parents_digest`). It is a function of
  serials, and the serials of 1346 of 1368 numeric and 3466 of 3477 WKB rows differ between the
  stores;
- the GkWKB residual-reuse counters `rho_evals`, `rho_reused` and `stage_1_RHS_evaluations`, in 25
  rows.

Every value table of every class is bit-identical, as the tool's JSON shows.

The agent stopped with the lock held and asked. The user's decision, relayed verbatim: *"R9/R10
differences: Those columns cannot be expected to be identical. So I think that's just an
incorrectly-worded prompt. Do whatever you were already planning to do with the lock. Only this
project is running code on the Mac Studio right now."* The orchestrator's note sets the boundary:
these four kinds are not the stop, and any other kind outside `QuadSourceIntegral` would still be.
None occurred: `scripts/classify_non_qsi.py` leaves 0 differences unclassified. The stop lasted
from about 01:58 to 02:05 UTC.

#### 2. R9 ran from a worktree — STRUCTURALLY REQUIRED

Prompt §5.2 calls for a worktree, and the dispatch gave it. The worktree has no venv, so R9 used
the main checkout's `venv/bin/python` by absolute path, with cwd and `PYTHONPATH` set to the
worktree. Its `--database` and launch log were absolute paths into the main checkout, so nothing
was written in the worktree.
- The registry recorded the run under the main checkout's `var/runs/`, as a worktree run is.
- The manifest records `git_head` `e6489519…` and the worktree's script path.
- `lsof` showed the cwd of the driver and of a Ray actor to be the worktree.
- The QSI values differ from R10's, which confirms that each tree ran its own code.

#### 3. `PYTHONDONTWRITEBYTECODE=1` on R9 only — IMPLEMENTATION CHOICE

This keeps `__pycache__` from being written into the worktree, which the dispatch says must not be
written. It changes no computation. R10 used R8's launch as written. Afterwards the worktree's
`git status --short --ignored` was empty, and `find -newer` the launch marker listed nothing.

#### 4. A shared `--job-name levin-phase-floor-05` — IMPLEMENTATION CHOICE

R8 used `handover-12`. R9 and R10 use the same job name, so that it could not become a value
difference. The job name reaches only `label`, which differs anyway, by its timestamp.

#### 5. The comparison's text form was run with `--max-rows 1000000` — IMPLEMENTATION CHOICE

The default prints 20 rows per table. The JSON, which §5.4 asks for, carries every row whatever the
setting.

#### 6. Three read-only scripts beyond the tabulation — IMPLEMENTATION CHOICE

These were written to characterise deviation 1's differences for the question:
- `classify_non_qsi.py`;
- `gkwkb_reuse_rows.py`;
- `parent_serials.py`, run after the answer to state precisely what `parents_digest` digests, as
  the orchestrator asked.

Each opens the stores sqlite `mode=ro`. The ten store files were hashed before the comparison and
at the end, and are identical.

#### 7. The first tabulation run failed — IMPLEMENTATION CHOICE (a fix to the agent's own script)

`tabulate_qsi.py` raised `TypeError` summing `WKB_Levin_num_regions`, which is NULL on the 291
rows without a WKB part. It was fixed to count NULL as 0 and re-run, exit 0. Only the second run's
output is kept.

#### 8. A lock `status` look made inside a compound command — UNINTENDED DRIFT

While stopped, one `status --ray-temp-dir /tmp/ray-sgk` was run in a command with `diff` and `git
status`, and not by `>>`. It is a look; it changed nothing. It was repeated correctly as row 5 of
the lock transcript.

#### 9. A stray `touch -r … /dev/null` — UNINTENDED DRIFT

A final `find -newer` command carried a leftover `touch -r state/r9_launch_marker /dev/null`. It
failed with "Permission denied" and wrote nothing.

#### 10. The stores' fingerprint was not re-taken — IMPLEMENTATION CHOICE

The dispatch allowed `store fingerprint` without `--write` "if useful". It was not run. Both
fingerprints come from the launcher at the end of each build. R9's files are byte-identical from
just after its build to the end. R10's files had their first hash taken at the start of turn 3,
after its fingerprint and before any read, and are identical to the end.

### Verification performed

| check | result | where |
|---|---|---|
| gate | both trees at their SHAs, clean; 13 runs finished; lock free; no 05 store | `out/gate.txt` |
| R9 | `done`, exit 0, pid 47770, `1b332ffa…`, `e6489519…`, `git_dirty` false | `out/r9_end_check.txt` |
| R10 | `done`, exit 0, pid 49752, `1b332ffa…`, `53084dcb…`, `git_dirty` false | `out/r10_end_check.txt` |
| worktree unwritten | `status --ignored` empty; `find -newer` empty, after both runs | `out/r9_end_check.txt`, `out/r10_end_check.txt` |
| value tables | 7 tables, 360,462 rows, all bit-identical | `out/store_value_diff_summary.txt` |
| non-QSI row columns | 4 kinds, 0 unclassified | `out/classify_non_qsi.txt` |
| `parents_digest` | a digest of serials; recomputed 291/291 in each store | `out/parent_serials.txt` |
| QSI tabulation | pairing = tool's identities; 0 `total` mismatches against the tool | `out/qsi_tabulation.txt`, `out/qsi_rows.csv` |
| readers | 6 × exit 0; store identical | `out/readers.txt` |
| stores | 10 files identical from before the comparison to the end | `state/store_sha256_before_diff.txt`, `state/stores_sha256_final.txt` |
| lock | acquire 0, set-pid 0 ×2, ray-processes 0 ×2, status 0 ×2, release 0 | `out/lock.log` |

### Observations not acted on

- **The tool has no notion of non-value columns.** Proposed as
  `[05-store-value-diff-compares-volatile-columns]` in `issues.md`. The laptop opens the issue for
  the prompt's wording and the tool.
- **One row moves by 28× its declared error.** At
  $(k,q,r,z_{\rm resp}) = (10^6, 3.162\times10^5, 10^6, 329.64)$, $\lvert\Delta{\rm total}\rvert$
  is 28.2 × `total_abserr`$_A$ and 29.4 × `total_abserr`$_B$, a relative 4.0e-5. Both are
  unconverged, at depth 6 → 7. Every other row moves by less than its declared error under either store (the next largest is 0.60 × `total_abserr`$_A$). Recorded, not
  judged. §6's L3 is where the declared error's honesty is scored.
- **The residual-memo counters depend on scheduling.** That is inferred from the two-way swaps. No
  same-code pair exists to prove it (`issues.md` P2).
- **A `levin-phase-floor-05-wider` run was `alive` in the registry at the end.** It is §6's
  Ray-free run, which is not this agent's. It was not touched.
- **The size of `out/`.** `store_value_diff.json` is 41 MB and `.txt` 10 MB. Whether to copy them
  back whole is the orchestrator's call. `store_value_diff_summary.txt`, `classify_non_qsi.txt` and
  `qsi_rows.csv` carry what the records cite.

### State handed on

- The lock is free (`status`: the lock directory does not exist; no Ray processes of
  `/tmp/ray-sgk`).
- The registry holds 16 runs. R9 and R10 are `done`; §6's `wider` is alive and is not this agent's.
- R9's store, primary `cfba7ac4…`, and R10's, primary `557066be…`, are in `var/datastores/`. Both
  carry fingerprint `1b332ffa…`, and neither has a sidecar.
- The main checkout is at `53084dc`, clean. The worktree is at `e648951`, clean and unwritten.
- The records' links assume they are copied to `docs/levin-phase-floor/05/rehearsal/`.

## §6 — the wider re-measurement (Mac Studio)

**Subject:** prompt 05 §6, run on the Mac Studio (mps026298) by the re-measurement agent. It was
Ray-free and took no lock. It ran alongside the R9/R10 agent, which held the lock for its own
builds.
- **Trees:**
  - $p$ is the code commit's parent `e648951`, in the git worktree
    `/Users/ds283/Documents/Code/SecondaryGWKit-lpf05-e648951` (detached);
  - $c$ is the code commit `120fdfe`, in the main checkout at the dispatch SHA `53084dc`, which adds
    only records.
- **Run:** `levin-phase-floor-05-wider-20261026T020005`. It ended `done`, 109/109 units, exit 0,
  02:00:05Z–02:11:54Z.

**Commit:** none here. The records are committed by the orchestrator in 05's records commit.

**Result:**
- **Gate L0 holds**, on both of its clauses and both rows.
- **L1 fails** (9 of 24 rows) and **L1q fails** (1 of 7), as written. Every failure is the tenfold
  region clause. 20 of the 21 failing rows are not capped under $p$ at all; the 21st is R5, capped
  by one 55-region chain.
- **L2 holds** (8 of 8, largest ratio 0.652).
- **L3 fails** (29 of 31). It fails on two uncapped mid rows, at 1.29 and 1.70.
- **On the 11 rows still capped under $p$**, the reading the user asked for:
  - the cap goes on every row: no group keeps a region at depth ≥ 18 under $c$;
  - regions fall to 0.16–4.7 % of $p$'s on 10 rows, and to 34.5 % on R5;
  - L1's rule holds on 9 of 10 and L1q's on 1 of 1;
  - L3's rule holds on 11 of 11.
- **No unit went over budget.**
- **The store is byte-identical**, fingerprint `0e32f24a…`, `problems: none`.
- **B1 alone moves 36 of 39 stored totals beyond their declared error** (issues W1).

### Predictions and verdicts

Prompt §6's predictions, unchanged. The same bytes are `state/predictions.md` and
`state/prompt_section6_predictions.raw`. `state/predictions.md` was written at 01:57:36Z, after gate
L0 held (01:57:24Z) and before the launch (02:00:05Z).

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

**Descriptive, not scored:**
- $|{\rm total}_c - {\rm total}_p| /$ `total_abserr`$_p$ on the depth-20 rows. This is 04's N4,
  wider, and now with the modulus correction in it too;
- cost and regions by class and band;
- `total_converged` before and after;
- the low band against the mid band.

**Verdicts** (`out/wider_05.txt`; "depth-20 row" read as the sample's depth-20 rows, as the user
agreed):

| | Number | Verdict |
|---|---|---|
| **Gate L0 (a)** | `QuadSource` at $p$: 360/360 stored nodes bit-identical in all nine columns, exactly B1's 3 nodes added, none removed, on both rows. The same in all 109 run units | holds |
| **Gate L0 (b)** | two in-process runs: result, group calls and region arrays bit-identical, on both rows | holds |
| **L1** | 9 of 24 rows true. Clause 1 holds on all 24. Clause 2 fails on 15: `low-G+r-1`, `low-G+r-2`, `low-G-r-1`, `low-G-2`, the nine drawn mid rows and R2 (none capped under $p$; $c/p$ 0.73–1.09), and R5 (capped, $c/p$ = 0.345) | **fails** |
| **L1q** | 1 of 7 rows true (R7, $c/p$ = 0.0016). It fails on the six drawn `Gqr` rows, none capped under $p$ ($c/p$ 1.00–1.35). The depth-18 regions' $z'$ the rule asks for: none exist, with 0 regions at depth ≥ 18 under $c$ and under $p$ on all six (maximum depth 2–7). So these failures are no evidence of a floor in $T$ | **fails** |
| **L2** | 8 of 8 controls true. Ratios: 0.161, 0.652, 0.0109, 0.00409, 4.3e-12, 6.6e-10, and 0 and 0 (the `NULL` controls are bit-identical) | **holds** |
| **L3** | 29 of 31 true. False on `mid-G-r-3` (ratio 1.29, relative 2.7e-6) and `mid-G-3` (1.70, 3.0e-7), neither capped under $p$. True on all 11 rows capped under $p$ (largest 0.166) | **fails** |

**The rows capped under $p$** (not scored; the user's reading, below). There are 11 of the 31:
`low-G+r-3`, `low-G-r-2`, `low-G-r-3`, `low-G-1`, `low-G-3`, R1, R3, R4, R5, R6, R7.

| row | $p$ depth, regions | $c$ 1e-7 depth, regions | $c/p$ | depth ≥ 18 under $c$, in $p$'s scored groups | L1 rule |
|---|---|---|---|---|---|
| `low-G+r-3` | 20, 1340 | 8, 33 | 0.0246 | 0 | true |
| `low-G-r-2` | 20, 15068 | 7, 42 | 0.0028 | 0 | true |
| `low-G-r-3` | 20, 6822 | 8, 40 | 0.0059 | 0 | true |
| `low-G-1` | 20, 3902 | 11, 34 | 0.0087 | 0 | true |
| `low-G-3` | 20, 831 | 7, 39 | 0.0469 | 0 | true |
| R1 | 20, 22852 | 8, 63 | 0.0028 | 0 | true |
| R3 | 20, 10457 | 5, 18 | 0.0017 | 0 | true |
| R4 | 20, 42830 | 8, 83 | 0.0019 | 0 | true |
| R5 | 20, 55 | 5, 19 | 0.345 | (no scored group) | **false** (clause 2) |
| R6 | 20, 5130 | 8, 28 | 0.0055 | 0 | true |
| R7 (`Gqr`) | 20, 16660 | 8, 27 | 0.0016 | 0 | true (L1q) |

**Descriptive:**
- **N4, wider:** $|c-p|$/`total_abserr`$_p$ is 0.0225–0.634 on the capped rows, and 2.48 on
  `low-G-3`. On the uncapped rows it is ≤ 0.932.
- **Cost:** low band, drawn rows, 28,054 → 288 regions and 101 → 4.5 s. Mid band, drawn rows,
  352 → 321 regions and 3.7 → 3.5 s (none capped). R1–R7, 97,999 → 250 regions and 545 → 2.4 s.
- **`total_converged`:** on the depth-20 rows it is False under $c$ on 21 of 31 at 1e-7 and 19 of 31
  at 1e-8. On 9 of the 11 capped rows it stays False.
- **Low band against mid band:** every drawn row still capped under $p$ is in the low band (5 of
  12). None of the 12 drawn mid rows is capped. 6 of 03's 7 mid rows are.

**The user's decision**, relayed by the laptop orchestrator after the launch, verbatim: *"§6
re-measurement: Carry on as you propose. I'm not sure I see all the consequences of this yet. I
guess we only care about rows that are capped under the new system (which I assume is what "p"
means)"*.
- The orchestrator clarified that $p$ is the code commit's parent, B1's code without 05's change.
  So "capped under $p$" means still capped after B1 and before 05.
- L1 and L1q are scored as written over the sample's rows. The capped-under-$p$ reading is reported
  beside them, unscored, and prominently.

### What shipped

Nothing is committed. Everything is under `var/levin-phase-floor-05/wider/`, and the checkpoints and
pickles are in the run directory.

**Scripts** (`scripts/`), each with a docstring naming its source and every change:
- `prov.py`: the two trees, their SHAs, and the header every output carries;
- `hash_store.py`: 04's, plus a journal/`-wal`/`-shm` check;
- `draw_sample.py`: the rule of `state/sample.md`;
- `harness.py`: 04's, with `QuadSource`'s `compute_quad_source.remote` and `ray.wait` stood in;
- `rebuild.py`: 04's, with `QuadSource` recomputed in-process and compared with the stored row
  (`qs_check`);
- `dispatch.py`: 04's `phases.py` cut down to its pass-through dispatcher and budget. It counts
  calls by kind and by `exact_remainder`;
- `worker.py`: 04's `run_remove.py` unit loop, one tree per process;
- `run_wider.py` and `launch.py`: the registered run;
- `l0.py`: gate L0;
- `analyse.py`: the scoring;
- `rec.sh`: records each command with its output.

**Outputs** (`out/`):
- `sample.txt`;
- `L0.txt` and `L0_final.txt`;
- `wider_05.txt`: every row, every group of the capped rows, the scores, and the B1 table;
- `fingerprint_05_wider.txt`;
- `cmd/`: every command run and its output, the failed attempts included.

**State** (`state/`):
- `sample.md`, `sample.json`, `predictions.md` and `prompt_section6_predictions.raw`;
- `L0/` and `L0_final/`: the gate's spec, ledger and pickles;
- `smoke/` and `smoke_attempt1_failed/`;
- `store_files_before`, `_prelaunch`, `_after` and `_after_fingerprint` (`.txt`);
- `launch.txt`, `launch_check.txt` and `registry_after.txt`;
- `analysis.json`.

**The run directory:** `var/runs/levin-phase-floor-05-wider-20261026T020005/`, 6.6 MB.
- `manifest.json`, `status.json`, `checkpoint.jsonl` (109 records), `stdout.log`, and `stderr.log`
  (15 scipy `IntegrationWarning`s, W9);
- `spec_{p,c}.json` and `worker_{p,c}_{hello,context,bye}.json`;
- 109 `unit_*.pkl`, the region arrays of every group call.

### Deviations from the prompt

1. **The `NULL`-depth controls are drawn from the store, not the census.** **STRUCTURALLY
   REQUIRED.** The census holds only rows with a Levin sub-interval, so it has no `NULL` row. The
   store was read with `sqlite3 mode=ro`, from the same 9280-row population (`state/sample.md`).
2. **Band controls must have a group containing $G$.** **IMPLEMENTATION CHOICE.** The cheapest seam
   rows otherwise have only an `r` group, on which the change cannot act. It is fixed in
   `state/sample.md` before any row ran.
3. **The top group's class.** It follows 01's T2 (the largest `elapsed`), and `Gqr` means the top
   group's regime is `Gqr`. The census writes `G-r` with an ASCII hyphen. **IMPLEMENTATION CHOICE.**
4. **How `QuadSource` is recomputed in-process.** **IMPLEMENTATION CHOICE**, within the prompt's
   "as `main.py` would compute it", with no production change.
   - The object is the factory's miss branch with `main.py`'s arguments, then `compute()`, then
     production's `store()`. `compute_quad_source.remote` and `ray.wait` are stood in.
   - The read-back is formed as `build()` forms it from a stored row, carrying the stored row's
     serial, because the `QuadSourceIntegral` row is keyed on it.
5. **The worktree.** **STRUCTURALLY REQUIRED** (the prompt runs two trees; the orchestrator set
   it up).
   - $p$'s code ran from the worktree through `PYTHONPATH`, with `-B` and
     `PYTHONDONTWRITEBYTECODE=1`.
   - Nothing was written there: `git status --short --ignored` is empty before and after.
   - The scripts and all outputs live in the main checkout's `var/`. So the manifest's `git_head` is
     `53084dc`, the $c$ tree (04's I4).
   - Each checkpoint payload records its own tree's `HEAD` and status, and every unit's evidence of
     where its modules were loaded from.
6. **One driver and one worker per tree.** One process cannot import two trees. The registered
   driver imports neither, and starts a $p$ worker and then a $c$ worker. Their unit records come
   over a pipe. **STRUCTURALLY REQUIRED.**
7. **The launch is `nohup sh -c '<driver>; exit $?'`,** so `sh` stays the driver's parent (CLAUDE.md
   rule 5). The `sh` pid (54763) was recorded first, then the driver's (54766). **IMPLEMENTATION
   CHOICE.**
8. **The budget** is 3× the row's $p$ wall time, measured as 04 measured it (around `compute()` and
   `ray.get`, after the rebuild). It is raised from the dispatcher at `theta_mod_2pi`'s entry.
   **IMPLEMENTATION CHOICE.** Rows without a Levin call can never be over budget. None was.
9. **L0 was run twice.** **IMPLEMENTATION CHOICE.**
   - The first run held (01:57:24Z). The predictions were written after it.
   - The smoke test then found that `worker.py` crashed on a `None` max depth (the `NULL` rows).
     The fix touches only that bookkeeping.
   - L0 was re-run with the final scripts (`out/L0_final.txt`) and gave an identical body. The
     launch followed.
10. **Two attempts failed before results** and are kept in `out/cmd/`:
    - the first L0 run (a module-lookup error in the worker's evidence code);
    - the first smoke test (the `None` max depth).
    **UNINTENDED DRIFT** in the harness, caught before any scored unit.
11. **The smoke test ran $c$ units** on `low-G+r-1` and `ctrl-NULL-1` before the run. It was after
    `state/predictions.md` was written, so nothing in the predictions could be influenced.
    **IMPLEMENTATION CHOICE.**
12. **"Depth-20 row" in L1, L1q and L3** is read as the sample's depth-20 rows (stored depth 20),
    since §6's "The rows" defines them that way. The rows at depth 20 under $p$ are reported beside
    them, unscored. This was proposed at the launch report, and the user agreed ("Carry on as you
    propose"; quoted above). The user said the capped-under-$p$ reading is the one that matters.
    **Recorded as the user's decision**, made after the launch and before any scoring; it changes
    no rule. **STRUCTURALLY REQUIRED** *(classified on committing by the laptop orchestrator, on
    the user's agreement of 2026-10-26, review check 10)*: L1's rule as written presumes a row
    capped under $p$ (`issues.md` W3), so a reading had to be fixed, and the user fixed it.
13. **The analysis was run twice.** The second run only adds, for L1q's failing rows, the count of
    depth-18 regions (zero), which the rule asks to report. The rest is identical
    (`out/cmd/17_analyse_v1.txt`). **IMPLEMENTATION CHOICE.**
14. **R9/R10's Ray builds shared the host** with the first minutes of this work. Wall times, and so
    the budgets, carry that load. No unit came near its budget: the largest $c$ wall is 0.80 s.
    **Recorded; not a deviation of method.** **IMPLEMENTATION CHOICE** *(classified on committing
    by the laptop orchestrator, on the user's agreement of 2026-10-26, review check 10)*: the
    orchestrator note ran §5 and §6 side by side, as the prompt allows; it changed no measurement.

### Verification performed

- **The trees.**
  - Main checkout at `53084dc` and the worktree at `e648951`, both with empty status, at the start,
    before launch, on every checkpoint record, and at the end (`state/registry_after.txt`).
  - The worktree's `git status --short --ignored` is empty.
- **Module provenance.** Each worker checked that `ComputeTargets`, `primitive_phase`,
  `QuadSource`, `QuadSourceIntegral`, `GkSourcePolicyData`, `StoreFixtures`, `main.py` and the
  original `theta_mod_2pi` came from its own tree, and refused to run otherwise.
- **Which phase each tree used.** The dispatcher counted the `theta_mod_2pi` calls:
  - under $p$, $G$ 24,192,372 and $T$ 14,924,742, with no `exact_remainder` attribute;
  - under $c$, $G$ 268,136 with `exact_remainder=True` and $T$ 167,562 with `False`.
- **The store.**
  - All six files' SHA-256, size and mtime are identical before, before launch, after the run and
    after the fingerprint. No journal, `-wal` or `-shm` file appeared.
  - `store fingerprint`, without `--write`, gives `0e32f24a…`, `problems: none`, "matches the
    fingerprint taken … at b90752d".
- **The run.**
  - It ended `done`, 109/109, in the planned order, exit 0, with none over budget and none in
    error.
  - Ray was never initialised: each worker checks at its end.
  - No lock was taken by this work.
  - The registry shows nothing `running` after it.
- **`QuadSource`.** L0 (a)'s comparison held in every one of the 109 units, not only on the gate's
  two rows.
- **Groups.** $p$ and $c$ made the same sequence of Levin group calls (regime, label, `x_span` bit
  for bit) on every row. So the sub-intervals and groups match one to one.

### Observations not acted on

All are in `issues.md`:
- **W1:** B1 moves 36/39 stored A3 v2 totals beyond their declared error, by ratios of 5.2 to
  1.5e5. One total changes sign.
- **W2:** under B1's code, 20 of the 31 stored depth-20 rows are no longer capped, R2 among them.
- **W3:** L1 and L1q's tenfold clause fails by construction on rows not capped under $p$.
- **W4:** R5's cap under $p$ is a single 55-region chain.
- **W5:** L3 fails on two uncapped mid rows: the error declared at 1e-7 misses the change to 1e-8.
- **W6:** on `low-G-3`, $|c-p|$ is 2.48 × `total_abserr`$_p$.
- **W7:** `total_converged` stays False on most depth-20 rows under $c$.
- **W8:** $c$'s region counts on R1–R7 against 04's arm A.
- **W9:** scipy `IntegrationWarning`s in both trees.

### State handed on

- **Nothing is running.** The run is `done`. No lock was taken.
- **The store is unchanged**, fingerprint `0e32f24a…`.
- **The worktree** `SecondaryGWKit-lpf05-e648951` is untouched, and is the orchestrator's to
  remove.
- **For the orchestrator:**
  - copy `out/`, `scripts/`, `state/` (the `.pkl` files under `state/L0*/` and `state/smoke*/` may
    be left out), `log.md`, `verification.md` and `issues.md` to `docs/levin-phase-floor/05/wider/`;
  - append this section to `logs/05-reanchor-g-phase.md`;
  - put `verification.md` in `VERIFICATION.md` §5. Its links assume the `05/wider/` location.
- **For the board:**
  - on the rows still capped under $p$, layer 1(a) removes the cap on all 11, and L3 holds on all
    11;
  - the scored L1, L1q and L3 fail only on rows that B1 had already uncapped, or on R5's short
    chain;
  - W1 is to be opened as an issue, as the laptop proposed.
