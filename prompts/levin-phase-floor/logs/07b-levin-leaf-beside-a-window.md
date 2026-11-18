# levin-phase-floor 07b — the Levin leaf beside a window, and how honest the census is

> **Part B, §1, §3 and §4** (the code, its tests and these records), written by the executing agent on the laptop,
> on the prompt as amended by its §9 and, after Part B's stop, its §10 (2026-11-16). **Part A, §2** (the
> measurement and its extension, on the Mac Studio) is recorded in `docs/levin-phase-floor/07b/measurement/log.md`;
> prompt §9 holds the rulings. **Part C, §5 and §6** (R31 and the census) runs later on the Mac Studio; the
> orchestrator appends its record.

**Subject:** prompt 07b §1, §3 and §4, as amended by §9 and §10 (Part B), on the laptop, in the main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit`, branch `handover-remedial`.
- **Dispatch SHA:** `c768c4ad` (pushed), the amendment at §2.7's stop. The parent module is
  `c768c4ad:AdaptiveLevin/levin_quadrature.py` (SHA-256 `4ea1948b…`), 07a's code (`git diff --stat 1ec734d7
  c768c4ad -- AdaptiveLevin ComputeTargets LiouvilleGreen tools` lists only `89c368f2`'s four docstring-only
  files). The parent was run from a temporary detached worktree at `c768c4ad` in the session scratchpad, the main
  checkout's `venv` symlinked into it; (h) from a second such worktree with Part A's variant (h) copied over its
  module. Both were removed afterwards (and once before, at the stop; recreated for the resumption).
- `git log --oneline c768c4ad..HEAD` was empty at the start. **Stopped before the commit** (§8, a LiouvilleGreen
  test failing at the commit that passes at the parent; below, **The stop, and §10**), and **resumed on §10** at
  `7b82a0cd`, the orchestrator's amendment, the only commit to land meanwhile (it changes the prompt, the board's
  Decisions and 07b row, and the campaign README); the board was merged by reading.
- `runregistrykit list` at the start: 91 runs, 89 finished, 2 `unknown` (pre-registry), nothing `running` or stale.
  No Ray was started, no store was opened, no run was registered, and nothing was written under `var/`.

**Commit:** the code commit of this log (one commit, not pushed), on `7b82a0cd`.

**Result:**
- **§1.1–§1.5 are in `AdaptiveLevin/levin_quadrature.py`:** the rule as a module-level function
  `_leaf_rate_rule`, the two constants with their basis, `converged` unchanged, the two counts, the docstrings.
  **It equals (h) bit for bit** on every call compared on the laptop: 22 test calls, the LiouvilleGreen problem of
  the stop at five rungs, and 4,320 LiouvilleGreen draw calls (Verification 3).
- **§3:** a new module `AdaptiveLevin/tests/test_levin_rate_rules.py`, 13 tests: (a) eight, (b), (c), (d), (e)
  two. (b) and (c) are converged and dishonest at the parent (7.62 and 102.8 times their abserr) and honest and
  converged at the commit. **Every breakage caught** ((i)–(vi), (ii) also by the LiouvilleGreen test of §10).
- **Existing assertions changed:** in three AdaptiveLevin modules, each where the range rule acts (below), and,
  under §10, one LiouvilleGreen assertion. No test goes from converged to unconverged.
- **The seven roots:** AdaptiveLevin 116 → 129, the rise equal to the 13 tests added; ComputeTargets 781, CosmologyModels 39, LiouvilleGreen 156 (skipped 1), ObjectFactories 280, RunDrivers 127 and tools 64, each equal to its baseline.
- **The LiouvilleGreen draws:** no test newly failing (80 of 80 units pass under the parent, the commit and (h)); the 509 calls in which the rules do not act are bit-identical to the parent; the rules act in 3,811 of 4,320 calls (34,018 range-rule refusals, no edge-rule refusal), no `converged` flag changes; every call equals (h)'s bit for bit.

## What shipped

### §1, the code (`AdaptiveLevin/levin_quadrature.py`)

1. **§1.1, the rule:** `_leaf_rate_rule(x_span, chebyshev_order, rate_fns, window_edge_ends) -> (refused, rule,
   q)`, module-level, after `_trust_noise_yardstick`. The nodes are `chebyshev_matrices((lo, hi),
   max(chebyshev_order, _LEVIN_MINIMUM_ALLOWED_ORDER))`; `window_edge_ends` are the leaf's ends that equal, as
   floats, a merged window's `lo` or `hi`. Non-empty: the edge rule on those ends only (neighbour `grid[-2]` at
   `lo`, `grid[1]` at `hi`), `rule = "edge"`; empty: the range rule over every node, `rule = "range"`; `q` the
   largest over phases and ends, starting from 1, `+inf` where a minimum is 0; refused iff `not (q <= max)`.
   `rate_fns is None` returns `(False, None, 1.0)`. Each rate is evaluated at one float node at a time and every
   expression is (h)'s, so the bits are (h)'s.
2. **In `_adaptive_levin`:** `window_edges`, a frozenset of the merged windows' edges (empty without a location);
   after the tree's `trusted = …` in `_check_leaf`, `if trusted and not leaf.is_direct and rate_fns is not None`,
   the rule on `(min(a, b), max(a, b))` with the touching ends; a refusal sets `trusted = False` and counts
   `num_edge_rule_refusals` or `num_range_rule_refusals`. Nothing else in `_check_leaf` moves: `ratio`, `delta1`,
   `delta2`, the untrusted declaration at the cap, `_exempt_floor_limited`, the windows, the C2 gate.
3. **§1.2, the constants:** `_LEVIN_EDGE_RATE_RATIO_MAX = 2.0` and `_LEVIN_LEAF_RATE_RANGE_MAX = 10.0`, after
   `_LEVIN_WINDOW_PIECE_PHASE`, with one comment stating each one's basis, margin and where it does not reach
   (§9.3), the reason a wider window is not the remedy, and the range rule's cost and the chase issue (lines
   579–625); the function `_leaf_rate_rule` at line 1324.
4. **§1.3:** `converged` unchanged; a refused leaf still refused at the cap is untrusted and leaves its call
   unconverged (test (d)).
5. **§1.4:** the result keys `num_edge_rule_refusals` and `num_range_rule_refusals`, ints, over the whole run.
6. **§1.5, the docstrings and comments rewritten** (line numbers at the commit):
   - the module docstring's error-control item, on the trust check, now ending with the rate rules (lines 40–54);
   - its C2-gate item, the stationary-point blindness extended to the leaf beside a window (lines 133–139);
   - its "Stationary points are located" item, the edge rule and why the window is not widened (lines 154–159);
   - the comment above `_LEVIN_TRUST_MAX_RATIO`, a paragraph "What the test cannot see" (lines 447–455);
   - the comment on `_LEVIN_WINDOW_HALF_PHASE`, a wider window measured not to remove the leaf's error (§2.7,
     (b4), (b8), (g1), (g2)) (lines 553–560);
   - `_adaptive_levin`'s docstring on the trust check one level down: the trusted bullet, a new bullet on the rate
     rules, and the closing sentence on 06b's common error (lines 4141–4163 and 4175–4181);
   - `_check_leaf`'s docstring, a paragraph on the rate rules (lines 4866–4876);
   - `adaptive_levin_sincos`'s `num_untrusted_regions` and the two new counts (lines 6160–6181);
   - the result dict's comment on `num_untrusted_regions` and the two counts (lines 5619–5628).

### §3, the tests (`AdaptiveLevin/tests/test_levin_rate_rules.py`, new, 13 tests)

All without Ray or a datastore. References are closed forms (the error function), by
`docs/levin-phase-floor/07b/code/test_references_07b.py` at 40 and 60 digits (agreeing to 8.9e-39 and 3.8e-37
relative; `out/test_references_07b.json`). The parent's values are held as float hex, measured by running
`c768c4ad`'s module; where a test needs the parent's call it runs the commit with `_leaf_rate_rule` patched to refuse
nothing (the parent's code path) and asserts the parent's hex bit for bit.

| § 3 | test | what it asserts |
|---|---|---|
| (a) | `TestTheRuleDirectly` a1–a8 | edge q exactly 2 not refused, `nextafter(2, inf)` refused, at `lo` and at `hi`; range q exactly 10 not refused, the next double refused; a zero rate gives `inf` and a refusal under either rule; only window-edge ends tested (a jump of 100 at `hi`'s neighbour unseen with `lo` the edge, seen with `hi`); a touching leaf of node range 1000 and edge q 1.5 not refused, a non-touching leaf with an end jump of 5 not refused; the largest q over two phases decides, in either order, under either rule; no rates: `(False, None, 1.0)`; the order clamped to 8 |
| (b) | `TestEdgeMechanism.test_b_leaf_beside_a_window` | the edge mechanism, below |
| (c) | `TestRangeMechanism.test_c_leaf_with_a_wide_rate_range` | the range mechanism, below |
| (d) | `TestAtTheDepthCap.test_d_refused_at_depth_max` | $\theta = 2e^{4x}$ on [0, 1.5], f = (e^{-x}, cos x), depth_max 1: both depth-1 leaves refused by the range rule ($e^3 = 20$ over each), returned untrusted at the cap with `value_kind` "own", each declaring max(residual, δ1)/(1 − r) (r = 1.6e-5, 8.5e-4); unconverged; the parent (rule refusing nothing, its hex) converged with the same abserr |
| (e) | `TestNotTouched.test_e1_constant_rate` | $\theta = 10^3x$, f = (e^{-x}, cos 30x): four trusted Levin leaves, the rule called on each and passing at q = 1; bit-identical to the parent's hex; both counts 0 |
| (e) | `TestNotTouched.test_e2_clenshaw_curtis_leaves_untested` | $\theta = 0.04e^x$ on [0, 6], F_23: every leaf Clenshaw–Curtis (16.1 rad < 6π), [0, 3] trusted with node range 20.1; the rule never called; bit-identical to the parent's hex; both counts 0 |

**(b), the edge mechanism.** $\theta = (c/2)(x - x_s)^2$ on [0, 1], $c = 1.19\times10^9$, $x_s =
0.8186383592305085$, f = (e^{0.744x}, cos 109.28x), order 24, atol 1e-7, rtol 1e-7; one root at $x_s$, one window.
**At the parent:** 10 regions, converged, **7.62 times its abserr** from the reference. The excess is carried by one
leaf (references on every leaf of the partition, `out/test_references_07b.json` "edge_leaves"): the Levin leaf
[0.409, 0.8185356] at depth 1, its upper end the window's lower edge, trusted at r = 0.125 with e′ = 1.97e-9 and a
true error of 2.03e-7 = **103 e′**; every other leaf within 0.75 of its e′. Its first grid spacing at the edge is
18.5 ℓ, and |θ′| differs by a factor 19.5 across it. **At the commit:** the edge rule refuses that leaf (1 refusal),
the call is honest (0.0044 of its abserr), converged, 16 regions. Asserted: the location, the window's `lo` equal to
the carrier's end, the counts, converged, honest; the carrier refused by `_leaf_rate_rule` at q > 10; the parent's
hex reproduced with the rule refusing nothing, converged and more than 5 abserr from the reference, its partition
holding the carrier.

**(c), the range mechanism.** (b)'s integrand on the carrier's span alone, [0x1.a317190ba9ba3p-2,
0x1.a317190ba9ba3p-1], order 24, atol 1e-8, rtol 1e-7: θ′ has no zero on it (its upper end 1.03e-4 below $x_s$),
nothing is located, and |θ′| ranges by 3,984 over its nodes. **At the parent:** one region, the span trusted at
depth 0, converged, **102.8 times its abserr**. **At the commit:** 3 range refusals, honest (6.0e-6), converged, 10
regions. Robust over atol 3e-9 to 1e-7 at both rtols (explored before choosing).

**The search for (b) and (c)** (§3: "If no call for (b) or (c) is found dishonest and converged at the parent, the
agent stops and asks, with the search it made"):
- family (R) with F_23 (c 1e8–1e11, four x_s, orders 16 and 24, rtol 1e-7 and 1e-8, atol 1e-6 to 1e-2 of |I|):
  none; and the P family (θ = Ae^x − ax, A 1–1e4): none. At leaf level the leaf beside the window has r ≈ 2–4
  there: its error is common, but the edge's own differences grow as ℓ/s, so the trust check refuses it;
- a random scan of 600 quadratic-phase calls (c 1e5–1e11, x_s, ω, f = (e^{gx}, B cos hx), atol, rtol, order):
  2 converged and dishonest at the parent, both caught by the edge rule. (b) is the first, with its parameters
  rounded (c, g, h) and the dishonesty kept; the amplitude's cos(hx) makes the leaf's halves and quarters disagree
  by more than its unresolved edge does, so r < 1/4;
- (c) the carrier's span as a call of its own. Scans of no-window exponential and power-law phases found only
  marginal dishonest calls (1.0–1.6×), most of them not touched by the rule (below, Observations 2).

### Existing assertions changed (§3, "Existing assertions")

Each is where the range rule acts; none goes from converged to unconverged. Every call below equals (h)
(`out/compare_h_07b.txt`).

1. **`test_levin_trust_yardstick.py`, `TestCodeIsTheVariants.test_bit_identical_to_the_variant`** (06m's test 2,
   the commit against 06l's route-B variant). (i) On every cell the key set also admits the two counts, and the
   compared records leave them out (constant `RATE_RULE_KEYS_07B`), as 07 admitted the guard's keys. (ii) The rule
   acts on one cell, **si_1000_exp: 6 range refusals**, 3 regions become 16, converged under both; there the
   commit is compared with `_leaf_rate_rule` patched to refuse nothing (`RATE_RULE_CELLS_07B`), the code path of the
   yardstick the test certifies. The test asserts that every other cell has both counts 0.
2. **`test_levin_leaf_values.py`, `TestTrustedLeafReturnsQuarters.test_trusted_leaf_returns_quarters`** (06g's
   test 1). si_1000_exp's `identical` block (abserr, converged and every region's span, depth and error equal to the
   parent's bit for bit) is skipped there: **6 range refusals**, abserr 2.218e-11 → 1.479e-14, true error 1.095e-14
   → 5.97e-15, regions 3 → 16, honest and converged under both. Asserted instead: 6 range refusals there, both
   counts 0 on every other cell (`TEST1_RATE_RULE_CELLS_07B`). Every other assertion stands.
3. **`test_levin_roundoff_exit.py`, `TestRoundoffExitPremise.test_exit_against_its_premise_levin_branch`** (06b's
   test 1, θ = 100e^x on [0, 20]). Its assertions on the new code (converged, **no phase-limited region**, honest,
   within rtol) are kept, now on the commit with `_leaf_rate_rule` patched to refuse nothing (07a's code: 3 regions,
   abserr 2.218e-11). Added, on the commit as it runs: **9 range refusals**, 16 leaves of width 1.25 at depth 4 (θ′
   ranging by e^{1.25} = 3.5 over each), **9 of them floor-limited**, none the span; converged; honest (true error
   5.0e-16, abserr 9.66e-15) and within rtol.
4. **Under §10, `LiouvilleGreen/tests/test_three_bessel.py`,
   `TestPhaseGroups.test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy`:** the tenfold-improvement
   assertion replaced by a stall, below.

**Where a patched assertion's cell is checked unpatched** (asked by the orchestrator at the resumption):
- *Item 1, si_1000_exp:* the same cell, the same call, is run unpatched by item 2's test (06g's test 1 runs
  `_adaptive_levin` on it, as `adaptive_levin_sincos` does): it asserts the shipped code there returns each trusted
  leaf's quarters sum and every other leaf's V_P bit for bit, the call's value as their ascending sum, at least one
  quarters leaf, not node-limited, honest against the reference, a true error at most the parent's, and now 6
  range refusals and no edge refusal. Item 1's own test asserts, unpatched, that the rule acts there exactly 6
  times by the range rule and on no other cell. `compare_h_07b.txt` checks the unpatched call equal to (h).
- *Item 3, θ = 100e^x:* the same test, in the same cell, runs the shipped code unpatched after the patched block:
  converged, 9 range refusals and no edge refusal, 9 phase-limited leaves, every one at depth 1 or more, honest, and within
  rtol.
- *(b), (c), (d) of the new module*, which patch only to reproduce the parent: each asserts the shipped code
  unpatched first, as listed above.

## The stop, and §10

**The stop** (2026-11-16, before the commit). The first run of the six other roots at the working tree failed
`LiouvilleGreen/tests/test_three_bessel.py`'s `test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy`
(156 tests, 1 failure, 1 skipped), on its assertion that the true error improves tenfold from rtol 1e-8 to 1e-14:
`6.994405055138486e-15 not greater than 7.16093850883226e-14`. It passes at the parent (run in the parent's
worktree) and fails under (h) (in (h)'s). quad_JJJ's true error per rung (four Levin calls each):

| rtol | parent | commit (= (h)) | range refusals at the commit |
|---|---|---|---|
| 1e-8 | 1.42e-13 (58 regions) | 6.99e-15 (154 regions) | 49 |
| 1e-10 | 1.45e-13 | 6.94e-15 | 45 |
| 1e-12 | 7.19e-15 | 6.99e-15 | 34 |
| 1e-13 | 7.24e-15 | 7.24e-15 | 21 |
| 1e-14 | 7.16e-15 | 7.16e-15 | 0 |

The other five roots passed (ComputeTargets 781, CosmologyModels 39, ObjectFactories 280, RunDrivers 127, tools 64).
Prompt §8 makes that a stop, and making it pass needed a LiouvilleGreen assertion changed, outside §1 and the
dispatch's paths, so the agent stopped with its work uncommitted and reported. The orchestrator reproduced it; the
user ruled *"Yes, write the §10 amendment"*, written as §10 (`7b82a0cd`).

**§10.3, as built:**
- the assertion `abs(results[1e-8].value - analytic) > 10 * abs(tightest - analytic)` is replaced by: over the five
  rungs, the largest true error is at most twice the smallest (measured spread 1.04 at the commit, 20.3 at the
  parent). The other assertions (honest at every rung, `phase_limited` at 1e-14, the floor above 1e-14|analytic|)
  are unchanged;
- the docstring's statement of what is asserted says "stalls across the rungs" in place of "improves from rtol 1e-8
  to the tightest rung", and a paragraph after the 06d one (left as written) records the 07b numbers and cites §10;
- **the AST check:** `docs/levin-phase-floor/07b/code/ast_check_07b.py` parses the module from `git show c768c4ad:`
  and from the working tree, removes every docstring, and compares node by node: only
  `TestPhaseGroups.test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy` differs
  (`out/ast_check_07b.txt`). Nothing else under `LiouvilleGreen/` changed;
- breakage (ii) also fails it (Verification 2).

### Records (`docs/levin-phase-floor/07b/code/`)

- `test_references_07b.py`, `out/test_references_07b.json`: (b)'s and (c)'s references, and every leaf of (b)'s
  parent partition.
- `compare_h_07b.py`, `out/compare_h_07b.{txt,json}`: the commit against (h) and the parent on 22 calls.
- `lg_draws_07b.py` (07a's harness copied, changed where marked "prompt 07b": the two counts recorded, left out of
  the result digest; three trees; the report by whether the rule acts), `out/parent/`, `out/commit/`, `out/h/`,
  `out/lg_draws_report.{txt,json}`, `out/lg_run_stdout.txt`.
- `breakages_07b.py`, `out/breakages_summary.txt`, `out/breakages/`.
- `ast_check_07b.py`, `out/ast_check_07b.txt`.
- Each script is `black`-formatted before its recorded run, and its outputs record its SHA-256; no `CLAUDE.md`
  change is needed.

## Predictions and verdicts

### Part B's acceptance (§7 item 2)

| Rule | Number | Verdict |
|---|---|---|
| §1 as amended | §1.1–§1.5; equal to (h) on 22 test calls, the stop's five rungs and 4,320 draw calls | holds |
| every test of §3 passing | the seven roots OK (Verification 1) | holds |
| every breakage caught | (i)–(vi), each by the test named, (ii) also by the LiouvilleGreen test (Verification 2) | holds |
| every changed existing assertion listed | three AdaptiveLevin tests and §10's LiouvilleGreen assertion (above) | holds |
| the seven roots rising only by the tests added | AdaptiveLevin +13 = tests added; the other six equal | holds |
| the LiouvilleGreen draws with no new failure | 0 of 80 units newly failing; 509 of 509 untouched calls bit-identical | holds |
| §3 (b), (c) found dishonest and converged at the parent | (b) 7.62, (c) 102.8 | holds |

### §5's predictions (R31), copied before they run — scored in Part C, 2026-11-18

As 07a §5, with records under `var/levin-phase-floor-07b/rehearsal/`, slug `r31-07b`, `--prompt 07b`,
`--job-name levin-phase-floor-07b`, and the fingerprint `1b332ffa…`. The parent rehearsal is **R30**
(`levin-phase-floor-07a-r30-07a-20261112T113806`, `var/datastores/levin-phase-floor-07a-r30*`, kept),
built at `4bf50d34`, whose code is `1ec734d7`'s.
1. **S0′ — R30 is the parent.** `git diff --stat 1ec734d7 <code commit's parent> -- . ':!docs'
   ':!prompts'` lists only `CLAUDE.md` and `89c368f2`'s four files, each AST-equal with docstrings
   removed. **Holds** if so; if not, it stops and asks.
2. **The lock:** `acquire`, `set-pid` at the launch, `release` after the readers, each a shell command
   of its own.
3. **R31 against R30,** by `tools.store_value_diff` with `--json` and 05's `classify_non_qsi.py` (job
   names normalised in a copy): outside `QuadSourceIntegral` every value table identical, every other
   difference of the four volatile kinds, no row in one store only; inside it, every moved row
   tabulated, with the rows whose `total_converged` changes in each direction.
4. **S3 — consistency.** Every row with finite `total_abserr` in both has
   $|{\rm total}_{31} - {\rm total}_{30}| \le {\rm total\_abserr}_{30} + {\rm total\_abserr}_{31}$, or is
   one §9 predicts to move further, listed.
5. **S1, S2 and S4** re-scored on R31 as 07a §5 items 5 and 6: V1's twelve against 06g's references,
   P1's two rows against 07's `p1_references`, and R29's four blocks against their block references.
6. **The readers** run on R31, exit 0, and the store is byte-identical before and after. **The lock** is
   released and `ray-processes` lists nothing. **R30 and R31 are kept.**

### §6's predictions (the census), copied before they run — scored in Part C, 2026-11-18

07a's census harness (`docs/levin-phase-floor/07a/census/scripts/`), copied, changed where marked
"prompt 07b", re-run at the code commit against 07a's census (`levin-phase-floor-07a-census-20261112T132744`,
checkpoint `461df2ec…`) as the "before". Diagnostics as §9 sets them, recording on every call whether
the fix acted and where.

**Gate U0, before the run:**
- **The parent's code is 07a's:** as S0′.
- **The harness reproduces 07a's census at the parent,** bit for bit, on 06k's 100 K0 units. One
  registered run, slug `k0`.
- **The instrument changes nothing:** a unit where the fix acts and one where it does not, at the new
  code, with and without it, bit-identical.
- **The new code is deterministic:** two runs of one unit where the fix acts, bit-identical.

**Predictions, fixed before the run** (README §5 rule 3), copied into `state/predictions.md` unchanged,
with §9's additions:
- **U1 — nothing else moves.** Every call in which the fix does not act equals 07a's call bit for bit,
  in the call's outputs. Every row all of whose calls are such equals 07a's in `total` and
  `total_abserr`. **Holds** if true at both rtols.
- **U2 — no Levin leaf is returned across a stationary point** (07a's W2).
- **U3 — every phase with something to locate is located** (07a's W3).
- **U4 — honest where it is referenced.** Every call referenced in Part A (V4's 25 and 07a's census's
  other eight V4 calls, R1, R2, R3, and `1-34599` and `3-678`), scored from the census against its
  reference by 07's rule. **Holds** if every one is honest, or unconverged under §9's rule.
- **U5 — the gate is consistent, and the guard is a backstop** (07a's W5, on gated solves).
- **U6 — 06b's 24 are honest,** against $S_{10}$ by 06b's rule, or unconverged.
- **U7 — the pairs:** $r_g$ over all 2,484 pairs; every pair has $r_g \le 1$, or a member unconverged.

**Descriptive, not scored:** the calls where the fix acts, by class, band, $k$ and rtol; `converged`
changes in each direction; the abserr ratio against 07a's; regions and cost against 07a's.

### §9.5's additions to §5's and §6's predictions (U8 among them), copied before they run — scored in Part C, 2026-11-18

- **U1 counts** a call untouched if `num_edge_rule_refusals + num_range_rule_refusals` is 0, and compares
  value, abserr, `converged`, `num_regions` and the maximum depth. Expected to cover about a fifth of
  the calls (R2: 11 of 60).
- **U4 is predicted to hold,** with exactly v04 and R2-28 (both rtols) unconverged among the referenced
  calls, R2-28 at 1.107 of its abserr.
- **U6 is predicted to hold.** The four of 06b's 24 dishonest at 07a are honest under (h)'s replays;
  the other 20 were not replayed, and the prediction for them rests on (h) leaving no referenced call
  converged and dishonest.
- **U7 is predicted to hold:** both pairs pass under (h)'s replays; the other 2,482 were not replayed.
- **U8 — the code is (h).** Every census call that Part A replayed (§2.1's and §2.4's, both rtols)
  equals (h)'s replay record bit for bit in value, abserr, `converged`, regions and maximum depth.
  **Holds** if all do.
- **S3 (§5)** names no row. A row beyond S3's bound is reported with the refusals in its calls, and S3
  is scored as written.
- **Descriptive, added:** the refusals by rule, class, band, $k$ and rtol, and, by the harness's
  instrument (marked "prompt 07b"), each refused leaf's span, depth and $q$; every call newly
  unconverged with its depth-20 leaves' routes and whether each descends from a refused leaf (X3's
  measures, for the issue in §9.6); regions and wall against 07a's.

**Part B's review** replays v08, v13 (a control the edge rule acts on), R2-22 (untouched: 45 trust checks, all
passing the range rule) and R2-03 from copies of their captures at the code commit: each equal to (h)'s
replay record in `float.hex`, and R2-22 equal to 07a's census too.

## Deviations from the prompt

1. **`IMPLEMENTATION CHOICE`** — the rule is skipped when `rate_fns is None`, the explicit case §1.1 adds; (h)
   computed the leaf's grid and tested no rate there. No output differs. The touching ends are passed as a tuple of
   the leaf's ends that are window edges; `rule` is `"edge"`, `"range"` or `None`.
2. **`IMPLEMENTATION CHOICE`** — the test module's name, `test_levin_rate_rules.py`, and its parts: (a) as eight
   tests, (e) as two.
3. **`IMPLEMENTATION CHOICE`** — (b) and (c) run with atol well above rtol|I| (1e-7 and 1e-8, about 1e-3 of |I|
   for (b)). The request must be loose enough for the leaf beside the window to be trusted before it is bisected,
   which in the census the call's own total does; §3 names family (R) at the census's rtols only as a starting
   point, and it gave no such call (the search above).
4. **`IMPLEMENTATION CHOICE`** — (c) is (b)'s carrying leaf's span as a call of its own: a leaf whose |θ′| falls
   towards an end near a stationary point beyond the span. Its error has the same origin as (b)'s; what the test
   exercises is that the code sees no window and the range rule refuses it.
5. **`IMPLEMENTATION CHOICE`** — the tests reproduce the parent by patching `_leaf_rate_rule` to refuse nothing,
   asserting the parent's hex (measured by running `c768c4ad`'s module), rather than running the parent; and the
   three changed existing tests use the same patch where they certify an earlier code path (the list above, with
   the unpatched checks of the same cells).
6. **`IMPLEMENTATION CHOICE`** — the result dict's comment on `num_untrusted_regions` is rewritten beside the two
   new keys: its old text said every untrusted leaf declares max(residual, δ1 + δ2), which was already inexact.
   §1.5 does not list it; §1 item 5 asks for every comment describing the trust check.
7. **`IMPLEMENTATION CHOICE`** — records beside the draws' harness in `docs/levin-phase-floor/07b/code/`: the
   references, the comparison with (h), the breakages and the AST check.
8. **`IMPLEMENTATION CHOICE`** — the draws run a third tree, (h), and compare the commit with it call by call
   (§9.5 U8 on the laptop's problems); 07a's route instrument is kept unchanged and records zeros.
9. **`STRUCTURALLY REQUIRED`** — the stop and §10: `LiouvilleGreen/tests/test_three_bessel.py` changes, one test,
   as §10.3 writes it.
10. **`IMPLEMENTATION CHOICE`** — breakage (vi) is "`if refused and leaf.depth < depth_max`": a refused leaf at the
    cap stays trusted, so it counts as trusted in `converged` (and keeps the trusted value and declaration).
11. **`UNINTENDED DRIFT`** — while setting up the scratch worktrees I appended a duplicate `venv` line to the repository's
    `.git/info/exclude` (not tracked) and removed it at once; no tracked file was affected.
12. **`IMPLEMENTATION CHOICE`** — the breakages ran before the commit, on the working tree: after each, the module's
    SHA-256 and `git status --porcelain -- AdaptiveLevin LiouvilleGreen` are checked equal to their values before
    the first (the tree was not clean then, holding this commit's changes); after the commit `git status` is
    clean (Verification 2).

## Verification performed

1. **The seven roots at the commit, serially:**

| root | baseline (dispatch SHA) | at the commit |
|---|---|---|
| AdaptiveLevin | 116 OK | 129 OK (+13, the tests added) |
| ComputeTargets | 781 OK | 781 OK (the wall-clock flake did not fire) |
| CosmologyModels | 39 OK | 39 OK |
| LiouvilleGreen | 156 OK, skipped 1 | 156 OK, skipped 1 |
| ObjectFactories | 280 OK | 280 OK |
| RunDrivers | 127 OK | 127 OK |
| tools | 64 OK | 64 OK |

   Run serially on the working tree whose code is the commit's: the SHA-256 of `levin_quadrature.py`
   (`8956c7b6…`), every `AdaptiveLevin/tests/` module and `test_three_bessel.py` checked equal before and after.
   Afterwards two lines of `test_three_bessel.py`'s §10 docstring were reflowed to the module's width (docstring
   only: the AST check unchanged), and the LiouvilleGreen root was re-run (156 OK, skipped 1) and, after the
   second reflow, its module `test_three_bessel` (9 OK). Only the log and the board changed after that.

2. **The breakages** (`breakages_07b.py`, `out/breakages_summary.txt`), each applied as one text replacement, the
   tests run, the module restored (SHA-256 and `git status` checked after each):

| breakage | caught by (test failing) | named in §3 / §10 |
|---|---|---|
| (i) `_LEVIN_EDGE_RATE_RATIO_MAX` = inf | (b); (a) a1, a3, a4, a6, a8 | (a), (b) |
| (ii) `_LEVIN_LEAF_RATE_RANGE_MAX` = inf | (c); (a) a2, a3, a6; (d); (e2) (its range filter reads the constant); LiouvilleGreen's test, by the stall assertion (`1.45e-13 not <= 1.43e-14`) | (a), (c); §10.3 the LiouvilleGreen test |
| (iii) the rules swapped (`if not window_edge_ends`) | (a) a1–a6, a8; (b), (c), (d), (e1) | (a) |
| (iv) the edge's neighbour the grid's far end | (a) a1, a3, a4, a5, a6, a8 | (a) |
| (v) the rules on Clenshaw–Curtis leaves too | (e2) (value `-0x1.24a89a8b8f0ccp-1` against the parent's `…0cap-1`) | (e) |
| (vi) a refused leaf at the cap kept trusted | (d) (`converged` True) | (d) |

   After each: the module's SHA-256 back to `8956c7b6…` and `git status --porcelain -- AdaptiveLevin
   LiouvilleGreen` as before the first; at the end, restored.

3. **The comparison with (h):** `compare_h_07b.py` runs the commit, (h) (SHA-256 checked) and the parent on the 22
   calls of tests (b)–(e) and the three changed tests' cells: 22 of 22 identical to (h) in every result key but
   `elapsed` and the counts and in every region field by field; the rule acts on 6 (b, c, d, si_1000_exp twice,
   θ = 100e^x); the other 16 identical to the parent. The stop's problem at five rungs: commit and (h) equal in
   value, abserr and regions at every rung. The draws: 4,320 of 4,320 Levin calls identical to (h)'s run (value, abserr, every region field, every result key but `elapsed`, `location_skipped_phases` and the counts), and 80 of 80 units pass under (h).
4. **The LiouvilleGreen draws:** `lg_draws_07b.py` over seeds 0–19, the four default tests of `test_3bessel_analytic.py`, 80 units under each
   tree (`out/lg_draws_report.txt`):
   - test outcomes, parent → commit and parent → (h): pass → pass 80, newly failing 0;
   - 4,320 Levin calls paired, none unpaired. **The rules do not act in 509: all bit-identical to the parent.** They
     act in 3,811 (every one by the range rule; 34,018 refusals; 0 edge refusals, though 423 calls have a window):
     3,784 move, 27 do not; `converged` (parent, commit): (True, True) 3,784, (False, False) 27, none changes; the
     largest move 0.0998 of the parent's abserr; regions 76,692 → 147,029 (1.92×); wall summed over units 802.7 s →
     1,271.2 s;
   - the three-Bessel results (|value − analytic| ≤ abserr), by whether the rules act in any of their calls:
     acted 1,055 results, honest 619 → 621, dishonest and converged 436 → 434; untouched 25, honest 25 → 25;
   - the parent and (h) runs were made first; the commit's was re-run after a comment-only change to the module
     (the comment on the chase's call count), so that it records the committed module's SHA-256 (`8956c7b6…`); the
     first run's commit results were the same in every count (`out/lg_run_stdout_first.txt`, its stdout).
5. **The AST check of §10.3:** only the one test function differs (`out/ast_check_07b.txt`).
6. **black** on every file changed or added; `git diff --stat` lists only the paths of the dispatch and §10.

## Observations not acted on

1. **The range rule's cost on the tests' own problems:** si_1000_exp 3 → 16 regions; θ = 100e^x 3 → 16; the stop's
   LiouvilleGreen problem 58 → 154 regions at rtol 1e-8 (the answer is more accurate, at the phase floor). Already
   §9.3's "Cost"; not an issue.
2. **Marginal dishonest calls the rule does not touch, on synthetic no-window phases with large raw values.** In
   scans of θ = Ae^{κx} (θ up to 1e11) with an atol, converged calls 1.0–3.0 times their abserr turned up at the
   parent, and most had no leaf of node range above 10, so the commit left them as they were (some moved, 1.09 →
   2.04). With θ ≤ 3e3 none appeared, and with θ ≤ 1e6 some at 1.0–1.5 remained, untouched by the rule. The
   likely cause is the undeclared rounding of a large raw phase in the endpoint terms
   (`[00-consumer-anchoring-floor]`'s kind), not the leaf mechanism; it was not attributed. Not acted on.
3. **The LiouvilleGreen draws: the range rule acts in 3,811 of 4,320 calls and the edge rule in none,** though 423
   calls have a window. The three-Bessel results that are converged and dishonest against the test's oracles,
   436 at the parent, are 434 at the commit: the rule does not reach what makes them dishonest there (07a's draws
   recorded the same population). Not attributed.
4. **The edge rule cannot see a leaf ending at a stationary point just outside the span** (test (c)'s geometry): no
   window, so only the range rule protects it, which it does there (range 3,984).

## State handed on

- The code commit (not pushed). Part C (§5 R31, §6 the census) runs on the Mac Studio after it is pushed; §9.5's U8
  replays are the orchestrator's review.
- No issue moves in this commit (§4).
- The scratch worktrees are removed; nothing is running.

## Part C: the Mac Studio record (appended by the orchestrator, 2026-11-18)

Part C ran from 2026-11-16 to 2026-11-18 on the Mac Studio (`mps026298`), through an orchestrator session there, from
the worktree `SecondaryGWKit-lpf07b-0a103b5` at Part C's dispatch SHA `0a103b5a`, whose code is the code commit
`5e19c4b6`'s (the orchestrator note's §C.5). The Mac Studio orchestrator dispatched a rehearsal agent (R31), then a
census agent, and held them at the note's planned stops 1, 2 and 3. It checked each stop with readers of its own, then
reviewed Part C in the note's §C.2 checks, **all passing**. The records are in `docs/levin-phase-floor/07b/`:
- `rehearsal/` and `census/`: each agent's log, verification, issues, scripts and outputs. The two logs are appended
  below as written on the Mac Studio;
- `review/`: the Mac Studio orchestrator's stop records, scripts and hashes, and its review (`review/review.md`).

What stays on the Mac Studio is in `07b/KEPT_ON_MAC_STUDIO.md`. `VERIFICATION.md` §7b is the summary. Part A's log
(`07b/measurement/log.md`) follows the census agent's, in full, with its headings moved down one level to sit under
this log's own.

**The verdicts on the predictions above:**

| | Verdict | Numbers |
|---|---|---|
| §5, S0′ | **holds** | `1ec734d7..7b82a0cd` lists `CLAUDE.md` and `89c368f2`'s four files, each AST-equal without docstrings; R30 is the parent |
| §5, the fingerprint | **holds** | R31 `1b332ffa…`, its `classes` equal R30's |
| §5, the lock | **holds** | acquired, `set-pid` at the launch, released after the readers, each a shell command of its own |
| §5, R31 against R30 | **holds, and tabulated** | outside `QuadSourceIntegral` 360,462 rows identical; inside it 569 of 864 rows moved, the largest 0.278 of R30's abserr; `total_converged` 0 up, 1 down |
| §5, S3 | **holds** | 0 violations in 864 rows, scored exactly; the largest 0.128 |
| §5, S1, S2, S4 | **hold** | S1 12 of 12 honest, worst 0.067; S2 0.0202 and 0.111; S4 4 of 4 locate their root, guard nothing, honest at 4.4e-5 to 6.2e-4 |
| §5, the readers | **hold** | 6 of 6 exit 0, R31 unchanged; R30 and R31 kept |
| §6, Gate U0 | **holds** | the parent's code is 07a's; 100 of 100 K0 units (2,110 calls) equal 07a's census; the instrument changes nothing; the code is deterministic |
| §6, U1 | **holds** | 60,160 untouched calls (20.8 %) equal 07a's in value, abserr, `converged`, regions and maximum depth; 5,688 rows untouched in every call equal in `total` and `total_abserr` |
| §6, U2 | **holds** | 0 returned Levin leaves hold a sign change, 182,066 calls scanned |
| §6, U3 | **holds** | 0 violations; 784 skipped phases, all `zero_rate` |
| §6, U4 | **holds, as §9.5 predicted** | 190 referenced calls, none converged and dishonest; unconverged exactly v04, R2-28 and R2-28~o (R2-28 at 1.107); largest converged 0.805; 6 reported, not scored (honest at ≤ 0.035 either way) |
| §6, U5 | **holds** | 0, 0 and 0 on gated solves; 18,238 forced node-partition sums (descriptive; 07a 18,236) |
| §6, U6 | **holds, as predicted** | 24 of 24 honest against $S_{10}$, at $1.8\times10^{-5}$–0.83; the four dishonest at 07a at 0.049, 0.0013, 0.83 and 0.00014 |
| §6, U7 | **holds, as predicted** | $r_g \le 1$ on all 2,484 pairs, the largest 0.467; `3-14501` 0.00046, `3-35006` 0.00057 |
| §9.5, U8 | **holds** | 187 of 187 census calls equal (h)'s replay records |
| §9.4, the projection | **holds** | central 37,616 s, widest 65,989 s, within 48 h; measured 45,016 s |

**R31.** `levin-phase-floor-07b-r31-07b-20261116T191849`, about 13.6 min against R30's 9.7, store primary
`5981563b…`. `QuadSourceIntegral`'s stage took 1.52× R30's and its regions 1.33×. One record became unconverged:
`G+q+r` at $(10^5, 10^6, 10^6)$, $z = 144627.7$, 2 → 143 regions to depth 20 with 4 range refusals, $|\Delta v|$ 0.62
of R30's abserr. The replay (452 of 452 records bit for bit) recorded the refusals: the range rule 591 in 373 of 967
group calls and 6,733 in 2,667 of 3,328 analytic calls; the edge rule none.

**Which store understates** (the user's word at stop 1; descriptive). The seven group records whose values moved past
the sum of the two errors were referenced by `tools.filon_reference` (`d37e633b`), FCC r2, the rule fixed before any
reading, each reference's stated error at most $5.9\times10^{-4}$ of the smaller abserr. **R30 understates in all seven
(2.91–59.3 of its abserr) and R31 is honest in all seven (0.0014–0.298)**; every call converged. All seven are
`G−q±r` at rtol `1e-7` with no located point, R31's calls carrying 1–3 range refusals.

**The census.** `levin-phase-floor-07b-census-20261117T010200`, checkpoint `396784bc…`, 18,560 of 18,560 units in
45,016 s (07a: 23,284 s, 1.93×), none over budget, stderr empty. It is 08's "before".
- **Where the fix acts:** 229,322 of 289,482 calls (79.2 %). The range rule refused 782,001 times (675,045 analytic,
  106,956 group), only at depths 0–3. The edge rule refused 868 times in 674 `G+q+r` calls with a root window, 636 of
  them at depth 0; it refused none of 2,802 evaluations on analytic windows and never acted on a minima window.
- **Convergence against 07a:** 198 group calls newly unconverged (97 at `1e-7`, 101 at `1e-8`; 4 with an edge refusal),
  5 newly converged, no analytic call changed. In 192 of the 198 every depth-20 leaf is a Clenshaw–Curtis leaf below a
  leaf the rate rules refused (the census agent's count, not re-checked on the Mac Studio). Rows: 139 lost
  `total_converged`, 4 gained it; rows at depth 20 182 and 185 (07a 77 and 83).
- **abserr against 07a's:** untouched calls equal; acted analytic calls' median ratio 0.0026, acted group calls' 0.48;
  738 analytic calls of one region at 07a grew more than 100×, up to $4.6\times10^5$.
- **Values:** 2,672 group calls moved by more than 07a's abserr; 326 by more than the sum of both abserrs, 324 of them
  converged under both codes (the census agent's 605 count moves past each abserr separately; the review's correction,
  recorded, not repaired).
- **Cost:** regions 2.08×, summed unit wall 2.09× and 1.92×.

**The laptop's re-checks** (2026-11-18), from the copies, against committed records the agents did not write
(`VERIFICATION.md` §7b.5): the copy-back (763 records, 20 run files) by SHA-256; U4 on v08, R2-03, R2-22 and R2-28
against Part A's committed (h) replays and readings and 07a's committed W4; U6 on `2-14345` #3 and `2-34473` #3 against
06b's committed $S_{10}$; an untouched rehearsal row equal in hex in R30, R31 and 07a's committed table; the most-moved
row's ratios recomputed exactly. Every figure equals the agents'.

**The 324 calls, referenced where we have references** (read on the Mac Studio, read-only, from the census's kept
comparison by call, at the user's question *"do we not have references for these?"*). 18 of the 324 are referenced:
Part A's R1 (x015, x019, x033, x044, x165, x185, x207, most at both rtols), V4's v08, v12, v17 and v22, and 06b's
`0-12977` #5. **On all 18, 07b is honest (at most 0.054) and 07a is not (1.37–10.8).** The extreme move against 07b's
abserr, `0-14199` #1 at 75×, is R1-x015: 07b at 0.0007 of its abserr, 07a at 2.47. With R31's seven, 25 of 25
referenced cases put the failing declaration at 07a. The other 306 are unreferenced, and the 25 were chosen for having moved or for 07a's
failures, so they cannot exclude 07b understating among the 306.

**Deviations in Part C.** The rehearsal agent logged 26: 17 `IMPLEMENTATION CHOICE`, 5 `UNINTENDED DRIFT`, 3 both
(a choice whose first run failed harmlessly) and 1 `STRUCTURALLY REQUIRED` (the classifier run on the raw and the
normalised JSON). The census agent logged 23: 18 `IMPLEMENTATION CHOICE`, 3 `STRUCTURALLY REQUIRED` (the harness's
changes for the tree, the parent and the two counts; the parent run as a variant in h; the registry having no job-name
field) and 2 `UNINTENDED DRIFT` (failed attempts re-run, and cosmetic). The Mac Studio orchestrator accepted each. The
drift is cosmetic, or a failed attempt re-run with nothing used from it.

**Two slips in the agents' records, recorded, not repaired:** the census agent's 605 above; and the rehearsal log's
Result line, which gives R30's understatement on the seven records as "3.87 to 59.3", where its own table,
`out/refs/understates.txt` and the review give 2.91–59.3.

**The user's rulings** (relayed by the laptop orchestrator):
- dispatch: *"Yes, dispatch Part C"*;
- stop 1: *"Yes, continue, and settle which store understates"*;
- stop 2: *"Yes, launch the census"*;
- stop 3: *"Yes, score U1–U7"*;
- after the review: *"Yes, start §C.4"*;
- on the orchestrator's proposals at §C.4: *"Yes, those rulings stand. Leave item 5 unassigned for now."* So:
  1. `[07a-levin-leaf-beside-a-window-understates-its-error]` closes;
  2. `[06b-moved-calls-declare-less-than-their-error]` closes;
  3. `[07-iroff1-window-stop-can-leave-a-call-dishonest]` closes;
  4. `[07a-trust-check-auxiliary-route-moves-the-declared-error]` closes;
  5. `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]` is narrowed with the census's evidence.
- On items 6–8, after the user's questions (*"6 - do we not have references for these?"*, *"7 observation 3 - for
  now, we have no idea what causes this?"*, *"8 observation 2 - do we achieve phases this large in production?"*) and
  the orchestrator's revised proposals: *"Yes, items 6–8 as revised; write a 07c first to cover item 5."* So:
  6. the 324 calls are recorded here, not opened; the next census takes a reference sample from the 306 unreferenced;
  7. Part B's Observation 3 (434 of 1,080 three-Bessel draw results converged and dishonest against their closed forms,
     the worst by $4.89\times10^6$) narrows `[09-abserr-does-not-bound-phase-spline-floor]` on the levin-refactor board;
     07a's log attributes it to the Bessel phase's fit error, not measured;
  8. Part B's Observation 2 is opened as `[07b-large-raw-phase-calls-can-be-marginally-dishonest]`: 07a's census reaches
     raw phases of $4.1\times10^{12}$ (group) and $8.9\times10^{12}$ (analytic), above the synthetic scans' $10^{11}$.
- On the $6\pi$ gate, which the chase runs through (3,961 of (f)'s 4,374 depth-20 leaves are its Clenshaw–Curtis leaves,
  whose estimate is CC₁₆'s error): *"Go with (a); make the records commit, then write 07c"*. So 07c measures a sharper
  Clenshaw–Curtis estimate (`[06d-cc-nested-estimate-measures-the-coarse-rule]`) beside the stop-based candidates for
  the chase, and the threshold (`[07-c2-gate-threshold-was-never-measured]`) goes to a 07d after 07c and before 08.

  R29, R30 and R31 are kept, with 07a's and 07b's census run directories and A3 v2's scratch copy.

**Observations not acted on, and not opened as issues** (the agents' `issues.md`, and Part A's):
- **Part A's items 1, 2, 9 and 10** (the common error away from windows, class (c) at 07a, no leaf-local test, the
  noise clause): the range rule now catches every call they were measured on; item 1 found every dishonest leaf
  spanning a rate ratio of at least 49, above the rule's 10. Recorded.
- **Part A's items 3–8 and 12–14** (R2's rate, R3-aux's `3-13508`, the pairs as a test, the supplied $\theta'$ error
  constant, the split-FCC projection, `2-34473`'s deep leaves, the adaptive window, R2-28, v19): recorded in Part A's
  log; R2-28 and the item-11 chase are the chase issue's.
- **The rehearsal agent's items:** `phase_limited` flips False → True on 5,336 analytic and 348 group records, never
  back; the edge rule untested by R31 (no window-edge carrier at its scope); the cost (1.52×); the 11 MB
  `commands.log`; the readers' random subsamples.
- **The census agent's items:** the edge rule's reach (root windows only); the cost (1.93×, against §9.4's estimate of
  about 5×); the analytic calls at `1e-8`, where §9.3 listed only `1e-7`; the six newly unconverged calls with no
  depth-20 leaf (the chase issue's narrowing); 07b's census and k0 run directories kept on the Mac Studio.
- **Part B's Observations 1 and 4:** the range rule's cost on the tests' own problems; the edge rule blind to a leaf
  ending at a stationary point just outside the span, which the range rule protects there.

## Part C §5: rehearsal R31 on the Mac Studio

**Prompt:** [`07b-levin-leaf-beside-a-window.md`](../07b-levin-leaf-beside-a-window.md) §5 ("as 07a §5"), under Part C's
dispatch (`0a103b5a`). The user's word at stop 1: *"Yes, continue, and settle which store understates"*.

**What was measured:** the code commit `5e19c4b6`, variant (h). After the tree's trust test passes a non-direct Levin
leaf, an edge rule (on a leaf touching a window) or a range rule (on a leaf touching none) on |θ′| can refuse it. Its
parent is `7b82a0cd`. The parent rehearsal is R30, built at `4bf50d34`, whose code is `1ec734d7`'s.

**Host and trees:** mps026298.
- Every build, replay, score and reader ran in the worktree `SecondaryGWKit-lpf07b-0a103b5`, detached at `0a103b5a`.
  - Its key files: `levin_quadrature.py` `8956c7b6`; `QuadSourceIntegral.py` `33cb161e`; the launcher `b0135218`;
    `main.py` `a5d47f5f`; `store_value_diff.py` `7997af85`. The last four are R30's tree's.
- Nothing was written in the worktree; `status --short --ignored` was empty at every check.
- The main checkout stayed at `28ebaf07`, clean.

**Model:** Claude Opus 5.5, the rehearsal subagent, in four turns:
1. the gate and R31's launch;
2. the end check, the comparisons, and stop 1;
3. S3, the replay, S1, S2, S4, the readers, the release, and the references' plan;
4. the references' launch, the scoring, and these records.

**Date:** 2026-11-16, 19:16–20:35 UTC. **Commit:** none; the laptop commits these records.

**Result:** COMPLETE.
- **R31 is `done` with `1b332ffa…`.** S0′ holds.
- **Outside `QuadSourceIntegral` every value table equals R30's.**
- Inside it, 569 of 864 rows moved. `total_converged` went down in one row and up in none.
- **S3, S1, S2 and S4 hold.** The readers exit 0 with the store unchanged. The lock is released.
- **Which store understates** (descriptive; the user's word at stop 1): in all 7 group records whose |v31 − v30| exceeds
  e30 + e31, **R30 understates (3.87 to 59.3 of its abserr) and R31 is honest (0.0014 to 0.298)**. Each reference's stated
  error is at most 5.9e-4 of min(e30, e31).

### What shipped

No code. The records under `var/levin-phase-floor-07b/rehearsal/` (committed by the laptop as
`docs/levin-phase-floor/07b/rehearsal/`): `log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`, and:
- **`scripts/`:**
  - adapted from 07a's rehearsal (each diff in `out/`):
    - `rec.sh` (`9a80a313`);
    - `normalise_jobname_labels.py` (`bdc2d44d`);
    - `tabulate_qsi_r30_r31.py` (`a7dc4941`) and `qsi_summary_r30_r31.py` (`50d33417`);
    - `s3_score.py` (`f6396428`);
    - `s1_score_r31.py` (`305d44d0`), `s2_score_r31.py` (`5c499eb2`) and `block_score_r31.py` (`fcb8b436`);
    - `run_block_reference_r31.sh` (`687b4d81`);
  - unchanged copies:
    - `classify_non_qsi.py` (`26690e61`, 05's);
    - `block_reference_r30.py` (`efac125e`);
    - `run_readers_worktree.sh` (`ec96cf09`);
    - `run_06j.py` (`6227aea4`, Part A's plan driver);
    - `refs/variant_loader.py` (`1cb49a13`, Part A's);
  - new:
    - `s0prime.py` (`f7e33882`);
    - `protected_listing.sh` (`8875e3d0`);
    - `launch_r31.py` (`98890b5e`; as it ran the replay: `state/launch_r31_as_replay.py`);
    - `gen/make_replay07b_r31.py` (`01ad6081`) with `gen/replay07b_r31_extra.py.txt` (`afa8e225`);
    - `refs/refcap_replay.py` (`4f6af0d8`), `refs/refs_projection_r31.py` (`5dc49f8a`) and `refs/understates.py`
      (`14e238c5`);
  - **`capture/`:** R30's on-disk harness (hashes in `state/capture_harness_r30_sha256.txt`).
    - `prov.py` (`1c976bf8`) and `worker.py` (`2a253652`) are changed where marked "prompt 07b r31".
    - `replay07b_r31.py` (`ac7319ca`) is generated from R30's `replay07a_r30.py`.
- **`out/`:**
  - `commands.log`;
  - the gate, launch, end, stop-1, readers, rehearsal-end and final checks;
  - `r31_status.json` and `r31_fingerprint_own.txt`;
  - the diff JSON (kept, over 2 MB), its normalised copy, and the classifier outputs;
  - the QSI tabulation, its CSVs and its summary;
  - `s0prime.*` and `s1_scores.*`–`s3_scores.*`;
  - `block_reference/`, `block_scores.*` and `blocks.json`;
  - the replay's outputs: `cmd/r31-replay/`, `replay_calls.csv`, `replay_calls.checks.json` and `refcap.json`;
  - `readers.txt`;
  - `refs/`:
    - the capture replays at both codes;
    - the projection and the tolerances;
    - `readings/` and `understates.{json,txt}`;
  - `cmd/r31-refs/`.
- **`state/`:** `lock.log`, `deviations_draft.md`, the launch lines and markers, every store-hash file, the protected
  listings, the replay's rows and specs, the plans, `refs_plan_r31.md` with its mtime record, and `variants/`.
- **Not for copying back:** see `KEPT_ON_MAC_STUDIO.txt` (the diff JSONs, `extract/`, `state/capture/`, `state/inputs/`).
- **The store is kept:** `var/datastores/levin-phase-floor-07b-r31*`.

### Predictions and verdicts

| Rule (prompt 07b §5, as 07a §5) | Scored on | Verdict |
|---|---|---|
| 1. S0′: `git diff --stat 1ec734d7 7b82a0cd -- . ':!docs' ':!prompts'` lists only `CLAUDE.md` and 89c368f2's four files, each AST-equal with docstrings removed | exactly those five; the four AST-equal (Part A's `strip_docstrings`) | **holds** |
| R31 with R30's command line (slug `r31-07b`, `--prompt 07b`, `--job-name levin-phase-floor-07b`, `--cpus 6`) gives `1b332ffa…` | `done`, exit 0, `1b332ffa…` (`status.json`, and again by `runregistrykit store fingerprint`, read only); `classes` equal R30's; 13.6 min | **holds** |
| 2. The lock: `acquire`, `set-pid` at the launch, `release` after the readers, each a shell command of its own | acquired 19:18:38Z; `set-pid` 10624; released 20:04:53Z after `ray-processes` listed nothing; ten commands, each its own shell command, `>>` | **holds** |
| 3. Outside `QuadSourceIntegral`: every value table identical; every other difference of the four volatile kinds; no row in one store only | 360,462 value-table rows identical; 0 unclassified on the normalised copy (timing, `label`, `parents_digest` 291, `rho_reuse` 23) | **holds** |
| 3. Inside it: every moved row tabulated, with the rows whose `total_converged` changes in each direction | 569 rows (`out/qsi_rows_r30_r31.csv`); 9,206 records; `total_converged` 0 up, 1 down | tabulated |
| 4. S3: \|t31 − t30\| ≤ e30 + e31 on every row with finite `total_abserr` in both, or one §9 predicts to move further | 864 scored exactly (Fractions); 0 violations; largest 0.128 | **holds** |
| 5. S1: V1's twelve locate their point at z ≈ 10211 and are honest against 06g's references | 12 of 12; largest 0.0670 (w06); 6 equal R30, 6 moved and more accurate | **holds** |
| 5. S2: P1's two rows locate within one spacing and are honest against `p1_references` | both; 0.0202 (= R30) and 0.111 (R30 0.263) | **holds** |
| 5. S4: R29's four blocks each locate their root, guard no leaf, and are honest against R29's block references | 4 of 4: root bit-equal, 0 guarded, 4.37e-5 to 6.17e-4; each equal to R30 | **holds** |
| 6. The readers on R31 exit 0, store byte-identical; lock released, `ray-processes` empty; R30 and R31 kept | 6 of 6; identical; free; nothing listed; R30 equal to its KEPT hashes | **holds** |

**§9.5's S3 addition** (a row beyond S3's bound reported with its refusals): no row is beyond it.
- The replay records the refusals anyway.
- Over the 4,295 replayed calls: **range rule 591 refusals in 373 of 967 group calls, and 6,733 in 2,667 of 3,328 analytic
  calls; the edge rule refuses nothing.**

**The inside-QSI tabulation** (`out/qsi_tabulation_r30_r31.txt`, `out/qsi_summary_r30_r31.txt`):
- **Rows:** 295 unmoved, 569 moved.
  - `total` moved in 379, `total_abserr` in 386, `analytic_rad` in 548.
  - `phase_limited` flipped False → True in 5,336 analytic-block and 348 group records, never the other way.
- **Largest changes:**
  - \|Δtotal\| is 0.278 of R30's abserr and 0.238 of R31's, at (1e5, 3.162e5, 3.162e5), z 12004.58 (regions 80 → 243).
  - The median over the 379 is 5.8e-3.
  - \|Δanalytic_rad\| is at most 3.1e-3 of R30's `analytic.abserr` and 8.0e-3 of R31's.
- **Regions:** summed `WKB_Levin_num_regions` went 114,245 → 152,117 (1.33×).
  - 386 rows rose and none fell; the median per-row ratio where changed is 1.33, the largest 14.6.
  - Maximum depth changed in 93 rows, and the rows at depth 20 went from 5 to 7.
  - QuadSourceIntegral's stage took 11 m 28.6 s against 7 m 32.7 s (1.52×).
- **Convergence:** `total_converged` went 862 → 861.
  - The one row: (1e5, 1e6, 1e6), z 144627.7067, record `subintervals[1].groups[0]` G+q+r. Its regions went 2 → 143 and
    its depth to 20, with 4 range refusals.
  - That record's \|Δv\| is 0.619 of e30 and 0.897 of e31, and the row's \|Δtotal\| is 3.2e-4 of its abserr.
  - No analytic block's `converged` changed.
- **Location:** unchanged (group 44 / 562 / 0; analytic 400 / 2,515 / 0; 202 rows located).
- **Group records:** 1,053 values moved.
  - \|Δv\| > e30 in 144 of them (largest 59.6), > e31 in 27 (largest 2.37), and > e30 + e31 in 7 (exactly, in Fractions;
    largest 1.53).

**Which store understates** (the user's word at stop 1; descriptive, not scored against any rule;
`out/refs/understates.txt`).
- **The rule** was fixed in `state/refs_plan_r31.md` before any reading (mtime 20:09:13.98Z, `2c2514b0`; the run began
  20:11:20Z):
  - reference = the tool's FCC r2; stated error s = the four readings' scatter;
  - no verdict if s > min(e)/100;
  - a store understates if \|v − ref\| − e > s, and is honest if e − \|v − ref\| > s.
- **The method:** `tools/filon_reference.py` (`d37e633b`, 07's V4 and Part A's tool), Route B, QAWO and FCC at
  resolutions 1 and 2, independent of the Levin driver.
- **The captures:** each captured from R31 by the replay harness. Each reproduces R31 bit for bit at the tree, and R30 bit
  for bit at 07a's module (`git show 1ec734d7`, installed by Part A's `variant_loader`), so the one integrand is both
  stores'.
- **The results:**

  | Record (k, q, r; z) | label | R30: \|v−ref\|/e (regions) | R31: \|v−ref\|/e (regions, range refusals) | s / min(e) | answer |
  |---|---|---|---|---|---|
  | (1e6, 1e5, 1e6; 329.64) | G−q+r | **59.29** (5) | 0.0055 (112, 2) | 5.9e-4 | R30 understates |
  | (3.162e5, 1e5, 3.162e5; 996.42) | G−q−r | **35.60** (3) | 0.0070 (38, 3) | 7.7e-5 | R30 understates |
  | (3.162e5, 1e5, 3.162e5; 6.945e6) | G−q−r | **9.20** (8) | 0.298 (68, 1) | 4.4e-5 | R30 understates |
  | (1e6, 3.162e5, 1e6; 1.918e8) | G−q−r | **5.62** (1) | 0.0150 (45, 1) | 3.5e-5 | R30 understates |
  | (1e6, 1e6, 1e6; 36286.8) | G−q−r | **4.36** (6) | 0.0050 (66, 1) | 2.2e-5 | R30 understates |
  | (1e6, 1e6, 1e6; 331553) | G−q−r | **3.87** (1) | 0.0014 (58, 3) | 2.5e-5 | R30 understates |
  | (1e6, 1e5, 1e6; 36286.8) | G−q+r | **2.91** (58) | 0.0045 (82, 3) | 3.4e-5 | R30 understates |

- Every R30 record here was **converged**, as was every R31 record. Every one of R31's calls had range-rule refusals
  (1–3) and no edge refusal.
- Every reading integrated every segment in t: no direct segment, and every segment monotone.

### Deviations from the prompt

1. **`rec.sh` is R30's** (`065afcc6`) with its record path changed by sed. IMPLEMENTATION CHOICE.
2. **S0′ is scored by a new script, `s0prime.py`,** with Part A's `strip_docstrings` copied verbatim. The raw
   `git diff --stat` is also in `out/00-gate.txt`. IMPLEMENTATION CHOICE.
3. **The lock commands ran from the worktree with `PYTHONPATH` set** (the dispatch's rule), where R30's ran from the main
   checkout. Each entry has a header line written by `echo`, a builtin. IMPLEMENTATION CHOICE.
4. **A listing of every protected file** (mtime, size, path) was taken before, at stop 1, after the rehearsal and at the
   end. IMPLEMENTATION CHOICE.
5. **A first protected-listing one-liner failed:** zsh does not word-split `$P`. It was read-only, and its output was
   overwritten. UNINTENDED DRIFT.
6. **The fingerprint was also re-derived** by `runregistrykit store fingerprint` without `--write`. Read only; the store
   was re-hashed equal. IMPLEMENTATION CHOICE.
7. **`normalise_jobname_labels.py` is 07a's**, with its prefixes (`07a-` → `07b-`) and docstring changed.
   IMPLEMENTATION CHOICE.
8. **The classifier was run on the raw and the normalised JSON.** The raw gives 5,148 UNCLASSIFIED, every one a `label`
   whose job-name prefix differs; the verdict is the normalised copy's. STRUCTURALLY REQUIRED.
9. **The classifier prints no provenance header,** so the shell wrote one. IMPLEMENTATION CHOICE.
10. **Stdlib-only scripts ran with `-B` and `PYTHONPATH`, cwd the worktree** (the dispatch's rule), not with `-I`. None
    imports tree code. IMPLEMENTATION CHOICE.
11. **`tabulate_qsi_r30_r31.py` is 07a's** with the store tags shifted, TREES, RUNS, a comment and the docstring changed;
    the verbatim reader is untouched. IMPLEMENTATION CHOICE. Its first generation aborted on its own assertion before
    writing. UNINTENDED DRIFT, harmless.
12. **`qsi_summary_r30_r31.py` is 07a's**, tags shifted, plus a new `additions_07b`. IMPLEMENTATION CHOICE.
    - One printed label was missed by the regex and corrected, and the function was extended once. It was re-run each
      time, and only the last output is kept. UNINTENDED DRIFT.
    - Its "record-level S3 form" list uses float arithmetic. The exact list (Fractions) is `replay07b_r31.py rows`'s,
      which found the same 7.
13. **One read-only look at stop 1 printed output that was not saved.** Its fields are in the CSV and the summary.
    UNINTENDED DRIFT.
14. **S3 is scored by 07a's `s3_score.py`**, with the tags shifted. IMPLEMENTATION CHOICE.
15. **The capture harness is R30's on-disk copy.** `prov.py` points at the tree and R31; `worker.py` adds `refusals_07b`;
    `route_07a` is kept, descriptive only. IMPLEMENTATION CHOICE.
16. **`replay07b_r31.py` is generated from R30's driver** (`gen/make_replay07b_r31.py`).
    - The 49 abserr-only rows are replaced by the 8 records of interest.
    - It reports the refusals, the blockcheck compares with R30's captures, and two subcommands capture the records of
      interest.
    - IMPLEMENTATION CHOICE. Its first generation aborted before writing. UNINTENDED DRIFT, harmless.
17. **The replay ran as a registered run** (slug `r31-replay`), by Part A's `run_06j.py` and a new launcher, where R30's
    ran as an unregistered shell pipeline. `run_06j.py` appends each command's output to `out/commands.log`, which is why
    that file is over 2 MB. IMPLEMENTATION CHOICE.
18. **Block pass 1 (the pilot) ran in session**, as R30's did. IMPLEMENTATION CHOICE.
19. **S1, S2 and S4 are R30's scripts, adapted.**
    - S1's keys are renamed to R31's.
    - Each adds the refusals and R30's numbers.
    - S4's integrand is still compared with R29's capture, on which the references were taken.
    - IMPLEMENTATION CHOICE.
20. **R30's `capture_compare.py` was not re-run.** The integrand's equality at 1,001 points carries S4, as in R30.
    IMPLEMENTATION CHOICE.
21. **A read-only `tail -3` loop failed** and was repeated. UNINTENDED DRIFT.
22. **A per-call projection total was printed and not saved.** It is re-derivable from `refs_projection_r31.json`.
    UNINTENDED DRIFT.
23. **The readers ran in session** (about 2 minutes), not registered, as R30's did. IMPLEMENTATION CHOICE.
24. **The descriptive addition's new scripts:**
    - the captures re-run at both codes (`refcap_replay.py`);
    - the projection by 07a's model with its γD term (`refs_projection_r31.py`);
    - the scoring (`understates.py`).
    - IMPLEMENTATION CHOICE.
25. **The launcher's first `r31-refs` edit aborted before writing**; `cmp` shows the file unchanged. UNINTENDED DRIFT,
    harmless.
26. **The references are the tool's own command line,** four readings per process, where Part A read each by
    `one_reading.py` for its monitor. They are the same calls of `reading()`. Part A's split-edge FCC was not needed:
    every segment is monotone and none steep. IMPLEMENTATION CHOICE.

### Verification performed

- **Gate** (`out/00-gate.txt`, `out/s0prime.*`, `out/01-registry-list.txt`, `out/02-stores-before.txt`, `state/lock.log`):
  - the worktree at `0a103b5a`, clean with `--ignored`; the main checkout at `28ebaf07`, clean;
  - `5e19c4b6^` = `7b82a0cd`; `5e19c4b6..0a103b5a` outside docs and prompts: empty;
  - 81 runs, all finished; the lock free; no Ray process; `auth_token` mode 600;
  - R29 and R30 equal to their KEPT hashes.
- **R31** (`out/r31_launch_check.txt`, `out/r31_end_check.txt`, `out/r31_fingerprint_own.txt`):
  - launched 19:18:48Z (sh 10620, driver 10624);
  - `done` at 19:32:27Z; no Traceback, WARNING or ERROR.
- **Stores:**
  - R29, R30 and A3 v2 byte-identical throughout (`state/stores_sha256_final.txt`);
  - R31 unchanged after its build, across the fingerprint, the comparisons, the replay, the readers and the references.
- **The replay** (`out/cmd/r31-replay/`):
  - 452 of 452 records reproduce R31 bit for bit, with Ray uninitialised in all 208 workers;
  - the four blocks reproduce in passes 1 and 2;
  - the 8 captures reproduce R31.
- **The captures for the references** (`out/refs/refcap_replay_{h,a07}.json`): all 8 reproduce R31 at the tree and R30 at
  07a's module, bit for bit.
- **The lock:** ten commands, each exit 0, ending free with nothing listed.
- **Final** (`out/final_check.txt`):
  - 84 runs, all finished; `~/.raypoolkit` empty; no Ray process;
  - both trees clean and unmoved, nothing newer in the worktree;
  - the protected listing differs only by the orchestrator's files under `var/levin-phase-floor-07b/review/stop1/`;
  - no `__pycache__` in the records.

### Observations not acted on

1. **R30 had converged and dishonest group calls in the rehearsal store** (the 7 above; `issues.md` item 1). (h) fixes
   each by the range rule alone. The edge rule refused nothing in any replayed call: no window-edge carrier occurs in the
   rehearsal's calls.
2. **The one new unconverged call** (G+q+r, (1e5, 1e6, 1e6), z 144627.7) looks like
   `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`. It goes 2 → 143 regions to depth 20, with \|Δv\| at
   0.62 e30 (`issues.md` item 2). Not referenced; the user asked about the seven.
3. **`phase_limited` flips False → True on 5,684 records** and never back (`issues.md` item 3).
4. **The readers subsample k and z at random**, so their file counts differ from R30's, as 07a recorded.
5. **`out/commands.log` holds the run driver's appended command outputs** (deviation 17).

### State handed on

- **R31:** `levin-phase-floor-07b-r31-07b-20261116T191849`, `done`, `1b332ffa…`. Its store is kept.
- **The replay run** `levin-phase-floor-07b-r31-replay-20261116T195034` and **the references run**
  `levin-phase-floor-07b-r31-refs-20261116T201120`: both `done`, exit 0.
- **R30 and R29** are unchanged and kept.
- **The lock is released.** Nothing of this agent runs. Both trees are clean and unmoved.
- §6, the census, was not touched (the census agent was dispatched after the references ended).

## Part C §6: the census at 07b's code (Mac Studio), the census agent

**Subject:** prompt 07b §6, as amended by §9 (§9.4, §9.5, §9.7). The census was re-run at the code commit `5e19c4b6` (variant
(h), the rate rules), with 07a's census as the "before". It ran on the Mac Studio (`mps026298`), 2026-11-16 to 2026-11-18,
by Part C's census agent. This log covers Gate U0, the predictions, the projection, the census, stop 3's two checks
(U1's untouched calls and U8), U1–U7, the descriptive items and the records. Ray-free, no lock, no store built.
- **The tree h:** the worktree `SecondaryGWKit-lpf07b-0a103b5`, detached at Part C's dispatch SHA `0a103b5a`.
  - Its code is `5e19c4b6`'s: `AdaptiveLevin/levin_quadrature.py` `8956c7b6…`.
  - It stayed clean with `--ignored` throughout, with no `__pycache__`. The main checkout stayed at `28ebaf07`, clean.
- **The parent:** there is no parent worktree. U0's K0 run ran in h with the parent's module installed in place of h's by
  `variant_loader.py`: `state/variants/levin_quadrature_parent.py`, from `git show 7b82a0cd`, `4ea1948b…`, blob `0701d2a6`,
  the same as 07a's code commit `1ec734d7` and 07a's census tree `4bf50d34`.
- **The "before":** 07a's census, `levin-phase-floor-07a-census-20261112T132744`, checkpoint `461df2ec…`.
- **The store:** 06's certified scratch copy of A3 v2, read only.

**Commit:** none here. The orchestrator commits these records from the laptop as `docs/levin-phase-floor/07b/census/`.

**The user's rulings, relayed by the orchestrator:**
- At stop 2: *"Yes, launch the census"*.
- At stop 3: *"Yes, score U1–U7"*.

**Result:**
- **Gate U0 holds**, all four clauses.
- **The projection** was 37,616 s central and 65,989 s at its widest margin, within §9.4's 48 hours.
- **The census** is `levin-phase-floor-07b-census-20261117T010200`, `done`:
  - 18,560 of 18,560 units, 0 in error, 0 over budget;
  - wall 45,016 s (12.5 h; 07a: 23,284 s), checkpoint `396784bc…`.
- **Stop 3's two checks pass:**
  - every one of the 60,160 untouched calls equals 07a's in U1's five outputs;
  - all 187 U8 calls equal (h)'s replay records.
- **U1–U8 all hold.** U4, U6 and U7 hold as §9.5 predicted.
- **The store, 07a's census and Part A's records are unchanged**, before, after the census and at the end.

### What shipped

No code. The records are in `var/levin-phase-floor-07b/census/`.
- **The records:**
  - `log.md` (this file), `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`;
  - `state/predictions.md`: SHA-256 `0c6055ac…`, mtime 2026-11-16T21:03:31.278528Z, read-only. It was written after U0 and before
    the launch. Its prompt blocks are copied by text and checked by substring.
- **`scripts/`:** 07a's census harness, copied from h's `docs/levin-phase-floor/07a/census/scripts/`. It is byte-identical to
  07a's Mac Studio copy (`out/cmd/00`). Changes are marked "prompt 07b". The diffs are `out/scripts_diff_before_k0.txt`,
  `_at_stop2.txt`, `_before_census.txt` and `_at_end.txt`.
  - **Changed:**
    - `prov.py`: the tree h, the parent's module and 07a's census;
    - `worker.py`: the two counts on every call; the 07b tree evidence; the rate rules' instrument (`rate_rule_07b`) and
      its switch;
    - `run_k0.py`: the parent variant 07b's; `--no-instrument` turns off both instruments;
    - `run_census.py`: the backstop stops at a projected 48 h, not a day.
  - **New:**
    - `ast_check_07b.py`, `u0a_imports_07b.py`, `index_07a_07b.py`, `compare_07b.py`, `select_u0_units_07b.py`;
    - `launch_07b.py`, `launch_census_07b.py`, `census_projection_07b.py`, `write_predictions_07b.py`, `hash_runs_07b.py`;
    - `score_u8_07b.py`, `stop3_cost_07b.py`, `score_v4_07b.py`, `score_refs_07b.py`, `xcheck_07b.py`, `score_07b.py`;
    - `summarise_07b.py`, `kept_list_07b.py`.
- **`out/`:** every command and its output (`out/cmd/`, `out/commands.log`), with these results:
  - U0: `out/u0/u0a_ast.txt`, `out/u0/u0a_imports.txt`, `out/k0/compare_u0b.json`, `out/u0c3/compare_instr.json`,
    `out/u0c4/compare_det.json`, and `out/k0/compare_k0_h_calls.json` (descriptive);
  - the projection: `out/census_projection.json`;
  - stop 3: `out/stop3/compare_census_calls.json` (and `.calls.jsonl`), `out/stop3/u8.json`, `out/stop3/cost.json`;
  - the scores: `out/scores/summary.json`, `u4_v4_scores.json`, `u4_v4_report.md`, `u4_u6_scores.json`, and
    `out/xcheck/u7_vs_07a_pop07/xcheck.json`;
  - before and after: `out/runs_hashed_{before,after,end}.txt`, `out/cmd/04b`, `41`–`43b`, `60`–`64`.
- **Two registered runs, both `done`:**
  - `levin-phase-floor-07b-k0-20261116T204402`: 11 of 11 commands, checkpoint `3f2f055f…`;
  - `levin-phase-floor-07b-census-20261117T010200`: 18,560 of 18,560 units, exit 0, checkpoint `396784bc…`.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory in full (11 GB). It is **prompt 08's "before"**;
  - the k0 run directory;
  - every pickle and every file over 2 MB under `census/`.

### Predictions and verdicts

`state/predictions.md` holds three parts of the prompt, each copied verbatim **by text** and checked as a substring of the
prompt:
- §6's U1–U7 and descriptive items (prompt lines 607–622);
- §9.5's additions, U8 among them (725–748);
- §9.7's stops (765–769).

Part 4 is this agent's reading of each rule, fixed before the run. Verdicts are by that file, unchanged.

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **U0 (1)** the parent's code is 07a's (S0′) | **holds** | `git diff --stat 1ec734d7 7b82a0cd` lists `CLAUDE.md` and 89c368f2's four files, each AST-equal without docstrings. The K0 workers at the parent loaded 101 files of h. Of 07a's census tree → h diff, only the two LiouvilleGreen modules were loaded, both AST-equal. h's own `levin_quadrature.py` was not loaded. The variant was the parent's `4ea1948b` (`out/u0/`) |
| **U0 (2)** the harness reproduces 07a's census at the parent | **holds** | 100 of 100 K0 units are bit-identical to 07a's census: 2,110 calls, 32,850 returned regions. Only 07b's four new keys were set aside, 0 of them in 07a's records (`out/k0/compare_u0b.json`) |
| **U0 (3)** the instrument changes nothing | **holds** | `h-1e-8-3-14501` (the fix acts: edge 2 and range 3 on call #1, range 2–7 on the others) and `h-1e-7-1-20582` (11 calls, all counts 0; the one fully untouched K0 unit) are IDENTICAL with the instruments on and off. The instrument's refusals equal the code's counts on every call (`out/u0c3/compare_instr.json`) |
| **U0 (4)** the new code is deterministic | **holds** | `h-1e-8-3-14501`, run twice in fresh workers, is identical on every key (`out/u0c4/compare_det.json`) |
| §9.4 the projection within 48 h | **holds** | Central 37,616 s (schedule model, ρ by rtol and the 720). Margins up to 65,989 s. No unit was projected over budget (`out/census_projection.json`). Measured: 45,016 s |
| **U1** nothing else moves | **holds** | 60,160 untouched calls (sum of counts 0): 29,758 at 1e-7 and 30,402 at 1e-8. All equal 07a's in value, abserr, `converged`, `num_regions` and max depth (hex), and on the whole call record. The 2,844 + 2,844 rows untouched in every call equal 07a's in `total` and `total_abserr`. 0 calls lack the counts. Coverage 20.8% of calls, about a fifth as §9.5 expected |
| **U2** no Levin leaf across a stationary point | **holds** | 0 Levin leaves listed. Window leaves listed: 1,682 (analytic, 1e-7), 526 (group, each rtol) |
| **U3** every phase with something to locate is located | **holds** | 23,998 calls scored, 0 violations. 784 set aside (392 group calls per rtol), every one `zero_rate` with no sign change |
| **U4** honest where referenced | **holds** (as §9.5 predicted) | 190 scorings, 0 converged and dishonest, 181 honest, 3 unconverged, 6 reported and not scored (stated error above min(abserr)/100; ratio ≤ 0.035 on each). The 190 are: V4's 33 on 07's readings; Part A's R1 60, R2 60, R3 (W7 6, aux 12, own 6, pairs 4) and item 3, on Part A's references; 07a's rows 1-34599 and 3-678 (8 calls) on 07a's references. **Unconverged: exactly v04, R2-28 and R2-28~o**, R2-28 at **1.1071** and 1.1071 of its abserr, as predicted. Largest honest ratio 0.805 |
| **U5** the gate is consistent; the guard a backstop (gated solves, 07a's ruling) | **holds** | 0 node-partition sums on gated solves, 0 Clenshaw–Curtis regions with a Levin half, 0 guarded regions. Descriptive: 18,238 forced solves took the node-partition sum (07a: 18,236) |
| **U6** 06b's 24 honest or unconverged | **holds** (as predicted) | 24 of 24 honest; ratios 1.8×10⁻⁵–0.83, all converged and all scored. The four dishonest at 07a (`0-12977` #5, `1-2174` #2, `2-14345` #3, `2-34473` #3) are honest |
| **U7** the pairs | **holds** (as predicted) | 2,484 of 2,484 pairs have r_g ≤ 1, 0 unpaired; largest 0.467. `3-14501` 0.00046 (07a 1.385); `3-35006` 0.00057 (07a 1.084). 14 pairs have an unconverged member |
| **U8** the code is (h) | **holds** | 187 of 187 equal (h)'s replay record in all five outputs. Each record also equals its summary row. They are 185 distinct census calls: `R1-x019-8`/`v12` and `R1-x317-7`/`v07` are each the same call |

### Descriptive (not scored)

- **Where the fix acts:** 229,322 of 289,482 calls (79.2%).

  | rtol, kind | calls | acted | edge (calls / refusals) | range (calls / refusals) |
  |---|---|---|---|---|
  | 1e-7 analytic | 107,416 | 93,932 | 0 / 0 | 93,932 / 351,789 |
  | 1e-7 group | 37,325 | 21,051 | 340 / 438 | 21,049 / 53,794 |
  | 1e-8 analytic | 107,416 | 93,665 | 0 / 0 | 93,665 / 323,256 |
  | 1e-8 group | 37,325 | 20,674 | 334 / 430 | 20,667 / 53,162 |

  - By band: seam 68% of calls acted, mid 91%, low 92%. Every k acts at 78–83%.
  - By 07a's class: untouched 80%, aux_only 97%, own 54%, window_both 47%, window_h_only (the minima windows 07a newly
    located) 0%.
  - Tables by rtol × kind × band × k: `out/scores/summary.json` `where_the_fix_acts`.
- **The edge rule:**
  - 868 refusals in 674 calls, every one a `G+q+r` call with a root window, all in the 720. 636 refusals are at depth 0
    (the seed leaf beside the window). Median q 656, range 2.0006–5464.
  - It was evaluated 2,802 times on analytic-call windows and refused none.
  - It never acted on a minima window.
- **The range rule:**
  - 782,001 refusals: 675,045 on analytic calls, 106,956 on group calls.
  - Only at depths 0–3: 62,145, 184,209, 302,056 and 233,591.
  - q from just above 10 to above 10⁹.
  - The instrument recorded every evaluation: 0 unrecorded, and in every call the instrument's refusals equal the counts.
- **`converged`:**
  - **198 group calls newly unconverged** (97 at 1e-7, 101 at 1e-8; 4 with an edge refusal).
    - In 192 of them every depth-20 leaf (668 leaves) is a Clenshaw–Curtis leaf descending from a leaf the rate rules
      refused. This is `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`.
    - The other 6 have no leaf at depth 20 and no untrusted leaf. Their abserr is 1.001–1.33× 07a's (the budget not met).
  - 5 group calls (1e-8) became converged.
  - No analytic call is unconverged under either code.
  - Rows: 139 `total_converged` true → false, 4 false → true.
  - Unconverged group calls: 07a 82 + 96, 07b 179 + 192 (1e-7 + 1e-8).
  - Row depth 20: 182 / 185 rows (07a 77 / 83).
- **abserr against 07a's:**
  - Untouched calls: equal.
  - Acted calls: median 0.0010 (1e-7 analytic), 0.0081 (1e-8 analytic), 0.40 / 0.51 (group). It grew in about 4% of
    acted analytic calls and about 20% of acted group calls.
  - 738 acted calls' abserr grew by more than 100×, up to 4.6×10⁵. These are analytic calls of one region at 07a, two at
    07b.
- **Values:**
  - 2,672 group calls moved by more than 07a's abserr. **605 moved beyond both codes' abserrs**; 603 of them are converged
    under both codes. No analytic call moved beyond 07a's abserr. None of the 605 is referenced.
  - Row totals: 11,235 rows differ; the move over 07a's `total_abserr` has median 6.6×10⁻⁴ and maximum 1.065, 2 rows above 1.
  - 07b against 07a on 07's 2,484 pairs: 220 calls listed, all class (c).
- **Windows:** stopped 1,247 (1e-7) and 1,236 (1e-8), as at 07a.
- **Cost:**

  | | 07b | 07a | ratio |
  |---|---|---|---|
  | census wall | 45,016 s | 23,284 s | 1.93× |
  | unit wall, 1e-7 | 52,154 s | 24,952 s | 2.09× |
  | unit wall, 1e-8 | 54,667 s | 28,513 s | 1.92× |
  | unit time with post sections, 1e-7 | 95,096 s | 50,696 s | 1.88× |
  | unit time with post sections, 1e-8 | 77,572 s | 39,379 s | 1.97× |
  | regions, all calls | 6,039,639 | 2,909,134 | 2.08× |
  | regions, 1e-7 analytic | | | 2.64× |
  | regions, 1e-7 group | | | 2.00× |
  | regions, 1e-8 analytic | | | 2.22× |
  | regions, 1e-8 group | | | 1.86× |
  | largest unit wall | 40.1 s | 38.4 s | |

  The wall came in 1.20× the central projection, inside its margin. 0 units were over budget.

### Deviations from the prompt

1. **The harness changes** (`prov.py`, `worker.py`, `run_k0.py`, `run_census.py`; `out/scripts_diff_at_end.txt`).
   STRUCTURALLY REQUIRED for the tree, the parent and the two counts. Item 16 covers the backstop.
2. **The parent ran as a variant from `state/variants/` in h**, as the dispatch directs. STRUCTURALLY REQUIRED.
3. **New scripts** (listed under What shipped). IMPLEMENTATION CHOICE.
4. **U0 (3)/(4)'s units:** `h-1e-8-3-14501` was fixed by name (Part A's `p14501_8` and `n0`). The quiet unit was chosen
   from K0 at h by a rule fixed before the run: every call untouched, then the most calls. Only one K0 unit qualified.
   IMPLEMENTATION CHOICE.
5. **The rate rules' instrument** records every evaluation, not only refusals. It reads a leaf's depth from the calling
   frame's `leaf`. IMPLEMENTATION CHOICE.
6. **`--no-instrument` turns off 07a's and 07b's instruments together.** Mode `instr` sets aside both instruments' keys but
   not the two counts. IMPLEMENTATION CHOICE.
7. **`compare_07b.py` adds an `oracle` category** (`analytic_rad`, `analytic_06h`) for the unit's LiouvilleGreen Levin
   calls, which are not recorded calls. IMPLEMENTATION CHOICE.
8. **The projection** adds a schedule simulation, calibrated on 07a's own wall, and stated margins to 07a's model.
   IMPLEMENTATION CHOICE.
9. **`predictions.md` adds §9.7's block and parts 4–5.** IMPLEMENTATION CHOICE.
10. **Smoke tests ran in the session** (`state/smoke/`, `out/cmd/11`–`17`), not scored. IMPLEMENTATION CHOICE.
11. **Part A's whole `measurement/` was hashed** (7,645 files) in place of "the captures you use". U8 reads Part A's replay
    records, and no Part A capture is used. IMPLEMENTATION CHOICE.
12. **The registry has no job-name field.** The run ids carry the prefix `levin-phase-floor-07b`. STRUCTURALLY REQUIRED.
13. **The A3 v2 fingerprint was not recomputed.** The copy is checked by SHA-256 against its certification, as 07a did.
    IMPLEMENTATION CHOICE.
14. **Failed attempts, each re-run.** UNINTENDED DRIFT (failed attempts).
    - zsh did not word-split `$P` (`out/cmd/04`, `05`; re-run as `04b`, `05b`).
    - Two commands written with `$T:scripts` never ran, because zsh reads `:s` as a modifier. They left no record.
    - The first smoke instr comparison set aside the wrong keys (`15`, re-run as `15b`).
    - `compare_07b` raised a KeyError on a leftover reading key (`16`, re-run as `16b`).
    - `score_v4_07b.py` failed because `out/scores/` did not exist (`50`, re-run as `50b`).
15. **Cosmetic:** a GNU `stat` call failed, and `runregistrykit list | head` raised BrokenPipe in `out/cmd/27`. UNINTENDED
    DRIFT (cosmetic).
16. **`run_census.py`'s backstop stops at a projected 48 h** (§9.4) rather than 06k's day. It never acted; the largest
    projection it reported was 44,854 s. IMPLEMENTATION CHOICE.
17. **`launch_census_07b.py`**, 07a's launcher with the 07b module check, prompt and purpose. IMPLEMENTATION CHOICE.
18. **U4's sets.**
    - "07a's census's other eight V4 calls" is read as z00 and Z3's seven, scored with z12 on 06g's readings
      (`score_v4_07b.py`, 07a's W4 method).
    - Part A's 06b W2 calls are scored under U6, not U4.
    - `i3-34599-h` (Part A's item 3) and 07a's `r1-34599-1e-7-2` are the same census call, scored once under each source.
    - IMPLEMENTATION CHOICE.
19. **U4's "scored" rule** is the reference's stated error ≤ min(abserr_07b, abserr_07a)/100, 07a's W4 form. Six calls are
    therefore reported, not scored; Part A's (h) column leaves the same six unscored. IMPLEMENTATION CHOICE.
20. **"Class" in the descriptive tables** is read as 07a's W1 class of the call, beside its kind, a window at 07b, and the
    720. IMPLEMENTATION CHOICE.
21. **`score_07b.py` reads every unit's pickle** (07a's read them only when needed), to aggregate the rate rules'
    evaluations. IMPLEMENTATION CHOICE.
22. **U7's population is 07's 2,484 pairs** (07's census marks), with 07a's census as the before, as 07a scored W6.
    IMPLEMENTATION CHOICE.
23. **Stop 3 and the scoring ran in the session through `rec.sh`**, not as registered runs. The dispatch names only `k0`
    and `census`. The longest took about 3 minutes. IMPLEMENTATION CHOICE.

### Verification performed

`verification.md` has the account. In brief:
- **The trees:** h is at `0a103b5a`, clean with `--ignored` at every check and at the end. The main checkout is at
  `28ebaf07`, clean.
- **Every census worker:** module `8956c7b6` from the tree, no variant, h clean at its SHA, no Ray, no foreign module
  (18,560 of 18,560 units' evidence; 16 workers).
- **The scratch copy:** EQUAL to its certification and to its before-hash, after the census and at the end. `shasum -c`
  against 06e's after-hashes is OK. No `-wal` or `-shm` files.
- **07a's census and Part A's records:** 07a's census is 149 of 149 files equal to 07a's KEPT list; Part A's 7,645 files
  and 07a's 653 working-record files are byte-identical before, after and at the end.
- **The registry:** 86 runs, all finished. No lock was taken.

### Observations not acted on

1. **The chase, measured in the census.** 192 of the 198 newly unconverged group calls end with every depth-20 leaf
   (668 in all) a Clenshaw–Curtis leaf below a leaf the rate rules refused. 139 rows lose `total_converged`.
2. **605 group calls moved beyond both codes' abserrs**, 603 of them converged under both codes.
   - Examples: `h-1e-7-0-12710` `G-q-r` (63.6× 07a's abserr, 1 region → 52) and `h-1e-8-1-34595` `G-q+r`.
   - Under one of the two codes each such call is dishonest. None is referenced. U4's referenced calls, which include
     07a's dishonest R1 rows, are all honest at 07b.
3. **abserr grew by more than 100× in 738 acted calls**, up to 4.6×10⁵. They are analytic calls with one region at 07a and
   two at 07b. Whether 07a's one-region declaration was dishonest is not measured.
4. **The edge rule's reach:**
   - It acts only on `G+q+r` root windows.
   - It was evaluated 2,802 times on analytic windows and refused none.
   - It never acted on a minima window: 07a's 2,360 newly located minima-window calls are untouched.
   - This matches §9.3's "does not reach".
5. **The range rule refuses only at depths 0–3.**
6. **One row untouched in every call** differs from 07a only in its LiouvilleGreen oracle (`analytic_rad`/`analytic_06h`),
   which is not among U1's outputs.
7. **The projection underestimated the wall by 1.20×**, inside its margin. 07a's census wall underlies it, and the acted
   analytic calls weigh more than K0's sample showed.

### State handed on

- **Nothing is running.** Both registered runs are `done`: k0 and census. No lock was taken.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory in full (150 files, 11 GB; **prompt 08's "before"**);
  - the k0 run directory;
  - every pickle and every file over 2 MB under `census/`.
- **Read only and unchanged:** the scratch copy, 07a's census and Part A's records, by hash.
- **Issues:** proposed in `issues.md`, for the laptop and the user.

## levin-phase-floor 07b, Part A (section 2): the measurement — log of the measuring agent

**Subject:** prompt 07b Part A, the leaf beside a window: section 2.1's replays at 07a's code, section 2.2's mechanism,
section 2.3's variants and section 2.4's references. Run on the Mac Studio (mps026298). The worktree is
`/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d`, at the dispatch SHA
`436ed5d43ace32fcfaf5e7517de93978144d824a`, whose code is 07a's (`1ec734d7`, plus `89c368f2`'s docstrings). Records are
in `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/measurement/`.
**Commit:** none. The orchestrator commits the records, on the laptop.
**Result (2026-11-14): the planned stop of section 2.6, with one stop condition of section 8 missed (deviation 16).**
- **The gate** holds before and after (`out/gate/`).
- **Section 2.1:** every replay at 07a's code reproduces 07a's census record bit for bit, 187 of 187, at both rtols for
  every capture.
- **Section 2.2** is done on the five failures, item 3, item 4's four calls and the 21 controls. Every leaf reference
  closes against its call.
- **Section 2.3:** seven variants ((a), (b4), (b8), (c), (d), (e), (f)) are scored on all 187 calls.
- **Section 2.4:** all 74 references are taken (200 readings), and every one is scorable but R2-27 and R3a-0-33666-a13.
- **Section 2.5's tables** are in `out/tables/tables_07b.md` and `out/tables/table3_variants.md`.

**One variant qualifies: (f).** It is not tied to windows. It untrusts every trusted Levin leaf whose |θ′| ranges by
more than 10 over its own nodes, at 2.7× (a)'s regions and wall time, with 61 calls unconverged against (a)'s 1.

**The mechanism.**
- **Beside a window (v08, v12, v17; v13, v18 and v23 among the controls):** a common error (M1). It lies in the eighth
  beside the window, and falls with Φ_w at about Φ_w^(-3/2).
- **Away from windows (v09, item 3, item 4):** the error is common to the leaf, halves and quarters. It does not fall as
  r predicts (M2's signature), and the eighths break the pattern.
- **M3 is excluded on every carrying leaf** by recomputing θ′ from the captured phase.

**Section 8, missed.** The projection of section 2.4's readings left out each Route B call's direct (stationary-point)
segment. It said 9.92 h. By 07a's model, as the prompt specifies, it is about 307 h: beyond 36 h, a stop before the
rest runs. I did not stop. The readings took 28.9 h elapsed, and Part A about 31.5 h.

### What shipped

Nothing in the tree. Its `git status --short --ignored` was empty at every check, and the main checkout stayed at
`28ebaf07`. The records, all under `measurement/`:
- **`scripts/`:**
  - 07a's census harness, copied whole from the tree's `docs/levin-phase-floor/07a/census/scripts/` (225 files,
    `out/scripts_sha256_as_copied.txt`). Every change is marked "prompt 07b": `prov.py`, `rec.sh` and
    `x1_capture_07b.py` (from `x1_capture_07a.py`, its title).
  - 07a Part A's scripts, in `scripts/from_07a_measurement/`.
  - The new scripts: `gate_code_07b.py`, `gate_hashes.py` (07a's, `TREE` changed), `w2_rescore_07b.py`,
    `select_07b.py`, `capture_list_07b.py`, `make_calls_07b.py`, `extract_census_refs_07b.py`, `replay_07b.py`,
    `run_replays_07b.py`, `make_variants_07b.py`, `leafprobe_07b.py`, `probe_all_07b.sh`, `leafspec_07b.py`,
    `leafref_07b.py`, `leafproj_07b.py`, `mech_07b.py`, `m3_theta_07b.py`, `refs_projection_07b.py`,
    `slice_refs_07b.py`, `launch_07b.py`, `score_07b.py` and `tables_07b.py`.
- **`scripts/variants/`:**
  - `levin_quadrature_07b_{b4,b8,c,d,e,f}.py`, each with its diff against the tree in `out/variants/<v>.diff` and its
    SHA-256 in `out/variants/variants.json`: b4 `877faea9…`, b8 `ef89cf1f…`, c `fd125a8d…`, d `06c7f0d8…`,
    e `7df87cac…`, f `fbe2ea9d…`.
  - `levin_quadrature_07.py`, 07's module (`4e5da371…`), for item 3.
  - The first (f), never run, in `scripts/variants/superseded/`.
- **`state/`:**
  - `seeds_07b.json` (`6d99b42e…`), written before any draw.
  - `selection_07b.json` (`4b94c181…`), saved before any reading.
  - `rows.json` (07a's, `aa523d28…`); `calls_07b.json` and `census_wanted_07b.json` (the manifest);
    `census_refs_07b.pkl` (188 of 07a's census records); `capture_h_selection.json`.
  - The leaf specs in `state/leafspecs/`, the reference inputs and specs in `state/refs/`, the plans and their
    purpose/scope texts, `plan_07b.md` (the plan, fixed before any reading) and `deviations_draft.md`.
- **`out/`:**
  - `gate/`; `w2/`; `captures/` (46 captures).
  - `replays/<variant>/` (187 per variant, with summaries), and `replays/not_a_test/`.
  - `probe/`; `leafrefs/`; `mech/` (`mech_all.{json,md}`); `m3/`; `refs/` (the projection, the inputs, the 200
    readings); `tables/`.
  - `cmd/`: every command with its output, by `rec.sh`.
- **Registered runs** (campaign `levin-phase-floor`, prompt `07b`, in the main checkout's `var/runs/`):
  - `levin-phase-floor-07b-pilot-20261113T141257`: done, 1/1, 59 min;
  - `levin-phase-floor-07b-leafrefs-20261113T151923`: done, 29/29, 32 min;
  - `levin-phase-floor-07b-refs-20261113T160925`: done, 200/200, 28 h 52 min.
- **The records:** `log.md` (this file), `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`. The last lists
  166 files: 165 pickles and `state/rows.json`, 117.6 MB.

#### The variants (section 2.3), each with its basis

Every rule-based variant (c)–(f) acts after `trusted = …` on a Levin leaf the tree trusts, and only then. A leaf it
untrusts is bisected as any untrusted leaf is. Each variant was checked on the five no-window calls n0–n4: (b4), (b8),
(c), (d) and (e) are bit-identical to (a) on all five, and (f) on n0 only, by design.
- **(b4), (b8):** `_LEVIN_WINDOW_HALF_PHASE = 4π` and `8π`. This changes every call with a window.
- **(c), the eighths:** solve the eighths, S3. With δ₃ = |Q − S3| and r₃ = δ₃/δ₂, the leaf stays trusted only if
  r₃ < 1/4, or if r₃ < 1 and δ₂ ≤ ν_B. It then declares with max(r, r₃).
  - Basis: M2's test one level further, the check two levels down that the prompt names.
- **(d), an independent check:** the leaf's own Levin antiderivative is evaluated across the 4π of phase next to the
  window edge, with p by barycentric interpolation on the leaf's nodes. It is compared there with the nested
  Clenshaw–Curtis pair, in pieces of at most 2π. The leaf is untrusted if the two differ by more than
  e_leaf + e_cc.
  - Basis (`make_variants_07b.py`): 4π keeps each piece in the pair's regime (below 6π), and reaches past the eighth
    beside the window, where section 2.2 finds the error (deviation 12).
- **(e), the edge's resolution:** the leaf is untrusted if |θ′| differs by more than a factor 2 between the window-edge
  node and its neighbour on the solve's Chebyshev grid (`chebyshev_matrices` at the working order).
  - Basis: next to a stationary point ℓ = |θ′/θ″| at the window edge is about 2Φ_w/|θ′|. The endpoint term's
    correction, of relative size 1/(2Φ_w), is missed while the first spacing is many ℓ, and a factor 2 is a spacing of
    about ℓ.
  - Measured: on every window-edge seed carrier the factor is 259–905 (spacing/ℓ 285–1010). On the honest deep touching
    leaves of the controls, spacing/ℓ is at most 1.17.
- **(f), not tied to windows:** every trusted Levin leaf (not a direct one) is untrusted if max|θ′|/min|θ′| over its
  own Chebyshev nodes exceeds 10 (`_F_RANGE_07B`).
  - Basis, as written into the variant from section 2.2's tables then (88 probed leaves): every dishonest leaf (11)
    spans a factor of 49 or more, and none spanning 30 or less is dishonest. 10 is a decade below the smallest dishonest
    range.
  - With the final tables (deviation 18; 73 scored trusted Levin leaves): the same 11, the smallest end-to-end ratio
    49.4, and none at 30 or below.
  - Honest leaves span ratios up to 3.4×10⁴, so (f) also untrusts many honest leaves: that is its cost.

### Predictions and verdicts

Part A has no scored predictions of its own: those belong to Parts B and C. Its questions, with what was measured:

| Question (prompt) | Measured on | Verdict |
|---|---|---|
| §2.1: replays at 07a's code reproduce its census bit for bit | 187 calls (V4 25, R1 60, R2 60, R3 22, W2 8, item 3, pairs 4, n0–n4 5) | **holds**, 187 of 187; each capture also reproduces at the other rtol |
| §2.1 item 3: 07's capture of 1-34599 replays at 07's module to 07's census | `i3-34599-07` | **holds** |
| §0.3: does 06b's under-declaration persist at 07a's code? | 24 W2 calls against S₁₀ | **4 of 24** dishonest, all converged (2.20, 1.36, 6.01, 1.39), against 17 of 24 at 06b's code |
| §0.3: is it H3's mechanism? | the four, through §2.2 | **not the window mechanism:** no carrying leaf touches a window. The error is common and not falling as r predicts (below) |
| §0.1: are the 20 honest calls' touching leaves honest, or merely small? | 40 touching Levin leaves of the 20 calls and z12 | **honest at leaf level but one:** v23's seed carrier is −1.06 e′, inside an honest call (0.767). The deep touching leaves are all ≤ 0.36 e′, with spacing/ℓ ≤ 1.17 |
| §0.1: are the pairs this defect? | 3-14501 and 3-35006 index 1 | **at 07a the 1e-7 member is converged and dishonest in both** (1.18, 1.08). (b)–(e) bring 14501's two members together (r_g ~1e-6) but leave both dishonest (2.0–2.4). 35006's r_g > 1 survives every variant but (f) |
| §0.4: how honest is the census on R1 and R2? | R1 30 rows × 2 rtols; R2 30 calls × 2 rtols | **R1: 9 of 30 rows converged and dishonest at both rtols** (1.19–10.8). **R2: none** (largest 0.805) |
| §2.4 R1: 07 or 06m? | 30 rows × 2 rtols | **07 nearer in 60 of 60.** 06m's ratios run from 1.04 to 2.8×10⁵ |

### Deviations from the prompt

1. **07a's census harness copied whole into `scripts/`** from the tree's `docs/levin-phase-floor/07a/census/scripts/`,
   byte-identical to the Mac Studio's copy. 07a Part A's replay and attribution scripts are in
   `scripts/from_07a_measurement/`. `prov.py` is changed where marked "prompt 07b", and `x1_capture_07b.py` comes from
   `x1_capture_07a.py` (its title). IMPLEMENTATION CHOICE.
2. **The first capture failed before starting.** `out/cmd/06_capture_h.txt`: `state/rows.json` (07a's copy of 06g's,
   `aa523d28`) had not been copied. It was copied, SHA checked, and the capture re-run as `06b`. UNINTENDED DRIFT
   (a failed attempt).
3. **`make_calls_07b.py`'s first manifest took every no-window call to be a group call.** n4 is analytic J2 (index 7),
   and the census extraction refused it. The kind is now read from 07a's checkpoint; the first manifest is kept in
   `state/superseded/`. UNINTENDED DRIFT.
4. **Two failed first attempts in the replays:**
   - the first tree replay command did not run, because zsh does not word-split a variable;
   - the first 07 replay of `i3-34599-07` failed on `own_err`, which 07's module has none of, and wrote no record. Its
     log is in `out/replay_logs_failed_first/`.

   Fixed (`getattr`) and re-run. UNINTENDED DRIFT (failed attempts).
5. **`i3-34599-07` was included in the tree run by mistake.** It is 07's capture, to be compared with 07's census, and
   at the tree it differs by design. The record is moved to `out/replays/not_a_test/`. It shows that 07's capture of
   1-34599, replayed at 07a's code, returns 07a's census value bit for bit. UNINTENDED DRIFT (cosmetic).
6. **R3's four shrinkers are counted as four distinct (row, kind, index):** 0-33666 J4, 3-39138 #2, 3-39138 #3 and
   2-21486 J2. The literal four smallest calls (0-33666 J4 at both rtols, 3-39138 #2 and #3) are all covered, the second
   rtol by the shared integrand. IMPLEMENTATION CHOICE.
7. **R1's captures are 07's cross-check captures**, re-checked by a replay at 07a's code at both rtols rather than
   captured afresh. The prompt allows either. IMPLEMENTATION CHOICE.
8. **Integrand identity across rtols** is shown by replaying one capture at the other rtol's (rtol, atol) and
   reproducing 07a's census record there bit for bit. This replaces 07a's 1001-point comparison of two captures.
   IMPLEMENTATION CHOICE.
9. **`refs_projection_07b.py` was started while the pilot's six workers ran.** That made a seventh busy process for
   about 11 min. It was stopped and re-run after the pilot. UNINTENDED DRIFT.
10. **Working directories.** The capture workers and the registered runs' jobs run with cwd in the records directory
    (07a's harness design). Every script is invoked by absolute path, with `PYTHONPATH` starting with the tree and never
    by `python -m`, so the cwd is not on `sys.path`. Interactive commands ran with cwd in the worktree.
    IMPLEMENTATION CHOICE.
11. **M3's phase perturbation:** production declares no `theta_abserr`. The stated proxy is the ulp of the raw phase at
    the window edge (2⁻¹³ rad at |θ| ~ 6.8×10¹¹ for v08). IMPLEMENTATION CHOICE.
12. **Variant (d)'s design.** It compares the leaf's own Levin antiderivative across a 4π sub-span beside the window
    with the nested Clenshaw–Curtis pair there. It does not use the sub-span's own Levin solve, which resolves the edge
    by its narrowness and so would not test the leaf. IMPLEMENTATION CHOICE.
13. **Variants (e) and (f) were written from section 2.2's first findings** (v08's carrier; items 3 and 4 without
    windows), before section 2.2 was complete. IMPLEMENTATION CHOICE.
14. **Split edges for the Route B FCC readings.**
    - Section 2.2's piece references, and section 2.4's Route B FCC readings, are read by `leafref_07b.py` with
      `--split-ratio 1.8`. Every monotone segment whose |θ′| varies by more than 1.8 over 33 samples is bisected until
      none does. The tool then integrates it in t, not directly: its own rule sends a segment of slope ratio above 2 to
      the direct path.
    - Section 2.4's Route B FCC is therefore not `one_reading.py`'s FCC. QAWO and Route A are read by `one_reading.py`
      unchanged. The reference is the split FCC r2, and its stated error is the spread of every reading taken.
    - Validated on v08's carrying leaf against the production-edge reading (`out/leafrefs/pilot_v08.json`): at most
      3.4×10⁻⁴³ difference per piece, against e′ 1.7×10⁻³⁶, at 24× less wall time.
    - IMPLEMENTATION CHOICE.
15. **The second projection run also overlapped running work.** `refs_projection_07b.py` was re-run while the
    leafrefs run was going, a seventh process again. It hung for about 50 min building Route A QAWO panel lists, and
    was killed. The script was then patched to count panels without building them and to project the split FCC
    readings, and re-run alone (`17b`, `17c`, `34`, `34b`). The Route A pilot lookup's key was also corrected.
    UNINTENDED DRIFT.
16. **Section 2.4's projection left out the direct segments, and section 8's 36-hour stop was not taken.**
    - **What went wrong.** The split-FCC projection (`refs_projection_07b.py`, `fccsplit_r1/_r2`) costed each Route B
      call at its measured CPU per segment and nothing more. Splitting cannot make a segment holding the stationary
      point monotone, and the tool integrates that segment directly, at a cost proportional to its phase span (07a's
      model's γ D_f term).
    - **The numbers.** I projected 9.92 h. By 07a's refitted model, which the prompt specifies, the readings project
      306.8 h. Split FCC plus 07a's direct term gives 309.0 h. Either is beyond 36 h, which section 8 makes a stop to
      report before the rest runs, with what would be dropped to fit.
    - **What happened instead.** The run went on: 28.9 h elapsed, 55.4 h of job wall over three slots of two workers.
      R1-x196's split FCC alone took 23.2 h (one direct segment, D_f = 5.1×10⁶ rad); projected 0.12 h, and 07a's
      model says 162 h. Part A's own wall time, from the gate (2026-11-13 13:49Z) to this log, is about 31.5 h, under
      36 h.
    - **Why it matters.** The orchestrator was not asked before the run, and the projection in my progress notes was
      wrong by a factor of 3 in fact and 31 by the prompt's model.
    - UNINTENDED DRIFT.
17. **M3's "input recomputed more precisely"** is `m3_theta_07b.py`, written after section 2.2's tables. It compares θ′
    with the derivative of the captured phase itself (the phase the references integrate against): pointwise by
    degree-24 interpolation, and integrated (∫θ′ by 24-point Gauss–Legendre against the phase's difference) per
    segment.
    - The integrated test was added after the pointwise test's noise floor proved too close to δ* on v09.
    - Two first runs failed: an empty `max`, and a `None` printed. They are kept in `out/m3/failed_first/`. A one-call
      trial, `out/m3/trial_v08.json`, is kept.
    - IMPLEMENTATION CHOICE (the method); UNINTENDED DRIFT (the failed runs).
18. **`tables_07b.py`'s first output is superseded** (`out/tables/superseded/`). Its rate-range summary counted
    untrusted leaves, and leaves whose references miss the 100× rule. It now scores trusted leaves with reference
    accuracy ≤ e′/100 and lists the rest. UNINTENDED DRIFT.
19. **The first (f) was replaced before any replay of it.** It was the window-edge resolution rule made general (every
    trusted Levin leaf's end nodes), and was never run. The rate-range rule replaced it on section 2.2's complete data.
    The first is kept in `scripts/variants/superseded/` and `out/variants/superseded/`. IMPLEMENTATION CHOICE.
20. **R2's population** is every Levin call of 07a's census at both rtols, group and analytic: 289,482 calls, 07a's
    stop-3 `compare_census_calls.calls.jsonl`. The draw is uniform, numpy `default_rng(710702)`, and the draw order is
    kept so that a cut to 20 would take the first 20. No cut was needed. IMPLEMENTATION CHOICE.
21. **M1's window replays compare leaves, not spans.** Each call's touching leaf on the same side of the window is
    compared at Φ_w = 2π, 4π and 8π, because the leaf's span changes with Φ_w. The fall is clean only where the leaf
    stays the seed region (v08, v12, v17, v13, v18). IMPLEMENTATION CHOICE.
22. **`mech_all` v1 is superseded by v2,** once v09's no-window carrier (leaf 38) was probed and referenced. v1 is kept
    as `out/mech/superseded_mech_all_v1.*`. IMPLEMENTATION CHOICE.
23. **The closure checks are recomputed into `tables_07b.py`.** They were first made interactively in the session, and
    are now recorded in a file. IMPLEMENTATION CHOICE.

### Verification performed

`verification.md` has the detail. In brief:
- **The gate:** code AST-equal to `1ec734d7` but `89c368f2`'s docstrings; the module `4ea1948b…`; 07a's census checkpoint
  `461df2ec…`. Captures 78/78 and the scratch copy's 5/5 files by hash, before and after.
- **Captures:** 46 of 46 equal to 07a's census.
- **Replays:** 187 of 187 reproduce at the tree, at both rtols; item 3 reproduces at 07's module.
- **Leaf probes:** every probe reproduces its replay's leaf by hex.
- **Closure:**
  - item 3's and item 4's summed leaf references against the call's reference: at most 1.72×10⁻⁴ of abserr;
  - V4's leaf references against 07a Part A's on the same 49 spans: at most 0.0039 e′.
- **References:** every reading's spread is at most min(abserr)/100, but R2-27 (0.107) and R3a-0-33666-a13 (0.014),
  which are reported and not scored.
- **The registry:** no `07b` run is running or stale; 80 runs, all finished.

### Observations not acted on

1. **R1: 9 of 30 class-(c) rows are converged and dishonest at 07a's code, at both rtols,** and at 07's code by the
   same ratios. The rows: 0-14199 2.47, 0-14299 3.08, 0-1777 10.8, 0-1942 3.31, 1-34528 1.37, 1-35423 3.73,
   1-38145 2.03, 2-14011 1.86 and 2-34787 1.19.
   - Under (e) all nine are honest. Under (d) all nine are honest or unconverged (three unconverged). (c) fixes seven of
     nine, all but 1-34528 (2.01) and 2-34787 (1.19). (b4) fixes 4 and (b8) 5.
   - They are not attributed leaf by leaf here.
   - The R1 sample is stratified from 396 rows, so this is a rate on class (c), not on the census.
2. **R3-aux `3-13508` G+q−r #2 at 1e-7 is converged and dishonest at 07's code (1.04) and honest at 07a's (0.020).**
   This bears on `[07a-trust-check-auxiliary-route-moves-the-declared-error]`. `3-843` J2 #5 is honest at both (0.039 at
   07, 0.556 at 07a).
3. **W7's three calls are honest at both codes:** 07a 6×10⁻⁶ to 1.4×10⁻⁴ at their own rtol (0.026 at most at the
   other), and 07 0.105–0.145. This bears on `[07-iroff1-window-stop-can-leave-a-call-dishonest]`.
4. **The pairs:** at 07a, 3-14501's and 3-35006's r_g > 1 is the 1e-7 member's own dishonesty (1.18, 1.08). Variants
   (b4)–(e) bring 14501's members to agree within 10⁻⁶ e₇ while both are about 2× their abserr, so they agree and are
   both wrong. A pair test alone would pass them.
5. **v23's seed carrier is dishonest at leaf level (−1.06 e′) inside an honest call (0.767).** The mechanism is present
   in the controls, and only the other leaves' slack hides it.
6. **W2-2-34473-3's deep leaves (depth 8–11, about 10⁻⁴¹ e′) have references only 2–45× e′ accurate.** 33 Levin leaves
   are not scored at leaf level. The call's excess (1.39) closes against S₁₀ at 2.5×10⁻⁵ of abserr, and its scored
   trusted carrier is leaf 102 (−1.40 e′).
7. **The declared θ′ error is far larger than the actual one.** `_CC_THETA_PRIME_REL_ERROR_SUPPLIED = 2⁻¹⁰` against a
   measured consistency of θ′ with the captured phase of at most 4×10⁻⁹ per segment (integrated; 6×10⁻⁷ pointwise at
   the noise floor). A perturbation at the declared error moves the carrying leaves' values 5.7–10⁴× their true error,
   so the prompt's M3 test at the declared error cannot tell M3 from the others. The recomputation can.
8. **(f)'s cost** is concentrated in R1 (44 of its 61 unconverged) and V4 (15). The unconverged calls end with
   Clenshaw–Curtis leaves near the windows at depth 20, untrusted at the noise level. That is the window-leaf chase of
   07a Part A's `issues.md` item 3, reached by (f) through the bisection it adds.
9. **(b4) and (b8) are not monotone improvements.** Each makes a previously honest call dishonest: (b4) v19 2.17; (b8)
   v07 1.33, v19 1.55 and R1-x317 1.33. b8 also leaves v12 at 4.94.
10. **The (b) scan of the seed carriers** falls at 2.6–4.2× per doubling of Φ_w (v08 2.67, 2.81; v12 2.61, 2.76; v17
    4.18, 3.98; v13 1.89, 2.34; v18 2.21, 2.49). Φ_w^(-3/2) would give 2.83.
11. **Python 3.14 processes, idle since about 28 days before this run, sat on the Mac Studio.** They are not this
    campaign's and were not touched.

### State handed on

- **Nothing is running.** The three registered runs are done, and no `07b` run is running or stale.
- **The worktree is clean** (`git status --short --ignored` empty), and the main checkout is at `28ebaf07`.
- **The records** are in `var/levin-phase-floor-07b/measurement/`, with the pickles and large files listed in
  `KEPT_ON_MAC_STUDIO.txt` and kept on this host.
- **The tables of record:**
  - `out/tables/tables_07b.md` (`1cd49bd0…`): tables 1, 2 and 4, the window scan, the rate-range finding and the
    closure;
  - `out/tables/table3_variants.md` (`200642c5…`): table 3;
  - `out/tables/tables_07b.json` (`1b27eec2…`);
  - `out/tables/scores_final.{json,md}` (`32085de9…`, `8791023c…`): every variant's honesty per call.
- **The references** score both 07a's code now and 07b's code at Part C, since every reading depends on the integrand
  alone: `out/refs/readings/` and `out/tables/scores_final.json` ``refs``.
- **Open for the user at section 2.6:** the choice of variant (only (f) qualifies) and deviation 16.

### Extension (§2.7)

**Subject:** prompt 07b §2.7, the extension of Part A, dispatched 2026-11-15 at `a510e4ad` (the prompt read from the
main checkout with `git --no-optional-locks show`; the code is unchanged, and the worktree stayed at `436ed5d4`). Run on
the Mac Studio (mps026298), 2026-11-15 22:47Z to 23:40Z. Records under `out/ext/`, `scripts/ext/` and `state/ext/`;
Part A's files are unchanged but for this section and those appended to `verification.md` and `issues.md`.
**Commit:** none. The orchestrator commits the records.
**Result: §2.7.4's stop.** The tables are complete; the projection was reported before anything long ran; no §8
condition arose.

**Which variants qualify (§2.5 item 4, unchanged).** Of the six new candidates: **(h)** and **(i2)**. With Part A's
(f), three qualify in all.

| variant | qualifies | converged & dishonest (design / held-out) | unconverged | (X4) unconverged within abserr | unconverged above abserr | regions | wall |
|---|---|---|---|---|---|---|---|
| (a) | no | 32 (14 / 18) | 1 | 1 | 0 | 7,993 | 167 s |
| (e) | no | 11 (11 / 0) | 1 | 1 | 0 | 1.31x | 1.29x |
| (f) | **yes** | 0 | 61 | 59 | 2 (R2-28 at both rtols, 1.107) | 2.74x | 2.76x |
| g1 | no | 24 (14 / 10) | 0 | 0 | 0 | 0.96x | 1.09x |
| g2 | no | 12 (12 / 0) | 0 | 0 | 0 | 1.40x | 1.49x |
| **h** | **yes** | 0 | 3 | 1 (v04) | 2 (R2-28 at both rtols, 1.107) | 1.89x | 1.76x |
| g2h | no | 1 (v19, design) | 2 | 0 | 2 (R2-28) | 1.92x | 1.95x |
| i | no | 3 (W2-14345 at both rtols / R1-x241-7) | 4 | 2 | 1 (W2-12977, 1.32) | 1.39x | 1.24x |
| **i2** | **yes** | 0 | 59 | 57 | 2 (R2-28) | 2.68x | 2.68x |

Every row's pairs: (h), g2h, (i) and (i2) pass both pairs; g1 and g2 fail 3-35006 (r_g 1.15). The full table, Table 3
per call, and the regions/depth/wall per call are `out/ext/tables/scores_ext.md`.

**What (X1)-(X3) find.**
- **(X1) (e)'s remainder:** each of (e)'s 11 converged-and-dishonest calls is carried by **one trusted Levin leaf that
  touches no window**, carrying the call's whole excess (the next leaf at most 4×10⁻⁴ of abserr; the window leaves
  together at most 1.5×10⁻⁷ of abserr, by subtraction). The carriers span |θ′| ratios of 49.4 to 1.4×10⁵ end to end.
  The six W2 calls' carriers are trusted by r < 1/4 (r 0.08–0.25); the pairs' and v09's by the noise clause (r
  0.38–0.81). (e) runs the tree's code on 9 of the 11 (all but the two 3-14501 members). `3-14501`'s two members under (e) share one carrier (leaf 41 and
  leaf 36, the same span, err(Q) −2.47 e′), which is why they agree to 10⁻⁶ e₇ and are both 2.0 of their abserr.
  `out/ext/x1/x1_table.md`.
- **(X2) The mechanism away from windows:** **nothing the code can compute from the leaf's own solves separates the
  six from the honest trusted leaves of rate range above 30.** Of 18 candidates on Part A's 73 scored leaves (26 honest
  above 30), the best is |Q(2N) − Q(N)|/e′ (N = 24, the calls' order): the six 1.13–4.08, honest above 30 up to 1.49
  (margin 0.76, one honest leaf above the six's smallest). Other orders, the eighths (r₃, |S₃ − Q|/e′), the tail of p's
  Chebyshev coefficients, the operator's conditioning and the rate range all overlap more. **A higher collocation order
  does not remove the six's error:** err(Q)/e′ at N′ = 16, 32, 40, 48 moves erratically (W2-12977: −2.2, −1.91, −2.4,
  −5.8, −5.3), and only v22 among the window carriers falls with N. `out/ext/x2/x2_tables.md`.
- **(X3) (f)'s chase:** of the 4,374 leaves at depth 20 in (f)'s 61 unconverged calls, 4,372 lie inside a Levin leaf
  (f)'s rule untrusted (506 such leaves, 373 of them touching a window). They are the C2 gate's fallback leaves (3,961)
  and Levin leaves (411), none a window piece, so **07a's iroff1 stop can count none of them** (0 of 4,374). The cause,
  from `fdiag`'s records of every bisection (4,558 of fallback leaves inside such spans): the trust check runs at the
  noise level (δ₁ about 10⁻⁴³ against requests about 10⁻³³, r 0.3–8), the leaf is untrusted; in 4,009 its residual is
  above its floor, so the round-off exit is not considered, and in 548 the exit's halving test refuses it. |v − ref|/abserr
  with converged set aside is at most 0.41 on 59 of the 61, and 1.107 on R2-28 at both rtols.
  `out/ext/x3/x3_chase.md`.
- **(i) and (i2), the chase addressed:** (i) counts each (f)-untrusted leaf's subtree by 07a's rule; because (f)
  untrusts the root seed leaf of whole calls, the count reaches 6 after 9–11 bisections of Levin leaves not at the noise
  level, and the stop returns them at V_P with the r-based declaration (f) exists to refuse: W2-14345 1.55, R1-x241-7
  2.44. (i2) counts and stops only the subtree's fallback leaves; it qualifies but leaves 59 calls unconverged, because
  its stop, unlike a window's, does not reach the fallback leaves made after it by the subtree's Levin leaves (52 stops
  fired). That is a further variant, beyond the six.
- **(g), the adaptive window:** on the census calls **89 of the 98 calls with a window reach the 8π cap with the edge
  test still failing** (178 of 184 Levin seed edges; spacing/ℓ there 1.04–2.2×10³, rate ratio 2.04–2.1×10³; at a
  passing edge spacing/ℓ at most 0.0068). So g1 is (b8) on those calls (its converged-and-dishonest list is (b8)'s,
  v07, v19 and R1-x317 included), and g2 is (e) at 8π, with v19 dishonest (1.55) where (e) has it honest. On family (R)
  at rtol 10⁻⁹ every case is honest and converged under (a), g1, g2 and g2h; the windows reach 8π on every case with
  c ≥ 10⁶ and stay at 2π at c = 10⁴, and at 8π value and abserr agree with 07's 8π rung (07's module) to the digits 07 printed. `out/ext/g/g_windows.md`,
  `out/ext/family_r/family_r.md`.

**I recommend nothing.**

#### What shipped (extension)

- **`scripts/ext/`:** `rec.sh`, `run_ext.py`, `launch_ext.py`, `replay_ext.py`, `run_replays_ext.py`,
  `leafprobe_ext.py` (Part A's, copied, changed only where marked: paths to `out/ext/`, deviation 27);
  `x1_spec.py`, `leafproj_ext.py`, `x1_table.py`, `x2_candidates.py`, `x2_tables.py`, `x3_chase.py`,
  `make_variants_ext.py`, `score_ext.py`, `family_r_ext.py`, `g_windows.py` (new).
- **`scripts/ext/variants/`:** `levin_quadrature_07b_{g1,g2,h,g2h,i,i2}.py`, the six candidates, and `..._fdiag.py`, an
  instrument; each with its diff in `out/ext/variants/<v>.diff` and SHA-256 in `out/ext/variants/variants.json`
  (`045681c4…`): g1 `4561826a…`, g2 `1550582a…`, h `47512022…`, g2h `a8bd0028…`, i `59100647…`, i2 `3973f45d…`,
  fdiag `a7581707…`.
- **`state/ext/`:** `x1/` (the six specs and `plan.json` `7b032d5c…`), the leafrefs run's plan, purpose and scope.
- **`out/ext/`:** `x1/` (`leafproj_x1.json` `d5da9581…`, `probe/`, `mech_x1.json` `ce696497…`, `x1_table.md`
  `ea035206…`), `leafrefs/` (6 readings), `x2/` (`x2_tables.md` `5d53199f…`), `x3/` (`x3_chase.md` `2f41fbf9…`),
  `replays/` (g1, g2, h, g2h, i, i2 on 187 calls each with instruments; fdiag on 61), `tables/` (`scores_ext.md`
  `f53936d7…`, `scores_ext.json` `508e946a…`), `family_r/` (`family_r.md` `49a41f4a…`), `g/` (`g_windows.md`
  `18147b21…`), `cmd/` and `commands.log`.
- **Registered run:** `levin-phase-floor-07b-ext-leafrefs-20261115T225140`, done, 6/6, 16 min.
- **No new pickle or file over 2 MB:** `KEPT_ON_MAC_STUDIO.txt` is unchanged.

#### The variants (§2.7.2), each with its basis, and the leaves it changes

- **g1, g2 (the adaptive window).** Before the loop, each window's half-width is doubled from 2π while the Levin seed
  leaf beside it fails (e)'s edge test (factor 2 across the first spacing), to Φ_max = 8π; the location is re-run with
  a per-point half-width, windows merging and cut into pieces of at most 2π as the tree does. A seed the C2 gate sends to
  the fallback is not tested. Basis of the cap: §2.7.2's (07's family (R) at rtol 10⁻⁹ clean to 8π; 16π never
  measured); it reaches family (R) only. At the cap g1 trusts the leaf as the tree does; g2 applies (e) on the edges of
  windows still failing at the cap. They change every call whose seed edge fails at 2π (89 of 98 with windows);
  n0–n4 are bit-identical to (a).
- **h ((e) plus the no-window rule).** (e) on Levin leaves touching a window; on the others (f)'s rate range over the
  leaf's nodes at most 10. X2 found no finer rule. Basis: Part A's table, every dishonest leaf spanning 49.4 or more over
  its nodes and none at 30 or less dishonest; 10 is 4.9 below 49.4. It reaches only the 73 scored leaves of the V4,
  item 3 and W2 calls; the R1, R2 and R3 classes (the held-out set) are outside it. It changes window-touching trusted
  Levin leaves (as (e)) and every non-touching trusted Levin leaf of range above 10; not tied to windows (n0 identical,
  n1–n4 differ).
- **g2h.** g2's adaptive window and capped-edge (e), with h's rule on leaves touching no window.
- **i ((f) with the chase addressed).** Each (f)-untrusted leaf's subtree is counted by 07a's iroff1 rule, unchanged
  (`_window_bisection_counts`, 6 counts), and stopped as `_stop_window` stops a window. Basis: X3; 07a's rule for
  window leaves at the same noise level.
- **i2.** (i) counting and stopping only the subtree's fallback leaves, chosen after (i)'s replays (deviation 33).

#### Predictions and verdicts (extension)

| Question (§2.7) | Measured on | Verdict |
|---|---|---|
| X1: what carries (e)'s 11? | the 11, every non-window leaf referenced | one trusted Levin leaf touching no window, per call, the whole excess |
| X2: does another collocation order separate the six? | 73 scored leaves, N′ = 16, 32, 40, 48 | **no**; margin 0.76 at best (2N); nor does any other candidate |
| X3: why does (f) chase to depth 20? | 61 calls, fdiag | noise-level trust checks on fallback and Levin leaves inside (f)-untrusted spans, beyond 07a's window count |
| (g): does the adaptive window resolve the edge by 8π? | 98 census calls with windows; family (R) | **rarely** on the census (89 of 98 capped and failing); family (R) clean |
| X4: are unconverged calls accurate? | every variant | (f) 59 of 61, (i2) 57 of 59, (h) 1 of 3; R2-28 is unconverged and 1.107 under (f), (h), g2h and (i2) |

#### Deviations (extension), numbered from Part A's

24. **`out/ext/cmd/x02_leafproj.txt` failed:** it named `state/leafspecs/v4_v08.json`, which does not exist (v08's spec is
    `pilot_v08.json`). Re-run as `x02b`. UNINTENDED DRIFT (a failed attempt).
25. **The projection reports two numbers:** 07a's model as specified (0.27 h of job CPU with its margin, γ D_f computed
    for every segment and 0 there) and the same calibrated on Part A's own split readings, which run 5.2–7.6× above it
    (2.1 h job CPU, 0.6 h wall). The run took 16 min. IMPLEMENTATION CHOICE.
26. **X1's references read only the non-window leaves,** with halves, quarters and eighths of every trusted Levin leaf;
    the window leaves' summed error is by subtraction from the call's reference. No span holds a stationary point, so no
    segment is read directly (deviation 16's cause). IMPLEMENTATION CHOICE.
27. **Part A's driver, launcher, recorder, replay runner, replay and leaf probe are copied into `scripts/ext/`,** changed
    only in their paths (to `out/ext/`), the variant lookup (the extension's `variants.json` first) and, in the probe, the
    override's hex conversion. Part A's own write to `out/cmd/`, `out/commands.log`, `out/replays/`; §2.7.4 forbids
    changing those. STRUCTURALLY REQUIRED.
28. **`fdiag`, an instrument, not a candidate:** (f) with two records added (every bisection; every round-off exit
    considered). Its replays equal (f)'s on all 61 calls (regions, value, abserr, every leaf). It is not counted among
    the six. IMPLEMENTATION CHOICE.
29. **X2's first run assumed order 16;** the calls run at 24, so "24" was the order itself, and a Clenshaw-Curtis
    quarter has no p sample (the tree run failed). Fixed to N − 8, N + 8, N + 16, 2N; the first script and the 07
    output are in `out/ext/x2/failed_first/`. UNINTENDED DRIFT.
30. **X2's candidates are computed on every trusted Levin leaf of Part A's `mech_all` and tabulated on the 73 scored.**
    C4 is the complexified operator's 2-norm condition number. IMPLEMENTATION CHOICE.
31. **(g)'s design:** "the leaf beside it" is the seed leaf between windows, tested before the loop at the working order;
    a seed sent to the fallback by the C2 gate is not tested; every point of a failing merged window has its half-width
    doubled; g2 applies (e) only on the edges of windows that end at the cap still failing. IMPLEMENTATION CHOICE.
32. **(h)'s threshold is (f)'s, 10,** from Part A's basis; X2 offered nothing finer, as §2.7.2 directs. IMPLEMENTATION
    CHOICE.
33. **(i2) was designed after (i)'s replays,** from (i)'s failure, and takes the sixth slot. After its replays I found
    that its stop, unlike a window's (`_add_leaf`), does not floor-limit the fallback leaves made later by the stopped
    subtree's Levin leaves; that is why 59 calls stay unconverged. Correcting it would be a seventh variant, and is not
    done. IMPLEMENTATION CHOICE (the design); the gap is reported, not fixed.
34. **Family (R) is run by `family_r_ext.py`,** which keeps 07's `calibrate.py` settings (f, phase, atol 10⁻¹⁴|ref|,
    rtol 10⁻⁹, depth_max 20, one process per call) and its closed-form reference (`reference.py`, read only) but not its
    Recorder or §2.5 accounts; run under (a), g1, g2 and g2h. IMPLEMENTATION CHOICE.
35. **Two failed attempts to probe the other-rtol W2 calls:** Part A's probe passed the override's hex strings as floats,
    then my copy converted its unit names too. Their records are in `out/ext/cmd/failed_first/` (their entries stay in
    `out/ext/commands.log`). UNINTENDED DRIFT.
36. **`g_windows.py` failed twice on a syntax error** before running; records in `out/ext/cmd/failed_first/`.
    UNINTENDED DRIFT.
37. **Two outputs superseded:** `out/ext/tables/superseded/scores_ext_v1.*` (scored before (i2) existed) and
    `out/ext/x1/superseded/x1_table.*` (v09's partial reading labelled "by subtraction"). IMPLEMENTATION CHOICE.
38. **"Identical to (a)" in the extended Table 4 compares region arrays as well as value and abserr.** Part A's compared
    value and abserr only; (c) gives 124 here against Part A's 128 (four calls with the same value and abserr and
    different regions); every other Part A variant agrees. IMPLEMENTATION CHOICE.
39. **The split of Table 4 is the prompt's:** design V4, W2 (both rtols), item 3 and the pairs; held-out R1, R2 and R3
    (both rtols). n0–n4 are not scored. No change.

#### Verification performed (extension)

`verification.md` § Extension has the detail. In brief: the worktree's `git status --short --ignored` empty and the main
checkout at `28ebaf07` at the end; nothing under `measurement/` outside `out/ext/`, `scripts/ext/` and `state/ext/`
newer than the extension's start but these three records; every X1 and X2 probe reproduces its leaf by hex; `fdiag`
equals (f) on 61 calls; the registry 81 runs, all finished.

#### Observations not acted on (extension)

12. **R2-28 is honest in one region at (a) (0.000) and unconverged at 1.107 of its abserr under (f), (h), g2h and (i2)**,
    296 regions: the rate-range rule makes it worse. Not attributed.
13. **v19 is dishonest under every variant that widens its window** ((b4) 2.17, (b8), g1, g2 1.55, g2h 1.54) and honest
    under (a), (e), (h). Its carrier is not attributed (it would need new leaf references).
14. **The noise clause trusts the pairs' and v09's carriers** (r 0.38–0.81, δ₁ ≤ ν_B) while they are 1.17–2.66 e′ wrong.
15. **The window stop's own reach:** X3's 4,374 depth-20 leaves are all outside any window; a stop that should cover
    them must follow the leaves made after it, as `_add_leaf` does for windows (deviation 33).
16. **(c)'s "identical" count:** four calls keep value and abserr under (c) with different region arrays (deviation 38).

#### State handed on (extension)

- **Nothing is running.** The one extension run is done; 81 runs, all finished.
- **The tables of record:** `out/ext/tables/scores_ext.md` (Tables 3 and 4 extended, with (X4) and the split),
  `out/ext/x1/x1_table.md`, `out/ext/x2/x2_tables.md`, `out/ext/x3/x3_chase.md`, `out/ext/g/g_windows.md`,
  `out/ext/family_r/family_r.md`.
- **Open for the user at §2.7.4:** the choice among (f), (h) and (i2), the three that qualify, or none.
