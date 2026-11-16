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

### §5's predictions (R31), copied before they run — not yet scored

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

### §6's predictions (the census), copied before they run — not yet scored

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

### §9.5's additions to §5's and §6's predictions (U8 among them), copied before they run — not yet scored

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
